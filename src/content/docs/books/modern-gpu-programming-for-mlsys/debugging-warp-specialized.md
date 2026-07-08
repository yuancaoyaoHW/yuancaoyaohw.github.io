---
title: 调试线程束特化核函数
sidebar:
  order: 170
---

# 调试线程束特化核函数

{ref}`chap_gemm_advanced` 中的 GEMM 第 7-9 步重叠了 TMA 加载、`tcgen05` MMA 和 TMEM/SMEM 写回。同样的调试方法适用于 Flash Attention 的交接：识别角色，识别每个角色拥有的存储，然后对照该模型验证生成的 CUDA。

不要一开始就重写核函数（kernel）。首先确保运行是有效的，然后检视生成的 CUDA。在排除了环境和编译期问题后，这些核函数的运行时失败通常归结为一个损坏的交接：未初始化的屏障（barrier）、错误的到达计数、藏在角色守卫内的集体操作、过期的屏障相位，或者存储在生产者使其写入可见之前就被复用。

## 调试核函数之前

先排除运行时上下文问题：

```bash
python -c "import tvm, tvm.tirx; print(tvm.__file__, tvm.__version__)"
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"

这些核函数面向 Blackwell（`sm_100a`）。如果 Python 导入了过期的 TVM 检出版，或 GPU 不是 Blackwell 级别，请在修改核函数之前先解决。然后运行核函数的最小正确性检查（如 `run_correctness()`），再看性能。

## 调试工作流

1. 在仍然失败的最小形状上复现失败。如果失败是非法内存访问，下次运行前重启 Python。
2. 如果编译失败，在阅读运行时同步代码之前，先检查已安装的 API、目标、`dispatch=` 和缓冲区作用域。
3. 保存 `inspect_source("cuda")` 输出。在重新阅读 Python 之前，先在其中搜索角色守卫、`mbarrier_init`、`tcgen05`、`cp.async.bulk.tensor` 和 `cta_sync()`。
4. 为失败的核函数路径编写角色 / 存储 / 交接 / 生命周期表。
5. 对照该表检查生成的 CUDA：屏障初始化在角色分支之前，预期的 TMA 生产者，MMA 发射者，写回组，以及 warpgroup 专属分支内没有 CTA 级集体操作。
6. 将运行分类为死锁、崩溃、错误结果或正确但慢，然后使用下方匹配的章节。
7. 每次只改一个交接：init 计数、arrive/wait 相位、角色守卫、fence、TMA 存储排空、TMEM alloc/dealloc 或分块调度器推进。
8. 测量性能前先重新运行正确性检查。

## 交接清单

对于任何异步核函数，在修改代码之前先做一个小工作表：

| 项目 | 写下什么 |
|---|---|
| 角色 | 发射每个异步操作的确切线程、warp（线程束）、warpgroup 或 CTA。 |
| 存储 | 每一步每个分块的活动位置：GMEM、SMEM、TMEM 或寄存器。 |
| 交接 | 生产者、消费者、信号对象、到达计数、相位，以及使数据可见的 fence 或排空。 |
| 生命周期 | 每个存储槽可被复用、读回或释放的最早时刻。 |

然后对照工作表验证生成的 CUDA：

- 角色守卫与角色表匹配。
- 屏障初始化出现在守卫角色分支之前。
- 集体操作未被 lane、warp 或 warpgroup 守卫意外收窄。
- Arrive/wait 相位与交接表匹配。
- TMA 存储排空、TMEM dealloc 和 SMEM 复用仅发生在生命周期表允许之后。

对 TMA->MMA->写回 GEMM 流水线和 Flash Attention 中 score/softmax/value/correction 交接使用同一张工作表。

## 如果编译失败

在调试运行时同步之前先修复编译期失败：

| 症状 | 可能区域 | 首先检查 |
|---|---|---|
| 未知的 TIRx API 或属性错误 | 安装的 wheel 与教程代码不匹配 | 打印 `tvm.__file__` 和 `tvm.__version__`；将 API 名与 {ref}`chap_language_reference` 对比。 |
| 不支持的 `dispatch=` | 所选目标或原语不支持该路径 | 检查 `dispatch` 参数和目标能力；本教程中的 `tcgen05` 路径需要 Blackwell。 |
| 缓冲区作用域不匹配 | 缓冲区通过错误的硬件路径使用 | 检查工作表的存储行：TMEM 必须通过 `tcgen05` 访问，TMA 操作数必须使用兼容的 GMEM/SMEM 布局。 |
| 编译成功但生成的 CUDA 缺少预期路径 | 派发未按你预期的方式降低 | 在修改算法之前先检视生成的 CUDA 中的 `tcgen05` 和 `cp.async.bulk.tensor`。 |

## 检视生成的代码

对于任何编译后的核函数，保存 CUDA 以便搜索和对比：

```python
from pathlib import Path

cuda_source = ex.mod.imports[0].inspect_source("cuda")
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/my_kernel.cu").write_text(cuda_source, encoding="utf-8")
print(cuda_source)

生成的代码将 TIRx 构造映射到 CUDA，如下所示：

| TIRx | 生成的 CUDA |
|------|---------------|
| `wg_id == 0` | `(warp_id_in_cta >> 2) == 0` |
| `wg_id == 1` | `(warp_id_in_cta >> 2) == 1` |
| `warp_id == 0` | `(warp_id_in_cta & 3) == 0` |
| `warp_id == 3` | `(warp_id_in_cta & 3) == 3` |
| `lane_id == 0` | `(((int)threadIdx.x) % 32) == 0` |
| `.init()` 内部守卫 | `((int)threadIdx.x) < 1`（仅 CTA 线程 0） |
| `elect_sync()` | `tvm_builtin_elect_one_sync_op()` |

在阅读完整核函数之前先搜索这些字符串：

| 生成的 CUDA | 检查 |
|---|---|
| `if (threadIdx.x < 1)` | 单 CTA 线程守卫，通常是屏障初始化 |
| `mbarrier_init` | 屏障初始化存在且出现在角色分支之前 |
| `tcgen05` | Tensor Core 路径已生成 |
| `cp.async.bulk.tensor` | 拷贝降低为 TMA |
| `cta_sync();` | CTA 级屏障；它不能位于 `wg_id` 分支内 |

## 第 7 步参考骨架

一个正确编译的第 7 步核函数具有如下顶层形状。下方的守卫以角色名书写以便阅读；在生成的 CUDA 中，搜索上表对应的表达式。

```c
// (1) 屏障初始化：顶层，仅 CTA 线程 0
if (threadIdx.x < 1) {
  mbarrier_init(tma2mma[0..1], 1);
  mbarrier_init(mma2tma[0..1], 1);
  mbarrier_init(mma2ld, 1);
  mbarrier_init(ld2mma, 128);   // 由 WG0 全部 128 个线程到达
}

// (2) TMEM 分配：WG0 warp 0，发射 warp 的所有通道
if (wg_id == 0 && warp_id == 0) tcgen05_alloc(..., 512);

// (3) fence + cta_sync，然后相位初始化：生产者=1，消费者=0

// (4) 线程束特化循环
if (wg_id == 1 && warp_id == 3 && elect_sync) { /* TMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 1 && warp_id == 0 && elect_sync) { /* MMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 0)                                { /* WB   */ while(valid){ ... next_tile(); } }

// (5) 清理：发射 warp，无 lane 守卫
cta_sync();
if (warp_id == 0) { tcgen05_relinquish_alloc_permit(); tcgen05_dealloc(..., 512); }

在修改算法之前检查这些：

- 屏障初始化位于顶层，不在 `wg_id` 守卫内。
- `tcgen05_alloc` 和 `tcgen05_dealloc` 有 warp 守卫但无 lane 守卫；发射 warp 的所有通道参与。
- TMA 和 MMA 循环都迭代 `K_TILES` 次。
- 相位初始化为生产者=`1`，消费者=`0`。

## 症状映射表

从症状出发，但将其视为线索而非最终诊断：

| 线索 | 可能区域 | 首先检查 |
|---|---|---|
| 核函数挂起，然后运行时报告未定义的启动失败 | 死锁 | 屏障初始化位置、到达计数、`cta_sync()` 位置和 `next_tile()` 参与 |
| 非法内存访问、XID，或后续不相关的 CUDA 调用也失败 | 崩溃 / 中毒上下文 | 重启 Python，然后检查指针范围、存储生命周期和集体操作参与 |
| 128 行或分块大小的条带中出现错误行 | 同步竞争或分块索引不匹配 | 生产者/消费者相位、调度器推进以及哪个 warpgroup 拥有每个行条带 |
| `NaN` 或明显无效的值 | 描述符、操作数设置或未初始化累加 | SMEM/TMEM 描述符设置、交换/布局和累加器初始化 |
| 有限但有规律的错误值 | 过期或部分可见的数据 | 缺失 fence、缺失 TMA 存储排空，或存储在生命周期表允许之前被复用 |
| 输出正确但无预期加速 | 派发或资源问题 | 生成的 CUDA 路径、流水线深度、占用率和寄存器溢出 |

## 何时重启 Python

CUDA 错误并不总是自行清理。在非法内存访问、XID 或"CUDA 上下文中毒"错误之后，后续不相关的调用（如 `torch.randn`）可能持续失败。在测试下一个修复之前重启 Python 进程，否则你可能调试的是上一次崩溃而非当前代码。

## 死锁

按顺序检查这些：

- **到达计数与 init 计数不匹配。** 常见情况：`MBarrier.init(128)` 但 `arrive` 被 `if warp_id == 0: if lane_id == 0:` 守卫，因此只有 1 个线程到达，wait 永不返回。

  | 屏障 | init(count) | 谁到达 | 到达数 |
  |---|---|---|---|
  | `TMABar` (tma->mma) | 1 | TMA 引擎经 `arrive(stage, bytes)` | 1 |
  | `TCGen05Bar` (mma->tma, mma->ld) | 1 | MMA warp 经 `tcgen05.commit` | 1 |
  | `MBarrier` (ld->mma) | 128 | WG0 全部线程经 `arrive` | 128 |

- **屏障初始化嵌套在 `wg_id` 守卫内。** `.init()` 降低为 `if threadIdx.x < 1:`，即 CTA 线程 0。CTA 线程 0 属于 WG0，因此 `if wg_id == 1:` 阻止每个线程运行初始化。初始化必须在顶层；在 `inspect_source()` 中 `grep mbarrier_init` 验证。

- **`cta_sync()` 位于 warpgroup 分支内。** `cta_sync` 是 `__syncthreads()`，要求所有 CTA 线程。在 `if wg_id == 0:` 内，WG1 永远到不了。用 `T.cuda.warpgroup_sync(10)` 做单 warpgroup 屏障。

- **`tile_scheduler.next_tile()` 被某些消费者 warpgroup 线程跳过。** 调度器跟踪每线程状态；跳过它的线程可能永远循环。

- **TMA 和 MMA 对 K-tile 计数不一致。** 如果 MMA 做 `K_TILES - 1` 而非 `K_TILES`，屏障相位漂移，第二个外层分块死锁。

- **`PipelineState` 初始相位错误。** 生产者从 `phase=1` 开始使首次 wait 通过；消费者从 `phase=0` 开始使首次 wait 阻塞。如果两者从相同相位开始，首次交接可能立即死锁。

## 崩溃与上下文中毒

常见原因：

- **`pool.commit()` 之后 `pool.alloc`。** 屏障包装器在内部调用 `alloc`。正确顺序：`tmem_addr -> 屏障包装器 -> move_base_to(1024) -> Asmem / Bsmem / Dsmem -> commit()`。
- **`tcgen05.alloc` 或 `tcgen05.dealloc` 带 lane 守卫。** 发射 warp 必须所有通道参与。`if lane_id == 0:` 只运行一个线程，这是未定义行为。
- **`tcgen05.dealloc` 之前缺失 `cta_sync()`。** TMEM 在写回仍在读取时被释放。
- **GMEM 或 SMEM 越界访问。** 缩小到一个分块，检查调度器的 `m_idx` / `n_idx`，并检查当前形状是否为核函数分块或集群分块的倍数。

## 错误结果

在猜测之前先按模式分类错误输出。整行条带通常指向生产者/消费者相位、分块索引或角色归属不匹配。`NaN` 输出通常指向描述符设置、操作数设置或未初始化累加。有限但有规律的错误值通常意味着消费者读到了旧分块、半写分块或存储尚未排空的数据。

- **`tcgen05.commit` 在 `elect_sync` 之外。** 全部 32 个线程创建提交组；31 个空组立即向 mbarrier 发信号。TMA 可能在 MMA 读取之前覆盖 SMEM。
- **TMA 存储之前缺失 `fence.proxy_async("shared::cta")`。** TMA 引擎可能看不到线程的 SMEM 写入。
- **TMA 存储之后缺失 `cp_async.bulk.commit_group()` 加 `wait_group(0)`。** 下一个分块可能在存储排空之前复用 Dsmem。
- **持久化核函数在小尺寸（如 1024x1024）时间歇性失败。** 更大的尺寸可用更长的 K 循环掩盖竞争。重新检查分块间的相位重置和 TMA 存储的 commit/wait。
- **`fence.after_thread_sync()` 通常不是解法。** MMA 完成 mbarrier 已携带 release-acquire 语义。第 8、9 步在写回边上（`mma2ld.wait` 之后、首次 `tcgen05.ld` 之前）保守地添加了它；不要在 TMA 到 MMA 边上例行添加。

## 正确但慢

如果输出正确但性能远低于预期，使用同样的检视循环：

| 线索 | 可能区域 | 首先检查 |
|---|---|---|
| 生成的 CUDA 没有 `cp.async.bulk.tensor` | 拷贝未降低为 TMA | 检查 `dispatch="tma"`、目标能力和操作数布局 |
| 生成的 CUDA 没有 `tcgen05` 路径 | MMA 未降低为 Blackwell Tensor Core 指令 | 检查 `dispatch="tcgen05"`、目标能力和操作数布局 |
| TMA 和 MMA 不重叠 | 流水线太浅或相位使生产者/消费者串行化 | 检视生成的 CUDA 中 wait/arrive/advance 的顺序 |
| 小形状正确性良好但大形状速度差 | 寄存器溢出、占用率或暂存缓冲区压力 | 检查编译器资源报告；减小分块大小、分块写回或降低流水线深度 |

## 提交一个好的 Issue

如果失败通过了上述所有检查，在 [Apache TVM GitHub 仓库](https://github.com/apache/tvm/issues) 提交 issue 之前先缩减它。包含：

- `tvm.__file__` / `tvm.__version__` 输出和 GPU 能力；
- 复现失败的最小形状；
- 失败是编译期、死锁、崩溃、错误结果还是正确但慢；
- 最小核函数或 notebook 单元及其正确性检查；
- 保存的 `inspect_source("cuda")` 输出，或显示可疑守卫、屏障或派发路径的最小摘录。

