---
title: 构建分块 GEMM
sidebar:
  order: 120
---

:::note[概述]

- 从 TIRx 分块原语出发，构建一个正确的分块 GEMM（通用矩阵乘法），起点是单个输出分块。
- 第 1 步是单分块 GEMM，第 2 步增加 K 循环累加，第 3 步在 CTA（协作线程数组）之间进行空间分块以处理完整矩阵。
- 首先保证正确性；性能优化留给接下来的两章。
:::

GEMM 是本书围绕展开的工作负载。它位于线性层、注意力投影和卷积之下，这些操作占据了 GPU 的大部分时间，因此一个正确的 GEMM 与一个快速的 GEMM 之间的差异，就是让芯片大部分时间闲置与使其饱和之间的差异。

这个差距太大，无法一步跨越。一个达到饱和的核函数（kernel）需要同时调试内存搬运、累加、分块和 Tensor Core 调度，却没有可靠的对象可供对比。更稳妥的做法是从最小的能给出正确答案的核函数开始，然后每次增加一个决策来逐步扩展。

本章编写第一个正确的分块 GEMM。前面的章节抽象地介绍了 TIRx 的作用域（scope）/布局（layout）/调度（dispatch）模型；这里我们将其应用到一个真实的核函数上。我们从一个 128 x 128 的输出分块开始，将其扩展为一个能处理完整大小矩阵的核函数，依次加入 K 维累加和跨多个 CTA 的空间分块。

这是三步走完一条完整 GEMM 优化路径的第一章。在本章中我们构建一个正确的分块核函数并到此为止。下一章（[GEMM async](/books/modern-gpu-programming-for-mlsys/gemm-async/)）将线程拷贝替换为 TMA（张量内存加速器），并通过流水线（pipeline）将数据搬运与计算重叠，而 [advanced GEMM](/books/modern-gpu-programming-for-mlsys/gemm-advanced/) 则更进一步，引入线程束特化（warp specialization）和 CTA 集群（cluster）。每章都建立在前一章之上，因此核函数逐步累积特性而非重新开始。

把每一步理解为对一个单一契约的编辑会有帮助，这个契约有三个条款：哪个**作用域**执行操作、操作数分块使用哪种**布局**、哪条**调度**路径执行它。大多数步骤只有一个主要变化，因此我们用一张小卡片开头，点明这个变化并指出使复用安全所需的任何同步细节。第 1 步确立基线，其余路径在其上编辑。

## GEMM

GEMM 是位于线性层、注意力投影和许多卷积实现之下的稠密矩阵乘法，这正是为什么一个快速的 GEMM 核函数几乎在任何地方都能带来收益。本教程中的示例使用 $D = A B^{\top}$：

- $A$ 的形状为 $M \times K$。
- $B$ 的形状为 $N \times K$。
- $D$ 的形状为 $M \times N$。
- $D[m,n] = \sum_k A[m,k] \cdot B[n,k]$。

这个转置并非我们选择执行的额外操作；它源于数据的存储方式。示例将 $B$ 保持为 $N$ 个长度为 $K$ 的行，这是线性层权重通常采用的布局，因此沿 $K$ 收缩自然就读取了 $B^{\top}$，无需任何重排。

贯穿本教程，我们以 TFLOPS 为单位的吞吐量（throughput）来衡量一个核函数，将每次乘加的两个浮点运算计入挂钟时间：

$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{t_{\text{seconds}} \times 10^{12}}$$

### GEMM 数据通路

本教程中的每一项优化都归结为数据存放在哪里以及它如何流动，因此在编写任何代码之前先把这条路径画出来是值得的。本质上，一个 Blackwell GEMM 核函数围绕两项活动组织：在内存之间搬运分块，以及在其上进行计算。下图追踪了一个分块从输入到输出所经过的每一块内存：

![*Memory Data Flow*](/books/modern-gpu-programming-for-mlsys/img/memory_dataflow.png)

上图展示了每个后续优化都在编辑但从不替换的基线通路。
从左向右读：操作数分块首先从 GMEM 移动到 SMEM；随后 `tcgen05.mma`
消费 SMEM 操作数并将累加器写入 TMEM；最后收尾（epilogue）将 TMEM 读
回到寄存器中，再将结果存储到 GMEM。请记住这条链，因为下面的每一个步骤
改变的只是其中一跳*如何*发生；它从不改变这些跳本身。

## 优化路径

上面这条朴素的数据通路足以得到正确答案，但它让大部分硬件闲置。本教程的其余部分通过逐次添加 Blackwell 特性来弥合这一差距，每个特性都通过一个 TIRx 分块原语表达。我们将遵循的路径依次访问这些特性：

- **TMA 异步搬运**通过 Blackwell 的硬件拷贝路径移动 GMEM 与 SMEM 之间的分块，用屏障（barrier）跟踪完成情况。
- **软件流水线化**使用多个 SMEM 阶段，使下一个 K 分块的数据搬运可以与当前分块上的 Tensor Core 计算重叠。
- **持久化调度**保持一个固定数量的 CTA 池，每个 CTA 通过分块调度器处理许多输出分块，而不是每个分块启动一个 CTA。
- **线程束特化**将生产者、MMA 消费者和写回角色分配到不同的线程束组（warpgroup）。
- **CTA 集群**让两个 CTA 在一个更大的 Blackwell MMA 分块上协作。
- **多消费者执行**使用多个消费者线程束组同时计算分块的不同部分，提升计算密度。

---

## 第 1 步：顺序单分块 GEMM

最简单但仍能走完整条硬件路径的 GEMM 是计算单个输出分块的那个。所以我们从这里开始。第 1 步计算一个 128 x 128 的输出分块，K = 64，足够小以至于不需要任何循环，数据通路的每一部分恰好出现一次。没有重复，我们就可以在不得不推理循环之前，先孤立地观察每一跳。

> **本步骤所确立的内容：基线**
> - 作用域：一个由 128 个线程组成的单线程束组按顺序走完整条路径，一个阶段接一个阶段。
> - 布局：A 和 B 分块位于 SMEM，累加器位于 TMEM，结果通过寄存器暂存输出。
> - 调度：同步的 `Tx.copy` 承担加载，`tcgen05` 执行 MMA。

### 单分块数据流

基线契约确定之后，接下来要确定的是一个分块以何种顺序穿过它。这第一个核函数恰好走一次核心 GEMM 数据通路，即数据流图中的 GMEM -> SMEM -> TMEM -> 寄存器 -> GMEM 链，外层没有包裹任何循环。它分配工作内存，加载操作数，计算乘积，写回结果，然后自行清理：

1. **分配**：SMEM（池分配器），TMEM（`tcgen05.alloc`），mbarrier
2. **加载**：全部 128 个线程协作将 A 和 B 分块从 GMEM 拷贝到 SMEM（同步 `Tx.copy`）
3. **计算**：单个被选中的线程发出 `Tx.gemm_async` + `tcgen05.commit`；所有线程在 mbarrier 上等待
4. **写回**：线程束组将 TMEM 读入寄存器；每个线程将 fp32 转换为 fp16 并写入 GMEM
5. **释放**：TMEM 释放

### 第一个核函数的四个部分

完整的核函数只有几十行，但分块阅读更容易消化。我们分四部分阅读（内存分配、同步加载、MMA 调度和写回），然后再将它们组装成一个核函数。沿途出现的 API 名称是第二部分（[TIRx intro](/books/modern-gpu-programming-for-mlsys/tirx-intro/)、[TIRx layout API](/books/modern-gpu-programming-for-mlsys/tirx-layout-api/)）引入的 TIRx 分块原语词汇。

**内存分配。** 核函数首先为操作数切出共享内存（shared memory），并为 TMEM 地址和 mbarrier 预留槽位：

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")           # TMEM address (4 bytes)
mma_bar = pool.alloc((1,), "uint64", align=8)    # mbarrier (8 bytes)
pool.move_base_to(1024)                           # Skip to offset 1024
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)  # 128×64 fp16
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)  # 128×64 fp16
pool.commit()
```

这里有两个细节值得留意。`pool.move_base_to(1024)` 把 Asmem 和 Bsmem 推到偏移 1024 处，将低地址留给它们上方的小块元数据，使庞大的操作数分块落在一个干净的边界上。而 `layout=A_layout` 向 `tma_shared_layout` 请求一种交换（swizzle）后的 SMEM 布局，TMA 和 `tcgen05.mma` 都可以直接读取，这正是第二部分所描述的那种"布局即契约"义务。

**同步加载。** 缓冲区就位之后，操作数仍然需要到达 SMEM。在这第一个版本中，我们让 CTA 自己的线程来做拷贝：

```python
Tx.cta.copy(Asmem[:, :], A[:, :])
Tx.cta.copy(Bsmem[:, :], B[:, :])
T.cuda.cta_sync()
```

因为这里只有一个分块（M=N=128，K=64），拷贝全部的 A 和 B 就是全部的加载。`Tx.cta.copy(...)` 让 CTA 在那次拷贝上协作，每个线程负责自己那一份数据。随后的 `T.cuda.cta_sync()` 一举两得：它等待每个线程完成，并发布它们的共享内存写入，使得后续 MMA 读取 `Asmem` 和 `Bsmem` 时看到的是完整的分块而非半满的缓冲区。这种由线程驱动的拷贝也正是我们最先要替换的东西；下一章（[GEMM async](/books/modern-gpu-programming-for-mlsys/gemm-async/)）会把它换成 TMA。

**MMA 调度。** 操作数现在位于 SMEM 中，我们可以发出 MMA 了，并且由单个被选中的线程来做：

```python
if warp_id == 0:
    if T.ptx.elect_sync():
        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                      accum=False, dispatch="tcgen05", cta_group=1)
        T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
```

两层嵌套守卫分两步收窄了发出者。外层 `if warp_id == 0` 只保留线程束组的第 0 个线程束（warp），内层 `if T.ptx.elect_sync():` 随后在该线程束内选出一个活跃通道（lane）。两者合起来恰好留下一个线程来运行 `Tx.gemm_async` 和 `tcgen05.commit`。

有必要说清楚那个单线程做了什么、没做什么，因为自然的读法会产生误导。单个发出线程*并不*意味着单线程乘法。计算仍然是完整的分块级 MMA：硬件按 SMEM 操作数布局和 TMEM 累加器布局所描述的分块执行协作乘法。关键在于 `Tx.gemm_async` 是一个*分块操作*，而不是一条硬件指令。K = 64 的分块比硬件 MMA 的 K 原子（`MMA_K = 16`）更宽，因此这一个分块操作降低（lowering）为沿 K 步进的一小串原始 `tcgen05.mma` 指令，线程束组协作驱动每一条。之所以只让一个线程发出分块操作，是因为每条底层 `tcgen05.mma` 本身就是一条*单指令*协作操作：一次启动驱动该 K 原子上的分块 MMA。如果 128 个线程都发出这个序列，同样的工作只会被启动 128 次。最后，`accum=False` 标志告诉 MMA 覆写 TMEM 目标而非累加进去，这正是我们在此想要的，因为此时没有可延续的先前部分和。

**写回。** 乘积现在位于 TMEM 中，但调用方希望它以 fp16 形式回到 GMEM。因此收尾必须把结果经寄存器取回，并在途中做类型转换：

```python
Dreg = T.alloc_local((BLK_N,), acc_type)        # per-thread fp32 register row
Dreg_f16 = T.alloc_local((BLK_N,), d_type)      # same row, cast to fp16
Dreg_wg = Dreg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
Tx.cast(Dreg_f16[:], Dreg[:])
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

MMA 在 TMEM 中留下一个 128 x 128 的 fp32 累加器分块。采用 fp32 是有意为之：GEMM 沿 K 对许多乘积求和，用更高精度保存运行中的部分和可以压住本会累积的舍入误差。但 `D` 是 fp16，所以这些值不能直接输出。它们先进入寄存器，在那里收窄为 fp16，然后才到达 GMEM。

两个寄存器缓冲区扮演不同角色。`Dreg` 是每线程 `BLK_N` 个元素的缓冲区，而 `Dreg_wg` 是同样这些寄存器在选定布局下的线程束组级*视图*：

```python
TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)])
```

这个布局将分块的第一维映射到线程束组的线程上：线程 0 拥有第 0 行，线程 1 拥有第 1 行，以此类推直到第 127 行。第二维留在每个线程自己的寄存器缓冲区内，因此一个线程持有它那一行的所有列。线程束组有 128 个线程，分块有 128 行，128 x 128 的输出就整齐地划分为每线程一行。

在该视图下读出累加器正是 `Tx.wg.copy_async(Dreg_wg, tmem)` 所做的，它降低为 Blackwell 的 TMEM 加载路径 `tcgen05.ld`。由于该加载是异步的，`T.ptx.tcgen05.wait.ld()` 必须在任何线程触碰 `Dreg` 之前完成；否则线程会读到加载尚未填充的寄存器。

等待返回后，每个线程私有的 `Dreg[:]` 持有它那一逻辑输出行的 fp32 值。线程在 `Dreg_f16` 中将它们收窄为 fp16，算出自己负责的全局行号，

```python
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
```

并写入 `D[m_thr, n_st:n_st + BLK_N]`。各行在四个线程束间干净地划分：线程束 0 写第 0-31 行，线程束 1 写第 32-63 行，线程束 2 写第 64-95 行，线程束 3 写第 96-127 行。

### 完整核函数

现在我们把四部分重新拼成一个可运行的核函数（M=N=128，K=64）。先放导入：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

核函数包裹在后续步骤也使用的 `hgemm_vX(M, N, K)` 风格中。第 1 步以 `M=N=128, K=64` 运行，因此启动恰好包含一个输出分块：

```python
def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    # MMA_M/MMA_N/MMA_K document the underlying hardware MMA tile; they are not
    # passed to gemm_async (which derives the MMA shape from the operand and
    # accumulator tiles), so the later steps omit them.
    MMA_M, MMA_N, MMA_K = 128, 128, 16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # Step 1 is a single-tile kernel: M = BLK_M and N = BLK_N, so the grid
        # is 1x1. Starting with a 1x1 grid keeps the per-CTA tile offsets
        # (m_st, n_st) trivially zero; Steps 3+ generalise this to larger M / N.
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])      # single warpgroup, so wg_id is always 0 (unused below)
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
    
        # --- SMEM allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()
    
        # --- Barrier + TMEM init (warp 0 only) ---
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
    
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
    
        tmem = T.decl_buffer(
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )
    
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_mma: T.int32 = 0
    
        # --- Load: all threads copy global -> shared (synchronous).
        # With M=BLK_M and N=BLK_N the slices below cover the full matrices;
        # the slice form is kept so the diff to Step 3 (multi-tile) is minimal.
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()
    
        # --- Compute: single elected thread issues MMA ---
        if warp_id == 0:
            if T.ptx.elect_sync():
                Tx.gemm_async(
                    tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                    accum=False, dispatch="tcgen05", cta_group=1
                )
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
    
        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
    
        # --- Writeback: TMEM -> RF -> GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
    
        # --- Deallocate TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

随后的每个 GEMM 步骤都以同样的方式编译、运行和自检，因此我们在此完整地把这套脚手架说一遍，之后只展示核函数。要运行后续某个步骤，把对应的 `hgemm_vX` 和匹配的问题规模替换到下面即可。有一个注意事项值得记住：每个全新的 Python 会话只编译一个步骤，在尝试另一个之前重启，因为示例复用了内部名称，而编译器持有每会话状态。

```python
import torch

target = tvm.target.Target("cuda")
device = torch.device('cuda')  # gpu(0)

M, N, K = 128, 128, 64
kernel = hgemm_v1(M, N, K)
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

torch.cuda.empty_cache()
torch.cuda.synchronize()
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")

ITERS = 10
for _ in range(3):
    ex.mod(A_tensor, B_tensor, D_tensor)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(ITERS):
    ex.mod(A_tensor, B_tensor, D_tensor)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / ITERS
tflops = 2 * M * N * K / ms / 1e9
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

第 1 到第 3 步刻意在较小规模下运行（这里是 128×128，第 3 步是 256³），以便让这些最初的讲解易于跟随。[advanced GEMM](/books/modern-gpu-programming-for-mlsys/gemm-advanced/) 末尾的跨步骤*端到端结果*表则采取相反的做法：它在单一的 M=N=K=4096 规模下测量每一步（包括这个第 1 步算法），使其加速比可以直接对比。

### 单分块核函数的局限

这个核函数是正确的，这正是第 1 步的全部目的，但它只在非常狭窄的设定下正确。有四个局限是刻意植入的，优化路径的其余部分会逐一解除：

- 它只处理单个 K 分块，因此无法在很大的 K 上收缩。
- 它只处理单个输出分块，因此 M 和 N 被钉在 128。
- 它使用同步的 GMEM -> SMEM 拷贝而非 TMA。
- 它不把数据搬运与计算重叠，所以两者从不同时运行。

---

## 第 2 步：K 循环累加

要解除的第一个限制是最小的一个。第 1 步只处理单个 64 宽的 K 分块，而真实矩阵的收缩远超于此。在第 2 步中我们保留单个输出分块，但让 K 跨越多个 64 宽的块。

思路很直接：对每个块重复一次加载 -> MMA -> 等待序列，并让每个 MMA 累加到同一个 TMEM 槽中。真正的工作其实在于同步。在迭代间复用同一个 mbarrier 引入了本章第一个真正的正确性隐患。如果代码跟踪了错误的相位，一个等待可能在它的 MMA 真正完成*之前*就返回，悄无声息地破坏结果。下面的机制正好展示了这如何出错，以及如何避免。

> **本步骤所改变的内容：布局复用**
> - 作用域：不变，仍是单个线程束组。
> - 布局/复用：同一对 SMEM 分块和 TMEM 累加器槽在 K 循环中被复用。不分配新的存储；操作数分块流过一个固定的缓冲区对，累加器状态停留在一个 TMEM 槽中。
> - 同步：被复用的 MMA 屏障必须在每个 K 块上推进到正确的相位，否则后续的等待可能观察到更早一次的完成。
> - 调度：不变。

### K 循环机制

第 1 步在单个 64 宽的 K 分块上收缩；这里我们保留它的单个输出分块，但让 K 按矩阵所需运行任意长。为了覆盖大于 64 的 K，我们以 `BLK_K=64` 为步长遍历 K。每次迭代将下一片 A 和 B 的 K 切片加载到 SMEM 并发出 `Tx.gemm_async`。`accum` 标志把这些块缝合成一个点积：第一个块 `accum=False` 初始化 TMEM 累加器，此后每个块 `accum=True` 将该块的乘积累加到 TMEM 中已有的运行和上。

需要小心的是同步。我们为每次 MMA 完成复用同一个 mbarrier，而安全复用它归结为跟踪我们正在等待哪个屏障相位。一个 mbarrier 携带 1 位相位，要么 0 要么 1，每次预期到达落地时它翻转到另一个值。微妙之处在于等待条件本身：`try_wait(bar, phase)` 阻塞直到屏障内部相位与 `phase` 参数*不同*。所以我们传入的参数要命名我们预期留在身后的相位，而不是我们等待到达的相位：

| K 迭代 | 等待前的本地 `phase_mma` | `try_wait` 等待的内容 | 等待后的本地更新 |
|---|---:|---|---:|
| 0 | 0 | 屏障翻转到 1 | `phase_mma = 1` |
| 1 | 1 | 屏障翻转到 0 | `phase_mma = 0` |
| 2 | 0 | 屏障翻转到 1 | `phase_mma = 1` |

`phase_mma ^= 1` 这一行正是让这张表保持正确的原因。去掉它，第二次迭代仍然调用 `try_wait(bar, 0)`，但屏障在第一次 MMA 之后已经翻转到相位 1，于是等待看到不匹配并立即返回，而第二次 MMA 尚未完成。核函数随后读到一个半成品的累加器，在没有任何错误的情况下报告错误答案。这是一个能完美编译和运行的 bug，这也正是为什么相位翻转值得如此多的关注。

### 完整核函数

下面的完整核函数只是把 K 循环和相位翻转折叠进第 1 步。导入与之前相同：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

它包裹在 `hgemm_v2(M, N, K)` 中。网格仍是 `[1, 1]`，因为我们仍在计算单个输出分块；唯一增长的是它的 K 范围：

```python
def hgemm_v2(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])  # still one output tile (M=N=128)
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # === K-loop: iterate over K in chunks of BLK_K ===
        for i in T.serial(K_TILES):   # serial device loop (keeps the full-K A/B parameters correctly shaped)
            # Load the i-th K chunk
            Tx.cta.copy(Asmem[:, :], A[:, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[:, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            # MMA: accum=False for first tile, True for rest
            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            # Wait for MMA, then flip phase
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # === Writeback (same as Step 1) ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

---

## 第 3 步：空间分块（多 CTA）

K 循环解决了收缩维度，但 M 和 N 仍然钉在单个 128 x 128 分块上。真实输出远大于一个分块，因此基础核函数的最后一块是用许多分块同时覆盖 M 和 N。第 3 步启动一个二维的 CTA 网格，每个输出分块一个 CTA，让 GPU 并行计算所有分块。示例使用 M=N=K=256，得到一个 2x2 的分块网格，刚好让索引变得不平凡又不至于淹没它。

> **本步骤所改变的内容：作用域**
> - 作用域：一个二维 CTA 网格，每个 CTA 拥有一个 128 x 128 的输出分块。
> - 布局：不变；在 CTA 内部，这与第 2 步的 SMEM/TMEM/寄存器通路相同。
> - 调度：不变。

### 网格映射

网格形状直接来自分块：每个 128 x 128 输出分块对应一个 CTA，总共需要 `[M // BLK_M, N // BLK_N]` 个 CTA。与第 2 步相比唯一真正新增的工作，是教会每个 CTA 哪一片矩阵是*它的*待计算切片。

CTA `(bx, by)` 拥有这个输出区域：

```text
D[bx * BLK_M : (bx + 1) * BLK_M,
  by * BLK_N : (by + 1) * BLK_N]
```

为了产生它，该 CTA 的 K 循环反复加载它自己的 A 行带和 B 列带的对应 K 切片：

```text
A[bx * BLK_M : (bx + 1) * BLK_M, k : k + BLK_K]
B[by * BLK_N : (by + 1) * BLK_N, k : k + BLK_K]
```

索引直接来自 `D = A @ B.T` 约定：`bx` 选择 A 和 D 的行，而 `by` 选择 B 的行，转置之后它们成为 D 的列。

每个 CTA 一个分块是最简单的可行映射，但也很浪费。同一行的每个 CTA 都从 GMEM 重新加载相同的 A 分块，同一列的每个 CTA 都重新加载相同的 B 分块，完全不复用相邻 CTA 已经拉入的数据。我们暂时把这个浪费留在原地；持久化调度（[GEMM async](/books/modern-gpu-programming-for-mlsys/gemm-async/) 的第 6 步）会回到这里，让这些共享操作数在 L2 中保持热态。

**与你的 agent 一起尝试**：给定 `M=N=K=256`、`BLK_M=BLK_N=128` 和 `BLK_K=64`，让它追踪 CTA `(1, 0)` 和 CTA `(0, 1)`。对每个 CTA，列出 `m_st`、`n_st`、每次 K 迭代加载的 A 和 B 切片，以及写入的 D 区域。因为核函数计算的是 `D = A @ B.T`，哪些 B 行会成为 D 的列？

### 完整核函数

核函数再次是第 2 步，这次只有两处改动：网格形状和每 CTA 的偏移。内层 K 循环和写回原封不动。导入相同：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

网格变成 `[M // BLK_M, N // BLK_N]` 而非 `[1, 1]`，加载和存储现在按 CTA 自己的 `m_st` 和 `n_st` 偏移：

```python
def hgemm_v3(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 2D grid: one CTA per 128x128 output tile
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0

        # Per-CTA tile offsets
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # K-loop with offset A and B slices
        for i in T.serial(K_TILES):   # serial device loop (keeps the full-K A/B parameters correctly shaped)
            Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # Writeback to the correct output tile
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st:n_st+BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

## 练习

1. 在第 1-3 步中，`Tx.copy` 在 MMA 之前把 A 和 B 分块搬入 SMEM。为什么核函数在 `Tx.gemm_async` 读取这些 SMEM 分块之前需要 `T.cuda.cta_sync()`？
2. 在第 2 步中，如果把 `phase_mma ^= 1` 从 K 循环中移除会发生什么？核函数还会等待每次 MMA 吗，还是后续的等待可能过早通过？
3. 对 M=N=4096、BLK_M=BLK_N=128，第 3 步会启动多少个 CTA？哪些操作数分块在相邻 CTA 之间被逻辑复用，第 3 步是否利用了这种复用？

