---
title: TIRx 简介
sidebar:
  order: 100
---

:::note[概览]

- TIRx 是一个用于在 IR 层级编写 GPU 核函数（kernel）的 Python DSL（领域特定语言）：你直接命名硬件，但通过结构化的 IR 来实现。
- 每个分块（tiling）操作都由三个设计要素控制：*作用域*（scope，哪些线程）、*布局*（layout，数据存放在何处）和*调度*（dispatch，走哪条硬件路径）。
- 一个可运行的单 MMA GEMM（通用矩阵乘法）展示了这三者；本书的其余部分就是这三个设计要素在更大规模上的应用。
:::

:::note[运行示例]

这些示例需要一块 Blackwell GPU（`sm_100a`，例如 B200）。TIRx 编译器以 Apache TVM wheel 包中的 `tvm.tirx` 模块形式发布；请与 CUDA 版本的 PyTorch 一起安装：

```bash
pip install apache-tvm
```

用 `python -c "import tvm, tvm.tirx; print(tvm.__version__)"` 确认可以导入。同一套环境可以运行本书中每一个可运行的示例。
:::

第一部分讲解了硬件是什么。要让它进行计算，我们需要一种编程方法。

我们可以直接编写原始的 CUDA 或 PTX，许多快速的核函数正是这样编写的。问题在于，真正决定核函数行为的那些决策在那里很难看清：哪些线程执行某个操作、每一块数据存放在哪里、由哪条硬件路径执行它。这些选择被埋藏在内置函数参数、地址算术和约定之中。

TIRx（Tensor IR neXt）是一个 Python DSL，它将这三个决策提升到明处：**作用域**（scope，哪些线程执行操作）、**布局**（layout，操作数分块存放在哪里）和**调度**（dispatch，哪条硬件路径执行它）。它仍然直接命名硬件概念，包括线程、共享内存（shared memory）和张量内存、屏障（barrier）以及 `tcgen05` MMA。区别在于，这些选择现在变成了编译器可以降低（lowering）、检查和调度（schedule）的结构化 IR。

与其抽象地介绍这些概念，不如我们从一个完整的核函数入手：一个最小的单 MMA GEMM。我们先让它跑起来，然后再逐行回读，看看作用域、布局和调度各自如何塑造它，以及核函数是如何被编译的。核函数所依赖的张量布局模型在 [TIRx layout API](/books/modern-gpu-programming-for-mlsys/tirx-layout-api/) 中单独展开，完整的语言特性集在 [TIRx language reference](/books/modern-gpu-programming-for-mlsys/appendix/) 中介绍；这里我们只聚焦于这一个核函数和三个设计要素。

## 第一个核函数：单 MMA GEMM

我们承诺的核函数是一个最小化的 GEMM，精简到仍能驱动一个 Tensor Core 的最小版本。它计算 `D = A B^T` 的单个 128 x 128 输出分块，K = 64。整个计算从头到尾用一个 `Tx.gemm_async` 分块操作表达。（这一个分块操作并不映射到单条硬件指令：因为硬件 MMA 的 K 原子（K-atom）为 16，K=64 的分块会降低为一小段沿 K 步进的 `tcgen05.mma` 指令序列。DSL 的意义恰恰在于我们写的是分块，而不是序列。）围绕这个操作，核函数做一些常规杂务：它分配共享内存（SMEM）和张量内存（TMEM），将 A 和 B 从全局内存拷贝到共享内存，向 TMEM 累加器发出分块 MMA，通过寄存器（register）把累加器读回，并存储结果。虽然很小，这个核函数是我们在 [GEMM basics](/books/modern-gpu-programming-for-mlsys/gemm-basics/) 中攀登的 GEMM 阶梯的第 1 步，在那里它将以完整的走查回归。

每个 TIRx 核函数都从同样的少数几个导入开始，所以值得一开始就看一次：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

我们把核函数包在一个小构建器 `hgemm_v1(M, N, K)` 里，它接受问题形状并返回一个 `PrimFunc`。对于我们所选的形状 `M=N=128, K=64`，启动恰好包含一个输出分块，这正是让这第一个版本简单到可以一口气读完的原因：

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

在阅读核函数之前，我们先确保它能正常工作。我们编译它并对照 torch 参考实现检查输出。我们不必显式指定架构：架构（如 `sm_100a`）会从设备自动检测，所以目标 `"cuda"` 就够了，而 `tir_pipeline="tirx"` 正是选择 TIRx 降低流水线（pipeline）的开关。编译完成后，`ex.mod(...)` 直接接受 torch 张量，中间无需任何手动转换。

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
```

## 作用域、布局、调度

现在核函数可以运行了，我们可以回读它，问一问它的每一行究竟决定了什么。这样看，整个核函数就是沿着三个设计要素的一组选择。其中的每个操作都回答同样的三个问题——*谁*运行它、它的数据*在何处*、它*如何*执行——而这三个答案正是作用域、布局和调度。本节余下部分逐一讨论这些设计要素；下方的交互式演示让你看到每个设计要素控制的是哪些行。

<div style="overflow-x:auto;">
<iframe src="/books/modern-gpu-programming-for-mlsys/demo/tirx_dispatch.html?notitle" title="TIRx: scope, layout, dispatch" loading="lazy"
        style="width:960px; max-width:none; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;
"></iframe>
</div>

*交互：点击 Scope / Layout / Dispatch，高亮显示每个设计要素所控制的核函数代码行。*

使用演示时，留意三个问题：

- **作用域：谁运行这个操作？** `Tx.cta.copy(...)` 是 CTA（协作线程数组）作用域的，所以全部 128 个线程都参与 GMEM -> SMEM 的拷贝。`Tx.gemm_async(...)` 由一个被选中的线程发出一次，因为每条降低后的 `tcgen05.mma` 指令本身已经是一次协作式 MMA 启动。`Tx.wg.copy_async(...)` 是线程束组（warpgroup）作用域的，所以线程束组的 128 个线程逐行分担 TMEM 的回读。
- **布局：每个分块存放在哪里？** A 和 B 使用 `tcgen05.mma` 所期望的交换（swizzle）SMEM 布局。累加器以 `TLane`/`TCol` 布局存放在 TMEM 中。寄存器回读视图把行映射到 `tid_in_wg`，所以每个线程束组线程拥有一个行片段。
- **调度：哪条硬件路径执行它？** `Tx.gemm_async(..., dispatch="tcgen05", ...)` 选择 Blackwell Tensor Core 路径。拷贝操作也有调度选择：这第一个核函数使用普通线程拷贝，后续的 GEMM 步骤会在不改变周围作用域或布局的情况下把这些拷贝换成 TMA（张量内存加速器）。

**与你的 agent 一起尝试**：从第一个核函数中挑出三行：一个拷贝、一个 MMA、一个 TMEM 回读。让它按作用域、布局、调度为每行贴标签，然后检查答案是否与代码中的守卫、缓冲区（buffer）布局和 `dispatch=` 参数匹配。

## 编译是如何工作的

我们上面已经编译过核函数来测试它；现在我们仔细看看这一步做了什么。配方很短：把 `PrimFunc` 包进一个 `IRModule`，交给 `tvm.compile(mod, target=..., tir_pipeline="tirx")`。这会运行 TIRx 降低流水线，并返回一个你可以直接调用的 `Executable`。

```python
target = tvm.target.Target("cuda")
ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
```

至少大致了解 `tir_pipeline="tirx"` 启动了什么是有价值的。流水线的核心 pass `LowerTIRx` 会针对每个分块原语的作用域/布局/调度契约进行解析：我们刚才讨论的三个设计要素正是在这里被实际兑现为指令的。此后，通常的 host/设备分割和一个 finalize 步骤生成可启动的模块。如果你愿意，也可以在 `with target:` 块内编译，这让核函数可以拾取周围的目标上下文。

这个流程的一个好处是对你没有任何隐藏：结果可以在两个层级上被检查。你可以用 `.show()` 或 `.script()` 阅读 IR 本身，也可以直接从编译后的模块读到编译器最终生成的 CUDA C。

```python
kernel.show()                          # pretty-print the TIRx (TVMScript)
print(kernel.script())                 # ... the same, as a string

print(ex.mod.imports[0].inspect_source())
```

这只是一个概述。关于完整的降低故事——涵盖所有 pass、分块原语调度如何解析、host/设备分割如何完成——请见 [compiler internals](/books/modern-gpu-programming-for-mlsys/appendix/)。

## 接下来去哪里

一个核函数已足以认识作用域、布局和调度，并看到它们被编译和运行。三个设计要素中的每一个，以及这个核函数本身，都通向一个将其进一步展开的章节：

- [TIRx layout API](/books/modern-gpu-programming-for-mlsys/tirx-layout-api/)：张量布局模型（`TileLayout`、命名轴、swizzle），上面的操作数和累加器放置就是由它构建的。如果三个设计要素中布局让你觉得最神秘，从这里开始。
- [TIRx language reference](/books/modern-gpu-programming-for-mlsys/appendix/)：完整的语言特性集，涵盖解析器工具、数据类型、缓冲区和内存、控制流和线程同步，适合当你想要完整词汇表而非导览时。
- [GEMM basics](/books/modern-gpu-programming-for-mlsys/gemm-basics/)：这个核函数作为 GEMM 优化路径的第 1 步，通过 K 循环（loop）累加、空间分块、TMA 和线程束特化（warp specialization）逐步构建。如果你想看到同样的三个设计要素如何扩展到一个真实核函数，这是自然的下一站。

