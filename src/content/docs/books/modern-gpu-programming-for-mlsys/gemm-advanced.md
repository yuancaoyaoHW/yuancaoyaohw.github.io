---
title: 用线程束特化和集群扩展 GEMM
sidebar:
  order: 140
---

:::note[概述]

- 流水线化的 GEMM 仍然由一个线程束组顺序执行加载、MMA 和写回，本章移除这个瓶颈。
- 第 7 步将线程束特化为不同角色，第 8 步加入 2-CTA 集群，第 9 步加入多个消费者。
- 每一步移除一个串行瓶颈，最终接近 SOTA（state-of-the-art）吞吐量。
:::

上一章（[GEMM async](/books/modern-gpu-programming-for-mlsys/gemm-async/)）的流水线化 GEMM 很快，但它仍要求一个线程束组包揽一切：发出加载、运行 MMA、再写回结果。即便有了软件流水线，这一队线程仍然成为三个引擎交汇的地方。

症状显而易见。Tensor Core 运行时 TMA 单元安静下来，结果排空到内存时 Tensor Core 安静下来，每个引擎都通过同一组线程等待另一个。越过这一点的方法是停止让一队人马包揽一切。

我们在三个不断拓宽合作的步骤中推进这个想法。第 7 步（[warp specialization](/books/modern-gpu-programming-for-mlsys/gemm-advanced/)）将线程束特化为生产者、消费者和写回角色。第 8 步（[CTA cluster](/books/modern-gpu-programming-for-mlsys/gemm-advanced/)）把两个 CTA 连成一个集群，跨它们的共享内存共享操作数。第 9 步（[multi-consumer GEMM](/books/modern-gpu-programming-for-mlsys/gemm-advanced/)）加入第二个 MMA 消费者，使一块暂存的分块喂给两倍的数学运算。

把这三步看作同一模式在不同尺度上的应用会有帮助。第 7 步把完整流水线保留在一个 CTA 内：TMA 和 MMA 共享一个线程束组，而写回在另一个中运行。第 8 步跨 CTA 拓宽合作，产生一个跨越两个 CTA 的 256×256 分块。第 9 步进一步推高计算密度：集群输出增长到 512×256，每个暂存的 B 分块被两个消费者复用，我们到达本教程中最密集的变体。

有一件事在所有这些中保持不变。SMEM、TMEM 和寄存器布局仍然遵循我们在前两章建立的契约；改变的是*谁合作*，而非数据如何布局。第 8 步是合作作用域首次超越单个 CTA，因此它的操作数分块被拆分到两个 CTA 的共享内存中，一个布局沿 `cbx` 集群轴跨越两个 CTA。


## 第 7 步：线程束特化 + 流水线

单线程束组核函数把性能留在桌上的原因很简单：每个线程走同一条路径，加载、然后计算、然后写，所以在它加载时 Tensor Core 无事可做，在它计算时 TMA 引擎无事可做。修复方法是*线程束特化*。与其让一队线程依次做每件事，不如把每件事交给一个专用线程束，让这些线程束同时运行，由软件流水线缝合在一起。这是 GEMM 路径中最大的架构变化，本章的其余部分都建立在它之上。此处的基准使用 M=N=K=4096。

> **本步骤所改变的内容：作用域**
> - 作用域：一个线程束组按顺序走加载 → MMA → 写回，变为三个并发角色（TMA 生产者、MMA 消费者、写回），由满/空屏障连接。
> - 布局：不变，与第 6 步相同的 SMEM 阶段和 TMEM 累加器。
> - 调度：不变，TMA 加载，`tcgen05` MMA。

**主题。**

- 线程束特化：将不同线程束/线程束组专用于不同任务

- 高层屏障抽象：`TMABar`、`TCGen05Bar`、`MBarrier`

- `PipelineState` 用于自动的阶段/相位管理

- `warpgroup_sync` 屏障 ID 用于每线程束组的同步

（多阶段 SMEM 流水线和持久化的 `ClusterPersistentScheduler2D` 从第 5-6 步原样复用；这里只有作用域拆分是新的。）

### 从顺序到并发

在引入角色和屏障之前，先隔离出线程束特化所移除的调度瓶颈会有帮助。下图用第 4 步风格的顺序时间线作为第 4-6 步未特化核函数的紧凑参考，然后把它放在第 7 步线程束特化调度之上，使引擎利用率差异一目了然。

![Warp Specialization Timeline](/books/modern-gpu-programming-for-mlsys/img/warp_specialization_timeline.png)

上方是未特化的单线程束组模式：同一未特化线程组既拥有加载路径又拥有 MMA 路径，所以一个引擎很容易在另一个活跃时空闲。第 5 步和第 6 步用双缓冲和持久化调度改进了基线，但它们尚未把加载和计算拆分为独立的生产者和消费者角色。下方，特化打破了这种轮流。TMA 生产者在 MMA 消费者忙于计算时预取下一个分块，写回自行进行。生产者线程束 3 在消费者线程束 0 仍在处理当前 MMA 时发出下一次加载，所以两个引擎都不必等待另一个。加载/MMA 交接使用两个屏障：

- **`tma2mma`**（TMA → MMA）：表示加载的 SMEM 数据已就绪可供 MMA 消费。
- **`mma2tma`**（MMA → TMA）：表示 MMA 已读完一个缓冲区，TMA 可为下一次加载复用它。

图中一个细节乍看像错误：`mma2tma` 箭头跳前了一个阶段。原因是环形缓冲区。`PIPE_DEPTH=2` 有两个 SMEM 缓冲区，阶段 0 和阶段 1；TMA Load k=0 填充缓冲区 0，TMA Load k=1 填充缓冲区 1。当 MMA Compute k=0 读完缓冲区 0 时，它发出 `mma2tma` 表示缓冲区空闲，但真正想要回缓冲区 0 的加载是 TMA Load k=2，而非 k=1（后者用的是缓冲区 1）。这就是为什么 MMA Compute k=0 的 `mma2tma` 箭头一直延伸到 TMA Load k=2。释放跳过一个阶段，仅仅是因为环有两个槽位。

### 线程束角色

时间线展示了*为什么*我们要拆分工作；下一个问题是*谁*做每一部分。特化把三项工作（加载、计算、写回）分配给特定线程束，使它们能同时运行。当 `WG_NUMBER=2` 时，核函数使用两个线程束组（角色表中简写为 WG）：

| 角色 | 位置 | 工作 |
|-------|----------|-----|
| **TMA 生产者** | 线程束组 1，线程束 3 | 持续通过 TMA 加载 A 和 B 分块 |
| **MMA 消费者** | 线程束组 1，线程束 0 | 数据一就绪就运行 MMA |
| **写回** | 线程束组 0（全部线程束） | 读取 TMEM 结果，写入 GMEM |

### 4 个屏障

三个并发角色需要四个屏障，这四个整齐地分成两个相反方向。前向路径（TMA → MMA → 写回）表示数据*就绪*；它的消息是"你等待的分块到了"。反向路径（写回 → MMA → TMA）表示缓冲区*释放*："你想要的槽位又空了"。一旦你知道命名约定，名字就自解释了：每个都是 `source2destination`，所以 `tma2mma` 就是 TMA 用来通知 MMA 的屏障。

| 屏障 | 类型 | 方向 | 含义 |
|---------|------|-----------|---------|
| **tma2mma** | `TMABar` | TMA -> MMA | "SMEM 数据已就绪" |
| **mma2tma** | `TCGen05Bar` | MMA -> TMA | "SMEM 缓冲区可复用" |
| **mma2ld** | `TCGen05Bar` | MMA -> 写回 | "TMEM 结果已就绪" |
| **ld2mma** | `MBarrier` | 写回 -> MMA | "TMEM 对下一个分块空闲" |

为什么每个屏障都有它那样的*类型*？类型取决于生产者如何宣告它完成了。**TMA 加载**使用 `TMABar`，一种带字节计数的 mbarrier：TMA 硬件本身在搬运的字节落地后到达屏障，因此消费者无需任何线程轮询就知道数据就绪。**TMA 存储**无法使用它（存储没有可通知的对象），所以退回到 `cp_async.bulk.commit_group()` + `wait_group(0)`，发出线程只等待自己的写排空。**MMA 操作**使用 `TCGen05Bar`，`tcgen05.commit()` 指令在 MMA 完成时向它发出信号。

这里一个小细节会在第 8 步派上用场。`arrive` 调用传入 `cta_mask=0`，因为在单 CTA 核函数中没有其他 CTA 可通知。当第 8 步形成集群时，正是这个参数变成非零，成为唤醒合作 CTA 的机制。

### PipelineState

四个屏障告诉角色一个缓冲区*何时*就绪；仍然需要某样东西来跟踪流水线循环时每个角色在*哪个*缓冲区上。这种簿记正是 `PipelineState` 管理的。一个环形缓冲区同时携带两份簿记：我们当前在哪个槽位，以及我们在等待该槽位屏障的哪个"相位"。在流水线循环中手工跟踪两者正是滋生差一错误的那种事，而这里的差一会让整个核函数死锁。`PipelineState` 的存在就是把两者放在一起，这样你就不必自己跟踪：

```python
tma_ps = PipelineState(PIPE_DEPTH, phase=1)   # Producer starts ready (phase=1)
tma_ps.advance()                          # Advance to next stage
```

初始 `phase` 决定一个角色的第一次 `wait` 是让它运行还是让它阻塞，而在流水线两端正确答案恰好相反，这正是让人栽跟头的部分：
- `phase=1`（生产者）-> 第一次 `wait(phase=1)` 看到屏障仍在相位 0，因为 0 != 1 它**立即通过**。这正是我们想要的，因为缓冲区开始时为空，生产者应当能自由地立即开始填充它们。

- `phase=0`（消费者）-> 第一次 `wait(phase=0)` 看到屏障在相位 0，因为 0 == 0 它**阻塞**。同样是我们想要的，因为此时还没有数据，消费者在生产者到达之前无可读取。

给两端相同的起始相位会得到死锁，或更糟的静默损坏，所以这一个选择值得做对。

### `warpgroup_sync` 屏障 ID

特化引入了一个容易走入的同步隐患。一旦每个线程束组运行不同的代码路径，熟悉的 `cta_sync()` 就会死锁：它使用硬件屏障 #0 并坚持*每个* CTA 线程都到达，然而在线程束组分支内部只有其中一部分线程在场。我们需要的相反是一个作用域限于单个线程束组的屏障。GPU 给我们 16 个命名屏障（ID 0–15），所以核函数求助于 `warpgroup_sync(10)`，它只同步一个线程束组内部的线程。当多个线程束组各自需要在自己的组内同步时（如多消费者第 9 步中那样），它们通过 `warpgroup_sync(wg_id + 10)` 取得不同的 ID，从而永远不会在同一硬件屏障上相撞。

**实现。**

我们这里用 `PIPE_DEPTH=2`，这是能让加载和计算有哪怕一点重叠的最小深度。更深的深度隐藏更多内存延迟（latency），上限是 SMEM 预算；下面*当第 7 步出问题时*的讨论详细分析了那个权衡。有了现在手头的所有零件（角色、四个屏障、`PipelineState` 和线程束组作用域同步），我们可以组装完整核函数：

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.pipeline import TMABar, TCGen05Bar, MBarrier, PipelineState
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D

SM_COUNT = 148  # Number of SMs on NVIDIA B200 GPU
F16_SIZE = 2

def hgemm_v7(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    WG_NUMBER = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, 1)
        ld2mma  = MBarrier(pool, 1)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)

        # --- Barrier init ---
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128)   # all 128 Warpgroup 0 threads arrive
        pool.commit()

        # --- TMEM alloc + fence ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // BLK_M, num_n_tiles=N // BLK_N,
            l2_group_size=8, num_clusters=SM_COUNT)
        tile_scheduler.init(bx)
        m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
        n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

        # =============================================
        # Warpgroup 1: TMA Producer (warp 3) + MMA Consumer (warp 0)
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                # === TMA Producer ===
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=1,
                                  mbar=tma2mma.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=1,
                                  mbar=tma2mma.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            tma2mma.arrive(tma_ps.stage,
                                           (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                # === MMA Consumer ===
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        # Wait for TMEM to be free from previous tile's writeback
                        ld2mma.wait(ld_ps.stage, ld_ps.phase)
                        ld_ps.advance()

                        for k in range(K_TILES):
                            tma2mma.wait(mma_ps.stage, mma_ps.phase)
                            Tx.gemm_async(
                                tmem[:, :BLK_N],
                                Asmem[mma_ps.stage, :, :],
                                Bsmem[mma_ps.stage, :, :],
                                accum=(k != 0), dispatch="tcgen05", cta_group=1)
                            mma2tma.arrive(mma_ps.stage, cta_group=1, cta_mask=0)
                            mma_ps.advance()

                        # Signal results ready for writeback
                        mma2ld.arrive(0, cta_group=1, cta_mask=0)
                        tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0: Writeback
        # =============================================
        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((BLK_N,), d_type)

            while tile_scheduler.valid():
                # Wait for MMA results
                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()

                # Read TMEM -> registers (warpgroup scope)
                reg = T.alloc_local((BLK_N,), acc_type)
                reg_wg = reg.view(128, BLK_N,
                    layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
                Tx.wg.copy_async(reg_wg[:], tmem[:, :BLK_N])
                T.ptx.tcgen05.wait.ld()

                # Signal TMEM free (all 128 threads arrive)
                ld2mma.arrive(0, cta_id=0, pred=True)

                # Cast fp32 -> fp16
                Tx.cast(reg_f16[:], reg[:])

                # Write to Dsmem + TMA store
                Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                T.ptx.fence.proxy_async("shared::cta")
                T.cuda.warpgroup_sync(10)
                if warp_id == 0:
                    if lane_id == 0:
                        Tx.copy_async(D[m_st:m_st+BLK_M, n_st:n_st+BLK_N],
                                      Dsmem[:, :], dispatch="tma")
                        T.ptx.cp_async.bulk.commit_group()
                        T.ptx.cp_async.bulk.wait_group(0)
                T.cuda.warpgroup_sync(10)

                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

要运行这些核函数中的任何一个，复用我们在第 1 步（[GEMM basics](/books/modern-gpu-programming-for-mlsys/gemm-basics/)）展示过一次的编译/运行/检查脚手架：把 `hgemm_v1` 换成 `hgemm_v7`、`hgemm_v8` 或 `hgemm_v9`，并选择一个问题规模如 `M=N=K=4096`。记住集群化步骤需要 `M` 和 `N` 是其集群分块（第 8 步 `256×256`，第 9 步 `512×256`）的倍数，所以一个微小的 `128×128` 规模根本产生不了分块。每个全新的 Python 会话只编译一个步骤，切换步骤前重启内核，因为核函数复用内部名称而编译器持有每会话状态。每步的耗时收集在下文的*端到端结果*中。

### 收尾（写回）细节

第 7 步能负担一个令人愉悦的简单收尾。只有 `BLK_N=128` 列时，写回线程束组一次性把整个 TMEM 分块读入寄存器，然后发出一次 TMA 存储。第 8 步和第 9 步就没这个奢侈了，这也正是它们引入我们稍后添加的分块读写的原因，但目前序列是：

1. 等待 MMA：`mma2ld.wait(phase)`。本教程的第 8 步和第 9 步在此添加 `fence.after_thread_sync()` 作为保守的额外措施；MMA 完成的 mbarrier 已经覆盖了排序，大多数核函数（包括 CUTLASS）都省略它，所以第 7 步也一样。
2. 读 TMEM -> 寄存器（每线程 128 个 fp32，通过 `Tx.copy_async(reg_wg, tmem[:, :BLK_N])` 加 `T.ptx.tcgen05.wait.ld()` 实现的线程束组作用域）。
3. 通知 MMA：`ld2mma.arrive(0, cta_id=0, pred=True)`（全部 128 个线程到达）；TMEM 现在对下一个分块空闲。这两个 `arrive` 关键字参数在集群化步骤中重现：`cta_id` 命名通知*哪个 CTA 的*屏障副本（`0` = 本 CTA，本地屏障；第 8 步中合作的到达通过 `cta_mask` 改为指向 CTA-0），而 `pred` 是一个每线程谓词，门控该线程是否真正到达（此处为 `True`，所以每个写回线程都计入到达总数）。
4. 在寄存器中将 fp32 转为 fp16。
5. 写寄存器 -> Dsmem，然后 `fence.proxy_async("shared::cta") + warpgroup_sync(10)` 刷新。
6. 通过 `cp_async.bulk.commit_group() + wait_group(0)` 将 Dsmem 经 TMA 存到 GMEM。

第 8 步（`BLK_N=256`）和第 9 步（每个消费者 `MMA_N=256`）无法保持这种一次性形式，原因是寄存器压力。每线程读取 256 个 fp32 值意味着 256 × 4 = 1024 字节必须同时存在于每个线程的寄存器中，这有溢出到本地内存的风险，此外还迫使 Dsmem 缓冲区更大。所以这些步骤把写回拆成 `EPI_N` 列的块（`EPI_N=64`）：每次迭代只让 `EPI_N` 个 fp32 寄存器存活，并发出相应更小的 TMA 存储，用几条额外的存储指令换取一个保持舒适的寄存器预算。

**实现说明。**

- **持久化核函数**：`bx = T.cta_id([SM_COUNT])` --- 每个 SM 一个 CTA，循环处理分块

- **L2 友好的调度**：`ClusterPersistentScheduler2D` 为缓存局部性对分块排序

- 这种模式 --- 线程束特化加软件流水线 --- 在高性能 GEMM 核函数中很常见，包括 CUTLASS 式设计。

### 当第 7 步出问题时

第 7 步是第一个 TMA 加载、`tcgen05` MMA 和写回同时处于在途状态的 GEMM 核函数。同样的失败模式在第 8 步和第 9 步中重现：屏障计数不匹配、角色守卫放错位置、缺少屏障（fence），或暂存缓冲区在 TMA 存储排空之前被复用。这些情况的调试清单收集在 [debugging warp specialization](/books/modern-gpu-programming-for-mlsys/debugging-warp-specialized/) 中。

**流水线深度调优。** 第 7 步核函数在 `PIPE_DEPTH=2` 下运行，即最小值。把它推到 4 或 6 让 TMA 生产者能更远地跑在 MMA 消费者之前，隐藏更多内存延迟，但代价是消耗更多 SMEM，而 SMEM 是有限的。B200 每个 SM 提供 228 KB（见 [GPU execution model](/books/modern-gpu-programming-for-mlsys/gpu-execution-model/) 的*需要记住的数字*）。在 `BLK_M=BLK_N=128, BLK_K=64, fp16` 下，每个流水线阶段为 A 和 B 合计花费 `(128*64 + 128*64) * 2 = 32 KB`，`Dsmem` 写回暂存缓冲区再加 32 KB。这让 `PIPE_DEPTH=4` 大约 160 KB、`PIPE_DEPTH=6` 大约 224 KB，直逼预算。要想比这更深，就得重新思考写回暂存策略。

---

线程束特化让一个 CTA 的线程合作起来。下一步把这种合作拓宽到 CTA 自身的边界之外，让两个 CTA 在一个更大的分块上协作。


## 第 8 步：2-CTA 集群

第 7 步让引擎重叠起来，但每个 CTA 仍然孤立地计算自己的 128×128 分块，重新加载没有邻居能借用的操作数。第 8 步打破这种孤立。两个 CTA 连成一个集群，获得伸入彼此共享内存的能力，于是单次协作的 `tcgen05` MMA 产生一个跨越两者的 256×256 分块，而一次 B 的加载现在喂给两倍的 MMA 工作。与之前一样，M=N=K=4096。

> **本步骤所改变的内容：作用域 + 布局 + 调度**
> - 作用域：合作作用域现在跨越集群中的两个 CTA，而非一个。
> - 布局：操作数分块被拆分到两个 CTA 的 SMEM；CTA 0 拥有共享的完成屏障（`remote_view`）。
> - 调度：MMA 增加 `cta_group` / `cta_mask`，使 `tcgen05` 作为 2-CTA 协作操作运行。

**主题。**

- CTA 集群：多个 CTA 在一个更大的分块上协作

- 通过 `map_shared_rank` 跨 CTA 访问 SMEM

- `cta_group=2` 用于 256x256 集群分块上的协作 MMA

- 用 `cta_mask` 跨 CTA 发出屏障信号


### 集群分块形状

整个优化建立在一项硬件能力之上：当 `cta_group=2` 时，MMA 被允许读取由*两个* CTA 暂存的操作数分块，而不仅是它所在的那个。每个 CTA 加载一片 128 行的存储 B，转置后它成为 128 个逻辑输出列，而协作 MMA 把两片缝合回一个操作数。下图追踪两个 CTA 的 A 和 B 切片如何组合成单个 256×256 集群分块：

<div style="overflow-x:auto;">
<iframe src="/books/modern-gpu-programming-for-mlsys/demo/cta_cluster.html" title="A 2-CTA cluster: cooperative MMA via cross-CTA SMEM read" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>

*交互式：每个 CTA 拥有一片 A 行切片和一片存储 B 行切片，然后通过集群（DSMEM）读取另一个 CTA 的存储 B 切片。经过 `B.T` 后，两片存储 B 切片覆盖完整的输出列范围，因此这对 CTA 产生一个 256×256 输出分块。*

**为什么 A 和 B 被拆分到集群中**：要看出 256×256 分块如何划分，回想本教程把 GEMM 存储为 `D = A @ B.T`，其中存储 B 的形状为 `N x K`。集群中有两个 CTA 时，拆分自然落下：

- **A 被纵向拆分**：CTA-0 持有 A0（第 0-127 行），CTA-1 持有 A1（第 128-255 行）。堆叠：`[A0; A1]`（256 行）。
- **存储 B 按行拆分**：CTA-0 加载 B 的第 0-127 行，CTA-1 加载第 128-255 行。因为数学用的是 `B.T`，这两片存储的行切片成为逻辑右操作数的两片 128 列切片。
- 当 `cta_group=2` 时，MMA 硬件通过跨 CTA 共享内存访问从**两个** CTA 的 SMEM 读取 B，因此它看到完整的逻辑输出列范围。
- 结果：两个 CTA 在一个 256x256 输出分块上协作。每个 CTA 写该分块的一条 128x256 行带。

值得停下来看看为什么这是一次真正的胜利而非只是工作的重排。每个 CTA 仍然只加载 128×K 的 A 和 128×K 的 B，所以集群整体暂存约 2× 单个 CTA 的操作数，然而它产生一个 256×256 分块，承载约 4× 于 128×128 分块的输出 FLOPs。因此 MMA 每暂存操作数字节做约两倍的工作，因为每个 CTA 的 B 切片通过协作 MMA 被复用到另一个 CTA 的 A 切片上。换言之，算术强度大致翻倍，而这正是一个仍偏向内存的核函数所需要的杠杆：端到端表中约 2.2× 的加速来自把同样的字节喂给更多数学运算。

### 分块地址计算

现在集群是工作单元，分块调度器也必须以集群分块计数。它交回的每个 `(m_idx, n_idx)` 命名一个完整的 256×256 区域，集群内的两个 CTA 在它们之间划分该区域。把一个集群坐标翻译成每个 CTA 实际加载的每 CTA 切片看起来是这样：

```python
m_st = (m_idx * CTA_GROUP + cbx) * BLK_M
n_st = (n_idx * CTA_GROUP + cbx) * BLK_N
```

两个 CTA 在*同一个* 256×256 集群分块上工作，而单一坐标 `cbx`（CTA 在集群内的位置，0 或 1）正是挑出本 CTA 沿两条轴贡献的那个。`m_st` 选定本 CTA 拥有的输出行带，`n_st` 选定它喂给协作 MMA 的存储 B 切片，写回随后发出 256 列输出范围的两个 128 列半段。还要注意 `num_m_tiles = M // 256` 和 `num_n_tiles = N // 256` 计的是集群分块，而非单个 CTA 分块。

乍看 `cbx` 同时出现在 `m_st` 和 `n_st` 中，好像一个行偏移不知怎么漏进了列里，但两处用法都正确，值得理清为什么。在写回路径上，`cbx` 只属于 M 轴：每个 CTA 拥有一条不同的 128 行带（`m_st = (m_idx * CTA_GROUP + cbx) * BLK_M`，所以 CTA-0 写行 `m_idx*256 .. +128`，CTA-1 写接下来的 128 行），然而两个 CTA 都写集群分块的*完整* 256 输出列。这正是为什么存储从集群的 `n_idx` 推导其列（`n_st_epi = n_idx * 256 + no * 128`，看不到 `cbx`），而非从每 CTA 的 `n_st`。`n_st` 之所以带 `cbx`，是因为每个 CTA 向 MMA 加载不同的存储 B 行切片：在那里 `cbx` 是*加载*偏移，而非 CTA 的输出列偏移。

### 相对第 7 步的代码变化

相对第 7 步的 diff 有六处编辑，每一处编码我们刚描述的集群契约的一条：

```python
cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])   # cbx = CTA index within cluster (0 or 1)

Tx.gemm_async(..., cta_group=2)

B_remote = T.ptx.map_shared_rank(Bsmem, cta_id=1)

tma2mma_cta0 = T.decl_buffer(
    [CTA_GROUP], "uint64",
    data=T.ptx.map_shared_rank(tma2mma.ptr_to([0]), 0),
    scope="shared"
)

mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)

T.cuda.cluster_sync()
```


### 集群作用域的变化

这六处编辑都源于同一个转变：合作作用域现在是集群而非单个 CTA。下面的要点说明了这种拓宽在实践中意味着什么：每个 CTA 如何找到自己的位置、集群在谁的屏障上协调、以及哪个 CTA 真正发出协作 MMA。

- **集群 CTA ID**：`cbx` 告诉每个 CTA 它在集群中的位置（0 或 1）。CTA-0 处理 A 的第 0-127 行，CTA-1 处理第 128-255 行。

- **远程屏障视图**：在集群中，每个 CTA 有自己的 SMEM 和自己的屏障，这引出一个明显问题：如果 CTA-1 需要等待 CTA-0 产出的某样东西，它实际触碰的是谁的屏障？答案是提名 CTA-0 的屏障作为单一协调点，让集群中任何 CTA 都能触及它。`map_shared_rank(tma2mma.ptr_to([0]), 0)` 返回一个指向 CTA-0 屏障的集群范围指针，配以 TIRx 包装器 `tma2mma.remote_view(0)`，此后每次到达和等待都指向 CTA-0 的副本。

- **仅从 CTA-0 发出 MMA**：把 `cta_group=2` 读作并行开动两个引擎很诱人，但实际并非如此。CTA-0 恰好发出一次 `tcgen05.mma`，硬件随后驱动一次*单一协作* MMA，跨越两个 CTA，从两个 SM 的 SMEM 读取操作数，把累加器写到两个 SM 的 TMEM。CTA-1 根本不发出 MMA。（每个 SM 只有一个 `tcgen05` 引擎，所以 `cta_group=2` 是一次跨 SM MMA，而非两个引擎并排运行。）这就是为什么代码用 `if cbx == 0:` 守卫 MMA。

- **多播到达**：`tcgen05.commit(..., cta_group=2, cta_mask=3)` 仅由 CTA-0 发出，但通知两个 CTA 的屏障。`cta_mask=3`（二进制 `11`）意味着 CTA-0 和 CTA-1 都被命中。

- **ld2mma 初始计数**：`init(128 * CTA_GROUP)` --- 两个 CTA 的写回线程束组（各 128 线程）都到达。


**实现。**

```python
def hgemm_v8(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_M, MMA_N = 256, 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    WG_NUMBER = 2
    F16_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, 128))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, 1)
        ld2mma  = MBarrier(pool, 1)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, 128), d_type, layout=D_layout)

        # --- Barrier init ---
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128 * CTA_GROUP)  # both CTAs' writeback threads
        pool.commit()

        # --- TMEM alloc (cooperative) ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler (cluster tiles) ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // 256, num_n_tiles=N // 256,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
        tile_scheduler.init(bx // CTA_GROUP)
        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
        n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

        # --- Cross-CTA barrier view ---
        tma2mma_cta0 = tma2mma.remote_view(0)

        # =============================================
        # Warpgroup 1: TMA Producer (warp 3) + MMA Consumer (warp 0)
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_ps.stage,
                                    CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(ld_ps.stage, ld_ps.phase)
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.gemm_async(
                                    tmem[:, :MMA_N],
                                    Asmem[mma_ps.stage, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_ps.advance()

                            mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)
                            tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0: Writeback (256 columns in 2 x 128-column chunks)
        # =============================================
        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((128,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                for no in T.unroll(2):  # 2 chunks of 128 columns = 256 total
                    reg = T.alloc_local((128,), acc_type)
                    reg_wg = reg.view(128, 128,
                        layout=TileLayout(S[(128, 128) : (1@tid_in_wg, 1)]))
                    Tx.wg.copy_async(reg_wg[:], tmem[:, no * 128:(no + 1) * 128])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(10)
                    if warp_id == 0:
                        if lane_id == 0:
                            n_st_epi = T.meta_var(n_idx * 256 + no * 128)
                            Tx.copy_async(D[m_st:m_st+BLK_M, n_st_epi:n_st_epi+128],
                                          Dsmem[:, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(10)

                ld2mma.arrive(0, cta_id=0, pred=True)
                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

**2 个 CTA 带来的变化。**

- `CTA_GROUP = 2`，`MMA_N = BLK_N * CTA_GROUP = 256`

- `ld2mma.init(128 * CTA_GROUP)` --- 两个 CTA 的写回线程束组都到达

- TMA 到达字节数包含两个 CTA：`CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`

- `tcgen05.alloc` 和 `tcgen05.dealloc` 必须用 `cta_group=2`

- 写回把 256 个输出列拆成两个 128 列的块 --- 一次性读取全部 256 个 TMEM 列会超出寄存器容量。第 9 步把块进一步缩小到 `EPI_N=64`

- 末尾用 `cluster_sync()` 替换 `cta_sync()`（确保所有 CTA 在 TMEM 释放前都完成）

所有这些额外的算术强度直接体现在挂钟时间上：第 8 步在 4096³ 下达到 **0.104 ms**，约为同一规模下第 1 步算法 70 ms 的 676×（见端到端表）。核函数现在偏向计算受限，而这恰好为第 9 步铺路，我们在那里加入第二个 MMA 消费者以让更多 Tensor Core 工作处于在途状态。

如果第 8 步出来比第 7 步*更慢*，罪魁祸首几乎总是某条新的集群契约略微写错。有三件事最值得先检查：TMA 到达字节数是 `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`；调度器维度是 `num_m_tiles=M//256, num_n_tiles=N//256`（对应 256×256 集群分块）；写回发出两次 TMA 存储，每个 128 列块一次，每次都在 Dsmem 复用前排空。

---

集群提升了*跨* CTA 的复用。最后一步转向内部，通过给生产者第二个 MMA 消费者来提升*每个* CTA 内部的计算密度。


## 第 9 步：多消费者线程束特化

到第 8 步 MMA 确实忙起来了，但单个消费者线程束啃完一块暂存 B 分块只能那么快，而那块 B 分块整个时间就坐在 SMEM 里，任何愿意读它的人都可以用。最后的优化利用了这一点：它加入第二个 MMA 消费者，把一个*不同*的 A 块乘以*同一块* B 分块。每个 CTA 的计算密度翻倍，集群输出从 256×256 增长到 512×256。与之前一样，M=N=K=4096。

> **本步骤所改变的内容：作用域 + 布局**
> - 作用域：一个 MMA 消费者变成两个，由 `warp_id` 选择。
> - 布局：一块暂存 B 分块被两个消费者复用；A 增加一个消费者轴。
> - 调度：不变。

**主题。**

- 多个 MMA 线程束（消费者）以获得更高吞吐量（throughput）

- 多个写回线程束组，各自有独立的屏障槽位

- 本教程中最优化的 GEMM 变体所使用的结构


### 多消费者结构

加入第二个消费者意味着核函数现在有更多不同角色要安排：两个 MMA 线程束而非一个，以及一个匹配的第二个写回线程束组来排空额外的累加器。当 `NUM_CONSUMER=2` 且 `WG_NUMBER=3` 时，核函数现在跨越三个线程束组（角色表中简写为 WG）：

| 线程束组 | 线程束 | 角色 |
|-----------|------|------|
| **WG 2** | 线程束 0 | MMA 消费者 0：`Asmem[..., 0] x B` -> TMEM 列 `[0:256]` |
| **WG 2** | 线程束 1 | MMA 消费者 1：`Asmem[..., 1] x B` -> TMEM 列 `[256:512]` |
| **WG 2** | 线程束 3 | TMA 生产者：每阶段加载 2 块 A + 1 块 B |
| **WG 0** | 全部 | 消费者 0 的写回：读取 TMEM `[0:256]` |
| **WG 1** | 全部 | 消费者 1 的写回：读取 TMEM `[256:512]` |

整个安排 hinges on 一个不对称。每个消费者把自己的 A 块乘以*同一块*暂存 B 分块，所以单次 B 加载现在喂给 2× 的 MMA 工作，B 每有用 FLOP 的加载成本实际上减半。我们共享 B 而非 A 的原因是两个消费者覆盖不同的 M 行带：它们的 A 块是真正不同的数据，而 B 对两者相同。练习 3 要你说服自己这是唯一可行的共享。

### 相对第 8 步的变化

具体来说，支持第二个消费者在几处触及核函数，而每一处变化都可追溯到一个事实：现在每阶段要喂给和排空两块 A 和两个 TMEM 范围，而 B 保持共享。下面的编辑暂存了额外一块 A、给每个消费者自己的屏障槽位，并为更高的 512×256 集群分块调整分块寻址。

- `Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ...)` --- 每阶段 2 块 A，每个消费者一块

- TMA 同时加载 `Asmem[stage, 0]` 和 `Asmem[stage, 1]`，TMA 到达字节数现在是 `CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`（多一块 A）

- MMA 线程束的 `warp_id` 选择哪块 A 和哪个 TMEM 范围

- `mma2tma.init(NUM_CONSUMER)` --- 两个消费者每阶段都通知 TMA

- `mma2ld` 和 `ld2mma` 的 `depth=NUM_CONSUMER` --- 每个消费者用自己的屏障槽位（MMA 侧用 `warp_id`，写回侧用 `wg_id`）

- 分块地址：`m_st = (m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M` --- M 方向有额外的 `NUM_CONSUMER` 因子，因为每个集群分块现在在 M 上跨越 `NUM_CONSUMER` 个消费者。分块调度器用 `num_m_tiles = M // 256 // NUM_CONSUMER`（集群分块为 512x256）

- 写回用分块的 `EPI_N`，使每次迭代在寄存器中保持更少的 TMEM 回读值存活


**实现。**

```python
def hgemm_v9(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    NUM_CONSUMER = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N = BLK_N * CTA_GROUP   # 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    WG_NUMBER = 3
    F16_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (NUM_CONSUMER, BLK_M, EPI_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, NUM_CONSUMER)   # depth=2, one slot per consumer
        ld2mma  = MBarrier(pool, NUM_CONSUMER)     # depth=2, one slot per consumer
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((NUM_CONSUMER, BLK_M, EPI_N), d_type, layout=D_layout)

        # --- Barrier init ---
        tma2mma.init(1)
        mma2tma.init(NUM_CONSUMER)  # each stage expects 2 arrivals
        mma2ld.init(1)              # each slot gets 1 arrival
        ld2mma.init(128 * CTA_GROUP)  # both CTAs' writeback threads
        pool.commit()

        # --- TMEM alloc (cooperative) ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler (512x256 cluster tiles) ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // 256 // NUM_CONSUMER, num_n_tiles=N // 256,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
        tile_scheduler.init(bx // CTA_GROUP)
        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
        n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

        tma2mma_cta0 = tma2mma.remote_view(0)

        # =============================================
        # Warpgroup 2: TMA Producer (warp 3) + 2 MMA Consumers (warp 0, 1)
        # =============================================
        if wg_id == 2:
            if warp_id == 3:
                # === TMA Producer: loads 2 A blocks + 1 B block per stage ===
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    m_st_c1 = T.meta_var(m_st + CTA_GROUP * BLK_M)
                    Tx.copy_async(Asmem[tma_ps.stage, 0, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Asmem[tma_ps.stage, 1, :, :],
                                  A[m_st_c1:m_st_c1+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_ps.stage,
                                    CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id < NUM_CONSUMER:
                # === MMA Consumer: warp_id selects A block and TMEM range ===
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(warp_id, ld_ps.phase)
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.gemm_async(
                                    tmem[:, warp_id * MMA_N:warp_id * MMA_N + MMA_N],
                                    Asmem[mma_ps.stage, warp_id, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_ps.advance()

                            mma2ld.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)
                            tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0/1: Writeback (each reads its consumer's TMEM range)
        # =============================================
        elif wg_id < NUM_CONSUMER:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((EPI_N,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(wg_id, wb_ps.phase)  # wait for THIS consumer
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                # Read TMEM in EPI_N=64 column chunks (4 iterations for 256 cols)
                for i in T.unroll(MMA_N // EPI_N):
                    reg = T.alloc_local((EPI_N,), acc_type)
                    reg_wg = reg.view(128, EPI_N,
                        layout=TileLayout(S[(128, EPI_N) : (1@tid_in_wg, 1)]))
                    col_st = T.meta_var(wg_id * MMA_N + i * EPI_N)
                    col_end = T.meta_var(wg_id * MMA_N + i * EPI_N + EPI_N)
                    Tx.wg.copy_async(reg_wg[:], tmem[:, col_st:col_end])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(Dsmem[wg_id, warp_id * 32 + lane_id, :], reg_f16[:])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if warp_id == 0:
                        if lane_id == 0:
                            m_st_epi = T.meta_var(
                                (m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M)
                            n_st_epi = T.meta_var(n_idx * MMA_N + i * EPI_N)
                            Tx.copy_async(
                                D[m_st_epi:m_st_epi+BLK_M, n_st_epi:n_st_epi+EPI_N],
                                Dsmem[wg_id, :, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(wg_id + 10)

                ld2mma.arrive(wg_id, cta_id=0, pred=True)
                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

**实现说明。**

- 在这个第 9 步设计中，`mma2ld` 和 `ld2mma` 各是一个 `depth=NUM_CONSUMER` 的共享对象，而非每消费者单独的对象。槽位 0 把 MMA 线程束 0 连到线程束组 0，槽位 1 把 MMA 线程束 1 连到线程束组 1；MMA 侧用 `warp_id` 索引，写回侧用 `wg_id`。

## 端到端结果

下表报告了从朴素基线到线程束特化集群核函数的测量里程碑，并附 cuBLAS 参考。NVIDIA B200 上的参考数字，M=N=K=4096，fp16，锁定时钟，1000 次迭代计时基准：

| 步骤 | 技术 | 时间 | 加速 |
|------|-----------|------|---------|
| 1 | 同步加载 + MMA | 70 ms | 1× |
| 2 | K 循环累加 | --- | 处理大于一个分块的 K |
| 3 | 空间分块 | 53.6 ms | ~1.3× |
| 4 | TMA 异步加载 | 0.49 ms | ~142× |
| 5 | 软件流水线 | --- | 重叠加载 + 计算 |
| 6 | 持久化核函数 | --- | L2 缓存局部性 |
| 7 | 线程束特化 | 0.23 ms | ~309× |
| 8 | 2-CTA 集群 | 0.104 ms | ~676× |
| 9 | 多消费者 | 0.094 ms | ~744× |
| --- | cuBLAS（参考） | 0.094 ms | ~744× |

这张表里的每个时间，包括 70 ms 的第 1 步基线，都是在同一个 M=N=K=4096 规模下测量的，这正是让加速链端到端可比的原因。有必要精确说明那个 70 ms 到底是什么，因为它容易被误读。它*不是* [GEMM basics](/books/modern-gpu-programming-for-mlsys/gemm-basics/) 中那个单分块第 1 步核函数在 4096³ 下运行的结果；那个核函数只计算一个 128×128 分块，只在很小规模下运行。这 70 ms 是一个朴素的全规模基线，采用同样的顺序单分块方法并把它放大到完整的 4096³ 问题。第 1-3 步在 [GEMM basics](/books/modern-gpu-programming-for-mlsys/gemm-basics/) 中以小规模（128×128 和 256³）引入以让最初的讲解简单；这里的第 1 步和第 3 步行是它们的全规模基准对应物。其余的破折号（第 2、5、6 步）标记的是为结构展示但未单独计时的步骤。

把这些数字读作一次 B200 在受控条件下的参考运行，而非排行榜条目。嵌入每步的 `{.python .input}` 基准单元是烟雾基准：它们适合发现趋势，不适合声称峰值性能。

四种技术占了几乎全部增益：

1. **TMA 异步数据搬运**：硬件拷贝引擎替换软件拷贝（第 1 步 → 第 4 步约 142×）。正确地解读这个 142× 很重要：它反映的是从一个 128×128 单分块核函数（网格 1×1）一直走到一个带 K 循环、空间分块和许多 CTA 的完整分块并行核函数，*再加上* TMA；它不是 TMA 单独的贡献。孤立 TMA 意味着比较两个只在拷贝机制上不同的全规模核函数。
2. **软件流水线 + 线程束特化**：通过给加载和计算各自专用角色来重叠它们（第 4 步 → 第 7 步约 2.2×）。
3. **CTA 集群**：2-SM 协作 MMA 改善跨 CTA 的 B 分块复用（本基准中第 7 步 → 第 8 步约 2.2×）。
4. **多消费者**：两个 MMA 线程束以获得更高计算密度（第 8 步 → 第 9 步约 10%）。

在测得的里程碑处绘制，这同样的四项贡献描绘出从同步分块核函数下降到 cuBLAS 参考的轨迹。下图展示了选定的测量点：

![GEMM Optimization Journey](/books/modern-gpu-programming-for-mlsys/img/gemm_perf.png)

注意收益随着我们沿列表下行而缩小，这有结构性原因而非任何努力的减弱。早期步骤攻的是*内存*瓶颈（TMA 替换软件拷贝、集群提升算术强度），而 70 ms 大部分时间就花在那里，所以这些步骤回报最大。到第 8 步核函数已在 cuBLAS 的 ~10% 以内（0.104 对 0.094 ms）并接近*计算受限*，这意味着已几乎没有可隐藏的内存停顿；第 9 步的多消费者重叠恢复了所剩无几的大部分。约 10% 的最终收益正是接近计算上限时所应期望的：它是一个几乎已解决问题的的边际递减回报，而非弱优化的标志。

我们在本章构建的一切（TMA 加载、`tcgen05` MMA、TMEM 回读和线程束特化屏障）都直接带入下一章。Flash Attention 复用全部，然后通过在两个 MMA 阶段之间楔入一个在线 softmax（online softmax）步骤而非简单重复单个阶段来提高难度。


## 练习

1. 如果在第 7 步中把 TMA 和 MMA 两个 `PipelineState` 的初始 `phase` 都设为 `0` 会发生什么？画出死锁场景。
2. 在第 8 步中 `cta_group=2` 时，TMA 到达字节数是 `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`。既然每个 CTA 加载自己的数据，为什么还要乘以 `CTA_GROUP`？
3. 在第 9 步中，每个消费者处理不同的 M 行但同一块 B 分块。为什么共享 B（而非 A）是正确的选择？

**与你的 agent 一起尝试**：粘贴第 7 步核函数，让它追踪一个 K 分块穿过四个屏障（`tma2mma`、`mma2tma`、`mma2ld`、`ld2mma`）的过程。对每个屏障，问谁等待、谁到达、哪个分块变得可安全读取、以及之后哪个缓冲区变得可复用。

