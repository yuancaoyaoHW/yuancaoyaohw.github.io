---
title: GPU 执行模型
sidebar:
  order: 10
---

:::note[概览]

- 核函数（kernel）在一个线程层级（thread → warp → warpgroup → CTA → cluster → grid）上运行，横跨多个不同的内存空间（寄存器、SMEM、GMEM、TMEM）。
- 计算分为 CUDA 核与 Tensor Core；TMA（张量内存加速器）等专用引擎负责为它们搬运数据。
- 核函数是一条流水线，将数据在这些内存空间中逐级暂存，并在独立的计算引擎与数据搬运引擎之间交接工作；反复出现的目标是让这些引擎同时保持忙碌。
:::

要编写快速的 GPU 程序，重要的是理解硬件本身以及代码如何在该硬件上运行。本章概述 GPU 的执行模型：执行工作的线程层级、持有并搬运数据的内存空间，以及承担主要计算的引擎与数据搬运引擎。我们先逐一介绍这些部件，然后在一条 GEMM 流水线中把它们组合起来，使数据与执行如何流经硬件一目了然。本书后续几乎每一项优化，都是在这些相同部件之间重新安排工作的某种方式。

现代 GPU 还包含许多特化的硬件单元。作为初步感受，下方交互式演示展示了 Blackwell 流式多处理器内部的主要部件，随后我们再逐一深入每个部分。你可以点击每个部分查看其细节。

<div style="overflow-x:auto;">
<div style="overflow-x:auto;">
<iframe src="/books/modern-gpu-programming-for-mlsys/demo/sm_architecture.html" title="Blackwell SM architecture" loading="lazy"
        style="width:1320px; max-width:none; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
</div>

*交互式：Blackwell SM，展示其 warp/warpgroup、共享内存、Tensor Memory，以及 Tensor Core 与 TMA 引擎。*

## 执行层级

我们从执行工作的线程讲起。GPU 并不会把它成千上万的线程呈现为一个扁平的池子。相反，它把它们组织成一个嵌套层级，之所以如此是因为协作会在多个不同尺度上同时发生。每一级的存在都是为了在某一个尺度上让协作变得廉价。下图展示了 Blackwell 上的层级；你可以点击每一级来高亮它。

<div style="overflow-x:auto;">
<iframe src="/books/modern-gpu-programming-for-mlsys/demo/thread_hierarchy.html" title="Blackwell thread hierarchy" loading="lazy"
        style="width:900px; max-width:none; height:520px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>

*交互式：点击一个层级：thread → warp → warpgroup → CTA → cluster → grid。*

- **Thread（线程）**：标量执行单元。每个线程拥有自己的程序计数器和自己的寄存器，并通过其所在 warp 内的一个 lane ID 来标识。
- **Warp（线程束）**：32 个以 SIMT（*单指令多线程*）方式执行的线程。一个 warp 的各通道一起发射同一条指令，但每个通道仍保留自己的寄存器，并且可以被单独屏蔽，这正是让单个 warp 的各通道能够走不同分支的原因。
- **Warpgroup**：四个连续的 warp，即 128 个线程。Hopper 引入了 warpgroup 作为发射 warpgroup 级 MMA（`wgmma`）的单位，而在 Blackwell 上它又承担了第二个角色：它是 Tensor Memory 访问的协作单元，128 个线程一起把一个 TMEM tile 搬入或搬出寄存器。
- **CTA**（*协作线程数组*，CUDA 也称之为线程块）：硬件调度基本单位。一个 CTA 运行在单个 SM（流式多处理器）上，并在其中拥有私有的共享内存分配。多个 CTA 可以同时驻留在同一个 SM 上，此时它们按份瓜分该 SM 的共享内存容量。
- **Cluster（集群）**：一组可能分布在不同 SM 上的协作 CTA。集群中的 CTA 可以彼此同步，并且可以读写彼此的共享内存，这一能力被称为分布式共享内存。

这些层级值得细细体味，因为与早期架构不同，Blackwell 的关键操作**并非全部由同一组线程发射**。TMA 拷贝由单个线程启动，再由硬件执行。TMEM 到寄存器的加载是 warpgroup 分布式的：四个 warp 协作，每个搬运 TMEM tile 中属于自己的那一片。`tcgen05` MMA 由一个被选中的线程提交，而集群化 MMA 则一次跨越两个 CTA。因此每个操作都有其自身的天然粒度，运行该操作的线程集合就是我们所说的该操作的**作用域（scope）**，也是本书反复回到的三个反复出现的设计要素（作用域、布局与调度）中的第一个。

## 内存空间

该层级中的线程的速度只取决于送达它们的数据，所以我们接下来转向数据存放在哪里。不存在既大又快的单一内存；物理规律迫使容量与速度之间做出权衡。因此 GPU 提供了多个内存而非一个，每个在不同的点上权衡这一取舍，而核函数的工作就是让数据在其中流转。每个空间都有自己的容量、自己的延迟，以及自己的访问权限规则。

| 内存 | 归属 | 角色 | 备注 |
|--------|-----------|------|-------|
| **全局内存（GMEM）** | 设备范围 | 持久张量存储 | 大容量 HBM，由所有 SM 共享 |
| **共享内存（SMEM）** | 每 CTA（一个 SM） | tile 暂存 | 低延迟暂存区；B200 上每 SM 最多 228 KB |
| **Tensor Memory（TMEM）** | 每 CTA | MMA 累加器存储 | Blackwell 新增；由 `tcgen05` 使用 |
| **寄存器堆（RF）** | 每线程 | 标量与每线程 tile 片段 | 速度快；存放 epilogue/临时值 |

按序读来，这些空间描绘出一条路径。本书中几乎每个核函数的数据通路都是 **GMEM → SMEM →（计算）→ 寄存器 → SMEM → GMEM**，而对于 Tensor Core 核函数，TMEM 位于该路径的中间，在运算进行时持有累加器。

在这四者中，**Tensor Memory（TMEM）** 是唯一在 Blackwell 之前的硬件上没有对应物的，其完整细节留待 [Tensor Core](/books/modern-gpu-programming-for-mlsys/tensor-cores/) 再述。不过现在就理解它的动机是值得的。早期 GPU 把大型 MMA 累加器保存在寄存器中，在那里它们要争夺一项稀缺资源。Blackwell 则把 `tcgen05` 的累加器输出写到 TMEM——一个 CTA 作用域的二维暂存区，每个 CTA 有 128 lane × 最多 512 个 32位列（该数组物理上位于 SM 上）。随后核函数必须在 epilogue 之前显式地把 TMEM 读回寄存器。这一额外步骤并非免费，其两个后果将贯穿全书反复出现。其一是 TMEM 读取是**显式且 warpgroup 分布式**的，由一个 warpgroup 的四个 warp 协作完成。其二是 TMEM 与寄存器不同，必须被**显式分配和释放**。

### 集群范围内的分布式共享内存

集群是层级中唯一其成员可跨越多个 SM 的一级，这一可达性带来了其他层级所不具备的内存能力。一个 CTA 运行在一个 SM 上，并从该 SM 的共享内存中取用数据，但单个 CTA 的 SMEM 预算是有限的，而大 tile 往往需要比单个块所能提供更多的操作数存储或更多的复用。Hopper 的回答是**线程块集群**：一组比相互独立的块协作更紧密的 CTA，它们可以一起同步、读写彼此的共享内存，这一能力被称为**分布式共享内存（DSMEM）**。Blackwell 保留了集群并加以扩展，引入了动态调度（[cluster launch control](/books/modern-gpu-programming-for-mlsys/cluster-launch-control/)）和 2-CTA 协作 MMA。

DSMEM 让一个 CTA 可以直接寻址并访问对端 CTA 的共享内存。一个线程可以指名对端 SMEM 中的某个位置，并从自己的 SMEM 直接把一个 tile 批量拷贝到对端，在字节落位后举升一个完成屏障（[mbarrier](/books/modern-gpu-programming-for-mlsys/async-barriers/)）。第三部分中的 2-CTA 集群 GEMM 正是建立在这一机制之上，利用它在两个 CTA 之间共享操作数 tile，而无需把它们绕回全局内存。

下图展示了 CTA 集群所增加的额外 DSMEM 一跳；点击某一块可以看到每个 CTA 拥有什么、以及跨 CTA 读取发生在何处。

<div style="overflow-x:auto;">
<div style="overflow-x:auto;">
<iframe src="/books/modern-gpu-programming-for-mlsys/demo/cta_cluster.html" title="A 2-CTA cluster sharing distributed shared memory" loading="lazy"
        style="width:720px; max-width:none; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
</div>

*交互式：一个 2-CTA 集群，其中每个 CTA 拥有一半 A 和一半 B，通过集群（DSMEM）读取对方的 B，二者共同产出一个 256×256 输出 tile。*

## 计算：CUDA 核与 Tensor Core

线程和它们搬运的数据必须在一个算术单元上汇合，而一个 SM 提供的不是一种而是两种截然不同的数学引擎。两者之间的分工塑造了几乎所有核函数的写法，它们扮演互补的角色。

- **CUDA 核**是通用 SIMT ALU。它们运行标量与向量指令，处理索引算术、逐元素运算、归约以及控制流，即围绕繁重矩阵工作的粘合逻辑。
- **Tensor Core** 是固定功能单元，以 *tile* 粒度执行稠密矩阵乘加，在单条指令中计算 $D = AB + C$。

这一划分之所以重要，是因为 Tensor Core 提供的算术吞吐量远高于 CUDA 核——在 FLOP/s 上约高一个数量级或更多——因此稠密线性代数（GEMM、卷积和注意力）只有在运行于 Tensor Core 上时才能达到峰值性能。获得性能因此在很大程度上就是让这些 Tensor Core 不饿着。随 GPU 世代变化的是 Tensor Core *如何*被编程以及其结果*落在哪里*。Hopper 引入了异步 warpgroup MMA（`wgmma.mma_async`）；Blackwell 的第五代 Tensor Core `tcgen05` 将其累加器放在 Tensor Memory 而非寄存器中，我们在 [Tensor Core](/books/modern-gpu-programming-for-mlsys/tensor-cores/) 中专门讨论。

集群从两个方面扩展了这些引擎，这两个方面贯穿 GEMM 各章。**2-CTA 协作 MMA** 让两个 CTA 各自贡献其 SMEM 操作数，共同组成一个更大的单次 Tensor Core MMA tile。**TMA 多播**让数据搬运引擎的一次加载即可将同一 GMEM tile 同时送达多个 CTA，从而消除各自单独加载本会产生的冗余全局流量。二者都建立在前文介绍的分布式共享内存之上。

## GEMM 数据流水线

到目前为止我们已经分别介绍了各硬件单元。要看到它们如何协同，可以以一条典型的通用矩阵乘法（GEMM）流水线为例。下方交互式演示展示了一个三级 GEMM tile 流水线所涉及的单元；点击诸如 `tma load` 的某个动作，可高亮它跨越各硬件单元所走的数据通路。

<div style="overflow-x:auto;">
<div style="overflow-x:auto;">
<iframe src="/books/modern-gpu-programming-for-mlsys/demo/pipeline_arch.html" title="Blackwell GEMM data pipeline" loading="lazy"
        style="width:1320px; max-width:none; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
</div>

*交互式：Blackwell 上的 load → MMA → epilogue 流水线；点击某个动作以追踪它跨越各硬件单元的数据通路。*

单个 GEMM tile 流经三个阶段。

1. **加载。** TMA 拷贝（[TMA](/books/modern-gpu-programming-for-mlsys/tma/)）把一个 A 或 B 操作数 tile 从 GMEM 流入 SMEM。一个线程发射该拷贝，并预先记录预期到达多少字节。随着字节落位，TMA 引擎报告其进度，只有当所有预期字节全部交付后，完成屏障才翻转。
2. **计算。** `tcgen05` MMA（[Tensor Core](/books/modern-gpu-programming-for-mlsys/tensor-cores/)）从 SMEM 中读出操作数 tile，并将乘积累加到一个 TMEM tile 中。它由一个被选中的线程发射，并在算术完成时向屏障发出信号。
3. **Epilogue。** warpgroup 把 TMEM 累加器读回寄存器，将结果转换为输出 dtype，并写入 GMEM——通常会先暂存到 SMEM 再发起一次 TMA 存储。

这样写出来三个阶段看似严格串行，但慢核函数与快核函数的全部差别就在于**重叠（overlap）**。一个朴素的核函数确实按顺序执行各步（加载、等待、计算、等待、存储），于是每个引擎在等待前一个引擎时都闲着。快的核函数则把它们流水线化：当 Tensor Core 正在计算 tile `k` 时，TMA 引擎已经在取 tile `k+1`，而 epilogue 正忙于排空 tile `k-1`，三个引擎同时保持忙碌。让三个异步引擎安全地相互交接工作，正是屏障与阶段模型（[mbarrier](/books/modern-gpu-programming-for-mlsys/async-barriers/)）的职责，而第三部分的 GEMM 阶梯就建立在它之上。

## 接下来读什么

既然已经看到了高层图景，我们可以继续深入下列主要机制的章节：

- [Tensor Core](/books/modern-gpu-programming-for-mlsys/tensor-cores/) 详细讲解 `tcgen05` 计算与 Tensor Memory。
- [TMA](/books/modern-gpu-programming-for-mlsys/tma/) 讲述基于 TMA 的异步数据搬运。
- [mbarrier](/books/modern-gpu-programming-for-mlsys/async-barriers/) 介绍协调这些引擎的 mbarrier 与阶段模型。

