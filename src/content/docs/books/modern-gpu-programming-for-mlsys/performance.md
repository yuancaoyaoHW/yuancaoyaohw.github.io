---
title: 什么让核函数变快
sidebar:
  order: 20
---

# 什么让核函数变快

> **概览**

- Roofline 模型给核函数一个性能天花板。这个天花板由内存带宽或计算吞吐量决定。
- 算术强度决定适用哪个天花板。它是每搬运一个字节所完成的有用算术工作量。
- 低算术强度意味着核函数受内存限制。主要出路是搬运更少的字节、更多地复用数据、融合操作，或使用更小的 dtype。
- 高算术强度意味着核函数可能受计算限制。那么主要任务就是让 Tensor Core 保持忙碌。
- 在现代 GPU 核函数中，主要杠杆是重叠。只要依赖图允许，TMA、Tensor Core、epilogue 与存储应当同时运行。

核函数只有相对于一个天花板才谈得上快慢。像 330 TFLOP/s 这样的数字单看可能很大，但在一个能在稠密 fp16 或 bf16 Tensor Core 工作上维持约 2 PFLOP/s 的 GPU 上，它的含义截然不同。没有天花板，就很难判断一个核函数是接近硬件极限，还是让芯片大部分仍处于空闲。

Roofline 模型给出了这个天花板。它把核函数分为两种基本活动：搬运字节和做算术。如果核函数无法足够快地搬运数据，内存带宽就成为限制。如果核函数有足够的数据复用和足够的算术工作量，计算吞吐量就成为限制。

本章的数字以 NVIDIA B200 为运行示例。沿用 {ref}`chap_background` 的约定，我们使用取整的天花板来推理：稠密 fp16 或 bf16 Tensor Core 吞吐量约 2 PFLOP/s，HBM3e 带宽约 8 TB/s。精确数值取决于具体器件、时钟、功耗限制与测量设置，因此应将它们视为数量级意义上的极限，而非数据手册常数。

## Roofline 模型

每个核函数都在搬运数据并做算术。Roofline 模型以这两条路径中较慢者对核函数设界。

计算天花板是硬件的最大算术吞吐量。对于 B200 上的 Tensor Core GEMM，相关天花板是 Tensor Core 吞吐量。对于标量或逐元素核函数，相关天花板则可能是 CUDA 核吞吐量或其他功能单元。

内存天花板是带宽乘以算术强度。如果一个核函数每搬运一个字节只做很少的算术，内存带宽就限制性能。如果它每字节做很多运算，内存就不太可能成为限制因素。

基本的 Roofline 界是：

```text
attainable FLOP/s <= min(peak FLOP/s, memory bandwidth * arithmetic intensity)

算术强度是：

```text
arithmetic intensity = useful FLOPs / bytes moved

必须指明内存层级。对于 HBM roofline，字节是 HBM 字节。对于 L2 roofline，是 L2 字节。对于 SMEM roofline，是共享内存字节。本章中默认 roofline 是 HBM roofline。

在 roofline 图上，x 轴是算术强度，单位为每字节 FLOP。y 轴是可达性能。内存屋顶是一条斜线：

```text
performance = bandwidth * arithmetic intensity

计算屋顶是一条水平线：

```text
performance = peak FLOP/s

二者在脊点相交：

```text
ridge point = peak FLOP/s / bandwidth

对于此处使用的 B200 取整数值：

```text
ridge point ≈ 2000 TFLOP/s / 8 TB/s
            ≈ 250 FLOP/byte

算术强度低于该值的核函数在 HBM roofline 下受内存限制。它无法达到峰值 Tensor Core 吞吐量，因为它无法每秒交付足够的字节来喂养那么多算术。

算术强度高于该值的核函数可能受计算限制。此时内存流量不再是首要限制。剩余的任务是把计算单元驱动得足够好，以逼近那条水平屋顶。

Roofline 模型有用的部分不是图本身。有用的部分在于它告诉程序员哪一种资源是约束。受内存限制的核函数不会因为其数学指令稍微变好而变快。受计算限制的核函数不会因为节省了几个无关字节而变快。第一步是知道核函数在脊点的哪一侧。

![A B200 roofline with example workloads, showing the memory roof, the compute roof, and the ridge point](/books/modern-gpu-programming-for-mlsys/img/roofline.png)

## 常见工作负载的算术强度

算术强度往往首先是算法性质，其次才是实现细节。通常在编写核函数之前就能做出粗略估计。

### 逐元素与归约

逐元素核函数（如 GELU）和归约式核函数（如 RMSNorm）读写大张量，而每个元素只做少量 FLOP。

它们的算术强度低。它们位于脊点远左侧。这类核函数的最佳版本通常试图逼近内存带宽屋顶，而非 Tensor Core 计算屋顶。

对于这些核函数，重要的问题是机械性的：

```text
Are the loads and stores coalesced?
Are bytes moved only once?
Can the operation be fused with a producer or consumer?
Can the dtype be smaller?
Can TMA or vectorized accesses help?

如果没有复用、也没有融合机会，内存屋顶就是真正的天花板。

### GEMM

GEMM 是相反的情形。其算术强度随问题规模增长，因为每个加载的 tile 可被许多乘加运算复用。

对于 `M = N = K` 的方阵 fp16 矩阵乘，理想算术强度约为：

```text
AI ≈ 2N^3 / (3 * 2N^2)
   = N / 3 FLOP/byte

这一估计假设 A 和 B 各读一次、C 写一次、beta 为零、片上复用完美、且没有额外的元数据、填充或冗余流量。真实核函数搬运的数据多于这一理想模型。但该估计仍然有用。

在 `N = 4096` 时：

```text
AI ≈ 4096 / 3
   ≈ 1365 FLOP/byte

这远在 B200 约 250 FLOP/byte 的脊点右侧。因此大型 GEMM 在 HBM roofline 下受计算限制。目标不仅仅是减少 HBM 流量。目标是使用 Tensor Core、让它们保持被喂养，并将数据搬运与计算重叠，从而使计算屋顶变得可达。

这正是一个朴素 GEMM 即便算术强度高也可能很慢的原因。算法允许高性能，但实现可能让 Tensor Core 闲置。

### 注意力

注意力介于这两个极端之间。其算术强度取决于序列长度、头维度、分块、掩码，以及中间张量是否被实体化。

标准注意力中的关键问题是分数矩阵。如果核函数把分数矩阵写到 HBM 再读回，它就把一个大中间体搬过了内存。Flash Attention（{ref}`chap_flash_attention`）通过把相关 tile 保留在片上、避免这次 HBM 往返来提升算术强度。

因此注意力优化一部分是 roofline 问题，一部分是调度问题。算法被改变以使更少的字节进入 HBM。然后核函数被调度以使剩余的搬运与计算重叠。

## 当算术强度低时

如果一个核函数位于脊点左侧，它受内存限制。Tensor Core 或 CUDA 核可能闲置，因为瓶颈是字节，而非算术指令。

有两种应对之策。

第一种应对是提高算术强度。这是杠杆更高的路径，因为它能把核函数推向受计算限制的区域。

最重要的技术是融合。低算术强度的一个常见来源是把一个中间张量写到 HBM、并在下一个操作中立即读回。把生产者与消费者融合后，该中间体就留在寄存器、SMEM 或 TMEM 中。HBM 往返消失了。

例子包括：

```text
GEMM plus elementwise epilogue
normalization folded into a neighboring op
attention computed without materializing the full score matrix

第二种技术是为复用而分块。如果一个 tile 被加载一次并在驱逐前使用多次，每个字节就支撑更多算术工作。GEMM 的高算术强度正来自这种复用。其他工作负载只要有对 tile 的重复使用，就可以借用同样的思路。

第三种技术是减少每个数值的字节数。从 fp32 走向 fp16、fp8 或 fp4 会减少流量并提高每字节的 FLOP。当格式需要元数据、缩放因子或额外转换工作时，实际收益会小于原始 dtype 之比。块缩放 fp8 和 fp4 就是这样的例子。即便如此，更小的 dtype 仍往往是把核函数在 roofline 上右移的最直接手段之一。

第二种应对是接受内存屋顶并努力达到它。有些核函数没有足够的工作可融合、也没有足够的复用可利用。纯拷贝、简单的逐元素操作，或对大张量的单遍归约，可能在根本上是受内存限制的。

此时目标不是击败屋顶。目标是饱和它。

这意味着：

```text
move each byte once
avoid redundant reads
use coalesced or vectorized accesses
use TMA for regular bulk tiles
keep enough memory requests in flight
use smaller storage dtypes when the algorithm allows it

一旦受内存限制的核函数达到了内存屋顶，进一步的计算优化就无济于事。变快的唯一办法是改变算法以搬运更少的字节。

## 优化阶梯

Roofline 说的是什么是可能的。它没有说达到该极限有多容易。

一个大型 fp16 GEMM 在理论上是受计算限制的。那只意味着 HBM 屋顶不是主要限制。它并不意味着任何实现都能达到 Tensor Core 屋顶。弥合差距需要正确的指令、布局、暂存、同步与调度。

第三部分中的 GEMM 核函数在 B200 上以一系列步骤展示了这一点（{ref}`chap_gemm_advanced`）。每一步保持相同的基本算法，但改变 tile 的计算或调度方式。

GEMM 阶梯中第一个大的可测跃迁，是从线程拷贝的分块路径走向 TMA 支撑的路径。TMA 把规则的 GMEM → SMEM tile 搬运从 CTA 线程上卸下，让核函数通过硬件管理的批量拷贝来喂养 Tensor Core。

在那第一次跃迁之后，主要改进来自重叠与调度。TMA 把未来的 tile 带入共享内存。`tcgen05.mma` 异步运行。Epilogue 排空先前结果。软件流水线与线程束特化安排这些部件，使硬件引擎同时活跃。

也没有规则规定每个中间步骤本身必须更快。像线程束特化这样的步骤可能暂时把资源花在一个不会立即改进数字的结构上。但如果它使后续的、更简单结构无法表达的重叠成为可能，它仍然可能是正确的一步。

![The GEMM optimization journey on B200: measured points from a synchronous tiled baseline through TMA, warp specialization, CTA clusters, and multi-consumer execution](/books/modern-gpu-programming-for-mlsys/img/gemm_perf.png)

## 重叠是主要杠杆

一旦 GEMM 受计算限制且已使用 Tensor Core，剩余差距通常来自空闲时间。

一个简单核函数可能这样做：

```text
load tile k
compute tile k
store tile k
load tile k + 1
compute tile k + 1
store tile k + 1

这种调度让硬件闲置。加载运行时 Tensor Core 在等。Tensor Core 运行时拷贝引擎可能闲置。存储排空时两者都可能等着。

流水线化核函数则试图把相互独立的阶段一起运行：

```text
load tile k + 1
compute tile k
store tile k - 1

这就是本书后续所用 Blackwell 核函数结构背后的核心思想。TMA 处理异步数据搬运。`tcgen05.mma` 处理异步 Tensor Core 工作。Epilogue 与存储处理输出侧。`mbarrier` 对象把各阶段连接起来，使每个消费者只在它所需的数据真正被需要时才等待。

要点不是消除依赖。要点是围绕依赖进行调度。tile `k` 的 MMA 在 tile `k` 加载完成之前无法开始。tile `k` 的 epilogue 在 tile `k` 的 MMA 完成之前无法读累加器。但 tile `k+1` 的加载通常可以在 tile `k` 的 MMA 进行时运行，而 tile `k-1` 的存储通常可以同时排空。

这就是为什么后续如此多的章节聚焦于异步机制：

```text
TMA for global memory to shared memory movement
mbarriers for load completion and resource handoff
tcgen05 for asynchronous Tensor Core compute
TMEM for long-lived accumulators
warp specialization to separate producer and consumer roles
clusters for larger cooperative tiles and multicast

它们是不同的机制，但服务于同一个调度目标：让有用的工作在不止一条硬件路径上同时运行。

## 占用率与资源压力

重叠不是唯一的延迟隐藏机制。更古老、更通用的机制是占用率（occupancy）。

占用率是驻留在一个 SM 上的工作量。如果一个 warp 停顿，调度器可以运行另一个就绪的 warp。这通过保持一个可用独立 warp 池来隐藏延迟。

占用率受每 SM 资源限制。主要限制是寄存器、共享内存、warp 槽位和 CTA 槽位。一个每线程使用大量寄存器或每 CTA 使用大量共享内存的核函数可能占用率低，因为只有少量 CTA 或 warp 能塞进该 SM。

许多现代 Tensor Core 核函数有意地以降低占用率的方式花费资源。多级共享内存流水线消耗 SMEM。大型寄存器片段消耗寄存器。TMEM 分配消耗 Tensor Memory 容量。线程束特化可能把整个 warp 保留给生产者或消费者角色。

这种权衡是刻意的。这些核函数不是通过驻留许多无关 warp 来隐藏延迟，而是通过在较少数量的驻留 CTA 内部进行显式重叠来隐藏延迟。一个低占用率核函数只要其流水线保持 TMA、Tensor Core 与存储忙碌，仍然可以很快。

两种方式并非普遍地更优。有些核函数需要高占用率，因为它们有不规则的内存访问或有限的显式重叠。另一些需要深度暂存与特化，因为那是高效喂养 Tensor Core 的唯一途径。正确的问题不是占用率是否高。正确的问题是活跃的硬件单元是否被保持忙碌。

## 这为后续带来什么

本书其余部分不断回到同一个诊断：

```text
Which roof is this kernel under?
What resource is binding?
What change moves the kernel closer to that roof?

对于受内存限制的核函数，答案通常是更少的字节和更好的带宽利用。这意味着融合、合并访问、向量化访问、在适用处使用 TMA，以及更小的 dtype。

对于受计算限制的 GEMM，答案是先上 Tensor Core，再谈重叠。核函数必须暂存操作数、发射异步 MMA 工作、保持流水线填满，并在不使计算路径停顿的情况下排空结果。

对于 Flash Attention，第一步是通过把分数与概率 tile 保留在片上来提高算术强度。之后，它使用与 GEMM 相同的重叠工具：分块数据搬运、共享内存暂存、异步计算，以及谨慎的资源交接。

这给出了一个实用的优化工作流。估计算术强度。定位屋顶。判断核函数是受内存限制还是受计算限制。然后优化真正决定天花板的那项资源。

没有这一步，核函数优化就沦为猜测。有了它，每一项改动都有理由：要么提高算术强度，要么把内存路径推向带宽峰值，要么减少计算屋顶下的空闲时间。

