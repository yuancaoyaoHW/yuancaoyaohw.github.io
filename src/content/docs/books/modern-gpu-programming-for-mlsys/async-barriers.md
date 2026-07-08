---
title: 异步协调：mbarrier
sidebar:
  order: 80
---

:::note[概述]

- TMA（张量内存加速器）和 Tensor Core 是异步的，因此发出工作并不等同于完成工作，消费者需要显式的完成信号。
- mbarrier 就是那个信号：生产者到达，消费者等待，它追踪到达计数和（对于 TMA）字节数。
- 每个屏障携带一个*阶段（phase）*，每轮翻转；在正确的阶段上等待正是安全放行消费者的关键。
:::

TMA（[TMA](/books/modern-gpu-programming-for-mlsys/tma/)）和 Tensor Core（[Tensor Core](/books/modern-gpu-programming-for-mlsys/tensor-cores/)）操作是异步的。当核函数（kernel）发出一次 TMA 加载或 `tcgen05` MMA 时，发起线程不会等待操作完成。指令只是被提交给硬件引擎；实际的数据移动或矩阵运算与程序的其余部分并行继续。

这很有用，因为它让内存移动和计算重叠。它也意味着程序顺序不足以证明数据已就绪。一条后续指令可能在更早的异步操作完成之前就运行。如果 TMA 仍在写入一个共享内存分块时 MMA 就开始读取它，MMA 会读到不完整的数据。如果收尾阶段（epilogue）在 Tensor Core 完成写入累加器之前读取 TMEM（张量内存），它会读到错误的值。如果核函数在错误的条件上等待，它可能永远无法取得进展。

因此，核函数在每次异步交接处都需要一个显式的完成信号。`mbarrier` 就是那个信号。生产者在其工作完成时到达屏障，消费者在使用已产生的数据之前等待屏障。同一机制用于 TMA 到 MMA 的交接、MMA 到收尾阶段的交接，以及跨流水线（pipeline）阶段的缓冲区重用。

屏障不仅仅是一次性的标志。它携带一个阶段位，该阶段位在屏障每完成一轮到达时改变。阶段正是让一个屏障可以在许多循环迭代中重用、而不把一次迭代的完成与另一次迭代的完成混淆的关键。

## mbarrier

`mbarrier`，即内存屏障（memory barrier）的简称，是存储在共享内存中的硬件同步（synchronization）对象。概念上，它包含两块状态：一个到达计数器和一个阶段位。计数器告诉屏障当前轮次还缺少多少次到达。阶段位告诉核函数屏障当前处于哪一轮。

<div style="overflow-x:auto;">
<div style="overflow-x:auto;">
<iframe src="/books/modern-gpu-programming-for-mlsys/demo/mbarrier_mechanism.html" title="mbarrier data structure and APIs" loading="lazy"
        style="width:1320px; max-width:none; height:620px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
</div>

*交互演示：`mbarrier` 状态视图，展示到达计数器、阶段位以及 `init`、`arrive` 和 `wait` 操作；点击某个字段以聚焦它。*

屏障从初始化开始。在 `init` 期间，核函数设置此屏障应期望多少次到达。屏障以阶段 0 开始，其计数器加载为该预期到达数。从那时起，屏障就在等待所有必需的生产者或某个资源的使用者报告它们已完成。

一次到达减少了屏障仍在等待的工作量。核函数的不同部分可以以不同方式到达屏障，这一区别很重要。

对于 TMA 加载，通常的到达路径是 tx 计数到达。诸如 `mbarrier.arrive.expect_tx(bytes)` 这样的操作做两件事。首先，它算作发起线程在屏障上的到达。其次，它记录 TMA 引擎预期要传输的字节数。屏障并非仅因为发起线程已到达就完成。它还等待 TMA 引擎在传输结束时排空该字节数。阶段只有在两个条件都满足时才翻转：正常到达计数已降为零，且待处理的 tx 字节数已降为零。

这就是为什么 `expect_tx` 不应被理解为"又一次普通到达"。它为异步拷贝设置了一个字节预算。硬件随后通过 complete-tx 更新来核算实际拷贝的完成。屏障只有在到达和字节传输都完成后才完成。

对于 Tensor Core 工作，到达路径不同。`tcgen05` MMA 不会仅因为 MMA 已发出就自动推进屏障。核函数必须显式地将一次屏障到达附加到提交路径上，例如通过 `tcgen05.commit.mbarrier::arrive` 操作。当该已提交的组完成时，Tensor Core 侧执行屏障到达。如果核函数忘了那次提交到达，等待屏障的消费者将永远等待。

普通线程也可以直接到达屏障。这用于普通线程代码作为生产者时，或一组线程宣布它已完成使用某个资源时。例如，在消费者读完一个共享内存缓冲区后，它可以到达一个屏障，告诉生产者该缓冲区可以重用。

等待是同一协议的消费者侧。消费者等待直到屏障已完成当前迭代所期望的阶段。只有那时才安全读取数据或重用受该屏障保护的资源。

要点是，异步硬件不仅先于程序运行；它还通过屏障回报完成。TMA 可以发信号表示一个共享内存分块已就绪。Tensor Core 工作可以发信号表示 TMEM 结果已就绪。普通线程可以发信号表示某个缓冲区不再被使用。屏障赋予所有这些情况相同的生产者-消费者形态：生产者到达，消费者等待。

## 阶段追踪

屏障通常不为单次使用而分配。一个流水线化的 K 循环可能执行同一交接数百次，为每次迭代分配一个新的共享内存屏障并不实际。相反，核函数保留一小组固定的屏障，并随着循环推进重用它们。

阶段位正是使该重用安全的关键。

<div style="overflow-x:auto;">
<div style="overflow-x:auto;">
<iframe src="/books/modern-gpu-programming-for-mlsys/demo/phase_tracking.html" title="mbarrier phase tracking" loading="lazy"
        style="width:1320px; max-width:none; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
</div>

*交互演示：一个在若干流水线迭代中重用的屏障，展示阶段位在每轮完成后翻转。*

每当屏障完成其当前轮次的所有到达时，它翻转阶段：阶段 0 变为阶段 1，阶段 1 变为阶段 0，如此往复。等待操作检查消费者所期望的阶段。该期望阶段由核函数保存在寄存器中。在一个阶段成功等待一轮之后，核函数在为下一轮使用屏障之前切换其本地阶段值。

这防止了核函数将旧的完成误认为新的完成。假设一个屏障曾被用于一次 TMA 加载并已完成。如果下一个循环迭代在不追踪阶段的情况下重用同一屏障，消费者可能观察到前一次完成并错误地假设新加载已就绪。阶段位将这两轮分开。迭代 0 等待一个阶段，迭代 1 等待相反的阶段，迭代 2 再次等待第一个阶段，模式如此继续。

在真实的流水线中，簿记通常按阶段进行。核函数有固定数量的共享内存阶段、与之匹配的固定数量的屏障，以及寄存器中的一小组阶段值。随着循环推进，每个逻辑迭代映射到一个物理阶段，阶段值告诉等待操作它在等待该物理屏障的哪一轮。

这就是为什么后续 GEMM 代码不需要每个 K 分块一个屏障（[GEMM async](/books/modern-gpu-programming-for-mlsys/gemm-async/)）。它需要每个可重用阶段一个屏障，加上阶段追踪。阶段索引选择共享内存缓冲区和屏障。阶段值区分该阶段的当前使用与前一次使用。

**与你的 agent 一起尝试**：给它一个两阶段流水线，要求它追踪四次迭代。对于每次迭代，列出阶段索引、本地阶段值、屏障何时翻转，以及如果在阶段重用前不切换阶段会出什么问题。

## 同步规则

一旦屏障和阶段机制清晰，Tensor Core 核函数中的同步模式就相当机械化了。每当一条路径产生数据或释放一条将被另一条路径消费的资源时，交接必须被显式化。

有三种常见情况。

第一种情况是线程代码为异步引擎产生数据。如果线程写入共享内存，而后续的 TMA 存储或 MMA 指令读取该共享内存，核函数必须使线程写入在引擎读取它们之前可见。这需要适当的线程级同步或屏障（fence）。确切的指令取决于交接的作用域（scope），但原因总是相同：引擎不得在产生数据的线程完成写入之前观察到该共享内存缓冲区。

第二种情况是 TMA 为 MMA 产生数据。一次 TMA 加载异步地填充一个共享内存分块。MMA 路径不能仅因为 TMA 指令已发出就推断分块已就绪。TMA 操作必须与一个 `mbarrier` 关联，且 MMA 路径必须在读取分块之前等待该屏障。

第三种情况是 MMA 为收尾阶段产生数据。`tcgen05` MMA 将其结果异步写入 TMEM。在 Tensor Core 完成相关工作之前，收尾阶段不能安全读取累加器。因此 MMA 提交路径到达一个完成屏障，收尾阶段在读取 TMEM 之前等待该屏障。

<div style="overflow-x:auto;">
<div style="overflow-x:auto;">
<iframe src="/books/modern-gpu-programming-for-mlsys/demo/mbarrier_tma_timeline.html" title="mbarrier signalling TMA completion" loading="lazy"
        style="width:1320px; max-width:none; height:700px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
</div>

*交互演示：一次 TMA 加载通过 `mbarrier` 发出完成信号。MMA 路径在读取共享内存分块前等待屏障。Tensor Core 到收尾阶段的交接遵循相同形态，只是由 Tensor Core 提交路径而非 TMA 执行到达。*

同一思想也适用于资源重用。屏障不仅是数据就绪信号。它也可以是"资源已空闲"的信号。一个共享内存阶段在旧分块的所有消费者都用完它之前不能被覆写。一个 TMEM 区域在前一个使用者完成读取或写入之前不能被重用。在这些情况下，到达意味着"我已用完此资源"，等待意味着"现在可以安全地为下一阶段重用此资源了"。

这是阅读流水线化 GEMM 核函数中同步的正确方式。那些等待和到达并非作为防御性编程散布各处。每一次都标记一次具体的所有权转移：一个分块变得就绪、一个累加器变得可读，或一个缓冲区变得可重用。一旦识别出这些交接，控制流就变得容易追踪得多。

