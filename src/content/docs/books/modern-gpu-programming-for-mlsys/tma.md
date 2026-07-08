---
title: 异步数据移动：TMA
sidebar:
  order: 50
---

:::note[概述]

- TMA（张量内存加速器）是一种硬件引擎，用于在全局内存与共享内存之间进行异步分块拷贝。一个线程发出拷贝指令，引擎负责搬运数据。
- TMA 拷贝由一个张量映射描述符（tensor-map descriptor）描述。该描述符告诉引擎全局张量的形状、步长、分块坐标以及共享内存的交换（swizzle）模式。
- 在加载路径上，TMA 可以在写入共享内存时对分块进行交换，使分块直接落入 Tensor Core 所期望的布局。
- TMA 加载通过带有字节数追踪的 `mbarrier` 完成信号通知。TMA 存储使用提交组（commit group）和等待组（wait group）。
:::

只有当数据准备就绪可供消费时，Tensor Core 才能发挥作用。在 GEMM（通用矩阵乘法）或注意力核函数（kernel）中，一旦流水线（pipeline）填满，计算可能是计算受限的（[performance](/books/modern-gpu-programming-for-mlsys/performance/)），但只有当下一个操作数分块按时到达时，流水线才能保持填满状态。

移动分块的传统方式是让线程自行拷贝。每个线程计算地址、从全局内存发起加载，并将值存入共享内存。这能工作，但它把线程束（warp）指令耗费在地址算术和拷贝簿记上，而非用于计算。它还使得拷贝路径对那些本应喂给 Tensor Core 的线程束的指令流可见。

张量内存加速器（Tensor Memory Accelerator，简称 TMA）将这项工作移入硬件拷贝引擎。一个线程发出分块拷贝指令，拷贝引擎随后在全局内存与共享内存之间异步地搬运一个矩形分块。在引擎搬运数据的同时，CTA（协作线程数组）的其余部分可以继续进行其他工作。

TMA 还处理了布局问题的部分内容。Tensor Core 不仅需要共享内存中有正确的值，还需要它们处于正确的共享内存布局中。在加载路径上，TMA 可以在写入分块时应用共享内存交换。这使得分块可以直接落入后续 MMA 所期望的布局。

<div style="overflow-x:auto;">
<div style="overflow-x:auto;">
<iframe src="/books/modern-gpu-programming-for-mlsys/demo/tma_intro.html" title="TMA: the Tensor Memory Accelerator" loading="lazy"
        style="width:1320px; max-width:none; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
</div>

*交互演示：TMA 将一个分块从全局内存拷贝到共享内存。切换交换模式，并悬停查看某个源单元格以观察它在共享内存中的落点。*

## 一个线程发起，硬件搬运分块

一次 TMA 拷贝始于一个发起线程。该线程并不遍历分块中的所有元素。它向硬件提供拷贝的描述，然后由 TMA 引擎执行传输。

主要输入是张量映射描述符。该描述符描述了全局张量以及如何从中读取分块。它记录了诸如张量形状、步长、元素大小、分块形状和交换模式等信息。发起线程还提供分块应落入的共享内存地址。

指令发出后，拷贝异步运行。发起线程可以继续执行。CTA 中的其他线程也可以继续执行。传输现在是 TMA 引擎的责任，而非普通加载和存储指令的循环。

这为核函数提供了两种不同的方式来表达同一个逻辑操作——"拷贝这个分块"。

一种路径是线程拷贝。线程协作从全局内存加载并存入共享内存。这赋予核函数对每次访问的直接控制，但会消耗线程指令和用于地址计算的寄存器。

另一种路径是 TMA 拷贝。一个线程发出传输指令，硬件拷贝引擎执行矩形拷贝。这是大型规则分块的自然路径，尤其是 Tensor Core 核函数所使用的操作数分块。

这两条路径有不同的同步规则和不同的性能表现。在二者之间选择是一个调度（dispatch）决策。布局告诉核函数它想要的内存排列。作用域（scope）告诉它哪些线程或 CTA 参与其中。调度决定拷贝是由普通线程代码实现还是由 TMA 实现。

## 交换布局

仅仅搬运分块是不够的。分块还必须以一种 Tensor Core 能高效读取的布局放入共享内存。

这正是 TMA 交换的用武之地。当 TMA 将分块写入共享内存时，它可以对共享内存地址模式进行置换。全局内存分块仍然是一个逻辑矩形，但共享内存中的目标布局可以被交换。

交换模式是 TMA 描述符的一部分。一旦描述符设置完成，发起线程不必手动应用交换。引擎在数据落入共享内存时自动应用它。

重要的要求是一致性。TMA 描述符、共享内存分块布局以及后续 MMA 指令必须都描述相同的布局（[data layout](/books/modern-gpu-programming-for-mlsys/data-layout/)）。如果 TMA 以一种交换写入分块，而 MMA 以另一种交换读取它，硬件仍然会精确执行被要求的操作，只是字节会因计算而被错误排列。

这正是布局记法不再仅仅是簿记工具的关键所在。DSL（领域特定语言）使用的布局必须与 TMA 描述符和 Tensor Core 指令使用的硬件布局相匹配。例如，如果核函数说某个操作数分块以 128 字节交换布局存储，那么 TMA 描述符必须使用匹配的交换模式，MMA 调度也必须期望相同的共享内存排列。上面的演示允许你在无交换和 128 字节交换之间切换；悬停某个源元素即可看到应用交换后的落点。

理解交换的一个有用方式是：TMA 并没有改变逻辑分块，它改变的是逻辑元素在共享内存中的物理落点。后续 MMA 仍然消费相同的逻辑 A 或 B 分块。交换只决定该分块如何在共享内存的各 bank 之间排列。

## 用于分块与交换的 3D TMA

普通的 TMA 拷贝搬运一个扁平的 2D 分块，但 Tensor Core 所需的共享内存布局通常被*分块（tiled）*成交换原子（即 [data layout](/books/modern-gpu-programming-for-mlsys/data-layout/) 中的 8 x 128 字节原子）。TMA 通过一个额外的描述符维度来处理这一点。**3D TMA** 将共享内存盒子描述为 `(group, row, col)`，其中 group 维度跨越各原子，内层两维在一个原子内寻址。一次 3D 拷贝既逐原子地铺设分块（分块），又在每个原子内应用交换，因此数据到达时已经处于 MMA 所期望的布局，无需单独的分块或交换步骤。

<div style="overflow-x:auto;">
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="/books/modern-gpu-programming-for-mlsys/demo/tma_3d.html" title="Tiling and swizzling with 3D TMA" loading="lazy"
        style="width:1320px; max-width:none; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
</div>

*交互演示：以 (group, row, col) 寻址的 3D TMA 拷贝，分块进入交换后的共享内存。*

选择交换*格式*与此分块密切相关。更宽的交换将一列分散到更多 bank 上，因此 128 字节交换在适用时是默认选择，但 N 字节原子需要分块的连续维度填满它。因此，由于形状约束而较小的分块无法使用 128 字节交换，必须降级到 64 字节或 32 字节：经验法则是选择分块能填满的最大交换（[data layout](/books/modern-gpu-programming-for-mlsys/data-layout/)）。下面的演示直接展示了该约束：16 x 16 分块上的 128 字节交换只有在将分块拆分为与原子匹配的 16 x 8 组时才变得无冲突。

<div style="overflow-x:auto;">
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="/books/modern-gpu-programming-for-mlsys/demo/tiling_constraint.html" title="Swizzle imposes a tiling constraint" loading="lazy"
        style="width:1320px; max-width:none; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
</div>
<script>
(function () {
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    document.querySelectorAll('iframe.demo-tma3d').forEach(function (f) {
      if (e.source === f.contentWindow) f.style.height = d.height + 'px';
    });
  });
})();
</script>

*交互演示：16 x 16 分块上的 128 字节交换，在分块为 16 x 8 组后无冲突。*

## 完成：加载

拷贝是异步的，因此仅仅发出指令是不够的。消费者不能因为 TMA 指令已经发出就读取共享内存分块。只有在引擎完成写入字节后，分块才是可安全读取的。

对于 TMA 加载，完成信号是 `mbarrier`（[mbarrier](/books/modern-gpu-programming-for-mlsys/async-barriers/)）。

通常的序列是：

1. 为流水线阶段初始化或重用一个 `mbarrier`；
2. 告诉屏障 TMA 传输预期要写入多少字节；
3. 发出 TMA 加载；
4. 让 TMA 引擎在字节到达时更新屏障；
5. 让消费者在读取共享内存分块前等待该屏障阶段。

字节数通过如下操作设置：

```text
mbarrier.arrive.expect_tx(bytes)
```

它完成两项工作。它记录预期的传输大小，同时也执行发起线程在屏障上的到达。屏障并非仅因为这次调用就完成。它仍需等待 TMA 引擎报告预期字节已经到达。

随着传输进行，引擎对屏障执行 complete-tx 更新。屏障阶段只有在两个条件都满足时才会翻转：到达计数已满足，且待处理字节数降为零。

消费者随后等待该屏障。一旦对预期阶段的等待完成，共享内存分块就准备就绪。此时 MMA 路径可以安全读取它。

![TMA load synchronization flow](/books/modern-gpu-programming-for-mlsys/img/tma_sync_flow.png)

这是其他异步生产者-消费者交接所使用的同一屏障模型。生产者是 TMA 引擎。消费者是 MMA 路径或任何读取共享内存分块的其他代码。屏障是它们之间显式的交接点。

## 完成：存储

TMA 存储沿相反方向移动数据，从共享内存到全局内存。它们也是异步的，但完成机制不同。

TMA 加载通常喂给同一核函数内的消费者。MMA 路径需要知道共享内存分块何时就绪。这就是加载路径使用 `mbarrier` 的原因。

TMA 存储通常将最终数据写出至全局内存。往往没有立即的核函数内消费者等待存储的结果。核函数主要需要知道何时可以安全重用共享内存缓冲区或结束存储序列。

为此，TMA 存储使用提交组和等待组。核函数发出一个或多个存储，提交该组，随后等待该组排空。等待完成后，从核函数的角度该组中的存储已完成，存储所使用的共享内存区域可以被安全重用。

因此规则很简单：

```text
TMA load:  wait through an mbarrier with byte-count tracking
TMA store: wait through a commit group and wait group
```

这两种机制在不同的交接点服务于同一目的。加载需要使共享内存分块对后续消费者可见。存储需要确保在核函数重用源存储或依赖存储已排空之前，外发传输已完成。

## TMA 对流水线的重要性

TMA 在作为流水线一部分时最为有用。核函数可以在 Tensor Core 计算当前分块的同时发出未来分块的加载。加载在后台运行。计算在前台运行。当未来分块变为当前分块时，屏障将二者连接起来。

一个典型的 GEMM 循环反复使用这种结构。共享内存的一个阶段持有当前被 MMA 消费的分块。另一个阶段正被 TMA 填充。随着循环推进，角色轮换。在 MMA 读取一个阶段之前，它等待该阶段的加载屏障。在 TMA 覆写一个阶段之前，核函数确保前一个消费者已用它完毕。

这就是为什么 TMA 和 `mbarrier` 在 Blackwell 和 Hopper 风格的核函数中通常一起出现。TMA 为核函数提供异步拷贝引擎。屏障为核函数提供一种精确方式，以知晓拷贝的字节何时就绪。

