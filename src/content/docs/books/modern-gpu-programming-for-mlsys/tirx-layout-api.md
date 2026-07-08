---
title: TIRx 布局 API
sidebar:
  order: 110
---

# TIRx 布局 API

> **概览**

- TIRx 布局 API 把 {ref}`chap_data_layout` 中的布局记号变成编译器对象。主要对象是 `TileLayout`、`SwizzleLayout` 和 `ComposeLayout`。
- `TileLayout` 描述在命名硬件轴上的仿射放置。它由分片规范 `S[...]`、副本规范 `R[...]` 和可选偏移构成。
- 一个布局把一个逻辑坐标映射到一个或多个物理坐标。`layout.apply()` 求值这个映射。
- `SwizzleLayout` 描述用于避免 bank 冲突（bank conflict）的基于 XOR 的共享内存交换。`ComposeLayout` 把一个交换叠加到一个分块布局之上。
- 诸如 `tmem_datapath_layout`、`tcgen05_atom_layout` 和 `wg_local_layout` 这样的现成构造器覆盖了在核函数中反复出现的硬件布局。

{ref}`chap_data_layout` 介绍了本书通篇使用的记号：一个分块形状、一组在命名轴上的步幅，以及一个用于被复制（而非被划分）的值的可选复制项。本章把那个记号变成编译器使用的 API。

目标是让页面上的记号和核函数中的代码看起来几乎一样。当你写一个诸如以下的布局时：

```python
S[(128, 256) : (1@TLane, 1@TCol)]

你不仅仅是在写一段解释。你正在构造一个 `TileLayout` 对象，它可以被附加到缓冲区（buffer）上。此后，每个触及该缓冲区的分块操作都可以从布局中读取它的放置。放置被写一次、检查一次，并被编译器复用。

布局既可以在从池中分配时附加，也可以在声明缓冲区时附加：

```python
pool.alloc(shape, dtype, layout=layout)

T.decl_buffer(shape, dtype, scope=scope, layout=layout)

从那时起，缓冲区就携带了它的物理放置。分块操作不必重复每个元素存放在哪里。

布局对象位于一个模块中：

```python
from tvm.tirx.layout import (
    TileLayout,
    SwizzleLayout,
    ComposeLayout,
    S,
    R,
    laneid,
    warpid,
    tid_in_wg,
    TLane,
    TCol,
    m,
    tcgen05_atom_layout,
    tmem_datapath_layout,
)

这个 API 背后有一个核心思想。一个布局不必把一个逻辑索引映射到单个物理地址。它把一个逻辑索引映射到在命名轴上的一组物理坐标。在通常情况下这组坐标只有一个元素。当存在复制时，同一个逻辑元素有多个物理放置。

这就是布局模型有三个部分的原因：分片、副本和偏移。分片放置元素。副本把它复制到额外的坐标。偏移移动整个放置。

## 通过示例看布局

下面的示例展示了 API 的基本形状。

TMEM 中的累加器可以写成在 TMEM 轴上的直接放置：

```python
acc = TileLayout(S[(128, 256) : (1@TLane, 1@TCol)])

这里逻辑行映射到 `TLane`，逻辑列映射到 `TCol`。在 {ref}`chap_tmem` 中，硬件坐标被称为 Lane 和 Col。在 TIRx 布局记号中，那些硬件轴被写作 `TLane` 和 `TCol`。

一个块缩放 MMA 的缩放因子布局使用复制：

```python
scale_factor_layout = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
)

分片把一个 32 行的组放置在 TMEM 中。副本以 32 个通道的步幅把该组重复四次，于是这个 32 行组在完整的 128 通道 TMEM 空间中可见。

一个 Tensor Core 寄存器片段可以分布在通道和线程束（warp）之间：

```python
frag = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
)

同一个物理轴可以出现不止一次。在这个示例中，两个不同的迭代（iter）都对 `laneid` 有贡献。没有显式轴的步幅使用默认的内存轴 `m`。

在真实核函数中，常见的硬件布局通常来自构造器：

```python
acc = tmem_datapath_layout("D", 128, 256)

ld = tcgen05_atom_layout("32x32b", (128, 64), "float32")

这些构造器返回普通的 `TileLayout` 对象。它们是便利工具，不是一套单独的机制。你可以检查返回的布局，把它与其他布局组合，或者在形状不寻常时手动写出底层的 `S[...]` 和 `R[...]` 形式。

## 交互式演示

在讲解机制之前，有一些具体的东西可以摆弄会很有帮助。下面的演示让你选择一个预设布局，编辑逻辑形状和 `S` 或 `R` 项，选择一个 dtype 和交换模式，然后点击一个元素查看哪个或哪些物理坐标拥有它。


<p>
  <a class="reference external" href="/books/modern-gpu-programming-for-mlsys/_static/tirx-layout-demo/index.html"
     target="_blank" rel="noopener"
     style="display:inline-block; padding:10px 18px; background:#3b82f6;
     color:#fff !important; font-weight:700; border-radius:8px;
     text-decoration:none;">▶ Open the demo full screen ↗</a>
</p>
<iframe id="tirx-layout-demo-frame" src="/books/modern-gpu-programming-for-mlsys/_static/tirx-layout-demo/index.html?notitle"
        style="width:100%; height:1040px; border:1px solid #dfe1e6;
        border-radius:10px; margin:10px 0 6px; display:block;"
        title="TIRx interactive layout demo" loading="lazy"></iframe>
<script>
// The demo (viz-base.js) posts its content height; size the iframe to fit so
// there is no inner scrollbar. This demo is responsive (fills the width), so
// only the height follows content.
(function () {
  var f = document.getElementById('tirx-layout-demo-frame');
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    if (f && e.source === f.contentWindow) f.style.height = d.height + 'px';
  });
})();
</script>

演示很有用，因为 API 的大部分只是演示所展示内容的一个精确版本。一个逻辑元素进入布局。布局把它展平，按它的迭代拆分，在命名轴上累加坐标，然后在需要时应用复制。

## TileLayout

`TileLayout` 是主要的仿射布局对象。它通常用与正文中相同的记号书写：

```python
TileLayout(S[shape : strides])

`S` 项是分片规范。你可以这样读它：取一个这个形状的逻辑分块，并用这些在命名轴上的步幅放置它。

当一个值需要出现在多个位置时，分片规范用一个副本规范扩展：

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride])

也可以加上一个可选偏移：

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride] + offset)

在表面之下，这些部分由迭代表示。一个迭代是一个三元组：

```text
(extent, stride, axis)

它描述沿一个命名轴的步进式行走。范围（extent）告诉迭代有多少个位置。步幅告诉每步移动多远。轴告诉正在改变哪个硬件坐标。

一个布局有三部分。

### 分片

分片，或 `D`，是由 `S[...]` 构建的部分。它把逻辑索引划分到一个或多个迭代上，并产生基础物理坐标。

例如：

```python
S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]

有四个分片迭代。它们的范围是 `8`、`2`、`4` 和 `2`。它们的步幅把数据放置在 `laneid`、`warpid`、再次 `laneid`，以及默认内存轴 `m` 上。

这推广了普通的形状-步幅规则。区别在于步幅被附加到命名硬件轴上，而不是单个扁平地址上。

### 副本

副本，或 `R`，描述同一个逻辑元素的额外物理副本。副本迭代独立于逻辑索引。它们枚举硬件空间中的额外偏移。

例如：

```python
R[2 : 4@warpid]

创建两个副本，在 `warpid` 轴上相隔四个线程束。

复制不是为了方便的把戏。它描述真实的硬件行为。一些数据在线程束、通道或内存区域之间广播。逻辑到物理的映射自然支持这一点，因为一个逻辑元素可以映射到一组物理坐标。

### 偏移

偏移，或 `O`，是加到每个结果上的固定坐标。

例如：

```python
5@warpid

把整个放置在 `warpid` 轴上移动五个。

偏移用于把一个分块放置在选定的基础坐标、为一个区域的独占使用预留，或描述一个在同一资源中接在另一个分块之后开始的分块。

### 把各部分组合起来

一个布局按顺序应用这三个部分。

首先，分片计算基础坐标。然后副本把那个坐标扇出到零个或多个额外副本。最后，偏移移动每个坐标。

对于逻辑坐标 `x`，结果是：

```text
L(x) = { D(x) + r + O | r in R }

如果没有副本，`R` 只包含零偏移，所以结果是单元素集。如果有副本，结果对每个副本位置包含一个坐标。

在 TIRx 语法中，一个完整布局可以看起来像这样：

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)

从左到右读，分片放置逻辑分块，副本在相隔四个线程束 ID 处创建第二个副本，偏移把整个放置移动到从 `warpid = 5` 开始。

如果迭代已经作为对象被构建，同样的布局可以直接构造：

```python
TileLayout.from_iters(shard, replica, offset)

大多数用户代码使用 `S[...]` 和 `R[...]` 记号，因为它更接近数学形式。

## 命名轴

布局中的轴不是匿名维度。每个轴命名一个真实的硬件坐标或一个编译器级放置坐标。

示例包括：

```text
bx, by, bz
cbx, cby, cbz
tx
warpid
laneid
wgid
tid_in_wg
wid_in_wg
m
P, F
Bank
TLane, TCol

诸如 `bx`、`by` 和 `bz` 这样的网格（grid）轴把工作分布在各 CTA 之间。诸如 `cbx`、`cby` 和 `cbz` 这样的集群（cluster）轴把工作放置在一个 CTA 集群内。诸如 `tx`、`warpid`、`laneid`、`tid_in_wg` 和 `wid_in_wg` 这样的线程（thread）轴描述 CTA 或线程束组内部的所有权。轴 `m` 是默认的线性内存轴。`P` 和 `F` 用于二维暂存式放置。`Bank` 命名共享内存 bank。`TLane` 和 `TCol` 是 TMEM Lane 和 Col 坐标的 TIRx 布局名称。

轴名称是布局的一部分。这很重要，因为具有相同整数值的两个坐标可能意味着不同的硬件事物。`1@tx` 与 `1@tid_in_wg` 不同。`1@laneid` 与 `1@TLane` 不同。布局把这些含义保持显式。

## 正向映射

求值一个布局意味着取一个逻辑坐标并计算它在物理上落在何处。API 方法是：

```python
layout.apply(*coord)

对于一个没有复制的布局，结果是一个坐标字典。有复制时，结果是一组坐标字典。一个坐标字典把轴名称映射到整数位置，例如：

```python
{"laneid": 7, "warpid": 2, "m": 1}

求值规则有四步。

首先，以行优先顺序展平逻辑坐标。对于一个逻辑坐标：

```text
x = (x0, x1, ..., xr-1)

在一个逻辑形状内：

```text
(S0, S1, ..., Sr-1)

扁平索引是：

```text
flat = x0 * S1 * S2 * ... * Sr-1
     + x1 * S2 * ... * Sr-1
     + ...
     + xr-2 * Sr-1
     + xr-1

其次，把那个扁平索引按分片范围拆分。如果分片范围是：

```text
(e0, e1, ..., en-1)

那么拆分产生分量：

```text
c0, c1, ..., cn-1

使用与分片范围相同的行优先顺序。

第三，用每个分量的步幅把它累加到它的轴上。如果分片迭代 `k` 有范围 `ek`、步幅 `sk`、轴 `ak`，那么分量 `ck` 贡献：

```text
ck * sk @ ak

对同一个轴的所有贡献被加在一起。然后加上偏移。

第四，应用副本迭代。每个副本迭代贡献一个独立于逻辑坐标的额外偏移。如果有多个副本迭代，布局枚举所有组合。

这条规则的一个有用推论是，布局不必硬编码输入形状。它需要的是逻辑分块具有与分片范围乘积相同的元素总数。一旦成立，展平和拆分就定义了映射。

## 案例研究：Tensor Core 寄存器分块

考虑一个逻辑 `(8, 16)` 分块分布在两个各 32 通道的线程束上。每个通道拥有一个小的寄存器片段。寄存器槽由默认内存轴 `m` 表示。

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)

从 `(8, 16)` 分块取一个逻辑元素 `(i, j)`。

行优先扁平索引是：

```text
flat = 16 * i + j

按分片范围 `(8, 2, 4, 2)` 拆分得到：

```text
c0 = i
c1 = floor(j / 8)
c2 = floor(j / 2) mod 4
c3 = j mod 2

分片贡献是：

```text
laneid = 4 * c0 + c2
warpid = c1
m      = c3

加上偏移 `5@warpid` 后，这变成：

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5
m      = j mod 2

副本项：

```python
R[2 : 4@warpid]

给 `warpid` 加上 `0` 或 `4`。所以完整的映射是：

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5 + 4 * r, where r in {0, 1}
m      = j mod 2

分片把分块放置在线程束 5 和 6 上。副本然后把它复制到线程束 9 和 10。因此同一个逻辑元素出现在两个线程束位置。

这个示例说明了为什么模型使用一组物理坐标。复制不能自然地用一个从物理坐标到逻辑坐标的函数表示。它能自然地用一个从一个逻辑坐标到多个物理坐标的函数表示。

## 案例研究：Blackwell 张量内存

同样的布局模型适用于内存放置。轴不必是线程轴。它们可以是内存轴。

TMEM 由硬件 Lane 和 Col 坐标寻址。在 TIRx 布局记号中，这些轴被写作 `TLane` 和 `TCol`。

考虑这个布局：

```python
layout = TileLayout(
    S[(2, 128, 112) : (112@TCol, 1@TLane, 1@TCol)]
)

如果逻辑分块形状是 `(2, 128, 112)`，拆分分量就是逻辑坐标本身。对于元素 `(a, l, c)`，映射是：

```text
TLane = l
TCol  = 112 * a + c

步幅为 `1@TLane` 的范围-128 迭代填满 128 个 TMEM Lane 行。步幅为 `112@TCol` 的范围-2 迭代和步幅为 `1@TCol` 的范围-112 迭代一起覆盖 224 列：

```text
TCol in [0, 224)

这个 224 列的跨度是有意的。TMEM 布局不必是 2 的幂。一个块缩放 FP8 GEMM 可能选择 224 列的累加器，因为完整的 256 列分块不会为两个累加器阶段加上缩放因子留下足够的 TMEM 容量。布局 API 可以直接表达那个形状。

## 缩放因子布局

上面的累加器布局是纯放置。每个逻辑累加器元素映射到一个 TMEM 坐标。块缩放 MMA 的缩放因子不同，因为同一个物理组可能需要在多个线程束窗口中可见。这正是复制变得有用的地方。

一个紧凑的缩放因子布局可以写成：

```python
scale = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)]
    + R[4 : 32@TLane]
)

分片把一个 32 行的缩放因子组放置在 TMEM 中：

```text
TLane = r
TCol  = s

对于一个逻辑缩放坐标 `(r, s)`。

副本项创建四个相隔 32 通道的副本：

```text
TLane = r + 32 * q, where q in {0, 1, 2, 3}
TCol  = s

所以这个 32 行组在 TMEM 通道 0 到 31、32 到 63、64 到 95、96 到 127 处可见。这就是 `warpx4` 广播模式（{ref}`chap_layout_generations`）。四个线程束大小的 TMEM 通道窗口中的每一个都看到同一个缩放因子组。

在完整的块缩放 MMA 布局中，这个原子与在 M 行和 K 缩放因子组上的外层迭代组合。多个缩放因子也可能被打包进一个 32 位 `TCol` 单元，取决于缩放因子的 dtype。例如，fp8 缩放因子可以把四个值打包进一个 32 位列单元。可选的零步幅复用和流水线深度迭代随后可以描述跨多个 MMA 的缩放复用和双缓冲（double buffering）。

重要的部分是同一个 `TileLayout` 模型描述两种情况。累加器是 TMEM 中的单个放置。缩放因子是同一 TMEM 地址空间中的复制放置。

## 现成布局

大多数核函数不会手写每个硬件布局。TIRx 为经常出现的布局提供了构造器。

```python
tmem_datapath_layout(datapath, rows, cols)

返回由 `tcgen05.mma` 写入的 TMEM 累加器布局。`datapath` 参数选择行放置模式。例如，`"D"` 对应 `M = 128` 的恒等式风格放置，而 `"F"` 对应 `M = 64` 的分散放置。

```python
tcgen05_atom_layout(instr_shape, tensor_shape, dtype)

返回由 `tcgen05.ld` 或 `tcgen05.st` 原子移动的寄存器分块布局。指令形状的示例包括 `.32x32b`、`.16x64b`、`.16x128b` 及相关形式。在 DSL 层级，这是一个线程束组分布的分块。在降低（lowering）期间它变成四条线程束协作的 `tcgen05.ld` 或 `tcgen05.st` 指令，每条线程束各一个，每条线程束处理自己的 32 个 TMEM 通道。

```python
wg_local_layout(cols, rows=128)

返回一个线程束组局部的寄存器分块，通常是 `tid_in_wg` 上每线程一行。

这些助手的存在是为了避免手动重写常见的硬件映射。它们并不隐藏模型。每个助手返回一个由上述同样的 `S` 和 `R` 部分构建的普通 `TileLayout`。

## SwizzleLayout 和 ComposeLayout

`TileLayout` 是仿射的。它可以表达在命名轴上的步幅、复制和偏移。这对许多放置已经足够，包括线程片段、TMEM 分块和紧凑的缩放因子布局。

共享内存交换需要别的东西。用于避免 bank 冲突的交换不是仿射步幅模式。它是对线性共享内存地址的基于 XOR 的置换。

因此 TIRx 把交换保持为一个单独的布局对象：

```python
SwizzleLayout(...)

并把它与分块布局组合：

```python
ComposeLayout(swizzle, tile)

分块布局首先产生一个线性内存地址。交换然后置换那个地址。把这两层保持分离比把 XOR 置换硬塞进仿射布局模型更干净。

## 为什么需要交换

共享内存被划分为 32 个 bank，每个 bank 字持有 4 字节。当一次访问的各通道触及同一个 bank 中的不同地址时，该访问会被 bank 冲突串行化。

一个普通的行优先分块会在结构上造成这种冲突。考虑一个行优先布局的 `(8, 64)` float16 分块：

```python
TileLayout(S[(8, 64) : (64@m, 1@m)])

逻辑元素 `(i, j)` 的线性元素地址是：

```text
m = 64 * i + j

每行是 64 个 float16 值，即 128 字节。这恰好是一整条共享内存 bank 线。如果一个线程束以固定 `j` 沿一列向下读，每行步进前进一整条 128 字节线。bank 索引重复，所以列读在各行间塌缩到同一个 bank 上。

交换通过让低位地址比特依赖更高行比特来改变这一点。一列原本会反复落在同一个 bank 上，现在被散布到不同 bank。

## 交换变换

一个 `SwizzleLayout` 由三个整数参数控制：

```text
per_element = M
swizzle_len = B
atom_len    = S

输入是一个线性元素地址 `m`。

`m` 的低 `M` 比特保持不变。这保留了一小组连续元素。更高比特被下移到一个临时值：

```text
x = m >> M

然后 `x` 的 `[S, S + B)` 位组被异或进 `x` 的 `[0, B)` 位组。交换后的地址随后通过把不变的低 `M` 比特放回而形成。

等价地：

```text
mask = (1 << B) - 1

low  = m & ((1 << M) - 1)
x    = m >> M
x2   = x ^ ((x >> S) & mask)

addr = (x2 << M) | low

要使布局良构，`S` 必须至少为 `B`。

这个变换的意义不在于改变分块中有哪些逻辑元素。它改变这些元素在共享内存中落在何处。MMA 仍然读同一个逻辑分块。交换使物理 bank 模式更好。

## 选择交换参数

在正常使用中，交换参数从 dtype 和共享内存交换模式中选择。常见模式是 32 字节、64 字节和 128 字节交换。

`per_element` 参数被选择使得一小组向量大小的元素保持连续。对于 float16，一个 16 字节向量包含 8 个元素，所以：

```text
M = log2(8) = 3

对于 128 字节交换，布局使用：

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

这保持 16 字节向量组完整，同时仍然对更大的共享内存地址模式进行足够的置换以打破列 bank 冲突。

大多数代码不应手动推导这些参数。dtype 和描述符（descriptor）模式通常决定它们。对程序员而言重要的是，TIRx 布局中的交换、TMA 描述符和 MMA 期望三者匹配。

因此一个交换过的共享内存分配看起来像：

```python
tile = TileLayout(S[(8, 64) : (64@m, 1@m)])
swizzle = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

layout = ComposeLayout(swizzle, tile)

组合后的布局就是被附加到共享内存缓冲区的东西。

## 一个元素的 bank 和线

要看交换是否有帮助，把交换后的元素地址转换回共享内存 bank。

令 `addr` 为交换后的元素地址，`b` 为以字节为单位的元素大小。字节地址是：

```text
byte = addr * b

bank 是：

```text
bank = floor(byte / 4) mod 32

128 字节 bank 线是：

```text
line = floor(byte / 128)

对于 float16，`b = 2`，所以 bank 公式变成：

```text
bank = floor(addr / 2) mod 32

这是下面工作示例中使用的公式。

## 工作示例：`(8, 64)` float16 分块上的 128B 交换

回到行优先 float16 分块：

```text
m = 64 * i + j

使用：

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

变换变成：

```text
x    = m >> 3
addr = ((x ^ ((x >> 3) & 7)) << 3) | (m & 7)

由于：

```text
m = 64 * i + j

我们可以写：

```text
q = floor(j / 8)
r = j mod 8

而交换后的地址是：

```text
addr = 64 * i + 8 * (q xor i) + r

现在看列 `j = 0`。那么 `q = 0` 且 `r = 0`，所以：

```text
addr = 72 * i

对于 float16，bank 是：

```text
bank = floor(addr / 2) mod 32

所以八行映射到：

```text
i = 0: bank 0
i = 1: bank 4
i = 2: bank 8
i = 3: bank 12
i = 4: bank 16
i = 5: bank 20
i = 6: bank 24
i = 7: bank 28

这一列现在触及八个不同的 bank。冲突消失了。

没有交换时，同一列的地址是：

```text
m = 64 * i

因此：

```text
bank = floor(64 * i / 2) mod 32 = 0

每一行都落在 bank 0 上，所以访问被串行化。交换只改变物理放置，但这已足以把列访问变成无冲突的。

这个保证取决于以交换被设计的方式使用它。dtype、交换宽度和访问形状必须与 TMA 和 MMA 描述符模式匹配。一个 128 字节 float16 交换是围绕相关的 16 字节行块和 Tensor Core 访问模式设计的。它并不承诺任意共享内存访问都变得无冲突。本章顶部的演示让这一点可见：选择一个 dtype 和交换模式，观察一列在没有交换时塌缩到一个 bank 上，然后在应用匹配的交换后散布到 bank 视图各处。

## 设计理据

布局 API 遵循三个设计选择。

第一，它支持一般形状。硬件分块并不总是 2 的幂。全局张量、共享内存阶段、TMEM 累加器和缩放因子缓冲区常常具有来自容量限制或算法选择的形状。布局模型把这些形状视为正常。

第二，映射从逻辑坐标到物理坐标。这个方向很重要，因为复制很常见。一个逻辑元素可能存在于多个物理位置。逻辑到物理的映射直接把它表示为一组坐标。

第三，硬件轴是显式的。布局不使用匿名维度并依赖上下文稍后解释它们。`tx`、`tid_in_wg`、`laneid`、`warpid`、`TLane` 和 `TCol` 之间的区别被写进布局本身。

合法性和可行性检查不是布局对象单独的工作。一个布局可以说数据放置在何处。更高层的分块原语决定一个给定操作能否合法且高效地使用那个放置。这种分离使布局 API 保持小巧，同时仍给编译器足够的信息来调度真实的硬件操作。

