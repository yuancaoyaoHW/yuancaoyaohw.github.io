---
title: 数据布局及其记法
sidebar:
  order: 30
---

# 数据布局及其记法

> **概览**

- *数据布局*把张量的逻辑索引映射到物理位置，它决定合并访问、bank 冲突，以及某个引擎能否读取一个 tile。
- 本书用一种记法书写布局：`S[(shape) : (strides)]`，带有命名轴（`@laneid`、`@TLane`、……）和一个用于广播或被复制数据的复制项 `R[...]`。
- swizzle 是对地址的一次 XOR 重映射，用于消除共享内存 bank 冲突。

同样的数字，以不同物理排列写入内存，在同一块 GPU 上可以相差一个数量级。

原因是张量的逻辑索引并不说明它的字节实际落在何处。硬件对该放置极为敏感：它决定了 32 个通道的加载是合并成一次事务，还是散成 32 次；它们的地址是落入不同内存 bank，还是相撞并串行化；甚至决定了一个 tile 是否匹配 Tensor Core 能读取的字节排列。

机器学习程序通常以逻辑形状描述张量。**数据布局**补上了缺失的物理部分：它说明具有逻辑索引 `(i, j, …)` 的元素住在哪里——是在内存、寄存器，还是某种其他硬件存储中。

本章介绍现代 GPU 编程中出现的主要布局。为使讨论易于处理，我们发展出一套紧凑的**记法**，用以描述机器学习系统所遇到的各种情形下的布局。最后以**交换（swizzle）**收尾，这是让对一个 tile 的行向与列向访问同时高效的机制。

## 形状-步长模型

在进入 GPU 专有布局之前，值得从最简单的布局讲起，因为本章其余一切都建立在它之上。其核心只有两样东西：一个**形状**和一组匹配的**步长**。我们把这对写作 `S[(shape) : (strides)]`，要找到某个逻辑索引的所在，就拿该索引与步长做点积。一个行主序的 4×4 矩阵看起来像这样：

```text
S[(4, 4) : (4, 1)]        addr(i, j) = i·4 + j·1

这无非是经典的形状/步长模型的紧凑写法（CuTe 记法的行主序简化），后续一切都由它构建而成。

事实上，你几乎肯定已经用过这个模型。任何写过 PyTorch 或 NumPy 的人都用过，因为这些库里的张量*正是*一个形状加上对一个扁平存储缓冲区的步长：

```python
import torch
t = torch.arange(12).reshape(3, 4)
t.shape        # torch.Size([3, 4])
t.stride()     # (4, 1)        ← exactly S[(3, 4) : (4, 1)]

一旦你以这种方式看待张量，就不难理解为什么那么多"重塑"操作根本不碰数据。它们只是改写步长、在同样的存储上返回一个**视图**，而最清晰的例子是转置，或 permute：

```python
tt = t.permute(1, 0)               # or t.T
tt.shape                           # torch.Size([4, 3])
tt.stride()                        # (1, 4)        ← strides swapped, no data moved
tt.data_ptr() == t.data_ptr()      # True, same bytes

这里 `t.permute(1, 0)` 就是 `S[(4, 3) : (1, 4)]`，落在*同一块*内存上：转置纯粹是步长的改变，一个字节都没动。`reshape` 或 `view` 在连续张量上的故事相同：在旧存储之上的新形状和新步长。（NumPy 的行为完全相同；唯一区别是它的 `.strides` 按字节而非按元素计数。）

这正是布局在 GPU 上的工作方式，本章其余部分其实都是同一个想法的一系列变体：一个 tile 的映射（无论到内存，还是通过我们即将引入的命名轴到通道与寄存器）是对一个固定缓冲区的步长规则，所以重新安排一个 tile 通常是改变*布局*而非拷贝。不过我们应当谨慎对待这一推理的边界。零拷贝的故事对单一线性地址空间上的逻辑视图成立得很干净；在 GPU 上它只在新视图与现有字节排列和归属安排兼容时才适用。一旦你改变了哪个线程或寄存器拥有某个元素，或改变了 SMEM swizzle，你通常就需要真正的数据搬运：加载、存储、shuffle、`ldmatrix`、转置。

## Tile 布局

到目前为止我们描述的是整个张量的布局。然而 GPU 核函数很少一次处理整个矩阵；它们处理更小的 tile，这些 tile 由硬件的不同部分加载、变换并计算。好消息是分块并不要求任何新东西。它仍然只是一个布局，只不过现在多了几个维度。把一个 8×8 矩阵切成 2×4 的 tile，就得到一个 4 维布局，坐标为 `(tile_row, row_in_tile, tile_col, col_in_tile)`，步长选择使每个 tile 保持连续：

```text
S[(4, 2, 2, 4) : (16, 4, 8, 1)]

一个逻辑 `(i, j)` 先变成 `(i//2, i%2, j//4, j%4)`，再穿过步长。值得注意的是，这套记法根本不需要任何特殊的"tile"概念就能表达分块：它与之前的形状-步长模型完全相同，只是把索引拆成了外层与内层坐标。

下方的交互式可视化展示了一个逻辑矩阵索引如何被分解为 tile 坐标，再映射到物理地址。


<iframe src="/books/modern-gpu-programming-for-mlsys/demo/tiled_layout.html" title="Tile layout: interactive address computation" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>

*交互式：点击一个单元格查看其分块索引与地址。*

## 命名轴

到目前为止 `S[...]` 中的每个步长都命名了一个到线性内存的偏移，我们把地址当作那里的一个位置。然而在 GPU 上，数据可以住在不止一个地方：除内存外，一个 tile 可以分布在 warp 通道、线程寄存器，或 TMEM 通道与列上。为统一描述所有这些，我们用**命名轴**扩展记法。想法是让每个步长系数带一个轴标签，说明它在哪个空间中移动：`@m` 表示普通内存，`@laneid` 表示 warp 通道，`@reg` 表示寄存器，`@warpid` 表示 warp，`@TLane` / `@TCol` 表示 TMEM 坐标。有了这些标签，单一布局就不仅能描述数据在内存中的位置，还能描述它如何在操作它的硬件资源上分布。

一旦内存标签显式化，内存中一个行主序的 8×16 tile 就只是

```text
S[(8, 16) : (16@m, 1@m)]

当布局描述的是*分布在线程间*而非排布在内存中的数据时，标签才真正显出价值。取 `S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]`：它不指向线性内存，而是把行和列映射到 lane ID 和每通道寄存器。这里 `laneid` 指 warp 内的 warp 通道索引，即 `thread_index % warp_size`。这正是你将在 {ref}`chap_layout_generations` 中遇到的 Tensor Core 寄存器片段。

下方的交互式可视化展示了布局如何把张量元素分布到 warp 通道与每通道寄存器，而非放置在线性内存中。


<iframe src="/books/modern-gpu-programming-for-mlsys/demo/thread_register.html" title="Thread + register layout via named axes" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>

*交互式：一个基于 `@laneid` 与 `@reg` 的布局；点击一个单元格查看哪个通道/寄存器持有它。*

## 分布式布局

命名轴之所以如此有用，是因为它们让我们能统一描述系统多个层级上的放置，包括*跨整个设备*的放置。我们刚把它们用于单个 GPU 内部的通道与寄存器，但同一想法也向外延伸：`@gpuid_x`、`@gpuid_y` 这样的轴可以说数据住在 GPU 网格中的何处，有了它们，记法就能刻画分布式训练与推理中出现的分片模式。轴尚不能刻画的一件事是*复制*——被拷贝到不止一处数据——所以我们加入记法 `R[n : stride]`，其中 `R` 标记被复制的维度。例如 `R[2 : 1@gpuid_x]` 描述沿 `@gpuid_x` 轴的复制。把两者结合，单一表达式既能把一个张量在 2×2 GPU 网格上分片，又能沿一条轴复制它：

```text
S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)] + R[2 : 1@gpuid_x]

下方演示在一个小 GPU 网格上展示了这种分片与复制合二为一的模式。点击任意单元格查看哪个设备持有它，并观察 `@gpuid_x` 复制如何把一份相同的副本放到配对设备上；按钮在全分片、分片+副本、分片+偏移这几种布局间切换。


<iframe src="/books/modern-gpu-programming-for-mlsys/demo/tile_distributed.html" title="Distributed layout across a GPU mesh" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>

*交互式：一个分布在 2×2 GPU 网格上的布局；点击一个单元格查看哪些设备持有它。*

### 核函数内复制模式：TMEM 中的缩放因子

我们刚才为 GPU 网格引入的复制维度 `R[...]` 不只关乎多个设备。同一构造恰好也能描述完全发生在单个核函数内部的事情：硬件*跨通道广播*的数据。Blackwell 的块缩放 MMA（{ref}`chap_layout_generations`）就是一个好例子。其缩放因子住在 TMEM 中，那里一个 128 行的缩放向量只存在 **32 个 TMEM 通道**里——逻辑行 `r` 落到 TMEM 通道 `r % 32`，`r // 32` 沿列方向走。这 32 个被存储的 TMEM 通道随后**沿 TMEM 的 `TLane` 轴被复制**，从 32 扩展到 128 个 TMEM 通道，使读取 warpgroup 中四个 warp 各自都能在自己那 32 通道的 TMEM 窗口里找到一份副本。这是一个 `warpx4` 广播，我们用复制维度书写。读取本身由这些 warp 的线程执行：

```text
S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]

这给出四个副本，步长为 32 个 TMEM 通道：TMEM 通道 `l`、`l+32`、`l+64`、`l+96` 都持有同一个缩放值。和之前一样，复制维度不携带新数据；它只是说"同一个值，坐在四个 TMEM 通道位置上"，正像刚才 `@gpuid_x` 把一行跨 GPU 网格广播一样。

下方交互式演示把两步一起展示：压缩进 32 个 TMEM 通道，然后 `warpx4` 广播到 128 个读取通道。


<iframe src="/books/modern-gpu-programming-for-mlsys/demo/sf_tmem.html" title="Scale factors in TMEM: packing and warpx4 replication" loading="lazy"
        style="width:100%; min-width:1040px; height:560px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>

*交互式：点击一个缩放因子 `SFA[m, sf]`；它压缩进 TMEM 的通道 `m mod 32`、列 `(m // 32)·4 + sf`，然后沿 `TLane` 轴 `warpx4` 广播到四个通道副本（`l`、`l+32`、`l+64`、`l+96`），每个 warp 的 32 通道窗口各一份。*

每列内部的字节打包（`scale_vec` 的 1X/2X/4X 模式）以及 `cta_group::2` 切分在 {ref}`chap_layout_generations` 中讲述。

已了解 CuTe 的读者可以把本章的记法视为 CuTe 的行主序变体，扩展了显式的硬件命名轴和专门的复制结构。

## Swizzle 布局

本章最后一个布局的存在是为了解决一个特定的硬件问题。GPU 的共享内存被组织成内存 bank，当不同通道落在不同 bank 上时访问最快。当若干通道反而落到*同一* bank 内的不同地址时，硬件别无选择只能把它们串行化，我们就付出一次 **bank 冲突**的代价。

在张量程序中这很难避免，因为内存并非以纯线性顺序访问。处理矩阵时，我们常常需要读同一 tile 的行切片和列切片，这就造成真正的张力：对行向访问高效的布局往往在列向访问时产生 bank 冲突，而有利于列的布局又伤害行。**交换（swizzle）**正是为打破这一张力而设计的技术。

swizzle 背后的想法是置换地址映射，通常通过把列索引与行做 XOR，使行访问和列访问*都*最终散布到各 bank 上。它提供的无冲突保证是具体的：它对匹配的元素位宽、swizzle 模式和访问模式（某个引擎描述符所期望的那种）成立，而对任意的元素位宽或对齐并不成立。

下方的第一个交互式演示使之具体化。点击一个列索引，观察每个元素落入哪个 bank：在左侧的朴素行主序 tile 中，一列把全部八个元素灌进同一个 bank，于是该读取串行成八个周期；在右侧的 XOR-swizzle 布局中，同一列散布到八个不同 bank，一个周期即可读取。


<iframe src="/books/modern-gpu-programming-for-mlsys/demo/swizzle_8x8.html" title="8x8 XOR swizzle" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>

*交互式：一个 8×8 tile，朴素行主序下按列发生 bank 冲突，经 XOR swizzle 后无冲突。*

这个小小的 8×8 例子抓住了核心想法，但真实 GPU 内存比这个玩具图景有更多 bank。要让 swizzle 在满规模下生效，我们不是把整个 tile 当作一个整体对象。相反，我们把内存切成小段，并在每一段内部施加 swizzle 模式。实践中最常见的是 `SWIZZLE_128B`，围绕 128 字节段组织，使同样的行/列重映射技巧自然地嵌入一个 32-bank 内存系统。

下方交互式演示展示了这一个具体的硬件 swizzle `SWIZZLE_128B`，以便在我们跨格式推广之前就能看到逐段重复的模式。


<iframe src="/books/modern-gpu-programming-for-mlsys/demo/swizzle_128B.html" title="SWIZZLE_128B layout" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>

*交互式：128 字节段内的 `SWIZZLE_128B` 模式；逐步查看读取周期，可见 `physical_sector = logical_sector XOR row` 把每一列散布到不同 bank。*

同一想法可以推广到这个 128 字节情形之外。为简化可视化，从现在起我们用一个色块代表一个段，而非画出各个 bank。一般而言，硬件定义一个小的重复**原子**，置换就在其上施加，不同 swizzle 模式选择不同原子大小。`SWIZZLE_128B` 用 8 × 128 B 原子，`SWIZZLE_64B` 用 8 × 64 B 原子，`SWIZZLE_32B` 用 8 × 32 B 原子；整个 tile 再由当前所用原子平铺而成。

最后的交互式演示让你在这些格式间切换（包括一个 16 B 交错模式）、选取一种数据类型，并悬停任意单元格以直接查看一个原子内部的元素排列——这正是推理某条加载/存储指令期望哪种 swizzle 时所需的细节层级。


<iframe src="/books/modern-gpu-programming-for-mlsys/demo/swizzle_atom_general.html" title="Swizzle atom layout per format (128B/64B/32B)" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>

*交互式：选取一个 swizzle 格式（与数据类型）查看其原子形状（8 × N B）；悬停一个单元格查看其元素如何被置换。*

该选哪种模式？经验法则是优先选 tile 能填满的*最大*原子。一个 N 字节原子要求 tile 的连续维度至少为 N 字节，且为其倍数，因此 `SWIZZLE_128B` 只在行长至少 128 字节（即 64 个 `float16` 元素）时适用。当条件满足时它是默认选择，因为其 8 × 128 B 原子覆盖一整条 128 字节 bank 线，从而一次把一列散布到全部 32 个 bank，在 fp16 下同时给 8 行 8 列无冲突访问。但当问题形状迫使连续维度变小时，tile 就填不满一个 128 B 原子，你便退到 `SWIZZLE_64B` 或 `SWIZZLE_32B`——行仍能覆盖的最大原子。

你从不需要手算这些置换地址，值得精确说明的是 swizzle 与 `S[...]` 记法的关系：它*不*属于那张仿射映射。它是叠加在其上的一个独立、非仿射层。`S[...]` 布局把一个元素放在一个线性内存（`@m`）地址上，swizzle 随后置换该地址，在 TIRx 布局 API 中写作 `ComposeLayout(swizzle, tile)`（{ref}`chap_tirx_layout_api`）。你的工作只是在每条触及该 tile 的操作上选一个一致的模式，让复合布局完成其余工作。

这个复合布局也正是硬件所填充的，swizzle 与分块就在此处合流。TMA 描述符是多维的，所以一个三维盒子就能同时描述 tile 的原子分块与每个原子内部的 swizzle；一次 TMA 加载就逐原子地铺设 tile，并在写共享内存时施加 swizzle（{ref}`chap_tma`），无需单独的 swizzle 步骤。每个引擎要求*哪种* swizzle 是世代特定的，那是下一章的主题。

