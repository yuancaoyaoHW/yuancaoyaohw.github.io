// Single source of truth for blog post list.
// Imported by src/pages/index.astro (homepage) and src/pages/blog/index.astro.
// Add a new entry here ONCE — both pages update automatically.
export const blogPosts = [
  {
    title: 'PRESTO：扩散投机解码的前缀对齐树起草',
    href: '/blog/presto/',
    description: 'arXiv 2607.22634 深度解读：扩散语言模型天生适合做多路径候选起草器，但块内位置边际分布对树中前缀选择不敏感（构树阶段前缀盲）。PRESTO 在边际对数分数上叠加前缀校正项（log q_d + λ log ρ_d）+ 优先级树展开，dFlash 平均 1.5× 吞吐加速。含 5 段交互式 SVG 动画：线性 vs 树起草、前缀盲性、λ 校正、Beam Search 增量打分、二次自投机流水线。',
    tag: 'PRESTO · Speculative Decoding',
    date: '2026-07-29',
  },
  {
    title: 'PTStore：把 KV 缓存变成 CDN',
    href: '/blog/ptstore/',
    description: 'arXiv 2607.22648 深度解读：PTStore 借鉴 CDN 思维——增量存储让前缀像 trie 无冗余生长、扁平元数据免遍历查询、热度复制用冗余换 locality、GDSF 驱逐频率×大小、bulk RDMA 零拷贝拉散落张量。长文档 Q&A 比基线快 5–6×，含 7 段交互式 SVG 动画。',
    tag: 'KV Cache · Distributed Inference',
    date: '2026-07-29',
  },
  {
    title: 'RTP-LLM原理与源码分析',
    href: '/blog/rtp-llm-deep-read/',
    description: 'arXiv 2605.29639 深度解读：阿里 RTP-LLM 端到端生产级推理引擎——Prefill-Decode 物理解耦 + 四级层次化 KV Cache + 统一哈希全局调度。论文 × rtp-llm 源码全对照（6097 文件 / 3 并行子代理辅助），含 Mermaid 架构图、CUDA kernel 定位与 TTFT/吞吐/加载基准对比。',
    tag: 'RTP-LLM · Inference Engine',
    date: '2026-07-29',
  },
  {
    title: '推理时共识 · 微调安全的新防线',
    href: '/blog/consensus-decoding/',
    description: 'arXiv 2607.23394 深度解读：Inference-Time Consensus——对每个数据源单独微调 reference model，解码时聚合 next-token 分布，只有源们共识才保留行为。Lemma 1 证明 KL 正则化只是软共识（几何/算术均值，无 veto），含四步交互式 SVG 动画可视化 minimum aggregation 的 veto 机制 + 语义平滑动画。',
    tag: 'LLM Alignment · Fine-tuning Safety',
    date: '2026-07-28',
  },
  {
    title: '全局计算，局部物化 · Sparse Event-KV 内存契约',
    href: '/blog/sparse-event-kv/',
    description: 'arXiv 2607.23693 深度解读：当 source 被丢弃后，下游 event 的 KV 行如何独立承载其状态（semantic materialization）。99:0 donor-following 分裂、trigger/landing/access 三段契约、刻意写入 vs 自然收割对比，含四步交互式 SVG 动画可视化物化机制。',
    tag: 'KV Cache · LLM Serving',
    date: '2026-07-28',
  },
  {
    title: 'KV Cache Offload 领域有影响力论文调研',
    href: '/blog/kv-cache-offload/',
    description: '从 PagedAttention 到 vAttention——KV cache offload 领域 12 篇高影响力论文分层推荐,3 条技术主线(分页管理 / CPU-SSD offload / 压缩池化)与建议阅读顺序。Semantic Paper Radar 技能聚合 arXiv + OpenAlex 检索。',
    tag: 'KV Cache · Offload',
    date: '2026-07-25',
  },
  {
    title: 'Megakernel 与 Rubin：软件抽象与硬件底座',
    href: '/blog/event-tensor-megakernel/',
    description: '整理 megakernel（GPU 编程模型）与 Rubin（NVIDIA 下一代 GPU 架构）的互补关系，含 Event Tensor 原论文（MLSys 2026）Paper Parse 双模深读报告与知乎社区讨论。',
    tag: 'Megakernel · Rubin',
    date: '2026-07-22',
  },
  {
    title: 'Latent MoE 技术报告整理',
    href: '/blog/latent-moe/',
    description: '整理 NVIDIA LatentMoE 原论文 + Nemotron 3 应用 + 知乎社区讨论，含 Paper Parse 双模深读报告（Part A 专业解析 + Part B 核心逻辑提炼）和 MathJax 渲染。',
    tag: 'MoE · NVIDIA',
    date: '2026-07-21',
  },
  {
    title: 'DFlash 原理与 vLLM 适配实现',
    href: '/blog/dflash-vllm/',
    description: '从扩散模型到投机解码——DFlash 原理解析与 vLLM 代码落地。围绕两个核心问题展开：什么是扩散模型，以及为什么 draft 用扩散而 target 不用。含 MathJax 数学公式与 Mermaid 流程图。',
    tag: 'DFlash · vLLM',
    date: '2026-06-30',
  },
  {
    title: 'DeepSpec 与 DSpark 深度讲解',
    href: '/blog/deepspec-dspark/',
    description: 'Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation——含交互式 SVG 动画的技术分析报告，覆盖 DSpark / DFlash / Eagle3 三种算法与 Qwen3 / Gemma4 两大模型族。',
    tag: 'DeepSpec · DSpark',
    date: '2026-06-28',
  },
  {
    title: 'mHC 流形约束超连接论文详解',
    href: '/blog/mhc-hyper-connections/',
    description: 'Manifold-Constrained Hyper-Connections——含交互式动画的 DeepSeek 论文深度解读，让残差流既「宽」又「稳」。',
    tag: 'mHC · Hyper-Connections',
    date: '2026-06-27',
  },
];
