/* PTStore animation engine — Variant A (step-based, total redraw) */
(function () {
  "use strict";

  const SVGNS = "http://www.w3.org/2000/svg";

  class AnimEngine {
    constructor(svgRoot, opts) {
      this.svg = svgRoot;
      this.steps = opts.steps || [];
      this.idx = -1;
      this.timer = null;
      this.speed = 1;
      this.playing = false;
      // dynamic layer
      this.dyn = document.createElementNS(SVGNS, "g");
      this.dyn.setAttribute("id", "dynamic");
      this.svg.appendChild(this.dyn);
      // static layer (drawn once, persistent across steps if needed)
      this.stat = document.createElementNS(SVGNS, "g");
      this.svg.insertBefore(this.stat, this.dyn);
    }
    clear() { while (this.dyn.firstChild) this.dyn.removeChild(this.dyn.firstChild); }
    clearStatic() { while (this.stat.firstChild) this.stat.removeChild(this.stat.firstChild); }
    draw(tag, attrs, text) {
      const el = document.createElementNS(SVGNS, tag);
      for (const k in attrs) el.setAttribute(k, attrs[k]);
      if (text != null) { el.textContent = text; }
      this.dyn.appendChild(el);
      return el;
    }
    drawStatic(tag, attrs, text) {
      const el = document.createElementNS(SVGNS, tag);
      for (const k in attrs) el.setAttribute(k, attrs[k]);
      if (text != null) el.textContent = text;
      this.stat.appendChild(el);
      return el;
    }
    el(id) { return this.svg.querySelector("#" + id); }
    update(idOrNode, attrs, text) {
      const node = typeof idOrNode === "string" ? this.el(idOrNode) : idOrNode;
      if (!node) return;
      if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
      if (text != null) node.textContent = text;
    }
    delay() { return 1100 / this.speed; }
    goTo(i) {
      if (i < 0 || i >= this.steps.length) return;
      this.idx = i;
      const step = this.steps[i];
      this.clear();
      step.render(this);
      const desc = document.getElementById(this.descId);
      if (desc) {
        desc.innerHTML = '<span class="step-counter">' + (i + 1) + " / " + this.steps.length + "</span>" + (step.desc || "");
        if (window.MathJax && MathJax.typesetPromise) {
          MathJax.typesetPromise([desc]).catch(function(){});
        }
      }
    }
    next() { if (this.idx < this.steps.length - 1) this.goTo(this.idx + 1); }
    prev() { if (this.idx > 0) this.goTo(this.idx - 1); }
    reset() { this.pause(); this.goTo(0); }
    play() {
      if (this.playing) { this.pause(); return; }
      if (this.idx === -1) this.goTo(0);
      this.playing = true;
      const btn = this.svg.parentNode.querySelector(".btn-play");
      if (btn) btn.textContent = "⏸ 暂停";
      const self = this;
      this.timer = setInterval(function () {
        if (self.idx >= self.steps.length - 1) {
          self.pause();
          return;
        }
        self.next();
      }, self.delay());
    }
    pause() {
      this.playing = false;
      if (this.timer) { clearInterval(this.timer); this.timer = null; }
      const btn = this.svg.parentNode.querySelector(".btn-play");
      if (btn) btn.textContent = "▶ 播放";
    }
    setSpeed(s) { this.speed = s; if (this.playing) { this.pause(); this.play(); } }
  }

  /* ---------- shared palette ---------- */
  const C = {
    gpu: "#1f6feb", gpuFill: "#ddf4ff",
    host: "#fb8f00", hostFill: "#fff8c5",
    ssd: "#6b6b6b", ssdFill: "#f0f0f0",
    net: "#cf222e", netFill: "#ffebe9",
    ok: "#1a7f37", okFill: "#dafbe4",
    hot: "#cf222e", cold: "#0969da",
    text: "#1a1a1a", muted: "#6b6b6b",
    line: "#d0d7de", line2: "#bbb"
  };

  function rect(e, x, y, w, h, label, fill, stroke, opts) {
    opts = opts || {};
    e.draw("rect", { x: x, y: y, width: w, height: h, rx: opts.rx != null ? opts.rx : 4,
      fill: fill || "none", stroke: stroke || C.line, "stroke-width": opts.sw || 1.5 });
    if (label) {
      e.draw("text", { x: x + w / 2, y: y + h / 2 + (opts.dy || 4), "text-anchor": "middle",
        "font-size": opts.fs || 12, fill: opts.tc || C.text, "font-weight": opts.fw || "normal" }, label);
    }
  }

  function arrow(e, x1, y1, x2, y2, color, label) {
    e.draw("line", { x1: x1, y1: y1, x2: x2, y2: y2, stroke: color || C.net, "stroke-width": 2,
      "marker-end": "url(#arrow)", "stroke-dasharray": "4 3" });
    if (label) {
      e.draw("text", { x: (x1 + x2) / 2, y: (y1 + y2) / 2 - 4, "text-anchor": "middle",
        "font-size": 10, fill: color || C.net }, label);
    }
  }

  function ensureDefs(e) {
    if (e.svg.querySelector("#arrow")) return;
    const defs = document.createElementNS(SVGNS, "defs");
    const m = document.createElementNS(SVGNS, "marker");
    m.setAttribute("id", "arrow");
    m.setAttribute("viewBox", "0 0 10 10");
    m.setAttribute("refX", "8");
    m.setAttribute("refY", "5");
    m.setAttribute("markerWidth", "6");
    m.setAttribute("markerHeight", "6");
    m.setAttribute("orient", "auto");
    const p = document.createElementNS(SVGNS, "path");
    p.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    p.setAttribute("fill", C.net);
    m.appendChild(p);
    defs.appendChild(m);
    e.svg.insertBefore(defs, e.svg.firstChild);
  }

  /* =========================================================
     Scene 1 — 三种缓存作用域对比 (problem)
     ========================================================= */
  function scene1build() {
    return [
      { title: "vLLM Vanilla", desc: "前缀只在 <b>单 GPU 块缓存</b> 内复用。GPU1 算过的前缀，GPU2 要重算。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        // 4 GPUs each with own cache
        for (let i = 0; i < 4; i++) {
          const x = 80 + i * 170;
          rect(e, x, 100, 120, 70, "GPU " + (i + 1) + "\nKV cache", C.gpuFill, C.gpu, { fs: 11, fw: "bold" });
          e.draw("text", { x: x + 60, y: 200, "text-anchor": "middle", "font-size": 11, fill: C.muted }, "独占 HBM");
        }
        e.draw("text", { x: 410, y: 60, "text-anchor": "middle", "font-size": 14, fill: C.text, "font-weight": "bold" }, "vLLM Vanilla：每 GPU 一份 KV 缓存");
        e.draw("text", { x: 410, y: 250, "text-anchor": "middle", "font-size": 12, fill: C.err }, "✗ 跨 GPU/节点 无复用");
      }},
      { title: "vLLM Prefix", desc: "flush 到 <b>主机内存</b>，同节点 GPU 可共享，但仍局限于单节点。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        // host memory bar
        rect(e, 80, 200, 600, 50, "Host Memory (单节点)", C.hostFill, C.host, { fs: 12, fw: "bold" });
        for (let i = 0; i < 4; i++) {
          const x = 110 + i * 150;
          rect(e, x, 100, 100, 60, "GPU " + (i + 1), C.gpuFill, C.gpu, { fs: 11, fw: "bold" });
          arrow(e, x + 50, 160, x + 50, 200, C.host, "flush");
        }
        e.draw("text", { x: 410, y: 60, "text-anchor": "middle", "font-size": 14, fill: C.text, "font-weight": "bold" }, "LMCache / vLLM Prefix：同节点主机内存共享");
        e.draw("text", { x: 410, y: 290, "text-anchor": "middle", "font-size": 12, fill: C.warm }, "△ 跨节点 仍需重算");
      }},
      { title: "PTStore", desc: "<b>跨节点</b> 分布式存储 + 热度复制：节点 A 的前缀，节点 B 可远程拉取并缓存副本。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        // two nodes
        for (let n = 0; n < 2; n++) {
          const x = 60 + n * 380;
          rect(e, x, 80, 320, 220, "", "#fff", C.line, { sw: 1, rx: 8 });
          e.draw("text", { x: x + 160, y: 100, "text-anchor": "middle", "font-size": 13, fill: C.text, "font-weight": "bold" }, "Node " + (n + 1));
          rect(e, x + 30, 120, 260, 50, "Host Mem (owned + replica)", C.hostFill, C.host, { fs: 11, fw: "bold" });
          for (let g = 0; g < 2; g++) {
            rect(e, x + 40 + g * 130, 200, 100, 60, "GPU", C.gpuFill, C.gpu, { fs: 11, fw: "bold" });
          }
        }
        // network link between nodes
        arrow(e, 380, 145, 460, 145, C.net, "RDMA 复制");
        arrow(e, 460, 155, 380, 155, C.net, "");
        e.draw("text", { x: 410, y: 50, "text-anchor": "middle", "font-size": 14, fill: C.ok, "font-weight": "bold" }, "PTStore：跨节点前缀复用 + 本地热度复制");
        e.draw("text", { x: 410, y: 330, "text-anchor": "middle", "font-size": 12, fill: C.ok }, "✓ 5–6× 更高效 (长文档 Q&A)");
      }}
    ];
  }

  /* =========================================================
     Scene 2 — trie 式增量前缀存储 (lcp)
     Layout: title y=30 | tensors y=60-96 | gap | servers y=300-390
     | inner tensors y=310-336 | bottom text y=410-440
     Arrows go from tensor bottom (y=96) straight down to server top
     (y=300) — the middle gap (y=100-290) is kept clear of text so
     arrows never cross labels.
     ========================================================= */
  function scene2build() {
    const SVW = 820, SVH = 460;
    const TENSOR_Y = 60, TENSOR_H = 38;
    const SERVER_Y = 300, SERVER_H = 90;
    const INNER_Y = 312, INNER_H = 26;
    const BOTTOM_Y = 420;
    // 5 tensors centered: total width 5*70 + 4*20 = 430, start x = (820-430)/2 = 195
    const T_START = 195, T_W = 70, T_GAP = 20;
    // 3 servers: each 240 wide, gap 15, start x = (820 - 3*240 - 2*15)/2 = 25
    const S_START = 25, S_W = 240, S_GAP = 15;

    function tensorX(i) { return T_START + i * (T_W + T_GAP); }
    function serverX(s) { return S_START + s * (S_W + S_GAP); }

    function drawServers(e, owned, highlightNew) {
      for (let s = 0; s < 3; s++) {
        const x = serverX(s);
        // server box (no label in rect, draw separately at bottom)
        rect(e, x, SERVER_Y, S_W, SERVER_H, "", C.hostFill, C.host, { fs: 11, fw: "bold" });
        e.draw("text", { x: x + S_W / 2, y: SERVER_Y + SERVER_H - 8, "text-anchor": "middle", "font-size": 11, "font-weight": "bold", fill: C.text }, "Server " + (s + 1) + " (owned)");
        // inner tensors (centered in upper part of server box)
        const items = owned[s];
        const innerW = 36, innerGap = 6;
        const totalW = items.length * innerW + (items.length - 1) * innerGap;
        const innerStart = x + (S_W - totalW) / 2;
        const innerY = SERVER_Y + 12;
        for (let i = 0; i < items.length; i++) {
          const ix = innerStart + i * (innerW + innerGap);
          const isHighlight = highlightNew && s === 1 && (items[i] === "T4" || items[i] === "T5");
          rect(e, ix, innerY, innerW, INNER_H, items[i],
            isHighlight ? C.okFill : "#fff",
            isHighlight ? C.ok : C.host, { fs: 10, fw: "bold" });
        }
      }
    }

    return [
      { title: "新请求到达", desc: "一个新推理请求到达，其 prompt 对应的 KV 张量序列：[T1, T2, T3, T4, T5]" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        e.draw("text", { x: 410, y: 30, "text-anchor": "middle", "font-size": 14, "font-weight": "bold", fill: C.text }, "新对象：T1 T2 T3 T4 T5");
        const labels = ["T1", "T2", "T3", "T4", "T5"];
        for (let i = 0; i < 5; i++) {
          rect(e, tensorX(i), TENSOR_Y, T_W, TENSOR_H, labels[i], C.gpuFill, C.gpu, { fs: 12, fw: "bold" });
        }
        e.draw("text", { x: 410, y: 180, "text-anchor": "middle", "font-size": 13, fill: C.muted }, "↓ 查询 LCP（最长公共前缀）");
        e.draw("text", { x: 410, y: 210, "text-anchor": "middle", "font-size": 12, fill: C.muted }, "(下一步：各 server 本地匹配)");
      }},
      { title: "LCP 查询", desc: "两级并行 reduce：各 server 在本地 owned cache 并行匹配，全局 reduce 得 <b>最长公共前缀 = [T1,T2,T3]</b>。增量 = [T4, T5]。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        // servers at bottom
        const owned = [["T1", "T2", "T3"], ["T1", "T2"], ["T1", "T2", "T3", "T4"]];
        drawServers(e, owned, false);
        // new object on top
        e.draw("text", { x: 410, y: 30, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: C.text }, "新对象：T1 T2 T3 T4 T5");
        const labels = ["T1", "T2", "T3", "T4", "T5"];
        for (let i = 0; i < 5; i++) {
          const isCommon = i < 3;
          rect(e, tensorX(i), TENSOR_Y, T_W, TENSOR_H, labels[i],
            isCommon ? C.okFill : C.gpuFill,
            isCommon ? C.ok : C.gpu, { fs: 12, fw: "bold" });
        }
        // arrows from common tokens straight down to server row (clear vertical path)
        for (let i = 0; i < 3; i++) {
          const tx = tensorX(i) + T_W / 2;
          arrow(e, tx, TENSOR_Y + TENSOR_H, tx, SERVER_Y, C.ok, "");
        }
        // "match" labels placed just below arrows start, left-aligned to avoid crossing
        e.draw("text", { x: tensorX(0) + T_W / 2, y: TENSOR_Y + TENSOR_H + 16, "text-anchor": "middle", "font-size": 10, fill: C.ok }, "match");
        // bottom result text (below servers, clear of arrows)
        e.draw("text", { x: 410, y: BOTTOM_Y, "text-anchor": "middle", "font-size": 12, fill: C.ok, "font-weight": "bold" }, "LCP = [T1, T2, T3]  →  增量 = [T4, T5]");
        e.draw("text", { x: 410, y: BOTTOM_Y + 18, "text-anchor": "middle", "font-size": 11, fill: C.muted }, "查询只传 tensor ID（几字节），不传张量本体");
      }},
      { title: "增量存储", desc: "增量 [T4, T5] 被 <b>随机分配到某一 server</b>（轻量负载均衡、无需同步），成为该 server 的 owned 张量。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        const owned = [["T1", "T2", "T3"], ["T4", "T5"], ["T1", "T2", "T3", "T4"]];
        drawServers(e, owned, true);
        // new object with metadata
        e.draw("text", { x: 410, y: 30, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: C.text }, "新对象的扁平元数据：[T1, T2, T3, T4, T5]");
        const labels = ["T1", "T2", "T3", "T4", "T5"];
        for (let i = 0; i < 5; i++) {
          const isNew = i >= 3;
          rect(e, tensorX(i), TENSOR_Y, T_W, TENSOR_H, labels[i],
            isNew ? C.okFill : C.hostFill,
            isNew ? C.ok : C.host, { fs: 12, fw: "bold" });
        }
        // arrow from T4/T5 (new increment) down to Server 2
        const t4x = tensorX(3) + T_W / 2;
        const s2x = serverX(1) + S_W / 2;
        // vertical then diagonal to server 2 center
        arrow(e, t4x, TENSOR_Y + TENSOR_H, t4x, 200, C.ok, "");
        arrow(e, t4x, 200, s2x, SERVER_Y, C.ok, "store 增量");
        // bottom text (clear of arrows)
        e.draw("text", { x: 410, y: BOTTOM_Y, "text-anchor": "middle", "font-size": 12, fill: C.ok, "font-weight": "bold" }, "Server 2 成为 T4, T5 的 owner");
        e.draw("text", { x: 410, y: BOTTOM_Y + 18, "text-anchor": "middle", "font-size": 11, fill: C.muted }, "前缀像 trie 一样无冗余地分叉生长");
      }}
    ];
  }

  /* =========================================================
     Scene 3 — 架构全景 + store/load 流 (arch)
     ========================================================= */
  function scene3build() {
    return [
      { title: "架构", desc: "每个节点一个 server，聚合 host mem + SSD。Owned cache 保证持久化；Replication cache 缓存热门副本。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        // 3 nodes
        for (let n = 0; n < 3; n++) {
          const x = 30 + n * 260;
          rect(e, x, 80, 230, 280, "", "#fff", C.line, { sw: 1, rx: 8 });
          e.draw("text", { x: x + 115, y: 100, "text-anchor": "middle", "font-size": 12, "font-weight": "bold", fill: C.text }, "Node " + (n + 1));
          // GPU clients
          rect(e, x + 20, 120, 80, 40, "GPU", C.gpuFill, C.gpu, { fs: 11, fw: "bold" });
          rect(e, x + 130, 120, 80, 40, "GPU", C.gpuFill, C.gpu, { fs: 11, fw: "bold" });
          // server
          rect(e, x + 20, 190, 190, 50, "Server (owned + replica)", C.hostFill, C.host, { fs: 10, fw: "bold" });
          // owned vs replica split
          rect(e, x + 20, 250, 90, 30, "Owned", C.okFill, C.ok, { fs: 10, fw: "bold" });
          rect(e, x + 120, 250, 90, 30, "Replica", "#ffe8e8", C.hot, { fs: 10, fw: "bold" });
          // SSD
          rect(e, x + 50, 300, 130, 30, "SSD / PFS", C.ssdFill, C.ssd, { fs: 10, fw: "bold" });
          // GPU-server arrow
          arrow(e, x + 60, 160, x + 60, 190, C.gpu, "");
          arrow(e, x + 170, 160, x + 170, 190, C.gpu, "");
        }
        e.draw("text", { x: 410, y: 50, "text-anchor": "middle", "font-size": 14, "font-weight": "bold", fill: C.text }, "PTStore 分布式架构");
        e.draw("text", { x: 410, y: 385, "text-anchor": "middle", "font-size": 11, fill: C.muted }, "绿色 = 本文贡献：扁平元数据 + 热度复制 + GDSF + bulk RDMA");
      }},
      { title: "Store 流程", desc: "client 发起新对象 store：1) 广播 LCP 查询；2) 各 server 本地并行匹配；3) 增量随机落到某 server，写入 owned cache。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        e.draw("text", { x: 410, y: 30, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: C.text }, "Store 流程");
        // client
        rect(e, 340, 60, 140, 40, "Client (GPU)", C.gpuFill, C.gpu, { fs: 11, fw: "bold" });
        // servers
        for (let s = 0; s < 3; s++) {
          const x = 80 + s * 220;
          rect(e, x, 180, 160, 60, "Server " + (s + 1), C.hostFill, C.host, { fs: 11, fw: "bold" });
        }
        // step 1: broadcast LCP query
        for (let s = 0; s < 3; s++) {
          const x = 80 + s * 220 + 80;
          arrow(e, 410, 100, x, 180, C.accent || "#0969da", "LCP query");
        }
        e.draw("text", { x: 410, y: 130, "text-anchor": "middle", "font-size": 11, fill: "#0969da" }, "① 广播 LCP（只传 tensor ID）");
        // step 2: local match
        e.draw("text", { x: 410, y: 270, "text-anchor": "middle", "font-size": 11, fill: C.muted }, "② 各 server 本地并行匹配 owned cache");
        // step 3: increment to server 2
        arrow(e, 410, 100, 300, 180, C.ok, "increment");
        e.draw("text", { x: 410, y: 310, "text-anchor": "middle", "font-size": 11, fill: C.ok }, "③ 增量随机分配 → Server 2 (owner)");
      }},
      { title: "Load 流程", desc: "client load：遍历对象的 tensor ID 列表，查本地 replication cache；命中的直接用，没命中的按 ID 远程拉（bulk RDMA）。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        e.draw("text", { x: 410, y: 30, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: C.text }, "Load 流程");
        // client
        rect(e, 340, 60, 140, 40, "Client (GPU)", C.gpuFill, C.gpu, { fs: 11, fw: "bold" });
        // local replica cache
        rect(e, 320, 130, 180, 40, "本地 Replica Cache", "#ffe8e8", C.hot, { fs: 10, fw: "bold" });
        // servers
        for (let s = 0; s < 3; s++) {
          const x = 80 + s * 220;
          rect(e, x, 240, 160, 60, "Server " + (s + 1), C.hostFill, C.host, { fs: 11, fw: "bold" });
        }
        // step 1: check local
        arrow(e, 410, 100, 410, 130, C.hot, "① 查本地");
        e.draw("text", { x: 510, y: 155, "font-size": 10, fill: C.ok }, "hit ✓");
        // step 2: remote fetch for misses
        arrow(e, 410, 170, 180, 240, C.net, "② 远程拉 (miss)");
        arrow(e, 410, 170, 400, 240, C.net, "");
        e.draw("text", { x: 410, y: 330, "text-anchor": "middle", "font-size": 11, fill: C.muted }, "命中的免 I/O；未命中的走 bulk RDMA 并行拉");
      }}
    ];
  }

  /* =========================================================
     Scene 4 — 扁平元数据 vs trie 遍历 (metadata)
     ========================================================= */
  function scene4build() {
    return [
      { title: "传统 trie", desc: "要重建对象 O 的组成，必须从根走叶子遍历多层节点，跨节点时还要分布式走 trie——开销随深度增长。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        e.draw("text", { x: 200, y: 40, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: C.err }, "传统 trie：多层遍历");
        // trie nodes
        const nodes = [
          { x: 200, y: 80, l: "T1" },
          { x: 130, y: 150, l: "T2" }, { x: 270, y: 150, l: "T2" },
          { x: 90, y: 220, l: "T3" }, { x: 170, y: 220, l: "T4" }, { x: 270, y: 220, l: "T3" }
        ];
        const edges = [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5]];
        for (const ed of edges) {
          const a = nodes[ed[0]], b = nodes[ed[1]];
          e.draw("line", { x1: a.x, y1: a.y + 14, x2: b.x, y2: b.y - 14, stroke: C.line, "stroke-width": 1.5 });
        }
        for (const n of nodes) {
          rect(e, n.x - 22, n.y - 14, 44, 28, n.l, "#fff", C.line, { fs: 11, fw: "bold" });
        }
        e.draw("text", { x: 200, y: 280, "text-anchor": "middle", "font-size": 11, fill: C.err }, "✗ 跨节点遍历 + 多次 RPC");

        // right side: flat
        e.draw("text", { x: 620, y: 40, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: C.ok }, "PTStore：扁平 ID 列表");
        const ids = ["T1@S2", "T2@S1", "T3@S2", "T4@S3"];
        for (let i = 0; i < ids.length; i++) {
          rect(e, 530 + (i % 2) * 100, 70 + Math.floor(i / 2) * 50, 90, 36, ids[i], C.okFill, C.ok, { fs: 11, fw: "bold" });
        }
        e.draw("text", { x: 620, y: 195, "text-anchor": "middle", "font-size": 11, fill: C.ok }, "✓ 一次遍历 + 本地查");
        e.draw("text", { x: 620, y: 215, "text-anchor": "middle", "font-size": 10, fill: C.muted }, "ID = 张量 + owner server");
      }},
      { title: "两级并行 reduce", desc: "<b>第一级</b>：每个 server 在本地 owned cache 并行匹配。<b>第二级</b>：全局 reduce 汇总最长匹配。因为只传 ID（几字节），大规模分布式也带宽轻量。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        e.draw("text", { x: 410, y: 40, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: C.text }, "LCP 查询：两级并行 reduce");
        // client
        rect(e, 350, 70, 120, 36, "Client", C.gpuFill, C.gpu, { fs: 11, fw: "bold" });
        // servers
        for (let s = 0; s < 4; s++) {
          const x = 70 + s * 180;
          rect(e, x, 170, 140, 50, "Server " + (s + 1) + "\n本地匹配", C.hostFill, C.host, { fs: 10, fw: "bold" });
          arrow(e, 410, 106, x + 70, 170, "#0969da", "");
        }
        e.draw("text", { x: 410, y: 140, "text-anchor": "middle", "font-size": 11, fill: "#0969da" }, "① 各 server 本地并行匹配");
        // reduce
        for (let s = 0; s < 4; s++) {
          const x = 70 + s * 180;
          arrow(e, x + 70, 220, 410, 270, C.ok, "");
        }
        rect(e, 320, 270, 180, 36, "全局 reduce → LCP", C.okFill, C.ok, { fs: 11, fw: "bold" });
        e.draw("text", { x: 410, y: 320, "text-anchor": "middle", "font-size": 11, fill: C.ok }, "② 汇总最长公共前缀");
      }}
    ];
  }

  /* =========================================================
     Scene 5 — 热度复制与 trade-off (replication)
     ========================================================= */
  function scene5build() {
    return [
      { title: "冷启动", desc: "新对象 store 后，增量分散在多个 owner 上。Load 时需联系多个远程 owner——即使并行，延迟也高于本地 host memory。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        // local node
        rect(e, 40, 80, 200, 200, "", "#fff", C.line, { rx: 8 });
        e.draw("text", { x: 140, y: 100, "text-anchor": "middle", "font-size": 12, "font-weight": "bold", fill: C.text }, "本地 Node");
        rect(e, 60, 120, 160, 40, "GPU", C.gpuFill, C.gpu, { fs: 11, fw: "bold" });
        rect(e, 60, 180, 160, 40, "Host Mem (空)", "#fff", C.host, { fs: 10, fw: "bold" });
        rect(e, 60, 240, 160, 30, "Replica Cache (空)", "#fff", C.hot, { fs: 9, fw: "bold" });
        // remote owners
        for (let i = 0; i < 3; i++) {
          const x = 320 + i * 150;
          rect(e, x, 120, 120, 60, "Remote\nOwner " + (i + 1), C.hostFill, C.host, { fs: 10, fw: "bold" });
          arrow(e, 220, 140, x, 140, C.net, "RDMA");
        }
        e.draw("text", { x: 410, y: 50, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: C.err }, "冷启动：每个张量都要远程拉");
        e.draw("text", { x: 410, y: 310, "text-anchor": "middle", "font-size": 11, fill: C.net }, "✗ 3 次远程 RDMA = 高延迟");
      }},
      { title: "热度复制", desc: "访问频次超过阈值的张量被复制到本地 replica cache。后续 load 命中本地，免远程 I/O。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        rect(e, 40, 80, 200, 200, "", "#fff", C.line, { rx: 8 });
        e.draw("text", { x: 140, y: 100, "text-anchor": "middle", "font-size": 12, "font-weight": "bold", fill: C.text }, "本地 Node");
        rect(e, 60, 120, 160, 40, "GPU", C.gpuFill, C.gpu, { fs: 11, fw: "bold" });
        // replica cache now has hot tensors
        rect(e, 60, 180, 160, 40, "Host Mem", C.hostFill, C.host, { fs: 10, fw: "bold" });
        rect(e, 60, 180, 45, 40, "T1", "#ffe8e8", C.hot, { fs: 9, fw: "bold" });
        rect(e, 105, 180, 45, 40, "T2", "#ffe8e8", C.hot, { fs: 9, fw: "bold" });
        rect(e, 60, 240, 160, 30, "Replica Cache (T1, T2 副本)", "#ffe8e8", C.hot, { fs: 9, fw: "bold" });
        // only 1 remote fetch for cold T3
        rect(e, 320, 120, 120, 60, "Remote\nOwner 3", C.hostFill, C.host, { fs: 10, fw: "bold" });
        arrow(e, 220, 140, 320, 140, C.net, "RDMA (T3)");
        e.draw("text", { x: 410, y: 50, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: C.ok }, "热度复制后：T1, T2 命中本地");
        e.draw("text", { x: 410, y: 310, "text-anchor": "middle", "font-size": 11, fill: C.ok }, "✓ 仅 1 次远程 RDMA（冷张量 T3）");
      }},
      { title: "trade-off", desc: "Replica cache 和 owned cache <b>抢同一块内存</b>。复制太多 → owned 容量缩小 → 冷张量更可能落到远端慢层。PTStore 用 <b>可配置阈值 + 上下限</b> 平衡。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        // memory bar split
        rect(e, 60, 120, 700, 80, "", "#fff", C.line, { rx: 4 });
        // owned portion
        rect(e, 60, 120, 420, 80, "", C.okFill, C.ok, { sw: 1 });
        e.draw("text", { x: 270, y: 165, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: C.ok }, "Owned Cache (持久化保证)");
        // replica portion
        rect(e, 480, 120, 280, 80, "", "#ffe8e8", C.hot, { sw: 1 });
        e.draw("text", { x: 620, y: 165, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: C.hot }, "Replica Cache (可驱逐)");
        // threshold marker
        e.draw("line", { x1: 480, y1: 110, x2: 480, y2: 210, stroke: "#0969da", "stroke-width": 2, "stroke-dasharray": "4 3" });
        e.draw("text", { x: 480, y: 105, "text-anchor": "middle", "font-size": 10, fill: "#0969da" }, "阈值");
        // lower/upper limits
        e.draw("line", { x1: 540, y1: 220, x2: 540, y2: 240, stroke: C.muted, "stroke-width": 1 });
        e.draw("text", { x: 540, y: 255, "text-anchor": "middle", "font-size": 9, fill: C.muted }, "下限");
        e.draw("line", { x1: 740, y1: 220, x2: 740, y2: 240, stroke: C.muted, "stroke-width": 1 });
        e.draw("text", { x: 740, y: 255, "text-anchor": "middle", "font-size": 9, fill: C.muted }, "上限");
        e.draw("text", { x: 410, y: 50, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: C.text }, "内存 trade-off：复制 vs owned");
        e.draw("text", { x: 410, y: 290, "text-anchor": "middle", "font-size": 11, fill: C.warm }, "先驱逐 replica（触及下限前），再 flush owned 到慢层");
      }}
    ];
  }

  /* =========================================================
     Scene 6 — LRU vs GDSF (eviction) — REDESIGNED
     One consistent cache layout across all steps: 4 tensor slots
     in a row, same positions throughout. Each step highlights
     which tensor gets evicted + shows the consequence.
     Step 1: the prefix structure (why front = high freq)
     Step 2: LRU evicts T_front (bad — kills hot tensor)
     Step 3: GDSF evicts T_big (good — score-based)
     ========================================================= */
  function scene6build() {
    const SVW = 820, SVH = 400;
    // 4 slots centered: each 150 wide, gap 20, total = 4*150+3*20=660, start=80
    const S_W = 150, S_H = 80, S_GAP = 20;
    const S_START = (SVW - 4 * S_W - 3 * S_GAP) / 2;
    const SLOT_Y = 130;
    const TITLE_Y = 30, SUBTITLE_Y = 60;
    const EVICT_Y = 250, RESULT_Y = 300, NOTE_Y = 330;

    function slotX(i) { return S_START + i * (S_W + S_GAP); }

    // tensor data: name, freq, size, desc, color, fill
    // T_front/T_mid = high freq (hot), T_tail = medium, T_big = low freq large
    const tensors = [
      { name: "T_front", freq: "高频", size: "小", desc: "公共前缀", c: C.hot, cf: "#ffe8e8", score: 5.0 },
      { name: "T_mid",   freq: "高频", size: "小", desc: "中间段",   c: C.hot, cf: "#ffe8e8", score: 4.5 },
      { name: "T_tail",  freq: "中频", size: "中", desc: "尾部",     c: C.warm, cf: "#fff8c5", score: 1.8 },
      { name: "T_big",   freq: "低频", size: "大", desc: "大块冷数据", c: C.cold, cf: "#ddf4ff", score: 0.2 }
    ];

    function drawCache(e, evictIdx, evictColor, showScores) {
      // cache container border
      e.draw("rect", { x: S_START - 12, y: SLOT_Y - 12, width: 4 * S_W + 3 * S_GAP + 24, height: S_H + 24, rx: 6, fill: "#fbfbfa", stroke: C.line, "stroke-width": 1, "stroke-dasharray": "6 4" });
      e.draw("text", { x: S_START, y: SLOT_Y - 18, "font-size": 10, fill: C.muted }, "Replica Cache（已满）");
      for (let i = 0; i < 4; i++) {
        const t = tensors[i];
        const x = slotX(i);
        const isEvict = (i === evictIdx);
        const fill = isEvict ? "#f0f0f0" : t.cf;
        const stroke = isEvict ? "#999" : t.c;
        const opacity = isEvict ? 0.4 : 1;
        // box
        e.draw("rect", { x: x, y: SLOT_Y, width: S_W, height: S_H, rx: 4, fill: fill, stroke: stroke, "stroke-width": isEvict ? 1 : 2, opacity: opacity });
        // name
        e.draw("text", { x: x + S_W / 2, y: SLOT_Y + 20, "text-anchor": "middle", "font-size": 12, "font-weight": "bold", fill: isEvict ? "#999" : C.text }, t.name);
        // freq + size
        e.draw("text", { x: x + S_W / 2, y: SLOT_Y + 38, "text-anchor": "middle", "font-size": 10, fill: isEvict ? "#999" : t.c }, t.freq + " · " + t.size);
        // desc
        e.draw("text", { x: x + S_W / 2, y: SLOT_Y + 54, "text-anchor": "middle", "font-size": 9, fill: isEvict ? "#bbb" : C.muted }, t.desc);
        // score (step 3 only)
        if (showScores) {
          e.draw("text", { x: x + S_W / 2, y: SLOT_Y + 70, "text-anchor": "middle", "font-size": 9, fill: isEvict ? "#999" : C.muted }, "score=" + t.score);
        }
        // evict arrow pointing down from evicted slot
        if (isEvict) {
          arrow(e, x + S_W / 2, SLOT_Y + S_H + 8, x + S_W / 2, EVICT_Y, evictColor, "");
          // X mark on evicted box
          e.draw("text", { x: x + S_W / 2, y: SLOT_Y - 2, "text-anchor": "middle", "font-size": 14, "font-weight": "bold", fill: evictColor }, "✗");
        }
      }
    }

    return [
      { title: "前缀结构", desc: "前缀结构里靠前的张量天然高频：它们属于更多对象的公共前缀。T_front 和 T_mid 是高频热点，T_big 是低频大块冷数据。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        e.draw("text", { x: SVW / 2, y: TITLE_Y, "text-anchor": "middle", "font-size": 14, "font-weight": "bold", fill: C.text }, "前缀结构：位置决定访问频率");
        e.draw("text", { x: SVW / 2, y: SUBTITLE_Y, "text-anchor": "middle", "font-size": 11, fill: C.muted }, "靠前的张量属于更多对象的公共前缀 → 天然高频");
        drawCache(e, -1, null, false);
        // frequency bars under each slot
        for (let i = 0; i < 4; i++) {
          const t = tensors[i];
          const x = slotX(i) + S_W / 2;
          const barW = 60, barH = 6;
          const freqRatio = t.score / 5.0;
          e.draw("rect", { x: x - barW / 2, y: SLOT_Y + S_H + 16, width: barW, height: barH, rx: 2, fill: "#eee", stroke: C.line, "stroke-width": 0.5 });
          e.draw("rect", { x: x - barW / 2, y: SLOT_Y + S_H + 16, width: barW * freqRatio, height: barH, rx: 2, fill: t.c });
        }
        e.draw("text", { x: SVW / 2, y: RESULT_Y, "text-anchor": "middle", "font-size": 11, fill: C.muted }, "↑ 频率条：T_front/T_mid 高频，T_big 低频");
        e.draw("text", { x: SVW / 2, y: NOTE_Y, "text-anchor": "middle", "font-size": 11, fill: C.warm }, "缓存已满，新张量要进来 → 该驱逐谁？");
      }},
      { title: "LRU 驱逐", desc: "LRU 只看「最近用没用」。T_big 刚被访问过（虽然低频），LRU 反而驱逐了最久没动但高频的 <b>T_front</b>——热点被杀。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        e.draw("text", { x: SVW / 2, y: TITLE_Y, "text-anchor": "middle", "font-size": 14, "font-weight": "bold", fill: C.err }, "LRU：只看时间，不看频率");
        e.draw("text", { x: SVW / 2, y: SUBTITLE_Y, "text-anchor": "middle", "font-size": 11, fill: C.muted }, "「最近用没用」≠ 「该不该留」");
        // LRU evicts T_front (index 0) — most recently untouched despite high freq
        drawCache(e, 0, C.err, false);
        // "LRU 驱逐" label next to arrow
        const ex = slotX(0) + S_W / 2;
        e.draw("text", { x: ex + 30, y: EVICT_Y - 15, "font-size": 11, fill: C.err, "font-weight": "bold" }, "LRU 驱逐");
        e.draw("text", { x: SVW / 2, y: RESULT_Y, "text-anchor": "middle", "font-size": 12, fill: C.err, "font-weight": "bold" }, "✗ 杀掉了高频的 T_front");
        e.draw("text", { x: SVW / 2, y: NOTE_Y, "text-anchor": "middle", "font-size": 11, fill: C.muted }, "下次访问 T_front → 被迫远程拉（高延迟）");
      }},
      { title: "GDSF 驱逐", desc: "GDSF 按 <code>frequency × (1/size)</code> 排序：高频小张量 score 高、优先保留；低频大张量 score 低、先驱逐。<b>T_big 被驱逐</b>。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        e.draw("text", { x: SVW / 2, y: TITLE_Y, "text-anchor": "middle", "font-size": 14, "font-weight": "bold", fill: C.ok }, "GDSF：频率 × 大小");
        e.draw("text", { x: SVW / 2, y: SUBTITLE_Y, "text-anchor": "middle", "font-size": 11, fill: C.muted }, "score = frequency ÷ size  →  低分先驱逐");
        // GDSF evicts T_big (index 3) — lowest score
        drawCache(e, 3, C.ok, true);
        const ex = slotX(3) + S_W / 2;
        e.draw("text", { x: ex - 10, y: EVICT_Y - 15, "text-anchor": "end", "font-size": 11, fill: C.ok, "font-weight": "bold" }, "GDSF 驱逐");
        e.draw("text", { x: SVW / 2, y: RESULT_Y, "text-anchor": "middle", "font-size": 12, fill: C.ok, "font-weight": "bold" }, "✓ 驱逐低频大张量 T_big");
        e.draw("text", { x: SVW / 2, y: NOTE_Y, "text-anchor": "middle", "font-size": 11, fill: C.muted }, "高频小张量保留 → 减少远程 I/O");
      }}
    ];
  }

  /* =========================================================
     Scene 7 — bulk RDMA 零拷贝 (rdma)
     ========================================================= */
  function scene7build() {
    return [
      { title: "逐段拷贝", desc: "朴素做法：把分散的远程张量逐个拷到本地连续区，再传一次。多次 RDMA setup + 内存拷贝。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        e.draw("text", { x: 410, y: 40, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: C.err }, "逐段拷贝：多次 setup + 拷贝" );
        // remote server with scattered tensors
        rect(e, 40, 80, 200, 180, "Remote Server", C.hostFill, C.host, { fs: 11, fw: "bold" });
        const scattered = [{ x: 60, y: 110, l: "T1" }, { x: 130, y: 150, l: "T2" }, { x: 70, y: 200, l: "T3" }];
        for (const s of scattered) {
          rect(e, s.x, s.y, 50, 30, s.l, "#fff", C.host, { fs: 10, fw: "bold" });
        }
        // local buffer
        rect(e, 500, 130, 220, 60, "本地连续缓冲区", C.gpuFill, C.gpu, { fs: 11, fw: "bold" });
        // 3 separate RDMA ops
        for (let i = 0; i < 3; i++) {
          arrow(e, 240, 95 + i * 45, 500, 145 + i * 10, C.err, "RDMA " + (i + 1));
        }
        e.draw("text", { x: 410, y: 290, "text-anchor": "middle", "font-size": 11, fill: C.err }, "✗ 3 次 RDMA setup + 3 次拷贝" );
      }},
      { title: "bulk RDMA 零拷贝", desc: "PTStore：每个张量给 (offset, size) 段，<b>一次 RDMA RPC</b> 并行读所有散落段，直接写到目标 GPU 内存——零拷贝。RDMA 段随张量缓存，省下次 setup。" , render: function (e) {
        ensureDefs(e);
        e.clearStatic();
        e.draw("text", { x: 410, y: 40, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: C.ok }, "bulk RDMA：一次 RPC，零拷贝" );
        // remote server with scattered tensors
        rect(e, 40, 80, 200, 180, "Remote Server", C.hostFill, C.host, { fs: 11, fw: "bold" });
        const scattered = [{ x: 60, y: 110, l: "T1" }, { x: 130, y: 150, l: "T2" }, { x: 70, y: 200, l: "T3" }];
        for (const s of scattered) {
          rect(e, s.x, s.y, 50, 30, s.l, "#fff", C.host, { fs: 10, fw: "bold" });
          // segment markers
          e.draw("text", { x: s.x + 25, y: s.y - 5, "text-anchor": "middle", "font-size": 8, fill: C.ok }, "(off,size)");
        }
        // GPU memory (target, scattered write)
        rect(e, 500, 80, 220, 180, "GPU Memory (scattered write)", C.gpuFill, C.gpu, { fs: 11, fw: "bold" });
        // single bulk RDMA
        arrow(e, 240, 170, 500, 170, C.ok, "1× bulk RDMA");
        // scattered destinations in GPU mem
        const dests = [{ x: 520, y: 110, l: "T1" }, { x: 600, y: 150, l: "T2" }, { x: 530, y: 200, l: "T3" }];
        for (const d of dests) {
          rect(e, d.x, d.y, 50, 30, d.l, "#fff", C.gpu, { fs: 10, fw: "bold" });
        }
        // cached segment table
        rect(e, 500, 270, 220, 30, "RDMA 段表 (cached)", C.codeBg || "#f6f8fa", C.muted, { fs: 10, fw: "bold" });
        e.draw("text", { x: 410, y: 330, "text-anchor": "middle", "font-size": 11, fill: C.ok }, "✓ 1 次 RPC，直接写 GPU，段表复用" );
      }}
    ];
  }

  /* ---------- wiring ---------- */
  const builders = {
    "svg1": scene1build,
    "svg2": scene2build,
    "svg3": scene3build,
    "svg4": scene4build,
    "svg5": scene5build,
    "svg6": scene6build,
    "svg7": scene7build
  };
  const engines = {};

  function initScene(svgId) {
    if (engines[svgId]) return engines[svgId];
    const svg = document.getElementById(svgId);
    if (!svg) return null;
    const buildFn = builders[svgId];
    if (!buildFn) return null;
    const eng = new AnimEngine(svg, { steps: buildFn() });
    eng.descId = "desc" + svgId.replace("svg", "");
    engines[svgId] = eng;
    eng.goTo(0);
    return eng;
  }

  document.addEventListener("DOMContentLoaded", function () {
    // init all immediately (avoid pitfall #7: blank SVGs until scrolled)
    Object.keys(builders).forEach(function (svgId) { initScene(svgId); });

    // bind controls per frame
    document.querySelectorAll(".anim-frame").forEach(function (frame) {
      const svg = frame.querySelector("svg");
      if (!svg) return;
      const svgId = svg.id;
      frame.addEventListener("click", function (ev) {
        const eng = engines[svgId];
        if (!eng) return;
        const t = ev.target;
        if (t.classList.contains("btn-play")) eng.play();
        else if (t.classList.contains("btn-next")) eng.next();
        else if (t.classList.contains("btn-prev")) eng.prev();
        else if (t.classList.contains("btn-reset")) eng.reset();
      });
      const speed = frame.querySelector(".speed input");
      if (speed) {
        speed.addEventListener("input", function () {
          eng.setSpeed(parseFloat(speed.value));
        });
      }
    });

    // scroll-spy TOC
    const tocLinks = document.querySelectorAll(".toc a[href^='#']");
    const observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          tocLinks.forEach(function (l) { l.classList.remove("active"); });
          const link = document.querySelector('.toc a[href="#' + id + '"]');
          if (link) { link.classList.add("active"); link.scrollIntoView({ block: "nearest" }); }
        }
      });
    }, { rootMargin: "-30% 0px -50% 0px", threshold: 0 });
    document.querySelectorAll("section[id]").forEach(function (s) { observer.observe(s); });

    // back to top
    const btt = document.getElementById("btt");
    window.addEventListener("scroll", function () {
      if (window.scrollY > 600) btt.classList.add("show");
      else btt.classList.remove("show");
    });
    btt.addEventListener("click", function () { window.scrollTo({ top: 0, behavior: "smooth" }); });

    // typeset math after a beat
    setTimeout(function () {
      if (window.MathJax && MathJax.typesetPromise) {
        MathJax.typesetPromise().catch(function () {});
      }
    }, 500);
  });
})();
