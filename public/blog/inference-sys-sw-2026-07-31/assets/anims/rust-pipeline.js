/* 动画① Rust 服务层 vs Python 服务层 — Token 流转对比
 * 左半：Python 路径（GIL 瓶颈，多次分配）
 * 右半：Rust 路径（无锁 shard，零分配稳态）
 */
(function () {
  function svgFactory() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 720 380");
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    defs.innerHTML =
      '<marker id="arr1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">' +
      '<path d="M 0 0 L 10 5 L 0 10 z" fill="#8b949e"/></marker>' +
      '<marker id="arr1g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">' +
      '<path d="M 0 0 L 10 5 L 0 10 z" fill="#3fb950"/></marker>' +
      '<marker id="arr1p" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">' +
      '<path d="M 0 0 L 10 5 L 0 10 z" fill="#bc8cff"/></marker>';
    svg.appendChild(defs);
    const bg = document.createElementNS("http://www.w3.org/2000/svg", "g");
    bg.setAttribute("id", "static");
    svg.appendChild(bg);
    const dyn = document.createElementNS("http://www.w3.org/2000/svg", "g");
    dyn.setAttribute("id", "dynamic");
    svg.appendChild(dyn);
    return svg;
  }

  function buildSteps() {
    const PY = { x: 20, y: 40, w: 320, h: 320, label: "Python 服务层" };
    const RS = { x: 380, y: 40, w: 320, h: 320, label: "Rust 服务层" };
    const STAGES = [
      { py: { x: 40, y: 80, w: 100, h: 40, l: "HTTP API" }, rs: { x: 400, y: 80, w: 100, h: 40, l: "HTTP API (Rust)" } },
      { py: { x: 40, y: 140, w: 100, h: 40, l: "Tokenizer" }, rs: { x: 400, y: 140, w: 100, h: 40, l: "Tokenizer Pool" } },
      { py: { x: 40, y: 200, w: 100, h: 40, l: "Scheduler" }, rs: { x: 400, y: 200, w: 100, h: 40, l: "Egress Ring" } },
      { py: { x: 40, y: 260, w: 100, h: 40, l: "Detokenizer" }, rs: { x: 400, y: 260, w: 100, h: 40, l: "Detok Shards" } }
    ];

    function drawBase(e) {
      e.clear();
      // Container boxes
      [PY, RS].forEach(function (c) {
        e.draw("rect", { x: c.x, y: c.y, width: c.w, height: c.h, rx: 10,
          fill: "#161b22", stroke: "#30363d", "stroke-width": 1.5, "stroke-dasharray": "5,3" });
        e.draw("text", { x: c.x + c.w / 2, y: c.y + 22, "text-anchor": "middle",
          "font-size": 14, "font-weight": 700, fill: c === PY ? "#8b949e" : "#bc8cff" }, c.label);
      });
      // Divider
      e.draw("line", { x1: 360, y1: 50, x2: 360, y2: 350, stroke: "#30363d", "stroke-width": 1, "stroke-dasharray": "3,3" });
      e.draw("text", { x: 360, y: 370, "text-anchor": "middle", "font-size": 11, fill: "#6e7681" }, "← Python GIL | 无锁 shard →");
    }

    function drawStages(e, highlightPy, highlightRs) {
      STAGES.forEach(function (s, i) {
        var isPyHi = highlightPy === i;
        var isRsHi = highlightRs === i;
        // Python side
        e.draw("rect", { x: s.py.x, y: s.py.y, width: s.py.w, height: s.py.h, rx: 6,
          fill: isPyHi ? "#3d2418" : "#1c2128", stroke: isPyHi ? "#db6d28" : "#30363d", "stroke-width": isPyHi ? 2.5 : 1.5,
          class: isPyHi ? "tween" : "" });
        e.draw("text", { x: s.py.x + s.py.w / 2, y: s.py.y + 24, "text-anchor": "middle",
          "font-size": 12, "font-weight": isPyHi ? 700 : 500, fill: isPyHi ? "#db6d28" : "#8b949e" }, s.py.l);
        // Rust side
        e.draw("rect", { x: s.rs.x, y: s.rs.y, width: s.rs.w, height: s.rs.h, rx: 6,
          fill: isRsHi ? "#1a2d1a" : "#1c2128", stroke: isRsHi ? "#3fb950" : "#30363d", "stroke-width": isRsHi ? 2.5 : 1.5,
          class: isRsHi ? "tween" : "" });
        e.draw("text", { x: s.rs.x + s.rs.w / 2, y: s.rs.y + 24, "text-anchor": "middle",
          "font-size": 12, "font-weight": isRsHi ? 700 : 500, fill: isRsHi ? "#3fb950" : "#8b949e" }, s.rs.l);
      });
    }

    function drawGIL(e, active) {
      e.draw("rect", { x: 160, y: 80, width: 50, height: 220, rx: 8,
        fill: active ? "#3d2418" : "#161b22", stroke: active ? "#f85149" : "#30363d",
        "stroke-width": active ? 2.5 : 1.5, class: "tween" });
      e.draw("text", { x: 185, y: 195, "text-anchor": "middle", "font-size": 11, "font-weight": 600,
        fill: active ? "#f85149" : "#6e7681", transform: "rotate(-90 185 195)" }, active ? "GIL ⚡ BLOCKED" : "GIL");
    }

    function drawShards(e, active) {
      // Rust shard routing — no lock
      for (var i = 0; i < 4; i++) {
        var x = 520 + i * 45;
        e.draw("rect", { x: x, y: 80, width: 35, height: 220, rx: 4,
          fill: active ? "#1a2d1a" : "#161b22", stroke: active ? "#3fb950" : "#30363d", "stroke-width": 1.5, class: "tween" });
        e.draw("text", { x: x + 17, y: 195, "text-anchor": "middle", "font-size": 9,
          fill: active ? "#3fb950" : "#6e7681", transform: "rotate(-90 " + (x+17) + " 195)" }, "shard " + i);
      }
      if (active) {
        e.draw("text", { x: 600, y: 320, "text-anchor": "middle", "font-size": 11, "font-weight": 600, fill: "#3fb950" }, "↑ 无锁分片 ↑");
      }
    }

    var steps = [
      { title: "对比开始：Python vs Rust 服务层", desc: "同样的 4 层链路，架构完全不同",
        render(e) { drawBase(e); drawStages(e, -1, -1); drawGIL(e, false); drawShards(e, false); }
      },
      { title: "Python 路径：Token 进入 → HTTP API", desc: "GIL 串行化所有请求的 token 处理",
        render(e) { drawBase(e); drawStages(e, 0, -1); drawGIL(e, true); drawShards(e, false);
          e.flyPacket("pkt1", 30, 90, 40, 100, "tok", "#fff8c5", "#d29922"); }
      },
      { title: "Python：Tokenizer 分配对象 + 持锁", desc: "每次 tokenize 创建 Python 对象，GIL 阻塞并行",
        render(e) { drawBase(e); drawStages(e, 1, -1); drawGIL(e, true); drawShards(e, false);
          e.flyPacket("pkt2", 40, 150, 40, 160, "ids", "#fff8c5", "#d29922"); }
      },
      { title: "Rust 路径：同一请求 → HTTP API (Rust)", desc: "零分配解析，直接写入 ring buffer",
        render(e) { drawBase(e); drawStages(e, -1, 0); drawGIL(e, false); drawShards(e, true);
          e.flyPacket("pkt3", 390, 90, 400, 100, "tok", "#ddf4ff", "#58a6ff"); }
      },
      { title: "Rust：Tokenizer Pool — pinned thread", desc: "OS 线程池从 flume channel 拉取，无 GIL",
        render(e) { drawBase(e); drawStages(e, -1, 1); drawGIL(e, false); drawShards(e, true);
          e.flyPacket("pkt4", 400, 150, 400, 160, "ids", "#ddf4ff", "#58a6ff"); }
      },
      { title: "Rust：Egress Ring → Detok Shards", desc: "按 Rid::shard 哈希路由，每 shard 单写入者——无锁",
        render(e) { drawBase(e); drawStages(e, -1, 3); drawGIL(e, false); drawShards(e, true);
          e.flyPacket("pkt5", 400, 270, 520, 195, "out", "#ddf4ff", "#3fb950");
          e.draw("text", { x: 470, y: 210, "font-size": 10, fill: "#3fb950", "font-weight": 600 }, "hash → shard"); }
      },
      { title: "结果：Rust 路径稳态零分配", desc: "Python 每次请求分配+释放对象；Rust Vec::clear 复用 bucket",
        render(e) { drawBase(e); drawStages(e, -1, -1); drawGIL(e, false); drawShards(e, true);
          e.draw("rect", { x: 380, y: 40, width: 320, height: 320, rx: 10, fill: "none", stroke: "#3fb950", "stroke-width": 3, class: "tween" });
          e.draw("text", { x: 540, y: 380, "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: "#3fb950" }, "✓ 零分配稳态 · 无锁 · pinned thread"); }
      }
    ];
    return steps;
  }

  // Register with global animation system
  window.INF_ANIMS = window.INF_ANIMS || {};
  window.INF_ANIMS[1] = { factory: svgFactory, steps: buildSteps };
})();
