/* 动画③ KV Cache 生产硬化 — EFA 竞态修复 + Token Dropping
 * 上半: Mooncake EFA CUDA context 竞态
 * 下半: LMCache Query ring buffer + smart token dropping
 */
(function () {
  function svgFactory() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 720 420");
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    defs.innerHTML =
      '<marker id="arr3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">' +
      '<path d="M 0 0 L 10 5 L 0 10 z" fill="#8b949e"/></marker>' +
      '<marker id="arr3r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">' +
      '<path d="M 0 0 L 10 5 L 0 10 z" fill="#f85149"/></marker>' +
      '<marker id="arr3g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">' +
      '<path d="M 0 0 L 10 5 L 0 10 z" fill="#3fb950"/></marker>';
    svg.appendChild(defs);
    var bg = document.createElementNS("http://www.w3.org/2000/svg", "g");
    bg.setAttribute("id", "static");
    svg.appendChild(bg);
    var dyn = document.createElementNS("http://www.w3.org/2000/svg", "g");
    dyn.setAttribute("id", "dynamic");
    svg.appendChild(dyn);
    return svg;
  }

  function buildSteps() {
    var steps = [
      { title: "KV Cache 生产硬化", desc: "Mooncake EFA 竞态修复 + LMCache 精准 token dropping",
        render(e) {
          e.clear();
          e.draw("rect", { x: 20, y: 20, width: 680, height: 180, rx: 10, fill: "#161b22", stroke: "#db6d28", "stroke-width": 2, "stroke-dasharray": "5,3" });
          e.draw("text", { x: 360, y: 45, "text-anchor": "middle", "font-size": 15, "font-weight": 700, fill: "#db6d28" }, "Mooncake EFA CUDA Context 竞态");
          e.draw("rect", { x: 20, y: 220, width: 680, height: 180, rx: 10, fill: "#161b22", stroke: "#39d0d8", "stroke-width": 2, "stroke-dasharray": "5,3" });
          e.draw("text", { x: 360, y: 245, "text-anchor": "middle", "font-size": 15, "font-weight": 700, fill: "#39d0d8" }, "LMCache Query Ring Buffer + Token Dropping");
        }
      },
      { title: "Bug: fi_mr_regattr 间歇性失败", desc: "std::async 线程无 CUDA context -> dmabuf export 返回 CUDA_ERROR_INVALID_CONTEXT",
        render(e) {
          e.clear();
          // Top section
          e.draw("rect", { x: 20, y: 20, width: 680, height: 180, rx: 10, fill: "#161b22", stroke: "#db6d28", "stroke-width": 2 });
          e.draw("text", { x: 360, y: 45, "text-anchor": "middle", "font-size": 14, "font-weight": 700, fill: "#db6d28" }, "Mooncake EFA: fi_mr_regattr 间歇性失败");
          // Thread pool
          e.draw("text", { x: 100, y: 70, "text-anchor": "middle", "font-size": 11, "font-weight": 600, fill: "#8b949e" }, "std::async 线程池");
          for (var i = 0; i < 4; i++) {
            var tx = 40 + i * 45;
            e.draw("rect", { x: tx, y: 80, width: 35, height: 30, rx: 4, fill: "#1c2128", stroke: "#8b949e", "stroke-width": 1.5 });
            e.draw("text", { x: tx + 17, y: 100, "text-anchor": "middle", "font-size": 9, fill: "#8b949e" }, "T" + i);
          }
          // Thread 2 gets reused for device 3
          e.draw("rect", { x: 130, y: 80, width: 35, height: 30, rx: 4, fill: "#3d2418", stroke: "#f85149", "stroke-width": 2, class: "tween" });
          e.draw("text", { x: 147, y: 100, "text-anchor": "middle", "font-size": 9, "font-weight": 700, fill: "#f85149" }, "T1!");
          // Arrow to EFA
          e.draw("path", { d: "M 200 95 L 260 95", stroke: "#f85149", "stroke-width": 2, fill: "none", "marker-end": "url(#arr3r)" });
          e.draw("text", { x: 230, y: 88, "text-anchor": "middle", "font-size": 9, fill: "#f85149" }, "no ctx" );
          // EFA box
          e.draw("rect", { x: 260, y: 70, width: 180, height: 50, rx: 6, fill: "#3d2418", stroke: "#f85149", "stroke-width": 2, class: "tween" });
          e.draw("text", { x: 350, y: 90, "text-anchor": "middle", "font-size": 11, "font-weight": 600, fill: "#f85149" }, "fi_mr_regattr()" );
          e.draw("text", { x: 350, y: 108, "text-anchor": "middle", "font-size": 10, fill: "#f85149" }, "Operation not supported" );
          // Arrow to misleading error
          e.draw("path", { d: "M 440 95 L 500 95", stroke: "#f85149", "stroke-width": 2, fill: "none", "marker-end": "url(#arr3r)" });
          e.draw("rect", { x: 500, y: 70, width: 180, height: 50, rx: 6, fill: "#3d2418", stroke: "#f85149", "stroke-width": 2, class: "tween" });
          e.draw("text", { x: 590, y: 90, "text-anchor": "middle", "font-size": 11, "font-weight": 600, fill: "#f85149" }, "remote session" );
          e.draw("text", { x: 590, y: 108, "text-anchor": "middle", "font-size": 10, fill: "#f85149" }, "is not alive!" );
          // Root cause
          e.draw("rect", { x: 260, y: 135, width: 420, height: 55, rx: 6, fill: "#161b22", stroke: "#d29922", "stroke-width": 1.5 });
          e.draw("text", { x: 470, y: 155, "text-anchor": "middle", "font-size": 11, "font-weight": 600, fill: "#d29922" }, "Root cause: cuMemGetHandleForAddressRange() 需要 current context" );
          e.draw("text", { x: 470, y: 173, "text-anchor": "middle", "font-size": 10, fill: "#d29922" }, "std::async 线程可能从未触碰 CUDA API -> ENOTSUP -> 错误指向远端" );
          e.draw("text", { x: 470, y: 188, "text-anchor": "middle", "font-size": 10, fill: "#8b949e" }, "libfabric 吞掉真实原因" );
          // Bottom placeholder
          e.draw("text", { x: 360, y: 310, "text-anchor": "middle", "font-size": 12, fill: "#6e7681" }, "下一步: 修复方案" );
        }
      },
      { title: "Fix: cuDevicePrimaryCtxRetain 绑定 context", desc: "按 device ordinal 检查当前 context，不匹配时 re-bind",
        render(e) {
          e.clear();
          e.draw("rect", { x: 20, y: 20, width: 680, height: 180, rx: 10, fill: "#161b22", stroke: "#3fb950", "stroke-width": 2 });
          e.draw("text", { x: 360, y: 45, "text-anchor": "middle", "font-size": 14, "font-weight": 700, fill: "#3fb950" }, "Fix: bindCudaContextIfNeeded(device_ordinal)" );
          // Flow
          e.draw("rect", { x: 40, y: 65, width: 120, height: 40, rx: 6, fill: "#1a2d1a", stroke: "#3fb950", "stroke-width": 2, class: "tween" });
          e.draw("text", { x: 100, y: 89, "text-anchor": "middle", "font-size": 10, "font-weight": 600, fill: "#3fb950" }, "cuDeviceGet" );
          e.draw("path", { d: "M 160 85 L 200 85", stroke: "#3fb950", "stroke-width": 1.5, fill: "none", "marker-end": "url(#arr3g)" });
          e.draw("rect", { x: 200, y: 65, width: 140, height: 40, rx: 6, fill: "#1a2d1a", stroke: "#3fb950", "stroke-width": 2, class: "tween" });
          e.draw("text", { x: 270, y: 89, "text-anchor": "middle", "font-size": 10, "font-weight": 600, fill: "#3fb950" }, "cuCtxGetCurrent" );
          e.draw("path", { d: "M 340 85 L 380 85", stroke: "#3fb950", "stroke-width": 1.5, fill: "none", "marker-end": "url(#arr3g)" });
          e.draw("rect", { x: 380, y: 65, width: 160, height: 40, rx: 6, fill: "#1a2d1a", stroke: "#3fb950", "stroke-width": 2, class: "tween" });
          e.draw("text", { x: 460, y: 82, "text-anchor": "middle", "font-size": 10, "font-weight": 600, fill: "#3fb950" }, "device matches?" );
          e.draw("text", { x: 460, y: 97, "text-anchor": "middle", "font-size": 9, fill: "#8b949e" }, "CUdevice compare" );
          e.draw("path", { d: "M 540 85 L 580 85", stroke: "#3fb950", "stroke-width": 1.5, fill: "none", "marker-end": "url(#arr3g)" });
          e.draw("rect", { x: 580, y: 65, width: 100, height: 40, rx: 6, fill: "#1a2d1a", stroke: "#3fb950", "stroke-width": 2, class: "tween" });
          e.draw("text", { x: 630, y: 89, "text-anchor": "middle", "font-size": 10, "font-weight": 600, fill: "#3fb950" }, "PrimaryCtx" );
          // Result
          e.draw("rect", { x: 40, y: 120, width: 640, height: 50, rx: 6, fill: "#1a2d1a", stroke: "#3fb950", "stroke-width": 1.5 });
          e.draw("text", { x: 360, y: 140, "text-anchor": "middle", "font-size": 11, "font-weight": 600, fill: "#3fb950" }, "fi_mr_regattr() 成功 -> buffer 正确注册到所有 NIC" );
          e.draw("text", { x: 360, y: 158, "text-anchor": "middle", "font-size": 10, fill: "#8b949e" }, "retain 故意不释放 -> 避免拆掉进程级 context" );
          // Bottom: transition
          e.draw("text", { x: 360, y: 250, "text-anchor": "middle", "font-size": 12, fill: "#39d0d8" }, "下一步: LMCache Token Dropping" );
        }
      },
      { title: "LMCache: Query Ring Buffer 捕获", desc: "每层 attention 的 Q tensor 写入 paged ring buffer，复用 KV 传输 kernel",
        render(e) {
          e.clear();
          // Top section (Mooncake done)
          e.draw("rect", { x: 20, y: 20, width: 680, height: 50, rx: 10, fill: "#161b22", stroke: "#3fb950", "stroke-width": 1 });
          e.draw("text", { x: 360, y: 48, "text-anchor": "middle", "font-size": 12, "font-weight": 600, fill: "#3fb950" }, "Mooncake EFA: fixed (see above)" );
          // Bottom section
          e.draw("rect", { x: 20, y: 80, width: 680, height: 320, rx: 10, fill: "#161b22", stroke: "#39d0d8", "stroke-width": 2 });
          e.draw("text", { x: 360, y: 105, "text-anchor": "middle", "font-size": 14, "font-weight": 700, fill: "#39d0d8" }, "LMCache QRingBuffer" );
          // vLLM worker
          e.draw("rect", { x: 40, y: 125, width: 200, height: 250, rx: 8, fill: "#0d1117", stroke: "#30363d", "stroke-width": 1.5 });
          e.draw("text", { x: 140, y: 145, "text-anchor": "middle", "font-size": 12, "font-weight": 600, fill: "#8b949e" }, "vLLM Worker" );
          // Attention layers
          for (var i = 0; i < 4; i++) {
            e.draw("rect", { x: 60, y: 160 + i * 50, width: 160, height: 40, rx: 4, fill: "#1c2128", stroke: "#30363d", "stroke-width": 1, class: "tween" });
            e.draw("text", { x: 140, y: 185 + i * 50, "text-anchor": "middle", "font-size": 10, fill: "#8b949e" }, "Attn Layer " + i );
          }
          // Highlight current layer
          e.draw("rect", { x: 60, y: 160, width: 160, height: 40, rx: 4, fill: "none", stroke: "#39d0d8", "stroke-width": 2.5, class: "tween" });
          e.draw("text", { x: 140, y: 185, "text-anchor": "middle", "font-size": 10, "font-weight": 700, fill: "#39d0d8" }, "Attn Layer 0 [Q]" );
          // Arrow to ring buffer
          e.draw("path", { d: "M 220 180 L 280 180", stroke: "#39d0d8", "stroke-width": 2, fill: "none", "marker-end": "url(#arr3)" });
          e.draw("text", { x: 250, y: 175, "text-anchor": "middle", "font-size": 9, fill: "#39d0d8" }, "scatter" );
          // Ring buffer
          e.draw("rect", { x: 280, y: 125, width: 200, height: 250, rx: 8, fill: "#0d1117", stroke: "#39d0d8", "stroke-width": 2 });
          e.draw("text", { x: 380, y: 145, "text-anchor": "middle", "font-size": 12, "font-weight": 600, fill: "#39d0d8" }, "Q Ring Buffer" );
          // Paged blocks
          for (var r = 0; r < 4; r++) {
            for (var c = 0; c < 4; c++) {
              var bx = 300 + c * 40;
              var by = 160 + r * 45;
              e.draw("rect", { x: bx, y: by, width: 35, height: 35, rx: 3, fill: r === 0 && c === 0 ? "#0d3b3b" : "#1c2128", stroke: r === 0 && c === 0 ? "#39d0d8" : "#30363d", "stroke-width": r === 0 && c === 0 ? 2 : 1, class: "tween" });
            }
          }
          e.draw("text", { x: 380, y: 350, "text-anchor": "middle", "font-size": 10, fill: "#8b949e" }, "paged: [num_layers, blocks, block_size, hidden]" );
          // Arrow to LMCache
          e.draw("path", { d: "M 480 250 L 540 250", stroke: "#39d0d8", "stroke-width": 2, fill: "none", "marker-end": "url(#arr3)" });
          e.draw("text", { x: 510, y: 245, "text-anchor": "middle", "font-size": 9, fill: "#39d0d8" }, "STORE" );
          // LMCache server
          e.draw("rect", { x: 540, y: 200, width: 140, height: 100, rx: 8, fill: "#0d1117", stroke: "#39d0d8", "stroke-width": 2 });
          e.draw("text", { x: 610, y: 225, "text-anchor": "middle", "font-size": 12, "font-weight": 600, fill: "#39d0d8" }, "LMCache" );
          e.draw("text", { x: 610, y: 245, "text-anchor": "middle", "font-size": 10, fill: "#8b949e" }, "MP Server" );
          e.draw("text", { x: 610, y: 265, "text-anchor": "middle", "font-size": 9, fill: "#8b949e" }, "model##query" );
          // Key: reused paged-KV kernels
          e.draw("rect", { x: 540, y: 320, width: 140, height: 35, rx: 6, fill: "#1a2d1a", stroke: "#3fb950", "stroke-width": 1.5 });
          e.draw("text", { x: 610, y: 342, "text-anchor": "middle", "font-size": 9, "font-weight": 600, fill: "#3fb950" }, "reuses KV transfer kernels" );
        }
      },
      { title: "Token Dropping: 随机 -> 精准", desc: "用 Q tensor 决定哪些 token 该丢，decode 吞吐 1.5x-1.7x",
        render(e) {
          e.clear();
          // Left: random dropping (before)
          e.draw("rect", { x: 20, y: 60, width: 320, height: 160, rx: 10, fill: "#161b22", stroke: "#8b949e", "stroke-width": 2, "stroke-dasharray": "5,3" });
          e.draw("text", { x: 180, y: 85, "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: "#8b949e" }, "Before: Random Token Dropping" );
          // Token boxes - some randomly dropped
          for (var i = 0; i < 10; i++) {
            var dropped = (i === 2 || i === 5 || i === 7);
            var bx = 40 + i * 28;
            e.draw("rect", { x: bx, y: 110, width: 22, height: 30, rx: 3,
              fill: dropped ? "#3d2418" : "#1c2128",
              stroke: dropped ? "#f85149" : "#8b949e",
              "stroke-width": 1.5, class: "tween" });
            e.draw("text", { x: bx + 11, y: 128, "text-anchor": "middle", "font-size": 9,
              fill: dropped ? "#f85149" : "#8b949e" },
              dropped ? "X" : "t" );
          }
          e.draw("text", { x: 180, y: 165, "text-anchor": "middle", "font-size": 11, fill: "#8b949e" }, "random drop: some important tokens lost" );
          e.draw("text", { x: 180, y: 190, "text-anchor": "middle", "font-size": 11, "font-weight": 600, fill: "#8b949e" }, "1.5x-1.7x throughput (but lossy)" );

          // Right: smart dropping (after)
          e.draw("rect", { x: 380, y: 60, width: 320, height: 160, rx: 10, fill: "#161b22", stroke: "#39d0d8", "stroke-width": 2, "stroke-dasharray": "5,3" });
          e.draw("text", { x: 540, y: 85, "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: "#39d0d8" }, "After: Smart Dropping (Q-based)" );
          // Q attention scores determine importance
          for (var i = 0; i < 10; i++) {
            var important = (i === 0 || i === 3 || i === 6);
            var dropped2 = (i === 2 || i === 5 || i === 7);
            var bx = 400 + i * 28;
            if (!dropped2) {
              e.draw("rect", { x: bx, y: 110, width: 22, height: 30, rx: 3,
                fill: important ? "#0d3b3b" : "#1c2128",
                stroke: important ? "#39d0d8" : "#8b949e",
                "stroke-width": important ? 2 : 1, class: "tween" });
              e.draw("text", { x: bx + 11, y: 128, "text-anchor": "middle", "font-size": 9,
                fill: important ? "#39d0d8" : "#8b949e" }, "t" );
            } else {
              e.draw("rect", { x: bx, y: 110, width: 22, height: 30, rx: 3,
                fill: "#1c2128", stroke: "#30363d", "stroke-width": 1, "stroke-dasharray": "2,2", class: "tween" });
              e.draw("text", { x: bx + 11, y: 128, "text-anchor": "middle", "font-size": 9, fill: "#6e7681" }, "drop" );
            }
          }
          e.draw("text", { x: 540, y: 165, "text-anchor": "middle", "font-size": 11, fill: "#39d0d8" }, "Q scores identify low-impact tokens" );
          e.draw("text", { x: 540, y: 190, "text-anchor": "middle", "font-size": 11, "font-weight": 600, fill: "#39d0d8" }, "1.5x-1.7x throughput (minimized loss)" );

          // Signal
          e.draw("rect", { x: 40, y: 250, width: 640, height: 50, rx: 8, fill: "#1a2d1a", stroke: "#3fb950", "stroke-width": 1.5 });
          e.draw("text", { x: 360, y: 275, "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: "#3fb950" }, "KV Cache 传输组件: 从插件 -> 独立系统层" );
          e.draw("text", { x: 360, y: 293, "text-anchor": "middle", "font-size": 11, fill: "#8b949e" }, "独立 release cycle + HA 机制 + 7x24 生产硬化" );
        }
      },
      { title: "结论: 推理引擎三方向系统软件化", desc: "Rust 服务层(吞吐上限) + Kernel 专用化(成本下限) + KV Cache 加固(规模化站稳)",
        render(e) {
          e.clear();
          // Three pillars
          var pillars = [
            { x: 40, y: 60, w: 200, h: 280, title: "Rust 服务层", color: "#bc8cff", desc: "吞吐上限", items: ["无锁 shard", "零分配稳态", "pinned thread", "ring buffer"] },
            { x: 260, y: 60, w: 200, h: 280, title: "Kernel 专用化", color: "#3fb950", desc: "成本下限", items: ["减法: -68 行", "  -> 1.88x + 448MiB", "加法: +309 行", "  -> +22% 吞吐"] },
            { x: 480, y: 60, w: 200, h: 280, title: "KV Cache 加固", color: "#db6d28", desc: "规模化站稳", items: ["CUDA ctx 竞态", "SIGBUS 硬崩", "Q ring buffer", "smart dropping"] }
          ];
          pillars.forEach(function (p) {
            e.draw("rect", { x: p.x, y: p.y, width: p.w, height: p.h, rx: 10, fill: "#161b22", stroke: p.color, "stroke-width": 2.5, class: "tween" });
            e.draw("text", { x: p.x + p.w / 2, y: p.y + 25, "text-anchor": "middle", "font-size": 14, "font-weight": 700, fill: p.color }, p.title );
            e.draw("text", { x: p.x + p.w / 2, y: p.y + 45, "text-anchor": "middle", "font-size": 12, fill: "#8b949e" }, p.desc );
            p.items.forEach(function (item, i) {
              e.draw("text", { x: p.x + 20, y: p.y + 80 + i * 30, "font-size": 11, fill: "#8b949e" }, "- " + item );
            });
          });
          // Bottom conclusion
          e.draw("rect", { x: 40, y: 360, width: 640, height: 40, rx: 8, fill: "#1c2128", stroke: "#58a6ff", "stroke-width": 1.5 });
          e.draw("text", { x: 360, y: 385, "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: "#58a6ff" }, "推理引擎正从三个方向同时被系统软件化" );
        }
      }
    ];
    return steps;
  }

  window.INF_ANIMS = window.INF_ANIMS || {};
  window.INF_ANIMS[3] = { factory: svgFactory, steps: buildSteps };
})();
