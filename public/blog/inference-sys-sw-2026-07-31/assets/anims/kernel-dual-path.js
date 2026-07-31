/* 动画② Kernel 优化双路 — 减法 vs 加法
 * 左：vLLM "减法"路径 — 消除冗余 kernel 调用/内存分配
 * 右：SGLang "加法"路径 — 铺设 SM90+DSv4+FP8 专用 fast path
 */
(function () {
  function svgFactory() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 720 400");
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    defs.innerHTML =
      '<marker id="arr2g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">' +
      '<path d="M 0 0 L 10 5 L 0 10 z" fill="#3fb950"/></marker>' +
      '<marker id="arr2o" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">' +
      '<path d="M 0 0 L 10 5 L 0 10 z" fill="#db6d28"/></marker>' +
      '<marker id="arr2r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">' +
      '<path d="M 0 0 L 10 5 L 0 10 z" fill="#f85149"/></marker>';
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
    var steps = [
      { title: "Kernel 优化双路", desc: "vLLM 做减法（68 行 → 1.88x + 448MiB） · SGLang 做加法（309 行 → +22%）",
        render(e) {
          e.clear();
          // Left: 减法
          e.draw("rect", { x: 20, y: 40, width: 330, height: 340, rx: 10, fill: "#161b22", stroke: "#3fb950", "stroke-width": 2, "stroke-dasharray": "5,3" });
          e.draw("text", { x: 185, y: 65, "text-anchor": "middle", "font-size": 15, "font-weight": 700, fill: "#3fb950" }, "减法 · vLLM");
          e.draw("text", { x: 185, y: 84, "text-anchor": "middle", "font-size": 12, fill: "#8b949e" }, "从已有路径消除浪费");
          // Right: 加法
          e.draw("rect", { x: 370, y: 40, width: 330, height: 340, rx: 10, fill: "#161b22", stroke: "#db6d28", "stroke-width": 2, "stroke-dasharray": "5,3" });
          e.draw("text", { x: 535, y: 65, "text-anchor": "middle", "font-size": 15, "font-weight": 700, fill: "#db6d28" }, "加法 · SGLang");
          e.draw("text", { x: 535, y: 84, "text-anchor": "middle", "font-size": 12, fill: "#8b949e" }, "铺设专用 fast path");
        }
      },
      { title: "减法 #1：vLLM #50298 — 消去 torch.full kernel", desc: "combine_topk_swa_indices 传 out tensor，避免函数内部分配",
        render(e) {
          e.clear();
          // Before
          e.draw("text", { x: 40, y: 110, "font-size": 13, "font-weight": 600, fill: "#f85149" }, "Before: 每次分配");
          e.draw("rect", { x: 40, y: 120, width: 130, height: 35, rx: 6, fill: "#3d2418", stroke: "#f85149", "stroke-width": 2, class: "tween" });
          e.draw("text", { x: 105, y: 142, "text-anchor": "middle", "font-size": 11, fill: "#f85149" }, "torch.full ⚡ kernel");
          e.draw("rect", { x: 40, y: 165, width: 130, height: 35, rx: 6, fill: "#3d2418", stroke: "#f85149", "stroke-width": 2, class: "tween" });
          e.draw("text", { x: 105, y: 187, "text-anchor": "middle", "font-size": 11, fill: "#f85149" }, "torch.empty ⚡ alloc");
          e.draw("text", { x: 105, y: 220, "text-anchor": "middle", "font-size": 11, fill: "#6e7681" }, "每次 prefill chunk 都跑");
          // Arrow
          e.draw("path", { d: "M 180 155 L 220 155", stroke: "#3fb950", "stroke-width": 2, fill: "none", "marker-end": "url(#arr2g)" });
          // After
          e.draw("text", { x: 240, y: 110, "font-size": 13, "font-weight": 600, fill: "#3fb950" }, "After: 传 out tensor");
          e.draw("rect", { x: 240, y: 120, width: 100, height: 80, rx: 6, fill: "#1a2d1a", stroke: "#3fb950", "stroke-width": 2, class: "tween" });
          e.draw("text", { x: 290, y: 155, "text-anchor": "middle", "font-size": 12, "font-weight": 600, fill: "#3fb950" }, "workspace");
          e.draw("text", { x: 290, y: 175, "text-anchor": "middle", "font-size": 10, fill: "#3fb950" }, "预分配复用");
          // Result
          e.draw("text", { x: 185, y: 260, "text-anchor": "middle", "font-size": 22, "font-weight": 800, fill: "#3fb950" }, "1.88x ⚡");
          e.draw("text", { x: 185, y: 285, "text-anchor": "middle", "font-size": 12, fill: "#8b949e" }, "kernel 提速 · 0 精度损失");
          e.draw("text", { x: 185, y: 310, "text-anchor": "middle", "font-size": 13, fill: "#3fb950" }, "+44 -23 lines · 3 files");
          // Right side placeholder
          e.draw("text", { x: 535, y: 200, "text-anchor": "middle", "font-size": 12, fill: "#6e7681" }, "→ 下一步看加法路径");
        }
      },
      { title: "减法 #2：vLLM #50312 — 消去 PP buffer 冗余分配", desc: "MTP 未启用时也分配了 448 MiB buffer，现在条件分配",
        render(e) {
          e.clear();
          // Before
          e.draw("text", { x: 40, y: 110, "font-size": 13, "font-weight": 600, fill: "#f85149" }, "Before: 无条件分配");
          e.draw("rect", { x: 40, y: 120, width: 280, height: 50, rx: 6, fill: "#3d2418", stroke: "#f85149", "stroke-width": 2, class: "tween" });
          e.draw("text", { x: 180, y: 140, "text-anchor": "middle", "font-size": 11, fill: "#f85149" }, "torch.empty(8192, 7168×4, bf16)");
          e.draw("text", { x: 180, y: 158, "text-anchor": "middle", "font-size": 10, fill: "#f85149" }, "即使 MTP 未启用也分配！");
          // Arrow
          e.draw("path", { d: "M 180 200 L 180 220", stroke: "#3fb950", "stroke-width": 2, fill: "none", "marker-end": "url(#arr2g)" });
          // After
          e.draw("text", { x: 40, y: 245, "font-size": 13, "font-weight": 600, fill: "#3fb950" }, "After: 检查 speculative config");
          e.draw("rect", { x: 40, y: 255, width: 280, height: 50, rx: 6, fill: "#1a2d1a", stroke: "#3fb950", "stroke-width": 2, class: "tween" });
          e.draw("text", { x: 180, y: 275, "text-anchor": "middle", "font-size": 11, fill: "#3fb950" }, "if needs_mtp: torch.empty(...)");
          e.draw("text", { x: 180, y: 293, "text-anchor": "middle", "font-size": 10, fill: "#3fb950" }, "use_eagle() or uses_draft_model()");
          // Result
          e.draw("text", { x: 185, y: 340, "text-anchor": "middle", "font-size": 20, "font-weight": 800, fill: "#3fb950" }, "448 MiB 节省");
          e.draw("text", { x: 185, y: 365, "text-anchor": "middle", "font-size": 12, fill: "#8b949e" }, "NVIDIA + AMD + XPU 三路径同修");
          // Right side
          e.draw("text", { x: 535, y: 200, "text-anchor": "middle", "font-size": 12, fill: "#6e7681" }, "→ 下一步看加法路径");
        }
      },
      { title: "加法：SGLang #29016 — SM90 DeepGEMM MegaMoE", desc: "为 H100 + DSv4 + FP8 铺设专用 MoE A2A 路径",
        render(e) {
          e.clear();
          // Stack: SM90 + DSv4 + FP8
          var stack = [
            { y: 90, l: "SM90 (H100)", c: "#58a6ff", bg: "#0d2240" },
            { y: 130, l: "DeepSeek-V4", c: "#bc8cff", bg: "#2a1a3d" },
            { y: 170, l: "FP8 Quantization", c: "#d29922", bg: "#3d2e0d" },
            { y: 210, l: "DeepGEMM MegaMoE", c: "#db6d28", bg: "#3d2418" },
            { y: 250, l: "JIT Pre-Dispatch Kernel", c: "#3fb950", bg: "#1a2d1a" }
          ];
          stack.forEach(function (s, i) {
            e.draw("rect", { x: 400, y: s.y, width: 270, height: 32, rx: 6,
              fill: s.bg, stroke: s.c, "stroke-width": 2, class: "tween" });
            e.draw("text", { x: 535, y: s.y + 21, "text-anchor": "middle",
              "font-size": 12, "font-weight": 600, fill: s.c }, s.l);
            if (i < stack.length - 1) {
              e.draw("path", { d: "M 535 " + (s.y + 32) + " L 535 " + (s.y + 38), stroke: s.c, "stroke-width": 1.5, fill: "none" });
            }
          });
          // Lock icon
          e.draw("text", { x: 535, y: 300, "text-anchor": "middle", "font-size": 20 }, "🔒");
          e.draw("text", { x: 535, y: 325, "text-anchor": "middle", "font-size": 12, "font-weight": 600, fill: "#db6d28" }, "三元绑定");
          e.draw("text", { x: 535, y: 345, "text-anchor": "middle", "font-size": 11, fill: "#8b949e" }, "模型/硬件换代 → 重写");
          // Result
          e.draw("text", { x: 535, y: 375, "text-anchor": "middle", "font-size": 16, "font-weight": 800, fill: "#db6d28" }, "+22.2% 吞吐 · -18.9% TPOT");
          // Left: contrast
          e.draw("text", { x: 185, y: 110, "text-anchor": "middle", "font-size": 13, "font-weight": 600, fill: "#3fb950" }, "减法路径");
          e.draw("text", { x: 185, y: 160, "text-anchor": "middle", "font-size": 12, fill: "#8b949e" }, "通用改进");
          e.draw("text", { x: 185, y: 185, "text-anchor": "middle", "font-size": 12, fill: "#8b949e" }, "不绑定模型/硬件");
          e.draw("text", { x: 185, y: 210, "text-anchor": "middle", "font-size": 12, fill: "#8b949e" }, "68 行改动");
          e.draw("text", { x: 185, y: 250, "text-anchor": "middle", "font-size": 12, fill: "#8b949e" }, "可维护性: ✅ 高");
          // Arrow
          e.draw("text", { x: 360, y: 200, "text-anchor": "middle", "font-size": 16, fill: "#6e7681" }, "vs");
        }
      },
      { title: "两条路的本质对比", desc: "减法用最少代码拿最大收益 · 加法用专用路径换极致性能但绑死三元",
        render(e) {
          e.clear();
          // Left summary
          e.draw("rect", { x: 40, y: 80, width: 280, height: 280, rx: 10, fill: "#1a2d1a", stroke: "#3fb950", "stroke-width": 2, class: "tween" });
          e.draw("text", { x: 180, y: 110, "text-anchor": "middle", "font-size": 15, "font-weight": 700, fill: "#3fb950" }, "减法 ✅");
          var leftItems = ["68 行 → 1.88x + 448 MiB", "通用改进，不绑定硬件", "不增加维护负担", "onboarding 速度不受影响"];
          leftItems.forEach(function (t, i) {
            e.draw("text", { x: 60, y: 145 + i * 35, "font-size": 12, fill: "#8b949e" }, "• " + t);
          });
          e.draw("text", { x: 180, y: 320, "text-anchor": "middle", "font-size": 14, "font-weight": 700, fill: "#3fb950" }, "可维护性: 高");

          // Right summary
          e.draw("rect", { x: 400, y: 80, width: 280, height: 280, rx: 10, fill: "#3d2418", stroke: "#db6d28", "stroke-width": 2, class: "tween" });
          e.draw("text", { x: 540, y: 110, "text-anchor": "middle", "font-size": 15, "font-weight": 700, fill: "#db6d28" }, "加法 ⚠️");
          var rightItems = ["309 行 → +22% 吞吐", "SM90+DSv4+FP8 三元绑定", "DeepGEMM 库依赖", "换代需要重写专用路径"];
          rightItems.forEach(function (t, i) {
            e.draw("text", { x: 420, y: 145 + i * 35, "font-size": 12, fill: "#8b949e" }, "• " + t);
          });
          e.draw("text", { x: 540, y: 320, "text-anchor": "middle", "font-size": 14, "font-weight": 700, fill: "#db6d28" }, "可维护性: 低");

          // Question
          e.draw("text", { x: 360, y: 200, "text-anchor": "middle", "font-size": 13, "font-weight": 600, fill: "#d29922" }, "每个模型都需要");
          e.draw("text", { x: 360, y: 220, "text-anchor": "middle", "font-size": 13, "font-weight": 600, fill: "#d29922" }, "自己的 kernel 快速路径？" );
        }
      }
    ];
    return steps;
  }

  window.INF_ANIMS = window.INF_ANIMS || {};
  window.INF_ANIMS[2] = { factory: svgFactory, steps: buildSteps };
})();
