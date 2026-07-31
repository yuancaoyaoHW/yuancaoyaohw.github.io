/* Main controller — loads animations from INF_ANIMS registry, wires up buttons */
(function () {
  "use strict";

  function init() {
    // Wait for all scripts to load
    if (!window.INF_ANIMS || !window.AnimEngine) {
      setTimeout(init, 50);
      return;
    }

    Object.keys(window.INF_ANIMS).forEach(function (key) {
      var num = parseInt(key, 10);
      var config = window.INF_ANIMS[num];
      if (!config || !config.factory || !config.steps) return;

      var svgId = "svg" + num;
      var svg = document.getElementById(svgId);
      if (!svg) return;

      // Replace empty svg with factory output
      var realSvg = config.factory();
      svg.parentNode.replaceChild(realSvg, svg);
      realSvg.setAttribute("id", svgId);
      realSvg.style.width = "100%";
      realSvg.style.maxWidth = "720px";

      var engine = new AnimEngine(realSvg, {
        steps: config.steps(),
        onStepChange: function (idx, step) {
          var titleEl = document.getElementById("a" + num + "-title");
          var descEl = document.getElementById("a" + num + "-desc");
          if (titleEl) titleEl.textContent = step.title || "";
          if (descEl) descEl.textContent = step.desc || "";
        }
      });

      // Wire up buttons
      var stage = document.getElementById("anim" + num);
      if (stage) {
        var buttons = stage.querySelectorAll(".anim-btn");
        buttons.forEach(function (btn) {
          btn.addEventListener("click", function () {
            var act = btn.getAttribute("data-act");
            if (act === "play") {
              if (engine.playing) {
                engine.pause();
                btn.textContent = "▶ 播放";
                btn.classList.remove("active");
              } else {
                engine.play();
                btn.textContent = "⏸ 暂停";
                btn.classList.add("active");
              }
            } else if (act === "prev") {
              engine.prev();
            } else if (act === "next") {
              engine.next();
            }
          });
        });
      }

      // Render first step
      engine.goTo(0);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
