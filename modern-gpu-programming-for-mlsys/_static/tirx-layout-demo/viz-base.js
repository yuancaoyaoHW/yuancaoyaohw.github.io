// Shared behavior for all viz HTMLs
document.addEventListener('DOMContentLoaded', function() {
  var p = new URLSearchParams(location.search);
  if (p.has('notitle')) document.body.classList.add('notitle');

  // ── Zoom controls ──────────────────────────────────────
  // Inject a zoom toolbar and wrap all body content in #zoom-wrap.
  // This works for every demo without modifying individual HTML files.
  (function setupZoom() {
    // Don't double-wrap
    if (document.getElementById('zoom-wrap')) return;

    // Create zoom toolbar
    var zoomBar = document.createElement('div');
    zoomBar.className = 'zoom-bar';
    zoomBar.innerHTML =
      '<span class="zlbl">缩放</span>' +
      '<button class="zbtn" id="zout" title="缩小">−</button>' +
      '<span class="zval" id="zval">100%</span>' +
      '<button class="zbtn" id="zin" title="放大">+</button>' +
      '<button class="zbtn" id="zreset" title="重置" style="width:auto;padding:0 10px;font-size:12px;">重置</button>';

    // Create zoom wrapper — move all body children into it
    var wrap = document.createElement('div');
    wrap.id = 'zoom-wrap';

    // Move all existing body children (except script tags) into the wrapper
    var children = Array.prototype.slice.call(document.body.childNodes);
    children.forEach(function(child) {
      if (child.nodeType === 1 && child.tagName === 'SCRIPT') return;
      if (child.nodeType === 1 && child.id === 'zoom-wrap') return;
      wrap.appendChild(child);
    });

    // Insert toolbar then wrapper at the top of body
    document.body.insertBefore(wrap, document.body.firstChild);
    document.body.insertBefore(zoomBar, wrap);

    // Zoom state
    var zoom = 1.0;
    var MIN_ZOOM = 0.4;
    var MAX_ZOOM = 2.0;
    var STEP = 0.1;
    var zval = document.getElementById('zval');

    function applyZoom() {
      wrap.style.transform = 'scale(' + zoom + ')';
      zval.textContent = Math.round(zoom * 100) + '%';
      // Notify parent iframe resizer about height change
      if (window.parent !== window) {
        setTimeout(function() {
          var h = document.body.scrollHeight;
          window.parent.postMessage({ type: 'demoHeight', height: h }, '*');
        }, 250);
      }
    }

    document.getElementById('zin').addEventListener('click', function() {
      zoom = Math.min(MAX_ZOOM, zoom + STEP);
      applyZoom();
    });
    document.getElementById('zout').addEventListener('click', function() {
      zoom = Math.max(MIN_ZOOM, zoom - STEP);
      applyZoom();
    });
    document.getElementById('zreset').addEventListener('click', function() {
      zoom = 1.0;
      applyZoom();
    });

    // Mouse wheel zoom (Ctrl+scroll)
    document.addEventListener('wheel', function(e) {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        var delta = e.deltaY < 0 ? STEP : -STEP;
        zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom + delta));
        applyZoom();
      }
    }, { passive: false });
  })();

  // Forward arrow keys to parent (reveal.js) when embedded
  if (window.parent !== window) {
    document.addEventListener('keydown', function(e) {
      if ([37, 38, 39, 40, 27, 32].indexOf(e.keyCode) !== -1) {
        // Left, Up, Right, Down, Escape, Space
        window.parent.postMessage({ type: 'revealKey', keyCode: e.keyCode }, '*');
      }
    });
  }
});

// Auto-height: when embedded in the book (demo-embed.js), the demo measures its
// OWN content height and posts it to the parent, which sizes the iframe to fit so
// there is never an inner scrollbar. This is push-based on purpose — the demo
// catches its own DOM changes (a click that appends rows, expands a panel, …),
// which a parent watching the iframe's <body> from outside can miss. Measuring
// body.scrollHeight (not documentElement, which is floored to the viewport) lets
// the reported height grow AND shrink with the content.
(function () {
  if (window.parent === window) return;   // only when embedded
  var lastH = 0;
  function report() {
    var b = document.body, de = document.documentElement;
    var h = (b ? b.scrollHeight : 0) || (de ? de.scrollHeight : 0) || 0;
    if (h && Math.abs(h - lastH) > 1) {
      lastH = h;
      window.parent.postMessage({ type: 'demoHeight', height: h }, '*');
    }
  }
  var scheduled = false;
  function schedule() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(function () { scheduled = false; report(); });
  }
  // documentElement exists even while we are still in <head>, so observers can be
  // attached immediately; the first read happens in the rAF after layout.
  try { new ResizeObserver(schedule).observe(document.documentElement); } catch (e) {}
  try {
    new MutationObserver(schedule).observe(document.documentElement, {
      subtree: true, childList: true, attributes: true, characterData: true
    });
  } catch (e) {}
  document.addEventListener('DOMContentLoaded', schedule);
  window.addEventListener('load', schedule);
  // Clicks often trigger async content changes; re-measure right after.
  window.addEventListener('click', function () { setTimeout(schedule, 0); }, true);
  // Catch late settling (fonts, deferred render).
  [100, 300, 600, 1200].forEach(function (t) { setTimeout(schedule, t); });
})();
