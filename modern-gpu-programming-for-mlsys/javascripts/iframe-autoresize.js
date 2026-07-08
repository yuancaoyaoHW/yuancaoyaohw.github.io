// Auto-resize iframes when their content changes (e.g. zoom)
(function() {
  function resizeIframe(iframe) {
    try {
      var h = iframe.contentWindow.document.body.scrollHeight;
      if (h && h > 0) {
        iframe.style.height = h + 'px';
      }
    } catch(e) {}
  }

  function initAll() {
    document.querySelectorAll('iframe').forEach(function(iframe) {
      if (iframe.dataset.autoResize) return;
      iframe.dataset.autoResize = '1';
      iframe.addEventListener('load', function() { resizeIframe(iframe); });
      // Listen for height messages from the iframe
      window.addEventListener('message', function(e) {
        if (e.data && e.data.type === 'demoHeight' && e.source === iframe.contentWindow) {
          iframe.style.height = e.data.height + 'px';
        }
      });
      // Initial resize
      setTimeout(function() { resizeIframe(iframe); }, 500);
      setTimeout(function() { resizeIframe(iframe); }, 1500);
    });
  }

  if (document.readyState !== 'loading') {
    initAll();
  } else {
    document.addEventListener('DOMContentLoaded', initAll);
  }
  // Re-check periodically for late-loading iframes
  setTimeout(initAll, 2000);
})();
