/* 主题初始化：在 <head> 里同步执行，首帧前设好 data-theme，避免闪烁。
   与博客共用 localStorage 键 "theme"（"light" / "dark"），嵌入博客时跟随博客主题。 */
(function () {
  var t = null;
  try { t = localStorage.getItem('theme'); } catch (e) {}
  if (t !== 'light' && t !== 'dark') t = matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', t);
})();
