// 可折叠侧边栏控制器
// 左侧导航 + 右侧目录均可折叠，状态存 localStorage
(function () {
  'use strict';

  const LEFT_KEY = 'sl-sidebar-collapsed';
  const RIGHT_KEY = 'sl-toc-collapsed';

  function init() {
    const html = document.documentElement;
    const isWide = window.matchMedia('(min-width: 50rem)').matches;
    if (!isWide) return; // 移动端不注入按钮

    // 创建左侧按钮
    const btnLeft = document.createElement('button');
    btnLeft.className = 'sidebar-toggle sidebar-toggle--left';
    btnLeft.setAttribute('aria-label', '折叠左侧导航');
    btnLeft.title = '折叠/展开左侧导航';
    btnLeft.innerHTML = '\u2039'; // ‹ 左尖括号
    btnLeft.dataset.target = 'sidebar';

    // 创建右侧按钮
    const btnRight = document.createElement('button');
    btnRight.className = 'sidebar-toggle sidebar-toggle--right';
    btnRight.setAttribute('aria-label', '折叠右侧目录');
    btnRight.title = '折叠/展开右侧目录';
    btnRight.innerHTML = '\u203a'; // › 右尖括号
    btnRight.dataset.target = 'toc';

    document.body.appendChild(btnLeft);
    document.body.appendChild(btnRight);

    // 恢复上次状态
    if (localStorage.getItem(LEFT_KEY) === '1') {
      html.setAttribute('data-sidebar-collapsed', '');
      btnLeft.innerHTML = '\u203a'; // 收起后显示 ›
    }
    if (localStorage.getItem(RIGHT_KEY) === '1') {
      html.setAttribute('data-toc-collapsed', '');
      btnRight.innerHTML = '\u2039'; // 收起后显示 ‹
    }

    // 点击事件
    btnLeft.addEventListener('click', function () {
      const collapsed = html.toggleAttribute('data-sidebar-collapsed');
      localStorage.setItem(LEFT_KEY, collapsed ? '1' : '0');
      btnLeft.innerHTML = collapsed ? '\u203a' : '\u2039';
    });

    btnRight.addEventListener('click', function () {
      const collapsed = html.toggleAttribute('data-toc-collapsed');
      localStorage.setItem(RIGHT_KEY, collapsed ? '1' : '0');
      btnRight.innerHTML = collapsed ? '\u2039' : '\u203a';
    });
  }

  // 等 DOM 就绪
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
