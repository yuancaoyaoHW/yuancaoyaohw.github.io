/* RetailSim 讲解站 · 共享脚本：顶栏、目录、翻页、滚动进度、出现动效、选项卡、数字滚动 */
(function () {
  const PAGES = [
    { f: 'index.html', cn: '总览', g: 'home', c: '--sim', d: '项目是什么、模块地图、阅读路线' },
    { f: 'flow-day.html', cn: '一个交易日', g: 'flow', c: '--orch', d: '盘前、盘中、盘后各时段谁在做什么' },
    { f: 'flow-journey.html', cn: '一条新闻的旅程', g: 'flow', c: '--social', d: '一条快讯如何变成帖子、态度与成交' },
    { f: 'flow-run.html', cn: '运行生命周期', g: 'flow', c: '--val', d: '配置、预注册、运行、重放与分叉' },
    { f: 'sim-publishers.html', cn: '发布者演示', g: 'demo', c: '--social', d: 'L0–L4 发布者对同一事件的动态反应' },
    { f: 'sim-personas.html', cn: '散户决策演示', g: 'demo', c: '--sim', d: '不同类型散户面对同一消息怎样决策' },
    { f: 'sim-chat.html', cn: '群聊演示', g: 'demo', c: '--social', d: '四个熟人群里每个散户说了什么' },
    { f: 'mod-schemas.html', cn: 'schemas', g: 'base', c: '--orch', d: '模块之间传递的数据类型' },
    { f: 'mod-prompts.html', cn: 'prompts', g: 'base', c: '--llm', d: '提示词分层、版本与缓存' },
    { f: 'mod-llm.html', cn: 'llm', g: 'base', c: '--llm', d: '网关、缓存、重试与成本' },
    { f: 'mod-orchestrator.html', cn: 'orchestrator', g: 'base', c: '--orch', d: '仿真时钟、激活与事件日志' },
    { f: 'mod-collect.html', cn: 'collect', g: 'data', c: '--data', d: '真实帖子前向存档与去标识化' },
    { f: 'mod-data.html', cn: 'data', g: 'data', c: '--data', d: '点时行情、复权与 available_at' },
    { f: 'mod-events.html', cn: 'events', g: 'data', c: '--data', d: '事件卡、量级打分与 L0/L1 改写' },
    { f: 'mod-population.html', cn: 'population', g: 'sim', c: '--sim', d: '人设卡、MBTI 与合成人口' },
    { f: 'mod-agents.html', cn: 'agents', g: 'sim', c: '--sim', d: '认知循环、内核修正与记忆' },
    { f: 'mod-social.html', cn: 'social', g: 'sim', c: '--social', d: '渠道、网络、推荐流与传播' },
    { f: 'mod-market.html', cn: 'market', g: 'sim', c: '--market', d: 'A 股交易规则与成交判定' },
    { f: 'mod-accounts.html', cn: 'accounts', g: 'sim', c: '--acct', d: '账户、出入金与战绩' },
    { f: 'mod-outputs.html', cn: 'outputs', g: 'out', c: '--out', d: '帖子、态度、决策与账户类表' },
    { f: 'mod-validation.html', cn: 'validation', g: 'out', c: '--val', d: 'V1–V5 验收、基线与校准' },
  ];
  const GROUPS = [['home', '总览'], ['flow', '流程'], ['demo', '演示'], ['base', '基础层'], ['data', '数据层'], ['sim', '模拟层'], ['out', '产出层']];
  window.RS_PAGES = PAGES;
  const here = location.pathname.split('/').pop() || 'index.html';
  const idx = Math.max(0, PAGES.findIndex(p => p.f === here));
  const cur = PAGES[idx];
  document.documentElement.style.setProperty('--accent', `var(${cur.c})`);

  // 顶栏：按组折叠，点击或悬停展开该组页面
  const bar = document.createElement('div');
  bar.className = 'topbar';
  bar.innerHTML = `<a class="brand" href="index.html"><span class="logo"></span>RetailSim<span class="ref" style="font-weight:400">v0.3 讲解</span></a>
    <nav class="links">${GROUPS.map(([g, n]) => g === 'home'
      ? `<a href="index.html" class="${here === 'index.html' ? 'on' : ''}">${n}</a>`
      : `<button type="button" class="grp${cur.g === g ? ' on' : ''}" data-g="${g}">${n}<span class="car">▾</span></button>`).join('')}</nav>
    <span class="here">${cur.g === 'home' ? '' : cur.cn}</span>
    <div class="navpanel" hidden></div>`;
  document.body.prepend(bar);
  const panel = bar.querySelector('.navpanel');
  let openG = null, hideT = 0;
  function openGroup(g) {
    clearTimeout(hideT);
    if (openG === g) return;
    openG = g;
    panel.innerHTML = PAGES.filter(p => p.g === g).map(p =>
      `<a href="${p.f}" class="${p.f === here ? 'on' : ''}" style="--c:var(${p.c})"><b>${p.cn}</b><small>${p.d}</small></a>`).join('');
    panel.hidden = false;
    bar.querySelectorAll('.grp').forEach(b => b.classList.toggle('open', b.dataset.g === g));
  }
  function closeGroup() { openG = null; panel.hidden = true; bar.querySelectorAll('.grp').forEach(b => b.classList.remove('open')); }
  bar.querySelectorAll('.grp').forEach(b => {
    b.addEventListener('click', e => { e.stopPropagation(); openG === b.dataset.g ? closeGroup() : openGroup(b.dataset.g); });
    b.addEventListener('mouseenter', () => { if (matchMedia('(hover:hover)').matches) openGroup(b.dataset.g); });
  });
  bar.addEventListener('mouseleave', () => { if (matchMedia('(hover:hover)').matches) hideT = setTimeout(closeGroup, 250); });
  panel.addEventListener('mouseenter', () => clearTimeout(hideT));
  document.addEventListener('click', e => { if (!bar.contains(e.target)) closeGroup(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeGroup(); });
  const prog = document.createElement('div');
  prog.className = 'progress';
  document.body.prepend(prog);

  // 目录：页面有 <aside class="toc"> 时，从 main section[id] > h2 生成
  const toc = document.querySelector('aside.toc');
  const secs = [...document.querySelectorAll('main section[id]')].filter(s => s.querySelector('h2'));
  if (toc) {
    toc.innerHTML = '<b>本页目录</b>' + secs.map(s => {
      const h = s.querySelector('h2').cloneNode(true);
      h.querySelectorAll('.num').forEach(n => n.remove());
      return `<a href="#${s.id}">${h.textContent.trim()}</a>`;
    }).join('');
  }

  // 翻页 + 页脚
  const main = document.querySelector('main');
  if (main && !document.querySelector('.pager')) {
    const prev = PAGES[idx - 1], next = PAGES[idx + 1];
    const pg = document.createElement('div');
    pg.className = 'pager';
    pg.innerHTML = (prev ? `<a href="${prev.f}"><small>← 上一页</small>${prev.cn}</a>` : '<span></span>') +
      (next ? `<a class="next" href="${next.f}"><small>下一页 →</small>${next.cn}</a>` : '<span></span>');
    main.appendChild(pg);
  }
  const ft = document.createElement('footer');
  ft.className = 'site';
  ft.textContent = 'RetailSim · 依据设计文档 v0.3 与讨论记录整理 · 设计草案，尚未实现';
  document.body.appendChild(ft);

  // 滚动进度与目录高亮
  const tocLinks = toc ? [...toc.querySelectorAll('a')] : [];
  function onScroll() {
    const h = document.documentElement;
    prog.style.width = (h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight) * 100) + '%';
    let active = 0;
    secs.forEach((s, i) => { if (s.getBoundingClientRect().top < 140) active = i; });
    tocLinks.forEach((a, i) => a.classList.toggle('on', i === active));
  }
  addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // 出现动效
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
  }), { threshold: 0.12 });
  document.querySelectorAll('.reveal').forEach(el => io.observe(el));

  // 选项卡：<div class="tabs"><div class="tab-bar"><button data-t="a">…</button></div><div class="tab-pane" data-t="a">…</div></div>
  document.querySelectorAll('.tabs').forEach(t => {
    const btns = [...t.querySelectorAll(':scope > .tab-bar button')];
    const panes = [...t.querySelectorAll(':scope > .tab-pane')];
    function show(k) {
      btns.forEach(b => b.classList.toggle('on', b.dataset.t === k));
      panes.forEach(p => p.classList.toggle('on', p.dataset.t === k));
    }
    btns.forEach(b => b.addEventListener('click', () => show(b.dataset.t)));
    if (btns[0]) show((btns.find(b => b.classList.contains('on')) || btns[0]).dataset.t);
  });

  // 数字滚动：<span class="count" data-to="550" data-dec="0">0</span>
  const cio = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    const el = e.target, to = +el.dataset.to, dec = +(el.dataset.dec || 0), t0 = performance.now();
    (function step(t) {
      const k = Math.min(1, (t - t0) / 1200), v = to * (1 - Math.pow(1 - k, 3));
      el.textContent = v.toLocaleString('zh-CN', { minimumFractionDigits: dec, maximumFractionDigits: dec });
      if (k < 1) requestAnimationFrame(step);
    })(t0);
    cio.unobserve(el);
  }), { threshold: 0.5 });
  document.querySelectorAll('.count').forEach(el => cio.observe(el));
})();
