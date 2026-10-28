// 主程序：渲染器、层级导航、拾取、信息面板、工具栏
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { LEVELS, INFO } from './data.js';
import { buildRack } from './rack.js';
import { buildCompute, buildSuperchip, buildGpu } from './detail.js';
import { buildSwitch, buildPower, buildCooling } from './infra.js';
import { buildFabric, buildCluster } from './topo.js';

const BUILD = {
  rack: buildRack, compute: buildCompute, superchip: buildSuperchip, gpu: buildGpu,
  switch: buildSwitch, fabric: buildFabric, power: buildPower, cooling: buildCooling, cluster: buildCluster,
};
const DEFAULT_INFO = {
  rack: 'rack', compute: 'ct', superchip: 'board', gpu: 'b200', switch: 'sw',
  fabric: 'fabric', power: 'shelf', cooling: 'cdu', cluster: 'cluster',
};
const $ = id => document.getElementById(id);
if (self !== top) document.documentElement.classList.add('embed');   // 嵌入博客时外层已有操作提示

// ---------------- 渲染器 / 场景 ----------------
const stage = $('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
stage.appendChild(renderer.domElement);
const css = new CSS2DRenderer();
Object.assign(css.domElement.style, { position: 'absolute', inset: '0', pointerEvents: 'none' });
stage.appendChild(css.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070b12);
scene.add(new THREE.HemisphereLight(0xbfd8ff, 0x1a1f28, 1.1));
scene.add(new THREE.AmbientLight(0xffffff, 0.25));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(12, 25, 18); scene.add(sun);
const rim = new THREE.DirectionalLight(0x6fa8ff, 0.6);
rim.position.set(-15, 10, -20); scene.add(rim);
const grid = new THREE.GridHelper(80, 80, 0x1b2636, 0x111a26);
grid.position.y = -0.02; grid.userData.noPick = true; scene.add(grid);

const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 500);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.maxDistance = 120;

function resize() {
  const w = stage.clientWidth || innerWidth, h = stage.clientHeight || innerHeight;
  renderer.setSize(w, h); css.setSize(w, h);
  // 宽屏时右侧信息面板（~374px）会遮挡场景：把投影中心左移半个面板宽，拾取/标签同样走 projectionMatrix
  const d = w > 1100 ? 187 : w > 820 ? 157 : 0;
  camera.aspect = (w + 2 * d) / h;
  if (d) camera.setViewOffset(w + 2 * d, h, 2 * d, 0, w, h); else camera.clearViewOffset();
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();

// ---------------- 信息面板 ----------------
let cur = null, view = null, curKey = null;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function showInfo(key, title) {
  const d = INFO[key]; if (!d) return;
  curKey = key;
  $('i-kicker').textContent = d.k || '';
  $('i-title').textContent = title || d.t;
  $('i-sub').textContent = title && title !== d.t ? d.t + (d.s ? ' · ' + d.s : '') : (d.s || '');
  $('i-desc').innerHTML = d.d || '';   // 描述来自本地 data.js 常量
  $('i-specs').innerHTML = (d.sp || []).map(([a, b]) => `<tr><td>${esc(a)}</td><td>${esc(b)}</td></tr>`).join('');
  const go = $('i-go');
  if (d.go && d.go !== cur && BUILD[d.go]) {
    go.hidden = false; go.textContent = `深入查看：${LEVELS[d.go].name} →`;
    go.onclick = () => enter(d.go);
  } else go.hidden = true;
}

// ---------------- 面包屑 / 工具栏 / 图例 ----------------
function crumbs() {
  const chain = [];
  for (let k = cur; k; k = LEVELS[k].parent) chain.unshift(k);
  const nav = $('crumbs'); nav.innerHTML = '';
  chain.forEach((k, i) => {
    if (i) { const s = document.createElement('span'); s.className = 'sep'; s.textContent = '›'; nav.appendChild(s); }
    const b = document.createElement('button'); b.textContent = LEVELS[k].name;
    if (k === cur) b.setAttribute('aria-current', 'page'); else b.onclick = () => enter(k);
    nav.appendChild(b);
  });
  if (cur === 'rack') {                         // 机架层额外给出各子系统入口
    const s = document.createElement('span'); s.className = 'sep'; s.textContent = '|'; nav.appendChild(s);
    for (const k of ['fabric', 'power', 'cooling', 'cluster']) {
      const b = document.createElement('button'); b.textContent = LEVELS[k].name; b.onclick = () => enter(k);
      nav.appendChild(b);
    }
  }
}

function toolbar(tools) {
  const tb = $('toolbar'); tb.innerHTML = '';
  for (const t of tools || []) {
    if (t.grp) { const g = document.createElement('div'); g.className = 'grp'; g.textContent = t.grp; tb.appendChild(g); continue; }
    const b = document.createElement('button'); b.type = 'button'; b.textContent = t.label;
    if (t.toggle) {
      let on = !!t.on;
      b.setAttribute('aria-pressed', String(on));
      if (on) t.toggle(true);
      b.onclick = () => { on = !on; b.setAttribute('aria-pressed', String(on)); t.toggle(on); };
    } else b.onclick = () => t.run();
    tb.appendChild(b);
  }
}

function legend(items) {
  const lg = $('legend'); lg.innerHTML = '';
  for (const [c, txt] of items || []) {
    const d = document.createElement('div'), i = document.createElement('i');
    i.style.background = '#' + c.toString(16).padStart(6, '0');
    d.append(i, document.createTextNode(txt)); lg.appendChild(d);
  }
  lg.hidden = !(items && items.length);
}

// ---------------- 层级切换 ----------------
function dispose(v) {
  v.labels.forEach(l => { l.element.remove(); l.removeFromParent(); });
  v.root.removeFromParent();
}
function enter(level) {
  if (!BUILD[level] || level === cur) return;
  if (view) dispose(view);
  cur = level;
  view = BUILD[level]({ enter });
  scene.add(view.root);
  const c = view.camera || { pos: [10, 10, 10], target: [0, 0, 0] };
  camera.position.set(...c.pos); controls.target.set(...c.target); controls.update();
  grid.visible = level !== 'fabric';
  crumbs(); toolbar(view.tools); legend(view.legend);
  showInfo(DEFAULT_INFO[level]);
  tip.style.display = 'none';
  history.replaceState(null, '', '#' + level);
}

// ---------------- 拾取 ----------------
const tip = $('tip');
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
function visibleChain(o) { for (; o; o = o.parent) if (!o.visible) return false; return true; }
function pick(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  ray.params.Line.threshold = 0.05;
  for (const h of ray.intersectObject(view.root, true)) {
    const o = h.object;
    if (o.userData.noPick || o.userData.isLabel || o.isLine || o.isPoints || !visibleChain(o)) continue;
    let k = o;
    while (k && !(k.userData.key && INFO[k.userData.key])) k = k.parent;
    if (!k) continue;
    let t = o; while (t && !t.userData.title && t !== k) t = t.parent;
    const title = (t && t.userData.title) || k.userData.title || INFO[k.userData.key].t;
    return { obj: o, keyObj: k, key: k.userData.key, title };
  }
  return null;
}

let downAt = null;
renderer.domElement.addEventListener('pointerdown', e => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointermove', e => {
  if (!view || e.buttons) { tip.style.display = 'none'; return; }
  const p = pick(e);
  renderer.domElement.style.cursor = p ? 'pointer' : '';
  if (!p) { tip.style.display = 'none'; return; }
  const go = INFO[p.key].go;
  tip.textContent = p.title + (go && go !== cur ? '  · 双击深入' : '');
  tip.style.display = 'block';
  tip.style.left = e.clientX + 14 + 'px'; tip.style.top = e.clientY + 14 + 'px';
});
renderer.domElement.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
renderer.domElement.addEventListener('click', e => {
  if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;  // 拖动不算点击
  const p = pick(e);
  if (p) showInfo(p.key, p.title); else showInfo(DEFAULT_INFO[cur]);
  view.onSelect?.(p ? p.obj : null);
});
renderer.domElement.addEventListener('dblclick', e => {
  const p = pick(e);
  const go = p && INFO[p.key].go;
  if (go && go !== cur) enter(go);
});
addEventListener('keydown', e => {
  if (e.key === 'Escape' && LEVELS[cur]?.parent) enter(LEVELS[cur].parent);
});

// ---------------- 主循环 ----------------
const clock = new THREE.Clock();
let first = true;
function loop() {
  const dt = Math.min(clock.getDelta(), 0.1), t = clock.elapsedTime;
  view?.update?.(t, dt);
  controls.update();
  renderer.render(scene, camera); css.render(scene, camera);
  if (first) { first = false; $('loading').style.display = 'none'; }
  requestAnimationFrame(loop);
}

const start = location.hash.slice(1);
addEventListener('hashchange', () => { const k = location.hash.slice(1); if (BUILD[k]) enter(k); });
enter(BUILD[start] ? start : 'rack');
loop();
