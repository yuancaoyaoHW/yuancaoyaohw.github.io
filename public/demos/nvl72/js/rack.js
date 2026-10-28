// LEVEL 0：整机架。坐标单位 = 10 cm；x 宽、y 高、z 前(+)/后(-)
import * as THREE from 'three';
import { C, mat, box, cyl, group, label, curve, tube, flow, ease, clamp01 } from './kit.js';

export const U = 0.445;
const Y0 = 0.6, TW = 4.4, TD = 8, TH = U * 0.92, ZC = 0.6;
const ZF = ZC + TD / 2, ZR = ZC - TD / 2;           // 托盘前/后面 z
const CART_X = [-1.65, -0.95, 0.95, 1.65];          // 4 个 NVLink 电缆盒 / 托盘后连接器的 x
const MAN_X = 2.55, MAN_Z = -4.6, UQD_X = 2.05;
const yOf = u => Y0 + u * U + U / 2;

// 自下而上的 U 位布局（示意）
function layout() {
  const s = [];
  let u = 3;
  for (let i = 4; i >= 1; i--) s.push({ type: 'shelf', n: i, grp: 'B', u: u++ });
  u++;
  for (let n = 18; n >= 11; n--) s.push({ type: 'ct', n, u: u++ });
  for (let n = 9; n >= 1; n--) s.push({ type: 'sw', n, u: u++ });
  for (let n = 10; n >= 1; n--) s.push({ type: 'ct', n, u: u++ });
  u++;
  for (let i = 4; i >= 1; i--) s.push({ type: 'shelf', n: i, grp: 'A', u: u++ });
  u += 4;
  s.push({ type: 'mgmt', n: 1, u: u++ });
  return s;
}

function rearCommon(g, connW, connColor) {
  for (const x of CART_X) {
    const c = box(connW, 0.3, 0.16, connColor, { parent: g, pos: [x, 0, ZR - 0.08], key: 'nvconn', m: 0.6 });
    c.userData.title = 'NVLink 盲插连接器';
    box(connW * 0.8, 0.06, 0.02, C.gold, { parent: g, pos: [x, 0.05, ZR - 0.17], m: 1, r: 0.3, key: 'nvconn' });
  }
  box(0.3, 0.26, 0.22, C.bus, { parent: g, pos: [0, 0, ZR - 0.11], key: 'pdb', m: 0.9, r: 0.3 });
  cyl(0.09, 0.3, C.cold, { parent: g, pos: [-UQD_X, 0, ZR - 0.15], rot: [Math.PI / 2, 0, 0], key: 'uqd' });
  cyl(0.09, 0.3, C.hot, { parent: g, pos: [UQD_X, 0, ZR - 0.15], rot: [Math.PI / 2, 0, 0], key: 'uqd' });
}

function computeTray(parent, s) {
  const g = group(parent, 'ct', [0, yOf(s.u), 0]);
  g.userData.title = `计算托盘 #${s.n}`;
  box(TW, TH, TD, C.tray, { parent: g, pos: [0, 0, ZC], m: 0.6, r: 0.45 });
  box(TW, TH, 0.05, C.trayFace, { parent: g, pos: [0, 0, ZF + 0.025] });
  const zf = ZF + 0.06;
  for (let i = 0; i < 4; i++) {                       // 4 × ConnectX OSFP
    box(0.28, 0.22, 0.06, C.cage, { parent: g, pos: [-1.95 + i * 0.34, 0, zf], key: 'cx', m: 0.8, r: 0.3 });
    box(0.2, 0.12, 0.02, C.port, { parent: g, pos: [-1.95 + i * 0.34, 0, zf + 0.035], key: 'cx' });
  }
  for (let i = 0; i < 6; i++)                          // E1.S
    box(0.13, 0.34, 0.05, 0x3b4452, { parent: g, pos: [-0.45 + i * 0.17, 0, zf], key: 'e1s' });
  for (let i = 0; i < 2; i++)                          // BlueField-3
    box(0.24, 0.16, 0.06, 0x6b2b20, { parent: g, pos: [0.85 + i * 0.3, 0, zf], key: 'bf3' });
  box(0.14, 0.12, 0.05, C.port, { parent: g, pos: [1.6, 0, zf] });
  box(0.05, 0.05, 0.03, C.led, { parent: g, pos: [2.0, 0.08, zf], e: C.led, ei: 0.9 });
  box(0.3, 0.03, 0.02, C.nvidia, { parent: g, pos: [1.9, -0.12, zf], e: C.nvidia, ei: 0.4 });
  rearCommon(g, 0.45, 0x23272f);
  return g;
}

function switchTray(parent, s) {
  const g = group(parent, 'sw', [0, yOf(s.u), 0]);
  g.userData.title = `NVSwitch 托盘 #${s.n}`;
  box(TW, TH, TD, C.sw, { parent: g, pos: [0, 0, ZC], m: 0.6, r: 0.4 });
  box(TW, TH, 0.05, C.swFace, { parent: g, pos: [0, 0, ZF + 0.025] });
  const zf = ZF + 0.06;
  box(1.6, 0.05, 0.02, C.nvidia, { parent: g, pos: [-0.6, 0, zf], e: C.nvidia, ei: 0.6 });
  for (let i = 0; i < 2; i++) box(0.14, 0.12, 0.05, C.port, { parent: g, pos: [1.1 + i * 0.2, 0, zf], key: 'mgmt' });
  for (let i = 0; i < 3; i++) box(0.04, 0.04, 0.03, i ? 0x3ab0ff : C.led, { parent: g, pos: [1.7 + i * 0.1, 0, zf], e: i ? 0x3ab0ff : C.led, ei: 0.8 });
  rearCommon(g, 0.55, 0x1a2230);
  return g;
}

function shelf(parent, s) {
  const g = group(parent, 'shelf', [0, yOf(s.u), 0]);
  g.userData.title = `电源架 ${s.grp}${s.n}（${s.grp} 路）`;
  box(TW, TH, TD, C.psu, { parent: g, pos: [0, 0, ZC] });
  for (let i = 0; i < 6; i++) {
    const x = -1.83 + i * 0.73;
    box(0.68, TH * 0.9, 0.05, 0x22262d, { parent: g, pos: [x, 0, ZF + 0.03], key: 'psu' });
    cyl(0.13, 0.03, 0x0b0d10, { parent: g, pos: [x - 0.12, 0, ZF + 0.06], rot: [Math.PI / 2, 0, 0], key: 'psu' });
    box(0.04, 0.04, 0.03, C.led, { parent: g, pos: [x + 0.22, 0.08, ZF + 0.06], e: C.led, ei: 0.8 });
  }
  box(0.3, 0.26, 0.22, C.bus, { parent: g, pos: [0, 0, ZR - 0.11], m: 0.9, r: 0.3, key: 'busbar' });
  box(0.4, 0.28, 0.2, 0x111111, { parent: g, pos: [-1.8, 0, ZR - 0.1], key: 'feed' });
  box(0.3, 0.06, 0.02, s.grp === 'A' ? 0xff8a3a : 0xffd23a, { parent: g, pos: [-1.8, 0.1, ZR - 0.21], key: 'feed' });
  return g;
}

function mgmt(parent, s) {
  const g = group(parent, 'mgmt', [0, yOf(s.u), 0]);
  g.userData.title = '管理交换机';
  box(TW, TH, TD * 0.5, 0x262b33, { parent: g, pos: [0, 0, ZC + TD * 0.25] });
  for (let i = 0; i < 24; i++) box(0.11, 0.1, 0.03, C.port, { parent: g, pos: [-1.9 + i * 0.13, 0, ZF + 0.02] });
  return g;
}
const cableMat = new THREE.LineBasicMaterial({ color: C.copper, transparent: true, opacity: 0.32 });
const cableHi = new THREE.LineBasicMaterial({ color: 0x9dff3a, transparent: true, opacity: 0.95 });

// 一根 NVLink 铜缆：托盘后连接器 → 电缆盒内竖直走线 → 交换托盘后连接器
function cablePath(c, i, j, yi, yj) {
  const x0 = CART_X[c] + (j - 4) * 0.03, x = CART_X[c] + (j - 4) * 0.045;
  const zd = -3.95 - ((i * 7 + j * 3) % 10) * 0.05, zs = ZR - 0.18;
  const d = Math.sign(yj - yi) * Math.min(0.5, Math.abs(yj - yi) / 3);
  return curve([[x0, yi, zs], [x, yi, zd + 0.15], [x, yi + d, zd], [x, yj - d, zd], [x, yj, zd + 0.15], [x0 + 0.01, yj, zs]]);
}

export function buildRack(ctx) {
  const root = new THREE.Group();
  const slots = layout(), labels = [];
  const H = Y0 + 48 * U, RW = 6, RD = 12;
  const L = (t, p, cls) => { const l = label(t, p, root, cls); labels.push(l); return l; };

  // 机架框架与外壳
  const frame = group(root, 'rack');
  for (const sx of [-1, 1]) for (const sz of [-1, 1])
    box(0.16, H, 0.16, C.frame, { parent: frame, pos: [sx * (RW / 2 - 0.08), H / 2, sz * (RD / 2 - 0.08)], m: 0.7 });
  for (const sx of [-1, 1]) for (const z of [ZF - 0.05, ZR + 0.6])
    box(0.12, H - 0.6, 0.08, 0x3a414c, { parent: frame, pos: [sx * 2.3, H / 2, z], m: 0.8 });
  box(RW, 0.3, RD, C.frame, { parent: frame, pos: [0, 0.15, 0] });
  box(RW, 0.2, RD, C.frame, { parent: frame, pos: [0, H + 0.1, 0] });
  const shell = group(root, 'rack');
  for (const sx of [-1, 1]) box(0.04, H, RD, 0x30445e, { parent: shell, pos: [sx * RW / 2, H / 2, 0], op: 0.1, m: 0.2 });
  box(RW, 0.04, RD, 0x30445e, { parent: shell, pos: [0, H + 0.22, 0], op: 0.12 });

  const trays = [], ctY = {}, swY = {};
  const mk = { ct: computeTray, sw: switchTray, shelf, mgmt };
  for (const s of slots) {
    const o = mk[s.type](root, s);
    o.userData.slot = s; o.userData.base = o.position.clone();
    trays.push(o);
    if (s.type === 'ct') ctY[s.n] = o.position.y;
    if (s.type === 'sw') swY[s.n] = o.position.y;
  }
  const shelves = slots.filter(s => s.type === 'shelf');
  const yBus0 = yOf(shelves[0].u) - 0.25, yBus1 = yOf(shelves[shelves.length - 1].u) + 0.25;
  const ctMin = ctY[18] - 0.35, ctMax = ctY[1] + 0.35;

  // 直流母排
  box(0.26, yBus1 - yBus0, 0.18, C.bus, { parent: root, pos: [0, (yBus0 + yBus1) / 2, ZR - 0.31], key: 'busbar', m: 0.95, r: 0.25 });
  // NVLink 电缆盒（spine）
  for (const x of CART_X)
    box(0.55, ctMax - ctMin, 0.75, 0x1b2330, { parent: root, pos: [x, (ctMin + ctMax) / 2, -4.2], key: 'spine', op: 0.22, m: 0.3 });
  // 铜缆：18 计算托盘 × 4 电缆盒 × 9 交换托盘 = 648 束（每束代表 2 条链路）
  const cableGroups = {}, nvCurves = [];
  for (let i = 1; i <= 18; i++) {
    const pos = [];
    for (let c = 0; c < 4; c++) for (let j = 1; j <= 9; j++) {
      const cv = cablePath(c, i, j, ctY[i], swY[j]);
      const p = cv.getPoints(18);
      for (let k = 0; k < p.length - 1; k++) pos.push(p[k].x, p[k].y, p[k].z, p[k + 1].x, p[k + 1].y, p[k + 1].z);
      if ((i + c * 3 + j) % 9 === 0) nvCurves.push(cv);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    const ls = new THREE.LineSegments(g, cableMat);
    ls.userData.noPick = true;
    cableGroups[i] = ls; root.add(ls);
  }
  // 液冷歧管：左蓝（供水）右红（回水），对应托盘后部左侧冷 UQD / 右侧热 UQD
  const mY0 = yBus0 - 0.6, mY1 = ctMax + 0.4;
  for (const [sx, col] of [[-1, C.cold], [1, C.hot]]) {
    cyl(0.16, mY1 - mY0, col, { parent: root, pos: [sx * MAN_X, (mY0 + mY1) / 2, MAN_Z], key: 'manifold', m: 0.5, r: 0.35 });
    for (const s of slots) if (s.type === 'ct' || s.type === 'sw') {   // 每托盘一段软管 + 机架侧 UQD 母头
      const y = yOf(s.u);
      tube(curve([[sx * UQD_X, y, ZR - 0.32], [sx * UQD_X, y, ZR - 0.6], [sx * (MAN_X - 0.1), y, MAN_Z + 0.35], [sx * MAN_X, y, MAN_Z]]),
        0.035, col, { parent: root, key: 'manifold', seg: 12, rs: 6 });
      cyl(0.08, 0.1, 0x8c96a3, { parent: root, pos: [sx * UQD_X, y, ZR - 0.36], rot: [Math.PI / 2, 0, 0], key: 'uqd', m: 0.9, r: 0.3 });
    }
  }
  // CDU（机架旁）与管路：CDU → 蓝歧管底部；红歧管底部 → CDU；CDU ↔ 机房一次回路
  const CX = -6.2, CZ = -2;
  const cdu = group(root, 'cdu', [CX, 0, CZ]);
  cdu.userData.title = 'CDU 冷却液分配单元';
  box(3, 7, 4, 0x2c3440, { parent: cdu, pos: [0, 3.5, 0], m: 0.5 });
  box(2.6, 1.2, 0.05, 0x111821, { parent: cdu, pos: [0, 5.6, 2.03] });
  for (let i = 0; i < 2; i++) cyl(0.45, 1.2, 0x4a5563, { parent: cdu, pos: [-0.6 + i * 1.2, 1.2, 2.2], key: 'pump', m: 0.8, r: 0.3 });
  box(1.8, 2, 0.4, 0x6a7686, { parent: cdu, pos: [0, 3.4, 2.2], key: 'hx', m: 0.8, r: 0.35 });
  const cSup = curve([[CX + 1.5, 0.9, CZ - 1.2], [-3.2, 0.5, MAN_Z], [-MAN_X, 0.5, MAN_Z], [-MAN_X, mY0, MAN_Z]]);
  const cRet = curve([[MAN_X, mY0, MAN_Z], [MAN_X, 0.35, MAN_Z - 0.4], [-3.2, 0.35, MAN_Z - 0.5], [CX + 1.5, 1.6, CZ - 1.6]]);
  tube(cSup, 0.12, C.cold, { parent: root, key: 'cdu' });
  tube(cRet, 0.12, C.hot, { parent: root, key: 'cdu' });
  const fSup = curve([[CX - 0.6, 7, CZ], [CX - 0.6, 9.5, CZ], [CX - 0.6, 10.5, CZ - 6]]);
  const fRet = curve([[CX + 0.6, 10.5, CZ - 6], [CX + 0.6, 9.5, CZ], [CX + 0.6, 7, CZ]]);
  tube(fSup, 0.14, 0x58d0c0, { parent: root, key: 'facility' });
  tube(fRet, 0.14, 0xff9a5a, { parent: root, key: 'facility' });

  // A/B 交流馈线：A 从顶部进入上方电源架，B 从地板进入下方电源架
  const feedA = curve([[-1.8, H + 4, ZR - 0.6], [-1.8, H + 0.6, ZR - 0.6], [-1.8, yOf(39), ZR - 0.4], [-1.8, yOf(36), ZR - 0.25]]);
  const feedB = curve([[-1.8, -1.5, ZR - 0.6], [-1.8, 0.4, ZR - 0.6], [-1.8, yOf(3), ZR - 0.4], [-1.8, yOf(6), ZR - 0.25]]);
  tube(feedA, 0.07, 0xff8a3a, { parent: root, key: 'feed' });
  tube(feedB, 0.07, 0xffd23a, { parent: root, key: 'feed' });

  // ---- 流动叠加层 ----
  const fxNv = flow(nvCurves, { count: 3, color: 0x9dff3a, size: 0.035, speed: 0.3 });
  const pCurves = [feedA, feedB,
    curve([[0, yBus0, ZR - 0.42], [0, (yBus0 + ctMax) / 2, ZR - 0.42], [0, ctMax, ZR - 0.42]]),
    curve([[0, yBus1, ZR - 0.42], [0, (yBus1 + ctMin) / 2, ZR - 0.42], [0, ctMin, ZR - 0.42]])];
  for (const s of slots) if ((s.type === 'ct' && s.n % 3 === 1) || (s.type === 'sw' && s.n % 4 === 1))
    pCurves.push(curve([[0, yOf(s.u), ZR - 0.4], [0, yOf(s.u), ZR + 0.3], [0, yOf(s.u), ZC]]));
  const fxPw = flow(pCurves, { count: 5, color: 0xffd23a, size: 0.05, speed: 0.22 });
  const cCurves = [cSup, cRet, fSup, fRet];
  for (const s of slots) if (s.type === 'ct' && s.n % 4 === 2) {        // 冷→托盘内→热 的回路
    const y = yOf(s.u);
    cCurves.push(curve([[-MAN_X, y, MAN_Z], [-UQD_X, y, ZR - 0.4], [-1.4, y + 0.05, -1], [0, y + 0.05, 1.8], [1.4, y + 0.05, -1], [UQD_X, y, ZR - 0.4], [MAN_X, y, MAN_Z]]));
  }
  const fxCo = flow(cCurves, { count: 7, color: 0x9fe3ff, size: 0.06, speed: 0.12 });
  for (const f of [fxNv, fxPw, fxCo]) { f.obj.visible = false; root.add(f.obj); }
  // ---- 标签 ----
  const lx = -3.4;
  L('电源架 A 路 ×4', [lx, yOf(37.5), ZF], 'o');
  L('计算托盘 #1–#10', [lx, yOf(29.5), ZF], 'g');
  L('NVSwitch 托盘 ×9', [lx, yOf(20), ZF], 'b');
  L('计算托盘 #11–#18', [lx, yOf(11.5), ZF], 'g');
  L('电源架 B 路 ×4', [lx, yOf(4.5), ZF], 'o');
  L('管理交换机', [lx, yOf(44), ZF], 's');
  L('NVLink 铜缆盒 ×4', [CART_X[3] + 0.3, ctMax + 0.6, -4.2], 'g');
  L('直流母排', [0, yBus1 + 0.5, ZR - 0.3], 'o');
  L('供水歧管', [-MAN_X, mY1 + 0.5, MAN_Z], 'b');
  L('回水歧管', [MAN_X, mY1 + 0.5, MAN_Z], 'o');
  L('CDU', [CX, 7.6, CZ], 'b big');
  L('A 路 415V AC', [-1.8, H + 3, ZR - 0.6], 'o s');
  L('B 路 415V AC', [-1.8, -1.2, ZR - 0.6], 'o s');
  L('机房冷却水', [CX, 10.8, CZ - 6], 's');

  // ---- 交互状态 ----
  const st = { asm: -1, pulled: null, hiTray: null, shell: true };
  function highlight(n) {
    for (const k in cableGroups) {
      const on = n == null || +k === n;
      cableGroups[k].material = n == null ? cableMat : (on ? cableHi : cableMat);
      cableGroups[k].visible = n == null || on;
    }
    st.hiTray = n;
  }
  function trayOf(o) { while (o && !o.userData.slot) o = o.parent; return o; }

  function update(t, dt) {
    fxNv.update(t); fxPw.update(t); fxCo.update(t);
    for (const o of trays) {
      let tz = o.userData.base.z, ty = o.userData.base.y;
      if (o === st.pulled) tz += 3.2;
      if (st.asm >= 0) {                          // 出厂组装：托盘按 U 位依次从前方推入
        const k = trays.indexOf(o), p = ease(clamp01((st.asm - k * 0.12) / 0.9));
        tz += (1 - p) * 14; ty += (1 - p) * 1.5;
        o.visible = st.asm - k * 0.12 > -0.2;
      }
      o.position.z += (tz - o.position.z) * Math.min(1, dt * 8);
      o.position.y += (ty - o.position.y) * Math.min(1, dt * 8);
      if (st.asm >= 0 && st.asm - trays.indexOf(o) * 0.12 < 1) { o.position.z = tz; o.position.y = ty; }
    }
    if (st.asm >= 0) {
      st.asm += dt;
      const doneAt = trays.length * 0.12 + 1;
      const cp = clamp01((st.asm - doneAt + 1) / 1.2);  // 托盘到位后布线出现
      for (const k in cableGroups) cableGroups[k].visible = cp > (+k / 18) * 0.9;
      if (st.asm > doneAt + 1.4) { st.asm = -1; highlight(st.hiTray); }
    }
  }

  const tools = [
    { grp: '动画' },
    { label: '▶ 出厂组装过程', run: () => { st.pulled = null; st.asm = 0; } },
    { label: 'NVLink 数据流', toggle: v => { fxNv.obj.visible = v; } },
    { label: '供电流向', toggle: v => { fxPw.obj.visible = v; } },
    { label: '冷却液流向', toggle: v => { fxCo.obj.visible = v; } },
    { grp: '显示' },
    { label: '隐藏铜缆', toggle: v => { for (const k in cableGroups) cableGroups[k].visible = !v && (st.hiTray == null || +k === st.hiTray); } },
    { label: '隐藏外壳', toggle: v => { shell.visible = !v; } },
    { label: '全部推回托盘', run: () => { st.pulled = null; highlight(null); } },
  ];

  return {
    root, labels,
    camera: { pos: [17, 15, 21], target: [-1, 9.5, 0] },
    tools,
    legend: [[C.copper, 'NVLink 铜缆（选中计算托盘时高亮其链路）'], [C.cold, '供水（冷）'], [C.hot, '回水（热）'],
      [C.bus, '直流母排 ~50V'], [0xff8a3a, 'A 路交流'], [0xffd23a, 'B 路交流']],
    update,
    onSelect(o) {                                 // 选中托盘：抽出 3 格并高亮该托盘的链路
      const tr = trayOf(o);
      if (!tr) { st.pulled = null; highlight(null); return; }
      st.pulled = tr === st.pulled ? null : tr;
      highlight(tr.userData.slot.type === 'ct' && st.pulled ? tr.userData.slot.n : null);
    },
  };
}
