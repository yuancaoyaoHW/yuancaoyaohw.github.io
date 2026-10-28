// LEVEL 1：NVSwitch 托盘 / 供电系统 / 液冷系统
import * as THREE from 'three';
import { C, box, cyl, group, label, curve, tube, line, flow, coldPlate } from './kit.js';

function view(root) {
  const labels = [];
  return { labels, L: (t, p, cls, parent = root) => { const l = label(t, p, parent, cls); labels.push(l); return l; } };
}
const addFlow = (root, curves, o) => { const f = flow(curves, o); f.obj.visible = o.visible ?? true; root.add(f.obj); return f; };

// ---------------- NVSwitch 托盘（顶盖打开） ----------------
// 与计算托盘同一坐标：x 宽 9、z 深 16（前 +8 / 后 -8）
export function buildSwitch() {
  const root = new THREE.Group(), { labels, L } = view(root);
  const W = 9, D = 16, ZF = 8, ZR = -8;
  const CX = [-3.38, -1.95, 1.95, 3.38];               // 4 个后部连接器，对应机架 4 个电缆盒
  const SX = [-2.2, 2.2], SZ = -0.8;
  box(W, 0.2, D, C.sw, { parent: root, pos: [0, -0.1, 0], key: 'sw', m: 0.6, r: 0.45 });
  for (const sx of [-1, 1]) box(0.12, 0.9, D, C.sw, { parent: root, pos: [sx * (W / 2 - 0.06), 0.45, 0], key: 'sw' });
  box(W, 0.9, 0.12, C.swFace, { parent: root, pos: [0, 0.45, ZF - 0.06], key: 'sw' });
  // 前面板：管理口 / 指示灯 / 管理模块
  for (let i = 0; i < 2; i++) box(0.36, 0.28, 0.1, C.port, { parent: root, pos: [2.3 + i * 0.5, 0.45, ZF + 0.05], key: 'mgmt' });
  for (let i = 0; i < 3; i++) box(0.1, 0.1, 0.06, i ? 0x3ab0ff : C.led, { parent: root, pos: [3.6 + i * 0.22, 0.45, ZF + 0.04], e: i ? 0x3ab0ff : C.led, ei: 0.8 });
  box(3.2, 0.1, 0.05, C.nvidia, { parent: root, pos: [-1.6, 0.45, ZF + 0.03], e: C.nvidia, ei: 0.6 });
  const bmc = box(2.2, 0.1, 1.6, C.pcb, { parent: root, pos: [2.6, 0.06, 5.6], key: 'mgmt', m: 0.3 });
  bmc.userData.title = '交换托盘管理模块（BMC + 管理 CPU）';
  box(0.6, 0.1, 0.6, C.die, { parent: root, pos: [2.6, 0.15, 5.6], key: 'mgmt' });
  // 主板 + 2 颗 NVSwitch 芯片
  box(8.4, 0.08, 10, C.pcb, { parent: root, pos: [0, 0.04, -1.5], m: 0.3, r: 0.7 });
  const chips = [];
  for (const [i, x] of SX.entries()) {
    box(3, 0.14, 3, 0x2a3140, { parent: root, pos: [x, 0.15, SZ], key: 'nvsw', m: 0.5 });
    const c = box(1.8, 0.16, 1.8, C.die, { parent: root, pos: [x, 0.3, SZ], key: 'nvsw', m: 0.6, r: 0.3, material: new THREE.MeshStandardMaterial({ color: C.die, metalness: 0.6, roughness: 0.3, emissive: 0x3ab0ff, emissiveIntensity: 0 }) });
    c.userData.title = `NVLink Switch 芯片 ${i}`;
    chips.push(c);
    for (let k = 0; k < 8; k++)                          // 板上 VRM
      box(0.3, 0.18, 0.3, 0x3b4452, { parent: root, pos: [x - 1.4 + (k % 4) * 0.4 + (k < 4 ? 0 : 0.2), 0.12, SZ + (k < 4 ? 2 : -2)], key: 'pdb' });
  }
  const plates = SX.map(x => {
    const p = coldPlate(root, 3.2, 3.2, [x, 0.42, SZ], 'coldplate');
    p.userData.base = 0.42; p.userData.title = 'NVSwitch 冷板';
    return p;
  });
  // 后部：4 个 NVLink 盲插连接器、配电板/母排夹、UQD
  for (const x of CX) {
    const c = box(1.1, 0.7, 0.5, 0x1a2230, { parent: root, pos: [x, 0.38, ZR + 0.3], key: 'nvconn', m: 0.6 });
    c.userData.title = 'NVLink 盲插连接器（36 条链路）';
    box(0.9, 0.12, 0.04, C.gold, { parent: root, pos: [x, 0.5, ZR + 0.03], key: 'nvconn', m: 1, r: 0.3 });
  }
  box(0.8, 0.6, 0.6, C.bus, { parent: root, pos: [0, 0.32, ZR + 0.35], key: 'pdb', m: 0.9, r: 0.3 });
  cyl(0.2, 0.6, C.cold, { parent: root, pos: [-4, 0.4, ZR + 0.3], rot: [Math.PI / 2, 0, 0], key: 'uqd' });
  cyl(0.2, 0.6, C.hot, { parent: root, pos: [4, 0.4, ZR + 0.3], rot: [Math.PI / 2, 0, 0], key: 'uqd' });

  // 每颗芯片 → 每个连接器 18 条链路（画 3 根代表），共 2 × 4 × 18 = 144 端口
  const nvCurves = [], inCurves = [], outCurves = [];
  SX.forEach((x, si) => CX.forEach((cx, ci) => {
    for (let k = 0; k < 3; k++) {
      const o = (k - 1) * 0.12, sx = x + (ci - 1.5) * 0.35 + o;
      const cv = curve([[sx, 0.32, SZ - 0.95], [sx, 0.6, SZ - 2.5], [cx + o + si * 0.1 - 0.05, 0.6, ZR + 2], [cx + o + si * 0.1 - 0.05, 0.42, ZR + 0.55]]);
      tube(cv, 0.035, C.link, { parent: root, key: 'nvconn', seg: 24, rs: 5 });
      nvCurves.push(cv);
      if (k === 1) {
        inCurves.push(curve(cv.getPoints(12).reverse().map(p => p.toArray())));
        outCurves.push(cv);
      }
    }
  }));
  // 液冷：冷 UQD → 冷板0 → 冷板1 → 热 UQD（串联示意）
  const cool = [
    curve([[-4, 0.4, ZR + 0.6], [-4, 0.9, -5], [SX[0] - 0.96, 0.9, SZ - 1.5], [SX[0] - 0.96, 0.64, SZ - 1.12]]),
    curve([[SX[0] + 0.96, 0.64, SZ - 1.12], [SX[0] + 0.96, 1.0, SZ - 1.6], [SX[1] - 0.96, 1.0, SZ - 1.6], [SX[1] - 0.96, 0.64, SZ - 1.12]]),
    curve([[SX[1] + 0.96, 0.64, SZ - 1.12], [SX[1] + 0.96, 0.9, SZ - 1.5], [4, 0.9, -5], [4, 0.4, ZR + 0.6]]),
  ];
  cool.forEach((cv, i) => tube(cv, 0.07, i === 0 ? C.cold : i === 2 ? C.hot : 0xa060ff, { parent: root, key: 'coldplate', seg: 32 }));
  // 供电：母排夹 → 配电 → 各芯片 VRM
  const pw = SX.flatMap(x => [curve([[0, 0.3, ZR + 0.7], [x * 0.5, 0.2, -4], [x, 0.2, SZ - 2]]), curve([[0, 0.3, ZR + 0.7], [x * 0.3, 0.2, -2], [x, 0.2, SZ + 2]])]);
  pw.forEach(cv => tube(cv, 0.05, C.bus, { parent: root, key: 'pdb', seg: 24 }));

  const fxNv = addFlow(root, nvCurves, { count: 3, color: 0x9dff3a, size: 0.07, speed: 0.35 });
  const fxIn = addFlow(root, inCurves, { count: 4, color: 0x9dff3a, size: 0.1, speed: 0.6, visible: false });
  const fxOut = addFlow(root, outCurves, { count: 4, color: 0xffa040, size: 0.1, speed: 0.6, visible: false });
  const fxCo = addFlow(root, cool, { count: 8, color: 0x9fe3ff, size: 0.08, speed: 0.2, visible: false });
  const fxPw = addFlow(root, pw, { count: 5, color: 0xffd23a, size: 0.07, speed: 0.3, visible: false });

  L('NVLink Switch 芯片 ×2', [0, 1.6, SZ], 'b big');
  L('每颗 72 端口', [SX[1], 1.1, SZ + 1.6], 'b s');
  L('NVLink 盲插连接器 ×4（→ 机架电缆盒）', [0, 1.4, ZR + 0.3], 'g');
  L('冷 UQD', [-4, 1.0, ZR + 0.3], 'b s');
  L('热 UQD', [4, 1.0, ZR + 0.3], 'o s');
  L('母排取电夹', [0, 1.0, ZR + 0.9], 'o s');
  L('管理模块 / BMC', [2.6, 0.7, 5.6], 's');
  L('前面板：管理口 · 状态灯', [1.2, 1.2, ZF], 's');
  const sharpL = L('SHARP：在交换芯片内求和', [0, 2.3, SZ], 'o');
  sharpL.visible = false;

  // 状态：SHARP 动画 0→1 阶段（汇入 → 芯片求和 → 广播）
  const st = { sharp: -1, lift: 0, liftT: 0 };
  function update(t, dt) {
    fxNv.update(t); fxCo.update(t); fxPw.update(t);
    st.lift += (st.liftT - st.lift) * Math.min(1, dt * 5);
    plates.forEach(p => { p.position.y = p.userData.base + st.lift * 2.2; });
    if (st.sharp >= 0) {
      st.sharp += dt / 4;                                  // 一轮 4 秒
      const ph = st.sharp % 1;
      fxIn.obj.visible = ph < 0.42;
      fxOut.obj.visible = ph > 0.58;
      fxIn.update(t); fxOut.update(t);
      const glow = ph > 0.38 && ph < 0.62 ? 1.4 : 0.15;
      chips.forEach(c => { c.material.emissiveIntensity += (glow - c.material.emissiveIntensity) * Math.min(1, dt * 10); });
    }
  }

  const tools = [
    { grp: '拆解' },
    { label: '抬起冷板', toggle: v => { st.liftT = v ? 1 : 0; } },
    { grp: '动画' },
    { label: 'NVLink 端口流量', on: true, toggle: v => { fxNv.obj.visible = v; } },
    { label: '▶ SHARP all-reduce', toggle: v => {
      st.sharp = v ? 0 : -1; sharpL.visible = v; fxNv.obj.visible = !v;
      if (!v) { fxIn.obj.visible = fxOut.obj.visible = false; chips.forEach(c => { c.material.emissiveIntensity = 0; }); }
    } },
    { label: '冷却液流向', toggle: v => { fxCo.obj.visible = v; } },
    { label: '供电流向', toggle: v => { fxPw.obj.visible = v; } },
  ];
  return {
    root, labels, tools, update,
    camera: { pos: [9, 11, 13], target: [0, 0, -1] },
    legend: [[C.link, 'NVLink 端口走线（每根代表 6 条链路）'], [0x9dff3a, 'SHARP：GPU 数据汇入'], [0xffa040, 'SHARP：求和结果广播'],
      [C.cold, '供水'], [C.hot, '回水'], [C.bus, '50V 直流']],
  };
}

// ---------------- 供电系统 ----------------
// 示意：上 A 路 4 个电源架、下 B 路 4 个电源架，中间托盘；背部竖直母排（z 负方向为机架背面）
export function buildPower() {
  const root = new THREE.Group(), { labels, L } = view(root);
  const SW = 8, SH = 0.6, SD = 5, ZB = -2.95;           // 电源架宽/高/深，母排 z
  const yA = [9.2, 9.85, 10.5, 11.15], yB = [0.5, 1.15, 1.8, 2.45];
  const trayY = Array.from({ length: 9 }, (_, i) => 3.3 + i * 0.6);   // 9 片代表 27 个托盘
  const ledMat = () => new THREE.MeshStandardMaterial({ color: C.led, emissive: C.led, emissiveIntensity: 0.9 });
  const shelves = [];
  const mkShelf = (y, grp, n) => {
    const g = group(root, 'shelf', [0, y, 0]);
    g.userData.title = `电源架 ${grp}${n}（${grp} 路）`;
    box(SW, SH, SD, C.psu, { parent: g, m: 0.5 });
    const psus = [], leds = [];
    for (let i = 0; i < 6; i++) {
      const x = -3.3 + i * 1.32;
      const p = group(g, 'psu', [x, 0, 0]);
      p.userData.title = `PSU ${grp}${n}-${i + 1}（5.5 kW）`;
      box(1.2, SH * 0.86, 0.3, 0x22262d, { parent: p, pos: [0, 0, SD / 2 + 0.1] });
      cyl(0.17, 0.05, 0x0b0d10, { parent: p, pos: [-0.25, 0, SD / 2 + 0.27], rot: [Math.PI / 2, 0, 0] });
      const led = ledMat();
      box(0.08, 0.08, 0.04, C.led, { parent: p, pos: [0.4, 0.12, SD / 2 + 0.27], material: led });
      p.userData.base = p.position.z;
      psus.push(p); leds.push(led);
    }
    box(0.7, 0.4, 0.3, C.bus, { parent: g, pos: [0, 0, -SD / 2 - 0.15], key: 'busbar', m: 0.9, r: 0.3 });
    box(0.8, 0.42, 0.3, 0x111111, { parent: g, pos: [-3.3, 0, -SD / 2 - 0.15], key: 'feed' });
    shelves.push({ g, grp, psus, leds, y });
  };
  yA.forEach((y, i) => mkShelf(y, 'A', i + 1));
  yB.forEach((y, i) => mkShelf(y, 'B', 4 - i));
  // 托盘（示意 9 片），每片后部一个母排取电夹
  trayY.forEach((y, i) => {
    const sw = i >= 3 && i <= 5;
    const t = box(SW, 0.36, SD, sw ? C.sw : C.tray, { parent: root, pos: [0, y, 0], key: sw ? 'sw' : 'ct', m: 0.5, op: 0.85 });
    t.userData.title = sw ? 'NVSwitch 托盘（示意）' : '计算托盘（示意）';
    box(0.6, 0.3, 0.35, C.bus, { parent: root, pos: [0, y, -SD / 2 - 0.17], key: 'pdb', m: 0.9, r: 0.3 });
  });
  // 竖直直流母排
  const bus = box(0.4, yA[3] - yB[0] + 0.6, 0.3, C.bus, { parent: root, pos: [0, (yA[3] + yB[0]) / 2, ZB], key: 'busbar', m: 0.95, r: 0.25 });
  bus.userData.title = '直流母排（约 50V）';
  // 机房 PDU 与 A/B 交流馈线
  const pduA = box(1.4, 2, 1, 0x2a2f38, { parent: root, pos: [-7, 14, -3.4], key: 'feed' });
  pduA.userData.title = 'A 路 PDU（机房配电）';
  const pduB = box(1.4, 2, 1, 0x2a2f38, { parent: root, pos: [-7, -2, -3.4], key: 'feed' });
  pduB.userData.title = 'B 路 PDU（机房配电）';
  const acA = [], acB = [];
  for (const y of yA) acA.push(curve([[-7, 13, -3.4], [-5.2, y + 0.6, -3.4], [-4.2, y, -3.1], [-3.3, y, -2.85]]));
  for (const y of yB) acB.push(curve([[-7, -1, -3.4], [-5.2, y - 0.6, -3.4], [-4.2, y, -3.1], [-3.3, y, -2.85]]));
  const matA = new THREE.MeshStandardMaterial({ color: 0xff8a3a, metalness: 0.4, roughness: 0.5 });
  acA.forEach(cv => tube(cv, 0.08, 0, { parent: root, key: 'feed', seg: 24, material: matA }));
  acB.forEach(cv => tube(cv, 0.08, 0xffd23a, { parent: root, key: 'feed', seg: 24 }));
  // 托盘内部放大：取电夹 → 配电板 PDB → 多相 VRM → GPU/CPU（右侧示意）
  const DX = 11, DY = 6.2;
  const plat = box(6.5, 0.12, 6, C.pcb, { parent: root, pos: [DX, DY, 0], key: 'pdb', m: 0.3, op: 0.95 });
  plat.userData.title = '托盘内部供电链路（放大示意）';
  box(0.8, 0.5, 0.6, C.bus, { parent: root, pos: [DX, DY + 0.3, -2.7], key: 'pdb', m: 0.9, r: 0.3 });
  const pdb = box(2.4, 0.2, 1, 0x3b4452, { parent: root, pos: [DX, DY + 0.16, -1.4], key: 'pdb' });
  pdb.userData.title = '配电板 PDB（50V 分配）';
  const chipX = [DX - 2.25, DX - 0.75, DX + 0.75, DX + 2.25], chainCv = [];
  chipX.forEach((x, i) => {
    for (let k = 0; k < 3; k++) box(0.3, 0.26, 0.3, 0x56606e, { parent: root, pos: [x - 0.4 + k * 0.4, DY + 0.19, 0.6], key: 'pdb' });
    const g = box(1, 0.16, 1, i === 1 ? C.grace : C.die, { parent: root, pos: [x, DY + 0.14, 1.9], key: i === 1 ? 'grace' : 'b200' });
    g.userData.title = i === 1 ? 'Grace CPU（示意）' : 'B200 GPU（示意）';
    chainCv.push(curve([[DX, DY + 0.4, -2.4], [DX, DY + 0.3, -1.4], [x, DY + 0.3, -0.3], [x, DY + 0.36, 0.6], [x, DY + 0.26, 1.9]]));
  });
  const zoomCv = curve([[0.3, trayY[4], -2.6], [4, trayY[4] + 0.4, -2.8], [DX - 1, DY + 0.5, -3], [DX, DY + 0.4, -2.7]]);
  line(zoomCv, 0x8899aa, root, 40, 0.5);

  // 流动：交流 → 电源架；直流：电源架 → 母排 → 托盘
  const yMid = 5.7;
  const dcA = yA.map(y => curve([[0, y, -2.7], [0.12, y, ZB - 0.2], [0.12, (y + yMid) / 2, ZB - 0.22], [0.12, yMid, ZB - 0.22]]));
  const dcB = yB.map(y => curve([[0, y, -2.7], [-0.12, y, ZB - 0.2], [-0.12, (y + yMid) / 2, ZB - 0.22], [-0.12, yMid, ZB - 0.22]]));
  const toTray = trayY.map(y => curve([[0, y, ZB], [0, y, -2.4], [0, y + 0.2, 0]]));
  const fxAcA = addFlow(root, acA, { count: 5, color: 0xffb070, size: 0.09, speed: 0.25 });
  const fxAcB = addFlow(root, acB, { count: 5, color: 0xffe680, size: 0.09, speed: 0.25 });
  const fxDcA = addFlow(root, dcA, { count: 6, color: 0xffd23a, size: 0.08, speed: 0.22 });
  const fxDcB = addFlow(root, dcB, { count: 6, color: 0xffd23a, size: 0.08, speed: 0.22 });
  const fxTray = addFlow(root, toTray, { count: 3, color: 0xffd23a, size: 0.07, speed: 0.3 });
  const fxChain = addFlow(root, chainCv, { count: 5, color: 0xfff0a0, size: 0.07, speed: 0.3 });

  L('A 路 PDU · 415V 三相交流', [-7, 15.4, -3.4], 'o');
  L('B 路 PDU · 415V 三相交流', [-7, -3.4, -3.4], 'o');
  const lA = L('电源架 A ×4 · 负载 50%', [-5.6, yA[3] + 0.8, 2.5], 'o');
  const lB = L('电源架 B ×4 · 负载 50%', [-5.6, yB[0] - 0.8, 2.5], 'o');
  L('计算 / 交换托盘（示意）', [-5.4, trayY[4], 2.5], 'g');
  L('直流母排 ≈50V · 数千安', [0, yA[3] + 1.1, ZB], 'o');
  L('取电夹', [0.9, trayY[0], -2.9], 'o s');
  L('托盘内部放大', [DX, DY + 1.5, -2.7], 's big');
  L('取电夹 → 配电板', [DX + 1.8, DY + 0.6, -2.2], 's');
  L('多相 VRM：50V → ≈0.8V', [DX + 3.6, DY + 0.6, 0.6], 's');
  L('GPU · CPU · GPU · GPU', [DX, DY + 0.7, 2.6], 's');
  const lSwap = L('热插拔：其余 5 个 PSU 分担负载', [3.3, yB[1] + 0.7, 4.8], 'o s');
  lSwap.visible = false;

  const st = { aOff: false, swap: 0, swapT: 0 };
  const swapPsu = shelves[5].psus[2], swapLed = shelves[5].leds[2];
  function setAOff(v) {
    st.aOff = v;
    fxAcA.obj.visible = fxDcA.obj.visible = !v;
    matA.color.setHex(v ? 0x3a3330 : 0xff8a3a);
    for (const s of shelves) if (s.grp === 'A') s.leds.forEach(m => { m.color.setHex(v ? 0xff3030 : C.led); m.emissive.setHex(v ? 0xff3030 : C.led); });
    lA.element.textContent = v ? '电源架 A ×4 · 断电（0%）' : '电源架 A ×4 · 负载 50%';
    lB.element.textContent = v ? '电源架 B ×4 · 独自承担 100%' : '电源架 B ×4 · 负载 50%';
  }
  function update(t, dt) {
    fxAcA.update(t); fxAcB.update(t); fxDcA.update(t); fxTray.update(t); fxChain.update(t);
    fxDcB.update(st.aOff ? t * 2 : t);                    // A 路断电：B 路电流翻倍
    st.swap += (st.swapT - st.swap) * Math.min(1, dt * 4);
    swapPsu.position.z = swapPsu.userData.base + st.swap * 1.8;
    const on = st.swap < 0.05;
    swapLed.emissiveIntensity = on ? 0.9 : 0;
  }
  const tools = [
    { grp: '故障演示' },
    { label: '⚡ A 路断电', toggle: setAOff },
    { label: '热插拔一个 PSU', toggle: v => { st.swapT = v ? 1 : 0; lSwap.visible = v; } },
    { grp: '动画' },
    { label: '交流流向', on: true, toggle: v => { fxAcA.obj.visible = v && !st.aOff; fxAcB.obj.visible = v; } },
    { label: '直流流向', on: true, toggle: v => { fxDcA.obj.visible = v && !st.aOff; fxDcB.obj.visible = v; fxTray.obj.visible = fxChain.obj.visible = v; } },
  ];
  return {
    root, labels, tools, update,
    camera: { pos: [14, 12, 22], target: [2, 6, 0] },
    legend: [[0xff8a3a, 'A 路 415V 交流'], [0xffd23a, 'B 路 415V 交流'], [C.bus, '≈50V 直流母排 / 取电夹'], [C.led, 'PSU 正常'], [0xff3030, 'PSU 失电']],
  };
}

// ---------------- 液冷系统 ----------------
// 示意：6 个打开顶盖的托盘（代表 27 个）· 背部供/回水歧管 · UQD · CDU · 机房一次回路
export function buildCooling() {
  const root = new THREE.Group(), { labels, L } = view(root);
  const N = 6, TW = 5.6, TD = 7, UX = 2.4, MX = 3.2, MZ = -4.6, UNP = 2;
  const ty = i => 1.4 + i * 1.15;
  const mY0 = 0.6, mY1 = ty(N - 1) + 0.9;
  const GPU = [[-2.1, 1.2], [-0.7, 1.2], [0.7, 1.2], [2.1, 1.2]], CPU = [[-1.4, -1.2], [1.4, -1.2]];
  // 托盘内回路：冷 UQD → CPU0 → GPU0 → GPU1 ‖ GPU2 → GPU3 → CPU1 → 热 UQD（串联示意）
  const inPts = y => [[-UX, y, -3.7], [-UX, y + 0.3, -2.6], [CPU[0][0], y + 0.35, CPU[0][1]], [GPU[0][0], y + 0.35, GPU[0][1]], [GPU[1][0], y + 0.35, GPU[1][1]]];
  const outPts = y => [[GPU[2][0], y + 0.35, GPU[2][1]], [GPU[3][0], y + 0.35, GPU[3][1]], [CPU[1][0], y + 0.35, CPU[1][1]], [UX, y + 0.3, -2.6], [UX, y, -3.7]];
  const hoseIn = y => [[-(MX - 0.15), y, MZ], [-UX, y, -4.35], [-UX, y, -4.0]];
  const hoseOut = y => [[UX, y, -4.0], [UX, y, -4.35], [MX - 0.15, y, MZ]];

  const trays = [], cold = [], warm = [], unpCold = [], unpWarm = [];
  let unpUqd = null;
  for (let i = 0; i < N; i++) {
    const y = ty(i), g = group(root, 'ct', [0, y, 0]);
    g.userData.title = i === UNP ? '计算托盘（演示拔出）' : '计算托盘（示意）';
    g.userData.base = 0;
    box(TW, 0.1, TD, C.tray, { parent: g, pos: [0, -0.05, -0.1], m: 0.5 });
    for (const sx of [-1, 1]) box(0.08, 0.4, TD, C.tray, { parent: g, pos: [sx * (TW / 2 - 0.04), 0.15, -0.1], m: 0.5 });
    box(TW, 0.4, 0.06, C.trayFace, { parent: g, pos: [0, 0.15, TD / 2 - 0.1] });
    box(5.2, 0.05, 6, C.pcb, { parent: g, pos: [0, 0.03, 0], m: 0.3, r: 0.7 });
    for (const [x, z] of GPU) {
      box(0.9, 0.1, 0.9, C.die, { parent: g, pos: [x, 0.1, z], key: 'b200' });
      coldPlate(g, 1.1, 1.1, [x, 0.2, z], 'coldplate').userData.title = 'GPU 冷板';
    }
    for (const [x, z] of CPU) {
      box(0.8, 0.1, 0.8, C.grace, { parent: g, pos: [x, 0.1, z], key: 'grace' });
      coldPlate(g, 1.0, 1.0, [x, 0.2, z], 'coldplate').userData.title = 'CPU 冷板';
    }
    tube(curve(inPts(0)), 0.06, C.cold, { parent: g, key: 'coldplate', seg: 40 });
    tube(curve(outPts(0)), 0.06, C.hot, { parent: g, key: 'coldplate', seg: 40 });
    tube(curve([[GPU[1][0], 0.35, GPU[1][1]], [0, 0.42, 1.2], [GPU[2][0], 0.35, GPU[2][1]]]), 0.06, 0xa060ff, { parent: g, key: 'coldplate', seg: 12 });
    // 托盘侧 UQD（公头），随托盘移动
    for (const [sx, col] of [[-1, C.cold], [1, C.hot]])
      cyl(0.11, 0.3, col, { parent: g, pos: [sx * UX, 0, -3.7], rot: [Math.PI / 2, 0, 0], key: 'uqd', m: 0.7, r: 0.3 });
    // 机架侧 UQD（母头）+ 到歧管的软管，固定不动
    for (const sx of [-1, 1]) {
      const q = cyl(0.13, 0.22, 0x8c96a3, { parent: root, pos: [sx * UX, y, -4.0], rot: [Math.PI / 2, 0, 0], key: 'uqd', m: 0.9, r: 0.3, unique: i === UNP });
      if (i === UNP) (unpUqd ??= []).push(q);
      tube(curve((sx < 0 ? hoseIn : hoseOut)(y)), 0.05, sx < 0 ? C.cold : C.hot, { parent: root, key: 'manifold', seg: 12, rs: 6 });
    }
    const ci = curve([...hoseIn(y), ...inPts(y).slice(1)]), co = curve([...outPts(y).slice(0, -1), ...hoseOut(y)]);
    (i === UNP ? unpCold : cold).push(ci);
    (i === UNP ? unpWarm : warm).push(co);
    trays.push(g);
  }
  // 歧管：左供水（蓝）右回水（红）
  for (const [sx, col] of [[-1, C.cold], [1, C.hot]]) {
    const m = cyl(0.18, mY1 - mY0, col, { parent: root, pos: [sx * MX, (mY0 + mY1) / 2, MZ], key: 'manifold', m: 0.5, r: 0.35 });
    m.userData.title = sx < 0 ? '供水歧管（冷）' : '回水歧管（热）';
  }
  // CDU（机架旁）：冗余泵 ×2 + 板式换热器 + 储液罐
  const CX = -8.6, CZ = -2.2;
  const cdu = group(root, 'cdu', [CX, 0, CZ]);
  cdu.userData.title = 'CDU 冷却液分配单元';
  box(3, 6.4, 3, 0x2c3440, { parent: cdu, pos: [0, 3.2, 0], m: 0.5, op: 0.35 });
  box(2.6, 0.9, 0.05, 0x111821, { parent: cdu, pos: [0, 5.4, 1.53] });
  const pumpMats = [], rotors = [];
  for (let i = 0; i < 2; i++) {
    const p = group(cdu, 'pump', [-0.7 + i * 1.4, 1.0, 0.4]);
    p.userData.title = `循环泵 ${i ? 'B' : 'A'}（N+1）`;
    const pm = new THREE.MeshStandardMaterial({ color: 0x4a5563, metalness: 0.8, roughness: 0.3, emissive: 0xff2020, emissiveIntensity: 0 });
    cyl(0.5, 1.1, 0, { parent: p, material: pm });
    rotors.push(box(0.8, 0.08, 0.12, 0xd0d6de, { parent: p, pos: [0, 0.6, 0], m: 0.9 }));
    pumpMats.push(pm);
  }
  const hx = box(2, 2.2, 0.6, 0x6a7686, { parent: cdu, pos: [0, 3.2, 0.3], key: 'hx', m: 0.8, r: 0.35 });
  hx.userData.title = '板式换热器';
  for (let k = 0; k < 9; k++) box(0.04, 2.1, 0.62, k % 2 ? C.cold : 0x58d0c0, { parent: cdu, pos: [-0.8 + k * 0.2, 3.2, 0.31], key: 'hx' });
  cyl(0.45, 1.2, 0x5a6573, { parent: cdu, pos: [0.9, 4.9, -0.8], key: 'cdu' });
  // 二次回路：CDU → 供水歧管底；回水歧管底 → CDU
  const sup = curve([[CX + 1.5, 1.0, CZ - 0.6], [-5.5, 0.3, MZ + 0.3], [-MX, 0.3, MZ], [-MX, mY0, MZ]]);
  const ret = curve([[MX, mY0, MZ], [MX, 0.15, MZ - 0.5], [-5.5, 0.15, MZ - 0.6], [CX + 1.5, 1.8, CZ - 1.2]]);
  tube(sup, 0.12, C.cold, { parent: root, key: 'manifold' });
  tube(ret, 0.12, C.hot, { parent: root, key: 'manifold' });
  const upS = curve([[-MX, mY0, MZ], [-MX, (mY0 + mY1) / 2, MZ], [-MX, mY1 - 0.1, MZ]]);
  const dnR = curve([[MX, mY1 - 0.1, MZ], [MX, (mY0 + mY1) / 2, MZ], [MX, mY0, MZ]]);
  // 一次回路：CDU ↔ 机房冷却塔 / 干冷器
  const fS = curve([[CX - 0.8, 11, CZ - 7], [CX - 0.8, 9, CZ], [CX - 0.8, 6.4, CZ]]);
  const fR = curve([[CX + 0.8, 6.4, CZ], [CX + 0.8, 9, CZ], [CX + 0.8, 11, CZ - 7]]);
  tube(fS, 0.14, 0x58d0c0, { parent: root, key: 'facility' });
  tube(fR, 0.14, 0xff9a5a, { parent: root, key: 'facility' });
  const tower = box(4, 1.6, 2.4, 0x3a4250, { parent: root, pos: [CX, 11.6, CZ - 7.6], key: 'facility', op: 0.7 });
  tower.userData.title = '机房冷却塔 / 干冷器（示意）';

  const fxC = addFlow(root, [sup, upS, ...cold], { count: 6, color: 0x9fe3ff, size: 0.09, speed: 0.18 });
  const fxW = addFlow(root, [dnR, ret, ...warm], { count: 6, color: 0xffa080, size: 0.09, speed: 0.18 });
  const fxUC = addFlow(root, unpCold, { count: 6, color: 0x9fe3ff, size: 0.09, speed: 0.18 });
  const fxUW = addFlow(root, unpWarm, { count: 6, color: 0xffa080, size: 0.09, speed: 0.18 });
  const fxF = addFlow(root, [fS, fR], { count: 8, color: 0xc8fff0, size: 0.1, speed: 0.12 });

  L('供水歧管 ≈25–30 °C', [-MX, mY1 + 0.6, MZ], 'b');
  L('回水歧管 ≈40–45 °C', [MX, mY1 + 0.6, MZ], 'o');
  L('托盘内：CPU → GPU 冷板串联', [0, ty(N - 1) + 1.2, 1.2], 'g');
  L('UQD 快接头（托盘侧公头 · 机架侧母头）', [UX + 2.6, ty(0) - 0.4, -4.0], 's');
  L('CDU', [CX, 7.2, CZ], 'b big');
  L('冗余泵 A / B', [CX, 1.0, CZ + 2.2], 's');
  L('板式换热器', [CX, 3.2, CZ + 1.9], 's');
  L('二次回路（TCS）', [-5.5, 0.6, MZ + 0.4], 'b s');
  L('一次回路（FWS）· 冷却塔', [CX, 12.9, CZ - 7.6], 's');
  const lUnp = L('UQD 断开：双向阀门自动关闭，无滴漏', [0, ty(UNP) + 0.9, 3.5], 'o');
  const lPump = L('泵 B 故障 → 泵 A 自动接管', [CX, -0.4, CZ + 1.5], 'o s');
  lUnp.visible = lPump.visible = false;

  const st = { pull: 0, pullT: 0, pumpFail: false };
  const unpTray = trays[UNP];
  function update(t, dt) {
    fxC.update(t); fxW.update(t); fxF.update(t); fxUC.update(t); fxUW.update(t);
    st.pull += (st.pullT - st.pull) * Math.min(1, dt * 4);
    unpTray.position.z = st.pull * 3;
    const connected = st.pull < 0.03;
    fxUC.obj.visible = fxUW.obj.visible = connected && fxC.obj.visible;
    unpUqd.forEach(q => q.material.color.setHex(connected ? 0x8c96a3 : 0xffb020));
    rotors[0].rotation.y = t * 12;
    if (!st.pumpFail) rotors[1].rotation.y = t * 12;
    pumpMats[1].emissiveIntensity = st.pumpFail ? 0.6 + 0.4 * Math.sin(t * 8) : 0;
  }
  const tools = [
    { grp: '故障演示' },
    { label: '拔出一个托盘（UQD 断开）', toggle: v => { st.pullT = v ? 1 : 0; lUnp.visible = v; } },
    { label: '泵 B 故障（N+1）', toggle: v => { st.pumpFail = v; lPump.visible = v; } },
    { grp: '动画' },
    { label: '冷却液流向', on: true, toggle: v => { fxC.obj.visible = fxW.obj.visible = fxF.obj.visible = v; } },
  ];
  return {
    root, labels, tools, update,
    camera: { pos: [9, 12, 15], target: [-2.5, 4, -1.5] },
    legend: [[C.cold, '二次回路供水（冷）'], [C.hot, '二次回路回水（热）'], [0xa060ff, '冷板间串联管'],
      [0x58d0c0, '一次回路供水'], [0xff9a5a, '一次回路回水'], [0xffb020, 'UQD 已断开（阀门关闭）']],
  };
}
