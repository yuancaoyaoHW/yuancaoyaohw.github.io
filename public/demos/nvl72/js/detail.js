// LEVEL 1–3：计算托盘 → GB200 超级芯片 → B200 GPU
import * as THREE from 'three';
import { C, box, cyl, group, label, curve, tube, flow, coldPlate } from './kit.js';

function view(root) {
  const labels = [];
  return { labels, L: (t, p, cls, parent = root) => { const l = label(t, p, parent, cls); labels.push(l); return l; } };
}
const addFlow = (root, curves, o) => { const f = flow(curves, o); f.obj.visible = o.visible ?? true; root.add(f.obj); return f; };

// ---------------- 计算托盘（顶盖打开） ----------------
// 坐标：x 宽 9、z 深 16（前 +8 / 后 -8），y 向上
export function buildCompute() {
  const root = new THREE.Group(), { labels, L } = view(root);
  const W = 9, D = 16, ZF = D / 2, ZR = -D / 2;
  box(W, 0.15, D, C.tray, { parent: root, pos: [0, 0, 0], m: 0.6, r: 0.5, key: 'ct' });
  for (const sx of [-1, 1]) box(0.1, 0.8, D, C.tray, { parent: root, pos: [sx * W / 2, 0.4, 0], key: 'ct' });
  box(W, 0.8, 0.12, C.trayFace, { parent: root, pos: [0, 0.4, ZF], key: 'ct' });
  box(W, 0.8, 0.12, C.trayFace, { parent: root, pos: [0, 0.4, ZR], key: 'ct' });
  const lid = box(W, 0.06, D, C.tray, { parent: root, pos: [0, 0.84, 0], op: 0.18, key: 'ct' });
  lid.visible = false;

  // 前面板：4×OSFP（ConnectX）、6×E1.S、2×BlueField-3、BMC
  const nicX = [-3.9, -3.2, -2.5, -1.8];
  nicX.forEach((x, i) => {
    box(0.55, 0.42, 0.14, C.cage, { parent: root, pos: [x, 0.4, ZF + 0.08], key: 'cx', m: 0.85, r: 0.3 });
    box(0.4, 0.22, 0.03, C.port, { parent: root, pos: [x, 0.4, ZF + 0.16], key: 'cx' });
    box(0.6, 0.05, 1.6, C.nic, { parent: root, pos: [x, 0.3, ZF - 1.1], key: 'cx', m: 0.3 });          // 网卡板
    box(0.35, 0.08, 0.35, 0x222630, { parent: root, pos: [x, 0.37, ZF - 1.3], key: 'cx', m: 0.6 });
  });
  for (let i = 0; i < 6; i++) box(0.24, 0.62, 2.2, 0x3b4452, { parent: root, pos: [-0.9 + i * 0.3, 0.42, ZF - 1.15], key: 'e1s' });
  const bfX = [1.6, 2.5];
  for (const x of bfX) {
    box(0.6, 0.3, 0.12, 0x6b2b20, { parent: root, pos: [x, 0.4, ZF + 0.08], key: 'bf3' });
    box(0.8, 0.05, 1.8, C.bf3, { parent: root, pos: [x, 0.3, ZF - 1.2], key: 'bf3', m: 0.3 });
  }
  box(0.3, 0.25, 0.1, C.port, { parent: root, pos: [3.5, 0.4, ZF + 0.08], key: 'mgmt' });

  // 两块 Bianca 板：每块 1 Grace + 2 B200
  const gpus = [], graces = [], plates = [];
  for (const [bi, bx] of [[0, -2.25], [1, 2.25]]) {
    const b = group(root, 'board', [bx, 0.1, 0.3]);
    b.userData.title = `GB200 超级芯片 #${bi}`;
    box(4.1, 0.06, 6.8, C.pcb, { parent: b, pos: [0, 0.05, 0], m: 0.2, r: 0.8, key: 'board' });
    box(1.1, 0.12, 1.1, C.grace, { parent: b, pos: [0, 0.14, 2], key: 'grace', m: 0.5, r: 0.4 });
    graces.push(new THREE.Vector3(bx, 0.3, 2.3));
    for (const sx of [-1, 1]) for (let k = 0; k < 4; k++)   // Grace 两侧 LPDDR5X
      box(0.5, 0.06, 0.36, 0x1a1d24, { parent: b, pos: [sx * 1.25, 0.11, 1.25 + k * 0.5], key: 'lpddr' });
    for (const [gi, gx] of [[0, -1], [1, 1]]) {
      const g = group(b, 'b200', [gx, 0.1, -1.4]);
      g.userData.title = `B200 GPU（托盘内 GPU${bi * 2 + gi}）`;
      box(1.5, 0.06, 1.7, 0x1f5a3d, { parent: g, pos: [0, 0, 0], key: 'b200' });
      box(1.2, 0.08, 0.6, C.die, { parent: g, pos: [0, 0.07, 0], key: 'b200', m: 0.6 });
      for (const hx of [-0.45, -0.15, 0.15, 0.45]) for (const hz of [-0.5, 0.5])
        box(0.22, 0.09, 0.28, C.hbm, { parent: g, pos: [hx, 0.08, hz], key: 'hbm' });
      gpus.push(new THREE.Vector3(bx + gx, 0.3, 0.3 - 1.4));
    }
    // 冷板：GPU×2 + Grace×1
    for (const [px, pz, w, d] of [[-1, -1.4, 1.5, 1.6], [1, -1.4, 1.5, 1.6], [0, 2, 1.2, 1.2]]) {
      const p = coldPlate(root, w, d, [bx + px, 0.42, 0.3 + pz], 'coldplate');
      Object.assign(p.userData, { base: p.position.y, w, d }); plates.push(p);
    }
  }
  // 后面板：4 个 NVLink 盲插连接器（GPU k → 连接器 k）、配电板、冷/热 UQD
  const connX = [-3.3, -1.9, 1.9, 3.3];
  connX.forEach((x, k) => {
    const c = box(0.9, 0.5, 0.25, 0x23272f, { parent: root, pos: [x, 0.4, ZR + 0.15], key: 'nvconn', m: 0.6 });
    c.userData.title = `NVLink 连接器 ${k}（GPU${k} 的 18 条链路）`;
    box(0.75, 0.08, 0.03, C.gold, { parent: root, pos: [x, 0.45, ZR - 0.0], key: 'nvconn', m: 1, r: 0.3 });
  });
  box(1.2, 0.5, 0.7, 0x4a3b22, { parent: root, pos: [0, 0.35, ZR + 0.45], key: 'pdb', m: 0.6 });
  box(0.4, 0.4, 0.3, C.bus, { parent: root, pos: [0, 0.4, ZR - 0.1], key: 'pdb', m: 0.9, r: 0.3 });
  const uqdC = [-4.1, 0.45, ZR - 0.1], uqdH = [4.1, 0.45, ZR - 0.1];
  cyl(0.18, 0.5, C.cold, { parent: root, pos: uqdC, rot: [Math.PI / 2, 0, 0], key: 'uqd', m: 0.7, r: 0.3 });
  cyl(0.18, 0.5, C.hot, { parent: root, pos: uqdH, rot: [Math.PI / 2, 0, 0], key: 'uqd', m: 0.7, r: 0.3 });

  // NVLink 扁平铜缆束：GPU → 后部连接器（走在冷却管下方）
  const nvCurves = gpus.map((g, k) => curve([[g.x, 0.24, g.z - 0.85], [g.x, 0.24, -4], [connX[k], 0.26, -6.5], [connX[k], 0.35, ZR + 0.3]]));
  nvCurves.forEach(cv => tube(cv, 0.09, C.link, { parent: root, key: 'gpulinks', m: 0.3, r: 0.6, seg: 40 }));

  // NVLink-C2C：Grace ↔ 两颗 B200
  const c2cCurves = [];
  for (let bi = 0; bi < 2; bi++) for (let gi = 0; gi < 2; gi++) {
    const gr = graces[bi], gp = gpus[bi * 2 + gi];
    const cv = curve([[gr.x + (gi ? 0.3 : -0.3), 0.3, gr.z - 0.6], [gp.x, 0.3, (gr.z + gp.z) / 2], [gp.x, 0.3, gp.z + 0.9]]);
    tube(cv, 0.05, C.c2c, { parent: root, key: 'c2c', e: C.c2c, ei: 0.35, seg: 24 });
    c2cCurves.push(cv);
  }
  // PCIe：GPU k → ConnectX k；Grace i → BlueField-3 i
  const pcie = [];
  gpus.forEach((g, k) => pcie.push(curve([[g.x, 0.3, g.z + 0.9], [g.x, 0.3, 4.4], [nicX[k], 0.3, 5.2], [nicX[k], 0.33, ZF - 1.9]])));
  graces.forEach((gr, i) => pcie.push(curve([[gr.x, 0.3, gr.z + 0.6], [gr.x, 0.3, 4.6], [bfX[i], 0.3, 5.0], [bfX[i], 0.33, ZF - 2.1]])));
  pcie.forEach((cv, i) => tube(cv, 0.03, i < 4 ? C.ib : C.eth, { parent: root, key: i < 4 ? 'cx' : 'bf3', seg: 32, rs: 6 }));

  // 托盘内液冷：冷集管 → 每块冷板入口 → 出口 → 热集管（并联）
  const yH = 0.75, zCold = -5, zHot = -5.6;
  tube(curve([[uqdC[0], uqdC[1], ZR + 0.2], [uqdC[0], yH, -6.5], [uqdC[0], yH, zCold], [3.4, yH, zCold]]), 0.08, C.cold, { parent: root, key: 'coldplate', seg: 32 });
  tube(curve([[-3.4, yH, zHot], [uqdH[0], yH, zHot], [uqdH[0], yH, -6.9], [uqdH[0], uqdH[1], ZR + 0.2]]), 0.08, C.hot, { parent: root, key: 'coldplate', seg: 32 });
  const coCurves = [];
  for (const p of plates) {
    const { w, d } = p.userData, x = p.position.x, z = p.position.z;
    const inP = [x - w * 0.3, 0.6, z - d * 0.35], outP = [x + w * 0.3, 0.6, z - d * 0.35];
    const cin = curve([[inP[0], yH, zCold], [inP[0], yH, (zCold + inP[2]) / 2], inP]);
    const cout = curve([outP, [outP[0], yH, (zHot + outP[2]) / 2], [outP[0], yH, zHot]]);
    p.userData.hoses = [tube(cin, 0.04, C.cold, { parent: root, key: 'coldplate', seg: 20, rs: 6 }),
      tube(cout, 0.04, C.hot, { parent: root, key: 'coldplate', seg: 20, rs: 6 })];
    coCurves.push(curve([[uqdC[0], uqdC[1], ZR + 0.2], [uqdC[0], yH, -6.5], [inP[0], yH, zCold], inP,
      [x, 0.55, z], outP, [outP[0], yH, zHot], [uqdH[0], yH, -6.9], [uqdH[0], uqdH[1], ZR + 0.2]]));
  }
  // 供电：配电板 → 两块主板的 VRM
  const pwCurves = [-2.25, 2.25].map(bx => curve([[0, 0.3, ZR + 0.8], [bx * 0.5, 0.22, -5.5], [bx, 0.22, -3.2], [bx, 0.22, -2.6]]));
  pwCurves.forEach(cv => tube(cv, 0.06, C.bus, { parent: root, key: 'pdb', m: 0.8, r: 0.35, seg: 24 }));
  for (const bx of [-2.25, 2.25]) for (let i = 0; i < 6; i++)
    box(0.22, 0.12, 0.22, 0x30343c, { parent: root, pos: [bx - 0.7 + i * 0.28, 0.22, -2.5], key: 'pdb' });   // VRM 相

  const fxNv = addFlow(root, nvCurves, { count: 8, color: 0x9dff3a, size: 0.07, speed: 0.35 });
  const fxIo = addFlow(root, [...c2cCurves, ...pcie], { count: 5, color: 0x9fffff, size: 0.05, speed: 0.4 });
  const fxCo = addFlow(root, coCurves, { count: 10, color: 0x9fe3ff, size: 0.07, speed: 0.08, visible: false });
  const fxPw = addFlow(root, pwCurves, { count: 6, color: 0xffd23a, size: 0.07, speed: 0.3, visible: false });
  L('前面板 · 4×ConnectX OSFP', [-2.85, 1.3, ZF], 'b');
  L('E1.S NVMe ×6', [-0.15, 1.3, ZF], 's');
  L('BlueField-3 ×2', [2.05, 1.3, ZF], 'o s');
  L('GB200 #0', [-2.25, 1.4, 0.3], 'g');
  L('GB200 #1', [2.25, 1.4, 0.3], 'g');
  gpus.forEach((g, k) => L(`GPU${k}`, [g.x, 0.95, g.z], 's'));
  graces.forEach((g, i) => L(`Grace ${i}`, [g.x, 0.95, g.z], 'b s'));
  connX.forEach((x, k) => L(`连接器 ${k}`, [x, 1.0, ZR], 'g s'));
  L('每颗 GPU 18 条 NVLink → 连接器 → 电缆盒 → 9 交换托盘 × 2 芯片', [0, 1.9, -6.2], 'g');
  L('母排取电 → 配电板', [0, 1.1, ZR - 0.2], 'o s');
  L('UQD 供水', [uqdC[0], 1.1, ZR], 'b s');
  L('UQD 回水', [uqdH[0], 1.1, ZR], 'o s');
  L('NVLink-C2C 900 GB/s', [-2.25, 0.9, 0.6], 's');

  const st = { open: 0, openT: 0 };
  function update(t, dt) {
    fxNv.update(t); fxIo.update(t); fxCo.update(t); fxPw.update(t);
    st.open += (st.openT - st.open) * Math.min(1, dt * 5);
    for (const p of plates) {
      p.position.y = p.userData.base + st.open * 1.6;
      p.rotation.x = -st.open * 0.25;
      for (const h of p.userData.hoses) h.visible = st.open < 0.05;
    }
    fxCo.obj.visible = fxCo.on && st.open < 0.05;
  }
  const tools = [
    { grp: '拆解' },
    { label: '拆开冷板（看芯片）', toggle: v => { st.openT = v ? 1 : 0; } },
    { label: '盖上顶盖', toggle: v => { lid.visible = v; } },
    { grp: '动画' },
    { label: 'NVLink 数据流', on: true, toggle: v => { fxNv.obj.visible = v; } },
    { label: 'C2C / PCIe 数据流', on: true, toggle: v => { fxIo.obj.visible = v; } },
    { label: '冷却液流向', toggle: v => { fxCo.on = v; } },
    { label: '供电流向', toggle: v => { fxPw.obj.visible = v; } },
  ];
  return {
    root, labels, tools, update,
    camera: { pos: [9, 11, 15], target: [0, 0, 0] },
    legend: [[C.link, 'NVLink 5（GPU → 后部连接器）'], [C.c2c, 'NVLink-C2C（Grace ↔ B200）'], [C.ib, 'PCIe → ConnectX（计算网）'],
      [C.eth, 'PCIe → BlueField-3（存储/前端）'], [C.cold, '供水'], [C.hot, '回水'], [C.bus, '50V 直流']],
  };
}

// ---------------- GB200 超级芯片（一块 Bianca 板） ----------------
function gpuPackage(parent, pos, key = 'b200') {
  const g = group(parent, key, pos);
  box(4, 0.15, 3.6, 0x1f5a3d, { parent: g, pos: [0, 0, 0], key: 'interposer', m: 0.3 });     // 封装基板
  box(3.2, 0.08, 2.9, 0x6b7380, { parent: g, pos: [0, 0.11, 0], key: 'interposer', m: 0.7, r: 0.3 });  // 中介层
  for (const sx of [-1, 1]) box(1.25, 0.16, 1.3, C.die, { parent: g, pos: [sx * 0.67, 0.23, 0], key: 'die', m: 0.6, r: 0.35 });
  box(0.08, 0.17, 1.3, C.c2c, { parent: g, pos: [0, 0.24, 0], key: 'nvhbi', e: C.c2c, ei: 0.6 });
  for (const hx of [-1.15, -0.4, 0.4, 1.15]) for (const hz of [-1, 1])
    box(0.55, 0.2, 0.42, C.hbm, { parent: g, pos: [hx, 0.25, hz * 1.05], key: 'hbm', m: 0.5 });
  return g;
}

export function buildSuperchip() {
  const root = new THREE.Group(), { labels, L } = view(root);
  const W = 11, D = 16;
  box(W, 0.15, D, C.pcb, { parent: root, pos: [0, 0, 0], key: 'board', m: 0.2, r: 0.8 });
  // Grace
  const GZ = 4.2;
  box(3.2, 0.15, 3.2, 0x1d3a7a, { parent: root, pos: [0, 0.15, GZ], key: 'grace', m: 0.3 });
  box(2.2, 0.2, 2.2, C.grace, { parent: root, pos: [0, 0.32, GZ], key: 'grace', m: 0.6, r: 0.35 });
  const memPts = [];
  for (const sx of [-1, 1]) for (let k = 0; k < 4; k++) {
    const p = [sx * 2.5, 0.22, GZ - 1.5 + k * 1];
    box(1.1, 0.14, 0.7, 0x1a1d24, { parent: root, pos: p, key: 'lpddr', m: 0.4 });
    memPts.push(p);
  }
  // 两颗 B200
  const GX = 2.7, GZB = -3.2, gp = [];
  for (const sx of [-1, 1]) { gpuPackage(root, [sx * GX, 0.15, GZB]); gp.push([sx * GX, 0.4, GZB]); }
  // C2C
  const c2c = gp.map(([x, , z]) => curve([[x * 0.3, 0.32, GZ - 1.1], [x * 0.5, 0.3, (GZ + z) / 2], [x, 0.3, z + 1.8]]));
  c2c.forEach(cv => tube(cv, 0.12, C.c2c, { parent: root, key: 'c2c', e: C.c2c, ei: 0.4, seg: 32 }));
  // 每颗 GPU 18 条 NVLink 走到板后沿的连接器
  const nvl = [];
  gp.forEach(([x, , z], k) => {
    for (let i = 0; i < 18; i++) {
      const dx = (i - 8.5) * 0.12;
      nvl.push(curve([[x + dx * 0.8, 0.2, z - 1.8], [x + dx, 0.2, z - 3], [x * 0.85 + dx, 0.2, -7.6]]));
    }
    box(2.6, 0.4, 0.4, 0x23272f, { parent: root, pos: [x * 0.85, 0.3, -7.75], key: 'gpulinks', m: 0.6 });
  });
  const nvGeo = [];
  for (const cv of nvl) { const p = cv.getPoints(12); for (let k = 0; k < p.length - 1; k++) nvGeo.push(...p[k].toArray(), ...p[k + 1].toArray()); }
  const nvG = new THREE.BufferGeometry(); nvG.setAttribute('position', new THREE.Float32BufferAttribute(nvGeo, 3));
  const nvLines = new THREE.LineSegments(nvG, new THREE.LineBasicMaterial({ color: C.link, transparent: true, opacity: 0.85 }));
  nvLines.userData.key = 'gpulinks'; root.add(nvLines);
  // PCIe：GPU → 前沿（去 ConnectX），Grace → 前沿（去 BlueField-3）
  const pcie = [...gp.map(([x, , z]) => curve([[x + x * 0.25, 0.2, z + 1.8], [x * 1.4, 0.2, 2], [x * 1.4, 0.2, 7.8]])),
    curve([[0, 0.2, GZ + 1.6], [0, 0.2, 7.8]])];
  pcie.forEach((cv, i) => tube(cv, 0.05, i < 2 ? C.ib : C.eth, { parent: root, key: i < 2 ? 'cx' : 'bf3', seg: 24, rs: 6 }));
  // 冷板（默认抬起，显示芯片）
  const plates = [[-GX, GZB, 4.2, 3.8], [GX, GZB, 4.2, 3.8], [0, GZ, 3.2, 3.2]].map(([x, z, w, d]) => {
    const p = coldPlate(root, w, d, [x, 0.65, z], 'coldplate');
    p.userData.base = 0.65; return p;
  });
  // GPU 直接按指针读 LPDDR5X：LPDDR → Grace 内存控制器 → C2C → GPU
  const memCurves = memPts.filter((_, i) => i % 2 === 0).map(([x, , z], i) => {
    const g = gp[x < 0 ? 0 : 1];
    return curve([[x, 0.4, z], [x * 0.3, 0.5, GZ], [g[0] * 0.3, 0.4, GZ - 1.1], [g[0] * 0.5, 0.4, (GZ + g[2]) / 2], [g[0], 0.5, g[2] + 1.6]]);
  });
  const fxC2c = addFlow(root, c2c, { count: 10, color: 0x9fffff, size: 0.1, speed: 0.45 });
  const fxMem = addFlow(root, memCurves, { count: 6, color: 0xffffff, size: 0.09, speed: 0.25, visible: false });
  const fxNv = addFlow(root, nvl.filter((_, i) => i % 3 === 0), { count: 3, color: 0x9dff3a, size: 0.06, speed: 0.5 });

  L('Grace CPU · 72 核', [0, 1.2, GZ], 'b');
  L('LPDDR5X ~480 GB', [-2.5, 0.9, GZ + 1.8], 's');
  L('LPDDR5X', [2.5, 0.9, GZ + 1.8], 's');
  gp.forEach(([x, , z], k) => L(`B200 GPU${k} · 192 GB HBM3e`, [x, 1.4, z], 'g'));
  L('NVLink-C2C 900 GB/s（缓存一致）', [0, 1.1, 0.3], 'b');
  L('每 GPU 18 × NVLink5 → 托盘后部连接器', [0, 0.9, -7.2], 'g');
  L('PCIe → ConnectX', [-GX * 1.4, 0.7, 6.8], 's');
  L('PCIe → BlueField-3', [0, 0.7, 7.2], 's');

  const st = { open: 1, openT: 1 };
  return {
    root, labels,
    camera: { pos: [10, 13, 15], target: [0, 0, 0] },
    tools: [
      { grp: '拆解' },
      { label: '装上冷板', toggle: v => { st.openT = v ? 0 : 1; } },
      { grp: '动画' },
      { label: 'C2C 数据流', on: true, toggle: v => { fxC2c.obj.visible = v; } },
      { label: 'GPU 读 CPU 内存', toggle: v => { fxMem.obj.visible = v; } },
      { label: 'NVLink 出口', on: true, toggle: v => { fxNv.obj.visible = v; } },
    ],
    legend: [[C.c2c, 'NVLink-C2C'], [C.link, 'NVLink 5（36 条 / 板）'], [C.ib, 'PCIe → ConnectX'], [C.eth, 'PCIe → BlueField-3'], [0xffffff, 'GPU 访问 LPDDR5X 的路径']],
    update(t, dt) {
      fxC2c.update(t); fxMem.update(t); fxNv.update(t);
      st.open += (st.openT - st.open) * Math.min(1, dt * 5);
      for (const p of plates) p.position.y = p.userData.base + st.open * 2.2;
    },
  };
}

// ---------------- B200 GPU（封装放大） ----------------
export function buildGpu() {
  const root = new THREE.Group(), { labels, L } = view(root);
  box(14, 0.4, 12.4, 0x1f5a3d, { parent: root, pos: [0, 0.2, 0], key: 'interposer', m: 0.3, r: 0.7 });      // 封装基板
  box(11.4, 0.2, 9.8, 0x7a8290, { parent: root, pos: [0, 0.5, 0], key: 'interposer', m: 0.75, r: 0.3 });    // CoWoS-L 中介层
  for (const hx of [-3.6, -1.3, 1.3, 3.6]) for (const hz of [-2.6, 2.6])                              // 局部硅桥 LSI：晶粒 ↔ HBM
    box(0.9, 0.04, 0.5, 0x9aa4b2, { parent: root, pos: [hx, 0.62, hz], key: 'interposer', m: 0.9 });
  // 两颗计算晶粒
  const dies = [];
  for (const [di, sx] of [[0, -1], [1, 1]]) {
    const d = group(root, 'die', [sx * 2.45, 0.85, 0]);
    d.userData.title = `计算晶粒 ${di}`; d.userData.base = d.position.x;
    box(4.5, 0.5, 4.6, C.die, { parent: d, m: 0.6, r: 0.3 });
    for (let r = 0; r < 8; r++) for (let c = 0; c < 6; c++)            // SM 阵列示意
      box(0.55, 0.04, 0.42, 0x4c5566, { parent: d, pos: [-1.6 + c * 0.64, 0.27, -1.85 + r * 0.5], e: 0x2a6a20, ei: 0.25 });
    box(4.1, 0.05, 0.3, 0x6b7a90, { parent: d, pos: [0, 0.28, 0], e: 0x3ab0ff, ei: 0.3 });   // L2 带
    dies.push(d);
  }
  const hbi = box(0.3, 0.52, 4.2, C.c2c, { parent: root, pos: [0, 0.86, 0], key: 'nvhbi', e: C.c2c, ei: 0.7 });
  // 8 个 HBM3e 堆叠：逻辑底座 + 8 层 DRAM + TSV
  const stacks = [];
  for (const hx of [-3.6, -1.3, 1.3, 3.6]) for (const hz of [-3.55, 3.55]) {
    const s = group(root, 'hbm', [hx, 0.6, hz]);
    s.userData.title = 'HBM3e 堆叠（8-Hi · 24 GB）';
    const layers = [box(1.9, 0.12, 1.4, 0x2f3540, { parent: s, pos: [0, 0.06, 0], m: 0.5 })];
    for (let k = 0; k < 8; k++)
      layers.push(box(1.8, 0.07, 1.3, k % 2 ? 0x6a7484 : 0x5a6474, { parent: s, pos: [0, 0.17 + k * 0.08, 0], m: 0.55, r: 0.35 }));
    const tsv = group(s, null);
    for (const tx of [-0.5, -0.17, 0.17, 0.5]) for (const tz of [-0.3, 0.3])
      cyl(0.025, 1, C.gold, { parent: tsv, pos: [tx, 0, tz], m: 1, r: 0.3, seg: 6 });
    tsv.visible = false;
    Object.assign(s.userData, { layers, tsv });
    stacks.push(s);
  }
  // 封装边缘的对外引脚区：18 × NVLink（后沿）、C2C（前沿左）、PCIe（前沿右）
  const nvPads = [];
  for (let i = 0; i < 18; i++) {
    const x = -5.95 + i * 0.7;
    box(0.5, 0.05, 0.5, C.gold, { parent: root, pos: [x, 0.43, -5.8], key: 'gpulinks', m: 1, r: 0.3 });
    nvPads.push([x, 0.5, -5.8]);
  }
  box(3.2, 0.05, 0.5, C.c2c, { parent: root, pos: [-3, 0.43, 5.8], key: 'c2c', e: C.c2c, ei: 0.5 });
  box(1.6, 0.05, 0.5, C.ib, { parent: root, pos: [3.6, 0.43, 5.8], key: 'cx', e: C.ib, ei: 0.4 });

  // 数据流
  const hbmCurves = stacks.map(s => {
    const p = s.position, dx = p.x < 0 ? -2.45 : 2.45;
    return curve([[p.x, 1.25, p.z], [p.x, 1.25, p.z * 0.6], [(p.x + dx) / 2, 1.2, p.z * 0.25]]);
  });
  const hbiCurves = [-1.4, -0.5, 0.5, 1.4].flatMap((z, i) => {
    const a = curve([[-2.2, 1.18, z], [0, 1.25, z], [2.2, 1.18, z]]);
    const b = curve([[2.2, 1.18, z + 0.2], [0, 1.25, z + 0.2], [-2.2, 1.18, z + 0.2]]);
    return [a, b];
  });
  const nvCurves = nvPads.map(([x, y, z], i) => {
    const dx = i < 9 ? -2.45 : 2.45;
    return curve([[dx + (i % 9 - 4) * 0.35, 1.15, -1.8], [x, 0.7, -4.6], [x, y, z]]);
  });
  const c2cCurve = [curve([[-2.45, 1.15, 1.8], [-3, 0.7, 4.6], [-3, 0.5, 5.8]]), curve([[2.45, 1.15, 1.8], [-1.8, 0.7, 4.8], [-2.4, 0.5, 5.8]])];
  const fxHbm = addFlow(root, hbmCurves, { count: 8, color: 0xffffff, size: 0.09, speed: 0.6 });
  const fxHbi = addFlow(root, hbiCurves, { count: 6, color: 0x9fffff, size: 0.08, speed: 0.7 });
  const fxNv = addFlow(root, nvCurves, { count: 3, color: 0x9dff3a, size: 0.07, speed: 0.4 });
  const fxC2c = addFlow(root, c2cCurve, { count: 5, color: 0x9fffff, size: 0.08, speed: 0.35, visible: false });

  L('计算晶粒 0', [-2.45, 1.7, -1], 'g');
  L('计算晶粒 1', [2.45, 1.7, -1], 'g');
  L('NV-HBI 10 TB/s', [0, 1.6, 1.4], 'b');
  L('HBM3e ×8 · 每个 24 GB · ~1 TB/s', [0, 1.9, 3.55], 's');
  L('CoWoS-L 中介层', [5.4, 0.8, 4.4], 's');
  L('封装基板', [6.6, 0.5, -5.6], 's');
  L('18 × NVLink5 端口 → 18 颗 NVSwitch', [0, 0.9, -6.4], 'g');
  L('NVLink-C2C → Grace', [-3, 0.9, 6.3], 'b s');
  L('PCIe → ConnectX', [3.6, 0.9, 6.3], 's');
  const s0 = stacks[1];                                      // 前排左侧堆叠：拆开后的分层标注
  const exLabels = [L('8 层 DRAM（24 Gb/层）', [1.6, 2.6, 0], 's', s0), L('逻辑底座', [1.6, 0.1, 0], 's', s0), L('TSV 硅通孔', [-1.6, 1.4, 0], 'o s', s0)];
  exLabels.forEach(l => { l.visible = false; });

  const st = { ex: 0, exT: 0, sp: 0, spT: 0 };
  return {
    root, labels,
    camera: { pos: [9, 11, 13], target: [0, 0.5, 0] },
    tools: [
      { grp: '拆解' },
      { label: '拆开 HBM 堆叠', toggle: v => { st.exT = v ? 1 : 0; } },
      { label: '分开两颗晶粒', toggle: v => { st.spT = v ? 1 : 0; } },
      { grp: '动画' },
      { label: 'HBM → 晶粒', on: true, toggle: v => { fxHbm.obj.visible = v; } },
      { label: 'NV-HBI 跨晶粒', on: true, toggle: v => { fxHbi.obj.visible = v; } },
      { label: 'NVLink 出口', on: true, toggle: v => { fxNv.obj.visible = v; } },
      { label: 'C2C 出口', toggle: v => { fxC2c.obj.visible = v; } },
    ],
    legend: [[C.hbm, 'HBM3e 堆叠'], [C.c2c, 'NV-HBI / NVLink-C2C'], [C.gold, 'NVLink 端口 / TSV'], [C.ib, 'PCIe'], [0xffffff, 'HBM 读写']],
    update(t, dt) {
      fxHbm.update(t); fxHbi.update(t); fxNv.update(t); fxC2c.update(t);
      const k = Math.min(1, dt * 5);
      st.ex += (st.exT - st.ex) * k; st.sp += (st.spT - st.sp) * k;
      for (const s of stacks) {
        const gap = st.ex * 0.32;
        s.userData.layers.forEach((l, i) => { l.position.y = (i ? 0.17 + (i - 1) * 0.08 : 0.06) + i * gap; });
        s.userData.tsv.visible = st.ex > 0.05;
        s.userData.tsv.position.y = (0.75 + 8 * gap) / 2;
        s.userData.tsv.scale.y = 0.75 + 8 * gap;
      }
      exLabels.forEach(l => { l.visible = st.ex > 0.5; });
      exLabels[0].position.y = 0.6 + 8 * st.ex * 0.32 + 0.4;
      dies.forEach((d, i) => { d.position.x = d.userData.base + (i ? 1 : -1) * st.sp * 1.2; });
      hbi.scale.x = 0.3 + st.sp * 2.4;
    },
  };
}
