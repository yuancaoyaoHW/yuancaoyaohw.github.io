// LEVEL 1：NVLink 全互联拓扑 / 集群组织
import * as THREE from 'three';
import { C, mat, box, cyl, group, label, curve, line, flow, ease, clamp01 } from './kit.js';

function view(root) {
  const labels = [];
  return { labels, L: (t, p, cls, parent = root) => { const l = label(t, p, parent, cls); labels.push(l); return l; } };
}
const addFlow = (root, curves, o) => { const f = flow(curves, o); f.obj.visible = o.visible ?? false; root.add(f.obj); return f; };
const segGeo = pairs => {                     // pairs: [[Vector3, Vector3], ...] → LineSegments 几何
  const a = [];
  for (const [p, q] of pairs) a.push(p.x, p.y, p.z, q.x, q.y, q.z);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(a, 3));
  return g;
};

// ---------------- NVLink 全互联：72 GPU × 18 NVSwitch ----------------
// GPU 排成下方大圆环（18 组 × 4，组间留缝），NVSwitch 排成上方小圆环（9 对，对间留缝）
export function buildFabric() {
  const root = new THREE.Group();
  const { labels, L } = view(root);
  const RG = 10, RS = 4.2, YS = 7.5;
  const gpuAng = g => { const t = Math.floor(g / 4); return ((g + t * 0.9) / (72 + 18 * 0.9)) * Math.PI * 2; };
  const swAng = s => { const p = Math.floor(s / 2); return ((s + p * 0.8 + 0.4) / (18 + 9 * 0.8)) * Math.PI * 2; };
  const gpuPos = [], swPos = [], gpuMesh = [], swMesh = [], swMats = [];
  const sph = new THREE.SphereGeometry(1, 16, 12);

  for (let g = 0; g < 72; g++) {
    const a = gpuAng(g), p = new THREE.Vector3(Math.cos(a) * RG, 0, Math.sin(a) * RG);
    gpuPos.push(p);
    const m = new THREE.Mesh(sph, mat(C.nvidia, { e: C.nvidia, ei: 0.35, m: 0.2, r: 0.4 }));
    m.scale.setScalar(0.17); m.position.copy(p);
    const tr = Math.floor(g / 4) + 1;
    m.userData = { key: 'gpunode', gpu: g, title: `GPU ${g} · 计算托盘 #${tr} 的 GPU${g % 4}` };
    root.add(m); gpuMesh.push(m);
  }
  for (let t = 0; t < 18; t++) {                       // 计算托盘底座（每 4 颗 GPU 一个）
    const a = (gpuAng(t * 4) + gpuAng(t * 4 + 3)) / 2;
    const b = box(1.55, 0.12, 0.5, C.tray, { parent: root, pos: [Math.cos(a) * RG, -0.25, Math.sin(a) * RG], key: 'ct', m: 0.5 });
    b.rotation.y = -a + Math.PI / 2;
    b.userData.title = `计算托盘 #${t + 1}（GPU ${t * 4}–${t * 4 + 3}）`; b.userData.tray = t;
    L(`CT${t + 1}`, [Math.cos(a) * (RG + 1), -0.4, Math.sin(a) * (RG + 1)], 's');
  }
  for (let s = 0; s < 18; s++) {
    const a = swAng(s), p = new THREE.Vector3(Math.cos(a) * RS, YS, Math.sin(a) * RS);
    swPos.push(p);
    const sm = mat(C.eth, { e: C.eth, ei: 0.3, m: 0.3, r: 0.4, unique: true });
    const m = box(0.42, 0.22, 0.42, 0, { material: sm, parent: root, pos: [p.x, p.y, p.z] });
    m.rotation.y = -a;
    const tr = Math.floor(s / 2) + 1;
    m.userData = { key: 'swnode', sw: s, title: `NVSwitch ${s} · 交换托盘 #${tr} 的芯片 ${s % 2}` };
    swMesh.push(m); swMats.push(sm);
  }
  for (let p = 0; p < 9; p++) {                         // 交换托盘底座（每 2 颗芯片一个）
    const a = (swAng(p * 2) + swAng(p * 2 + 1)) / 2;
    const b = box(1.25, 0.08, 0.6, C.sw, { parent: root, pos: [Math.cos(a) * RS, YS - 0.2, Math.sin(a) * RS], key: 'sw', m: 0.5 });
    b.rotation.y = -a + Math.PI / 2;
    b.userData.title = `NVSwitch 托盘 #${p + 1}（芯片 ${p * 2}、${p * 2 + 1}）`; b.userData.swt = p;
    L(`SW${p + 1}`, [Math.cos(a) * (RS + 0.9), YS + 0.4, Math.sin(a) * (RS + 0.9)], 's b');
  }

  // 1296 条链路：GPU g 端口 s ↔ NVSwitch s 端口 g
  const links = [];
  for (let g = 0; g < 72; g++) for (let s = 0; s < 18; s++) links.push([gpuPos[g], swPos[s], g, s]);
  const baseMat = new THREE.LineBasicMaterial({ color: C.nvidia, transparent: true, opacity: 0.09, depthWrite: false });
  const hiMat = new THREE.LineBasicMaterial({ color: 0xd4ff6a, transparent: true, opacity: 0.95 });
  const p2pMat = new THREE.LineBasicMaterial({ color: 0xffd23a, transparent: true, opacity: 0.9 });
  const base = new THREE.LineSegments(segGeo(links), baseMat);
  const hi = new THREE.LineSegments(new THREE.BufferGeometry(), hiMat);
  for (const l of [base, hi]) { l.userData.noPick = true; root.add(l); }

  // 中心轴示意
  cyl(0.03, YS, 0x2a3446, { parent: root, pos: [0, YS / 2, 0], op: 0.5 }).userData.noPick = true;

  L('72 × B200 GPU（18 个计算托盘 × 4）', [0, -1.4, RG + 1.2], 'g big');
  L('18 × NVSwitch 芯片（9 个交换托盘 × 2）', [0, YS + 1.6, 0], 'b big');
  const lInfo = L('', [0, YS / 2, 0], 'o');
  lInfo.visible = false;
  const setInfo = t => { lInfo.element.textContent = t; lInfo.visible = !!t; };

  function highlight(pred) {
    hi.geometry.dispose();
    hi.geometry = pred ? segGeo(links.filter(pred)) : new THREE.BufferGeometry();
    baseMat.opacity = pred ? 0.035 : 0.09;
  }
  function paintNodes(gSet, sSet) {
    gpuMesh.forEach((m, i) => m.scale.setScalar(gSet && gSet.has(i) ? 0.28 : 0.17));
    swMesh.forEach((m, i) => { const k = sSet && sSet.has(i) ? 1.4 : 1; m.scale.set(0.42 * k, 0.22 * k, 0.42 * k); });
  }
  const ALLG = new Set(gpuPos.map((_, i) => i)), ALLS = new Set(swPos.map((_, i) => i));

  // 点对点：GPU A → 18 颗 NVSwitch 并行 → GPU B（每条路径都只经 1 跳）
  const GA = 2, GB = 45;
  const p2pCurves = swPos.map(s => curve([gpuPos[GA].toArray(), s.toArray(), gpuPos[GB].toArray()]));
  const fxP2P = addFlow(root, p2pCurves, { count: 4, color: 0xffd23a, size: 0.09, speed: 0.35 });
  const p2pLines = new THREE.LineSegments(segGeo(links.filter(l => l[2] === GA || l[2] === GB)), p2pMat);
  p2pLines.visible = false; p2pLines.userData.noPick = true; root.add(p2pLines);

  // SHARP all-reduce：72 GPU 各发 1 份 → 交换芯片内求和 → 广播回 72 GPU
  const dotMat = new THREE.MeshBasicMaterial({ color: 0x9dff3a });
  const dots = new THREE.InstancedMesh(sph, dotMat, links.length);
  dots.visible = false; dots.userData.noPick = true; root.add(dots);
  const dm = new THREE.Object3D();
  const st = { sh: -1, p2p: false };

  function update(t, dt) {
    fxP2P.update(t);
    let glow = 0.3;
    if (st.sh >= 0) {
      st.sh += dt;
      const s = st.sh;
      let u = null;
      if (s < 1.6) { u = ease(s / 1.6); dotMat.color.setHex(0x9dff3a); setInfo('① 72 颗 GPU 同时把各自数据分片发往 18 颗 NVSwitch'); }
      else if (s < 2.6) { glow = 1.4 + Math.sin(s * 14) * 0.5; setInfo('② NVSwitch 芯片内完成求和（SHARP 网内归约）'); }
      else if (s < 4.2) { u = 1 - ease((s - 2.6) / 1.6); dotMat.color.setHex(0x7fe8ff); setInfo('③ 归约结果从交换芯片广播回每颗 GPU'); }
      else if (s < 6.5) setInfo('完成：每颗 GPU 只发送 1 次、接收 1 次；环形算法需 2×(72−1)=142 步');
      else { st.sh = -1; setInfo(''); }
      dots.visible = u != null;
      if (u != null) {
        links.forEach(([g, w], i) => {
          dm.position.lerpVectors(g, w, u);
          dm.scale.setScalar(0.05); dm.updateMatrix(); dots.setMatrixAt(i, dm.matrix);
        });
        dots.instanceMatrix.needsUpdate = true;
      }
    }
    swMats.forEach(m => { m.emissiveIntensity = glow; });
  }

  function select(o) {
    while (o && o.userData.gpu == null && o.userData.sw == null && o.userData.tray == null && o.userData.swt == null) o = o.parent;
    const d = o ? o.userData : {};
    if (d.gpu != null) {
      highlight(l => l[2] === d.gpu); paintNodes(new Set([d.gpu]), ALLS);
      setInfo(`GPU ${d.gpu}：18 条 NVLink → 18 颗 NVSwitch 各 1 条`);
    } else if (d.sw != null) {
      highlight(l => l[3] === d.sw); paintNodes(ALLG, new Set([d.sw]));
      setInfo(`NVSwitch ${d.sw}：72 个端口 → 72 颗 GPU 各 1 条`);
    } else if (d.tray != null) {
      const gs = new Set([0, 1, 2, 3].map(k => d.tray * 4 + k));
      highlight(l => gs.has(l[2])); paintNodes(gs, ALLS);
      setInfo(`计算托盘 #${d.tray + 1}：4 GPU × 18 = 72 条链路`);
    } else if (d.swt != null) {
      const ss = new Set([d.swt * 2, d.swt * 2 + 1]);
      highlight(l => ss.has(l[3])); paintNodes(ALLG, ss);
      setInfo(`交换托盘 #${d.swt + 1}：2 芯片 × 72 = 144 条链路`);
    } else { highlight(null); paintNodes(null, null); setInfo(''); }
  }

  const tools = [
    { grp: '动画' },
    { label: '▶ SHARP all-reduce', run: () => { select(null); st.sh = 0; } },
    { label: '点对点传输 GPU2 → GPU45', toggle: v => {
      fxP2P.obj.visible = p2pLines.visible = v;
      paintNodes(v ? new Set([GA, GB]) : null, v ? ALLS : null);
      setInfo(v ? '18 条单跳路径并行：GPU2 → 每颗 NVSwitch → GPU45，合计 1.8 TB/s' : '');
    } },
    { grp: '显示' },
    { label: '清除高亮', run: () => select(null) },
  ];
  return {
    root, labels, tools, update, onSelect: select,
    camera: { pos: [0, 15, 22], target: [0, 3, 0] },
    legend: [[C.nvidia, 'GPU / NVLink 5 链路（1296 条）'], [C.eth, 'NVSwitch 芯片'], [0xd4ff6a, '选中节点的链路'], [0xffd23a, '点对点传输路径']],
  };
}

// ---------------- 集群组织：1 个 SU = 8 × NVL72 ----------------
// 计算网按轨道优化：每个机架第 r 号网卡（每托盘 GPU r 的 ConnectX）→ 第 r 条轨道的 2 台 leaf
// 8 机架 × 18 托盘 = 144 张 / 轨道 → 2 台 144 口 leaf（各 72 下行 + 72 上行）→ 8 × 72 = 576 上行 = 4 台 spine × 144 口
export function buildCluster() {
  const root = new THREE.Group();
  const { labels, L } = view(root);
  const RX = i => -10.5 + i * 3;
  const RAIL = [0xb36bff, 0x8f4de0, 0xd49bff, 0x6d3bb8];
  const mkLines = (pairs, color, op) => {
    const l = new THREE.LineSegments(segGeo(pairs), new THREE.LineBasicMaterial({ color, transparent: true, opacity: op, depthWrite: false }));
    l.userData.noPick = true; root.add(l); return l;
  };
  const V = (x, y, z) => new THREE.Vector3(x, y, z);

  // 8 个机架
  const rackTop = [];
  for (let i = 0; i < 8; i++) {
    const g = group(root, 'sunit', [RX(i), 0, 0]);
    g.userData.title = `NVL72 机架 #${i + 1}（72 GPU）`;
    box(1.6, 6, 2.4, C.frame, { parent: g, pos: [0, 3, 0], m: 0.6 });
    box(1.4, 1.5, 0.04, C.tray, { parent: g, pos: [0, 4.4, 1.22] });
    box(1.4, 0.9, 0.04, C.sw, { parent: g, pos: [0, 3.2, 1.22] });
    box(1.4, 1.3, 0.04, C.tray, { parent: g, pos: [0, 2.05, 1.22] });
    box(0.08, 4, 0.05, C.nvidia, { parent: g, pos: [0.62, 3.2, 1.25], e: C.nvidia, ei: 0.7 });  // NVLink 域标识
    for (const y of [5.6, 0.7]) box(1.4, 0.4, 0.04, C.psu, { parent: g, pos: [0, y, 1.22] });
    L(`#${i + 1}`, [RX(i), 6.5, 1.2], 's');
    rackTop.push(r => V(RX(i) - 0.45 + r * 0.3, 6, -0.4));
  }

  // 计算网：4 轨道 × 2 leaf，4 台 spine
  const leafPos = [], spinePos = [];
  for (let r = 0; r < 4; r++) for (let k = 0; k < 2; k++) {
    const p = V(-10.5 + (r * 2 + k) * 3, 11, -1.5);
    leafPos.push({ p, r });
    const b = box(2.2, 0.4, 1.2, 0, { material: mat(RAIL[r], { e: RAIL[r], ei: 0.25 }), parent: root, pos: p.toArray(), key: 'rail' });
    b.userData.title = `轨道 ${r} · Leaf ${k + 1}（接 8 个机架的第 ${r} 号网卡）`;
  }
  for (let i = 0; i < 4; i++) {
    const p = V(-6.75 + i * 4.5, 16.5, -2.5);
    spinePos.push(p);
    const b = box(2.8, 0.5, 1.4, C.ib, { parent: root, pos: p.toArray(), key: 'spinesw', e: C.ib, ei: 0.15 });
    b.userData.title = `Spine 交换机 #${i + 1}`;
  }
  const railLines = RAIL.map((col, r) => {
    const pairs = [];
    for (let i = 0; i < 8; i++) for (const lf of leafPos) if (lf.r === r) pairs.push([rackTop[i](r), V(lf.p.x, lf.p.y - 0.2, lf.p.z)]);
    return mkLines(pairs, col, 0.45);
  });
  const upPairs = [];
  for (const lf of leafPos) for (const s of spinePos) upPairs.push([V(lf.p.x, lf.p.y + 0.2, lf.p.z), V(s.x, s.y - 0.25, s.z)]);
  const upLines = mkLines(upPairs, C.ib, 0.3);

  // 存储 / 前端网（BlueField-3）
  const fe = V(15.5, 8, 2.5);
  box(2, 0.4, 1.2, C.eth, { parent: root, pos: fe.toArray(), key: 'storage', e: C.eth, ei: 0.2 }).userData.title = '存储 / 前端交换机';
  const stor = group(root, 'storage', [19, 0, 2.5]);
  for (let j = 0; j < 5; j++) box(1.8, 0.9, 1.6, 0x2b3a4d, { parent: stor, pos: [0, 0.6 + j * 1.05, 0], m: 0.5 });
  for (let j = 0; j < 5; j++) box(1.5, 0.08, 0.02, C.eth, { parent: stor, pos: [0, 0.75 + j * 1.05, 0.82], e: C.eth, ei: 0.8 });
  const stPairs = [];
  for (let i = 0; i < 8; i++) stPairs.push([V(RX(i) + 0.5, 4.8, 1.2), V(fe.x - 1, fe.y, fe.z)]);
  stPairs.push([V(fe.x + 1, fe.y, fe.z), V(19, 5.4, 2.5)]);
  mkLines(stPairs, C.eth, 0.35);
  const user = box(1.4, 0.8, 1.4, 0x34404f, { parent: root, pos: [15.5, 12, 2.5], key: 'storage' });
  user.userData.title = '用户入口 / 推理前端';
  mkLines([[V(15.5, 8.2, 2.5), V(15.5, 11.6, 2.5)]], C.eth, 0.6);

  // 带外管理网
  const oob = V(-15.5, 4, 2.5);
  box(2, 0.35, 1.2, C.mgmt, { parent: root, pos: oob.toArray(), key: 'oob', e: C.mgmt, ei: 0.15 }).userData.title = '带外管理核心交换机';
  box(1.6, 2.2, 1.6, 0x2c3440, { parent: root, pos: [-15.5, 1.1, 2.5], key: 'oob' }).userData.title = '管理节点（BCM / SLURM 控制器 / 监控）';
  const oPairs = [];
  for (let i = 0; i < 8; i++) oPairs.push([V(RX(i) - 0.5, 5.9, 1.2), V(oob.x + 1, oob.y, oob.z)]);
  mkLines(oPairs, C.mgmt, 0.35);
  // NVL576：8 机架组成一个 NVLink 域（第二级 NVLink 交换，示意）
  const n576 = group(root, 'nvl576');
  box(25.5, 7.4, 4, C.nvidia, { parent: n576, pos: [0, 3.4, 0], op: 0.08, e: C.nvidia, ei: 0.3 });
  box(25.5, 0.5, 1, C.nvidia, { parent: n576, pos: [0, 7.6, 0], e: C.nvidia, ei: 0.5, op: 0.6 });
  n576.visible = false;
  const l576 = L('NVL576：8 机架 × 72 = 576 GPU 一个 NVLink 域（示意）', [0, 8.4, 0], 'g');
  l576.visible = false;

  // AI 工厂：更多 SU 通过 core 层互联（虚影）
  const ghosts = group(root, null);
  for (const [gx, gz] of [[-30, -16], [0, -22], [30, -16]]) {
    for (let i = 0; i < 8; i++) box(1.6, 6, 2.4, 0x3a4658, { parent: ghosts, pos: [gx - 10.5 + i * 3, 3, gz], op: 0.25 });
    box(20, 0.4, 1.2, C.ib, { parent: ghosts, pos: [gx, 12, gz], op: 0.35 });
  }
  const cores = [-9, 0, 9].map(x => V(x, 21, -12));
  for (const c of cores) box(3.2, 0.6, 1.6, 0xd49bff, { parent: ghosts, pos: c.toArray(), op: 0.7 });
  const gPairs = [];
  for (const c of cores) {
    for (const s of spinePos) gPairs.push([V(c.x, c.y - 0.3, c.z), V(s.x, s.y + 0.25, s.z)]);
    for (const gx of [-30, 0, 30]) gPairs.push([V(c.x, c.y - 0.3, c.z), V(gx, 12.2, gx ? -16 : -22)]);
  }
  const gl = mkLines(gPairs, 0xd49bff, 0.3);
  gl.removeFromParent(); ghosts.add(gl);
  ghosts.visible = false;
  const lFac = L('AI 工厂：多个 SU 经 core 层组网 → 数万 GPU', [0, 22.5, -12], 'b');
  lFac.visible = false;

  // ---- 流动 ----
  // 计算网：机架 1 轨道 0 → leaf → spine → leaf → 机架 6（跨 leaf 时经 spine）；同轨道同 leaf 只需 1 跳
  const cc = [];
  for (let r = 0; r < 4; r++) {
    const l0 = leafPos[r * 2], l1 = leafPos[r * 2 + 1], s = spinePos[r];
    cc.push(curve([rackTop[r](r).toArray(), l0.p.toArray(), s.toArray(), l1.p.toArray(), rackTop[7 - r](r).toArray()]));
    cc.push(curve([rackTop[r + 2](r).toArray(), l0.p.toArray(), rackTop[r + 4](r).toArray()]));
  }
  const fxC = addFlow(root, cc, { count: 5, color: 0xe3c4ff, size: 0.12, speed: 0.18 });
  const sc = [];
  for (let i = 0; i < 8; i += 2) sc.push(curve([[RX(i) + 0.5, 4.8, 1.2], fe.toArray(), [19, 5.4, 2.5]]));
  sc.push(curve([[15.5, 11.6, 2.5], fe.toArray(), [RX(7) + 0.5, 4.8, 1.2]]));
  const fxS = addFlow(root, sc, { count: 4, color: 0x7fd0ff, size: 0.12, speed: 0.2 });
  const mc = [];
  for (let i = 0; i < 8; i += 2) mc.push(curve([oob.toArray(), [RX(i) - 0.5, 5.9, 1.2]]));
  const fxM = addFlow(root, mc, { count: 3, color: 0xffe27a, size: 0.1, speed: 0.25 });

  // ---- 标签 ----
  L('扩展单元 SU：8 × NVL72 = 576 GPU', [0, -0.8, 2.5], 'g big');
  L('计算网 Leaf（4 轨道 × 2）', [-14.5, 11, -1.5], 'b');
  L('计算网 Spine', [-11, 16.5, -2.5], 'b');
  L('存储 / 前端网', [15.5, 8.8, 2.5], 'b');
  L('并行存储', [19, 6.3, 2.5], 's');
  L('用户入口', [15.5, 12.8, 2.5], 's');
  L('带外管理网', [-15.5, 4.7, 2.5], 'o');
  for (let r = 0; r < 4; r++) L(`轨道 ${r}`, [-9 + r * 6, 11.7, -1.5], 's');

  function select(o) {
    while (o && o.userData.key !== 'rail' && o.parent) o = o.parent;
    const hit = o && o.userData.key === 'rail' ? leafPos.find(lf => lf.p.distanceTo(o.position) < 0.01) : null;
    railLines.forEach((l, r) => { l.material.opacity = !hit ? 0.45 : r === hit.r ? 0.95 : 0.06; });
    upLines.material.opacity = hit ? 0.1 : 0.3;
  }

  const tools = [
    { grp: '流量' },
    { label: '计算网（东西向）', toggle: v => { fxC.obj.visible = v; } },
    { label: '存储 / 前端（南北向）', toggle: v => { fxS.obj.visible = v; } },
    { label: '带外管理', toggle: v => { fxM.obj.visible = v; } },
    { grp: '规模' },
    { label: 'NVL576 多机架 NVLink 域', toggle: v => { n576.visible = l576.visible = v; } },
    { label: 'AI 工厂（多个 SU）', toggle: v => { ghosts.visible = lFac.visible = v; } },
  ];
  return {
    root, labels, tools, onSelect: select,
    update(t) { fxC.update(t); fxS.update(t); fxM.update(t); },
    camera: { pos: [20, 17, 30], target: [0, 7, -2] },
    legend: [[C.nvidia, '机架内 NVLink 域（72 GPU）'], [RAIL[0], '计算网轨道 0–3（深浅区分，点击 leaf 高亮）'], [C.ib, 'Leaf ↔ Spine'],
      [C.eth, '存储 / 前端网'], [C.mgmt, '带外管理网']],
  };
}
