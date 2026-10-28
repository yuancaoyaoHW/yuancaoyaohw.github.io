// 公共几何/材质/标签/流动动画工具
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

export const C = {
  frame: 0x1c222c, tray: 0x2a313d, trayFace: 0x15191f, sw: 0x1e2a3a, swFace: 0x101820,
  psu: 0x343a44, nvidia: 0x76b900, copper: 0xc87533, cold: 0x2f9bff, hot: 0xff4a3a,
  bus: 0xd8a24a, gold: 0xd4af37, pcb: 0x0e3b2a, die: 0x3a3f4a, hbm: 0x56606e,
  port: 0x0a0c10, cage: 0x8c96a3, led: 0x6cff6c, grace: 0x2b5bd7, nic: 0x5a2d82,
  bf3: 0x8a3a2a, link: 0x76b900, c2c: 0x00e5ff, ib: 0xb36bff, eth: 0x3ab0ff, mgmt: 0xffd23a,
};

const cache = new Map();
export function mat(color, o = {}) {
  const k = [color, o.m, o.r, o.e, o.op, o.ei].join('|');
  if (!o.unique && cache.has(k)) return cache.get(k);
  const m = new THREE.MeshStandardMaterial({
    color, metalness: o.m ?? 0.45, roughness: o.r ?? 0.55,
    emissive: o.e ?? 0x000000, emissiveIntensity: o.ei ?? 1,
    transparent: o.op != null && o.op < 1, opacity: o.op ?? 1,
    depthWrite: !(o.op != null && o.op < 1),
  });
  if (!o.unique) cache.set(k, m);
  return m;
}

const boxGeo = new THREE.BoxGeometry(1, 1, 1);
export function box(w, h, d, color, o = {}) {
  const m = new THREE.Mesh(boxGeo, o.material || mat(color, o));
  m.scale.set(w, h, d);
  if (o.pos) m.position.set(...o.pos);
  if (o.key) m.userData.key = o.key;
  if (o.parent) o.parent.add(m);
  return m;
}
export function cyl(r, h, color, o = {}) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, o.r2 ?? r, h, o.seg ?? 20), o.material || mat(color, o));
  if (o.pos) m.position.set(...o.pos);
  if (o.rot) m.rotation.set(...o.rot);
  if (o.key) m.userData.key = o.key;
  if (o.parent) o.parent.add(m);
  return m;
}
export function group(parent, key, pos) {
  const g = new THREE.Group();
  if (key) g.userData.key = key;
  if (pos) g.position.set(...pos);
  if (parent) parent.add(g);
  return g;
}

// HTML 标签（CSS2D），cls: g/b/o/s/big
export function label(text, pos, parent, cls = '') {
  const el = document.createElement('div');
  el.className = 'lbl ' + cls;
  el.textContent = text;
  const o = new CSS2DObject(el);
  o.position.set(...pos);
  o.userData.isLabel = true;
  parent.add(o);
  return o;
}

// 管线/电缆：points 为 [x,y,z] 数组
export function curve(points, closed = false) {
  return new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p)), closed, 'catmullrom', 0.2);
}
export function tube(cv, r, color, o = {}) {
  const m = new THREE.Mesh(new THREE.TubeGeometry(cv, o.seg ?? 64, r, o.rs ?? 8, cv.closed), o.material || mat(color, o));
  if (o.key) m.userData.key = o.key;
  if (o.parent) o.parent.add(m);
  m.userData.curve = cv;
  return m;
}
export function line(cv, color, parent, n = 32, opacity = 0.6) {
  const g = new THREE.BufferGeometry().setFromPoints(cv.getPoints(n));
  const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  if (parent) parent.add(l);
  return l;
}

// 沿曲线移动的发光粒子：flow(curves, {count, color, size, speed})
const dotGeo = new THREE.SphereGeometry(1, 8, 6);
export function flow(curves, o = {}) {
  const per = o.count ?? 6, total = curves.length * per;
  const im = new THREE.InstancedMesh(dotGeo, new THREE.MeshBasicMaterial({ color: o.color ?? 0xffffff }), total);
  im.userData.noPick = true;
  const dm = new THREE.Object3D(), s = o.size ?? 0.05, sp = o.speed ?? 0.25;
  const phase = Array.from({ length: total }, () => Math.random());
  const update = (t) => {
    let i = 0;
    for (const cv of curves) for (let k = 0; k < per; k++, i++) {
      const u = ((k / per) + t * sp + phase[i] * (o.jitter ?? 0)) % 1;
      cv.getPointAt(u, dm.position);
      dm.scale.setScalar(s);
      dm.updateMatrix();
      im.setMatrixAt(i, dm.matrix);
    }
    im.instanceMatrix.needsUpdate = true;
  };
  update(0);
  return { obj: im, update };
}

// 端口阵列：在 parent 上按网格放置小方块
export function ports(parent, n, cols, origin, step, size, color, o = {}) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const c = i % cols, r = Math.floor(i / cols);
    out.push(box(size[0], size[1], size[2], color, {
      ...o, parent, pos: [origin[0] + c * step[0], origin[1] - r * step[1], origin[2]],
    }));
  }
  return out;
}

// 散热鳍片/冷板表面
export function coldPlate(parent, w, d, pos, key) {
  const g = group(parent, key, pos);
  box(w, 0.06, d, C.copper, { parent: g, m: 0.9, r: 0.3 });
  box(w * 0.92, 0.05, d * 0.92, 0xb06a30, { parent: g, pos: [0, 0.055, 0], m: 0.9, r: 0.35 });
  cyl(0.05, 0.16, C.cold, { parent: g, pos: [-w * 0.3, 0.14, -d * 0.35] });
  cyl(0.05, 0.16, C.hot, { parent: g, pos: [w * 0.3, 0.14, -d * 0.35] });
  return g;
}

// 缓动
export const ease = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
export const clamp01 = t => Math.max(0, Math.min(1, t));
