/* ============================================================
 * DisDP Explainer — Variant A (step-based) animation engine
 * Light theme. Matches mHC design system (CSS vars).
 * ============================================================ */
(function(){
"use strict";
var SVG_NS = "http://www.w3.org/2000/svg";

/* ---- Engine ---- */
function AnimEngine(svgRoot, opts){
  opts = opts || {};
  this.root = svgRoot;
  this.svgId = svgRoot.getAttribute("id") || "svg";
  this.steps = opts.steps || [];
  this.idx = 0; this.playing = false; this.timer = null; this.speed = 1.0;
  this._renderToken = 0; this._currentLayer = null;
  this.frame = opts.frame || null;
  this._initDefs(); this._bindControls();
}
AnimEngine.prototype._uid = function(s){ return this.svgId + "-" + s; };
AnimEngine.prototype._initDefs = function(){
  if(this.root.querySelector("defs")) return;
  var self = this;
  var defs = document.createElementNS(SVG_NS, "defs");
  var mk = function(id, color){
    var m = document.createElementNS(SVG_NS, "marker");
    m.setAttribute("id", id);
    m.setAttribute("markerWidth","9"); m.setAttribute("markerHeight","9");
    m.setAttribute("refX","7"); m.setAttribute("refY","4");
    m.setAttribute("orient","auto"); m.setAttribute("markerUnits","strokeWidth");
    var p = document.createElementNS(SVG_NS, "path");
    p.setAttribute("d","M0,0 L7,4 L0,8 Z");
    p.setAttribute("fill", color);
    m.appendChild(p); defs.appendChild(m);
  };
  mk(this._uid("arr"),  "var(--sv-text-muted)");
  mk(this._uid("arrA"), "var(--sv-blue)");
  mk(this._uid("arrG"), "var(--sv-green)");
  mk(this._uid("arrR"), "var(--sv-red)");
  mk(this._uid("arrO"), "var(--sv-orange)");
  mk(this._uid("arrP"), "var(--sv-purple)");
  mk(this._uid("arrT"), "var(--sv-teal)");
  this.root.appendChild(defs);
};
AnimEngine.prototype.draw = function(tag, attrs, parent){
  var target = parent || this._currentLayer || this.root;
  var el = document.createElementNS(SVG_NS, tag);
  if(attrs){
    for(var k in attrs){
      if(!Object.prototype.hasOwnProperty.call(attrs,k)) continue;
      var v = attrs[k];
      if(v === null || v === undefined) continue;
      if(k === "className"){ el.setAttribute("class", v); continue; }
      if(k === "style" && typeof v === "object"){
        for(var sk in v){ if(Object.prototype.hasOwnProperty.call(v,sk)) el.style[sk] = v[sk]; }
        continue;
      }
      if(typeof v === "boolean"){ if(v) el.setAttribute(k,k); continue; }
      el.setAttribute(k, v);
    }
  }
  target.appendChild(el);
  return el;
};
AnimEngine.prototype._newLayer = function(){
  var olds = Array.prototype.slice.call(this.root.querySelectorAll("g.anim-layer"));
  if(olds.length > 1){
    olds.slice(0, olds.length - 1).forEach(function(l){ if(l.parentNode) l.remove(); });
    olds = Array.prototype.slice.call(this.root.querySelectorAll("g.anim-layer"));
  }
  var layer = document.createElementNS(SVG_NS, "g");
  layer.setAttribute("class","anim-layer");
  layer.style.opacity = "0";
  this.root.appendChild(layer);
  var self = this;
  requestAnimationFrame(function(){
    requestAnimationFrame(function(){ layer.style.opacity = "1"; });
  });
  olds.forEach(function(old){
    old.style.opacity = "0";
    setTimeout(function(){ if(old.parentNode) old.remove(); }, 320);
  });
  this._currentLayer = layer;
  return layer;
};
AnimEngine.prototype.goTo = function(i){
  if(this.steps.length === 0){ this._updateStatus(); return; }
  var clamped = Math.max(0, Math.min(this.steps.length - 1, Math.floor(i)));
  this._renderToken++; this.idx = clamped; this._newLayer();
  try{
    var step = this.steps[clamped];
    if(step && typeof step.render === "function") step.render(this);
  }catch(err){
    console.error("["+this.svgId+"] render error at step "+clamped+":", err);
    var t = this.draw("text",{x:410,y:180,"text-anchor":"middle",class:"anim-err","font-size":"13"});
    t.textContent = "该步骤渲染出错: " + (err && err.message ? err.message : String(err));
  }
  this._updateStatus();
};
AnimEngine.prototype.next = function(){ this.pause(); this.goTo(this.idx + 1); };
AnimEngine.prototype.prev = function(){ this.pause(); this.goTo(this.idx - 1); };
AnimEngine.prototype.reset = function(){ this.pause(); this.goTo(0); };
AnimEngine.prototype.play = function(){
  if(this.playing){ this.pause(); return; }
  if(this.idx >= this.steps.length - 1) this.goTo(0);
  this.playing = true; this._setPlayBtn("⏸ 暂停");
  var self = this;
  var tick = function(){
    if(!self.playing) return;
    if(self.idx >= self.steps.length - 1){ self.pause(); self._setPlayBtn("▶ 播放"); return; }
    self.goTo(self.idx + 1);
    self.timer = setTimeout(tick, 1200 / self.speed);
  };
  tick();
};
AnimEngine.prototype.pause = function(){
  this.playing = false;
  if(this.timer){ clearTimeout(this.timer); this.timer = null; }
  this._setPlayBtn("▶ 播放");
};
AnimEngine.prototype.setSpeed = function(s){ this.speed = s; };
AnimEngine.prototype._setPlayBtn = function(txt){
  if(this.frame){ var b = this.frame.querySelector(".btn-play"); if(b) b.textContent = txt; }
};
AnimEngine.prototype._updateStatus = function(){
  if(!this.frame) return;
  var n = this.steps.length;
  var cur = this.frame.querySelector(".step-cur");
  var tot = this.frame.querySelector(".step-total");
  if(cur) cur.textContent = String(this.idx + 1);
  if(tot) tot.textContent = String(n);
  var prev = this.frame.querySelector(".btn-prev");
  var next = this.frame.querySelector(".btn-next");
  if(prev){ prev.disabled = this.idx <= 0; prev.setAttribute("aria-disabled", this.idx<=0?"true":"false"); }
  if(next){ next.disabled = this.idx >= n-1; next.setAttribute("aria-disabled", this.idx>=n-1?"true":"false"); }
  var stTitle = this.frame.querySelector(".step-title");
  var st = this.steps[this.idx];
  if(stTitle && st && st.title) stTitle.textContent = st.title;
  var fill = this.frame.querySelector(".progress-fill");
  if(fill && n > 0) fill.style.width = ((this.idx + 1) / n * 100) + "%";
  var desc = this.frame.querySelector(".anim-desc");
  if(desc && st && st.desc) desc.textContent = st.desc;
};
AnimEngine.prototype._bindControls = function(){
  if(!this.frame) return;
  var self = this;
  var prev = this.frame.querySelector(".btn-prev");
  var next = this.frame.querySelector(".btn-next");
  var play = this.frame.querySelector(".btn-play");
  var reset = this.frame.querySelector(".btn-reset");
  var slider = this.frame.querySelector('input[type=range]');
  var speedVal = this.frame.querySelector(".speed-val");
  if(prev) prev.addEventListener("click", function(){ self.prev(); });
  if(next) next.addEventListener("click", function(){ self.next(); });
  if(play) play.addEventListener("click", function(){ self.play(); });
  if(reset) reset.addEventListener("click", function(){ self.reset(); });
  if(slider) slider.addEventListener("input", function(){
    var v = parseFloat(slider.value);
    self.setSpeed(v);
    if(speedVal) speedVal.textContent = v.toFixed(1) + "×";
  });
  this.frame.addEventListener("keydown", function(e){
    if(e.key === "ArrowRight"){ e.preventDefault(); self.next(); }
    else if(e.key === "ArrowLeft"){ e.preventDefault(); self.prev(); }
    else if(e.key === " "){ e.preventDefault(); self.play(); }
    else if(e.key === "Home"){ e.preventDefault(); self.reset(); }
  });
};

/* ---- Drawing primitives ---- */
function r(e, x, y, w, h, opts){
  opts = opts || {};
  return e.draw("rect", Object.assign({x:x,y:y,width:w,height:h,rx:opts.rx||4,fill:opts.fill||"none",stroke:opts.stroke||"none","stroke-width":opts.sw||1}, opts.attrs));
}
function t(e, x, y, str, opts){
  opts = opts || {};
  var lines = String(str).split("\n");
  var el = e.draw("text", Object.assign({x:x,y:y,"text-anchor":opts.ta||"middle",class:opts.cls||"label","font-size":opts.fs||11,fill:opts.fill}, opts.attrs), null);
  if(lines.length === 1){ el.textContent = str; }
  else {
    var fs = opts.fs || 11;
    lines.forEach(function(line, i){
      var ts = document.createElementNS(SVG_NS, "tspan");
      ts.setAttribute("x", x);
      ts.setAttribute("dy", i === 0 ? 0 : fs * 1.15);
      ts.textContent = line;
      el.appendChild(ts);
    });
  }
  return el;
}
function line(e, x1, y1, x2, y2, opts){
  opts = opts || {};
  return e.draw("line", {x1:x1,y1:y1,x2:x2,y2:y2,stroke:opts.stroke||"var(--sv-text-muted)","stroke-width":opts.sw||1.5,"stroke-dasharray":opts.dash||"none","marker-end":opts.marker||null});
}
function arrow(e, x1, y1, x2, y2, color, markerSuffix){
  return e.draw("line", {x1:x1,y1:y1,x2:x2,y2:y2,stroke:color,"stroke-width":1.8,"marker-end":"url(#"+e._uid(markerSuffix)+")"});
}
function box(e, x, y, w, h, label, opts){
  opts = opts || {};
  var rect = r(e, x, y, w, h, {fill:opts.fill||"var(--sv-surface)",stroke:opts.stroke||"var(--sv-border)",sw:1.5,rx:5,attrs:opts.attrs});
  if(label){
    var fs = opts.fs || 12;
    var ty = y + h/2 + fs/3;
    t(e, x + w/2, ty, label, {fs:fs, fill:opts.tfill||"var(--sv-text)", cls:opts.tcls||"label"});
  }
  return rect;
}

/* ============================================================
 * Scene 1 — Three DP architectures
 * ============================================================ */
function scene1Build(){
  var W = 820, H = 360;
  var steps = [];
  function drawBase(e){
    t(e, W/2, 22, "Worker GPU 上的资源摆放", {fs:14, fill:"var(--sv-text)", cls:"title-txt"});
    var cols = [
      {x:60,  c:"var(--sv-red)", name:"ZeRO-Infinity"},
      {x:310, c:"var(--sv-orange)", name:"SwitchML"},
      {x:560, c:"var(--sv-green)", name:"DisDP"}
    ];
    cols.forEach(function(c){
      t(e, c.x + 100, 50, c.name, {fs:12, fill:c.c, cls:"title-txt"});
      r(e, c.x, 60, 200, 280, {fill:"var(--sv-canvas-bg)", stroke:"var(--sv-border)", sw:1, rx:6});
    });
    return cols;
  }
  steps.push({
    title: "ZeRO-Infinity：计算+网络+存储全挤在 GPU",
    short: "ZeRO",
    desc: "worker GPU 上同时跑 GEMM（计算）+ NCCL AllGather/ReduceScatter（网络）+ CPU optimizer（存储）。三种资源互相干扰，MFU 仅 15%。",
    render: function(e){
      var cols = drawBase(e); var c = cols[0];
      box(e, c.x+20, 80, 160, 240, "", {fill:"var(--sv-red-bg)", stroke:"var(--sv-red)", sw:1.5, rx:5});
      t(e, c.x+100, 98, "GPU", {fs:12, fill:"var(--sv-red)", cls:"title-txt"});
      box(e, c.x+35, 115, 130, 45, "GEMM 计算", {fill:"var(--sv-red-bg-soft)", stroke:"var(--sv-red)", sw:1, rx:4, fs:11, tfill:"var(--sv-red)"});
      box(e, c.x+35, 170, 130, 45, "NCCL 通信核", {fill:"var(--sv-red-bg-soft)", stroke:"var(--sv-red)", sw:1, rx:4, fs:11, tfill:"var(--sv-red)"});
      box(e, c.x+35, 225, 130, 45, "Optimizer (CPU)", {fill:"var(--sv-red-bg-soft)", stroke:"var(--sv-red)", sw:1, rx:4, fs:10, tfill:"var(--sv-red)"});
      arrow(e, c.x+100, 160, c.x+100, 170, "var(--sv-red)", "arrR");
      arrow(e, c.x+100, 215, c.x+100, 225, "var(--sv-red)", "arrR");
      t(e, c.x+100, 290, "⚠ 三者争 SM + 显存带宽", {fs:10, fill:"var(--sv-red)", cls:"label"});
      t(e, c.x+100, 305, "MFU 15%", {fs:12, fill:"var(--sv-red)", cls:"title-txt"});
    }
  });
  steps.push({
    title: "SwitchML：只把 in-network 聚合挪到交换机",
    short: "SwitchML",
    desc: "SmartSwitch 做 in-network aggregation，减少了 AllReduce 流量。但 AG/RS 仍由 GPU 端 NCCL 管理数据块，计算-通信干扰未除；且 AG/RS 单向流量不变，通信时间难降。",
    render: function(e){
      var cols = drawBase(e); var c = cols[1];
      box(e, c.x+20, 80, 160, 200, "", {fill:"var(--sv-orange-bg)", stroke:"var(--sv-orange)", sw:1.5, rx:5});
      t(e, c.x+100, 98, "GPU", {fs:12, fill:"var(--sv-orange)", cls:"title-txt"});
      box(e, c.x+35, 115, 130, 45, "GEMM 计算", {fill:"var(--sv-orange-bg-soft)", stroke:"var(--sv-orange)", sw:1, rx:4, fs:11, tfill:"var(--sv-orange)"});
      box(e, c.x+35, 170, 130, 45, "NCCL 通信核", {fill:"var(--sv-orange-bg-soft)", stroke:"var(--sv-orange)", sw:1, rx:4, fs:11, tfill:"var(--sv-orange)"});
      arrow(e, c.x+100, 160, c.x+100, 170, "var(--sv-orange)", "arrO");
      t(e, c.x+100, 250, "Optimizer (CPU)", {fs:10, fill:"var(--sv-orange)", cls:"label"});
      box(e, c.x+30, 295, 140, 35, "SmartSwitch (聚合)", {fill:"var(--sv-green-bg)", stroke:"var(--sv-green)", sw:1.5, rx:4, fs:10, tfill:"var(--sv-green)"});
      arrow(e, c.x+100, 250, c.x+100, 295, "var(--sv-green)", "arrG");
      t(e, c.x+100, 350, "部分解耦，仍受干扰", {fs:10, fill:"var(--sv-orange)", cls:"label"});
    }
  });
  steps.push({
    title: "DisDP：计算/网络/存储三路全解耦",
    short: "DisDP",
    desc: "GEMM 留 GPU；collectives 卸到 FPGA SmartNIC；optimizer 卸到单台 PS。GPU 只做计算，MFU ~59%。100Gbps 解耦 > 4.8Tbps 聚合。",
    render: function(e){
      var cols = drawBase(e); var c = cols[2];
      box(e, c.x+20, 80, 160, 70, "", {fill:"var(--sv-green-bg)", stroke:"var(--sv-green)", sw:1.5, rx:5});
      t(e, c.x+100, 98, "GPU", {fs:12, fill:"var(--sv-green)", cls:"title-txt"});
      box(e, c.x+35, 115, 130, 30, "GEMM 计算", {fill:"var(--sv-green-bg-soft)", stroke:"var(--sv-green)", sw:1, rx:4, fs:11, tfill:"var(--sv-green)"});
      box(e, c.x+20, 165, 160, 50, "FPGA SmartNIC (通信)", {fill:"var(--sv-blue-bg)", stroke:"var(--sv-blue)", sw:1.5, rx:5, fs:11, tfill:"var(--sv-blue)"});
      box(e, c.x+20, 230, 160, 45, "PS (Optimizer)", {fill:"var(--sv-purple-bg)", stroke:"var(--sv-purple)", sw:1.5, rx:5, fs:11, tfill:"var(--sv-purple)"});
      box(e, c.x+20, 290, 160, 35, "SmartSwitch (聚合)", {fill:"var(--sv-teal-bg)", stroke:"var(--sv-teal)", sw:1.5, rx:5, fs:10, tfill:"var(--sv-teal)"});
      arrow(e, c.x+100, 150, c.x+100, 165, "var(--sv-blue)", "arrA");
      arrow(e, c.x+100, 215, c.x+100, 230, "var(--sv-purple)", "arrP");
      arrow(e, c.x+100, 275, c.x+100, 290, "var(--sv-teal)", "arrT");
      t(e, c.x+100, 350, "全解耦，MFU ~59%", {fs:12, fill:"var(--sv-green)", cls:"title-txt"});
    }
  });
  return steps;
}

/* ============================================================
 * Scene 2 — DisDP dataflow
 * ============================================================ */
function scene2Build(){
  var W = 820, H = 380;
  var steps = [];
  function drawBase(e){
    t(e, W/2, 22, "DisDP 训练迭代数据流", {fs:14, fill:"var(--sv-text)", cls:"title-txt"});
    for(var i=0; i<3; i++){
      var x = 60 + i*200;
      box(e, x, 60, 160, 50, "GPU " + (i+1), {fill:"var(--sv-green-bg)", stroke:"var(--sv-green)", sw:1.5, rx:4, fs:11, tfill:"var(--sv-green)"});
      box(e, x, 120, 160, 35, "SmartNIC " + (i+1), {fill:"var(--sv-blue-bg)", stroke:"var(--sv-blue)", sw:1, rx:4, fs:10, tfill:"var(--sv-blue)"});
    }
    t(e, 30, 110, "...", {fs:14, fill:"var(--sv-text-muted)", cls:"label", ta:"start"});
    box(e, 50, 200, 580, 40, "SmartSwitch (聚合 / 广播 / heartbeat 聚合)", {fill:"var(--sv-teal-bg)", stroke:"var(--sv-teal)", sw:1.5, rx:4, fs:11, tfill:"var(--sv-teal)"});
    box(e, 250, 290, 280, 50, "Parameter Server (Adam)", {fill:"var(--sv-purple-bg)", stroke:"var(--sv-purple)", sw:1.5, rx:4, fs:12, tfill:"var(--sv-purple)"});
    box(e, 590, 290, 90, 50, "SSD", {fill:"var(--sv-orange-bg)", stroke:"var(--sv-orange)", sw:1, rx:4, fs:11, tfill:"var(--sv-orange)"});
  }
  steps.push({
    title: "① 前向阶段：PS 广播参数 → GPU",
    short: "前向 pull",
    desc: "PS 从 SSD 读参数，push 到 SmartNIC；SmartSwitch 广播到所有 worker 的 SmartNIC；worker pull 参数到 GPU 显存。GPU 开始前向 GEMM。",
    render: function(e){
      drawBase(e);
      arrow(e, 590, 315, 530, 315, "var(--sv-orange)", "arrO");
      arrow(e, 390, 290, 390, 240, "var(--sv-purple)", "arrP");
      t(e, 410, 268, "push 参数", {fs:10, fill:"var(--sv-purple)", cls:"label", ta:"start"});
      for(var i=0;i<3;i++){ var x = 140 + i*200; arrow(e, x, 200, x, 155, "var(--sv-teal)", "arrT"); }
      t(e, 180, 178, "广播", {fs:10, fill:"var(--sv-teal)", cls:"label"});
      box(e, 60, 60, 160, 50, "GPU 1 ★", {fill:"var(--sv-green-bg-med)", stroke:"var(--sv-green)", sw:2, rx:4, fs:11, tfill:"var(--sv-green)"});
    }
  });
  steps.push({
    title: "② GPU 前向 GEMM（无通信干扰）",
    short: "前向 GEMM",
    desc: "GPU 跑前向 GEMM。注意：SmartNIC pull 与 GPU GEMM 通过 PCIe DMA 并发，几乎无干扰（论文图 4 实测）。这是 DisDP 的核心优势所在。",
    render: function(e){
      drawBase(e);
      box(e, 60, 60, 160, 50, "GPU 1 GEMM", {fill:"var(--sv-green-bg-med)", stroke:"var(--sv-green)", sw:2, rx:4, fs:11, tfill:"var(--sv-green)"});
      box(e, 260, 60, 160, 50, "GPU 2 GEMM", {fill:"var(--sv-green-bg-med)", stroke:"var(--sv-green)", sw:2, rx:4, fs:11, tfill:"var(--sv-green)"});
      box(e, 460, 60, 160, 50, "GPU 3 GEMM", {fill:"var(--sv-green-bg-med)", stroke:"var(--sv-green)", sw:2, rx:4, fs:11, tfill:"var(--sv-green)"});
      line(e, 140, 110, 140, 120, {stroke:"var(--sv-green)", sw:1.5, dash:"4,3"});
      t(e, 700, 90, "DMA\n无干扰", {fs:10, fill:"var(--sv-green)", cls:"label", ta:"start"});
    }
  });
  steps.push({
    title: "③ 反向阶段：GPU push 梯度 → SmartSwitch 聚合",
    short: "反向 push",
    desc: "反向传播时，每个 worker 把本层部分梯度 push 到 SmartNIC；SmartSwitch 对所有 worker 的梯度做 in-network aggregation，产生聚合梯度。GPU 同时继续反向 GEMM。",
    render: function(e){
      drawBase(e);
      for(var i=0;i<3;i++){ var x = 140 + i*200; arrow(e, x, 155, x, 200, "var(--sv-blue)", "arrA"); }
      t(e, 200, 180, "push 部分梯度", {fs:10, fill:"var(--sv-blue)", cls:"label", ta:"start"});
      box(e, 50, 200, 580, 40, "SmartSwitch ★ 聚合梯度中", {fill:"var(--sv-teal-bg-med)", stroke:"var(--sv-teal)", sw:2, rx:4, fs:11, tfill:"var(--sv-teal)"});
      box(e, 60, 60, 160, 50, "GPU 1 反向", {fill:"var(--sv-green-bg-med)", stroke:"var(--sv-green)", sw:1.5, rx:4, fs:11, tfill:"var(--sv-green)"});
    }
  });
  steps.push({
    title: "④ SmartSwitch 聚合梯度 → PS，PS 跑 Adam",
    short: "PS Adam",
    desc: "SmartSwitch 把聚合梯度 pull 给 PS。PS 用步骤中心流水线跑 out-of-core Adam：SSD 读模型状态 → CPU Adam → 写回 SSD + 产出新参数。PS 与 GPU 反向并行。",
    render: function(e){
      drawBase(e);
      arrow(e, 390, 240, 390, 290, "var(--sv-teal)", "arrT");
      t(e, 400, 268, "pull 聚合梯度", {fs:10, fill:"var(--sv-teal)", cls:"label", ta:"start"});
      box(e, 250, 290, 280, 50, "PS ★ Adam 流水线", {fill:"var(--sv-purple-bg-med)", stroke:"var(--sv-purple)", sw:2, rx:4, fs:11, tfill:"var(--sv-purple)"});
      arrow(e, 590, 305, 530, 305, "var(--sv-orange)", "arrO");
      arrow(e, 530, 325, 590, 325, "var(--sv-orange)", "arrO");
      t(e, 560, 295, "读/写", {fs:9, fill:"var(--sv-orange)", cls:"label"});
      t(e, 50, 260, "heartbeat 聚合\n(每周期 1 ACK)", {fs:9, fill:"var(--sv-teal)", cls:"label", ta:"start"});
    }
  });
  steps.push({
    title: "⑤ PS push 新参数 → SmartSwitch 广播 → worker pull",
    short: "广播新参数",
    desc: "PS 把新参数 push 回 SmartSwitch，SmartSwitch 广播到所有 worker，worker pull 到 GPU。一次迭代完成，全程 GPU 只做 GEMM，通信完全在 SmartNIC/SmartSwitch/PS 之间。",
    render: function(e){
      drawBase(e);
      arrow(e, 390, 290, 390, 240, "var(--sv-purple)", "arrP");
      t(e, 400, 268, "push 新参数", {fs:10, fill:"var(--sv-purple)", cls:"label", ta:"start"});
      for(var i=0;i<3;i++){ var x = 140 + i*200; arrow(e, x, 200, x, 155, "var(--sv-teal)", "arrT"); }
      t(e, 200, 180, "广播参数", {fs:10, fill:"var(--sv-teal)", cls:"label", ta:"start"});
      box(e, 60, 60, 160, 50, "GPU 1 pull", {fill:"var(--sv-green-bg)", stroke:"var(--sv-green)", sw:1.5, rx:4, fs:11, tfill:"var(--sv-green)"});
      t(e, W/2, 360, "★ 全程 GPU 无通信核干扰", {fs:12, fill:"var(--sv-green)", cls:"title-txt"});
    }
  });
  return steps;
}

/* ============================================================
 * Scene 3 — SoC vs FPGA SmartNIC
 * ============================================================ */
function scene3Build(){
  var W = 820, H = 340;
  var steps = [];
  steps.push({
    title: "SoC SmartNIC（BlueField）：两条瓶颈导致 20% 线路利用率",
    short: "SoC 瓶颈",
    desc: "Off-path 架构：流量 host↔Arm↔网络 来回搬运。① Arm-交换 PCIe 链路带宽不够（BF-2 需 400Gbps，仅 250Gbps）；② Arm 显存带宽不够（需 800Gbps，仅 204.8Gbps），每方向 2× 访问。结果只有 20% 线路利用率。",
    render: function(e){
      t(e, W/2, 22, "SoC SmartNIC（Off-path，BlueField）", {fs:14, fill:"var(--sv-red)", cls:"title-txt"});
      box(e, 60, 70, 120, 50, "Host (GPU)", {fill:"var(--sv-red-bg)", stroke:"var(--sv-red)", sw:1.5, rx:4, fs:11, tfill:"var(--sv-red)"});
      box(e, 280, 70, 100, 50, "内部交换", {fill:"var(--sv-orange-bg)", stroke:"var(--sv-orange)", sw:1.5, rx:4, fs:10, tfill:"var(--sv-orange)"});
      box(e, 460, 70, 100, 50, "Arm 核心", {fill:"var(--sv-red-bg)", stroke:"var(--sv-red)", sw:1.5, rx:4, fs:10, tfill:"var(--sv-red)"});
      box(e, 460, 150, 100, 35, "Arm DRAM", {fill:"var(--sv-red-bg-med)", stroke:"var(--sv-red)", sw:1, rx:4, fs:10, tfill:"var(--sv-red)"});
      box(e, 640, 70, 120, 50, "网络", {fill:"var(--sv-grey-bg)", stroke:"var(--sv-text-muted)", sw:1.5, rx:4, fs:11, tfill:"var(--sv-text)"});
      arrow(e, 180, 90, 280, 90, "var(--sv-red)", "arrR");
      arrow(e, 380, 90, 460, 90, "var(--sv-red)", "arrR");
      arrow(e, 510, 120, 510, 150, "var(--sv-red)", "arrR");
      arrow(e, 460, 90, 380, 90, "var(--sv-orange)", "arrO");
      arrow(e, 560, 90, 640, 90, "var(--sv-red)", "arrR");
      t(e, 330, 140, "瓶颈1: Arm-交换链路\nBF-2 需400Gbps 仅250Gbps", {fs:9, fill:"var(--sv-red)", cls:"label"});
      t(e, 580, 170, "瓶颈2: 2× DRAM 访问\n需800Gbps 仅204.8Gbps", {fs:9, fill:"var(--sv-red)", cls:"label", ta:"start"});
      t(e, W/2, 240, "实测 BF-2 仅 20% 线路利用率", {fs:13, fill:"var(--sv-red)", cls:"title-txt"});
      t(e, W/2, 270, "BF-3 同样未解决（需 1600Gbps，仅 716.8Gbps）", {fs:11, fill:"var(--sv-text-muted)", cls:"label"});
    }
  });
  steps.push({
    title: "FPGA SmartNIC（U50）：on-path 流水线 + 片上 SRAM → 线速",
    short: "FPGA 线速",
    desc: "On-path 架构：包在硬件流水线里处理，不经 Arm、不经内部交换。流水线级之间用片上 SRAM 暂存，避免 off-chip DRAM 带宽瓶颈。push/pull 独立处理单元并发，打到线速。",
    render: function(e){
      t(e, W/2, 22, "FPGA SmartNIC（On-path，U50）", {fs:14, fill:"var(--sv-green)", cls:"title-txt"});
      box(e, 60, 70, 120, 50, "Host (GPU)", {fill:"var(--sv-green-bg)", stroke:"var(--sv-green)", sw:1.5, rx:4, fs:11, tfill:"var(--sv-green)"});
      var stages = ["Stage 1\nDMA", "Stage 2\n格式转换", "Stage 3\npush/pull\n处理", "Stage 4\n网络"];
      for(var i=0; i<4; i++){
        var x = 220 + i*110;
        box(e, x, 70, 90, 70, stages[i], {fill:"var(--sv-green-bg-soft)", stroke:"var(--sv-green)", sw:1.5, rx:4, fs:9, tfill:"var(--sv-green)"});
        if(i < 3){ t(e, x+95, 100, "SRAM", {fs:8, fill:"var(--sv-blue)", cls:"label"}); arrow(e, x+90, 100, x+110, 100, "var(--sv-blue)", "arrA"); }
      }
      box(e, 660, 70, 100, 50, "网络", {fill:"var(--sv-grey-bg)", stroke:"var(--sv-text-muted)", sw:1.5, rx:4, fs:11, tfill:"var(--sv-text)"});
      arrow(e, 610, 95, 660, 95, "var(--sv-green)", "arrG");
      arrow(e, 180, 95, 220, 95, "var(--sv-green)", "arrG");
      t(e, W/2, 180, "无内部交换争用 · 无 2× DRAM 访问", {fs:12, fill:"var(--sv-green)", cls:"title-txt"});
      t(e, W/2, 210, "片上 SRAM 暂存 → 不碰 off-chip DRAM", {fs:11, fill:"var(--sv-text-muted)", cls:"label"});
      t(e, W/2, 240, "push / pull 独立硬件模块 → 双工并发", {fs:11, fill:"var(--sv-text-muted)", cls:"label"});
      t(e, W/2, 280, "★ 线路利用率 → 100%", {fs:14, fill:"var(--sv-green)", cls:"title-txt"});
    }
  });
  steps.push({
    title: "对比小结：为什么必须是 FPGA",
    short: "小结",
    desc: "SoC SmartNIC 的两大瓶颈源于芯片架构（off-path + lookaside Arm + DRAM staging），靠下一代硬件也难解。FPGA 的 on-path + SRAM 流水线从架构层面绕开这两个瓶颈，是 DisDP 能打到线速的前提。",
    render: function(e){
      t(e, W/2, 30, "SoC vs FPGA SmartNIC 对比", {fs:14, fill:"var(--sv-text)", cls:"title-txt"});
      var rows = [
        ["维度", "SoC (BlueField)", "FPGA (U50)"],
        ["架构", "Off-path, lookaside", "On-path, 流水线"],
        ["内部交换争用", "有（Arm-交换链路）", "无"],
        ["显存访问", "2× off-chip DRAM", "片上 SRAM 暂存"],
        ["线路利用率", "20%", "~100%"],
        ["可编程性", "C/C++ 易开发", "FPGA 需 HDL"]
      ];
      var y0 = 60, rh = 32, cw = [120, 220, 220];
      for(var r_i=0; r_i<rows.length; r_i++){
        var row = rows[r_i]; var x = 120;
        for(var c_i=0; c_i<row.length; c_i++){
          var fill = r_i===0 ? "var(--sv-orange-bg)" : (c_i===1 ? "var(--sv-red-bg)" : "var(--sv-green-bg)");
          var stroke = r_i===0 ? "var(--sv-orange)" : (c_i===1 ? "var(--sv-red)" : "var(--sv-green)");
          var tfill = r_i===0 ? "var(--sv-orange)" : (c_i===1 ? "var(--sv-red)" : "var(--sv-green)");
          box(e, x, y0 + r_i*rh, cw[c_i], rh, row[c_i], {fill:fill, stroke:stroke, sw:1, rx:3, fs:10, tfill:tfill});
          x += cw[c_i];
        }
      }
    }
  });
  return steps;
}

/* ============================================================
 * Scene 4 — Many-to-one reliable protocol
 * ============================================================ */
function scene4Build(){
  var W = 820, H = 360;
  var steps = [];
  steps.push({
    title: "朴素方案：每个 worker 与 PS 维持 one-to-one reliable 连接",
    short: "ACK 风暴",
    desc: "每个 worker 发一个数据包，PS 就要回一个 ACK。N 个 worker → PS 每轮收 N 个 ACK。32 worker 吞吐降到 30Gbps，64 worker 降到 18Gbps。",
    render: function(e){
      t(e, W/2, 22, "朴素：每包 ACK → ACK 风暴耗尽 PS IOPS", {fs:13, fill:"var(--sv-red)", cls:"title-txt"});
      for(var i=0; i<5; i++){
        var x = 60 + i*130;
        box(e, x, 60, 100, 40, "Worker " + (i+1), {fill:"var(--sv-grey-bg)", stroke:"var(--sv-text-muted)", sw:1, rx:3, fs:10, tfill:"var(--sv-text)"});
        arrow(e, x+50, 100, 360, 200, "var(--sv-blue)", "arrA");
        arrow(e, 360, 210, x+50, 100, "var(--sv-red)", "arrR");
      }
      t(e, 200, 160, "数据包", {fs:10, fill:"var(--sv-blue)", cls:"label"});
      t(e, 480, 160, "ACK × N", {fs:10, fill:"var(--sv-red)", cls:"label", ta:"start"});
      box(e, 280, 200, 200, 50, "Parameter Server", {fill:"var(--sv-red-bg)", stroke:"var(--sv-red)", sw:2, rx:4, fs:11, tfill:"var(--sv-red)"});
      t(e, 380, 280, "PS IOPS 被 ACK 耗尽", {fs:12, fill:"var(--sv-red)", cls:"title-txt"});
      t(e, 380, 305, "32 worker → 30Gbps", {fs:11, fill:"var(--sv-text-muted)", cls:"label"});
      t(e, 380, 325, "64 worker → 18Gbps", {fs:11, fill:"var(--sv-text-muted)", cls:"label"});
    }
  });
  steps.push({
    title: "方案A 不可行：让 SmartSwitch 做 reliable endpoint",
    short: "Switch 不行",
    desc: "让 SmartSwitch 自己跑 reliable 协议也不行：Tofino 流水线 ≤20 级，而 reliable RDMA/TCP 需 >50 级。交换机硬件资源不足以做 endpoint。",
    render: function(e){
      t(e, W/2, 22, "方案A：SmartSwitch 做 endpoint → 硬件不够", {fs:13, fill:"var(--sv-red)", cls:"title-txt"});
      box(e, 280, 80, 200, 60, "SmartSwitch (Tofino)", {fill:"var(--sv-red-bg)", stroke:"var(--sv-red)", sw:2, rx:4, fs:11, tfill:"var(--sv-red)"});
      for(var i=0;i<20;i++){ r(e, 100 + i*30, 160, 26, 25, {fill:"var(--sv-red-bg)", stroke:"var(--sv-red)", sw:0.5, rx:2}); }
      t(e, W/2, 200, "Tofino: ≤20 级", {fs:11, fill:"var(--sv-red)", cls:"label"});
      for(var i=0;i<50;i++){ r(e, 100 + i*12, 230, 10, 20, {fill:"var(--sv-green-bg)", stroke:"var(--sv-green)", sw:0.5, rx:1}); }
      t(e, W/2, 265, "reliable RDMA/TCP: >50 级", {fs:11, fill:"var(--sv-green)", cls:"label"});
      t(e, W/2, 310, "✗ 交换机硬件不足以做 reliable endpoint", {fs:13, fill:"var(--sv-red)", cls:"title-txt"});
    }
  });
  steps.push({
    title: "DisDP 方案：周期 heartbeat + SmartSwitch 聚合 ACK",
    short: "heartbeat 聚合",
    desc: "每个 worker 周期发 heartbeat（含 Ack + Credit）。SmartSwitch 维护 heartbeat 表，对各 worker 做最小聚合，只向 PS 发 1 个聚合 heartbeat。PS 每周期只收发各 1 个 ACK，与 worker 数无关。",
    render: function(e){
      t(e, W/2, 22, "DisDP：SmartSwitch 聚合 heartbeat → PS 每周期仅 1 ACK", {fs:13, fill:"var(--sv-green)", cls:"title-txt"});
      for(var i=0; i<5; i++){
        var x = 60 + i*130;
        box(e, x, 60, 100, 40, "Worker " + (i+1), {fill:"var(--sv-grey-bg)", stroke:"var(--sv-text-muted)", sw:1, rx:3, fs:10, tfill:"var(--sv-text)"});
        arrow(e, x+50, 100, 360, 150, "var(--sv-teal)", "arrT");
      }
      t(e, 200, 135, "heartbeat × N", {fs:10, fill:"var(--sv-teal)", cls:"label"});
      box(e, 280, 150, 200, 50, "SmartSwitch ★ 聚合", {fill:"var(--sv-teal-bg)", stroke:"var(--sv-teal)", sw:2, rx:4, fs:11, tfill:"var(--sv-teal)"});
      arrow(e, 380, 200, 380, 250, "var(--sv-green)", "arrG");
      t(e, 390, 230, "1 个聚合 heartbeat", {fs:10, fill:"var(--sv-green)", cls:"label", ta:"start"});
      box(e, 280, 250, 200, 45, "Parameter Server", {fill:"var(--sv-green-bg)", stroke:"var(--sv-green)", sw:1.5, rx:4, fs:11, tfill:"var(--sv-green)"});
      t(e, 380, 320, "PS IOPS 不随 worker 数增长", {fs:12, fill:"var(--sv-green)", cls:"title-txt"});
    }
  });
  steps.push({
    title: "流控（Credit）+ 可靠（Ack + RTO 重传）",
    short: "流控+可靠",
    desc: "流控：发包到 TX Credit 上限就停，等 heartbeat 更新。可靠：超 RTO(~250μs) 未推进就重传，SmartSwitch 仅转发重传包。最小聚合保证最慢的 worker 不被甩开。",
    render: function(e){
      t(e, W/2, 22, "heartbeat 携带 Ack + Credit", {fs:13, fill:"var(--sv-text)", cls:"title-txt"});
      box(e, 60, 70, 140, 60, "Worker", {fill:"var(--sv-grey-bg)", stroke:"var(--sv-text-muted)", sw:1.5, rx:4, fs:11, tfill:"var(--sv-text)"});
      box(e, 340, 70, 140, 60, "SmartSwitch", {fill:"var(--sv-teal-bg)", stroke:"var(--sv-teal)", sw:1.5, rx:4, fs:11, tfill:"var(--sv-teal)"});
      box(e, 620, 70, 140, 60, "PS", {fill:"var(--sv-green-bg)", stroke:"var(--sv-green)", sw:1.5, rx:4, fs:11, tfill:"var(--sv-green)"});
      box(e, 200, 160, 100, 35, "Ack", {fill:"var(--sv-purple-bg)", stroke:"var(--sv-purple)", sw:1, rx:3, fs:11, tfill:"var(--sv-purple)"});
      box(e, 320, 160, 100, 35, "Credit", {fill:"var(--sv-purple-bg)", stroke:"var(--sv-purple)", sw:1, rx:3, fs:11, tfill:"var(--sv-purple)"});
      t(e, 310, 150, "heartbeat 内容", {fs:10, fill:"var(--sv-purple)", cls:"label"});
      t(e, 100, 220, "流控: 发到 TX Credit 上限即停", {fs:11, fill:"var(--sv-blue)", cls:"label", ta:"start"});
      t(e, 100, 245, "  等 heartbeat 更新 Credit 再继续", {fs:11, fill:"var(--sv-text-muted)", cls:"label", ta:"start"});
      t(e, 450, 220, "可靠: 超 RTO(~250μs) 未推进 → 重传", {fs:11, fill:"var(--sv-red)", cls:"label", ta:"start"});
      t(e, 450, 245, "  SmartSwitch 仅转发重传包，不处理", {fs:11, fill:"var(--sv-text-muted)", cls:"label", ta:"start"});
      t(e, W/2, 290, "最小聚合 → 最慢 worker 不被甩开，公平性有保障", {fs:12, fill:"var(--sv-green)", cls:"title-txt"});
    }
  });
  return steps;
}

/* ============================================================
 * Scene 5 — Layer-centric vs Step-centric optimizer pipeline
 * ============================================================ */
function scene5Build(){
  var W = 820, H = 360;
  var steps = [];
  steps.push({
    title: "层中心流水线：每层固定 32 线程 → 并行度被卡死",
    short: "层中心",
    desc: "传统做法给每层分配固定线程。但 Step 5（CPU Adam）要 32 线程才线速，其余步骤只需 1 线程。每层都占 32 线程 → 少量层就把 CPU 线程耗尽，流水线出现气泡。",
    render: function(e){
      t(e, W/2, 22, "层中心：每层 32 线程，气泡多", {fs:14, fill:"var(--sv-red)", cls:"title-txt"});
      var stepNames = ["① SSD→CPU", "② push", "③ pull", "④ SSD→CPU", "⑤ Adam(32线程)", "⑥ 写回SSD"];
      var colors = ["var(--sv-text-muted)","var(--sv-blue)","var(--sv-teal)","var(--sv-text-muted)","var(--sv-red)","var(--sv-orange)"];
      for(var i=0;i<6;i++){
        var x = 40 + i*120; var w = i===4 ? 110 : 90;
        box(e, x, 60, w, 40, stepNames[i], {fill:"var(--sv-red-bg)", stroke:colors[i], sw:1, rx:3, fs:9, tfill:colors[i]});
      }
      t(e, 30, 50, "Layer 1", {fs:10, fill:"var(--sv-text-muted)", cls:"label", ta:"start"});
      for(var i=0;i<6;i++){
        var x = 40 + i*120; var w = i===4 ? 110 : 90;
        box(e, x + 30, 120, w, 40, stepNames[i], {fill:"var(--sv-red-bg)", stroke:colors[i], sw:1, rx:3, fs:9, tfill:colors[i]});
      }
      t(e, 30, 110, "Layer 2", {fs:10, fill:"var(--sv-text-muted)", cls:"label", ta:"start"});
      box(e, 70, 180, 90, 40, stepNames[0], {fill:"var(--sv-red-bg)", stroke:colors[0], sw:1, rx:3, fs:9, tfill:colors[0]});
      t(e, 30, 170, "Layer 3", {fs:10, fill:"var(--sv-text-muted)", cls:"label", ta:"start"});
      t(e, 380, 250, "⚠ 气泡: 线程不够，层等不到调度", {fs:12, fill:"var(--sv-red)", cls:"title-txt"});
      t(e, 380, 275, "32 线程/层 × 3 层 = 96 线程，机器只有 ~32", {fs:11, fill:"var(--sv-text-muted)", cls:"label"});
      t(e, 380, 300, "→ 大量层排队，无法线速消费梯度", {fs:11, fill:"var(--sv-text-muted)", cls:"label"});
    }
  });
  steps.push({
    title: "步骤中心流水线：按步骤分线程，层在步骤间轮转",
    short: "步骤中心",
    desc: "DisDP 按步骤而非按层分配线程：Step 5 分 32 线程，其余步骤分少线程。层从 Step 1 流到 Step 6。37 个线程就能跑满整条流水线，无气泡。",
    render: function(e){
      t(e, W/2, 22, "步骤中心：按步骤分线程，37 线程跑满", {fs:14, fill:"var(--sv-green)", cls:"title-txt"});
      var stepNames = ["① SSD→CPU\n(1线程)", "② push\n(1线程)", "③ pull\n(1线程)", "④ SSD→CPU\n(1线程)", "⑤ Adam\n(32线程)", "⑥ 写回\n(1线程)"];
      var colors = ["var(--sv-text-muted)","var(--sv-blue)","var(--sv-teal)","var(--sv-text-muted)","var(--sv-red)","var(--sv-orange)"];
      for(var i=0;i<6;i++){
        var x = 40 + i*125; var w = 100;
        box(e, x, 60, w, 50, stepNames[i], {fill:"var(--sv-green-bg-soft)", stroke:colors[i], sw:1.5, rx:3, fs:9, tfill:colors[i]});
      }
      var layerYs = [120, 160, 200]; var layerOffsets = [0, 1, 2];
      for(var l=0; l<3; l++){
        var li = layerOffsets[l]; var x = 40 + li*125 + 50;
        box(e, x-30, layerYs[l], 60, 30, "Layer " + (l+1), {fill:"var(--sv-green-bg)", stroke:"var(--sv-green)", sw:1, rx:3, fs:9, tfill:"var(--sv-green)"});
        if(li < 5){ arrow(e, x+30, layerYs[l]+15, x+95, layerYs[l]+15, "var(--sv-green)", "arrG"); }
      }
      for(var i=0;i<5;i++){ arrow(e, 40+i*125+100, 85, 40+(i+1)*125, 85, "var(--sv-text-muted)", "arr"); }
      t(e, W/2, 250, "层在步骤间轮转，每步骤线程数独立", {fs:12, fill:"var(--sv-green)", cls:"title-txt"});
      t(e, W/2, 275, "1+1+1+1+32+1 = 37 线程", {fs:12, fill:"var(--sv-purple)", cls:"title-txt"});
      t(e, W/2, 300, "→ 无气泡，线速消费聚合梯度", {fs:11, fill:"var(--sv-text-muted)", cls:"label"});
    }
  });
  steps.push({
    title: "反向阶段资源需求（100Gbps 线速）",
    short: "资源需求",
    desc: "反向阶段 6 步合计：99 GFLOPS 计算、349 GB/s 显存带宽、81.4 GB/s SSD 带宽。单台 Intel 6730P（PCIe Gen5 + 12 SSD）即可满足，服务任意多 worker。",
    render: function(e){
      t(e, W/2, 22, "反向阶段资源需求", {fs:14, fill:"var(--sv-text)", cls:"title-txt"});
      var rows = [
        ["资源", "需求", "6730P 满足?"],
        ["计算", "99 GFLOPS", "✓"],
        ["显存带宽", "349 GB/s", "✓"],
        ["SSD 带宽", "81.4 GB/s", "✓ (12 SSD)"],
        ["网络", "100Gbps", "✓"]
      ];
      var y0 = 60, rh = 38, cw = [180, 200, 200];
      for(var r_i=0; r_i<rows.length; r_i++){
        var row = rows[r_i]; var x = 130;
        for(var c_i=0; c_i<row.length; c_i++){
          var fill = r_i===0 ? "var(--sv-orange-bg)" : "var(--sv-green-bg-soft)";
          var stroke = r_i===0 ? "var(--sv-orange)" : "var(--sv-green)";
          var tfill = r_i===0 ? "var(--sv-orange)" : (c_i===2 ? "var(--sv-green)" : "var(--sv-text)");
          box(e, x, y0 + r_i*rh, cw[c_i], rh, row[c_i], {fill:fill, stroke:stroke, sw:1, rx:3, fs:11, tfill:tfill});
          x += cw[c_i];
        }
      }
      t(e, W/2, 290, "★ 单台 PS 即可线速服务任意多 worker", {fs:12, fill:"var(--sv-green)", cls:"title-txt"});
    }
  });
  steps.push({
    title: "对比小结：层中心 vs 步骤中心",
    short: "小结",
    desc: "层中心受限于「最重步骤的线程数 × 并发层数」，步骤中心把线程按步骤实际需求分配，最大化并行度。这是 DisDP 能用单台 PS 替代传统 13 台 CPU 机器的关键。",
    render: function(e){
      t(e, W/2, 30, "层中心 vs 步骤中心", {fs:14, fill:"var(--sv-text)", cls:"title-txt"});
      var rows = [
        ["维度", "层中心", "步骤中心"],
        ["分配单位", "按层", "按步骤"],
        ["最重步骤线程", "32 (Adam)", "32 (Adam)"],
        ["其余步骤线程", "32 (浪费)", "1"],
        ["并发层最大数", "~1 (32线程/层)", "多层轮转"],
        ["总线程需求", "96+ (3层)", "37"],
        ["气泡", "多", "无"],
        ["PS 台数", "13 台", "1 台"]
      ];
      var y0 = 60, rh = 30, cw = [140, 240, 240];
      for(var r_i=0; r_i<rows.length; r_i++){
        var row = rows[r_i]; var x = 100;
        for(var c_i=0; c_i<row.length; c_i++){
          var fill = r_i===0 ? "var(--sv-orange-bg)" : (c_i===1 ? "var(--sv-red-bg)" : "var(--sv-green-bg)");
          var stroke = r_i===0 ? "var(--sv-orange)" : (c_i===1 ? "var(--sv-red)" : "var(--sv-green)");
          var tfill = r_i===0 ? "var(--sv-orange)" : (c_i===1 ? "var(--sv-red)" : "var(--sv-green)");
          box(e, x, y0 + r_i*rh, cw[c_i], rh, row[c_i], {fill:fill, stroke:stroke, sw:1, rx:3, fs:10, tfill:tfill});
          x += cw[c_i];
        }
      }
    }
  });
  return steps;
}

/* ============================================================
 * Init
 * ============================================================ */
var engines = {};
function initScene(id, builder){
  var svg = document.getElementById(id);
  if(!svg) return;
  var frame = document.getElementById(id.replace("svg","anim"));
  var engine = new AnimEngine(svg, {frame: frame, steps: builder()});
  engine.goTo(0);
  engines[id] = engine;
}

document.addEventListener("DOMContentLoaded", function(){
  initScene("svg1", scene1Build);
  initScene("svg2", scene2Build);
  initScene("svg3", scene3Build);
  initScene("svg4", scene4Build);
  initScene("svg5", scene5Build);

  // TOC scroll-spy
  var tocLinks = document.querySelectorAll(".toc-inner a[href^='#']");
  var sections = [];
  tocLinks.forEach(function(link){
    var id = link.getAttribute("href").slice(1);
    var sec = document.getElementById(id);
    if(sec) sections.push({id:id, el:sec, link:link});
  });
  var spyObserver = new IntersectionObserver(function(entries){
    entries.forEach(function(entry){
      if(entry.isIntersecting){
        tocLinks.forEach(function(l){ l.classList.remove("active"); });
        var match = sections.find(function(s){ return s.el === entry.target; });
        if(match) match.link.classList.add("active");
      }
    });
  }, {rootMargin: "-80px 0px -70% 0px"});
  sections.forEach(function(s){ spyObserver.observe(s.el); });

  // Back to top
  var btt = document.querySelector(".back-to-top");
  if(btt){
    window.addEventListener("scroll", function(){
      if(window.scrollY > 600) btt.classList.add("show");
      else btt.classList.remove("show");
    });
    btt.addEventListener("click", function(){
      window.scrollTo({top:0, behavior:"smooth"});
    });
  }
});
})();
