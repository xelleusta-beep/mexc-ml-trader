/**
 * WebGL Parallax Background — MEXC Trading System
 * Profesyonel partikul sistemi + derinlik katmanlari
 * Mouse hareketine tepki veren interaktif arka plan
 */
(function(){
  const canvas = document.createElement('canvas');
  canvas.id = 'parallax-canvas';
  canvas.style.cssText = 'position:fixed;inset:0;z-index:0;pointer-events:none;opacity:0.6';
  document.body.prepend(canvas);

  const gl = canvas.getContext('webgl', {alpha:true, antialias:true, premultipliedAlpha:false});
  if(!gl){console.warn('WebGL not supported');return;}

  let W = 0, H = 0, mouseX = 0.5, mouseY = 0.5;
  let time = 0;

  function resize(){
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
    gl.viewport(0, 0, W, H);
    initParticles();
  }

  window.addEventListener('resize', resize);
  document.addEventListener('mousemove', e=>{
    mouseX = e.clientX / W;
    mouseY = e.clientY / H;
  });

  // ── Shader Sources ──────────────────────────────────────────────────────
  const vsSource = `
    attribute vec2 aPos;
    attribute float aSize;
    attribute float aLayer;
    attribute float aPhase;
    uniform float uTime;
    uniform vec2 uMouse;
    uniform vec2 uRes;
    varying float vAlpha;
    varying float vLayer;
    varying float vPhase;
    void main(){
      float depth = aLayer;
      float parallaxX = (uMouse.x - 0.5) * depth * 60.0;
      float parallaxY = (uMouse.y - 0.5) * depth * 40.0;
      float sway = sin(uTime * 0.3 + aPhase * 6.28) * 8.0 * depth;
      float drift = cos(uTime * 0.2 + aPhase * 3.14) * 5.0 * depth;
      vec2 pos = aPos + vec2(parallaxX + sway, parallaxY + drift);
      pos = pos / uRes * 2.0 - 1.0;
      pos.y *= -1.0;
      gl_Position = vec4(pos, 0.0, 1.0);
      gl_PointSize = aSize * (1.0 + depth * 0.5);
      vAlpha = 0.15 + depth * 0.25;
      vLayer = depth;
      vPhase = aPhase;
    }
  `;

  const fsSource = `
    precision mediump float;
    varying float vAlpha;
    varying float vLayer;
    varying float vPhase;
    uniform float uTime;
    void main(){
      vec2 pc = gl_PointCoord * 2.0 - 1.0;
      float d = length(pc);
      if(d > 1.0) discard;
      float glow = exp(-d * 2.5);
      vec3 color1 = vec3(0.0, 0.83, 1.0);
      vec3 color2 = vec3(0.486, 0.227, 0.929);
      vec3 color = mix(color1, color2, vPhase);
      float pulse = 0.7 + 0.3 * sin(uTime * 0.5 + vPhase * 6.28);
      gl_FragColor = vec4(color, glow * vAlpha * pulse);
    }
  `;

  // ── Compile & Link ──────────────────────────────────────────────────────
  function compileShader(src, type){
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)){
      console.error(gl.getShaderInfoLog(s));
      return null;
    }
    return s;
  }

  const vs = compileShader(vsSource, gl.VERTEX_SHADER);
  const fs = compileShader(fsSource, gl.FRAGMENT_SHADER);
  if(!vs || !fs) return;

  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if(!gl.getProgramParameter(prog, gl.LINK_STATUS)){
    console.error(gl.getProgramInfoLog(prog));
    return;
  }
  gl.useProgram(prog);

  // ── Attributes & Uniforms ───────────────────────────────────────────────
  const aPos   = gl.getAttribLocation(prog, 'aPos');
  const aSize  = gl.getAttribLocation(prog, 'aSize');
  const aLayer = gl.getAttribLocation(prog, 'aLayer');
  const aPhase = gl.getAttribLocation(prog, 'aPhase');
  const uTime  = gl.getUniformLocation(prog, 'uTime');
  const uMouse = gl.getUniformLocation(prog, 'uMouse');
  const uRes   = gl.getUniformLocation(prog, 'uRes');

  let posBuf, sizeBuf, layerBuf, phaseBuf, N;

  function initParticles(){
    const layers = [
      {count: 120, sizeMin: 1.5, sizeMax: 3.0, layerMin: 0.1, layerMax: 0.3},
      {count: 80,  sizeMin: 2.0, sizeMax: 4.5, layerMin: 0.3, layerMax: 0.6},
      {count: 40,  sizeMin: 3.0, sizeMax: 6.0, layerMin: 0.6, layerMax: 1.0},
    ];

    let allPos = [], allSize = [], allLayer = [], allPhase = [];
    layers.forEach(l=>{
      for(let i=0; i<l.count; i++){
        allPos.push(Math.random() * W, Math.random() * H);
        allSize.push(l.sizeMin + Math.random() * (l.sizeMax - l.sizeMin));
        allLayer.push(l.layerMin + Math.random() * (l.layerMax - l.layerMin));
        allPhase.push(Math.random());
      }
    });

    N = allPos.length / 2;

    function makeBuf(data, loc){
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
      return buf;
    }

    if(posBuf) gl.deleteBuffer(posBuf);
    if(sizeBuf) gl.deleteBuffer(sizeBuf);
    if(layerBuf) gl.deleteBuffer(layerBuf);
    if(phaseBuf) gl.deleteBuffer(phaseBuf);

    posBuf   = makeBuf(allPos, aPos);
    sizeBuf  = makeBuf(allSize, aSize);
    layerBuf = makeBuf(allLayer, aLayer);
    phaseBuf = makeBuf(allPhase, aPhase);
  }

  // ── Grid Lines (secondary layer) ────────────────────────────────────────
  const gridVS = `
    attribute vec2 aPos;
    uniform vec2 uRes;
    uniform float uTime;
    uniform vec2 uMouse;
    varying float vAlpha;
    void main(){
      float parallaxX = (uMouse.x - 0.5) * 15.0;
      float parallaxY = (uMouse.y - 0.5) * 10.0;
      vec2 p = aPos + vec2(parallaxX, parallaxY);
      p = p / uRes * 2.0 - 1.0;
      p.y *= -1.0;
      gl_Position = vec4(p, 0.0, 1.0);
      vAlpha = 0.035;
    }
  `;
  const gridFS = `
    precision mediump float;
    varying float vAlpha;
    void main(){ gl_FragColor = vec4(0.39, 0.7, 0.93, vAlpha); }
  `;

  let gridProg;
  let gridBuf;
  try{
    const gvs = compileShader(gridVS, gl.VERTEX_SHADER);
    const gfs = compileShader(gridFS, gl.FRAGMENT_SHADER);
    gridProg = gl.createProgram();
    gl.attachShader(gridProg, gvs);
    gl.attachShader(gridProg, gfs);
    gl.linkProgram(gridProg);
    if(!gl.getProgramParameter(gridProg, gl.LINK_STATUS)) gridProg = null;
  }catch(e){gridProg=null;}

  function initGrid(){
    if(!gridProg) return;
    const step = 60;
    const verts = [];
    for(let x=0; x<=W+step; x+=step){ verts.push(x,0, x,H); }
    for(let y=0; y<=H+step; y+=step){ verts.push(0,y, W,y); }
    if(gridBuf) gl.deleteBuffer(gridBuf);
    gridBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
  }

  // ── Render Loop ─────────────────────────────────────────────────────────
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  function render(){
    time += 0.016;
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Grid
    if(gridProg && gridBuf){
      gl.useProgram(gridProg);
      const gPos = gl.getAttribLocation(gridProg, 'aPos');
      gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf);
      gl.enableVertexAttribArray(gPos);
      gl.vertexAttribPointer(gPos, 2, gl.FLOAT, false, 0, 0);
      gl.uniform2f(gl.getUniformLocation(gridProg, 'uRes'), W, H);
      gl.uniform1f(gl.getUniformLocation(gridProg, 'uTime'), time);
      gl.uniform2f(gl.getUniformLocation(gridProg, 'uMouse'), mouseX, mouseY);
      const vCount = gl.getBufferParameter(gridBuf, gl.ARRAY_BUFFER) / 8;
      gl.drawArrays(gl.LINES, 0, vCount);
    }

    // Particles
    gl.useProgram(prog);
    gl.uniform1f(uTime, time);
    gl.uniform2f(uMouse, mouseX, mouseY);
    gl.uniform2f(uRes, W, H);

    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuf);
    gl.enableVertexAttribArray(aSize);
    gl.vertexAttribPointer(aSize, 1, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, layerBuf);
    gl.enableVertexAttribArray(aLayer);
    gl.vertexAttribPointer(aLayer, 1, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, phaseBuf);
    gl.enableVertexAttribArray(aPhase);
    gl.vertexAttribPointer(aPhase, 1, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.POINTS, 0, N);

    requestAnimationFrame(render);
  }

  resize();
  initGrid();
  render();
})();
