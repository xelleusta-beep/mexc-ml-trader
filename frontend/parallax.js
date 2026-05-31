/**
 * Sci-Fi WebGL Background — MEXC Trading System v2
 * Bilim-kurgu temali partikul agi + parallax + glow efektleri
 */
(function(){
  const canvas = document.createElement('canvas');
  canvas.id = 'webgl-bg';
  canvas.style.cssText = 'position:fixed;inset:0;z-index:0;width:100%;height:100%;pointer-events:none';
  document.body.prepend(canvas);

  const gl = canvas.getContext('webgl',{alpha:true,antialias:true,premultipliedAlpha:false});
  if(!gl){console.warn('WebGL yok');return;}

  let W=0,H=0,mx=0.5,my=0.5,t=0;
  const resize=()=>{W=canvas.width=innerWidth;H=canvas.height=innerHeight;gl.viewport(0,0,W,H);init()};
  addEventListener('resize',resize);
  addEventListener('mousemove',e=>{mx=e.clientX/W;my=e.clientY/H});

  // ── Particle Shader ──────────────────────────────────────────────────
  const pVS=`
    attribute vec2 aPos;
    attribute float aSize;
    attribute float aDepth;
    attribute float aPhase;
    uniform float uTime;
    uniform vec2 uMouse;
    uniform vec2 uRes;
    varying float vAlpha;
    varying float vPhase;
    varying float vDepth;
    void main(){
      float d=aDepth;
      vec2 parallax=(uMouse-0.5)*d*80.0;
      float sway=sin(uTime*0.4+aPhase*6.283)*12.0*d;
      float drift=cos(uTime*0.3+aPhase*3.141)*8.0*d;
      vec2 p=aPos+vec2(parallax.x+sway,parallax.y+drift);
      p=p/uRes*2.0-1.0;
      p.y*=-1.0;
      gl_Position=vec4(p,0.0,1.0);
      gl_PointSize=aSize*(1.0+d*0.8);
      vAlpha=0.3+d*0.5;
      vPhase=aPhase;
      vDepth=d;
    }
  `;
  const pFS=`
    precision mediump float;
    varying float vAlpha;
    varying float vPhase;
    varying float vDepth;
    uniform float uTime;
    void main(){
      vec2 pc=gl_PointCoord*2.0-1.0;
      float dist=length(pc);
      if(dist>1.0)discard;
      float glow=exp(-dist*2.0);
      vec3 c1=vec3(0.0,0.83,1.0);
      vec3 c2=vec3(0.486,0.227,0.929);
      vec3 c3=vec3(0.0,1.0,0.5);
      float t2=sin(uTime*0.2)*0.5+0.5;
      vec3 col=mix(mix(c1,c2,vPhase),c3,t2*0.3);
      float pulse=0.6+0.4*sin(uTime*0.6+vPhase*6.283);
      gl_FragColor=vec4(col,glow*vAlpha*pulse);
    }
  `;

  // ── Line Shader ──────────────────────────────────────────────────────
  const lVS=`
    attribute vec2 aPos;
    uniform vec2 uRes;
    uniform vec2 uMouse;
    uniform float uTime;
    varying float vAlpha;
    void main(){
      vec2 parallax=(uMouse-0.5)*20.0;
      vec2 p=(aPos+parallax)/uRes*2.0-1.0;
      p.y*=-1.0;
      gl_Position=vec4(p,0.0,1.0);
      vAlpha=0.08+0.04*sin(uTime*0.5);
    }
  `;
  const lFS=`
    precision mediump float;
    varying float vAlpha;
    void main(){
      gl_FragColor=vec4(0.0,0.6,1.0,vAlpha);
    }
  `;

  function compile(src,type){
    const s=gl.createShader(type);
    gl.shaderSource(s,src);
    gl.compileShader(s);
    if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)){console.error(gl.getShaderInfoLog(s));return null}
    return s;
  }

  function link(vs,fs){
    const p=gl.createProgram();
    gl.attachShader(p,vs);
    gl.attachShader(p,fs);
    gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS)){console.error(gl.getProgramInfoLog(p));return null}
    return p;
  }

  let pProg,lProg;
  let pBuf,sBuf,dBuf,phBuf,lBuf;
  let N=0;

  function init(){
    // Particle program
    const pvs=compile(pVS,gl.VERTEX_SHADER);
    const pfs=compile(pFS,gl.FRAGMENT_SHADER);
    pProg=link(pvs,pfs);

    // Line program
    const lvs=compile(lVS,gl.VERTEX_SHADER);
    const lfs=compile(lFS,gl.FRAGMENT_SHADER);
    lProg=link(lvs,lfs);

    // Particles — 3 katman
    const layers=[
      {n:200,sMin:1.5,sMax:3.5,dMin:0.05,dMax:0.25},
      {n:120,sMin:2.5,sMax:5.0,dMin:0.25,dMax:0.55},
      {n:60,sMin:4.0,sMax:8.0,dMin:0.55,dMax:1.0},
    ];

    let allP=[],allS=[],allD=[],allPh=[];
    layers.forEach(l=>{
      for(let i=0;i<l.n;i++){
        allP.push(Math.random()*W,Math.random()*H);
        allS.push(l.sMin+Math.random()*(l.sMax-l.sMin));
        allD.push(l.dMin+Math.random()*(l.dMax-l.dMin));
        allPh.push(Math.random());
      }
    });
    N=allP.length/2;

    function mkBuf(data){
      const b=gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER,b);
      gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.STATIC_DRAW);
      return b;
    }

    if(pBuf)gl.deleteBuffer(pBuf);
    if(sBuf)gl.deleteBuffer(sBuf);
    if(dBuf)gl.deleteBuffer(dBuf);
    if(phBuf)gl.deleteBuffer(phBuf);

    pBuf=mkBuf(allP);
    sBuf=mkBuf(allS);
    dBuf=mkBuf(allD);
    phBuf=mkBuf(allPh);

    // Grid lines
    const grid=[];
    const step=80;
    for(let x=0;x<=W+step;x+=step){grid.push(x,0,x,H)}
    for(let y=0;y<=H+step;y+=step){grid.push(0,y,W,y)}
    if(lBuf)gl.deleteBuffer(lBuf);
    lBuf=mkBuf(grid);
  }

  // ── Render ───────────────────────────────────────────────────────────
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);

  function frame(){
    t+=0.016;
    gl.clearColor(0.02,0.03,0.06,1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Grid
    if(lProg&&lBuf){
      gl.useProgram(lProg);
      const aPos=gl.getAttribLocation(lProg,'aPos');
      gl.bindBuffer(gl.ARRAY_BUFFER,lBuf);
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos,2,gl.FLOAT,false,0,0);
      gl.uniform2f(gl.getUniformLocation(lProg,'uRes'),W,H);
      gl.uniform2f(gl.getUniformLocation(lProg,'uMouse'),mx,my);
      gl.uniform1f(gl.getUniformLocation(lProg,'uTime'),t);
      const n=gl.getBufferParameter(lBuf,gl.ARRAY_BUFFER)/8;
      gl.drawArrays(gl.LINES,0,n);
    }

    // Particles
    if(pProg&&pBuf){
      gl.useProgram(pProg);
      gl.uniform1f(gl.getUniformLocation(pProg,'uTime'),t);
      gl.uniform2f(gl.getUniformLocation(pProg,'uMouse'),mx,my);
      gl.uniform2f(gl.getUniformLocation(pProg,'uRes'),W,H);

      const aPos=gl.getAttribLocation(pProg,'aPos');
      const aSize=gl.getAttribLocation(pProg,'aSize');
      const aDepth=gl.getAttribLocation(pProg,'aDepth');
      const aPhase=gl.getAttribLocation(pProg,'aPhase');

      gl.bindBuffer(gl.ARRAY_BUFFER,pBuf);
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos,2,gl.FLOAT,false,0,0);

      gl.bindBuffer(gl.ARRAY_BUFFER,sBuf);
      gl.enableVertexAttribArray(aSize);
      gl.vertexAttribPointer(aSize,1,gl.FLOAT,false,0,0);

      gl.bindBuffer(gl.ARRAY_BUFFER,dBuf);
      gl.enableVertexAttribArray(aDepth);
      gl.vertexAttribPointer(aDepth,1,gl.FLOAT,false,0,0);

      gl.bindBuffer(gl.ARRAY_BUFFER,phBuf);
      gl.enableVertexAttribArray(aPhase);
      gl.vertexAttribPointer(aPhase,1,gl.FLOAT,false,0,0);

      gl.drawArrays(gl.POINTS,0,N);
    }

    requestAnimationFrame(frame);
  }

  resize();
  frame();
})();
