/* ════════════════════════════════════════════════════
   A. BACKGROUND: Static elegant particle grid (gesture OFF)
════════════════════════════════════════════════════ */
(function(){
  const c=document.getElementById('bg-canvas');
  const ctx=c.getContext('2d');
  let W,H,pts=[];
  function resize(){ W=c.width=innerWidth; H=c.height=innerHeight; init(); }
  function init(){
    pts=[];
    const cols=Math.ceil(W/90), rows=Math.ceil(H/90);
    for(let r=0;r<=rows;r++) for(let col=0;col<=cols;col++){
      pts.push({ x:col*(W/cols)+(Math.random()-0.5)*18, y:r*(H/rows)+(Math.random()-0.5)*18, vx:(Math.random()-0.5)*0.15, vy:(Math.random()-0.5)*0.15, a:Math.random()*0.25+0.05 });
    }
  }
  function draw(){
    ctx.clearRect(0,0,W,H);
    // Draw connections
    for(let i=0;i<pts.length;i++){
      const p=pts[i];
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<-20||p.x>W+20) p.vx*=-1;
      if(p.y<-20||p.y>H+20) p.vy*=-1;
      for(let j=i+1;j<pts.length;j++){
        const q=pts[j];
        const d=Math.hypot(p.x-q.x,p.y-q.y);
        if(d<120){
          ctx.strokeStyle=`rgba(0,180,216,${(1-d/120)*0.07})`;
          ctx.lineWidth=0.5;
          ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y); ctx.stroke();
        }
      }
      ctx.beginPath(); ctx.arc(p.x,p.y,1.2,0,Math.PI*2);
      ctx.fillStyle=`rgba(0,180,216,${p.a})`; ctx.fill();
    }
    requestAnimationFrame(draw);
  }
  window.addEventListener('resize',resize);
  resize(); draw();
})();


/* ════════════════════════════════════════════════════
   B. THREE.JS MORPHING SPHERE (gesture ON)
════════════════════════════════════════════════════ */
let sphereScene, sphereCamera, sphereRenderer, sphereMesh, particleSystem;
let handNormX=0.5, handNormY=0.5; // normalized 0-1 hand position
let sphereActive=false;

function initSphere(){
  if(sphereScene) return; // already init
  const canvas=document.getElementById('sphere-canvas');
  sphereScene=new THREE.Scene();
  sphereCamera=new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 1000);
  sphereCamera.position.z=5;
  sphereRenderer=new THREE.WebGLRenderer({canvas, alpha:true, antialias:true});
  sphereRenderer.setSize(innerWidth, innerHeight);
  sphereRenderer.setPixelRatio(Math.min(devicePixelRatio,2));

  // ── Morphing sphere ──
  const geo=new THREE.IcosahedronGeometry(1.6, 5);
  const mat=new THREE.MeshPhongMaterial({
    color:0x00b4d8, wireframe:true, transparent:true, opacity:0.3
  });
  sphereMesh=new THREE.Mesh(geo,mat);
  sphereScene.add(sphereMesh);

  // Store original positions for morphing
  const posArr=geo.attributes.position.array;
  geo.userData.origPositions=new Float32Array(posArr);

  // ── Inner glowing sphere ──
  const innerGeo=new THREE.SphereGeometry(1.4, 32,32);
  const innerMat=new THREE.MeshPhongMaterial({color:0x001a2e, transparent:true, opacity:0.85});
  const innerMesh=new THREE.Mesh(innerGeo,innerMat);
  sphereScene.add(innerMesh);

  // ── Particles ──
  const pGeo=new THREE.BufferGeometry();
  const pCount=1800;
  const pPos=new Float32Array(pCount*3);
  const pVel=new Float32Array(pCount*3);
  for(let i=0;i<pCount;i++){
    const r=2.5+Math.random()*3;
    const theta=Math.random()*Math.PI*2;
    const phi=Math.acos(2*Math.random()-1);
    pPos[i*3]=r*Math.sin(phi)*Math.cos(theta);
    pPos[i*3+1]=r*Math.sin(phi)*Math.sin(theta);
    pPos[i*3+2]=r*Math.cos(phi);
    pVel[i*3]=(Math.random()-0.5)*0.004;
    pVel[i*3+1]=(Math.random()-0.5)*0.004;
    pVel[i*3+2]=(Math.random()-0.5)*0.004;
  }
  pGeo.setAttribute('position',new THREE.BufferAttribute(pPos,3));
  pGeo.userData.vel=pVel;
  particleSystem=new THREE.Points(pGeo, new THREE.PointsMaterial({color:0x00f2ff,size:0.04,transparent:true,opacity:0.6}));
  sphereScene.add(particleSystem);

  // ── Lights ──
  sphereScene.add(new THREE.AmbientLight(0x001a2e,2));
  const pLight=new THREE.PointLight(0x00f2ff,2,20);
  pLight.position.set(3,3,3);
  sphereScene.add(pLight);
  const pLight2=new THREE.PointLight(0x7c3aed,1.5,20);
  pLight2.position.set(-3,-2,2);
  sphereScene.add(pLight2);

  window.addEventListener('resize',()=>{
    sphereCamera.aspect=innerWidth/innerHeight;
    sphereCamera.updateProjectionMatrix();
    sphereRenderer.setSize(innerWidth,innerHeight);
  });

  animateSphere();
}

function animateSphere(){
  requestAnimationFrame(animateSphere);
  if(!sphereActive) return;

  const t=Date.now()*0.001;
  const geo=sphereMesh.geometry;
  const pos=geo.attributes.position.array;
  const orig=geo.userData.origPositions;

  // Hand controls sphere distortion
  const offsetX=(handNormX-0.5)*2.2;
  const offsetY=(handNormY-0.5)*-2.2;
  const dist=Math.hypot(offsetX,offsetY);

  for(let i=0;i<pos.length;i+=3){
    const ox=orig[i], oy=orig[i+1], oz=orig[i+2];
    const nx=ox/1.6, ny=oy/1.6, nz=oz/1.6; // unit normal
    // Morph: push vertices toward hand direction
    const dot=nx*offsetX+ny*offsetY;
    const warp=Math.sin(t*1.2+i*0.15)*0.12+(dot*0.35)*Math.max(0,1-dist*0.5);
    pos[i]  =ox+nx*warp;
    pos[i+1]=oy+ny*warp;
    pos[i+2]=oz+nz*Math.sin(t+i*0.08)*0.1;
  }
  geo.attributes.position.needsUpdate=true;

  // Rotate sphere gently, hand tilts it
  sphereMesh.rotation.y=t*0.25+offsetX*0.3;
  sphereMesh.rotation.x=t*0.12+offsetY*0.2;

  // Animate particles — drift + attract slightly to hand direction
  const pPos=particleSystem.geometry.attributes.position.array;
  const pVel=particleSystem.geometry.userData.vel;
  const attract=new THREE.Vector3(offsetX*0.8,offsetY*0.8,0);
  for(let i=0;i<pPos.length;i+=3){
    pPos[i]  +=pVel[i]  +(attract.x-pPos[i]  )*0.0008;
    pPos[i+1]+=pVel[i+1]+(attract.y-pPos[i+1])*0.0008;
    pPos[i+2]+=pVel[i+2]+(attract.z-pPos[i+2])*0.0004;
    // Keep in bounds
    const r=Math.hypot(pPos[i],pPos[i+1],pPos[i+2]);
    if(r>6||r<1.8){pVel[i]*=-1;pVel[i+1]*=-1;pVel[i+2]*=-1;}
  }
  particleSystem.geometry.attributes.position.needsUpdate=true;
  particleSystem.rotation.y=t*0.05;

  sphereRenderer.render(sphereScene,sphereCamera);
}


/* ════════════════════════════════════════════════════
   C. GESTURE TOGGLE
════════════════════════════════════════════════════ */
let gestureOn=false;
let handsInited=false;
let cameraInited=false;

function toggleGesture(){
  gestureOn=!gestureOn;
  const btn=document.getElementById('gesture-toggle');
  const lbl=document.getElementById('toggle-label');
  const sphereC=document.getElementById('sphere-canvas');
  const normalBg=document.getElementById('normal-bg');
  const bgCanvas=document.getElementById('bg-canvas');
  const skeletonC=document.getElementById('hand-skeleton');

  if(gestureOn){
    btn.classList.add('active');
    lbl.textContent='GESTURE ON';
    // Switch background
    normalBg.classList.add('hidden-bg');
    bgCanvas.style.opacity='0.2';
    sphereActive=true;
    sphereC.classList.add('active');
    skeletonC.style.opacity='1';
    // Init Three sphere + MediaPipe
    initSphere();
    if(!handsInited) initGestureEngine();
  } else {
    btn.classList.remove('active');
    lbl.textContent='GESTURE OFF';
    normalBg.classList.remove('hidden-bg');
    bgCanvas.style.opacity='1';
    sphereActive=false;
    sphereC.classList.remove('active');
    skeletonC.style.opacity='0';
    document.getElementById('gesture-cursor').style.opacity='0';
    document.getElementById('cursor2').style.opacity='0';
    // Reset grab counter
    airGrabCount=0; updateGrabBadge(0);
    setLog('IDLE','rgba(100,140,180,0.55)');
  }
}


/* ════════════════════════════════════════════════════
   D. FOLDER BROWSER (preserved + improved)
════════════════════════════════════════════════════ */
let folders=[], looseFiles=[], selected=[];

function triggerFolder(){ document.getElementById('folderInput').click(); }
function triggerFiles(){ document.getElementById('filesInput').click(); }

function onFolderLoad(rawFiles){
  const map={};
  let done=0,total=rawFiles.length;
  if(!total) return;
  Array.from(rawFiles).forEach(f=>{
    const parts=f.webkitRelativePath.split('/');
    const folder=parts.length>1?parts[parts.length-2]:'ROOT';
    if(!map[folder]) map[folder]=[];
    const r=new FileReader();
    r.onload=e=>{
      map[folder].push({name:f.name,url:e.target.result,obj:f});
      if(++done===total) buildFolders(map);
    };
    r.readAsDataURL(f);
  });
}
function buildFolders(map){
  Object.entries(map).forEach(([name,files])=>{
    const ex=folders.find(f=>f.name===name);
    if(ex) ex.files=ex.files.concat(files);
    else folders.push({name,open:false,files});
  });
  renderTree();
}
function onFilesLoad(rawFiles){
  let done=0,total=rawFiles.length;
  if(!total) return;
  Array.from(rawFiles).forEach(f=>{
    const r=new FileReader();
    r.onload=e=>{ looseFiles.push({name:f.name,url:e.target.result,obj:f}); if(++done===total) renderTree(); };
    r.readAsDataURL(f);
  });
}
function clearAll(){ folders=[]; looseFiles=[]; selected=[]; renderTree(); renderChips(); }
function selAll(){
  selected=[];
  folders.forEach(fo=>fo.files.forEach(f=>selected.push(f)));
  looseFiles.forEach(f=>selected.push(f));
  renderTree(); renderChips();
}
function toggleFile(pool,idx){
  const file=pool[idx];
  const si=selected.findIndex(s=>s.name===file.name&&s.url===file.url);
  if(si>=0) selected.splice(si,1); else selected.push(file);
  renderTree(); renderChips();
} 
function isSelected(file){ return selected.some(s=>s.name===file.name&&s.url===file.url); }
function getExt(name){ return (name.split('.').pop()||'').toUpperCase(); }

function makeFileRow(file,pool,fi,fj){
  const sel=isSelected(file);
  const row=document.createElement('div');
  row.className='file-row g-clickable'+(sel?' is-sel':'');
  row.dataset.dragPool=pool; row.dataset.dragFi=fi;
  if(fj!==undefined) row.dataset.dragFj=fj;
  row.innerHTML=`<img src="${file.url}" style="width:18px;height:18px;border-radius:3px;object-fit:cover;flex-shrink:0;border:1px solid rgba(0,242,255,.18);">
    <span class="row-label">${file.name}</span><span class="row-badge">${getExt(file.name)}</span>
    ${sel?'<i class="fas fa-check" style="color:var(--cyan);font-size:9px;margin-left:2px;"></i>':''}
    <svg class="d-ring" viewBox="0 0 20 20"><circle cx="10" cy="10" r="7"/></svg>`;
  row.addEventListener('click',()=>{ toggleFile(pool==='loose'?looseFiles:folders[fi].files, pool==='loose'?fi:fj); });
  return row;
}
function renderTree(){
  const tree=document.getElementById('folder-tree');
  while(tree.firstChild) tree.removeChild(tree.firstChild);
  const hasContent=folders.length>0||looseFiles.length>0;
  if(!hasContent){ const emp=document.getElementById('tree-empty'); emp.style.display='block'; tree.appendChild(emp); return; }
  folders.forEach((fo,fi)=>{
    const fr=document.createElement('div');
    fr.className='folder-row g-clickable'+(fo.open?' is-open':'');
    fr.innerHTML=`<i class="fas fa-chevron-right"></i><i class="fas fa-folder${fo.open?'-open':''}"></i>
      <span class="row-label">${fo.name}</span><span class="row-badge">${fo.files.length}</span>
      <svg class="d-ring" viewBox="0 0 20 20"><circle cx="10" cy="10" r="7"/></svg>`;
    fr.addEventListener('click',()=>{ fo.open=!fo.open; renderTree(); });
    tree.appendChild(fr);
    if(fo.open){
      const wrap=document.createElement('div'); wrap.className='children-wrap open';
      fo.files.forEach((file,fj)=>wrap.appendChild(makeFileRow(file,'folder',fi,fj)));
      tree.appendChild(wrap);
    }
  });
  if(looseFiles.length>0){
    const hdr=document.createElement('div'); hdr.className='folder-row'; hdr.style.cssText='cursor:default;margin-top:6px;';
    hdr.innerHTML=`<i class="fas fa-layer-group" style="color:#38bdf8;font-size:12px;"></i>
      <span class="row-label" style="color:rgba(0,242,255,.6);">LOOSE FILES</span><span class="row-badge">${looseFiles.length}</span>`;
    tree.appendChild(hdr);
    looseFiles.forEach((file,fi)=>tree.appendChild(makeFileRow(file,'loose',fi,undefined)));
  }
}
function renderChips(){
  document.getElementById('sel-count').textContent=selected.length;
  document.getElementById('send-btn').style.display=selected.length>0?'block':'none';
  document.getElementById('sel-list').innerHTML=selected.map((f,i)=>`
    <div class="sel-chip"><span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:150px;">${f.name}</span>
    <span class="chip-x" onclick="removeSel(${i})"><i class="fas fa-times"></i></span></div>`).join('');
}
function removeSel(i){ selected.splice(i,1); renderTree(); renderChips(); }
function sendToQueue(){
  if(!selected.length) return;
  selected.forEach(f=>addFileToQueue(f.obj,f.url));
  showTip(`${selected.length} SCAN(S) → QUEUE`,1800);
}


/* ════════════════════════════════════════════════════
   E. QUEUE
════════════════════════════════════════════════════ */
let queue=[];
function addToQueue(files){ Array.from(files).forEach(f=>{ const r=new FileReader(); r.onload=e=>addFileToQueue(f,e.target.result); r.readAsDataURL(f); }); }
function addFileToQueue(fileObj,url){
  if(queue.find(q=>q.name===fileObj.name&&q.size===fileObj.size)) return;
  queue.push(fileObj);
  const list=document.getElementById('file-queue');
  const p=list.querySelector('p'); if(p) p.remove();
  const item=document.createElement('div');
  item.style.cssText='display:flex;align-items:center;justify-content:space-between;padding:6px 8px;background:rgba(0,242,255,0.05);border-left:2px solid rgba(0,242,255,0.5);border-radius:0 6px 6px 0;margin-bottom:3px;font-size:9px;';
  item.innerHTML=`<div style="display:flex;align-items:center;gap:7px;overflow:hidden;">
    <img src="${url}" style="width:16px;height:16px;border-radius:3px;object-fit:cover;flex-shrink:0;">
    <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${fileObj.name}</span></div>
    <span style="color:rgba(0,242,255,0.45);flex-shrink:0;margin-left:4px;font-family:'Space Mono';">#${queue.length}</span>`;
  list.appendChild(item);
  document.getElementById('process-btn').style.display='block';
}


/* ════════════════════════════════════════════════════
   F. GESTURE ENGINE v5.0
════════════════════════════════════════════════════ */
const ALPHA=0.16, ALPHA_FAST=0.32;
const PINCH_CLOSE=0.047, PINCH_OPEN=0.085, DWELL=1400;
const SCROLL_MULT=5.5, FRICTION=0.88;

let smX=-1,smY=-1,smGX=-1,smGY=-1,sm2X=-1,sm2Y=-1;
let isPinching=false,lastAction=0,pinchConfirm=0;
let dwellEl=null,dwellStart=0;
let dragMode=false,dragFile=null,dragSource=null;
let lastScrollY=null,scrollVel=0,scrollRafId=null;

// ── AIR GRAB OFF COUNTER ──
let airGrabCount=0;       // how many air grabs done so far (resets after 2s gap)
let grabWasOnFile=false;  // was the last pinch on a file/button (not empty air)?
let grabResetTimer=null;  // timeout to reset count if grabs are too far apart

function updateGrabBadge(n){
  const badge=document.getElementById('grab-off-badge');
  ['pip1','pip2','pip3'].forEach((id,i)=>{
    document.getElementById(id).classList.toggle('filled', i<n);
  });
  if(n>0){ badge.classList.add('show'); clearTimeout(updateGrabBadge._t); updateGrabBadge._t=setTimeout(()=>badge.classList.remove('show'),2200); }
  else { badge.classList.remove('show'); }
}

// Viewer zoom state
let viewerOpen=false,viewerZoom=1,viewerImg=null;
let viewerCtx=null,viewerW=0,viewerH=0;
let lastPinchDist=null; // for viewer pinch zoom
let zoomPinching=false;

const resultModal=document.getElementById('result-modal');
const cursor2El=document.getElementById('cursor2');
const scrollInd=document.getElementById('scroll-indicator');
const scrollThumb=document.getElementById('scroll-thumb');
const cursor=document.getElementById('gesture-cursor');

function ema(prev,next,a){ return prev<0?next:prev+a*(next-prev); }
function updateScrollThumb(){ scrollThumb.style.top=(resultModal.scrollTop/Math.max(1,resultModal.scrollHeight-resultModal.clientHeight)*68)+'%'; }
function momentumScroll(){ if(Math.abs(scrollVel)<0.4){scrollVel=0;return;} resultModal.scrollTop+=scrollVel; scrollVel*=FRICTION; updateScrollThumb(); scrollRafId=requestAnimationFrame(momentumScroll); }
function stopMomentum(){ cancelAnimationFrame(scrollRafId); scrollVel=0; }
function showScrollIndicator(){ scrollInd.classList.add('visible'); }
function hideScrollIndicator(){ scrollInd.classList.remove('visible'); cursor2El.style.opacity='0'; }

const tooltip=document.getElementById('g-tooltip');
function showTip(msg,ms=1100){ tooltip.textContent=msg; tooltip.classList.add('show'); clearTimeout(showTip._t); showTip._t=setTimeout(()=>tooltip.classList.remove('show'),ms); }
function setLog(txt,col='rgba(100,140,180,0.55)'){ const el=document.getElementById('log-val'); el.style.color=col; el.textContent=txt; }

function showGhost(file,x,y){
  document.getElementById('ghost-img').src=file.url;
  document.getElementById('ghost-label').textContent=file.name.length>22?file.name.substring(0,20)+'…':file.name;
  const g=document.getElementById('drag-ghost'); g.classList.add('active');
  smGX=x;smGY=y; g.style.left=(x+18)+'px'; g.style.top=(y-20)+'px';
}
function moveGhost(x,y){
  smGX=ema(smGX,x,ALPHA_FAST); smGY=ema(smGY,y,ALPHA_FAST);
  const g=document.getElementById('drag-ghost'); g.style.left=(smGX+18)+'px'; g.style.top=(smGY-20)+'px';
}
function hideGhost(){ document.getElementById('drag-ghost').classList.remove('active'); smGX=-1;smGY=-1; }
const dropZone=document.getElementById('drop-upload-zone');
function overDropZone(x,y){ const r=dropZone.getBoundingClientRect(); return x>=r.left&&x<=r.right&&y>=r.top&&y<=r.bottom; }
function commitDrop(file){ dropZone.classList.add('drag-over'); setTimeout(()=>dropZone.classList.remove('drag-over'),600); addFileToQueue(file.obj,file.url); showTip('DROP → QUEUED ✓',1200); setLog('DROPPED','#4ade80'); }
function cancelDrag(){ if(dragSource) dragSource.classList.remove('is-grabbed'); dragMode=false;dragFile=null;dragSource=null; hideGhost(); dropZone.classList.remove('drag-over'); }

// Draw skeleton on hand-skeleton canvas
const skelCanvas=document.getElementById('hand-skeleton');
const skelCtx=skelCanvas.getContext('2d');
function resizeSkel(){ skelCanvas.width=innerWidth; skelCanvas.height=innerHeight; }
resizeSkel(); window.addEventListener('resize',resizeSkel);
const CONNECTIONS=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]];
function drawSkeleton(landmarks){
  skelCtx.clearRect(0,0,innerWidth,innerHeight);
  if(!gestureOn||!landmarks) return;
  CONNECTIONS.forEach(([a,b])=>{
    const pa=landmarks[a],pb=landmarks[b];
    const x1=(1-pa.x)*innerWidth,y1=pa.y*innerHeight;
    const x2=(1-pb.x)*innerWidth,y2=pb.y*innerHeight;
    skelCtx.strokeStyle='rgba(0,242,255,0.18)'; skelCtx.lineWidth=1;
    skelCtx.beginPath(); skelCtx.moveTo(x1,y1); skelCtx.lineTo(x2,y2); skelCtx.stroke();
  });
  landmarks.forEach((lm,i)=>{
    const x=(1-lm.x)*innerWidth,y=lm.y*innerHeight;
    skelCtx.beginPath(); skelCtx.arc(x,y,i===8||i===4?4:2,0,Math.PI*2);
    skelCtx.fillStyle=i===8||i===4?'rgba(0,242,255,0.7)':'rgba(0,242,255,0.3)'; skelCtx.fill();
  });
}

function initGestureEngine(){
  handsInited=true;
  const videoEl=document.getElementById('webcam');
  const hands=new Hands({locateFile:f=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`});
  hands.setOptions({maxNumHands:2,modelComplexity:1,minDetectionConfidence:0.75,minTrackingConfidence:0.75});

  hands.onResults(results=>{
    if(!gestureOn){ skelCtx.clearRect(0,0,innerWidth,innerHeight); return; }
    const allHands=results.multiHandLandmarks||[];
    const modalOpen=resultModal.classList.contains('open');

    // Update hand position for sphere morphing
    if(allHands.length>0){
      handNormX=1-allHands[0][8].x;
      handNormY=allHands[0][8].y;
    }
    drawSkeleton(allHands[0]||null);

    // Sensor status
    const sensorSt=document.getElementById('sensor-stat');

    // ═══ VIEWER PINCH ZOOM (priority when viewer open) ═══
    if(viewerOpen && allHands.length>=1){
      const h=allHands[0];
      const x=(1-h[8].x)*innerWidth, y=h[8].y*innerHeight;
      smX=ema(smX,x,ALPHA); smY=ema(smY,y,ALPHA);
      cursor.style.opacity='1'; cursor.style.transform=`translate3d(${smX-14}px,${smY-14}px,0)`;

      const pinch=Math.hypot(h[8].x-h[4].x,h[8].y-h[4].y);
      const pxDist=pinch*innerWidth; // pixel distance approximation

      if(allHands.length===2){
        // Two-hand spread = zoom out, squeeze = zoom in
        const h2=allHands[1];
        const p2x=(1-h2[8].x)*innerWidth,p2y=h2[8].y*innerHeight;
        sm2X=ema(sm2X,p2x,ALPHA); sm2Y=ema(sm2Y,p2y,ALPHA);
        cursor2El.style.opacity='1'; cursor2El.style.transform=`translate3d(${sm2X-10}px,${sm2Y-10}px,0)`;
        const d=Math.hypot(smX-sm2X,smY-sm2Y);
        if(lastPinchDist!==null){
          const dz=(d-lastPinchDist)*0.008;
          applyZoom(viewerZoom+dz);
        }
        lastPinchDist=d;
      } else {
        lastPinchDist=null; cursor2El.style.opacity='0';
        // Single pinch: close=zoom in, open=zoom out
        const isPinchClosed=pinch<PINCH_CLOSE;
        const isPinchOpen=pinch>PINCH_OPEN;
        if(isPinchClosed){ applyZoom(viewerZoom+0.025); setLog('ZOOM IN ＋','#4ade80'); }
        if(isPinchOpen && zoomPinching){ zoomPinching=false; }
        if(isPinchClosed) zoomPinching=true;
        // Open hand sustained = zoom out
        if(pinch>0.12) applyZoom(viewerZoom-0.012);
      }

      // Palm close viewer
      const palm=h[8].y<h[6].y&&h[12].y<h[10].y&&h[16].y<h[14].y&&h[20].y<h[18].y;
      if(palm&&Date.now()-lastAction>3000){ lastAction=Date.now(); closeViewer(); }
      return;
    } else if(viewerOpen){
      lastPinchDist=null; cursor2El.style.opacity='0';
    }

    // ═══ TWO-FINGER SCROLL ═══
    if(allHands.length===2&&modalOpen){
      stopMomentum();
      const h0=allHands[0],h1=allHands[1];
      const r0x=(1-h0[8].x)*innerWidth,r0y=h0[8].y*innerHeight;
      const r1x=(1-h1[8].x)*innerWidth,r1y=h1[8].y*innerHeight;
      smX=ema(smX,r0x,ALPHA); smY=ema(smY,r0y,ALPHA);
      sm2X=ema(sm2X,r1x,ALPHA); sm2Y=ema(sm2Y,r1y,ALPHA);
      cursor.style.opacity='1'; cursor.style.transform=`translate3d(${smX-14}px,${smY-14}px,0)`;
      cursor2El.style.opacity='1'; cursor2El.style.transform=`translate3d(${sm2X-10}px,${sm2Y-10}px,0)`;
      const avgY=(smY+sm2Y)/2;
      if(lastScrollY!==null){ const d=(avgY-lastScrollY)*SCROLL_MULT; if(Math.abs(d)>0.8){ scrollVel=d; resultModal.scrollTop+=d; updateScrollThumb(); setLog(d>0?'SCROLL ↓':'SCROLL ↑','#fbbf24'); showScrollIndicator(); } }
      lastScrollY=avgY;
      sensorSt.textContent='2-HAND'; sensorSt.style.cssText='background:rgba(245,158,11,0.12);color:#fbbf24;border:1px solid rgba(245,158,11,0.3);';
      return;
    } else {
      if(lastScrollY!==null&&Math.abs(scrollVel)>1) scrollRafId=requestAnimationFrame(momentumScroll);
      lastScrollY=null; sm2X=-1; sm2Y=-1; cursor2El.style.opacity='0';
      if(!modalOpen) hideScrollIndicator();
    }

    if(!allHands.length){
      cursor.style.opacity='0';
      sensorSt.textContent='OFFLINE'; sensorSt.style.cssText='background:rgba(239,68,68,0.12);color:#f87171;border:1px solid rgba(239,68,68,0.3);';
      setLog('NO HAND','rgba(100,140,180,0.55)'); dwellEl=null;
      if(dragMode) cancelDrag(); smX=-1; smY=-1; return;
    }

    const hand=allHands[0];
    const rawX=(1-hand[8].x)*innerWidth, rawY=hand[8].y*innerHeight;
    smX=ema(smX,rawX,ALPHA); smY=ema(smY,rawY,ALPHA);
    const x=smX,y=smY;
    cursor.style.opacity='1'; cursor.style.transform=`translate3d(${x-14}px,${y-14}px,0)`;
    tooltip.style.left=(x+22)+'px'; tooltip.style.top=(y-14)+'px';
    sensorSt.textContent='ONLINE'; sensorSt.style.cssText='background:rgba(74,222,128,0.12);color:#4ade80;border:1px solid rgba(74,222,128,0.3);';

    // Palm reset — NO page reload, just visual flash + log
    const palm=hand[8].y<hand[6].y&&hand[12].y<hand[10].y&&hand[16].y<hand[14].y&&hand[20].y<hand[18].y;
    if(palm&&Date.now()-lastAction>3000){
      if(dragMode) cancelDrag();
      lastAction=Date.now();
      setLog('PALM FLASH','#fbbf24'); showTip('✋ PALM — GRAB×3 TO TURN OFF',1800);
      cursor.style.boxShadow='0 0 40px #ffcc00';
      setTimeout(()=>{ cursor.style.boxShadow='0 0 12px var(--cyan)'; },600);
    }

    const pinch=Math.hypot(hand[8].x-hand[4].x,hand[8].y-hand[4].y);
    const isPinchClosed=pinch<PINCH_CLOSE;
    const isPinchOpen=pinch>PINCH_OPEN;

    // Drag mode (must come before pinch section)
    if(dragMode){
      if(!isPinchOpen){
        moveGhost(x,y); cursor.style.borderColor='#f59e0b'; cursor.style.boxShadow='0 0 22px #f59e0b'; setLog('DRAGGING','#f59e0b');
        if(overDropZone(x,y)){ dropZone.classList.add('drag-over'); showTip('RELEASE TO DROP',400); } else dropZone.classList.remove('drag-over');
      } else {
        if(overDropZone(x,y)) commitDrop(dragFile); else { showTip('CANCELLED',700); setLog('IDLE','rgba(100,140,180,0.55)'); }
        cancelDrag(); lastAction=Date.now();
      }
      return;
    }

    // ── AIR GRAB COUNTER — 3 grabs in empty air = turn gesture OFF ──
    // A "grab" = pinch on empty space (not on a file row or button), then release
    if(isPinchClosed){
      cursor.style.borderColor='#bc13fe'; cursor.style.boxShadow='0 0 22px #bc13fe'; setLog('PINCH','#bc13fe');
      pinchConfirm++;
      if(pinchConfirm>=2&&!isPinching&&Date.now()-lastAction>640){
        isPinching=true; lastAction=Date.now(); dwellEl=null;
        const el=document.elementFromPoint(x,y);
        if(el){
          const fr=el.closest('.file-row[data-drag-pool]');
          const cl=el.closest('.g-clickable,button,[onclick]');
          if(fr){
            // File grab → drag as before
            const pool=fr.dataset.dragPool,fi=parseInt(fr.dataset.dragFi),fj=fr.dataset.dragFj!==undefined&&fr.dataset.dragFj!==''?parseInt(fr.dataset.dragFj):null;
            let file=null;
            if(pool==='folder'&&fj!==null&&folders[fi]) file=folders[fi].files[fj];
            else if(pool==='loose') file=looseFiles[fi];
            if(file){ dragMode=true;dragFile=file;dragSource=fr; fr.classList.add('is-grabbed'); showGhost(file,x,y); setLog('GRABBED','#f59e0b'); showTip('DRAG TO DROP ZONE',1000); }
            grabWasOnFile=true;
          } else if(cl){
            // Button click
            cl.click(); showTip('PINCH SELECT',700);
            grabWasOnFile=true;
          } else {
            // AIR GRAB — on empty space
            grabWasOnFile=false;
          }
        } else {
          grabWasOnFile=false;
        }
      }
    } else {
      pinchConfirm=0;
      if(isPinchOpen){
        // On release — if the pinch was an air grab, count it
        if(isPinching&&!grabWasOnFile&&!dragMode){
          airGrabCount++;
          updateGrabBadge(airGrabCount);
          setLog(`AIR GRAB ${airGrabCount}/3`,'#f59e0b');
          if(airGrabCount>=3){
            // Triple air grab → turn gesture OFF
            airGrabCount=0; updateGrabBadge(0);
            showTip('GESTURE OFF ✓',1200);
            setLog('OFF','rgba(100,140,180,0.55)');
            setTimeout(()=>toggleGesture(),400);
          }
        }
        isPinching=false;
        cursor.style.borderColor='var(--cyan)'; cursor.style.boxShadow='0 0 12px var(--cyan)';
        // Reset grab count if too much time passes between grabs (2 seconds)
        clearTimeout(grabResetTimer);
        if(airGrabCount>0&&airGrabCount<3){
          grabResetTimer=setTimeout(()=>{ airGrabCount=0; updateGrabBadge(0); },2000);
        }
      }
    }
    // A "grab" = pinch on empty space (not on a file row or button), then release
    if(isPinchClosed){
      cursor.style.borderColor='#bc13fe'; cursor.style.boxShadow='0 0 22px #bc13fe'; setLog('PINCH','#bc13fe');
      pinchConfirm++;
      if(pinchConfirm>=2&&!isPinching&&Date.now()-lastAction>640){
        isPinching=true; lastAction=Date.now(); dwellEl=null;
        const el=document.elementFromPoint(x,y);
        if(el){
          const fr=el.closest('.file-row[data-drag-pool]');
          const cl=el.closest('.g-clickable,button,[onclick]');
          if(fr){
            // File grab → drag as before
            const pool=fr.dataset.dragPool,fi=parseInt(fr.dataset.dragFi),fj=fr.dataset.dragFj!==undefined&&fr.dataset.dragFj!==''?parseInt(fr.dataset.dragFj):null;
            let file=null;
            if(pool==='folder'&&fj!==null&&folders[fi]) file=folders[fi].files[fj];
            else if(pool==='loose') file=looseFiles[fi];
            if(file){ dragMode=true;dragFile=file;dragSource=fr; fr.classList.add('is-grabbed'); showGhost(file,x,y); setLog('GRABBED','#f59e0b'); showTip('DRAG TO DROP ZONE',1000); }
            grabWasOnFile=true;
          } else if(cl){
            // Button click
            cl.click(); showTip('PINCH SELECT',700);
            grabWasOnFile=true;
          } else {
            // AIR GRAB — on empty space
            grabWasOnFile=false;
          }
        } else {
          grabWasOnFile=false;
        }
      }
    } else {
      pinchConfirm=0;
      if(isPinchOpen){
        // On release — if the pinch was an air grab, count it
        if(isPinching&&!grabWasOnFile&&!dragMode){
          airGrabCount++;
          updateGrabBadge(airGrabCount);
          setLog(`AIR GRAB ${airGrabCount}/3`,'#f59e0b');
          if(airGrabCount>=3){
            // Triple air grab → turn gesture OFF
            airGrabCount=0; updateGrabBadge(0);
            showTip('GESTURE OFF ✓',1200);
            setLog('OFF','rgba(100,140,180,0.55)');
            setTimeout(()=>toggleGesture(),400);
          }
        }
        isPinching=false;
        cursor.style.borderColor='var(--cyan)'; cursor.style.boxShadow='0 0 12px var(--cyan)';
        // Reset grab count if too much time passes between grabs (2 seconds)
        clearTimeout(grabResetTimer);
        if(airGrabCount>0&&airGrabCount<3){
          grabResetTimer=setTimeout(()=>{ airGrabCount=0; updateGrabBadge(0); },2000);
        }
      }
    }

    // Dwell hover
    if(isPinchOpen&&!dragMode){
      const el=document.elementFromPoint(x,y);
      const target=el?el.closest('.g-clickable'):null;
      document.querySelectorAll('.g-hover').forEach(e=>e.classList.remove('g-hover'));
      if(target){
        target.classList.add('g-hover'); setLog('HOVER','var(--cyan)');
        if(dwellEl!==target){ dwellEl=target; dwellStart=Date.now(); }
        else{
          const prog=Math.min((Date.now()-dwellStart)/DWELL,1);
          const ring=target.querySelector('.d-ring');
          if(ring){ ring.classList.add('active'); const c=ring.querySelector('circle'); if(c) c.style.strokeDashoffset=44-(44*prog); }
          if(Date.now()-dwellStart>=DWELL&&Date.now()-lastAction>DWELL+50){ lastAction=Date.now(); dwellEl=null; target.click(); showTip('DWELL ✓',700); setLog('DWELL CLICK','#4ade80'); }
        }
      } else {
        dwellEl=null;
        document.querySelectorAll('.d-ring').forEach(r=>{ r.classList.remove('active'); const c=r.querySelector('circle'); if(c) c.style.strokeDashoffset='44'; });
        setLog('IDLE','rgba(100,140,180,0.55)');
      }
    }
  });

  new Camera(document.getElementById('webcam'),{onFrame:async()=>await hands.send({image:document.getElementById('webcam')}),width:640,height:480}).start();
}


/* ════════════════════════════════════════════════════
   G. FULLSCREEN VIEWER + WATER RIPPLE ZOOM
════════════════════════════════════════════════════ */
const viewerCanvas=document.getElementById('viewer-canvas');
const vCtx=viewerCanvas.getContext('2d');

// Ripple buffer
let rippleBuffer1=null,rippleBuffer2=null,rippleW=0,rippleH=0;
let rippleAnimId=null;

function initRippleBuffers(w,h){
  rippleW=w; rippleH=h;
  rippleBuffer1=new Float32Array(w*h);
  rippleBuffer2=new Float32Array(w*h);
}

function addRipple(x,y,strength=400){
  const ix=Math.floor(x),iy=Math.floor(y);
  for(let dy=-4;dy<=4;dy++) for(let dx=-4;dx<=4;dx++){
    const nx=ix+dx,ny=iy+dy;
    if(nx>=0&&nx<rippleW&&ny>=0&&ny<rippleH){
      const d=Math.hypot(dx,dy);
      if(d<5) rippleBuffer1[ny*rippleW+nx]+=strength*(1-d/5);
    }
  }
}

function stepRipple(){
  for(let y=1;y<rippleH-1;y++){
    for(let x=1;x<rippleW-1;x++){
      const i=y*rippleW+x;
      rippleBuffer2[i]=(
        (rippleBuffer1[(y-1)*rippleW+x]+rippleBuffer1[(y+1)*rippleW+x]+
         rippleBuffer1[y*rippleW+(x-1)]+rippleBuffer1[y*rippleW+(x+1)])/2
      )-rippleBuffer2[i];
      rippleBuffer2[i]*=0.94;
    }
  }
  const tmp=rippleBuffer1; rippleBuffer1=rippleBuffer2; rippleBuffer2=tmp;
}

let viewerOffscreenCanvas=null, viewerOffCtx=null, viewerOffImg=null;

function renderViewer(){
  if(!viewerOpen||!viewerImg) return;
  const cw=viewerCanvas.width, ch=viewerCanvas.height;

  // Draw base image at current zoom into offscreen
  viewerOffCtx.clearRect(0,0,cw,ch);
  const sw=cw/viewerZoom, sh=ch/viewerZoom;
  const sx=(viewerW/2)-sw/2, sy=(viewerH/2)-sh/2;
  viewerOffCtx.drawImage(viewerImg, Math.max(0,sx),Math.max(0,sy), Math.min(sw,viewerW),Math.min(sh,viewerH), 0,0,cw,ch);

  // Apply ripple displacement
  stepRipple();
  const src=viewerOffCtx.getImageData(0,0,cw,ch);
  const dst=vCtx.createImageData(cw,ch);
  for(let y=1;y<ch-1;y++){
    for(let x=1;x<cw-1;x++){
      const i=y*cw+x;
      let dx=Math.floor(rippleBuffer1[i]-rippleBuffer1[i+1]);
      let dy=Math.floor(rippleBuffer1[i]-rippleBuffer1[i+cw]);
      dx=Math.max(-8,Math.min(8,dx));
      dy=Math.max(-8,Math.min(8,dy));
      const sx2=Math.max(0,Math.min(cw-1,x+dx));
      const sy2=Math.max(0,Math.min(ch-1,y+dy));
      const si=(sy2*cw+sx2)*4, di=i*4;
      dst.data[di]=src.data[si]; dst.data[di+1]=src.data[si+1];
      dst.data[di+2]=src.data[si+2]; dst.data[di+3]=src.data[si+3];
    }
  }
  vCtx.putImageData(dst,0,0);

  // Cyan vignette overlay
  const grad=vCtx.createRadialGradient(cw/2,ch/2,cw*0.28,cw/2,ch/2,cw*0.7);
  grad.addColorStop(0,'rgba(0,0,0,0)');
  grad.addColorStop(1,'rgba(0,10,30,0.55)');
  vCtx.fillStyle=grad; vCtx.fillRect(0,0,cw,ch);

  rippleAnimId=requestAnimationFrame(renderViewer);
}

function openViewer(imgSrc){
  viewerOpen=true;
  viewerZoom=1.0; lastPinchDist=null;
  const overlay=document.getElementById('viewer-overlay');
  overlay.classList.add('open');

  const img=new Image();
  img.onload=()=>{
    viewerImg=img; viewerW=img.naturalWidth; viewerH=img.naturalHeight;
    // Size canvas to window
    const maxW=Math.min(innerWidth*0.88,900);
    const maxH=Math.min(innerHeight*0.82,700);
    const ar=viewerW/viewerH;
    let cw,ch;
    if(ar>maxW/maxH){ cw=maxW; ch=maxW/ar; } else { ch=maxH; cw=maxH*ar; }
    viewerCanvas.width=Math.floor(cw); viewerCanvas.height=Math.floor(ch);
    viewerOffscreenCanvas=document.createElement('canvas');
    viewerOffscreenCanvas.width=Math.floor(cw); viewerOffscreenCanvas.height=Math.floor(ch);
    viewerOffCtx=viewerOffscreenCanvas.getContext('2d');
    initRippleBuffers(Math.floor(cw),Math.floor(ch));
    cancelAnimationFrame(rippleAnimId);
    renderViewer();
    // Initial ripple burst
    addRipple(cw/2,ch/2,600);
  };
  img.src=imgSrc;
  updateZoomBadge();
}

function closeViewer(){
  viewerOpen=false;
  cancelAnimationFrame(rippleAnimId);
  document.getElementById('viewer-overlay').classList.remove('open');
  cursor.style.opacity='0'; viewerImg=null; lastPinchDist=null;
}

function applyZoom(z){
  viewerZoom=Math.max(0.5,Math.min(6,z));
  updateZoomBadge();
  if(viewerCanvas.width>0) addRipple(viewerCanvas.width/2,viewerCanvas.height/2,Math.abs(z-viewerZoom+viewerZoom)*60+80);
}
function updateZoomBadge(){ document.getElementById('zoom-level-badge').textContent=`ZOOM ${viewerZoom.toFixed(1)}×`; }

// Mouse click on canvas = ripple
viewerCanvas.addEventListener('click',e=>{
  const r=viewerCanvas.getBoundingClientRect();
  addRipple(e.clientX-r.left,e.clientY-r.top,350);
});

/* ════════════════════════════════════════════════════
   H. PROCESS + TYPEWRITER + RESULT MODAL
════════════════════════════════════════════════════ */
function typeWriterEffect(id, text, speed=90){
  let i=0;
  const el=document.getElementById(id);
  (function t(){ if(i<text.length){ el.innerHTML+=text.charAt(i++); setTimeout(t,speed); } })();
}

function closeResultModal(){
  resultModal.classList.remove('open');
  hideScrollIndicator();
}

// Tumor type → color mapping
const TUMOR_COLORS = {
  'Glioma'     : '#f87171',   // red
  'Meningioma' : '#fbbf24',   // amber
  'Pituitary'  : '#a78bfa',   // violet
  'No Tumor'   : '#4ade80',   // green
};

// Confidence → severity label
function getSeverityLabel(tumorType, confidence){
  if(tumorType==='No Tumor') return { label:'CLEAR', color:'#4ade80' };
  if(confidence>=90)         return { label:'HIGH CERTAINTY', color:'#f87171' };
  if(confidence>=75)         return { label:'MODERATE', color:'#fbbf24' };
  return                            { label:'LOW CERTAINTY', color:'#94a3b8' };
}

async function processBatch(){
  if(!queue.length){
    showTip('NO SCANS IN QUEUE', 1400);
    return;
  }

  const area=document.getElementById('result-content');
  resultModal.classList.add('open');
  resultModal.scrollTop=0;
  showScrollIndicator();
  updateScrollThumb();

  // ── Loading screen ──
  area.innerHTML=`
    <div style="text-align:center;padding:80px 0;">
      <div style="font-family:'Orbitron';font-size:14px;letter-spacing:0.4em;
                  color:rgba(0,242,255,0.6);animation:fadeIn 0.8s ease;">
        MAPPING NEURAL ACTIVATION...
      </div>
      <div style="margin-top:10px;font-family:'Space Mono';font-size:10px;
                  color:rgba(0,242,255,0.35);letter-spacing:0.12em;">
        MODEL: ${selectedModel.toUpperCase()}
      </div>
      <div style="margin-top:20px;width:200px;height:2px;
                  background:rgba(0,242,255,0.1);
                  margin-left:auto;margin-right:auto;
                  border-radius:1px;overflow:hidden;">
        <div style="width:40%;height:100%;background:var(--cyan);
                    animation:slide 1.2s ease infinite;"></div>
      </div>
      <style>
        @keyframes slide{
          0%  { transform:translateX(-100%) }
          100%{ transform:translateX(350%)  }
        }
      </style>
    </div>`;

  let loadedCount=0;

  for(let i=0; i<queue.length; i++){
    const fd=new FormData();
    fd.append('file', queue[i]);

    try{
      // ── API call with selected model ──
      const r=await fetch(
  `https://mri-brain-tumour-detection-with-gesture-voqw.onrender.com/predict?model_key=${selectedModel}`,
  { method:'POST', body:fd }
);

      if(!r.ok){
        const err=await r.json();
        throw new Error(err.detail || `HTTP ${r.status}`);
      }

      const d=await r.json();

      if(loadedCount===0) area.innerHTML='';
      loadedCount++;

      document.getElementById('result-count').textContent=`${loadedCount} SCAN(S)`;
      setLog(`PROCESSED ${loadedCount}/${queue.length}`, '#4ade80');

      const objUrl     = URL.createObjectURL(queue[i]);
      const heatmapSrc = `data:image/jpeg;base64,${d.heatmap}`;
      const rid        = `res-${i}`;
      const tumorColor = TUMOR_COLORS[d.tumor_type] || 'var(--cyan)';
      const severity   = getSeverityLabel(d.tumor_type, d.confidence);

      // All-scores bar HTML
      const scoresHtml = Object.entries(d.all_scores||{}).map(([name, pct])=>`
        <div style="margin-bottom:6px;">
          <div style="display:flex;justify-content:space-between;
                      font-family:'Space Mono';font-size:9px;
                      color:rgba(180,200,220,0.7);margin-bottom:3px;">
            <span>${name}</span><span>${pct}%</span>
          </div>
          <div style="height:3px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;">
            <div style="height:100%;width:${pct}%;
                        background:${TUMOR_COLORS[name]||'var(--cyan)'};
                        border-radius:2px;transition:width 0.8s ease;"></div>
          </div>
        </div>`).join('');

      const row=document.createElement('div');
      row.className='result-row fade-in';
      row.innerHTML=`
        <!-- ── Images ── -->
        <div style="display:flex;gap:16px;flex-shrink:0;">

          <div style="text-align:center;">
            <div style="font-family:'Orbitron';font-size:8px;letter-spacing:0.15em;
                        color:rgba(100,140,180,0.6);margin-bottom:8px;">01_INPUT</div>
            <div class="scan-thumb" onclick="openViewer('${objUrl}')">
              <img src="${objUrl}"
                   style="width:180px;height:180px;object-fit:cover;filter:grayscale(0.4);">
              <div class="thumb-label">INPUT SCAN</div>
              <div class="thumb-zoom"><i class="fas fa-expand-alt"></i></div>
            </div>
          </div>

          <div style="text-align:center;">
            <div style="font-family:'Orbitron';font-size:8px;letter-spacing:0.15em;
                        color:rgba(0,242,255,0.6);margin-bottom:8px;">02_GRAD_CAM</div>
            <div class="scan-thumb" onclick="openViewer('${heatmapSrc}')">
              <img src="${heatmapSrc}"
                   style="width:180px;height:180px;object-fit:cover;">
              <div class="thumb-label">ACTIVATION MAP</div>
              <div class="thumb-zoom"><i class="fas fa-expand-alt"></i></div>
            </div>
          </div>

        </div>

        <!-- ── Result data ── -->
        <div style="flex:1;min-width:0;">

          <!-- Tumor type typewriter -->
          <h3 id="${rid}"
              style="font-family:'Orbitron';font-size:48px;font-weight:900;
                     color:${tumorColor};line-height:1;letter-spacing:-1px;
                     margin-bottom:12px;text-shadow:0 0 30px ${tumorColor}55;"
              class="typewriter-text"></h3>

          <!-- Severity + model tag -->
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:18px;flex-wrap:wrap;">
            <span style="font-family:'Orbitron';font-size:9px;letter-spacing:0.14em;
                         padding:4px 12px;border-radius:5px;
                         background:${severity.color}18;
                         border:1px solid ${severity.color}55;
                         color:${severity.color};">
              ${severity.label}
            </span>
            <span style="font-family:'Orbitron';font-size:9px;letter-spacing:0.12em;
                         padding:4px 12px;border-radius:5px;
                         background:rgba(0,242,255,0.06);
                         border:1px solid rgba(0,242,255,0.2);
                         color:rgba(0,242,255,0.55);">
              ${d.model_used || selectedModel.toUpperCase()}
            </span>
          </div>

          <!-- Confidence + info -->
          <div style="display:flex;align-items:center;gap:16px;
                      flex-wrap:wrap;margin-bottom:20px;">
            <div style="padding:12px 24px;
                        background:${tumorColor}18;
                        border:1px solid ${tumorColor}55;
                        border-radius:10px;
                        font-family:'Orbitron';font-size:22px;
                        color:${tumorColor};">
              ${d.confidence}%
            </div>
            <div style="font-size:10px;color:rgba(100,140,180,0.7);
                        font-family:'Space Mono';line-height:1.8;">
              CERTAINTY INDEX<br>
              GRAD-CAM VERIFIED<br>
              DEVICE: ${d.device||'N/A'}
            </div>
          </div>

          <!-- All-class probability bars -->
          <div style="margin-bottom:16px;">
            <div style="font-family:'Orbitron';font-size:8px;letter-spacing:0.16em;
                        color:rgba(0,242,255,0.4);margin-bottom:10px;">
              CLASS PROBABILITY DISTRIBUTION
            </div>
            ${scoresHtml}
          </div>

          <!-- Hint -->
          <div style="font-size:9px;font-family:'Space Mono';
                      color:rgba(0,242,255,0.3);letter-spacing:0.1em;line-height:1.9;">
            → CLICK IMAGE TO OPEN FULLSCREEN VIEWER<br>
            → USE GESTURE PINCH TO ZOOM WITH WATER EFFECT
          </div>

        </div>`;

      area.appendChild(row);
      typeWriterEffect(rid, d.tumor_type);

    } catch(e){
      if(loadedCount===0) area.innerHTML='';
      loadedCount++;

      const errRow=document.createElement('div');
      errRow.style.cssText='text-align:center;padding:60px;';
      errRow.innerHTML=`
        <div style="font-family:'Space Mono';color:rgba(239,68,68,0.8);font-size:12px;line-height:2;">
          <i class="fas fa-exclamation-triangle" style="font-size:28px;display:block;margin-bottom:16px;opacity:0.6;"></i>
          ERROR — ${e.message || 'No response from 127.0.0.1:8000'}
        </div>
        <div style="font-size:10px;color:rgba(100,140,180,0.4);margin-top:10px;font-family:'Space Mono';">
          Scan ${i+1}: ${queue[i].name}<br>
          Make sure FastAPI backend is running
        </div>`;
      area.appendChild(errRow);
    }
  }

  // ── Final summary bar ──
  if(loadedCount>0){
    const summary=document.createElement('div');
    summary.style.cssText=`
      text-align:center;padding:32px 0 16px;
      font-family:'Orbitron';font-size:9px;letter-spacing:0.2em;
      color:rgba(0,242,255,0.35);border-top:1px solid rgba(0,242,255,0.08);
      margin-top:32px;`;
    summary.innerHTML=`
      SWEEP COMPLETE — ${loadedCount} SCAN(S) PROCESSED
      <span style="margin-left:16px;color:rgba(0,242,255,0.2);">|</span>
      <span style="margin-left:16px;">MODEL: ${selectedModel.toUpperCase()}</span>`;
    area.appendChild(summary);
  }

  setLog('SWEEP COMPLETE', '#4ade80');
}
/* ════════════════════════════════════════════════════
   I. MODEL SELECTOR — gesture-friendly custom dropdown
════════════════════════════════════════════════════ */
let selectedModel      = 'cnn';
let modelDropdownOpen  = false;

const MODEL_INFO = {
  cnn: {
    label   : '⬡ CUSTOM CNN — 92.00% TEST ACC',
    arch    : '4-block CNN + BatchNorm',
    params  : '~8.8M trainable',
    val_acc : '96.70%',
  },
  resnet18: {
    label   : '⬡ RESNET18 PRETRAINED — 94.81% TEST ACC',
    arch    : 'ResNet18 pretrained backbone',
    params  : '~11.7M trainable',
    val_acc : '97.32%',
  }
};

function toggleModelDropdown(){
  modelDropdownOpen = !modelDropdownOpen;
  const panel  = document.getElementById('model-options');
  const arrow  = document.getElementById('model-arrow');
  const trigger = document.getElementById('model-trigger');

  if(modelDropdownOpen){
    panel.classList.add('open');
    arrow.textContent           = '▴';
    trigger.style.borderColor   = 'var(--cyan)';
    trigger.style.boxShadow     = '0 0 16px rgba(0,242,255,0.2)';
  } else {
    panel.classList.remove('open');
    arrow.textContent           = '▾';
    trigger.style.borderColor   = 'rgba(0,242,255,0.28)';
    trigger.style.boxShadow     = 'none';
  }
}

function selectModel(key){
  selectedModel = key;
  const info    = MODEL_INFO[key];

  // Update trigger label
  document.getElementById('model-trigger-label').textContent = info.label;

  // Update info badge
  document.getElementById('model-badge').innerHTML = `
    <span style="color:rgba(0,242,255,0.5);">&gt; arch:</span> ${info.arch}<br>
    <span style="color:rgba(0,242,255,0.5);">&gt; params:</span> ${info.params}<br>
    <span style="color:rgba(0,242,255,0.5);">&gt; val_acc:</span>
    <span style="color:#4ade80;">${info.val_acc}</span>
  `;

  // Mark active option
  document.querySelectorAll('.model-opt').forEach(btn => {
    btn.classList.toggle('is-active', btn.dataset.key === key);
  });

  // Close dropdown
  modelDropdownOpen = false;
  const panel   = document.getElementById('model-options');
  const arrow   = document.getElementById('model-arrow');
  const trigger = document.getElementById('model-trigger');
  panel.classList.remove('open');
  arrow.textContent         = '▾';
  trigger.style.borderColor = 'rgba(0,242,255,0.28)';
  trigger.style.boxShadow   = 'none';

  // Pulse trigger to confirm
  trigger.style.borderColor = 'var(--cyan)';
  trigger.style.boxShadow   = '0 0 20px rgba(0,242,255,0.35)';
  setTimeout(() => {
    trigger.style.borderColor = 'rgba(0,242,255,0.28)';
    trigger.style.boxShadow   = 'none';
  }, 700);

  setLog(`MODEL → ${key.toUpperCase()}`, 'var(--cyan)');
  showTip(`MODEL: ${key.toUpperCase()} SELECTED ✓`, 1400);
}

// Close dropdown if gesture cursor moves far away from it
function closeDropdownIfAway(x, y){
  if(!modelDropdownOpen) return;
  const wrap = document.getElementById('model-dropdown-wrap');
  if(!wrap) return;
  const r = wrap.getBoundingClientRect();
  const padded = 60;
  if(x < r.left-padded || x > r.right+padded ||
     y < r.top-padded  || y > r.bottom+padded){
    modelDropdownOpen = false;
    document.getElementById('model-options').classList.remove('open');
    document.getElementById('model-arrow').textContent          = '▾';
    document.getElementById('model-trigger').style.borderColor  = 'rgba(0,242,255,0.28)';
    document.getElementById('model-trigger').style.boxShadow    = 'none';
  }
}

// Close on outside mouse click (non-gesture users)
document.addEventListener('click', e => {
  if(!e.target.closest('#model-dropdown-wrap')) {
    modelDropdownOpen = false;
    const panel = document.getElementById('model-options');
    if(panel) panel.classList.remove('open');
    const arrow = document.getElementById('model-arrow');
    if(arrow) arrow.textContent = '▾';
  }
});

async function syncModelsFromBackend(){
  try{
    const r    = await fetch('https://mri-brain-tumour-detection-with-gesture-voqw.onrender.com/models');
    const data = await r.json();

    data.models.forEach(m => {
      const btn = document.querySelector(`.model-opt[data-key="${m.key}"]`);
      if(btn && !m.loaded){
        btn.style.opacity = '0.45';
        btn.querySelector('div').textContent += ' [OFFLINE]';
      }
    });

    if(data.default){
      selectModel(data.default);
    }
  } catch(e){
    console.warn('Backend not reachable for model sync:', e);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  setTimeout(syncModelsFromBackend, 800);
});