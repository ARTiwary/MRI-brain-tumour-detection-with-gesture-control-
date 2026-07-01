/* ════════════════════════════════════════════════════
   A. BACKGROUND — subtle professional particle grid
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
      pts.push({
        x:col*(W/cols)+(Math.random()-0.5)*18,
        y:r*(H/rows)+(Math.random()-0.5)*18,
        vx:(Math.random()-0.5)*0.12,
        vy:(Math.random()-0.5)*0.12,
        a:Math.random()*0.18+0.04
      });
    }
  }
  function draw(){
    ctx.clearRect(0,0,W,H);
    for(let i=0;i<pts.length;i++){
      const p=pts[i];
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<-20||p.x>W+20) p.vx*=-1;
      if(p.y<-20||p.y>H+20) p.vy*=-1;
      for(let j=i+1;j<pts.length;j++){
        const q=pts[j];
        const d=Math.hypot(p.x-q.x,p.y-q.y);
        if(d<120){
          ctx.strokeStyle=`rgba(100,116,139,${(1-d/120)*0.1})`;
          ctx.lineWidth=0.5;
          ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y); ctx.stroke();
        }
      }
      ctx.beginPath(); ctx.arc(p.x,p.y,1.4,0,Math.PI*2);
      ctx.fillStyle=`rgba(100,116,139,${p.a})`; ctx.fill();
    }
    requestAnimationFrame(draw);
  }
  window.addEventListener('resize',resize);
  resize(); draw();
})();


/* ════════════════════════════════════════════════════
   B. THREE.JS MORPHING SPHERE (gesture ON)
════════════════════════════════════════════════════ */
let sphereScene,sphereCamera,sphereRenderer,sphereMesh,particleSystem;
let handNormX=0.5,handNormY=0.5;
let sphereActive=false;

function initSphere(){
  if(sphereScene) return;
  const canvas=document.getElementById('sphere-canvas');
  sphereScene=new THREE.Scene();
  sphereCamera=new THREE.PerspectiveCamera(60,innerWidth/innerHeight,0.1,1000);
  sphereCamera.position.z=5;
  sphereRenderer=new THREE.WebGLRenderer({canvas,alpha:true,antialias:true});
  sphereRenderer.setSize(innerWidth,innerHeight);
  sphereRenderer.setPixelRatio(Math.min(devicePixelRatio,2));

  const geo=new THREE.IcosahedronGeometry(1.6,5);
  const mat=new THREE.MeshPhongMaterial({color:0x2563eb,wireframe:true,transparent:true,opacity:0.25});
  sphereMesh=new THREE.Mesh(geo,mat);
  sphereScene.add(sphereMesh);
  const posArr=geo.attributes.position.array;
  geo.userData.origPositions=new Float32Array(posArr);

  const innerGeo=new THREE.SphereGeometry(1.4,32,32);
  const innerMat=new THREE.MeshPhongMaterial({color:0xeff6ff,transparent:true,opacity:0.6});
  sphereScene.add(new THREE.Mesh(innerGeo,innerMat));

  const pGeo=new THREE.BufferGeometry();
  const pCount=1400;
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
  particleSystem=new THREE.Points(pGeo,new THREE.PointsMaterial({color:0x3b82f6,size:0.04,transparent:true,opacity:0.5}));
  sphereScene.add(particleSystem);

  sphereScene.add(new THREE.AmbientLight(0xeff6ff,2));
  const pLight=new THREE.PointLight(0x2563eb,2,20);
  pLight.position.set(3,3,3);
  sphereScene.add(pLight);
  const pLight2=new THREE.PointLight(0x1e3a5f,1.5,20);
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
  const offsetX=(handNormX-0.5)*2.2;
  const offsetY=(handNormY-0.5)*-2.2;
  const dist=Math.hypot(offsetX,offsetY);
  for(let i=0;i<pos.length;i+=3){
    const ox=orig[i],oy=orig[i+1],oz=orig[i+2];
    const nx=ox/1.6,ny=oy/1.6,nz=oz/1.6;
    const dot=nx*offsetX+ny*offsetY;
    const warp=Math.sin(t*1.2+i*0.15)*0.12+(dot*0.35)*Math.max(0,1-dist*0.5);
    pos[i]=ox+nx*warp; pos[i+1]=oy+ny*warp; pos[i+2]=oz+nz*Math.sin(t+i*0.08)*0.1;
  }
  geo.attributes.position.needsUpdate=true;
  sphereMesh.rotation.y=t*0.25+offsetX*0.3;
  sphereMesh.rotation.x=t*0.12+offsetY*0.2;
  const pPos=particleSystem.geometry.attributes.position.array;
  const pVel=particleSystem.geometry.userData.vel;
  const attract=new THREE.Vector3(offsetX*0.8,offsetY*0.8,0);
  for(let i=0;i<pPos.length;i+=3){
    pPos[i]  +=pVel[i]  +(attract.x-pPos[i]  )*0.0008;
    pPos[i+1]+=pVel[i+1]+(attract.y-pPos[i+1])*0.0008;
    pPos[i+2]+=pVel[i+2]+(attract.z-pPos[i+2])*0.0004;
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
let gestureOn=false,handsInited=false;

function toggleGesture(){
  gestureOn=!gestureOn;
  const btn=document.getElementById('gesture-toggle');
  const lbl=document.getElementById('toggle-label');
  const sphereC=document.getElementById('sphere-canvas');
  const normalBg=document.getElementById('normal-bg');
  const bgCanvas=document.getElementById('bg-canvas');
  const skeletonC=document.getElementById('hand-skeleton');
  if(gestureOn){
    btn.classList.add('active'); lbl.textContent='Gesture on';
    normalBg.classList.add('hidden-bg'); bgCanvas.style.opacity='0.1';
    sphereActive=true; sphereC.classList.add('active'); skeletonC.style.opacity='1';
    initSphere();
    if(!handsInited) initGestureEngine();
  } else {
    btn.classList.remove('active'); lbl.textContent='Gesture off';
    normalBg.classList.remove('hidden-bg'); bgCanvas.style.opacity='0.3';
    sphereActive=false; sphereC.classList.remove('active'); skeletonC.style.opacity='0';
    document.getElementById('gesture-cursor').style.opacity='0';
    document.getElementById('cursor2').style.opacity='0';
    airGrabCount=0; updateGrabBadge(0);
    setLog('Idle');
  }
}


/* ════════════════════════════════════════════════════
   D. FOLDER BROWSER
════════════════════════════════════════════════════ */
let folders=[],looseFiles=[],selected=[];

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
    r.onload=e=>{ map[folder].push({name:f.name,url:e.target.result,obj:f}); if(++done===total) buildFolders(map); };
    r.readAsDataURL(f);
  });
}
function buildFolders(map){
  Object.entries(map).forEach(([name,files])=>{
    const ex=folders.find(f=>f.name===name);
    if(ex) ex.files=ex.files.concat(files); else folders.push({name,open:false,files});
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
  row.innerHTML=`<img src="${file.url}" style="width:18px;height:18px;border-radius:3px;object-fit:cover;flex-shrink:0;border:1px solid var(--border);">
    <span class="row-label">${file.name}</span><span class="row-badge">${getExt(file.name)}</span>
    ${sel?'<i class="fas fa-check" style="color:var(--navy-mid);font-size:9px;margin-left:2px;"></i>':''}
    <svg class="d-ring" viewBox="0 0 20 20"><circle cx="10" cy="10" r="7"/></svg>`;
  row.addEventListener('click',()=>{ toggleFile(pool==='loose'?looseFiles:folders[fi].files,pool==='loose'?fi:fj); });
  return row;
}
function renderTree(){
  const tree=document.getElementById('folder-tree');
  while(tree.firstChild) tree.removeChild(tree.firstChild);
  const hasContent=folders.length>0||looseFiles.length>0;
  if(!hasContent){
    const emp=document.getElementById('tree-empty');
    if(emp){ emp.style.display='block'; tree.appendChild(emp); }
    return;
  }
  folders.forEach((fo,fi)=>{
    const fr=document.createElement('div');
    fr.className='folder-row g-clickable'+(fo.open?' is-open':'');
    fr.innerHTML=`<i class="fas fa-chevron-right"></i><i class="fas fa-folder${fo.open?'-open':''}"></i>
      <span class="row-label">${fo.name}</span><span class="row-badge">${fo.files.length}</span>
      <svg class="d-ring" viewBox="0 0 20 20"><circle cx="10" cy="10" r="7"/></svg>`;
    fr.addEventListener('click',()=>{ fo.open=!fo.open; renderTree(); });
    tree.appendChild(fr);
    if(fo.open){
      const wrap=document.createElement('div');
      fo.files.forEach((file,fj)=>wrap.appendChild(makeFileRow(file,'folder',fi,fj)));
      tree.appendChild(wrap);
    }
  });
  if(looseFiles.length>0){
    const hdr=document.createElement('div');
    hdr.className='folder-row';
    hdr.style.cssText='cursor:default;margin-top:6px;';
    hdr.innerHTML=`<i class="fas fa-layer-group" style="color:var(--navy-mid);font-size:12px;"></i>
      <span class="row-label" style="color:var(--navy-mid);font-size:11px;font-weight:600;">Loose files</span>
      <span class="row-badge">${looseFiles.length}</span>`;
    tree.appendChild(hdr);
    looseFiles.forEach((file,fi)=>tree.appendChild(makeFileRow(file,'loose',fi,undefined)));
  }
}
function renderChips(){
  document.getElementById('sel-count').textContent=selected.length;
  document.getElementById('send-btn').style.display=selected.length>0?'block':'none';
  document.getElementById('sel-list').innerHTML=selected.map((f,i)=>`
    <div class="sel-chip">
      <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:150px;font-size:10px;">${f.name}</span>
      <span class="chip-x" onclick="removeSel(${i})"><i class="fas fa-times"></i></span>
    </div>`).join('');
}
function removeSel(i){ selected.splice(i,1); renderTree(); renderChips(); }
function sendToQueue(){
  if(!selected.length) return;
  selected.forEach(f=>addFileToQueue(f.obj,f.url));
  showTip(`${selected.length} scan(s) added to queue`);
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
  item.className='queue-item';
  item.innerHTML=`<img src="${url}" style="width:20px;height:20px;border-radius:4px;object-fit:cover;flex-shrink:0;border:1px solid var(--border);">
    <span class="qi-name">${fileObj.name}</span>
    <span style="color:var(--text-4);font-size:10px;font-family:'JetBrains Mono';flex-shrink:0;">#${queue.length}</span>`;
  list.appendChild(item);
  document.getElementById('process-btn').style.display='block';
  const sensor=document.getElementById('sensor-stat');
  sensor.textContent='Ready'; sensor.className='status-pill status-online';
}


/* ════════════════════════════════════════════════════
   F. BRAIN MRI FILTER
   Rejects images where all three models give very low
   max-confidence on all tumor classes (suggests the
   image is likely not a brain MRI at all). The filter
   runs after the API call using the returned scores,
   so it never requires a separate classifier.
════════════════════════════════════════════════════ */
const CONFIDENCE_FLOOR = 30; // below this on ALL classes = suspicious
function looksLikeNonBrainImage(allScores){
  if(!allScores) return false;
  const vals=Object.values(allScores);
  const maxConf=Math.max(...vals);
  return maxConf < CONFIDENCE_FLOOR;
}


/* ════════════════════════════════════════════════════
   G. GESTURE ENGINE
════════════════════════════════════════════════════ */
const ALPHA=0.16,ALPHA_FAST=0.32;
const PINCH_CLOSE=0.047,PINCH_OPEN=0.085,DWELL=1400;
const SCROLL_MULT=5.5,FRICTION=0.88;

let smX=-1,smY=-1,smGX=-1,smGY=-1,sm2X=-1,sm2Y=-1;
let isPinching=false,lastAction=0,pinchConfirm=0;
let dwellEl=null,dwellStart=0;
let dragMode=false,dragFile=null,dragSource=null;
let lastScrollY=null,scrollVel=0,scrollRafId=null;
let airGrabCount=0,grabWasOnFile=false,grabResetTimer=null;

function updateGrabBadge(n){
  const badge=document.getElementById('grab-off-badge');
  ['pip1','pip2','pip3'].forEach((id,i)=>{ document.getElementById(id).classList.toggle('filled',i<n); });
  if(n>0){ badge.classList.add('show'); clearTimeout(updateGrabBadge._t); updateGrabBadge._t=setTimeout(()=>badge.classList.remove('show'),2200); }
  else badge.classList.remove('show');
}

let viewerOpen=false,viewerZoom=1,viewerImg=null;
let lastPinchDist=null,zoomPinching=false;

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
function showTip(msg,ms=1200){ tooltip.textContent=msg; tooltip.classList.add('show'); clearTimeout(showTip._t); showTip._t=setTimeout(()=>tooltip.classList.remove('show'),ms); }
function setLog(txt,col='var(--text-4)'){ const el=document.getElementById('log-val'); el.style.color=col; el.textContent=txt; }

function showGhost(file,x,y){
  document.getElementById('ghost-img').src=file.url;
  document.getElementById('ghost-label').textContent=file.name.length>22?file.name.substring(0,20)+'…':file.name;
  const g=document.getElementById('drag-ghost'); g.classList.add('active');
  smGX=x; smGY=y; g.style.left=(x+18)+'px'; g.style.top=(y-20)+'px';
}
function moveGhost(x,y){
  smGX=ema(smGX,x,ALPHA_FAST); smGY=ema(smGY,y,ALPHA_FAST);
  const g=document.getElementById('drag-ghost'); g.style.left=(smGX+18)+'px'; g.style.top=(smGY-20)+'px';
}
function hideGhost(){ document.getElementById('drag-ghost').classList.remove('active'); smGX=-1; smGY=-1; }
const dropZone=document.getElementById('drop-upload-zone');
function overDropZone(x,y){ const r=dropZone.getBoundingClientRect(); return x>=r.left&&x<=r.right&&y>=r.top&&y<=r.bottom; }
function commitDrop(file){ dropZone.classList.add('drag-over'); setTimeout(()=>dropZone.classList.remove('drag-over'),600); addFileToQueue(file.obj,file.url); showTip('Dropped — added to queue'); setLog('Dropped','var(--success)'); }
function cancelDrag(){ if(dragSource) dragSource.classList.remove('is-grabbed'); dragMode=false; dragFile=null; dragSource=null; hideGhost(); dropZone.classList.remove('drag-over'); }

/* hand skeleton overlay */
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
    skelCtx.strokeStyle='rgba(37,99,235,0.2)'; skelCtx.lineWidth=1;
    skelCtx.beginPath(); skelCtx.moveTo(x1,y1); skelCtx.lineTo(x2,y2); skelCtx.stroke();
  });
  landmarks.forEach((lm,i)=>{
    const x=(1-lm.x)*innerWidth,y=lm.y*innerHeight;
    skelCtx.beginPath(); skelCtx.arc(x,y,i===8||i===4?4:2,0,Math.PI*2);
    skelCtx.fillStyle=i===8||i===4?'rgba(37,99,235,0.7)':'rgba(37,99,235,0.3)'; skelCtx.fill();
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
    if(allHands.length>0){ handNormX=1-allHands[0][8].x; handNormY=allHands[0][8].y; }
    drawSkeleton(allHands[0]||null);

    const sensorSt=document.getElementById('sensor-stat');

    /* ─ viewer pinch zoom ─ */
    if(viewerOpen&&allHands.length>=1){
      const h=allHands[0];
      const x=(1-h[8].x)*innerWidth,y=h[8].y*innerHeight;
      smX=ema(smX,x,ALPHA); smY=ema(smY,y,ALPHA);
      cursor.style.opacity='1'; cursor.style.transform=`translate3d(${smX-14}px,${smY-14}px,0)`;
      const pinch=Math.hypot(h[8].x-h[4].x,h[8].y-h[4].y);
      if(allHands.length===2){
        const h2=allHands[1];
        const p2x=(1-h2[8].x)*innerWidth,p2y=h2[8].y*innerHeight;
        sm2X=ema(sm2X,p2x,ALPHA); sm2Y=ema(sm2Y,p2y,ALPHA);
        cursor2El.style.opacity='1'; cursor2El.style.transform=`translate3d(${sm2X-10}px,${sm2Y-10}px,0)`;
        const d=Math.hypot(smX-sm2X,smY-sm2Y);
        if(lastPinchDist!==null) applyZoom(viewerZoom+(d-lastPinchDist)*0.008);
        lastPinchDist=d;
      } else {
        lastPinchDist=null; cursor2El.style.opacity='0';
        if(pinch<PINCH_CLOSE) applyZoom(viewerZoom+0.025);
        if(pinch>0.12) applyZoom(viewerZoom-0.012);
      }
      const palm=h[8].y<h[6].y&&h[12].y<h[10].y&&h[16].y<h[14].y&&h[20].y<h[18].y;
      if(palm&&Date.now()-lastAction>3000){ lastAction=Date.now(); closeViewer(); }
      return;
    } else if(viewerOpen){ lastPinchDist=null; cursor2El.style.opacity='0'; }

    /* ─ two-hand scroll ─ */
    if(allHands.length===2&&modalOpen){
      stopMomentum();
      const h0=allHands[0],h1=allHands[1];
      smX=ema(smX,(1-h0[8].x)*innerWidth,ALPHA); smY=ema(smY,h0[8].y*innerHeight,ALPHA);
      sm2X=ema(sm2X,(1-h1[8].x)*innerWidth,ALPHA); sm2Y=ema(sm2Y,h1[8].y*innerHeight,ALPHA);
      cursor.style.opacity='1'; cursor.style.transform=`translate3d(${smX-14}px,${smY-14}px,0)`;
      cursor2El.style.opacity='1'; cursor2El.style.transform=`translate3d(${sm2X-10}px,${sm2Y-10}px,0)`;
      const avgY=(smY+sm2Y)/2;
      if(lastScrollY!==null){ const d=(avgY-lastScrollY)*SCROLL_MULT; if(Math.abs(d)>0.8){ scrollVel=d; resultModal.scrollTop+=d; updateScrollThumb(); setLog(d>0?'Scrolling ↓':'Scrolling ↑','var(--warning)'); showScrollIndicator(); } }
      lastScrollY=avgY;
      sensorSt.textContent='2 hands'; sensorSt.className='status-pill status-online';
      return;
    } else {
      if(lastScrollY!==null&&Math.abs(scrollVel)>1) scrollRafId=requestAnimationFrame(momentumScroll);
      lastScrollY=null; sm2X=-1; sm2Y=-1; cursor2El.style.opacity='0';
      if(!modalOpen) hideScrollIndicator();
    }

    if(!allHands.length){
      cursor.style.opacity='0';
      sensorSt.textContent='Offline'; sensorSt.className='status-pill status-offline';
      setLog('No hand detected'); dwellEl=null;
      if(dragMode) cancelDrag(); smX=-1; smY=-1; return;
    }

    const hand=allHands[0];
    const rawX=(1-hand[8].x)*innerWidth,rawY=hand[8].y*innerHeight;
    smX=ema(smX,rawX,ALPHA); smY=ema(smY,rawY,ALPHA);
    cursor.style.opacity='1'; cursor.style.transform=`translate3d(${smX-14}px,${smY-14}px,0)`;
    tooltip.style.left=(smX+22)+'px'; tooltip.style.top=(smY-14)+'px';
    sensorSt.textContent='Online'; sensorSt.className='status-pill status-online';

    const palm=hand[8].y<hand[6].y&&hand[12].y<hand[10].y&&hand[16].y<hand[14].y&&hand[20].y<hand[18].y;
    if(palm&&Date.now()-lastAction>3000){
      if(dragMode) cancelDrag();
      lastAction=Date.now();
      setLog('Palm flash','var(--warning)'); showTip('Palm detected — grab × 3 in air to turn off gesture');
      cursor.style.boxShadow='0 0 0 6px rgba(217,119,6,0.3)';
      setTimeout(()=>{ cursor.style.boxShadow=''; },600);
    }

    const pinch=Math.hypot(hand[8].x-hand[4].x,hand[8].y-hand[4].y);
    const isPinchClosed=pinch<PINCH_CLOSE;
    const isPinchOpen=pinch>PINCH_OPEN;

    if(dragMode){
      if(!isPinchOpen){
        moveGhost(smX,smY);
        cursor.style.border='2px solid var(--warning)';
        cursor.style.boxShadow='0 0 0 4px rgba(217,119,6,0.2)';
        setLog('Dragging','var(--warning)');
        if(overDropZone(smX,smY)){ dropZone.classList.add('drag-over'); showTip('Release to drop',400); } else dropZone.classList.remove('drag-over');
      } else {
        if(overDropZone(smX,smY)) commitDrop(dragFile); else { showTip('Cancelled'); setLog('Idle'); }
        cancelDrag(); lastAction=Date.now();
      }
      return;
    }

    if(isPinchClosed){
      cursor.style.border='2px solid var(--navy-mid)';
      cursor.style.boxShadow='0 0 0 4px rgba(37,99,235,0.2)';
      cursor.style.background='rgba(37,99,235,0.12)';
      setLog('Pinch','var(--navy-mid)');
      pinchConfirm++;
      if(pinchConfirm>=2&&!isPinching&&Date.now()-lastAction>640){
        isPinching=true; lastAction=Date.now(); dwellEl=null;
        const el=document.elementFromPoint(smX,smY);
        if(el){
          const fr=el.closest('.file-row[data-drag-pool]');
          const cl=el.closest('.g-clickable,button,[onclick]');
          if(fr){
            const pool=fr.dataset.dragPool,fi=parseInt(fr.dataset.dragFi);
            const fj=fr.dataset.dragFj!==undefined&&fr.dataset.dragFj!==''?parseInt(fr.dataset.dragFj):null;
            let file=null;
            if(pool==='folder'&&fj!==null&&folders[fi]) file=folders[fi].files[fj];
            else if(pool==='loose') file=looseFiles[fi];
            if(file){ dragMode=true; dragFile=file; dragSource=fr; fr.classList.add('is-grabbed'); showGhost(file,smX,smY); setLog('Grabbed','var(--warning)'); showTip('Drag to drop zone',1000); }
            grabWasOnFile=true;
          } else if(cl){ cl.click(); showTip('Selected'); grabWasOnFile=true; }
          else grabWasOnFile=false;
        } else grabWasOnFile=false;
      }
    } else {
      pinchConfirm=0;
      if(isPinchOpen){
        if(isPinching&&!grabWasOnFile&&!dragMode){
          airGrabCount++;
          updateGrabBadge(airGrabCount);
          setLog(`Air grab ${airGrabCount}/3`,'var(--warning)');
          if(airGrabCount>=3){ airGrabCount=0; updateGrabBadge(0); showTip('Gesture off'); setLog('Off'); setTimeout(()=>toggleGesture(),400); }
        }
        isPinching=false;
        cursor.style.border='2px solid var(--navy-mid)';
        cursor.style.boxShadow='';
        cursor.style.background='rgba(37,99,235,0.08)';
        clearTimeout(grabResetTimer);
        if(airGrabCount>0&&airGrabCount<3) grabResetTimer=setTimeout(()=>{ airGrabCount=0; updateGrabBadge(0); },2000);
      }
    }

    /* dwell hover */
    if(isPinchOpen&&!dragMode){
      const el=document.elementFromPoint(smX,smY);
      const target=el?el.closest('.g-clickable'):null;
      document.querySelectorAll('.g-hover').forEach(e=>e.classList.remove('g-hover'));
      if(target){
        target.classList.add('g-hover'); setLog('Hover');
        if(dwellEl!==target){ dwellEl=target; dwellStart=Date.now(); }
        else {
          const prog=Math.min((Date.now()-dwellStart)/DWELL,1);
          const ring=target.querySelector('.d-ring');
          if(ring){ ring.classList.add('active'); const c=ring.querySelector('circle'); if(c) c.style.strokeDashoffset=44-(44*prog); }
          if(Date.now()-dwellStart>=DWELL&&Date.now()-lastAction>DWELL+50){ lastAction=Date.now(); dwellEl=null; target.click(); showTip('Selected ✓'); setLog('Dwell click','var(--success)'); }
        }
      } else {
        dwellEl=null;
        document.querySelectorAll('.d-ring').forEach(r=>{ r.classList.remove('active'); const c=r.querySelector('circle'); if(c) c.style.strokeDashoffset='44'; });
        setLog('Idle');
      }
    }
  });

  new Camera(document.getElementById('webcam'),{onFrame:async()=>await hands.send({image:document.getElementById('webcam')}),width:640,height:480}).start();
}


/* ════════════════════════════════════════════════════
   H. FULLSCREEN VIEWER + WATER RIPPLE ZOOM
════════════════════════════════════════════════════ */
const viewerCanvas=document.getElementById('viewer-canvas');
const vCtx=viewerCanvas.getContext('2d');
let rippleBuffer1=null,rippleBuffer2=null,rippleW=0,rippleH=0;
let rippleAnimId=null;
let viewerOffscreenCanvas=null,viewerOffCtx=null;

function initRippleBuffers(w,h){ rippleW=w; rippleH=h; rippleBuffer1=new Float32Array(w*h); rippleBuffer2=new Float32Array(w*h); }
function addRipple(x,y,strength=400){
  const ix=Math.floor(x),iy=Math.floor(y);
  for(let dy=-4;dy<=4;dy++) for(let dx=-4;dx<=4;dx++){
    const nx=ix+dx,ny=iy+dy;
    if(nx>=0&&nx<rippleW&&ny>=0&&ny<rippleH){ const d=Math.hypot(dx,dy); if(d<5) rippleBuffer1[ny*rippleW+nx]+=strength*(1-d/5); }
  }
}
function stepRipple(){
  for(let y=1;y<rippleH-1;y++) for(let x=1;x<rippleW-1;x++){
    const i=y*rippleW+x;
    rippleBuffer2[i]=((rippleBuffer1[(y-1)*rippleW+x]+rippleBuffer1[(y+1)*rippleW+x]+rippleBuffer1[y*rippleW+(x-1)]+rippleBuffer1[y*rippleW+(x+1)])/2)-rippleBuffer2[i];
    rippleBuffer2[i]*=0.94;
  }
  const tmp=rippleBuffer1; rippleBuffer1=rippleBuffer2; rippleBuffer2=tmp;
}
function renderViewer(){
  if(!viewerOpen||!viewerImg) return;
  const cw=viewerCanvas.width,ch=viewerCanvas.height;
  viewerOffCtx.clearRect(0,0,cw,ch);
  const sw=cw/viewerZoom,sh=ch/viewerZoom;
  const sx=(viewerImg.naturalWidth/2)-sw/2,sy=(viewerImg.naturalHeight/2)-sh/2;
  viewerOffCtx.drawImage(viewerImg,Math.max(0,sx),Math.max(0,sy),Math.min(sw,viewerImg.naturalWidth),Math.min(sh,viewerImg.naturalHeight),0,0,cw,ch);
  stepRipple();
  const src=viewerOffCtx.getImageData(0,0,cw,ch);
  const dst=vCtx.createImageData(cw,ch);
  for(let y=1;y<ch-1;y++) for(let x=1;x<cw-1;x++){
    const i=y*cw+x;
    let dx=Math.max(-8,Math.min(8,Math.floor(rippleBuffer1[i]-rippleBuffer1[i+1])));
    let dy=Math.max(-8,Math.min(8,Math.floor(rippleBuffer1[i]-rippleBuffer1[i+cw])));
    const sx2=Math.max(0,Math.min(cw-1,x+dx)),sy2=Math.max(0,Math.min(ch-1,y+dy));
    const si=(sy2*cw+sx2)*4,di=i*4;
    dst.data[di]=src.data[si]; dst.data[di+1]=src.data[si+1]; dst.data[di+2]=src.data[si+2]; dst.data[di+3]=src.data[si+3];
  }
  vCtx.putImageData(dst,0,0);
  /* subtle vignette */
  const grad=vCtx.createRadialGradient(cw/2,ch/2,cw*0.3,cw/2,ch/2,cw*0.72);
  grad.addColorStop(0,'rgba(0,0,0,0)'); grad.addColorStop(1,'rgba(0,0,0,0.35)');
  vCtx.fillStyle=grad; vCtx.fillRect(0,0,cw,ch);
  rippleAnimId=requestAnimationFrame(renderViewer);
}
function openViewer(imgSrc){
  viewerOpen=true; viewerZoom=1.0; lastPinchDist=null;
  document.getElementById('viewer-overlay').classList.add('open');
  const img=new Image();
  img.onload=()=>{
    viewerImg=img;
    const maxW=Math.min(innerWidth*0.88,900),maxH=Math.min(innerHeight*0.82,700);
    const ar=img.naturalWidth/img.naturalHeight;
    let cw,ch;
    if(ar>maxW/maxH){ cw=maxW; ch=maxW/ar; } else { ch=maxH; cw=maxH*ar; }
    viewerCanvas.width=Math.floor(cw); viewerCanvas.height=Math.floor(ch);
    viewerOffscreenCanvas=document.createElement('canvas');
    viewerOffscreenCanvas.width=Math.floor(cw); viewerOffscreenCanvas.height=Math.floor(ch);
    viewerOffCtx=viewerOffscreenCanvas.getContext('2d');
    initRippleBuffers(Math.floor(cw),Math.floor(ch));
    cancelAnimationFrame(rippleAnimId); renderViewer();
    addRipple(cw/2,ch/2,600);
  };
  img.src=imgSrc; updateZoomBadge();
}
function closeViewer(){ viewerOpen=false; cancelAnimationFrame(rippleAnimId); document.getElementById('viewer-overlay').classList.remove('open'); cursor.style.opacity='0'; viewerImg=null; lastPinchDist=null; }
function applyZoom(z){ viewerZoom=Math.max(0.5,Math.min(6,z)); updateZoomBadge(); if(viewerCanvas.width>0) addRipple(viewerCanvas.width/2,viewerCanvas.height/2,80); }
function updateZoomBadge(){ document.getElementById('zoom-level-badge').textContent=`${viewerZoom.toFixed(1)}×`; }
viewerCanvas.addEventListener('click',e=>{ const r=viewerCanvas.getBoundingClientRect(); addRipple(e.clientX-r.left,e.clientY-r.top,350); });


/* ════════════════════════════════════════════════════
   I. PROCESS BATCH + RESULT MODAL
════════════════════════════════════════════════════ */
function typeWriterEffect(id,text,speed=80){
  let i=0; const el=document.getElementById(id);
  (function t(){ if(i<text.length){ el.textContent+=text.charAt(i++); setTimeout(t,speed); } })();
}
function closeResultModal(){ resultModal.classList.remove('open'); hideScrollIndicator(); }

/* tumor class colors */
const TUMOR_COLORS={ 'Glioma':'#dc2626','Meningioma':'#d97706','Pituitary':'#7c3aed','No Tumor':'#059669' };
const TUMOR_BG={ 'Glioma':'#fee2e2','Meningioma':'#fef3c7','Pituitary':'#ede9fe','No Tumor':'#d1fae5' };

function getSeverity(type,conf){
  if(type==='No Tumor') return {label:'Clear',color:'var(--success)',bg:'var(--success-bg)'};
  if(conf>=90) return {label:'High confidence',color:'var(--danger)',bg:'var(--danger-bg)'};
  if(conf>=70) return {label:'Moderate confidence',color:'var(--warning)',bg:'var(--warning-bg)'};
  return {label:'Low confidence',color:'var(--text-3)',bg:'var(--bg-panel)'};
}

/* known Glioma sensitivity per model from medical-grade evaluation */
const GLIOMA_SENSITIVITY={ cnn:80.71,resnet18:83.75,efficientnet_b0:85.75 };

async function processBatch(){
  if(!queue.length){ showTip('No scans in queue'); return; }
  const area=document.getElementById('result-content');
  resultModal.classList.add('open');
  resultModal.scrollTop=0;
  showScrollIndicator(); updateScrollThumb();

  area.innerHTML=`
    <div style="text-align:center;padding:80px 0;">
      <div style="width:48px;height:48px;border:3px solid var(--border);border-top-color:var(--navy-mid);
                  border-radius:50%;animation:spin 0.8s linear infinite;margin:0 auto 20px;"></div>
      <div style="font-size:15px;font-weight:600;color:var(--text-1);">Running diagnostic sweep…</div>
      <div style="font-size:12px;color:var(--text-4);margin-top:6px;">Model: ${MODEL_INFO[selectedModel]?.label||selectedModel}</div>
      <style>@keyframes spin{to{transform:rotate(360deg)}}</style>
    </div>`;

  let loadedCount=0;

  for(let i=0;i<queue.length;i++){
    const fd=new FormData(); fd.append('file',queue[i]);
    try{
      const r=await fetch(
        `http://127.0.0.1:8000/predict?model_key=${selectedModel}&explain=true`,
        {method:'POST',body:fd}
      );
      if(!r.ok){ const err=await r.json(); throw new Error(err.detail||`HTTP ${r.status}`); }
      const d=await r.json();

      /* non-brain image filter */
      if(looksLikeNonBrainImage(d.all_scores)){
        if(loadedCount===0) area.innerHTML='';
        loadedCount++;
        const warn=document.createElement('div');
        warn.style.cssText='padding:32px;text-align:center;';
        warn.innerHTML=`
          <div style="background:var(--warning-bg);border:1px solid rgba(217,119,6,0.3);border-radius:12px;padding:24px;">
            <i class="fas fa-exclamation-triangle" style="color:var(--warning);font-size:24px;margin-bottom:12px;display:block;"></i>
            <div style="font-size:14px;font-weight:600;color:var(--text-1);margin-bottom:6px;">Image rejected</div>
            <div style="font-size:12px;color:var(--text-3);">${queue[i].name} — model confidence too low across all classes. This image may not be a brain MRI scan.</div>
          </div>`;
        area.appendChild(warn);
        continue;
      }

      if(loadedCount===0) area.innerHTML='';
      loadedCount++;
      document.getElementById('result-count').textContent=`${loadedCount} scan${loadedCount!==1?'s':''}`;
      setLog(`Processed ${loadedCount}/${queue.length}`,'var(--success)');

      const objUrl=URL.createObjectURL(queue[i]);
      const heatSrc=`data:image/jpeg;base64,${d.heatmap}`;
      const rid=`res-${i}`;
      const tColor=TUMOR_COLORS[d.tumor_type]||'var(--navy-mid)';
      const tBg=TUMOR_BG[d.tumor_type]||'var(--navy-pale)';
      const sev=getSeverity(d.tumor_type,d.confidence);
      const gliomaSens=GLIOMA_SENSITIVITY[selectedModel];
      const showGliomaWarning=d.tumor_type==='Glioma'&&gliomaSens&&gliomaSens<95;

      const barsHtml=Object.entries(d.all_scores||{}).map(([name,pct])=>`
        <div class="pred-bar-row">
          <span class="pred-bar-label" style="font-size:11px;color:var(--text-2);width:96px;flex-shrink:0;">${name}</span>
          <div class="pred-bar-track">
            <div class="pred-bar-fill${name===d.tumor_type?' top':''}" style="width:${pct}%;background:${TUMOR_COLORS[name]||'var(--navy-mid)'}"></div>
          </div>
          <span class="pred-bar-pct">${pct}%</span>
        </div>`).join('');

      const row=document.createElement('div');
      row.className='result-row fade-in';
      row.innerHTML=`
        <div style="display:flex;gap:16px;flex-shrink:0;flex-direction:column;align-items:center;">
          <div>
            <div style="font-size:10px;font-weight:600;letter-spacing:0.08em;color:var(--text-4);text-transform:uppercase;margin-bottom:6px;text-align:center;">Original scan</div>
            <div class="scan-thumb" onclick="openViewer('${objUrl}')">
              <img src="${objUrl}" style="width:176px;height:176px;object-fit:cover;">
              <div class="thumb-label">Input</div>
              <div class="thumb-zoom"><i class="fas fa-expand-alt"></i></div>
            </div>
          </div>
          <div>
            <div style="font-size:10px;font-weight:600;letter-spacing:0.08em;color:var(--text-4);text-transform:uppercase;margin-bottom:6px;text-align:center;">Grad-CAM</div>
            <div class="scan-thumb" onclick="openViewer('${heatSrc}')">
              <img src="${heatSrc}" style="width:176px;height:176px;object-fit:cover;">
              <div class="thumb-label">Activation map</div>
              <div class="thumb-zoom"><i class="fas fa-expand-alt"></i></div>
            </div>
          </div>
        </div>

        <div class="pred-card">
          <div class="pred-label">Diagnosis</div>
          <div class="pred-class" id="${rid}" style="color:${tColor};"></div>
          <div class="pred-conf">${d.confidence}% model confidence · ${d.device||'CPU'}</div>

          <div style="display:flex;gap:8px;align-items:center;margin-bottom:16px;flex-wrap:wrap;">
            <span style="font-size:11px;font-weight:600;padding:4px 10px;border-radius:20px;
                         background:${sev.bg};color:${sev.color};border:1px solid ${sev.color}33;">
              ${sev.label}
            </span>
            <span style="font-size:10px;font-weight:600;padding:4px 10px;border-radius:20px;
                         background:var(--navy-pale);color:var(--navy-mid);border:1px solid var(--navy-border);">
              ${d.model_used||selectedModel}
            </span>
          </div>

          <div class="pred-label" style="margin-bottom:8px;">Class probability distribution</div>
          <div class="pred-bars">${barsHtml}</div>

          ${showGliomaWarning?`
          <div class="caution-banner">
            <i class="fas fa-exclamation-triangle"></i>
            <span>Glioma sensitivity for this model is ${gliomaSens}% on held-out test data — below the 95% benchmark target. A negative result does not rule out glioma.</span>
          </div>`:''}

          ${d.explanation?`<div class="explain-box">${d.explanation}</div>`:''}

          <div style="margin-top:12px;font-size:10px;color:var(--text-4);">
            <i class="fas fa-info-circle"></i>&nbsp;
            Click images to open fullscreen viewer · Use gesture pinch to zoom
          </div>
        </div>`;

      area.appendChild(row);
      typeWriterEffect(rid,d.tumor_type);

    } catch(e){
      if(loadedCount===0) area.innerHTML='';
      loadedCount++;
      const errRow=document.createElement('div');
      errRow.style.cssText='padding:40px;';
      errRow.innerHTML=`
        <div style="background:var(--danger-bg);border:1px solid rgba(220,38,38,0.25);
                    border-radius:12px;padding:24px;text-align:center;">
          <i class="fas fa-exclamation-triangle" style="color:var(--danger);font-size:22px;display:block;margin-bottom:12px;"></i>
          <div style="font-size:14px;font-weight:600;color:var(--text-1);margin-bottom:6px;">Request failed</div>
          <div style="font-size:12px;color:var(--text-3);">${e.message||'Cannot reach http://127.0.0.1:8000 — make sure the FastAPI backend is running.'}</div>
          <div style="font-size:11px;color:var(--text-4);margin-top:8px;">Scan ${i+1}: ${queue[i].name}</div>
        </div>`;
      area.appendChild(errRow);
    }
  }

  if(loadedCount>0){
    const summary=document.createElement('div');
    summary.style.cssText='text-align:center;padding:32px 0 16px;font-size:11px;color:var(--text-4);border-top:1px solid var(--border);margin-top:32px;';
    summary.textContent=`Sweep complete — ${loadedCount} scan${loadedCount!==1?'s':''} processed · Model: ${MODEL_INFO[selectedModel]?.label||selectedModel}`;
    area.appendChild(summary);
  }
  setLog('Sweep complete','var(--success)');
}


/* ════════════════════════════════════════════════════
   J. MODEL SELECTOR
════════════════════════════════════════════════════ */
let selectedModel='efficientnet_b0';
let modelDropdownOpen=false;

const MODEL_INFO={
  cnn:{
    label:'Custom CNN — 89.94% test acc',
    arch:'4-block CNN + BatchNorm',
    params:'~8.8M trainable',
    val_acc:'93.84%',
  },
  resnet18:{
    label:'ResNet-18 — 95.12% test acc',
    arch:'ResNet-18 pretrained backbone',
    params:'~11.3M trainable',
    val_acc:'99.02%',
  },
  efficientnet_b0:{
    label:'EfficientNet-B0 — 95.75% test acc',
    arch:'EfficientNet-B0 pretrained',
    params:'~5.3M trainable',
    val_acc:'99.11%',
  }
};

function toggleModelDropdown(){
  modelDropdownOpen=!modelDropdownOpen;
  const panel=document.getElementById('model-options');
  const arrow=document.getElementById('model-arrow');
  const trigger=document.getElementById('model-trigger');
  if(modelDropdownOpen){
    panel.classList.add('open'); arrow.textContent='▴';
    trigger.style.borderColor='var(--navy-mid)';
    trigger.style.boxShadow='0 0 0 3px rgba(37,99,235,0.12)';
  } else {
    panel.classList.remove('open'); arrow.textContent='▾';
    trigger.style.borderColor=''; trigger.style.boxShadow='';
  }
}

function selectModel(key){
  selectedModel=key;
  const info=MODEL_INFO[key];
  if(!info) return;
  document.getElementById('model-trigger-label').textContent=info.label;
  document.getElementById('model-badge').innerHTML=`
    <span class="mb-key">arch: </span><span>${info.arch}</span><br>
    <span class="mb-key">params: </span><span>${info.params}</span><br>
    <span class="mb-key">val_acc: </span><span class="mb-good">${info.val_acc}</span>`;
  document.querySelectorAll('.model-opt').forEach(btn=>btn.classList.toggle('is-active',btn.dataset.key===key));
  modelDropdownOpen=false;
  const panel=document.getElementById('model-options');
  const arrow=document.getElementById('model-arrow');
  const trigger=document.getElementById('model-trigger');
  panel.classList.remove('open'); arrow.textContent='▾';
  trigger.style.borderColor=''; trigger.style.boxShadow='';
  setLog(`Model: ${key}`,'var(--navy-mid)');
  showTip(`Model set to ${info.label}`);
}

document.addEventListener('click',e=>{
  if(!e.target.closest('#model-dropdown-wrap')){
    modelDropdownOpen=false;
    const p=document.getElementById('model-options'); if(p) p.classList.remove('open');
    const a=document.getElementById('model-arrow'); if(a) a.textContent='▾';
  }
});

async function syncModelsFromBackend(){
  try{
    // const r=await fetch('https://your-render-url.onrender.com/models');
    const r=await fetch('http://127.0.0.1:8000/models');
    const data=await r.json();
    data.models.forEach(m=>{
      const btn=document.querySelector(`.model-opt[data-key="${m.key}"]`);
      if(btn&&!m.loaded){
        btn.style.opacity='0.45';
        const d=btn.querySelector('div'); if(d) d.textContent+=' [offline]';
      }
    });
    if(data.default) selectModel(data.default);
  } catch(e){ console.warn('Backend not reachable for model sync:',e); }
}

window.addEventListener('DOMContentLoaded',()=>{ setTimeout(syncModelsFromBackend,800); });