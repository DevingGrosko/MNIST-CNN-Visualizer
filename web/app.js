import { initPalette } from './palette.js';
import { pauseTrainingMovie, resetTrainingMovie } from './training-player.js';
import {forward,cloneWeights} from './engine.js';
import {$,$$,f,pct,heatmap,color,esc} from './render.js';
import {STEPS,renderLesson} from './lessons.js';
import {GUIDE} from './guide.js';
import { PAGES, pageForStep, resolvePage } from './pages.js';
const state={step:1,channel:0,inputChannel:0,x:13,y:13,index:0,neuron:0,classIndex:7,target:7,snapshot:'trained',sampleIndex:0,source:'sample',scanning:false,trainingSteps:0};
let currentPage = resolvePage(location.pathname);
if (Number.isInteger(currentPage.step)) state.step = currentPage.step;
let calculationTimer=null;
let timer=null,drawContext=null,drawing=false,lastPoint=null,drawFrame=null;
const app={state,
  lesson(){if (!Number.isInteger(currentPage.step)) {paintInput(); return;} const element=document.activeElement;const active=element?.id?'#'+element.id:['data-channel','data-neuron','data-index','data-slice','data-connection','data-softmax-digit','data-checkpoint','data-inspect-cell'].filter(k=>element?.hasAttribute(k)).map(k=>`[${k}="${element.getAttribute(k)}"]`)[0];renderLesson(app);if(active&&$(active))$(active).focus({preventScroll:true});paintInput();},
  focus(x,y){this.stopCalculation();state.reveal=9;state.x=x;state.y=y;this.lesson();},
  channel(c){this.stopCalculation();state.reveal=9;state.channel=c;this.lesson();},
  setStep(index, focus=false) {
    const page = pageForStep(index);
    if (!page) throw new Error('Invalid lesson');
    navigate(page.id, {focus});
  },
  stopCalculation() {
    clearInterval(calculationTimer);
    calculationTimer = null;
    state.calculating = false;
  },
  replayProducts() {
    if (state.calculating) { this.stopCalculation(); this.lesson(); return; }
    this.stopScan();
    state.calculating = true;
    if ((state.reveal ?? 9) === 9) state.reveal = 0;
    state.term = Math.max(0,state.reveal-1);
    this.lesson();
    calculationTimer = setInterval(() => {
      state.reveal = Math.min(9,state.reveal+1);
      state.term = state.reveal-1;
      if (state.reveal === 9) this.stopCalculation();
      this.lesson();
    }, 650);
  },
  stopScan(){clearInterval(timer);timer=null;state.scanning=false;},
  scan(n){if(state.scanning){this.stopScan();this.lesson();return;}state.scanning=true;this.lesson();timer=setInterval(()=>{const i=(state.y*n+state.x+1)%(n*n);state.x=i%n;state.y=Math.floor(i/n);this.lesson();},250);},
  update(){state.tensors=forward(state.pixels,state.weights);renderPrediction();renderNetwork();renderContext();this.lesson();},
  refreshTrainingState(){state.tensors=forward(state.pixels,state.weights);renderPrediction();renderNetwork();renderContext();renderInputControls();updateModelControls();},
  resetTraining(){resetTrainingMovie(this);this.stopCalculation();state.reveal=9;},
  loadSample(index){this.stopScan();state.sampleIndex=index;state.source='sample';const sample=state.samples[index];state.pixels=Float64Array.from(sample.pixels,p=>p/255);state.target=sample.label;state.classIndex=sample.label;this.resetTraining();renderInputControls();this.update();},
  setSnapshot(snapshot){if(!['trained','initial'].includes(snapshot))throw new Error('Invalid snapshot');this.stopScan();state.snapshot=snapshot;state.weights=cloneWeights(state.model[snapshot]);state.trainingSteps=0;this.resetTraining();updateModelControls();this.update();},
  editPixel(index,value){state.pixels[index]=value;state.source='edited';this.resetTraining();renderInputControls();this.update();},
};
function motion(){return matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth';}
function updateModelControls(){const trained=state.snapshot==='trained';$('#trained').classList.toggle('selected',trained);$('#trained').setAttribute('aria-pressed',trained);$('#untrained').classList.toggle('selected',!trained);$('#untrained').setAttribute('aria-pressed',!trained);$('#accuracy').textContent=state.trainingSteps?`${state.trainingSteps} local weight update${state.trainingSteps===1?'':'s'}`:trained?`${pct(state.model.meta.history.at(-1).accuracy)} test accuracy`:'Random initialization';}
function renderInputControls(){const sample=state.samples[state.sampleIndex],isSample=state.source==='sample';$('#input-canvas').hidden=state.source==='draw';$('#draw-canvas').hidden=state.source!=='draw';$('#sample-controls').hidden=!isSample;$('#draw-controls').hidden=state.source!=='draw';$('#sample-mode').classList.toggle('selected',isSample);$('#sample-mode').setAttribute('aria-pressed',isSample);$('#draw-mode').classList.toggle('selected',state.source==='draw');$('#draw-mode').setAttribute('aria-pressed',state.source==='draw');$('.digit-picker').innerHTML=Array.from({length:10},(_,i)=>`<button data-digit="${i}" class="${sample.label===i?'selected':''}" aria-label="Load a sample of digit ${i}" aria-pressed="${sample.label===i}">${i}</button>`).join('');$$('[data-digit]').forEach(b=>b.onclick=()=>app.loadSample(state.samples.findIndex(s=>s.label===Number(b.dataset.digit))));$('#sample-label').textContent=`Test #${sample.id}`;$('#input-note').textContent=state.source==='sample'?'784 pixels. One grayscale channel.':state.source==='draw'?'Your drawing stays on this device.':state.source==='training'?`MNIST training #${state.trainingExample.id} · label ${state.target}`:'Your image stays on this device.';}
function paintInput(){if(!state.pixels)return;heatmap($('#input-canvas'),state.pixels,28,{gray:true,...(state.step===1?{x:state.x,y:state.y,patch:true}:{})});}
function renderPrediction(){const probs=state.tensors.softmax,winner=Array.from(probs).indexOf(Math.max(...probs));$('#predicted-digit').textContent=winner;$('#prediction-percent').textContent=pct(probs[winner]);$('#probabilities').innerHTML=Array.from(probs,(p,i)=>`<button class="probability ${i===winner?'winner':''} ${i===state.classIndex?'selected':''}" data-class="${i}" aria-label="Inspect digit ${i}, probability ${pct(p)}"><span class="prob-label">${i}</span><span class="prob-track"><span class="prob-fill" style="display:block;width:${p*100}%"></span></span><span class="prob-value">${pct(p)}</span></button>`).join('');$$('[data-class]').forEach(b=>b.onclick=()=>{state.classIndex=Number(b.dataset.class);app.setStep(10,true);});const blank=state.pixels.every(p=>p===0);$('#prediction-note').textContent=blank?'Blank input · draw or choose a digit.':state.snapshot==='initial'&&!state.trainingSteps?'Untrained network · scores are not learned.':state.source==='sample'?`True label: ${state.samples[state.sampleIndex].label} · ${winner===state.samples[state.sampleIndex].label?'Correct prediction':'A real model mistake'}`:state.source==='training'?`Training example · true label ${state.target}`:'Custom digit · the model may be uncertain or wrong.';}
function neuralSvg(output=false){const n=output?10:8;let lines='',nodes='';for(let i=0;i<6;i++)for(let j=0;j<n;j++)lines+=`<line x1="20" y1="${14+i*18}" x2="78" y2="${8+j*104/(n-1)}" stroke="var(--negative)" opacity="${.07+(i+j)%4*.02}"/>`;for(let i=0;i<6;i++)nodes+=`<circle cx="20" cy="${14+i*18}" r="4" fill="var(--bg)" stroke="var(--negative)" stroke-width="1"/>`;for(let j=0;j<n;j++){const v=output?state.tensors.softmax[j]:Math.min(1,state.tensors.hidden[j*16]/3);nodes+=`<circle cx="78" cy="${8+j*104/(n-1)}" r="${output?4:5}" fill="${color(v,1)}" stroke="${output?'var(--accent)':'var(--negative)'}" stroke-width="1"/>`;}return `<svg class="node-svg" viewBox="0 0 100 120" aria-hidden="true">${lines}${nodes}</svg>`;}
function renderNetwork(){const block=state.step<=3?1:state.step<=7?4:state.step===8?8:10;$('#network-flow').innerHTML=`<button class="network-block ${block===1?'selected':''}" data-step="1" aria-label="Explore the first convolution block"><div class="tensor-stack">${[0,1,2].map(i=>`<canvas data-map="pool1" data-ch="${i}" aria-hidden="true"></canvas>`).join('')}</div><span class="net-name">Conv block 1</span><span class="net-shape">8 × 14 × 14</span><span class="net-op">Conv → ReLU → Pool</span></button><span class="net-arrow" aria-hidden="true">→</span><button class="network-block ${block===4?'selected':''}" data-step="4" aria-label="Explore the second convolution block"><div class="tensor-stack smaller">${[2,6,9].map(i=>`<canvas data-map="pool2" data-ch="${i}" aria-hidden="true"></canvas>`).join('')}</div><span class="net-name">Conv block 2</span><span class="net-shape">16 × 7 × 7</span><span class="net-op">Conv → ReLU → Pool</span></button><span class="net-arrow" aria-hidden="true">→</span><button class="network-block classification ${block===8?'selected':''}" data-step="8" aria-label="Explore the fully connected layer">${neuralSvg()}<span class="net-name">Dense</span><span class="net-shape">784 → 128</span><span class="net-op">Linear → ReLU</span></button><span class="net-arrow" aria-hidden="true">→</span><button class="network-block classification ${block===10?'selected':''}" data-step="10" aria-label="Explore the output probabilities">${neuralSvg(true)}<span class="net-name">Output</span><span class="net-shape">128 → 10</span><span class="net-op">Linear → Softmax</span></button>`;$$('#network-flow [data-step]').forEach(b=>b.onclick=()=>app.setStep(Number(b.dataset.step),true));$$('[data-map]').forEach(c=>{const n=c.dataset.map==='pool1'?14:7,ch=Number(c.dataset.ch);heatmap(c,state.tensors[c.dataset.map].slice(ch*n*n,(ch+1)*n*n),n);});$('#network-note-copy').textContent=state.step>=8?'Every output neuron connects to all 128 hidden activations. The diagram shows a subset of the hidden neurons; inspect the full weight tables below.':'The first block detects local strokes. The second combines them across eight channels. The stacks preview three channels; inspect every channel below.';}
function renderNav() {
  $('#stage-nav').innerHTML = STEPS.map((step, index) => {
    const group = index === 0 ? '01 / EXTRACT FEATURES' : index === 7 ? '02 / MAKE A PREDICTION' : index === 11 ? '03 / LEARN FROM ERROR' : '';
    const page = pageForStep(index);
    return `${group ? `<p class="toc-group">${group}</p>` : ''}<a class="stage-button ${currentPage.id === page.id ? 'selected' : ''}" href="./${page.file}" data-route="${page.id}" ${currentPage.id === page.id ? 'aria-current="page"' : ''}><span>${String(index + 1).padStart(2, '0')}</span>${page.label}</a>`;
  }).join('');
  const selectedLink=$('.stage-button.selected');
  const chapterList=$('#stage-nav');
  if(selectedLink && chapterList.scrollWidth>chapterList.clientWidth) chapterList.scrollLeft=Math.max(0,selectedLink.offsetLeft-chapterList.offsetLeft-chapterList.clientWidth/2);
  $$('.overview-link, .notes-link, .site-header [data-route]').forEach(link => {
    const active = link.dataset.route === currentPage.id || (link.dataset.route === 'input' && Number.isInteger(currentPage.step) && link.closest('.site-header'));
    link.classList.toggle('active', Boolean(active));
    if (active) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  });
}
function navigate(id, {push=true, focus=true}={}) {
  const page = PAGES.find(page => page.id === id);
  if (!page) throw new Error('Unknown page');
  app.stopScan();
  app.stopCalculation?.();
  pauseTrainingMovie(app);
  document.body.classList.remove('movie-wide');
  if (push && currentPage.id !== page.id) history.pushState({page: page.id}, '', './' + page.file);
  currentPage = page;
  if (Number.isInteger(page.step)) state.step = page.step;
  document.body.dataset.page = page.id;
  document.title = `${page.label} — CNN, Explained`;
  $('#page-title').textContent = page.title;
  $('#page-deck').textContent = page.deck;
  $('#page-eyebrow').textContent = Number.isInteger(page.step) ? `CHAPTER ${String(page.step + 1).padStart(2, '0')} / ${page.step < 7 ? 'FEATURE EXTRACTION' : page.step < 11 ? 'CLASSIFICATION' : 'LEARNING'}` : page.id === 'guide' ? 'REFERENCE / FIELD NOTES' : 'AN INTERACTIVE TEXTBOOK / OVERVIEW';
  $('#overview-page').hidden = page.id !== 'overview';
  $('#lesson-page').hidden = !Number.isInteger(page.step);
  $('#guide').hidden = page.id !== 'guide';
  $('#about-model').hidden = page.id !== 'guide';
  $('#start-tour').hidden = page.id !== 'overview';
  $('.model-bar').hidden = page.id === 'guide';
  if (Number.isInteger(page.step)) {
    const previous = page.step === 0 ? PAGES[0] : pageForStep(page.step - 1);
    const next = page.step === 11 ? PAGES.at(-1) : pageForStep(page.step + 1);
    $('#previous-step').href = './' + previous.file;
    $('#previous-step').dataset.route = previous.id;
    $('#previous-step').innerHTML = `<span>← PREVIOUS</span>${previous.label}`;
    $('#next-step').href = './' + next.file;
    $('#next-step').dataset.route = next.id;
    $('#next-step').innerHTML = `<span>NEXT →</span>${next.label}`;
    $('#step-count').textContent = `${page.step + 1} / 12`;
  }
  renderNav();
  if (state.tensors) { app.lesson(); renderContext(); }
  if (focus) { window.scrollTo({top: 0, behavior: motion()}); $('#page-title').focus({preventScroll:true}); }
  $('#announcement').textContent = page.title;
}
function renderContext() {
  if (!state.pixels) return;
  heatmap($('#context-image'), state.pixels, 28, {gray:true});
  const isSample = state.source === 'sample';
  $('#context-caption').textContent = isSample ? `MNIST test #${state.samples[state.sampleIndex].id}` : state.source==='training'?`MNIST training #${state.trainingExample.id}`:'Your custom image';
  $('#context-sample').innerHTML = `${isSample ? '' : `<option value="-1">${state.source==='training'?'Training digit '+state.target:'Custom'}</option>`}` + Array.from({length:10}, (_,digit) => `<option value="${digit}" ${isSample && digit === state.samples[state.sampleIndex].label ? 'selected' : ''}>Digit ${digit}</option>`).join('');
  const probabilities = state.tensors.softmax;
  const winner = probabilities.indexOf(Math.max(...probabilities));
  $('#context-prediction').textContent = `${winner} · ${pct(probabilities[winner])}`;
}
function wireNavigation() {
  document.addEventListener('click', event => {
    const link = event.target.closest('a[data-route]');
    if (!link || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    navigate(link.dataset.route);
  });
  addEventListener('popstate', () => navigate(resolvePage(location.pathname).id, {push:false}));
  $('#context-sample').onchange = event => {
    const digit = Number(event.target.value);
    if (digit >= 0) app.loadSample(state.samples.findIndex(sample => sample.label === digit));
  };
}
function beginDrawing(){app.stopScan();state.source='draw';app.resetTraining();renderInputControls();drawContext=$('#draw-canvas').getContext('2d');drawContext.fillStyle='#000';drawContext.fillRect(0,0,280,280);state.pixels=new Float64Array(784);app.update();}
function updateDrawing(){const mini=document.createElement('canvas');mini.width=mini.height=28;const c=mini.getContext('2d',{willReadFrequently:true});c.drawImage($('#draw-canvas'),0,0,28,28);const rgba=c.getImageData(0,0,28,28).data;state.pixels=Float64Array.from({length:784},(_,i)=>rgba[4*i]/255);app.resetTraining();app.update();}
function drawPoint(e){const canvas=$('#draw-canvas'),r=canvas.getBoundingClientRect(),p=[(e.clientX-r.left)*280/r.width,(e.clientY-r.top)*280/r.height];drawContext.strokeStyle='#fff';drawContext.fillStyle='#fff';drawContext.lineWidth=19;drawContext.lineCap='round';drawContext.lineJoin='round';drawContext.beginPath();if(lastPoint){drawContext.moveTo(...lastPoint);drawContext.lineTo(...p);drawContext.stroke();}else{drawContext.arc(...p,9.5,0,Math.PI*2);drawContext.fill();}lastPoint=p;if(!drawFrame)drawFrame=requestAnimationFrame(()=>{drawFrame=null;updateDrawing();});}
function fitCanvas(source){const c=source.getContext('2d',{willReadFrequently:true}),{data}=c.getImageData(0,0,source.width,source.height);let x0=source.width,y0=source.height,x1=-1,y1=-1;for(let y=0;y<source.height;y++)for(let x=0;x<source.width;x++){if(data[(y*source.width+x)*4]>25){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);}}const output=document.createElement('canvas');output.width=output.height=280;const ctx=output.getContext('2d');ctx.fillStyle='#000';ctx.fillRect(0,0,280,280);if(x1>=x0){const w=x1-x0+1,h=y1-y0+1,scale=200/Math.max(w,h);ctx.drawImage(source,x0,y0,w,h,(280-w*scale)/2,(280-h*scale)/2,w*scale,h*scale);}return output;}
function wireDrawing(){const canvas=$('#draw-canvas');canvas.onpointerdown=e=>{if(state.source!=='draw')return;e.preventDefault();canvas.setPointerCapture(e.pointerId);drawing=true;lastPoint=null;drawPoint(e);};canvas.onpointermove=e=>{if(drawing)drawPoint(e);};const end=()=>{drawing=false;lastPoint=null;};canvas.onpointerup=end;canvas.onpointercancel=end;canvas.onlostpointercapture=end;$('#clear-drawing').onclick=beginDrawing;$('#use-drawing').onclick=()=>{drawContext.drawImage(fitCanvas(canvas),0,0);updateDrawing();};$('#upload-image').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>15*1024*1024)throw new Error('Choose an image smaller than 15 MB.');const bitmap=await createImageBitmap(file);const source=document.createElement('canvas');source.width=source.height=280;const c=source.getContext('2d',{willReadFrequently:true});c.fillStyle='#fff';c.fillRect(0,0,280,280);const scale=Math.min(280/bitmap.width,280/bitmap.height);c.drawImage(bitmap,(280-bitmap.width*scale)/2,(280-bitmap.height*scale)/2,bitmap.width*scale,bitmap.height*scale);bitmap.close();const pixels=c.getImageData(0,0,280,280);let border=0,count=0;for(let y=0;y<280;y++)for(let x=0;x<280;x++)if(x<5||y<5||x>274||y>274){const i=(y*280+x)*4;border+=(pixels.data[i]+pixels.data[i+1]+pixels.data[i+2])/3;count++;}const invert=border/count>127;for(let i=0;i<pixels.data.length;i+=4){let v=.299*pixels.data[i]+.587*pixels.data[i+1]+.114*pixels.data[i+2];if(invert)v=255-v;pixels.data[i]=pixels.data[i+1]=pixels.data[i+2]=v;pixels.data[i+3]=255;}c.putImageData(pixels,0,0);beginDrawing();drawContext.drawImage(fitCanvas(source),0,0);updateDrawing();state.source='upload';renderInputControls();paintInput();renderPrediction();$('#upload-status').textContent='Image loaded and centered. Select Learning to set its correct label.';}catch(error){$('#upload-status').textContent=error.message||'This image could not be decoded. Try PNG or JPEG.';}};}
function webMCP(){const context=document.modelContext;if(!context?.registerTool)return;const lifecycle=new AbortController();addEventListener('pagehide',()=>lifecycle.abort(),{once:true});const tools=[{name:'inspect_cnn',description:'Read the current CNN input, layer, prediction, and real class scores.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true},execute:()=>({layer:STEPS[state.step].id,source:state.source,model:state.snapshot,probabilities:Array.from(state.tensors.softmax),logits:Array.from(state.tensors.logits)})},{name:'explore_cnn_sample',description:'Load an MNIST test digit and navigate to a CNN lesson in the visible lab.',inputSchema:{type:'object',properties:{digit:{type:'integer',minimum:0,maximum:9},step:{type:'integer',minimum:0,maximum:11}},required:['digit','step'],additionalProperties:false},annotations:{readOnlyHint:false},execute:input=>{if(!Number.isInteger(input?.digit)||input.digit<0||input.digit>9||!Number.isInteger(input?.step)||input.step<0||input.step>11)throw new Error('Expected digit 0–9 and step 0–11.');app.loadSample(state.samples.findIndex(s=>s.label===input.digit));app.setStep(input.step,true);return {layer:STEPS[state.step].id,probabilities:Array.from(state.tensors.softmax)};}}];for(const tool of tools){try{Promise.resolve(context.registerTool(tool,{signal:lifecycle.signal})).catch(()=>{});}catch{ /* Optional browser capability; the visual lab remains usable. */ }}}
async function init(){try{const responses=await Promise.all(['./assets/model.json','./assets/samples.json'].map(url=>fetch(new URL(url,import.meta.url))));if(responses.some(r=>!r.ok))throw new Error('The model files could not be loaded. Reload the page or try again shortly.');[state.model,state.samples]=await Promise.all(responses.map(r=>r.json()));state.weights=cloneWeights(state.model.trained);app.loadSample(state.samples.findIndex(s=>s.label===7));updateModelControls();renderNav();$('#lab').hidden=false;$('#loading').hidden=true;$('#trained').onclick=()=>app.setSnapshot('trained');$('#untrained').onclick=()=>app.setSnapshot('initial');$('#sample-mode').onclick=()=>app.loadSample(state.sampleIndex);$('#draw-mode').onclick=beginDrawing;$('#next-sample').onclick=()=>{const label=state.samples[state.sampleIndex].label,indices=state.samples.map((s,i)=>s.label===label?i:-1).filter(i=>i>=0);app.loadSample(indices[(indices.indexOf(state.sampleIndex)+1)%indices.length]);};wireNavigation();navigate(currentPage.id,{push:false,focus:false});wireDrawing();webMCP();$('#guide-content').innerHTML=GUIDE.map(([title,body])=>`<details><summary>${title}</summary><p>${body}</p></details>`).join('');const meta=state.model.meta;$('#model-facts').innerHTML=[['Architecture','1 → 8 → 16 channels'],['Classifier','784 → 128 → 10'],['Training',`${meta.epochs} epochs · Adam`],['Exported test accuracy',`${pct(meta.history.at(-1).accuracy)} / 10,000 images`],['Dataset','MNIST · 60,000 training digits'],['Execution','100% in your browser']].map(([k,v])=>`<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');}catch(error){$('#loading').hidden=true;$('#load-error').hidden=false;$('#load-error').innerHTML=`${esc(error.message)} <button class="button" onclick="location.reload()">Retry</button>`;$('#start-tour').disabled=true;$('#trained').disabled=true;$('#untrained').disabled=true;}}
initPalette();
navigate(currentPage.id, {push:false,focus:false});
init();
