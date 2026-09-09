import { getPalette, signedRGB, inkOn } from './palette.js';
import { loss } from './engine.js';
import { TrainingSession, PHASES, WEIGHT_LAYOUTS, gradientTerms, conv2MosaicIndex, conv2WeightIndex } from './training-session.js';
import { $, $$, f, pct, heatmap, color } from './render.js';

const players=new WeakMap();
const PHASE_INDEX=Object.fromEntries(PHASES.map((phase,index)=>[phase.id,index]));
const GRADIENT_PHASE={fc2:6,fc1:7,conv2:8,conv1:9};
let trainingData;

function exampleFromState(s) {
  return {id:s.source==='sample'?s.samples[s.sampleIndex].id:s.trainingExample?.id??null,
    label:s.target,pixels:Array.from(s.pixels,v=>Math.round(v*255)),split:s.source==='sample'?'test':s.source==='training'?'train':'custom'};
}
function playerFor(app) {
  if(!players.has(app)) {
    players.set(app,{session:new TrainingSession(app.state.weights,exampleFromState(app.state),{snapshot:app.state.snapshot}),
      playing:false,timer:null,speed:1,goal:1,mode:'repeat',selection:{key:'conv1Weight',index:4},baseSteps:app.state.trainingSteps,notice:'Choose a cell in any matrix. Its exact numbers appear below.',wide:true,error:''});
  }
  return players.get(app);
}
export function pauseTrainingMovie(app) {
  const player=players.get(app);
  if(!player)return;
  clearTimeout(player.timer);player.timer=null;player.playing=false;
}
export function resetTrainingMovie(app) { pauseTrainingMovie(app);players.delete(app); }

function commit(app,player,frame) {
  const s=app.state;
  s.weights=player.session.weights;s.trainingSteps=player.baseSteps+player.session.completed;
  s.pixels=Float64Array.from(frame.example.pixels,v=>v/255);s.target=frame.example.label;s.classIndex=s.target;
  if(frame.example.split==='train'){s.source='training';s.trainingExample=frame.example;}
  app.refreshTrainingState();
}
function advance(app,player) {
  const event=player.session.advance();
  if(event.phase===0&&event.frame.number===1&&!player.inspected){player.selection={key:'conv1Weight',index:event.frame.gradients.weights.conv1Weight.reduce((best,value,index,values)=>Math.abs(value)>Math.abs(values[best])?index:best,0)};}
  if(event.committed)commit(app,player,event.frame);
  app.lesson();
  return event;
}
function schedule(app,player) {
  if(!player.playing||document.body.dataset.page!=='learn')return;
  player.timer=setTimeout(()=>{
    try {
      advance(app,player);
      if(player.session.phase===PHASES.length-1&&player.session.completed>=player.goal){
        pauseTrainingMovie(app);
        player.notice=player.session.completed===1?'One complete update. Inspect the changed cells, then continue to update 10.':`${player.session.completed} updates complete. Scrub the checkpoints to compare the matrices.`;
        app.lesson();return;
      }
      schedule(app,player);
    }catch(error){pauseTrainingMovie(app);player.loading=false;player.error=error.message;app.lesson();}
  },Math.max(180,760/player.speed));
}
function play(app,player,goal) {
  pauseTrainingMovie(app);
  player.session.review=null;player.goal=goal;player.playing=true;player.error='';
  if(!player.session.frame||player.session.phase===PHASES.length-1)advance(app,player);
  else app.lesson();
  schedule(app,player);
}
async function newRun(app,player,{untrained=false}={}) {
  pauseTrainingMovie(app);
  const request=player.request=(player.request||0)+1;
  if(player.mode==='stream'&&!trainingData){
    player.loading=true;player.notice='Loading 200 examples from the MNIST training split…';app.lesson();
    const response=await fetch('./assets/training-samples.json');
    if(!response.ok)throw new Error('Training examples could not be loaded. Try the repeated-image mode.');
    const data=await response.json();
    if(data.split!=='train'||!Array.isArray(data.samples))throw new Error('Invalid training stream.');
    trainingData=data.samples;
  }
  player.loading=false;
  if(players.get(app)!==player||player.request!==request||document.body.dataset.page!=='learn')return;
  const s=app.state,snapshot=untrained?'initial':s.snapshot;
  const weights=untrained?s.model.initial:s.weights;
  player.baseSteps=untrained?0:s.trainingSteps;
  player.session=new TrainingSession(weights,exampleFromState(s),{rate:player.session.rate,mode:player.mode,trainingSamples:trainingData||[],snapshot});
  s.weights=player.session.weights;s.snapshot=snapshot;s.trainingSteps=player.baseSteps;
  player.goal=1;player.inspected=false;player.term=0;
  player.notice='Ready. Prediction → loss → gradients → changed matrices.';player.error='';
  app.refreshTrainingState();app.lesson();
}

function isGradientVisible(display,column) {
  return display.frame&&(display.phase==='result'||PHASE_INDEX[display.phase]>=GRADIENT_PHASE[column]);
}
function rangeMaximum(values) {let max=0;for(const value of values)max=Math.max(max,Math.abs(value));return max||1;}
function paintRect(canvas,values,rows,cols,{domain,changes,flash=false,selected=-1}={}) {
  canvas.width=cols;canvas.height=rows;
  const context=canvas.getContext('2d'),image=context.createImageData(cols,rows),max=domain||rangeMaximum(values);
  const maxChange=changes?rangeMaximum(changes):1;
  for(let index=0;index<values.length;index++){
    const rgb=flash&&changes&&Math.abs(changes[index])>maxChange*.35?getPalette().rgb.change:signedRGB(values[index],max);
    image.data.set([...rgb,255],index*4);
  }
  context.putImageData(image,0,0);
  if(selected>=0){context.fillStyle=getPalette().colors.text;context.fillRect(selected%cols,Math.floor(selected/cols),1,1);}
}
function paintKernel(canvas,values,{domain,changes,flash,selected}) {
  canvas.width=180;canvas.height=180;
  const context=canvas.getContext('2d'),maxChange=changes?rangeMaximum(changes):1;
  for(let i=0;i<9;i++) {
    const row=Math.floor(i/3),col=i%3;
    const cellColor=flash&&changes&&Math.abs(changes[i])>maxChange*.35?getPalette().rgb.change:signedRGB(values[i],domain);
    context.fillStyle=`rgb(${cellColor.join(',')})`;
    context.fillRect(col*60,row*60,60,60);
    context.strokeStyle=getPalette().colors.line;context.lineWidth=1;context.strokeRect(col*60,row*60,60,60);
    context.fillStyle=inkOn(cellColor);
    context.font='24px monospace';context.textAlign='center';context.textBaseline='middle';
    context.fillText(f(values[i],2),col*60+30,row*60+31);
    if(i===selected){context.strokeStyle=getPalette().colors.text;context.lineWidth=5;context.strokeRect(col*60+3,row*60+3,54,54);}
  }
}
function wireRect(canvas,rows,cols,key,app,player) {
  canvas.tabIndex=0;canvas.setAttribute('role','img');
  const select=(row,col)=>selectWeight(app,player,key,row*cols+col);
  canvas.onclick=event=>{const box=canvas.getBoundingClientRect();select(Math.min(rows-1,Math.max(0,Math.floor((event.clientY-box.top)/box.height*rows))),Math.min(cols-1,Math.max(0,Math.floor((event.clientX-box.left)/box.width*cols))));};
  canvas.onkeydown=event=>{
    const moves={ArrowLeft:-1,ArrowRight:1,ArrowUp:-cols,ArrowDown:cols};
    if(!(event.key in moves))return;event.preventDefault();
    const current=player.selection.key===key?player.selection.index:0;
    const index=Math.max(0,Math.min(valuesLength(rows,cols)-1,current+moves[event.key]));
    select(Math.floor(index/cols),index%cols);
  };
}
const valuesLength=(rows,cols)=>rows*cols;
function conv2Mosaic(values) {
  const output=new Float64Array(24*48);
  for(let i=0;i<values.length;i++)output[conv2MosaicIndex(i)]=values[i];
  return output;
}

function kernelShelf(key,gradient=false) {
  return `<div class="movie-kernels">${Array.from({length:8},(_,index)=>`<button data-movie-kernel="${index}" data-weight-key="${key}"><canvas data-kernel="${index}" data-gradient="${gradient}" aria-label="${gradient?'Gradient for':'Weights of'} first convolution filter ${index}"></canvas><span>${index}</span></button>`).join('')}</div>`;
}
function activationGroup(key,channels,size,label) {
  return `<div class="activation-label">${label}<span>${channels} × ${size} × ${size}</span></div><div class="movie-feature-grid">${Array.from({length:channels},(_,channel)=>`<canvas data-activation="${key}" data-channel-index="${channel}" data-size="${size}" role="img" aria-label="${label}, channel ${channel}"></canvas>`).join('')}</div>`;
}
function weightPanel(key,player,display) {
  const layout=WEIGHT_LAYOUTS[key],column=layout.column;
  const phase=PHASES.find(phase=>phase.id===display.phase);
  const active=phase?.column===column||phase?.column==='all';
  const gradients=isGradientVisible(display,column);
  const weights=key==='conv1Weight'?kernelShelf(key):`<canvas class="movie-weight-texture ${key==='conv2Weight'?'conv2-mosaic':''}" id="weights-${key}" aria-label="Full ${layout.label} weight matrix, ${layout.rows} by ${layout.cols}; click to magnify a cell"></canvas>`;
  const gradientView=key==='conv1Weight'?kernelShelf(key,true):`<canvas class="movie-weight-texture ${key==='conv2Weight'?'conv2-mosaic':''}" id="gradients-${key}" aria-label="Gradient matrix for ${layout.label}"></canvas>`;
  const activations=column==='conv1'?activationGroup('conv1',8,28,'Before ReLU')+activationGroup('relu1',8,28,'After ReLU')+activationGroup('pool1',8,14,'After pooling'):
    column==='conv2'?activationGroup('conv2',16,14,'Before ReLU')+activationGroup('relu2',16,14,'After ReLU')+activationGroup('pool2',16,7,'After pooling'):
    column==='fc1'?`<div class="activation-label">784 features → 128 neurons</div><canvas class="movie-vector" id="movie-dense" role="img" aria-label="All 128 dense pre-activations"></canvas><div class="activation-label">After ReLU</div><canvas class="movie-vector" id="movie-hidden" role="img" aria-label="All 128 hidden activations after ReLU"></canvas>`:`<div class="activation-label">Class probabilities</div><div id="movie-loss-equation" class="movie-loss-equation"></div><div id="movie-probabilities"></div><div class="activation-label">∂Loss / ∂logit</div><div class="movie-logit-gradient" id="movie-logit-gradient"></div>`;
  return `<article class="movie-stage ${active?'stage-active':''} ${phase?.direction==='backward'&&active?'stage-backward':''} ${display.phase==='update'?'stage-updating':''}" data-stage="${column}"><header><span>${column==='conv1'?'01':column==='conv2'?'02':column==='fc1'?'03':'04'}</span><h3>${layout.label}</h3><small>${layout.rows.toLocaleString()} × ${layout.cols.toLocaleString()}</small></header><div class="matrix-section"><div class="activation-label">Weights W<span>${key==='conv1Weight'?'8 filters':key==='conv2Weight'?'4 × 4 filters · 8 slices each':'full matrix'}</span></div>${weights}<div class="movie-bias-row"><span>Bias</span><canvas data-bias="${layout.bias}" aria-label="All ${layout.label} biases"></canvas></div></div><div class="matrix-section gradient-section ${gradients?'gradient-revealed':''}"><div class="activation-label">Gradients ∂L / ∂W<span>${gradients?'calculated':'waiting for backward pass'}</span></div><div class="gradient-content">${gradientView}</div></div><div class="activation-section">${activations}</div></article>`;
}

export function renderTrainingMovie(app) {
  const player=playerFor(app),session=player.session,display=session.display(),phase=PHASES.find(phase=>phase.id===display.phase);
  const frame=display.frame;
  const currentLoss=loss(display.tensors.logits,display.example.label);
  const correct=display.example.label,probability=display.tensors.softmax[correct];
  const active=player.session.phase;
  const checkpoints=[0,...session.frames.map(frame=>frame.number)];
  $('#lesson').innerHTML=`<div class="training-cinema ${player.wide?'cinema-expanded':''}">
    <div class="movie-toolbar"><div class="movie-example"><canvas id="movie-input" role="img" aria-label="The image being used for this training step"></canvas><div><span class="context-label">${display.example.split==='train'?'TRAINING EXAMPLE':display.example.split==='test'?'TEST IMAGE · EDUCATIONAL DEMO':'YOUR IMAGE'}</span><strong>Correct digit: ${correct}</strong><span class="movie-example-id">${display.example.id===null?'Custom image':'MNIST #'+display.example.id}</span></div></div><div class="movie-stats"><div><span>UPDATE</span><strong>${session.review?session.review.number:session.completed}<small> / ${Math.max(10,player.goal)}</small></strong></div><div><span>LOSS</span><strong>${f(currentLoss,4)}</strong></div><div><span>p(${correct})</span><strong>${pct(probability)}</strong></div></div><button class="text-button" id="expand-movie">${player.wide?'Exit wide view':'Expand matrix wall ↗'}</button></div>
    <div class="movie-controls"><button class="button primary" id="movie-demo">▶ Watch the first update</button><button class="button" id="movie-play">${player.playing?'Ⅱ Pause':session.frame?'▶ Resume / play one':'▶ Train once'}</button><button class="button" id="movie-ten">${session.completed<10?'Play to 10 updates →':'Play 10 more →'}</button><button class="button" id="movie-next">Next scene ›</button><div class="movie-speed"><label>Speed <select id="movie-speed">${[.5,1,2,4].map(speed=>`<option value="${speed}" ${speed===player.speed?'selected':''}>${speed}×</option>`).join('')}</select></label><label>Learning rate <select id="movie-rate">${[.001,.01,.05,.1].map(rate=>`<option value="${rate}" ${rate===session.rate?'selected':''}>${rate}</option>`).join('')}</select></label></div></div>
    <div class="movie-setup"><label>Learn from <select id="movie-mode"><option value="repeat" ${player.mode==='repeat'?'selected':''}>This image, repeatedly</option><option value="stream" ${player.mode==='stream'?'selected':''}>A stream of training digits</option></select></label><label>Target label <select id="movie-target">${Array.from({length:10},(_,i)=>`<option value="${i}" ${i===session.example.label?'selected':''}>${i}</option>`).join('')}</select></label><button class="text-button" id="movie-reset">Reset to random weights</button><span>“Watch” starts untrained and pauses after update 1. Then continue to 10.</span></div>
    <div class="movie-narration ${phase?.direction==='backward'?'narration-backward':''}"><div class="scene-number">${phase?String(PHASE_INDEX[phase.id]+1).padStart(2,'0'):'00'}<span>/ 12</span></div><div><h3>${session.review?`Checkpoint ${session.review.number}: inspect the matrices`:phase?.title||'A network before its next lesson.'}</h3><p>${session.review?'This is a saved view. Resuming returns to the live training run; inspecting a checkpoint does not change its weights.':phase?.caption||'Keep every matrix in view. Play an update to watch the error travel backward and the cells change.'}</p></div><span class="movie-direction">${phase?.direction==='backward'?'← BACKPROP':phase?.direction==='update'?'W → W′':'FORWARD →'}</span></div>
    <div class="scene-track" aria-label="Training cycle progress">${PHASES.map((item,index)=>`<span class="${!session.review&&index===active?'current':''} ${!session.review&&index<active?'past':''}" title="${item.title}" ${index===active?'aria-current="step"':''}>${index+1}</span>`).join('')}</div>
    <div class="movie-status" role="status">${player.error||player.notice}</div>
    <div class="update-scorecard">${frame&&(display.phase==='update'||display.phase==='result')?`<span><strong>${frame.changed.toLocaleString()}</strong> parameters changed</span><span>Loss <strong>${f(frame.beforeLoss,4)} → ${f(frame.afterLoss,4)}</strong></span><span>p(${correct}) <strong>${pct(frame.before.softmax[correct])} → ${pct(frame.after.softmax[correct])}</strong></span>`:'<span>One full cycle: predict → measure error → backpropagate → update → predict again.</span>'}</div>
    <div class="movie-wall">${Object.keys(WEIGHT_LAYOUTS).map(key=>weightPanel(key,player,display)).join('')}</div>
    <div class="wall-legend"><span><i class="positive-swatch"></i> Positive · ${getPalette().positiveName.toLowerCase()}</span><span><i class="negative-swatch"></i> Negative · ${getPalette().negativeName.toLowerCase()}</span><span><i class="change-swatch"></i> ${getPalette().changeName} flash · largest changes</span><span>All channels are shown. Click a matrix to magnify its numbers.</span></div>
    <section id="movie-inspector" class="movie-inspector"></section>
    <section class="movie-timeline"><div class="worked-heading"><div><span class="annotation-number">↔</span><h4>Compare before training, after 1, after 10.</h4></div><button class="text-button" id="movie-live">Return to live run</button></div><div class="movie-checkpoints">${checkpoints.map(number=>`<button data-checkpoint="${number}" class="${session.review?.number===number?'selected':''}"><small>${number===0?'START':number===1?'FIRST UPDATE':number===10?'TEN UPDATES':'UPDATE'}</small><strong>${number}</strong></button>`).join('')}</div><div id="movie-loss-history"></div><p class="movie-footnote">Each update is real single-image SGD, run in your browser. One update is not an epoch. ${player.mode==='repeat'?'Repeating one image makes its learning easy to see; it does not measure general digit accuracy.':'The stream cycles through 200 MNIST training images. Each loss pair compares the same image before and after its update.'} Checkpoints 0, 1, 10 and the latest 20 updates are retained.</p></section>
  </div>`;
  document.body.classList.toggle('movie-wide',player.wide);
  heatmap($('#movie-input'),display.tensors.input,28,{gray:true});
  drawWall(app,player);renderInspector(app,player);renderLossHistory(player);
  wireControls(app,player);
}

function selectWeight(app,player,key,index) {
  pauseTrainingMovie(app);player.selection={key,index};player.term=0;player.inspected=true;
  player.notice='Paused for inspection. The outlined cell is selected in all four numeric matrices.';
  app.lesson();
}
function drawWall(app,player) {
  const display=player.session.display(),frame=display.frame,flash=display.phase==='update';
  for(const [key,layout] of Object.entries(WEIGHT_LAYOUTS)) {
    const values=display.weights[key],gradients=frame?.gradients.weights[key]||new Float64Array(values.length);
    const domain=Math.max(rangeMaximum(frame?.beforeWeights[key]||values),rangeMaximum(frame?.afterWeights[key]||values));
    const selected=player.selection.key===key?player.selection.index:-1;
    if(key==='conv1Weight') {
      $$('[data-kernel]').forEach(canvas=>{
        const filter=Number(canvas.dataset.kernel),gradient=canvas.dataset.gradient==='true';
        const data=(gradient?gradients:values).slice(filter*9,filter*9+9);
        paintKernel(canvas,data,{domain:gradient?rangeMaximum(gradients):domain,changes:frame?.delta[key].slice(filter*9,filter*9+9),flash:!gradient&&flash,selected:selected>=filter*9&&selected<filter*9+9?selected%9:-1});
        canvas.parentElement.onclick=()=>selectWeight(app,player,key,filter*9+4);
      });
    } else {
      for(const prefix of ['weights','gradients']) {
        const canvas=$(`#${prefix}-${key}`),isGradient=prefix==='gradients',data=isGradient?gradients:values;
        if(key==='conv2Weight') {
          const mosaic=conv2Mosaic(data),mapped=selected<0?-1:conv2MosaicIndex(selected);
          paintRect(canvas,mosaic,24,48,{domain:isGradient?rangeMaximum(gradients):domain,changes:frame?conv2Mosaic(frame.delta[key]):null,flash:!isGradient&&flash,selected:mapped});
          canvas.tabIndex=0;
          canvas.onclick=event=>{const box=canvas.getBoundingClientRect();const row=Math.min(23,Math.max(0,Math.floor((event.clientY-box.top)/box.height*24))),col=Math.min(47,Math.max(0,Math.floor((event.clientX-box.left)/box.width*48)));selectWeight(app,player,key,conv2WeightIndex(row,col));};
          canvas.onkeydown=event=>{const moves={ArrowLeft:-1,ArrowRight:1,ArrowUp:-3,ArrowDown:3};if(event.key in moves){event.preventDefault();selectWeight(app,player,key,Math.max(0,Math.min(1151,Math.max(0,selected)+moves[event.key])));}};
        } else {
          paintRect(canvas,data,layout.rows,layout.cols,{domain:isGradient?rangeMaximum(gradients):domain,changes:frame?.delta[key],flash:!isGradient&&flash,selected});
          wireRect(canvas,layout.rows,layout.cols,key,app,player);
        }
      }
    }
    const biasCanvas=$(`[data-bias="${layout.bias}"]`),bias=display.weights[layout.bias];
    paintRect(biasCanvas,bias,1,bias.length,{changes:frame?.delta[layout.bias],flash,selected:player.selection.key===layout.bias?player.selection.index:-1});
    wireRect(biasCanvas,1,bias.length,layout.bias,app,player);
  }
  $$('[data-activation]').forEach(canvas=>{
    const key=canvas.dataset.activation,n=Number(canvas.dataset.size),channel=Number(canvas.dataset.channelIndex);
    const data=display.tensors[key].slice(channel*n*n,(channel+1)*n*n);
    // Fixed before/after scale makes activation changes comparable during an update.
    const domain=frame?Math.max(rangeMaximum(frame.before[key]),rangeMaximum(frame.after[key])):rangeMaximum(display.tensors[key]);
    paintRect(canvas,data,n,n,{domain});
  });
  for(const key of ['dense','hidden'])paintRect($(`#movie-${key}`),display.tensors[key],8,16);
  $('#movie-probabilities').innerHTML=Array.from(display.tensors.softmax,(p,i)=>`<div class="movie-prob ${i===display.example.label?'correct':''}"><b>${i}</b><span><i style="width:${p*100}%"></i></span><strong>${pct(p)}</strong></div>`).join('');
  $('#movie-loss-equation').innerHTML=`L = −ln p(${display.example.label})<br><strong>−ln(${display.tensors.softmax[display.example.label].toPrecision(4)}) = ${f(loss(display.tensors.logits,display.example.label),4)}</strong>`;
  const revealed=isGradientVisible(display,'fc2');
  $('#movie-logit-gradient').innerHTML=Array.from({length:10},(_,i)=>`<span class="${i===display.example.label?'correct':''}"><small>${i}</small>${revealed?f(frame.gradients.logits[i],3):'—'}</span>`).join('');
}

function selectionWindow(key,index) {
  const bias=key.endsWith('Bias'),weightKey=key.replace('Bias','Weight'),layout=WEIGHT_LAYOUTS[weightKey];
  if(bias) {
    const start=Math.max(0,Math.min(index-1,layout.rows-3));
    return {indices:Array.from({length:Math.min(3,layout.rows)},(_,i)=>start+i),cols:3,label:`${layout.label} bias [${index}]`};
  }
  if(key.startsWith('conv')) {
    const slice=Math.floor(index/9),inputChannels=key==='conv1Weight'?1:8;
    return {indices:Array.from({length:9},(_,i)=>slice*9+i),cols:3,label:`${layout.label} · filter ${Math.floor(slice/inputChannels)}${inputChannels>1?' · input channel '+slice%inputChannels:''} · cell [${Math.floor(index%9/3)}, ${index%3}]`};
  }
  const row=Math.floor(index/layout.cols),col=index%layout.cols;
  const startRow=Math.max(0,Math.min(row-1,layout.rows-3)),startCol=Math.max(0,Math.min(col-1,layout.cols-3));
  return {indices:Array.from({length:9},(_,i)=>(startRow+Math.floor(i/3))*layout.cols+startCol+i%3),cols:3,label:`${layout.label} · output neuron ${row} ← input ${col} · nearby 3 × 3 window`};
}
const exact=value=>value===0?'0.000000':Math.abs(value)<.000001?value.toExponential(2):f(value,6);
function renderInspector(app,player) {
  const session=player.session,display=session.display(),{key,index}=player.selection;
  const cell=session.cell(key,index),window=selectionWindow(key,index),column=key.replace(/Weight|Bias/,'');
  const ready=isGradientVisible(display,column),updated=display.phase==='update'||display.phase==='result';
  const matrices=[['before','Old weight W',true],['gradient','Gradient ∂L / ∂W',ready],['delta','Change −η × gradient',ready],['after','New weight W′',updated]];
  $('#movie-inspector').innerHTML=`<div class="inspector-heading"><div><span class="context-label">ONE CELL, FOUR VIEWS</span><h3>The numbers inside the colour.</h3><p>${window.label}</p></div><div class="inspector-select"><label>Matrix <select id="inspect-matrix">${Object.entries(WEIGHT_LAYOUTS).flatMap(([weight,layout])=>[[weight,layout.label+' weights'],[layout.bias,layout.label+' biases']]).map(([value,label])=>`<option value="${value}" ${value===key?'selected':''}>${label}</option>`).join('')}</select></label><label>Flat index <input type="number" id="inspect-index" min="0" max="${display.weights[key].length-1}" value="${index}"></label></div></div>
    <div class="numeric-matrices">${matrices.map(([property,title,visible])=>`<figure><figcaption>${title}</figcaption><div class="numeric-grid ${property==='after'&&updated?'numbers-updated':''}" style="grid-template-columns:repeat(${window.cols},minmax(0,1fr))">${window.indices.map(i=>{const value=session.cell(key,i)[property];return `<button data-inspect-cell="${i}" class="${i===index?'selected':''} ${visible&&value<0?'negative':''} ${property==='after'&&updated&&session.cell(key,i).delta!==0?'changed':''}" title="${key}[${i}]${visible?' = '+value:''}">${visible?exact(value):'· · ·'}</button>`;}).join('')}</div><p>${property==='before'?'Values before this update.':property==='gradient'?'How loss changes per unit of weight.':property==='delta'?'Positive: add. Negative: subtract.':'The actual values after SGD.'}</p></figure>`).join('')}</div>
    <div class="selected-equation"><span>${exact(cell.before)}</span><b>−</b><span>${cell.rate}</span><b>×</b><span class="derivative-number">(${ready?exact(cell.gradient):'gradient'})</span><b>=</b><strong>${updated?exact(cell.after):'new weight'}</strong></div>
    <div class="gradient-recipe" id="gradient-recipe"></div>
    <div class="cell-milestones"><span>This same parameter over time</span>${[0,1,10].map(number=>{const checkpoint=session.frames.find(frame=>frame.number===number);const value=number===0?session.baseWeights[key][index]:checkpoint?.afterWeights[key][index];return `<div><small>${number===0?'BEFORE TRAINING':'AFTER '+number+' UPDATE'+(number===1?'':'S')}</small><strong>${value===undefined?'—':exact(value)}</strong></div>`;}).join('')}</div><p class="movie-footnote">Numbers are rounded for display; calculations use full precision. Some gradients are exactly zero because an input is zero or a ReLU gate blocks the signal.</p>`;
  $('#inspect-matrix').onchange=event=>selectWeight(app,player,event.target.value,0);
  $('#inspect-index').onchange=event=>{const value=Number(event.target.value);if(Number.isInteger(value)&&value>=0&&value<display.weights[key].length)selectWeight(app,player,key,value);else event.target.value=index;};
  $$('[data-inspect-cell]').forEach(button=>button.onclick=()=>selectWeight(app,player,key,Number(button.dataset.inspectCell)));
  renderGradientRecipe(app,player,ready);
}
function renderGradientRecipe(app,player,ready) {
  const display=player.session.display(),{key,index}=player.selection;
  const node=$('#gradient-recipe');
  if(!ready){node.innerHTML='<p>The backward pass will reveal where this gradient comes from. Use Next scene to follow it one stage at a time.</p>';return;}
  const terms=gradientTerms(display.frame,key,index),term=Math.min(player.term||0,terms.products.length-1);
  node.innerHTML=`<div><h4>Where did this gradient come from?</h4><p>${terms.note}</p>${terms.size>1?`<canvas id="gradient-contributions" role="img" tabindex="0" aria-label="Every spatial contribution to this weight gradient. Click a cell or use arrow keys."></canvas><span class="contribution-caption">${terms.size} × ${terms.size} local products · click a position</span>`:''}</div><div class="gradient-arithmetic"><span class="context-label">${terms.size>1?'AT OUTPUT POSITION ['+Math.floor(term/terms.size)+', '+term%terms.size+']':'ONE CONNECTION'}</span><div class="operand-pair"><div><small>${terms.leftLabel}</small><strong>${exact(terms.left[term])}</strong></div><b>×</b><div><small>${terms.rightLabel}</small><strong>${exact(terms.right[term])}</strong></div></div><div class="gradient-product">= ${exact(terms.products[term])}</div><div class="gradient-total"><span>${terms.size>1?'Sum of all '+terms.products.length+' products':'Weight gradient'}</span><strong>${exact(terms.total)}</strong></div><p>${terms.size>1?'Positive and negative contributions can cancel. Padding contributes zero.':'This value enters the update equation above.'}</p></div>`;
  if(terms.size>1) {
    const canvas=$('#gradient-contributions');heatmap(canvas,terms.products,terms.size,{x:term%terms.size,y:Math.floor(term/terms.size)});
    const select=position=>{pauseTrainingMovie(app);player.term=position;app.lesson();$('#gradient-contributions').focus({preventScroll:true});};
    canvas.onclick=event=>{const box=canvas.getBoundingClientRect();select(Math.min(terms.size-1,Math.max(0,Math.floor((event.clientY-box.top)/box.height*terms.size)))*terms.size+Math.min(terms.size-1,Math.max(0,Math.floor((event.clientX-box.left)/box.width*terms.size))));};
    canvas.onkeydown=event=>{const moves={ArrowLeft:-1,ArrowRight:1,ArrowUp:-terms.size,ArrowDown:terms.size};if(event.key in moves){event.preventDefault();select(Math.max(0,Math.min(terms.products.length-1,term+moves[event.key])));}};
  }
}
function renderLossHistory(player) {
  const points=player.session.history.slice(-10),max=Math.max(1,...points.flatMap(point=>[point.before,point.after]));
  $('#movie-loss-history').innerHTML=`<div class="loss-history-heading"><h4>Did this update help this image?</h4><span><i></i> Before <i></i> After</span></div>${points.length?`<div class="loss-pairs">${points.map(point=>`<div class="loss-pair"><span>Update ${point.number}<small>digit ${point.label}</small></span><div><i style="width:${point.before/max*100}%"></i><i style="width:${point.after/max*100}%"></i></div><strong>${f(point.before,4)} → ${f(point.after,4)}</strong></div>`).join('')}</div>`:'<p class="movie-footnote">The before-and-after loss for each update will appear here.</p>'}`;
}
function wireControls(app,player) {
  const bind=(id,handler)=>{$(id).onclick=async()=>{try{await handler();}catch(error){pauseTrainingMovie(app);player.loading=false;player.error=error.message;app.lesson();}};};
  const canPlay=()=>players.get(app)===player&&document.body.dataset.page==='learn';
  if(player.loading)$$('.movie-controls button,.movie-controls select,.movie-setup button,.movie-setup select').forEach(control=>control.disabled=true);
  bind('#movie-demo',async()=>{await newRun(app,player,{untrained:true});if(canPlay())play(app,player,1);});
  bind('#movie-play',()=>{if(player.playing){pauseTrainingMovie(app);app.lesson();}else play(app,player,player.session.phase===10?player.session.completed:player.session.completed+1);});
  bind('#movie-ten',()=>play(app,player,player.session.completed<10?10:player.session.completed+10));
  bind('#movie-next',()=>{pauseTrainingMovie(app);advance(app,player);});
  bind('#movie-reset',()=>newRun(app,player,{untrained:true}));
  bind('#expand-movie',()=>{player.wide=!player.wide;app.lesson();});
  bind('#movie-live',()=>{pauseTrainingMovie(app);player.session.review=null;app.lesson();});
  $('#movie-speed').onchange=event=>{player.speed=Number(event.target.value);if(player.playing){clearTimeout(player.timer);schedule(app,player);}};
  const setup=handler=>async event=>{try{handler(event);await newRun(app,player);}catch(error){player.loading=false;player.mode=player.session.mode;player.error=error.message;pauseTrainingMovie(app);app.lesson();}};
  $('#movie-rate').onchange=setup(event=>{player.session.rate=Number(event.target.value);});
  $('#movie-mode').onchange=setup(event=>{player.mode=event.target.value;});
  $('#movie-target').disabled=player.mode==='stream';
  $('#movie-target').onchange=setup(event=>{app.state.target=Number(event.target.value);});
  $$('[data-checkpoint]').forEach(button=>button.onclick=()=>{pauseTrainingMovie(app);player.session.selectCheckpoint(Number(button.dataset.checkpoint));app.lesson();});
}
