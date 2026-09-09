import { convolutionCell } from './engine.js';
import { productTrace, groupedContributions, softmaxTrace } from './math-trace.js';
import { $, $$, f, pct, heatmap, mapEvents, options, color } from './render.js';

const termMatrix = (values, title, kind, padding=[]) => `<div class="worked-matrix"><p class="matrix-title">${title}</p><div class="matrix" style="grid-template-columns:repeat(3,1fr)">${values.map((value,index)=>`<button class="matrix-cell ${value<0?'negative':value===0?'zero':''} ${padding[index]?'padded':''}" data-term="${index}" data-kind="${kind}" aria-label="${title}, row ${Math.floor(index/3)}, column ${index%3}, value ${value}"><small>${Math.floor(index/3)},${index%3}</small><span>${f(value,3)}</span></button>`).join('')}</div></div>`;

function accumulationSVG(trace, reveal, selected) {
  const values = [0,...trace.terms.map(term=>term.sum)];
  const min = Math.min(...values), max = Math.max(...values), span = max-min || 1;
  const y = value => 102-(value-min)/span*70;
  const x = i=>32+i*67;
  return `<svg class="accumulation-chart" viewBox="0 0 670 142" role="img" aria-label="Running sum of the nine products in row-major order"><path d="M24 ${y(0)}H650" stroke="#c9c4b8" stroke-dasharray="4 4"/><text x="13" y="${y(0)+4}" class="chart-zero">0</text><polyline points="${values.slice(0,reveal+1).map((value,i)=>`${x(i)},${y(value)}`).join(' ')}" fill="none" stroke="#a63b32" stroke-width="2"/>${values.map((value,i)=>`<g opacity="${i<=reveal?1:.23}"><circle cx="${x(i)}" cy="${y(value)}" r="${i===selected+1?5:3}" fill="${i===selected+1?'#292622':'#a63b32'}"/><text x="${x(i)}" y="${y(value)-13}" text-anchor="middle">${f(value,3)}</text><text x="${x(i)}" y="131" text-anchor="middle">${i?'+'+i:'start'}</text></g>`).join('')}</svg>`;
}

export function renderWorkedConvolution(app) {
  const s=app.state,layer=s.step===1?1:2,n=layer===1?28:14,channels=layer===1?8:16;
  s.channel=Math.min(s.channel,channels-1);s.x=Math.min(s.x,n-1);s.y=Math.min(s.y,n-1);
  const input=s.tensors[layer===1?'input':'pool1'];
  const output=s.tensors[`conv${layer}`];
  const inputChannel=layer===1?0:s.inputChannel;
  const calc=convolutionCell(input,s.weights,layer,s.channel,s.y,s.x);
  const slice=calc.slices[inputChannel],trace=productTrace(slice.patch,slice.kernel);
  const reveal=s.reveal??9;
  s.term=Math.min(8,s.term??4);
  $('#lesson-work').innerHTML=`
    <div class="work-toolbar"><h4>A single output at row ${s.y}, column ${s.x}</h4><div class="controls"><label>Output filter <select id="filter-select">${options(channels,s.channel)}</select></label>${layer===2?`<label>Input channel <select id="input-channel">${options(8,inputChannel)}</select></label>`:''}<button class="button" id="scan-kernel">${s.scanning?'Pause scan':'Slide the filter →'}</button></div></div>
    <div class="convolution-flow">
      <figure><figcaption><b>A.</b> ${layer===1?'The input image':'Input channel '+inputChannel}<span>${n} × ${n}</span></figcaption><canvas id="conv-source" class="heatmap-large" aria-label="Select a patch center in the input feature map"></canvas><p>The outlined 3 × 3 patch supplies nine values.</p></figure>
      <div class="flow-arrow" aria-hidden="true">→</div>
      <figure class="kernel-figure"><figcaption><b>B.</b> Learned filter<span>3 × 3</span></figcaption>${termMatrix(slice.kernel,'W[u, v]','mini-weight')}<p>These weights are reused at every position.</p></figure>
      <div class="flow-arrow" aria-hidden="true">→</div>
      <figure><figcaption><b>C.</b> Output channel ${s.channel}<span>${n} × ${n}</span></figcaption><canvas id="feature-focus" class="heatmap-large" aria-label="Select the convolution output cell to inspect"></canvas><p>The marked cell is <strong>${f(calc.total,5)}</strong>.</p></figure>
    </div>
    <div class="work-position"><span>Move by clicking either map or using its arrow keys.</span><div class="map-coordinates"><label>Row <input id="row-input" type="number" min="0" max="${n-1}" value="${s.y}"></label><label>Column <input id="col-input" type="number" min="0" max="${n-1}" value="${s.x}"></label></div></div>
    <div class="worked-heading"><div><span class="annotation-number">01</span><h4>Multiply matching cells</h4></div><p>Select a cell to follow the same term across all three matrices.</p></div>
    <div class="worked-equation">${termMatrix(slice.patch,'INPUT PATCH · X', 'input',slice.padding)}<span class="math-op">⊙</span>${termMatrix(slice.kernel,'FILTER · W','weight')}<span class="math-op">=</span>${termMatrix(slice.products,'ELEMENTWISE PRODUCTS','product')}</div>
    <div class="term-equation" id="term-equation" aria-live="polite"></div>
    <div class="worked-heading"><div><span class="annotation-number">02</span><h4>Add the nine products</h4></div><div class="controls"><button id="replay-products" class="button">${s.calculating?'Pause arithmetic':'Replay arithmetic'}</button><button id="advance-product" class="button" ${reveal===9?'disabled':''}>Next term →</button></div></div>
    <div class="product-strip">${trace.terms.map(term=>`<button data-term="${term.index}" data-kind="strip" class="product-term ${term.index<reveal?'revealed':''}"><small>TERM ${term.index+1}</small><span>${term.index<reveal?f(term.product,4):'—'}</span></button>`).join('')}</div>
    <div class="arithmetic-progress"><label for="product-progress">Products included</label><input id="product-progress" type="range" min="0" max="9" value="${reveal}"><span>${reveal} / 9</span><strong>Σ = ${f(reveal?trace.terms[reveal-1].sum:0,5)}</strong></div>
    <div class="trace-plot">${accumulationSVG(trace,reveal,s.term)}</div>
    ${layer===2?`<div class="worked-heading"><div><span class="annotation-number">03</span><h4>Combine all eight channels</h4></div><p>Each channel contributes its own sum of nine products.</p></div><div class="channel-ledger">${calc.slices.map((channel,i)=>`<button data-slice="${i}" class="${i===inputChannel?'selected':''}"><small>CHANNEL ${i}</small><strong>${f(channel.sum,5)}</strong><span>9 products</span></button>`).join('')}</div>`:''}
    <div class="bias-equation"><div><span class="context-label">${layer===1?'ALL NINE PRODUCTS':'ALL EIGHT CHANNEL SUMS'}</span><strong>${f(calc.total-calc.bias,5)}</strong></div><span>+</span><div><span class="context-label">ONE LEARNED BIAS</span><strong>${f(calc.bias,5)}</strong></div><span>=</span><div class="final-answer"><span class="context-label">FINAL OUTPUT z[${s.channel}, ${s.y}, ${s.x}]</span><strong>${f(calc.total,5)}</strong></div></div>
    <p class="math-note">The final output includes every term, even when the replay above is paused partway through. ${layer===2?'The replay examines the selected input channel; its subtotal joins the other seven above.':''} Numeric labels are rounded. Red encodes positive values and blue negative values; the black outline marks a selection.</p>`;
  const sourceMap=input.slice(inputChannel*n*n,(inputChannel+1)*n*n);
  heatmap($('#conv-source'),sourceMap,n,{gray:layer===1,x:s.x,y:s.y,patch:true});
  heatmap($('#feature-focus'),output.slice(s.channel*n*n,(s.channel+1)*n*n),n,{x:s.x,y:s.y});
  for(const id of ['#conv-source','#feature-focus'])mapEvents($(id),n,(x,y)=>{app.stopCalculation();s.reveal=9;app.focus(x,y);});
  $('#filter-select').onchange=e=>{app.stopCalculation();s.reveal=9;app.channel(Number(e.target.value));};
  const chooseSlice=i=>{app.stopCalculation();s.inputChannel=i;s.reveal=9;app.lesson();};
  if(layer===2)$('#input-channel').onchange=e=>chooseSlice(Number(e.target.value));
  $$('[data-slice]').forEach(button=>button.onclick=()=>chooseSlice(Number(button.dataset.slice)));
  $('#row-input').onchange=e=>{app.stopCalculation();app.focus(s.x,Math.max(0,Math.min(n-1,Math.trunc(Number(e.target.value)||0))));};
  $('#col-input').onchange=e=>{app.stopCalculation();app.focus(Math.max(0,Math.min(n-1,Math.trunc(Number(e.target.value)||0))),s.y);};
  $('#scan-kernel').onclick=()=>{app.stopCalculation();s.reveal=9;app.scan(n);};
  $('#replay-products').onclick=()=>app.replayProducts();
  $('#advance-product').onclick=()=>{app.stopCalculation();s.reveal=Math.min(9,(s.reveal??0)+1);s.term=Math.max(0,s.reveal-1);app.lesson();};
  $('#product-progress').oninput=e=>{app.stopCalculation();const value=Number(e.target.value);$('#product-progress').nextElementSibling.textContent=`${value} / 9`;};
  $('#product-progress').onchange=e=>{s.reveal=Number(e.target.value);s.term=Math.max(0,s.reveal-1);app.lesson();};
  function selectTerm(index) {
    s.term=index;
    $('.trace-plot').innerHTML=accumulationSVG(trace,s.reveal??9,index);
    $$('[data-term]').forEach(button=>{const selected=Number(button.dataset.term)===index;button.classList.toggle('term-selected',selected);button.setAttribute('aria-pressed',String(selected));});
    const term=trace.terms[index],row=Math.floor(index/3),col=index%3;
    $('#term-equation').innerHTML=`<span class="term-reference">TERM ${index+1} · [${row}, ${col}]</span><span class="input-ink">${f(term.input,5)}</span><span>×</span><span class="weight-ink">${f(term.weight,5)}</span><span>=</span><strong>${f(term.product,6)}</strong>${slice.padding[index]?'<small>This input is a padded zero outside the image.</small>':''}`;
    heatmap($('#conv-source'),sourceMap,n,{gray:layer===1,x:s.x,y:s.y,patch:true});
    const px=s.x+col-1,py=s.y+row-1;
    if(px>=0&&py>=0&&px<n&&py<n){const context=$('#conv-source').getContext('2d');context.strokeStyle='#d8a526';context.lineWidth=2.4;context.strokeRect(px*8+1.2,py*8+1.2,5.6,5.6);}
  }
  $$('[data-term]').forEach(button=>{button.onclick=()=>selectTerm(Number(button.dataset.term));button.onfocus=()=>selectTerm(Number(button.dataset.term));});
  selectTerm(s.term);
}

function fanInSVG(input, weights, output, bias, selected, target) {
  const start=Math.max(0,Math.min(input.length-6,selected-2));
  const rows=Array.from({length:6},(_,i)=>start+i);
  const scale=Math.max(...rows.map(i=>Math.abs(input[i]*weights[i])),.001);
  return `<svg class="fan-in-svg" viewBox="0 0 860 350" role="img" aria-label="Six actual connections into neuron ${target}; the output sum includes all ${input.length} connections"><text x="70" y="20">INPUT ACTIVATION</text><text x="270" y="20">WEIGHT</text><text x="433" y="20">PRODUCT</text><text x="680" y="20">ALL ${input.length} TERMS + BIAS</text>${rows.map((i,j)=>{const y=55+j*48,product=input[i]*weights[i],active=i===selected;return `<g data-connection="${i}" class="connection-line ${active?'chosen':''}" tabindex="0" role="button" aria-label="Inspect connection ${i}"><path d="M164 ${y}H402 M515 ${y}L638 178" fill="none" stroke="${product<0?'#315d7d':'#a63b32'}" stroke-width="${active?3:1+Math.abs(product)/scale*1.5}" opacity="${active?1:.55}"/><rect x="12" y="${y-16}" width="150" height="33" fill="${active?'#ece8df':'#fffefa'}" stroke="#cec9bd"/><text x="21" y="${y+5}">x[${i}]  ${f(input[i],3)}</text><rect x="248" y="${y-12}" width="95" height="25" fill="#fffefa"/><text x="259" y="${y+5}" fill="#315d7d">× ${f(weights[i],4)}</text><rect x="402" y="${y-16}" width="110" height="33" fill="${product<0?'#e2eaf0':'#f3e2dc'}" stroke="${active?'#292622':'#cec9bd'}"/><text x="417" y="${y+5}">${f(product,5)}</text></g>`;}).join('')}<circle cx="661" cy="178" r="28" fill="#f5f3ed" stroke="#292622"/><text x="651" y="187" class="sigma-symbol">Σ</text><path d="M689 178H732" stroke="#292622"/><rect x="731" y="154" width="120" height="48" fill="#f3e2dc" stroke="#a63b32"/><text x="745" y="184" class="fan-result">${f(output,5)}</text><text x="599" y="236">bias ${f(bias,5)}</text><text x="25" y="341">Shown: inputs ${start}–${start+5}. The remaining ${input.length-6} connections are included in the output.</text></svg>`;
}

function contributionBars(groups,bias) {
  const max=Math.max(...groups.map(g=>Math.abs(g.value)),Math.abs(bias),.001);
  return `<div class="contribution-bars"><div class="contribution-key"><span>← Decreases the score</span><span>Increases the score →</span></div>${[...groups,{label:'bias',value:bias}].map(group=>`<div class="contribution-bar-row"><span>${group.label}</span><div class="signed-track"><i style="left:${group.value<0?50-Math.abs(group.value)/max*50:50}%;width:${Math.abs(group.value)/max*50}%;background:${group.value<0?'#315d7d':'#a63b32'}"></i></div><strong>${f(group.value,5)}</strong></div>`).join('')}</div>`;
}

export function enhanceDense(app) {
  const s=app.state,isLogit=s.step===9,target=isLogit?s.classIndex:s.neuron;
  const input=s.tensors[isLogit?'hidden':'flatten'],n=input.length;
  const weights=s.weights[isLogit?'fc2Weight':'fc1Weight'].slice(target*n,(target+1)*n);
  const bias=s.weights[isLogit?'fc2Bias':'fc1Bias'][target];
  const trace=productTrace(input,weights,bias);
  s.connection=Math.max(0,Math.min(n-1,s.connection??0));
  const selected=trace.terms[s.connection];
  const work=$('#lesson-work');
  const table=$('.table-scroll',work);
  if(table){const detail=document.createElement('section');detail.className='all-connections';detail.innerHTML=`<h4>All ${n} connections</h4>`;table.replaceWith(detail);detail.append(table);}
  const grid=$('.dense-grid',work);
  if(grid){const detail=document.createElement('section');detail.className='all-neurons';detail.innerHTML='<h4>Choose from all 128 hidden neurons</h4>';grid.replaceWith(detail);detail.append(grid);}
  const diagram=document.createElement('section');diagram.className='neuron-worked';
  diagram.innerHTML=`<div class="worked-heading"><div><span class="annotation-number">01</span><h4>Follow an actual connection</h4></div><label class="connection-control">Input index <input id="connection-index" type="number" min="0" max="${n-1}" value="${s.connection}"></label></div><div class="fan-scroll">${fanInSVG(input,weights,trace.total,bias,s.connection,target)}</div><div class="term-equation"><span class="term-reference">CONNECTION ${s.connection} → ${target}</span><span>${f(selected.input,5)}</span><span>×</span><span class="weight-ink">${f(selected.weight,5)}</span><span>=</span><strong>${f(selected.product,6)}</strong></div><div class="worked-heading"><div><span class="annotation-number">02</span><h4>See what pushes the score up or down</h4></div><p>Ten largest contributions; the rest are added together.</p></div>${contributionBars(groupedContributions(input,weights,10),bias)}<div class="bias-equation"><div><span class="context-label">SUM OF ${n} PRODUCTS</span><strong>${f(trace.subtotal,5)}</strong></div><span>+</span><div><span class="context-label">BIAS</span><strong>${f(bias,5)}</strong></div><span>=</span><div class="final-answer"><span class="context-label">PRE-ACTIVATION</span><strong>${f(trace.total,5)}</strong></div>${isLogit?'':`<span>→</span><div><span class="context-label">AFTER RELU</span><strong>${f(Math.max(0,trace.total),5)}</strong></div>`}</div>`;
  $('.work-toolbar',work).after(diagram);
  $('#connection-index').onchange=e=>{s.connection=Math.max(0,Math.min(n-1,Math.trunc(Number(e.target.value)||0)));app.lesson();};
  $$('[data-connection]').forEach(node=>{const choose=()=>{s.connection=Number(node.dataset.connection);app.lesson();};node.onclick=choose;node.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}};});
  // The old shifted bar chart obscured negative logits; the signed contribution
  // plot now carries the visual explanation, while exact scores stay in Softmax.
  if(isLogit){$('.chart-bars',work)?.remove();$('.map-caption',work)?.remove();}
}

export function renderWorkedSoftmax(app) {
  const s=app.state,z=s.tensors.logits,trace=softmaxTrace(z),selected=s.classIndex;
  const columns=[
    {title:'1. Raw score',formula:'zᵢ',values:Array.from(z),kind:'logit'},
    {title:'2. Subtract max',formula:`zᵢ − ${f(trace.max,3)}`,values:trace.shifted,kind:'shift'},
    {title:'3. Exponentiate',formula:'exp(zᵢ − max)',values:trace.exponentials,kind:'exp'},
    {title:'4. Divide by the sum',formula:`exp(…) / ${f(trace.denominator,4)}`,values:trace.probabilities,kind:'prob'},
  ];
  $('#lesson-work').innerHTML=`<div class="work-toolbar"><h4>One transformation per column</h4><label>Follow digit <select id="softmax-class">${options(10,selected)}</select></label></div><div class="softmax-pipeline">${columns.map((column,c)=>`<div class="softmax-column"><h4>${column.title}</h4><p class="mono">${column.formula}</p>${column.values.map((v,i)=>`<button data-softmax-digit="${i}" class="softmax-cell ${i===selected?'selected':''}"><small>${i}</small><span>${column.kind==='prob'?pct(v):column.kind==='exp'&&v<.0001?v.toExponential(2):f(v,4)}</span>${column.kind==='prob'?`<i style="width:${v*100}%"></i>`:''}</button>`).join('')}<p class="column-foot">${c===0?'Scores can have either sign.':c===1?'The largest becomes exactly zero.':c===2?`All values are positive. Σ = ${f(trace.denominator,5)}`:'All ten probabilities add to 100%.'}</p></div>`).join('')}</div><div class="worked-heading"><div><span class="annotation-number">→</span><h4>Follow the highlighted row: digit ${selected}</h4></div></div><div class="softmax-fraction"><span>p<sub>${selected}</sub> =</span><math xmlns="http://www.w3.org/1998/Math/MathML" aria-label="Exponential of selected logit minus maximum, divided by the sum of exponentials"><mfrac><mrow><mi>exp</mi><mo>(</mo><mn>${f(z[selected],4)}</mn><mo>−</mo><mn>${f(trace.max,4)}</mn><mo>)</mo></mrow><mn>${f(trace.denominator,6)}</mn></mfrac></math><span>=</span><strong>${pct(trace.probabilities[selected])}</strong></div><div class="probability-whole" role="img" aria-label="The ten class probabilities partition one hundred percent">${trace.probabilities.map((value,i)=>`<span style="width:${value*100}%;background:${i===selected?'#a63b32':i%2?'#bab3a5':'#797366'}" title="Digit ${i}: ${pct(value)}">${value>.07?i:''}</span>`).join('')}</div><p class="math-note">Click any row to follow a different digit. The maximum is subtracted from every score, so it cancels out in the fraction. It changes numerical stability, not the predicted probabilities.</p>`;
  const choose=index=>{s.classIndex=index;app.lesson();};
  $('#softmax-class').onchange=e=>choose(Number(e.target.value));
  $$('[data-softmax-digit]').forEach(button=>button.onclick=()=>choose(Number(button.dataset.softmaxDigit)));
}

export function enhanceRelu(app) {
  const s=app.state,layer=s.step===2?1:2,n=layer===1?28:14;
  const x=s.tensors[`conv${layer}`][s.channel*n*n+s.y*n+s.x],y=Math.max(0,x);
  const extent=Math.max(1,Math.abs(x)*1.35);
  const px=v=>200+v/extent*155,py=v=>143-v/extent*108;
  const plot=document.createElement('div');plot.className='relu-function';
  plot.innerHTML=`<div><p class="eyebrow">THE FUNCTION, AT YOUR SELECTED VALUE</p><h4>ReLU(x) = max(0, x)</h4><p>${x<0?'The selected point lies on the flat segment. The output is zero.':x>0?'The selected point lies on the diagonal. The output equals the input.':'The selected point is at the corner. PyTorch uses derivative zero here.'}</p><p class="mono">Local derivative: <strong>${x>0?1:0}</strong></p></div><svg viewBox="0 0 430 290" role="img" aria-label="ReLU graph with selected input ${f(x,4)} and output ${f(y,4)}"><path d="M30 143H398 M200 270V15" stroke="#cec9bd"/><path d="M45 143H200L355 35" fill="none" stroke="#a63b32" stroke-width="3"/><path d="M${px(x)} 143V${py(y)}H200" fill="none" stroke="#315d7d" stroke-dasharray="5 4"/><circle cx="${px(x)}" cy="${py(y)}" r="6" fill="#292622"/><text x="${Math.min(340,Math.max(35,px(x)+9))}" y="${Math.max(20,py(y)-15)}">(${f(x,3)}, ${f(y,3)})</text><text x="385" y="165">x</text><text x="210" y="22">ReLU(x)</text><text x="207" y="160">0</text><text x="35" y="278">Left: slope 0. Right: slope 1.</text></svg>`;
  $('.value-callout',$('#lesson-work')).after(plot);
}

export function enhancePooling(app) {
  const s=app.state,layer=s.step===3?1:2,n=layer===1?28:14;
  const start=s.channel*n*n+2*s.y*n+2*s.x,input=s.tensors[`relu${layer}`];
  const values=[input[start],input[start+1],input[start+n],input[start+n+1]],winner=values.indexOf(Math.max(...values));
  const panel=document.createElement('section');panel.className='pool-routing';
  panel.innerHTML=`<div><p class="eyebrow">FORWARD VALUE / BACKWARD DERIVATIVE</p><h4>The winning position is remembered.</h4><p>If a gradient <i>g</i> arrives from the next layer, it returns only to position [${Math.floor(winner/2)}, ${winner%2}] inside this window.</p></div><div class="pool-switch">${values.map((v,i)=>`<div class="${i===winner?'winner':''}"><span>${f(v,4)}</span><small>${i===winner?'← receives g':'receives 0'}</small></div>`).join('')}</div><span class="routing-arrow">← <i>g</i></span>`;
  $('#lesson-work').append(panel);
}
