import { forward, backward, cloneWeights, sgd, loss } from './engine.js';

export const PHASES = [
  {id:'input', title:'Read the image', column:'input', direction:'forward', caption:'These 784 brightness values are the input. The label tells us what the answer should be.'},
  {id:'conv1', title:'Find small patterns', column:'conv1', direction:'forward', caption:'Eight learned filters produce eight maps. ReLU keeps positive values; pooling keeps local maxima.'},
  {id:'conv2', title:'Combine the patterns', column:'conv2', direction:'forward', caption:'Sixteen filters combine all eight input channels. Each output value sums 72 products and one bias.'},
  {id:'dense', title:'Connect the evidence', column:'fc1', direction:'forward', caption:'The 784 pooled features feed 128 neurons. Every row of the dense matrix produces one weighted sum.'},
  {id:'output', title:'Make a prediction', column:'fc2', direction:'forward', caption:'Ten class scores become probabilities. The highlighted class is the correct label.'},
  {id:'loss', title:'Measure the mistake', column:'fc2', direction:'backward', caption:'Loss is −log of the probability assigned to the correct digit. Higher target probability means lower loss.'},
  {id:'back-output', title:'Send error through the output weights', column:'fc2', direction:'backward', caption:'For each class, the logit gradient is probability minus its one-hot label. Multiplying by each hidden activation gives the weight gradient.'},
  {id:'back-dense', title:'Send error through the dense matrix', column:'fc1', direction:'backward', caption:'Multiply the incoming gradient by each input feature. ReLU blocks gradients where the pre-activation was nonpositive.'},
  {id:'back-conv2', title:'Send error through the second filters', column:'conv2', direction:'backward', caption:'Pooling routes gradients to its saved maxima. Every shared filter weight adds gradient contributions from all its spatial positions.'},
  {id:'back-conv1', title:'Send error through the first filters', column:'conv1', direction:'backward', caption:'The chain rule reaches the first eight filters. Each cell now has a derivative telling us how to change its weight.'},
  {id:'update', title:'Change every weight and bias', column:'all', direction:'update', caption:'New weight = old weight − learning rate × gradient. Outlined cells are changing; a warm flash marks the largest changes in each panel.'},
  {id:'result', title:'Predict again with the new matrices', column:'all', direction:'result', caption:'The same image passes through the updated network. Compare its loss and probability before and after this single update.'},
];
export const WEIGHT_LAYOUTS = {
  conv1Weight: {label:'Conv 1',rows:8,cols:9,bias:'conv1Bias',column:'conv1'},
  conv2Weight: {label:'Conv 2',rows:16,cols:72,bias:'conv2Bias',column:'conv2'},
  fc1Weight: {label:'Dense',rows:128,cols:784,bias:'fc1Bias',column:'fc1'},
  fc2Weight: {label:'Output',rows:10,cols:128,bias:'fc2Bias',column:'fc2'},
};

export function makeTrainingFrame(weights, example, rate, number) {
  const pixels=Float64Array.from(example.pixels,v=>v/255);
  const before=forward(pixels,weights);
  const gradients=backward(before,weights,example.label);
  const afterWeights=sgd(weights,gradients.weights,rate);
  if(Object.values(afterWeights).some(values=>values.some(value=>!Number.isFinite(value))))throw new Error('This step overflowed. Reset the weights and choose a smaller learning rate.');
  const after=forward(pixels,afterWeights);
  if(!Number.isFinite(loss(after.logits,example.label)))throw new Error('This step overflowed. Reset the weights and choose a smaller learning rate.');
  const delta={};
  let changed=0,maximumChange=0;
  for(const key of Object.keys(weights)) {
    delta[key]=Float64Array.from(weights[key],(value,index)=>afterWeights[key][index]-value);
    for(const change of delta[key]){if(change!==0)changed++;maximumChange=Math.max(maximumChange,Math.abs(change));}
  }
  return {number,example:{...example,pixels:Array.from(example.pixels)},rate,beforeWeights:weights,afterWeights,before,after,gradients,delta,changed,maximumChange,
    beforeLoss:loss(before.logits,example.label),afterLoss:loss(after.logits,example.label)};
}

/** UI-independent playback state. A completed frame commits exactly once. */
export class TrainingSession {
  constructor(weights, example, {rate=.05,mode='repeat',trainingSamples=[],snapshot='trained',retention=20}={}) {
    this.baseWeights=cloneWeights(weights);
    this.weights=this.baseWeights;
    this.example={...example,pixels:Array.from(example.pixels)};
    this.rate=rate;this.mode=mode;this.trainingSamples=trainingSamples;this.snapshot=snapshot;
    this.retention=retention;this.frames=[];this.history=[];this.completed=0;
    this.phase=-1;this.frame=null;this.review=null;
    this.initial=forward(Float64Array.from(example.pixels,v=>v/255),this.weights);
  }
  nextExample() {
    if(this.mode==='stream') {
      if(!this.trainingSamples.length)throw new Error('The training stream has not loaded.');
      const sample=this.trainingSamples[this.completed%this.trainingSamples.length];
      return {...sample,split:'train'};
    }
    return this.example;
  }
  advance() {
    this.review=null;
    if(!this.frame || this.phase===PHASES.length-1) {
      this.frame=makeTrainingFrame(this.weights,this.nextExample(),this.rate,this.completed+1);
      this.phase=0;
      return {committed:false,frame:this.frame,phase:this.phase};
    }
    this.phase++;
    let committed=false;
    if(PHASES[this.phase].id==='update') {
      this.weights=this.frame.afterWeights;
      this.completed++;
      this.frames.push(this.frame);
      this.history.push({number:this.frame.number,label:this.frame.example.label,before:this.frame.beforeLoss,after:this.frame.afterLoss});
      const keep=this.frames.filter(frame=>frame.number<=1||frame.number===10||frame.number>this.completed-this.retention);
      this.frames=keep;
      committed=true;
    }
    return {committed,frame:this.frame,phase:this.phase};
  }
  selectCheckpoint(number) {
    if(number===0){this.review={number:0};return;}
    const frame=this.frames.find(frame=>frame.number===number);
    if(!frame)throw new Error('This checkpoint is no longer retained.');
    this.review={number,frame};
  }
  display() {
    if(this.review?.number===0) return {number:0,phase:'initial',tensors:this.initial,weights:this.baseWeights,frame:null,example:this.example};
    if(this.review?.frame)return {number:this.review.number,phase:'result',tensors:this.review.frame.after,weights:this.review.frame.afterWeights,frame:this.review.frame,example:this.review.frame.example};
    if(!this.frame)return {number:0,phase:'initial',tensors:this.initial,weights:this.weights,frame:null,example:this.example};
    const phase=PHASES[this.phase].id,after=phase==='update'||phase==='result';
    return {number:this.frame.number,phase,tensors:after?this.frame.after:this.frame.before,weights:after?this.frame.afterWeights:this.frame.beforeWeights,frame:this.frame,example:this.frame.example};
  }
  cell(key,index) {
    const display=this.display(),frame=display.frame;
    if(!display.weights[key]||!Number.isInteger(index)||index<0||index>=display.weights[key].length)throw new Error('Invalid weight index');
    const before=frame?frame.beforeWeights[key][index]:display.weights[key][index];
    return {before,gradient:frame?frame.gradients.weights[key][index]:0,delta:frame?frame.delta[key][index]:0,after:frame?frame.afterWeights[key][index]:before,rate:frame?.rate??this.rate};
  }
}

/** The operands whose sum is one exact parameter derivative. */
export function gradientTerms(frame,key,index) {
  const column=key.replace(/Weight|Bias/,'');
  const bias=key.endsWith('Bias');
  if(column.startsWith('conv')) {
    const n=column==='conv1'?28:14,channels=column==='conv1'?1:8;
    const out=bias?index:Math.floor(index/(channels*9));
    const inputChannel=bias?0:Math.floor(index/9)%channels;
    const ky=Math.floor(index%9/3),kx=index%3;
    const source=frame.before[column==='conv1'?'input':'pool1'];
    const left=new Float64Array(n*n),right=new Float64Array(n*n);
    for(let y=0;y<n;y++)for(let x=0;x<n;x++) {
      const position=y*n+x,iy=y+ky-1,ix=x+kx-1;
      left[position]=frame.gradients[column][out*n*n+position];
      right[position]=bias?1:iy>=0&&iy<n&&ix>=0&&ix<n?source[inputChannel*n*n+iy*n+ix]:0;
    }
    const products=left.map((value,i)=>value*right[i]);
    return {left,right,products,size:n,total:products.reduce((a,b)=>a+b,0),leftLabel:'upstream gradient',rightLabel:bias?'bias multiplier':'input at shifted position',note:'The same parameter is reused at every output position. Add all these local contributions.'};
  }
  const cols=column==='fc1'?784:128,row=bias?index:Math.floor(index/cols),col=index%cols;
  const left=Float64Array.of(frame.gradients[column==='fc1'?'hidden':'logits'][row]);
  const right=Float64Array.of(bias?1:frame.before[column==='fc1'?'flatten':'hidden'][col]);
  return {left,right,products:Float64Array.of(left[0]*right[0]),size:1,total:left[0]*right[0],leftLabel:column==='fc1'?'gradient after the ReLU gate':'p(class) − one-hot label',rightLabel:bias?'bias multiplier':'input activation',note:column==='fc1'?'ReLU passes the incoming gradient only when this neuron’s pre-activation is positive.':'The class error multiplies this connection’s input activation.'};
}

/** Tile sixteen filters in four columns; each filter has eight 3×3 slices. */
export function conv2MosaicIndex(index) {
  const filter=Math.floor(index/72),channel=Math.floor(index/9)%8,ky=Math.floor(index%9/3),kx=index%3;
  return (Math.floor(filter/4)*6+Math.floor(channel/4)*3+ky)*48+(filter%4)*12+(channel%4)*3+kx;
}
export function conv2WeightIndex(row,col) {
  const filter=Math.floor(row/6)*4+Math.floor(col/12),channel=Math.floor(row%6/3)*4+Math.floor(col%12/3);
  return (filter*8+channel)*9+(row%3)*3+col%3;
}
