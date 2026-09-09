/** Transparent, dependency-free inference for ForwardStep.Forward.
 * Tensor layout: channel, row, column. Dense weights: output, input.
 * Convolution is PyTorch's cross-correlation (the kernel is not flipped).
 */
export const SHAPES = {input:[1,28,28], conv1:[8,28,28], relu1:[8,28,28], pool1:[8,14,14], conv2:[16,14,14], relu2:[16,14,14], pool2:[16,7,7], flatten:[784], dense:[128], hidden:[128], logits:[10], softmax:[10]};
export function cloneWeights(w) { return Object.fromEntries(Object.entries(w).map(([k,v])=>[k, Float64Array.from(v)])); }
export function conv(input, weights, bias, inChannels, outChannels, size) {
  const output = new Float64Array(outChannels * size * size);
  for(let o=0;o<outChannels;o++) for(let y=0;y<size;y++) for(let x=0;x<size;x++) {
    let sum=bias[o];
    for(let c=0;c<inChannels;c++) for(let ky=0;ky<3;ky++) for(let kx=0;kx<3;kx++) {
      const iy=y+ky-1, ix=x+kx-1;
      if(iy>=0 && iy<size && ix>=0 && ix<size) sum+=input[c*size*size+iy*size+ix]*weights[(o*inChannels+c)*9+ky*3+kx];
    }
    output[o*size*size+y*size+x]=sum;
  }
  return output;
}
export function relu(input) { return input.map(v=>Math.max(0,v)); }
export function pool(input, channels, size) {
  const n=size/2, output=new Float64Array(channels*n*n), switches=new Int32Array(output.length);
  for(let c=0;c<channels;c++) for(let y=0;y<n;y++) for(let x=0;x<n;x++) {
    const indices=[c*size*size+2*y*size+2*x,c*size*size+2*y*size+2*x+1,c*size*size+(2*y+1)*size+2*x,c*size*size+(2*y+1)*size+2*x+1];
    const winner=indices.reduce((a,b)=>input[b]>input[a]?b:a);
    const index=c*n*n+y*n+x;
    output[index]=input[winner]; switches[index]=winner;
  }
  return {output,switches};
}
export function linear(input, weights, bias) {
  return Float64Array.from(bias, (b,o)=>{let sum=b; for(let i=0;i<input.length;i++)sum+=input[i]*weights[o*input.length+i]; return sum;});
}
export function softmax(logits) {
  const max=Math.max(...logits), exp=logits.map(v=>Math.exp(v-max)), sum=exp.reduce((a,b)=>a+b,0);
  return exp.map(v=>v/sum);
}
export function forward(pixels, w) {
  if(pixels.length!==784 || Array.from(pixels).some(v=>!Number.isFinite(v))) throw new Error('Expected 784 finite pixels');
  const input=Float64Array.from(pixels), conv1=conv(input,w.conv1Weight,w.conv1Bias,1,8,28), relu1=relu(conv1);
  const p1=pool(relu1,8,28), pool1=p1.output, conv2=conv(pool1,w.conv2Weight,w.conv2Bias,8,16,14), relu2=relu(conv2);
  const p2=pool(relu2,16,14), pool2=p2.output, flatten=pool2.slice();
  const dense=linear(flatten,w.fc1Weight,w.fc1Bias), hidden=relu(dense), logits=linear(hidden,w.fc2Weight,w.fc2Bias);
  return {input,conv1,relu1,pool1,conv2,relu2,pool2,flatten,dense,hidden,logits,softmax:softmax(logits),switch1:p1.switches,switch2:p2.switches};
}
export function convolutionCell(input,w,layer,channel,y,x) {
  const n=layer===1?28:14, channels=layer===1?1:8;
  const weights=w[`conv${layer}Weight`], slices=[];
  let total=w[`conv${layer}Bias`][channel];
  for(let c=0;c<channels;c++) {
    const patch=[],kernel=[],products=[],padding=[];
    for(let ky=0;ky<3;ky++)for(let kx=0;kx<3;kx++) {
      const iy=y+ky-1,ix=x+kx-1,padded=iy<0||iy>=n||ix<0||ix>=n;
      const value=padded?0:input[c*n*n+iy*n+ix],weight=weights[(channel*channels+c)*9+ky*3+kx];
      patch.push(value);kernel.push(weight);products.push(value*weight);padding.push(padded);
    }
    const sum=products.reduce((a,b)=>a+b,0);total+=sum;
    slices.push({patch,kernel,products,padding,sum});
  }
  return {slices,bias:w[`conv${layer}Bias`][channel],total};
}
export function loss(logits, target) {
  if(!Number.isInteger(target)||target<0||target>9)throw new Error('Target must be a digit 0–9');
  const max=Math.max(...logits);
  return Math.log(logits.reduce((s,v)=>s+Math.exp(v-max),0))+max-logits[target];
}
function linearBackward(input, weights, gradient) {
  const dw=new Float64Array(weights.length),dx=new Float64Array(input.length);
  for(let o=0;o<gradient.length;o++)for(let i=0;i<input.length;i++) {
    dw[o*input.length+i]=gradient[o]*input[i];dx[i]+=gradient[o]*weights[o*input.length+i];
  }
  return {dw,db:gradient.slice(),dx};
}
function poolBackward(gradient,switches,length) {
  const dx=new Float64Array(length);gradient.forEach((g,i)=>dx[switches[i]]+=g);return dx;
}
function convBackward(input,weights,gradient,inChannels,outChannels,n) {
  const dx=new Float64Array(input.length),dw=new Float64Array(weights.length),db=new Float64Array(outChannels);
  for(let o=0;o<outChannels;o++)for(let y=0;y<n;y++)for(let x=0;x<n;x++) {
    const g=gradient[o*n*n+y*n+x];db[o]+=g;
    for(let c=0;c<inChannels;c++)for(let ky=0;ky<3;ky++)for(let kx=0;kx<3;kx++) {
      const iy=y+ky-1,ix=x+kx-1;
      if(iy>=0&&iy<n&&ix>=0&&ix<n) {
        const ii=c*n*n+iy*n+ix,wi=(o*inChannels+c)*9+ky*3+kx;
        dw[wi]+=g*input[ii];dx[ii]+=g*weights[wi];
      }
    }
  }
  return {dx,dw,db};
}
export function backward(s,w,target) {
  loss(s.logits,target);
  const dz=s.softmax.slice();dz[target]-=1;
  const fc2=linearBackward(s.hidden,w.fc2Weight,dz),dh=fc2.dx.map((g,i)=>s.dense[i]>0?g:0);
  const fc1=linearBackward(s.flatten,w.fc1Weight,dh),dp2=poolBackward(fc1.dx,s.switch2,s.relu2.length);
  const dz2=dp2.map((g,i)=>s.conv2[i]>0?g:0),c2=convBackward(s.pool1,w.conv2Weight,dz2,8,16,14);
  const dp1=poolBackward(c2.dx,s.switch1,s.relu1.length),dz1=dp1.map((g,i)=>s.conv1[i]>0?g:0);
  const c1=convBackward(s.input,w.conv1Weight,dz1,1,8,28);
  return {weights:{conv1Weight:c1.dw,conv1Bias:c1.db,conv2Weight:c2.dw,conv2Bias:c2.db,fc1Weight:fc1.dw,fc1Bias:fc1.db,fc2Weight:fc2.dw,fc2Bias:fc2.db},input:c1.dx,logits:dz,hidden:dh,conv1:dz1,conv2:dz2,loss:loss(s.logits,target)};
}
export function sgd(w,gradients,rate) {
  if(!Number.isFinite(rate)||rate<=0||rate>0.1)throw new Error('Learning rate must be in (0, 0.1]');
  return Object.fromEntries(Object.entries(w).map(([key,v])=>[key,Float64Array.from(v,(x,i)=>x-rate*gradients[key][i])]));
}
