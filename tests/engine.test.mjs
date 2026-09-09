import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {forward,cloneWeights,convolutionCell,pool,softmax,loss,backward,sgd,SHAPES} from '../web/engine.js';
const model=JSON.parse(readFileSync(new URL('../web/assets/model.json',import.meta.url)));
const samples=JSON.parse(readFileSync(new URL('../web/assets/samples.json',import.meta.url)));
const fixtures=JSON.parse(readFileSync(new URL('./pytorch-reference.json',import.meta.url)));
const pixels=s=>Float64Array.from(s.pixels,v=>v/255);
const near=(actual,expected,tolerance=1e-5)=>assert.ok(Math.abs(actual-expected)<=tolerance,`${actual} != ${expected} (tolerance ${tolerance})`);
for(const fixture of fixtures) test(`PyTorch parity: ${fixture.snapshot} checkpoint, test sample ${fixture.sampleId}`,()=>{
  const sample=samples.find(s=>s.id===fixture.sampleId),result=forward(pixels(sample),model[fixture.snapshot]);
  for(const [key,expected] of Object.entries(fixture.stages)) {
    assert.equal(result[key].length,expected.length,key);
    assert.equal(result[key].length,SHAPES[key].reduce((a,b)=>a*b),key+' shape');
    let error=0;result[key].forEach((v,i)=>error=Math.max(error,Math.abs(v-expected[i])));
    assert.ok(error<3e-5,`${key}: maximum difference ${error}`);
  }
});
test('Inspectors include zero padding and every Conv2 input channel, with one bias',()=>{
  const s=forward(pixels(samples[0]),model.trained);
  for(const layer of [1,2])for(const channel of [0,layer===1?7:15])for(const [x,y] of [[0,0],[3,5],[layer===1?27:13,layer===1?27:13]]) {
    const n=layer===1?28:14,cell=convolutionCell(s[layer===1?'input':'pool1'],model.trained,layer,channel,y,x);
    near(cell.total,s[`conv${layer}`][channel*n*n+y*n+x],1e-12);
    assert.equal(cell.slices.length,layer===1?1:8);
    if(x===0&&y===0)assert.equal(cell.slices[0].padding.filter(Boolean).length,5);
  }
});
test('Max pool preserves channels, downsamples, and routes ties to the first maximum',()=>{
  const p=pool(Float64Array.of(1,4,4,2,8,1,0,7),2,2);
  assert.deepEqual(Array.from(p.output),[4,8]);assert.deepEqual(Array.from(p.switches),[1,4]);
});
test('Stable softmax and cross-entropy remain finite for extreme logits',()=>{
  const z=Float64Array.from([10000,9999,9998,-10000,-9999,0,1,2,3,4]);
  const p=softmax(z);near(p.reduce((a,b)=>a+b),1,1e-14);
  assert.ok(p.every(Number.isFinite));near(loss(z,0),-Math.log(p[0]),1e-12);
  const shifted=softmax(z.map(v=>v+500));p.forEach((v,i)=>near(v,shifted[i],1e-14));
  assert.ok(Number.isFinite(loss(z,3)));
});
// Tiny deterministic input perturbations avoid max-pool ties, where central
// finite differences need not match the chosen subgradient.
test('Full backward pass agrees with finite differences in every weight and bias tensor',()=>{
  const w=cloneWeights(model.trained),input=pixels(samples[1]).map((v,i)=>v+0.001*(1+Math.sin(i*1.2345))),target=3,s=forward(input,w),g=backward(s,w,target),epsilon=1e-7;
  for(const [key,values] of Object.entries(w)) {
    const indices=[0,Math.floor(values.length/2),values.length-1,g.weights[key].reduce((best,v,i)=>Math.abs(v)>Math.abs(g.weights[key][best])?i:best,0)];
    for(const i of new Set(indices)) {
      const saved=values[i];values[i]=saved+epsilon;const plus=loss(forward(input,w).logits,target);
      values[i]=saved-epsilon;const minus=loss(forward(input,w).logits,target);values[i]=saved;
      near(g.weights[key][i],(plus-minus)/(2*epsilon),1e-4);
    }
  }
  for(const i of [185,325,430]) {
    const old=input[i];input[i]=old+epsilon;const plus=loss(forward(input,w).logits,target);
    input[i]=old-epsilon;const minus=loss(forward(input,w).logits,target);input[i]=old;
    near(g.input[i],(plus-minus)/(2*epsilon),1e-4);
  }
});
test('A small real SGD step reduces the chosen-image loss without mutating the checkpoint',()=>{
  const w=cloneWeights(model.initial),input=pixels(samples[0]),s=forward(input,w),g=backward(s,w,samples[0].label),before=loss(s.logits,samples[0].label);
  const updated=sgd(w,g.weights,.001),after=loss(forward(input,updated).logits,samples[0].label);
  assert.ok(after<before,`${after} must be below ${before}`);
  assert.deepEqual(w.conv1Weight,Float64Array.from(model.initial.conv1Weight));
  near(g.logits.reduce((a,b)=>a+b),0,1e-12);
});
test('Model provenance, parameter count, and examples stay consistent',()=>{
  assert.equal(Object.values(model.trained).reduce((s,w)=>s+w.length,0),103018);
  assert.equal(model.meta.parameters,103018);assert.equal(model.meta.trainingExamples,60000);assert.equal(model.meta.testExamples,10000);
  for(let i=0;i<10;i++)assert.equal(samples.filter(s=>s.label===i).length,5);
  assert.ok(samples.every(s=>s.pixels.length===784&&s.pixels.every(v=>Number.isInteger(v)&&v>=0&&v<=255)));
  assert.equal(new Set(samples.map(s=>s.id)).size,samples.length);
});
test('Malformed inference and training inputs fail explicitly',()=>{
  assert.throws(()=>forward([1,2],model.trained));
  assert.throws(()=>loss(new Float64Array(10),-1));
  assert.throws(()=>sgd(model.trained,{},-0.01));
});
