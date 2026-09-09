import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { productTrace, groupedContributions, softmaxTrace, chainForOutputWeight } from '../web/math-trace.js';
import { forward, convolutionCell, backward } from '../web/engine.js';
const model=JSON.parse(readFileSync(new URL('../web/assets/model.json',import.meta.url))).trained;
const samples=JSON.parse(readFileSync(new URL('../web/assets/samples.json',import.meta.url)));
const input=Float64Array.from(samples[0].pixels,v=>v/255);
const tensors=forward(input,model);
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-9,`${a} != ${b}`);

test('Worked convolution terms accumulate in matrix order and include each padded zero',()=>{
  for(const [layer,channel,y,x] of [[1,0,0,0],[1,7,12,16],[2,15,0,13],[2,4,5,7]]) {
    const cell=convolutionCell(tensors[layer===1?'input':'pool1'],model,layer,channel,y,x);
    let channelSum=0;
    for(const slice of cell.slices) {
      const trace=productTrace(slice.patch,slice.kernel);
      assert.equal(trace.terms.length,9);
      trace.terms.forEach((term,i)=>{
        assert.equal(term.index,i);
        near(term.product,slice.products[i]);
        near(term.previous+(slice.products[i]),term.sum);
        if(slice.padding[i])assert.equal(term.input,0);
      });
      near(trace.subtotal,slice.sum);
      channelSum+=trace.subtotal;
    }
    near(channelSum+cell.bias,cell.total);
  }
});

test('Dense fan-in and signed contribution groups account for the complete neuron',()=>{
  for(const [key,inputKey,count,target] of [['fc1','flatten',784,127],['fc2','hidden',128,7]]) {
    const values=tensors[inputKey],weights=model[key+'Weight'].slice(target*count,(target+1)*count),bias=model[key+'Bias'][target];
    const trace=productTrace(values,weights,bias);
    near(trace.total,tensors[key==='fc1'?'dense':'logits'][target]);
    const groups=groupedContributions(values,weights,10);
    assert.equal(groups.length,11);
    assert.equal(groups.at(-1).label,`${count-10} others`);
    near(groups.reduce((sum,g)=>sum+g.value,0)+bias,trace.total);
  }
});

test('Every visual softmax column matches the inference calculation',()=>{
  const trace=softmaxTrace(tensors.logits);
  trace.shifted.forEach((v,i)=>near(v,tensors.logits[i]-trace.max));
  trace.exponentials.forEach((v,i)=>near(v,Math.exp(trace.shifted[i])));
  trace.probabilities.forEach((v,i)=>near(v,tensors.softmax[i]));
  near(trace.probabilities.reduce((sum,v)=>sum+v,0),1);
  const extreme=softmaxTrace([10000,9999,-10000]);
  assert.ok(extreme.probabilities.every(Number.isFinite));
});

test('The worked chain rule gives the same output-weight gradient as full backpropagation',()=>{
  const target=7,gradient=backward(tensors,model,target);
  for(const klass of [0,target,9])for(const neuron of [0,37,127]) {
    const index=klass*128+neuron;
    const chain=chainForOutputWeight(tensors.hidden[neuron],tensors.softmax[klass],model.fc2Weight[index],klass===target);
    near(chain.weightGradient,gradient.weights.fc2Weight[index]);
    near(chain.outputDerivative,gradient.logits[klass]);
  }
});

test('Worked dot products reject mismatched dimensions and non-finite terms',()=>{
  assert.throws(()=>productTrace([1],[1,2]));
  assert.throws(()=>productTrace([Infinity],[2]));
});
