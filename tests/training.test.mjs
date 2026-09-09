import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { TrainingSession, makeTrainingFrame, gradientTerms, PHASES, conv2MosaicIndex, conv2WeightIndex } from '../web/training-session.js';
import { cloneWeights, forward, backward, sgd, loss } from '../web/engine.js';
const model=JSON.parse(readFileSync(new URL('../web/assets/model.json',import.meta.url)));
const samples=JSON.parse(readFileSync(new URL('../web/assets/samples.json',import.meta.url)));
const stream=JSON.parse(readFileSync(new URL('../web/assets/training-samples.json',import.meta.url)));
const example={...samples.find(s=>s.label===7),split:'test'};
const near=(actual,expected,tolerance=1e-12)=>assert.ok(Math.abs(actual-expected)<tolerance,`${actual} != ${expected}`);
const complete=session=>{for(let i=0;i<PHASES.length;i++)session.advance();};

test('Playback commits exactly at the update scene, never during forward/backward',()=>{
  const session=new TrainingSession(model.initial,example),original=session.weights;
  for(let i=0;i<10;i++){const event=session.advance();assert.equal(event.committed,false);assert.equal(session.completed,0);assert.equal(session.weights,original);assert.equal(session.display().weights,original);}
  assert.equal(session.advance().committed,true);
  assert.equal(session.completed,1);assert.equal(session.history.length,1);
  assert.equal(session.display().weights,session.frame.afterWeights);
  assert.equal(session.advance().committed,false);assert.equal(session.completed,1);
  assert.equal(session.advance().committed,false);assert.equal(session.completed,1);
  assert.equal(session.frame.beforeWeights,session.weights);assert.equal(session.frame.number,2);
});

test('Ten animated updates match ten direct full-network SGD steps and teach the repeated digit',()=>{
  const session=new TrainingSession(model.initial,example,{rate:.05});
  let direct=cloneWeights(model.initial);
  const pixels=Float64Array.from(example.pixels,v=>v/255);
  const initial=forward(pixels,direct),initialLoss=loss(initial.logits,example.label);
  for(let i=0;i<10;i++) {
    direct=sgd(direct,backward(forward(pixels,direct),direct,example.label).weights,.05);
    complete(session);
    assert.equal(session.completed,i+1);
    for(const key of Object.keys(direct))assert.deepEqual(session.weights[key],direct[key]);
  }
  const after=forward(pixels,session.weights);
  assert.ok(loss(after.logits,example.label)<initialLoss*.1);
  assert.ok(after.softmax[example.label]>initial.softmax[example.label]);
  for(const key of Object.keys(direct))assert.ok(direct[key].some((v,i)=>v!==model.initial[key][i]),`${key} must change`);
});

test('Checkpoint review is immutable and resuming uses the latest live weights',()=>{
  const session=new TrainingSession(model.initial,example);
  for(let i=0;i<10;i++)complete(session);
  const live=session.weights;
  session.selectCheckpoint(0);assert.equal(session.display().weights,session.baseWeights);
  session.selectCheckpoint(1);assert.equal(session.display().weights,session.frames[0].afterWeights);
  near(session.cell('conv1Weight',4).after,session.frames[0].afterWeights.conv1Weight[4]);
  assert.equal(session.completed,10);assert.equal(session.weights,live);
  session.advance();assert.equal(session.review,null);assert.equal(session.frame.beforeWeights,live);assert.equal(session.frame.number,11);
});

test('The matrix movie delta and each displayed gradient recipe reconcile for every parameter group',()=>{
  const frame=makeTrainingFrame(cloneWeights(model.initial),example,.05,1);
  for(const [key,values] of Object.entries(frame.beforeWeights)) {
    for(const index of [...new Set([0,Math.floor(values.length/2),values.length-1])]) {
      const terms=gradientTerms(frame,key,index);
      near(terms.total,frame.gradients.weights[key][index]);
      near(frame.afterWeights[key][index],values[index]-.05*terms.total);
      near(frame.delta[key][index],frame.afterWeights[key][index]-values[index]);
    }
  }
  for(const key of ['conv1Weight','conv2Weight'])for(let i=0;i<frame.beforeWeights[key].length;i++)near(gradientTerms(frame,key,i).total,frame.gradients.weights[key][i]);
});

test('The optional stream contains real training-split images, cycles in order, and uses each correct label',()=>{
  assert.equal(stream.split,'train');assert.equal(stream.samples.length,200);
  const counts=Array(10).fill(0);
  for(const sample of stream.samples){counts[sample.label]++;assert.equal(sample.pixels.length,784);assert.ok(sample.pixels.every(v=>Number.isInteger(v)&&v>=0&&v<=255));}
  assert.deepEqual(counts,Array(10).fill(20));
  const session=new TrainingSession(model.initial,example,{mode:'stream',trainingSamples:stream.samples.slice(0,3)});
  for(let i=0;i<5;i++){complete(session);assert.equal(session.frame.example.id,stream.samples[i%3].id);assert.equal(session.frame.example.split,'train');assert.equal(session.history[i].label,stream.samples[i%3].label);near(session.history[i].before,loss(session.frame.before.logits,session.frame.example.label));}
});

test('Longer runs retain the initial, first, tenth, and recent snapshots with bounded matrix storage',()=>{
  const session=new TrainingSession(model.initial,example,{retention:3});
  for(let i=0;i<15;i++)complete(session);
  assert.deepEqual(session.frames.map(frame=>frame.number),[1,10,13,14,15]);
  session.selectCheckpoint(0);assert.equal(session.display().number,0);
  assert.throws(()=>session.selectCheckpoint(2));
  assert.throws(()=>session.cell('conv1Weight',-1));
  assert.throws(()=>session.cell('missing',0));
});

test('All 128 kernel slices retain their identity when packed into the clickable Conv2 wall',()=>{
  const visited=new Set();
  for(let index=0;index<1152;index++) {
    const position=conv2MosaicIndex(index);
    assert.ok(position>=0&&position<1152);
    visited.add(position);
    assert.equal(conv2WeightIndex(Math.floor(position/48),position%48),index);
  }
  assert.equal(visited.size,1152);
  assert.equal(conv2WeightIndex(0,12),72); // next filter to the right
  assert.equal(conv2WeightIndex(3,0),36); // fifth input-channel slice
  assert.equal(conv2WeightIndex(6,0),288); // next filter row
});
