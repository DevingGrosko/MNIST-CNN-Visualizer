import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { PAGES, pageForStep, resolvePage } from '../web/pages.js';
import { pageDocument } from '../scripts/pages.mjs';
const template=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');

test('The overview, twelve lessons, and field notes have unique real page URLs',()=>{
  assert.equal(PAGES.length,14);
  assert.equal(new Set(PAGES.map(p=>p.file)).size,14);
  assert.equal(new Set(PAGES.map(p=>p.id)).size,14);
  for(let step=0;step<12;step++)assert.equal(pageForStep(step).step,step);
  assert.equal(pageForStep(-1),undefined);
  assert.equal(pageForStep(12),undefined);
});

test('Every static document has its own title, heading, metadata, and initial view',()=>{
  for(const page of PAGES) {
    const document=pageDocument(template,page);
    assert.ok(document.includes(`<body data-page="${page.id}">`));
    assert.ok(document.includes(`<title>${page.label.replaceAll('&','&amp;')} — CNN, Explained</title>`));
    assert.ok(document.includes(page.title));
    assert.ok(document.includes('id="page-title" tabindex="-1"'));
    assert.ok(document.includes('./app.js'));
    assert.ok(document.includes('./styles.css'));
  }
});

test('Lesson routing resolves correctly under the GitHub repository subpath',()=>{
  for(const page of PAGES)assert.equal(resolvePage('/MNIST-CNN-Visualizer/'+page.file),page);
  assert.equal(resolvePage('/MNIST-CNN-Visualizer/'),PAGES[0]);
  assert.equal(resolvePage('/'),PAGES[0]);
});
