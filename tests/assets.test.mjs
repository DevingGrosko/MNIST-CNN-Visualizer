import test from 'node:test';
import assert from 'node:assert/strict';
import { assetRevision, versionDocumentAssets } from '../scripts/assets.mjs';
import { pageDocument, PAGES } from '../scripts/pages.mjs';
import { readFileSync } from 'node:fs';

test('Changing styles, modules, or model data gives the whole release new cache keys',()=>{
  const files=[{path:'styles.css',content:Buffer.from('old')},{path:'app.js',content:Buffer.from('code')},{path:'assets/model.json',content:Buffer.from('{}')}];
  const first=assetRevision(files);
  assert.equal(first,assetRevision([...files].reverse()));
  for(let i=0;i<files.length;i++)assert.notEqual(first,assetRevision(files.map((file,index)=>index===i?{...file,content:Buffer.from('new')}:file)));
});

test('Versioned documents keep lesson URLs stable and put all executable assets in one release',()=>{
  const template=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
  const assetRoot='./_app/abc123';
  for(const page of PAGES) {
    const html=pageDocument(template,page,{assetRoot});
    for(const file of ['app.js','styles.css','training.css','favicon.svg'])assert.ok(html.includes(`${assetRoot}/${file}`));
    assert.ok(html.includes('href="./input.html"'));
    assert.ok(html.includes('https://github.com/DevingGrosko/MNIST-CNN-Visualizer'));
    assert.equal((html.match(/\.\/_app\/abc123\/favicon\.svg/g)||[]).length,2);
  }
  assert.equal(versionDocumentAssets(template),template);
});

test('Relative imports and data URLs remain inside the release at the GitHub repository subpath',()=>{
  const root='https://devinggrosko.github.io/MNIST-CNN-Visualizer/_app/abc123/';
  for(const path of ['web/app.js','web/training-player.js']) {
    const source=readFileSync(new URL('../'+path,import.meta.url),'utf8');
    for(const [,module] of source.matchAll(/from\s*['"]([^'"]+)['"]/g))assert.ok(new URL(module,root+'app.js').href.startsWith(root));
    assert.ok(source.includes('import.meta.url'));
  }
  assert.equal(new URL('./assets/model.json',root+'app.js').href,root+'assets/model.json');
});
