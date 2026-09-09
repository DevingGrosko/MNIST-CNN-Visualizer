import { PAGES, pageDocument } from './pages.mjs';
import { assetRevision } from './assets.mjs';
import { rm, cp, readFile, writeFile, readdir, mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
for (const path of ['web/index.html','web/app.js','web/palette.js','web/engine.js','web/styles.css','web/training.css','web/training-player.js','web/training-session.js','web/assets/training-samples.json','web/assets/model.json','web/assets/samples.json']) await readFile(path);
const data = JSON.parse(await readFile('web/assets/model.json', 'utf8'));
if (data.trained.fc1Weight.length !== 784 * 128) throw new Error('Invalid model weights');
async function sourceFiles(directory='web',prefix='') {
  const files=[];
  for(const entry of await readdir(directory,{withFileTypes:true})) {
    const path=join(prefix,entry.name),source=join(directory,entry.name);
    if(entry.isDirectory())files.push(...await sourceFiles(source,path));
    else files.push({path,content:await readFile(source)});
  }
  return files;
}
const files=await sourceFiles(),revision=assetRevision(files),assetRoot='./_app/'+revision;
await rm('dist', {recursive:true, force:true});
// Keep old asset paths available for HTML already cached by existing visitors.
await cp('web', 'dist', {recursive:true});
for(const file of files.filter(file=>!file.path.endsWith('.html'))) {
  const target=join('dist',assetRoot,file.path);
  await mkdir(dirname(target),{recursive:true});
  await writeFile(target,file.content);
}
const template = await readFile('web/index.html', 'utf8');
await Promise.all(PAGES.map(page => writeFile('dist/' + page.file, pageDocument(template, page,{assetRoot}))));
await writeFile('dist/release.json',JSON.stringify({revision,assetRoot}));
console.log(`Built ${PAGES.length} static pages; release ${revision}`);
