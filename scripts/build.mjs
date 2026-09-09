import { rm, cp, readFile } from 'node:fs/promises';
for (const path of ['web/index.html','web/app.js','web/engine.js','web/styles.css','web/assets/model.json','web/assets/samples.json']) await readFile(path);
const data = JSON.parse(await readFile('web/assets/model.json', 'utf8'));
if (data.trained.fc1Weight.length !== 784 * 128) throw new Error('Invalid model weights');
await rm('dist', {recursive:true, force:true});
await cp('web', 'dist', {recursive:true});
console.log('Built static website in dist/');
