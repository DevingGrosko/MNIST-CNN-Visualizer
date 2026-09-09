import { createHash } from 'node:crypto';

/** One release directory keeps HTML, CSS, modules, and model assets in sync. */
export function assetRevision(files) {
  const hash=createHash('sha256');
  for(const {path,content} of [...files].sort((a,b)=>a.path.localeCompare(b.path))) {
    hash.update(path+'\0'+content.length+'\0');
    hash.update(content);
  }
  return hash.digest('hex').slice(0,16);
}
export function versionDocumentAssets(html,assetRoot='.') {
  return html.replace(/((?:src|href)=["'])\.\/([^"'?#]+\.(?:css|js|svg))(["'])/g,
    (_,prefix,file,quote)=>`${prefix}${assetRoot}/${file}${quote}`);
}
