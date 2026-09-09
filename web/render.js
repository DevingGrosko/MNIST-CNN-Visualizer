import { signedRGB, getPalette } from './palette.js';
export const $ = (s, root=document)=>root.querySelector(s);
export const $$ = (s, root=document)=>Array.from(root.querySelectorAll(s));
export const f = (v,d=3)=>Math.abs(v)<0.5*10**(-d)?(0).toFixed(d):Number(v).toFixed(d);
export const pct = v=>(100*v).toFixed(v>=.9995?2:1)+'%';
export const esc = value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function color(value,max=1,gray=false) {
  if(gray){const v=Math.round(Math.max(0,Math.min(1,value))*235+8);return `rgb(${v},${v+Math.round((255-v)*.012)},${v+Math.round((255-v)*.035)})`;}
  return `rgb(${signedRGB(value,max).join(',')})`;
}
export function heatmap(canvas,data,n,{gray=false,x,y,patch=false,region=1}={}) {
  if(!canvas)return;
  const c=canvas.getContext('2d'),scale=8;canvas.width=n*scale;canvas.height=n*scale;
  const max=data.reduce((m,v)=>Math.max(m,Math.abs(v)),0);
  for(let row=0;row<n;row++)for(let col=0;col<n;col++) {c.fillStyle=color(data[row*n+col],max,gray);c.fillRect(col*scale,row*scale,scale,scale);}
  if(Number.isInteger(x)&&Number.isInteger(y)) {
    c.strokeStyle=getPalette().colors.text;c.lineWidth=1.6;
    const startX=patch?Math.max(0,x-1):x,startY=patch?Math.max(0,y-1):y;
    const endX=patch?Math.min(n,x+2):Math.min(n,x+region),endY=patch?Math.min(n,y+2):Math.min(n,y+region);
    c.strokeRect(startX*scale+.8,startY*scale+.8,(endX-startX)*scale-1.6,(endY-startY)*scale-1.6);
    canvas.dataset.x=x;canvas.dataset.y=y;
  }
}
export function mapEvents(canvas,n,onSelect) {
  canvas.tabIndex=0;canvas.setAttribute('role','img');
  canvas.addEventListener('click',e=>{const r=canvas.getBoundingClientRect();onSelect(Math.min(n-1,Math.max(0,Math.floor((e.clientX-r.left)/r.width*n))),Math.min(n-1,Math.max(0,Math.floor((e.clientY-r.top)/r.height*n))));});
  canvas.addEventListener('keydown',e=>{
    const moves={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]};
    if(!moves[e.key])return;e.preventDefault();const [dx,dy]=moves[e.key];
    onSelect(Math.max(0,Math.min(n-1,Number(canvas.dataset.x||0)+dx)),Math.max(0,Math.min(n-1,Number(canvas.dataset.y||0)+dy)));
  });
}
export function matrix(values,title,{padding=[],maxIndex=-1}={}) {
  const n=Math.sqrt(values.length);
  return `<div class="matrix-block"><p class="matrix-title">${title}</p><div class="matrix" style="grid-template-columns:repeat(${n},1fr)">${values.map((v,i)=>`<div class="matrix-cell ${v<0?'negative':v===0?'zero':''} ${padding[i]?'padded':''} ${i===maxIndex?'max-cell':''}" title="${v}${padding[i]?' (zero padding)':''}">${f(v,2)}</div>`).join('')}</div></div>`;
}
export function options(n,selected,prefix='') { return Array.from({length:n},(_,i)=>`<option value="${i}" ${i===selected?'selected':''}>${prefix}${i}</option>`).join(''); }
