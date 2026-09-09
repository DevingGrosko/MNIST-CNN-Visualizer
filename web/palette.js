/** Shared palette tokens for CSS, SVG, and numeric canvas views. */
export const PALETTES = {
  botanical: {
    name: 'Botanical',
    swatches: ['#8381A6','#D48662','#8B322C','#723715','#26160C'],
    colors: {
      bg:'#F7F1E9',surface:'#FFFCF7',raised:'#EDE2D6',line:'#CFC0B1','line-soft':'#E5DAD0',
      text:'#26160C',muted:'#685449',dim:'#766052',accent:'#8B322C',earth:'#723715',
      positive:'#8B322C',negative:'#8381A6','positive-ink':'#8B322C','negative-ink':'#585578',
      'positive-soft':'#F3DFD3','negative-soft':'#E8E5F0',selected:'#EFCEBD',
      change:'#D48662','change-ink':'#723715','change-soft':'#F2D2BF',neutral:'#AB9686','neutral-dark':'#786557',
    },
    positiveName:'Burgundy',negativeName:'Lavender',changeName:'Clay',
  },
  twilight: {
    name: 'Twilight',
    swatches: ['#0C0B31','#48306C','#51499B','#B76BA5','#E2CAE5'],
    colors: {
      bg:'#F5F0F8',surface:'#FDFBFE',raised:'#E9DDEE',line:'#C9B8D3','line-soft':'#E0D5E8',
      text:'#0C0B31',muted:'#5D506C',dim:'#70607D',accent:'#51499B',earth:'#48306C',
      positive:'#B76BA5',negative:'#51499B','positive-ink':'#8B427B','negative-ink':'#51499B',
      'positive-soft':'#F0DDEC','negative-soft':'#E4E0F3',selected:'#E2CAE5',
      change:'#B76BA5','change-ink':'#48306C','change-soft':'#E2CAE5',neutral:'#A496B4','neutral-dark':'#726380',
    },
    positiveName:'Rose',negativeName:'Indigo',changeName:'Rose',
  },
};
export const rgb = hex => [1,3,5].map(start=>parseInt(hex.slice(start,start+2),16));
for(const palette of Object.values(PALETTES))palette.rgb=Object.fromEntries(Object.entries(palette.colors).map(([key,value])=>[key,rgb(value)]));
let active='botanical';
export const getPalette = () => PALETTES[active];
export function setPalette(name) { if(!Object.hasOwn(PALETTES,name))return false;active=name;return true; }
export function signedRGB(value,max=1) {
  const p=getPalette().rgb,t=Math.min(1,Math.abs(value)/(max||1)),end=value<0?p.negative:p.positive;
  return p.bg.map((v,i)=>Math.round(v+(end[i]-v)*t));
}
const luminance = values => values.map(value=>value/255).map(value=>value<=.04045?value/12.92:((value+.055)/1.055)**2.4).reduce((sum,value,i)=>sum+value*[.2126,.7152,.0722][i],0);
export function inkOn(values) {
  const background=luminance(values),dark=luminance(getPalette().rgb.text),light=luminance(getPalette().rgb.surface);
  const contrast=foreground=>(Math.max(background,foreground)+.05)/(Math.min(background,foreground)+.05);
  const best=Math.max(contrast(dark),contrast(light));
  if(best<4.5)return background>.179?'#000000':'#FFFFFF';
  return contrast(dark)>=contrast(light)?getPalette().colors.text:getPalette().colors.surface;
}
export function signLegend() {
  const p=getPalette();
  return `${p.positiveName} encodes positive values; ${p.negativeName.toLowerCase()} encodes negative values. The dark outline marks a selection.`;
}
export function initPalette(onChange) {
  const root=document.documentElement;
  let saved;
  try { saved=localStorage.getItem('cnn-palette'); } catch { /* Optional local preference. */ }
  const requested=new URL(location.href).searchParams.get('palette');
  setPalette(Object.hasOwn(PALETTES,requested)?requested:saved);
  const controls=document.querySelector('#palette-controls');
  controls.innerHTML=Object.entries(PALETTES).map(([key,p])=>`<button type="button" id="palette-${key}" data-palette="${key}" aria-label="Use the ${p.name} color palette"><span class="palette-swatches" aria-hidden="true">${p.swatches.map(hex=>`<i style="background:${hex}"></i>`).join('')}</span><span>${p.name}</span></button>`).join('');
  function apply() {
    const palette=getPalette();root.dataset.palette=active;
    for(const [key,value] of Object.entries(palette.colors))root.style.setProperty('--'+key,value);
    controls.querySelectorAll('[data-palette]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.palette===active)));
    document.querySelector('meta[name="theme-color"]').content=palette.colors.bg;
    const c=palette.colors;
    const icon=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" fill="${c.bg}"/><path d="M12 13h40v40H12zM25 13v40M39 13v40M12 26h40M12 40h40" fill="none" stroke="${c.text}" stroke-width="2"/><path fill="${c.accent}" d="M26 27h12v12H26z"/></svg>`;
    const iconURL='data:image/svg+xml,'+encodeURIComponent(icon);
    document.querySelector('link[rel="icon"]').href=iconURL;
    document.querySelector('.brand img').src=iconURL;
  }
  apply();
  controls.querySelectorAll('[data-palette]').forEach(button=>button.onclick=()=>{
    if(!setPalette(button.dataset.palette))return;
    try {localStorage.setItem('cnn-palette',active);} catch { /* In-memory switching still works. */ }
    const url=new URL(location.href);url.searchParams.set('palette',active);history.replaceState(history.state,'',url);
    apply();onChange();
  });
}
