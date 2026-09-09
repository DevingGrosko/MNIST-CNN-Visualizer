/** Botanical palette shared by the page, diagrams, and numeric canvases. */
const BOTANICAL = {
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
  };
export const rgb = hex => [1,3,5].map(start=>parseInt(hex.slice(start,start+2),16));
BOTANICAL.rgb=Object.fromEntries(Object.entries(BOTANICAL.colors).map(([key,value])=>[key,rgb(value)]));
export const getPalette = () => BOTANICAL;
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
export function initPalette() {
  const root=document.documentElement;
  root.dataset.palette='botanical';
  for(const [key,value] of Object.entries(BOTANICAL.colors))root.style.setProperty('--'+key,value);
  // Existing comparison links still open their lesson, using the final palette.
  const url=new URL(location.href);
  if(url.searchParams.has('palette')) {
    url.searchParams.delete('palette');
    history.replaceState(history.state,'',url);
  }
}
