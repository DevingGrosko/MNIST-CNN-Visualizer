/** Navy, rose, coral, and peach shared by the page and all numeric views. */
const ROSE_DESERT = {
  name: 'Rose Desert',
  swatches: ['#355070','#6D597A','#B56576','#E56B6F','#EAAC8B'],
  colors: {
    "bg": "#FFF7F2",
    "surface": "#FFFCFA",
    "raised": "#F0E2E0",
    "line": "#CFBBBF",
    "line-soft": "#E7D9DB",
    "text": "#283A53",
    "muted": "#60576B",
    "dim": "#6D597A",
    "accent": "#355070",
    "earth": "#6D597A",
    "header": "#355070",
    "sidebar": "#6D597A",
    "positive": "#E56B6F",
    "negative": "#355070",
    "positive-ink": "#A23B45",
    "negative-ink": "#355070",
    "positive-soft": "#F8DDE0",
    "negative-soft": "#E4EAF1",
    "selected": "#F2CBB6",
    "change": "#EAAC8B",
    "change-ink": "#355070",
    "change-soft": "#F7D9C9",
    "neutral": "#B56576",
    "neutral-dark": "#6D597A"
},
  positiveName:'Coral',negativeName:'Navy',changeName:'Peach',
};
export const rgb = hex => [1,3,5].map(start=>parseInt(hex.slice(start,start+2),16));
ROSE_DESERT.rgb=Object.fromEntries(Object.entries(ROSE_DESERT.colors).map(([key,value])=>[key,rgb(value)]));
export const getPalette = () => ROSE_DESERT;
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
  root.dataset.palette='rose-desert';
  for(const [key,value] of Object.entries(ROSE_DESERT.colors))root.style.setProperty('--'+key,value);
  // Existing comparison links still open their lesson, using the final palette.
  const url=new URL(location.href);
  if(url.searchParams.has('palette')) {
    url.searchParams.delete('palette');
    history.replaceState(history.state,'',url);
  }
}
