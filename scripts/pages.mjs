import { PAGES } from '../web/pages.js';
export { PAGES };
const escape = text => text.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;');
export function pageDocument(template, page) {
  return template
    .replace(/<title>.*?<\/title>/, `<title>${escape(page.label)} — CNN, Explained</title>`)
    .replace(/<meta name="description" content="[^"]*">/, `<meta name="description" content="${escape(page.deck)}">`)
    .replace('<body data-page="overview">', `<body data-page="${page.id}">`)
    .replace(/(<h1 id="page-title"[^>]*>).*?(<\/h1>)/, `$1${escape(page.title)}$2`)
    .replace(/(<p class="intro-copy" id="page-deck">).*?(<\/p>)/, `$1${escape(page.deck)}$2`);
}
