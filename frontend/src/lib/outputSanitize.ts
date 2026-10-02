/**
 * Allow-list sanitiser for output-document HTML.
 *
 * The HTML is first captured from the app's own DOM, but it is also restored
 * from autosave and from .ustat project files, which can come from anyone,
 * and it is rendered with dangerouslySetInnerHTML. So it is never trusted:
 * the input is parsed and a NEW tree is built from allowed tags and allowed
 * attributes only. Building up rather than stripping down means a construct
 * nobody anticipated is simply not copied.
 */

const ALLOWED_TAGS = new Set([
  "div", "p", "span", "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption",
  "h1", "h2", "h3", "h4", "h5", "h6", "strong", "b", "em", "i", "u", "sup", "sub",
  "br", "hr", "ul", "ol", "li", "code", "pre", "small", "img",
]);

/** Dropped together with everything inside them. */
const DROPPED_TAGS = new Set([
  "script", "style", "svg", "math", "iframe", "object", "embed", "template", "noscript",
  "link", "meta", "base", "button", "input", "select", "option", "textarea", "form",
  "dialog", "canvas", "video", "audio", "source", "picture",
]);

// Tailwind utility names, including arbitrary values (text-[10px], w-1/2).
const CLASS_RE = /^[\w\s\-:./[\]%#()]*$/;
const SPAN_RE = /^\d{1,3}$/;
// Only images this app produced: PNG or JPEG embedded as base64.
const IMG_SRC_RE = /^data:image\/(png|jpeg);base64,[A-Za-z0-9+/=]+$/;

function copyAttributes(from: Element, to: Element): void {
  const tag = to.tagName.toLowerCase();
  const cls = from.getAttribute("class");
  if (cls && cls.length <= 500 && CLASS_RE.test(cls)) to.setAttribute("class", cls);
  for (const name of ["title", "alt"]) {
    const v = from.getAttribute(name);
    if (v) to.setAttribute(name, v.slice(0, 500));
  }
  if (tag === "td" || tag === "th") {
    for (const name of ["colspan", "rowspan"]) {
      const v = from.getAttribute(name);
      if (v && SPAN_RE.test(v)) to.setAttribute(name, v);
    }
  }
  if (tag === "img") {
    const src = from.getAttribute("src") ?? "";
    if (IMG_SRC_RE.test(src)) to.setAttribute("src", src);
  }
}

function rebuild(node: Node, out: Node, doc: Document): void {
  for (const child of Array.from(node.childNodes)) {
    if (child.nodeType === Node.TEXT_NODE) {
      out.appendChild(doc.createTextNode(child.textContent ?? ""));
      continue;
    }
    if (child.nodeType !== Node.ELEMENT_NODE) continue;
    const el = child as Element;
    const tag = el.tagName.toLowerCase();
    if (DROPPED_TAGS.has(tag)) continue;
    if (tag === "img" && !IMG_SRC_RE.test(el.getAttribute("src") ?? "")) continue;
    if (ALLOWED_TAGS.has(tag)) {
      const copy = doc.createElement(tag);
      copyAttributes(el, copy);
      rebuild(el, copy, doc);
      out.appendChild(copy);
    } else {
      // Unknown but harmless wrappers (section, label, details...): keep
      // their content, lose the element.
      rebuild(el, out, doc);
    }
  }
}

export function sanitizeOutputHtml(html: string): string {
  const parsed = new DOMParser().parseFromString(`<body>${html}</body>`, "text/html");
  const doc = document.implementation.createHTMLDocument("");
  const root = doc.createElement("div");
  rebuild(parsed.body, root, doc);
  return root.innerHTML;
}
