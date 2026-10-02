/**
 * An output item's sanitised HTML as the flat blocks the Word export takes:
 * headings, paragraphs, tables (rows of cell text) and figures.
 *
 * Walks the tree once. A heading, table or figure is emitted whole and its
 * subtree is not visited again; any other element whose children are only
 * text and inline markup becomes one paragraph. Summary cards (a label over a
 * value, "N" / "300") come out as "N: 300" instead of two one-word lines.
 */
import type { OutputItem } from "./outputDoc";

export type OutputBlock =
  | { type: "heading"; text: string; level: number }
  | { type: "paragraph"; text: string }
  | { type: "table"; rows: string[][]; header_rows: number }
  | { type: "image"; image: string };

const BLOCK_TAGS = new Set([
  "div", "p", "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption",
  "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "pre", "img", "hr",
]);

const clean = (s: string | null | undefined): string => (s ?? "").replace(/\s+/g, " ").trim();

function hasBlockChild(el: Element): boolean {
  return Array.from(el.children).some((c) => BLOCK_TAGS.has(c.tagName.toLowerCase()));
}

function tableRows(table: Element): { rows: string[][]; header_rows: number } {
  const trs = Array.from(table.querySelectorAll("tr"));
  const rows = trs.map((tr) => Array.from(tr.children).map((cell) => clean(cell.textContent)));
  let header_rows = 0;
  for (const tr of trs) {
    const inHead = tr.parentElement?.tagName.toLowerCase() === "thead";
    const allTh = tr.children.length > 0 && Array.from(tr.children).every((c) => c.tagName.toLowerCase() === "th");
    if (inHead || allTh) header_rows += 1;
    else break;
  }
  return { rows: rows.filter((r) => r.some(Boolean)), header_rows };
}

/** A card: exactly two leaf children, both short, read as "label: value". */
function asLabelValue(el: Element): string | null {
  const kids = Array.from(el.children);
  if (kids.length !== 2 || kids.some(hasBlockChild)) return null;
  const [a, b] = kids.map((k) => clean(k.textContent));
  if (!a || !b || a.length > 40 || b.length > 40) return null;
  return `${a}: ${b}`;
}

function walk(el: Element, out: OutputBlock[]): void {
  const tag = el.tagName.toLowerCase();
  if (/^h[1-6]$/.test(tag)) {
    const text = clean(el.textContent);
    if (text) out.push({ type: "heading", text, level: Number(tag[1]) });
    return;
  }
  if (tag === "table") {
    const t = tableRows(el);
    if (t.rows.length) out.push({ type: "table", ...t });
    return;
  }
  if (tag === "img") {
    const src = el.getAttribute("src");
    if (src) out.push({ type: "image", image: src });
    return;
  }
  const card = el.children.length === 2 ? asLabelValue(el) : null;
  if (card) {
    out.push({ type: "paragraph", text: card });
    return;
  }
  if (!hasBlockChild(el)) {
    const text = clean(el.textContent);
    if (text) out.push({ type: "paragraph", text: tag === "li" ? `• ${text}` : text });
    return;
  }
  // Mixed content: inline text between block children is a paragraph of its
  // own, written where it stands rather than lost.
  let inline = "";
  const flush = () => {
    const text = clean(inline);
    if (text) out.push({ type: "paragraph", text });
    inline = "";
  };
  for (const node of Array.from(el.childNodes)) {
    if (node.nodeType === Node.ELEMENT_NODE && BLOCK_TAGS.has((node as Element).tagName.toLowerCase())) {
      flush();
      walk(node as Element, out);
    } else {
      inline += ` ${node.textContent ?? ""}`;
    }
  }
  flush();
}

export function htmlToBlocks(html: string): OutputBlock[] {
  const body = new DOMParser().parseFromString(`<body><div>${html}</div></body>`, "text/html").body;
  const out: OutputBlock[] = [];
  walk(body.firstElementChild as Element, out);
  return out;
}

export function outputDocxPayload(items: OutputItem[], datasetName: string) {
  return {
    title: `uSTAT output${datasetName ? `: ${datasetName}` : ""}`,
    filename: `${(datasetName || "ustat").replace(/\.[^.]+$/, "")}_output`,
    items: items.map((item) => ({
      title: item.title,
      meta: [item.tab, new Date(item.createdAt).toLocaleString()].filter(Boolean).join(" · "),
      note: item.note,
      blocks: htmlToBlocks(item.html),
    })),
  };
}
