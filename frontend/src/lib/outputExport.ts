/**
 * The output document as one self-contained HTML file.
 *
 * The app's own stylesheet is embedded, so the tables look as they did on
 * screen (they were copied with their Tailwind classes), and plots are PNG
 * data URLs already, so the file opens anywhere with nothing to fetch. A
 * print stylesheet keeps each item on its own page break boundary, which is
 * what "Print / PDF" relies on.
 */
import type { OutputItem } from "./outputDoc";

function appCss(): string {
  const parts: string[] = [];
  for (const sheet of Array.from(document.styleSheets)) {
    try {
      parts.push(Array.from(sheet.cssRules).map((r) => r.cssText).join("\n"));
    } catch {
      // A cross-origin sheet (a web font) cannot be read; the layout does not
      // depend on it.
    }
  }
  return parts.join("\n");
}

function escapeText(s: string): string {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

const PRINT_CSS = `
  body { background: #fff; margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #111827; }
  .output-doc { max-width: 960px; margin: 0 auto; padding: 32px 24px; }
  .output-doc > header { border-bottom: 1px solid #e5e7eb; margin-bottom: 24px; padding-bottom: 12px; }
  .output-item { margin-bottom: 36px; break-inside: avoid-page; }
  .output-item h2 { font-size: 18px; font-weight: 600; margin: 0 0 2px; }
  .output-meta { font-size: 11px; color: #6b7280; margin-bottom: 10px; }
  .output-note { font-size: 13px; background: #f9fafb; border-left: 3px solid #6366f1; padding: 8px 12px; margin: 10px 0; white-space: pre-wrap; }
  .output-body img { max-width: 100%; height: auto; }
  .output-body table { border-collapse: collapse; }
  @media print {
    .output-doc { padding: 0; }
    .output-item { break-before: auto; }
  }
`;

export function buildOutputHtml(items: OutputItem[], datasetName: string): string {
  const created = new Date().toLocaleString();
  const body = items
    .map((item) => {
      const when = new Date(item.createdAt).toLocaleString();
      const note = item.note.trim() ? `<div class="output-note">${escapeText(item.note)}</div>` : "";
      return `<section class="output-item">
  <h2>${escapeText(item.title)}</h2>
  <div class="output-meta">${escapeText([item.tab, when].filter(Boolean).join(" · "))}</div>
  ${note}
  <div class="output-body">${item.html}</div>
</section>`;
    })
    .join("\n");
  const title = `uSTAT output${datasetName ? `: ${datasetName}` : ""}`;
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeText(title)}</title>
<style>${appCss()}</style>
<style>${PRINT_CSS}</style>
</head>
<body>
<main class="output-doc">
<header><h1 style="font-size:22px;font-weight:700;margin:0">${escapeText(title)}</h1>
<div class="output-meta">${items.length} item${items.length === 1 ? "" : "s"} · exported ${escapeText(created)}</div></header>
${body}
</main>
</body>
</html>`;
}

export function downloadOutputHtml(items: OutputItem[], datasetName: string): void {
  const blob = new Blob([buildOutputHtml(items, datasetName)], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${(datasetName || "ustat").replace(/\.[^.]+$/, "")}_output.html`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** Open the document in a new window and bring up the print dialog (Save as PDF). */
export function printOutput(items: OutputItem[], datasetName: string): boolean {
  const win = window.open("", "_blank");
  if (!win) return false;
  win.document.open();
  win.document.write(buildOutputHtml(items, datasetName));
  win.document.close();
  win.addEventListener("load", () => win.print(), { once: true });
  return true;
}
