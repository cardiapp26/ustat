/**
 * ResultExporter – standardized CSV / XLSX / 300 DPI PNG export toolbar.
 * Sits at the top-right of any result panel.
 *
 * Usage (table):
 *   <ResultExporter title="Summary" headers={["Variable","N","Mean"]} rows={data} />
 *
 * Usage (plot):
 *   <ResultExporter title="ROC Curve" plotRef={ref} />
 *
 * Usage (both):
 *   <ResultExporter title="Cox Results" headers={h} rows={r} plotRef={ref} />
 */
import { useState } from "react";
import { Download } from "lucide-react";
import { plotlyToTiffBlob, downloadBlob } from "../lib/tiffEncoder";
import { withRegisteredPlotCapture } from "../lib/plotCapture";
import type { PlotRef, PlotCaptureHandle } from "../lib/plotTypes";
import { provenanceLines, type Provenance } from "../lib/engine/provenance";

/** Minimal shape of the Plotly module / graph-div fields we call. */
interface PlotlyToImage {
  toImage?: (gd: HTMLElement, opts: Record<string, unknown>) => Promise<string>;
}

/** Resolve the mounted Plotly graph div from a duck-typed plot handle. */
function resolveGraphDiv(plotRef: PlotRef): HTMLElement | undefined {
  const candidates: unknown[] = [];
  const r: PlotCaptureHandle | null = plotRef.current;
  if (r) {
    candidates.push(r.el);
    candidates.push(r);
    candidates.push(r.elRef?.current);
    const qs = r.querySelector;
    if (typeof qs === "function") {
      const query = qs as (selector: string) => Element | null;
      candidates.push(query.call(r, ".plotly-graph-div") || query.call(r, ".js-plotly-plot"));
    }
  }
  return candidates.find(
    (c): c is HTMLElement => !!c && !!(c as { _fullLayout?: unknown })._fullLayout,
  );
}

/** Resolve the Plotly `toImage` implementation for a mounted graph div. */
async function resolvePlotly(el: HTMLElement): Promise<PlotlyToImage> {
  let Plotly: PlotlyToImage | undefined = (el as { _Plotly?: PlotlyToImage })._Plotly;
  if (!Plotly?.toImage) {
    const mod = (await import("plotly.js/dist/plotly")) as unknown as PlotlyToImage & { default?: PlotlyToImage };
    Plotly = mod?.toImage ? mod : mod?.default;
  }
  if (!Plotly?.toImage) throw new Error("plotly.js toImage not available");
  return Plotly;
}

interface Props {
  title: string;
  /** Column headers for CSV/XLSX export */
  headers?: string[];
  /** Table rows for CSV/XLSX export */
  rows?: (string | number | null | undefined)[][];
  /** Plotly chart element ref for PNG export */
  plotRef?: PlotRef;
  /**
   * The result no longer matches the data / filter / settings it was computed
   * under. Export is refused while this is true: a CSV or a 300 dpi PNG leaves
   * the app with none of the context that would let a reader tell it is out of
   * date, and it is a figure in a paper by the time anyone notices.
   */
  stale?: boolean;
  /** Why it is stale, for the tooltip on the disabled buttons. */
  staleReason?: string;
  /**
   * What computed the result, appended to every table export.
   *
   * A CSV leaves the app with none of the surrounding interface, so a figure
   * whose engine and library versions are only on screen becomes a figure with
   * no provenance the moment it is exported -- which is the moment it starts
   * travelling towards a methods section.
   */
  provenance?: Provenance | null;
  className?: string;
}

/** Provenance as trailing rows of a table export: a blank line, then a label
 *  per fact, so a spreadsheet shows them as a footer rather than as data. */
function provenanceRows(
  provenance: Provenance | null | undefined,
  width: number,
): (string | number | null | undefined)[][] {
  const lines = provenanceLines(provenance);
  if (!lines.length) return [];
  const pad = (cells: (string | null)[]) =>
    [...cells, ...Array(Math.max(0, width - cells.length)).fill(null)];
  return [
    pad([]),
    pad(["Provenance", `Exported ${new Date().toISOString()}`]),
    ...lines.map((line) => {
      const colon = line.indexOf(":");
      return colon > 0
        ? pad([line.slice(0, colon), line.slice(colon + 1).trim()])
        : pad([line]);
    }),
  ];
}

function downloadCSV(filename: string, headers: string[], rows: (string | number | null | undefined)[][]) {
  const escape = (v: string | number | null | undefined) =>
    `"${String(v ?? "").replace(/"/g, '""')}"`;
  const lines = [headers, ...rows].map((r) => r.map(escape).join(","));
  const blob = new Blob([lines.join("\r\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename + ".csv"; a.click();
  URL.revokeObjectURL(url);
}

async function downloadXLSX(filename: string, headers: string[], rows: (string | number | null | undefined)[][]) {
  // xlsx package ships both ESM (named exports) and CJS (default export) builds.
  // Resolve whichever shape Vite delivers in this environment.
  interface XlsxUtils {
    aoa_to_sheet: (data: unknown[][]) => unknown;
    book_new: () => unknown;
    book_append_sheet: (wb: unknown, ws: unknown, name: string) => void;
  }
  interface XlsxModule {
    utils?: XlsxUtils;
    write: (wb: unknown, opts: Record<string, unknown>) => ArrayBuffer;
  }
  const mod = (await import("xlsx")) as XlsxModule & { default?: XlsxModule };
  const XLSX: XlsxModule | undefined = mod?.utils ? mod : mod?.default;
  if (!XLSX?.utils?.aoa_to_sheet) {
    throw new Error("xlsx module loaded but utils.aoa_to_sheet not available");
  }
  const ws = XLSX.utils.aoa_to_sheet([headers, ...rows]);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Results");
  // Use write() + blob for macOS Safari compatibility (writeFile can fail)
  const wbout = XLSX.write(wb, { bookType: "xlsx", type: "array" });
  const blob = new Blob([wbout], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename + ".xlsx";
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 100);
}

async function downloadPNG(plotRef: PlotRef, filename: string) {
  // Resolve to a Plotly graph div. react-plotly.js exposes `.el`; raw refs
  // give the DOM node directly; some wrappers nest it one deeper. A graph
  // div is recognisable by the `_fullLayout` property Plotly attaches at
  // mount time.
  const el = resolveGraphDiv(plotRef);
  if (!el) {
    throw new Error("plot is not mounted yet — wait for the chart to render and try again");
  }
  // Prefer the Plotly instance react-plotly.js already attached to the
  // gd — reuses the exact bundle that drew the chart and bypasses the
  // ESM tree-shake bug entirely. Fall back to the dist subpath only when
  // _Plotly isn't present.
  const Plotly = await resolvePlotly(el);
  // scale 3.125 ≈ 300 DPI (96 PPI × 3.125 = 300). Plotly.downloadImage
  // crashes on plotly.js@3 in production builds (tree-shaking strips an
  // internal dep) — use toImage + anchor-click instead.
  const dataUrl: string = await Plotly.toImage!(el, {
    format: "png",
    width: 1200,
    height: 700,
    scale: 3.125,
    setBackground: "opaque",
  });
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${filename}.png`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function downloadTIFF(plotRef: PlotRef, filename: string) {
  // Resolve the Plotly graph div the same way downloadPNG does.
  const el = resolveGraphDiv(plotRef);
  if (!el) {
    throw new Error("plot is not mounted yet — wait for the chart to render and try again");
  }
  const blob = await plotlyToTiffBlob(el, { width: 1200, height: 700, dpi: 300 });
  downloadBlob(blob, `${filename}.tiff`);
}

/** Render the chart to a PNG blob — shared by copy + download paths. */
async function renderPlotPngBlob(plotRef: PlotRef): Promise<Blob> {
  const el = resolveGraphDiv(plotRef);
  if (!el) throw new Error("plot is not mounted yet — wait for the chart to render and try again");
  const Plotly = await resolvePlotly(el);
  const dataUrl: string = await Plotly.toImage!(el, {
    format: "png",
    width: 1200,
    height: 700,
    scale: 3.125,
    setBackground: "opaque",
  });
  const res = await fetch(dataUrl);
  return await res.blob();
}

/** Copy the chart to the clipboard as PNG (system clipboard). */
async function copyPlotToClipboard(plotRef: PlotRef) {
  if (typeof ClipboardItem === "undefined" || !navigator.clipboard?.write) {
    throw new Error("Clipboard API not available in this browser");
  }
  const blob = await renderPlotPngBlob(plotRef);
  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
}

/** Copy the table to the clipboard as TSV (pastes into Excel / Word / Sheets). */
async function copyTableToClipboard(headers: string[], rows: (string | number | null | undefined)[][]) {
  if (!navigator.clipboard?.writeText) {
    throw new Error("Clipboard API not available in this browser");
  }
  const esc = (v: string | number | null | undefined) => {
    const s = String(v ?? "");
    // Tabs / newlines inside a cell break TSV — wrap the cell in quotes.
    return /[\t\n\r"]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const tsv = [headers, ...rows].map((r) => r.map(esc).join("\t")).join("\n");
  await navigator.clipboard.writeText(tsv);
}

export default function ResultExporter({
  title, headers, rows, plotRef, stale = false, staleReason, provenance, className = "",
}: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  // "Copied" pill flashes for ~1.5 s on a successful copy.
  const [copyToast, setCopyToast] = useState<string | null>(null);

  const safeTitle = title.replace(/[^\w\s-]/g, "").replace(/\s+/g, "_").slice(0, 50) || "export";
  const hasTable = headers && rows;
  /** The table plus its provenance footer -- what actually gets written out. */
  const exportRows = () =>
    headers && rows ? [...rows, ...provenanceRows(provenance, headers.length)] : [];
  const hasPlot = !!plotRef;

  const blockedTitle = staleReason
    ? `Recompute first — this result predates ${staleReason}`
    : "Recompute first — this result is out of date";

  const handle = async (format: "csv" | "xlsx" | "png" | "tiff" | "copy-table" | "copy-plot") => {
    if (busy || stale) return;
    setBusy(format);
    setErr(null);
    try {
      if (format === "csv" && headers && rows) downloadCSV(safeTitle, headers, exportRows());
      if (format === "xlsx" && headers && rows) await downloadXLSX(safeTitle, headers, exportRows());
      if (format === "png" && plotRef) {
        await withRegisteredPlotCapture(plotRef, () => downloadPNG(plotRef, safeTitle));
      }
      if (format === "tiff" && plotRef) {
        await withRegisteredPlotCapture(plotRef, () => downloadTIFF(plotRef, safeTitle));
      }
      if (format === "copy-table" && headers && rows) {
        await copyTableToClipboard(headers, exportRows());
        setCopyToast("Table copied — paste into Excel / Word");
        setTimeout(() => setCopyToast(null), 1500);
      }
      if (format === "copy-plot" && plotRef) {
        await withRegisteredPlotCapture(plotRef, () => copyPlotToClipboard(plotRef));
        setCopyToast("Chart copied to clipboard");
        setTimeout(() => setCopyToast(null), 1500);
      }
    } catch (e) {
      console.error("Export error:", e);
      const msg = e instanceof Error ? e.message : String(e);
      setErr(`${format.toUpperCase()} export failed: ${msg}`);
    } finally {
      setBusy(null);
    }
  };

  if (!hasTable && !hasPlot) return null;

  return (
    <div className={`flex items-center gap-1 ${className}`}>
      <span
        className={`text-[10px] mr-0.5 flex items-center gap-0.5 ${stale ? "text-amber-600" : "text-gray-400"}`}
        title={stale ? blockedTitle : undefined}
      >
        <Download size={10} /> {stale ? "Export blocked" : "Export"}
      </span>
      {hasTable && (
        <>
          <button
            onClick={() => handle("csv")}
            disabled={!!busy || stale}
            title={stale ? blockedTitle : undefined}
            className="px-2 py-0.5 text-[10px] font-medium rounded border border-gray-200 bg-white text-gray-600 hover:bg-gray-50 hover:text-indigo-600 disabled:opacity-40 transition-colors"
          >
            {busy === "csv" ? "…" : "CSV"}
          </button>
          <button
            onClick={() => handle("xlsx")}
            disabled={!!busy || stale}
            title={stale ? blockedTitle : undefined}
            className="px-2 py-0.5 text-[10px] font-medium rounded border border-gray-200 bg-white text-gray-600 hover:bg-gray-50 hover:text-indigo-600 disabled:opacity-40 transition-colors"
          >
            {busy === "xlsx" ? "…" : "XLSX"}
          </button>
          <button
            onClick={() => handle("copy-table")}
            disabled={!!busy || stale}
            title={stale ? blockedTitle : "Copy table to clipboard as TSV — paste into Excel / Word / Google Sheets"}
            className="px-2 py-0.5 text-[10px] font-medium rounded border border-gray-200 bg-white text-gray-600 hover:bg-emerald-50 hover:text-emerald-700 hover:border-emerald-200 disabled:opacity-40 transition-colors"
          >
            {busy === "copy-table" ? "…" : "⧉ Copy"}
          </button>
        </>
      )}
      {hasPlot && (
        <>
          <button
            onClick={() => handle("png")}
            disabled={!!busy || stale}
            title={stale ? blockedTitle : undefined}
            className="px-2 py-0.5 text-[10px] font-medium rounded border border-gray-200 bg-white text-gray-600 hover:bg-gray-50 hover:text-indigo-600 disabled:opacity-40 transition-colors"
          >
            {busy === "png" ? "…" : "PNG 300dpi"}
          </button>
          <button
            onClick={() => handle("tiff")}
            disabled={!!busy || stale}
            title={stale ? blockedTitle : "Baseline uncompressed RGB TIFF (journal-ready, larger file)"}
            className="px-2 py-0.5 text-[10px] font-medium rounded border border-gray-200 bg-white text-gray-600 hover:bg-gray-50 hover:text-indigo-600 disabled:opacity-40 transition-colors"
          >
            {busy === "tiff" ? "…" : "TIFF 300dpi"}
          </button>
          <button
            onClick={() => handle("copy-plot")}
            disabled={!!busy || stale}
            title={stale ? blockedTitle : "Copy chart to clipboard as PNG — paste into PowerPoint / Word / Slack"}
            className="px-2 py-0.5 text-[10px] font-medium rounded border border-gray-200 bg-white text-gray-600 hover:bg-emerald-50 hover:text-emerald-700 hover:border-emerald-200 disabled:opacity-40 transition-colors"
          >
            {busy === "copy-plot" ? "…" : "⧉ Copy chart"}
          </button>
        </>
      )}
      {copyToast && (
        <span className="text-[10px] font-medium px-2 py-0.5 rounded bg-emerald-600 text-white shadow ml-1 whitespace-nowrap">
          {copyToast}
        </span>
      )}
      {err && (
        <span
          title={err}
          className="text-[10px] text-red-600 ml-1 max-w-[280px] truncate"
        >
          {err}
        </span>
      )}
    </div>
  );
}
