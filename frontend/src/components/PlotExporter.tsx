/**
 * PlotExporter – floating ↓ button that downloads any Plotly chart as PNG/SVG.
 * Plotly is imported lazily inside the handler so this file adds no extra
 * top-level dependency on plotly.js (the chart component already loads it).
 */
import { useState, useEffect } from "react";
import { plotlyToTiffBlob, downloadBlob } from "../lib/tiffEncoder";
import type { PlotRef, PlotCaptureHandle } from "../lib/plotTypes";
import { staleExportTitle, useStaleGuard } from "../lib/staleGuard";

type ExportFmt = "png" | "svg" | "tiff" | "jpeg";

/** Minimal shape of the Plotly module / graph-div fields we call. */
interface PlotlyToImage {
  toImage?: (gd: HTMLElement, opts: Record<string, unknown>) => Promise<string>;
}

interface Props {
  plotRef: PlotRef;
  title?: string;
  className?: string;
  defaultWidth?: number;
  defaultHeight?: number;
  /** Run just before the figure is captured (copy or download) and restored
   *  right after — used to transiently strip labels from the export only. */
  onBeforeCapture?: () => void | Promise<void>;
  onAfterCapture?: () => void | Promise<void>;
}

export default function PlotExporter({
  plotRef,
  title = "chart",
  className = "",
  defaultWidth,
  defaultHeight,
  onBeforeCapture,
  onAfterCapture,
}: Props) {
  const [open, setOpen]       = useState(false);
  const [width, setWidth]     = useState(defaultWidth ?? 1200);
  const [height, setHeight]   = useState(defaultHeight ?? 700);
  const [fmt, setFmt]         = useState<ExportFmt>("png");
  const [dpi, setDpi]         = useState(300);
  const [busy, setBusy]       = useState(false);
  // Inside an out-of-date result the figure cannot leave the app.
  const guard = useStaleGuard();
  
  useEffect(() => {
    if (defaultWidth) setWidth(defaultWidth);
  }, [defaultWidth]);

  useEffect(() => {
    if (defaultHeight) setHeight(defaultHeight);
  }, [defaultHeight]);

  // "Copied to clipboard" pill is shown for ~1.5s on success so the user
  // doesn't have to guess whether the click did anything.
  const [copyToast, setCopyToast] = useState<string | null>(null);


  const safeTitle = title.replace(/[^\w\s-]/g, "").replace(/\s+/g, "_").slice(0, 40) || "chart";
  const getEl = (): HTMLElement | null => {
    const ref: PlotCaptureHandle | null = plotRef.current;
    if (!ref) return null;
    // react-plotly.js component instance → .el
    if (ref.el) return ref.el;
    // Direct DOM element with .data (already a plotly div)
    if (ref.data) return ref as unknown as HTMLElement;
    // Wrapper div → find the plotly div inside
    if (ref instanceof HTMLElement) {
      const plotlyDiv = ref.querySelector(".js-plotly-plot") ?? ref.querySelector("[class*='plotly']");
      if (plotlyDiv) return plotlyDiv as HTMLElement;
      // If the ref IS the container, find first child div with data
      for (const child of ref.querySelectorAll("div")) {
        if ((child as { data?: unknown }).data) return child as HTMLElement;
      }
      return ref;
    }
    return null;
  };

  /** Render the chart to a PNG blob at the current width/height/dpi. */
  const renderPngBlob = async (): Promise<Blob | null> => {
    const el = getEl();
    if (!el) return null;
    // Reuse the gd's own Plotly instance — same fix as downloadImage.
    let Plotly: PlotlyToImage | undefined = (el as { _Plotly?: PlotlyToImage })._Plotly;
    if (!Plotly?.toImage) {
      const mod = (await import("plotly.js/dist/plotly")) as unknown as PlotlyToImage & { default?: PlotlyToImage };
      Plotly = mod?.toImage ? mod : mod?.default;
    }
    if (!Plotly?.toImage) throw new Error("plotly.js toImage not available");
    const scale = dpi / 72;
    // Plotly's supported opaque mode composites transparent paper onto white.
    const dataUrl: string = await Plotly.toImage(el, { format: "png", width, height, scale, setBackground: "opaque" });
    const res = await fetch(dataUrl);
    return await res.blob();
  };

  /** Copy the rendered chart to the system clipboard as a PNG image. */
  const copyImage = async () => {
    if (busy || guard.stale) return;
    setBusy(true);
    try {
      await onBeforeCapture?.();
      let blob: Blob | null;
      try { blob = await renderPngBlob(); } finally { await onAfterCapture?.(); }
      if (!blob) throw new Error("plot is not mounted yet");
      // ClipboardItem + write is supported in modern Chromium, Safari 13.4+,
      // and Firefox 127+. The user click is the gesture required by Safari.
      if (typeof ClipboardItem === "undefined" || !navigator.clipboard?.write) {
        throw new Error("Clipboard API not available in this browser");
      }
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
      setCopyToast("Copied to clipboard");
      setTimeout(() => setCopyToast(null), 1500);
    } catch (e: unknown) {
      console.error("PlotExporter copy failed:", e);
      alert(`Copy chart failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  const downloadImage = async () => {
    if (guard.stale) return;
    const el = getEl();
    if (!el) return;
    setBusy(true);
    await onBeforeCapture?.();
    try {
      if (fmt === "tiff") {
        // Plotly cannot emit TIFF natively — rasterise to PNG at the
        // requested DPI, then encode an uncompressed baseline RGB TIFF.
        const blob = await plotlyToTiffBlob(el, { width, height, dpi });
        downloadBlob(blob, `${safeTitle}.tiff`);
      } else {
        // Use Plotly.toImage instead of Plotly.downloadImage — the latter
        // crashes on plotly.js@3 in production builds with "Cannot read
        // properties of undefined (reading 'prototype')" because part of
        // its internal download chain got tree-shaken away. toImage just
        // returns a data URL / Blob, which we hand to an anchor click —
        // the same trustworthy pattern the TIFF and dataset exporters use.
        // Import the same UMD bundle that react-plotly.js uses
        // (plotly.js/dist/plotly) — the package-root entrypoint resolves
        // to an ESM build whose toImage / downloadImage chains were
        // tree-shaken in production and crash with "Cannot read properties
        // of undefined (reading 'prototype')".
        // Prefer the Plotly instance react-plotly.js already attached to
        // the gd — calling _Plotly.toImage reuses the exact bundle that
        // rendered the chart and bypasses the ESM tree-shake bug entirely.
        let Plotly: PlotlyToImage | undefined = (el as { _Plotly?: PlotlyToImage })._Plotly;
        if (!Plotly?.toImage) {
          const mod = (await import("plotly.js/dist/plotly")) as unknown as PlotlyToImage & { default?: PlotlyToImage };
          Plotly = mod?.toImage ? mod : mod?.default;
        }
        if (!Plotly?.toImage) {
          throw new Error("plotly.js toImage not available");
        }
        const scale = (fmt === "png" || fmt === "jpeg") ? dpi / 72 : 1;  // SVG vector
        const dataUrl: string = await Plotly.toImage(el, {
          format: fmt,
          width,
          height,
          setBackground: "opaque",
          ...(scale !== 1 ? { scale } : {}),
        });
        // Data URL → Blob → anchor click. Direct anchor.href = dataUrl works
        // for small charts but Safari truncates very large data URLs; round
        // through a Blob so any chart size downloads cleanly.
        const res = await fetch(dataUrl);
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${safeTitle}.${fmt}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    } catch (e: unknown) {
      console.error("PlotExporter download failed:", e);
      alert(`Chart export failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      await onAfterCapture?.();
      setBusy(false);
      setOpen(false);
    }
  };

  return (
    <div className={`absolute top-2 right-2 z-10 flex gap-1 ${className}`}>
      <button
        onClick={copyImage}
        disabled={busy || guard.stale}
        className="p-1.5 rounded-lg bg-white/80 border border-gray-200 shadow-sm text-gray-500 hover:text-emerald-600 hover:bg-white hover:border-emerald-200 transition-colors text-xs disabled:opacity-50"
        title={guard.stale ? staleExportTitle(guard.reason) : "Copy chart to clipboard as PNG"}
      >
        ⧉
      </button>
      <button
        onClick={() => setOpen(o => !o)}
        disabled={guard.stale}
        className="p-1.5 rounded-lg bg-white/80 border border-gray-200 shadow-sm text-gray-500 hover:text-indigo-600 hover:bg-white hover:border-indigo-200 transition-colors text-xs disabled:opacity-50"
        title={guard.stale ? staleExportTitle(guard.reason) : "Export chart"}
      >
        ↓
      </button>
      {copyToast && (
        <span className="absolute -bottom-7 right-0 text-[10px] font-medium px-2 py-1 rounded bg-emerald-600 text-white shadow whitespace-nowrap">
          {copyToast}
        </span>
      )}

      {open && !guard.stale && (
        <div className="absolute right-0 top-8 bg-white border border-gray-200 rounded-xl shadow-xl p-4 w-52 space-y-3 z-20">
          <p className="text-xs font-semibold text-gray-700">Export Chart</p>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] text-gray-400 block mb-0.5">Width px</label>
              <input type="number" value={width} onChange={e => setWidth(+e.target.value)}
                className="select w-full text-xs py-0.5" min={400} max={4000} step={100} />
            </div>
            <div>
              <label className="text-[10px] text-gray-400 block mb-0.5">Height px</label>
              <input type="number" value={height} onChange={e => setHeight(+e.target.value)}
                className="select w-full text-xs py-0.5" min={200} max={3000} step={100} />
            </div>
          </div>

          <div className="grid grid-cols-4 gap-0 rounded overflow-hidden border border-gray-200">
            {(["png", "tiff", "jpeg", "svg"] as const).map(f => (
              <button key={f} onClick={() => setFmt(f)}
                className={`text-xs py-1 transition-colors ${fmt === f ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`}>
                {f.toUpperCase()}
              </button>
            ))}
          </div>

          {(fmt === "png" || fmt === "tiff" || fmt === "jpeg") && (
            <div>
              <label className="text-[10px] text-gray-400 block mb-0.5">DPI (resolution)</label>
              <div className="flex rounded overflow-hidden border border-gray-200">
                {[150, 300, 600].map(d => (
                  <button key={d} onClick={() => setDpi(d)}
                    className={`flex-1 text-xs py-1 transition-colors ${dpi === d ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`}>
                    {d}
                  </button>
                ))}
              </div>
              {fmt === "tiff" && (
                <p className="text-[9px] text-gray-400 mt-1 leading-tight">
                  Uncompressed RGB baseline TIFF. Journal-ready; file sizes are larger than PNG.
                </p>
              )}
            </div>
          )}

          <button onClick={downloadImage} disabled={busy} className="btn-primary w-full text-xs py-1.5">
            {busy ? "Exporting…" : `Download ${fmt.toUpperCase()}`}
          </button>

          <button onClick={() => setOpen(false)} className="w-full text-[10px] text-gray-400 hover:text-gray-700 py-0.5">
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}
