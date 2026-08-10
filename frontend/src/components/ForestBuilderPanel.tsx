import { useState, useRef, useMemo, useEffect } from "react";
import { usePlotLayout } from "../plotStyle";
import Plot from "../PlotComponent";
import PlotExporter from "./PlotExporter";
import ThreeCol from "./ThreeCol";
import { useStore, isNumericKind } from "../store";
import { fmtPubP } from "../lib/format";
import type { PlotCaptureHandle, PlotData } from "../lib/plotTypes";
import type { Data, Layout } from "plotly.js";

// ── Types ───────────────────────────────────────────────────────────────────

interface ForestRowInput {
  label: string;
  est: number | null;
  ci_low: number | null;
  ci_high: number | null;
  p: number | null;
  extra: string;          // free-text annotation: "(48 events)", "n=110", "OR", etc.
}

type ForestLayout = {
  customTitle: string;
  customSubtitle: string;
  leftHeader: string;   // bold column header above the row labels (e.g. "Time horizon")
  rightHeader: string;  // bold header above the value column (default "HR (95% CI)")
  xLabel: string;
  xScale: "log" | "linear";
  nullLine: number;        // 1 for HR/OR, 0 for β/Δ
  colorBy: "all_one" | "significance";
  markerStyle: "square" | "circle" | "diamond";
  showValueColumns: boolean;
  showExtra: boolean;
  pointColor: string;
  sigColor: string;
  nonSigColor: string;
  height: number;
  width?: number;   // explicit px width; undefined → fill the column
};

const DEFAULT_LAYOUT: ForestLayout = {
  customTitle: "",
  customSubtitle: "",
  leftHeader: "",
  rightHeader: "",
  xLabel: "Hazard ratio (95% CI), log scale",
  xScale: "log",
  nullLine: 1,
  colorBy: "significance",
  markerStyle: "square",
  showValueColumns: true,
  showExtra: true,
  pointColor: "#475569",
  sigColor: "#dc2626",
  nonSigColor: "#475569",
  height: 480,
};

const PRESETS: { name: string; rows: ForestRowInput[]; layout: Partial<ForestLayout> }[] = [
  {
    name: "Sensitivity — model specifications",
    rows: [
      { label: "Unadjusted",                          est: 2.03, ci_low: 1.02, ci_high: 4.03, p: 0.04,  extra: "" },
      { label: "Adjusted: age + sex + urea",          est: 1.27, ci_low: 0.61, ci_high: 2.63, p: 0.52,  extra: "" },
      { label: "Adjusted: age + EF + diabetes",       est: 1.38, ci_low: 0.69, ci_high: 2.78, p: 0.36,  extra: "" },
      { label: "Full model (age+sex+urea+EF+DM)",     est: 1.43, ci_low: 0.69, ci_high: 2.96, p: 0.34,  extra: "" },
      { label: "Landmark: deaths >30 d excluded",     est: 1.48, ci_low: 0.66, ci_high: 3.32, p: 0.34,  extra: "" },
      { label: "Landmark: deaths >180 d excluded",    est: 1.17, ci_low: 0.50, ci_high: 2.72, p: 0.72,  extra: "" },
    ],
    layout: {
      customTitle: "LDL <100 mg/dL (ref >130): hazard ratio for all-cause mortality across models",
      customSubtitle: "(n=388, 48 events; red = 95% CI excludes 1)",
      xLabel: "Hazard ratio (95% CI), log scale",
    },
  },
  {
    name: "Multiple endpoints / time horizons",
    rows: [
      { label: "30-day mortality",   est: 2.55, ci_low: 0.52, ci_high: 12.65, p: 0.25,    extra: "(6 events)"  },
      { label: "6-month mortality",  est: 2.57, ci_low: 0.83, ci_high:  7.98, p: 0.10,    extra: "(12 events)" },
      { label: "1-year mortality",   est: 5.67, ci_low: 2.15, ci_high: 14.92, p: 0.0005,  extra: "(19 events)" },
      { label: "2-year mortality",   est: 6.13, ci_low: 2.35, ci_high: 15.95, p: 0.0003,  extra: "(20 events)" },
      { label: "Overall (all-cause)", est: 2.36, ci_low: 1.34, ci_high:  4.17, p: 0.003,  extra: "(48 events)" },
    ],
    layout: {
      customTitle: "LDL <100 vs ≥100 mg/dL: crude hazard ratio for mortality by time horizon",
      customSubtitle: "(unadjusted; red = 95% CI excludes 1)",
      xLabel: "Hazard ratio (95% CI), log scale",
    },
  },
];

const emptyRow = (): ForestRowInput => ({
  label: "", est: null, ci_low: null, ci_high: null, p: null, extra: "",
});

/** A row the user has actually put something in. The sheet always carries a
 *  blank starter row, which must not survive an appending hand-off as a gap. */
const isFilled = (r: ForestRowInput): boolean =>
  r.label.trim() !== "" || r.est != null || r.ci_low != null || r.ci_high != null || r.p != null;

// ── Component ───────────────────────────────────────────────────────────────

/** What survives leaving the tab. Switching to Models and back unmounts this
 *  panel, and a half-built figure is not something to lose on a tab click —
 *  nor can rows be accumulated from several fits if they vanish in between. */
interface ForestSheet { rows: ForestRowInput[]; layout: ForestLayout; returnTo: { tab: string; label: string } | null }

export default function ForestBuilderPanel() {
  const themedBase = usePlotLayout();
  const sheet = useStore.getState().panelCache.forestBuilder as ForestSheet | undefined;
  const setPanelCache = useStore((s) => s.setPanelCache);
  const [rows, setRows] = useState<ForestRowInput[]>(sheet?.rows ?? [emptyRow()]);
  const [layout, setLayout] = useState<ForestLayout>(sheet?.layout ?? DEFAULT_LAYOUT);
  const plotRef = useRef<HTMLDivElement | null>(null);
  const plotCaptureRef = useRef<PlotCaptureHandle | null>(null);
  // Resizable plot box: the user can drag the bottom-right corner. A
  // ResizeObserver writes the dragged height back into layout.height (so it
  // persists across re-renders) and nudges Plotly to re-fit its width.
  const boxRef = useRef<HTMLDivElement>(null);
  const manualResizeRef = useRef(false);
  const firstObsRef = useRef(true);
  const [pastebox, setPastebox] = useState("");
  const [pasteOpen, setPasteOpen] = useState(false);

  const session = useStore((s) => s.session);
  const forestHandoff = useStore((s) => s.forestHandoff);
  const forestHandoffLayout = useStore((s) => s.forestHandoffLayout);
  const forestHandoffAppend = useStore((s) => s.forestHandoffAppend);
  const setForestHandoff = useStore((s) => s.setForestHandoff);
  const setActiveTab = useStore((s) => s.setActiveTab);
  // Where to jump back to (set when a panel handed off these rows), so the
  // user can return to the source model with their settings intact.
  const [returnTo, setReturnTo] = useState<{ tab: string; label: string } | null>(sheet?.returnTo ?? null);

  // Keep the sheet where a remount can find it.
  useEffect(() => {
    setPanelCache("forestBuilder", { rows, layout, returnTo });
  }, [rows, layout, returnTo, setPanelCache]);

  // Consume a cross-panel handoff (e.g. from the Cox time-horizon panel):
  // load the supplied rows + layout, then clear the buffer so a later
  // visit doesn't re-inject stale rows.
  useEffect(() => {
    if (forestHandoff && forestHandoff.length > 0) {
      const incoming = forestHandoff.map((r) => ({ ...r }));
      // An appending sender adds to what is already on the sheet, so two fits
      // (a continuous exposure and its dichotomised form, say) end up in one
      // figure. The starter blank row is dropped rather than kept as a gap.
      const kept = forestHandoffAppend ? rows.filter(isFilled) : [];
      setRows(kept.length ? [...kept, ...incoming] : incoming);
      // Layout belongs to whoever started the sheet: overwriting it on the
      // second hand-off would throw away a title the user has since edited.
      if (forestHandoffLayout && !kept.length) {
        const { returnTab, returnLabel, ...rest } = forestHandoffLayout;
        setLayout((l) => ({
          ...l,
          xScale: "log",
          nullLine: 1,
          ...rest,
        }));
        setReturnTo(returnTab ? { tab: returnTab, label: returnLabel || "← Back" } : null);
      }
      setForestHandoff(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [forestHandoff]);

  const [mapLabel, setMapLabel] = useState("");
  const [mapEst, setMapEst] = useState("");
  const [mapCiLow, setMapCiLow] = useState("");
  const [mapCiHigh, setMapCiHigh] = useState("");
  const [mapP, setMapP] = useState("");
  const [mapExtra, setMapExtra] = useState("");

  const allCols = useMemo(() => session?.columns?.map((c) => c.name) ?? [], [session]);
  const numCols = useMemo(() => session?.columns?.filter((c) => isNumericKind(c.kind)).map((c) => c.name) ?? [], [session]);

  useEffect(() => {
    if (!session || !session.columns) return;
    const cols = session.columns;
    
    // Auto-map Label
    const labelCol = cols.find((c) => {
      const n = c.name.toLowerCase();
      return n.includes("label") || n.includes("study") || n.includes("name") || n.includes("subgroup") || n.includes("variable");
    })?.name || cols[0]?.name || "";
    setMapLabel(labelCol);

    // Auto-map Est
    const estCol = cols.find((c) => {
      const n = c.name.toLowerCase();
      return isNumericKind(c.kind) && (n === "est" || n === "hr" || n === "or" || n === "rr" || n === "estimate" || n === "coef" || n === "mean" || n === "beta");
    })?.name || cols.find((c) => isNumericKind(c.kind))?.name || "";
    setMapEst(estCol);

    // Auto-map CI Low
    const ciLowCol = cols.find((c) => {
      const n = c.name.toLowerCase();
      return isNumericKind(c.kind) && (n.includes("low") || n.includes("lower") || n.includes("min") || n === "ci_l" || n === "cilow");
    })?.name || "";
    setMapCiLow(ciLowCol);

    // Auto-map CI High
    const ciHighCol = cols.find((c) => {
      const n = c.name.toLowerCase();
      return isNumericKind(c.kind) && (n.includes("high") || n.includes("upper") || n.includes("max") || n === "ci_h" || n === "cihigh");
    })?.name || "";
    setMapCiHigh(ciHighCol);

    // Auto-map p-value
    const pCol = cols.find((c) => {
      const n = c.name.toLowerCase();
      return isNumericKind(c.kind) && (n === "p" || n === "p_val" || n === "pval" || n === "pvalue" || n.includes("sig"));
    })?.name || "";
    setMapP(pCol);

    // Auto-map Extra
    const extraCol = cols.find((c) => {
      const n = c.name.toLowerCase();
      return n.includes("extra") || n.includes("event") || n.includes("note") || n === "n" || n.includes("weight");
    })?.name || "";
    setMapExtra(extraCol);
  }, [session]);

  const handleLoadFromDataset = () => {
    if (!session || !mapLabel || !mapEst || !mapCiLow || !mapCiHigh) return;

    const loadedRows: ForestRowInput[] = [];
    const previewData = session.preview ?? [];

    for (const row of previewData) {
      const labelVal = String(row[mapLabel] ?? "");
      const estVal = row[mapEst] != null ? parseFloat(String(row[mapEst])) : null;
      const ciLowVal = row[mapCiLow] != null ? parseFloat(String(row[mapCiLow])) : null;
      const ciHighVal = row[mapCiHigh] != null ? parseFloat(String(row[mapCiHigh])) : null;

      // Optional p-value
      let pVal: number | null = null;
      if (mapP && row[mapP] != null) {
        const parsedP = parseFloat(String(row[mapP]));
        if (!isNaN(parsedP)) pVal = parsedP;
      }

      // Optional extra
      let extraVal = "";
      if (mapExtra && row[mapExtra] != null) {
        extraVal = String(row[mapExtra]);
      }

      loadedRows.push({
        label: labelVal,
        est: Number.isNaN(estVal as number) ? null : estVal,
        ci_low: Number.isNaN(ciLowVal as number) ? null : ciLowVal,
        ci_high: Number.isNaN(ciHighVal as number) ? null : ciHighVal,
        p: pVal,
        extra: extraVal,
      });
    }

    if (loadedRows.length > 0) {
      setRows(loadedRows);
    }
  };

  const updateRow = (i: number, patch: Partial<ForestRowInput>) => {
    setRows((prev) => prev.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
  };
  const addRow = () => setRows((p) => [...p, emptyRow()]);
  const delRow = (i: number) => setRows((p) => p.filter((_, idx) => idx !== i));
  const moveRow = (i: number, delta: -1 | 1) => {
    setRows((prev) => {
      const j = i + delta;
      if (j < 0 || j >= prev.length) return prev;
      const out = prev.slice();
      [out[i], out[j]] = [out[j], out[i]];
      return out;
    });
  };
  const clearAll = () => { setRows([emptyRow()]); setLayout(DEFAULT_LAYOUT); };

  const loadPreset = (idx: number) => {
    const p = PRESETS[idx];
    setRows(p.rows.map((r) => ({ ...r })));
    setLayout({ ...DEFAULT_LAYOUT, ...p.layout });
  };

  // Parse CSV / TSV paste: label, est, ci_low, ci_high[, p, extra]
  const applyPaste = () => {
    const lines = pastebox.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    if (lines.length === 0) return;
    const next: ForestRowInput[] = [];
    for (const line of lines) {
      const parts = line.split(/\t|,/).map((s) => s.trim());
      if (parts.length < 4) continue;
      const [label, est, lo, hi, p, extra] = parts;
      // Skip header-looking lines
      if (isNaN(parseFloat(est))) continue;
      next.push({
        label,
        est: parseFloat(est),
        ci_low: parseFloat(lo),
        ci_high: parseFloat(hi),
        p: p != null && p !== "" && !isNaN(parseFloat(p)) ? parseFloat(p) : null,
        extra: extra ?? "",
      });
    }
    if (next.length) setRows(next);
    setPasteOpen(false);
    setPastebox("");
  };

  // Valid rows only
  const validRows = useMemo(
    () => rows.filter((r) => r.label && r.est != null && r.ci_low != null && r.ci_high != null),
    [rows],
  );

  // Helpers
  const fmtCI = (r: ForestRowInput) =>
    `${r.est!.toFixed(2)} (${r.ci_low!.toFixed(2)}–${r.ci_high!.toFixed(2)})`;
  const rowColor = (r: ForestRowInput): string => {
    if (layout.colorBy === "all_one") return layout.pointColor;
    const lo = r.ci_low ?? 0;
    const hi = r.ci_high ?? 0;
    const includesNull = lo <= layout.nullLine && hi >= layout.nullLine;
    return includesNull ? layout.nonSigColor : layout.sigColor;
  };

  const symbol = layout.markerStyle;

  // Plotly data
  const n = validRows.length;
  const yIdx = validRows.map((_, i) => n - 1 - i); // top-down order
  const colors = validRows.map((r) => rowColor(r));

  // Left margin grows to fit the longest row label (and the left header), so
  // long labels like "Fully adjusted — <100 vs >130" aren't clipped. The
  // Width slider sizes the plotting area, not this label gutter.
  const labelMargin = useMemo(() => {
    const maxChars = Math.max(
      (layout.leftHeader ?? "").length,
      ...validRows.map((r) => (r.label ?? "").length),
      8,
    );
    return Math.min(380, Math.max(130, Math.round(maxChars * 6.6) + 26));
  }, [validRows, layout.leftHeader]);

  // Clean log-axis ticks. Plotly's default log axis labels every minor tick
  // (0.06, 0.07, … shown as bare "6 7 8") which clutters the figure. Pick a
  // 1–2–5 sequence spanning the data and label those only.
  const logTicks = useMemo(() => {
    if (layout.xScale !== "log" || validRows.length === 0) return null;
    const vals: number[] = [layout.nullLine];
    for (const r of validRows) {
      if (r.ci_low != null && r.ci_low > 0) vals.push(r.ci_low);
      if (r.ci_high != null && r.ci_high > 0) vals.push(r.ci_high);
      if (r.est != null && r.est > 0) vals.push(r.est);
    }
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    if (!(lo > 0) || !(hi > 0) || !isFinite(lo) || !isFinite(hi)) return null;
    const ticks: number[] = [];
    const startDec = Math.floor(Math.log10(lo));
    const endDec = Math.ceil(Math.log10(hi));
    for (let d = startDec; d <= endDec; d++) {
      for (const m of [1, 2, 5]) {
        const v = m * Math.pow(10, d);
        if (v >= lo * 0.9 && v <= hi * 1.1) ticks.push(v);
      }
    }
    if (ticks.length === 0) return null;
    const fmt = (v: number) => {
      if (v >= 1) return String(Math.round(v * 100) / 100).replace(/\.0+$/, "");
      // strip trailing zeros for sub-1 values: 0.10 → 0.1, 0.50 → 0.5
      return String(parseFloat(v.toPrecision(2)));
    };
    return { tickvals: ticks, ticktext: ticks.map(fmt) };
  }, [layout.xScale, layout.nullLine, validRows]);

  // Height that snugly fits the rows — used as the initial size so the plot
  // isn't stretched tall over a few rows. The user can override by dragging.
  const fittedHeight = Math.min(
    660,
    Math.max(220, 70 + (layout.customTitle ? 46 : 0) + Math.max(n, 1) * 72),
  );
  useEffect(() => {
    if (!manualResizeRef.current) {
      setLayout((l) => (l.height === fittedHeight ? l : { ...l, height: fittedHeight }));
    }
  }, [fittedHeight]);

  // Sync user drags (bottom-right corner) back into layout.height + refit width.
  useEffect(() => {
    const el = boxRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      if (firstObsRef.current) { firstObsRef.current = false; return; }
      manualResizeRef.current = true;
      const h = Math.round(el.clientHeight);
      setLayout((l) => (Math.abs(l.height - h) <= 2 ? l : { ...l, height: h }));
      window.dispatchEvent(new Event("resize"));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // One trace PER ROW. Plotly's error_x.color only accepts a single color
  // string (an array is silently ignored → default black whiskers), so to
  // colour each confidence interval to match its marker we emit a separate
  // trace per point.
  const forestTraces: PlotData[] = validRows.map((r, i) => {
    const c = colors[i];
    const yi = yIdx[i];
    return {
      type: "scatter",
      mode: "markers",
      x: [r.est],
      y: [yi],
      error_x: {
        type: "data", symmetric: false,
        array: [r.ci_high! - r.est!],
        arrayminus: [r.est! - r.ci_low!],
        thickness: 2.2, width: 7,
        color: c,
      },
      marker: {
        size: 12,
        symbol,
        color: c,
        line: { color: "#ffffff", width: 0.5 },
      },
      hovertemplate:
        `<b>${r.label}</b><br>${fmtCI(r)}${r.p != null ? `<br>${fmtPubP(r.p)}` : ""}${r.extra ? `<br>${r.extra}` : ""}<extra></extra>`,
      showlegend: false,
    };
  });

  // Value-column annotations (right side)
  const showCols = layout.showValueColumns;
  // The value column ("HR (CI), p" + optional extra) is a FIXED-PX gutter in
  // the right margin, sized to its longest content — so the left-anchored text
  // never runs off the SVG edge on a narrow embedded plot (it would clip
  // 'p=0.04' → 'p=0.0' otherwise). Headers/markers stay in the plot area.
  const valText = (r: ForestRowInput) =>
    `${fmtCI(r)}${r.p != null ? `, ${fmtPubP(r.p)}` : ""}`;
  const maxValChars = Math.max(
    (layout.rightHeader ?? "HR (95% CI)").length,
    ...validRows.map((r) => valText(r).length),
    10,
  );
  const maxExtraChars = (showCols && layout.showExtra)
    ? Math.max(0, ...validRows.map((r) => (r.extra ?? "").length))
    : 0;
  const rightGutter = showCols
    ? Math.min(440, Math.round((maxValChars + (maxExtraChars ? maxExtraChars + 2 : 0)) * 6.6) + 34)
    : 24;
  // Plot area fills the paper; the value column lives in the right margin
  // gutter (TX1 just past x=1), so markers use the full width either way.
  const forestRight = 1.0;
  const TX1 = 1.02;   // HR (95% CI), p — just past the plot's right edge, in the gutter
  const TX2_PX = Math.round((maxValChars + 2) * 6.6);  // extra column offset (px) from TX1

  const annotations: PlotData[] = [];
  // Column headers live in the TOP MARGIN (yref paper, pixel yshift above the
  // plot area). The top margin is sized below to fit them — a fractional
  // y like 1.04 scales with plot height and gets clipped on short plots.
  const showHeaders = !!layout.leftHeader || showCols;
  if (layout.leftHeader) {
    annotations.push({
      xref: "paper", yref: "paper", x: 0, y: 1,
      xanchor: "left", yanchor: "bottom", xshift: -(labelMargin - 10), yshift: 14,
      text: `<b>${layout.leftHeader}</b>`, showarrow: false,
      font: { size: 12.5, color: "#1f2937" },
    });
  }
  // Row labels: drawn as annotations (NOT y-axis ticks) so they LEFT-align at
  // the figure's left edge like a publication forest plot — Plotly tick labels
  // can only right-align against the axis.
  validRows.forEach((r, i) => {
    const c = rowColor(r);
    const sig = layout.colorBy === "significance" && c === layout.sigColor;
    annotations.push({
      xref: "paper", yref: "y", x: 0, y: yIdx[i],
      xanchor: "left", yanchor: "middle", xshift: -(labelMargin - 10),
      text: r.label ?? "", showarrow: false,
      font: { size: 12, color: sig ? c : "#374151" },
    });
  });
  if (showCols) {
    annotations.push({
      xref: "paper", yref: "paper", x: TX1, y: 1,
      xanchor: "left", yanchor: "bottom", yshift: 14,
      text: `<b>${layout.rightHeader || "HR (95% CI)"}</b>`, showarrow: false,
      font: { size: 12.5, color: "#1f2937" },
    });
    validRows.forEach((r, i) => {
      const c = rowColor(r);
      const sig = layout.colorBy === "significance" && c === layout.sigColor;
      const v = valText(r);
      const yi = yIdx[i];
      annotations.push({
        xref: "paper", yref: "y", x: TX1, y: yi,
        xanchor: "left", yanchor: "middle",
        text: sig ? `<b>${v}</b>` : v,
        showarrow: false,
        font: { size: 11, color: c },
      });
      if (layout.showExtra && r.extra) {
        annotations.push({
          xref: "paper", yref: "y", x: TX1, y: yi, xshift: TX2_PX,
          xanchor: "left", yanchor: "middle",
          text: r.extra,
          showarrow: false,
          font: { size: 11, color: c },
        });
      }
    });
  }

  // Title block
  const titleHtml = layout.customSubtitle
    ? `${layout.customTitle}<br><span style="font-size:11px;color:#6b7280">${layout.customSubtitle}</span>`
    : layout.customTitle;

  const leftCol = (
    <div className="space-y-4 max-h-[82vh] overflow-y-auto pr-1">
      {returnTo && (
        <button
          onClick={() => setActiveTab(returnTo.tab)}
          className="w-full px-3 py-2 text-xs font-medium rounded-lg border border-indigo-300 text-indigo-700 bg-indigo-50 hover:bg-indigo-100 transition-colors"
        >
          {returnTo.label}
        </button>
      )}
      {/* Load from Active Dataset */}
      <div className="panel space-y-3 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-700">Load from Active Dataset</h3>
          {session && (
            <span className="text-[9px] bg-indigo-50 text-indigo-700 px-1.5 py-0.5 rounded font-medium border border-indigo-100">
              {session.filename}
            </span>
          )}
        </div>

        {!session ? (
          <p className="text-[11px] text-gray-400 leading-relaxed">
            No active dataset loaded. Please upload a dataset in the <strong>Data</strong> tab first to map columns directly.
          </p>
        ) : (
          <div className="space-y-2 text-xs">
            <div className="grid grid-cols-2 gap-2">
              <label className="flex flex-col gap-0.5">
                <span className="text-gray-500 text-[10px]">Label Column</span>
                <select value={mapLabel} onChange={(e) => setMapLabel(e.target.value)}
                  className="w-full text-xs border border-gray-300 rounded px-1 py-1 focus:outline-none focus:border-indigo-400 bg-white">
                  <option value="">— select —</option>
                  {allCols.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>

              <label className="flex flex-col gap-0.5">
                <span className="text-gray-500 text-[10px]">Estimate Column</span>
                <select value={mapEst} onChange={(e) => setMapEst(e.target.value)}
                  className="w-full text-xs border border-gray-300 rounded px-1 py-1 focus:outline-none focus:border-indigo-400 bg-white">
                  <option value="">— select —</option>
                  {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <label className="flex flex-col gap-0.5">
                <span className="text-gray-500 text-[10px]">CI Low Column</span>
                <select value={mapCiLow} onChange={(e) => setMapCiLow(e.target.value)}
                  className="w-full text-xs border border-gray-300 rounded px-1 py-1 focus:outline-none focus:border-indigo-400 bg-white">
                  <option value="">— select —</option>
                  {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>

              <label className="flex flex-col gap-0.5">
                <span className="text-gray-500 text-[10px]">CI High Column</span>
                <select value={mapCiHigh} onChange={(e) => setMapCiHigh(e.target.value)}
                  className="w-full text-xs border border-gray-300 rounded px-1 py-1 focus:outline-none focus:border-indigo-400 bg-white">
                  <option value="">— select —</option>
                  {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <label className="flex flex-col gap-0.5">
                <span className="text-gray-500 text-[10px]">p-value (Opt)</span>
                <select value={mapP} onChange={(e) => setMapP(e.target.value)}
                  className="w-full text-xs border border-gray-300 rounded px-1 py-1 focus:outline-none focus:border-indigo-400 bg-white">
                  <option value="">— none —</option>
                  {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>

              <label className="flex flex-col gap-0.5">
                <span className="text-gray-500 text-[10px]">Extra Col (Opt)</span>
                <select value={mapExtra} onChange={(e) => setMapExtra(e.target.value)}
                  className="w-full text-xs border border-gray-300 rounded px-1 py-1 focus:outline-none focus:border-indigo-400 bg-white">
                  <option value="">— none —</option>
                  {allCols.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
            </div>

            <button
              onClick={handleLoadFromDataset}
              disabled={!mapLabel || !mapEst || !mapCiLow || !mapCiHigh}
              className="w-full mt-2 text-xs bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-200 disabled:text-gray-400 text-white rounded py-1.5 transition-colors font-medium flex items-center justify-center gap-1.5 shadow-sm">
              📥 Load Dataset Rows
            </button>
          </div>
        )}
      </div>

      <div className="panel space-y-3 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
        <h3 className="text-sm font-semibold text-gray-700">Presets</h3>
        <div className="space-y-1.5">
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => loadPreset(i)}
              className="w-full text-left text-xs px-2 py-1 rounded border border-gray-300 text-gray-700 hover:bg-indigo-50 hover:border-indigo-300 transition-colors">
              {p.name}
            </button>
          ))}
          <button onClick={clearAll}
            className="w-full text-left text-xs px-2 py-1 rounded border border-orange-300 text-orange-600 hover:bg-orange-50 transition-colors">
            ✕ Clear all
          </button>
        </div>
      </div>

      <div className="panel space-y-2 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
        <h3 className="text-sm font-semibold text-gray-700">Title</h3>
        <input
          value={layout.customTitle}
          onChange={(e) => setLayout((l) => ({ ...l, customTitle: e.target.value }))}
          placeholder="Plot title (optional)"
          className="w-full text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400"
        />
        <input
          value={layout.customSubtitle}
          onChange={(e) => setLayout((l) => ({ ...l, customSubtitle: e.target.value }))}
          placeholder="Subtitle (optional)"
          className="w-full text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400"
        />
        <input
          value={layout.xLabel}
          onChange={(e) => setLayout((l) => ({ ...l, xLabel: e.target.value }))}
          placeholder="X axis label"
          className="w-full text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400"
        />
        <div className="grid grid-cols-2 gap-2">
          <input
            value={layout.leftHeader}
            onChange={(e) => setLayout((l) => ({ ...l, leftHeader: e.target.value }))}
            placeholder="Left header (e.g. Time horizon)"
            className="w-full text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400"
          />
          <input
            value={layout.rightHeader}
            onChange={(e) => setLayout((l) => ({ ...l, rightHeader: e.target.value }))}
            placeholder="Right header (HR (95% CI))"
            className="w-full text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400"
          />
        </div>
        <p className="text-[10px] text-gray-400">Left header sits above the row labels; right header above the value column.</p>
      </div>

      <div className="panel space-y-2 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
        <h3 className="text-sm font-semibold text-gray-700">Scale & reference</h3>
        <div className="inline-flex rounded-md border border-gray-300 overflow-hidden text-[10px] w-full">
          {(["log", "linear"] as const).map((s) => (
            <button key={s}
              onClick={() => setLayout((l) => ({ ...l, xScale: s, nullLine: s === "log" ? 1 : 0 }))}
              className={`flex-1 py-1 ${layout.xScale === s ? "bg-indigo-600 text-white" : "bg-white hover:bg-gray-50 text-gray-600"}`}>
              {s === "log" ? "Log (HR/OR/RR)" : "Linear (β/Δ)"}
            </button>
          ))}
        </div>
        <label className="flex items-center justify-between text-[11px] text-gray-600">
          <span>Null reference line at:</span>
          <input type="number" step="any"
            value={layout.nullLine}
            onChange={(e) => setLayout((l) => ({ ...l, nullLine: parseFloat(e.target.value) || 0 }))}
            className="w-16 text-xs border border-gray-300 rounded px-1 py-0.5 focus:outline-none focus:border-indigo-400 text-right"
          />
        </label>
      </div>

      <div className="panel space-y-2 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
        <h3 className="text-sm font-semibold text-gray-700">Color</h3>
        <div className="inline-flex rounded-md border border-gray-300 overflow-hidden text-[10px] w-full">
          {(["significance", "all_one"] as const).map((m) => (
            <button key={m}
              onClick={() => setLayout((l) => ({ ...l, colorBy: m }))}
              className={`flex-1 py-1 ${layout.colorBy === m ? "bg-indigo-600 text-white" : "bg-white hover:bg-gray-50 text-gray-600"}`}>
              {m === "significance" ? "By significance" : "Single color"}
            </button>
          ))}
        </div>
        {layout.colorBy === "significance" ? (
          <div className="grid grid-cols-2 gap-2 text-[10px]">
            <label className="flex flex-col gap-0.5">
              <span className="text-gray-500">CI excludes null</span>
              <input type="color" value={layout.sigColor}
                onChange={(e) => setLayout((l) => ({ ...l, sigColor: e.target.value }))}
                className="w-full h-6 border border-gray-300 rounded cursor-pointer" />
            </label>
            <label className="flex flex-col gap-0.5">
              <span className="text-gray-500">Includes null</span>
              <input type="color" value={layout.nonSigColor}
                onChange={(e) => setLayout((l) => ({ ...l, nonSigColor: e.target.value }))}
                className="w-full h-6 border border-gray-300 rounded cursor-pointer" />
            </label>
          </div>
        ) : (
          <label className="flex items-center justify-between text-[10px] text-gray-500">
            <span>Point color</span>
            <input type="color" value={layout.pointColor}
              onChange={(e) => setLayout((l) => ({ ...l, pointColor: e.target.value }))}
              className="w-10 h-6 border border-gray-300 rounded cursor-pointer" />
          </label>
        )}
      </div>

      <div className="panel space-y-2 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
        <h3 className="text-sm font-semibold text-gray-700">Marker & value columns</h3>
        <div className="inline-flex rounded-md border border-gray-300 overflow-hidden text-[10px] w-full">
          {(["square", "circle", "diamond"] as const).map((s) => (
            <button key={s}
              onClick={() => setLayout((l) => ({ ...l, markerStyle: s }))}
              className={`flex-1 py-1 ${layout.markerStyle === s ? "bg-indigo-600 text-white" : "bg-white hover:bg-gray-50 text-gray-600"}`}>
              {s}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-[11px] text-gray-600">
          <input type="checkbox" checked={layout.showValueColumns}
            onChange={(e) => setLayout((l) => ({ ...l, showValueColumns: e.target.checked }))}
            className="accent-indigo-500" />
          Show <strong>HR (95% CI), p</strong> column
        </label>
        <label className="flex items-center gap-2 text-[11px] text-gray-600">
          <input type="checkbox" checked={layout.showExtra}
            onChange={(e) => setLayout((l) => ({ ...l, showExtra: e.target.checked }))}
            className="accent-indigo-500" />
          Show extra annotation column (events / n / etc.)
        </label>
        <label className="flex items-center justify-between text-[11px] text-gray-600">
          <span>Plot height</span>
          <input type="number" min={200} max={1200} step={20}
            value={layout.height}
            onChange={(e) => setLayout((l) => ({ ...l, height: parseInt(e.target.value) || 480 }))}
            className="w-16 text-xs border border-gray-300 rounded px-1 py-0.5 focus:outline-none focus:border-indigo-400 text-right"
          />
        </label>
      </div>

      <div className="panel space-y-2 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
        <h3 className="text-sm font-semibold text-gray-700">Bulk paste (CSV/TSV)</h3>
        <p className="text-[10px] text-gray-500 leading-snug">
          One row per line:<br/>
          <code className="text-[10px]">label, est, ci_low, ci_high[, p][, extra]</code>
        </p>
        {pasteOpen ? (
          <>
            <textarea
              value={pastebox}
              onChange={(e) => setPastebox(e.target.value)}
              placeholder={"Unadjusted, 2.03, 1.02, 4.03, 0.04\nFull model, 1.43, 0.69, 2.96, 0.34, (48 events)"}
              rows={6}
              className="w-full text-[10px] font-mono border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400"
            />
            <div className="flex gap-1 mt-2">
              <button onClick={applyPaste} className="flex-1 text-xs bg-indigo-600 text-white rounded py-1 hover:bg-indigo-700 transition-colors">
                Apply
              </button>
              <button onClick={() => { setPasteOpen(false); setPastebox(""); }} className="text-xs text-gray-500 hover:text-red-500 px-2 py-1 transition-colors">
                Cancel
              </button>
            </div>
          </>
        ) : (
          <button onClick={() => setPasteOpen(true)}
            className="w-full text-xs px-2 py-1.5 rounded border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors">
            📋 Paste rows
          </button>
        )}
      </div>
    </div>
  );

  const middleCol = (
    <div className="space-y-4">
      {validRows.length === 0 ? (
        <div className="panel min-h-[480px] flex flex-col items-center justify-center text-slate-400 p-8 text-center bg-white border border-gray-200 shadow-sm rounded-2xl relative overflow-hidden">
          {/* High-Fidelity SVG Preview Illustration */}
          <div className="w-full max-w-lg mb-6 opacity-85 hover:opacity-100 transition-opacity duration-300 bg-slate-50/50 p-4 rounded-xl border border-slate-100 shadow-inner">
            <svg width="100%" height="220" viewBox="0 0 480 220" fill="none" xmlns="http://www.w3.org/2000/svg" className="mx-auto select-none">
              <defs>
                <linearGradient id="forestSig" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#ef4444" />
                  <stop offset="100%" stopColor="#dc2626" />
                </linearGradient>
                <linearGradient id="forestNonSig" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#64748b" />
                  <stop offset="100%" stopColor="#475569" />
                </linearGradient>
                <linearGradient id="forestPooled" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#6366f1" />
                  <stop offset="100%" stopColor="#4f46e5" />
                </linearGradient>
              </defs>

              {/* Header row labels */}
              <text x="20" y="20" fill="#475569" fontSize="9" fontWeight="bold" fontFamily="system-ui" textAnchor="start">Study or Subgroup</text>
              <text x="200" y="20" fill="#475569" fontSize="9" fontWeight="bold" fontFamily="system-ui" textAnchor="middle">Hazard Ratio (95% CI)</text>
              <text x="400" y="20" fill="#475569" fontSize="9" fontWeight="bold" fontFamily="system-ui" textAnchor="start">Weight / Estimate</text>

              {/* Grid Horizontal Separators */}
              <line x1="20" y1="28" x2="460" y2="28" stroke="#cbd5e1" strokeWidth="1" />
              <line x1="20" y1="185" x2="460" y2="185" stroke="#cbd5e1" strokeWidth="1" />

              {/* Null reference line (dashed) at x=200 */}
              <line x1="200" y1="30" x2="200" y2="180" stroke="#94a3b8" strokeDasharray="3 3" strokeWidth="1.5" />

              {/* Study 1 */}
              <text x="20" y="55" fill="#334155" fontSize="10" fontFamily="system-ui">Model A (Unadjusted)</text>
              {/* CI Line */}
              <line x1="220" y1="51" x2="340" y2="51" stroke="#ef4444" strokeWidth="1.8" />
              <line x1="220" y1="47" x2="220" y2="55" stroke="#ef4444" strokeWidth="1.8" />
              <line x1="340" y1="47" x2="340" y2="55" stroke="#ef4444" strokeWidth="1.8" />
              {/* Square marker at est x=270 */}
              <rect x="264" y="45" width="12" height="12" rx="1.5" fill="url(#forestSig)" />
              <text x="400" y="55" fill="#1e293b" fontSize="10" fontFamily="system-ui" fontWeight="medium">2.03 (1.02–4.03)</text>

              {/* Study 2 */}
              <text x="20" y="90" fill="#334155" fontSize="10" fontFamily="system-ui">Model B (Adjusted: Age + Sex)</text>
              {/* CI Line */}
              <line x1="140" y1="86" x2="260" y2="86" stroke="#64748b" strokeWidth="1.8" />
              <line x1="140" y1="82" x2="140" y2="90" stroke="#64748b" strokeWidth="1.8" />
              <line x1="260" y1="82" x2="260" y2="90" stroke="#64748b" strokeWidth="1.8" />
              {/* Square marker at est x=190 */}
              <rect x="184" y="80" width="12" height="12" rx="1.5" fill="url(#forestNonSig)" />
              <text x="400" y="90" fill="#64748b" fontSize="10" fontFamily="system-ui">1.27 (0.61–2.63)</text>

              {/* Study 3 */}
              <text x="20" y="125" fill="#334155" fontSize="10" fontFamily="system-ui">Model C (Full Adjusted)</text>
              {/* CI Line */}
              <line x1="210" y1="121" x2="310" y2="121" stroke="#ef4444" strokeWidth="1.8" />
              <line x1="210" y1="117" x2="210" y2="125" stroke="#ef4444" strokeWidth="1.8" />
              <line x1="310" y1="117" x2="310" y2="125" stroke="#ef4444" strokeWidth="1.8" />
              {/* Square marker at est x=250 */}
              <rect x="244" y="115" width="12" height="12" rx="1.5" fill="url(#forestSig)" />
              <text x="400" y="125" fill="#1e293b" fontSize="10" fontFamily="system-ui" fontWeight="medium">1.43 (1.09–2.96)</text>

              {/* Pooled Effect (Diamond) */}
              <text x="20" y="165" fill="#1e293b" fontSize="10" fontWeight="bold" fontFamily="system-ui">Pooled Estimate (Overall)</text>
              {/* Diamond centered at x=235 (range 210 to 260) */}
              <polygon points="235,155 250,161 235,167 220,161" fill="url(#forestPooled)" stroke="#4f46e5" strokeWidth="1" />
              <text x="400" y="165" fill="#4f46e5" fontSize="10" fontWeight="bold" fontFamily="system-ui">1.36 (1.10–1.68)</text>

              {/* Bottom scale labels */}
              <text x="100" y="205" fill="#94a3b8" fontSize="8" fontFamily="system-ui" textAnchor="middle">0.5</text>
              <text x="200" y="205" fill="#475569" fontSize="8" fontWeight="bold" fontFamily="system-ui" textAnchor="middle">1.0 (Null)</text>
              <text x="300" y="205" fill="#94a3b8" fontSize="8" fontFamily="system-ui" textAnchor="middle">2.0</text>
              <text x="400" y="205" fill="#94a3b8" fontSize="8" fontFamily="system-ui" textAnchor="middle">4.0</text>
              <line x1="100" y1="192" x2="100" y2="196" stroke="#cbd5e1" strokeWidth="1" />
              <line x1="200" y1="192" x2="200" y2="196" stroke="#cbd5e1" strokeWidth="1" />
              <line x1="300" y1="192" x2="300" y2="196" stroke="#cbd5e1" strokeWidth="1" />
              <line x1="400" y1="192" x2="400" y2="196" stroke="#cbd5e1" strokeWidth="1" />
              <line x1="60" y1="192" x2="440" y2="192" stroke="#cbd5e1" strokeWidth="1" />
            </svg>
          </div>
          <span className="text-sm font-bold text-slate-800 tracking-tight">Interactive Forest Plot Builder</span>
          <p className="text-xs text-slate-400 max-w-md mt-1.5 leading-relaxed text-center">
            Create publication-ready forest plots for sensitivity analyses, multiple endpoints, or meta-analyses. Use the controls on the left to select presets, customize styling and scales, or paste CSV/TSV data, and manage individual data rows on the right.
          </p>
        </div>
      ) : (
        <div className="panel relative space-y-2 bg-white border border-gray-200 shadow-sm rounded-2xl p-4">
          {/* PlotExporter floats top-right (absolute); keep it a direct child
              of the relative panel so it anchors correctly. */}
          <PlotExporter plotRef={plotCaptureRef} title={`Forest_custom`} />
          <div className="flex items-center justify-between pr-16">
            <h3 className="text-sm font-semibold text-gray-700">Forest plot</h3>
          </div>
          {/* Size controls — explicit sliders. CSS corner-drag is unreliable
              over a Plotly plot (its drag layer eats the pointer), so these
              are the dependable way to resize width + height. */}
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 pb-1">
            <label className="flex items-center gap-1.5 text-[10px] text-gray-500">
              <span className="font-medium">Width</span>
              <input type="range" min={520} max={1400} step={20}
                value={layout.width ?? 900}
                onChange={(e) => setLayout((l) => ({ ...l, width: Number(e.target.value) }))}
                className="accent-indigo-500" style={{ width: 120 }} />
              <span className="tabular-nums w-9">{layout.width ?? "auto"}</span>
              {layout.width != null && (
                <button onClick={() => setLayout((l) => ({ ...l, width: undefined }))}
                  className="text-indigo-500 hover:text-indigo-700">auto</button>
              )}
            </label>
            <label className="flex items-center gap-1.5 text-[10px] text-gray-500">
              <span className="font-medium">Height</span>
              <input type="range" min={220} max={900} step={20}
                value={layout.height}
                onChange={(e) => { manualResizeRef.current = true; setLayout((l) => ({ ...l, height: Number(e.target.value) })); }}
                className="accent-indigo-500" style={{ width: 120 }} />
              <span className="tabular-nums w-9">{layout.height}</span>
            </label>
          </div>
          {/* Quick label editor — rename axis / column headers right here
              before exporting (also available in the left controls). */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-1.5 pb-1">
            <input
              value={layout.xLabel}
              onChange={(e) => setLayout((l) => ({ ...l, xLabel: e.target.value }))}
              placeholder="X-axis label"
              className="text-[11px] border border-gray-200 rounded px-2 py-1 focus:outline-none focus:border-indigo-400"
            />
            <input
              value={layout.leftHeader}
              onChange={(e) => setLayout((l) => ({ ...l, leftHeader: e.target.value }))}
              placeholder="Left header (e.g. Time horizon)"
              className="text-[11px] border border-gray-200 rounded px-2 py-1 focus:outline-none focus:border-indigo-400"
            />
            <input
              value={layout.rightHeader}
              onChange={(e) => setLayout((l) => ({ ...l, rightHeader: e.target.value }))}
              placeholder="Right header (HR (95% CI))"
              className="text-[11px] border border-gray-200 rounded px-2 py-1 focus:outline-none focus:border-indigo-400"
            />
          </div>
          {/* Resizable box — bottom-right corner handle (CSS resize). */}
          <div
            ref={boxRef}
            style={{
              height: layout.height,
              width: layout.width != null ? layout.width : "100%",
              minHeight: 220,
              maxWidth: "100%",
              overflow: "hidden",
            }}
          >
          <div ref={plotRef} style={{ width: "100%", height: "100%" }}>
            <Plot
              data={forestTraces as unknown as Data[]}
              layout={{
                ...themedBase,
                autosize: true,
                // Extra top room when column headers are shown — they render
                // in the margin (yref paper + yshift), not the plot area.
                // Right margin = the value-column gutter so 'HR (CI), p' text
                // never clips off the SVG edge.
                margin: { t: (layout.customTitle ? 60 : 30) + (showHeaders ? 30 : 0), r: rightGutter, b: 60, l: labelMargin },
                title: layout.customTitle
                  ? { text: titleHtml, font: { size: 13, color: "#1f2937" }, x: 0.5, xanchor: "center" }
                  : undefined,
                xaxis: {
                  type: layout.xScale,
                  gridcolor: "#e5e7eb",
                  zeroline: false,
                  domain: [0, forestRight],
                  title: { text: layout.xLabel, font: { size: 11, color: "#374151" } },
                  // Clean 1–2–5 ticks on log scale; suppress Plotly's noisy
                  // minor-tick labels (the "6 7 8 9" clutter) and minor grid.
                  ...(logTicks
                    ? {
                        tickmode: "array",
                        tickvals: logTicks.tickvals,
                        ticktext: logTicks.ticktext,
                        minor: { showgrid: false, ticks: "" },
                      }
                    : {}),
                },
                yaxis: {
                  // Labels are drawn as left-aligned annotations in the left
                  // margin (see above) — native tick labels would right-align
                  // against the axis, unlike a publication forest plot.
                  showticklabels: false,
                  ticks: "",
                  range: [-0.6, n - 0.4],
                  gridcolor: "transparent",
                  zeroline: false,
                },
                shapes: [
                  {
                    type: "line", x0: layout.nullLine, x1: layout.nullLine,
                    xref: "x", yref: "paper", y0: 0, y1: 1,
                    line: { color: "#9ca3af", dash: "dash", width: 1.4 },
                  },
                ],
                annotations: annotations as unknown as Layout["annotations"],
                showlegend: false,
              }}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
              config={{ responsive: true, displaylogo: false }}
              onInitialized={(_: object, gd: HTMLElement) => { plotCaptureRef.current = gd as unknown as PlotCaptureHandle; }}
              onUpdate={(_: object, gd: HTMLElement) => { plotCaptureRef.current = gd as unknown as PlotCaptureHandle; }}
            />
          </div>
          </div>
        </div>
      )}
    </div>
  );

  const rightCol = (
    <div className="panel space-y-2 bg-white border border-gray-200 shadow-sm rounded-2xl p-4 max-h-[82vh] overflow-y-auto">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">Rows</h3>
        <span className="text-[10px] text-gray-400">{validRows.length} of {rows.length} valid</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-[10px] text-gray-500 border-b border-gray-200">
              <th className="text-left px-1 py-1 w-8">#</th>
              <th className="text-left px-1 py-1">Label</th>
              <th className="text-right px-1 py-1 w-16">Est</th>
              <th className="text-right px-1 py-1 w-16">CI low</th>
              <th className="text-right px-1 py-1 w-16">CI high</th>
              <th className="text-right px-1 py-1 w-16"><i>p</i></th>
              <th className="text-left px-1 py-1 w-32">Extra</th>
              <th className="px-1 py-1 w-16"></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                <td className="px-1 py-0.5 text-gray-400">{i + 1}</td>
                <td className="px-1 py-0.5">
                  <input value={r.label} placeholder="Label"
                    onChange={(e) => updateRow(i, { label: e.target.value })}
                    className="w-full border border-transparent rounded px-1 py-0.5 focus:outline-none focus:border-indigo-400 hover:border-gray-300" />
                </td>
                {(["est", "ci_low", "ci_high", "p"] as const).map((k) => (
                  <td key={k} className="px-1 py-0.5">
                    <input type="number" step="any"
                      value={r[k] == null ? "" : r[k] as number}
                      placeholder="—"
                      onChange={(e) => {
                        const v = e.target.value === "" ? null : parseFloat(e.target.value);
                        updateRow(i, { [k]: Number.isNaN(v as number) ? null : v } as Partial<ForestRowInput>);
                      }}
                      className="w-full text-right border border-transparent rounded px-1 py-0.5 focus:outline-none focus:border-indigo-400 hover:border-gray-300" />
                  </td>
                ))}
                <td className="px-1 py-0.5">
                  <input value={r.extra}
                    placeholder="(48 events)"
                    onChange={(e) => updateRow(i, { extra: e.target.value })}
                    className="w-full border border-transparent rounded px-1 py-0.5 focus:outline-none focus:border-indigo-400 hover:border-gray-300" />
                </td>
                <td className="px-1 py-0.5 text-right whitespace-nowrap">
                  <button onClick={() => moveRow(i, -1)} className="text-gray-400 hover:text-indigo-600 px-0.5" title="Move up">▲</button>
                  <button onClick={() => moveRow(i, +1)} className="text-gray-400 hover:text-indigo-600 px-0.5" title="Move down">▼</button>
                  <button onClick={() => delRow(i)} className="text-gray-400 hover:text-red-500 px-0.5" title="Delete">✕</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <button onClick={addRow}
        className="w-full mt-2 text-xs px-2 py-1 rounded border border-dashed border-gray-300 text-gray-500 hover:border-indigo-300 hover:text-indigo-600 hover:bg-indigo-50 transition-colors">
        + Add row
      </button>
    </div>
  );

  return (
    <ThreeCol
      storageKey="ForestBuilderPanel"
      left={leftCol}
      middle={middleCol}
      right={rightCol}
    />
  );
}
