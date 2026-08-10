/* eslint-disable react-refresh/only-export-components -- label helper co-located with the plot component */
import { useState, useRef } from "react";
import { usePlotLayout } from "../../plotStyle";
import Plot from "../../PlotComponent";
import { useStore } from "../../store";
import PlotExporter from "../PlotExporter";
import { fmtP } from "../../lib/format";
import type { PlotCaptureHandle, PlotData } from "../../lib/plotTypes";
import type { Coefficient, ORRow, ForestResult } from "./shared";

// Geometry and axis styling only; the background, font and colourway are the
// global chart theme's to decide and are merged in at render.
const FOREST_BASE = {
  xaxis: {
    type: "log" as const,
    gridcolor: "#e5e7eb",
    zeroline: false,
    tickfont: { size: 10 },
  },
  yaxis: { gridcolor: "transparent", zeroline: false, tickfont: { size: 11 } },
  shapes: [{
    type: "line" as const,
    x0: 1, x1: 1,
    xref: "x" as const, yref: "paper" as const,
    y0: 0, y1: 1,
    line: { color: "#ef4444", dash: "dot" as const, width: 1.5 },
  }],
};

/**
 * Auto-label a variable name with a clinical suffix:
 * - "Gender_Male"   → "Gender_Male  [Male vs. ref]"
 * - "Platelet (per 10000 units)" → keep as-is
 * - "Age"           → "Age  [per 1 unit ↑]"
 */
export function varLabel(v: string): string {
  // Already has a unit hint → leave it alone
  if (v.includes("per ") || v.includes(" vs.") || v.includes("[")) return v;
  // Dummy from pd.get_dummies: "BaseName_Level"
  const dummyMatch = v.match(/^(.+)_([^_]+)$/);
  if (dummyMatch) {
    return `${v}  [${dummyMatch[2]} vs. ref]`;
  }
  // Binary-looking name (contains common medical binary keywords)
  const binaryKeywords = /^(DM|HT|AF|STEMI|NSTEMI|UAP|smoking|malign|redo|beating|Hypert|Heart|Chronic|periferik|occluded|Lima|LIMA)/i;
  if (binaryKeywords.test(v)) {
    return `${v}  [Yes vs. No]`;
  }
  return `${v}  [per 1 unit ↑]`;
}

/**
 * Marker size proportional to statistical precision (1/log-CI-width).
 * Narrow CI → larger square; wide CI → smaller square.
 */
function precisionSize(est: number, lo: number, hi: number): number {
  if (!est || !lo || !hi || lo <= 0 || hi <= lo) return 9;
  const logW = Math.log(hi) - Math.log(lo);
  if (logW <= 0) return 16;
  // exp decay: very narrow CI (logW~0.1) → ~15, wide CI (logW~3) → ~7
  const sz = 6 + 10 * Math.exp(-logW * 0.9);
  return Math.min(16, Math.max(6, sz));
}

type ForestLayout = "overlay" | "split";
type ForestColorMode = "series" | "significance";

export function ForestPlot({ result, modelType, outcome }: {
  result: ForestResult;
  modelType: string;
  outcome?: string;
}) {
  const forestRef = useRef<PlotCaptureHandle | null>(null);
  const themed = usePlotLayout();
  const isORTable = modelType === "ortable" || modelType === "firth_ortable";
  const isCox     = modelType === "cox";
  const metric    = isCox ? "HR" : "OR";
  const showGrid  = useStore((s) => s.showGrid);

  // ── User-tunable layout options ───────────────────────────────────────────
  const [opts, setOpts] = useState<{
    layout: ForestLayout;
    colorBy: ForestColorMode;
    showValueColumns: boolean;
    customTitle: string;
    customSubtitle: string;
    customXLabel: string;
    markerStyle: "square" | "circle" | "diamond";
    height: number;
    sigColor: string;
    nonSigColor: string;
    showLegend: boolean;
    showArrows: boolean;
  }>({
    layout: "overlay",
    colorBy: "series",
    showValueColumns: true,
    customTitle: "",
    customSubtitle: "",
    customXLabel: "",
    markerStyle: "square",
    height: 0,                  // 0 = auto from row count
    sigColor: "#dc2626",
    nonSigColor: "#475569",
    showLegend: true,
    showArrows: true,
  });
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Significance color helper honours user-supplied palette.
  const sigColor = (
    est: number | null | undefined,
    lo: number | null | undefined,
    hi: number | null | undefined,
  ): string => {
    if (est == null || lo == null || hi == null) return "#9ca3af";
    const includesOne = lo <= 1 && hi >= 1;
    return includesOne ? opts.nonSigColor : opts.sigColor;
  };

  // Inline toolbar with the full option set. Advanced controls live behind
  // a disclosure to keep the bar compact.
  const Toolbar = (
    <div className="mb-2 text-[10px] text-gray-600 space-y-1.5">
      <div className="flex flex-wrap items-center gap-2">
        {isORTable && (
          <div className="inline-flex rounded-md border border-gray-300 overflow-hidden">
            {(["overlay", "split"] as const).map((v) => (
              <button key={v}
                onClick={() => setOpts((o) => ({ ...o, layout: v }))}
                className={`px-2 py-0.5 text-[10px] ${opts.layout === v ? "bg-indigo-600 text-white" : "bg-white hover:bg-gray-50 text-gray-600"}`}>
                {v === "overlay" ? "Overlay (one panel)" : "Split (two panels)"}
              </button>
            ))}
          </div>
        )}
        <div className="inline-flex rounded-md border border-gray-300 overflow-hidden">
          {(["series", "significance"] as const).map((v) => (
            <button key={v}
              onClick={() => setOpts((o) => ({ ...o, colorBy: v }))}
              className={`px-2 py-0.5 text-[10px] ${opts.colorBy === v ? "bg-indigo-600 text-white" : "bg-white hover:bg-gray-50 text-gray-600"}`}>
              {v === "series" ? "Color by series" : "Color by significance"}
            </button>
          ))}
        </div>
        <div className="inline-flex rounded-md border border-gray-300 overflow-hidden">
          {(["square", "circle", "diamond"] as const).map((v) => (
            <button key={v}
              onClick={() => setOpts((o) => ({ ...o, markerStyle: v }))}
              className={`px-2 py-0.5 text-[10px] ${opts.markerStyle === v ? "bg-indigo-600 text-white" : "bg-white hover:bg-gray-50 text-gray-600"}`}>
              ▪ {v}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-1 cursor-pointer">
          <input type="checkbox" checked={opts.showValueColumns}
            onChange={(e) => setOpts((o) => ({ ...o, showValueColumns: e.target.checked }))}
            className="accent-indigo-500" />
          Value columns
        </label>
        <label className="flex items-center gap-1 cursor-pointer">
          <input type="checkbox" checked={opts.showLegend}
            onChange={(e) => setOpts((o) => ({ ...o, showLegend: e.target.checked }))}
            className="accent-indigo-500" />
          Legend
        </label>
        <label className="flex items-center gap-1 cursor-pointer">
          <input type="checkbox" checked={opts.showArrows}
            onChange={(e) => setOpts((o) => ({ ...o, showArrows: e.target.checked }))}
            className="accent-indigo-500" />
          Direction arrows
        </label>
        <button
          onClick={() => setAdvancedOpen((v) => !v)}
          className="ml-auto text-[10px] px-2 py-0.5 rounded border border-gray-300 text-gray-600 hover:bg-gray-50">
          {advancedOpen ? "▾ Hide advanced" : "▸ Show advanced"}
        </button>
      </div>
      {advancedOpen && (
        <div className="flex flex-wrap items-center gap-2 p-2 rounded border border-gray-200 bg-gray-50">
          <input type="text"
            placeholder="Custom title…"
            value={opts.customTitle}
            onChange={(e) => setOpts((o) => ({ ...o, customTitle: e.target.value }))}
            className="text-[10px] border border-gray-300 rounded px-2 py-0.5 w-60 focus:outline-none focus:border-indigo-400"
          />
          <input type="text"
            placeholder="Subtitle…"
            value={opts.customSubtitle}
            onChange={(e) => setOpts((o) => ({ ...o, customSubtitle: e.target.value }))}
            className="text-[10px] border border-gray-300 rounded px-2 py-0.5 w-60 focus:outline-none focus:border-indigo-400"
          />
          <input type="text"
            placeholder={`X-axis: ${metric} (95% CI), log scale`}
            value={opts.customXLabel}
            onChange={(e) => setOpts((o) => ({ ...o, customXLabel: e.target.value }))}
            className="text-[10px] border border-gray-300 rounded px-2 py-0.5 w-52 focus:outline-none focus:border-indigo-400"
          />
          <label className="flex items-center gap-1 text-[10px]">
            Height
            <input type="number" min={0} step={20}
              value={opts.height || ""}
              placeholder="auto"
              onChange={(e) => setOpts((o) => ({ ...o, height: parseInt(e.target.value) || 0 }))}
              className="w-14 text-[10px] border border-gray-300 rounded px-1 py-0.5 text-right focus:outline-none focus:border-indigo-400"
            />
          </label>
          {opts.colorBy === "significance" && (
            <>
              <label className="flex items-center gap-1 text-[10px]">
                CI excludes 1
                <input type="color" value={opts.sigColor}
                  onChange={(e) => setOpts((o) => ({ ...o, sigColor: e.target.value }))}
                  className="w-6 h-5 cursor-pointer border border-gray-300 rounded" />
              </label>
              <label className="flex items-center gap-1 text-[10px]">
                Includes 1
                <input type="color" value={opts.nonSigColor}
                  onChange={(e) => setOpts((o) => ({ ...o, nonSigColor: e.target.value }))}
                  className="w-6 h-5 cursor-pointer border border-gray-300 rounded" />
              </label>
            </>
          )}
        </div>
      )}
    </div>
  );

  // ── Shared helpers ────────────────────────────────────────────────────────
  const fmtCI = (
    est: number | null | undefined,
    lo: number | null | undefined,
    hi: number | null | undefined,
  ) =>
    est == null ? "—" : `${est.toFixed(2)} (${lo?.toFixed(2)}–${hi?.toFixed(2)})`;

  // Base props for layout annotations
  const AB = { showarrow: false, xanchor: "left" as const, yanchor: "middle" as const };
  const HDR = { size: 9, color: "#374151" };

  // Directional arrow labels below x-axis (within forest domain).
  // forestRight = right edge of forest domain in paper coords.
  // Pushed below the x-axis tick numbers (which sit at roughly y = -0.05)
  // so "◀ Reduces risk" and "Increases risk ▶" no longer overlap with
  // 0.4, 0.6, … 2.4 tick labels.
  const dirAnnotations = (forestRight: number, yPos = -0.20) => [
    {
      ...AB, xref: "paper" as const, yref: "paper" as const,
      x: 0.02, y: yPos, xanchor: "left" as const,
      text: `◀ ${isCox ? "Reduces hazard" : "Reduces risk"}`,
      font: { size: 9, color: "#10b981" }, showarrow: false,
    },
    {
      ...AB, xref: "paper" as const, yref: "paper" as const,
      x: forestRight - 0.01, y: yPos, xanchor: "right" as const,
      text: `${isCox ? "Increases hazard" : "Increases risk"} ▶`,
      font: { size: 9, color: "#ef4444" }, showarrow: false,
    },
  ];

  // ── OR Table (dual trace) ─────────────────────────────────────────────────
  if (isORTable) {
    const rows       = result.table ?? [];
    const n          = rows.length;
    const yIdx       = Object.fromEntries(rows.map((r, i) => [r.variable, i]));
    const uniValid   = rows.filter((r): r is ORRow & { uni_or: number } => r.uni_or != null && r.uni_or > 0);
    const multiValid = rows.filter((r): r is ORRow & { multi_or: number } => r.multi_or != null && r.multi_or > 0);
    const plotH      = Math.max(320, n * 58 + 120);
    const splitLayout = opts.layout === "split";

    // Color resolvers honour the colorBy toggle:
    //   • "series"        — uni=slate / multi=emerald (existing palette)
    //   • "significance"  — gray when CI includes 1, red when it doesn't
    const uniColor = (r: ORRow) =>
      opts.colorBy === "significance"
        ? sigColor(r.uni_or, r.uni_ci_low, r.uni_ci_high)
        : (r.uni_p != null && r.uni_p < 0.05 ? "#6366f1" : "#6b7280");
    const multiColor = (r: ORRow) =>
      opts.colorBy === "significance"
        ? sigColor(r.multi_or, r.multi_ci_low, r.multi_ci_high)
        : (r.multi_p != null && r.multi_p < 0.05 ? "#10b981" : "#6b7280");

    // Forest x-domain depends on layout: overlay puts text columns to the
    // right of a single forest; split spreads two forests side-by-side and
    // skips the text columns by default (they can still be re-enabled).
    const forestRight = splitLayout
      ? 1.0           // entire width filled by two side-by-side forests
      : (opts.showValueColumns ? 0.47 : 0.95);
    const TX1 = splitLayout ? null : 0.49;
    const TX2 = splitLayout ? null : 0.76;
    const showCols = !splitLayout && opts.showValueColumns;

    const annotations: object[] = [
      ...(showCols && TX1 && TX2
        ? [
            { ...AB, xref: "paper", yref: "paper", x: TX1, y: 1.055,
              text: "<b>OR (95% CI)</b>", font: HDR },
            { ...AB, xref: "paper", yref: "paper", x: TX2, y: 1.055,
              text: "<b><i>p</i></b>", font: HDR },
            { ...AB, xref: "paper", yref: "paper", x: TX1, y: 1.012,
              text: opts.colorBy === "significance"
                ? "● Uni   ◆ Multi (red = CI excl. 1)"
                : "● Uni   ◆ Multi", font: { size: 8, color: "#4b5563" } },
          ]
        : []),
      ...(splitLayout
        ? [
            { ...AB, xref: "paper" as const, yref: "paper" as const, x: 0.215, y: 1.045,
              xanchor: "center" as const,
              text: "<b>Unadjusted</b>", font: HDR },
            { ...AB, xref: "paper" as const, yref: "paper" as const, x: 0.755, y: 1.045,
              xanchor: "center" as const,
              text: "<b>Adjusted (mutually adjusted model)</b>", font: HDR },
          ]
        : []),
      ...(splitLayout || !opts.showArrows ? [] : dirAnnotations(forestRight)),
    ];

    if (showCols && TX1 && TX2) {
      rows.forEach((r, i) => {
        if (r.uni_or != null) {
          const col = uniColor(r);
          annotations.push(
            { ...AB, xref: "paper", yref: "y", x: TX1, y: i + 0.18,
              text: fmtCI(r.uni_or, r.uni_ci_low, r.uni_ci_high), font: { size: 9, color: col } },
            { ...AB, xref: "paper", yref: "y", x: TX2, y: i + 0.18,
              text: fmtP(r.uni_p), font: { size: 9, color: col } },
          );
        }
        if (r.multi_or != null) {
          const col = multiColor(r);
          annotations.push(
            { ...AB, xref: "paper", yref: "y", x: TX1, y: i - 0.18,
              text: fmtCI(r.multi_or, r.multi_ci_low, r.multi_ci_high), font: { size: 9, color: col } },
            { ...AB, xref: "paper", yref: "y", x: TX2, y: i - 0.18,
              text: fmtP(r.multi_p), font: { size: 9, color: col } },
          );
        }
      });
    }

    // Trace builders shared by both layouts
    const uniMarker = opts.markerStyle === "square" ? "square" : "circle";
    const multiMarker = opts.markerStyle === "square" ? "square" : "diamond";
    const uniTrace: PlotData = {
      name: splitLayout ? "Unadjusted" : "Univariate",
      type: "scatter", mode: "markers",
      x: uniValid.map((r) => r.uni_or),
      y: uniValid.map((r) => splitLayout ? yIdx[r.variable] : yIdx[r.variable] + 0.18),
      error_x: {
        type: "data", symmetric: false,
        array:      uniValid.map((r) => (r.uni_ci_high ?? r.uni_or) - r.uni_or),
        arrayminus: uniValid.map((r) => r.uni_or - (r.uni_ci_low ?? r.uni_or)),
        color: opts.colorBy === "series" ? "#6366f1" : "#9ca3af",
        thickness: 2, width: 7,
      },
      marker: {
        size: uniValid.map((r) => precisionSize(r.uni_or, r.uni_ci_low ?? 0, r.uni_ci_high ?? 0)),
        symbol: uniMarker,
        color: uniValid.map((r) => uniColor(r)),
        line: { color: "#d1d5db", width: 1 },
      },
      hovertemplate: uniValid.map((r) =>
        `<b>${r.variable}</b> (Unadjusted)<br>OR: ${r.uni_or?.toFixed(3)}<br>95% CI: ${r.uni_ci_low?.toFixed(3)} – ${r.uni_ci_high?.toFixed(3)}<br><i>p</i> = ${fmtP(r.uni_p)}<extra></extra>`
      ),
      ...(splitLayout ? { xaxis: "x", yaxis: "y" } : {}),
    };
    const multiTrace: PlotData = {
      name: splitLayout ? "Adjusted" : "Multivariate",
      type: "scatter", mode: "markers",
      x: multiValid.map((r) => r.multi_or),
      y: multiValid.map((r) => splitLayout ? yIdx[r.variable] : yIdx[r.variable] - 0.18),
      error_x: {
        type: "data", symmetric: false,
        array:      multiValid.map((r) => (r.multi_ci_high ?? r.multi_or) - r.multi_or),
        arrayminus: multiValid.map((r) => r.multi_or - (r.multi_ci_low ?? r.multi_or)),
        color: opts.colorBy === "series" ? "#10b981" : "#9ca3af",
        thickness: 2, width: 7,
      },
      marker: {
        size: multiValid.map((r) => precisionSize(r.multi_or, r.multi_ci_low ?? 0, r.multi_ci_high ?? 0)),
        symbol: multiMarker,
        color: multiValid.map((r) => multiColor(r)),
        line: { color: "#d1d5db", width: 1 },
      },
      hovertemplate: multiValid.map((r) =>
        `<b>${r.variable}</b> (Adjusted)<br>OR: ${r.multi_or?.toFixed(3)}<br>95% CI: ${r.multi_ci_low?.toFixed(3)} – ${r.multi_ci_high?.toFixed(3)}<br><i>p</i> = ${fmtP(r.multi_p)}<extra></extra>`
      ),
      ...(splitLayout ? { xaxis: "x2", yaxis: "y" } : {}),
    };

    // Split layout uses two x-axes (xaxis [0, 0.43] / xaxis2 [0.50, 0.93])
    // anchored to the same y-axis, both log-scaled with the same null line.
    const splitShapes = splitLayout
      ? [
          { type: "line" as const, x0: 1, x1: 1, xref: "x" as const, yref: "paper" as const, y0: 0, y1: 1,
            line: { color: "#9ca3af", dash: "dash" as const, width: 1.2 } },
          { type: "line" as const, x0: 1, x1: 1, xref: "x2" as const, yref: "paper" as const, y0: 0, y1: 1,
            line: { color: "#9ca3af", dash: "dash" as const, width: 1.2 } },
        ]
      : [
          ...FOREST_BASE.shapes,
          ...(showCols
            ? [{ type: "line" as const, xref: "paper" as const, yref: "paper" as const,
                x0: 0.48, x1: 0.48, y0: 0, y1: 1,
                line: { color: "#e5e7eb", width: 1 } }]
            : []),
        ];

    const effHeight = opts.height || plotH;
    const titleHtml = opts.customTitle && opts.customSubtitle
      ? `${opts.customTitle}<br><span style="font-size:11px;color:#6b7280">${opts.customSubtitle}</span>`
      : opts.customTitle;
    const xLabel = opts.customXLabel
      || `${metric === "OR" ? "Odds Ratio" : "Hazard Ratio"} (95% CI)${splitLayout ? "" : (outcome ? ` — Outcome: ${outcome}` : "")}, log scale`;
    // Push legend BELOW the x-axis title so it never sits inside the data
    // area. Bottom margin grows when the legend is on to make room. Floor
    // raised so the "◀ Reduces risk / Increases risk ▶" annotations at
    // y=-0.20 land below the x-axis tick numbers rather than on top of
    // them.
    const bottomPad = opts.showLegend ? 130 : 80;

    return (
      <div className="relative" ref={(el) => { forestRef.current = el as unknown as PlotCaptureHandle | null; }}>
      {Toolbar}
      <PlotExporter plotRef={forestRef} title={`Forest_${metric}_${outcome ?? "model"}`} />
      <Plot
        data={[uniTrace, multiTrace]}
        layout={{
          ...themed,
          ...themed,
        ...FOREST_BASE,
          height: effHeight,
          autosize: true,
          margin: { t: opts.customTitle ? (opts.customSubtitle ? 70 : 50) : 30, r: 20, b: bottomPad, l: 180 },
          title: opts.customTitle
            ? { text: titleHtml, font: { size: 12, color: "#1f2937" }, x: 0.5, xanchor: "center" as const }
            : undefined,
          xaxis: {
            ...FOREST_BASE.xaxis,
            showgrid: showGrid,
            domain: splitLayout ? [0, 0.43] : [0, forestRight],
            title: { text: xLabel, font: { size: 10, color: "#374151" } },
          },
          ...(splitLayout
            ? {
                xaxis2: {
                  ...FOREST_BASE.xaxis,
                  showgrid: showGrid,
                  domain: [0.50, 0.93],
                  anchor: "y" as const,
                  title: { text: opts.customXLabel || `${metric === "OR" ? "Odds Ratio" : "Hazard Ratio"} (95% CI), log scale${outcome ? ` — Outcome: ${outcome}` : ""}`, font: { size: 10, color: "#374151" } },
                },
              }
            : {}),
          yaxis: {
            ...FOREST_BASE.yaxis,
            tickvals: rows.map((_, i) => i),
            ticktext: rows.map((r) => varLabel(r.variable)),
            autorange: "reversed" as const,
            range: [-0.5, n - 0.5],
          },
          shapes: splitShapes,
          annotations,
          showlegend: opts.showLegend,
          // Legend sits below the x-axis title (yref=paper, y<0) so it never
          // overlaps short forests with few rows.
          legend: {
            font: { color: "#374151", size: 11 },
            bgcolor: "rgba(255,255,255,0.95)",
            bordercolor: "#e5e7eb", borderwidth: 1,
            orientation: "h" as const,
            x: 0.5, xanchor: "center" as const,
            y: -0.32, yanchor: "top" as const,
          },
        }}
        style={{ width: "100%", height: effHeight }}
        useResizeHandler
        config={{ responsive: true, displaylogo: false, displayModeBar: false }}
      />
      </div>
    );
  }

  // ── Single model — logistic or cox ────────────────────────────────────────
  const coefs    = (result.coefficients ?? []).filter((c: Coefficient) => c.variable !== "const");
  const n        = coefs.length;
  if (n === 0) return null;

  const estimates = coefs.map((c: Coefficient) => isCox ? c.hr         : c.odds_ratio);
  const ciLow     = coefs.map((c: Coefficient) => isCox ? c.hr_ci_low  : c.or_ci_low);
  const ciHigh    = coefs.map((c: Coefficient) => isCox ? c.hr_ci_high : c.or_ci_high);
  const pVals     = coefs.map((c: Coefficient) => c.p);
  const labels    = coefs.map((c: Coefficient) => c.variable);
  const estimatesForPlot = estimates.map((e) => e ?? null);
  const COLOR     = isCox ? "#10b981" : "#6366f1";
  const COLOR_SIG = isCox ? "#34d399" : "#818cf8";
  const plotH     = Math.max(260, n * 46 + 120);

  // Color resolver honours the colorBy toggle for the single-model branch.
  const rowColor = (i: number) =>
    opts.colorBy === "significance"
      ? sigColor(estimates[i], ciLow[i], ciHigh[i])
      : (pVals[i] < 0.05 ? COLOR_SIG : "#6b7280");

  // xaxis.domain depends on whether the value columns are visible.
  const forestRight = opts.showValueColumns ? 0.55 : 0.95;
  const TX1 = 0.57;
  const TX2 = 0.80;

  const annotations: object[] = [
    ...(opts.showValueColumns
      ? [
          { ...AB, xref: "paper", yref: "paper", x: TX1, y: 1.06,
            text: `<b>${metric} (95% CI)</b>`, font: HDR },
          { ...AB, xref: "paper", yref: "paper", x: TX2, y: 1.06,
            text: "<b><i>p</i></b>", font: HDR },
        ]
      : []),
    ...(opts.showArrows ? dirAnnotations(forestRight) : []),
    // Per-variable rows (only when value columns are shown)
    ...(opts.showValueColumns
      ? coefs.map((_: Coefficient, i: number) => {
          const col = rowColor(i);
          return [
            { ...AB, xref: "paper", yref: "y", x: TX1, y: i,
              text: fmtCI(estimates[i], ciLow[i], ciHigh[i]), font: { size: 9, color: col } },
            { ...AB, xref: "paper", yref: "y", x: TX2, y: i,
              text: fmtP(pVals[i]), font: { size: 9, color: col } },
          ];
        }).flat()
      : []),
  ];

  const effHeight = opts.height || plotH;
  const titleHtml = opts.customTitle && opts.customSubtitle
    ? `${opts.customTitle}<br><span style="font-size:11px;color:#6b7280">${opts.customSubtitle}</span>`
    : opts.customTitle;
  const xLabelSingle = opts.customXLabel || (isCox
    ? `Hazard Ratio (95% CI), log scale${outcome ? ` — Outcome: ${outcome}` : ""}`
    : `Odds Ratio (95% CI), log scale${outcome ? ` — Outcome: ${outcome}` : ""}`);

  return (
    <div className="relative" ref={(el) => { forestRef.current = el as unknown as PlotCaptureHandle | null; }}>
    {Toolbar}
    <PlotExporter plotRef={forestRef} title={`Forest_${metric}_${outcome ?? "model"}`} />
    <Plot
      data={[{
        type: "scatter", mode: "markers",
        x: estimatesForPlot,
        y: coefs.map((_: Coefficient, i: number) => i),
        error_x: {
          type: "data", symmetric: false,
          array:      estimates.map((e, i: number) => e == null ? 0 : (ciHigh[i] ?? e) - e),
          arrayminus: estimates.map((e, i: number) => e == null ? 0 : e - (ciLow[i]  ?? e)),
          color: opts.colorBy === "series" ? COLOR : "#9ca3af",
          thickness: 2.5, width: 9,
        },
        marker: {
          size: estimates.map((_, i: number) => precisionSize(estimates[i] ?? 0, ciLow[i] ?? 0, ciHigh[i] ?? 0)),
          symbol: opts.markerStyle,
          color: coefs.map((_: Coefficient, i: number) => rowColor(i)),
          line: { color: "#d1d5db", width: 1 },
        },
        hovertemplate: coefs.map((_: Coefficient, i: number) =>
          `<b>${labels[i]}</b><br>${metric}: ${estimates[i]?.toFixed(3)}<br>95% CI: ${ciLow[i]?.toFixed(3)} – ${ciHigh[i]?.toFixed(3)}<br><i>p</i> = ${fmtP(pVals[i])}<extra></extra>`
        ),
        name: isCox ? "Hazard Ratio" : "Odds Ratio",
        showlegend: opts.showLegend,
      }]}
      layout={{
        ...FOREST_BASE,
        height: effHeight,
        autosize: true,
        margin: { t: opts.customTitle ? (opts.customSubtitle ? 70 : 50) : 20, r: 20, b: opts.showLegend ? 130 : 90, l: 180 },
        title: opts.customTitle
          ? { text: titleHtml, font: { size: 12, color: "#1f2937" }, x: 0.5, xanchor: "center" as const }
          : undefined,
        xaxis: {
          ...FOREST_BASE.xaxis,
          showgrid: showGrid,
          domain: [0, forestRight],
          title: { text: xLabelSingle, font: { size: 10, color: "#374151" } },
        },
        yaxis: {
          ...FOREST_BASE.yaxis,
          tickvals: coefs.map((_: Coefficient, i: number) => i),
          ticktext: labels.map((l: string) => varLabel(l)),
          autorange: "reversed" as const,
          range: [-0.5, n - 0.5],
        },
        shapes: [
          ...FOREST_BASE.shapes,
          ...(opts.showValueColumns
            ? [{ type: "line" as const, xref: "paper" as const, yref: "paper" as const,
                x0: 0.56, x1: 0.56, y0: 0, y1: 1,
                line: { color: "#e5e7eb", width: 1 } }]
            : []),
        ],
        annotations,
        showlegend: opts.showLegend,
        legend: {
          font: { color: "#374151", size: 11 },
          bgcolor: "rgba(255,255,255,0.95)",
          bordercolor: "#e5e7eb", borderwidth: 1,
          orientation: "h" as const,
          x: 0.5, xanchor: "center" as const,
          y: -0.32, yanchor: "top" as const,
        },
      }}
      style={{ width: "100%", height: effHeight }}
      useResizeHandler
      config={{ responsive: true, displaylogo: false, displayModeBar: false }}
    />
    </div>
  );
}
