import { useEffect, useLayoutEffect, useState, useCallback, useRef, type ReactNode } from "react";
import { Pencil, Trash2 } from "lucide-react";
import {
  useStore,
  paletteOf,
  isNumericKind,
  runColumnStructureMutation,
} from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { usePalette, usePlotLayout } from "../plotStyle";
import api, { deleteColumn, renameColumn } from "../api";
import ResultExporter from "./ResultExporter";
import TitledPlot from "./TitledPlot";
import { fmtP } from "../lib/format";
import { labelFor } from "../lib/valueLabels";
import { categoryColors } from "../lib/categoryColors";
import type { PlotData, PlotCaptureHandle } from "../lib/plotTypes";

// ── Result shapes returned by the descriptive / column-summary endpoints ─────

interface HistBin { bin_start: number; bin_end: number; count: number }
interface QQPoint { x: number; y: number }
interface OutlierPoint { row: number; value: number }
interface CatRow { value: string | number; count: number; pct?: number }
interface NormalityDeviant {
  row: number; value: number; z: number; qq_x: number; abs_residual: number;
}

interface ColumnSummary {
  type?: "numeric" | "categorical";
  histogram: HistBin[];
  raw_values?: number[];
  outliers?: OutlierPoint[];
  normality_deviants?: NormalityDeviant[];
  qq: QQPoint[];
  categories?: CatRow[];
  n?: number;
  n_categories?: number;
  missing?: number;
  display_decimals?: number;
  mean?: number;
  std?: number;
  median?: number;
  min?: number;
  max?: number;
  q1?: number;
  q3?: number;
  iqr?: number;
  whisker_low?: number;
  whisker_high?: number;
  skewness?: number;
  kurtosis?: number;
  normal?: boolean;
  normality_label?: string;
  normality_test?: string;
  normality_p?: number;
  shapiro_p?: number;
}

type ChartTab = "histogram" | "boxplot" | "violin" | "qq";

interface ScatterResult {
  points: Record<string, unknown>[];
  color?: unknown;
  regression: {
    line_x?: number[];
    line_y?: number[];
    r?: number;
    r2?: number;
    p?: number | null;
    slope?: number;
    intercept?: number;
    note?: string;
  };
}

interface ColMeta {
  name: string;
  kind: string;
  hist?: null;
  shapiro_p?: number;
  top2?: null;
}

// ── Inline sparkline SVG (real histogram / category bars) ────────────────────

interface SparkData { type: string; data: number[]; }

function Sparkline({ spark }: { spark: SparkData }) {
  const W = 40, H = 12;
  const { type, data } = spark;
  const pal = usePalette();
  if (!data || data.length === 0) return null;
  const max = Math.max(...data);
  if (max === 0) return null;

  if (type === "numeric") {
    const bw = W / data.length;
    return (
      <svg width={W} height={H} style={{ display: "block", flexShrink: 0 }}>
        {data.map((v, i) => {
          const bh = Math.max(1, (v / max) * H);
          return (
            <rect key={i} x={i * bw} y={H - bh}
              width={Math.max(bw - 0.5, 0.5)} height={bh}
              fill={pal[0]} opacity={0.7} rx={0.5} />
          );
        })}
      </svg>
    );
  }

  // categorical → proportional horizontal bars
  const total = data.reduce((a, b) => a + b, 0);
  const CATS = pal;
  const segments = data.reduce<{ i: number; x: number; bw: number }[]>((acc, v, i) => {
    const bw = (v / total) * W;
    const x = acc.length ? acc[acc.length - 1].x + acc[acc.length - 1].bw : 0;
    return [...acc, { i, x, bw }];
  }, []);
  return (
    <svg width={W} height={H} style={{ display: "block", flexShrink: 0 }}>
      {segments.map(({ i, x, bw }) => (
        <rect key={i} x={x} y={0} width={Math.max(bw - 0.5, 0.5)} height={H}
          fill={CATS[i % CATS.length]} opacity={0.8} />
      ))}
    </svg>
  );
}

// ── Normality Deviants component with right-click context menu ────────────────

interface NormalityDeviant { row: number; value: number; z: number; abs_residual: number; }

function NormalityDeviants({ deviants, onDelete }: { deviants: NormalityDeviant[]; onDelete: () => void }) {
  const [contextMenu, setContextMenu] = useState<{ row: number; x: number; y: number } | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const handleContextMenu = (e: React.MouseEvent, row: number) => {
    e.preventDefault();
    setContextMenu({ row, x: e.clientX, y: e.clientY });
  };

  const handleDeleteRow = async (row: number) => {
    try {
      await useStore.getState().deleteRow(row);
      // Refresh the stats immediately
      onDelete();
    } catch (err) {
      console.error("Error deleting row:", err);
    }
    setContextMenu(null);
  };

  // Close context menu on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setContextMenu(null);
      }
    };
    if (contextMenu) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [contextMenu]);

  return (
    <div className="mt-2 px-3 py-2 bg-orange-50 border border-orange-100 rounded-lg">
      <p className="text-xs font-semibold text-orange-700 mb-2 flex items-center gap-1">
        <span>🔶 Top Normality Deviants (Q-Q Deviation)</span>
        <span className="font-normal text-[10px] text-orange-400">(Shows why the distribution failed normality)</span>
      </p>
      <div className="flex flex-wrap gap-1.5 relative">
        {deviants.map((e) => (
          <div
            key={e.row}
            className="group cursor-pointer relative flex items-center gap-1 text-[10px] font-mono bg-white text-orange-800 border border-orange-200 rounded px-2 py-0.5 shadow-sm hover:border-orange-400 hover:bg-orange-100 transition-all"
            onClick={async () => {
              try {
                await useStore.getState().deleteRow(e.row);
                onDelete();
              } catch { /* row delete is best-effort */ }
            }}
            onContextMenu={(ev) => handleContextMenu(ev, e.row)}
            title="Click to delete or right-click for menu"
          >
            <span className="opacity-0 w-0 overflow-hidden group-hover:w-auto group-hover:opacity-100 transition-all mr-0.5 text-orange-600">🗑</span>
            <span className="text-orange-400 font-bold">#{e.row}</span>
            <span className="w-px h-2.5 bg-orange-100 mx-0.5"></span>
            <span className="font-semibold">{e.value.toFixed(2)}</span>
            <span className="text-[9px] text-orange-400 ml-0.5">z={e.z > 0 ? "+" : ""}{e.z.toFixed(2)}</span>

            {/* Tooltip on hover */}
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block bg-gray-800 text-white text-[9px] px-2 py-1 rounded whitespace-nowrap z-10">
              Row {e.row} | Resid: {e.abs_residual.toFixed(3)}
            </div>
          </div>
        ))}

        {/* Context Menu */}
        {contextMenu && (
          <div
            ref={menuRef}
            className="fixed bg-white border border-gray-200 rounded-lg shadow-lg z-50 py-1 min-w-max"
            style={{ top: `${contextMenu.y}px`, left: `${contextMenu.x}px` }}
          >
            <button
              onClick={() => handleDeleteRow(contextMenu.row)}
              className="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50 flex items-center gap-2 transition-colors"
            >
              <span>🗑</span> Delete Row {contextMenu.row}
            </button>
            <div className="border-t border-gray-100 my-1"></div>
            <button
              onClick={() => setContextMenu(null)}
              className="w-full text-left px-4 py-2 text-sm text-gray-500 hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// Geometry and axis colours only. The background, the font and the colourway
// come from the global chart theme at render — kept here as literals, they
// were three settings the theme picker could not reach.
const BASE_LAYOUT = {
  margin: { t: 24, r: 16, b: 48, l: 56 },
  xaxis: { gridcolor: "#e5e7eb", zerolinecolor: "#d1d5db" },
  yaxis: { gridcolor: "#e5e7eb", zerolinecolor: "#d1d5db" },
};
const SUMMARY_CHART_HEIGHT = 360;

// ── Main chart for numeric columns ──────────────────────────────────────────

function NumericView({ summary, loadSummary, selected }: { summary: ColumnSummary; loadSummary: (col: string) => void; selected: string }) {
  const themedBase = { ...usePlotLayout(), ...BASE_LAYOUT };
  const chartTab = useStore((s) => s.descriptiveTab);
  const showGrid = useStore((s) => s.showGrid);
  const pal = usePalette();
  const histRef = useRef<PlotCaptureHandle | null>(null);
  const boxRef = useRef<PlotCaptureHandle | null>(null);
  const violinRef = useRef<PlotCaptureHandle | null>(null);
  const qqRef = useRef<PlotCaptureHandle | null>(null);
  const P = pal[0]; // primary color

  const histData = [{
    type: "bar" as const,
    x: summary.histogram.map((b) => (b.bin_start + b.bin_end) / 2),
    y: summary.histogram.map((b) => b.count),
    width: summary.histogram.map((b) => b.bin_end - b.bin_start),
    marker: { color: P, opacity: 0.85 },
    name: "Count",
    hovertemplate: "Range: %{customdata[0]}–%{customdata[1]}<br>Count: %{y}<extra></extra>",
    customdata: summary.histogram.map((b) => [b.bin_start.toFixed(2), b.bin_end.toFixed(2)]),
  }];

  const outliers: { row: number; value: number }[] = summary.outliers ?? [];
  const rawVals: number[] = summary.raw_values ?? [];

  const summaryHover =
    `<b>Distribution Summary</b><br>` +
    `Median: ${summary.median?.toFixed(2)}<br>` +
    `Q1: ${summary.q1?.toFixed(2)}  Q3: ${summary.q3?.toFixed(2)}<br>` +
    `Whisker: ${(summary.whisker_low ?? 0).toFixed(2)} – ${(summary.whisker_high ?? 0).toFixed(2)}<br>` +
    `Min: ${summary.min?.toFixed(2)}  Max: ${summary.max?.toFixed(2)}<br>` +
    `Mean ± SD: ${summary.mean?.toFixed(2)} ± ${summary.std?.toFixed(2)}<extra></extra>`;

  // ── Box trace ─────────────────────────────────────────────────────────────
  // Give the box an EXPLICIT x category so Plotly uses a category axis.
  // Then scatter traces with the same x value co-locate perfectly.
  // hoverinfo:"none" kills the ugly per-stat labels Plotly shows by default.
  const boxTrace: PlotData = {
    type: "box" as const,
    x: rawVals.map(() => "Distribution"),   // explicit category → category axis
    y: rawVals,
    name: "Distribution",
    boxmean: true,
    boxpoints: false,                       // we draw outliers ourselves
    marker: { color: P, size: 5 },
    line: { color: P },
    fillcolor: "rgba(99,102,241,0.15)",
    hoverinfo: "none" as const,             // suppress "(Distribution, max: 85)" labels
  };

  // ── Invisible summary hover scatter ──────────────────────────────────────
  // Large transparent marker at median; triggers hover anywhere over the box.
  const summaryScatter: PlotData = {
    type: "scatter" as const,
    mode: "markers" as const,
    x: ["Distribution"],
    y: [summary.median],
    marker: { opacity: 0.001, size: 80, color: "rgba(0,0,0,0)" },
    hovertemplate: summaryHover,
    showlegend: false,
  };

  // ── Outlier scatter ────────────────────────────────────────────────────────
  // Same x category → overlaid perfectly on the box.
  const outlierTrace: PlotData[] = outliers.length > 0 ? [{
    type: "scatter" as const,
    mode: "markers" as const,
    x: outliers.map(() => "Distribution"),
    y: outliers.map((o) => o.value),
    customdata: outliers.map((o) => [o.row, o.value.toFixed(4)]),
    hovertemplate: "<b>Outlier</b><br>Row: %{customdata[0]}<br>Value: %{customdata[1]}<extra></extra>",
    marker: { color: "#ef4444", size: 8, symbol: "circle-open", line: { width: 2, color: "#ef4444" } },
    name: "Outlier",
    showlegend: false,
  }] : [];

  const boxData: PlotData[] = [boxTrace, summaryScatter, ...outlierTrace];


  const normalityDeviants: NormalityDeviant[] =
    summary.normality_deviants ?? [];

  const qqData = [
    {
      type: "scatter" as const, mode: "markers" as const,
      x: summary.qq.map((p) => p.x),
      y: summary.qq.map((p) => p.y),
      marker: { color: P, size: 4 },
      name: "Observed",
      hovertemplate: "Theoretical: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>",
    },
    (() => {
      const xs = summary.qq.map((p) => p.x);
      const ys = summary.qq.map((p) => p.y);
      const xMin = Math.min(...xs), xMax = Math.max(...xs);
      const yMin = Math.min(...ys), yMax = Math.max(...ys);
      return {
        type: "scatter" as const, mode: "lines" as const,
        x: [xMin, xMax], y: [yMin, yMax],
        line: { color: "#9ca3af", width: 1, dash: "dash" as const },
        name: "Reference",
        hoverinfo: "skip" as const,
      };
    })(),
    // Normality deviants overlay (The ones that trigger the "Non-normal" warning)
    ...(normalityDeviants.length > 0 ? [{
      type: "scatter" as const,
      mode: "markers" as const,
      x: normalityDeviants.map((e) => e.qq_x),
      y: normalityDeviants.map((e) => e.value),
      customdata: normalityDeviants.map((e) => [e.row, e.value.toFixed(4), e.z.toFixed(3)]),
      hovertemplate:
        "<b>Disrupts normality</b><br>" +
        "Row: %{customdata[0]}<br>" +
        "Value: %{customdata[1]}<br>" +
        "Z-score: %{customdata[2]}<extra></extra>",
      marker: { 
        color: "#f97316", 
        size: 8, 
        symbol: "diamond", 
        line: { width: 1, color: "#ea580c" },
        opacity: 0.8
      },
      name: "Deviant",
      showlegend: false,
    }] : []),
  ];

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Chart type tabs removed — now controlled exclusively by the 5 top-level sub-tabs under Descriptive */}

      {/* Histogram */}
      {chartTab === "histogram" && (
        <div className="relative">
        <TitledPlot plotRefOut={histRef} storageKey="desc:hist"
          data={histData}
          layout={{ ...themedBase, autosize: true, height: SUMMARY_CHART_HEIGHT, bargap: 0.02,
            xaxis: { ...BASE_LAYOUT.xaxis, showgrid: showGrid, title: { text: "Value" } },
            yaxis: { ...BASE_LAYOUT.yaxis, showgrid: showGrid, title: { text: "Count" } },
          }}
          config={{ responsive: true, displaylogo: false, displayModeBar: false }}
          defaultTitle=""
          defaultSubtitle=""
          defaultXAxis="Value"
          defaultYAxis="Count" />
        </div>
      )}

      {/* Box Plot */}
      {chartTab === "boxplot" && (
        <div className="relative">
        <TitledPlot plotRefOut={boxRef} storageKey="desc:boxplot"
          data={boxData}
          layout={{
            ...themedBase,
            autosize: true,
            height: SUMMARY_CHART_HEIGHT,
            yaxis: { ...BASE_LAYOUT.yaxis, showgrid: showGrid, title: { text: "Value" } },
            xaxis: { ...BASE_LAYOUT.xaxis, showticklabels: false, zeroline: false, showgrid: false },
            showlegend: false,
            annotations: [
              {
                x: 0.5, y: 1.0,
                xref: "paper" as const, yref: "paper" as const,
                text: `IQR = ${summary.iqr?.toFixed(2)}  ·  Skew = ${summary.skewness?.toFixed(3)}` +
                      (outliers.length > 0 ? `  ·  <b style="color:#ef4444">${outliers.length} outlier</b>` : ""),
                showarrow: false,
                font: { color: "#6b7280", size: 11 },
                xanchor: "center" as const,
                yanchor: "bottom" as const,
              },
            ],
          }}
          config={{ responsive: true, displaylogo: false, displayModeBar: false }}
          defaultTitle=""
          defaultSubtitle=""
          defaultXAxis=""
          defaultYAxis="Value" />
        {outliers.length > 0 && (
          <div className="mt-2 px-3 py-2 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-xs font-semibold text-red-600 mb-1">
              ⚠️ {outliers.length} outlier (1.5 × IQR rule)
            </p>
            <div className="flex flex-wrap gap-1">
              {outliers.slice(0, 50).map((o) => (
                <span
                  key={o.row}
                  className="inline-block text-[10px] font-mono bg-red-100 text-red-700 border border-red-200 rounded px-1.5 py-0.5 cursor-pointer hover:bg-red-200"
                  title={`Click to delete Row ${o.row}: ${o.value}`}
                  onClick={async () => {
                     try {
                        await useStore.getState().deleteRow(o.row);
                        import("../api").then((api) => api.refreshSession(useStore.getState().session!.session_id));
                     } catch { /* outlier delete is best-effort */ }
                  }}
                >
                  <span className="opacity-50 hover:opacity-100 mr-1">🗑</span>
                  #{o.row} · {o.value.toFixed(2)}
                </span>
              ))}
              {outliers.length > 50 && (
                <span className="text-[10px] text-red-400 italic">…and {outliers.length - 50} more</span>
              )}
            </div>
          </div>
        )}
        </div>
      )}

      {/* Violin Plot */}
      {chartTab === "violin" && (
        <div className="relative">
        <TitledPlot plotRefOut={violinRef} storageKey="desc:violin"
          data={[{
            type: "violin",
            y: summary.raw_values ?? [],
            name: "Distribution",
            box: { visible: true },
            meanline: { visible: true },
            line: { color: P },
            fillcolor: P + "25",
            points: (summary.raw_values?.length ?? 0) < 200 ? "all" : false,
            jitter: 0.3,
            pointpos: -1.5,
            marker: { color: P, size: 3, opacity: 0.5 },
            hovertemplate:
              `Median: ${summary.median?.toFixed(2)}<br>` +
              `Mean: ${summary.mean?.toFixed(2)}<br>` +
              `SD: ${summary.std?.toFixed(2)}<br>` +
              `IQR: ${summary.q1?.toFixed(2)}–${summary.q3?.toFixed(2)}<extra></extra>`,
          }]}
          layout={{
            ...themedBase,
            autosize: true,
            height: SUMMARY_CHART_HEIGHT,
            yaxis: { ...BASE_LAYOUT.yaxis, showgrid: showGrid, title: { text: "Value" } },
            xaxis: { ...BASE_LAYOUT.xaxis, showticklabels: false, zeroline: false, showgrid: false },
            showlegend: false,
            annotations: [
              {
                x: 0.5, y: 1.0,
                xref: "paper" as const, yref: "paper" as const,
                text: `IQR = ${summary.iqr?.toFixed(2)}  ·  Skew = ${summary.skewness?.toFixed(3)}`,
                showarrow: false,
                font: { color: "#6b7280", size: 11 },
                xanchor: "center" as const,
                yanchor: "bottom" as const,
              },
            ],
          }}
          config={{ responsive: true, displaylogo: false, displayModeBar: false }}
          defaultTitle=""
          defaultSubtitle=""
          defaultXAxis=""
          defaultYAxis="Value" />
        </div>
      )}

      {/* Q-Q Plot */}
      {chartTab === "qq" && (
        <div className="relative">
        <TitledPlot plotRefOut={qqRef} storageKey="desc:qq"
          data={qqData}
          layout={{ ...themedBase, autosize: true, height: SUMMARY_CHART_HEIGHT,
            xaxis: { ...BASE_LAYOUT.xaxis, showgrid: showGrid, title: { text: "Theoretical quantiles" } },
            yaxis: { ...BASE_LAYOUT.yaxis, showgrid: showGrid, title: { text: "Sample quantiles" } },
          }}
          config={{ responsive: true, displaylogo: false, displayModeBar: false }}
          defaultTitle="Q-Q Plot (Normality)"
          defaultSubtitle=""
          defaultXAxis="Theoretical quantiles"
          defaultYAxis="Sample quantiles" />
        {/* List of most deviating values (The ones responsible for Non-normal label) */}
        {!summary.normal && normalityDeviants.length > 0 && (
          <NormalityDeviants deviants={normalityDeviants} onDelete={() => loadSummary(selected)} />
        )}
        </div>
      )}
    </div>
  );
}

// ── Main chart for categorical columns ──────────────────────────────────────

function CategoricalView({ summary }: { summary: ColumnSummary }) {
  const themedBase = { ...usePlotLayout(), ...BASE_LAYOUT };
  const showGrid = useStore((s) => s.showGrid);
  const donutRef = useRef<PlotCaptureHandle | null>(null);
  const barRef = useRef<PlotCaptureHandle | null>(null);
  const cats: CatRow[] = (summary.categories ?? []).slice(0, 20);
  // Six hard-coded colours could not cover eleven histology levels: the
  // biggest slice and one of the smallest both came out purple.
  const colors = categoryColors(paletteOf(useStore.getState().plotTheme), cats.length);

  const donutData = [{
    type: "pie" as const,
    values: cats.map((c) => c.count),
    labels: cats.map((c) => c.value),
    hole: 0.5,
    marker: { colors: colors },
    // Count AND percent on the slice. A percentage on its own cannot be
    // reported: "22.6%" of an unstated denominator is not a result, and the
    // convention this panel states in its own summary line is n (%).
    textinfo: "value+percent" as const,
    // Plotly stacks the leader lines of adjacent thin slices downward. With a
    // 10px bottom margin the last four labels of an eleven-level column were
    // drawn 108px below the plot and clipped away entirely. automargin shrinks
    // the pie until its outside labels fit instead of letting them fall off.
    automargin: true,
    hovertemplate: "%{label}: %{value} (%{percent})<extra></extra>",
  }];

  const barData = [{
    type: "bar" as const,
    x: cats.map((c) => c.count),
    y: cats.map((c) => c.value),
    orientation: "h" as const,
    marker: { color: paletteOf(useStore.getState().plotTheme)[0], opacity: 0.85 },
    text: cats.map((c) => `${c.count}`),
    textposition: "outside" as const,
    hovertemplate: "%{y}: %{x}<extra></extra>",
  }];

  return (
    <div className="flex flex-col gap-3 h-full">
      <TitledPlot plotRefOut={donutRef} storageKey="desc:cat:donut"
        data={donutData}
        layout={{
          ...themedBase,
          // A donut has no plotting area to tint; the pie sits on the card.
          plot_bgcolor: "transparent",
          // automargin grows these as the labels need; the values are the
          // floor, not the budget. The right side still reserves room for the
          // legend, which automargin does not account for.
          margin: { t: 24, r: 160, b: 24, l: 10 },
          autosize: true,
          height: SUMMARY_CHART_HEIGHT,
          legend: { font: { color: "#374151" }, bgcolor: "transparent" },
        }}
        config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        defaultTitle=""
        defaultSubtitle=""
        defaultXAxis=""
        defaultYAxis=""
      />
      <TitledPlot plotRefOut={barRef} storageKey="desc:cat:bar"
        data={barData}
        layout={{ ...themedBase, autosize: true, height: SUMMARY_CHART_HEIGHT,
          xaxis: { ...BASE_LAYOUT.xaxis, showgrid: showGrid, title: { text: "Count" } },
          yaxis: { ...BASE_LAYOUT.yaxis, showgrid: showGrid, automargin: true },
          margin: { ...BASE_LAYOUT.margin, l: 90 },
        }}
        config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        defaultTitle=""
        defaultSubtitle=""
        defaultXAxis="Count"
        defaultYAxis=""
      />
    </div>
  );
}

// ── Scatter view ─────────────────────────────────────────────────────────────

// Use global palette — falls back to default if not set
const _getPalette = () => paletteOf(useStore.getState().plotTheme);
const SYMBOLS  = ["circle","square","diamond","triangle-up","cross","star","hexagram","pentagon"] as const;

function ScatterView({
  sessionId,
  numCols,
  catCols,
  defaultX,
}: {
  sessionId: string;
  numCols: string[];
  catCols: string[];
  defaultX: string;
}) {
  const themedBase = { ...usePlotLayout(), ...BASE_LAYOUT };
  const showGrid = useStore((s) => s.showGrid);
  const [xCol,    setXCol]    = usePersistedPanelState<string>("descriptive_numeric", "xCol", defaultX || numCols[0] || "");
  const [yCol,    setYCol]    = usePersistedPanelState<string>("descriptive_numeric", "yCol", numCols.find((c) => c !== defaultX) ?? "");
  const [color,   setColor]   = usePersistedPanelState<string>("descriptive_numeric", "color", "");
  const [shape,   setShape]   = usePersistedPanelState<string>("descriptive_numeric", "shape", "");
  const scatterCache = useStore((s) => s.panelCache.descriptive_numeric) as
    | Record<string, unknown>
    | undefined;
  const [data,    setData]    = useState<ScatterResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);
  const prevKey = useRef("");
  const scatterRequestIdRef = useRef(0);
  const scatterRef = useRef<PlotCaptureHandle | null>(null);

  // Structural edits are initiated by the parent Summary list. Its shared
  // cache remap must also update these already-mounted local selectors.
  useEffect(() => {
    const cachedX = typeof scatterCache?.xCol === "string" ? scatterCache.xCol : xCol;
    const cachedY = typeof scatterCache?.yCol === "string" ? scatterCache.yCol : yCol;
    const cachedColor = typeof scatterCache?.color === "string" ? scatterCache.color : color;
    const cachedShape = typeof scatterCache?.shape === "string" ? scatterCache.shape : shape;
    const nextX = numCols.includes(cachedX) ? cachedX : (numCols[0] ?? "");
    const nextY = numCols.includes(cachedY) ? cachedY : "";
    const nextColor = catCols.includes(cachedColor) ? cachedColor : "";
    const nextShape = catCols.includes(cachedShape) ? cachedShape : "";
    if (nextX !== xCol) setXCol(nextX);
    if (nextY !== yCol) setYCol(nextY);
    if (nextColor !== color) setColor(nextColor);
    if (nextShape !== shape) setShape(nextShape);
  }, [scatterCache, numCols, catCols, xCol, yCol, color, shape, setXCol, setYCol, setColor, setShape]);

  useEffect(() => {
    if (!xCol || !yCol) {
      scatterRequestIdRef.current += 1;
      prevKey.current = "";
      // eslint-disable-next-line react-hooks/set-state-in-effect -- clear stale fetch result
      setData((d) => (d === null ? d : null));
      setLoading(false);
      return;
    }
    const key = `${xCol}|${yCol}|${color}|${shape}`;
    if (key === prevKey.current) return;
    prevKey.current = key;
    const requestId = ++scatterRequestIdRef.current;
    setLoading(true); setError(null);
    api.post("/api/charts/scatter", {
      session_id: sessionId, x: xCol, y: yCol,
      color: color || undefined,
      shape: shape || undefined,
    })
      .then((r) => {
        if (scatterRequestIdRef.current === requestId) setData(r.data);
      })
      .catch((e) => {
        if (scatterRequestIdRef.current === requestId) {
          setError(e.response?.data?.detail ?? e.message);
          setData(null);
        }
      })
      .finally(() => {
        if (scatterRequestIdRef.current === requestId) setLoading(false);
      });
  }, [xCol, yCol, color, shape, sessionId]);

  const fmt = (v: number | null | undefined, d = 3) =>
    typeof v === "number" ? (Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(2) : v.toFixed(d)) : "—";

  const traces: PlotData[] = [];
  if (data) {
    const pts = data.points;
    const shapeUniq: string[] = shape
      ? Array.from(new Set(pts.map((p) => String(p[shape] ?? "null"))))
      : [];
    const symbolOf = (v: string) => SYMBOLS[shapeUniq.indexOf(v) % SYMBOLS.length] ?? "circle";

    if (color && data.color) {
      const groups: Record<string, { x: unknown[]; y: unknown[]; shapeLabels: string[] }> = {};
      pts.forEach((p) => {
        const g = String(p[color] ?? "null");
        if (!groups[g]) groups[g] = { x: [], y: [], shapeLabels: [] };
        groups[g].x.push(p[xCol]);
        groups[g].y.push(p[yCol]);
        if (shape) groups[g].shapeLabels.push(String(p[shape] ?? "null"));
      });
      Object.entries(groups).forEach(([g, vals], i) => {
        traces.push({
          type: "scatter", mode: "markers",
          x: vals.x, y: vals.y,
          name: g,
          marker: {
            color: _getPalette()[i % _getPalette().length],
            size: 7, opacity: 0.78,
            symbol: shape ? vals.shapeLabels.map(symbolOf) : "circle",
          },
          text: shape ? vals.shapeLabels : undefined,
          hovertemplate:
            `<b>${color}</b>: ${g}` +
            (shape ? `<br><b>${shape}</b>: %{text}` : "") +
            `<br>${xCol}: %{x}<br>${yCol}: %{y}<extra></extra>`,
        });
      });
    } else if (shape) {
      const groups: Record<string, { x: unknown[]; y: unknown[] }> = {};
      pts.forEach((p) => {
        const g = String(p[shape] ?? "null");
        if (!groups[g]) groups[g] = { x: [], y: [] };
        groups[g].x.push(p[xCol]);
        groups[g].y.push(p[yCol]);
      });
      Object.entries(groups).forEach(([g, vals], i) => {
        traces.push({
          type: "scatter", mode: "markers",
          x: vals.x, y: vals.y,
          name: g,
          marker: { color: _getPalette()[0], size: 7, opacity: 0.78, symbol: SYMBOLS[i % SYMBOLS.length] },
          hovertemplate: `<b>${shape}</b>: ${g}<br>${xCol}: %{x}<br>${yCol}: %{y}<extra></extra>`,
        });
      });
    } else {
      traces.push({
        type: "scatter", mode: "markers",
        x: pts.map((p) => p[xCol]),
        y: pts.map((p) => p[yCol]),
        name: "Data",
        marker: { color: _getPalette()[0], size: 6, opacity: 0.7, symbol: "circle" },
        hovertemplate: `${xCol}: %{x}<br>${yCol}: %{y}<extra></extra>`,
      });
    }

    const reg = data.regression;
    if ((reg.line_x?.length ?? 0) > 0) {
      traces.push({
        type: "scatter", mode: "lines",
        x: reg.line_x, y: reg.line_y,
        name: "Fit",
        line: { color: "#ef4444", width: 2, dash: "dash" },
        hoverinfo: "skip",
        showlegend: false,
      });
    }
  }

  const hasGrouping = !!(color || shape);

  return (
    <div className="flex flex-col gap-4 h-full p-4 overflow-y-auto">
      <div className="flex gap-3 flex-wrap flex-shrink-0">
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider">X axis</label>
          <select className="select text-xs min-w-[150px]" value={xCol}
            onChange={(e) => setXCol(e.target.value)}>
            {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider">Y axis</label>
          <select className="select text-xs min-w-[150px]" value={yCol}
            onChange={(e) => setYCol(e.target.value)}>
            <option value="">— pick Y variable —</option>
            {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider">🎨 Color by</label>
          <select className="select text-xs min-w-[150px]" value={color}
            onChange={(e) => setColor(e.target.value)}>
            <option value="">— none —</option>
            {catCols.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider">◆ Shape by</label>
          <select className="select text-xs min-w-[150px]" value={shape}
            onChange={(e) => setShape(e.target.value)}>
            <option value="">— none —</option>
            {catCols.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      </div>

      {!yCol && (
        <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
          Select a continuous variable for the Y axis
        </div>
      )}
      {loading && (
        <div className="flex-1 flex items-center justify-center text-gray-400 animate-pulse">
          Computing…
        </div>
      )}
      {error && (
        <div className="text-red-500 text-xs bg-red-50 rounded-lg p-3">{error}</div>
      )}

      {data && !loading && (
        <>
          <div className="flex gap-3 flex-wrap flex-shrink-0">
            {[
              { key: "n",         label: <i>n</i>,                       value: String(data.points.length) },
              { key: "r",         label: "r" as ReactNode,         value: fmt(data.regression.r) },
              { key: "r2",        label: "r²" as ReactNode,        value: fmt(data.regression.r2) },
              { key: "p",         label: <i>p</i>,                       value: data.regression.p == null ? "—" : data.regression.p < 0.001 ? "<0.001" : fmt(data.regression.p) },
              { key: "slope",     label: "slope" as ReactNode,     value: fmt(data.regression.slope) },
              { key: "intercept", label: "intercept" as ReactNode, value: fmt(data.regression.intercept) },
            ].map(({ key, label, value }) => (
              <div key={key} className="flex flex-col items-center bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 min-w-[60px]">
                <span className="text-[10px] text-gray-400 mb-0.5">{label}</span>
                <span className="text-xs font-mono font-semibold text-gray-800">{value}</span>
              </div>
            ))}
            {data.regression.r != null ? (
              <div className={`flex items-center px-3 py-2 rounded-lg border text-xs font-semibold
                ${Math.abs(data.regression.r) > 0.7
                  ? "bg-indigo-50 border-indigo-200 text-indigo-700"
                  : Math.abs(data.regression.r) > 0.4
                    ? "bg-amber-50 border-amber-200 text-amber-700"
                    : "bg-gray-50 border-gray-200 text-gray-500"}`}>
                {Math.abs(data.regression.r) > 0.7 ? "Strong" :
                 Math.abs(data.regression.r) > 0.4 ? "Moderate" : "Weak"}
                {" "}{data.regression.r >= 0 ? "positive" : "negative"} correlation
              </div>
            ) : (
              <div className="flex items-center px-3 py-2 rounded-lg border border-gray-200 text-xs text-gray-400">
                {data.regression.note ?? "Regression unavailable"}
              </div>
            )}
          </div>

          <div className="flex-1" style={{ minHeight: 320 }}>
            <TitledPlot plotRefOut={scatterRef} storageKey={`desc:scatter:${xCol}:${yCol}`}
              data={traces}
              layout={{
                ...themedBase,
                autosize: true,
                height: SUMMARY_CHART_HEIGHT,
                xaxis: { ...BASE_LAYOUT.xaxis, showgrid: showGrid, title: { text: xCol } },
                yaxis: { ...BASE_LAYOUT.yaxis, showgrid: showGrid, title: { text: yCol } },
                legend: { font: { color: "#374151", size: 11 }, bgcolor: "rgba(249,250,251,0.9)", bordercolor: "#e5e7eb", borderwidth: 1 },
                showlegend: hasGrouping,
                annotations: data.regression.r != null ? [{
                  x: 0.03, y: 0.97,
                  xref: "paper" as const, yref: "paper" as const,
                  text: `r = ${data.regression.r.toFixed(3)}   <i>p</i> = ${fmtP(data.regression.p)}`,
                  showarrow: false,
                  font: { color: "#374151", size: 11 },
                  bgcolor: "rgba(249,250,251,0.9)",
                  bordercolor: "#e5e7eb",
                  borderwidth: 1,
                  borderpad: 5,
                  align: "left" as const,
                  xanchor: "left" as const,
                  yanchor: "top" as const,
                }] : [],
              }}
              config={{ responsive: true, displaylogo: false, displayModeBar: false }}
              defaultTitle=""
              defaultSubtitle=""
              defaultXAxis={xCol}
              defaultYAxis={yCol}
            />
          </div>
        </>
      )}
    </div>
  );
}

// ── Main panel ───────────────────────────────────────────────────────────────

const KIND_CYCLE: Record<string, "numeric" | "categorical" | "text" | "date"> = {
  numeric: "categorical",
  categorical: "text",
  text: "date",
  date: "numeric",
};

const KIND_STYLE: Record<string, { label: string; cls: string }> = {
  numeric:     { label: "N", cls: "bg-blue-100 text-blue-700" },
  categorical: { label: "C", cls: "bg-purple-100 text-purple-700" },
  text:        { label: "T", cls: "bg-gray-100 text-gray-500" },
  date:        { label: "D", cls: "bg-purple-100 text-purple-700" },
};

const COLUMN_LIST_MIN_WIDTH = 224;
const COLUMN_LIST_DEFAULT_WIDTH = 320;
const COLUMN_LIST_MAX_WIDTH = 560;
// Distribution plot needs 520 px; surrounding p-4 adds 32 px. Keep a small
// extra gutter so the divider never clips the plot at the minimum window size.
const RESULT_PANE_MIN_WIDTH = 560;
// …but never at the cost of freezing the divider. Reserving 560 px
// unconditionally meant that below a 784 px window the maximum collapsed onto
// the minimum: aria-valuemin and aria-valuemax both read 224, and the divider
// was a control that could not move at all. That is precisely the window size
// at which the column names are too clipped to read and widening the list
// matters most. The plot carries its own width slider, so a narrower result
// pane is a trade the user can make; a divider that does nothing is not.
const COLUMN_LIST_MIN_TRAVEL = 120;

function getColumnListMaxWidth() {
  if (typeof window === "undefined") return COLUMN_LIST_MAX_WIDTH;
  return Math.max(
    COLUMN_LIST_MIN_WIDTH + COLUMN_LIST_MIN_TRAVEL,
    Math.min(COLUMN_LIST_MAX_WIDTH, window.innerWidth - RESULT_PANE_MIN_WIDTH),
  );
}

function clampColumnListWidth(width: number) {
  return Math.max(
    COLUMN_LIST_MIN_WIDTH,
    Math.min(getColumnListMaxWidth(), width),
  );
}

function persistColumnListWidth(width: number) {
  try {
    localStorage.setItem("uStat.descriptiveColumnListW", String(width));
  } catch {
    // localStorage can be unavailable in private or embedded contexts.
  }
}

export default function DescriptivePanel() {
  const session = useStore((s) => s.session);
  const updateColumnKind = useStore((s) => s.updateColumnKind);
  const reorderColumns   = useStore((s) => s.reorderColumns);
  const renameSessionColumn = useStore((s) => s.renameSessionColumn);
  const removeSessionColumn = useStore((s) => s.removeSessionColumn);
  // Per-column decimal overrides set in the Data tab. The backend has
  // already auto-detected integer columns (it returns `display_decimals`
  // on /api/stats/descriptive), but user overrides from the store win.
  const columnDecimals = useStore((s) => s.columnDecimals);
  const [dragIdx,  setDragIdx]  = useState<number | null>(null);
  const [dropIdx,  setDropIdx]  = useState<number | null>(null);
  const [colMeta, setColMeta] = useState<ColMeta[]>([]);
  const [sparklines, setSparklines] = useState<Record<string, SparkData>>({});
  const [nameTip, setNameTip] = useState<{ text: string; x: number; y: number } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const selectedRef = useRef<string | null>(null);
  const [summary, setSummary] = useState<ColumnSummary | null>(null);
  // Drives which distribution sub-tabs are offered: box plot, violin and Q-Q
  // are numeric-only. While a summary is still loading this stays false, so
  // the tab bar is never disabled on a column whose kind is not known yet.
  const summaryIsCategorical = summary?.type === "categorical";
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [renameCol, setRenameCol] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [busyColumn, setBusyColumn] = useState<string | null>(null);
  const [columnActionError, setColumnActionError] = useState<string | null>(null);
  const [columnListWidth, setColumnListWidth] = useState(() => {
    if (typeof window === "undefined") return COLUMN_LIST_DEFAULT_WIDTH;
    try {
      const stored = Number.parseInt(
        localStorage.getItem("uStat.descriptiveColumnListW")
          || String(COLUMN_LIST_DEFAULT_WIDTH),
        10,
      );
      return clampColumnListWidth(stored || COLUMN_LIST_DEFAULT_WIDTH);
    } catch {
      return clampColumnListWidth(COLUMN_LIST_DEFAULT_WIDTH);
    }
  });
  const [columnListMaxWidth, setColumnListMaxWidth] = useState(
    getColumnListMaxWidth,
  );
  const [view, setView] = usePersistedPanelState<"distribution" | "scatter">("descriptive", "view", "distribution");
  const chartTab = useStore((s) => s.descriptiveTab);
  const setChartTab = useStore((s) => s.setDescriptiveTab);

  const columnListResizeRef = useRef<{
    pointerId: number;
    startX: number;
    startWidth: number;
    currentWidth: number;
    previousCursor: string;
    previousUserSelect: string;
  } | null>(null);
  const stopColumnListResizeRef = useRef<(() => void) | null>(null);
  const renameInputRef = useRef<HTMLInputElement | null>(null);
  const skipRenameCommitRef = useRef(false);
  const renameCommitInFlightRef = useRef(false);

  const onColumnListResizeMove = useCallback((e: PointerEvent) => {
    const resize = columnListResizeRef.current;
    if (!resize || e.pointerId !== resize.pointerId) return;
    const nextWidth = clampColumnListWidth(
      resize.startWidth + e.clientX - resize.startX,
    );
    resize.currentWidth = nextWidth;
    setColumnListWidth(nextWidth);
  }, []);

  const startColumnListResize = (e: React.PointerEvent) => {
    if (!e.isPrimary || e.button !== 0) return;
    e.preventDefault();
    stopColumnListResizeRef.current?.();
    columnListResizeRef.current = {
      pointerId: e.pointerId,
      startX: e.clientX,
      startWidth: columnListWidth,
      currentWidth: columnListWidth,
      previousCursor: document.body.style.cursor,
      previousUserSelect: document.body.style.userSelect,
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    const stopResize = (event?: Event) => {
      const resize = columnListResizeRef.current;
      if (
        resize
        && event
        && "pointerId" in event
        && event.pointerId !== resize.pointerId
      ) {
        return;
      }
      columnListResizeRef.current = null;
      stopColumnListResizeRef.current = null;
      document.removeEventListener("pointermove", onColumnListResizeMove);
      document.removeEventListener("pointerup", stopResize);
      document.removeEventListener("pointercancel", stopResize);
      window.removeEventListener("blur", stopResize);
      if (resize) {
        document.body.style.cursor = resize.previousCursor;
        document.body.style.userSelect = resize.previousUserSelect;
        persistColumnListWidth(resize.currentWidth);
      }
    };

    stopColumnListResizeRef.current = stopResize;
    document.addEventListener("pointermove", onColumnListResizeMove);
    document.addEventListener("pointerup", stopResize);
    document.addEventListener("pointercancel", stopResize);
    window.addEventListener("blur", stopResize);
  };

  const resetColumnListWidth = () => {
    const nextWidth = clampColumnListWidth(COLUMN_LIST_DEFAULT_WIDTH);
    setColumnListWidth(nextWidth);
    persistColumnListWidth(nextWidth);
  };

  const onColumnListResizeKeyDown = (e: React.KeyboardEvent) => {
    let nextWidth: number | null = null;
    if (e.key === "ArrowLeft") nextWidth = columnListWidth - 16;
    if (e.key === "ArrowRight") nextWidth = columnListWidth + 16;
    if (e.key === "Home") nextWidth = COLUMN_LIST_MIN_WIDTH;
    if (e.key === "End") nextWidth = columnListMaxWidth;
    if (e.key === "Enter") nextWidth = COLUMN_LIST_DEFAULT_WIDTH;
    if (nextWidth != null) {
      e.preventDefault();
      const clampedWidth = clampColumnListWidth(nextWidth);
      setColumnListWidth(clampedWidth);
      persistColumnListWidth(clampedWidth);
    }
  };

  useEffect(() => {
    const onWindowResize = () => {
      const nextMax = getColumnListMaxWidth();
      setColumnListMaxWidth(nextMax);
      setColumnListWidth((current) => Math.min(current, nextMax));
    };
    window.addEventListener("resize", onWindowResize);
    return () => {
      window.removeEventListener("resize", onWindowResize);
      stopColumnListResizeRef.current?.();
    };
  }, []);

  // Dedicated resizer for Scatter Plot tab (divider on the RIGHT edge of the plot area)
  // Drag right → scatter grows (correct direction)
  const [scatterPlotWidth, setScatterPlotWidth] = useState(() => {
    if (typeof window !== "undefined") {
      const v = parseInt(localStorage.getItem("uStat.scatterPlotW") || "920", 10);
      return Math.max(520, Math.min(1400, v || 920));
    }
    return 920;
  });

  const scatterPlotResizeRef = useRef<{ startX: number; startW: number } | null>(null);

  const onScatterPlotResizeMove = useCallback((e: PointerEvent) => {
    const d = scatterPlotResizeRef.current;
    if (!d) return;
    const dx = e.clientX - d.startX;
    // Divider is on the RIGHT edge → positive dx (drag right) must GROW the width
    const next = Math.max(520, Math.min(1400, d.startW + dx));
    setScatterPlotWidth(next);
  }, []);

  const startScatterPlotResize = (e: React.PointerEvent) => {
    e.preventDefault();
    scatterPlotResizeRef.current = { startX: e.clientX, startW: scatterPlotWidth };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    const onUp = () => {
      const d = scatterPlotResizeRef.current;
      scatterPlotResizeRef.current = null;
      document.removeEventListener("pointermove", onScatterPlotResizeMove);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      if (d) {
        try {
          localStorage.setItem("uStat.scatterPlotW", String(scatterPlotWidth));
        } catch { /* localStorage unavailable in private mode */ }
      }
    };
    document.addEventListener("pointermove", onScatterPlotResizeMove);
    document.addEventListener("pointerup", onUp, { once: true });
  };

  const resetScatterPlotWidth = () => setScatterPlotWidth(920);

  // 2D resizable container for the main Distribution plot (user wants red drag lines on right + bottom)
  const [distPlotW, setDistPlotW] = useState(() => {
    if (typeof window !== "undefined") {
      const v = parseInt(localStorage.getItem("uStat.distPlotW") || "920", 10);
      return Math.max(520, Math.min(1400, v || 920));
    }
    return 920;
  });
  const [distPlotH, setDistPlotH] = useState(() => {
    if (typeof window !== "undefined") {
      const v = parseInt(localStorage.getItem("uStat.distPlotH") || "520", 10);
      return Math.max(320, Math.min(900, v || 520));
    }
    return 520;
  });

  const distResizeRef = useRef<{ startX: number; startW: number; startY: number; startH: number; mode: "right" | "bottom" } | null>(null);

  const onDistResizeMove = useCallback((e: PointerEvent) => {
    const d = distResizeRef.current;
    if (!d) return;
    if (d.mode === "right") {
      const dx = e.clientX - d.startX;
      const nextW = Math.max(520, Math.min(1400, d.startW + dx));
      setDistPlotW(nextW);
    } else {
      const dy = e.clientY - d.startY;
      const nextH = Math.max(320, Math.min(900, d.startH + dy));
      setDistPlotH(nextH);
    }
  }, []);

  const startDistResize = (mode: "right" | "bottom") => (e: React.PointerEvent) => {
    e.preventDefault();
    distResizeRef.current = {
      startX: e.clientX,
      startW: distPlotW,
      startY: e.clientY,
      startH: distPlotH,
      mode,
    };
    document.body.style.cursor = mode === "right" ? "col-resize" : "row-resize";
    document.body.style.userSelect = "none";
    const onUp = () => {
      distResizeRef.current = null;
      document.removeEventListener("pointermove", onDistResizeMove);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      try {
        localStorage.setItem("uStat.distPlotW", String(distPlotW));
        localStorage.setItem("uStat.distPlotH", String(distPlotH));
      } catch { /* localStorage unavailable in private mode */ }
    };
    document.addEventListener("pointermove", onDistResizeMove);
    document.addEventListener("pointerup", onUp, { once: true });
  };

  useEffect(() => {
    if (!session) return;
    // Fetch real sparkline histograms for all columns
    api.get(`/api/stats/${session.session_id}/sparklines`).then((r) => {
      setSparklines(r.data as Record<string, SparkData>);
    });
    api.get(`/api/stats/${session.session_id}/descriptive`).then((r) => {
      const numStats = r.data as Record<string, { normality_p?: number }>;
      const metas: ColMeta[] = session.columns.map((c) => {
        if (c.kind === "numeric" && numStats[c.name]) {
          const s = numStats[c.name];
          return { name: c.name, kind: "numeric", hist: null, shapiro_p: s.normality_p };
        }
        return { name: c.name, kind: c.kind, top2: null };
      });
      setColMeta(metas);
    });
    // Re-load metadata only on a new dataset — `session` object identity
    // changes on every cell edit, but the metadata only needs to refresh
    // when the underlying session_id changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.session_id]);

  // Guards a fast column switch: only the response matching the MOST RECENT
  // request is allowed to update state, so a slow stale request can't
  // overwrite the summary for whatever column is selected now.
  const summaryRequestIdRef = useRef(0);

  const loadSummary = useCallback((colName: string, kindOverride?: string) => {
    const currentSession = useStore.getState().session;
    if (!currentSession || currentSession.session_id !== session?.session_id) return;
    const sessionId = currentSession.session_id;
    const kind = kindOverride
      ?? currentSession.columns.find((c) => c.name === colName)?.kind
      ?? undefined;
    const requestId = ++summaryRequestIdRef.current;
    selectedRef.current = colName;
    setSelected(colName);
    setSummary(null);
    setSummaryLoading(true);
    api.get(`/api/stats/${sessionId}/column_summary`, { params: { column: colName, kind } })
      .then((r) => {
        if (summaryRequestIdRef.current !== requestId) return; // superseded by a newer request
        const latestSession = useStore.getState().session;
        if (latestSession?.session_id !== sessionId) return;
        const rawSummary = r.data as ColumnSummary;
        if (rawSummary && rawSummary.type === "categorical" && rawSummary.categories) {
          const colMeta = latestSession.columns.find((c) => c.name === colName);
          const vLabels = colMeta?.value_labels ?? {};
          const relabeled: ColumnSummary = {
            ...rawSummary,
            categories: rawSummary.categories.map((c) => ({
              ...c,
              // The endpoint stringifies a float64 code as "0.0" while the
              // labels are keyed "0", so an exact lookup missed every whole
              // number and the donut showed raw codes for a labelled column.
              value: labelFor(vLabels, c.value, String(c.value)),
            })),
          };
          setSummary(relabeled);
          return;
        }
        setSummary(rawSummary);
      })
      .catch(() => {
        // Falls back to the existing "Select a column" empty state below —
        // no crash, no stale data, just nothing rendered for this column.
        if (summaryRequestIdRef.current === requestId) setSummary(null);
      })
      .finally(() => {
        if (summaryRequestIdRef.current === requestId) setSummaryLoading(false);
      });
  }, [session?.session_id]);

  useLayoutEffect(() => {
    if (!renameCol) return;
    renameInputRef.current?.focus();
    renameInputRef.current?.select();
  }, [renameCol]);

  const markColumnMutation = () => {
    useStore.setState((state) => ({
      undoDepth: state.undoDepth + 1,
      redoDepth: 0,
      columnMutationRedo: [],
    }));
  };

  const startColumnRename = (name: string) => {
    if (busyColumn) return;
    setColumnActionError(null);
    skipRenameCommitRef.current = false;
    setRenameCol(name);
    setRenameDraft(name);
  };

  const cancelColumnRename = () => {
    skipRenameCommitRef.current = true;
    setRenameCol(null);
    setRenameDraft("");
  };

  const commitColumnRename = async () => {
    if (
      !session
      || !renameCol
      || renameCommitInFlightRef.current
      || skipRenameCommitRef.current
    ) {
      skipRenameCommitRef.current = false;
      return;
    }
    const oldName = renameCol;
    const newName = renameDraft.trim();
    if (!newName || newName === oldName) {
      setRenameCol(null);
      return;
    }
    if (session.columns.some((column) => column.name === newName)) {
      setColumnActionError(`Column "${newName}" already exists.`);
      renameInputRef.current?.select();
      return;
    }

    renameCommitInFlightRef.current = true;
    setBusyColumn(oldName);
    setColumnActionError(null);
    const sessionId = session.session_id;
    try {
      await runColumnStructureMutation(sessionId, async () => {
        const res = await renameColumn(sessionId, oldName, newName);
        if (useStore.getState().session?.session_id !== sessionId) return;
        renameSessionColumn(oldName, newName, res.data.case_filter);
        setColMeta((current) => current.map((meta) =>
          meta.name === oldName ? { ...meta, name: newName } : meta
        ));
        setSparklines((current) => {
          if (!(oldName in current)) return current;
          const next = { ...current, [newName]: current[oldName] };
          delete next[oldName];
          return next;
        });
        if (selectedRef.current === oldName) {
          summaryRequestIdRef.current += 1;
          selectedRef.current = newName;
          setSelected(newName);
          setSummary(null);
          setSummaryLoading(false);
          loadSummary(newName);
        }
        setNameTip(null);
        setRenameCol(null);
        setRenameDraft("");
        markColumnMutation();
      });
    } catch (error: unknown) {
      const detail =
        (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        ?? (error instanceof Error ? error.message : String(error));
      setColumnActionError(`Rename failed: ${detail}`);
    } finally {
      renameCommitInFlightRef.current = false;
      setBusyColumn(null);
    }
  };

  const handleColumnDelete = async (name: string) => {
    if (!session || busyColumn) return;
    if (!window.confirm(`Delete column "${name}"? You can undo this from the Data tab.`)) {
      return;
    }

    setBusyColumn(name);
    setColumnActionError(null);
    const sessionId = session.session_id;
    try {
      await runColumnStructureMutation(sessionId, async () => {
        const res = await deleteColumn(sessionId, name);
        if (useStore.getState().session?.session_id !== sessionId) return;
        const currentSession = useStore.getState().session;
        const currentIndex = currentSession?.columns.findIndex(
          (column) => column.name === name,
        ) ?? -1;
        const remainingNames = currentSession?.columns
          .filter((column) => column.name !== name)
          .map((column) => column.name) ?? [];
        const nextName = remainingNames[
          Math.min(Math.max(currentIndex, 0), remainingNames.length - 1)
        ];

        removeSessionColumn(name, res.data.case_filter);
        setColMeta((current) => current.filter((meta) => meta.name !== name));
        setSparklines((current) => {
          if (!(name in current)) return current;
          const next = { ...current };
          delete next[name];
          return next;
        });
        if (renameCol === name) cancelColumnRename();
        if (selectedRef.current === name) {
          summaryRequestIdRef.current += 1;
          selectedRef.current = null;
          setSelected(null);
          setSummary(null);
          setSummaryLoading(false);
          if (nextName) loadSummary(nextName);
        }
        setNameTip(null);
        markColumnMutation();
      });
    } catch (error: unknown) {
      const detail =
        (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        ?? (error instanceof Error ? error.message : String(error));
      setColumnActionError(`Delete failed: ${detail}`);
    } finally {
      setBusyColumn(null);
    }
  };

  useEffect(() => {
    if (session && !selected && session.columns.length > 0) {
      loadSummary(session.columns[0].name);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.session_id]);

  if (!session) return null;

  const numCols = session.columns.filter((c) => isNumericKind(c.kind) && !c.analysis_excluded).map((c) => c.name);
  const catCols = session.columns.filter((c) => !isNumericKind(c.kind) && !c.analysis_excluded).map((c) => c.name);

  const filtered = session.columns.filter((c) =>
    c.name.toLowerCase().includes(search.toLowerCase())
  );

  // Per-column decimal resolver. Resolution order:
  //   1. Explicit `d` argument (lets callers force a precision for things
  //      like p-values).
  //   2. User override from the Data-tab decimals control.
  //   3. Server-supplied `display_decimals` from the active summary block
  //      (auto-detected integer columns → 0).
  //   4. Fallback 2.
  const colDecimals = (col: string | null | undefined): number => {
    if (col && col in columnDecimals) return columnDecimals[col];
    if (
      col &&
      summary &&
      summary.type === "numeric" &&
      typeof summary.display_decimals === "number"
    ) {
      return summary.display_decimals;
    }
    return 2;
  };

  const fmt = (v: number | null | undefined, d?: number) => {
    if (typeof v !== "number") return "—";
    if (Math.abs(v) < 0.0001 && v !== 0) return v.toExponential(2);
    const dd = typeof d === "number" ? d : colDecimals(selected);
    return v.toFixed(dd);
  };

  return (
    <div className="flex gap-0 h-full" style={{ minHeight: 0 }}>

      {/* ── Left: column list ── */}
      <div
        className="relative flex-shrink-0 flex flex-col border-r border-gray-200 bg-white overflow-hidden"
        style={{ width: columnListWidth }}
      >
        <div className="p-2 border-b border-gray-200">
          <input
            className="select w-full text-xs"
            placeholder="Search columns…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {columnActionError && (
            <p role="alert" className="mt-1 text-[10px] leading-tight text-red-600">
              {columnActionError}
            </p>
          )}
        </div>
        <div className="overflow-y-auto flex-1">
          {filtered.map((c) => {
            const meta = colMeta.find((m) => m.name === c.name);
            const isActive = selected === c.name;
            const realIdx = session!.columns.findIndex((sc) => sc.name === c.name);
            const isDragOver = dropIdx === realIdx && dragIdx !== realIdx;
            return (
              <div
                key={c.name}
                data-testid={`summary-column-${c.name}`}
                draggable={!renameCol && !busyColumn}
                onDragStart={(e) => {
                  if (renameCol || busyColumn) {
                    e.preventDefault();
                    return;
                  }
                  setDragIdx(realIdx);
                  e.dataTransfer.effectAllowed = "move";
                }}
                onDragOver={(e) => { e.preventDefault(); setDropIdx(realIdx); }}
                onDragLeave={() => { if (dropIdx === realIdx) setDropIdx(null); }}
                onDrop={(e) => {
                  e.preventDefault();
                  if (dragIdx !== null && dragIdx !== realIdx) {
                    void reorderColumns(dragIdx, realIdx).catch((error: unknown) => {
                      const detail = error instanceof Error ? error.message : String(error);
                      setColumnActionError(`Reorder failed: ${detail}`);
                    });
                  }
                  setDragIdx(null);
                  setDropIdx(null);
                }}
                onDragEnd={() => { setDragIdx(null); setDropIdx(null); }}
                onClick={() => { setView("distribution"); loadSummary(c.name); }}
                className={`group flex items-center justify-between px-3 py-2 cursor-grab active:cursor-grabbing border-b border-gray-100 transition-colors select-none
                  ${dragIdx === realIdx ? "opacity-40" : ""}
                  ${busyColumn === c.name ? "opacity-50 pointer-events-none" : ""}
                  ${isDragOver ? "border-t-2 border-t-indigo-500" : ""}
                  ${isActive ? "bg-indigo-50 border-l-2 border-l-indigo-500" : "hover:bg-gray-50"}`}
              >
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <span className="text-gray-300 text-[8px] flex-shrink-0">⠿</span>
                  <span
                    title={`Type: ${c.kind} — click to change`}
                    onClick={(e) => {
                      e.stopPropagation();
                      const next = KIND_CYCLE[c.kind] ?? "numeric";
                      updateColumnKind(c.name, next);
                      if (selected === c.name) loadSummary(c.name, next);
                    }}
                    className={`text-[9px] font-bold px-1 rounded flex-shrink-0 cursor-pointer hover:opacity-70
                      ${KIND_STYLE[c.kind]?.cls ?? "bg-gray-100 text-gray-500"}`}>
                    {KIND_STYLE[c.kind]?.label ?? "?"}
                  </span>
                  {/* The list is narrow, so long names are clipped. Show the
                      full one on hover — same behaviour as the data grid's
                      header, and only when the text is actually cut off. */}
                  {renameCol === c.name ? (
                    <input
                      ref={renameInputRef}
                      aria-label={`Rename ${c.name}`}
                      value={renameDraft}
                      onChange={(e) => setRenameDraft(e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                      onPointerDown={(e) => e.stopPropagation()}
                      onBlur={() => void commitColumnRename()}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          void commitColumnRename();
                        }
                        if (e.key === "Escape") {
                          e.preventDefault();
                          cancelColumnRename();
                        }
                      }}
                      className="min-w-0 flex-1 rounded border border-indigo-300 bg-white px-1.5 py-0.5 text-xs text-gray-800 outline-none focus:ring-1 focus:ring-indigo-400"
                    />
                  ) : (
                    <span
                      className="text-xs text-gray-700 truncate"
                      title="Double-click to rename"
                      onDoubleClick={(e) => {
                        e.stopPropagation();
                        startColumnRename(c.name);
                      }}
                      onMouseEnter={(e) => {
                        const el = e.currentTarget;
                        if (el.scrollWidth <= el.clientWidth) return;
                        const r = el.getBoundingClientRect();
                        setNameTip({
                          text: c.label && c.label !== c.name ? `${c.name} — ${c.label}` : c.name,
                          x: r.left,
                          y: r.bottom + 6,
                        });
                      }}
                      onMouseLeave={() => setNameTip(null)}
                    >
                      {c.name}
                    </span>
                  )}
                </div>
                <div className="ml-1 flex flex-shrink-0 items-center gap-1">
                  {sparklines[c.name] ? (
                    <div className="flex-shrink-0">
                      <Sparkline spark={sparklines[c.name]} />
                    </div>
                  ) : meta && (
                    <div className="w-10 h-3 bg-gray-100 rounded flex-shrink-0 animate-pulse" />
                  )}
                  {renameCol !== c.name && (
                    <div className="flex items-center opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                      <button
                        type="button"
                        aria-label={`Rename ${c.name}`}
                        title="Rename column"
                        onPointerDown={(e) => e.stopPropagation()}
                        onClick={(e) => {
                          e.stopPropagation();
                          startColumnRename(c.name);
                        }}
                        className="rounded p-1 text-gray-400 hover:bg-indigo-100 hover:text-indigo-600"
                      >
                        <Pencil size={12} />
                      </button>
                      <button
                        type="button"
                        aria-label={`Delete ${c.name}`}
                        title="Delete column"
                        onPointerDown={(e) => e.stopPropagation()}
                        onClick={(e) => {
                          e.stopPropagation();
                          void handleColumnDelete(c.name);
                        }}
                        className="rounded p-1 text-gray-400 hover:bg-red-100 hover:text-red-600"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        <div className="p-2 border-t border-gray-200 text-xs text-gray-400 text-center">
          {session.columns.length} columns · {session.rows} rows
        </div>
      </div>

      {/* The divider is its own flex item, not an overlay pinned inside the
          list. As a 1 px transparent strip on top of the list's right edge it
          was a real control that could not be hit: the target was one pixel
          wide, invisible until the pointer was already on it, and it sat over
          the row's own rename and delete buttons. Here it owns its space,
          takes a normal-sized grab area, and shows a hairline so it reads as
          draggable before you touch it. */}
      <div
        role="separator"
        aria-label="Resize column list"
        aria-orientation="vertical"
        aria-valuemin={COLUMN_LIST_MIN_WIDTH}
        aria-valuemax={columnListMaxWidth}
        aria-valuenow={columnListWidth}
        tabIndex={0}
        onPointerDown={startColumnListResize}
        onDoubleClick={resetColumnListWidth}
        onKeyDown={onColumnListResizeKeyDown}
        className="group relative z-20 flex w-2 flex-shrink-0 cursor-col-resize touch-none items-center justify-center bg-gray-100 hover:bg-indigo-100 active:bg-indigo-200 focus:bg-indigo-100 focus:outline-none"
        title="Drag left or right to resize · Arrow keys resize · Enter or double-click resets"
      >
        <span className="h-8 w-0.5 rounded bg-gray-300 transition-colors group-hover:bg-indigo-400 group-active:bg-indigo-500" />
      </div>

      {/* ── Right: view area ── */}
      <div className="flex-1 flex flex-col overflow-hidden bg-white">

        {/* ── Sub-tabs under Descriptive: Histogram | Box Plot | Violin | Q-Q Plot | Scatter Plot ── */}
        <div className="flex items-center gap-1 px-4 py-1.5 border-b border-gray-200 flex-shrink-0 bg-gray-50 overflow-x-auto">
          {[
            { id: "histogram", label: "Histogram" },
            { id: "boxplot",   label: "Box Plot" },
            { id: "violin",    label: "Violin" },
            { id: "qq",        label: "Q-Q Plot" },
            { id: "scatter",   label: "Scatter Plot" },
          ].map(({ id, label }) => {
            // A box plot, a violin and a Q-Q plot all need an ordered numeric
            // scale; none of them is defined for an unordered category. The
            // categorical view drew its own chart whatever the tab said, so
            // all four tabs rendered the identical pie-and-bar while the tab
            // bar highlighted whichever one had been clicked — a live control
            // that did nothing, and looked like the chart had failed to load.
            const numericOnly = id === "boxplot" || id === "violin" || id === "qq";
            const disabled = numericOnly && summaryIsCategorical;
            // Highlight what is actually on screen: for a categorical column
            // that is always the category breakdown.
            const shownTab = summaryIsCategorical ? "histogram" : chartTab;
            const isActive = (id === "scatter"
              ? view === "scatter"
              : view === "distribution" && shownTab === id);
            return (
              <button
                key={id}
                disabled={disabled}
                title={disabled
                  ? `${label} needs an ordered numeric scale — ${selected} is categorical`
                  : undefined}
                onClick={() => {
                  if (id === "scatter") {
                    setView("scatter");
                  } else {
                    setView("distribution");
                    setChartTab(id as ChartTab);
                  }
                }}
                className={`px-3 py-1 rounded-md text-xs font-medium whitespace-nowrap transition-colors
                  ${disabled
                    ? "text-gray-300 cursor-not-allowed"
                    : isActive
                      ? "bg-indigo-600 text-white shadow-sm"
                      : "text-gray-600 hover:text-gray-800 hover:bg-gray-200"}`}
              >
                {label}
              </button>
            );
          })}
        </div>

        {/* ── Scatter Plot view: clean scatter with red draggable resize line on the right (matching the other Descriptive sub-tabs) ── */}
        {view === "scatter" && (
          <div className="p-4">
            <div 
              className="relative border border-gray-200 rounded-lg bg-white shadow-sm overflow-hidden"
              style={{ width: `${scatterPlotWidth}px`, minWidth: 520, maxWidth: '100%' }}
            >
              <ScatterView
                key={session.session_id}
                sessionId={session.session_id}
                numCols={numCols}
                catCols={catCols}
                defaultX={selected && numCols.includes(selected) ? selected : (numCols[0] ?? "")}
              />

              {/* Red vertical resize line on the right (drag to change scatter width) */}
              <div
                onPointerDown={startScatterPlotResize}
                onDoubleClick={resetScatterPlotWidth}
                className="absolute top-0 bottom-0 w-[5px] right-0 cursor-col-resize bg-red-500/70 hover:bg-red-600 active:bg-red-700 transition-colors z-20"
                title="Drag the red line to resize the scatter plot width • Double-click to reset"
              />
            </div>

            <div className="text-[10px] text-gray-400 mt-1">
              Drag the red line on the right edge to resize the scatter plot width (like the other Descriptive tabs)
            </div>
          </div>
        )}

        {/* ── Distribution view ── */}
        {view === "distribution" && (
          <>
            {summaryLoading && (
              <div className="flex-1 flex items-center justify-center text-gray-400 animate-pulse">
                Computing distribution…
              </div>
            )}
            {!summaryLoading && !summary && (
              <div className="flex-1 flex items-center justify-center text-gray-400">
                Select a column to view distribution
              </div>
            )}
            {!summaryLoading && summary && (
              <>
                {/* Header - compacted to a single row */}
                <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200 flex-shrink-0">
                  <div className="flex items-center gap-2 text-sm">
                    <span className="font-semibold text-gray-900">Distribution of</span>
                    <span className="font-semibold text-indigo-600">{selected}</span>
                    <span className="text-gray-400">·</span>
                    <span className="text-gray-600">
                      {summary.type === "numeric" ? "Continuous" : "Categorical"} · <i>n</i>={summary.n}
                    </span>
                    {(summary.missing ?? 0) > 0 && (
                      <span className="text-amber-600 text-xs">· {summary.missing} missing</span>
                    )}
                    {summary.type === "numeric" && summary.normality_p != null && (
                      <span className={`text-xs ${summary.normal ? "text-emerald-600" : "text-amber-600"}`}>
                        · {summary.normal ? "Normal" : "Non-normal"} (<i>p</i>={fmtP(summary.normality_p)})
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3">
                    <ResultExporter
                      title={`Summary_${selected}`}
                      headers={summary.type === "numeric"
                        ? ["Statistic", "Value"]
                        : ["Category", "Count", "Percent"]}
                      rows={summary.type === "numeric"
                        ? (() => {
                            // Exports keep one extra digit of precision over
                            // the on-screen display, but never less than the
                            // user's column rule (integer columns stay integer).
                            const dCol = colDecimals(selected);
                            const dExp = Math.max(dCol, dCol === 0 ? 0 : 4);
                            const fix = (x: number | undefined) =>
                              typeof x === "number" ? x.toFixed(dExp) : "";
                            return [
                              ["N", summary.n],
                              ["Missing", summary.missing],
                              ["Mean", fix(summary.mean)],
                              ["SD", fix(summary.std)],
                              ["Median", fix(summary.median)],
                              ["Q1", fix(summary.q1)],
                              ["Q3", fix(summary.q3)],
                              ["IQR", fix(summary.iqr)],
                              ["Min", fix(summary.min)],
                              ["Max", fix(summary.max)],
                              ["Skewness", summary.skewness?.toFixed(4) ?? ""],
                              ["Kurtosis", summary.kurtosis?.toFixed(4) ?? ""],
                              ["Normality test", summary.normality_test ?? ""],
                              ["Normality p",
                                fmtP(summary.normality_p ?? summary.shapiro_p)],
                            ];
                          })()
                        : (summary.categories ?? []).map((c) => [
                            c.value, c.count,
                            c.pct != null ? `${c.pct.toFixed(1)}%` : "",
                          ])}
                    />
                  {summary.type === "numeric" && (
                    <div className={`px-3 py-1.5 rounded-lg text-xs font-semibold border
                      ${summary.normal
                        ? "bg-green-50 border-green-300 text-green-700"
                        : "bg-red-50 border-red-300 text-red-600"}`}>
                      {summary.normality_label}
                      <span className="font-normal text-gray-400 ml-1">
                        ({summary.normality_test ?? "Shapiro-Wilk"} <i>p</i> = {fmt(summary.normality_p ?? summary.shapiro_p, 3)})
                      </span>
                      <div className="text-[10px] font-normal text-gray-400 mt-0.5">
                        {(summary.n ?? 0) < 50 ? "n < 50 → Shapiro-Wilk" : "n ≥ 50 → Kolmogorov-Smirnov"}
                      </div>
                    </div>
                  )}
                  {summary.type === "categorical" && (
                    <div className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-purple-300 bg-purple-50 text-purple-700">
                      {summary.n_categories} categories
                    </div>
                  )}
                  </div>
                </div>

                {/* Stats strip (numeric) — inline single-line for max vertical space */}
                {summary.type === "numeric" && (
                  <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 px-4 py-1.5 border-b border-gray-200 text-xs flex-shrink-0">
                    {[
                      ["Mean", fmt(summary.mean)],
                      ["SD", fmt(summary.std)],
                      ["Median", fmt(summary.median)],
                      ["Q1", fmt(summary.q1)],
                      ["Q3", fmt(summary.q3)],
                      ["IQR", fmt(summary.iqr)],
                      ["Min", fmt(summary.min)],
                      ["Max", fmt(summary.max)],
                      ["Skew", fmt(summary.skewness)],
                    ].map(([k, v], i) => (
                      <span key={k as string} className="whitespace-nowrap">
                        {i > 0 && <span className="text-gray-300 mr-3">·</span>}
                        <span className="text-gray-400">{k}</span>{" "}
                        <span className="font-mono font-semibold text-gray-800">{v}</span>
                      </span>
                    ))}
                  </div>
                )}
                {/* Interpretation guidance */}
                {summary.type === "numeric" && (
                  <div className="px-4 py-1.5 border-b border-gray-100 bg-amber-50 flex-shrink-0">
                    <p className="text-[10px] text-amber-800 leading-relaxed">
                      {(() => {
                        // Narrative inherits the column's display rule
                        // (integer column \u2192 no decimals), so the suggested
                        // report-this string is publication-ready.
                        const d = colDecimals(selected);
                        const f = (v?: number) =>
                          typeof v === "number" ? v.toFixed(d) : "\u2014";
                        const pNorm = fmtP(summary.normality_p);
                        return summary.normal
                          ? <>Normal distribution ({summary.normality_test}, <i>p</i>={pNorm}) \u2014 report Mean \u00B1 SD ({f(summary.mean)} \u00B1 {f(summary.std)}).</>
                          : <>Non-normal ({summary.normality_test}, <i>p</i>={pNorm}) \u2014 report Median [IQR] ({f(summary.median)} [{f(summary.q1)}\u2013{f(summary.q3)}]).</>;
                      })()}
                      {Math.abs(summary.skewness ?? 0) > 2 ? " Highly skewed \u2014 consider log-transformation." :
                       Math.abs(summary.skewness ?? 0) > 1 ? " Moderately skewed." : ""}
                    </p>
                  </div>
                )}
                {summary.type === "categorical" && (
                  <div className="px-4 py-1.5 border-b border-gray-100 bg-amber-50 flex-shrink-0">
                    <p className="text-[10px] text-amber-800 leading-relaxed">
                      {summary.categories?.length} categories, <i>n</i> = {summary.n}. Report as <i>n</i> (%). Most frequent: {summary.categories?.[0]?.value} ({summary.categories?.[0]?.pct}%).
                      {(summary.missing ?? 0) > 0 ? ` Missing: ${summary.missing} (${((summary.missing ?? 0) / ((summary.n ?? 0) + (summary.missing ?? 0)) * 100).toFixed(1)}%).` : ""}
                    </p>
                  </div>
                )}

                {/* Charts - now inside a 2D resizable box with red drag lines on right + bottom (as requested) */}
                <div className="p-4">
                  <div
                    className="relative border border-gray-200 rounded-lg bg-white shadow-sm overflow-hidden"
                    style={{ width: `${distPlotW}px`, height: `${distPlotH}px`, maxHeight: "calc(100vh - 240px)", minWidth: 520, minHeight: 320 }}
                  >
                    {/* The actual distribution plot content */}
                    <div className="absolute inset-0 overflow-auto p-4">
                      {summary.type === "numeric" && <NumericView summary={summary} loadSummary={loadSummary} selected={selected ?? ""} />}
                      {summary.type === "categorical" && <CategoricalView summary={summary} />}
                    </div>

                    {/* Right vertical red resize line (drag to change width) */}
                    <div
                      onPointerDown={startDistResize("right")}
                      className="absolute top-0 bottom-0 w-[5px] right-0 cursor-col-resize bg-red-500/70 hover:bg-red-600 active:bg-red-700 z-20"
                      title="Drag to resize plot width"
                    />

                    {/* Bottom horizontal red resize line (drag to change height) */}
                    <div
                      onPointerDown={startDistResize("bottom")}
                      className="absolute left-0 right-0 h-[5px] bottom-0 cursor-row-resize bg-red-500/70 hover:bg-red-600 active:bg-red-700 z-20"
                      title="Drag to resize plot height"
                    />

                    {/* Small corner handle for convenience */}
                    <div
                      onPointerDown={startDistResize("right")} // diagonal would be nicer but this is simple
                      className="absolute bottom-0 right-0 w-3 h-3 bg-red-500/80 cursor-nwse-resize z-30 rounded-tl"
                      title="Drag corner to resize both"
                    />
                  </div>

                  <div className="text-[10px] text-gray-400 mt-1">
                    Drag the red lines on the right and bottom to resize the plot area • Changes are remembered
                  </div>
                </div>
              </>
            )}
          </>
        )}
      </div>

      {/* Full column name for a list entry whose text is clipped. Fixed and
          pointer-events-none so it escapes the list's own scroll clipping
          without intercepting the click that selects the column. */}
      {nameTip && (
        <div
          role="tooltip"
          className="fixed z-50 pointer-events-none max-w-sm rounded-lg bg-gray-900/95 px-2.5 py-1.5 text-xs text-white shadow-lg"
          style={{ left: Math.min(nameTip.x, window.innerWidth - 320), top: nameTip.y }}
        >
          {nameTip.text}
        </div>
      )}
    </div>
  );
}
