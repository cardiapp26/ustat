/**
 * PSMPanel — Propensity Score Matching
 *
 * Pipeline:
 * 1. Logistic regression (Treatment ~ Covariates) → propensity scores
 * 2. Nearest-neighbor 1:1 matching with caliper (0.2 × SD of PS)
 * 3. SMD balance assessment before / after
 * 4. Love Plot — the publication-standard balance visualization
 * 5. Optional outcome analysis on matched cohort
 */
import { useState, useRef, useMemo } from "react";
import { useStore, paletteOf, analysisCols, type Session } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { runPSM, getSessionInfo } from "../api";
import { Tip } from "./Tip";
import TitledPlot from "./TitledPlot";
import ResultExporter from "./ResultExporter";
import { fmtP } from "../lib/format";
import { useResizableRightCol } from "../hooks/useResizableRightCol";
import { exportDataset } from "../lib/exportDataset";
import type { PlotData, PlotLayout, PlotCaptureHandle } from "../lib/plotTypes";

interface OutcomeCoefficient {
  variable: string;
  estimate?: number;
  se?: number;
  z?: number;
  p: number;
  or?: number;
  or_low?: number;
  or_high?: number;
  hr?: number;
  hr_low?: number;
  hr_high?: number;
}

interface OutcomeResult {
  type?: string;
  model?: string;
  n?: number;
  n_events?: number;
  n_informative_sets?: number;
  n_uninformative_sets?: number;
  concordance?: number;
  aic?: number;
  bic?: number;
  coefficients: OutcomeCoefficient[];
  method_note?: string;
  error?: string;
}

interface RosenbaumCurvePoint {
  gamma: number;
  p_upper: number;
}

interface RosenbaumResult {
  applicable?: boolean;
  reason?: string;
  discordant_pairs?: number;
  b?: number;
  c?: number;
  p_unbiased?: number;
  critical_gamma?: number | null;
  gamma_max?: number;
  alpha?: number;
  curve?: RosenbaumCurvePoint[];
}

interface PSMResult {
  balance_achieved: boolean;
  n_total: number;
  n_treated: number;
  n_control: number;
  n_matched_pairs: number;
  n_matched_controls: number;
  n_unmatched: number;
  n_trimmed_common_support: number;
  caliper_used?: number;
  caliper_scale?: string;
  common_support?: { lo?: number; hi?: number };
  matched_session_id?: string;
  smd_before: Record<string, number>;
  smd_after: Record<string, number>;
  avg_smd_before: number;
  avg_smd_after: number;
  reduction_pct: number;
  variance_ratio_after?: Record<string, number | null>;
  variance_ratio_before?: Record<string, number | null>;
  ks_p_after?: Record<string, number | null>;
  ps_distribution?: {
    treated_unmatched: number[];
    control_unmatched: number[];
    treated_matched: number[];
    control_matched: number[];
  };
  outcome_result?: OutcomeResult;
  rosenbaum?: RosenbaumResult;
  matching_warning?: string;
}

const smdColor = (smd: number) =>
  smd < 0.10 ? "text-emerald-600" : smd < 0.20 ? "text-amber-500" : "text-red-500";

// Geometry only. Background, font and colourway follow the global chart
// theme at render; hard-coded here they made the theme picker a no-op.
const PLOT_BASE = {
  margin: { t: 30, r: 24, b: 56, l: 130 },
};

function LovePlot({
  smdBefore,
  smdAfter,
  threshold,
  showConnectors,
  showGrid,
}: {
  smdBefore: Record<string, number>;
  smdAfter: Record<string, number>;
  threshold: number;
  showConnectors: boolean;
  showGrid: boolean;
}) {
  const plotTheme = useStore((s) => s.plotTheme);
  // Subscribed, not read once: this is what repaints when the palette changes.
  const themedBase: Record<string, unknown> = {
    ...PLOT_BASE,
    paper_bgcolor: "transparent",
    plot_bgcolor: plotTheme.plotBg,
    font: { family: plotTheme.fontFamily, color: "#374151", size: plotTheme.fontSize },
    colorway: paletteOf(plotTheme),
  };
  const plotRef = useRef<PlotCaptureHandle | null>(null);
  const covariates = Object.keys(smdBefore).reverse(); // bottom-to-top

  const xMax = Math.max(0.4, ...Object.values(smdBefore), ...Object.values(smdAfter)) * 1.15;

  const traces: PlotData[] = [
    {
      type: "scatter",
      mode: "markers",
      name: "Unmatched cohort",
      x: covariates.map((c) => smdBefore[c]),
      y: covariates,
      marker: { symbol: "square", size: 11, color: "#ef4444" },
      hovertemplate: "<b>%{y}</b><br>SMD (before) = %{x:.4f}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "markers",
      name: "Matched cohort",
      x: covariates.map((c) => smdAfter[c]),
      y: covariates,
      marker: { symbol: "circle", size: 11, color: "#3b82f6" },
      hovertemplate: "<b>%{y}</b><br>SMD (after) = %{x:.4f}<extra></extra>",
    },
  ];

  if (showConnectors) {
    for (const cov of covariates) {
      traces.push({
        type: "scatter",
        mode: "lines",
        x: [smdBefore[cov], smdAfter[cov]],
        y: [cov, cov],
        line: { color: "#94a3b8", width: 1.2, dash: "dot" },
        showlegend: false,
        hoverinfo: "skip",
      });
    }
  }

  const layout: PlotLayout = {
    ...themedBase,
    autosize: true,
    height: Math.max(260, covariates.length * 52 + 80),
    xaxis: {
      title: { text: "Standardized Mean Difference (SMD)" },
      range: [0, xMax],
      gridcolor: showGrid ? "#e5e7eb" : "transparent",
      zeroline: false,
    },
    yaxis: {
      gridcolor: "transparent",
      automargin: true,
    },
    legend: {
      x: 1, y: 0, xanchor: "right", yanchor: "bottom",
      bgcolor: "rgba(249,250,251,0.9)",
      bordercolor: "#e5e7eb", borderwidth: 1,
      font: { size: 11 },
    },
    shapes: [
      {
        type: "line",
        x0: threshold, x1: threshold,
        y0: 0, y1: 1,
        xref: "x", yref: "paper",
        line: { color: "#dc2626", width: 1.5, dash: "dash" },
      },
    ],
    annotations: [
      {
        x: threshold, y: 1.02,
        xref: "x", yref: "paper",
        text: `Threshold (${threshold})`,
        showarrow: false,
        font: { color: "#dc2626", size: 10 },
        xanchor: "center",
      },
    ],
  };

  return (
    <TitledPlot
      plotRefOut={plotRef}
      storageKey="psm:love"
      data={traces}
      layout={layout}
      config={{ responsive: true, displaylogo: false, displayModeBar: false }}
      defaultTitle=""
      defaultSubtitle=""
      defaultXAxis="Standardized Mean Difference (SMD)"
      defaultYAxis=""
    />
  );
}

function PSOverlapPlot({
  psDist,
  showGrid,
}: {
  psDist: { treated_unmatched: number[]; control_unmatched: number[]; treated_matched: number[]; control_matched: number[] };
  showGrid: boolean;
}) {
  const plotRef = useRef<PlotCaptureHandle | null>(null);
  const plotTheme = useStore((s) => s.plotTheme);
  const themedBase: Record<string, unknown> = {
    ...PLOT_BASE,
    paper_bgcolor: "transparent",
    plot_bgcolor: plotTheme.plotBg,
    font: { family: plotTheme.fontFamily, color: "#374151", size: plotTheme.fontSize },
    colorway: paletteOf(plotTheme),
  };
  return (
    <TitledPlot
      plotRefOut={plotRef}
      storageKey="psm:ps-overlap"
      data={[
        {
          type: "histogram",
          name: "Treated (unmatched)",
          x: psDist.treated_unmatched,
          opacity: 0.55,
          marker: { color: "#ef4444" },
          nbinsx: 25,
          hovertemplate: "PS: %{x:.3f}<br>Count: %{y}<extra></extra>",
        },
        {
          type: "histogram",
          name: "Control (unmatched)",
          x: psDist.control_unmatched,
          opacity: 0.55,
          marker: { color: "#3b82f6" },
          nbinsx: 25,
          hovertemplate: "PS: %{x:.3f}<br>Count: %{y}<extra></extra>",
        },
      ]}
      layout={{
        ...themedBase,
        barmode: "overlay",
        autosize: true,
        height: 230,
        xaxis: {
          title: { text: "Propensity Score" },
          range: [0, 1],
          gridcolor: showGrid ? "#e5e7eb" : "transparent",
        },
        yaxis: { title: { text: "Count" }, gridcolor: showGrid ? "#e5e7eb" : "transparent" },
        legend: { x: 1, y: 1, xanchor: "right", bgcolor: "rgba(249,250,251,0.9)", bordercolor: "#e5e7eb", borderwidth: 1 },
        annotations: [{
          x: 0.5, y: 1.08, xref: "paper", yref: "paper",
          text: "Propensity Score Overlap (Common Support)",
          showarrow: false, font: { color: "#374151", size: 12 },
        }],
      } as PlotLayout}
      config={{ responsive: true, displaylogo: false, displayModeBar: false }}
      defaultTitle=""
      defaultSubtitle=""
      defaultXAxis="Propensity Score"
      defaultYAxis="Count"
    />
  );
}

export default function PSMPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <PSMPanelBody session={session} />;
}

function PSMPanelBody({ session }: { session: Session }) {
  const showGrid = useStore((s) => s.showGrid);
  const setSession = useStore((s) => s.setSession);
  const setOriginalSession = useStore((s) => s.setOriginalSession);
  const { w: rightColW, onDragStart: onResizeStart, onReset: onResizeReset } =
    useResizableRightCol("PSMPanel.result", 480);

  const allCols = analysisCols(session.columns).map((c) => c.name);

  const binaryCols = useMemo(() =>
    allCols.filter((col) => {
      const vals = new Set(session.preview.map((r) => r[col]).filter((v) => v != null));
      return vals.size === 2 && [...vals].every((v) => v === 0 || v === 1);
    }),
    // Recompute only per dataset, not on every preview/columns reidentification —
    // the binary-column shape is a property of the loaded session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [session.session_id]
  );

  // Form State — genuine selections persisted across tab switches (per-session cache)
  const [treatCol, setTreatCol] = usePersistedPanelState<string>("psm", "treatCol", binaryCols[0] ?? allCols[0] ?? "");
  const [outcomeCol, setOutcomeCol] = usePersistedPanelState<string>("psm", "outcomeCol", "");
  const [covariates, setCovariates] = usePersistedPanelState<string[]>("psm", "covariates", []);
  const [caliper, setCaliper] = usePersistedPanelState<number>("psm", "caliper", 0.2);
  const [caliperScale, setCaliperScale] = usePersistedPanelState<"logit" | "raw">("psm", "caliperScale", "logit");
  const [trimCommonSupport, setTrimCommonSupport] = usePersistedPanelState<boolean>("psm", "trimCommonSupport", false);
  const [ratio, setRatio] = usePersistedPanelState<number>("psm", "ratio", 1);
  const [randomState, setRandomState] = usePersistedPanelState<number>("psm", "randomState", 42);
  const [scoreMethod, setScoreMethod] = usePersistedPanelState<"logistic" | "probit" | "gbm">("psm", "scoreMethod", "logistic");
  const [matchingMethod, setMatchingMethod] = usePersistedPanelState<"greedy" | "optimal">("psm", "matchingMethod", "greedy");
  const [exactMatch, setExactMatch] = usePersistedPanelState<string[]>("psm", "exactMatch", []);
  const [outcomeType, setOutcomeType] = usePersistedPanelState<"binary" | "survival">("psm", "outcomeType", "binary");
  const [survDuration, setSurvDuration] = usePersistedPanelState<string>("psm", "survDuration", "");
  const [survEvent, setSurvEvent] = usePersistedPanelState<string>("psm", "survEvent", "");
  const [computeRosenbaum, setComputeRosenbaum] = usePersistedPanelState<boolean>("psm", "computeRosenbaum", false);
  const [rosenbaumGammaMax] = useState<number>(3.0);
  const [covFilter, setCovFilter] = useState("");

  // Result & UI
  const [result, setResult] = useState<PSMResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.10);
  const [showConnectors, setShowConnectors] = useState(true);

  const toggleCov = (c: string) =>
    setCovariates(covariates.includes(c) ? covariates.filter((x) => x !== c) : [...covariates, c]);

  const run = async () => {
    if (covariates.length === 0) { setError("Select at least one covariate"); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await runPSM({
        session_id: session.session_id,
        treatment_col: treatCol,
        covariates,
        outcome_col: outcomeType === "binary" ? (outcomeCol || undefined) : undefined,
        caliper,
        caliper_scale: caliperScale,
        trim_common_support: trimCommonSupport,
        ratio,
        random_state: Number.isFinite(randomState) ? randomState : undefined,
        score_method: scoreMethod,
        matching_method: matchingMethod,
        exact_match: exactMatch.length > 0 ? exactMatch : undefined,
        outcome_type: outcomeType,
        survival_duration_col: outcomeType === "survival" ? (survDuration || undefined) : undefined,
        survival_event_col: outcomeType === "survival" ? (survEvent || undefined) : undefined,
        compute_rosenbaum: outcomeType === "binary" && ratio === 1 && computeRosenbaum,
        rosenbaum_gamma_max: rosenbaumGammaMax,
      });
      setResult(res.data as PSMResult);
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
      setError(typeof msg === "string" ? msg : (e instanceof Error ? e.message : "PSM failed"));
    } finally { setLoading(false); }
  };

  const smdExportHeaders = ["Covariate", "SMD Before", "SMD After", "Reduction %", "Balanced (<0.10)"];
  const smdExportRows = result
    ? Object.keys(result.smd_before).map((cov) => [
        cov,
        result.smd_before[cov].toFixed(4),
        result.smd_after[cov].toFixed(4),
        (((result.smd_before[cov] - result.smd_after[cov]) / result.smd_before[cov]) * 100).toFixed(1) + "%",
        result.smd_after[cov] < 0.10 ? "Yes" : "No",
      ])
    : [];

  const availableCovs = allCols.filter(
    (c) => c !== treatCol && c !== outcomeCol &&
           c.toLowerCase().includes(covFilter.toLowerCase())
  );

  return (
    <div className="flex gap-4 min-h-0">
      {/* ── Left sidebar ─────────────────────────────────────────────────── */}
      <div className="w-64 flex-shrink-0 space-y-3 overflow-y-auto">
        {/* Header */}
        <div className="panel bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-200 space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xl">🧬</span>
            <h2 className="text-sm font-bold text-indigo-800">
              Propensity Score Matching (PSM)
            </h2>
          </div>
          <p className="text-[10px] text-indigo-600 leading-snug">
            Mimics an RCT from observational data by balancing confounders between treated and control groups via 1:k matching.
          </p>
        </div>

        {/* Treatment variable */}
        <div className="panel space-y-2">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            Treatment Variable
            <Tip wide text="The binary intervention variable: 1 = Treated, 0 = Control. Must be coded 0/1." />
          </label>
          <select
            className="select w-full"
            value={treatCol}
            onChange={(e) => { setTreatCol(e.target.value); setResult(null); }}>
            {binaryCols.length > 0
              ? binaryCols.map((c) => <option key={c} value={c}>{c}</option>)
              : allCols.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          {!binaryCols.includes(treatCol) && (
            <p className="text-[10px] text-amber-600">⚠ Column may not be binary (0/1)</p>
          )}
        </div>

        {/* Outcome variable */}
        <div className="panel space-y-2">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            Outcome Variable <span className="normal-case font-normal text-gray-400">(optional)</span>
            <Tip wide text="The endpoint to analyse in the matched cohort. If binary, conditional logistic or GEE logistic is run automatically. Leave blank to only assess balance." />
          </label>
          <select
            className="select w-full"
            value={outcomeCol}
            onChange={(e) => { setOutcomeCol(e.target.value); setResult(null); }}>
            <option value="">— Skip outcome analysis —</option>
            {allCols.filter((c) => c !== treatCol).map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>

        {/* Covariates */}
        <div className="panel space-y-2">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            Covariates (Confounders)
            <Tip wide text="Baseline patient characteristics that influence both treatment and outcome. Include all known confounders." />
          </label>
          <input
            type="text"
            placeholder="Filter covariates…"
            className="select w-full text-xs py-1"
            value={covFilter}
            onChange={(e) => setCovFilter(e.target.value)}
          />
          <div className="flex gap-1">
            <button onClick={() => setCovariates(availableCovs)}
              className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">All</button>
            <button onClick={() => setCovariates([])}
              className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">None</button>
          </div>
          <div className="max-h-52 overflow-y-auto space-y-0.5 border border-gray-200 rounded-lg p-1">
            {availableCovs.map((c) => (
              <label key={c} className="flex items-center gap-1.5 text-xs px-1 py-0.5 rounded hover:bg-gray-50 cursor-pointer">
                <input type="checkbox" className="accent-indigo-500"
                  checked={covariates.includes(c)} onChange={() => toggleCov(c)} />
                <span className="text-gray-700 truncate">{c}</span>
              </label>
            ))}
          </div>
          <p className="text-[10px] text-gray-400">{covariates.length} selected</p>
        </div>

        {/* Caliper */}
        <div className="panel space-y-2">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            Caliper
            <Tip wide text="Maximum allowed PS distance for a match, as a fraction of the SD of propensity scores. Medical standard = 0.2." />
          </label>
          <div className="flex gap-2 items-center">
            <input type="range" min="0.05" max="0.50" step="0.05"
              className="flex-1 accent-indigo-500"
              value={caliper}
              onChange={(e) => setCaliper(parseFloat(e.target.value))} />
            <span className="font-mono text-sm font-semibold text-indigo-700 w-10 text-right">{caliper}</span>
          </div>
          <div className="flex justify-between text-[9px] text-gray-400">
            <span>0.05 (strict)</span>
            <span className="text-indigo-500">0.20 ★ standard</span>
            <span>0.50 (loose)</span>
          </div>
          <div className="pt-2 border-t border-gray-100 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
                Caliper scale
                <Tip wide text="Austin (2011) recommends matching on the LOGIT of the propensity score: 'logit' is the publication standard." />
              </span>
              <div className="flex gap-1">
                {(["logit", "raw"] as const).map((s) => (
                  <button key={s} onClick={() => setCaliperScale(s)}
                    className={`px-2 py-0.5 text-[10px] rounded border transition-colors ${caliperScale === s ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-500 hover:bg-gray-50"}`}>
                    {s === "logit" ? "Logit ★" : "Raw PS"}
                  </button>
                ))}
              </div>
            </div>
            <label className="flex items-start gap-2 cursor-pointer">
              <input type="checkbox" className="accent-indigo-500 mt-0.5"
                checked={trimCommonSupport} onChange={(e) => setTrimCommonSupport(e.target.checked)} />
              <span className="text-[10px] text-gray-600 leading-tight">
                <span className="font-medium">Trim to common support</span>
                <Tip wide text="Exclude treated and control units with PS outside the overlap region before matching." />
                <span className="block text-gray-400">Crump 2009 trimming</span>
              </span>
            </label>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-gray-500 font-medium">Ratio</span>
              <select className="select text-[10px] py-0.5 flex-1" value={ratio} onChange={(e) => setRatio(parseInt(e.target.value, 10))}>
                {[1, 2, 3, 4, 5].map((k) => <option key={k} value={k}>1 : {k}</option>)}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-gray-500 font-medium">Seed</span>
              <input type="number" className="select text-[10px] py-0.5 flex-1"
                value={randomState} onChange={(e) => setRandomState(parseInt(e.target.value, 10))} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-gray-500 font-medium">Score model</span>
              <select className="select text-[10px] py-0.5" value={scoreMethod}
                onChange={(e) => setScoreMethod(e.target.value as "logistic" | "probit" | "gbm")}>
                <option value="logistic">Logistic ★</option>
                <option value="probit">Probit</option>
                <option value="gbm">GBM</option>
              </select>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
                Matching
                <Tip wide text="Greedy: standard caliper nearest neighbour. Optimal: Hungarian algorithm minimises total distance." />
              </span>
              <div className="flex gap-1">
                {(["greedy", "optimal"] as const).map((m) => (
                  <button key={m} onClick={() => setMatchingMethod(m)}
                    className={`px-2 py-0.5 text-[10px] rounded border transition-colors ${matchingMethod === m ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-500 hover:bg-gray-50"}`}>
                    {m === "greedy" ? "Greedy ★" : "Optimal"}
                  </button>
                ))}
              </div>
            </div>
            <div className="space-y-1">
              <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
                Exact match strata
              </span>
              <div className="max-h-20 overflow-y-auto border border-gray-200 rounded p-1 space-y-0.5">
                {allCols.filter((c) => c !== treatCol).slice(0, 100).map((c) => (
                  <label key={c} className="flex items-center gap-1 text-[10px] px-1 py-0.5 rounded hover:bg-gray-50 cursor-pointer">
                    <input type="checkbox" className="accent-indigo-500"
                      checked={exactMatch.includes(c)}
                      onChange={() => setExactMatch(exactMatch.includes(c) ? exactMatch.filter((x) => x !== c) : [...exactMatch, c])} />
                    <span className="text-gray-700 truncate">{c}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Outcome type */}
        <div className="panel space-y-2">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            Outcome type
            <Tip wide text="Binary: conditional logistic / GEE logistic on the matched cohort. Survival: stratified Cox PH." />
          </label>
          <div className="flex gap-1">
            {(["binary", "survival"] as const).map((t) => (
              <button key={t} onClick={() => { setOutcomeType(t); setResult(null); }}
                className={`flex-1 px-2 py-1 text-[11px] rounded border transition-colors ${outcomeType === t ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-500 hover:bg-gray-50"}`}>
                {t === "binary" ? "Binary" : "Survival (Cox)"}
              </button>
            ))}
          </div>
          {outcomeType === "survival" && (
            <div className="space-y-1.5">
              <div>
                <span className="text-[10px] text-gray-500 font-medium">Duration column</span>
                <select className="select w-full text-xs py-1" value={survDuration} onChange={(e) => setSurvDuration(e.target.value)}>
                  <option value="">— select —</option>
                  {allCols.filter((c) => c !== treatCol).map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <span className="text-[10px] text-gray-500 font-medium">Event column (0/1)</span>
                <select className="select w-full text-xs py-1" value={survEvent} onChange={(e) => setSurvEvent(e.target.value)}>
                  <option value="">— select —</option>
                  {allCols.filter((c) => c !== treatCol).map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            </div>
          )}
          {outcomeType === "binary" && ratio === 1 && outcomeCol && (
            <label className="flex items-start gap-2 cursor-pointer pt-1 border-t border-gray-100">
              <input type="checkbox" className="accent-indigo-500 mt-0.5"
                checked={computeRosenbaum} onChange={(e) => setComputeRosenbaum(e.target.checked)} />
              <span className="text-[10px] text-gray-600 leading-tight">
                <span className="font-medium">Rosenbaum bounds</span>
                <Tip wide text="Sensitivity analysis to unmeasured confounding. Computes the critical Γ — hidden bias size nullifying treatment p < 0.05." />
                <span className="block text-gray-400">1:1 binary only · Γ up to {rosenbaumGammaMax.toFixed(1)}</span>
              </span>
            </label>
          )}
        </div>

        {/* Run */}
        <button
          className="btn-primary w-full py-3 text-sm font-semibold flex items-center justify-center gap-2"
          onClick={run} disabled={loading || covariates.length === 0}>
          {loading
            ? <><span className="animate-spin inline-block">⏳</span> Matching…</>
            : <><span>🔗</span> Run PSM</>}
        </button>
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-3 py-2 text-xs text-red-600">{error}</div>
        )}
      </div>

      {/* ── Main content ─────────────────────────────────────────────────── */}
      <div className="flex-1 min-w-0 overflow-y-auto space-y-4">
        {result ? (
          <div
            className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_var(--right-col)] gap-4 auto-rows-min items-start xl:grid-flow-dense relative"
            style={{ "--right-col": `${rightColW}px` } as React.CSSProperties}
          >
            <div
              role="separator"
              aria-orientation="vertical"
              onPointerDown={onResizeStart}
              onDoubleClick={onResizeReset}
              className="hidden xl:block absolute top-0 bottom-0 w-1.5 rounded-full bg-gray-300/60 hover:bg-indigo-400/80 cursor-col-resize z-20 transition-colors"
              style={{ right: `calc(${rightColW}px + 5px)` }}
            />
            {/* Summary banner */}
            <div className={`panel border-2 xl:col-start-2 ${result.balance_achieved ? "border-emerald-300 bg-emerald-50" : "border-amber-300 bg-amber-50"}`}>
              <div className="flex items-start gap-3">
                <span className="text-2xl flex-shrink-0">{result.balance_achieved ? "✅" : "⚠️"}</span>
                <div className="flex-1">
                  <p className={`font-bold text-sm ${result.balance_achieved ? "text-emerald-800" : "text-amber-800"}`}>
                    {result.balance_achieved
                      ? "Balance achieved — all SMDs < 0.10. Publication-ready."
                      : "Partial balance — some SMDs ≥ 0.10. Consider widening caliper or adjusting covariates."}
                  </p>
                  <p className="text-xs text-gray-600 mt-1">
                    Matched {result.n_matched_pairs} treated : {result.n_matched_controls} control pairs
                    ({result.n_unmatched} treated patients unmatched and excluded).
                  </p>
                  <p className="text-[11px] text-gray-500 mt-1">
                    Caliper = {result.caliper_used?.toFixed(4)} on the <b>{result.caliper_scale ?? "logit"}</b> scale ({caliper} × SD = {result.caliper_used?.toFixed(4)}).
                    {result.n_trimmed_common_support > 0 && (
                      <> · {result.n_trimmed_common_support} units trimmed (common support [{result.common_support?.lo?.toFixed(3)}, {result.common_support?.hi?.toFixed(3)}]).</>
                    )}
                  </p>
                </div>
              </div>

              {/* Key stats */}
              <div className="grid grid-cols-5 gap-2 mt-3">
                {[
                  { label: "Total N", val: result.n_total },
                  { label: "Treated", val: result.n_treated },
                  { label: "Controls", val: result.n_control },
                  { label: "Matched Pairs", val: result.n_matched_pairs, highlight: true },
                  { label: "Unmatched", val: result.n_unmatched, warn: result.n_unmatched > 0 },
                ].map(({ label, val, highlight, warn }) => (
                  <div key={label} className={`rounded-lg px-2 py-2 text-center border ${
                    highlight ? "bg-indigo-50 border-indigo-200" :
                    warn && val > 0 ? "bg-amber-50 border-amber-200" :
                    "bg-white border-gray-200"}`}>
                    <p className="text-[9px] text-gray-400 uppercase tracking-wide">{label}</p>
                    <p className={`text-lg font-bold ${highlight ? "text-indigo-700" : warn && val > 0 ? "text-amber-600" : "text-gray-800"}`}>
                      {val}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Matched Cohort Actions */}
            <div className="panel border border-indigo-200 bg-indigo-50/20 xl:col-start-2 space-y-3">
              <div className="flex items-center gap-2">
                <span className="text-lg">📋</span>
                <h4 className="text-sm font-bold text-indigo-900">Matched Cohort Actions</h4>
              </div>
              <p className="text-xs text-gray-600">
                You can download the balanced matched patient cohort directly, or load it as the active dataset in uSTAT to run any other analysis (e.g. t-tests, survival curves, factor analysis).
              </p>
              <div className="flex flex-wrap gap-2 pt-1">
                <button
                  onClick={async () => {
                    if (!result.matched_session_id) return;
                    try {
                      setLoading(true);
                      const res = await getSessionInfo(result.matched_session_id);
                      setOriginalSession(session);
                      setSession(res.data);
                      // Switch to data tab so the user sees the new matched cohort patient list
                      useStore.getState().setActiveTab("data");
                      alert("Successfully loaded matched cohort! The entire app is now filtered and updated to the matched sample.");
                    } catch (e: unknown) {
                      const detail = (e as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
                      alert("Failed to load matched cohort: " + (typeof detail === "string" ? detail : (e instanceof Error ? e.message : String(e))));
                    } finally {
                      setLoading(false);
                    }
                  }}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm flex items-center gap-1.5 transition-colors cursor-pointer"
                >
                  🔍 View & Analyze Matched Cohort in App
                </button>
                <button
                  onClick={() => {
                    if (!result.matched_session_id) return;
                    exportDataset(
                      { session_id: result.matched_session_id, filename: "psm_matched_cohort" },
                      session.columns.concat({ name: "match_set_id", kind: "categorical", dtype: "object" }),
                      "csv"
                    );
                  }}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 flex items-center gap-1.5 transition-colors cursor-pointer"
                >
                  📥 Export as CSV
                </button>
                <button
                  onClick={() => {
                    if (!result.matched_session_id) return;
                    exportDataset(
                      { session_id: result.matched_session_id, filename: "psm_matched_cohort" },
                      session.columns.concat({ name: "match_set_id", kind: "categorical", dtype: "object" }),
                      "xlsx"
                    );
                  }}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 flex items-center gap-1.5 transition-colors cursor-pointer"
                >
                  📊 Export as Excel (.xlsx)
                </button>
                <button
                  onClick={() => {
                    if (!result.matched_session_id) return;
                    exportDataset(
                      { session_id: result.matched_session_id, filename: "psm_matched_cohort" },
                      session.columns.concat({ name: "match_set_id", kind: "categorical", dtype: "object" }),
                      "sav"
                    );
                  }}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 flex items-center gap-1.5 transition-colors cursor-pointer"
                >
                  💿 Export as SPSS (.sav)
                </button>
              </div>
            </div>

            {/* Love Plot */}
            <div className="panel space-y-3 xl:col-start-1">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div>
                  <h3 className="text-sm font-bold text-gray-800">Love Plot: Covariate Balance</h3>
                  <p className="text-[10px] text-gray-400">
                    Thomas Love plot. Publication-required visual proof of balance. All matched points (blue ●) must lie left of the threshold.
                  </p>
                </div>
                <div className="flex gap-3">
                  {[
                    { label: "Avg Unmatched SMD", val: result.avg_smd_before, color: "text-red-600" },
                    { label: "Avg Matched SMD", val: result.avg_smd_after, color: "text-emerald-600" },
                    { label: "Reduction", val: `${result.reduction_pct}%`, color: "text-indigo-600" },
                  ].map(({ label, val, color }) => (
                    <div key={label} className="text-center bg-gray-50 border border-gray-200 rounded-lg px-3 py-1.5">
                      <p className="text-[9px] text-gray-400 uppercase tracking-wide">{label}</p>
                      <p className={`text-base font-bold font-mono ${color}`}>
                        {typeof val === "number" ? val.toFixed(3) : val}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              <LovePlot
                smdBefore={result.smd_before}
                smdAfter={result.smd_after}
                threshold={threshold}
                showConnectors={showConnectors}
                showGrid={showGrid}
              />

              <div className="flex flex-wrap items-center gap-6 pt-2 border-t border-gray-100">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500">Balance Threshold</span>
                  <input type="range" min="0.05" max="0.25" step="0.01"
                    className="w-28 accent-indigo-500"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))} />
                  <span className="font-mono text-sm font-bold text-indigo-700 w-10">{threshold.toFixed(2)}</span>
                </div>
                <label className="flex items-center gap-2 cursor-pointer">
                  <span className="text-xs text-gray-500">Show Connectors</span>
                  <div
                    className={`w-9 h-5 rounded-full transition-colors cursor-pointer ${showConnectors ? "bg-indigo-600" : "bg-gray-300"}`}
                    onClick={() => setShowConnectors((v) => !v)}>
                    <div className={`w-4 h-4 bg-white rounded-full shadow mt-0.5 transition-transform ${showConnectors ? "translate-x-4" : "translate-x-0.5"}`} />
                  </div>
                </label>
                <div className="flex items-center gap-3 text-xs ml-auto">
                  <span className="flex items-center gap-1"><span className="w-3 h-3 bg-red-400 rounded-sm inline-block" /> Unmatched</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 bg-blue-500 rounded-full inline-block" /> Matched</span>
                  <span className="flex items-center gap-1"><span className="border-l-2 border-dashed border-red-500 h-3 inline-block" /> Threshold</span>
                </div>
              </div>
            </div>

            {/* SMD balance table */}
            <div className="panel space-y-2 xl:col-start-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-gray-700">
                  SMD Balance Table
                  <Tip wide text="Standardized Mean Difference measures imbalance. Golden standard: SMDs after matching must be < 0.10 (Austin, 2009)." />
                </h3>
                <ResultExporter
                  title="PSM_SMD_Balance"
                  headers={smdExportHeaders}
                  rows={smdExportRows}
                />
              </div>
              <div className="overflow-auto rounded-lg border border-gray-200">
                <table className="w-full text-xs border-collapse">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                      <th className="text-left px-3 py-2 font-medium">Covariate</th>
                      <th className="text-right px-3 py-2 font-medium">SMD Before</th>
                      <th className="text-right px-3 py-2 font-medium">SMD After</th>
                      <th className="text-right px-3 py-2 font-medium" title="Rubin's variance ratio. Target 0.5–2.0.">Var Ratio</th>
                      <th className="text-right px-3 py-2 font-medium" title="Two-sample KS test p-value.">KS <i>p</i> (after)</th>
                      <th className="text-right px-3 py-2 font-medium">Reduction</th>
                      <th className="text-center px-3 py-2 font-medium">Balanced</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(result.smd_before).map((cov) => {
                      const before = result.smd_before[cov];
                      const after = result.smd_after[cov];
                      const reduction = before > 0 ? ((before - after) / before * 100).toFixed(1) : "—";
                      const balanced = after < threshold;
                      const vr = result.variance_ratio_after?.[cov] as number | null | undefined;
                      const vrBefore = result.variance_ratio_before?.[cov] as number | null | undefined;
                      const ksAfter = result.ks_p_after?.[cov] as number | null | undefined;
                      const vrOk = vr == null || (vr >= 0.5 && vr <= 2.0);
                      return (
                        <tr key={cov} className="border-b border-gray-100 hover:bg-gray-50">
                          <td className="px-3 py-1.5 font-mono text-gray-800">{cov}</td>
                          <td className={`px-3 py-1.5 text-right font-mono ${smdColor(before)}`}>{before.toFixed(4)}</td>
                          <td className={`px-3 py-1.5 text-right font-mono font-semibold ${smdColor(after)}`}>{after.toFixed(4)}</td>
                          <td className={`px-3 py-1.5 text-right font-mono ${vrOk ? "text-gray-600" : "text-amber-600 font-semibold"}`}
                              title={vrBefore != null ? `Before: ${vrBefore.toFixed(3)}` : ""}>
                            {vr != null ? vr.toFixed(3) : "—"}
                          </td>
                          <td className={`px-3 py-1.5 text-right font-mono ${ksAfter != null && ksAfter < 0.05 ? "text-red-600 font-semibold" : "text-gray-500"}`}>
                            {ksAfter != null ? fmtP(ksAfter) : "—"}
                          </td>
                          <td className="px-3 py-1.5 text-right text-gray-500">{reduction}%</td>
                          <td className="px-3 py-1.5 text-center">
                            <span className={`inline-block text-[9px] font-semibold border rounded-full px-1.5 py-0.5 ${
                              balanced ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-red-50 text-red-600 border-red-200"
                            }`}>
                              {balanced ? "✓" : "✗"}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                  <tfoot className="bg-gray-50 border-t-2 border-gray-200">
                    <tr>
                      <td className="px-3 py-1.5 font-semibold text-gray-700">Average</td>
                      <td className={`px-3 py-1.5 text-right font-mono font-semibold ${smdColor(result.avg_smd_before)}`}>{result.avg_smd_before.toFixed(4)}</td>
                      <td className={`px-3 py-1.5 text-right font-mono font-semibold ${smdColor(result.avg_smd_after)}`}>{result.avg_smd_after.toFixed(4)}</td>
                      <td className="px-3 py-1.5 text-gray-400" colSpan={2}>—</td>
                      <td className="px-3 py-1.5 text-right text-indigo-600 font-semibold">{result.reduction_pct}%</td>
                      <td className="px-3 py-1.5 text-center">
                        <span className={`inline-block text-[9px] font-semibold border rounded-full px-1.5 py-0.5 ${
                          result.balance_achieved ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-amber-50 text-amber-700 border-amber-200"
                        }`}>
                          {result.balance_achieved ? "All ✓" : "Partial"}
                        </span>
                      </td>
                    </tr>
                  </tfoot>
                </table>
              </div>
            </div>

            {/* PS Overlap */}
            {result.ps_distribution && (
              <div className="panel space-y-2 xl:col-start-1">
                <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
                  Propensity Score Overlap
                </h3>
                <PSOverlapPlot psDist={result.ps_distribution} showGrid={showGrid} />
              </div>
            )}

            {/* Outcome analysis */}
            {result.outcome_result && !result.outcome_result.error && (() => {
              const t: string = result.outcome_result.type ?? "";
              const isCoxKind = t.startsWith("stratified_cox") || t.startsWith("weighted_cox");
              const isClogit = t.startsWith("conditional_logistic");
              return (
              <div className="panel space-y-3 xl:col-start-2">
                <h3 className="text-sm font-semibold text-gray-700">
                  Outcome Analysis — Matched Cohort
                  <span className="ml-2 text-[10px] font-normal text-indigo-600 bg-indigo-50 border border-indigo-200 rounded-full px-2 py-0.5">
                    {result.outcome_result.model}
                  </span>
                </h3>
                <div className="grid grid-cols-3 gap-2">
                  {isCoxKind ? (
                    [
                      ["n (matched)", result.outcome_result.n],
                      ["Events", result.outcome_result.n_events],
                      ["C-index", result.outcome_result.concordance?.toFixed(3)],
                    ].map(([k, v]) => (
                      <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-sm">{v}</p>
                      </div>
                    ))
                  ) : isClogit ? (
                    [
                      ["n (in fit)", result.outcome_result.n],
                      ["Informative sets", result.outcome_result.n_informative_sets],
                      ["Concordant sets", result.outcome_result.n_uninformative_sets],
                    ].map(([k, v]) => (
                      <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-sm">{v}</p>
                      </div>
                    ))
                  ) : (
                    [
                      ["n (matched)", result.outcome_result.n],
                      ["AIC", result.outcome_result.aic?.toFixed(2)],
                      ["BIC", result.outcome_result.bic?.toFixed(2)],
                    ].map(([k, v]) => (
                      <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-sm">{v}</p>
                      </div>
                    ))
                  )}
                </div>
                <div className="overflow-auto rounded-lg border border-gray-200">
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                        {(isCoxKind
                          ? ["Variable", "HR", "95% CI", "β", "SE", "z", "p"]
                          : ["Variable", "OR", "95% CI", "β", "SE", "z", "p"]
                        ).map((h: string) => (
                          <th key={h} className="px-2 py-2 text-left font-medium">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.outcome_result.coefficients.map((c) => {
                        const effect = isCoxKind ? c.hr : c.or;
                        const lo = isCoxKind ? c.hr_low : c.or_low;
                        const hi = isCoxKind ? c.hr_high : c.or_high;
                        return (
                          <tr key={c.variable} className={`border-b border-gray-100 ${c.p < 0.05 ? "hover:bg-indigo-50/30" : "hover:bg-gray-50"}`}>
                            <td className="px-2 py-1.5 font-mono text-gray-800">{c.variable}</td>
                            <td className={`px-2 py-1.5 font-mono font-semibold ${c.p < 0.05 ? "text-indigo-700" : "text-gray-600"}`}>{effect?.toFixed(3)}</td>
                            <td className="px-2 py-1.5 font-mono text-gray-500">[{lo?.toFixed(3)}, {hi?.toFixed(3)}]</td>
                            <td className="px-2 py-1.5 font-mono text-gray-600">{c.estimate?.toFixed(4)}</td>
                            <td className="px-2 py-1.5 font-mono text-gray-500">{c.se?.toFixed(4)}</td>
                            <td className="px-2 py-1.5 font-mono text-gray-500">{c.z?.toFixed(3)}</td>
                            <td className="px-2 py-1.5">
                              <span className={`inline-block font-mono px-1.5 py-0.5 rounded text-[10px] ${
                                c.p < 0.05 ? "bg-indigo-100 text-indigo-700 font-semibold" : "text-gray-400"
                              }`}>{fmtP(c.p)}</span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="text-[10px] text-gray-400 whitespace-pre-line">
                  {result.outcome_result.method_note ?? (
                    isCoxKind
                      ? "HR = Hazard Ratio (exp(β))."
                      : "OR = Odds Ratio (exp(β))."
                  )}
                </p>
              </div>
              );
            })()}

            {result.outcome_result?.error && (
              <div className="panel bg-red-50 border border-red-200 text-xs text-red-600 xl:col-start-2">
                Outcome analysis failed: {result.outcome_result.error}
              </div>
            )}

            {/* Rosenbaum bounds */}
            {result.rosenbaum && (
              <div className="panel space-y-2 xl:col-start-2">
                <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
                  Rosenbaum Bounds — Sensitivity to Hidden Bias
                </h3>
                {result.rosenbaum.applicable === false ? (
                  <p className="text-xs text-gray-500">{result.rosenbaum.reason}</p>
                ) : (
                  <>
                    <div className="grid grid-cols-4 gap-2">
                      <div className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">Discordant pairs</p>
                        <p className="font-semibold text-gray-800 text-sm">{result.rosenbaum.discordant_pairs} ({result.rosenbaum.b} / {result.rosenbaum.c})</p>
                      </div>
                      <div className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">p (Γ = 1)</p>
                        <p className="font-semibold text-gray-800 text-sm">{fmtP(result.rosenbaum.p_unbiased)}</p>
                      </div>
                      <div className={`rounded-lg p-2 text-center border ${result.rosenbaum.critical_gamma != null && result.rosenbaum.critical_gamma > 1.5 ? "bg-emerald-50 border-emerald-200" : "bg-amber-50 border-amber-200"}`}>
                        <p className="text-[10px] text-gray-500">Critical Γ</p>
                        <p className={`font-semibold text-sm ${result.rosenbaum.critical_gamma != null && result.rosenbaum.critical_gamma > 1.5 ? "text-emerald-700" : "text-amber-700"}`}>
                          {result.rosenbaum.critical_gamma != null ? result.rosenbaum.critical_gamma.toFixed(2) : `> ${result.rosenbaum.gamma_max}`}
                        </p>
                      </div>
                      <div className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">α</p>
                        <p className="font-semibold text-gray-800 text-sm">{result.rosenbaum.alpha}</p>
                      </div>
                    </div>
                    <div className="overflow-auto rounded-lg border border-gray-200 max-h-48">
                      <table className="w-full text-[11px] border-collapse">
                        <thead className="sticky top-0 bg-gray-50 border-b border-gray-200 text-gray-500">
                          <tr>
                            <th className="text-left px-2 py-1">Γ</th>
                            <th className="text-right px-2 py-1">Upper-bound one-sided p</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.rosenbaum.curve?.map((r) => (
                            <tr key={r.gamma} className={r.p_upper > (result.rosenbaum?.alpha ?? 0.05) ? "bg-amber-50" : ""}>
                              <td className="px-2 py-0.5 font-mono">{r.gamma.toFixed(2)}</td>
                              <td className="px-2 py-0.5 text-right font-mono">{r.p_upper.toFixed(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="text-[10px] text-gray-400">
                      Reference: Rosenbaum PR (2002). <em>Observational Studies</em>.
                    </p>
                  </>
                )}
              </div>
            )}

            {result.matching_warning && (
              <div className="panel bg-amber-50 border border-amber-200 text-xs text-amber-800 xl:col-start-2">
                {result.matching_warning}
              </div>
            )}
          </div>
        ) : (
          /* Empty state */
          <div className="space-y-4">
            <div className="panel bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-100">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-3xl">🧬</span>
                <div>
                  <h2 className="text-base font-bold text-indigo-900">Propensity Score Matching</h2>
                  <p className="text-xs text-indigo-600">Advanced Epidemiology — Observational Causal Inference</p>
                </div>
              </div>
              <p className="text-sm text-gray-700 leading-relaxed">
                PSM mimics a Randomized Controlled Trial from observational data by balancing baseline characteristics between treated and control groups. It is the accepted gold standard for non-randomized cardiology studies.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {[
                { icon: "🎯", title: "Step 1 — Propensity Score",
                  body: "Logistic regression (Treatment ~ Covariates) estimates each patient's probability of receiving treatment given baseline profile. This is their Propensity Score (PS)." },
                { icon: "🔗", title: "Step 2 — Nearest-Neighbour Matching",
                  body: "Each treated patient is matched to the control with the closest PS. Caliper = 0.2 × SD(PS) — the medical standard prevents poor matches from degrading balance." },
                { icon: "📊", title: "Step 3 — Love Plot (SMD)",
                  body: "Standardized Mean Differences are calculated before and after matching for every covariate. ALL SMDs must be < 0.10 for the match to be publication-ready." },
                { icon: "🏥", title: "Step 4 — Outcome Analysis",
                  body: "Logistic regression or Cox regression is run on the balanced matched cohort. Treatment effects estimated here are free from measured confounding." },
              ].map(({ icon, title, body }) => (
                <div key={title} className="panel flex gap-3 border-t-4 border-indigo-200">
                  <span className="text-2xl flex-shrink-0">{icon}</span>
                  <div>
                    <p className="text-xs font-bold text-gray-800 mb-1">{title}</p>
                    <p className="text-xs text-gray-500 leading-relaxed">{body}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="panel bg-amber-50 border border-amber-200 space-y-1.5">
              <p className="text-xs font-bold text-amber-800">⚠ Key Assumptions</p>
              {[
                ["No unmeasured confounders", "PSM only balances variables you include. Hidden confounders (not in your dataset) cannot be removed."],
                ["Binary treatment", "The treatment variable must be 0/1. Continuous or multi-level treatments require other methods."],
                ["Common support", "Treated and control propensity score distributions must overlap substantially. No overlap = no valid matches."],
              ].map(([title, body]) => (
                <div key={title} className="flex gap-1.5 text-[10px] text-amber-700">
                  <span className="flex-shrink-0 font-semibold">{title}:</span>
                  <span>{body}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
