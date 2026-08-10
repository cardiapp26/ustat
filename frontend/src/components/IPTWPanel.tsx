/**
 * IPTWPanel — Inverse Probability of Treatment Weighting
 *
 * Pipeline:
 * 1. Propensity Score estimation (Logistic regression / GBM)
 * 2. Calculate weights (ATE, ATT, or Overlap)
 * 3. SMD balance assessment and Love Plot
 * 4. Outcome analysis (Weighted GLM or Weighted Cox PH)
 */
import { useState, useRef, useMemo } from "react";
import { useStore, paletteOf, analysisCols, type Session } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { runIPTW, getSessionInfo } from "../api";
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
  concordance?: number;
  aic?: number;
  bic?: number;
  coefficients: OutcomeCoefficient[];
  method_note?: string;
  error?: string;
}

interface IPTWResult {
  balance_achieved: boolean;
  estimand?: string;
  stabilize?: boolean;
  se_method?: string;
  score_method?: string;
  n_total: number;
  n_treated: number;
  n_control: number;
  n_trimmed_common_support: number;
  matched_session_id?: string;
  weight_truncation?: { n_trimmed?: number };
  weight_summary?: {
    ess_treated?: number;
    ess_control?: number;
    min?: number;
    median?: number;
    max?: number;
  };
  weight_distribution?: { treated: number[]; control: number[] };
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
}

const smdColor = (smd: number | undefined | null) =>
  smd == null || !Number.isFinite(smd) ? "text-gray-400"
    : smd < 0.10 ? "text-emerald-600" : smd < 0.20 ? "text-amber-500" : "text-red-500";

// Geometry only. Background, font and colourway follow the global chart theme
// at render; hard-coded here they made the theme picker a no-op on this panel.
const PLOT_BASE = {
  margin: { t: 30, r: 24, b: 56, l: 130 },
};

/** The theme-driven half of the layout. Subscribed, so the figure repaints
 *  the moment the palette or font changes rather than at the next render. */
function useThemedBase(): Record<string, unknown> {
  const t = useStore((s) => s.plotTheme);
  return {
    ...PLOT_BASE,
    paper_bgcolor: "transparent",
    plot_bgcolor: t.plotBg,
    font: { family: t.fontFamily, color: "#374151", size: t.fontSize },
    colorway: paletteOf(t),
  };
}

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
  const themedBase = useThemedBase();
  const plotRef = useRef<PlotCaptureHandle | null>(null);
  const covariates = Object.keys(smdBefore).reverse(); // bottom-to-top

  const xMax = Math.max(0.4, ...Object.values(smdBefore), ...Object.values(smdAfter)) * 1.15;

  const traces: PlotData[] = [
    {
      type: "scatter",
      mode: "markers",
      name: "Unweighted cohort",
      x: covariates.map((c) => smdBefore[c]),
      y: covariates,
      marker: { symbol: "square", size: 11, color: "#ef4444" },
      hovertemplate: "<b>%{y}</b><br>SMD (before) = %{x:.4f}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "markers",
      name: "Weighted cohort",
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
      storageKey="iptw:love"
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
  const themedBase = useThemedBase();
  const plotRef = useRef<PlotCaptureHandle | null>(null);
  return (
    <TitledPlot
      plotRefOut={plotRef}
      storageKey="iptw:ps-overlap"
      data={[
        {
          type: "histogram",
          name: "Treated",
          x: psDist.treated_unmatched,
          opacity: 0.55,
          marker: { color: "#ef4444" },
          nbinsx: 25,
          hovertemplate: "PS: %{x:.3f}<br>Count: %{y}<extra></extra>",
        },
        {
          type: "histogram",
          name: "Control",
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
      } as PlotLayout}
      config={{ responsive: true, displaylogo: false, displayModeBar: false }}
      defaultTitle="Propensity Score Overlap"
      defaultSubtitle=""
      defaultXAxis="Propensity Score"
      defaultYAxis="Count"
    />
  );
}

export default function IPTWPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <IPTWPanelBody session={session} />;
}

function IPTWPanelBody({ session }: { session: Session }) {
  const themedBase = useThemedBase();
  const showGrid = useStore((s) => s.showGrid);
  const setSession = useStore((s) => s.setSession);
  const setOriginalSession = useStore((s) => s.setOriginalSession);
  const { w: rightColW, onDragStart: onResizeStart, onReset: onResizeReset } =
    useResizableRightCol("IPTWPanel.result", 480);

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

  // Form State
  const [treatCol, setTreatCol] = usePersistedPanelState<string>("iptw", "treatCol", binaryCols[0] ?? allCols[0] ?? "");
  const [outcomeCol, setOutcomeCol] = usePersistedPanelState<string>("iptw", "outcomeCol", "");
  const [covariates, setCovariates] = usePersistedPanelState<string[]>("iptw", "covariates", []);
  const [trimCommonSupport, setTrimCommonSupport] = usePersistedPanelState<boolean>("iptw", "trimCommonSupport", false);
  const [randomState, setRandomState] = usePersistedPanelState<number>("iptw", "randomState", 42);
  const [scoreMethod, setScoreMethod] = usePersistedPanelState<"logistic" | "probit" | "gbm">("iptw", "scoreMethod", "logistic");
  const [outcomeType, setOutcomeType] = usePersistedPanelState<"binary" | "survival">("iptw", "outcomeType", "binary");
  const [survDuration, setSurvDuration] = usePersistedPanelState<string>("iptw", "survDuration", "");
  const [survEvent, setSurvEvent] = usePersistedPanelState<string>("iptw", "survEvent", "");
  const [covFilter, setCovFilter] = useState("");

  // IPTW options
  const [estimand, setEstimand] = usePersistedPanelState<"ate" | "att" | "overlap">("iptw", "estimand", "ate");
  const [stabilize, setStabilize] = usePersistedPanelState<boolean>("iptw", "stabilize", true);
  const [weightTruncation, setWeightTruncation] = usePersistedPanelState<"percentile" | "hard" | "none">("iptw", "weightTruncation", "percentile");
  const [weightTruncLo, setWeightTruncLo] = usePersistedPanelState<number>("iptw", "weightTruncLo", 0.01);
  const [weightTruncHi, setWeightTruncHi] = usePersistedPanelState<number>("iptw", "weightTruncHi", 0.99);
  const [weightTruncMax, setWeightTruncMax] = usePersistedPanelState<number>("iptw", "weightTruncMax", 10);
  const [seMethod, setSeMethod] = usePersistedPanelState<"robust" | "bootstrap">("iptw", "seMethod", "robust");
  const [bootstrapReps, setBootstrapReps] = usePersistedPanelState<number>("iptw", "bootstrapReps", 500);

  // Result & UI
  const [result, setResult] = useState<IPTWResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.10);
  const [showConnectors, setShowConnectors] = useState(true);
  const weightDistRef = useRef<PlotCaptureHandle | null>(null);

  const toggleCov = (c: string) =>
    setCovariates(covariates.includes(c) ? covariates.filter((x) => x !== c) : [...covariates, c]);

  const run = async () => {
    if (covariates.length === 0) { setError("Select at least one covariate"); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await runIPTW({
        session_id: session.session_id,
        treatment_col: treatCol,
        covariates,
        outcome_col: outcomeType === "binary" ? (outcomeCol || undefined) : undefined,
        imputation: undefined,
        random_state: Number.isFinite(randomState) ? randomState : undefined,
        score_method: scoreMethod,
        estimand,
        stabilize,
        trim_common_support: trimCommonSupport,
        weight_truncation: weightTruncation,
        weight_truncation_lo: weightTruncLo,
        weight_truncation_hi: weightTruncHi,
        weight_truncation_max: weightTruncMax,
        outcome_type: outcomeType,
        survival_duration_col: outcomeType === "survival" ? (survDuration || undefined) : undefined,
        survival_event_col: outcomeType === "survival" ? (survEvent || undefined) : undefined,
        se_method: seMethod,
        bootstrap_reps: seMethod === "bootstrap" ? bootstrapReps : undefined,
      });
      setResult(res.data as IPTWResult);
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
      setError(typeof msg === "string" ? msg : (e instanceof Error ? e.message : "IPTW failed"));
    } finally { setLoading(false); }
  };

  // Older backend builds returned smd_before/smd_after as scalar averages.
  // Normalise to per-covariate maps so the table/Love plot never crash on
  // a stale API response.
  const smdBeforeMap: Record<string, number> =
    result && typeof result.smd_before === "object" && result.smd_before !== null ? result.smd_before : {};
  const smdAfterMap: Record<string, number> =
    result && typeof result.smd_after === "object" && result.smd_after !== null ? result.smd_after : {};
  const fmtNum = (v: number | undefined | null, d: number) =>
    typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—";

  const smdExportHeaders = ["Covariate", "SMD Before", "SMD After", "Reduction %", "Balanced (<0.10)"];
  const smdExportRows = Object.keys(smdBeforeMap).map((cov) => [
    cov,
    fmtNum(smdBeforeMap[cov], 4),
    fmtNum(smdAfterMap[cov], 4),
    smdBeforeMap[cov] > 0
      ? (((smdBeforeMap[cov] - (smdAfterMap[cov] ?? 0)) / smdBeforeMap[cov]) * 100).toFixed(1) + "%"
      : "—",
    (smdAfterMap[cov] ?? Infinity) < 0.10 ? "Yes" : "No",
  ]);

  const availableCovs = allCols.filter(
    (c) => c !== treatCol && c !== outcomeCol &&
           c.toLowerCase().includes(covFilter.toLowerCase())
  );

  return (
    <div className="flex gap-4 min-h-0">
      {/* ── Left sidebar ─────────────────────────────────────────────────── */}
      <div className="w-64 flex-shrink-0 space-y-3 overflow-y-auto">
        {/* Header */}
        <div className="panel bg-gradient-to-br from-indigo-50 to-emerald-50 border-indigo-200 space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xl">⚖️</span>
            <h2 className="text-sm font-bold text-indigo-800">
              Inverse Probability of Treatment Weighting (IPTW)
            </h2>
          </div>
          <p className="text-[10px] text-indigo-600 leading-snug">
            Reweights every unit by the inverse of its propensity score. Keeps the full sample; supports ATE / ATT / overlap estimands; pairs with weighted GLM or weighted Cox.
          </p>
        </div>

        {/* IPTW options */}
        <div className="panel space-y-3">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            IPTW Options
            <Tip wide text="Choose the estimand (target population), whether to stabilise the weights, and how to truncate the upper tail. Stabilisation rescales by P(T=1) to bring weights closer to 1 and reduces SE inflation." />
          </label>

          <div className="space-y-1">
            <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
              Estimand
              <Tip wide text="ATE — average effect across the whole population. ATT — effect on those actually exposed. Overlap — concentrates inference on the overlap region; naturally bounded and robust to extreme PS." />
            </span>
            <div className="flex gap-1">
              {(["ate", "att", "overlap"] as const).map((e) => (
                <button key={e} onClick={() => setEstimand(e)}
                  className={`flex-1 px-2 py-1 text-[10px] rounded border transition-colors ${estimand === e ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-500 hover:bg-gray-50"}`}>
                  {e === "ate" ? "ATE ★" : e === "att" ? "ATT" : "Overlap"}
                </button>
              ))}
            </div>
          </div>

          <label className="flex items-start gap-2 cursor-pointer">
            <input type="checkbox" className="accent-indigo-500 mt-0.5"
              checked={stabilize} onChange={(e) => setStabilize(e.target.checked)} />
            <span className="text-[10px] text-gray-600 leading-tight">
              <span className="font-medium">Stabilised weights</span>
              <Tip wide text="Rescales weights by the marginal P(T=1) (and 1−P(T=1) for controls). Reduces extreme-weight impact without changing the point estimate." />
              <span className="block text-gray-400">Multiplies by marginal P(T=1)</span>
            </span>
          </label>

          <div className="space-y-1 pt-2 border-t border-gray-100">
            <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
              Weight truncation
              <Tip wide text="Truncating the weight distribution controls the influence of units near the PS support boundaries. Percentile (e.g. 1st/99th) is recommended. Hard cap clips at an absolute maximum." />
            </span>
            <div className="flex gap-1">
              {(["percentile", "hard", "none"] as const).map((m) => (
                <button key={m} onClick={() => setWeightTruncation(m)}
                  className={`flex-1 px-2 py-1 text-[10px] rounded border transition-colors ${weightTruncation === m ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-500 hover:bg-gray-50"}`}>
                  {m === "percentile" ? "Pctile ★" : m === "hard" ? "Hard" : "None"}
                </button>
              ))}
            </div>
            {weightTruncation === "percentile" && (
              <div className="grid grid-cols-2 gap-1.5">
                <label className="flex items-center gap-1 text-[10px] text-gray-500">
                  Lo
                  <input type="number" min="0" max="0.5" step="0.005" className="select text-[10px] py-0.5 flex-1"
                    value={weightTruncLo}
                    onChange={(e) => setWeightTruncLo(parseFloat(e.target.value))} />
                </label>
                <label className="flex items-center gap-1 text-[10px] text-gray-500">
                  Hi
                  <input type="number" min="0.5" max="1" step="0.005" className="select text-[10px] py-0.5 flex-1"
                    value={weightTruncHi}
                    onChange={(e) => setWeightTruncHi(parseFloat(e.target.value))} />
                </label>
              </div>
            )}
            {weightTruncation === "hard" && (
              <label className="flex items-center gap-1 text-[10px] text-gray-500">
                Max weight
                <input type="number" min="1" step="0.5" className="select text-[10px] py-0.5 flex-1"
                  value={weightTruncMax}
                  onChange={(e) => setWeightTruncMax(parseFloat(e.target.value))} />
              </label>
            )}
          </div>

          <div className="space-y-1 pt-2 border-t border-gray-100">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
                Score model
                <Tip wide text="Model used to estimate propensity scores. Logistic = standard parametric. Probit = probit link. GBM = gradient-boosted trees." />
              </span>
              <select className="select text-[10px] py-0.5" value={scoreMethod}
                onChange={(e) => setScoreMethod(e.target.value as "logistic" | "probit" | "gbm")}>
                <option value="logistic">Logistic ★</option>
                <option value="probit">Probit</option>
                <option value="gbm">GBM</option>
              </select>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-gray-500 font-medium">Seed</span>
              <input type="number" className="select text-[10px] py-0.5 flex-1"
                value={randomState} onChange={(e) => setRandomState(parseInt(e.target.value, 10))} />
            </div>
            <label className="flex items-start gap-2 cursor-pointer">
              <input type="checkbox" className="accent-indigo-500 mt-0.5"
                checked={trimCommonSupport} onChange={(e) => setTrimCommonSupport(e.target.checked)} />
              <span className="text-[10px] text-gray-600 leading-tight">
                <span className="font-medium">Trim to common support</span>
                <Tip wide text="Exclude treated and control units with PS outside the overlap region before weighting." />
                <span className="block text-gray-400">Crump 2009 trimming</span>
              </span>
            </label>
          </div>

          <div className="space-y-1 pt-2 border-t border-gray-100">
            <span className="text-[10px] text-gray-500 font-medium flex items-center gap-1">
              Standard error
              <Tip wide text="Robust — HC1 sandwich. Bootstrap — refit propensity scores and outcome model on resampled subjects." />
            </span>
            <div className="flex gap-1">
              {(["robust", "bootstrap"] as const).map((m) => (
                <button key={m} onClick={() => setSeMethod(m)}
                  className={`flex-1 px-2 py-1 text-[10px] rounded border transition-colors ${seMethod === m ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-500 hover:bg-gray-50"}`}>
                  {m === "robust" ? "Robust ★" : "Bootstrap"}
                </button>
              ))}
            </div>
            {seMethod === "bootstrap" && (
              <label className="flex items-center gap-1 text-[10px] text-gray-500">
                Reps
                <input type="number" min="50" max="5000" step="50" className="select text-[10px] py-0.5 flex-1"
                  value={bootstrapReps}
                  onChange={(e) => setBootstrapReps(parseInt(e.target.value, 10))} />
              </label>
            )}
          </div>
        </div>

        {/* Treatment variable */}
        <div className="panel space-y-2">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            Treatment Variable
            <Tip wide text="The binary intervention variable (0/1)." />
          </label>
          <select
            className="select w-full"
            value={treatCol}
            onChange={(e) => { setTreatCol(e.target.value); setResult(null); }}>
            {binaryCols.length > 0
              ? binaryCols.map((c) => <option key={c} value={c}>{c}</option>)
              : allCols.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>

        {/* Outcome variable */}
        <div className="panel space-y-2">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            Outcome Variable <span className="normal-case font-normal text-gray-400">(optional)</span>
            <Tip wide text="The endpoint to analyse in the weighted cohort." />
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
            <Tip wide text="Baseline characteristics that influence both treatment and outcome." />
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

        {/* Outcome type */}
        <div className="panel space-y-2">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
            Outcome Type
          </label>
          <div className="flex gap-1">
            {(["binary", "survival"] as const).map((t) => (
              <button key={t} onClick={() => { setOutcomeType(t); setResult(null); }}
                className={`flex-1 px-2 py-1 text-[11px] rounded border transition-colors ${outcomeType === t ? "bg-indigo-600 text-white border-indigo-600" : "border-gray-300 text-gray-500 hover:bg-gray-50"}`}>
                {t === "binary" ? "Binary" : "Survival"}
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
        </div>

        {/* Run */}
        <button
          className="btn-primary w-full py-3 text-sm font-semibold flex items-center justify-center gap-2"
          onClick={run} disabled={loading || covariates.length === 0}>
          {loading
            ? <><span className="animate-spin inline-block">⏳</span> Weighting…</>
            : <><span>⚖️</span> Run IPTW</>}
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
                      : "Partial balance — some SMDs ≥ 0.10. Consider trimming weights, switching to overlap weights, or adjusting covariates."}
                  </p>
                  <p className="text-xs text-gray-600 mt-1">
                    IPTW · {(result.estimand ?? "ate").toUpperCase()} weights
                    {result.stabilize ? " (stabilised)" : ""}
                    {" · "}<i>n</i> = {result.n_total} ({result.n_treated} treated, {result.n_control} control)
                    {result.n_trimmed_common_support > 0 && <> · {result.n_trimmed_common_support} trimmed (common support)</>}
                    {(result.weight_truncation?.n_trimmed ?? 0) > 0 && <> · {result.weight_truncation?.n_trimmed} weights truncated</>}
                  </p>
                  <p className="text-[11px] text-gray-500 mt-1">
                    Score = {result.score_method} · ESS treated = {result.weight_summary?.ess_treated} · ESS control = {result.weight_summary?.ess_control} · max w = {result.weight_summary?.max}
                  </p>
                </div>
              </div>

              {/* Key stats */}
              <div className="grid grid-cols-5 gap-2 mt-3">
                {[
                  { label: "Total N", val: result.n_total },
                  { label: "Treated", val: result.n_treated },
                  { label: "Controls", val: result.n_control },
                  { label: "ESS Treated", val: result.weight_summary?.ess_treated, highlight: true },
                  { label: "ESS Control", val: result.weight_summary?.ess_control, highlight: true },
                ].map(({ label, val, highlight }) => (
                  <div key={label} className={`rounded-lg px-2 py-2 text-center border ${
                    highlight ? "bg-indigo-50 border-indigo-200" : "bg-white border-gray-200"}`}>
                    <p className="text-[9px] text-gray-400 uppercase tracking-wide">{label}</p>
                    <p className={`text-lg font-bold ${highlight ? "text-indigo-700" : "text-gray-800"}`}>
                      {val}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Weighted Cohort Actions */}
            <div className="panel border border-indigo-200 bg-indigo-50/20 xl:col-start-2 space-y-3">
              <div className="flex items-center gap-2">
                <span className="text-lg">📋</span>
                <h4 className="text-sm font-bold text-indigo-900">Weighted Cohort Actions</h4>
              </div>
              <p className="text-xs text-gray-600">
                You can download the balanced weighted patient cohort directly, or load it as the active dataset in uSTAT to run any other analysis (e.g. t-tests, survival curves, factor analysis).
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
                      alert("Successfully loaded weighted cohort! The entire app is now filtered and updated to the weighted sample with IPTW weights.");
                    } catch (e: unknown) {
                      const detail = (e as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
                      alert("Failed to load weighted cohort: " + (typeof detail === "string" ? detail : (e instanceof Error ? e.message : String(e))));
                    } finally {
                      setLoading(false);
                    }
                  }}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm flex items-center gap-1.5 transition-colors cursor-pointer"
                >
                  🔍 View & Analyze Weighted Cohort in App
                </button>
                <button
                  onClick={() => {
                    if (!result.matched_session_id) return;
                    exportDataset(
                      { session_id: result.matched_session_id, filename: "iptw_weighted_cohort" },
                      session.columns.concat({ name: "iptw_weight", kind: "numeric", dtype: "float64" }),
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
                      { session_id: result.matched_session_id, filename: "iptw_weighted_cohort" },
                      session.columns.concat({ name: "iptw_weight", kind: "numeric", dtype: "float64" }),
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
                      { session_id: result.matched_session_id, filename: "iptw_weighted_cohort" },
                      session.columns.concat({ name: "iptw_weight", kind: "numeric", dtype: "float64" }),
                      "sav"
                    );
                  }}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 flex items-center gap-1.5 transition-colors cursor-pointer"
                >
                  💿 Export as SPSS (.sav)
                </button>
              </div>
            </div>

            {/* Weight distribution */}
            {result.weight_distribution && (
              <div className="panel space-y-2 xl:col-start-1">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-gray-800">Weight Distribution</h3>
                    <p className="text-[10px] text-gray-400">
                      Per-unit IPTW weights by treatment group. Wider tails or large maxima indicate poor PS support.
                    </p>
                  </div>
                  <div className="text-[10px] text-gray-500 text-right">
                    min {result.weight_summary?.min} · median {result.weight_summary?.median} · max {result.weight_summary?.max}
                  </div>
                </div>
                <TitledPlot
                  plotRefOut={weightDistRef}
                  storageKey="iptw:weight-dist"
                  data={[
                    {
                      type: "histogram", name: "Treated",
                      x: result.weight_distribution.treated,
                      opacity: 0.65, marker: { color: "#6366f1" },
                    },
                    {
                      type: "histogram", name: "Control",
                      x: result.weight_distribution.control,
                      opacity: 0.65, marker: { color: "#10b981" },
                    },
                  ] as PlotData[]}
                  layout={{
                    ...themedBase,
                    barmode: "overlay",
                    height: 220, autosize: true,
                    margin: { t: 24, r: 20, b: 40, l: 60 },
                    xaxis: { title: { text: "IPTW weight" } },
                    yaxis: { title: { text: "Count" } },
                    legend: { font: { size: 10 } },
                  }}
                  config={{ responsive: true, displaylogo: false }}
                  defaultTitle="Weight Distribution"
                  defaultSubtitle=""
                  defaultXAxis="IPTW weight"
                  defaultYAxis="Count"
                />
              </div>
            )}

            {/* Love Plot */}
            <div className="panel space-y-3 xl:col-start-1">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div>
                  <h3 className="text-sm font-bold text-gray-800">Love Plot: Covariate Balance</h3>
                  <p className="text-[10px] text-gray-400">
                    Publication-standard balance visualization. All weighted points (blue ●) must lie left of the threshold line.
                  </p>
                </div>
                <div className="flex gap-3">
                  {[
                    { label: "Avg Unweighted SMD", val: result.avg_smd_before, color: "text-red-600" },
                    { label: "Avg Weighted SMD", val: result.avg_smd_after, color: "text-emerald-600" },
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
                smdBefore={smdBeforeMap}
                smdAfter={smdAfterMap}
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
                  <span className="flex items-center gap-1"><span className="w-3 h-3 bg-red-400 rounded-sm inline-block" /> Unweighted</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 bg-blue-500 rounded-full inline-block" /> Weighted</span>
                  <span className="flex items-center gap-1"><span className="border-l-2 border-dashed border-red-500 h-3 inline-block" /> Threshold</span>
                </div>
              </div>
            </div>

            {/* SMD balance table */}
            <div className="panel space-y-2 xl:col-start-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-gray-700">
                  SMD Balance Table
                  <Tip wide text="Standardized Mean Difference measures imbalance in each covariate between groups." />
                </h3>
                <ResultExporter
                  title="IPTW_SMD_Balance"
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
                      <th className="text-right px-3 py-2 font-medium" title="Two-sample KS test p-value after weighting.">KS <i>p</i> (after)</th>
                      <th className="text-right px-3 py-2 font-medium">Reduction</th>
                      <th className="text-center px-3 py-2 font-medium">Balanced</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(smdBeforeMap).map((cov) => {
                      const before = smdBeforeMap[cov];
                      const after = smdAfterMap[cov];
                      const reduction = before > 0 && Number.isFinite(after) ? ((before - after) / before * 100).toFixed(1) : "—";
                      const balanced = Number.isFinite(after) && after < threshold;
                      const vr = result.variance_ratio_after?.[cov] as number | null | undefined;
                      const vrBefore = result.variance_ratio_before?.[cov] as number | null | undefined;
                      const ksAfter = result.ks_p_after?.[cov] as number | null | undefined;
                      const vrOk = vr == null || (vr >= 0.5 && vr <= 2.0);
                      return (
                        <tr key={cov} className="border-b border-gray-100 hover:bg-gray-50">
                          <td className="px-3 py-1.5 font-mono text-gray-800">{cov}</td>
                          <td className={`px-3 py-1.5 text-right font-mono ${smdColor(before)}`}>{fmtNum(before, 4)}</td>
                          <td className={`px-3 py-1.5 text-right font-mono font-semibold ${smdColor(after)}`}>{fmtNum(after, 4)}</td>
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
                      <td className={`px-3 py-1.5 text-right font-mono font-semibold ${smdColor(result.avg_smd_before)}`}>{fmtNum(result.avg_smd_before, 4)}</td>
                      <td className={`px-3 py-1.5 text-right font-mono font-semibold ${smdColor(result.avg_smd_after)}`}>{fmtNum(result.avg_smd_after, 4)}</td>
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
              const isWeightedGLM = t.startsWith("weighted_glm");
              return (
              <div className="panel space-y-3 xl:col-start-2">
                <h3 className="text-sm font-semibold text-gray-700">
                  Outcome Analysis — Weighted Cohort
                  <span className="ml-2 text-[10px] font-normal text-indigo-600 bg-indigo-50 border border-indigo-200 rounded-full px-2 py-0.5">
                    {result.outcome_result.model}
                  </span>
                </h3>
                <div className="grid grid-cols-3 gap-2">
                  {isCoxKind ? (
                    [
                      ["n (weighted)", result.outcome_result.n],
                      ["Events", result.outcome_result.n_events],
                      ["C-index", result.outcome_result.concordance?.toFixed(3)],
                    ].map(([k, v]) => (
                      <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-sm">{v}</p>
                      </div>
                    ))
                  ) : isWeightedGLM ? (
                    [
                      ["n (weighted)", result.outcome_result.n],
                      ["Estimand", (result.estimand ?? "ate").toUpperCase()],
                      ["SE Method", (result.se_method === "bootstrap" ? "Bootstrap" : "Robust HC1")],
                    ].map(([k, v]) => (
                      <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
                        <p className="text-[10px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-sm">{v}</p>
                      </div>
                    ))
                  ) : (
                    [
                      ["n (weighted)", result.outcome_result.n],
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
          </div>
        ) : (
          /* Empty state */
          <div className="space-y-4">
            <div className="panel bg-gradient-to-br from-indigo-50 to-emerald-50 border-indigo-100">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-3xl">⚖️</span>
                <div>
                  <h2 className="text-base font-bold text-indigo-900">Inverse Probability of Treatment Weighting (IPTW)</h2>
                  <p className="text-xs text-indigo-600">Observational Causal Inference — Propensity Score Weighting</p>
                </div>
              </div>
              <p className="text-sm text-gray-700 leading-relaxed">
                IPTW is a causal inference technique that uses propensity scores to weight each individual in the dataset. This creates a synthetic sample where baseline characteristics are balanced between groups, allowing for unbiased estimation of treatment effects (e.g., ATE or ATT) without discarding data.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {[
                { icon: "🎯", title: "Step 1 — Propensity Score",
                  body: "A model (Logistic, Probit or GBM) predicts the probability of receiving treatment for each subject given their baseline covariates." },
                { icon: "⚖️", title: "Step 2 — Weight Calculation",
                  body: "Subject weights are computed as the inverse probability of their actual treatment assignment. Supports ATE (whole sample), ATT (exposed target), or Overlap (clinical equipoise)." },
                { icon: "📊", title: "Step 3 — Love Plot Balance Check",
                  body: "The weighted standardized mean differences (SMDs) are evaluated for all covariates. Values < 0.10 show excellent publication-ready balance." },
                { icon: "🏥", title: "Step 4 — Weighted Outcome GLM",
                  body: "A weighted regression model (GLM or Cox proportional hazards) is run on the weighted sample, using robust (sandwich) or bootstrap standard errors." },
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
                ["Positivity", "Every subject must have a non-zero probability of receiving both treatment options. Extreme weights (close to 0 or 1 PS) can destabilize the estimator."],
                ["Consistency (SUTVA)", "Assumes a subject's observed outcome under a given treatment is their true potential outcome under that treatment, with no interference between subjects and no multiple versions of treatment."],
                ["Exchangeability", "Assumes all confounding factors are measured and properly adjusted for in the propensity score model (no unmeasured confounding)."],
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
