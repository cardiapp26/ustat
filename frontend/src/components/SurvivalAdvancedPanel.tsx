import { useState, useEffect, useRef, useMemo, type ReactNode, type RefObject } from "react";
import type { Data, Annotations } from "plotly.js";
import Plot from "../PlotComponent";
import { useStore, analysisCols, isCategoricalKind, type Session } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { runFineGray, runEValue, runLandmark, runKM, runCox, runRMST, runRecurrentLWYY, runCoxHorizons, runCoxUniMulti, runCoxModelSpecs, runFrailty, runMultistate, runDynamicPrediction } from "../api";
import { usePlotLayout, usePalette, useTraceDefaults } from "../plotStyle";
import ResultExporter from "./ResultExporter";
import PlotExporter from "./PlotExporter";
import TitledPlot from "./TitledPlot";
import IntervalCensoredPanel from "./IntervalCensoredPanel";
import { Tip } from "./Tip";
import ThreeCol from "./ThreeCol";
import { fmtP, fmtPubP } from "../lib/format";
import type { PlotData, PlotLayout, PlotCaptureHandle, PlotRef } from "../lib/plotTypes";

// ── Loose result shapes for the survival API responses ────────────────────────
// The /api/survival_advanced/* and /api/models/survival/* endpoints return wide,
// duck-typed JSON. These interfaces capture exactly the fields read in this file
// so member access stays typed without resorting to `any`.

/** A Plotly figure as returned embedded in an API result. */
interface ApiPlot {
  data: PlotData[];
  layout: PlotLayout & {
    title?: { text?: string };
    xaxis?: { title?: { text?: string } };
    yaxis?: { title?: { text?: string } };
  };
}

/** A single (time, survival) point on a KM-style step curve. */
interface CurvePoint { time: number; survival: number }

/** Survival-at-landmark-time estimate for a group. */
interface SurvivalAtPoint {
  survival?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  reliable?: boolean;
  n_at_risk?: number | null;
}

interface LogRank { p?: number | null; chi2?: number | null }

interface PairwiseComparison {
  group_a: string;
  group_b: string;
  p?: number | null;
  p_adj?: number | null;
}

interface KMGroup {
  group: string | number;
  n?: number;
  events?: number;
  median_survival?: number | string | null;
  curve: CurvePoint[];
  censors?: CurvePoint[];
  at_risk?: (number | null)[];
  survival_at?: SurvivalAtPoint[];
}

interface KMStratum {
  label: string | number;
  n?: number;
  groups: KMGroup[];
  logrank?: LogRank;
}

interface KMResult {
  groups?: KMGroup[];
  logrank?: LogRank;
  pairwise?: { comparisons?: PairwiseComparison[]; correction?: string } | null;
  survival_times?: number[];
  median_follow_up?: { median?: number | null; q1?: number | null; q3?: number | null };
  risk_times?: number[];
  strata?: KMStratum[];
  n_total?: number;
}

/** Row produced by the log-rank screening scan. */
interface KMScanRow {
  variable: string;
  groups: number | null;
  logrank_p: number | null;
  chi2: number | null;
}

interface CoxCoefficient {
  variable: string;
  log_hr?: number;
  se?: number;
  hr?: number;
  hr_ci_low?: number;
  hr_ci_high?: number;
  p?: number | null;
}

interface CoxResult {
  coefficients?: CoxCoefficient[];
  n?: number;
  concordance?: number;
  log_likelihood?: number;
}

/** Row produced by the univariable Cox screening scan. */
interface CoxScanRow {
  variable: string;
  hr: number | null;
  hr_ci_low: number | null;
  hr_ci_high: number | null;
  p: number | null;
  n: number | null;
}

interface FineGrayCoefficient {
  variable: string;
  shr?: number;
  shr_low?: number;
  shr_high?: number;
  p?: number | null;
}

interface FineGrayResult extends ResultBlockData {
  plot?: ApiPlot;
  regression_result?: {
    model?: string;
    n?: number;
    n_events_of_interest?: number;
    n_competing?: number;
    n_censored?: number;
    concordance?: number;
    coefficients: FineGrayCoefficient[];
    method_note?: string;
  };
}

interface RmstGroupStat { n?: number; rmst?: number | string; ci_low?: number | string; ci_high?: number | string }
interface RmstContrast { group_a: string; group_b: string; delta_rmst?: number | string; p?: number | null }
interface RmstResult extends ResultBlockData {
  plot?: ApiPlot;
  rmst_by_group?: Record<string, RmstGroupStat>;
  rmst_mi_note?: string;
  contrasts?: RmstContrast[];
}

interface LwyyCoefficient {
  variable: string;
  rate_ratio?: number;
  rr_low?: number;
  rr_high?: number;
  p?: number | null;
}
interface LwyyResult extends ResultBlockData {
  plot?: ApiPlot;
  model?: string;
  n_subjects?: number;
  n_events?: number;
  events_per_subject?: number;
  coefficients?: LwyyCoefficient[];
}

/** Fixed-effect row of the shared frailty Cox fit. The backend emits
 *  `variable / estimate (log-HR) / hr / se / p` — there is no CI in the
 *  payload, so the 95% CI is derived here from estimate ± 1.96·se. */
interface FrailtyCoefficient {
  variable: string;
  estimate?: number | null;
  hr?: number | null;
  se?: number | null;
  p?: number | null;
}
interface FrailtyVarianceTest {
  lrt_statistic?: number | null;
  ordinary_chi_square_p?: number | null;
  chi_bar_square_p?: number | null;
  mixture?: string;
  interpretation?: string;
}
interface FrailtyResult extends ResultBlockData {
  plot?: ApiPlot;
  n_subjects?: number;
  n_clusters?: number;
  n_events?: number;
  theta?: number | null;
  theta_se?: number | null;
  frailty_distribution?: string;
  estimation_method?: string;
  coefficients?: FrailtyCoefficient[];
  frailty_variance_test?: FrailtyVarianceTest;
  concordance?: number | null;
  log_likelihood?: number | null;
  warnings?: string[];
  method_note?: string;
}

// ── Multi-state (long-format transition data) ────────────────────────────────
interface TransitionCoefficient { variable?: string; coef?: number; hr?: number; se?: number; p?: number }
interface TransitionFit {
  n_transitions?: number;
  n_events?: number;
  coefficients?: TransitionCoefficient[];
  concordance?: number | null;
  aic?: number | null;
}
interface MarkovTest {
  status?: string;
  reason?: string;
  coef?: number;
  hr?: number;
  se?: number;
  p_value?: number;
  markov_assumption_violated?: boolean;
  interpretation?: string;
}
interface MultistateResult extends ResultBlockData {
  transitions_estimated?: string[];
  results?: Record<string, TransitionFit>;
  markov_assumption_tests?: Record<string, MarkovTest>;
  model_type?: string;
  note?: string;
}
interface DynamicPredictionResult extends ResultBlockData {
  landmark_time?: number;
  current_state?: number;
  n_at_risk?: number;
  times?: number[];
  state_probabilities?: Record<string, number[]>;
  elos?: Record<string, number>;
  bootstrap?: Record<string, unknown>;
  microsimulation?: Record<string, unknown>;
  model_type?: string;
  note?: string;
  warnings?: string[];
}

interface EValueResult extends ResultBlockData {
  evalue_point?: number | string;
  evalue_ci?: number | string;
  interpretation?: string;
}

interface LandmarkCoxRow {
  variable: string;
  HR?: number | string;
  ci_low?: number | string;
  ci_high?: number | string;
  p?: number | null;
  fmi?: number | string | null;
  error?: string;
}
interface LandmarkResult extends ResultBlockData {
  plot?: ApiPlot;
  cox_mi_note?: string;
  cox_results?: LandmarkCoxRow[];
}

/** Forest row consumed by the Forest Builder handoff. */
interface ForestRow {
  label: string;
  est: number | null;
  ci_low: number | null;
  ci_high: number | null;
  p: number | null;
  extra: string;
}
interface CoxHorizonsResult {
  forest_rows?: ForestRow[];
  covariates?: string[];
  predictor?: string;
  interpretation?: string;
}

interface CoxUniMultiResult { rows: UMRow[]; n: number; n_events: number }

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Axios-style error envelope: `e.response.data.detail` may be a string or a
 *  list of validation messages. */
function errorDetail(e: unknown): unknown {
  if (e && typeof e === "object" && "response" in e) {
    const resp = (e as { response?: { data?: { detail?: unknown } } }).response;
    return resp?.data?.detail;
  }
  return undefined;
}

/** Extract a user-facing message from an unknown thrown value, falling back to
 *  the provided default when no `detail` is present. */
function errorMessage(e: unknown, fallback: string): string {
  const detail = errorDetail(e);
  return typeof detail === "string" ? detail : fallback;
}

// In the 2-grid layout only the selected method renders, so the Section is
// always expanded — the header is a static card title, no collapse toggle.
function Section({ title, description, children }: { title: string; description: string; children: React.ReactNode }) {
  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      <div className="px-5 py-3.5 bg-gray-50 border-b border-gray-100">
        <h3 className="text-sm font-semibold text-gray-800">{title}</h3>
        <p className="text-[11px] text-gray-400 mt-0.5">{description}</p>
      </div>
      <div className="px-5 py-4 space-y-4">{children}</div>
    </div>
  );
}

// ── Unadjusted-vs-adjusted paired Cox forest (publication "Figure 4") ─────────
const UM_GREY = "#9aa0a6";
const UM_BLUE = "#1f4e79";

interface UMRow {
  term: string; predictor: string; kind: string;
  category: string | null; reference: string | null;
  unadjusted: { hr: number; hr_ci_low: number; hr_ci_high: number; p: number } | null;
  adjusted: { hr: number; hr_ci_low: number; hr_ci_high: number; p: number } | null;
}

function CoxUniMultiForest({
  result, columns, plotRef, refs, loading, onChangeRef, onClose,
}: {
  result: { rows: UMRow[]; n: number; n_events: number };
  columns: { name: string; value_labels?: Record<string, string> }[];
  plotRef: RefObject<HTMLDivElement | null>;
  refs: Record<string, string>;
  loading: boolean;
  onChangeRef: (predictor: string, level: string) => void;
  onClose: () => void;
}) {
  const rows = result.rows;
  const [umW, setUmW] = useState<number | undefined>(undefined);   // px or auto
  const [umH, setUmH] = useState<number>(Math.max(320, rows.length * 64 + 130));
  const [umLabels, setUmLabels] = useState<Record<string, string>>({});  // per-row label overrides
  const vlab = (pname: string, code: string | null): string => {
    if (code == null) return "";
    const meta = columns.find((c) => c.name === pname);
    return meta?.value_labels?.[String(code)] ?? String(code);
  };
  const autoLabel = (r: UMRow): string =>
    r.kind === "category"
      ? `${vlab(r.predictor, r.category)} (vs ${vlab(r.predictor, r.reference)})`
      : r.predictor;
  const rowLabel = (r: UMRow): string =>
    umLabels[r.term]?.trim() ? umLabels[r.term] : autoLabel(r);
  const fmtHR = (e: UMRow["unadjusted"]): string =>
    e ? `${e.hr.toFixed(2)} (${e.hr_ci_low.toFixed(2)}–${e.hr_ci_high.toFixed(2)})` : "—";

  // Categorical predictors and their selectable levels (for reference choice).
  const catPreds = Array.from(new Set(rows.filter((r) => r.kind === "category").map((r) => r.predictor)));
  const levelsOf = (p: string): string[] => {
    const set = new Set<string>();
    rows.filter((r) => r.predictor === p && r.kind === "category").forEach((r) => {
      if (r.reference) set.add(r.reference);
      if (r.category) set.add(r.category);
    });
    return Array.from(set).sort((a, b) => {
      const x = Number(a), y = Number(b);
      return !isNaN(x) && !isNaN(y) ? x - y : a.localeCompare(b);
    });
  };
  const currentRef = (p: string): string =>
    refs[p] ?? (rows.find((r) => r.predictor === p && r.kind === "category")?.reference ?? "");

  // y descending so the first row sits at the top.
  const n = rows.length;
  const ys = rows.map((_, i) => n - i);
  const OFF = 0.18;

  // x-range from all finite estimates.
  const xs: number[] = [];
  for (const r of rows) {
    for (const e of [r.unadjusted, r.adjusted]) {
      if (e) { xs.push(e.hr_ci_low, e.hr_ci_high, e.hr); }
    }
  }
  const xmin = xs.length ? Math.min(...xs) : 0.5;
  const xmax = xs.length ? Math.max(...xs) : 2;
  const TICKS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16].filter((t) => t >= xmin * 0.6 && t <= xmax * 1.6);

  const mkTrace = (which: "unadjusted" | "adjusted") => {
    const xs2: number[] = [], ys2: number[] = [], plus: number[] = [], minus: number[] = [];
    rows.forEach((r, i) => {
      const e = r[which];
      if (!e) return;
      xs2.push(e.hr);
      ys2.push(ys[i] + (which === "unadjusted" ? OFF : -OFF));
      plus.push(e.hr_ci_high - e.hr);
      minus.push(e.hr - e.hr_ci_low);
    });
    return {
      x: xs2, y: ys2, type: "scatter", mode: "markers",
      name: which === "unadjusted" ? "Unadjusted" : "Adjusted",
      marker: {
        symbol: which === "unadjusted" ? "circle" : "square",
        size: which === "unadjusted" ? 11 : 10,
        color: which === "unadjusted" ? UM_GREY : UM_BLUE,
      },
      error_x: {
        type: "data", symmetric: false, array: plus, arrayminus: minus,
        color: which === "unadjusted" ? UM_GREY : UM_BLUE, thickness: 2, width: 4,
      },
    } as PlotData;
  };

  // Right-hand numeric columns: anchor at the plot-area right edge (x=1 paper)
  // and offset by a FIXED pixel amount so the columns never scale off-screen
  // with plot width (the previous paper-fraction x clipped the Adjusted column).
  const COL_U_SHIFT = 14;    // px right of plot area: Unadjusted column
  const COL_A_SHIFT = 168;   // px right of plot area: Adjusted column
  const RIGHT_MARGIN = 340;  // must exceed COL_A_SHIFT + text width so export fits
  const ann: PlotData[] = [
    { xref: "paper", yref: "paper", x: 1, y: 1, xanchor: "left", yanchor: "bottom",
      xshift: COL_U_SHIFT, yshift: 8,
      text: "<b>Unadjusted</b>", showarrow: false, font: { color: UM_GREY, size: 12 } },
    { xref: "paper", yref: "paper", x: 1, y: 1, xanchor: "left", yanchor: "bottom",
      xshift: COL_A_SHIFT, yshift: 8,
      text: "<b>Adjusted</b>", showarrow: false, font: { color: UM_BLUE, size: 12 } },
  ];
  rows.forEach((r, i) => {
    ann.push({ xref: "paper", yref: "y", x: 1, y: ys[i], xanchor: "left", yanchor: "middle",
      xshift: COL_U_SHIFT, text: fmtHR(r.unadjusted), showarrow: false,
      font: { color: UM_GREY, size: 11 } });
    ann.push({ xref: "paper", yref: "y", x: 1, y: ys[i], xanchor: "left", yanchor: "middle",
      xshift: COL_A_SHIFT, text: fmtHR(r.adjusted), showarrow: false,
      font: { color: UM_BLUE, size: 11 } });
  });

  return (
    <div className="rounded-lg border border-gray-200 mt-2">
      <div className="bg-gray-50 px-3 py-2 border-b border-gray-200 flex items-center justify-between">
        <p className="text-xs font-semibold text-gray-600">
          Unadjusted vs Adjusted hazard ratios — paired forest
          <span className="ml-2 font-normal text-gray-400"><i>n</i> = {result.n}, {result.n_events} events</span>
          <Tip wide text="Grey circle = univariable (unadjusted) HR; blue square = multivariable (adjusted) HR, both with 95% CI. Labels use the column's value labels. Drag Width/Height to size the figure. Export (↓ top-right) for the manuscript; the figure title is intentionally omitted so you can supply it as the figure legend." />
        </p>
        <button onClick={onClose} className="text-[10px] text-gray-400 hover:text-red-500">✕ Close</button>
      </div>

      {/* Size controls — same pattern as the KM plot */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 px-3 pt-2">
        <label className="flex items-center gap-1.5 text-[10px] text-gray-500">
          <span className="font-medium">Width</span>
          <input type="range" min={520} max={1400} step={20} value={umW ?? 820}
            onChange={(e) => setUmW(Number(e.target.value))} className="accent-indigo-500" style={{ width: 110 }} />
          <span className="tabular-nums w-8">{umW ?? "auto"}</span>
          {umW != null && <button onClick={() => setUmW(undefined)} className="text-indigo-500 hover:text-indigo-700">auto</button>}
        </label>
        <label className="flex items-center gap-1.5 text-[10px] text-gray-500">
          <span className="font-medium">Height</span>
          <input type="range" min={260} max={900} step={20} value={umH}
            onChange={(e) => setUmH(Number(e.target.value))} className="accent-indigo-500" style={{ width: 110 }} />
          <span className="tabular-nums w-8">{umH}</span>
        </label>
      </div>

      {/* Editable row labels — override the axis text for publication */}
      <div className="px-3 pt-2 space-y-1">
        <span className="text-[10px] text-gray-400 inline-flex items-center">Row labels
          <Tip wide text="Rename each row as it appears on the plot's left axis — e.g. 'AGE' → 'Age, per year', or a reference contrast → 'LDL-C <100 mg/dL (vs >130)'. Leave blank to keep the auto label. Reference level for categoricals is the lowest value code; rename here if you want a different wording." />
        </span>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {rows.map((r) => (
            <span key={r.term} className="inline-flex items-center gap-1">
              <input
                value={umLabels[r.term] ?? ""}
                onChange={(e) => {
                  const next = { ...umLabels };
                  if (e.target.value.trim()) next[r.term] = e.target.value; else delete next[r.term];
                  setUmLabels(next);
                }}
                placeholder={autoLabel(r)}
                className="flex-1 text-[11px] border border-gray-200 rounded px-2 py-0.5 focus:outline-none focus:border-indigo-400" />
            </span>
          ))}
        </div>
      </div>

      {/* Reference level per categorical predictor — align figure with Methods/text */}
      {catPreds.length > 0 && (
        <div className="px-3 pt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5">
          <span className="text-[10px] text-gray-400 inline-flex items-center">Reference group
            <Tip wide text="Choose the comparison (reference) level for each categorical predictor. All HRs are reported versus this level — set it to match your manuscript's Methods (e.g. LDL-C >130 mg/dL as reference so the headline '<100 vs >130' contrast appears). Refits both models on change." />
          </span>
          {catPreds.map((p) => (
            <label key={p} className="flex items-center gap-1 text-[10px] text-gray-500">
              <span className="font-medium">{p}</span>
              <select
                value={currentRef(p)}
                disabled={loading}
                onChange={(e) => onChangeRef(p, e.target.value)}
                className="select text-[11px] py-0.5 disabled:opacity-50"
              >
                {levelsOf(p).map((lvl) => (
                  <option key={lvl} value={lvl}>{vlab(p, lvl)}</option>
                ))}
              </select>
            </label>
          ))}
          {loading && <span className="text-[10px] text-indigo-500">refitting…</span>}
        </div>
      )}

      <div className="relative p-2" ref={plotRef}
        style={{ width: umW != null ? umW : "100%", height: umH, maxWidth: "100%" }}>
        <Plot
          data={[mkTrace("unadjusted"), mkTrace("adjusted")]}
          layout={{
            paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff",
            xaxis: {
              type: "log", title: { text: "Hazard ratio (log scale)" },
              tickvals: TICKS, ticktext: TICKS.map(String),
              zeroline: false,
            },
            yaxis: {
              tickvals: ys, ticktext: rows.map(rowLabel),
              range: [0.3, n + 0.7], showgrid: false, zeroline: false,
              automargin: true,   // grow left margin to fit long row labels
            },
            shapes: [{ type: "line", x0: 1, x1: 1, yref: "paper", y0: 0, y1: 1,
              line: { color: "#9aa0a6", dash: "dash", width: 1.2 } }],
            annotations: ann,
            showlegend: true,
            legend: { orientation: "h", x: 0.0, y: -0.14, font: { size: 11 } },
            margin: { t: 40, r: RIGHT_MARGIN, b: 56, l: 230 },
            autosize: true,
          }}
          config={{ responsive: true }}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
        />
        <PlotExporter plotRef={plotRef as unknown as PlotRef} title="Figure_unadjusted_vs_adjusted_HR" />
      </div>
    </div>
  );
}

// Method registry — drives the left nav + which Section renders on the right.
type SurvMethod =
  | "km" | "cox" | "timehorizon" | "landmark" | "rmst"
  | "finegray" | "lwyy" | "evalue" | "intervalcensored" | "frailty" | "multistate";

const SURV_METHODS: { id: SurvMethod; title: string; desc: string }[] = [
  { id: "km",          title: "Kaplan-Meier",        desc: "Survival curves + log-rank" },
  { id: "cox",         title: "Cox PH",              desc: "Hazard ratios" },
  { id: "timehorizon", title: "Time-horizon HR",     desc: "HR by follow-up window → forest" },
  { id: "landmark",    title: "Landmark",            desc: "Conditional on surviving to t" },
  { id: "rmst",        title: "RMST",                desc: "Restricted mean survival time" },
  { id: "finegray",    title: "Fine-Gray",           desc: "Competing risks (CIF)" },
  { id: "lwyy",        title: "Recurrent (LWYY)",    desc: "Repeated events" },
  { id: "intervalcensored", title: "Interval-censored", desc: "Event time bracketed [L, R]" },
  { id: "frailty",     title: "Shared Frailty",      desc: "Clustered / multi-centre survival" },
  { id: "multistate",  title: "Multi-state",         desc: "Transitions between clinical states" },
  { id: "evalue",      title: "E-value",             desc: "Unmeasured confounding" },
];

/** Ordinal (ordered categorical) is eligible wherever numeric OR categorical is. */
function kindMatches(kinds: string[] | undefined, kind: string): boolean {
  if (!kinds) return true;
  if (kinds.includes(kind)) return true;
  return kind === "ordinal" && (kinds.includes("numeric") || kinds.includes("categorical"));
}

/** Missing-data strategy toggle for the regression-style survival analyses.
 *  MICE → m datasets pooled by Rubin's rules (covariate imputation only). */
function ImputeToggle({ value, onChange }: { value: "listwise" | "mice"; onChange: (v: "listwise" | "mice") => void }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-gray-500 font-medium">Missing data</span>
      <div className="flex gap-1">
        {([["listwise", "Complete-case"], ["mice", "MICE (pooled)"]] as const).map(([v, lab]) => (
          <button key={v} type="button" onClick={() => onChange(v)}
            className={`text-[11px] px-2.5 py-1 rounded-lg border transition-colors ${
              value === v ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-gray-600 border-gray-200 hover:border-indigo-300"}`}>
            {lab}
          </button>
        ))}
      </div>
      {value === "mice" && (
        <span className="text-[10px] text-gray-400 leading-snug">m=10 chained-PMM datasets, Rubin's-rules pooling (covariates imputed; outcome kept complete-case).</span>
      )}
    </div>
  );
}

function VarSelect({ label, value, onChange, columns, kinds }: {
  label: string; value: string; onChange: (v: string) => void;
  columns: { name: string; kind: string }[]; kinds?: string[];
}) {
  const filtered = kinds ? columns.filter((c) => kindMatches(kinds, c.kind)) : columns;
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-gray-500 font-medium">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}
        className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400">
        <option value="">— select —</option>
        {filtered.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
      </select>
    </label>
  );
}

function MultiSelect({ label, columns, selected, onChange, kinds, excludeNames }: {
  label: string; columns: { name: string; kind: string }[];
  selected: string[]; onChange: (v: string[]) => void;
  kinds?: string[]; excludeNames?: string[];
}) {
  const filtered = (kinds ? columns.filter((c) => kindMatches(kinds, c.kind)) : columns)
    .filter((c) => !(excludeNames ?? []).includes(c.name));
  const toggle = (name: string) =>
    onChange(selected.includes(name) ? selected.filter((c) => c !== name) : [...selected, name]);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500 font-medium">{label}</span>
        <div className="flex items-center gap-2 text-[10px] text-gray-400">
          <span>{selected.length} selected</span>
          {selected.length > 0 && (
            <button type="button" onClick={() => onChange([])} className="hover:text-red-500">Clear</button>
          )}
        </div>
      </div>
      <div className="text-xs border border-gray-300 rounded-lg px-2 py-1.5 bg-white max-h-40 overflow-y-auto space-y-0.5">
        {filtered.length === 0 && <p className="text-[11px] text-gray-400 italic">No columns available</p>}
        {filtered.map((c) => {
          const isNum = c.kind === "numeric";
          const isChecked = selected.includes(c.name);
          return (
            <label key={c.name} className="flex items-center gap-2 cursor-pointer hover:bg-gray-50 px-1 py-0.5 rounded">
              <input type="checkbox" checked={isChecked} onChange={() => toggle(c.name)} className="accent-indigo-500 flex-shrink-0" />
              <span className={`flex-1 truncate ${isChecked ? "text-gray-900 font-medium" : "text-gray-700"}`}>{c.name}</span>
              <span className={`text-[9px] px-1 rounded flex-shrink-0 ${isNum ? "bg-blue-50 text-blue-600" : "bg-purple-50 text-purple-600"}`}>
                {isNum ? "N" : "C"}
              </span>
            </label>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Event columns for Cox must be binary 0/1. The session ColMeta `kind`
 * enum has no "binary" value AND a 0/1 column is frequently classified as
 * `categorical` (e.g. SEX, DM, HT), so a kind-based filter both leaks
 * continuous numerics and hides the real event columns. Detect binary
 * columns purely from the data: scan every column's preview values and
 * keep those whose non-null values are all numerically coercible with
 * ≤2 distinct values (0/1, 1/2, …) — regardless of `kind`.
 *
 * Falls back to the full column list only when nothing qualifies (e.g.
 * preview unavailable) so the user is never stuck with an empty dropdown.
 */
function binaryLikeColumns(
  columns: { name: string; kind: string }[],
  preview: Record<string, unknown>[] | undefined,
): { name: string; kind: string }[] {
  if (!preview || preview.length === 0) return columns;
  const out: { name: string; kind: string }[] = [];
  for (const c of columns) {
    const distinct = new Set<number>();
    let ok = true;
    let seen = 0;
    for (const row of preview) {
      const v = row[c.name];
      if (v === null || v === undefined || v === "") continue;
      const n = Number(v);
      if (!Number.isFinite(n)) { ok = false; break; }  // non-numeric → not an event col
      seen += 1;
      distinct.add(n);
      if (distinct.size > 2) { ok = false; break; }
    }
    if (ok && seen > 0 && distinct.size >= 1 && distinct.size <= 2) out.push(c);
  }
  return out.length > 0 ? out : columns;
}

// Okabe-Ito colourblind-safe palette + distinct line dashes for KM curves.
const OKABE_ITO = ["#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"];
const KM_DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"];

/** Nice round tick set 0..xmax (~7 ticks) for the number-at-risk columns. */
function niceRiskTimes(xmax: number): number[] {
  if (!(xmax > 0) || !isFinite(xmax)) return [];
  const raw = xmax / 6;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const ticks: number[] = [];
  for (let t = 0; t <= xmax + 1e-9; t += step) ticks.push(Math.round(t));
  return ticks;
}

/**
 * Compose a publication-style standard interpretation of a KM result
 * from the returned data: overall log-rank, landmark survival, pairwise
 * comparisons, and median survival. Returns "" when not enough is present.
 */
function buildKmNarrative(
  km: KMResult | null,
  labels: Record<string, string>,
  groupName: string,
): string {
  if (!km || !Array.isArray(km.groups) || km.groups.length === 0) return "";
  const lab = (g: string) => labels[g] ?? g;
  const groups = km.groups;
  const gv = groupName || "group";
  const parts: string[] = [];

  // 0. Median follow-up (reverse KM).
  const mfu = km.median_follow_up;
  if (mfu?.median != null) {
    const iqr = (mfu.q1 != null && mfu.q3 != null)
      ? ` (IQR ${mfu.q1.toFixed(0)}–${mfu.q3.toFixed(0)})` : "";
    parts.push(`Median follow-up was ${mfu.median.toFixed(0)}${iqr}.`);
  }

  // 1. Overall difference.
  if (groups.length >= 2 && km.logrank?.p != null) {
    const sig = km.logrank.p < 0.05;
    parts.push(
      `Kaplan–Meier survival ${sig ? "differed significantly" : "did not differ significantly"} ` +
      `across the ${groups.length} ${gv} groups (log-rank ${fmtPubP(km.logrank.p)}).`,
    );
  }

  // 2. Landmark survival at the requested time point(s).
  if (Array.isArray(km.survival_times) && km.survival_times.length > 0) {
    for (const t of km.survival_times) {
      const idx = km.survival_times.indexOf(t);
      let anyUnreliable = false;
      const frags = groups
        .map((g) => {
          const pt = (g.survival_at ?? [])[idx];
          if (!pt || pt.survival == null) return null;
          if (pt.reliable === false) anyUnreliable = true;
          return `${(pt.survival * 100).toFixed(1)}% in the ${lab(String(g.group))} group`;
        })
        .filter(Boolean);
      if (frags.length) {
        parts.push(
          `Estimated survival at t=${t} was ${frags.join(", ")}.` +
          (anyUnreliable ? " (Interpret the t=" + t + " estimate with caution — few patients remained at risk.)" : ""),
        );
      }
    }
  }

  // 3. Pairwise comparisons.
  const pw = km.pairwise?.comparisons;
  if (pw && pw.length) {
    const useAdj = pw.some((c) => c.p_adj != null);
    const val = (c: PairwiseComparison) => (useAdj ? c.p_adj : c.p);
    const sig = pw.filter((c) => {
      const p = val(c);
      return p != null && p < 0.05;
    });
    const ns = pw.filter((c) => {
      const p = val(c);
      return p != null && p >= 0.05;
    });
    const fmtPair = (c: PairwiseComparison) => `${lab(c.group_a)} vs ${lab(c.group_b)}, ${fmtPubP(val(c))}`;
    const bits: string[] = [];
    if (sig.length) bits.push(`significant differences were found for ${sig.map(fmtPair).join("; ")}`);
    if (ns.length) bits.push(`no significant difference between ${ns.map(fmtPair).join("; ")}`);
    if (bits.length) {
      parts.push(
        `Pairwise comparisons (${km.pairwise?.correction && km.pairwise.correction !== "none" ? km.pairwise.correction + "-adjusted" : "unadjusted"}) showed ${bits.join(", whereas ")}.`,
      );
    }
  }

  // 4. Median survival.
  const meds = groups.map((g) => ({ g: lab(String(g.group)), m: g.median_survival }));
  const reached = meds.filter((x) => x.m != null);
  if (reached.length === 0 && groups.length > 0) {
    parts.push("Median survival was not reached in any group.");
  } else if (reached.length) {
    parts.push(`Median survival: ${reached.map((x) => `${x.m} (${x.g})`).join(", ")}.`);
  }

  return parts.join(" ");
}

function RunButton({ onClick, loading, label }: { onClick: () => void; loading: boolean; label: string }) {
  return (
    <button onClick={onClick} disabled={loading}
      className="px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors">
      {loading ? "Running…" : label}
    </button>
  );
}

interface ResultBlockData {
  result_text?: string;
  assumptions?: { met?: boolean; name?: string; detail?: string }[];
  export_rows?: (string | number | null)[][];
  r_code?: string;
  test?: string;
}

function ResultBlock({ result }: { result: ResultBlockData | null | undefined }) {
  if (!result) return null;
  const assumptions = result.assumptions ?? [];
  const exportRows = result.export_rows ?? [];
  return (
    <div className="space-y-3 mt-3">
      {/* Result text */}
      {result.result_text && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-4 py-3 text-sm text-indigo-900">
          {result.result_text}
        </div>
      )}

      {/* Assumptions */}
      {assumptions.length > 0 && (
        <div className="space-y-1">
          {assumptions.map((a, i: number) => (
            <div key={i} className={`flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg ${a.met ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
              <span>{a.met ? "✓" : "⚠"}</span>
              <span className="font-medium">{a.name}</span>
              <span className="text-gray-500">— {a.detail}</span>
            </div>
          ))}
        </div>
      )}

      {/* Export rows as table */}
      {exportRows.length > 1 && (
        <div className="overflow-auto rounded-lg border border-gray-200">
          <table className="text-xs w-full">
            <thead>
              <tr className="bg-gray-50">
                {exportRows[0].map((h, i: number) => (
                  <th key={i} className="px-3 py-1.5 text-left text-gray-500 font-medium border-r border-gray-100 last:border-r-0">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {exportRows.slice(1).map((row, ri: number) => (
                <tr key={ri} className="border-t border-gray-100">
                  {row.map((v, ci: number) => (
                    <td key={ci} className="px-3 py-1 text-gray-700 border-r border-gray-100 last:border-r-0">{v ?? "—"}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* R code */}
      {result.r_code && (
        <details className="group">
          <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-600">R Code</summary>
          <pre className="mt-1 bg-gray-900 text-green-300 text-[11px] rounded-lg p-3 overflow-x-auto">{result.r_code}</pre>
        </details>
      )}

      {/* Exporter */}
      {exportRows.length > 1 && (
        <ResultExporter title={result.test ?? "result"} headers={exportRows[0].map((h) => h == null ? "" : String(h))} rows={exportRows.slice(1)} />
      )}
    </div>
  );
}

// ── Main Panel ───────────────────────────────────────────────────────────────

export default function SurvivalAdvancedPanel() {
  const session = useStore((s) => s.session);
  if (!session) return <p className="text-gray-400 text-sm p-6">Upload data first.</p>;
  return <SurvivalAdvancedPanelBody session={session} />;
}

function SurvivalAdvancedPanelBody({ session }: { session: Session }) {
  const columns = session.columns;
  // Columns offered in variable pickers: hide any flagged "exclude from
  // analysis". The full `columns` list is kept for display lookups of
  // already-saved results (value_labels of selected columns).
  const pickCols = useMemo(() => analysisCols(columns), [columns]);
  const sid = session.session_id;
  // 2-grid layout: left method nav, right active method panel.
  const [activeMethod, setActiveMethod] = usePersistedPanelState<SurvMethod>("survival", "activeMethod", "km");
  // Binary 0/1-like columns for Cox event selectors (the `kind` enum has
  // no "binary", so a numeric filter would leak continuous columns).
  const binaryCols = useMemo(
    () => binaryLikeColumns(pickCols, session?.preview),
    [pickCols, session?.preview],
  );
  // Cross-panel handoff to the Forest Builder + Visual-tab deep link.
  const setForestHandoff = useStore((s) => s.setForestHandoff);
  const setVisualSubTab = useStore((s) => s.setVisualSubTab);
  const setActiveTab = useStore((s) => s.setActiveTab);

  const baseLayout = usePlotLayout();
  const pal = usePalette();
  const traceDefaults = useTraceDefaults();

  const fgPlotRef = useRef<PlotCaptureHandle | null>(null);
  const lmPlotRef = useRef<PlotCaptureHandle | null>(null);

  // Fine-Gray state
  const [fgDuration, setFgDuration] = usePersistedPanelState("survival", "fgDuration", "");
  const [fgEvent, setFgEvent] = usePersistedPanelState("survival", "fgEvent", "");
  const [fgInterest, setFgInterest] = usePersistedPanelState("survival", "fgInterest", 1);
  const [fgGroup, setFgGroup] = usePersistedPanelState("survival", "fgGroup", "");
  const [fgPredictors, setFgPredictors] = usePersistedPanelState<string[]>("survival", "fgPredictors", []);
  const [fgPredFilter, setFgPredFilter] = useState("");
  const [fgResult, setFgResult] = useState<FineGrayResult | null>(null);
  const [fgLoading, setFgLoading] = useState(false);
  const [fgError, setFgError] = useState<string | null>(null);

  // E-value state
  const [evEst, setEvEst] = useState("");
  const [evLo, setEvLo] = useState("");
  const [evHi, setEvHi] = useState("");
  const [evType, setEvType] = usePersistedPanelState("survival", "evType", "OR");
  const [evP0, setEvP0] = useState("0.1");
  const [evResult, setEvResult] = useState<EValueResult | null>(null);
  const [evLoading, setEvLoading] = useState(false);
  const [evError, setEvError] = useState<string | null>(null);

  // KM state
  const [kmDuration, setKmDuration] = usePersistedPanelState("survival", "kmDuration", "");
  const [kmEvent, setKmEvent] = usePersistedPanelState("survival", "kmEvent", "");
  const [kmGroup, setKmGroup] = usePersistedPanelState("survival", "kmGroup", "");
  const [kmStratify, setKmStratify] = usePersistedPanelState("survival", "kmStratify", "");
  const [kmSurvTimes, setKmSurvTimes] = usePersistedPanelState("survival", "kmSurvTimes", "");          // e.g. "365, 1825"
  const [kmPairwise, setKmPairwise] = usePersistedPanelState("survival", "kmPairwise", false);
  const [kmCorrection, setKmCorrection] = usePersistedPanelState("survival", "kmCorrection", "holm");    // none|bonferroni|holm|bh
  const [kmResult, setKmResult] = useState<KMResult | null>(null);
  const [kmLoading, setKmLoading] = useState(false);
  const [kmError, setKmError] = useState<string | null>(null);
  const kmPlotRef = useRef<HTMLDivElement | null>(null);
  // KM screening state
  const [kmScanResult, setKmScanResult] = useState<KMScanRow[]>([]);
  const [kmScanLoading, setKmScanLoading] = useState(false);
  // Group rename state
  const [kmGroupLabels, setKmGroupLabels] = useState<Record<string, string>>({});
  const [kmCustomGroupTitle, setKmCustomGroupTitle] = useState("");
  const [kmCustomDurationTitle, setKmCustomDurationTitle] = useState("");
  // Publication-quality customisation
  const [kmCustomPlotTitle, setKmCustomPlotTitle] = useState("");      // editable plot title
  const [kmShowPInTitle, setKmShowPInTitle] = useState(true);          // append " (log-rank p=…)"
  const [kmShowNInLegend, setKmShowNInLegend] = useState(true);        // append " (n=…)"
  const [kmHidePrefix, setKmHidePrefix] = useState(true);              // drop "groupCol = " prefix
  const [kmAutoZoomY, setKmAutoZoomY] = useState(true);                // zoom Y to data range
  const [kmYAxisAsPct, setKmYAxisAsPct] = useState(false);             // % vs decimal Y
  const [kmYTitle, setKmYTitle] = useState("Survival probability");    // editable Y-axis label
  const [kmPlotH, setKmPlotH] = useState(420);                         // plot height px
  const [kmPlotW, setKmPlotW] = useState<number | undefined>(undefined); // plot width px (undefined = fill)
  const [kmRiskTable, setKmRiskTable] = useState(true);              // number-at-risk table
  const [kmColorblind, setKmColorblind] = useState(true);           // Okabe-Ito + line styles
  const [kmShowCensors, setKmShowCensors] = useState(true);         // censor tick marks
  const [kmGroupColors, setKmGroupColors] = useState<Record<string, string>>({}); // per-group color
  const [kmContextMenu, setKmContextMenu] = useState<{ type: "item"|"groupTitle"|"durationTitle"|"plotTitle"; group?: string; x: number; y: number } | null>(null);
  const [kmRenameValue, setKmRenameValue] = useState("");

  // Nudge Plotly to refit when the KM plot box is resized via sliders
  // (useResizeHandler only listens to window resize, not element resize).
  useEffect(() => {
    const id = window.setTimeout(() => window.dispatchEvent(new Event("resize")), 50);
    return () => window.clearTimeout(id);
  }, [kmPlotW, kmPlotH]);

  // Lazily augment the current KM result when the risk-table or censor-mark
  // toggles are switched on after the run: derive tick times from the
  // curves' x-max and re-fetch with risk_times / include_censors so the
  // extras render without forcing the user to re-run.
  useEffect(() => {
    if (!kmResult?.groups?.length) return;
    const needRisk = kmRiskTable && !kmResult.groups.some((g) => Array.isArray(g.at_risk));
    const needCens = kmShowCensors && !kmResult.groups.some((g) => Array.isArray(g.censors));
    if (!needRisk && !needCens) return;
    let xmax = 0;
    for (const g of kmResult.groups) {
      const c = g.curve;
      if (c?.length) xmax = Math.max(xmax, c[c.length - 1].time);
    }
    const times = niceRiskTimes(xmax);
    let cancelled = false;
    (async () => {
      try {
        const res = await runKM({
          session_id: sid, duration_col: kmDuration, event_col: kmEvent,
          group_col: kmGroup || undefined, stratify_col: kmStratify || undefined,
          survival_times: kmResult.survival_times?.length ? kmResult.survival_times : undefined,
          pairwise: !!kmResult.pairwise, pairwise_correction: kmCorrection,
          risk_times: kmRiskTable && times.length ? times : undefined,
          include_censors: kmShowCensors,
        });
        if (!cancelled) setKmResult(res.data);
      } catch { /* non-fatal — extras just won't show */ }
    })();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kmRiskTable, kmShowCensors, kmResult]);

  // Cox state
  const [coxDuration, setCoxDuration] = usePersistedPanelState("survival", "coxDuration", "");
  const [coxEvent, setCoxEvent] = usePersistedPanelState("survival", "coxEvent", "");
  const [coxPreds, setCoxPreds] = usePersistedPanelState<string[]>("survival", "coxPreds", []);
  const [coxInteractions, setCoxInteractions] = usePersistedPanelState<Array<[string, string]>>("survival", "coxInteractions", []);
  const [coxIxA, setCoxIxA] = useState<string>("");
  const [coxIxB, setCoxIxB] = useState<string>("");
  const [coxResult, setCoxResult] = useState<CoxResult | null>(null);
  const [coxLoading, setCoxLoading] = useState(false);
  const [coxError, setCoxError] = useState<string | null>(null);
  // Cox univariable screening state
  const [coxScanResult, setCoxScanResult] = useState<CoxScanRow[]>([]);
  const [coxScanLoading, setCoxScanLoading] = useState(false);
  // Unadjusted-vs-adjusted paired forest (publication "Figure 4")
  const [coxUMResult, setCoxUMResult] = useState<CoxUniMultiResult | null>(null);
  const [coxUMLoading, setCoxUMLoading] = useState(false);
  const [coxUMRefs, setCoxUMRefs] = useState<Record<string, string>>({});  // per-predictor reference level
  const coxUMRef = useRef<HTMLDivElement | null>(null);

  const runCoxUMForest = async (refs: Record<string, string>) => {
    setCoxUMLoading(true);
    try {
      const res = await runCoxUniMulti({
        session_id: sid, duration_col: coxDuration, event_col: coxEvent,
        predictors: coxPreds, references: Object.keys(refs).length ? refs : undefined,
      });
      setCoxUMResult(res.data);
    } catch (e: unknown) { setCoxError(errorMessage(e, "Forest failed")); }
    finally { setCoxUMLoading(false); }
  };

  // Model-specification (sensitivity) forest: one exposure across adjustment sets.
  type SpecRow = { label: string; covariates: string[] };
  const [specOpen, setSpecOpen] = usePersistedPanelState("survival", "specOpen", false);
  const [specExposure, setSpecExposure] = usePersistedPanelState("survival", "specExposure", "");
  const [specExposureRef, setSpecExposureRef] = usePersistedPanelState("survival", "specExposureRef", "");
  // For a multi-level categorical exposure: show only this contrast level
  // (e.g. <100) across models, so each row is one model — clean + matches text.
  const [specExposureLevel, setSpecExposureLevel] = usePersistedPanelState("survival", "specExposureLevel", "");
  const [specRows, setSpecRows] = usePersistedPanelState<SpecRow[]>("survival", "specRows", [
    { label: "Parsimonious", covariates: [] },
    { label: "Fully adjusted", covariates: [] },
  ]);
  const [specLoading, setSpecLoading] = useState(false);

  const runSpecForest = async () => {
    if (!coxDuration || !coxEvent || !specExposure) { setCoxError("Select duration, event, and an exposure."); return; }
    setSpecLoading(true); setCoxError(null);
    try {
      const meta = columns.find((c) => c.name === specExposure);
      const vlab = (code: string | null) => (code == null ? "" : (meta?.value_labels?.[String(code)] ?? String(code)));
      const res = await runCoxModelSpecs({
        session_id: sid, duration_col: coxDuration, event_col: coxEvent,
        exposure: specExposure,
        exposure_reference: specExposureRef || undefined,
        specs: specRows.filter((s) => s.label.trim()).map((s) => ({ label: s.label, covariates: s.covariates })),
        include_unadjusted: true,
      });
      const data = res.data;
      const terms = (data.exposure_terms ?? []) as Array<{ term: string; kind: string; category: string | null; reference: string | null }>;
      const multiTerm = terms.length > 1;
      // When a single contrast level is chosen, keep only that term → one row
      // per model with the MODEL NAME as the label (short, no clipping, matches
      // the manuscript). The contrast goes into the caption instead.
      const focusLevel = multiTerm && specExposureLevel ? specExposureLevel : null;
      type SpecTerm = {
        term: string; kind: string; category: string | null; reference: string | null;
        hr?: number | null; hr_ci_low?: number | null; hr_ci_high?: number | null; p?: number | null;
      };
      type SpecData = { label: string; n: number; n_events: number; terms: SpecTerm[] };
      const specsData = data.specs as SpecData[];
      // If every model shares the same n / events (no covariate missingness),
      // the per-row "(48/388 events)" is redundant — show it once in the
      // caption. Only keep it per-row when models actually differ.
      const neSet = new Set(specsData.map((s) => `${s.n_events}/${s.n}`));
      const sameNE = neSet.size === 1 && specsData[0]?.n > 0;
      const rows: Array<{ label: string; est: number | null; ci_low: number | null; ci_high: number | null; p: number | null; extra: string }> = [];
      for (const s of specsData) {
        for (const t of s.terms) {
          if (focusLevel && String(t.category) !== String(focusLevel)) continue;
          // No focus level → prefix the model name so each row is identifiable.
          // Prepend the exposure column name to each level so the contrast reads
          // as "LDL-C <100 vs LDL-C >130" rather than the bare "<100 vs >130",
          // which is meaningless once the row is detached from the panel header.
          const contrast = (!focusLevel && multiTerm && t.kind === "category")
            ? ` — ${specExposure} ${vlab(t.category)} vs ${specExposure} ${vlab(t.reference)}` : "";
          rows.push({
            label: `${s.label}${contrast}`,
            est: t.hr ?? null, ci_low: t.hr_ci_low ?? null, ci_high: t.hr_ci_high ?? null,
            p: t.p ?? null,
            extra: sameNE ? "" : (s.n ? `(${s.n_events}/${s.n} events)` : ""),
          });
        }
      }
      if (!rows.length) { setCoxError("No model fit — check exposure / covariates."); return; }
      const focusTerm = focusLevel ? terms.find((t) => String(t.category) === String(focusLevel)) : null;
      const exposureLine = focusTerm
        ? `Exposure: ${specExposure} (${vlab(focusTerm.category)} vs ${vlab(focusTerm.reference)})`
        : `Exposure: ${specExposure}`;
      const cohortLine = sameNE ? ` · n=${specsData[0].n}, ${specsData[0].n_events} events` : "";
      setForestHandoff(rows, {
        customTitle: "",
        customSubtitle: `${exposureLine} · adjusted HR across model specifications${cohortLine}`,
        xLabel: "Hazard ratio for all-cause mortality (95% CI, log scale)",
        leftHeader: "Model specification",
        rightHeader: "HR (95% CI)",
        returnTab: "models",
        returnLabel: "← Back to Cox model",
      });
      setVisualSubTab("forest");
      setActiveTab("visual");
    } catch (e: unknown) { setCoxError(errorMessage(e, "Model-spec forest failed")); }
    finally { setSpecLoading(false); }
  };


  // Time-horizon sensitivity (Cox at multiple administrative-censoring windows)
  const [chDuration, setChDuration] = usePersistedPanelState("survival", "chDuration", "");
  const [chEvent, setChEvent] = usePersistedPanelState("survival", "chEvent", "");
  const [chPredictor, setChPredictor] = usePersistedPanelState("survival", "chPredictor", "");
  const [chCovariates, setChCovariates] = usePersistedPanelState<string[]>("survival", "chCovariates", []);
  const [chHorizons, setChHorizons] = usePersistedPanelState("survival", "chHorizons", "365, 730");
  const [chLabels, setChLabels] = usePersistedPanelState("survival", "chLabels", "1 year, 2 years");
  const [chResult, setChResult] = useState<CoxHorizonsResult | null>(null);
  const [chLoading, setChLoading] = useState(false);
  const [chError, setChError] = useState<string | null>(null);

  // Missing-data strategy shared by the regression-style survival analyses
  // (Fine-Gray sHR, landmark Cox, RMST). 'mice' → m datasets pooled (Rubin).
  const [survImputation, setSurvImputation] = usePersistedPanelState<"listwise" | "mice">("survival", "imputation", "listwise");

  // RMST state — Restricted Mean Survival Time (PH-free alternative)
  const [rmstDuration, setRmstDuration] = usePersistedPanelState("survival", "rmstDuration", "");
  const [rmstEvent, setRmstEvent] = usePersistedPanelState("survival", "rmstEvent", "");
  const [rmstGroup, setRmstGroup] = usePersistedPanelState("survival", "rmstGroup", "");
  const [rmstTau, setRmstTau] = usePersistedPanelState<string>("survival", "rmstTau", "");
  const [rmstResult, setRmstResult] = useState<RmstResult | null>(null);
  const [rmstLoading, setRmstLoading] = useState(false);
  const [rmstError, setRmstError] = useState<string | null>(null);
  const rmstPlotRef = useRef<PlotCaptureHandle | null>(null);

  // Shared frailty state — clustered / multi-centre Cox with a random
  // cluster effect. Advanced knobs (nested/correlated clusters, parametric
  // baseline, estimation method, penalizer) stay at their backend defaults.
  const [frDuration, setFrDuration] = usePersistedPanelState("survival", "frailtyDuration", "");
  const [frEvent, setFrEvent] = usePersistedPanelState("survival", "frailtyEvent", "");
  const [frCluster, setFrCluster] = usePersistedPanelState("survival", "frailtyCluster", "");
  const [frPredictors, setFrPredictors] = usePersistedPanelState<string[]>("survival", "frailtyPredictors", []);
  const [frDist, setFrDist] = usePersistedPanelState("survival", "frailtyDistribution", "gamma");
  const [frailtyResult, setFrailtyResult] = useState<FrailtyResult | null>(null);
  const [frLoading, setFrLoading] = useState(false);
  const [frError, setFrError] = useState<string | null>(null);
  const frPlotRef = useRef<PlotCaptureHandle | null>(null);

  // Multi-state state. Unlike every other method here the input is LONG-FORMAT
  // transition data (one row per state transition), so the column pickers are
  // shared by both the transition-model fit and the landmark prediction, and
  // the two results are held separately so running one keeps the other.
  const [msId, setMsId] = usePersistedPanelState("survival", "msIdCol", "");
  const [msFrom, setMsFrom] = usePersistedPanelState("survival", "msFromCol", "");
  const [msTo, setMsTo] = usePersistedPanelState("survival", "msToCol", "");
  const [msEntry, setMsEntry] = usePersistedPanelState("survival", "msEntryCol", "");
  const [msExit, setMsExit] = usePersistedPanelState("survival", "msExitCol", "");
  const [msEvent, setMsEvent] = usePersistedPanelState("survival", "msEventCol", "");
  const [msPredictors, setMsPredictors] = usePersistedPanelState<string[]>("survival", "msPredictors", []);
  const [msLandmark, setMsLandmark] = usePersistedPanelState("survival", "msLandmark", "");
  const [msHorizon, setMsHorizon] = usePersistedPanelState("survival", "msHorizon", "5");
  const [msState, setMsState] = usePersistedPanelState("survival", "msCurrentState", "0");
  const [msBootstrap, setMsBootstrap] = usePersistedPanelState<boolean>("survival", "msBootstrap", false);
  const [msResult, setMsResult] = useState<MultistateResult | null>(null);
  const [dpResult, setDpResult] = useState<DynamicPredictionResult | null>(null);
  const [msLoading, setMsLoading] = useState(false);
  const [dpLoading, setDpLoading] = useState(false);
  const [msError, setMsError] = useState<string | null>(null);
  const dpPlotRef = useRef<PlotCaptureHandle | null>(null);

  // Recurrent-events LWYY state
  const [lwId, setLwId] = usePersistedPanelState("survival", "lwId", "");
  const [lwStart, setLwStart] = usePersistedPanelState("survival", "lwStart", "");
  const [lwStop, setLwStop] = usePersistedPanelState("survival", "lwStop", "");
  const [lwEvent, setLwEvent] = usePersistedPanelState("survival", "lwEvent", "");
  const [lwPreds, setLwPreds] = usePersistedPanelState<string[]>("survival", "lwPreds", []);
  const [lwGroup, setLwGroup] = usePersistedPanelState("survival", "lwGroup", "");
  const [lwResult, setLwResult] = useState<LwyyResult | null>(null);
  const [lwLoading, setLwLoading] = useState(false);
  const [lwError, setLwError] = useState<string | null>(null);
  const lwPlotRef = useRef<PlotCaptureHandle | null>(null);

  // Landmark state
  const [lmDuration, setLmDuration] = usePersistedPanelState("survival", "lmDuration", "");
  const [lmEvent, setLmEvent] = usePersistedPanelState("survival", "lmEvent", "");
  const [lmTime, setLmTime] = usePersistedPanelState("survival", "lmTime", "");
  const [lmGroup, setLmGroup] = usePersistedPanelState("survival", "lmGroup", "");
  const [lmPreds, setLmPreds] = usePersistedPanelState<string[]>("survival", "lmPreds", []);
  const [lmResult, setLmResult] = useState<LandmarkResult | null>(null);
  const [lmLoading, setLmLoading] = useState(false);
  const [lmError, setLmError] = useState<string | null>(null);

  // ── Fine-Gray handler
  const handleFineGray = async () => {
    if (!fgDuration || !fgEvent) { setFgError("Select duration and event columns"); return; }
    setFgResult(null); setFgError(null); setFgLoading(true);
    try {
      const res = await runFineGray({
        session_id: sid, duration_col: fgDuration, event_col: fgEvent,
        event_of_interest: fgInterest, group_col: fgGroup || undefined,
        predictors: fgPredictors.length > 0 ? fgPredictors : undefined,
        imputation: survImputation,
      });
      setFgResult(res.data);
    } catch (e: unknown) { setFgError(errorMessage(e, "Fine-Gray failed")); }
    finally { setFgLoading(false); }
  };
  const fgToggleP = (c: string) =>
    setFgPredictors(fgPredictors.includes(c) ? fgPredictors.filter((x) => x !== c) : [...fgPredictors, c]);

  // ── RMST handler
  const handleRMST = async () => {
    if (!rmstDuration || !rmstEvent) { setRmstError("Select duration and event columns"); return; }
    const tau = parseFloat(rmstTau);
    if (!Number.isFinite(tau) || tau <= 0) { setRmstError("Enter a positive time horizon τ"); return; }
    setRmstResult(null); setRmstError(null); setRmstLoading(true);
    try {
      const res = await runRMST({
        session_id: sid, duration_col: rmstDuration, event_col: rmstEvent,
        tau, group_col: rmstGroup || undefined,
        imputation: survImputation,
      });
      setRmstResult(res.data);
    } catch (e: unknown) { setRmstError(errorMessage(e, "RMST failed")); }
    finally { setRmstLoading(false); }
  };

  // ── Shared frailty handler
  const handleFrailty = async () => {
    if (!frDuration || !frEvent || !frCluster) { setFrError("Select duration, event, and cluster columns"); return; }
    if (frPredictors.length === 0) { setFrError("Select at least one predictor"); return; }
    setFrailtyResult(null); setFrError(null); setFrLoading(true);
    try {
      const res = await runFrailty({
        session_id: sid, duration_col: frDuration, event_col: frEvent,
        cluster_col: frCluster, predictors: frPredictors,
        frailty_distribution: frDist,
        imputation: survImputation,
      });
      setFrailtyResult(res.data);
    } catch (e: unknown) { setFrError(errorMessage(e, "Shared frailty model failed")); }
    finally { setFrLoading(false); }
  };

  // ── E-value handler
  const handleEValue = async () => {
    if (!evEst || !evLo || !evHi) { setEvError("Enter estimate and confidence interval"); return; }
    setEvResult(null); setEvError(null); setEvLoading(true);
    try {
      const res = await runEValue({
        estimate: parseFloat(evEst), ci_low: parseFloat(evLo), ci_high: parseFloat(evHi),
        measure_type: evType, baseline_risk: parseFloat(evP0),
      });
      setEvResult(res.data);
    } catch (e: unknown) { setEvError(errorMessage(e, "E-value failed")); }
    finally { setEvLoading(false); }
  };

  // ── Landmark handler
  const handleLandmark = async () => {
    if (!lmDuration || !lmEvent || !lmTime) { setLmError("Select duration, event, and landmark time"); return; }
    setLmResult(null); setLmError(null); setLmLoading(true);
    try {
      const res = await runLandmark({
        session_id: sid, duration_col: lmDuration, event_col: lmEvent,
        landmark_time: parseFloat(lmTime), group_col: lmGroup || undefined,
        predictors: lmPreds.length > 0 ? lmPreds : undefined,
        imputation: survImputation,
      });
      setLmResult(res.data);
    } catch (e: unknown) { setLmError(errorMessage(e, "Landmark analysis failed")); }
    finally { setLmLoading(false); }
  };

  // ── Auto-clear stale results when their inputs change. Without this the
  // user perceives a re-Run as 'nothing happened' because the previous
  // result panel stays on screen even when the new fetch fails or returns
  // a near-identical table.
  useEffect(() => { setFgResult(null); setFgError(null); }, [fgDuration, fgEvent, fgInterest, fgGroup, fgPredictors]);
  useEffect(() => { setRmstResult(null); setRmstError(null); }, [rmstDuration, rmstEvent, rmstGroup, rmstTau]);
  useEffect(() => { setLwResult(null); setLwError(null); }, [lwId, lwStart, lwStop, lwEvent, lwPreds, lwGroup]);
  useEffect(() => { setFrailtyResult(null); setFrError(null); }, [frDuration, frEvent, frCluster, frPredictors, frDist]);

  // ── Multi-state handlers. Both endpoints share the same long-format columns.
  const msColumnsPayload = () => ({
    session_id: sid,
    id_col: msId, from_state_col: msFrom, to_state_col: msTo,
    entry_col: msEntry, exit_col: msExit, event_col: msEvent,
    predictors: msPredictors,
  });
  const msColumnsMissing = () =>
    !msId || !msFrom || !msTo || !msEntry || !msExit || !msEvent;

  const handleMultistate = async () => {
    if (msColumnsMissing()) { setMsError("Select the id, from/to state, entry, exit and event columns"); return; }
    if (msPredictors.length === 0) { setMsError("Select at least one predictor"); return; }
    setMsResult(null); setMsError(null); setMsLoading(true);
    try {
      const res = await runMultistate({ ...msColumnsPayload(), imputation: survImputation });
      setMsResult(res.data);
    } catch (e: unknown) { setMsError(errorMessage(e, "Multi-state model failed")); }
    finally { setMsLoading(false); }
  };

  const handleDynamicPrediction = async () => {
    if (msColumnsMissing()) { setMsError("Select the id, from/to state, entry, exit and event columns"); return; }
    if (msPredictors.length === 0) { setMsError("Select at least one predictor"); return; }
    const landmark = parseFloat(msLandmark);
    if (!Number.isFinite(landmark) || landmark < 0) { setMsError("Enter a landmark time ≥ 0"); return; }
    const horizon = parseFloat(msHorizon);
    if (!Number.isFinite(horizon) || horizon <= 0) { setMsError("Enter a prediction horizon > 0"); return; }
    setDpResult(null); setMsError(null); setDpLoading(true);
    try {
      const res = await runDynamicPrediction({
        ...msColumnsPayload(),
        landmark_time: landmark,
        horizon,
        current_state: parseInt(msState, 10) || 0,
        run_bootstrap: msBootstrap,
      });
      setDpResult(res.data);
    } catch (e: unknown) { setMsError(errorMessage(e, "Dynamic prediction failed")); }
    finally { setDpLoading(false); }
  };

  useEffect(() => {
    setMsResult(null); setDpResult(null); setMsError(null);
  }, [msId, msFrom, msTo, msEntry, msExit, msEvent, msPredictors]);

  const handleLWYY = async () => {
    if (!lwId || !lwStart || !lwStop || !lwEvent || lwPreds.length === 0) {
      setLwError("Select id, start, stop, event and at least one predictor."); return;
    }
    setLwResult(null); setLwError(null); setLwLoading(true);
    try {
      const res = await runRecurrentLWYY({
        session_id: sid, id_col: lwId, start_col: lwStart, stop_col: lwStop,
        event_col: lwEvent, predictors: lwPreds, group_col: lwGroup || undefined,
      });
      setLwResult(res.data);
    } catch (e: unknown) {
      const detail = errorDetail(e);
      const msg = e instanceof Error ? e.message : "LWYY failed";
      setLwError(Array.isArray(detail)
        ? detail.map((m) => (m && typeof m === "object" && "msg" in m ? String((m as { msg?: unknown }).msg) : String(m))).join(", ")
        : (typeof detail === "string" ? detail : msg));
    } finally { setLwLoading(false); }
  };
  useEffect(() => { setEvResult(null); setEvError(null); }, [evEst, evLo, evHi, evType, evP0]);
  useEffect(() => { setLmResult(null); setLmError(null); }, [lmDuration, lmEvent, lmTime, lmGroup, lmPreds]);
  useEffect(() => { setKmResult(null); setKmError(null); }, [kmDuration, kmEvent, kmGroup, kmStratify]);
  useEffect(() => { setCoxResult(null); setCoxError(null); }, [coxDuration, coxEvent, coxPreds, coxInteractions]);

  const coxUMReady = (coxUMResult?.rows?.length ?? 0) > 0 ? coxUMResult : null;
  const chForestRows = chResult?.forest_rows ?? [];

  return (
    <div className="flex gap-4 max-w-[1400px] mx-auto">
      <nav className="w-52 shrink-0">
        <div className="panel space-y-2">
          <h3 className="text-sm font-semibold text-gray-700">Method</h3>
          {SURV_METHODS.map((m) => (
            <label key={m.id} className="flex items-start gap-2 cursor-pointer group">
              <input type="radio" name="surv-method" value={m.id}
                checked={activeMethod === m.id}
                onChange={() => setActiveMethod(m.id)}
                className="accent-indigo-500 mt-0.5" />
              <span className="leading-tight">
                <span className={`block text-sm font-medium ${activeMethod === m.id ? "text-indigo-700" : "text-gray-700"}`}>{m.title}</span>
                <span className="block text-[10px] text-gray-400">{m.desc}</span>
              </span>
            </label>
          ))}
        </div>
      </nav>
      <div className="flex-1 min-w-0 space-y-3">
      {/* ── Fine-Gray ── */}
      {activeMethod === "finegray" && (
      <Section title="Fine-Gray Competing Risks" description="Cumulative incidence function with competing events (Aalen-Johansen)">
        <ThreeCol
          storageKey="SurvivalAdvanced.FineGray"
          left={
            <>
              <div className="grid grid-cols-1 gap-2">
                <VarSelect label="Duration" value={fgDuration} onChange={setFgDuration} columns={pickCols} kinds={["numeric"]} />
                <VarSelect label="Event (0=censor, 1,2..=events)" value={fgEvent} onChange={setFgEvent} columns={pickCols} />
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Event of interest</span>
                  <input type="number" value={fgInterest} onChange={(e) => setFgInterest(Number(e.target.value))} min={1}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 w-20 focus:outline-none focus:border-indigo-400" />
                </label>
                <VarSelect label="Group (optional)" value={fgGroup} onChange={setFgGroup} columns={pickCols} kinds={["categorical"]} />
              </div>
              {/* Predictors for subdistribution-hazard regression (Fine-Gray 1999) */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500 font-medium">
                    Predictors for sHR regression
                    <span className="ml-1 text-[10px] text-gray-400">(optional)</span>
                  </span>
                  {fgPredictors.length > 0 && (
                    <button onClick={() => setFgPredictors([])}
                      className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-red-50 hover:text-red-500 hover:border-red-300">
                      Clear ({fgPredictors.length})
                    </button>
                  )}
                </div>
                <input type="text"
                  placeholder="Filter columns…"
                  value={fgPredFilter}
                  onChange={(e) => setFgPredFilter(e.target.value)}
                  className="w-full text-xs border border-gray-300 rounded-lg px-3 py-1 focus:outline-none focus:border-indigo-400" />
                <div className="max-h-32 overflow-y-auto border border-gray-200 rounded-lg p-1 space-y-0.5">
                  {pickCols
                    .map((c) => c.name)
                    .filter((n: string) =>
                      n !== fgDuration && n !== fgEvent && n !== fgGroup
                      && n.toLowerCase().includes(fgPredFilter.toLowerCase()))
                    .slice(0, 100)
                    .map((n: string) => (
                      <label key={n} className="flex items-center gap-1.5 text-xs px-1 py-0.5 rounded hover:bg-gray-50 cursor-pointer">
                        <input type="checkbox" className="accent-indigo-500"
                          checked={fgPredictors.includes(n)} onChange={() => fgToggleP(n)} />
                        <span className="text-gray-700 truncate">{n}</span>
                      </label>
                    ))}
                </div>
              </div>
              <div className="flex items-center gap-3">
                {fgPredictors.length > 0 && <ImputeToggle value={survImputation} onChange={setSurvImputation} />}
                <RunButton onClick={handleFineGray} loading={fgLoading} label="Run Fine-Gray" />
              </div>
              {fgError && <p className="text-xs text-red-500">{fgError}</p>}
            </>
          }
          middle={
            fgResult?.plot ? (
              <TitledPlot
                plotRefOut={fgPlotRef}
                storageKey="surv:finegray"
                data={fgResult.plot.data}
                layout={{ ...fgResult.plot.layout, ...baseLayout, title: fgResult.plot.layout.title }}
                config={{ responsive: true }}
                defaultTitle={fgResult.plot.layout?.title?.text ?? ""}
                defaultSubtitle=""
                defaultXAxis={fgResult.plot.layout?.xaxis?.title?.text ?? ""}
                defaultYAxis={fgResult.plot.layout?.yaxis?.title?.text ?? ""}
              />
            ) : (
              <div className="flex items-center justify-center h-[400px] border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
                Run Fine-Gray to render CIF
              </div>
            )
          }
          right={
            <>
              {/* ── Subdistribution-hazard regression sub-card ── */}
              {fgResult?.regression_result && (
                <div className="border border-indigo-200 bg-indigo-50/30 rounded-lg p-3 space-y-2">
                  <div className="flex items-baseline gap-2 flex-wrap">
                    <h4 className="text-sm font-semibold text-gray-800">sHR Regression (Fine-Gray)</h4>
                  </div>
                  <p className="text-[10px] text-gray-500">{fgResult.regression_result.model}</p>
                  <div className="grid grid-cols-3 gap-1">
                    {([
                      [<i>n</i>,                  fgResult.regression_result.n],
                      ["Events",             fgResult.regression_result.n_events_of_interest],
                      ["Competing",          fgResult.regression_result.n_competing],
                      ["Censored",           fgResult.regression_result.n_censored],
                      ["C-index",            fgResult.regression_result.concordance?.toFixed(3)],
                    ] as [ReactNode, ReactNode][]).map(([k, v], i) => (
                      <div key={i} className="bg-white border border-gray-200 rounded p-1.5 text-center">
                        <p className="text-[9px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-xs">{v}</p>
                      </div>
                    ))}
                  </div>
                  <div className="overflow-auto rounded border border-gray-200 bg-white">
                    <table className="w-full text-[11px] border-collapse">
                      <thead>
                        <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                          {["Variable", "sHR", "95% CI", "p"].map((h) => (
                            <th key={h} className="px-1.5 py-1 text-left font-medium">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {fgResult.regression_result.coefficients.map((c) => (
                          <tr key={c.variable} className="border-b border-gray-100">
                            <td className="px-1.5 py-1 font-mono text-gray-800 truncate max-w-[80px]">{c.variable}</td>
                            <td className={`px-1.5 py-1 font-mono font-semibold ${c.p != null && c.p < 0.05 ? "text-indigo-700" : "text-gray-600"}`}>{c.shr?.toFixed(2)}</td>
                            <td className="px-1.5 py-1 font-mono text-gray-500">
                              {c.shr_low != null && c.shr_high != null
                                ? `[${c.shr_low.toFixed(2)}, ${c.shr_high.toFixed(2)}]`
                                : "—"}
                            </td>
                            <td className="px-1.5 py-1">
                              <span className={`inline-block font-mono px-1 py-0.5 rounded text-[10px] ${
                                c.p != null && c.p < 0.05 ? "bg-indigo-100 text-indigo-700 font-semibold" : "text-gray-400"
                              }`}>
                                {fmtP(c.p)}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-[9px] text-gray-500 leading-relaxed">{fgResult.regression_result.method_note}</p>
                </div>
              )}
              <ResultBlock result={fgResult} />
            </>
          }
        />
      </Section>
      )}

      {/* ── RMST ── */}
      {activeMethod === "rmst" && (
      <Section title="Restricted Mean Survival Time (RMST)"
        description="Average event-free time over a fixed horizon τ — PH-free alternative to the hazard ratio. Robust when curves cross or the proportional-hazards assumption fails.">
        <ThreeCol
          storageKey="SurvivalAdvanced.RMST"
          left={
            <>
              <div className="grid grid-cols-1 gap-2">
                <VarSelect label="Duration" value={rmstDuration} onChange={setRmstDuration} columns={pickCols} kinds={["numeric"]} />
                <VarSelect label="Event (0/1)" value={rmstEvent} onChange={setRmstEvent} columns={binaryCols} />
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">τ (time horizon)</span>
                  <input type="number" min="0" step="any" value={rmstTau}
                    onChange={(e) => setRmstTau(e.target.value)}
                    placeholder="e.g. 1825"
                    className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" />
                </label>
                <VarSelect label="Group (optional)" value={rmstGroup} onChange={setRmstGroup} columns={pickCols} kinds={["categorical"]} />
              </div>
              <div className="flex items-center gap-3">
                {rmstGroup && <ImputeToggle value={survImputation} onChange={setSurvImputation} />}
                <RunButton onClick={handleRMST} loading={rmstLoading} label="Run RMST" />
              </div>
              {rmstError && <p className="text-xs text-red-500">{rmstError}</p>}
            </>
          }
          middle={
            rmstResult?.plot ? (
              <TitledPlot
                plotRefOut={rmstPlotRef}
                storageKey="surv:rmst"
                data={rmstResult.plot.data}
                layout={{ ...rmstResult.plot.layout, ...baseLayout, title: rmstResult.plot.layout.title }}
                config={{ responsive: true }}
                defaultTitle={rmstResult.plot.layout?.title?.text ?? ""}
                defaultSubtitle=""
                defaultXAxis={rmstResult.plot.layout?.xaxis?.title?.text ?? ""}
                defaultYAxis={rmstResult.plot.layout?.yaxis?.title?.text ?? ""}
              />
            ) : (
              <div className="flex items-center justify-center h-[400px] border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
                Run RMST to render KM curves
              </div>
            )
          }
          right={
            <>
              {rmstResult?.rmst_by_group && (
                <div className="overflow-auto rounded-lg border border-gray-200">
                  <table className="w-full text-[11px]">
                    <thead>
                      <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                        {(["Group", <i key="n">n</i>, "RMST", "95% CI"] as ReactNode[]).map((h, i) => (
                          <th key={i} className="px-1.5 py-1.5 text-left font-medium">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(rmstResult.rmst_by_group).map(([g, v]) => (
                        <tr key={g} className="border-b border-gray-100 hover:bg-gray-50">
                          <td className="px-1.5 py-1 font-mono text-gray-800 truncate max-w-[60px]">{g}</td>
                          <td className="px-1.5 py-1 font-mono text-gray-600">{v.n}</td>
                          <td className="px-1.5 py-1 font-mono font-semibold text-indigo-700">{v.rmst}</td>
                          <td className="px-1.5 py-1 font-mono text-gray-500">[{v.ci_low}, {v.ci_high}]</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {rmstResult?.rmst_mi_note && (
                <div className="text-[10px] text-teal-700 bg-teal-50 border border-teal-200 rounded px-2 py-1 mb-1">{rmstResult.rmst_mi_note}</div>
              )}
              {rmstResult?.contrasts && rmstResult.contrasts.length > 0 && (
                <div className="overflow-auto rounded-lg border border-indigo-200 bg-indigo-50/30">
                  <table className="w-full text-[11px]">
                    <thead>
                      <tr className="bg-white border-b border-indigo-200 text-gray-600">
                        {["A", "B", "ΔRMST", "p"].map((h) => (
                          <th key={h} className="px-1.5 py-1.5 text-left font-semibold">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rmstResult.contrasts.map((c, i) => (
                        <tr key={i} className={`border-b border-indigo-100 ${c.p != null && c.p < 0.05 ? "bg-indigo-50/60" : ""}`}>
                          <td className="px-1.5 py-1 font-mono text-gray-800 truncate max-w-[50px]">{c.group_a}</td>
                          <td className="px-1.5 py-1 font-mono text-gray-800 truncate max-w-[50px]">{c.group_b}</td>
                          <td className={`px-1.5 py-1 font-mono font-semibold ${c.p != null && c.p < 0.05 ? "text-indigo-700" : "text-gray-700"}`}>{c.delta_rmst}</td>
                          <td className="px-1.5 py-1">
                            <span className={`inline-block font-mono px-1 py-0.5 rounded text-[10px] ${
                              c.p != null && c.p < 0.05 ? "bg-indigo-100 text-indigo-700 font-semibold" : "text-gray-400"
                            }`}>
                              {fmtP(c.p)}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <ResultBlock result={rmstResult} />
            </>
          }
        />
      </Section>
      )}

      {/* ── Shared frailty — clustered / multi-centre survival ── */}
      {activeMethod === "frailty" && (
      <Section title="Shared Frailty Model"
        description="Cox model with a shared random effect (frailty) per cluster — for multi-centre trials, repeated measures within families, or any design where subjects are nested. θ quantifies the within-cluster dependence remaining after the fixed effects.">
        <ThreeCol
          storageKey="SurvivalAdvanced.Frailty"
          left={
            <>
              <div className="grid grid-cols-1 gap-2">
                <VarSelect label="Duration (time)" value={frDuration} onChange={setFrDuration} columns={pickCols} kinds={["numeric"]} />
                <VarSelect label="Event (0/1)" value={frEvent} onChange={setFrEvent} columns={binaryCols} />
                <VarSelect label="Cluster (centre / family)" value={frCluster} onChange={setFrCluster} columns={pickCols} kinds={["categorical"]} />
                <MultiSelect label="Predictors" columns={pickCols} selected={frPredictors} onChange={setFrPredictors}
                  excludeNames={[frDuration, frEvent, frCluster].filter(Boolean)} />
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Frailty distribution</span>
                  <select value={frDist} onChange={(e) => setFrDist(e.target.value)}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400">
                    <option value="gamma">Gamma</option>
                    <option value="inverse_gaussian">Inverse Gaussian</option>
                  </select>
                </label>
              </div>
              <div className="flex items-center gap-3">
                {frPredictors.length > 0 && <ImputeToggle value={survImputation} onChange={setSurvImputation} />}
                <RunButton onClick={handleFrailty} loading={frLoading} label="Run Shared Frailty" />
              </div>
              {frError && <p className="text-xs text-red-500">{frError}</p>}
            </>
          }
          middle={
            frailtyResult?.plot ? (
              <TitledPlot
                plotRefOut={frPlotRef}
                storageKey="surv:frailty"
                data={frailtyResult.plot.data}
                layout={{ ...frailtyResult.plot.layout, ...baseLayout, title: frailtyResult.plot.layout.title }}
                config={{ responsive: true }}
                defaultTitle={frailtyResult.plot.layout?.title?.text ?? ""}
                defaultSubtitle=""
                defaultXAxis={frailtyResult.plot.layout?.xaxis?.title?.text ?? ""}
                defaultYAxis={frailtyResult.plot.layout?.yaxis?.title?.text ?? ""}
              />
            ) : (
              <div className="flex items-center justify-center h-[400px] border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
                Run the shared frailty model to render the cluster frailty distribution
              </div>
            )
          }
          right={
            <>
              {frailtyResult && (
                <div className="overflow-auto rounded-lg border border-gray-200">
                  <table className="w-full text-[11px]">
                    <tbody>
                      <tr className="border-b border-gray-100">
                        <td className="px-1.5 py-1 text-gray-500">Subjects</td>
                        <td className="px-1.5 py-1 font-mono text-gray-800">{frailtyResult.n_subjects ?? "—"}</td>
                      </tr>
                      <tr className="border-b border-gray-100">
                        <td className="px-1.5 py-1 text-gray-500">Clusters</td>
                        <td className="px-1.5 py-1 font-mono text-gray-800">{frailtyResult.n_clusters ?? "—"}</td>
                      </tr>
                      <tr className="border-b border-gray-100">
                        <td className="px-1.5 py-1 text-gray-500">Events</td>
                        <td className="px-1.5 py-1 font-mono text-gray-800">{frailtyResult.n_events ?? "—"}</td>
                      </tr>
                      <tr className="border-b border-gray-100">
                        <td className="px-1.5 py-1 text-gray-500">Frailty variance θ</td>
                        <td className="px-1.5 py-1 font-mono font-semibold text-indigo-700">
                          {frailtyResult.theta ?? "—"}
                          {frailtyResult.frailty_variance_test?.chi_bar_square_p != null && (
                            <span className="ml-1.5 font-normal text-gray-500">
                              (<i>p</i> = {fmtP(frailtyResult.frailty_variance_test.chi_bar_square_p)})
                            </span>
                          )}
                        </td>
                      </tr>
                      <tr>
                        <td className="px-1.5 py-1 text-gray-500">Concordance</td>
                        <td className="px-1.5 py-1 font-mono text-gray-800">{frailtyResult.concordance ?? "—"}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}
              {frailtyResult?.coefficients && frailtyResult.coefficients.length > 0 && (
                <div className="overflow-auto rounded-lg border border-indigo-200 bg-indigo-50/30 mt-2">
                  <table className="w-full text-[11px]">
                    <thead>
                      <tr className="bg-white border-b border-indigo-200 text-gray-600">
                        {(["Variable", "HR", "95% CI", <i key="p">p</i>] as ReactNode[]).map((h, i) => (
                          <th key={i} className="px-1.5 py-1.5 text-left font-semibold">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {frailtyResult.coefficients.map((c, i) => {
                        // Backend returns log-HR (`estimate`) + `se` but no CI —
                        // derive the Wald 95% CI on the log scale, exponentiate.
                        const hasCI = c.estimate != null && c.se != null && Number.isFinite(c.se);
                        const lo = hasCI ? Math.exp(c.estimate! - 1.96 * c.se!) : null;
                        const hi = hasCI ? Math.exp(c.estimate! + 1.96 * c.se!) : null;
                        const sig = c.p != null && c.p < 0.05;
                        return (
                          <tr key={i} className={`border-b border-indigo-100 ${sig ? "bg-indigo-50/60" : ""}`}>
                            <td className="px-1.5 py-1 font-mono text-gray-800 truncate max-w-[70px]">{c.variable}</td>
                            <td className={`px-1.5 py-1 font-mono font-semibold ${sig ? "text-indigo-700" : "text-gray-700"}`}>
                              {c.hr != null ? c.hr : "—"}
                            </td>
                            <td className="px-1.5 py-1 font-mono text-gray-500">
                              {lo != null && hi != null ? `[${lo.toFixed(2)}, ${hi.toFixed(2)}]` : "—"}
                            </td>
                            <td className="px-1.5 py-1">
                              <span className={`inline-block font-mono px-1 py-0.5 rounded text-[10px] ${
                                sig ? "bg-indigo-100 text-indigo-700 font-semibold" : "text-gray-400"
                              }`}>
                                {fmtPubP(c.p)}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
              {frailtyResult?.warnings && frailtyResult.warnings.length > 0 && (
                <div className="mt-2 space-y-1">
                  {frailtyResult.warnings.map((w, i) => (
                    <div key={i} className="text-[10px] text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1">{w}</div>
                  ))}
                </div>
              )}
              {frailtyResult?.method_note && (
                <p className="text-[10px] text-gray-500 mt-2 leading-snug">{frailtyResult.method_note}</p>
              )}
              <ResultBlock result={frailtyResult} />
            </>
          }
        />
      </Section>
      )}

      {/* ── Multi-state transitions + landmark dynamic prediction ── */}
      {activeMethod === "multistate" && (
      <Section title="Multi-state Model"
        description="Transition-specific models across clinical states, plus state-occupancy prediction from a landmark time.">
        {/* Every other method here takes one row per subject; this one does not,
            so spell the layout out rather than let users feed it the wrong shape. */}
        <div className="mb-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-800 leading-relaxed">
          <strong>Needs long-format transition data</strong> — one row per state transition,
          not one row per subject. Columns: subject id, from-state, to-state, entry time,
          exit time, event (1 = the transition happened, 0 = censored), plus predictors.
          Example with 0 = healthy, 1 = illness, 2 = death: a patient who fell ill at t=2
          and died at t=5 contributes two rows — <code>0→1</code> (entry 0, exit 2, event 1)
          and <code>1→2</code> (entry 2, exit 5, event 1).
        </div>
        <ThreeCol
          storageKey="SurvivalAdvanced.Multistate"
          left={
            <>
              <VarSelect label="Subject id" value={msId} onChange={setMsId} columns={pickCols} />
              <VarSelect label="From state" value={msFrom} onChange={setMsFrom} columns={pickCols} />
              <VarSelect label="To state" value={msTo} onChange={setMsTo} columns={pickCols} />
              <VarSelect label="Entry time" value={msEntry} onChange={setMsEntry} columns={pickCols} />
              <VarSelect label="Exit time" value={msExit} onChange={setMsExit} columns={pickCols} />
              <VarSelect label="Event (1 = transition occurred)" value={msEvent} onChange={setMsEvent} columns={pickCols} />
              <MultiSelect label="Predictors" columns={pickCols} selected={msPredictors}
                onChange={setMsPredictors}
                excludeNames={[msId, msFrom, msTo, msEntry, msExit, msEvent].filter(Boolean)} />
              {msPredictors.length > 0 && <ImputeToggle value={survImputation} onChange={setSurvImputation} />}
              <RunButton onClick={handleMultistate} loading={msLoading} label="Fit transition models" />

              <div className="mt-3 pt-3 border-t border-gray-200 space-y-2">
                <p className="text-xs font-semibold text-gray-600">Landmark prediction</p>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Landmark time</span>
                  <input className="text-sm border border-gray-300 rounded-lg px-3 py-2"
                    value={msLandmark} onChange={(e) => setMsLandmark(e.target.value)} placeholder="e.g. 2" />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Horizon</span>
                  <input className="text-sm border border-gray-300 rounded-lg px-3 py-2"
                    value={msHorizon} onChange={(e) => setMsHorizon(e.target.value)} />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">State at the landmark</span>
                  <input className="text-sm border border-gray-300 rounded-lg px-3 py-2"
                    value={msState} onChange={(e) => setMsState(e.target.value)} />
                </label>
                <label className="flex items-center gap-2 text-xs text-gray-600">
                  <input type="checkbox" className="accent-indigo-500"
                    checked={msBootstrap} onChange={(e) => setMsBootstrap(e.target.checked)} />
                  Bootstrap CIs (slower)
                </label>
                <RunButton onClick={handleDynamicPrediction} loading={dpLoading} label="Predict from landmark" />
              </div>
              {msError && <p className="text-xs text-red-500 mt-2">{msError}</p>}
            </>
          }
          middle={
            dpResult?.state_probabilities && dpResult.times ? (
              <TitledPlot
                plotRefOut={dpPlotRef}
                storageKey="surv:multistate"
                data={Object.entries(dpResult.state_probabilities).map(([state, ys], i) => ({
                  x: dpResult.times, y: ys, type: "scatter", mode: "lines",
                  name: state.replace("state_", "State "),
                  line: { color: pal[i % pal.length], width: 2 },
                  ...traceDefaults,
                }))}
                layout={{
                  ...baseLayout,
                  title: `State occupancy from landmark t = ${dpResult.landmark_time}`,
                  xaxis: { title: "Time" },
                  yaxis: { title: "Probability", range: [0, 1] },
                }}
                config={{ responsive: true }}
                defaultTitle={`State occupancy from landmark t = ${dpResult.landmark_time}`}
                defaultSubtitle=""
                defaultXAxis="Time"
                defaultYAxis="Probability"
              />
            ) : (
              <div className="h-full min-h-[300px] flex items-center justify-center border-2 border-dashed border-gray-200 rounded-xl">
                <p className="text-xs text-gray-400 text-center px-6">
                  Run the landmark prediction to see state-occupancy curves.
                </p>
              </div>
            )
          }
          right={
            <>
              {msResult?.results && (
                <div className="space-y-3">
                  <p className="text-xs font-semibold text-gray-600">
                    Transitions estimated: {(msResult.transitions_estimated ?? []).join(", ") || "—"}
                  </p>
                  {Object.entries(msResult.results).map(([trans, fit]) => (
                    <div key={trans} className="border border-gray-200 rounded-lg overflow-hidden">
                      <div className="bg-gray-50 px-2 py-1 text-xs font-semibold text-gray-700">
                        {trans}
                        <span className="ml-2 font-normal text-gray-500">
                          {fit.n_events ?? "—"} events / {fit.n_transitions ?? "—"} records
                          {fit.concordance != null && ` · C = ${fit.concordance.toFixed(3)}`}
                        </span>
                      </div>
                      <table className="w-full text-[11px]">
                        <thead><tr className="bg-white text-gray-400">
                          <th className="px-2 py-1 text-left">Variable</th>
                          <th className="px-2 py-1 text-right">HR</th>
                          <th className="px-2 py-1 text-right">SE</th>
                          <th className="px-2 py-1 text-right"><i>p</i></th>
                        </tr></thead>
                        <tbody>
                          {(fit.coefficients ?? []).map((c) => (
                            <tr key={c.variable} className="border-t border-gray-100">
                              <td className="px-2 py-1">{c.variable}</td>
                              <td className="px-2 py-1 text-right font-mono">{c.hr?.toFixed(4) ?? "—"}</td>
                              <td className="px-2 py-1 text-right font-mono">{c.se?.toFixed(4) ?? "—"}</td>
                              <td className="px-2 py-1 text-right font-mono">{fmtPubP(c.p)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ))}
                </div>
              )}
              {msResult?.markov_assumption_tests && (
                <div className="space-y-1">
                  <p className="text-xs font-semibold text-gray-600">Markov assumption</p>
                  {Object.entries(msResult.markov_assumption_tests).map(([trans, t]) => (
                    <div key={trans} className={`rounded-lg px-2 py-1.5 text-[11px] ${
                      t.status === "Tested"
                        ? (t.markov_assumption_violated ? "bg-amber-50 text-amber-800" : "bg-green-50 text-green-800")
                        : "bg-gray-50 text-gray-500"}`}>
                      <span className="font-semibold">{trans}</span>{" — "}
                      {t.status === "Tested"
                        ? <>{t.markov_assumption_violated ? "violated" : "not violated"} ({fmtPubP(t.p_value)})</>
                        : (t.reason ?? t.status)}
                    </div>
                  ))}
                </div>
              )}
              <ResultBlock result={msResult} />

              {dpResult && (
                <div className="space-y-2 pt-2 border-t border-gray-200">
                  <p className="text-xs font-semibold text-gray-600">
                    Landmark prediction — {dpResult.n_at_risk ?? "—"} at risk in state {dpResult.current_state}
                  </p>
                  {dpResult.elos && (
                    <table className="w-full text-[11px]">
                      <thead><tr className="text-gray-400">
                        <th className="px-2 py-1 text-left">State</th>
                        <th className="px-2 py-1 text-right">Expected time in state</th>
                      </tr></thead>
                      <tbody>
                        {Object.entries(dpResult.elos).map(([st, v]) => (
                          <tr key={st} className="border-t border-gray-100">
                            <td className="px-2 py-1">State {st}</td>
                            <td className="px-2 py-1 text-right font-mono">{Number(v).toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                  {msBootstrap && dpResult.bootstrap && Object.keys(dpResult.bootstrap).length === 0 && (
                    <p className="text-[11px] text-gray-400">Bootstrap produced no intervals for this fit.</p>
                  )}
                  <ResultBlock result={dpResult} />
                </div>
              )}
            </>
          }
        />
      </Section>
      )}

      {/* ── Recurrent events — LWYY ── */}
      {activeMethod === "lwyy" && (
      <Section title="Recurrent Events (LWYY)"
        description="Modified Andersen-Gill model with Lin-Wei-Yang-Ying cluster-robust SE for recurrent events (e.g. repeat hospitalisations). Counting-process (start, stop, event] intervals; exp(β) = rate ratio.">
        <ThreeCol
          storageKey="SurvivalAdvanced.LWYY"
          left={
            <>
              <div className="grid grid-cols-1 gap-2">
                <VarSelect label="Subject id" value={lwId} onChange={setLwId} columns={pickCols} />
                <VarSelect label="Start (interval entry)" value={lwStart} onChange={setLwStart} columns={pickCols} kinds={["numeric"]} />
                <VarSelect label="Stop (interval / event time)" value={lwStop} onChange={setLwStop} columns={pickCols} kinds={["numeric"]} />
                <VarSelect label="Event (1 = event at stop)" value={lwEvent} onChange={setLwEvent} columns={pickCols} />
                <VarSelect label="Group for MCF plot (optional)" value={lwGroup} onChange={setLwGroup} columns={pickCols} kinds={["categorical"]} />
              </div>
              <MultiSelect label="Predictors" columns={pickCols} selected={lwPreds} onChange={setLwPreds}
                excludeNames={[lwId, lwStart, lwStop, lwEvent].filter(Boolean)} />
              <div className="flex items-center gap-3">
                <RunButton onClick={handleLWYY} loading={lwLoading} label="Run LWYY" />
              </div>
              {lwError && <p className="text-xs text-red-500">{lwError}</p>}
            </>
          }
          middle={
            lwResult?.plot ? (
              <TitledPlot
                plotRefOut={lwPlotRef}
                storageKey="surv:lwyy"
                data={lwResult.plot.data}
                layout={{ ...lwResult.plot.layout, ...baseLayout, title: lwResult.plot.layout.title }}
                config={{ responsive: true }}
                defaultTitle={lwResult.plot.layout?.title?.text ?? ""}
                defaultSubtitle=""
                defaultXAxis={lwResult.plot.layout?.xaxis?.title?.text ?? ""}
                defaultYAxis={lwResult.plot.layout?.yaxis?.title?.text ?? ""}
              />
            ) : (
              <div className="flex items-center justify-center h-[400px] border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
                Run LWYY to render the mean cumulative function
              </div>
            )
          }
          right={
            <>
              {lwResult && (
                <div className="border border-indigo-200 bg-indigo-50/30 rounded-lg p-3 space-y-2">
                  <h4 className="text-sm font-semibold text-gray-800">Rate-ratio table</h4>
                  <p className="text-[10px] text-gray-500">{lwResult.model}</p>
                  <div className="grid grid-cols-3 gap-1">
                    {[
                      ["Subjects", lwResult.n_subjects],
                      ["Events", lwResult.n_events],
                      ["Ev/subj", lwResult.events_per_subject?.toFixed(2)],
                    ].map(([k, v]) => (
                      <div key={String(k)} className="bg-white border border-gray-200 rounded p-1.5 text-center">
                        <p className="text-[9px] text-gray-400">{k}</p>
                        <p className="font-semibold text-gray-800 text-xs">{v}</p>
                      </div>
                    ))}
                  </div>
                  <div className="overflow-auto rounded border border-gray-200 bg-white">
                    <table className="w-full text-[11px] border-collapse">
                      <thead>
                        <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                          {["Variable", "RR", "95% CI", "p"].map((h) => (
                            <th key={h} className="px-1.5 py-1 text-left font-medium">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {lwResult.coefficients?.map((c) => (
                          <tr key={c.variable} className="border-b border-gray-100">
                            <td className="px-1.5 py-1 font-mono text-gray-800 truncate max-w-[80px]">{c.variable}</td>
                            <td className={`px-1.5 py-1 font-mono font-semibold ${c.p != null && c.p < 0.05 ? "text-indigo-700" : "text-gray-600"}`}>{c.rate_ratio?.toFixed(2)}</td>
                            <td className="px-1.5 py-1 font-mono text-gray-500">[{c.rr_low?.toFixed(2)}, {c.rr_high?.toFixed(2)}]</td>
                            <td className="px-1.5 py-1">
                              <span className={`inline-block font-mono px-1 py-0.5 rounded text-[10px] ${c.p != null && c.p < 0.05 ? "bg-indigo-100 text-indigo-700 font-semibold" : "text-gray-400"}`}>
                                {fmtP(c.p)}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              <ResultBlock result={lwResult} />
            </>
          }
        />
      </Section>
      )}

      {/* ── Interval-censored ── */}
      {activeMethod === "intervalcensored" && (
        <IntervalCensoredPanel session={session} />
      )}

      {/* ── E-value ── */}
      {activeMethod === "evalue" && (
      <Section title="E-value (Unmeasured Confounding)" description="Quantify the minimum confounding strength to explain away an observed effect">
        <div className="grid grid-cols-5 gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Measure</span>
            <select value={evType} onChange={(e) => setEvType(e.target.value)}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400">
              <option value="OR">OR</option>
              <option value="HR">HR</option>
              <option value="RR">RR</option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Estimate</span>
            <input type="number" step="0.01" value={evEst} onChange={(e) => setEvEst(e.target.value)}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" placeholder="e.g. 2.5" />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">CI Low</span>
            <input type="number" step="0.01" value={evLo} onChange={(e) => setEvLo(e.target.value)}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" placeholder="e.g. 1.2" />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">CI High</span>
            <input type="number" step="0.01" value={evHi} onChange={(e) => setEvHi(e.target.value)}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" placeholder="e.g. 5.1" />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Baseline risk (p₀)</span>
            <input type="number" step="0.01" value={evP0} onChange={(e) => setEvP0(e.target.value)} min={0.01} max={0.99}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" />
          </label>
        </div>
        <div className="flex items-center gap-3">
          <RunButton onClick={handleEValue} loading={evLoading} label="Calculate E-value" />
          {evError && <p className="text-xs text-red-500">{evError}</p>}
        </div>
        {evResult && (
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-4 py-3 text-center">
              <p className="text-[10px] text-indigo-400 uppercase tracking-wider font-semibold">E-value (point)</p>
              <p className="text-3xl font-bold text-indigo-700 mt-1">{evResult.evalue_point}</p>
            </div>
            <div className="bg-violet-50 border border-violet-200 rounded-xl px-4 py-3 text-center">
              <p className="text-[10px] text-violet-400 uppercase tracking-wider font-semibold">E-value (CI)</p>
              <p className="text-3xl font-bold text-violet-700 mt-1">{evResult.evalue_ci}</p>
            </div>
          </div>
        )}
        {evResult?.interpretation && (
          <div className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 text-sm text-gray-700">
            {evResult.interpretation}
          </div>
        )}
        <ResultBlock result={evResult} />
      </Section>
      )}

      {/* ── Landmark ── */}
      {activeMethod === "landmark" && (
      <Section title="Landmark Survival Analysis" description="Survival analysis conditional on surviving beyond a landmark time point">
        <ThreeCol
          storageKey="SurvivalAdvanced.Landmark"
          left={
            <>
              <div className="grid grid-cols-1 gap-2">
                <VarSelect label="Duration" value={lmDuration} onChange={setLmDuration} columns={pickCols} kinds={["numeric"]} />
                <VarSelect label="Event (0/1)" value={lmEvent} onChange={setLmEvent} columns={binaryCols} />
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-gray-500 font-medium">Landmark time</span>
                  <input type="number" step="1" value={lmTime} onChange={(e) => setLmTime(e.target.value)}
                    className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-400" placeholder="e.g. 30" />
                </label>
                <VarSelect label="Group (optional)" value={lmGroup} onChange={setLmGroup} columns={pickCols} kinds={["categorical"]} />
              </div>
              <MultiSelect label="Predictors for Cox (optional)" columns={pickCols} selected={lmPreds} onChange={setLmPreds} excludeNames={[lmDuration, lmEvent].filter(Boolean)} />
              <div className="flex items-center gap-3">
                {lmPreds.length > 0 && <ImputeToggle value={survImputation} onChange={setSurvImputation} />}
                <RunButton onClick={handleLandmark} loading={lmLoading} label="Run Landmark" />
              </div>
              {lmError && <p className="text-xs text-red-500">{lmError}</p>}
            </>
          }
          middle={
            lmResult?.plot ? (
              <TitledPlot
                plotRefOut={lmPlotRef}
                storageKey="surv:landmark"
                data={lmResult.plot.data}
                layout={{ ...lmResult.plot.layout, ...baseLayout, title: lmResult.plot.layout.title }}
                config={{ responsive: true }}
                defaultTitle={lmResult.plot.layout?.title?.text ?? ""}
                defaultSubtitle=""
                defaultXAxis={lmResult.plot.layout?.xaxis?.title?.text ?? ""}
                defaultYAxis={lmResult.plot.layout?.yaxis?.title?.text ?? ""}
              />
            ) : (
              <div className="flex items-center justify-center h-[400px] border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
                Run Landmark to render KM
              </div>
            )
          }
          right={
            <>
              {lmResult?.cox_mi_note && (
                <div className="text-[10px] text-teal-700 bg-teal-50 border border-teal-200 rounded px-2 py-1 mb-1">{lmResult.cox_mi_note}</div>
              )}
              {lmResult?.cox_results && lmResult.cox_results.length > 0 && !lmResult.cox_results[0].error && (
                <div className="overflow-auto rounded-lg border border-gray-200">
                  <table className="text-[11px] w-full">
                    <thead>
                      <tr className="bg-gray-50">
                        <th className="px-1.5 py-1.5 text-left text-gray-500">Variable</th>
                        <th className="px-1.5 py-1.5 text-left text-gray-500">HR</th>
                        <th className="px-1.5 py-1.5 text-left text-gray-500">95% CI</th>
                        <th className="px-1.5 py-1.5 text-left text-gray-500"><i>p</i></th>
                        {lmResult.cox_results[0]?.fmi != null && <th className="px-1.5 py-1.5 text-right text-gray-500" title="Fraction of missing information">FMI</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {lmResult.cox_results.map((r, i) => (
                        <tr key={i} className="border-t border-gray-100">
                          <td className="px-1.5 py-1 text-gray-700 font-medium truncate max-w-[80px]">{r.variable}</td>
                          <td className="px-1.5 py-1 text-gray-700 font-mono">{r.HR}</td>
                          <td className="px-1.5 py-1 text-gray-500 font-mono">{r.ci_low}–{r.ci_high}</td>
                          <td className={`px-1.5 py-1 font-mono ${r.p != null && r.p < 0.05 ? "text-indigo-600 font-semibold" : "text-gray-500"}`}>
                            {fmtP(r.p)}
                          </td>
                          {r.fmi != null && <td className="px-1.5 py-1 text-right text-gray-400 font-mono">{r.fmi}</td>}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <ResultBlock result={lmResult} />
            </>
          }
        />
      </Section>
      )}

      {/* ── Kaplan-Meier ── */}
      {activeMethod === "km" && (
      <Section title="Kaplan-Meier Survival" description="Visualise time-to-event data with survival curves and log-rank test">
        <div className="grid grid-cols-4 gap-3">
          <VarSelect label="Duration (time)" value={kmDuration} onChange={setKmDuration} columns={pickCols} kinds={["numeric"]} />
          <VarSelect label="Event (0/1)" value={kmEvent} onChange={setKmEvent} columns={binaryCols} />
          <VarSelect label="Group (optional)" value={kmGroup} onChange={setKmGroup} columns={pickCols} kinds={["categorical"]} />
          <VarSelect label="Stratify by (optional)" value={kmStratify} onChange={setKmStratify} columns={pickCols} kinds={["categorical"]} />
        </div>

        {/* Landmark survival + pairwise log-rank options */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">
              Survival at time(s) — comma-sep
              <Tip wide text="Landmark survival probabilities: the KM survival estimate (+ 95% CI) read off each curve at the time points you list, in the Duration column's unit. Days → '1825' gives 5-year survival; months → '60'; years → '5'. Multiple values allowed (e.g. '365, 1825'). Reported as e.g. '77.0% (95% CI 70–84)'." />
            </span>
            <input value={kmSurvTimes} onChange={(e) => setKmSurvTimes(e.target.value)}
              placeholder="365, 1825"
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400" />
            <span className="text-[10px] text-gray-400">In the Duration unit. Days → 1825 = 5-year survival.</span>
          </label>
          <label className="flex items-center gap-2 text-xs text-gray-600 pb-2">
            <input type="checkbox" checked={kmPairwise} onChange={(e) => setKmPairwise(e.target.checked)} className="accent-indigo-500" />
            Pairwise log-rank (≥3 groups)
            <Tip wide text="With 3+ groups the overall log-rank only says 'some difference exists'. Pairwise runs a log-rank for every group pair so you can state which pair drives it (e.g. '<100 vs 100–130, p=0.003; 100–130 vs >130, p=0.46'). Always apply a multiplicity correction → for ≥3 comparisons." />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">
              Multiplicity correction
              <Tip wide text="Adjusts pairwise p-values for testing several pairs at once (controls false positives). Holm = uniformly more powerful than Bonferroni, recommended default for confirmatory pairwise comparisons. Bonferroni = most conservative. Benjamini-Hochberg = controls false-discovery rate, for exploratory work. None = raw p (report only if pre-specified)." />
            </span>
            <select value={kmCorrection} onChange={(e) => setKmCorrection(e.target.value)} disabled={!kmPairwise}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400 disabled:opacity-50">
              <option value="none">None (raw p)</option>
              <option value="holm">Holm</option>
              <option value="bonferroni">Bonferroni</option>
              <option value="bh">Benjamini-Hochberg (FDR)</option>
            </select>
          </label>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <RunButton onClick={async () => {
            if (!kmDuration || !kmEvent) { setKmError("Select duration and event columns"); return; }
            setKmResult(null); setKmError(null); setKmLoading(true);
            try {
              const survTimes = kmSurvTimes.split(",").map((s) => parseFloat(s.trim())).filter((x) => !Number.isNaN(x) && x > 0);
              const res = await runKM({
                session_id: sid, duration_col: kmDuration, event_col: kmEvent,
                group_col: kmGroup || undefined, stratify_col: kmStratify || undefined,
                survival_times: survTimes.length ? survTimes : undefined,
                pairwise: kmPairwise && !!kmGroup,
                pairwise_correction: kmCorrection,
                include_censors: kmShowCensors,
              });
              setKmResult(res.data);
            } catch (e: unknown) { setKmError(errorMessage(e, "KM failed")); }
            finally { setKmLoading(false); }
          }} loading={kmLoading} label="Run Kaplan-Meier" />

          {/* Log-rank screening button */}
          {kmDuration && kmEvent && (
            <button
              disabled={kmScanLoading}
              onClick={async () => {
                const catCols = pickCols.filter((c) => isCategoricalKind(c.kind)).map((c) => c.name);
                if (catCols.length === 0) return;
                setKmScanLoading(true);
                const results: KMScanRow[] = [];
                for (const col of catCols) {
                  try {
                    const res = await runKM({ session_id: sid, duration_col: kmDuration, event_col: kmEvent, group_col: col });
                    results.push({
                      variable: col,
                      groups: res.data.groups?.length ?? 0,
                      logrank_p: res.data.logrank?.p ?? null,
                      chi2: res.data.logrank?.chi2 ?? null,
                    });
                  } catch { results.push({ variable: col, groups: null, logrank_p: null, chi2: null }); }
                }
                results.sort((a, b) => (a.logrank_p ?? 1) - (b.logrank_p ?? 1));
                setKmScanResult(results);
                setKmScanLoading(false);
              }}
              className="px-3 py-1.5 text-xs font-medium border border-indigo-300 text-indigo-600 rounded-lg hover:bg-indigo-50 disabled:opacity-50 transition-colors"
            >
              {kmScanLoading ? "Scanning…" : "🔍 Log-rank Scan"}
            </button>
          )}
          {kmError && <p className="text-xs text-red-500">{kmError}</p>}
        </div>

        {/* KM scan results */}
        {kmScanResult.length > 0 && (
          <div className="rounded-lg border border-gray-200 overflow-auto">
            <div className="bg-gray-50 px-3 py-2 border-b border-gray-200 flex items-center justify-between">
              <p className="text-xs font-semibold text-gray-600">Log-rank Scan — All Categorical Variables</p>
              <button onClick={() => setKmScanResult([])} className="text-[10px] text-gray-400 hover:text-red-500">✕ Close</button>
            </div>
            <table className="text-xs w-full">
              <thead><tr className="bg-gray-50">
                <th className="px-3 py-1.5 text-left text-gray-500">Variable</th>
                <th className="px-3 py-1.5 text-left text-gray-500">Groups</th>
                <th className="px-3 py-1.5 text-left text-gray-500">χ²</th>
                <th className="px-3 py-1.5 text-left text-gray-500">Log-rank p</th>
                <th className="px-3 py-1.5 text-left text-gray-500"></th>
              </tr></thead>
              <tbody>
                {kmScanResult.map((r, i) => (
                  <tr key={i} className={`border-t border-gray-100 ${r.logrank_p !== null && r.logrank_p < 0.05 ? "bg-indigo-50" : ""}`}>
                    <td className="px-3 py-1 font-medium text-gray-700">{r.variable}</td>
                    <td className="px-3 py-1 text-gray-500">{r.groups ?? "—"}</td>
                    <td className="px-3 py-1 text-gray-500">{r.chi2 != null ? r.chi2.toFixed(3) : "—"}</td>
                    <td className={`px-3 py-1 font-semibold ${r.logrank_p !== null && r.logrank_p < 0.05 ? "text-indigo-700" : "text-gray-500"}`}>
                      {r.logrank_p !== null ? fmtP(r.logrank_p) : "error"}
                    </td>
                    <td className="px-3 py-1">
                      {r.logrank_p !== null && r.logrank_p < 0.05 && (
                        <button onClick={() => { setKmGroup(r.variable); setKmScanResult([]); }}
                          className="text-[10px] text-indigo-500 hover:text-indigo-700 underline">
                          Add to chart
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {kmResult?.groups && (() => {
          // Resolve group display name: custom rename > value_labels > raw value
          const groupColMeta = columns.find((c) => c.name === kmGroup);
          const vLabels = groupColMeta?.value_labels ?? {};
          const resolveGroupName = (raw: string) =>
            kmGroupLabels[raw] ?? vLabels[raw] ?? raw;

          // Build legend label per group
          const legendLabel = (g: KMGroup) => {
            const resolved = resolveGroupName(String(g.group));
            const nSuffix = kmShowNInLegend ? ` (<i>n</i>=${g.n})` : "";
            if (!kmGroup) return `${resolved}${nSuffix}`;
            if (kmHidePrefix) return `${resolved}${nSuffix}`;
            return `${kmCustomGroupTitle || kmGroup} = ${resolved}${nSuffix}`;
          };

          // Compute Y-axis range
          let yRange: [number, number] = [0, 1.05];
          if (kmAutoZoomY) {
            let minSurv = 1;
            for (const g of kmResult.groups) {
              for (const p of g.curve) {
                if (typeof p.survival === "number" && p.survival < minSurv) minSurv = p.survival;
              }
            }
            const floor = Math.max(0, Math.floor((minSurv - 0.02) * 20) / 20);
            yRange = [floor, 1.0];
          }

          // Build plot title
          const lrP = kmResult.logrank?.p;
          const pStr = lrP == null ? null : fmtP(lrP);
          const baseTitle = kmCustomPlotTitle || (kmGroup ? `Kaplan–Meier survival by ${kmCustomGroupTitle || kmGroup}` : "Kaplan–Meier survival");
          const titleText = (kmShowPInTitle && pStr) ? `${baseTitle} (log-rank p=${pStr})` : baseTitle;

          // Number-at-risk rendered INSIDE the figure (Plotly annotations) so it
          // is captured by PNG/TIFF/SVG export — a separate HTML table would not
          // appear in the exported image. Group labels sit in the left margin
          // (resolveGroupName: manual override > value_labels > raw); counts are
          // placed at each risk time under the curve.
          const riskGroups = kmResult.groups.filter((g) => Array.isArray(g.at_risk));
          const riskTimes: number[] = Array.isArray(kmResult.risk_times) ? kmResult.risk_times : [];
          const showRisk = kmRiskTable && riskTimes.length > 0 && riskGroups.length > 0;
          const RISK_ROW_PX = 17;
          const RISK_HEADER_Y = 54;   // px below plot area: "NUMBER AT RISK" header
          const RISK_FIRST_ROW_Y = 72; // px below plot area: first group row
          const riskAnnotations: Record<string, unknown>[] = [];
          let riskLeftMargin = 68;
          let riskBottomMargin = 56;
          if (showRisk) {
            const riskLabels = riskGroups.map((g) => resolveGroupName(String(g.group)));
            const maxLen = riskLabels.reduce((m: number, s: string) => Math.max(m, s.length), 0);
            riskLeftMargin = Math.min(220, Math.max(80, maxLen * 6 + 16));
            riskBottomMargin = RISK_FIRST_ROW_Y + riskGroups.length * RISK_ROW_PX + 18;
            // Header (left margin)
            riskAnnotations.push({
              xref: "paper", yref: "paper", x: 0, y: 0, xanchor: "left", yanchor: "top",
              xshift: -riskLeftMargin + 6, yshift: -RISK_HEADER_Y,
              text: "NUMBER AT RISK", showarrow: false,
              font: { size: 10, color: "#6b7280" },
            });
            riskGroups.forEach((g, gi) => {
              const color = kmGroupColors[String(g.group)] ?? (kmColorblind ? OKABE_ITO[gi % OKABE_ITO.length] : pal[gi % pal.length]);
              const rowShift = -(RISK_FIRST_ROW_Y + gi * RISK_ROW_PX);
              // Group label in the left margin
              riskAnnotations.push({
                xref: "paper", yref: "paper", x: 0, y: 0, xanchor: "left", yanchor: "top",
                xshift: -riskLeftMargin + 6, yshift: rowShift,
                text: riskLabels[gi], showarrow: false,
                font: { size: 10, color },
              });
              // Counts under each risk time
              riskTimes.forEach((t: number, ti: number) => {
                const n = g.at_risk?.[ti];
                if (n == null) return;
                riskAnnotations.push({
                  xref: "x", yref: "paper", x: t, y: 0, xanchor: "center", yanchor: "top",
                  yshift: rowShift,
                  text: String(n), showarrow: false,
                  font: { size: 10, color: "#374151" },
                });
              });
            });
          }

          return (
          <>
            {/* Publication-style customisation strip */}
            <div className="flex flex-wrap items-center gap-3 text-[10px] text-gray-500 mb-2 px-1">
              <button
                onClick={(e) => {
                  setKmContextMenu({ type: "plotTitle", x: e.clientX, y: e.clientY });
                  setKmRenameValue(kmCustomPlotTitle || (kmGroup ? `Kaplan–Meier survival by ${kmCustomGroupTitle || kmGroup}` : "Kaplan–Meier survival"));
                }}
                className="text-[10px] px-2 py-0.5 rounded border border-gray-300 text-gray-600 hover:bg-gray-50"
              >✏️ Plot title</button>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmShowPInTitle} onChange={(e) => setKmShowPInTitle(e.target.checked)} className="accent-indigo-500" />
                log-rank p in title
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmShowNInLegend} onChange={(e) => setKmShowNInLegend(e.target.checked)} className="accent-indigo-500" />
                (n=…) in legend
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmHidePrefix} onChange={(e) => setKmHidePrefix(e.target.checked)} className="accent-indigo-500" />
                hide "{kmCustomGroupTitle || kmGroup || "group"} =" prefix
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmAutoZoomY} onChange={(e) => setKmAutoZoomY(e.target.checked)} className="accent-indigo-500" />
                zoom Y to data
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmYAxisAsPct} onChange={(e) => setKmYAxisAsPct(e.target.checked)} className="accent-indigo-500" />
                Y as %
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmRiskTable} onChange={(e) => setKmRiskTable(e.target.checked)} className="accent-indigo-500" />
                Number at risk
                <Tip wide text="Journal-standard 'number at risk' row under the curve: subjects still in follow-up (event-free and uncensored) in each group at evenly-spaced time points. Required by most journals (CONSORT/STROBE) so readers can judge how much data supports the tail of the curve." />
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmColorblind} onChange={(e) => setKmColorblind(e.target.checked)} className="accent-indigo-500" />
                Colour-blind + line styles
                <Tip wide text="Switches to the Okabe-Ito colour-blind-safe palette and gives each group a distinct line style (solid / dashed / dotted), so curves stay distinguishable in greyscale print and for colour-blind readers. Recommended for publication figures." />
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={kmShowCensors} onChange={(e) => setKmShowCensors(e.target.checked)} className="accent-indigo-500" />
                Censor marks
                <Tip wide text="Overlays a small '+' on each curve wherever a subject was censored (lost to follow-up / still event-free at last contact). Shows where information thins out; common in publication KM plots." />
              </label>
              {(kmCustomPlotTitle || Object.keys(kmGroupColors).length > 0 || Object.keys(kmGroupLabels).length > 0) && (
                <button
                  onClick={() => { setKmCustomPlotTitle(""); setKmGroupColors({}); setKmGroupLabels({}); }}
                  className="text-[10px] px-2 py-0.5 rounded border border-orange-300 text-orange-600 hover:bg-orange-50"
                >✕ Reset customisation</button>
              )}
            </div>

            {/* Axis labels + size — like the Forest builder */}
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 mb-2 px-1">
              <span className="text-[10px] text-gray-400 inline-flex items-center">Axis &amp; size<Tip wide text="Rename the X/Y axis titles for publication (e.g. 'Time since primary PCI (days)', 'Overall survival'). Drag Width/Height to size the figure; Width 'auto' fills the column. Export (↓ top-right of the plot) keeps these labels — SVG/PDF are vector, journal-ready." /></span>
              <input value={kmCustomDurationTitle} onChange={(e) => setKmCustomDurationTitle(e.target.value)}
                placeholder={`X-axis (Time (${kmDuration || "time"}))`}
                className="text-[11px] border border-gray-200 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" style={{ width: 200 }} />
              <input value={kmYTitle} onChange={(e) => setKmYTitle(e.target.value)}
                placeholder="Y-axis label"
                className="text-[11px] border border-gray-200 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" style={{ width: 160 }} />
              <label className="flex items-center gap-1.5 text-[10px] text-gray-500">
                <span className="font-medium">Width</span>
                <input type="range" min={420} max={1300} step={20} value={kmPlotW ?? 800}
                  onChange={(e) => setKmPlotW(Number(e.target.value))} className="accent-indigo-500" style={{ width: 100 }} />
                <span className="tabular-nums w-8">{kmPlotW ?? "auto"}</span>
                {kmPlotW != null && (
                  <button onClick={() => setKmPlotW(undefined)} className="text-indigo-500 hover:text-indigo-700">auto</button>
                )}
              </label>
              <label className="flex items-center gap-1.5 text-[10px] text-gray-500">
                <span className="font-medium">Height</span>
                <input type="range" min={260} max={760} step={20} value={kmPlotH}
                  onChange={(e) => setKmPlotH(Number(e.target.value))} className="accent-indigo-500" style={{ width: 100 }} />
                <span className="tabular-nums w-8">{kmPlotH}</span>
              </label>
            </div>

            {/* Editable group (legend) labels */}
            {kmGroup && kmResult.groups?.length > 0 && (
              <div className="flex flex-wrap items-center gap-2 mb-2 px-1">
                <span className="text-[10px] text-gray-400 inline-flex items-center">Legend labels<Tip wide text="Rename each group as it appears in the legend, number-at-risk row, tables, and the auto-narrative — e.g. code '1' → '<100 mg/dL'. Leave blank to keep the original. Also editable by right-clicking a row in the summary table." /></span>
                {kmResult.groups.map((g, i) => (
                  <span key={g.group} className="inline-flex items-center gap-1">
                    <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: kmGroupColors[String(g.group)] ?? (kmColorblind ? OKABE_ITO[i % OKABE_ITO.length] : pal[i % pal.length]) }} />
                    <input
                      value={kmGroupLabels[String(g.group)] ?? ""}
                      onChange={(e) => {
                        const next = { ...kmGroupLabels };
                        const v = e.target.value;
                        if (v.trim()) next[String(g.group)] = v; else delete next[String(g.group)];
                        setKmGroupLabels(next);
                      }}
                      placeholder={String(g.group)}
                      className="text-[11px] border border-gray-200 rounded px-2 py-0.5 focus:outline-none focus:border-indigo-400" style={{ width: 130 }} />
                  </span>
                ))}
              </div>
            )}

            {/* Title — rendered as HTML here (UI only); NOT embedded in the
                exported figure, so it can be supplied as the figure legend. */}
            {titleText && (
              <div className="text-center text-sm font-medium text-gray-700 mb-1">{titleText}</div>
            )}

            <div className="relative" ref={kmPlotRef} style={{ width: kmPlotW != null ? kmPlotW : "100%", height: kmPlotH, maxWidth: "100%" }}>
              <Plot
                data={[
                  ...kmResult.groups.map((g, i) => ({
                    x: g.curve.map((p) => p.time),
                    y: g.curve.map((p) => p.survival),
                    type: "scatter", mode: "lines",
                    name: legendLabel(g),
                    line: {
                      width: traceDefaults.lineWidth,
                      color: kmGroupColors[String(g.group)] ?? (kmColorblind ? OKABE_ITO[i % OKABE_ITO.length] : pal[i % pal.length]),
                      shape: "hv",
                      ...(kmColorblind ? { dash: KM_DASHES[i % KM_DASHES.length] } : {}),
                    },
                  })),
                  // Censor tick marks ('+') overlaid on each curve.
                  ...(kmShowCensors
                    ? kmResult.groups.flatMap((g, i) => {
                        if (!Array.isArray(g.censors) || g.censors.length === 0) return [];
                        const c = kmGroupColors[String(g.group)] ?? (kmColorblind ? OKABE_ITO[i % OKABE_ITO.length] : pal[i % pal.length]);
                        return [{
                          x: g.censors.map((p) => p.time),
                          y: g.censors.map((p) => p.survival),
                          type: "scatter", mode: "markers",
                          name: `${legendLabel(g)} (censored)`,
                          marker: { symbol: "cross-thin-open", size: 7, color: c, line: { width: 1.4, color: c } },
                          hoverinfo: "x", showlegend: false,
                        }];
                      })
                    : []),
                ] as unknown as Data[]}
                layout={{
                  ...baseLayout,
                  // Title is rendered as HTML above the plot (UI only) so it
                  // is NOT baked into the exported figure — the user supplies
                  // it as the figure legend in the manuscript.
                  title: undefined,
                  xaxis: {
                    ...(baseLayout.xaxis as Record<string, unknown>),
                     // Using custom duration title if available
                    title: { text: kmCustomDurationTitle ? kmCustomDurationTitle : `Time (${kmDuration})` },
                  },
                  yaxis: {
                    ...(baseLayout.yaxis as Record<string, unknown>),
                    title: { text: kmYTitle || "Survival probability" },
                    range: yRange,
                    tickformat: kmYAxisAsPct ? ".0%" : ".2f",
                  },
                  autosize: true,
                  // Solid white backgrounds so PNG/JPEG export isn't transparent
                  // (transparent renders as black in most image viewers).
                  paper_bgcolor: "#ffffff",
                  plot_bgcolor: "#ffffff",
                  margin: { t: 20, r: 20, b: riskBottomMargin, l: riskLeftMargin }, showlegend: true,
                  legend: { title: { text: kmCustomGroupTitle || kmGroup || "Group" } },
                  annotations: riskAnnotations,
                }}
                useResizeHandler
                config={{ responsive: true }} style={{ width: "100%", height: "100%" }}
              />
              <PlotExporter plotRef={kmPlotRef as unknown as PlotRef} title="KM_Survival" />
            </div>

            {/* Number at risk is embedded in the figure above (Plotly
                annotations) so it renders on-screen and in every export. */}

            {/* Compact Group summary table & Log-rank test */}
            <div className="overflow-hidden rounded-lg border border-gray-200 shadow-sm mt-2">
              <table className="text-xs w-full bg-white">
                <thead><tr className="bg-gray-50 border-b border-gray-200 bg-opacity-70">
                  <th className="px-3 py-1.5 text-left text-[9px] font-bold text-gray-500 uppercase tracking-wider cursor-context-menu"
                    onContextMenu={(e) => {
                      e.preventDefault();
                      setKmContextMenu({ type: "groupTitle", x: e.clientX, y: e.clientY });
                      setKmRenameValue(kmCustomGroupTitle || kmGroup || "Group");
                    }}
                  >
                    {kmCustomGroupTitle || kmGroup || "Group"}
                    <span className="ml-1 font-normal text-gray-400 normal-case tracking-normal">(right-click to rename)</span>
                  </th>
                  <th className="px-3 py-1.5 text-right text-[9px] font-bold text-gray-500 uppercase tracking-wider"><i>n</i></th>
                  <th className="px-3 py-1.5 text-right text-[9px] font-bold text-gray-500 uppercase tracking-wider">Events</th>
                  <th className="px-3 py-1.5 text-right text-[9px] font-bold text-gray-500 uppercase tracking-wider cursor-context-menu"
                    onContextMenu={(e) => {
                       e.preventDefault();
                       setKmContextMenu({ type: "durationTitle", x: e.clientX, y: e.clientY });
                       setKmRenameValue(kmCustomDurationTitle || kmDuration);
                    }}
                  >
                    Median ({kmCustomDurationTitle || kmDuration})
                    <span className="ml-1 font-normal text-gray-400 normal-case tracking-normal">(right-click to rename)</span>
                  </th>
                </tr></thead>
                <tbody className="divide-y divide-gray-100">
                  {kmResult.groups.map((g, i) => {
                    const label = resolveGroupName(String(g.group));
                    const isRenamed = label !== String(g.group);
                    return (
                      <tr key={i} className="hover:bg-indigo-50/30 transition-colors"
                        onContextMenu={(e) => {
                          e.preventDefault();
                          setKmContextMenu({ type: "item", group: String(g.group), x: e.clientX, y: e.clientY });
                          setKmRenameValue(label);
                        }}
                      >
                        <td className="px-3 py-1 cursor-context-menu select-none">
                          <span className="inline-flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                              style={{ background: pal[i % pal.length] }} />
                            <span className="text-[11px] font-medium text-gray-700">{label}</span>
                            {isRenamed && (
                              <span className="text-[9px] text-gray-400">({g.group})</span>
                            )}
                          </span>
                        </td>
                        <td className="px-3 py-1 text-[11px] font-medium text-gray-600 text-right">{g.n}</td>
                        <td className="px-3 py-1 text-[11px] font-medium text-gray-600 text-right">{g.events}</td>
                        <td className="px-3 py-1 text-[11px] font-medium text-gray-600 text-right">{g.median_survival ?? "NR"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              {/* Log-rank test embedded as a cohesive footer inside the same block */}
              {kmResult.logrank && (
                <div className={`px-3 py-1.5 text-[11px] border-t font-medium flex items-center justify-between ${kmResult.logrank.p != null && kmResult.logrank.p < 0.05 ? "bg-indigo-50 border-indigo-100 text-indigo-700" : "bg-gray-50 border-gray-100 text-gray-500"}`}>
                  <span>Log-rank test (overall)</span>
                  <span>
                    <i>p</i> = {fmtP(kmResult.logrank.p)}
                    {kmResult.logrank.p != null && kmResult.logrank.p < 0.05 ? " (Significant difference)" : " (No difference)"}
                  </span>
                </div>
              )}

              {/* Median follow-up (reverse Kaplan–Meier) */}
              {kmResult.median_follow_up?.median != null && (
                <div className="px-3 py-1.5 text-[11px] border-t border-gray-100 bg-gray-50/60 text-gray-600 flex items-center justify-between">
                  <span>Median follow-up (reverse KM)</span>
                  <span className="tabular-nums">
                    {kmResult.median_follow_up.median.toFixed(0)}
                    {kmResult.median_follow_up.q1 != null && kmResult.median_follow_up.q3 != null &&
                      ` [${kmResult.median_follow_up.q1.toFixed(0)}–${kmResult.median_follow_up.q3.toFixed(0)}]`}
                    {" "}{kmCustomDurationTitle || kmDuration}
                  </span>
                </div>
              )}

              {/* Landmark survival-at-time table */}
              {Array.isArray(kmResult.survival_times) && kmResult.survival_times.length > 0 && (
                <div className="border-t border-gray-100">
                  <div className="px-3 py-1.5 text-[10px] font-semibold text-gray-500 uppercase tracking-wide bg-gray-50">
                    Survival probability at time point(s)
                  </div>
                  <table className="w-full">
                    <thead>
                      <tr className="text-[10px] text-gray-400">
                        <th className="px-3 py-1 text-left font-medium">Group</th>
                        {kmResult.survival_times.map((t: number) => (
                          <th key={t} className="px-3 py-1 text-right font-medium">t = {t}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {kmResult.groups?.map((g) => (
                        <tr key={g.group} className="border-t border-gray-50">
                          <td className="px-3 py-1 text-[11px] text-gray-700">{kmGroupLabels[String(g.group)] ?? g.group}</td>
                          {(g.survival_at ?? []).map((pt, i) => {
                            const unreliable = pt.reliable === false;
                            return (
                              <td key={i} className={`px-3 py-1 text-[11px] text-right tabular-nums ${unreliable ? "text-gray-300" : "text-gray-600"}`}
                                title={pt.n_at_risk != null ? `${pt.n_at_risk} at risk${unreliable ? " — too few; estimate unstable" : ""}` : undefined}>
                                {pt.survival != null ? `${(pt.survival * 100).toFixed(1)}%` : "—"}
                                {pt.ci_low != null && pt.ci_high != null && (
                                  <span className={unreliable ? "text-gray-300" : "text-gray-400"}> ({(pt.ci_low * 100).toFixed(1)}–{(pt.ci_high * 100).toFixed(1)})</span>
                                )}
                                {unreliable && <span className="text-amber-500" title="Unstable: fewer than 10 at risk or beyond max follow-up">*</span>}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {kmResult.groups?.some((g) => (g.survival_at ?? []).some((p) => p.reliable === false)) && (
                    <div className="px-3 py-1 text-[9px] text-amber-600 italic">
                      * Unstable estimate — fewer than 10 patients at risk or beyond maximum follow-up. Interpret with caution or omit.
                    </div>
                  )}
                </div>
              )}

              {/* Pairwise log-rank comparisons */}
              {(kmResult.pairwise?.comparisons?.length ?? 0) > 0 && (
                <div className="border-t border-gray-100">
                  <div className="px-3 py-1.5 text-[10px] font-semibold text-gray-500 uppercase tracking-wide bg-gray-50 flex items-center justify-between">
                    <span>Pairwise log-rank</span>
                    {kmResult.pairwise?.correction && kmResult.pairwise.correction !== "none" && (
                      <span className="normal-case font-normal text-gray-400">{kmResult.pairwise.correction} adjusted</span>
                    )}
                  </div>
                  <table className="w-full">
                    <thead>
                      <tr className="text-[10px] text-gray-400">
                        <th className="px-3 py-1 text-left font-medium">Comparison</th>
                        <th className="px-3 py-1 text-right font-medium">p (raw)</th>
                        {(kmResult.pairwise?.comparisons ?? []).some((c) => c.p_adj != null) && (
                          <th className="px-3 py-1 text-right font-medium"><i>p</i> (adj)</th>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {(kmResult.pairwise?.comparisons ?? []).map((c, i) => {
                        const pShow = (p: number | null | undefined) => fmtP(p);
                        const pVal = c.p_adj ?? c.p;
                        const sig = pVal != null && pVal < 0.05;
                        const la = kmGroupLabels[c.group_a] ?? c.group_a;
                        const lb = kmGroupLabels[c.group_b] ?? c.group_b;
                        return (
                          <tr key={i} className={`border-t border-gray-50 ${sig ? "bg-indigo-50/40" : ""}`}>
                            <td className="px-3 py-1 text-[11px] text-gray-700">{la} vs {lb}</td>
                            <td className="px-3 py-1 text-[11px] text-gray-600 text-right tabular-nums">{pShow(c.p)}</td>
                            {kmResult.pairwise?.comparisons?.some((x) => x.p_adj != null) && (
                              <td className={`px-3 py-1 text-[11px] text-right tabular-nums ${sig ? "font-semibold text-indigo-700" : "text-gray-600"}`}>{pShow(c.p_adj)}</td>
                            )}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Auto-generated standard interpretation */}
              {(() => {
                const narrative = buildKmNarrative(kmResult, kmGroupLabels, kmCustomGroupTitle || kmGroup);
                if (!narrative) return null;
                return (
                  <div className="border-t border-gray-100 bg-amber-50/60 px-4 py-3">
                    <div className="text-[10px] font-semibold text-amber-700 uppercase tracking-wide mb-1">Interpretation (auto-generated)</div>
                    <p className="text-[12px] text-amber-900 leading-relaxed">{narrative}</p>
                    <p className="text-[9px] text-amber-600/80 mt-1.5 italic">Draft wording — verify against your data and reporting guidelines before publication.</p>
                  </div>
                );
              })()}

              {/* Right-click context menu (absolute body mount replacement) */}
              {kmContextMenu && (
                <>
                  <div 
                    className="fixed inset-0 z-40" 
                    onClick={() => setKmContextMenu(null)} 
                    onContextMenu={(e) => { e.preventDefault(); setKmContextMenu(null); }} 
                  />
                  <div
                    className="fixed bg-white border border-gray-200 rounded-lg shadow-xl z-50 p-3 min-w-[200px]"
                    style={{ top: kmContextMenu.y, left: kmContextMenu.x }}
                  >
                  <p className="text-[10px] text-gray-400 mb-1.5 font-medium uppercase tracking-wide">
                    {kmContextMenu.type === "item" && `Rename group "${kmContextMenu.group}"`}
                    {kmContextMenu.type === "groupTitle" && `Rename Legend Title`}
                    {kmContextMenu.type === "durationTitle" && `Rename Time Axis Title`}
                    {kmContextMenu.type === "plotTitle" && `Edit plot title`}
                  </p>
                  <input
                    autoFocus
                    className="w-full text-xs border border-gray-300 rounded px-2 py-1 mb-2 focus:outline-none focus:border-indigo-400"
                    value={kmRenameValue}
                    onChange={(e) => setKmRenameValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        if (kmContextMenu.type === "item" && kmContextMenu.group) {
                          setKmGroupLabels((prev) => ({ ...prev, [kmContextMenu.group!]: kmRenameValue }));
                        } else if (kmContextMenu.type === "groupTitle") {
                          setKmCustomGroupTitle(kmRenameValue);
                        } else if (kmContextMenu.type === "durationTitle") {
                          setKmCustomDurationTitle(kmRenameValue);
                        } else if (kmContextMenu.type === "plotTitle") {
                          setKmCustomPlotTitle(kmRenameValue);
                        }
                        setKmContextMenu(null);
                      }
                      if (e.key === "Escape") setKmContextMenu(null);
                    }}
                  />
                  {kmContextMenu.type === "item" && kmContextMenu.group && (
                    <div className="mb-2">
                      <p className="text-[10px] text-gray-400 mb-1 font-medium uppercase tracking-wide">Color</p>
                      <div className="flex items-center gap-1.5 flex-wrap">
                        {["#dc2626", "#16a34a", "#2563eb", "#ea580c", "#9333ea", "#0891b2", "#ca8a04", "#475569"].map((c) => (
                          <button key={c} title={c}
                            onClick={() => setKmGroupColors((prev) => ({ ...prev, [kmContextMenu.group!]: c }))}
                            className="w-5 h-5 rounded-full border border-gray-300 hover:scale-110 transition-transform"
                            style={{ background: c }}
                          />
                        ))}
                        <input type="color"
                          value={kmGroupColors[kmContextMenu.group] ?? "#6366f1"}
                          onChange={(e) => setKmGroupColors((prev) => ({ ...prev, [kmContextMenu.group!]: e.target.value }))}
                          className="w-6 h-6 cursor-pointer border border-gray-300 rounded"
                        />
                      </div>
                    </div>
                  )}
                  <div className="flex gap-2">
                    <button
                      onClick={() => {
                        if (kmContextMenu.type === "item" && kmContextMenu.group) {
                          setKmGroupLabels((prev) => ({ ...prev, [kmContextMenu.group!]: kmRenameValue }));
                        } else if (kmContextMenu.type === "groupTitle") {
                          setKmCustomGroupTitle(kmRenameValue);
                        } else if (kmContextMenu.type === "durationTitle") {
                          setKmCustomDurationTitle(kmRenameValue);
                        } else if (kmContextMenu.type === "plotTitle") {
                          setKmCustomPlotTitle(kmRenameValue);
                        }
                        setKmContextMenu(null);
                      }}
                      className="flex-1 text-xs bg-indigo-600 text-white rounded px-2 py-1 hover:bg-indigo-700"
                    >Save</button>
                    <button
                      onClick={() => {
                        if (kmContextMenu.type === "item" && kmContextMenu.group) {
                          const next = { ...kmGroupLabels };
                          delete next[kmContextMenu.group];
                          setKmGroupLabels(next);
                          const nc = { ...kmGroupColors };
                          delete nc[kmContextMenu.group];
                          setKmGroupColors(nc);
                        } else if (kmContextMenu.type === "groupTitle") {
                          setKmCustomGroupTitle("");
                        } else if (kmContextMenu.type === "durationTitle") {
                          setKmCustomDurationTitle("");
                        } else if (kmContextMenu.type === "plotTitle") {
                          setKmCustomPlotTitle("");
                        }
                        setKmContextMenu(null);
                      }}
                      className="text-xs text-gray-400 hover:text-red-500 px-2 py-1"
                    >Reset</button>
                  </div>
                </div>
                </>
              )}
            </div>
          </>
          );
        })()}

        {/* Stratified KM (small-multiples grid) — when stratify_col is set */}
        {kmResult?.strata && (() => {
          const strata: KMStratum[] = kmResult.strata;
          const miniH = 420;

          const stratColMeta = columns.find((c) => c.name === kmStratify);
          const stratLabels = stratColMeta?.value_labels ?? {};
          const groupColMeta = columns.find((c) => c.name === kmGroup);
          const grpLabels = groupColMeta?.value_labels ?? {};

          const buildTraces = (groups: KMGroup[]) =>
            groups.map((g, i) => ({
              x: g.curve.map((p) => p.time),
              y: g.curve.map((p) => p.survival),
              type: "scatter" as const, mode: "lines" as const,
              name: kmGroup
                ? `${kmCustomGroupTitle || kmGroup} = ${grpLabels[String(g.group)] ?? g.group}`
                : String(g.group),
              line: { width: traceDefaults.lineWidth, color: pal[i % pal.length], shape: "hv" as const },
            }));

          return (
            <div className="mt-3 space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-gray-700">
                  Stratified by <span className="text-indigo-600">{kmStratify}</span>
                  {kmGroup && <span className="text-gray-400 font-normal ml-2">— curves by {kmGroup}</span>}
                </h4>
                <span className="text-xs text-gray-400">{strata.length} strata · {kmResult.n_total ?? "?"} total</span>
              </div>
              <div className="grid gap-4 grid-cols-1">
                {strata.map((stratum) => {
                  const stratLabel = stratLabels[String(stratum.label)] ?? stratum.label;
                  const pAnnot = stratum.logrank?.p != null ? [{
                    xref: "paper", yref: "paper", x: 0.02, y: 0.98,
                    xanchor: "left", yanchor: "top",
                    text: `<i>p</i> = ${fmtP(stratum.logrank.p)}`,
                    showarrow: false,
                    font: { size: 11, color: stratum.logrank.p != null && stratum.logrank.p < 0.05 ? "#6366f1" : "#6b7280" },
                    bgcolor: "rgba(249,250,251,0.85)", borderpad: 3, bordercolor: "#e5e7eb", borderwidth: 1,
                  }] : [];
                  return (
                    <div key={stratum.label} className="border border-gray-200 rounded-lg overflow-hidden">
                      <div className="px-3 py-1.5 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
                        <span className="text-xs font-semibold text-gray-700">{kmStratify} = {stratLabel}</span>
                        <span className="text-[10px] text-gray-400"><i>n</i>={stratum.n}</span>
                      </div>
                      <Plot
                        data={buildTraces(stratum.groups)}
                        layout={{
                          ...baseLayout,
                          autosize: true,
                          height: miniH,
                          margin: { t: 10, r: 10, b: 40, l: 50 },
                          xaxis: { ...(baseLayout.xaxis as Record<string, unknown>), title: { text: kmCustomDurationTitle || kmDuration } },
                          yaxis: { ...(baseLayout.yaxis as Record<string, unknown>), range: [0, 1.05], tickformat: ".0%", title: { text: "Survival" } },
                          legend: { font: { size: 9 }, orientation: "h", y: -0.22 },
                          annotations: pAnnot as Partial<Annotations>[],
                        }}
                        style={{ width: "100%", height: miniH }}
                        config={{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"] }}
                        useResizeHandler
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })()}
      </Section>
      )}

      {/* ── Cox PH ── */}
      {activeMethod === "cox" && (
      <Section title="Cox Proportional Hazards" description="Regression for time-to-event data — outputs Hazard Ratios (HR)">
        <div className="grid grid-cols-2 gap-3">
          <VarSelect label="Duration (time)" value={coxDuration} onChange={setCoxDuration} columns={pickCols} kinds={["numeric"]} />
          <VarSelect label="Event (0/1)" value={coxEvent} onChange={setCoxEvent} columns={binaryCols} />
        </div>

        {/* Checkbox predictor list */}
        <div>
          <p className="text-xs text-gray-500 font-medium mb-1.5">
            Predictors
            {coxPreds.length > 0 && (
              <span className="ml-2 text-indigo-600 font-semibold">{coxPreds.length} selected</span>
            )}
            {coxPreds.length > 0 && (
              <button onClick={() => setCoxPreds([])} className="ml-2 text-[10px] text-gray-400 hover:text-red-500 underline">clear</button>
            )}
          </p>
          <div className="border border-gray-200 rounded-lg overflow-y-auto max-h-36 divide-y divide-gray-100">
            {pickCols.map((c) => (
              <label key={c.name} className={`flex items-center gap-2 px-3 py-1.5 cursor-pointer transition-colors text-xs
                ${coxPreds.includes(c.name) ? "bg-indigo-50 text-indigo-800" : "hover:bg-gray-50 text-gray-700"}`}>
                <input
                  type="checkbox"
                  checked={coxPreds.includes(c.name)}
                  onChange={(e) => {
                    if (e.target.checked) setCoxPreds([...coxPreds, c.name]);
                    else setCoxPreds(coxPreds.filter((p) => p !== c.name));
                  }}
                  className="accent-indigo-500"
                />
                <span className="font-medium">{c.name}</span>
                <span className="text-[10px] text-gray-400 ml-auto">{c.kind}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Interactions — pair selector (only meaningful when ≥2 predictors ticked) */}
        {coxPreds.length >= 2 && (
          <div>
            <p className="text-xs text-gray-500 font-medium mb-1.5 flex items-center gap-1">
              Interactions
              <Tip wide text="Add pairwise interaction terms to the linear Cox model — e.g. LDL × AGE. Numeric × numeric is the element-wise product, numeric × categorical expands across every dummy of the categorical, categorical × categorical multiplies every dummy pair. Use sparingly: each extra term costs degrees of freedom. The output table reports each interaction as 'A:B' with its own HR and p-value." />
              {coxInteractions.length > 0 && (
                <span className="ml-1 text-indigo-600 font-semibold">{coxInteractions.length} added</span>
              )}
            </p>
            <div className="flex flex-wrap items-center gap-1.5">
              <select
                value={coxIxA}
                onChange={(e) => setCoxIxA(e.target.value)}
                className="select text-xs py-1"
              >
                <option value="">First term…</option>
                {coxPreds.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
              <span className="text-gray-400 text-xs">×</span>
              <select
                value={coxIxB}
                onChange={(e) => setCoxIxB(e.target.value)}
                className="select text-xs py-1"
              >
                <option value="">Second term…</option>
                {coxPreds.filter((p) => p !== coxIxA).map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
              <button
                onClick={() => {
                  if (!coxIxA || !coxIxB || coxIxA === coxIxB) return;
                  const exists = coxInteractions.some(
                    ([a, b]) => (a === coxIxA && b === coxIxB) || (a === coxIxB && b === coxIxA),
                  );
                  if (exists) return;
                  setCoxInteractions([...coxInteractions, [coxIxA, coxIxB]]);
                  setCoxIxA(""); setCoxIxB("");
                }}
                disabled={!coxIxA || !coxIxB || coxIxA === coxIxB}
                className="text-xs px-2 py-1 rounded border border-indigo-300 text-indigo-600 hover:bg-indigo-50 disabled:opacity-40 transition-colors"
              >
                + Add
              </button>
            </div>
            {coxInteractions.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mt-1.5">
                {coxInteractions.map(([a, b], i) => (
                  <span
                    key={`${a}:${b}:${i}`}
                    className="inline-flex items-center gap-1 bg-amber-50 border border-amber-200 text-amber-800 text-[11px] rounded px-2 py-0.5"
                  >
                    {a} × {b}
                    <button
                      onClick={() => setCoxInteractions(coxInteractions.filter((_, idx) => idx !== i))}
                      className="text-amber-500 hover:text-red-500"
                      title="Remove"
                    >
                      ×
                    </button>
                  </span>
                ))}
                <button
                  onClick={() => setCoxInteractions([])}
                  className="text-[10px] text-gray-400 hover:text-red-500 underline"
                >
                  clear all
                </button>
              </div>
            )}
          </div>
        )}

        <div className="flex items-center gap-3 flex-wrap">
          <RunButton onClick={async () => {
            if (!coxDuration || !coxEvent || coxPreds.length === 0) { setCoxError("Select duration, event, and at least one predictor"); return; }
            setCoxResult(null); setCoxError(null); setCoxLoading(true);
            try {
              const res = await runCox({
                session_id: sid,
                duration_col: coxDuration,
                event_col: coxEvent,
                predictors: coxPreds,
                interactions: coxInteractions.length > 0 ? coxInteractions : undefined,
              });
              setCoxResult(res.data);
            } catch (e: unknown) { setCoxError(errorMessage(e, "Cox failed")); }
            finally { setCoxLoading(false); }
          }} loading={coxLoading} label="Run Cox Regression" />

          {/* Univariable screening button */}
          {coxDuration && coxEvent && coxPreds.length > 0 && (
            <button
              disabled={coxScanLoading}
              onClick={async () => {
                setCoxScanLoading(true);
                const results: CoxScanRow[] = [];
                for (const pred of coxPreds) {
                  try {
                    const res = await runCox({ session_id: sid, duration_col: coxDuration, event_col: coxEvent, predictors: [pred] });
                    const coef = res.data.coefficients?.[0];
                    results.push({
                      variable: pred,
                      hr: coef?.hr ?? null,
                      hr_ci_low: coef?.hr_ci_low ?? null,
                      hr_ci_high: coef?.hr_ci_high ?? null,
                      p: coef?.p ?? null,
                      n: res.data.n ?? null,
                    });
                  } catch { results.push({ variable: pred, hr: null, hr_ci_low: null, hr_ci_high: null, p: null, n: null }); }
                }
                results.sort((a, b) => (a.p ?? 1) - (b.p ?? 1));
                setCoxScanResult(results);
                setCoxScanLoading(false);
              }}
              className="px-3 py-1.5 text-xs font-medium border border-indigo-300 text-indigo-600 rounded-lg hover:bg-indigo-50 disabled:opacity-50 transition-colors"
            >
              {coxScanLoading ? "Scanning…" : "🔍 Univariable Scan"}
              <Tip wide text="Fits a separate Cox PH model for each predictor on its own (Surv(time,event) ~ X), then ranks them by p-value. Use it to triage which candidates are worth carrying into the multivariable model. Common rule: take everything with univariable p < 0.10 forward and let the multivariable fit decide what stays — variables can lose or gain significance once you adjust for confounders (e.g. SMOKER univariable p ≈ 1.0 but p < 0.001 once AGE is controlled for)." />
            </button>
          )}

          {/* Paired unadjusted-vs-adjusted forest (publication Figure 4) */}
          {coxDuration && coxEvent && coxPreds.length > 0 && (
            <button
              disabled={coxUMLoading}
              onClick={() => runCoxUMForest(coxUMRefs)}
              className="px-3 py-1.5 text-xs font-medium border border-indigo-300 text-indigo-600 rounded-lg hover:bg-indigo-50 disabled:opacity-50 transition-colors"
            >
              {coxUMLoading ? "Fitting…" : "📊 Unadjusted vs Adjusted forest"}
              <Tip wide text="Builds the publication-standard paired forest: each row shows the predictor's HR alone (unadjusted, univariable Cox — grey circle) next to its HR adjusted for every other predictor (multivariable Cox — blue square), with 95% CIs. Multi-level categorical predictors (those with value labels, e.g. LDL groups) expand to one contrast row per non-reference level; binary predictors (sex, diabetes) stay a single HR; continuous predictors report a per-unit HR. Large unadjusted→adjusted shifts flag confounding." />
            </button>
          )}
          {coxError && <p className="text-xs text-red-500">{coxError}</p>}
        </div>

        {coxUMReady && (
          <CoxUniMultiForest
            result={coxUMReady}
            /* full column list — used only to resolve value_labels for the saved result */
            columns={columns /* display only */}
            plotRef={coxUMRef}
            refs={coxUMRefs}
            loading={coxUMLoading}
            onChangeRef={(predictor, level) => {
              const next = { ...coxUMRefs };
              if (level) next[predictor] = level; else delete next[predictor];
              setCoxUMRefs(next);
              runCoxUMForest(next);
            }}
            onClose={() => setCoxUMResult(null)}
          />
        )}

        {/* Model-specification (sensitivity) forest builder — always visible so
            the exposure picker is discoverable; Build validates duration/event. */}
        {(
          <div className="rounded-lg border border-gray-200">
            <button onClick={() => setSpecOpen(!specOpen)}
              className="w-full flex items-center justify-between px-3 py-2 text-xs font-semibold text-gray-600 hover:bg-gray-50">
              <span className="inline-flex items-center gap-1">📋 Model-specification forest (sensitivity)
                <Tip wide text="One exposure's adjusted HR across several adjustment sets — the Figure 6 sensitivity forest. Pick the exposure (e.g. an LDL<100 indicator), define named model specs with different covariates (parsimonious / alternative / full), and run: each model's HR for the exposure becomes a forest row (an Unadjusted row is added automatically). Results are sent to the Forest Builder with full label/size/export controls." /></span>
              <span>{specOpen ? "▴" : "▾"}</span>
            </button>
            {specOpen && (
              <div className="px-3 pb-3 space-y-2 border-t border-gray-100 pt-2">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                  <VarSelect label="Exposure (HR tracked)" value={specExposure} onChange={(v) => { setSpecExposure(v); setSpecExposureRef(""); setSpecExposureLevel(""); }} columns={pickCols} />
                  {(() => {
                    const em = columns.find((c) => c.name === specExposure);
                    const vl = em?.value_labels ?? {};
                    const levels = Object.keys(vl);
                    if (levels.length < 2) return <><div /><div /></>;
                    const sorted = levels.sort((a, b) => Number(a) - Number(b));
                    const ref = specExposureRef || sorted[0];
                    return (
                      <>
                        <label className="flex flex-col gap-1">
                          <span className="text-xs text-gray-500 font-medium">Reference level</span>
                          <select value={specExposureRef} onChange={(e) => { setSpecExposureRef(e.target.value); setSpecExposureLevel(""); }} className="select text-xs py-1">
                            <option value="">(lowest code)</option>
                            {sorted.map((k) => <option key={k} value={k}>{vl[k]}</option>)}
                          </select>
                        </label>
                        <label className="flex flex-col gap-1">
                          <span className="text-xs text-gray-500 font-medium inline-flex items-center">Show only level
                            <Tip wide text="For a 3+ level exposure: show just this one contrast (e.g. <100 vs reference) across all models, so each forest row is one model labelled by its name — clean and matching the manuscript. Leave 'all levels' to plot every contrast (rows then carry the model name + contrast)." /></span>
                          <select value={specExposureLevel} onChange={(e) => setSpecExposureLevel(e.target.value)} className="select text-xs py-1">
                            <option value="">all levels</option>
                            {sorted.filter((k) => k !== ref).map((k) => <option key={k} value={k}>{vl[k]} vs {vl[ref]}</option>)}
                          </select>
                        </label>
                      </>
                    );
                  })()}
                </div>

                {specRows.map((s, i) => (
                  <div key={i} className="rounded border border-gray-200 p-2 space-y-1.5">
                    <div className="flex items-center gap-2">
                      <input value={s.label}
                        onChange={(e) => setSpecRows(specRows.map((r, j) => j === i ? { ...r, label: e.target.value } : r))}
                        placeholder={`Model ${i + 1} label`}
                        className="flex-1 text-[11px] border border-gray-200 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" />
                      <button onClick={() => setSpecRows(specRows.filter((_, j) => j !== i))}
                        className="text-[11px] text-gray-400 hover:text-red-500">remove</button>
                    </div>
                    <MultiSelect label="Covariates" columns={pickCols}
                      selected={s.covariates}
                      onChange={(next) => setSpecRows(specRows.map((r, j) => j === i ? { ...r, covariates: next } : r))}
                      excludeNames={[coxDuration, coxEvent, specExposure].filter(Boolean)} />
                  </div>
                ))}

                <div className="flex items-center gap-3 flex-wrap">
                  <button onClick={() => setSpecRows([...specRows, { label: `Model ${specRows.length + 1}`, covariates: [] }])}
                    className="text-[11px] px-2 py-1 rounded border border-indigo-200 text-indigo-600 hover:bg-indigo-50">+ Add model</button>
                  <button disabled={specLoading || !specExposure} onClick={runSpecForest}
                    className="px-3 py-1.5 text-xs font-medium bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50">
                    {specLoading ? "Fitting…" : "→ Build forest"}
                  </button>
                  <span className="text-[10px] text-gray-400">An “Unadjusted” row is added automatically.</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Cox univariable scan results */}
        {coxScanResult.length > 0 && (
          <div className="rounded-lg border border-gray-200 overflow-auto">
            <div className="bg-gray-50 px-3 py-2 border-b border-gray-200 flex items-center justify-between">
              <p className="text-xs font-semibold text-gray-600">Univariable Cox Scan — One Predictor at a Time</p>
              <button onClick={() => setCoxScanResult([])} className="text-[10px] text-gray-400 hover:text-red-500">✕ Close</button>
            </div>
            <table className="text-xs w-full">
              <thead><tr className="bg-gray-50">
                <th className="px-3 py-1.5 text-left text-gray-500">Variable</th>
                <th className="px-3 py-1.5 text-left text-gray-500">N (events)</th>
                <th className="px-3 py-1.5 text-left text-gray-500">HR</th>
                <th className="px-3 py-1.5 text-left text-gray-500">95% CI</th>
                <th className="px-3 py-1.5 text-left text-gray-500"><i>p</i></th>
              </tr></thead>
              <tbody>
                {coxScanResult.map((r, i) => (
                  <tr key={i} className={`border-t border-gray-100 ${r.p !== null && r.p < 0.05 ? "bg-indigo-50" : ""}`}>
                    <td className="px-3 py-1 font-medium text-gray-700">{r.variable}</td>
                    <td className="px-3 py-1 text-gray-500">{r.n ?? "—"}</td>
                    <td className="px-3 py-1 font-semibold text-gray-800">{r.hr != null ? r.hr.toFixed(3) : "—"}</td>
                    <td className="px-3 py-1 text-gray-500">
                      {r.hr_ci_low != null && r.hr_ci_high != null ? `${r.hr_ci_low.toFixed(3)} – ${r.hr_ci_high.toFixed(3)}` : "—"}
                    </td>
                    <td className={`px-3 py-1 font-semibold ${r.p !== null && r.p < 0.05 ? "text-indigo-700" : "text-gray-500"}`}>
                      {r.p !== null ? fmtP(r.p) : "error"}
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr className="border-t border-gray-200 bg-amber-50">
                  <td colSpan={5} className="px-3 py-1.5 text-[10px] text-amber-700">
                    💡 Add variables with p &lt; 0.10 to the multivariable Cox model — the adjustment may reveal effects that look null on their own.
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        )}

        {coxResult?.coefficients && (
          <div className="overflow-auto rounded-lg border border-gray-200">
            <table className="text-xs w-full">
              <thead>
                {/* Model summary row */}
                <tr className="bg-indigo-50 border-b border-indigo-100">
                  <td colSpan={2} className="px-3 py-1.5 text-indigo-700 font-medium">
                    N (events): <span className="font-bold">{coxResult.n}</span>
                  </td>
                  <td colSpan={2} className="px-3 py-1.5 text-indigo-700 font-medium">
                    C-index: <span className="font-bold">{coxResult.concordance?.toFixed(4)}</span>
                  </td>
                  <td colSpan={2} className="px-3 py-1.5 text-indigo-700 font-medium">
                    Log-Likelihood: <span className="font-bold">{coxResult.log_likelihood?.toFixed(2)}</span>
                  </td>
                </tr>
                <tr className="bg-gray-50">
                  <th className="px-3 py-1.5 text-left text-gray-500">Variable</th>
                  <th className="px-3 py-1.5 text-left text-gray-500">B</th>
                  <th className="px-3 py-1.5 text-left text-gray-500">SE</th>
                  <th className="px-3 py-1.5 text-left text-gray-500">HR</th>
                  <th className="px-3 py-1.5 text-left text-gray-500">95% CI</th>
                  <th className="px-3 py-1.5 text-left text-gray-500"><i>p</i></th>
                </tr>
              </thead>
              <tbody>
                {coxResult.coefficients?.map((c, i) => (
                  <tr key={i} className="border-t border-gray-100 hover:bg-gray-50">
                    <td className="px-3 py-1 font-medium text-gray-700">{c.variable}</td>
                    <td className="px-3 py-1 text-gray-600">{c.log_hr?.toFixed(4)}</td>
                    <td className="px-3 py-1 text-gray-600">{c.se?.toFixed(4)}</td>
                    <td className="px-3 py-1 font-semibold text-gray-800">{c.hr?.toFixed(4)}</td>
                    <td className="px-3 py-1 text-gray-500">{c.hr_ci_low?.toFixed(3)} – {c.hr_ci_high?.toFixed(3)}</td>
                    <td className={`px-3 py-1 ${c.p != null && c.p < 0.05 ? "text-indigo-600 font-semibold" : "text-gray-500"}`}>
                      {fmtP(c.p)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>
      )}

      {/* ───────────────────────────────────────────────────────────────────── */}
      {/* Time-horizon sensitivity → Forest plot                               */}
      {/* ───────────────────────────────────────────────────────────────────── */}
      {activeMethod === "timehorizon" && (
      <Section
        title="Time-horizon HR (forest)"
        description="Run the same Cox model at several follow-up windows (1-year, 2-year, full) and send the hazard ratios straight to the Forest Builder. Answers 'does the effect hold early vs. late?'."
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <VarSelect label="Duration / Time" value={chDuration} onChange={setChDuration} columns={pickCols} kinds={["numeric"]} />
          <VarSelect label="Event (0/1)" value={chEvent} onChange={setChEvent} columns={binaryCols} />
          <VarSelect label="Predictor (HR tracked)" value={chPredictor} onChange={setChPredictor} columns={pickCols} />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-2">
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Horizon cut-points (comma-sep)</span>
            <input value={chHorizons} onChange={(e) => setChHorizons(e.target.value)}
              placeholder="365, 730"
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400" />
            <span className="text-[10px] text-gray-400">
              In your Duration column's unit. Days → 365, 730 · months → 12, 24 · years → 1, 2.
            </span>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Labels (comma-sep, optional)</span>
            <input value={chLabels} onChange={(e) => setChLabels(e.target.value)}
              placeholder="1 year, 2 years"
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400" />
            <span className="text-[10px] text-gray-400">
              One label per cut-point. Blank → auto "≤ 365". Count must match cut-points.
            </span>
          </label>
        </div>

        <div className="mt-2">
          <MultiSelect label="Adjustment covariates (optional)" columns={pickCols}
            selected={chCovariates} onChange={setChCovariates}
            excludeNames={[chDuration, chEvent, chPredictor].filter(Boolean)} />
        </div>

        <div className="flex items-start gap-2 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 mt-1">
          <span className="text-amber-500 text-sm leading-none mt-0.5">ⓘ</span>
          <p className="text-[11px] text-amber-800 leading-relaxed">
            <b>Cut-points use the same time unit as your Duration column.</b> Follow-up in days → write
            {" "}<span className="font-mono">365, 730</span>; in months → <span className="font-mono">12, 24</span>;
            in years → <span className="font-mono">1, 2</span>. A <b>"Full follow-up"</b> row (all events,
            no censoring) is appended automatically. Each window applies administrative censoring at its
            cut-point, so shorter windows have fewer events and wider CIs.
          </p>
        </div>

        <div className="flex items-center gap-3 flex-wrap mt-2">
          <RunButton
            loading={chLoading}
            label="Run horizons"
            onClick={async () => {
              if (!chDuration || !chEvent || !chPredictor) {
                setChError("Select duration, event, and a predictor.");
                return;
              }
              const horizons = chHorizons.split(",").map((s) => parseFloat(s.trim())).filter((x) => !Number.isNaN(x) && x > 0);
              if (horizons.length === 0) {
                setChError("Enter at least one positive horizon cut-point.");
                return;
              }
              const labels = chLabels.split(",").map((s) => s.trim()).filter(Boolean);
              setChResult(null); setChError(null); setChLoading(true);
              try {
                const res = await runCoxHorizons({
                  session_id: sid,
                  duration_col: chDuration,
                  event_col: chEvent,
                  predictor: chPredictor,
                  covariates: chCovariates.length ? chCovariates : undefined,
                  horizons,
                  horizon_labels: labels.length === horizons.length ? labels : undefined,
                  include_full: true,
                });
                setChResult(res.data);
              } catch (e: unknown) {
                setChError(errorMessage(e, "Time-horizon analysis failed"));
              } finally {
                setChLoading(false);
              }
            }}
          />
          {chResult && chForestRows.length > 0 && (
            <button
              onClick={() => {
                const cov = (chResult.covariates ?? []) as string[];
                // Keep p + event counts in the figure — richer than the
                // bare reference look. Right header reflects that content.
                setForestHandoff(chForestRows, {
                  customTitle: "",
                  customSubtitle: cov.length ? `Adjusted for ${cov.join(" + ")}` : "(unadjusted; red = 95% CI excludes 1)",
                  xLabel: `${cov.length ? "Adjusted" : "Unadjusted"} hazard ratio for ${chResult.predictor} (95% CI), log scale`,
                  leftHeader: "Time horizon",
                  rightHeader: "HR (95% CI), p",
                });
                setVisualSubTab("forest");
                setActiveTab("visual");
              }}
              className="px-4 py-2 text-sm font-medium bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors"
            >
              → Send to Forest Builder
            </button>
          )}
        </div>

        {chError && <p className="text-sm text-red-500 mt-2">{chError}</p>}

        {chResult && (
          <div className="mt-3 space-y-3">
            <div className="overflow-x-auto rounded-xl border border-gray-200">
              <table className="w-full text-xs">
                <thead className="bg-gray-50 text-gray-500">
                  <tr>
                    <th className="text-left px-3 py-2 font-medium">Horizon</th>
                    <th className="text-right px-3 py-2 font-medium">n events</th>
                    <th className="text-right px-3 py-2 font-medium">HR</th>
                    <th className="text-right px-3 py-2 font-medium">95% CI</th>
                    <th className="text-right px-3 py-2 font-medium"><i>p</i></th>
                  </tr>
                </thead>
                <tbody>
                  {(chResult.forest_rows ?? []).map((row, i) => (
                    <tr key={i} className="border-t border-gray-100">
                      <td className="px-3 py-1.5 text-gray-800">{row.label}</td>
                      <td className="px-3 py-1.5 text-right text-gray-500">{(row.extra ?? "").replace(/[()]/g, "").replace(" events", "")}</td>
                      <td className="px-3 py-1.5 text-right font-semibold text-gray-900">{row.est != null ? row.est.toFixed(2) : "—"}</td>
                      <td className="px-3 py-1.5 text-right text-gray-600">
                        {row.ci_low != null && row.ci_high != null ? `${row.ci_low.toFixed(2)}–${row.ci_high.toFixed(2)}` : "—"}
                      </td>
                      <td className="px-3 py-1.5 text-right text-gray-600">{fmtP(row.p)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {chResult.interpretation && (
              <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-4 py-3 text-sm text-indigo-900">
                {chResult.interpretation}
              </div>
            )}
          </div>
        )}
      </Section>
      )}
      </div>
    </div>
  );
}
