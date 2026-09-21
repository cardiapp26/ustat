/* eslint-disable react-refresh/only-export-components -- shared helpers + small components bundled by design */
// Shared presentational helpers for the model result views.

// ── Shared model result types ────────────────────────────────────────────────
// These describe the (loosely-typed) JSON returned by the regression endpoints.
// Every field is optional because the exact shape depends on the model family
// (linear / logistic / Poisson / Cox / GEE / GLM). Components read whichever
// fields apply to their branch.

/** A single coefficient row across all model families. */
export interface Coefficient {
  variable: string;
  estimate?: number;
  se?: number;
  t?: number;
  z?: number;
  p: number;
  ci_low?: number;
  ci_high?: number;
  // Logistic
  log_odds?: number;
  odds_ratio?: number;
  or_ci_low?: number;
  or_ci_high?: number;
  // Cox
  log_hr?: number;
  hr?: number;
  hr_ci_low?: number;
  hr_ci_high?: number;
  // Poisson / Negative Binomial
  log_irr?: number;
  irr?: number;
  irr_ci_low?: number;
  irr_ci_high?: number;
  // Exponentiated estimate (GLM exp(β))
  exp_estimate?: number;
}

/** A univariate/multivariate odds-ratio comparison row (ORTable / forest). */
export interface ORRow {
  variable: string;
  uni_or?: number | null;
  uni_ci_low?: number | null;
  uni_ci_high?: number | null;
  uni_p?: number | null;
  multi_or?: number | null;
  multi_ci_low?: number | null;
  multi_ci_high?: number | null;
  multi_p?: number | null;
}

/** Per-predictor metadata returned alongside a prediction model. */
export interface PredictorInfo {
  type: "numeric" | "categorical";
  median?: number;
  mean?: number;
  min?: number;
  max?: number;
  categories?: string[];
}

/** Linear-prediction model result consumed by the interactive predictor. */
export interface PredictionResult {
  predictor_info?: Record<string, PredictorInfo>;
  coefficients?: Coefficient[];
  outcome?: string;
  residual_se?: number;
  df_resid?: number;
  n?: number;
  r_squared?: number;
  adj_r_squared?: number;
}

/** Classification metrics block for a logistic model summary. */
export interface Classification {
  accuracy: number;
  sensitivity: number;
  specificity: number;
  ppv: number;
  npv: number;
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}

/** A chi-square style fit/calibration test. */
export interface ChiSquareTest {
  chi2?: number;
  df?: number;
  p: number;
}

/** Logistic-model summary block (fit + calibration + classification). */
export interface ModelSummary {
  classification?: Classification;
  hosmer_lemeshow?: ChiSquareTest;
  omnibus?: ChiSquareTest;
  minus2ll?: number;
  cox_snell_r2?: number;
  nagelkerke_r2?: number;
  auc?: number;
}

/** Inputs the forest plot reads from a fitted regression result. */
export interface ForestResult {
  table?: ORRow[];
  coefficients?: Coefficient[];
}

/** One Cox term's HR statistics for a single model column. */
export interface HRStat {
  hr: number | null;
  hr_ci_low: number | null;
  hr_ci_high: number | null;
  p: number | null;
}

/** A Cox univariable/parsimonious/adjusted HR table row. */
export interface HRRow {
  term: string;
  predictor: string;
  kind: "numeric" | "category";
  category: string | null;
  reference: string | null;
  unadjusted: HRStat | null;
  parsimonious: HRStat | null;
  adjusted: HRStat | null;
}

/**
 * Loosely-typed regression result returned by the model endpoints. Every field
 * is optional because the exact shape depends on the model family. Structurally
 * compatible with the narrower result types each child view consumes
 * (PredictionResult / ForestResult).
 */
export interface ModelResult {
  model?: string;
  outcome?: string;
  n?: number;
  n_total?: number;
  n_excluded?: number;
  n_events?: number;
  n_events_pars?: number;
  n_pars?: number;
  n_multi?: number;
  imputation?: string;
  selection_method?: string;
  duration_col?: string;
  event_col?: string;
  r_squared?: number;
  adj_r_squared?: number;
  pseudo_r2?: number;
  f_stat?: number;
  aic?: number;
  bic?: number;
  concordance?: number;
  result_text?: string;
  coefficients?: Coefficient[];
  predictor_info?: Record<string, PredictorInfo>;
  residual_se?: number;
  df_resid?: number;
  table?: ORRow[];
  rows?: HRRow[];
  model_stats?: ModelSummary;
  omnibus?: ChiSquareTest;
  brant_proportional_odds?: BrantTest;
  use_firth?: boolean;
  ci_method?: string;
  method_note?: string;
}

export interface BrantTest {
  computed: boolean;
  reason?: string;
  note?: string;
  omnibus?: { chi2: number; df: number; p: number; violation: boolean };
  by_predictor?: { variable: string; chi2: number; df: number; p: number; violation: boolean }[];
}

export function adjustP(p: number, beta: number, nullHyp: string): number {
  if (nullHyp === "leq") return beta > 0 ? Math.min(p / 2, 1) : Math.min(1 - p / 2, 1);
  if (nullHyp === "geq") return beta < 0 ? Math.min(p / 2, 1) : Math.min(1 - p / 2, 1);
  return p; // "eq" = two-tailed default
}

// ── Mini bell-curve (sampling distribution of the estimator) ─────────────────
export function MiniNormalSVG({ beta, se, p }: { beta: number; se: number; p: number }) {
  if (!isFinite(beta) || !isFinite(se) || se <= 0)
    return <span className="text-amber-400 text-[11px]">⚠</span>;
  const W = 64, H = 24, span = 3.8 * se;
  const lo = beta - span, hi = beta + span;
  const N  = 60;
  const toSX = (x: number) => ((x - lo) / (hi - lo)) * W;
  const toSY = (y: number) => H - 2 - y * (H - 4);
  const pts = Array.from({ length: N + 1 }, (_, i) => {
    const x = lo + (hi - lo) * i / N;
    return [x, Math.exp(-0.5 * ((x - beta) / se) ** 2)] as [number, number];
  });
  const curve = pts.map(([x, y]) => `${toSX(x).toFixed(1)},${toSY(y).toFixed(1)}`).join(" ");
  const fill  = [`0,${H}`, ...pts.map(([x, y]) => `${toSX(x).toFixed(1)},${toSY(y).toFixed(1)}`), `${W},${H}`].join(" ");
  const zx    = toSX(0);
  const color = p < 0.001 ? "#3730a3" : p < 0.01 ? "#4338ca" : p < 0.05 ? "#6366f1" : "#9ca3af";
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
      <polygon points={fill}  fill={`${color}${p < 0.05 ? "22" : "0e"}`} />
      <polyline points={curve} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
      {zx >= 0 && zx <= W && (
        <line x1={zx.toFixed(1)} y1="1" x2={zx.toFixed(1)} y2={H}
          stroke="#9ca3af" strokeWidth="0.8" strokeDasharray="2,2" />
      )}
    </svg>
  );
}

// ── Significance bar ──────────────────────────────────────────────────────────
export function SigBar({ p }: { p: number }) {
  const pct   = p < 0.001 ? 100 : p < 0.01 ? 80 : p < 0.05 ? 55 : p < 0.1 ? 22 : 7;
  const color = p < 0.001 ? "#3730a3" : p < 0.01 ? "#4338ca" : p < 0.05 ? "#6366f1" : "#d1d5db";
  return (
    <div style={{ width: 56, height: 10, backgroundColor: "#f3f4f6", borderRadius: 3, overflow: "hidden" }}>
      <div style={{ width: `${pct}%`, height: "100%", backgroundColor: color }} />
    </div>
  );
}
