import { useState, useEffect, useMemo, useRef, type ReactNode } from "react";
import TitledPlot from "./TitledPlot";
import ResultExporter from "./ResultExporter";
import { useStore, PALETTES, isNumericKind, type Session } from "../store";
import { runROC, runROCCompare, runROCMultiCompare, runROCCombined } from "../api";
import { Tip, InfoBanner } from "./Tip";
import { MissingGuard, type ImputationStrategy } from "./MissingGuard";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { fmtP } from "../lib/format";
import type { PlotData, PlotCaptureHandle } from "../lib/plotTypes";

// ── Result shapes (loose views over the untyped API responses) ──────────────
/** A single point on an ROC curve. */
interface CurvePoint { fpr: number; tpr: number; }
/** Sens/Spec/PPV/etc. block returned for a cutoff. */
interface CutoffMetrics {
  cutoff?: number | null;
  sensitivity?: number | null;
  specificity?: number | null;
  ppv?: number | null;
  npv?: number | null;
  accuracy?: number | null;
  lr_pos?: number | null;
  lr_neg?: number | null;
  youden_j?: number | null;
  tp?: number | null;
  tn?: number | null;
  fp?: number | null;
  fn?: number | null;
}
/** Loose view of the /roc single-result payload. */
interface ROCResult {
  auc: number;
  auc_p?: number | null;
  auc_se?: number | null;
  auc_z?: number | null;
  ci_lower?: number | null;
  ci_upper?: number | null;
  curve: CurvePoint[];
  optimal?: CutoffMetrics;
  manual?: CutoffMetrics;
  optimal_cutoff?: number | null;
  sensitivity?: number | null;
  specificity?: number | null;
  direction_requested?: "auto" | "higher" | "lower";
  direction_used?: "higher" | "lower";
  direction_flipped?: boolean;
  result_text?: string;
  imputation?: string;
  n?: number;
  n_excluded?: number | null;
  n_positive?: number;
  n_negative?: number;
  tp?: number | null;
  tn?: number | null;
  fp?: number | null;
  fn?: number | null;
}
/** Loose view of the /roc_compare (DeLong) payload. */
interface ROCCompareResult {
  score_1: string;
  score_2: string;
  auc_1: number;
  auc_2: number;
  ci_1_low: number;
  ci_1_high: number;
  ci_2_low: number;
  ci_2_high: number;
  ci_diff_low: number;
  ci_diff_high: number;
  difference: number;
  z: number;
  p: number;
  n: number;
  significant: boolean;
  interpretation: string;
  curve_1?: CurvePoint[];
  curve_2?: CurvePoint[];
  direction_1_requested?: "auto" | "higher" | "lower";
  direction_2_requested?: "auto" | "higher" | "lower";
  direction_1_flipped?: boolean;
  direction_2_flipped?: boolean;
}
/** One row of the pairwise DeLong matrix. */
interface DelongPair {
  a: string;
  b: string;
  delta_auc: number;
  ci_low: number;
  ci_high: number;
  p_raw: number;
  p_adj: number;
  significant: boolean;
}
/** Loose view of the /roc_multi_compare payload. */
interface ROCMultiDelong {
  pairs?: DelongPair[];
  scores?: string[];
  n: number;
  n_pairs: number;
  p_adjust: string;
}
/** Shape of an axios-style error with a backend `detail` field. */
interface ApiError {
  message?: string;
  response?: { data?: { detail?: unknown } };
}

/** Extract a human-readable message from an unknown API rejection. */
function apiErrorMessage(e: unknown, fallback: string): string {
  const err = e as ApiError;
  const detail = err?.response?.data?.detail;
  if (Array.isArray(detail)) {
    return detail
      .map((m) => (m as { msg?: string }).msg ?? String(m))
      .join(", ");
  }
  if (typeof detail === "string") return detail;
  return err?.message ?? fallback;
}

// ── Helper to get current palette primary color ────────────────────────────
const _pal = () => PALETTES[useStore.getState().plotTheme.palette] ?? PALETTES.indigo;
const _p0 = () => _pal()[0];

// ── Constants ─────────────────────────────────────────────────────────────────

const PLOT_LAYOUT: Record<string, unknown> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "#ffffff",
  font: { color: "#374151", size: 12 },
  margin: { t: 40, r: 20, b: 50, l: 60 },
  xaxis: { gridcolor: "#e5e7eb", title: { text: "1 − Specificity (FPR)" }, range: [0, 1] },
  yaxis: { gridcolor: "#e5e7eb", title: { text: "Sensitivity (TPR)" }, range: [0, 1] },
};

const MULTI_PALETTE = [
  "#dc2626","#2563eb","#f59e0b","#16a34a",
  "#7c3aed","#0891b2","#be185d","#92400e",
  "#ea580c","#4f46e5",
];
const ROC_DASHES  = ["solid","dash","dot","dashdot"] as const;
const ROC_WIDTHS  = [1, 1.5, 2, 2.5, 3, 4];

// ── Helpers ───────────────────────────────────────────────────────────────────

const aucColor = (auc: number) =>
  auc >= 0.9 ? "text-green-600" : auc >= 0.8 ? "text-blue-600" : auc >= 0.7 ? "text-amber-600" : "text-red-500";
const aucLabel = (auc: number) =>
  auc >= 0.9 ? "Excellent" : auc >= 0.8 ? "Good" : auc >= 0.7 ? "Fair" : "Poor";
const fmtPct = (v?: number | null) => v == null ? "—" : `${(v * 100).toFixed(1)}%`;
const fmtAUC = (auc: number, lo?: number | null, hi?: number | null) =>
  lo != null && hi != null
    ? `AUC ${auc.toFixed(2)} (95% CI ${lo.toFixed(2)}–${hi.toFixed(2)})`
    : `AUC ${auc.toFixed(2)}`;

// ── ROC Guidance ────────────────────────────────────────────────────────────
const ROC_GUIDANCE = {
  single: {
    use: "Evaluate how well a single continuous predictor (biomarker, score) discriminates between two groups (e.g. disease vs. healthy, event vs. no event).",
    check: "Outcome must be binary 0/1. The predictor should have a reasonable spread of values. AUC = 0.5 means no better than chance.",
    interpret: "AUC 0.9-1.0 = Excellent, 0.8-0.9 = Good, 0.7-0.8 = Fair, <0.7 = Poor. Youden's index gives the optimal cut-off that maximises Sensitivity + Specificity. Report: AUC (95% CI).",
  },
  compare: {
    use: "Compare two biomarkers' discriminative ability using the DeLong test. Essential for proving a new marker outperforms an existing one.",
    check: "Both markers measured on the SAME patients (paired design). DeLong test is valid for correlated AUCs from the same sample.",
    interpret: "p < 0.05 means one AUC is significantly higher. Report: AUC\u2081 vs AUC\u2082, \u0394AUC (95% CI), DeLong p. The overlaid ROC plot visualises the difference.",
  },
  multi: {
    use: "Screen multiple biomarkers simultaneously to find the best single predictor. Overlay ROC curves for visual comparison.",
    check: "Each predictor is evaluated independently against the same binary outcome. The combined model uses cross-validated predictions to avoid overfitting bias.",
    interpret: "Compare AUCs and 95% CIs across predictors. Overlapping CIs suggest no significant difference. The combined model shows the joint discriminative power of all predictors together.",
  },
};

// ── Types ─────────────────────────────────────────────────────────────────────

interface CurveStyle { color: string; width: number; dash: string; }
interface MultiResult {
  col: string;
  auc: number;
  ci_lower?: number;
  ci_upper?: number;
  curve: { fpr: number; tpr: number }[];
  error?: string;
}

const defaultStyle = (i: number): CurveStyle => ({
  color: MULTI_PALETTE[i % MULTI_PALETTE.length],
  width: 2,
  dash: "solid",
});

// ── Metrics block (single mode) ───────────────────────────────────────────────

// Trimmed from full clinical explanations to one-line definitions so the
// hover popovers don't sprawl across the screen. The full glossary is still
// in the methods appendix for anyone who needs the full context.
const METRIC_TIPS: Record<string, string> = {
  "Cutoff":      "Threshold for predicting positive (score ≥ cutoff).",
  "Sensitivity": "TP rate — % of true cases correctly identified.",
  "Specificity": "TN rate — % of true non-cases correctly ruled out.",
  "PPV":         "Of predicted positives, % truly diseased. Depends on prevalence.",
  "NPV":         "Of predicted negatives, % truly healthy.",
  "Accuracy":    "(TP + TN) / total. Misleading when classes are imbalanced.",
  "LR+":         "Likelihood Ratio +: post-test odds ↑. >10 = strong evidence.",
  "LR−":         "Likelihood Ratio −: post-test odds ↓. <0.1 = strong evidence.",
  "Youden J":    "Sens + Spec − 1. Optimal cutoff maximises J.",
  "TP": "True Positives.", "TN": "True Negatives.",
  "FP": "False Positives (Type I error).",
  "FN": "False Negatives (Type II error).",
};

function MetricsBlock({ m, label }: { m: CutoffMetrics; label: string }) {
  return (
    <div className="space-y-0.5">
      <p className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold mt-2 mb-1">{label}</p>
      {[
        ["Cutoff",      m.cutoff],
        ["Sensitivity", m.sensitivity != null ? fmtPct(m.sensitivity) : "—"],
        ["Specificity", m.specificity != null ? fmtPct(m.specificity) : "—"],
        ["PPV",         m.ppv       != null ? fmtPct(m.ppv)       : "—"],
        ["NPV",         m.npv       != null ? fmtPct(m.npv)       : "—"],
        ["Accuracy",    m.accuracy  != null ? fmtPct(m.accuracy)  : "—"],
        ["LR+",         m.lr_pos    != null ? m.lr_pos.toFixed(2) : "—"],
        ["LR−",         m.lr_neg    != null ? m.lr_neg.toFixed(2) : "—"],
        ["Youden J",    m.youden_j  != null ? m.youden_j.toFixed(2) : "—"],
        ["TP", m.tp], ["TN", m.tn], ["FP", m.fp], ["FN", m.fn],
      ].map(([k, v]) => {
        const key = k as string;
        return (
        <div key={key} className="flex justify-between border-b border-gray-100 py-0.5 text-xs">
          <span className="text-gray-400 flex items-center">
            {key}
            {METRIC_TIPS[key] && <Tip text={METRIC_TIPS[key]} wide />}
          </span>
          <span className="text-gray-700 font-mono">{v}</span>
        </div>
        );
      })}
    </div>
  );
}

// ── StyleRow: one row of color/width/dash controls ────────────────────────────

function StyleRow({
  label, color, width, dash, onColor, onWidth, onDash,
}: {
  label: string; color: string; width: number; dash: string;
  onColor: (v: string) => void; onWidth: (v: number) => void; onDash: (v: string) => void;
}) {
  return (
    <div className="flex items-center gap-2 min-w-0">
      <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: color }} />
      <span className="text-xs text-gray-600 truncate flex-1 min-w-0" title={label}>{label}</span>
      <input type="color" value={color} onChange={(e) => onColor(e.target.value)}
        className="w-6 h-6 rounded cursor-pointer border border-gray-300 flex-shrink-0" />
      <select className="select text-xs py-0 px-1 flex-shrink-0" value={width}
        onChange={(e) => onWidth(Number(e.target.value))}>
        {ROC_WIDTHS.map((w) => <option key={w} value={w}>{w}px</option>)}
      </select>
      <select className="select text-xs py-0 px-1 flex-shrink-0" value={dash}
        onChange={(e) => onDash(e.target.value)}>
        {ROC_DASHES.map((d) => <option key={d} value={d}>{d}</option>)}
      </select>
    </div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function ROCPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <ROCPanelBody session={session} />;
}

function ROCPanelBody({ session }: { session: Session }) {
  const showGrid = useStore((s) => s.showGrid);

  const numCols = session.columns.filter((c) => isNumericKind(c.kind) && !c.analysis_excluded).map((c) => c.name);
  const allCols = session.columns.filter((c) => !c.analysis_excluded).map((c) => c.name);

  // Binary columns (≤ 2 unique non-null values, both ∈ {0, 1}) — ROC outcome
  // must be 0/1. Falls back to allCols if no binary column is detected so
  // the user can still type-override via the Dictionary modal.
  const binaryCols = useMemo(() => {
    const out: string[] = [];
    for (const col of session.columns) {
      if (col.analysis_excluded) continue;
      const vals = new Set<unknown>();
      for (const row of session.preview) {
        const v = row[col.name];
        if (v == null || v === "") continue;
        vals.add(typeof v === "number" ? v : Number(v));
        if (vals.size > 2) break;
      }
      const arr = [...vals];
      if (arr.length === 0 || arr.length > 2) continue;
      if (arr.every((v) => v === 0 || v === 1)) out.push(col.name);
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.session_id]);
  // Prefer a binary 0/1 column whose name hints at an outcome; otherwise the
  // first binary column; only fall back to allCols when no binary exists.
  const defaultOutcome =
    binaryCols.find((c) => /mortalite|death|event|outcome|binary|status/i.test(c))
    ?? binaryCols[0]
    ?? allCols.find((c) => /mortalite|death|event|outcome|binary|status/i.test(c))
    ?? allCols[0]
    ?? "";

  // ── Mode ──
  const [mode, setMode] = usePersistedPanelState<"single" | "multi">("roc", "mode", "single");

  // ── Shared ──
  const [outcomeCol, setOutcomeCol] = usePersistedPanelState<string>("roc", "outcomeCol", defaultOutcome);
  const rocSingleRef = useRef<PlotCaptureHandle | null>(null);
  const rocCompareRef = useRef<PlotCaptureHandle | null>(null);
  const rocMultiRef = useRef<PlotCaptureHandle | null>(null);

  // ── Single-curve state ──
  const [scoreCol,     setScoreCol]     = usePersistedPanelState<string>("roc", "scoreCol", numCols[0] ?? "");
  const [manualCutoff, setManualCutoff] = usePersistedPanelState<string>("roc", "manualCutoff", "");
  // Score direction for ROC. "auto" flips the score sign when the naive
  // AUC < 0.5 so protective biomarkers (albumin, eGFR, Hb) report the
  // correct ~0.69 instead of the inverted ~0.31 — see Heinze (2002) and
  // pROC's `direction="auto"` for the same convention.
  const [scoreDirection, setScoreDirection]   = usePersistedPanelState<"auto" | "higher" | "lower">("roc", "scoreDirection", "auto");
  const [scoreDirection2, setScoreDirection2] = usePersistedPanelState<"auto" | "higher" | "lower">("roc", "scoreDirection2", "auto");
  const [useManual,    setUseManual]    = usePersistedPanelState<boolean>("roc", "useManual", false);
  const [result,       setResult]       = usePersistedPanelState<ROCResult | null>("roc", "result", null);
  const [error,        setError]        = useState<string | null>(null);
  const [loading,      setLoading]      = useState(false);
  const [imputation,   setImputation]   = usePersistedPanelState<ImputationStrategy>("roc", "imputation", "listwise");

  const [showCompare, setShowCompare] = usePersistedPanelState<boolean>("roc", "showCompare", false);
  const [scoreCol2,   setScoreCol2]   = usePersistedPanelState<string>("roc", "scoreCol2", numCols[1] ?? numCols[0] ?? "");
  const [cmpResult,   setCmpResult]   = usePersistedPanelState<ROCCompareResult | null>("roc", "cmpResult", null);
  const [cmpError,    setCmpError]    = useState<string | null>(null);
  const [cmpLoading,  setCmpLoading]  = useState(false);

  const [singleStyle, setSingleStyle] = usePersistedPanelState<CurveStyle>("roc", "singleStyle", { color: _p0(), width: 2.5, dash: "solid" });
  const [chanceStyle, setChanceStyle] = usePersistedPanelState<CurveStyle>("roc", "chanceStyle", { color: "#9ca3af", width: 1,   dash: "dash"  });

  useEffect(() => {
    if (result) { setSingleStyle({ color: _p0(), width: 2.5, dash: "solid" }); }
    // Re-seed style only when a *new* curve arrives (auc or n change). Adding
    // `result` itself would also reset the user's manual style edits on every
    // unrelated re-render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result?.auc, result?.n]);

  // ── Multi-curve state ──
  const [multiCols,    setMultiCols]    = usePersistedPanelState<string[]>("roc", "multiCols", []);
  const [multiResults, setMultiResults] = usePersistedPanelState<MultiResult[]>("roc", "multiResults", []);
  const [multiStyles,  setMultiStyles]  = usePersistedPanelState<CurveStyle[]>("roc", "multiStyles", []);
  const [multiLoading, setMultiLoading] = useState(false);
  const [multiError,   setMultiError]   = useState<string | null>(null);
  // Multi-curve DeLong pairwise matrix (K-way generalisation of /roc_compare).
  const [multiDelong,    setMultiDelong]    = usePersistedPanelState<ROCMultiDelong | null>("roc", "multiDelong", null);
  const [multiDelongErr, setMultiDelongErr] = useState<string | null>(null);
  const [multiPAdjust,   setMultiPAdjust]   = usePersistedPanelState<"holm" | "bonferroni" | "none">("roc", "multiPAdjust", "holm");
  const [multiChance,  setMultiChance]  = usePersistedPanelState<CurveStyle>("roc", "multiChance", { color: "#9ca3af", width: 1, dash: "dash" });

  // Re-run pairwise DeLong with the new adjustment when the user changes
  // the dropdown — only valid when we already have per-curve results.
  // Declared AFTER the multi-curve state it closes over to avoid a
  // temporal-dead-zone ReferenceError on the dependency array.
  useEffect(() => {
    if (!multiResults.length || !session?.session_id) return;
    const successCols = multiResults.filter((r) => !r.error).map((r) => r.col);
    if (successCols.length < 2) return;
    let cancelled = false;
    (async () => {
      try {
        const dl = await runROCMultiCompare({
          session_id: session.session_id,
          score_columns: successCols,
          outcome_column: outcomeCol,
          directions: successCols.map(() => scoreDirection),
          p_adjust: multiPAdjust,
        });
        if (!cancelled) { setMultiDelong(dl.data); setMultiDelongErr(null); }
      } catch (e: unknown) {
        const msg = apiErrorMessage(e, "DeLong matrix failed");
        if (!cancelled) { setMultiDelong(null); setMultiDelongErr(msg); }
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [multiPAdjust]);

  // ── Combined model state ──
  const [showCombined,     setShowCombined]     = usePersistedPanelState<boolean>("roc", "showCombined", false);
  const [combinedCols,     setCombinedCols]     = usePersistedPanelState<string[]>("roc", "combinedCols", []);
  const [combinedName,     setCombinedName]     = usePersistedPanelState<string>("roc", "combinedName", "Combined Model");
  const [combinedResult,   setCombinedResult]   = usePersistedPanelState<MultiResult | null>("roc", "combinedResult", null);
  const [combinedStyle,    setCombinedStyle]    = usePersistedPanelState<CurveStyle>("roc", "combinedStyle", { color: "#dc2626", width: 3, dash: "solid" });
  const [combinedLoading,  setCombinedLoading]  = useState(false);
  const [combinedError,    setCombinedError]    = useState<string | null>(null);
  const [multiFilter,     setMultiFilter]     = useState("");
  const [combinedFilter,  setCombinedFilter]  = useState("");

  const toggleCombinedCol = (col: string) =>
    setCombinedCols(combinedCols.includes(col) ? combinedCols.filter((c) => c !== col) : [...combinedCols, col]);

  const toggleMultiCol = (col: string) => {
    setMultiCols(
      multiCols.includes(col) ? multiCols.filter((c) => c !== col) : [...multiCols, col]
    );
    setMultiResults([]);
  };

  const runCombined = async () => {
    if (!combinedCols.length || !outcomeCol) return;
    setCombinedLoading(true); setCombinedError(null); setCombinedResult(null);
    try {
      const res = await runROCCombined({
        session_id: session.session_id,
        predictor_columns: combinedCols,
        outcome_column: outcomeCol,
        model_name: combinedName || "Combined Model",
      });
      const d = res.data;
      setCombinedResult({ col: combinedName || "Combined Model", auc: d.auc, curve: d.curve });
    } catch (e: unknown) {
      setCombinedError(apiErrorMessage(e, "Failed"));
    } finally { setCombinedLoading(false); }
  };

  const runMulti = async () => {
    if (!multiCols.length || !outcomeCol) return;
    setMultiLoading(true); setMultiError(null); setMultiResults([]);
    setMultiDelong(null); setMultiDelongErr(null);
    setMultiStyles(multiCols.map((_, i) => defaultStyle(i)));
    // Also run combined model if enabled
    if (showCombined && combinedCols.length > 0) runCombined();
    try {
      const settled = await Promise.allSettled(
        multiCols.map((col) =>
          runROC({
            session_id: session.session_id,
            score_column: col,
            outcome_column: outcomeCol,
            imputation,
            direction: scoreDirection,
          })
        )
      );
      const results: MultiResult[] = settled.map((s, i) => {
        if (s.status === "fulfilled") {
          const d = s.value.data;
          return {
            col: multiCols[i],
            auc: d.auc,
            ci_lower: d.ci_lower,
            ci_upper: d.ci_upper,
            curve: d.curve,
          };
        } else {
          // Capture backend detail so the user sees what went wrong instead
          // of a generic "Failed" / "err" badge.
          const reason: unknown = (s as PromiseRejectedResult).reason;
          const msg = apiErrorMessage(reason, "Failed");
          return { col: multiCols[i], auc: 0, curve: [], error: msg };
        }
      });
      setMultiResults(results);

      // Pairwise DeLong matrix — only when at least 2 columns succeeded.
      const successCols = results.filter((r) => !r.error).map((r) => r.col);
      if (successCols.length >= 2) {
        try {
          const dl = await runROCMultiCompare({
            session_id: session.session_id,
            score_columns: successCols,
            outcome_column: outcomeCol,
            directions: successCols.map(() => scoreDirection),
            p_adjust: multiPAdjust,
          });
          setMultiDelong(dl.data);
          setMultiDelongErr(null);
        } catch (e: unknown) {
          setMultiDelong(null);
          setMultiDelongErr(apiErrorMessage(e, "DeLong matrix failed"));
        }
      } else {
        setMultiDelong(null);
        setMultiDelongErr(null);
      }
    } catch (e: unknown) {
      setMultiError(e instanceof Error ? e.message : "Request failed");
    } finally { setMultiLoading(false); }
  };

  // ── Single ROC run ──
  const run = async () => {
    if (!scoreCol || !outcomeCol) return;
    if (scoreCol === outcomeCol) { setError("Score and outcome columns must be different"); return; }
    setLoading(true); setError(null); setResult(null); setCmpResult(null);
    const mc = useManual && manualCutoff !== "" ? parseFloat(manualCutoff) : undefined;
    try {
      const res = await runROC({
        session_id: session.session_id,
        score_column: scoreCol,
        outcome_column: outcomeCol,
        imputation,
        direction: scoreDirection,
        ...(mc != null && !isNaN(mc) ? { manual_cutoff: mc } : {}),
      });
      setResult(res.data);
    } catch (e: unknown) {
      const err = e as ApiError;
      const msg = err.response?.data?.detail;
      setError(Array.isArray(msg)
        ? msg.map((m) => (m as { msg?: string }).msg).join(", ")
        : ((msg as string | undefined) ?? err.message ?? "Request failed"));
    } finally { setLoading(false); }
  };

  const runCompare = async () => {
    if (scoreCol === scoreCol2) { setCmpError("Select two different score columns"); return; }
    setCmpLoading(true); setCmpError(null); setCmpResult(null);
    try {
      const res = await runROCCompare({
        session_id: session.session_id,
        score_column_1: scoreCol,
        score_column_2: scoreCol2,
        outcome_column: outcomeCol,
        direction_1: scoreDirection,
        direction_2: scoreDirection2,
      });
      setCmpResult(res.data);
    } catch (e: unknown) {
      const err = e as ApiError;
      const msg = err.response?.data?.detail;
      setCmpError(Array.isArray(msg)
        ? msg.map((m) => (m as { msg?: string }).msg).join(", ")
        : ((msg as string | undefined) ?? "Comparison failed"));
    } finally { setCmpLoading(false); }
  };

  // ── Exports ──
  // Two-column "metric / value" table — round-trips cleanly through CSV /
  // XLSX. ROC curve points (FPR, TPR per row) come after the summary block
  // in the same sheet so the user gets one self-contained file.
  const singleExport = useMemo(() => {
    if (!result) return null;
    const opt = result.optimal ?? {};
    const headers = ["Metric", "Value"];
    const rows: (string | number | null)[][] = [
      ["Score column", scoreCol],
      ["Outcome column", outcomeCol],
      ["Direction requested", result.direction_requested ?? scoreDirection],
      ["Direction used", result.direction_used ?? "—"],
      ["Auto direction flipped", result.direction_requested === "auto" && result.direction_flipped ? "Yes" : "No"],
      ["n", result.n ?? "—"],
      ["Positives (1)", result.n_positive ?? "—"],
      ["Negatives (0)", result.n_negative ?? "—"],
      ["AUC", result.auc],
      ["AUC interpretation", aucLabel(result.auc)],
      ["AUC SE (DeLong)", result.auc_se ?? "—"],
      ["95% CI lower", result.ci_lower ?? "—"],
      ["95% CI upper", result.ci_upper ?? "—"],
      ["AUC Z (vs 0.5)", result.auc_z ?? "—"],
      ["AUC p (H₀: AUC = 0.5)", fmtP(result.auc_p)],
      ["", ""],
      ["Optimal cutoff (Youden J)", ""],
      ["Cutoff", opt.cutoff ?? "—"],
      ["Sensitivity", fmtPct(opt.sensitivity)],
      ["Specificity", fmtPct(opt.specificity)],
      ["PPV", fmtPct(opt.ppv)],
      ["NPV", fmtPct(opt.npv)],
      ["Accuracy", fmtPct(opt.accuracy)],
      ["LR+", opt.lr_pos ?? "—"],
      ["LR-", opt.lr_neg ?? "—"],
      ["Youden J", opt.youden_j != null ? Number(opt.youden_j).toFixed(2) : "—"],
      ["TP", opt.tp ?? "—"],
      ["TN", opt.tn ?? "—"],
      ["FP", opt.fp ?? "—"],
      ["FN", opt.fn ?? "—"],
    ];
    if (result.manual) {
      rows.push(["", ""], ["Manual cutoff", ""],
        ["Cutoff", result.manual.cutoff ?? null],
        ["Sensitivity", fmtPct(result.manual.sensitivity)],
        ["Specificity", fmtPct(result.manual.specificity)],
        ["PPV", fmtPct(result.manual.ppv)],
        ["NPV", fmtPct(result.manual.npv)],
        ["Accuracy", fmtPct(result.manual.accuracy)],
      );
    }
    rows.push(["", ""], ["ROC Curve (FPR, TPR)", ""]);
    for (const p of result.curve) {
      rows.push([p.fpr.toFixed(6), p.tpr.toFixed(6)]);
    }
    return { headers, rows };
  }, [result, scoreCol, outcomeCol, scoreDirection]);

  const multiExport = useMemo(() => {
    if (!multiResults.length) return null;
    const headers = ["Variable", "AUC", "CI Lower", "CI Upper"];
    const rows: (string | number | null)[][] = multiResults.map((r) =>
      [r.col, r.auc, r.ci_lower ?? "", r.ci_upper ?? ""]
    );
    // Append each curve's FPR/TPR pairs after the summary so all data lands
    // in one file — section header rows mark the boundary.
    multiResults.forEach((r) => {
      if (r.curve.length) {
        rows.push(["", "", "", ""], [`Curve — ${r.col}`, "FPR", "TPR", ""]);
        r.curve.forEach((p) => rows.push(["", p.fpr.toFixed(6), p.tpr.toFixed(6), ""]));
      }
    });
    return { headers, rows };
  }, [multiResults]);

  // PNG / TIFF flow through ResultExporter \u2014 kept only for any legacy
  // call site (none remain after this refactor).

  // ── Derived ──
  const activeMetrics = useManual && result?.manual
    ? result.manual
    : result?.optimal ?? (result ? {
        cutoff: result.optimal_cutoff, sensitivity: result.sensitivity, specificity: result.specificity,
        ppv: null, npv: null, accuracy: null, lr_pos: null, lr_neg: null, youden_j: null,
        tp: result.tp, tn: result.tn, fp: result.fp, fn: result.fn,
      } : null);

  const updateMultiStyle = (i: number, patch: Partial<CurveStyle>) =>
    setMultiStyles((prev) => prev.map((s, j) => j === i ? { ...s, ...patch } : s));

  // ── Multi-curve plot traces (reference first so it sits behind, then
  // individual ROC step curves, then combined-model curve on top) ──
  const multiTraces = [
    // Reference diagonal — rendered first so the data curves sit on top.
    {
      type: "scatter", mode: "lines",
      x: [0, 1], y: [0, 1],
      line: { color: multiChance.color, width: multiChance.width, dash: multiChance.dash },
      name: "Reference",
      hoverinfo: "skip" as const,
    },
    // Individual predictors — step curves (`shape: hv`) to match the
    // matplotlib reference aesthetic. Legend label is `name: AUC X.XX`
    // (no embedded CI to keep the entry short and readable).
    ...multiResults
      .filter((r) => !r.error && r.curve.length > 0)
      .map((r, i) => {
        const st = multiStyles[i] ?? defaultStyle(i);
        return {
          type: "scatter", mode: "lines",
          x: r.curve.map((p) => p.fpr),
          y: r.curve.map((p) => p.tpr),
          line: { color: st.color, width: st.width, dash: st.dash, shape: "hv" as const },
          name: `${r.col}: AUC ${Number(r.auc).toFixed(2)}`,
          hovertemplate: `${r.col}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>`,
        };
      }),
    // Combined model — drawn last so it sits visually on top of the predictors.
    ...(showCombined && combinedResult && !combinedResult.error && combinedResult.curve.length > 0 ? [{
      type: "scatter", mode: "lines",
      x: combinedResult.curve.map((p) => p.fpr),
      y: combinedResult.curve.map((p) => p.tpr),
      line: { color: combinedStyle.color, width: combinedStyle.width, dash: combinedStyle.dash, shape: "hv" as const },
      name: `${combinedResult.col}: AUC ${Number(combinedResult.auc).toFixed(2)}`,
      hovertemplate: `${combinedResult.col}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>`,
    }] : []),
  ];

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div className="flex gap-4 h-full">

      {/* ── Left sidebar (controls + DeLong) ────────────────────────────────── */}
      {/* Holds the score / outcome / direction / manual-cutoff controls plus
          the DeLong AUC-comparison card (single mode). The single-mode
          result card (AUC tile + paragraph + metric table) lives in the
          right-hand column. */}
      <div className="w-[340px] flex-shrink-0 flex flex-col gap-3 overflow-y-auto">

        {/* Mode toggle */}
        <div className="flex rounded-lg overflow-hidden border border-gray-300 sticky top-0 z-10 bg-white">
          <button
            onClick={() => setMode("single")}
            className={`flex-1 text-xs py-1.5 font-medium transition-colors
              ${mode === "single" ? "bg-indigo-600 text-white" : "text-gray-500 hover:bg-gray-50"}`}>
            Single curve
          </button>
          <button
            onClick={() => setMode("multi")}
            className={`flex-1 text-xs py-1.5 font-medium transition-colors
              ${mode === "multi" ? "bg-indigo-600 text-white" : "text-gray-500 hover:bg-gray-50"}`}>
            Multi-curve
          </button>
        </div>

        {/* Outcome column (shared) */}
        <div className="panel space-y-2">
          <label className="text-xs text-gray-400 block">
            Binary outcome <span className="text-gray-300">(must be 0/1)</span>
            {binaryCols.length === 0 && <span className="ml-1 text-[10px] text-amber-600">⚠ no binary column detected — recode in Dictionary</span>}
          </label>
          <select className="select w-full text-xs" value={outcomeCol}
            onChange={(e) => { setOutcomeCol(e.target.value); setResult(null); setMultiResults([]); }}>
            {(binaryCols.length > 0 ? binaryCols : allCols).map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>

        {/* ── SINGLE MODE controls ── */}
        {mode === "single" && (
          <>
            <div className="panel space-y-3">
              <h3 className="text-sm font-semibold text-gray-700">Score / predictor</h3>
              <select className="select w-full text-xs" value={scoreCol}
                onChange={(e) => { setScoreCol(e.target.value); setResult(null); }}>
                {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>

              <div>
                <label className="text-[10px] text-gray-400 flex items-center mb-1">
                  Direction
                  <Tip text="Auto: flip score when AUC<0.5 (protective markers). Higher: risk score. Lower: protective." />
                </label>
                <div className="flex gap-0 rounded overflow-hidden border border-gray-200">
                  {(["auto", "higher", "lower"] as const).map((d) => (
                    <button key={d}
                      onClick={() => { setScoreDirection(d); setResult(null); setCmpResult(null); }}
                      className={`flex-1 text-[10px] py-1 transition-colors ${
                        scoreDirection === d ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"
                      }`}>
                      {d === "auto" ? "Auto" : d === "higher" ? "Higher = event" : "Lower = event"}
                    </button>
                  ))}
                </div>
                {result?.direction_requested === "auto" && result.direction_flipped && (
                  <p role="status" className="text-[10px] text-emerald-700 mt-1">
                    ⚙ Auto direction flipped to lower = event: low values of <span className="font-mono">{scoreCol}</span> predict the event (AUC describes the protective direction).
                  </p>
                )}
              </div>

              <div>
                <label className="flex items-center gap-2 text-xs text-gray-500 cursor-pointer mb-1">
                  <input type="checkbox" className="accent-indigo-500" checked={useManual}
                    onChange={(e) => { setUseManual(e.target.checked); if (!e.target.checked) setManualCutoff(""); }} />
                  Manual cutoff
                </label>
                {useManual && (
                  <input type="number" step="any" placeholder="e.g. 42.5"
                    className="select w-full text-xs" value={manualCutoff}
                    onChange={(e) => setManualCutoff(e.target.value)} />
                )}
              </div>

              <MissingGuard
                sessionId={session.session_id}
                columns={[scoreCol, outcomeCol].filter(Boolean)}
                imputation={imputation}
                onImputation={setImputation}
              >
                <button className="btn-primary w-full" onClick={run}
                  disabled={loading || !scoreCol || !outcomeCol}>
                  {loading ? "Computing…" : "Run ROC"}
                </button>
              </MissingGuard>

              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-2">
                  <p className="text-red-600 text-xs">{error}</p>
                </div>
              )}
            </div>

            {/* Single results card lives in the right-hand column; the
                DeLong comparison card stays here in the left sidebar so
                the workflow (pick two scores → compare) sits next to the
                main score / outcome / direction controls instead of
                getting pushed underneath the results column. */}
            <div className="panel space-y-3">
              <button className="flex items-center w-full" onClick={() => setShowCompare((v) => !v)}>
                <span className="text-sm font-semibold text-gray-700">AUC Comparison (DeLong)</span>
                <span className="ml-auto text-gray-400 text-xs">{showCompare ? "▲" : "▼"}</span>
              </button>
              {showCompare && (
                <>
                  <p className="text-xs text-gray-400">Non-parametric test comparing two score columns.</p>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Score 1</label>
                    <p className="text-xs text-indigo-600 font-mono truncate bg-indigo-50 rounded px-2 py-1">{scoreCol || "—"}</p>
                  </div>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Score 2</label>
                    <select className="select w-full text-xs" value={scoreCol2}
                      onChange={(e) => { setScoreCol2(e.target.value); setCmpResult(null); }}>
                      {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] text-gray-400 block mb-1">
                      Score 2 direction
                      <Tip text="Auto-flip when AUC<0.5 (recommended for protective markers vs risk scores)." />
                    </label>
                    <div className="flex gap-0 rounded overflow-hidden border border-gray-200">
                      {(["auto", "higher", "lower"] as const).map((d) => (
                        <button key={d}
                          onClick={() => { setScoreDirection2(d); setCmpResult(null); }}
                          className={`flex-1 text-[10px] py-1 transition-colors ${
                            scoreDirection2 === d ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"
                          }`}>
                          {d === "auto" ? "Auto" : d === "higher" ? "H" : "L"}
                        </button>
                      ))}
                    </div>
                  </div>
                  {(
                    (cmpResult?.direction_1_requested === "auto" && cmpResult.direction_1_flipped)
                    || (cmpResult?.direction_2_requested === "auto" && cmpResult.direction_2_flipped)
                  ) && (
                    <div className="text-[10px] text-emerald-700 bg-emerald-50 border border-emerald-200 rounded px-2 py-1 leading-tight">
                      ⚙ DeLong recomputed with auto-flipped direction:
                      {cmpResult.direction_1_requested === "auto" && cmpResult.direction_1_flipped && (
                        <> <span className="font-mono">{scoreCol}</span> (lower = event)</>
                      )}
                      {cmpResult.direction_1_requested === "auto" && cmpResult.direction_1_flipped
                        && cmpResult.direction_2_requested === "auto" && cmpResult.direction_2_flipped && " · "}
                      {cmpResult.direction_2_requested === "auto" && cmpResult.direction_2_flipped && (
                        <><span className="font-mono">{scoreCol2}</span> (lower = event)</>
                      )}
                    </div>
                  )}
                  <button className="btn-primary w-full" onClick={runCompare}
                    disabled={cmpLoading || !scoreCol || !scoreCol2 || !outcomeCol || scoreCol === scoreCol2}>
                    {cmpLoading ? "Testing…" : "Run DeLong Test"}
                  </button>
                  {cmpError && <p className="text-red-500 text-xs">{cmpError}</p>}
                  {cmpResult && (
                    <div className="space-y-2 mt-1">
                      <div className={`text-xs px-2 py-1.5 rounded font-semibold border flex items-center gap-1.5
                        ${cmpResult.significant ? "border-green-300 bg-green-50 text-green-700" : "border-gray-200 bg-gray-50 text-gray-500"}`}>
                        {cmpResult.significant ? "✓ Significant difference (p < 0.05)" : "No significant difference (p ≥ 0.05)"}
                      </div>

                      <div className="grid grid-cols-3 gap-1 text-center">
                        <div className="bg-blue-50 rounded p-1.5">
                          <p className="text-[9px] text-blue-400 uppercase tracking-wide font-semibold">Baseline AUC</p>
                          <p className="text-sm font-bold font-mono text-blue-700">{cmpResult.auc_2.toFixed(3)}</p>
                          <p className="text-[9px] text-blue-400">{cmpResult.ci_2_low.toFixed(3)}–{cmpResult.ci_2_high.toFixed(3)}</p>
                        </div>
                        <div className="bg-rose-50 rounded p-1.5">
                          <p className="text-[9px] text-rose-400 uppercase tracking-wide font-semibold">New AUC</p>
                          <p className="text-sm font-bold font-mono text-rose-700">{cmpResult.auc_1.toFixed(3)}</p>
                          <p className="text-[9px] text-rose-400">{cmpResult.ci_1_low.toFixed(3)}–{cmpResult.ci_1_high.toFixed(3)}</p>
                        </div>
                        <div className={`rounded p-1.5 ${cmpResult.significant ? "bg-green-50" : "bg-gray-50"}`}>
                          <p className="text-[9px] text-gray-400 uppercase tracking-wide font-semibold">ΔAUC</p>
                          <p className={`text-sm font-bold font-mono ${cmpResult.significant ? "text-green-700" : "text-gray-600"}`}>
                            {cmpResult.difference > 0 ? "+" : ""}{cmpResult.difference.toFixed(3)}
                          </p>
                          <p className="text-[9px] text-gray-400">
                            {cmpResult.ci_diff_low.toFixed(3)} to {cmpResult.ci_diff_high.toFixed(3)}
                          </p>
                        </div>
                      </div>

                      {([
                        ["Z statistic", cmpResult.z.toFixed(3), false],
                        ["DeLong p-value", fmtP(cmpResult.p), true],
                        ["n (paired)", cmpResult.n, false],
                      ] as const).map(([k, v, isP]) => (
                        <div key={k} className="flex justify-between border-b border-gray-100 py-0.5 text-xs">
                          <span className="text-gray-400">{isP ? <>DeLong <i>p</i>-value</> : k}</span>
                          <span className={`font-mono ml-2 shrink-0 ${isP && cmpResult.significant ? "text-green-700 font-semibold" : "text-gray-700"}`}>{v}</span>
                        </div>
                      ))}

                      <div className="rounded bg-amber-50 border border-amber-200 px-2 py-2 text-[10px] text-amber-800 leading-relaxed">
                        <p className="font-semibold mb-0.5">Publication format (Q1/Q2):</p>
                        <p className="italic">{cmpResult.interpretation}</p>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </>
        )}

        {/* ── MULTI MODE controls ── */}
        {mode === "multi" && (
          <>
            <div className="panel space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-gray-700">Predictors</h3>
                <div className="flex gap-1">
                  <button onClick={() => { setMultiCols([...numCols]); setMultiResults([]); }}
                    className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">
                    All
                  </button>
                  <button onClick={() => { setMultiCols([]); setMultiResults([]); }}
                    className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">
                    None
                  </button>
                </div>
              </div>
              <input
                type="text"
                placeholder="Filter variables…"
                value={multiFilter}
                onChange={(e) => setMultiFilter(e.target.value)}
                className="select w-full text-xs py-1"
              />
              <div className="max-h-48 overflow-y-auto space-y-0.5 border border-gray-200 rounded-lg p-1">
                {numCols.filter((c) => c.toLowerCase().includes(multiFilter.toLowerCase())).map((col) => (
                  <label key={col} className="flex items-center gap-2 px-2 py-1 rounded hover:bg-gray-50 cursor-pointer">
                    <input type="checkbox" className="accent-indigo-500"
                      checked={multiCols.includes(col)}
                      onChange={() => toggleMultiCol(col)} />
                    <span className="text-xs text-gray-700 truncate">{col}</span>
                  </label>
                ))}
              </div>
              <p className="text-[10px] text-gray-400">{multiCols.length} selected</p>

              <button className="btn-primary w-full" onClick={runMulti}
                disabled={multiLoading || multiCols.length < 1 || !outcomeCol}>
                {multiLoading ? "Computing…" : `Run ${multiCols.length} ROC${multiCols.length !== 1 ? "s" : ""}`}
              </button>
              {multiError && <p className="text-red-500 text-xs">{multiError}</p>}
            </div>

            {/* ── Combined Model panel ── */}
            <div className="panel space-y-2">
              <button className="flex items-center w-full gap-2" onClick={() => setShowCombined((v) => !v)}>
                <input type="checkbox" checked={showCombined}
                  onChange={(e) => setShowCombined(e.target.checked)}
                  className="accent-indigo-500" onClick={(e) => e.stopPropagation()} />
                <span className="text-sm font-semibold text-gray-700">Combined Model</span>
                <span className="ml-auto text-gray-400 text-xs">{showCombined ? "▲" : "▼"}</span>
              </button>

              {showCombined && (
                <>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    Fits logistic regression on selected variables using cross-validated predictions (no overfitting bias) and plots the combined model ROC.
                  </p>

                  {/* Model name */}
                  <input
                    type="text"
                    placeholder="Combined Model"
                    value={combinedName}
                    onChange={(e) => setCombinedName(e.target.value)}
                    className="select w-full text-xs"
                  />

                  {/* Predictor checkboxes */}
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-400">Variables</span>
                      <div className="flex gap-1">
                        <button onClick={() => setCombinedCols([...allCols.filter((c) => c !== outcomeCol)])}
                          className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">All</button>
                        <button onClick={() => setCombinedCols([])}
                          className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">None</button>
                      </div>
                    </div>
                    <input
                      type="text"
                      placeholder="Filter variables…"
                      value={combinedFilter}
                      onChange={(e) => setCombinedFilter(e.target.value)}
                      className="select w-full text-xs py-1 mb-1"
                    />
                    <div className="max-h-36 overflow-y-auto space-y-0.5 border border-gray-200 rounded-lg p-1">
                      {allCols.filter((c) => c !== outcomeCol && c.toLowerCase().includes(combinedFilter.toLowerCase())).map((col) => (
                        <label key={col} className="flex items-center gap-2 px-2 py-0.5 rounded hover:bg-gray-50 cursor-pointer">
                          <input type="checkbox" className="accent-red-500"
                            checked={combinedCols.includes(col)}
                            onChange={() => toggleCombinedCol(col)} />
                          <span className="text-xs text-gray-700 truncate">{col}</span>
                        </label>
                      ))}
                    </div>
                    <p className="text-[10px] text-gray-400 mt-1">{combinedCols.length} predictor(s) selected</p>
                  </div>

                  <button className="btn-primary w-full" onClick={runCombined}
                    disabled={combinedLoading || combinedCols.length < 1 || !outcomeCol}>
                    {combinedLoading ? "Fitting model…" : "Run Combined Model"}
                  </button>
                  {combinedError && <p className="text-red-500 text-xs">{combinedError}</p>}

                  {combinedResult && !combinedResult.error && (
                    <div className="flex items-center justify-between border-t border-gray-100 pt-2">
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-0.5 rounded" style={{ background: combinedStyle.color, height: 3 }} />
                        <span className="text-xs text-gray-600 font-medium">{combinedResult.col}</span>
                      </div>
                      <span className={`text-sm font-bold font-mono ${aucColor(combinedResult.auc)}`}>
                        {combinedResult.auc}
                      </span>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* AUC Summary + pairwise DeLong matrix moved to the right
                results column (see `mode === "multi"` block below) so the
                multi-curve view gets the same 3-column controls / chart /
                results layout as single mode. */}
          </>
        )}
      </div>

      {/* ── Plot area (centre column) ───────────────────────────────────── */}
      <div className="flex-1 flex flex-col gap-3 min-h-0 overflow-y-auto">

        {/* ROC Guidance — compact collapsible strip (collapsed by default to
            keep the chart above the fold; expands to the 3-card detail). */}
        {(() => {
          const g = mode === "single" ? ROC_GUIDANCE.single : ROC_GUIDANCE.multi;
          return (
            <details className="panel bg-indigo-50 border-indigo-200 py-1.5 px-3 group">
              <summary className="flex items-center gap-2 cursor-pointer list-none text-[11px] text-indigo-800">
                <span className="font-semibold">💡 Guidance</span>
                <span className="text-indigo-400 truncate flex-1">{g.use}</span>
                <span className="text-indigo-400 group-open:rotate-90 transition-transform">▸</span>
              </summary>
              <div className="grid grid-cols-3 gap-3 mt-2">
                {[
                  { icon: "🎯", title: "Use when", text: g.use },
                  { icon: "✅", title: "Check", text: g.check },
                  { icon: "📖", title: "Interpret", text: g.interpret },
                ].map(({ icon, title, text }) => (
                  <div key={title} className="rounded-lg bg-white/60 border border-indigo-100 p-2">
                    <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider mb-0.5">{icon} {title}</p>
                    <p className="text-[11px] text-indigo-800 leading-snug">{text}</p>
                  </div>
                ))}
              </div>
            </details>
          );
        })()}

        {/* Style controls bar */}
        {mode === "single" && result && (
          <div className="panel flex flex-wrap items-center gap-4 py-2">
            <StyleRow
              label="ROC curve"
              color={singleStyle.color} width={singleStyle.width} dash={singleStyle.dash}
              onColor={(v) => setSingleStyle((s) => ({ ...s, color: v }))}
              onWidth={(v) => setSingleStyle((s) => ({ ...s, width: v }))}
              onDash={(v)  => setSingleStyle((s) => ({ ...s, dash:  v }))}
            />
            <div className="w-px h-5 bg-gray-200" />
            <StyleRow
              label="Chance line"
              color={chanceStyle.color} width={chanceStyle.width} dash={chanceStyle.dash}
              onColor={(v) => setChanceStyle((s) => ({ ...s, color: v }))}
              onWidth={(v) => setChanceStyle((s) => ({ ...s, width: v }))}
              onDash={(v)  => setChanceStyle((s) => ({ ...s, dash:  v }))}
            />
          </div>
        )}

        {mode === "multi" && multiResults.length > 0 && (
          <details className="panel py-1.5 px-3 group">
            <summary className="flex items-center gap-2 cursor-pointer list-none text-xs font-semibold text-gray-500">
              <span>🎨 Curve styles</span>
              <span className="flex items-center gap-1 flex-1 min-w-0">
                {multiResults.filter((r) => !r.error).map((r, i) => {
                  const st = multiStyles[multiResults.findIndex((x) => x.col === r.col)] ?? defaultStyle(i);
                  return <span key={r.col} className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: st.color }} title={r.col} />;
                })}
              </span>
              <span className="text-gray-400 group-open:rotate-90 transition-transform">▸</span>
            </summary>
            <div className="space-y-1.5 mt-2">
              {/* Combined model style row — shown first when enabled */}
              {showCombined && combinedResult && !combinedResult.error && (
                <>
                  <StyleRow
                    label={combinedResult.col}
                    color={combinedStyle.color} width={combinedStyle.width} dash={combinedStyle.dash}
                    onColor={(v) => setCombinedStyle((s) => ({ ...s, color: v }))}
                    onWidth={(v) => setCombinedStyle((s) => ({ ...s, width: v }))}
                    onDash={(v)  => setCombinedStyle((s) => ({ ...s, dash:  v }))}
                  />
                  <div className="border-t border-gray-200" />
                </>
              )}
              {multiResults.map((r, i) => {
                const st = multiStyles[i] ?? defaultStyle(i);
                return (
                  <StyleRow
                    key={r.col} label={r.col}
                    color={st.color} width={st.width} dash={st.dash}
                    onColor={(v) => updateMultiStyle(i, { color: v })}
                    onWidth={(v) => updateMultiStyle(i, { width: v })}
                    onDash={(v)  => updateMultiStyle(i, { dash:  v })}
                  />
                );
              })}
              <div className="border-t border-gray-100 pt-1.5">
                <StyleRow
                  label="Reference"
                  color={multiChance.color} width={multiChance.width} dash={multiChance.dash}
                  onColor={(v) => setMultiChance((s) => ({ ...s, color: v }))}
                  onWidth={(v) => setMultiChance((s) => ({ ...s, width: v }))}
                  onDash={(v)  => setMultiChance((s) => ({ ...s, dash:  v }))}
                />
              </div>
            </div>
          </details>
        )}

        {/* Plot */}
        <div className="flex-1 panel min-h-0 flex flex-col gap-4 overflow-y-auto" style={{ minHeight: 380 }}>

          {/* ── Single plot ── */}
          {mode === "single" && result && (() => {
            // Pick the cutoff point to highlight: optimal (Youden J) unless the
            // user clicked "Manual" with a cutoff in scope.
            const cutoffPoint = result.manual?.specificity != null && result.manual.sensitivity != null
              ? { x: 1 - result.manual.specificity, y: result.manual.sensitivity,
                  cutoff: Number(result.manual.cutoff), se: result.manual.sensitivity,
                  sp: result.manual.specificity, label: "Manual cutoff", color: "#f59e0b" }
              : result.optimal?.specificity != null && result.optimal.sensitivity != null
                ? { x: 1 - result.optimal.specificity, y: result.optimal.sensitivity,
                    cutoff: Number(result.optimal.cutoff), se: result.optimal.sensitivity,
                    sp: result.optimal.specificity, label: "Optimal cutoff", color: "#ef4444" }
                : result.sensitivity != null && result.specificity != null
                  ? { x: 1 - result.specificity, y: result.sensitivity,
                      cutoff: Number(result.optimal_cutoff), se: result.sensitivity,
                      sp: result.specificity, label: "Optimal cutoff", color: "#ef4444" }
                  : null;

            // AUC box: line 1 = AUC + 95% CI, line 2 = p.
            const aucCI = result.ci_lower != null && result.ci_upper != null
              ? `AUC = ${result.auc.toFixed(2)} (95% CI ${result.ci_lower.toFixed(2)}–${result.ci_upper.toFixed(2)})`
              : `AUC = ${result.auc.toFixed(2)}`;
            const aucP = result.auc_p != null
              ? `<i>p</i> = ${fmtP(result.auc_p)}`
              : null;
            const aucBoxText = aucP ? `${aucCI}<br>${aucP}` : aucCI;

            const annotations: Record<string, unknown>[] = [
              {
                x: 0.98, y: 0.04, xref: "paper" as const, yref: "paper" as const,
                text: aucBoxText,
                showarrow: false,
                font: { color: "#374151", size: 13 },
                align: "right" as const,
                bgcolor: "rgba(249,250,251,0.92)", bordercolor: "#9ca3af", borderwidth: 1, borderpad: 6,
                xanchor: "right" as const, yanchor: "bottom" as const,
              },
            ];
            if (cutoffPoint) {
              // Arrow from text label to the cutoff dot, like matplotlib annotate().
              annotations.push({
                x: cutoffPoint.x, y: cutoffPoint.y, xref: "x" as const, yref: "y" as const,
                ax: 50, ay: 70, axref: "pixel" as const, ayref: "pixel" as const,
                text: `${cutoffPoint.label} = ${cutoffPoint.cutoff.toFixed(2)}<br>(Se ${(cutoffPoint.se * 100).toFixed(1)}%, Sp ${(cutoffPoint.sp * 100).toFixed(1)}%)`,
                showarrow: true,
                arrowhead: 0, arrowsize: 1, arrowwidth: 1.2, arrowcolor: "#6b7280",
                font: { color: "#374151", size: 11 },
                align: "left" as const,
                bgcolor: "rgba(255,255,255,0)", borderpad: 0,
                xanchor: "left" as const, yanchor: "top" as const,
              });
            }

            return (
            <div className="relative flex-shrink-0" style={{ width: "100%", minHeight: 420 }}>
            <TitledPlot
              plotRefOut={rocSingleRef}
              storageKey={`roc:single:${scoreCol}:${outcomeCol}`}
              data={[
                {
                  type: "scatter", mode: "lines",
                  x: result.curve.map((p) => p.fpr),
                  y: result.curve.map((p) => p.tpr),
                  line: { color: singleStyle.color, width: singleStyle.width, dash: singleStyle.dash, shape: "hv" as const },
                  name: fmtAUC(result.auc, result.ci_lower, result.ci_upper),
                  fill: "tozeroy",
                  fillcolor: `${singleStyle.color}22`,
                  showlegend: false,
                  hovertemplate: "FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
                },
                {
                  type: "scatter", mode: "lines",
                  x: [0, 1], y: [0, 1],
                  line: { color: chanceStyle.color, width: chanceStyle.width, dash: chanceStyle.dash },
                  name: "Reference",
                  showlegend: false,
                  hoverinfo: "skip" as const,
                },
                ...(cutoffPoint ? [{
                  type: "scatter" as const, mode: "markers" as const,
                  x: [cutoffPoint.x], y: [cutoffPoint.y],
                  marker: { color: cutoffPoint.color, size: 12, symbol: "circle" as const,
                            line: { color: "#ffffff", width: 1.5 } },
                  name: `${cutoffPoint.label} = ${cutoffPoint.cutoff.toFixed(2)}`,
                  showlegend: false,
                  hovertemplate: `${cutoffPoint.label} = ${cutoffPoint.cutoff.toFixed(2)}<br>Se: ${(cutoffPoint.se * 100).toFixed(1)}%<br>Sp: ${(cutoffPoint.sp * 100).toFixed(1)}%<extra></extra>`,
                }] : []),
              ]}
              layout={{
                ...PLOT_LAYOUT,
                xaxis: {
                  ...(PLOT_LAYOUT.xaxis as object), showgrid: showGrid,
                  title: { text: "1 − Specificity (FPR)", font: { color: "#374151", size: 12 } },
                  range: [0, 1], zeroline: false,
                },
                yaxis: {
                  ...(PLOT_LAYOUT.yaxis as object), showgrid: showGrid,
                  title: { text: "Sensitivity (TPR)", font: { color: "#374151", size: 12 } },
                  range: [0, 1.05], zeroline: false,
                },
                autosize: true,
                showlegend: false,
                annotations,
              }}
              defaultTitle={`ROC — ${scoreCol} predicting ${outcomeCol}`}
              defaultSubtitle=""
              defaultXAxis="1 − Specificity (FPR)"
              defaultYAxis="Sensitivity (TPR)"
              config={{ responsive: true, displaylogo: false, displayModeBar: false }}
            />
            </div>
            );
          })()}

          {/* ── DeLong comparison plot (publication quality) ── */}
          {mode === "single" && cmpResult && cmpResult.curve_1 && cmpResult.curve_2 && (
            <div className="relative flex-shrink-0" style={{ width: "100%", minHeight: 420 }}>
            <TitledPlot
              plotRefOut={rocCompareRef}
              storageKey={`roc:delong:${cmpResult.score_1}:${cmpResult.score_2}`}
              data={[
                // Baseline model (blue dashed)
                {
                  type: "scatter", mode: "lines",
                  x: cmpResult.curve_2.map((p) => p.fpr),
                  y: cmpResult.curve_2.map((p) => p.tpr),
                  line: { color: "#2563eb", width: 2.5, dash: "dash" },
                  name: `Baseline (${cmpResult.score_2}): AUC = ${cmpResult.auc_2.toFixed(3)} (${cmpResult.ci_2_low.toFixed(3)}–${cmpResult.ci_2_high.toFixed(3)})`,
                },
                // New model (red solid)
                {
                  type: "scatter", mode: "lines",
                  x: cmpResult.curve_1.map((p) => p.fpr),
                  y: cmpResult.curve_1.map((p) => p.tpr),
                  line: { color: "#e11d48", width: 3, dash: "solid" },
                  name: `New model (${cmpResult.score_1}): AUC = ${cmpResult.auc_1.toFixed(3)} (${cmpResult.ci_1_low.toFixed(3)}–${cmpResult.ci_1_high.toFixed(3)})`,
                },
                // Chance line
                {
                  type: "scatter", mode: "lines",
                  x: [0, 1], y: [0, 1],
                  line: { color: "#9ca3af", width: 1.5, dash: "dot" },
                  name: "Reference (chance)",
                  showlegend: true,
                },
              ]}
              layout={{
                ...PLOT_LAYOUT,
                xaxis: { ...(PLOT_LAYOUT.xaxis as object), showgrid: showGrid },
                yaxis: { ...(PLOT_LAYOUT.yaxis as object), showgrid: showGrid },
                autosize: true,
                legend: {
                  font: { color: "#374151", size: 11 },
                  bgcolor: "rgba(249,250,251,0.95)",
                  bordercolor: "#e5e7eb", borderwidth: 1,
                  x: 0.5, y: 0.03,
                  xanchor: "left" as const, yanchor: "bottom" as const,
                },
                annotations: [
                  // DeLong p-value box inside plot (top-left — journal standard)
                  {
                    x: 0.02, y: 0.98,
                    xref: "paper" as const, yref: "paper" as const,
                    xanchor: "left" as const, yanchor: "top" as const,
                    text: [
                      `<b>ΔAUC = ${cmpResult.difference > 0 ? "+" : ""}${cmpResult.difference.toFixed(3)}</b>`,
                      `95% CI: ${cmpResult.ci_diff_low.toFixed(3)} to ${cmpResult.ci_diff_high.toFixed(3)}`,
                      `DeLong <i>p</i> = ${fmtP(cmpResult.p)}`,
                    ].join("<br>"),
                    showarrow: false,
                    font: { color: cmpResult.significant ? "#15803d" : "#6b7280", size: 11 },
                    bgcolor: cmpResult.significant ? "rgba(240,253,244,0.95)" : "rgba(249,250,251,0.95)",
                    bordercolor: cmpResult.significant ? "#86efac" : "#e5e7eb",
                    borderwidth: 1, borderpad: 6,
                    align: "left" as const,
                  },
                ],
              }}
              defaultTitle={`ROC Analysis: Model Comparison — ${cmpResult.score_1} vs. ${cmpResult.score_2}`}
              defaultSubtitle=""
              defaultXAxis="1 − Specificity (FPR)"
              defaultYAxis="Sensitivity (TPR)"
              config={{ responsive: true, displaylogo: false, displayModeBar: false }}
            />
            </div>
          )}

          {/* ── Multi-curve plot ── */}
          {mode === "multi" && (multiResults.length > 0 || (showCombined && combinedResult && !combinedResult.error)) && (
            <div className="relative flex-shrink-0" style={{ width: "100%", minHeight: 420 }}>
            <TitledPlot
              plotRefOut={rocMultiRef}
              storageKey={`roc:multi:${outcomeCol}`}
              data={multiTraces as PlotData[]}
              layout={{
                ...PLOT_LAYOUT,
                xaxis: {
                  ...(PLOT_LAYOUT.xaxis as object), showgrid: showGrid,
                  title: { text: "1 − Specificity (FPR)", font: { color: "#374151", size: 12 } },
                  range: [0, 1], zeroline: false,
                },
                yaxis: {
                  ...(PLOT_LAYOUT.yaxis as object), showgrid: showGrid,
                  title: { text: "Sensitivity (TPR)", font: { color: "#374151", size: 12 } },
                  range: [0, 1.05], zeroline: false,
                },
                autosize: true,
                legend: {
                  font: { color: "#374151", size: 11 },
                  bgcolor: "rgba(249,250,251,0.95)",
                  bordercolor: "#9ca3af", borderwidth: 1,
                  x: 0.98, y: 0.04, xanchor: "right" as const, yanchor: "bottom" as const,
                },
              }}
              defaultTitle={`ROC comparison: ${outcomeCol}`}
              defaultSubtitle=""
              defaultXAxis="1 − Specificity (FPR)"
              defaultYAxis="Sensitivity (TPR)"
              config={{ responsive: true, displaylogo: false, displayModeBar: false }}
            />
            </div>
          )}

          {/* ── Empty state ── */}
          {((mode === "single" && !result) || (mode === "multi" && !multiResults.length && !(showCombined && combinedResult && !combinedResult?.error))) && (
            <div className="h-full flex flex-col items-center justify-center gap-2 text-gray-400">
              <span className="text-3xl">📈</span>
              <span className="text-sm text-center">
                {mode === "single"
                  ? "Select a continuous score and a binary outcome (0/1), then click Run ROC"
                  : "Select predictors and a binary outcome, then click Run ROCs"}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* ── Right results column ───────────────────────────────────────────── */}
      {/* Single-mode results (AUC tile + paragraph + metric table) and the
          DeLong comparison card moved here from the left sidebar so the
          chart no longer dominates the viewport and the result numbers
          have a proper column to live in. Only renders for single mode. */}
      {mode === "single" && (
        <div className="w-[380px] flex-shrink-0 flex flex-col gap-3 overflow-y-auto">
          {result && (
            <div className="panel space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-gray-700">Results</h3>
                {singleExport && (
                  <ResultExporter
                    title={`ROC_${scoreCol}_vs_${outcomeCol}`}
                    headers={singleExport.headers}
                    rows={singleExport.rows}
                  />
                )}
              </div>

              <div className="flex flex-col items-center bg-gray-50 border border-gray-200 rounded-lg py-3">
                <span className="text-xs text-gray-400 mb-0.5 flex items-center">
                  AUC
                  <Tip text="P(score₊ > score₋). 0.5 = chance, 1.0 = perfect. ≥0.9 Excellent · ≥0.8 Good · ≥0.7 Fair." />
                </span>
                <span className={`text-2xl font-bold font-mono ${aucColor(result.auc)}`}>{result.auc}</span>
                <span className={`text-xs mt-0.5 ${aucColor(result.auc)}`}>{aucLabel(result.auc)}</span>
                {result.ci_lower != null && (
                  <span className="text-[10px] text-gray-400 mt-0.5">
                    95% CI {result.ci_lower} – {result.ci_upper}
                  </span>
                )}
                {result.auc_p != null && (
                  <span className="text-[10px] text-gray-500 mt-0.5">
                    H₀: AUC = 0.5{" "}
                    <span className="text-gray-400">|</span>{" "}
                    Z = {result.auc_z}{" "}
                    <span className="text-gray-400">|</span>{" "}
                    <span className={`font-mono ${result.auc_p < 0.05 ? "text-emerald-700 font-semibold" : "text-gray-500"}`}>
                      <i>p</i> = {fmtP(result.auc_p)}
                    </span>
                  </span>
                )}
              </div>

              {result.result_text && (
                <div className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 mt-2">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] font-semibold text-gray-400 uppercase">Results Paragraph</span>
                    <button onClick={() => navigator.clipboard.writeText(result.result_text ?? "")} className="text-[10px] px-2 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 transition-colors">Copy</button>
                  </div>
                  <p className="text-sm text-gray-700 leading-relaxed">{result.result_text}</p>
                </div>
              )}
              {result.n_excluded != null && result.n_excluded > 0 && (
                <InfoBanner>
                  {result.n_excluded} row{result.n_excluded !== 1 ? "s" : ""} excluded due to missing values
                  {result.imputation && result.imputation !== "listwise" ? ` (${result.imputation} imputation applied)` : " (listwise deletion)"}.
                </InfoBanner>
              )}
              {([[<i>n</i>, result.n], ["Positives", result.n_positive], ["Negatives", result.n_negative]] as [ReactNode, ReactNode][]).map(([k, v], i) => (
                <div key={i} className="flex justify-between border-b border-gray-100 py-0.5 text-xs">
                  <span className="text-gray-400">{k}</span>
                  <span className="text-gray-700 font-mono">{v}</span>
                </div>
              ))}

              {result.manual && (
                <div className="flex rounded overflow-hidden border border-gray-300 mt-2">
                  <button
                    className={`flex-1 text-xs py-1 transition-colors ${!useManual ? "bg-indigo-600 text-white" : "text-gray-500 hover:bg-gray-100"}`}
                    onClick={() => setUseManual(false)}>
                    Youden J
                    {!useManual && <Tip text="Maximises Sens + Spec − 1 (Youden J)." />}
                  </button>
                  <button
                    className={`flex-1 text-xs py-1 transition-colors ${useManual ? "bg-indigo-600 text-white" : "text-gray-500 hover:bg-gray-100"}`}
                    onClick={() => setUseManual(true)}>Manual</button>
                </div>
              )}

              {activeMetrics && (
                <MetricsBlock m={activeMetrics}
                  label={useManual && result.manual ? "At manual cutoff" : "At optimal cutoff (Youden J)"} />
              )}
            </div>
          )}

          {/* DeLong comparison moved back to the left sidebar — find it next
              to the single-mode controls. */}
        </div>
      )}

      {/* ── Right results column (multi mode) ───────────────────────────── */}
      {mode === "multi" && multiResults.length > 0 && (
        <div className="w-[380px] flex-shrink-0 flex flex-col gap-3 overflow-y-auto">
          {/* AUC Summary */}
          <div className="panel space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-700">AUC Summary</h3>
              {multiExport && (
                <ResultExporter
                  title={`ROC_multi_${outcomeCol}`}
                  headers={multiExport.headers}
                  rows={multiExport.rows}
                />
              )}
            </div>
            {/* Combined model in summary */}
            {showCombined && combinedResult && !combinedResult.error && (
              <div className="flex items-center justify-between gap-2 border-b-2 border-gray-200 pb-1.5 mb-1">
                <div className="flex items-center gap-1.5 min-w-0">
                  <div className="w-3 h-0.5 rounded flex-shrink-0" style={{ background: combinedStyle.color, height: 3 }} />
                  <span className="text-xs text-gray-700 font-semibold truncate">{combinedResult.col}</span>
                </div>
                <span className={`text-xs font-mono font-bold flex-shrink-0 ${aucColor(combinedResult.auc)}`}>
                  {combinedResult.auc}
                </span>
              </div>
            )}
            {[...multiResults]
              .sort((a, b) => b.auc - a.auc)
              .map((r) => {
                const origIdx = multiResults.findIndex((x) => x.col === r.col);
                const st = multiStyles[origIdx] ?? defaultStyle(origIdx);
                return (
                  <div key={r.col} className="flex items-center justify-between gap-2 border-b border-gray-100 pb-1">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: st.color }} />
                      <span className="text-xs text-gray-600 truncate">{r.col}</span>
                    </div>
                    {r.error ? (
                      <span className="text-red-500 text-xs truncate max-w-[160px]" title={r.error}>{r.error}</span>
                    ) : (
                      <span className={`text-xs font-mono font-semibold flex-shrink-0 ${aucColor(r.auc)}`}>
                        {r.auc}
                      </span>
                    )}
                  </div>
                );
              })}
          </div>

          {/* Pairwise DeLong matrix */}
          {multiDelongErr && (
            <div className="rounded border border-red-200 bg-red-50 px-2 py-1 text-[11px] text-red-600">
              DeLong: {multiDelongErr}
            </div>
          )}
          {multiDelong?.pairs && multiDelong.pairs.length > 0 && (
            <div className="panel space-y-1.5">
              <div className="flex items-center justify-between">
                <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
                  Pairwise DeLong
                  <Tip wide text="K-curve DeLong (1988) pairwise ΔAUC test on the same paired sample. Each row reports ΔAUC = AUC(A) − AUC(B), the DeLong 95% CI, the z-statistic, and both the raw p-value and the p-value adjusted for K(K−1)/2 pairwise comparisons." />
                </p>
                <select
                  value={multiPAdjust}
                  onChange={(e) => setMultiPAdjust(e.target.value as "holm" | "bonferroni" | "none")}
                  className="text-[10px] border border-gray-300 rounded px-1.5 py-0.5 bg-white">
                  <option value="holm">Holm</option>
                  <option value="bonferroni">Bonferroni</option>
                  <option value="none">No adjust</option>
                </select>
              </div>
              <div className="overflow-auto rounded-lg border border-gray-200">
                <table className="w-full text-[11px] border-collapse">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                      <th className="text-left px-1.5 py-1 font-medium">A</th>
                      <th className="text-left px-1.5 py-1 font-medium">B</th>
                      <th className="text-right px-1.5 py-1 font-medium">ΔAUC</th>
                      <th className="text-right px-1.5 py-1 font-medium">95% CI</th>
                      <th className="text-right px-1.5 py-1 font-medium"><i>p</i></th>
                    </tr>
                  </thead>
                  <tbody>
                    {multiDelong.pairs.map((pr: DelongPair, i: number) => (
                      <tr key={i} className={`border-b border-gray-100 ${pr.significant ? "bg-indigo-50/40" : ""}`}>
                        <td className="px-1.5 py-1 font-mono text-gray-700 truncate max-w-[80px]">{pr.a}</td>
                        <td className="px-1.5 py-1 font-mono text-gray-700 truncate max-w-[80px]">{pr.b}</td>
                        <td className={`px-1.5 py-1 font-mono text-right ${pr.significant ? "text-indigo-700 font-semibold" : "text-gray-600"}`}>
                          {pr.delta_auc >= 0 ? "+" : ""}{pr.delta_auc.toFixed(3)}
                        </td>
                        <td className="px-1.5 py-1 font-mono text-right text-gray-500 whitespace-nowrap">
                          [{pr.ci_low.toFixed(2)}, {pr.ci_high.toFixed(2)}]
                        </td>
                        <td className="px-1.5 py-1 text-right">
                          <span className={`inline-block font-mono px-1 py-0.5 rounded text-[10px] ${
                            pr.significant ? "bg-indigo-100 text-indigo-700 font-semibold" : "text-gray-400"
                          }`}
                            title={`raw p = ${pr.p_raw}`}>
                            {fmtP(pr.p_adj)}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[9px] text-gray-400 leading-snug">
                <i>n</i> = {multiDelong.n} · K = {multiDelong.scores?.length ?? 0} curves · {multiDelong.n_pairs} pairs · p-adjust: {multiDelong.p_adjust}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
