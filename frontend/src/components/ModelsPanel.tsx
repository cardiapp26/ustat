import { useState } from "react";
import { useStore, paletteOf } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { runLinear, runLogistic, runFirthLogistic, runKM, runCox, runLogisticTable, runPoisson, runCoxUniMulti, runOrdinal, runMultiOutcomeRegression } from "../api";
import { Tip, InfoBanner } from "./Tip";
import ResultExporter from "./ResultExporter";
import { fmtP, pCellTitle } from "../lib/format";
import { MissingGuard, type ImputationStrategy } from "./MissingGuard";
import { type ColMeta } from "../store";
import { CoefTable, ORTable, ForestPlot, PredictionPanel, CoefDetailPanel, ModelSummaryTable } from "./models/resultViews";
import CoxHRTable from "./models/CoxHRTable";
import { useModelData } from "./models/useModelData";
import type { ModelResult } from "./models/shared";

const _pal = () => paletteOf(useStore.getState().plotTheme);

/** Default figure title when a model's estimates are sent to the Forest
 *  Builder — the penalty and the adjustment are part of what the figure
 *  claims, so they belong in the title rather than only in the methods. */
const MODEL_FOREST_TITLE: Record<string, string> = {
  firth: "Firth penalized logistic regression",
  firth_ortable: "Firth penalized logistic regression",
  logistic: "Logistic regression",
  ortable: "Logistic regression",
  cox: "Cox proportional hazards",
  ordinal: "Ordinal logistic regression",
};

/** Send-to-Forest-Builder control, shared by both forest cards. */
function ForestBuilderButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      title="Add these estimates as rows in the Forest Builder, keeping whatever is already there — so a figure can combine several fits (e.g. a continuous exposure and its dichotomised form, which cannot share one model)."
      className="flex-shrink-0 whitespace-nowrap rounded-lg border border-indigo-200 bg-indigo-50 px-2.5 py-1 text-xs font-medium text-indigo-700 transition-colors hover:bg-indigo-100"
    >
      → Forest Builder
    </button>
  );
}

// ── Model guidance ──────────────────────────────────────────────────────────
const MODEL_GUIDANCE: Record<string, { use: string; check: string; interpret: string }> = {
  linear: {
    use: "Predict a continuous outcome from one or more predictors. Best for dose-response, biomarker prediction, and adjusted mean comparisons.",
    check: "Residuals vs Fitted should show no pattern (linearity). Q-Q plot should be roughly diagonal (normality). Scale-Location should be flat (homoscedasticity). Use Robust SE if heteroscedastic.",
    interpret: "Each coefficient = change in outcome per 1-unit increase in predictor, holding others constant. R\u00B2 = proportion of variance explained. Check p-values and 95% CIs.",
  },
  logistic: {
    use: "Model a binary outcome (0/1) — e.g. death, readmission, disease presence. Returns Odds Ratios (OR) with 95% CI.",
    check: "Outcome must be binary 0/1. Check for multicollinearity (VIF > 5). Sample size rule of thumb: \u2265 10 events per predictor variable (EPV).",
    interpret: "OR > 1 = higher odds of outcome. OR < 1 = protective. OR = 1 = no effect. Report: OR (95% CI), p-value. Pseudo-R\u00B2 is NOT comparable to linear R\u00B2.",
  },
  ortable: {
    use: "Publication-standard univariate + multivariate OR table. Shows each predictor's effect both alone and adjusted for all others.",
    check: "Same as logistic. The forest plot visually compares unadjusted vs adjusted ORs — large shifts suggest confounding.",
    interpret: "Univariate OR = crude effect. Multivariate OR = adjusted effect. If they differ substantially, the variable is confounded by others in the model.",
  },
  poisson: {
    use: "Model count outcomes (0, 1, 2, 3...) — e.g. number of events, hospital visits, complications. Returns Incidence Rate Ratios (IRR).",
    check: "Outcome must be non-negative integers. Check for overdispersion: if variance >> mean, use Negative Binomial instead. Robust SE helps with mild overdispersion.",
    interpret: "IRR > 1 = higher rate. IRR = 1.5 means 50% more events. Report: IRR (95% CI), p-value.",
  },
  km: {
    use: "Visualise time-to-event data. The survival curve shows the probability of surviving beyond each time point. Log-rank test compares curves between groups.",
    check: "Event column must be binary 0/1 (1 = event occurred). Duration must be positive. Censoring is assumed non-informative.",
    interpret: "Curves that separate early = strong effect. Log-rank p < 0.05 = significant difference between groups. Median survival = time at which 50% have had the event.",
  },
  cox: {
    use: "Regression for time-to-event data with multiple predictors. Returns Hazard Ratios (HR) — the multiplicative effect on the event rate.",
    check: "Proportional hazards assumption: HR should be constant over time (check Schoenfeld residuals). Event column must be binary 0/1.",
    interpret: "HR > 1 = higher hazard (worse prognosis). HR < 1 = protective. HR = 1 = no effect. Report: HR (95% CI), p-value.",
  },
  ordinal: {
    use: "Ordinal outcome with ≥3 ordered categories (e.g. NYHA I–IV, Killip, none/mild/severe). Proportional-odds model — one OR per predictor shared across the cumulative thresholds.",
    check: "Outcome must be an ordered categorical (mark it 'Ordered Categorical' in the Data tab). Proportional-odds assumption: each predictor's effect is constant across the category cut-points.",
    interpret: "OR > 1 = higher odds of being in a HIGHER category per unit increase. One OR per predictor (not one per category). Report OR (95% CI), p.",
  },
  hrtable: {
    use: "Publication HR table (Table 3): each predictor's univariable HR, its parsimonious-model HR (a subset you tick), and its fully-adjusted HR (all predictors together) side by side.",
    check: "Event column must be binary 0/1, duration positive. Tick which predictors enter the parsimonious column. Categorical predictors expand to one row per level vs the reference.",
    interpret: "Univariable = crude effect. Parsimonious = adjusted for the chosen subset. Fully adjusted = adjusted for everything. A blank (—) cell means the predictor was not in that model.",
  },
  multi_outcome: {
    use: "Regress multiple continuous outcomes together on the same predictors/covariates. Produces one consolidated coefficient table (rows = predictors in model order, columns = outcomes).",
    check: "All outcomes must be continuous. Predictors and covariates are mutually exclusive with outcomes. Use listwise or imputation as needed.",
    interpret: "B = unstandardized coefficient. SE = standard error. β = standardized (when toggle on). 95% CI and p per outcome. Bottom model-fit per outcome.",
  },
  rcs: {
    use: "Model non-linear (U/J-shaped) dose-response relationships using Restricted Cubic Splines. Essential for continuous biomarkers where the effect is not a straight line.",
    check: "Predictor must have enough unique values (\u2265 knots + 2). Logistic RCS requires binary 0/1 outcome. 4 knots is the standard default.",
    interpret: "The dose-response curve shows how OR (or outcome) changes across the predictor range. Reference value = OR 1.0. Non-linearity p-value tests whether the curve is significantly non-linear.",
  },
};


// ── Sparkline mini distribution bar ──────────────────────────────────────────
function SparklineMini({ data, type }: { data: number[]; type: string }) {
  const W = 44, H = 14;
  if (!data || data.length === 0) return null;
  const max = Math.max(...data);
  if (max === 0) return null;
  if (type === "numeric") {
    const bw = W / data.length;
    return (
      <svg width={W} height={H} style={{ display: "block", flexShrink: 0 }}>
        {data.map((v, i) => {
          const bh = Math.max(1, (v / max) * H);
          return <rect key={i} x={i * bw} y={H - bh} width={Math.max(bw - 0.5, 0.5)} height={bh} fill="#ef4444" opacity={0.65} />;
        })}
      </svg>
    );
  }
  // categorical → stacked horizontal proportion bars
  const total = data.reduce((a, b) => a + b, 0);
  const CATS  = _pal();
  // Cumulative left offset per segment, precomputed immutably (no reassigned
  // closure variable inside the render map) — react-hooks/immutability.
  const offsets = data.reduce<number[]>(
    (acc, _v, i) => [...acc, i === 0 ? 0 : acc[i - 1] + (data[i - 1] / total) * W],
    [],
  );
  return (
    <svg width={W} height={H} style={{ display: "block", flexShrink: 0 }}>
      {data.map((v, i) => {
        const w = (v / total) * W;
        return <rect key={i} x={offsets[i]} y={0} width={Math.max(w, 0.5)} height={H} fill={CATS[i % CATS.length]} />;
      })}
    </svg>
  );
}


export default function ModelsPanel() {
  const session  = useStore((s) => s.session);

  const { numCols: numColsRaw, allCols: allColsRaw, binaryCols: binaryColsRaw, missingCounts, sparklines } = useModelData(session);

  // Hide "exclude from analysis" columns from every variable picker in this panel.
  // The list comes from useModelData as plain names, so we filter against the set
  // of column names flagged analysis_excluded in the session metadata.
  // Computed before the early-return so it is null-safe and usable as hook defaults.
  const excludedSet = new Set((session?.columns ?? []).filter((c) => c.analysis_excluded).map((c) => c.name));
  const numCols    = numColsRaw.filter((c) => !excludedSet.has(c));
  const allCols    = allColsRaw.filter((c) => !excludedSet.has(c));
  const binaryCols = binaryColsRaw.filter((c) => !excludedSet.has(c));

  const [model, setModel] = usePersistedPanelState<string>("models", "model", "linear");
  const [outcome, setOutcome] = usePersistedPanelState<string>("models", "outcome", numCols[0] ?? "");
  const [predictors, setPredictors] = usePersistedPanelState<string[]>("models", "predictors", []);
  // HR Table (Cox uni/parsimonious/full): subset of predictors that enter the
  // parsimonious (middle) model column.
  const [parsimonious, setParsimonious] = usePersistedPanelState<string[]>("models", "parsimonious", []);
  // HR Table: per-predictor reference level (value code) for categorical
  // predictors. Default (omitted) = lowest code. e.g. LDL-C ref = ">130".
  const [references, setReferences] = usePersistedPanelState<Record<string, string>>("models", "references", {});
  // Pairwise interaction terms applied to linear / logistic / Cox / poisson.
  // Stored as [colA, colB]; rendered as a small picker below the predictor list.
  const [glmInteractions, setGlmInteractions] = usePersistedPanelState<Array<[string, string]>>("models", "glmInteractions", []);
  const [glmIxA, setGlmIxA] = useState<string>("");
  const [glmIxB, setGlmIxB] = useState<string>("");

  // ── New feature state ─────────────────────────────────────────────────────
  const [selectedCoefIdx, setSelectedCoefIdx] = useState<number | null>(null);
  const [nullHyp,   setNullHyp]   = useState("eq");    // eq | leq | geq
  const [robustSE,  setRobustSE]  = useState(false);
  const [scaleFactors, setScaleFactors] = useState<Record<string, string>>({}); // col → divisor string
  const [selection, setSelection] = usePersistedPanelState<string>("models", "selection", "p10"); // multivariate variable selection strategy
  const [durationCol, setDurationCol] = usePersistedPanelState<string>("models", "durationCol", numCols[0] ?? "");
  const [eventCol, setEventCol] = usePersistedPanelState<string>("models", "eventCol", binaryCols[0] ?? numCols[1] ?? "");
  const [groupCol, setGroupCol] = usePersistedPanelState<string>("models", "groupCol", "");
  const [stratifyCol, setStratifyCol] = usePersistedPanelState<string>("models", "stratifyCol", "");
  const cachedModels = useStore((s) => s.panelCache.models) as { result?: ModelResult | null } | undefined;
  const setCacheModels = useStore((s) => s.setPanelCache);
  const setForestHandoff = useStore((s) => s.setForestHandoff);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const setVisualSubTab = useStore((s) => s.setVisualSubTab);
  const [result, _setResultRaw] = useState<ModelResult | null>(cachedModels?.result ?? null);
  const setResult = (r: ModelResult | null) => { _setResultRaw(r); setCacheModels("models", { result: r }); };
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [imputation, setImputation] = useState<ImputationStrategy>("listwise");
  const [predFilter, setPredFilter] = useState("");

  // Multi-outcome regression state (separate to avoid clobbering single-outcome pickers)
  const [moOutcomes, setMoOutcomes] = usePersistedPanelState<string[]>("models", "moOutcomes", []);
  const [moPredictors, setMoPredictors] = usePersistedPanelState<string[]>("models", "moPredictors", []);
  const [moCovariates, setMoCovariates] = usePersistedPanelState<string[]>("models", "moCovariates", []);
  const [moStandardize, setMoStandardize] = useState(true);
  const [moRobust, setMoRobust] = useState(false);

  // All hooks above run unconditionally (react-hooks/rules-of-hooks). The
  // session guard sits here, after every hook is declared.
  if (!session) return null;

  const sid = session.session_id;

  const run = async () => {
    setLoading(true); setError(null); setResult(null); setSelectedCoefIdx(null);
    try {
      let res: { data: ModelResult };
      const sf = buildScaleFactors();
      const interactions = glmInteractions.length > 0 ? glmInteractions : undefined;
      if (model === "linear") res = await runLinear({ session_id: sid, outcome, predictors, imputation, robust_se: robustSE, interactions });
      else if (model === "logistic") res = await runLogistic({ session_id: sid, outcome, predictors, scale_factors: sf, imputation, robust_se: robustSE, interactions });
      else if (model === "firth") res = await runFirthLogistic({ session_id: sid, outcome, predictors, scale_factors: sf, imputation, interactions });
      else if (model === "ortable") res = await runLogisticTable({ session_id: sid, outcome, predictors, scale_factors: sf, selection, imputation });
      else if (model === "firth_ortable") res = await runLogisticTable({ session_id: sid, outcome, predictors, scale_factors: sf, selection, imputation, use_firth: true });
      else if (model === "poisson") res = await runPoisson({ session_id: sid, outcome, predictors, imputation, robust_se: robustSE });
      else if (model === "ordinal") res = await runOrdinal({ session_id: sid, outcome, predictors, imputation });
      else if (model === "multi_outcome") res = await runMultiOutcomeRegression({ session_id: sid, outcomes: moOutcomes, predictors: moPredictors, covariates: moCovariates, standardize: moStandardize, imputation, robust_se: moRobust });
      else if (model === "km") res = await runKM({ session_id: sid, duration_col: durationCol, event_col: eventCol, group_col: groupCol || undefined, stratify_col: stratifyCol || undefined, imputation });
      else if (model === "hrtable") {
        const refs = Object.fromEntries(Object.entries(references).filter(([col]) => predictors.includes(col)));
        res = await runCoxUniMulti({
          session_id: sid, duration_col: durationCol, event_col: eventCol, predictors,
          parsimonious: parsimonious.filter((p) => predictors.includes(p)),
          references: Object.keys(refs).length ? refs : undefined,
        });
      }
      else res = await runCox({ session_id: sid, duration_col: durationCol, event_col: eventCol, predictors, imputation, interactions });
      setResult(res.data);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      const fallback = e instanceof Error ? e.message : String(e);
      setError(typeof detail === "string" ? detail : (fallback ?? "Unknown error"));
    } finally { setLoading(false); }
  };

  const togglePredictor = (col: string) => {
    if (predictors.includes(col)) {
      // Removing — also clear its scale factor
      setScaleFactors((sf) => { const next = { ...sf }; delete next[col]; return next; });
      setPredictors(predictors.filter((c) => c !== col));
    } else {
      setPredictors([...predictors, col]);
    }
  };

  const setScaleFactor = (col: string, val: string) => {
    setScaleFactors((sf) => ({ ...sf, [col]: val }));
  };

  /** Build scale_factors object for API: only include valid factors != 1 */
  const buildScaleFactors = () => {
    const out: Record<string, number> = {};
    for (const [col, val] of Object.entries(scaleFactors)) {
      const n = parseFloat(val);
      if (!isNaN(n) && n > 0 && n !== 1) out[col] = n;
    }
    return Object.keys(out).length > 0 ? out : undefined;
  };

  /**
   * Hand this model's estimates to the Forest Builder, appending to whatever
   * is already on the sheet.
   *
   * A published forest often spans several fits — a continuous exposure and
   * its dichotomised form cannot share a model, so the figure is assembled
   * from two. Appending is what makes that a two-click job instead of typing
   * the estimates back in by hand.
   */
  const sendToForestBuilder = () => {
    if (!result) return;
    const isCox = model === "cox";
    const metric = isCox ? "HR" : "OR";
    const rows: Array<{ label: string; est: number | null; ci_low: number | null; ci_high: number | null; p: number | null; extra: string }> = [];

    if (result.coefficients?.length) {
      for (const c of result.coefficients) {
        if (c.variable === "const" || c.variable === "Intercept") continue;
        const est = isCox ? c.hr : c.odds_ratio;
        const lo = isCox ? c.hr_ci_low : c.or_ci_low;
        const hi = isCox ? c.hr_ci_high : c.or_ci_high;
        if (est == null) continue;
        rows.push({
          label: c.variable, est, ci_low: lo ?? null, ci_high: hi ?? null,
          p: c.p ?? null, extra: "",
        });
      }
    } else if (result.table?.length) {
      // Uni + multi table: the adjusted estimate is the one that goes in a
      // figure; fall back to the unadjusted when a term was not carried into
      // the multivariable model.
      for (const r of result.table) {
        const adjusted = r.multi_or != null;
        const est = adjusted ? r.multi_or : r.uni_or;
        if (est == null) continue;
        rows.push({
          label: r.variable,
          est,
          ci_low: (adjusted ? r.multi_ci_low : r.uni_ci_low) ?? null,
          ci_high: (adjusted ? r.multi_ci_high : r.uni_ci_high) ?? null,
          p: (adjusted ? r.multi_p : r.uni_p) ?? null,
          extra: adjusted ? "" : "unadjusted",
        });
      }
    }
    if (!rows.length) { setError("No estimates to send — fit a model first."); return; }

    setForestHandoff(rows, {
      customTitle: MODEL_FOREST_TITLE[model] ?? "",
      customSubtitle: result.outcome ? `Outcome: ${result.outcome}` : "",
      xLabel: `${isCox ? "Hazard" : "Odds"} ratio (95% CI, log scale)`,
      leftHeader: "Variable",
      rightHeader: `${metric} (95% CI)`,
      returnTab: "models",
      returnLabel: "← Back to the model",
    }, true);
    setVisualSubTab("forest");
    setActiveTab("visual");
  };

  // ── Multi-outcome pickers: enforce mutual exclusion ────────────────────────
  const toggleMoOutcome = (col: string) => {
    const isAdding = !moOutcomes.includes(col);
    const next = isAdding ? [...moOutcomes, col] : moOutcomes.filter((c) => c !== col);
    setMoOutcomes(next);
    if (isAdding) {
      // remove from others
      setMoPredictors((p) => p.filter((c) => c !== col));
      setMoCovariates((c) => c.filter((x) => x !== col));
    }
    setResult(null);
  };
  const toggleMoPredictor = (col: string) => {
    const isAdding = !moPredictors.includes(col);
    const next = isAdding ? [...moPredictors, col] : moPredictors.filter((c) => c !== col);
    setMoPredictors(next);
    if (isAdding) {
      setMoOutcomes((o) => o.filter((c) => c !== col));
      setMoCovariates((c) => c.filter((x) => x !== col));
    }
    setResult(null);
  };
  const toggleMoCovariate = (col: string) => {
    const isAdding = !moCovariates.includes(col);
    const next = isAdding ? [...moCovariates, col] : moCovariates.filter((c) => c !== col);
    setMoCovariates(next);
    if (isAdding) {
      setMoOutcomes((o) => o.filter((c) => c !== col));
      setMoPredictors((p) => p.filter((c) => c !== col));
    }
    setResult(null);
  };
  const clearMoAll = () => {
    setMoOutcomes([]); setMoPredictors([]); setMoCovariates([]); setResult(null);
  };

  const isSurvival  = false;  // KM/Cox moved to Survival Advanced tab
  const isORTable   = model === "ortable" || model === "firth_ortable";
  const isHRTable   = model === "hrtable";
  const isMultiOutcome = model === "multi_outcome";
  // Ordinal outcome → proportional-odds model is the right choice.
  const outcomeIsOrdinal = session.columns.some((c) => c.name === outcome && c.kind === "ordinal");
  const suggestOrdinal = outcomeIsOrdinal && model !== "ordinal" && !isHRTable;
  const hasRobustSE = model === "linear" || model === "logistic" || model === "poisson";

  const toggleParsimonious = (col: string) => {
    setParsimonious(parsimonious.includes(col) ? parsimonious.filter((c) => c !== col) : [...parsimonious, col]);
  };
  const setReference = (col: string, code: string) => {
    setReferences({ ...references, [col]: code });
  };
  // Column lookup for HR Table reference pickers (value labels per level).
  const colByName: Record<string, ColMeta> = Object.fromEntries(session.columns.map((c) => [c.name, c]));

  return (
    <div className="flex gap-4">
      <div className="w-64 flex-shrink-0 space-y-4">
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Model</h3>
          {([
            ["linear",   "Linear Regression",       "Predict a continuous outcome (e.g. blood pressure) from one or more predictors. Output: β coefficients, R², p-values."],
            ["multi_outcome", "Multi-outcome regression", "Simultaneous linear regression of ≥2 continuous outcomes on shared predictors/covariates. Produces consolidated table with B, SE, β, 95% CI, p per outcome (APA-style)."],
            ["logistic", "Logistic Regression",      "Predict a binary outcome (0/1, yes/no) — outputs Odds Ratios showing how each predictor changes the odds of the event."],
            ["ordinal",  "Ordinal Logistic (proportional odds)", "For an ordered categorical outcome with ≥3 levels (NYHA, Killip, none/mild/severe). Proportional-odds model: one OR per predictor shared across the cumulative thresholds. Mark the outcome 'Ordered Categorical' in the Data tab."],
            ["ortable",  "OR Table (Uni + Multi)",   "Run univariate logistic regression for each predictor separately, then all significant ones together in a multivariate model. Standard for clinical papers."],
            ["firth",    "Firth Logistic (penalized)", "Bias-corrected logistic regression (Firth 1993). Use when standard logistic fails or returns infinite ORs from rare events / separation. Same output shape as Logistic but with Jeffreys-prior penalty."],
            ["firth_ortable", "Firth OR Table (Uni + Multi)", "Same univariate + multivariate OR table as above but every cell is fitted via Firth's penalised likelihood — handles rare events and quasi-separation. Use for the LAR / albumin-style protective biomarker workflow when standard logistic returns ∞ or near-zero ORs."],
            ["hrtable",  "HR Table (Uni + Multi)",   "Cox survival version of the OR table (publication Table 3). Each predictor's univariable HR, its parsimonious-model HR (a subset you tick), and its fully-adjusted HR — side by side. Needs a duration + binary event column."],
            ["poisson",  "Poisson Regression",       "Count outcome model (e.g. number of events). Outputs Incidence Rate Ratios (IRR = eβ). Use when the outcome is a non-negative integer (event counts, re-admissions, etc.)."],
          ] as const).map(([v, l, desc]) => (
            <label key={v} className="flex items-start gap-2 cursor-pointer group">
              <input type="radio" name="model" value={v} checked={model === v} onChange={() => { setModel(v); setResult(null); setSelectedCoefIdx(null); }} className="accent-indigo-500 mt-0.5" />
              <span className="leading-tight">
                <span className="text-sm text-gray-700">{l}</span>
                <span className="block text-[11px] text-gray-400 leading-snug mt-0.5">{desc}</span>
              </span>
            </label>
          ))}
          {hasRobustSE && (
            <label className="flex items-center gap-2 cursor-pointer mt-1 pt-2 border-t border-gray-100">
              <input type="checkbox" checked={robustSE} onChange={(e) => setRobustSE(e.target.checked)} className="accent-indigo-500" />
              <span className="text-xs text-gray-600">
                Robust SE (HC3)
                <Tip text="Heteroscedasticity-consistent standard errors (HC3). Use when residuals may have unequal variance — common in clinical data. Does not change point estimates, only SEs and p-values." wide />
              </span>
            </label>
          )}
          {(model === "linear" || model === "logistic" || model === "firth" || model === "ortable" || model === "firth_ortable") && (
            <div className="mt-1 pt-2 border-t border-gray-100 text-[10px] text-indigo-600 bg-indigo-50 border border-indigo-200 rounded px-2 py-1 leading-snug">
              Need a non-linear continuous effect? Use the
              {" "}<span className="font-semibold">Restricted Cubic Spline</span> sub-tab —
              <code>rcs(X, k)</code> with Harrell or custom knots, Wald non-linearity test,
              OR / β / HR curve with 95 % CI, and optional spline × covariate or spline × spline
              interaction (LR test + 2D contour / 3D surface).
            </div>
          )}
        </div>

        <div className="panel space-y-3">
          {isHRTable ? (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Duration column</label>
                <select className="select w-full" value={durationCol} onChange={(e) => setDurationCol(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Event column (0/1)
                  {binaryCols.length === 0 && <span className="ml-1 text-[10px] text-amber-600">⚠ no binary 0/1 column detected</span>}
                </label>
                <select className="select w-full" value={eventCol} onChange={(e) => setEventCol(e.target.value)}>
                  {(binaryCols.length > 0 ? binaryCols : numCols).map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-400">Predictors</label>
                  <button onClick={() => { setPredictors([]); setParsimonious([]); setReferences({}); setResult(null); }} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-red-50 hover:text-red-500 hover:border-red-300 transition-colors">Clear all</button>
                </div>
                <div className="mb-2 text-[10px] text-indigo-600 bg-indigo-50 border border-indigo-200 rounded px-2 py-1 leading-snug">
                  Tick a predictor to include it. The <strong>★</strong> box marks which predictors
                  also enter the <strong>parsimonious</strong> (middle) model column. Univariable and
                  fully-adjusted columns always use every ticked predictor.
                </div>
                <input
                  type="text"
                  placeholder="Filter variables…"
                  value={predFilter}
                  onChange={(e) => setPredFilter(e.target.value)}
                  className="select w-full text-xs mb-1 py-1"
                />
                <div className="max-h-48 overflow-y-auto space-y-1">
                  {allCols
                    .filter((c) => c !== durationCol && c !== eventCol && c.toLowerCase().includes(predFilter.toLowerCase()))
                    .map((c) => {
                      const checked = predictors.includes(c);
                      const spk = sparklines[c];
                      // Categorical predictors expose value labels → offer a
                      // reference-level picker (e.g. LDL-C ref = ">130").
                      const vl = colByName[c]?.value_labels;
                      const levels = vl
                        ? Object.keys(vl).sort((a, b) => (Number(a) || 0) - (Number(b) || 0))
                        : [];
                      const showRef = checked && levels.length >= 2;
                      return (
                        <div key={c} className="space-y-0.5">
                          <div className="flex items-center gap-2 text-sm">
                            <label className="flex items-center gap-2 cursor-pointer flex-1 min-w-0">
                              <input type="checkbox" checked={checked} onChange={() => togglePredictor(c)} className="accent-indigo-500" />
                              <span className="text-gray-700 truncate flex-1">{c}</span>
                              {(missingCounts[c] ?? 0) > 0 && (
                                <span className="text-[9px] font-bold px-1 py-0.5 rounded bg-amber-100 text-amber-600 border border-amber-200 flex-shrink-0"
                                  title={`${missingCounts[c]} missing values`}>
                                  {missingCounts[c]}✕
                                </span>
                              )}
                              {spk && <SparklineMini data={spk.data} type={spk.type} />}
                            </label>
                            <label
                              className={`flex items-center gap-0.5 text-[10px] flex-shrink-0 ${checked ? "cursor-pointer text-amber-600" : "opacity-30"}`}
                              title="Include in the parsimonious model column"
                            >
                              <input
                                type="checkbox"
                                disabled={!checked}
                                checked={checked && parsimonious.includes(c)}
                                onChange={() => toggleParsimonious(c)}
                                className="accent-amber-500"
                              />
                              ★
                            </label>
                          </div>
                          {showRef && (
                            <div className="flex items-center gap-1 ml-5">
                              <span className="text-gray-400 text-[10px] flex-shrink-0">ref:</span>
                              <select
                                value={references[c] ?? levels[0]}
                                onChange={(e) => setReference(c, e.target.value)}
                                className="select text-[11px] py-0.5 flex-1 min-w-0"
                                title="Reference category — every other level is compared against this one"
                              >
                                {levels.map((code) => (
                                  <option key={code} value={code}>{vl?.[code] ?? code}</option>
                                ))}
                              </select>
                            </div>
                          )}
                        </div>
                      );
                    })}
                </div>
              </div>
            </>
          ) : isMultiOutcome ? (
            <>
              {/* Outcomes (≥1 continuous) */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-400">Outcomes (continuous, ≥1)</label>
                  <button onClick={clearMoAll} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-red-50 hover:text-red-500 hover:border-red-300 transition-colors">Clear all</button>
                </div>
                <input
                  type="text"
                  placeholder="Filter variables…"
                  value={predFilter}
                  onChange={(e) => setPredFilter(e.target.value)}
                  className="select w-full text-xs mb-1 py-1"
                />
                <div className="max-h-32 overflow-y-auto space-y-1 mb-2">
                  {numCols.filter((c) => c.toLowerCase().includes(predFilter.toLowerCase())).map((c) => {
                    const checked = moOutcomes.includes(c);
                    const spk = sparklines[c];
                    return (
                      <label key={c} className="flex items-center gap-2 text-sm cursor-pointer">
                        <input type="checkbox" checked={checked} onChange={() => toggleMoOutcome(c)} className="accent-indigo-500" />
                        <span className="text-gray-700 truncate flex-1">{c}</span>
                        {(missingCounts[c] ?? 0) > 0 && (
                          <span className="text-[9px] font-bold px-1 py-0.5 rounded bg-amber-100 text-amber-600 border border-amber-200 flex-shrink-0" title={`${missingCounts[c]} missing values`}>{missingCounts[c]}✕</span>
                        )}
                        {spk && <SparklineMini data={spk.data} type={spk.type} />}
                      </label>
                    );
                  })}
                </div>
              </div>

              {/* Predictors (≥1, mutually exclusive with outcomes) */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-400">Predictors (≥1)</label>
                </div>
                <input
                  type="text"
                  placeholder="Filter variables…"
                  value={predFilter}
                  onChange={(e) => setPredFilter(e.target.value)}
                  className="select w-full text-xs mb-1 py-1"
                />
                <div className="max-h-32 overflow-y-auto space-y-1 mb-2">
                  {allCols.filter((c) => !moOutcomes.includes(c) && c.toLowerCase().includes(predFilter.toLowerCase())).map((c) => {
                    const checked = moPredictors.includes(c);
                    const spk = sparklines[c];
                    return (
                      <label key={c} className="flex items-center gap-2 text-sm cursor-pointer">
                        <input type="checkbox" checked={checked} onChange={() => toggleMoPredictor(c)} className="accent-indigo-500" />
                        <span className="text-gray-700 truncate flex-1">{c}</span>
                        {(missingCounts[c] ?? 0) > 0 && (
                          <span className="text-[9px] font-bold px-1 py-0.5 rounded bg-amber-100 text-amber-600 border border-amber-200 flex-shrink-0" title={`${missingCounts[c]} missing values`}>{missingCounts[c]}✕</span>
                        )}
                        {spk && <SparklineMini data={spk.data} type={spk.type} />}
                      </label>
                    );
                  })}
                </div>
              </div>

              {/* Covariates (optional, mutually exclusive) */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-400">Covariates (optional)</label>
                </div>
                <input
                  type="text"
                  placeholder="Filter variables…"
                  value={predFilter}
                  onChange={(e) => setPredFilter(e.target.value)}
                  className="select w-full text-xs mb-1 py-1"
                />
                <div className="max-h-24 overflow-y-auto space-y-1">
                  {allCols.filter((c) => !moOutcomes.includes(c) && !moPredictors.includes(c) && c.toLowerCase().includes(predFilter.toLowerCase())).map((c) => {
                    const checked = moCovariates.includes(c);
                    const spk = sparklines[c];
                    return (
                      <label key={c} className="flex items-center gap-2 text-sm cursor-pointer">
                        <input type="checkbox" checked={checked} onChange={() => toggleMoCovariate(c)} className="accent-indigo-500" />
                        <span className="text-gray-700 truncate flex-1">{c}</span>
                        {(missingCounts[c] ?? 0) > 0 && (
                          <span className="text-[9px] font-bold px-1 py-0.5 rounded bg-amber-100 text-amber-600 border border-amber-200 flex-shrink-0" title={`${missingCounts[c]} missing values`}>{missingCounts[c]}✕</span>
                        )}
                        {spk && <SparklineMini data={spk.data} type={spk.type} />}
                      </label>
                    );
                  })}
                </div>
              </div>

              {/* Toggles specific to multi-outcome */}
              <div className="pt-2 mt-1 border-t border-gray-100 space-y-1">
                <label className="flex items-center gap-2 cursor-pointer text-xs text-gray-600">
                  <input type="checkbox" checked={moStandardize} onChange={(e) => setMoStandardize(e.target.checked)} className="accent-indigo-500" />
                  Standardized β
                  <Tip text="Report standardized coefficients (β) in addition to unstandardized B. When off, the β columns are omitted from the table." />
                </label>
                <label className="flex items-center gap-2 cursor-pointer text-xs text-gray-600">
                  <input type="checkbox" checked={moRobust} onChange={(e) => setMoRobust(e.target.checked)} className="accent-indigo-500" />
                  Robust SE (HC3)
                  <Tip text="Heteroscedasticity-consistent standard errors (HC3). Does not affect point estimates." />
                </label>
              </div>
            </>
          ) : isSurvival ? (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Duration column</label>
                <select className="select w-full" value={durationCol} onChange={(e) => setDurationCol(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Event column (0/1)
                  {binaryCols.length === 0 && <span className="ml-1 text-[10px] text-amber-600">⚠ no binary 0/1 column detected</span>}
                </label>
                <select className="select w-full" value={eventCol} onChange={(e) => setEventCol(e.target.value)}>
                  {(binaryCols.length > 0 ? binaryCols : numCols).map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              {model === "km" && (
                <>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Group column (optional)</label>
                    <select className="select w-full" value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
                      <option value="">None</option>
                      {allCols.map((c) => <option key={c}>{c}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Stratify by (optional)</label>
                    <select className="select w-full" value={stratifyCol} onChange={(e) => setStratifyCol(e.target.value)}>
                      <option value="">None</option>
                      {allCols.filter((c) => c !== groupCol && c !== durationCol && c !== eventCol).map((c) => <option key={c}>{c}</option>)}
                    </select>
                  </div>
                </>
              )}
              {model === "cox" && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs text-gray-400">Predictors</label>
                    <button onClick={() => { setPredictors([]); setResult(null); }} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-red-50 hover:text-red-500 hover:border-red-300 transition-colors">Clear all</button>
                  </div>
                  <div className="mb-2 text-[10px] text-indigo-600 bg-indigo-50 border border-indigo-200 rounded px-2 py-1 leading-snug">
                    Predictors enter the model linearly. For non-linear continuous effects (e.g.
                    J-shaped LDL ↔ mortality), use the
                    {" "}<span className="font-semibold">Restricted Cubic Spline</span> sub-tab —
                    it fits <code>rcs(X, 4)</code> in the same Cox PH framework with knot placement
                    (Harrell percentiles or custom), Wald non-linearity test, HR curve with 95% CI,
                    optional <code>rcs(X) × rcs(Y)</code> interaction, and 2D/3D HR surfaces.
                  </div>
                  <input
                    type="text"
                    placeholder="Filter variables…"
                    value={predFilter}
                    onChange={(e) => setPredFilter(e.target.value)}
                    className="select w-full text-xs mb-1 py-1"
                  />
                  <div className="max-h-40 overflow-y-auto space-y-1">
                    {allCols
                      .filter((c) => c !== durationCol && c !== eventCol && c.toLowerCase().includes(predFilter.toLowerCase()))
                      .map((c) => {
                        const spk = sparklines[c];
                        return (
                          <label key={c} className="flex items-center gap-2 text-sm cursor-pointer">
                            <input type="checkbox" checked={predictors.includes(c)} onChange={() => togglePredictor(c)} className="accent-indigo-500" />
                            <span className="text-gray-700 truncate flex-1">{c}</span>
                            {(missingCounts[c] ?? 0) > 0 && (
                              <span className="text-[9px] font-bold px-1 py-0.5 rounded bg-amber-100 text-amber-600 border border-amber-200 flex-shrink-0"
                                title={`${missingCounts[c]} missing values`}>
                                {missingCounts[c]}✕
                              </span>
                            )}
                            {spk && <SparklineMini data={spk.data} type={spk.type} />}
                          </label>
                        );
                      })}
                  </div>
                </div>
              )}
            </>
          ) : (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">
                  Outcome{isORTable && <span className="text-gray-400 ml-1">(binary 0/1)</span>}
                  {(missingCounts[outcome] ?? 0) > 0 && (
                    <span className="ml-1 text-[9px] font-bold px-1 py-0.5 rounded bg-amber-100 text-amber-600 border border-amber-200"
                      title={`${missingCounts[outcome]} missing values in outcome`}>
                      {missingCounts[outcome]} missing
                    </span>
                  )}
                </label>
                <select className="select w-full" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
                  {allCols.map((c) => <option key={c}>{c}</option>)}
                </select>
                {suggestOrdinal && (
                  <div className="mt-1 text-[10px] text-teal-700 bg-teal-50 border border-teal-200 rounded px-2 py-1.5 leading-snug flex items-start gap-1.5">
                    <span className="flex-1">
                      Outcome is ordinal — <strong>Ordinal Logistic (proportional odds)</strong> keeps
                      the category order (linear regression assumes equal spacing).
                    </span>
                    <button onClick={() => { setModel("ordinal"); setResult(null); }} className="flex-shrink-0 underline hover:text-teal-900">
                      Use ordinal
                    </button>
                  </div>
                )}
              </div>

              {isORTable && (
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Multivariate Selection</label>
                  <select className="select w-full text-xs" value={selection} onChange={(e) => setSelection(e.target.value)}>
                    <option value="all">All variables (Enter)</option>
                    <option value="p10">Univariate p &lt; 0.10 ★</option>
                    <option value="p05">Univariate p &lt; 0.05</option>
                    <option value="forward">Stepwise Forward</option>
                    <option value="backward">Stepwise Backward</option>
                  </select>
                </div>
              )}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-400">Predictors</label>
                  <button onClick={() => { setPredictors([]); setResult(null); }} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-red-50 hover:text-red-500 hover:border-red-300 transition-colors">Clear all</button>
                </div>
                <input
                  type="text"
                  placeholder="Filter variables…"
                  value={predFilter}
                  onChange={(e) => setPredFilter(e.target.value)}
                  className="select w-full text-xs mb-1 py-1"
                />
                <div className="max-h-48 overflow-y-auto space-y-1">
                  {allCols.filter((c) => c !== outcome && c.toLowerCase().includes(predFilter.toLowerCase())).map((c) => {
                    const checked = predictors.includes(c);
                    const showScale = checked && (model === "logistic" || model === "firth" || model === "ortable" || model === "firth_ortable");
                    const spk = sparklines[c];
                    return (
                      <div key={c} className="space-y-0.5">
                        <label className="flex items-center gap-2 text-sm cursor-pointer">
                          <input type="checkbox" checked={checked} onChange={() => togglePredictor(c)} className="accent-indigo-500" />
                          <span className="text-gray-700 truncate flex-1">{c}</span>
                          {(missingCounts[c] ?? 0) > 0 && (
                            <span className="text-[9px] font-bold px-1 py-0.5 rounded bg-amber-100 text-amber-600 border border-amber-200 flex-shrink-0"
                              title={`${missingCounts[c]} missing values`}>
                              {missingCounts[c]}✕
                            </span>
                          )}
                          {spk && <SparklineMini data={spk.data} type={spk.type} />}
                        </label>
                        {showScale && (
                          <div className="flex items-center gap-1 ml-5 mb-0.5">
                            <span className="text-gray-400 text-xs">÷</span>
                            <input
                              type="number"
                              min="0"
                              step="any"
                              placeholder="1 (no scaling)"
                              value={scaleFactors[c] ?? ""}
                              onChange={(e) => setScaleFactor(c, e.target.value)}
                              className="w-full text-xs bg-white border border-gray-300 rounded px-1.5 py-0.5 text-gray-700 placeholder-gray-300 focus:border-indigo-500 focus:outline-none"
                            />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}
          {/* Pairwise interactions — linear / logistic / cox accept them
              server-side (Poisson + OR-table do not yet). Hidden until at
              least 2 predictors are ticked. */}
          {(model === "linear" || model === "logistic" || model === "firth" || model === "cox") && predictors.length >= 2 && (
            <div className="panel space-y-1.5">
              <p className="text-xs text-gray-500 font-medium flex items-center gap-1">
                Interactions
                <Tip wide text="Add pairwise interaction terms (e.g. AGE × SEX). Numeric × numeric is the element-wise product; numeric × categorical expands across every dummy of the categorical; categorical × categorical multiplies every dummy pair. The output coefficient table reports each interaction as 'A:B' with its own effect estimate and p-value. Use sparingly — each extra term costs degrees of freedom." />
                {glmInteractions.length > 0 && (
                  <span className="ml-1 text-indigo-600 font-semibold">{glmInteractions.length} added</span>
                )}
              </p>
              <div className="flex flex-wrap items-center gap-1.5">
                <select value={glmIxA}
                  onChange={(e) => setGlmIxA(e.target.value)}
                  className="select text-xs py-1">
                  <option value="">First term…</option>
                  {predictors.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
                <span className="text-gray-400 text-xs">×</span>
                <select value={glmIxB}
                  onChange={(e) => setGlmIxB(e.target.value)}
                  className="select text-xs py-1">
                  <option value="">Second term…</option>
                  {predictors.filter((p) => p !== glmIxA).map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
                <button
                  onClick={() => {
                    if (!glmIxA || !glmIxB || glmIxA === glmIxB) return;
                    const exists = glmInteractions.some(
                      ([a, b]) => (a === glmIxA && b === glmIxB) || (a === glmIxB && b === glmIxA),
                    );
                    if (exists) return;
                    setGlmInteractions([...glmInteractions, [glmIxA, glmIxB]]);
                    setGlmIxA(""); setGlmIxB("");
                  }}
                  disabled={!glmIxA || !glmIxB || glmIxA === glmIxB}
                  className="text-xs px-2 py-1 rounded border border-indigo-300 text-indigo-600 hover:bg-indigo-50 disabled:opacity-40 transition-colors"
                >+ Add</button>
              </div>
              {glmInteractions.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-1">
                  {glmInteractions.map(([a, b], i) => (
                    <span key={`${a}:${b}:${i}`}
                      className="inline-flex items-center gap-1 bg-amber-50 border border-amber-200 text-amber-800 text-[11px] rounded px-2 py-0.5">
                      {a} × {b}
                      <button onClick={() => setGlmInteractions(glmInteractions.filter((_, idx) => idx !== i))}
                        className="text-amber-500 hover:text-red-500" title="Remove">×</button>
                    </span>
                  ))}
                  <button onClick={() => setGlmInteractions([])}
                    className="text-[10px] text-gray-400 hover:text-red-500 underline">clear all</button>
                </div>
              )}
            </div>
          )}
          <MissingGuard
            sessionId={sid}
            columns={isHRTable
              ? [durationCol, eventCol, ...predictors]
              : isSurvival
              ? [durationCol, eventCol, ...(model === "cox" ? predictors : [])]
              : isMultiOutcome
              ? [...moOutcomes, ...moPredictors, ...moCovariates]
              : [...predictors, outcome]}
            imputation={imputation}
            onImputation={setImputation}
          >
            <button className="btn-primary w-full" onClick={run} disabled={
              loading ||
              (isMultiOutcome ? (moOutcomes.length < 1 || moPredictors.length < 1) : (!isSurvival && predictors.length === 0) || (isORTable && predictors.length < 1))
            }>
              {loading ? "Fitting…" : "Fit Model"}
            </button>
          </MissingGuard>
          {error && (
            <div className="text-xs text-red-700 bg-red-50 border border-red-200 rounded px-2 py-1" role="alert">
              {error}
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 space-y-4">
        {/* Model guidance */}
        {MODEL_GUIDANCE[model] && (
          <div className="grid grid-cols-3 gap-3">
            {[
              { icon: "🎯", title: "Use when", text: MODEL_GUIDANCE[model].use },
              { icon: "✅", title: "Check", text: MODEL_GUIDANCE[model].check },
              { icon: "📖", title: "Interpret", text: MODEL_GUIDANCE[model].interpret },
            ].map(({ icon, title, text }) => (
              <div key={title} className="panel bg-indigo-50 border-indigo-200 p-3">
                <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider mb-1">{icon} {title}</p>
                <p className="text-xs text-indigo-800 leading-relaxed">{text}</p>
              </div>
            ))}
          </div>
        )}

        {result ? (
          isHRTable && result.rows ? (
            <div className="panel">
              <h4 className="font-semibold text-gray-900 mb-2">
                Univariable, Parsimonious &amp; Fully adjusted Cox HR Table
                <Tip wide text="Publication Table 3. Univariable = each predictor fitted alone. Parsimonious = the subset you ticked (★) fitted together. Fully adjusted = all predictors fitted together. A blank (—) cell means the predictor was not in that model." />
              </h4>
              <CoxHRTable
                rows={result.rows}
                columns={session.columns}
                n={result.n ?? 0}
                nEvents={result.n_events ?? 0}
                nPars={result.n_pars ?? 0}
                nEventsPars={result.n_events_pars ?? 0}
                durationCol={result.duration_col ?? ""}
                eventCol={result.event_col ?? ""}
              />
            </div>
          ) : isMultiOutcome ? (
            <MultiOutcomeResult result={result} standardize={moStandardize} />
          ) : (
          <div className="space-y-4">
            {/* Summary cards */}
            <div className="panel">
              <h4 className="font-semibold text-gray-900 mb-3">{result.model}</h4>
              <div className="grid grid-cols-3 gap-3">
                {[
                  ["N",          result.n,                       "Total number of observations used to fit the model."],
                  result.r_squared != null      && ["R²",        result.r_squared?.toFixed(4),      "Proportion of variance in the outcome explained by the model (0–1). Higher is better, but add predictors only if they genuinely help."],
                  result.adj_r_squared != null  && ["Adj R²",    result.adj_r_squared?.toFixed(4),  "R² adjusted for the number of predictors — penalises adding unhelpful variables. Prefer this over R² when comparing models."],
                  result.pseudo_r2 != null      && ["Pseudo R²", result.pseudo_r2?.toFixed(4),      "McFadden's Pseudo R² for logistic regression. Analogous to R² but not directly comparable. Values 0.2–0.4 indicate good fit."],
                  result.f_stat != null         && ["F-stat",    result.f_stat?.toFixed(3),         "F-test: tests whether the model as a whole explains significantly more variance than no predictors. Large F with small p = model is useful."],
                  result.aic != null            && ["AIC",       result.aic?.toFixed(2),            "Akaike Information Criterion — lower is better. Used to compare models: the model with the lowest AIC balances fit and complexity best."],
                  result.bic != null            && ["BIC",       result.bic?.toFixed(2),            "Bayesian Information Criterion — similar to AIC but applies a larger penalty for extra parameters. Prefer the model with the lower BIC."],
                  result.concordance != null    && ["C-index",   result.concordance?.toFixed(4),    "Concordance index for Cox models — equivalent to AUC. Probability that the model ranks a higher-risk patient above a lower-risk patient."],
                ].filter(Boolean).map((entry) => {
                  const [k, v, tip] = entry as [string, string | number, string?];
                  return (
                  <div key={k} className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                    <p className="text-xs text-gray-400 flex items-center">
                      {k}
                      {tip && <Tip text={tip} wide />}
                    </p>
                    <p className="text-gray-900 font-semibold">{v}</p>
                  </div>
                  );
                })}
              </div>
              {/* Missing-data exclusion notice */}
              {result.n_excluded != null && result.n_excluded > 0 && (
                <div className="mt-3">
                  <InfoBanner>
                    {result.n_excluded} row{result.n_excluded !== 1 ? "s" : ""} were excluded due to missing values
                    {result.imputation && result.imputation !== "listwise" ? ` (${result.imputation} imputation applied to numeric columns)` : " (listwise deletion)"}.
                    Model was fitted on <strong>{result.n}</strong> of <strong>{result.n_total ?? ((result.n ?? 0) + result.n_excluded)}</strong> rows.
                  </InfoBanner>
                </div>
              )}
              {/* Plain-English model fit interpretation */}
              {result.r_squared != null && (
                <div className="mt-3">
                  <InfoBanner>
                    The model explains <strong>{(result.r_squared * 100).toFixed(1)}%</strong> of the variance in <em>{result.outcome ?? "the outcome"}</em>.{" "}
                    {result.r_squared >= 0.7 ? "This is a strong fit." : result.r_squared >= 0.4 ? "This is a moderate fit — other factors likely also play a role." : "This is a weak fit — important predictors may be missing."}
                    {result.adj_r_squared != null && result.adj_r_squared < result.r_squared - 0.05 && " Note: Adjusted R² is notably lower than R², suggesting some predictors may not be contributing meaningfully."}
                  </InfoBanner>
                </div>
              )}
              {result.pseudo_r2 != null && !result.omnibus && (
                <div className="mt-3">
                  <InfoBanner>
                    Pseudo R² = {result.pseudo_r2?.toFixed(3)}.{" "}
                    {result.pseudo_r2 >= 0.4 ? "Excellent model fit." : result.pseudo_r2 >= 0.2 ? "Good model fit." : result.pseudo_r2 >= 0.1 ? "Moderate model fit." : "Weak model fit — consider adding more informative predictors."}
                  </InfoBanner>
                </div>
              )}

              {/* ── Brant test of the proportional-odds assumption (ordinal) ── */}
              {result.brant_proportional_odds?.computed && result.brant_proportional_odds.omnibus && (() => {
                const b = result.brant_proportional_odds!;
                const om = b.omnibus!;
                const violated = om.violation;
                return (
                  <div className="mt-3 panel">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-semibold text-gray-900">Proportional-odds assumption (Brant test)</h4>
                      <Tip wide text="The proportional-odds model assumes each predictor's effect is the same across every cumulative split of the ordinal outcome. Brant's test checks this. A significant result (p < 0.05) means the assumption is violated for at least one predictor — its single shared odds ratio is misleading and a partial-proportional-odds or multinomial model is preferable." />
                    </div>
                    <div className={`rounded-lg px-3 py-2 text-sm ${violated ? "bg-amber-50 text-amber-800 border border-amber-200" : "bg-emerald-50 text-emerald-800 border border-emerald-200"}`}>
                      <strong>{violated ? "⚠ Assumption violated" : "✓ Assumption supported"}</strong>
                      {" — omnibus χ² = "}{om.chi2.toFixed(2)}, df = {om.df}, p = {om.p < 0.001 ? "<0.001" : om.p.toFixed(3)}.
                      {violated && b.by_predictor && (() => {
                        const bad = b.by_predictor.filter((x) => x.violation).map((x) => x.variable);
                        return bad.length ? <> Flagged predictor(s): <strong>{bad.join(", ")}</strong>.</> : null;
                      })()}
                    </div>
                    {b.by_predictor && b.by_predictor.length > 0 && (
                      <table className="w-full text-xs mt-2">
                        <thead>
                          <tr className="text-gray-500 border-b border-gray-200">
                            <th className="text-left py-1 font-medium">Predictor</th>
                            <th className="text-right py-1 font-medium">χ²</th>
                            <th className="text-right py-1 font-medium">df</th>
                            <th className="text-right py-1 font-medium"><i>p</i></th>
                          </tr>
                        </thead>
                        <tbody>
                          {b.by_predictor.map((x) => (
                            <tr key={x.variable} className={`border-b border-gray-100 ${x.violation ? "bg-amber-50/50" : ""}`}>
                              <td className="py-1 font-mono text-gray-700">{x.variable}</td>
                              <td className="py-1 text-right tabular-nums">{x.chi2.toFixed(2)}</td>
                              <td className="py-1 text-right tabular-nums">{x.df}</td>
                              <td className={`py-1 text-right tabular-nums ${x.violation ? "text-amber-700 font-semibold" : "text-gray-600"}`}>
                                {x.p < 0.001 ? "<0.001" : x.p.toFixed(3)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                );
              })()}

              {/* ── SPSS-style Model Summary (logistic only) ── */}
              {result.omnibus && (
                <div className="mt-3">
                  <ModelSummaryTable s={result} />
                </div>
              )}
            </div>

            {/* Coefficients table + detail panel */}
            {result.coefficients && (
              <div className="panel">
                <div className="flex items-center justify-between mb-1">
                  <h4 className="font-semibold text-gray-900">
                    {model === "cox" ? "Coefficients (Hazard Ratios)" : (model === "logistic" || model === "firth" || model === "ordinal") ? "Coefficients (Odds Ratios)" : model === "poisson" ? "Coefficients (Incidence Rate Ratios)" : "Coefficients"}
                    {model === "linear" && <Tip text="Each β coefficient shows how much the outcome changes for a 1-unit increase in that predictor, holding all others constant. Significant predictors (p < 0.05) are highlighted." wide />}
                    {(model === "logistic" || model === "firth") && <Tip text="Odds Ratio (OR) > 1 means higher odds of the outcome; OR < 1 means lower odds. E.g. OR = 2.0 means the outcome is twice as likely per unit increase. 95% CI not crossing 1 = significant." wide />}
                    {model === "cox" && <Tip text="Hazard Ratio (HR) > 1 means a higher rate of the event over time; HR < 1 means a protective effect. E.g. HR = 1.5 means 50% higher event rate per unit increase." wide />}
                    {model === "poisson" && <Tip text="Incidence Rate Ratio (IRR) = eβ. IRR > 1 means higher event rate; IRR < 1 means lower rate. Use for count outcomes (hospital admissions, episodes, etc.)." wide />}
                  </h4>
                  {/* Null hypothesis radio */}
                  <div className="flex items-center gap-3 text-xs text-gray-500">
                    <span className="text-gray-400">H₀:</span>
                    {([["eq", "β = 0"], ["leq", "β ≤ 0"], ["geq", "β ≥ 0"]] as const).map(([v, lbl]) => (
                      <label key={v} className="flex items-center gap-1 cursor-pointer">
                        <input type="radio" name="nullhyp" value={v} checked={nullHyp === v}
                          onChange={() => { setNullHyp(v); setSelectedCoefIdx(null); }}
                          className="accent-indigo-500" />
                        <span>{lbl}</span>
                      </label>
                    ))}
                  </div>
                </div>
                <CoefTable
                  coefs={result.coefficients}
                  hrMode={model === "cox"}
                  allColumns={allCols}
                  selectedIdx={selectedCoefIdx}
                  onSelect={(i) => setSelectedCoefIdx((prev) => prev === i ? null : i)}
                  nullHyp={nullHyp}
                />
                {selectedCoefIdx != null && result.coefficients[selectedCoefIdx] && (
                  <div className="mt-3">
                    <CoefDetailPanel
                      coef={result.coefficients[selectedCoefIdx]}
                      nullHyp={nullHyp}
                      onClose={() => setSelectedCoefIdx(null)}
                    />
                  </div>
                )}
                <p className="text-[10px] text-gray-400 mt-2">Click a row to see the coefficient's sampling distribution.</p>
              </div>
            )}

            {/* Prediction Panel — linear only */}
            {model === "linear" && result.predictor_info && Object.keys(result.predictor_info).length > 0 && (
              <PredictionPanel result={result} />
            )}

            {/* Results text for all regression models */}
            {result.result_text && !result.table && !isMultiOutcome && (
              <div className="panel">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-gray-900">Results Paragraph</h4>
                  <button onClick={() => navigator.clipboard.writeText(result.result_text ?? "")} className="text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 transition-colors">Copy</button>
                </div>
                <p className="text-sm text-gray-700 leading-relaxed bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">{result.result_text}</p>
              </div>
            )}

            {/* OR Table (Uni + Multi) */}
            {result.table && (
              <div className="panel">
                <h4 className="font-semibold text-gray-900 mb-2">
                  Univariate &amp; Multivariate OR Table
                  <Tip text="Univariate: each predictor tested alone against the outcome. Multivariate: all selected predictors tested together, adjusting for each other. Compare both columns — a variable that is significant univariately but not multivariately may be confounded by another predictor." wide />
                </h4>
                <ORTable
                  rows={result.table}
                  outcome={result.outcome ?? ""}
                  selectionMethod={result.selection_method ?? ""}
                  nMulti={result.n_multi}
                  nTotal={result.n_total}
                />
              </div>
            )}

            {/* SPSS-style model stats for OR Table multivariate model */}
            {result.model_stats && (
              <div className="panel">
                <h4 className="font-semibold text-gray-900 mb-2">Multivariate Model Summary</h4>
                <ModelSummaryTable s={result.model_stats} />
              </div>
            )}

            {/* Auto-generated results text */}
            {result.result_text && !isMultiOutcome && (
              <div className="panel">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-gray-900">Results Paragraph</h4>
                  <button onClick={() => navigator.clipboard.writeText(result.result_text ?? "")} className="text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 transition-colors">Copy</button>
                </div>
                <p className="text-sm text-gray-700 leading-relaxed bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">{result.result_text}</p>
              </div>
            )}

            {/* Forest plot — logistic / firth / cox / ordinal (rendered after tables) */}
            {result.coefficients && (model === "logistic" || model === "firth" || model === "cox" || model === "ordinal") &&
              result.coefficients.filter((c) => c.variable !== "const").length > 0 && (
              <div className="panel">
                <div className="mb-2 flex items-start justify-between gap-3">
                  <h4 className="font-semibold text-gray-900">
                    Forest Plot
                    <Tip text="Each row shows one predictor. The square is the point estimate (OR or HR); the horizontal line is the 95% Confidence Interval. If the CI crosses 1 (the vertical dashed line), the effect is not statistically significant. Larger squares = more precise estimate." wide />
                    <span className="ml-2 text-xs font-normal text-gray-400">
                      {model === "cox" ? "HR" : "OR"} with 95% CI — colored = <i>p</i>&lt;0.05, square size = precision
                    </span>
                  </h4>
                  <ForestBuilderButton onClick={sendToForestBuilder} />
                </div>
                <ForestPlot result={result} modelType={model} outcome={result.outcome} />
              </div>
            )}

            {/* Forest plot — OR table (rendered after tables) */}
            {result.table && result.table.length > 0 && (
              <div className="panel">
                <div className="mb-2 flex items-start justify-between gap-3">
                  <h4 className="font-semibold text-gray-900">
                    Forest Plot
                    <span className="ml-2 text-xs font-normal text-gray-400">
                      ● Univariate &nbsp;◆ Multivariate — colored = <i>p</i>&lt;0.05, square size = precision
                    </span>
                  </h4>
                  <ForestBuilderButton onClick={sendToForestBuilder} />
                </div>
                <ForestPlot result={result} modelType={model} outcome={result.outcome} />
              </div>
            )}
          </div>
          )
        ) : (
          <div className="panel h-64 flex items-center justify-center text-gray-400">
            Configure and fit a model
          </div>
        )}
      </div>
    </div>
  );
}

// ── Inline Multi-outcome regression tab component (per task requirements) ────
// Defined in same file; uses checkbox pattern copied from Linear predictors picker.
// Renders APA consolidated table + result_text + model fit + ResultExporter.
type MultiOutcomeCell = {
  B?: number | null;
  SE?: number | null;
  beta?: number | null;
  ci?: unknown;
  p?: number | null;
};

type MultiOutcomeRow = {
  predictor: string;
  by_outcome: Record<string, MultiOutcomeCell>;
};

type MultiOutcomeFit = {
  n?: number;
  k?: number;
  r2?: number | null;
  adj_r2?: number | null;
  f?: number | null;
  f_p?: number | null;
};

type MultiOutcomeResultData = {
  outcomes?: unknown;
  predictors_order?: unknown;
  rows?: unknown;
  model_fit?: Record<string, MultiOutcomeFit>;
  n_by_outcome?: Record<string, number>;
  result_text?: string;
};

function MultiOutcomeResult({
  result,
  standardize,
}: {
  result: MultiOutcomeResultData | null;
  standardize: boolean;
}) {
  if (!result) return null;
  const outcomes: string[] = Array.isArray(result.outcomes) ? result.outcomes : [];
  const predictorsOrder: string[] = Array.isArray(result.predictors_order) ? result.predictors_order : [];
  const rows: MultiOutcomeRow[] = Array.isArray(result.rows) ? result.rows as MultiOutcomeRow[] : [];
  const modelFit: Record<string, MultiOutcomeFit> = result.model_fit || {};

  const showBeta = !!standardize;

  // Build export headers/rows (flat table)
  const exportHeaders: string[] = ["Predictor"];
  outcomes.forEach((oc) => {
    exportHeaders.push(`${oc}_B`, `${oc}_SE`);
    if (showBeta) exportHeaders.push(`${oc}_beta`);
    exportHeaders.push(`${oc}_CI`, `${oc}_p`);
  });
  const exportRows: (string | number | null | undefined)[][] = predictorsOrder.map((pred) => {
    const r = rows.find((x) => x.predictor === pred);
    const bo = (r && r.by_outcome) || {};
    const row: (string | number | null | undefined)[] = [pred];
    outcomes.forEach((oc) => {
      const cell = bo[oc] || {};
      row.push(cell.B ?? null, cell.SE ?? null);
      if (showBeta) row.push(cell.beta == null ? "—" : cell.beta);
      const ci = Array.isArray(cell.ci) && cell.ci.length === 2 ? `[${Number(cell.ci[0]).toFixed(3)}, ${Number(cell.ci[1]).toFixed(3)}]` : "—";
      row.push(ci, cell.p ?? null);
    });
    return row;
  });

  const fmt = (v: unknown, digits = 3) => (v == null || !isFinite(Number(v)) ? "—" : Number(v).toFixed(digits));

  return (
    <div className="panel space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-gray-900">Multi-outcome regression</h4>
        <ResultExporter title="multi_outcome_regression" headers={exportHeaders} rows={exportRows} />
      </div>

      {/* Plain-English result_text */}
      {result.result_text && (
        <div className="panel bg-gray-50 border border-gray-200 p-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] uppercase tracking-wider text-gray-500 font-medium">Results</span>
            <button
              onClick={() => navigator.clipboard.writeText(result.result_text ?? "")}
              className="text-[10px] px-2 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 transition-colors"
            >
              Copy
            </button>
          </div>
          <p className="text-sm text-gray-700 leading-relaxed">{result.result_text}</p>
        </div>
      )}

      {/* Consolidated APA-style table: rows=predictors, cols per outcome */}
      <div className="overflow-x-auto rounded border border-gray-200">
        <table className="min-w-full text-xs">
          <thead>
            <tr className="bg-gray-50">
              <th rowSpan={2} className="sticky left-0 z-10 bg-gray-50 border-b border-r border-gray-200 px-2 py-1 text-left font-semibold text-gray-700">Predictor</th>
              {outcomes.map((oc) => (
                <th
                  key={oc}
                  colSpan={showBeta ? 5 : 4}
                  className="border-b border-gray-200 px-2 py-1 text-center font-semibold text-gray-700"
                >
                  {oc}
                  {result.n_by_outcome && result.n_by_outcome[oc] != null && (
                    <span className="ml-1 text-[10px] font-normal text-gray-400"><i>n</i>={result.n_by_outcome[oc]}</span>
                  )}
                </th>
              ))}
            </tr>
            <tr className="bg-gray-50 text-[10px] text-gray-500">
              {outcomes.flatMap((_, oi) => {
                const parts = ["B", "SE"];
                if (showBeta) parts.push("β");
                parts.push("95% CI", "p");
                return parts.map((h, j) => (
                  <th key={`${oi}-${j}`} className="border-b border-r border-gray-200 px-1 py-0.5 text-center font-medium tabular-nums">
                    {h === "p" ? <i>p</i> : h}
                  </th>
                ));
              })}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {predictorsOrder.length === 0 ? (
              <tr><td colSpan={1 + outcomes.length * (showBeta ? 5 : 4)} className="px-3 py-2 text-gray-400">No rows.</td></tr>
            ) : (
              predictorsOrder.map((pred, i) => {
	                const r = rows.find((x) => x.predictor === pred) || { predictor: pred, by_outcome: {} };
                const isInt = pred === "(Intercept)" || /intercept/i.test(pred);
                return (
                  <tr key={i} className="hover:bg-gray-50/60">
                    <td className="sticky left-0 z-10 bg-white border-r border-gray-200 px-2 py-1 font-mono text-gray-800 whitespace-nowrap">{pred}</td>
                    {outcomes.flatMap((oc, oi) => {
                      const cell = (r.by_outcome || {})[oc] || {};
                      const B = fmt(cell.B);
                      const SE = fmt(cell.SE);
                      const beta = (cell.beta == null || isInt) ? "—" : fmt(cell.beta);
                      const ciStr = Array.isArray(cell.ci) && cell.ci.length === 2
                        ? `[${fmt(cell.ci[0])}, ${fmt(cell.ci[1])}]`
                        : "—";
                      const pval = fmtP(cell.p);
                      const t = pCellTitle(cell.p);
                      const tds = [
                        <td key={`b-${oi}`} className="px-1.5 py-1 text-right tabular-nums border-r border-gray-200">{B}</td>,
                        <td key={`se-${oi}`} className="px-1.5 py-1 text-right tabular-nums text-gray-600 border-r border-gray-200">{SE}</td>,
                      ];
                      if (showBeta) tds.push(<td key={`bta-${oi}`} className="px-1.5 py-1 text-right tabular-nums border-r border-gray-200">{beta}</td>);
                      tds.push(
                        <td key={`ci-${oi}`} className="px-1.5 py-1 text-right tabular-nums text-gray-600 border-r border-gray-200">{ciStr}</td>,
                        <td key={`p-${oi}`} className="px-1.5 py-1 text-right tabular-nums" title={t}>{pval}</td>
                      );
                      return tds;
                    })}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Model fit section (per outcome) */}
      {Object.keys(modelFit).length > 0 && (
        <div>
          <div className="text-xs font-semibold text-gray-600 mb-1">Model fit</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 text-[11px]">
            {outcomes.map((oc) => {
              const f = modelFit[oc];
              if (!f) return null;
              return (
                <div key={oc} className="bg-gray-50 border border-gray-200 rounded px-2 py-1.5">
                  <div className="font-medium">{oc} <span className="text-gray-400">· <i>n</i>={f.n} k={f.k}</span></div>
                  <div className="tabular-nums text-gray-700">
                    R²={fmt(f.r2, 3)} · adj-R²={fmt(f.adj_r2, 3)} · F={fmt(f.f, 2)} (<i>p</i>={fmtP(f.f_p)})
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
