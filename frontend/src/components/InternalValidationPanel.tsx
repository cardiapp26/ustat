import { useState, type ReactNode } from "react";
import { useStore } from "../store";
import {
  runModelValidation,
  runExternalValidationLogistic,
  runNriIdi,
  runExternalValidationSurvival,
} from "../api";
import ResultExporter from "./ResultExporter";
import { fmtP } from "../lib/format";

/**
 * Internal & external validation for a user-fitted prediction model.
 *
 * Internal: refit-based Harrell bootstrap optimism correction + optional k-fold
 * cross-validation for a logistic or Cox model — apparent vs optimism-corrected
 * AUC / C-index and calibration slope, with an overfitting verdict.
 *
 * External (logistic): apply the development model's predicted probabilities to a
 * fresh cohort and report discrimination (AUC + DeLong CI), calibration
 * (slope/intercept, Hosmer-Lemeshow, O/E, Brier), a calibration plot, and the
 * dev→val performance drop.
 *
 * Reclassification (NRI / IDI): compare two models' predicted probabilities on
 * the same cohort — net reclassification improvement at a risk cutoff and
 * integrated discrimination improvement, both with bootstrap CIs.
 *
 * External (survival): apply a survival model's linear predictor to a fresh
 * cohort — validation C-index, calibration slope/intercept, and (when time
 * points are supplied) time-dependent AUC / integrated Brier score.
 */
type TabId = "internal" | "external" | "reclassification" | "external_survival";

const TABS = [
  ["internal", "Internal (bootstrap + CV)"],
  ["external", "External (logistic)"],
  ["reclassification", "Reclassification (NRI / IDI)"],
  ["external_survival", "External (survival)"],
] as const;

const TAB_CONTENT: Record<TabId, () => ReactNode> = {
  internal: InternalTab,
  external: ExternalTab,
  reclassification: ReclassificationTab,
  external_survival: ExternalSurvivalTab,
};

export default function InternalValidationPanel() {
  const [tab, setTab] = useState<TabId>("internal");
  const Active = TAB_CONTENT[tab];

  return (
    <div className="space-y-3">
      <div className="flex gap-1">
        {TABS.map(([id, label]) => (
          <button key={id} onClick={() => setTab(id)}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
              tab === id ? "bg-white text-indigo-700 shadow-sm border border-gray-200" : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
            }`}>
            {label}
          </button>
        ))}
      </div>
      <Active />
    </div>
  );
}

const Tile = ({ label, value, sub, tone }: { label: string; value: string; sub?: ReactNode; tone?: string }) => (
  <div className="rounded-xl border border-gray-200 bg-white p-3">
    <div className="text-[10px] uppercase tracking-wider text-gray-500">{label}</div>
    <div className={`text-xl font-semibold mt-1 ${tone ?? "text-gray-900"}`}>{value}</div>
    {sub && <div className="text-[11px] text-gray-500 mt-0.5">{sub}</div>}
  </div>
);

const fmt = (x: unknown, d = 3) => (typeof x === "number" && isFinite(x) ? x.toFixed(d) : "—");
const signed = (x: unknown, d = 3) => (typeof x === "number" && isFinite(x) ? (x >= 0 ? "+" : "") + x.toFixed(d) : "—");

function errDetail(e: unknown, fallback: string): string {
  if (e && typeof e === "object") {
    const detail = (e as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
    if (typeof detail === "string") return detail;
  }
  return fallback;
}

type MetricKey = "auc" | "c_index";

interface MetricBlock { auc?: number; c_index?: number; calibration_slope?: number; brier?: number }
interface CvBlock { auc?: number; c_index?: number; calibration_slope?: number; brier?: number; folds: number }
interface InternalResult {
  interpretation: string; n: number; n_predictors: number; n_boot: number;
  apparent: MetricBlock; optimism: MetricBlock; corrected: MetricBlock;
  cv: CvBlock | null; overfit_gap: number;
}
interface ExternalResult {
  result_text: string; n: number;
  discrimination: { auc: number; auc_ci: [number, number]; se: number };
  calibration: {
    slope: number; intercept: number; oe_ratio: number | null;
    hosmer_lemeshow: { chi2: number; df: number; p: number } | null;
    brier: number; acceptable: boolean;
  };
  calibration_plot: Array<{ pred: number; obs: number; n: number }>;
  dev_vs_val: { auc_drop?: number; slope_shift?: number } | null;
}
interface NriIdiResult {
  n: number; cutoff_used: number; test: string;
  nri: {
    estimate: number; ci_low: number; ci_high: number;
    contribution_events: number; contribution_non_events: number;
  };
  idi: { estimate: number; ci_low: number; ci_high: number };
  reclassification_counts: {
    up_in_events: number; down_in_events: number;
    up_in_non_events: number; down_in_non_events: number;
  };
}
interface ExternalSurvivalResult {
  /** Present *instead of* every other field when the service bails out early. */
  error?: string;
  test?: string;
  n_validation?: number;
  validation_c_index?: number | null;
  validation_calibration_slope?: number | null;
  validation_calibration_intercept?: number | null;
  note?: string;
  integrated_brier_score?: { ibs?: number | null; n_time_points?: number; error?: string } | null;
  time_dependent_auc?: Array<{ time: number; auc: number; n_at_risk: number; n_events: number }> | null;
  performance_vs_dev?: { c_index_drop?: number; calibration_slope_shift?: number } | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal validation
// ─────────────────────────────────────────────────────────────────────────────

function InternalTab() {
  const session = useStore((s) => s.session);
  const allCols = (session?.columns ?? []).map((c) => c.name);
  const sid = session?.session_id ?? "";

  const [modelType, setModelType] = useState<"logistic" | "cox">("logistic");
  const [outcome, setOutcome] = useState("");
  const [durationCol, setDurationCol] = useState("");
  const [eventCol, setEventCol] = useState("");
  const [preds, setPreds] = useState<string[]>([]);
  const [nBoot, setNBoot] = useState(200);
  const [cv, setCv] = useState(true);
  const [folds, setFolds] = useState(5);
  const [result, setResult] = useState<InternalResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const isCox = modelType === "cox";
  const reserved = isCox ? [durationCol, eventCol] : [outcome];
  const togglePred = (c: string) =>
    setPreds(preds.includes(c) ? preds.filter((x) => x !== c) : [...preds, c]);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const payload: Record<string, unknown> = {
        session_id: sid, model_type: modelType, predictors: preds,
        n_boot: nBoot, cv_folds: cv ? folds : 0,
      };
      if (isCox) { payload.duration_col = durationCol; payload.event_col = eventCol; }
      else { payload.outcome = outcome; }
      const r = await runModelValidation(payload);
      setResult(r.data as InternalResult);
    } catch (e: unknown) {
      setError(errDetail(e, "Internal validation failed."));
    } finally { setLoading(false); }
  };

  const canRun = !!sid && preds.length > 0 && !loading &&
    (isCox ? !!durationCol && !!eventCol : !!outcome);

  const metric: MetricKey = isCox ? "c_index" : "auc";
  const metricLabel = isCox ? "C-index" : "AUC";
  const gap = result ? (result.overfit_gap ?? 0) : 0;
  const heavy = gap > 0.05;

  return (
    <div className="flex gap-4">
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="panel bg-indigo-50 border-indigo-200 space-y-1">
          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">What this does</p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            Apparent performance is <b>optimistic</b> — the model was scored on the same data it was fit on.
            This refits the model on <b>bootstrap resamples</b> (Harrell optimism) and/or <b>k-fold CV</b> to
            estimate honest, <b>optimism-corrected</b> {metricLabel} and calibration slope.
          </p>
          <p className="text-[10px] text-indigo-700 mt-1">A big apparent − corrected gap ⇒ overfitting; a slope « 1 ⇒ shrinkage advisable.</p>
        </div>

        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Internal Validation</h3>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Model type</label>
            <div className="flex gap-1">
              {([["logistic", "Logistic"], ["cox", "Cox PH"]] as const).map(([id, label]) => (
                <button key={id} onClick={() => { setModelType(id); setResult(null); }}
                  className={`flex-1 px-2 py-1 rounded-md text-xs font-medium border ${
                    modelType === id ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-gray-600 border-gray-300 hover:bg-gray-50"
                  }`}>{label}</button>
              ))}
            </div>
          </div>

          {isCox ? (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Duration / time</label>
                <select className="select w-full" value={durationCol} onChange={(e) => setDurationCol(e.target.value)}>
                  <option value="">— select —</option>
                  {allCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Event (1 = event, 0 = censored)</label>
                <select className="select w-full" value={eventCol} onChange={(e) => setEventCol(e.target.value)}>
                  <option value="">— select —</option>
                  {allCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
            </>
          ) : (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Outcome (binary 0/1)</label>
              <select className="select w-full" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
                <option value="">— select —</option>
                {allCols.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}

          <div>
            <label className="text-xs text-gray-400 block mb-1">Predictors</label>
            <div className="text-xs border border-gray-300 rounded-lg p-2 max-h-40 overflow-y-auto space-y-0.5">
              {allCols.filter((c) => !reserved.includes(c)).map((c) => (
                <label key={c} className="flex items-center gap-1.5 cursor-pointer">
                  <input type="checkbox" className="accent-indigo-500" checked={preds.includes(c)}
                    onChange={() => togglePred(c)} />
                  <span className="text-gray-700">{c}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Bootstrap resamples</label>
            <input type="number" min={50} max={500} step={50} className="select w-full"
              value={nBoot} onChange={(e) => setNBoot(Math.max(50, Math.min(500, +e.target.value || 200)))} />
          </div>

          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" className="accent-indigo-500" checked={cv} onChange={(e) => setCv(e.target.checked)} />
            <span className="text-xs text-gray-600">k-fold cross-validation</span>
          </label>
          {cv && (
            <div className="flex gap-1">
              {[5, 10].map((f) => (
                <button key={f} onClick={() => setFolds(f)}
                  className={`flex-1 px-2 py-1 rounded-md text-xs font-medium border ${
                    folds === f ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-gray-600 border-gray-300 hover:bg-gray-50"
                  }`}>{f}-fold</button>
              ))}
            </div>
          )}

          <button className="btn-primary w-full" onClick={run} disabled={!canRun}>
            {loading ? "Validating…" : "Run internal validation"}
          </button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>

      <div className="flex-1 min-w-0 space-y-4">
        {!result ? (
          <div className="panel h-64 flex items-center justify-center text-gray-400 text-sm text-center">
            Pick a model type, outcome/predictors, and run bootstrap + CV to get optimism-corrected performance.
          </div>
        ) : (
          <>
            <div className={`panel border ${heavy ? "border-amber-300 bg-amber-50" : "border-emerald-300 bg-emerald-50"}`}>
              <p className={`text-sm leading-relaxed ${heavy ? "text-amber-900" : "text-emerald-900"}`}>
                {result.interpretation}
              </p>
              <div className="text-[11px] text-gray-500 mt-2">
                <i>n</i> = {result.n} · {result.n_predictors} predictor term(s) · {result.n_boot} usable bootstraps
              </div>
            </div>

            <div>
              <div className="text-xs font-semibold text-gray-600 mb-1">Discrimination ({metricLabel})</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Tile label={`Apparent ${metricLabel}`} value={fmt(result.apparent?.[metric])} />
                <Tile label="Optimism" value={signed(result.optimism?.[metric])}
                  tone={heavy ? "text-amber-600" : "text-gray-900"} />
                <Tile label={`Corrected ${metricLabel}`} value={fmt(result.corrected?.[metric])} tone="text-emerald-600"
                  sub="optimism-corrected" />
                <Tile label={cv && result.cv ? `${result.cv.folds}-fold CV ${metricLabel}` : "CV"}
                  value={result.cv ? fmt(result.cv[metric]) : "off"}
                  tone={result.cv ? "text-indigo-600" : "text-gray-400"} />
              </div>
            </div>

            {!isCox && (
              <div>
                <div className="text-xs font-semibold text-gray-600 mb-1">Calibration & fit</div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <Tile label="Apparent slope" value={fmt(result.apparent?.calibration_slope, 2)} />
                  <Tile label="Corrected slope" value={fmt(result.corrected?.calibration_slope, 2)}
                    tone={(result.corrected?.calibration_slope ?? 1) < 0.9 ? "text-amber-600" : "text-emerald-600"}
                    sub={(result.corrected?.calibration_slope ?? 1) < 0.9 ? "shrinkage advisable" : "≈ 1, good"} />
                  <Tile label="Apparent Brier" value={fmt(result.apparent?.brier)} />
                  {result.cv && <Tile label="CV Brier" value={fmt(result.cv.brier)} tone="text-indigo-600" />}
                </div>
              </div>
            )}

            <div className="panel bg-gray-50 border-gray-200">
              <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">Overfitting gap (apparent − corrected {metricLabel})</div>
              <div className={`text-lg font-semibold ${heavy ? "text-amber-600" : "text-emerald-600"}`}>{signed(gap)}</div>
            </div>

            <ResultExporter title="internal_validation" />
          </>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// External validation (logistic)
// ─────────────────────────────────────────────────────────────────────────────

function CalibrationPlot({ points }: { points: Array<{ pred: number; obs: number; n: number }> }) {
  const S = 240, pad = 30;
  const sc = (v: number) => pad + v * (S - 2 * pad);
  const yc = (v: number) => S - pad - v * (S - 2 * pad);
  return (
    <svg width={S} height={S} className="bg-white rounded-xl border border-gray-200">
      {/* axes */}
      <line x1={pad} y1={S - pad} x2={S - pad} y2={S - pad} stroke="#9ca3af" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={S - pad} stroke="#9ca3af" strokeWidth={1} />
      {/* perfect-calibration diagonal */}
      <line x1={sc(0)} y1={yc(0)} x2={sc(1)} y2={yc(1)} stroke="#d1d5db" strokeDasharray="4 3" strokeWidth={1} />
      {/* points */}
      {points.map((p, i) => (
        <circle key={i} cx={sc(p.pred)} cy={yc(p.obs)} r={4} fill="#4f46e5" fillOpacity={0.75} />
      ))}
      <text x={S / 2} y={S - 6} textAnchor="middle" className="fill-gray-500" fontSize={10}>Predicted probability</text>
      <text x={10} y={S / 2} textAnchor="middle" transform={`rotate(-90 10 ${S / 2})`} className="fill-gray-500" fontSize={10}>Observed</text>
    </svg>
  );
}

function ExternalTab() {
  const session = useStore((s) => s.session);
  const allCols = (session?.columns ?? []).map((c) => c.name);
  const sid = session?.session_id ?? "";

  const [outcome, setOutcome] = useState("");
  const [probColumn, setProbColumn] = useState("");
  const [devAuc, setDevAuc] = useState("");
  const [devSlope, setDevSlope] = useState("");
  const [result, setResult] = useState<ExternalResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const payload: Record<string, unknown> = { session_id: sid, outcome, prob_column: probColumn };
      if (devAuc.trim() !== "") payload.dev_auc = parseFloat(devAuc);
      if (devSlope.trim() !== "") payload.dev_calibration_slope = parseFloat(devSlope);
      const r = await runExternalValidationLogistic(payload);
      setResult(r.data as ExternalResult);
    } catch (e: unknown) {
      setError(errDetail(e, "External validation failed."));
    } finally { setLoading(false); }
  };

  const canRun = !!sid && !!outcome && !!probColumn && probColumn !== outcome && !loading;
  const ok = result?.calibration?.acceptable;
  const hl = result?.calibration?.hosmer_lemeshow;

  return (
    <div className="flex gap-4">
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="panel bg-indigo-50 border-indigo-200 space-y-1">
          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">What this does</p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            Load the <b>validation cohort</b> as the active dataset, with a column of <b>predicted probabilities</b>
            from the development model applied to it. This reports whether <b>discrimination</b> (AUC) and
            <b> calibration</b> (slope/intercept, Hosmer-Lemeshow, O/E) hold up in the new population.
          </p>
          <p className="text-[10px] text-indigo-700 mt-1">Optionally enter the development AUC / slope to see the dev→val drop.</p>
        </div>

        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">External Validation (logistic)</h3>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Outcome (binary 0/1)</label>
            <select className="select w-full" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
              <option value="">— select —</option>
              {allCols.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Predicted probability column (0–1)</label>
            <select className="select w-full" value={probColumn} onChange={(e) => setProbColumn(e.target.value)}>
              <option value="">— select —</option>
              {allCols.filter((c) => c !== outcome).map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-gray-400 block mb-1">Dev AUC (opt.)</label>
              <input type="number" step="0.01" min="0" max="1" className="select w-full"
                value={devAuc} onChange={(e) => setDevAuc(e.target.value)} placeholder="e.g. 0.82" />
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1">Dev slope (opt.)</label>
              <input type="number" step="0.05" className="select w-full"
                value={devSlope} onChange={(e) => setDevSlope(e.target.value)} placeholder="e.g. 1.0" />
            </div>
          </div>

          <button className="btn-primary w-full" onClick={run} disabled={!canRun}>
            {loading ? "Validating…" : "Run external validation"}
          </button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>

      <div className="flex-1 min-w-0 space-y-4">
        {!result ? (
          <div className="panel h-64 flex items-center justify-center text-gray-400 text-sm text-center">
            Select the outcome and predicted-probability column from the validation cohort, then run.
          </div>
        ) : (
          <>
            <div className={`panel border ${ok ? "border-emerald-300 bg-emerald-50" : "border-amber-300 bg-amber-50"}`}>
              <p className={`text-sm leading-relaxed ${ok ? "text-emerald-900" : "text-amber-900"}`}>
                {result.result_text}
              </p>
              <div className="text-[11px] text-gray-500 mt-2"><i>n</i> = {result.n}</div>
            </div>

            <div>
              <div className="text-xs font-semibold text-gray-600 mb-1">Discrimination</div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <Tile label="AUC" value={fmt(result.discrimination?.auc)}
                  sub={result.discrimination?.auc_ci ? `95% CI ${fmt(result.discrimination.auc_ci[0])}–${fmt(result.discrimination.auc_ci[1])}` : undefined} />
                {result.dev_vs_val?.auc_drop != null &&
                  <Tile label="Dev→val AUC drop" value={signed(-result.dev_vs_val.auc_drop)}
                    tone={result.dev_vs_val.auc_drop > 0.05 ? "text-amber-600" : "text-emerald-600"}
                    sub="negative = worse in validation" />}
                {result.dev_vs_val?.slope_shift != null &&
                  <Tile label="Slope shift vs dev" value={signed(result.dev_vs_val.slope_shift, 2)} />}
              </div>
            </div>

            <div>
              <div className="text-xs font-semibold text-gray-600 mb-1">Calibration</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Tile label="Slope" value={fmt(result.calibration?.slope, 2)}
                  tone={ok ? "text-emerald-600" : "text-amber-600"} sub="ideal ≈ 1" />
                <Tile label="Intercept" value={fmt(result.calibration?.intercept, 2)} sub="ideal ≈ 0" />
                <Tile label="O / E ratio" value={fmt(result.calibration?.oe_ratio, 2)} sub="ideal ≈ 1" />
                <Tile label="Brier" value={fmt(result.calibration?.brier)} />
                {hl && <Tile label="Hosmer-Lemeshow"
                  value={`χ²=${fmt(hl.chi2, 1)}`}
                  sub={<><i>p</i> = {fmtP(hl.p)}</>}
                  tone={hl.p < 0.05 ? "text-amber-600" : "text-emerald-600"} />}
              </div>
            </div>

            {Array.isArray(result.calibration_plot) && result.calibration_plot.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-gray-600 mb-1">Calibration plot</div>
                <CalibrationPlot points={result.calibration_plot} />
              </div>
            )}

            <ResultExporter title="external_validation_logistic" />
          </>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Reclassification (NRI / IDI)
// ─────────────────────────────────────────────────────────────────────────────

function ReclassificationTab() {
  const session = useStore((s) => s.session);
  const allCols = (session?.columns ?? []).map((c) => c.name);
  const sid = session?.session_id ?? "";

  const [outcome, setOutcome] = useState("");
  const [probOld, setProbOld] = useState("");
  const [probNew, setProbNew] = useState("");
  const [cutoff, setCutoff] = useState("0.5");
  const [result, setResult] = useState<NriIdiResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const payload: Record<string, unknown> = {
        session_id: sid, outcome, prob_old: probOld, prob_new: probNew,
      };
      if (cutoff.trim() !== "") payload.cutoff = parseFloat(cutoff);
      const r = await runNriIdi(payload);
      setResult(r.data as NriIdiResult);
    } catch (e: unknown) {
      setError(errDetail(e, "NRI / IDI failed."));
    } finally { setLoading(false); }
  };

  const canRun = !!sid && !!outcome && !!probOld && !!probNew &&
    probOld !== probNew && probOld !== outcome && probNew !== outcome && !loading;

  const counts = result?.reclassification_counts;
  const nriPositive = (result?.nri?.estimate ?? 0) > 0;

  return (
    <div className="flex gap-4">
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="panel bg-indigo-50 border-indigo-200 space-y-1">
          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">What this does</p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            Compares an <b>old</b> and a <b>new</b> model on the same cohort using their predicted
            probabilities. <b>NRI</b> counts how many people move across the risk <b>cutoff</b> in the right
            direction (up for events, down for non-events); <b>IDI</b> measures the average gain in
            predicted-probability separation. Both come with bootstrap 95% CIs.
          </p>
          <p className="text-[10px] text-indigo-700 mt-1">Needs ≥ 100 complete observations and a binary outcome.</p>
        </div>

        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Reclassification (NRI / IDI)</h3>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Outcome (binary 0/1)</label>
            <select className="select w-full" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
              <option value="">— select —</option>
              {allCols.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Old model probability (0–1)</label>
            <select className="select w-full" value={probOld} onChange={(e) => setProbOld(e.target.value)}>
              <option value="">— select —</option>
              {allCols.filter((c) => c !== outcome).map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">New model probability (0–1)</label>
            <select className="select w-full" value={probNew} onChange={(e) => setProbNew(e.target.value)}>
              <option value="">— select —</option>
              {allCols.filter((c) => c !== outcome && c !== probOld).map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Risk cutoff</label>
            <input type="number" step="0.05" min="0" max="1" className="select w-full"
              value={cutoff} onChange={(e) => setCutoff(e.target.value)} placeholder="0.5" />
          </div>

          <button className="btn-primary w-full" onClick={run} disabled={!canRun}>
            {loading ? "Computing…" : "Run NRI / IDI"}
          </button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>

      <div className="flex-1 min-w-0 space-y-4">
        {!result ? (
          <div className="panel h-64 flex items-center justify-center text-gray-400 text-sm text-center">
            Select the outcome and the two predicted-probability columns (old vs new model), then run.
          </div>
        ) : (
          <>
            <div className={`panel border ${nriPositive ? "border-emerald-300 bg-emerald-50" : "border-amber-300 bg-amber-50"}`}>
              <p className={`text-sm leading-relaxed ${nriPositive ? "text-emerald-900" : "text-amber-900"}`}>
                {nriPositive
                  ? "The new model reclassifies people in the right direction overall (NRI > 0)."
                  : "The new model does not improve reclassification overall (NRI ≤ 0)."}
              </p>
              <div className="text-[11px] text-gray-500 mt-2">
                <i>n</i> = {result.n} · cutoff = {fmt(result.cutoff_used, 2)} · {result.test}
              </div>
            </div>

            <div>
              <div className="text-xs font-semibold text-gray-600 mb-1">Net Reclassification Improvement</div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <Tile label="NRI" value={signed(result.nri?.estimate)}
                  tone={nriPositive ? "text-emerald-600" : "text-amber-600"}
                  sub={`95% CI ${signed(result.nri?.ci_low)} – ${signed(result.nri?.ci_high)}`} />
                <Tile label="NRI in events" value={signed(result.nri?.contribution_events)}
                  sub="upward movement is good" />
                <Tile label="NRI in non-events" value={signed(result.nri?.contribution_non_events)}
                  sub="downward movement is good" />
              </div>
            </div>

            <div>
              <div className="text-xs font-semibold text-gray-600 mb-1">Integrated Discrimination Improvement</div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <Tile label="IDI" value={signed(result.idi?.estimate, 4)}
                  tone={(result.idi?.estimate ?? 0) > 0 ? "text-emerald-600" : "text-amber-600"}
                  sub={`95% CI ${signed(result.idi?.ci_low, 4)} – ${signed(result.idi?.ci_high, 4)}`} />
              </div>
            </div>

            {counts && (
              <div>
                <div className="text-xs font-semibold text-gray-600 mb-1">Reclassification counts (at cutoff)</div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <Tile label="Up in events" value={String(counts.up_in_events)} tone="text-emerald-600" />
                  <Tile label="Down in events" value={String(counts.down_in_events)} tone="text-amber-600" />
                  <Tile label="Up in non-events" value={String(counts.up_in_non_events)} tone="text-amber-600" />
                  <Tile label="Down in non-events" value={String(counts.down_in_non_events)} tone="text-emerald-600" />
                </div>
              </div>
            )}

            <ResultExporter title="nri_idi" />
          </>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// External validation (survival)
// ─────────────────────────────────────────────────────────────────────────────

/** Parse a comma/space separated list of times into finite positive numbers. */
const parseTimePoints = (raw: string): number[] =>
  raw.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean)
    .map((s) => parseFloat(s)).filter((v) => isFinite(v));

function ExternalSurvivalTab() {
  const session = useStore((s) => s.session);
  const allCols = (session?.columns ?? []).map((c) => c.name);
  const sid = session?.session_id ?? "";

  const [durationCol, setDurationCol] = useState("");
  const [eventCol, setEventCol] = useState("");
  const [lpCol, setLpCol] = useState("");
  const [timePoints, setTimePoints] = useState("");
  const [devC, setDevC] = useState("");
  const [devSlope, setDevSlope] = useState("");
  const [result, setResult] = useState<ExternalSurvivalResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const payload: Record<string, unknown> = {
        session_id: sid, duration_col: durationCol, event_col: eventCol, predicted_lp_col: lpCol,
      };
      const times = parseTimePoints(timePoints);
      if (times.length > 0) payload.time_points = times;
      const dev: Record<string, number> = {};
      if (devC.trim() !== "") dev.c_index = parseFloat(devC);
      if (devSlope.trim() !== "") dev.calibration_slope = parseFloat(devSlope);
      if (Object.keys(dev).length > 0) payload.dev_metrics = dev;
      const r = await runExternalValidationSurvival(payload);
      setResult(r.data as ExternalSurvivalResult);
    } catch (e: unknown) {
      setError(errDetail(e, "External survival validation failed."));
    } finally { setLoading(false); }
  };

  const canRun = !!sid && !!durationCol && !!eventCol && !!lpCol &&
    lpCol !== durationCol && lpCol !== eventCol && durationCol !== eventCol && !loading;

  // The service short-circuits with `{ error }` only — never render tiles then.
  const serviceError = result?.error;
  const tdAuc = result?.time_dependent_auc;
  const drop = result?.performance_vs_dev;

  return (
    <div className="flex gap-4">
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="panel bg-indigo-50 border-indigo-200 space-y-1">
          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">What this does</p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            Load the <b>validation cohort</b> as the active dataset, with the development survival model's
            <b> linear predictor</b> (risk score) stored as a column. This reports the validation
            <b> C-index</b> and <b>calibration</b> slope/intercept, plus time-dependent AUC when you supply
            time points.
          </p>
          <p className="text-[10px] text-indigo-700 mt-1">Optionally enter the development C-index / slope to see the dev→val drop.</p>
        </div>

        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">External Validation (survival)</h3>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Duration / time</label>
            <select className="select w-full" value={durationCol} onChange={(e) => setDurationCol(e.target.value)}>
              <option value="">— select —</option>
              {allCols.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Event (1 = event, 0 = censored)</label>
            <select className="select w-full" value={eventCol} onChange={(e) => setEventCol(e.target.value)}>
              <option value="">— select —</option>
              {allCols.filter((c) => c !== durationCol).map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Predicted linear predictor / risk score</label>
            <select className="select w-full" value={lpCol} onChange={(e) => setLpCol(e.target.value)}>
              <option value="">— select —</option>
              {allCols.filter((c) => c !== durationCol && c !== eventCol).map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Time points (opt., comma-separated)</label>
            <input type="text" className="select w-full"
              value={timePoints} onChange={(e) => setTimePoints(e.target.value)} placeholder="e.g. 12, 24, 60" />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-gray-400 block mb-1">Dev C-index (opt.)</label>
              <input type="number" step="0.01" min="0" max="1" className="select w-full"
                value={devC} onChange={(e) => setDevC(e.target.value)} placeholder="e.g. 0.74" />
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1">Dev slope (opt.)</label>
              <input type="number" step="0.05" className="select w-full"
                value={devSlope} onChange={(e) => setDevSlope(e.target.value)} placeholder="e.g. 1.0" />
            </div>
          </div>

          <button className="btn-primary w-full" onClick={run} disabled={!canRun}>
            {loading ? "Validating…" : "Run survival validation"}
          </button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>

      <div className="flex-1 min-w-0 space-y-4">
        {!result ? (
          <div className="panel h-64 flex items-center justify-center text-gray-400 text-sm text-center">
            Select duration, event, and the linear-predictor column from the validation cohort, then run.
          </div>
        ) : serviceError ? (
          <div className="panel border border-amber-300 bg-amber-50">
            <p className="text-sm text-amber-900 leading-relaxed">{serviceError}</p>
          </div>
        ) : (
          <>
            <div className="panel border border-emerald-300 bg-emerald-50">
              <p className="text-sm text-emerald-900 leading-relaxed">
                {result.note ?? "External validation complete."}
              </p>
              <div className="text-[11px] text-gray-500 mt-2"><i>n</i> = {result.n_validation ?? "—"}</div>
            </div>

            <div>
              <div className="text-xs font-semibold text-gray-600 mb-1">Discrimination & calibration</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Tile label="C-index" value={fmt(result.validation_c_index)} tone="text-emerald-600" />
                <Tile label="Calibration slope" value={fmt(result.validation_calibration_slope, 2)} sub="ideal ≈ 1" />
                <Tile label="Calibration intercept" value={fmt(result.validation_calibration_intercept, 2)} sub="ideal ≈ 0" />
                {result.integrated_brier_score?.ibs != null &&
                  <Tile label="Integrated Brier" value={fmt(result.integrated_brier_score.ibs)} sub="lower is better" />}
              </div>
            </div>

            {drop && (drop.c_index_drop != null || drop.calibration_slope_shift != null) && (
              <div>
                <div className="text-xs font-semibold text-gray-600 mb-1">Dev → validation</div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {drop.c_index_drop != null &&
                    <Tile label="C-index drop" value={signed(-drop.c_index_drop)}
                      tone={drop.c_index_drop > 0.05 ? "text-amber-600" : "text-emerald-600"}
                      sub="negative = worse in validation" />}
                  {drop.calibration_slope_shift != null &&
                    <Tile label="Slope shift vs dev" value={signed(drop.calibration_slope_shift, 2)} />}
                </div>
              </div>
            )}

            {Array.isArray(tdAuc) && tdAuc.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-gray-600 mb-1">Time-dependent AUC</div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {tdAuc.map((p) => (
                    <Tile key={p.time} label={`AUC at t = ${fmt(p.time, 2)}`} value={fmt(p.auc)}
                      sub={`${p.n_events} event(s) · ${p.n_at_risk} at risk`} />
                  ))}
                </div>
              </div>
            )}

            <ResultExporter title="external_validation_survival" />
          </>
        )}
      </div>
    </div>
  );
}
