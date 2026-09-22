import { useState, type ReactNode } from "react";
import { useStore } from "../store";
import { runAddedValue } from "../api";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { useStampedResult } from "../hooks/useStampedResult";
import ResultExporter from "./ResultExporter";
import StaleResultNotice from "./StaleResultNotice";
import StaleGuard from "./StaleGuard";
import { fmtP } from "../lib/format";
import { describeStale } from "../lib/resultStamp";

/**
 * Added Predictive Value — judge whether a new predictor genuinely improves a
 * model, beyond a significant coefficient p-value. Compares a base model with
 * base + new predictor(s) on discrimination (ΔAUC, DeLong), reclassification
 * (NRI, IDI), overall fit (LR test, ΔAIC, pseudo-R²), and calibration.
 */
interface Discrimination {
  auc_base?: number;
  auc_full?: number;
  delta_auc: number;
  delong_p?: number;
  significant?: boolean;
}

interface Reclassification {
  idi: number;
  idi_ci?: [number | string, number | string] | null;
  nri: number;
  nri_ci?: [number | string, number | string] | null;
  nri_events?: number;
  nri_nonevents?: number;
}

interface Fit {
  lr_stat?: number;
  lr_p?: number;
  delta_aic: number;
  nagelkerke_base?: number;
  nagelkerke_full?: number;
}

interface CalibrationSide {
  calibration_slope?: number;
  brier?: number | string;
}

interface Calibration {
  base: CalibrationSide;
  full: CalibrationSide;
  preserved?: boolean;
}

interface AddedValueResult {
  added_value?: boolean;
  result_text?: string;
  n?: number;
  n_excluded?: number;
  prediction_basis?: string;
  discrimination: Discrimination;
  reclassification: Reclassification;
  fit: Fit;
  calibration: Calibration;
}

export default function AddedValuePanel() {
  const session = useStore((s) => s.session);
  const columns = session?.columns ?? [];
  const sid = session?.session_id ?? "";
  const allCols = columns.map((c) => c.name);

  // Persisted alongside the result: it comes back from the panel cache on a
  // tab switch, and against selections reset to empty it would always read as
  // computed under other settings.
  const [outcome, setOutcome] = usePersistedPanelState<string>("added_value", "outcome", "");
  const [basePreds, setBasePreds] = usePersistedPanelState<string[]>("added_value", "basePreds", []);
  const [newPreds, setNewPreds] = usePersistedPanelState<string[]>("added_value", "newPreds", []);
  const [cv, setCv] = usePersistedPanelState<boolean>("added_value", "cv", false);
  // The request body minus the session: exactly what the comparison is fitted from.
  const runParams = {
    outcome, base_predictors: basePreds, new_predictors: newPreds,
    model_type: "logistic", cv_folds: cv ? 5 : 0, bootstrap: 400,
  };
  const {
    result, setResult, stale, staleReasons: staleWhy,
  } = useStampedResult<AddedValueResult>("added_value", runParams);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const toggle = (list: string[], set: (v: string[]) => void, c: string) =>
    set(list.includes(c) ? list.filter((x) => x !== c) : [...list, c]);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await runAddedValue({ session_id: sid, ...runParams });
      setResult(r.data);
    } catch (e: unknown) {
      setError((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Added-value analysis failed.");
    } finally { setLoading(false); }
  };

  const canRun = sid && outcome && basePreds.length > 0 && newPreds.length > 0 && !loading;

  const Tile = ({ label, value, sub, tone }: { label: string; value: string; sub?: ReactNode; tone?: string }) => (
    <div className="rounded-xl border border-gray-200 bg-white p-3">
      <div className="text-[10px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className={`text-xl font-semibold mt-1 ${tone ?? "text-gray-900"}`}>{value}</div>
      {sub && <div className="text-[11px] text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );

  return (
    <div className="flex gap-4">
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="panel bg-indigo-50 border-indigo-200 space-y-1">
          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">What this does</p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            A significant coefficient p-value does <b>not</b> prove a predictor adds value. This compares your
            base model with <b>base + the new predictor</b> and reports whether <b>discrimination</b> (AUC),
            <b> reclassification</b> (NRI/IDI) and <b>calibration</b> actually improve.
          </p>
          <p className="text-[10px] text-indigo-700 mt-1">Tip: enable cross-validation for an honest (optimism-free) ΔAUC.</p>
        </div>

        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Added Predictive Value</h3>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Outcome (binary 0/1)</label>
            <select className="select w-full" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
              <option value="">— select —</option>
              {allCols.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">Base model predictors (established)</label>
            <div className="text-xs border border-gray-300 rounded-lg p-2 max-h-32 overflow-y-auto space-y-0.5">
              {allCols.filter((c) => c !== outcome && !newPreds.includes(c)).map((c) => (
                <label key={c} className="flex items-center gap-1.5 cursor-pointer">
                  <input type="checkbox" className="accent-indigo-500" checked={basePreds.includes(c)}
                    onChange={() => toggle(basePreds, setBasePreds, c)} />
                  <span className="text-gray-700">{c}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">New predictor(s) to evaluate</label>
            <div className="text-xs border border-gray-300 rounded-lg p-2 max-h-32 overflow-y-auto space-y-0.5">
              {allCols.filter((c) => c !== outcome && !basePreds.includes(c)).map((c) => (
                <label key={c} className="flex items-center gap-1.5 cursor-pointer">
                  <input type="checkbox" className="accent-emerald-500" checked={newPreds.includes(c)}
                    onChange={() => toggle(newPreds, setNewPreds, c)} />
                  <span className="text-gray-700">{c}</span>
                </label>
              ))}
            </div>
          </div>

          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" className="accent-indigo-500" checked={cv} onChange={(e) => setCv(e.target.checked)} />
            <span className="text-xs text-gray-600">5-fold cross-validated ΔAUC (honest)</span>
          </label>

          <button className="btn-primary w-full" onClick={run} disabled={!canRun}>
            {loading ? "Running…" : "Assess added value"}
          </button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>

      <div className="flex-1 min-w-0 space-y-4">
        {!result ? (
          <div className="panel h-64 flex items-center justify-center text-gray-400 text-sm text-center">
            Pick an outcome, base predictors, and a new predictor — then assess whether it adds predictive value.
          </div>
        ) : (
          <>
            {stale && (
              <StaleResultNotice
                reasons={staleWhy}
                onRecompute={run}
                busy={loading}
                what="This added-value comparison"
              />
            )}
            <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
            <div className={`panel border ${result.added_value ? "border-emerald-300 bg-emerald-50" : "border-amber-300 bg-amber-50"}`}>
              <p className={`text-sm leading-relaxed ${result.added_value ? "text-emerald-900" : "text-amber-900"}`}>
                {result.result_text}
              </p>
              <div className="text-[11px] text-gray-500 mt-2">
                <i>n</i> = {result.n}{result.n_excluded ? ` (${result.n_excluded} excluded)` : ""} · predictions: {result.prediction_basis}
              </div>
            </div>

            <div>
              <div className="text-xs font-semibold text-gray-600 mb-1">Discrimination</div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <Tile label="AUC base" value={result.discrimination.auc_base?.toFixed(3) ?? "—"} />
                <Tile label="AUC full" value={result.discrimination.auc_full?.toFixed(3) ?? "—"} tone="text-emerald-600" />
                <Tile label="ΔAUC (DeLong)"
                  value={(result.discrimination.delta_auc >= 0 ? "+" : "") + result.discrimination.delta_auc?.toFixed(3)}
                  sub={<><i>p</i> = {fmtP(result.discrimination.delong_p)}</>}
                  tone={result.discrimination.significant ? "text-emerald-600" : "text-gray-900"} />
              </div>
            </div>

            <div>
              <div className="text-xs font-semibold text-gray-600 mb-1">Reclassification</div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <Tile label="IDI" value={(result.reclassification.idi >= 0 ? "+" : "") + result.reclassification.idi?.toFixed(4)}
                  sub={result.reclassification.idi_ci ? `95% CI ${result.reclassification.idi_ci[0]} to ${result.reclassification.idi_ci[1]}` : undefined} />
                <Tile label="Continuous NRI" value={(result.reclassification.nri >= 0 ? "+" : "") + result.reclassification.nri?.toFixed(4)}
                  sub={result.reclassification.nri_ci ? `95% CI ${result.reclassification.nri_ci[0]} to ${result.reclassification.nri_ci[1]}` : undefined} />
                <Tile label="NRI (events / non)"
                  value={`${result.reclassification.nri_events?.toFixed(3)} / ${result.reclassification.nri_nonevents?.toFixed(3)}`} />
              </div>
            </div>

            <div>
              <div className="text-xs font-semibold text-gray-600 mb-1">Overall fit & calibration</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Tile label="LR test"
                  value={`χ²=${result.fit.lr_stat?.toFixed(2)}`}
                  sub={<><i>p</i> = {fmtP(result.fit.lr_p)}</>} />
                <Tile label="ΔAIC" value={(result.fit.delta_aic >= 0 ? "+" : "") + result.fit.delta_aic?.toFixed(1)}
                  tone={result.fit.delta_aic < 0 ? "text-emerald-600" : "text-gray-900"} />
                <Tile label="Nagelkerke R²" value={`${result.fit.nagelkerke_base?.toFixed(3)} → ${result.fit.nagelkerke_full?.toFixed(3)}`} />
                <Tile label="Calibration (full)"
                  value={`slope ${result.calibration.full.calibration_slope?.toFixed(2)}`}
                  sub={`Brier ${result.calibration.base.brier} → ${result.calibration.full.brier}`}
                  tone={result.calibration.preserved ? "text-emerald-600" : "text-amber-600"} />
              </div>
            </div>

            <ResultExporter title="added_predictive_value" />
            </StaleGuard>
          </>
        )}
      </div>
    </div>
  );
}
