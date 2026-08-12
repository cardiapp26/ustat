/**
 * MultiModelPanel — one exposure, progressively adjusted, in one table.
 *
 * The staple of observational epidemiology: the exposure crude, then adjusted
 * for demographics, then for demographics plus clinical covariates. Readers
 * judge confounding by watching the estimate move across the columns, which
 * only works if every column is the same estimate of the same thing on the
 * same people — so the panel fits the models together on the shared complete
 * cases rather than leaving them to be assembled by hand.
 *
 * With a categorical exposure it also reports P for trend, refitting each
 * model with one number per level in place of the indicators.
 */
import { useMemo, useState } from "react";
import { analysisCols, isNumericKind, isCategoricalKind, useStore, type Session } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { runMultiModel } from "../api";
import { Tip } from "./Tip";
import ResultExporter from "./ResultExporter";
import { fmtP, pCellTitle } from "../lib/format";

interface Effect {
  level: string; reference?: boolean;
  beta?: number | null; se?: number | null; p?: number | null;
  ci_low?: number | null; ci_high?: number | null;
  ratio?: number | null; ratio_ci_low?: number | null; ratio_ci_high?: number | null;
}
interface ModelRow { label: string; covariates: string[]; effects: Effect[]; trend?: Effect }
export interface MultiModelResult {
  outcome: string; exposure: string; outcome_kind: "continuous" | "binary" | "survival";
  effect_label: string; exposure_categorical: boolean; levels: string[];
  trend_basis: string | null; n_used: number; n_dropped: number;
  models: ModelRow[]; warnings: string[]; result_text: string;
}

interface ModelSpec { label: string; covariates: string[] }

const DEFAULT_MODELS: ModelSpec[] = [
  { label: "Crude", covariates: [] },
  { label: "Model 1", covariates: [] },
  { label: "Model 2", covariates: [] },
];

export default function MultiModelPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <MultiModelPanelBody session={session} />;
}

function MultiModelPanelBody({ session }: { session: Session }) {
  const cols = useMemo(() => analysisCols(session.columns), [session.columns]);
  const numCols = useMemo(() => cols.filter((c) => isNumericKind(c.kind)).map((c) => c.name), [cols]);
  const catCols = useMemo(() => cols.filter((c) => isCategoricalKind(c.kind)).map((c) => c.name), [cols]);
  const allCols = useMemo(() => cols.map((c) => c.name), [cols]);

  const [kind, setKind] = usePersistedPanelState<"continuous" | "binary" | "survival">("multimodel", "kind", "continuous");
  const [outcome, setOutcome] = usePersistedPanelState<string>("multimodel", "outcome", "");
  const [exposure, setExposure] = usePersistedPanelState<string>("multimodel", "exposure", "");
  const [asLevels, setAsLevels] = usePersistedPanelState<boolean>("multimodel", "asLevels", false);
  const [timeCol, setTimeCol] = usePersistedPanelState<string>("multimodel", "time", "");
  const [models, setModels] = usePersistedPanelState<ModelSpec[]>("multimodel", "models", DEFAULT_MODELS);

  const [result, setResult] = useState<MultiModelResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setModel = (i: number, patch: Partial<ModelSpec>) =>
    setModels((prev) => prev.map((m, j) => (j === i ? { ...m, ...patch } : m)));
  const toggleCov = (i: number, c: string) =>
    setModel(i, {
      covariates: models[i].covariates.includes(c)
        ? models[i].covariates.filter((x) => x !== c)
        : [...models[i].covariates, c],
    });
  const addModel = () =>
    setModels((prev) => [...prev, { label: `Model ${prev.length}`, covariates: [...(prev[prev.length - 1]?.covariates ?? [])] }]);
  const removeModel = (i: number) => setModels((prev) => prev.filter((_, j) => j !== i));

  const run = async () => {
    if (!outcome || !exposure) { setError("Choose an outcome and an exposure."); return; }
    if (kind === "survival" && !timeCol) { setError("A time-to-event outcome needs a follow-up time column."); return; }
    setLoading(true); setError(null);
    try {
      const used = Array.from(new Set(models.flatMap((m) => m.covariates)));
      const res = await runMultiModel({
        session_id: session.session_id, outcome, exposure, outcome_kind: kind,
        time_col: kind === "survival" ? timeCol : undefined,
        models, exposure_categorical: asLevels,
        categorical: used.filter((c) => catCols.includes(c)),
      });
      setResult(res.data as MultiModelResult);
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(typeof msg === "string" ? msg : e instanceof Error ? e.message : "Analysis failed");
      setResult(null);
    } finally { setLoading(false); }
  };

  const ratio = result ? result.outcome_kind !== "continuous" : false;
  const fmtEffect = (e?: Effect) => {
    if (!e) return "—";
    if (e.reference) return "1.00 (reference)";
    if (e.beta == null) return "—";
    return ratio
      ? `${e.ratio?.toFixed(2)} (${e.ratio_ci_low?.toFixed(2)}–${e.ratio_ci_high?.toFixed(2)})`
      : `${e.beta.toFixed(3)} (${e.ci_low?.toFixed(3)}–${e.ci_high?.toFixed(3)})`;
  };

  const rowLevels = result
    ? (result.exposure_categorical ? result.levels : ["per unit"])
    : [];
  const exportRows = result ? [
    ...rowLevels.map((lv) => [
      lv,
      ...result.models.map((m) => {
        const e = m.effects.find((x) => x.level === lv);
        return e?.reference ? "1.00 (reference)" : `${fmtEffect(e)}${e?.p != null ? `, ${fmtP(e.p)}` : ""}`;
      }),
    ]),
    ...(result.models.some((m) => m.trend) ? [[
      "P for trend", ...result.models.map((m) => fmtP(m.trend?.p ?? null)),
    ]] : []),
  ] : [];

  return (
    <div className="flex gap-4">
      <div className="w-72 flex-shrink-0 space-y-3">
        <div className="panel space-y-3">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-gray-900">Progressive adjustment</h3>
            <Tip wide text="One exposure reported crude and then across increasingly adjusted models, in a single table. Every model is fitted on the same complete cases, so the estimate moves across the row because of adjustment rather than because the sample changed." />
          </div>

          <label className="block space-y-1">
            <span className="text-xs font-medium text-gray-500">Outcome type</span>
            <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
              value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}>
              <option value="continuous">Continuous (mean difference)</option>
              <option value="binary">Binary 0/1 (odds ratio)</option>
              <option value="survival">Time-to-event (hazard ratio)</option>
            </select>
          </label>

          <label className="block space-y-1">
            <span className="text-xs font-medium text-gray-500">Outcome</span>
            <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
              value={outcome} onChange={(e) => setOutcome(e.target.value)}>
              <option value="">— select —</option>
              {(kind === "continuous" ? numCols : allCols).map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>

          {kind === "survival" && (
            <label className="block space-y-1">
              <span className="text-xs font-medium text-gray-500">Follow-up time</span>
              <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
                value={timeCol} onChange={(e) => setTimeCol(e.target.value)}>
                <option value="">— select —</option>
                {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
          )}

          <label className="block space-y-1">
            <span className="text-xs font-medium text-gray-500">Exposure (the focus of the table)</span>
            <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
              value={exposure} onChange={(e) => setExposure(e.target.value)}>
              <option value="">— select —</option>
              {allCols.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>

          <label className="flex items-start gap-2 text-xs text-gray-600 cursor-pointer">
            <input type="checkbox" checked={asLevels} onChange={(e) => setAsLevels(e.target.checked)} className="mt-0.5" />
            <span>
              One row per level, against the lowest
              <span className="block text-[10px] text-gray-400">
                For quantile columns. Adds P for trend, refitting with one number per level.
              </span>
            </span>
          </label>
        </div>

        <div className="panel space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-gray-700">Models</h4>
            <button onClick={addModel} className="rounded border border-gray-200 px-2 py-0.5 text-[11px] text-gray-500 hover:bg-gray-50">
              + Add
            </button>
          </div>
          {models.map((m, i) => (
            <div key={i} className="rounded-lg border border-gray-200 p-2 space-y-1.5">
              <div className="flex items-center gap-1.5">
                <input value={m.label} onChange={(e) => setModel(i, { label: e.target.value })}
                  aria-label={`Model ${i + 1} label`}
                  className="min-w-0 flex-1 rounded border border-gray-300 px-1.5 py-0.5 text-xs outline-none focus:border-indigo-400" />
                <button aria-label={`Remove ${m.label}`} onClick={() => removeModel(i)}
                  disabled={models.length <= 1}
                  className="px-1 text-[11px] text-gray-400 hover:text-red-500 disabled:opacity-30">✕</button>
              </div>
              <div className="max-h-28 overflow-y-auto rounded border border-gray-100 p-1">
                {allCols.filter((c) => c !== outcome && c !== exposure && c !== timeCol).map((c) => (
                  <label key={c} className="flex items-center gap-1.5 px-0.5 py-0.5 text-[11px] text-gray-600 hover:bg-gray-50 rounded cursor-pointer">
                    <input type="checkbox" checked={m.covariates.includes(c)} onChange={() => toggleCov(i, c)} />
                    <span className="truncate">{c}</span>
                  </label>
                ))}
              </div>
              <p className="text-[10px] text-gray-400">
                {m.covariates.length ? `Adjusted for ${m.covariates.join(", ")}` : "Unadjusted"}
              </p>
            </div>
          ))}
          <button className="btn-primary w-full" onClick={run} disabled={loading || !outcome || !exposure}>
            {loading ? "Fitting…" : "Build the table"}
          </button>
          {error && <p className="text-xs text-red-500">{error}</p>}
        </div>
      </div>

      <div className="flex-1 min-w-0 space-y-3">
        {result ? (
          <>
            <div className="panel">
              <div className="mb-2 flex items-center justify-between">
                <h4 className="text-sm font-semibold text-gray-700">
                  {result.effect_label} for {result.exposure} — {result.outcome}
                </h4>
                <ResultExporter title="Progressive_adjustment"
                  headers={[result.exposure, ...result.models.map((m) => m.label)]} rows={exportRows} />
              </div>
              <div className="overflow-auto rounded border border-gray-200">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-gray-50 text-gray-500">
                      <th className="px-2 py-1.5 text-left">{result.exposure}</th>
                      {result.models.map((m) => (
                        <th key={m.label} className="px-2 py-1.5 text-right" title={m.covariates.length ? `Adjusted for ${m.covariates.join(", ")}` : "Unadjusted"}>
                          {m.label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rowLevels.map((lv) => (
                      <tr key={lv} className="border-t border-gray-100">
                        <th scope="row" className="px-2 py-1.5 text-left font-medium text-gray-700">{lv}</th>
                        {result.models.map((m) => {
                          const e = m.effects.find((x) => x.level === lv);
                          return (
                            <td key={m.label} className="px-2 py-1.5 text-right font-mono whitespace-nowrap">
                              {fmtEffect(e)}
                              {e && !e.reference && e.p != null && (
                                <span className="ml-1 text-gray-400" title={pCellTitle(e.p)}>{fmtP(e.p)}</span>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                    {result.models.some((m) => m.trend) && (
                      <tr className="border-t border-gray-200 bg-gray-50/60">
                        <th scope="row" className="px-2 py-1.5 text-left font-medium text-gray-700">
                          P for trend
                          <Tip text="Each model refitted with one number per level in place of the indicators, so a single coefficient carries the monotone trend." />
                        </th>
                        {result.models.map((m) => (
                          <td key={m.label} className="px-2 py-1.5 text-right font-mono" title={pCellTitle(m.trend?.p ?? null)}>
                            {fmtP(m.trend?.p ?? null)}
                          </td>
                        ))}
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              <p className="mt-1.5 text-[10px] text-gray-400">
                Hover a column heading for what it is adjusted for. n = {result.n_used} in every column.
                {result.trend_basis && ` P for trend scores each level by its ${result.trend_basis}.`}
              </p>
            </div>

            <div className="panel space-y-2">
              <h4 className="text-sm font-semibold text-gray-700">For the manuscript</h4>
              <p className="rounded border border-indigo-100 bg-white px-2 py-1.5 text-[11px] leading-relaxed text-gray-600">
                {result.result_text}
              </p>
              {result.warnings.map((w) => (
                <p key={w} className="rounded bg-amber-50 px-2 py-1.5 text-[11px] leading-relaxed text-amber-800">{w}</p>
              ))}
            </div>
          </>
        ) : (
          <div className="panel py-16 text-center text-gray-400">
            <p className="mb-2 text-lg">🧱</p>
            <p>Pick an exposure, then build up the adjustment sets on the left.</p>
            <p className="mx-auto mt-2 max-w-md text-xs">
              Every model is fitted on the rows that are complete for all of them, so a change across
              the columns is adjustment rather than a change of sample.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
