import { useState, type ReactNode } from "react";
import { useStore, isNumericKind, isCategoricalKind } from "../store";
import { runWeightedDescriptive } from "../api";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { useStampedResult } from "../hooks/useStampedResult";
import { describeStale } from "../lib/resultStamp";
import { Tip } from "./Tip";
import ResultExporter from "./ResultExporter";
import StaleResultNotice from "./StaleResultNotice";
import StaleGuard from "./StaleGuard";
import { fmtP } from "../lib/format";

interface WeightedRow {
  column: string;
  error?: string;
  n?: number;
  ess_kish?: number;
  w_mean?: number;
  w_sd?: number;
  ci_low?: number;
  ci_high?: number;
  w_median?: number;
  w_q1?: number;
  w_q3?: number;
  w_proportion?: number | null;
}

interface WeightedComparison {
  variable: string;
  group_a: string;
  group_b: string;
  w_mean_a: number | string;
  w_mean_b: number | string;
  diff: number | string;
  ci_low: number | string;
  ci_high: number | string;
  p: number;
}

interface Assumption {
  name: string;
  detail: string;
  met: boolean;
}

interface WeightedStatsResult {
  weight_col?: string;
  n?: number;
  results: WeightedRow[];
  comparison?: WeightedComparison;
  assumptions?: Assumption[];
  result_text?: string;
  export_rows?: string[][];
}

interface ApiErrorDetail {
  msg?: string;
}

export default function WeightedStatsPanel() {
  const session = useStore((s) => s.session);
  const columns = session?.columns ?? [];
  const sid = session?.session_id ?? "";
  const numCols = columns.filter((c) => isNumericKind(c.kind)).map((c) => c.name);
  const catCols = columns.filter((c) => isCategoricalKind(c.kind)).map((c) => c.name);

  // Persisted with the result, so a table restored after a tab switch comes
  // back with the columns it was computed from rather than blank selectors.
  const [weightCol, setWeightCol] = usePersistedPanelState("weighted_stats", "weightCol", "");
  const [valueCols, setValueCols] = usePersistedPanelState<string[]>("weighted_stats", "valueCols", []);
  const [groupCol, setGroupCol] = usePersistedPanelState("weighted_stats", "groupCol", "");
  const runParams = { weightCol, valueCols, groupCol };
  const {
    result, setResult, stale, staleReasons: staleWhy,
  } = useStampedResult<WeightedStatsResult>("weighted_stats", runParams);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggle = (c: string) =>
    setValueCols((v) => (v.includes(c) ? v.filter((x) => x !== c) : [...v, c]));

  const run = async () => {
    if (!weightCol || valueCols.length === 0) { setError("Pick a weight column and at least one value column."); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await runWeightedDescriptive({
        session_id: sid, weight_col: weightCol, value_cols: valueCols,
        group_col: groupCol || undefined,
      });
      setResult(res.data);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
      const message = e instanceof Error ? e.message : String(e);
      setError(Array.isArray(detail) ? detail.map((m: ApiErrorDetail) => m.msg ?? String(m)).join(", ")
        : (typeof detail === "string" ? detail : (message || "Failed")));
    } finally { setLoading(false); }
  };

  return (
    <div className="p-4 flex gap-4">
      {/* Controls */}
      <div className="w-72 flex-shrink-0 space-y-3">
        <div className="panel space-y-2">
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
            Weighted Descriptives
            <Tip wide text="Design-based descriptive statistics using a sampling / post-stratification weight column. Weighted mean, SD, SE, 95% CI, weighted quartiles, Kish effective sample size, and (for a 0/1 variable) a weighted proportion. With a 2-level group column it adds a weighted mean difference + two-sample weighted t-test. Weights-only — full complex-survey strata/cluster design is not yet modelled." />
          </h3>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Weight column</span>
            <select value={weightCol} onChange={(e) => { setWeightCol(e.target.value); setResult(null); }}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400">
              <option value="">— select —</option>
              {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>

          <div className="space-y-1.5">
            <span className="text-xs text-gray-500 font-medium">Value variables</span>
            <div className="max-h-56 overflow-y-auto border border-gray-200 rounded-lg p-1 space-y-0.5">
              {numCols.filter((c) => c !== weightCol).map((c) => (
                <label key={c} className="flex items-center gap-1.5 text-xs px-1 py-0.5 rounded hover:bg-gray-50 cursor-pointer">
                  <input type="checkbox" className="accent-indigo-500" checked={valueCols.includes(c)} onChange={() => toggle(c)} />
                  <span className="text-gray-700 truncate">{c}</span>
                </label>
              ))}
            </div>
            <p className="text-[10px] text-gray-400">{valueCols.length} selected</p>
          </div>

          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500 font-medium">Group (optional, 2 levels)</span>
            <select value={groupCol} onChange={(e) => setGroupCol(e.target.value)}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400">
              <option value="">— none —</option>
              {[...catCols, ...numCols].map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>

          <button onClick={run} disabled={loading}
            className="w-full px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors">
            {loading ? "Computing…" : "Compute weighted stats"}
          </button>
          {error && <p className="text-xs text-red-500">{error}</p>}
        </div>
      </div>

      {/* Results */}
      <div className="flex-1 min-w-0 space-y-3">
        {!result && !error && (
          <div className="flex items-center justify-center h-64 border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
            Pick a weight column + value variables, then compute
          </div>
        )}
        {result && stale && (
          <StaleResultNotice
            reasons={staleWhy}
            onRecompute={run}
            busy={loading}
            what="This weighted summary"
          />
        )}
        {result && (
          <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
            <div className="panel space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-gray-800">
                  Weighted statistics <span className="text-gray-400 font-normal">· weight = {result.weight_col} · <i>n</i> = {result.n}</span>
                </h4>
                {result.export_rows && (
                  <ResultExporter title={`Weighted_${result.weight_col}`} headers={result.export_rows[0]} rows={result.export_rows.slice(1)} />
                )}
              </div>
              <div className="overflow-auto rounded-lg border border-gray-200">
                <table className="w-full text-xs border-collapse">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                      {(["Variable", <i key="n">n</i>, "ESS", "W. mean", "W. SD", "95% CI", "Median", "[Q1, Q3]", "W. prop"] as ReactNode[]).map((h, i) => (
                        <th key={i} className="text-left px-2 py-1.5 font-medium">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.results.map((r: WeightedRow) => (
                      <tr key={r.column} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="px-2 py-1.5 font-mono text-gray-800">{r.column}</td>
                        {r.error ? (
                          <td colSpan={8} className="px-2 py-1.5 text-red-500">{r.error}</td>
                        ) : (
                          <>
                            <td className="px-2 py-1.5 font-mono text-gray-600">{r.n}</td>
                            <td className="px-2 py-1.5 font-mono text-gray-500">{r.ess_kish}</td>
                            <td className="px-2 py-1.5 font-mono font-semibold text-indigo-700">{r.w_mean?.toFixed(3)}</td>
                            <td className="px-2 py-1.5 font-mono text-gray-600">{r.w_sd?.toFixed(3)}</td>
                            <td className="px-2 py-1.5 font-mono text-gray-500">{r.ci_low?.toFixed(2)}–{r.ci_high?.toFixed(2)}</td>
                            <td className="px-2 py-1.5 font-mono text-gray-600">{r.w_median?.toFixed(3)}</td>
                            <td className="px-2 py-1.5 font-mono text-gray-500">[{r.w_q1?.toFixed(2)}, {r.w_q3?.toFixed(2)}]</td>
                            <td className="px-2 py-1.5 font-mono text-gray-600">
                              {r.w_proportion != null ? `${(r.w_proportion * 100).toFixed(1)}%` : "—"}
                            </td>
                          </>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {result.comparison && (
              <div className="panel space-y-2">
                <h4 className="text-sm font-semibold text-gray-800">
                  Weighted comparison — {result.comparison.variable}
                </h4>
                <div className="grid grid-cols-4 gap-2">
                  {([
                    [`${result.comparison.group_a}`, result.comparison.w_mean_a],
                    [`${result.comparison.group_b}`, result.comparison.w_mean_b],
                    ["Δ (95% CI)", `${result.comparison.diff} (${result.comparison.ci_low}–${result.comparison.ci_high})`],
                    [<>t-test <i>p</i></>, fmtP(result.comparison.p)],
                  ] as [ReactNode, ReactNode][]).map(([k, v], i) => (
                    <div key={i} className="bg-gray-50 border border-gray-200 rounded p-2 text-center">
                      <p className="text-[9px] text-gray-400">{k}</p>
                      <p className="font-semibold text-gray-800 text-xs font-mono">{v}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {result.assumptions?.map((a: Assumption, i: number) => (
              <div key={i} className={`flex items-start gap-2 text-xs px-3 py-1.5 rounded-lg ${a.met ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
                <span>{a.met ? "✓" : "⚠"}</span>
                <span><span className="font-medium">{a.name}</span> — {a.detail}</span>
              </div>
            ))}

            {result.result_text && (
              <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-3 py-2 text-xs text-indigo-900 leading-relaxed">
                {result.result_text}
              </div>
            )}
          </StaleGuard>
        )}
      </div>
    </div>
  );
}
