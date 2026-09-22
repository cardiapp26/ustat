import { useState } from "react";
import { useStore, isNumericKind, type Session } from "../store";
import { runCronbach } from "../api";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { useStampedResult } from "../hooks/useStampedResult";
import { describeStale } from "../lib/resultStamp";
import ResultExporter from "./ResultExporter";
import StaleResultNotice from "./StaleResultNotice";
import StaleGuard from "./StaleGuard";

interface ItemStat {
  item: string;
  mean?: number;
  sd?: number;
  item_total_r?: number | null;
  alpha_if_deleted?: number | null;
}

interface ScaleSummary {
  mean: number | string;
  sd: number | string;
  min: number | string;
  max: number | string;
  skewness: number | string;
}

interface ReliabilityResult {
  alpha: number;
  omega?: number | null;
  interpretation?: string;
  k?: number;
  n?: number;
  result_text?: string;
  scale_summary?: ScaleSummary;
  item_stats?: ItemStat[];
  r_code?: string;
  export_rows?: string[][];
}

export default function ReliabilityPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <ReliabilityPanelBody session={session} />;
}

function ReliabilityPanelBody({ session }: { session: Session }) {
  const numCols = session.columns.filter((c) => isNumericKind(c.kind)).map((c) => c.name);

  // Persisted beside the result: a cached alpha whose item list reset to
  // empty on remount would read as computed from different settings.
  const [items, setItems] = usePersistedPanelState<string[]>("reliability", "items", []);
  const {
    result, setResult, stale, staleReasons: staleWhy,
  } = useStampedResult<ReliabilityResult>("reliability", { items });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await runCronbach({ session_id: session.session_id, items });
      setResult(res.data);
    } catch (e: unknown) { setError((e as { response?: { data?: { detail?: string } } }).response?.data?.detail ?? "Error"); }
    finally { setLoading(false); }
  };

  const alphaColor = (a: number) =>
    a >= 0.9 ? "text-green-700 bg-green-50" : a >= 0.8 ? "text-blue-700 bg-blue-50" :
    a >= 0.7 ? "text-indigo-700 bg-indigo-50" : a >= 0.6 ? "text-amber-700 bg-amber-50" : "text-red-700 bg-red-50";

  return (
    <div className="flex gap-4">
      <div className="w-64 flex-shrink-0 space-y-4">
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Scale Items</h3>
          <p className="text-xs text-gray-400">Select the numeric columns that form your measurement scale (e.g. questionnaire items).</p>
          <select multiple className="select w-full h-48" value={items}
            onChange={(e) => setItems(Array.from(e.target.selectedOptions, o => o.value))}>
            {numCols.map((c) => <option key={c}>{c}</option>)}
          </select>
          <p className="text-[10px] text-gray-400">Hold Ctrl/Cmd to select multiple. Need at least 2 items.</p>
          <button className="btn-primary w-full" onClick={run} disabled={loading || items.length < 2}>
            {loading ? "Computing\u2026" : "Compute Reliability"}
          </button>
          {error && <p className="text-red-400 text-xs">{error}</p>}
        </div>

        <div className="panel bg-indigo-50 border-indigo-200 space-y-2">
          <p className="text-[10px] font-bold text-indigo-900 uppercase">Cronbach's Alpha</p>
          <p className="text-xs text-indigo-800">{"Measures internal consistency of a multi-item scale. Higher alpha = items measure the same construct. Rule of thumb: > 0.70 acceptable, > 0.80 good, > 0.90 excellent."}</p>
          <p className="text-[10px] font-bold text-indigo-900 uppercase mt-2">Alpha-if-deleted</p>
          <p className="text-xs text-indigo-800">{"If removing an item increases alpha, that item may not belong in the scale. Consider dropping items where alpha-if-deleted > overall alpha."}</p>
        </div>
      </div>

      <div className="flex-1 space-y-3">
        {result && stale && (
          <StaleResultNotice reasons={staleWhy} onRecompute={run} busy={loading} what="This reliability analysis" />
        )}
        {result ? (
          <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
          <div className="space-y-4">
            {/* Main result card */}
            <div className="panel">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-gray-900">Scale Reliability Report</h4>
                <ResultExporter title="Reliability_Analysis" headers={result.export_rows?.[0]} rows={result.export_rows?.slice(1)} />
              </div>

              <div className="flex flex-wrap items-center gap-4 mb-4">
                <div className={`text-2xl font-bold px-4 py-2 rounded-xl ${alphaColor(result.alpha)}`}>
                  α = {result.alpha?.toFixed(3)}
                </div>
                {result.omega !== undefined && result.omega !== null && (
                  <div className={`text-2xl font-bold px-4 py-2 rounded-xl ${alphaColor(result.omega)}`}>
                    ω = {result.omega?.toFixed(3)}
                  </div>
                )}
                <div>
                  <p className="text-sm font-medium text-gray-700">{result.interpretation}</p>
                  <p className="text-xs text-gray-400">{result.k} items, <i>n</i> = {result.n}</p>
                </div>
              </div>

              {result.result_text && (
                <div className="rounded-lg border border-indigo-100 bg-white px-3 py-2 text-xs text-gray-600 leading-relaxed mb-3">
                  <span className="text-indigo-400 mr-1">\uD83D\uDCAC</span> {result.result_text}
                </div>
              )}

              {/* Scale summary */}
              {result.scale_summary && (
                <div className="grid grid-cols-5 gap-2 text-center mb-4">
                  {[["Mean", result.scale_summary.mean], ["SD", result.scale_summary.sd],
                    ["Min", result.scale_summary.min], ["Max", result.scale_summary.max],
                    ["Skewness", result.scale_summary.skewness]].map(([label, val]) => (
                    <div key={label as string} className="bg-gray-50 rounded-lg px-2 py-1.5">
                      <p className="text-[10px] text-gray-400">{label}</p>
                      <p className="text-sm font-mono text-gray-700">{typeof val === "number" ? val.toFixed(3) : "\u2014"}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Item analysis table */}
            {(result.item_stats?.length ?? 0) > 0 && result.item_stats && (
              <div className="panel">
                <h4 className="text-sm font-semibold text-gray-700 mb-2">Item Analysis</h4>
                <div className="overflow-auto rounded border border-gray-200">
                  <table className="w-full text-xs">
                    <thead><tr className="bg-gray-50">
                      <th className="px-3 py-1.5 text-left">Item</th>
                      <th className="px-3 py-1.5 text-right">Mean</th>
                      <th className="px-3 py-1.5 text-right">SD</th>
                      <th className="px-3 py-1.5 text-right">Item-Total r</th>
                      <th className="px-3 py-1.5 text-right">\u03B1 if deleted</th>
                      <th className="px-3 py-1.5 text-center">Flag</th>
                    </tr></thead>
                    <tbody>
                      {result.item_stats.map((item: ItemStat) => {
                        const flagDrop = item.alpha_if_deleted != null && item.alpha_if_deleted > result.alpha;
                        const flagLowR = item.item_total_r != null && item.item_total_r < 0.3;
                        return (
                          <tr key={item.item} className={`border-t border-gray-100 ${flagDrop ? "bg-amber-50" : ""}`}>
                            <td className="px-3 py-1.5 font-medium text-gray-700">{item.item}</td>
                            <td className="px-3 py-1.5 text-right font-mono">{item.mean?.toFixed(3)}</td>
                            <td className="px-3 py-1.5 text-right font-mono">{item.sd?.toFixed(3)}</td>
                            <td className={`px-3 py-1.5 text-right font-mono ${flagLowR ? "text-red-600 font-semibold" : ""}`}>
                              {item.item_total_r?.toFixed(3)}
                            </td>
                            <td className={`px-3 py-1.5 text-right font-mono ${flagDrop ? "text-amber-700 font-semibold" : ""}`}>
                              {item.alpha_if_deleted?.toFixed(3)}
                            </td>
                            <td className="px-3 py-1.5 text-center">
                              {flagDrop && <span className="text-amber-600 text-[10px]" title="Removing this item increases alpha">\u26A0 drop?</span>}
                              {flagLowR && !flagDrop && <span className="text-red-500 text-[10px]" title="Low item-total correlation">low r</span>}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* R code */}
            {result.r_code && (
              <details className="text-xs panel">
                <summary className="text-gray-400 cursor-pointer hover:text-indigo-600">R code</summary>
                <pre className="mt-1 p-2 bg-gray-50 rounded-lg text-gray-600 font-mono text-[10px] whitespace-pre-wrap">{result.r_code}</pre>
              </details>
            )}
          </div>
          </StaleGuard>
        ) : (
          <div className="panel text-center text-gray-400 py-12">
            <p className="text-lg mb-2">\uD83D\uDCCB</p>
            <p>Select scale items from the left panel to compute reliability</p>
            <p className="text-xs mt-2">Cronbach's alpha measures how consistently your items measure the same underlying construct</p>
          </div>
        )}
      </div>
    </div>
  );
}
