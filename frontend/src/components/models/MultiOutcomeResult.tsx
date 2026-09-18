/** Multi-outcome regression result card: APA consolidated table +
 *  result_text + per-outcome model fit + ResultExporter. Extracted from
 *  ModelsPanel.tsx. */
import ResultExporter from "../ResultExporter";
import type { Provenance } from "../../lib/engine/provenance";
import { fmtP, pCellTitle } from "../../lib/format";

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

export type MultiOutcomeResultData = {
  outcomes?: unknown;
  predictors_order?: unknown;
  rows?: unknown;
  model_fit?: Record<string, MultiOutcomeFit>;
  n_by_outcome?: Record<string, number>;
  result_text?: string;
};

export default function MultiOutcomeResult({
  result,
  standardize,
  stale = false,
  staleReason,
  provenance,
}: {
  stale?: boolean;
  staleReason?: string;
  provenance?: Provenance | null;
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
        <ResultExporter title="multi_outcome_regression" headers={exportHeaders} rows={exportRows} stale={stale} staleReason={staleReason} provenance={provenance} />
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
