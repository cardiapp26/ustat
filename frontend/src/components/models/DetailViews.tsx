import { useRef, useState } from "react";
import { usePlotLayout } from "../../plotStyle";
import TitledPlot from "../TitledPlot";
import type { PlotCaptureHandle } from "../../lib/plotTypes";
import { fmtP } from "../../lib/format";
import { adjustP, type Coefficient, type PredictorInfo, type PredictionResult,
  type ModelSummary } from "./shared";

export function PredictionPanel({ result }: { result: PredictionResult }) {
  const predictorInfo: Record<string, PredictorInfo> = result.predictor_info ?? {};
  const coefs: Coefficient[] = result.coefficients ?? [];
  const outcome: string = result.outcome ?? "";
  const residualSe: number = result.residual_se ?? 0;
  const dfResid: number = result.df_resid ?? 100;

  // ── t quantile approximation (for PI) ─────────────────────────────────────
  const tQuantile = (ci: number) => {
    // good approximation for df > 30; exact for df → ∞
    if (dfResid > 200) return ci === 0.99 ? 2.576 : ci === 0.95 ? 1.96 : 1.645;
    const z = ci === 0.99 ? 2.576 : ci === 0.95 ? 1.96 : 1.645;
    return z * (1 + 1 / (4 * dfResid));  // simple correction
  };

  // ── Initialize slider values: numeric → median, categorical → first cat ───
  const initVals = () => {
    const v: Record<string, number | string> = {};
    for (const [col, info] of Object.entries(predictorInfo)) {
      if (info.type === "numeric") v[col] = info.median ?? info.mean ?? 0;
      else v[col] = info.categories?.[0] ?? "";
    }
    return v;
  };
  const [vals, setVals] = useState<Record<string, number | string>>(initVals);
  const [ciLevel, setCiLevel] = useState(0.95);
  const [showPI, setShowPI] = useState(false);
  const [sortByCat, setSortByCat] = useState(false);

  // ── Client-side prediction ─────────────────────────────────────────────────
  const predict = (overrides: Record<string, number | string> = {}) => {
    const v = { ...vals, ...overrides };
    let yhat = 0;
    for (const c of coefs) {
      const name: string = c.variable;
      const est: number = c.estimate ?? 0;
      if (name === "const" || name === "Intercept") {
        yhat += est;
      } else if (name in predictorInfo && predictorInfo[name].type === "numeric") {
        yhat += est * (Number(v[name]) || 0);
      } else {
        // Dummy variable — find parent by prefix match
        const parent = Object.keys(predictorInfo).find(
          (p) => predictorInfo[p]?.type === "categorical" && name.startsWith(p + "_")
        );
        if (parent) {
          const level = name.slice(parent.length + 1);
          yhat += est * (String(v[parent]) === level ? 1 : 0);
        } else {
          // Numeric predictor whose name is not in predictor_info (shouldn't happen but safe)
          if (name in v) yhat += est * (Number(v[name]) || 0);
        }
      }
    }
    return yhat;
  };

  const currentPred = predict();
  const tQ = tQuantile(ciLevel);
  const piHalf = tQ * residualSe * Math.sqrt(1 + 1 / Math.max(result.n ?? 100, 1));
  const piLow  = currentPred - piHalf;
  const piHigh = currentPred + piHalf;

  const exportCSV = () => {
    const rows: string[][] = [
      ["Variable", "Coefficient", "SE", "t", "p", "CI_low", "CI_high"],
      ...coefs.map((c: Coefficient) => [c.variable, c.estimate, c.se, c.t, c.p, c.ci_low, c.ci_high].map(String)),
      [],
      ["Outcome", outcome],
      ["R²", result.r_squared?.toFixed(4) ?? ""],
      ["Adj R²", result.adj_r_squared?.toFixed(4) ?? ""],
      ["N", String(result.n ?? "")],
      ["Residual SE", residualSe.toFixed(5)],
      [],
      ["--- Current Prediction ---"],
      ...Object.entries(vals).map(([k, v]) => [k, String(v)]),
      ["Predicted " + outcome, currentPred.toFixed(4)],
    ];
    const csv = rows.map((r) => r.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `Model_${outcome}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const numPreds = Object.entries(predictorInfo).filter(([, i]) => i.type === "numeric");
  const catPreds = Object.entries(predictorInfo).filter(([, i]) => i.type === "categorical");

  // Shared Plotly base layout. Background, font family and colourway come from
  // the global chart theme; only the small type size is local, because these
  // panels are half-height.
  const themed = usePlotLayout();
  const plotBase = {
    ...themed,
    font: { ...(themed.font as object), size: 10 },
    margin: { t: 32, r: 10, b: 44, l: 44 },
    showlegend: false,
  };

  return (
    <div className="panel space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-gray-900">
          Predicted <span className="text-indigo-600">{outcome}</span>
        </h4>
        <button
          onClick={exportCSV}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs text-indigo-600 border border-indigo-200 hover:bg-indigo-50 transition-colors"
        >
          ↓ Export Model (CSV)
        </button>
      </div>

      {/* Charts grid */}
      {(numPreds.length > 0 || catPreds.length > 0) && (
        <div className="grid grid-cols-2 gap-4">
          {/* Numeric predictor line charts */}
          {numPreds.map(([col, info]) => {
            const N = 120;
            const lo = info.min ?? 0, hi = info.max ?? 0;
            const xs = Array.from({ length: N + 1 }, (_, i) => lo + (hi - lo) * i / N);
            const ys = xs.map((x) => predict({ [col]: x }));
            const cx = Number(vals[col]);
            const cy = predict();
            return (
              <div key={col} className="space-y-1">
                <p className="text-xs font-medium text-gray-600 text-center">
                  Predicted <em>{outcome}</em> vs. {col}
                </p>
                <TitledPlot
                  storageKey={"predict:num:" + col}
                  defaultTitle=""
                  defaultSubtitle=""
                  defaultXAxis={col}
                  defaultYAxis={outcome}
                  data={[
                    { type: "scatter" as const, mode: "lines" as const, x: xs, y: ys,
                      line: { color: "#6366f1", width: 2 }, hovertemplate: `${col}: %{x:.2f}<br>Ŷ: %{y:.3f}<extra></extra>` },
                    ...(showPI ? [{
                      type: "scatter" as const, mode: "lines" as const,
                      x: [...xs, ...xs.slice().reverse()],
                      y: [...ys.map((y) => y + piHalf), ...ys.map((y) => y - piHalf).reverse()],
                      fill: "toself" as const, fillcolor: "rgba(99,102,241,0.10)",
                      line: { color: "transparent" }, hoverinfo: "skip" as const, showlegend: false,
                    }] : []),
                    { type: "scatter" as const, mode: "markers" as const, x: [cx], y: [cy],
                      marker: { color: "white", size: 11, line: { color: "#6366f1", width: 2.5 } },
                      hovertemplate: `${col} = ${cx.toFixed(1)}<br>Ŷ = ${cy.toFixed(3)}<extra></extra>` },
                  ]}
                  layout={{
                    ...plotBase, height: 200, autosize: true,
                    xaxis: { title: { text: col, font: { size: 10 } }, gridcolor: "#f3f4f6" },
                    yaxis: { title: { text: outcome, font: { size: 10 } }, gridcolor: "#f3f4f6", zeroline: false },
                  }}
                  config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                />
                {/* Slider */}
                <div className="flex items-center gap-2 px-1">
                  <input
                    type="number"
                    value={Number(vals[col]).toFixed(1)}
                    onChange={(e) => setVals((p) => ({ ...p, [col]: Number(e.target.value) }))}
                    className="w-16 text-xs border border-gray-300 rounded px-1.5 py-0.5 text-right font-mono"
                  />
                  <input
                    type="range"
                    min={lo} max={hi}
                    step={(hi - lo) / 200}
                    value={Number(vals[col])}
                    onChange={(e) => setVals((p) => ({ ...p, [col]: Number(e.target.value) }))}
                    className="flex-1 accent-indigo-500"
                  />
                </div>
              </div>
            );
          })}

          {/* Categorical predictor bar charts */}
          {catPreds.map(([col, info]) => {
            const cats: string[] = info.categories ?? [];
            const preds = cats.map((cat: string) => predict({ [col]: cat }));
            const selectedCat = String(vals[col]);
            const pairs = cats.map((cat: string, i: number) => ({ cat, pred: preds[i] }));
            const sorted = sortByCat ? [...pairs].sort((a, b) => b.pred - a.pred) : pairs;
            return (
              <div key={col} className="space-y-1">
                <p className="text-xs font-medium text-gray-600 text-center">
                  Predicted <em>{outcome}</em> by {col}
                </p>
                <TitledPlot
                  storageKey={"predict:cat:" + col}
                  defaultTitle=""
                  defaultSubtitle=""
                  defaultXAxis={outcome}
                  defaultYAxis=""
                  data={[{
                    type: "bar" as const,
                    orientation: "h" as const,
                    x: sorted.map((p) => p.pred),
                    y: sorted.map((p) => p.cat),
                    text: sorted.map((p) => p.pred.toFixed(2)),
                    textposition: "outside" as const,
                    marker: {
                      color: sorted.map((p) => p.cat === selectedCat ? "#ef4444" : "#9ca3af"),
                    },
                    hovertemplate: `%{y}: Ŷ = %{x:.4f}<extra></extra>`,
                  }]}
                  layout={{
                    ...plotBase, height: Math.max(160, cats.length * 38 + 60), autosize: true,
                    xaxis: { title: { text: outcome, font: { size: 10 } }, gridcolor: "#f3f4f6" },
                    yaxis: { gridcolor: "transparent", zeroline: false, autorange: "reversed" as const },
                  }}
                  config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                />
                <div className="flex items-center gap-2 px-1">
                  <select
                    value={selectedCat}
                    onChange={(e) => setVals((p) => ({ ...p, [col]: e.target.value }))}
                    className="select text-xs flex-1"
                  >
                    {cats.map((cat: string) => <option key={cat}>{cat}</option>)}
                  </select>
                  <label className="flex items-center gap-1 text-[10px] text-gray-500 cursor-pointer whitespace-nowrap">
                    <input type="checkbox" checked={sortByCat} onChange={(e) => setSortByCat(e.target.checked)} className="accent-indigo-500" />
                    Sort by predicted
                  </label>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Big predicted value display */}
      <div className="rounded-xl bg-gray-900 text-white p-6 text-center relative">
        <p className="text-xs text-gray-400 mb-1">Predicted {outcome} =</p>
        <p className="text-5xl font-bold tracking-tight">{currentPred.toFixed(2)}</p>
        {showPI && residualSe > 0 && (
          <p className="text-sm text-gray-400 mt-2">
            {(ciLevel * 100).toFixed(0)}% PI: [{piLow.toFixed(2)}, {piHigh.toFixed(2)}]
          </p>
        )}
      </div>

      {/* PI controls */}
      <div className="flex items-center gap-4 px-1">
        <label className="flex items-center gap-2 cursor-pointer text-xs text-gray-600">
          <input type="checkbox" checked={showPI} onChange={(e) => setShowPI(e.target.checked)} className="accent-indigo-500" />
          Show prediction intervals
        </label>
        <input
          type="range" min={0.80} max={0.99} step={0.01} value={ciLevel}
          onChange={(e) => setCiLevel(Number(e.target.value))}
          disabled={!showPI}
          className={`w-28 accent-indigo-500 ${!showPI ? "opacity-30" : ""}`}
        />
        <span className="text-xs text-gray-500 font-mono">%{(ciLevel * 100).toFixed(0)}</span>
      </div>
    </div>
  );
}

// ── Coefficient Detail Panel (Plotly normal distribution on click) ────────────
export function CoefDetailPanel({
  coef, nullHyp, onClose,
}: {
  coef: Coefficient; nullHyp: string; onClose: () => void;
}) {
  const coefPlotRef = useRef<PlotCaptureHandle | null>(null);
  const plotBase = usePlotLayout();
  const beta = coef.log_odds ?? coef.log_irr ?? coef.log_hr ?? coef.estimate ?? 0;
  const se   = coef.se ?? 1;
  const adjP = adjustP(coef.p, beta, nullHyp);

  if (!isFinite(beta) || !isFinite(se) || se <= 0) return null;

  const span = 4 * se;
  const lo   = beta - span, hi = beta + span;
  const N    = 200;
  const xs   = Array.from({ length: N + 1 }, (_, i) => lo + (hi - lo) * i / N);
  const ys   = xs.map((x) => Math.exp(-0.5 * ((x - beta) / se) ** 2) / (se * Math.sqrt(2 * Math.PI)));

  const fillX = [...xs, ...xs.slice().reverse()];
  const fillY = [...ys, ...xs.map(() => 0)];

  const col = adjP < 0.001 ? "#3730a3" : adjP < 0.01 ? "#4338ca" : adjP < 0.05 ? "#6366f1" : "#9ca3af";

  return (
    <div className="panel border border-indigo-100 bg-indigo-50/30 relative">
      <button onClick={onClose} className="absolute top-2 right-2 text-gray-400 hover:text-gray-700 text-xs">✕ close</button>
      <h5 className="text-xs font-semibold text-gray-600 mb-2">
        Coefficient Detail — <span className="font-mono text-indigo-700">{coef.variable}</span>
      </h5>
      <div className="flex gap-4 items-start">
        <TitledPlot
          plotRefOut={coefPlotRef}
          storageKey={"coefdetail:" + coef.variable}
          defaultTitle=""
          defaultSubtitle=""
          defaultXAxis="β (coefficient)"
          defaultYAxis="density"
          data={[
            { type: "scatter" as const, x: fillX, y: fillY, fill: "toself",
              fillcolor: `${col}22`, line: { color: "transparent" }, hoverinfo: "skip", showlegend: false },
            { type: "scatter" as const, x: xs, y: ys, mode: "lines" as const,
              line: { color: col, width: 2 }, name: "N(β, SE)", hovertemplate: "x: %{x:.4f}<br>density: %{y:.4f}<extra></extra>" },
            { type: "scatter" as const, x: [0, 0], y: [0, Math.max(...ys) * 1.05],
              mode: "lines" as const, line: { color: "#9ca3af", dash: "dot" as const, width: 1.5 },
              name: "H₀ = 0", hoverinfo: "skip" as const },
            { type: "scatter" as const, x: [beta, beta], y: [0, Math.max(...ys) * 1.05],
              mode: "lines" as const, line: { color: col, dash: "dash" as const, width: 1.5 },
              name: `β = ${beta.toFixed(4)}`, hoverinfo: "skip" as const },
          ]}
          layout={{
            ...plotBase,
            font: { ...(plotBase.font as object), size: 11 },
            height: 200,
            margin: { t: 10, r: 20, b: 40, l: 50 },
            xaxis: { title: { text: "β (coefficient)", font: { size: 10 } }, gridcolor: "#e5e7eb", zeroline: false },
            yaxis: { title: { text: "density", font: { size: 10 } }, gridcolor: "#e5e7eb", zeroline: false },
            legend: { font: { size: 10 }, x: 0.65, y: 0.95, xanchor: "left" as const, yanchor: "top" as const },
            showlegend: true,
          }}
          config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        />
        <div className="flex-shrink-0 space-y-2 min-w-[130px] pt-2">
          {[
            ["β", beta.toFixed(5)],
            ["SE", se.toFixed(5)],
            ["z / t", (coef.z ?? coef.t)?.toFixed(4) ?? "–"],
            ["p (adj)", fmtP(adjP)],
            ...(coef.ci_low != null && coef.ci_high != null ? [["95% CI", `${coef.ci_low.toFixed(3)} – ${coef.ci_high.toFixed(3)}`]] : []),
            ...(coef.or_ci_low != null && coef.or_ci_high != null ? [["OR CI", `${coef.or_ci_low.toFixed(3)} – ${coef.or_ci_high.toFixed(3)}`]] : []),
            ...(coef.hr_ci_low  != null && coef.hr_ci_high != null ? [["HR CI", `${coef.hr_ci_low.toFixed(3)} – ${coef.hr_ci_high.toFixed(3)}`]] : []),
            ...(coef.irr_ci_low != null && coef.irr_ci_high != null ? [["IRR CI", `${coef.irr_ci_low.toFixed(3)} – ${coef.irr_ci_high.toFixed(3)}`]] : []),
            ...(coef.odds_ratio != null ? [["OR", coef.odds_ratio.toFixed(4)]] : []),
            ...(coef.hr != null         ? [["HR", coef.hr.toFixed(4)]] : []),
            ...(coef.irr != null        ? [["IRR", coef.irr.toFixed(4)]] : []),
          ].map(([k, v]) => (
            <div key={k}>
              <p className="text-[10px] text-gray-400">{k}</p>
              <p className={`text-xs font-mono font-semibold ${adjP < 0.05 ? "text-indigo-700" : "text-gray-700"}`}>{v}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Compact SPSS-style Model Summary Table ──────────────────────────────────

export function ModelSummaryTable({ s }: { s: ModelSummary }) {
  const cl = s.classification;
  const hl = s.hosmer_lemeshow;
  const om = s.omnibus;

  return (
    <div className="overflow-auto rounded-xl border border-gray-200">
      <table className="text-xs w-full border-collapse">
        {/* Model Fit */}
        <thead>
          <tr className="bg-gray-50"><th colSpan={4} className="text-left px-3 py-1.5 text-[10px] font-semibold text-gray-500 uppercase tracking-wider border-b border-gray-200">Model Fit</th></tr>
          <tr className="bg-gray-50 text-gray-400">
            <th className="px-3 py-1 text-left font-medium border-b border-gray-100">Metric</th>
            <th className="px-3 py-1 text-right font-medium border-b border-gray-100">Value</th>
            <th className="px-3 py-1 text-left font-medium border-b border-gray-100" colSpan={2}>Interpretation</th>
          </tr>
        </thead>
        <tbody>
          {om && (
            <tr className="border-b border-gray-50">
              <td className="px-3 py-1.5 text-gray-600">Omnibus χ²</td>
              <td className="px-3 py-1.5 text-right font-mono text-gray-800">{om.chi2?.toFixed(3)} <span className="text-gray-400">(df={om.df})</span></td>
              <td className="px-3 py-1.5" colSpan={2}>
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${om.p < 0.05 ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}>
                  <i>p</i> = {fmtP(om.p)}
                </span>
                <span className="text-gray-400 ml-1.5 text-[10px]">{om.p < 0.05 ? "Model significant" : "Not significant"}</span>
              </td>
            </tr>
          )}
          <tr className="border-b border-gray-50">
            <td className="px-3 py-1.5 text-gray-600">-2 Log Likelihood</td>
            <td className="px-3 py-1.5 text-right font-mono text-gray-800">{s.minus2ll?.toFixed(3)}</td>
            <td className="px-3 py-1.5 text-gray-400 text-[10px]" colSpan={2}>Lower = better fit</td>
          </tr>
          <tr className="border-b border-gray-50">
            <td className="px-3 py-1.5 text-gray-600">Cox &amp; Snell R²</td>
            <td className="px-3 py-1.5 text-right font-mono text-gray-800">{s.cox_snell_r2?.toFixed(4)}</td>
            <td className="px-3 py-1.5 text-gray-400 text-[10px]" colSpan={2}>Max &lt; 1.0</td>
          </tr>
          <tr className="border-b border-gray-50 bg-indigo-50/30">
            <td className="px-3 py-1.5 text-indigo-700 font-medium">Nagelkerke R²</td>
            <td className="px-3 py-1.5 text-right font-mono font-bold text-indigo-700">{s.nagelkerke_r2?.toFixed(4)}</td>
            <td className="px-3 py-1.5" colSpan={2}>
              <span className="text-[10px] text-indigo-600">{(s.nagelkerke_r2 ?? 0) >= 0.4 ? "Excellent" : (s.nagelkerke_r2 ?? 0) >= 0.2 ? "Good" : (s.nagelkerke_r2 ?? 0) >= 0.1 ? "Moderate" : "Weak"} explanatory power</span>
            </td>
          </tr>
          {s.auc != null && (
            <tr className="border-b border-gray-50 bg-indigo-50/30">
              <td className="px-3 py-1.5 text-indigo-700 font-medium">AUC</td>
              <td className="px-3 py-1.5 text-right font-mono font-bold text-indigo-700">{s.auc?.toFixed(4)}</td>
              <td className="px-3 py-1.5" colSpan={2}>
                <span className="text-[10px] text-indigo-600">{s.auc >= 0.9 ? "Excellent" : s.auc >= 0.8 ? "Good" : s.auc >= 0.7 ? "Fair" : "Poor"} discrimination</span>
              </td>
            </tr>
          )}

          {/* Hosmer-Lemeshow */}
          {hl && (<>
            <tr className="bg-gray-50"><td colSpan={4} className="px-3 py-1.5 text-[10px] font-semibold text-gray-500 uppercase tracking-wider border-y border-gray-200">Calibration</td></tr>
            <tr className="border-b border-gray-50">
              <td className="px-3 py-1.5 text-gray-600">Hosmer-Lemeshow</td>
              <td className="px-3 py-1.5 text-right font-mono text-gray-800">{hl.chi2?.toFixed(3)} <span className="text-gray-400">(df={hl.df})</span></td>
              <td className="px-3 py-1.5" colSpan={2}>
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${hl.p >= 0.05 ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}>
                  <i>p</i> = {fmtP(hl.p)}
                </span>
                <span className="text-gray-400 ml-1.5 text-[10px]">{hl.p >= 0.05 ? "✓ Good calibration" : "⚠ Poor calibration"}</span>
              </td>
            </tr>
          </>)}

          {/* Classification */}
          {cl && (<>
            <tr className="bg-gray-50"><td colSpan={4} className="px-3 py-1.5 text-[10px] font-semibold text-gray-500 uppercase tracking-wider border-y border-gray-200">Classification (cutoff = 0.50)</td></tr>
            <tr className="border-b border-gray-50">
              <td className="px-3 py-1.5 text-gray-600">Accuracy</td>
              <td className="px-3 py-1.5 text-right font-mono font-semibold text-gray-800">{(cl.accuracy * 100).toFixed(1)}%</td>
              <td className="px-3 py-1.5 text-gray-400 text-[10px]" colSpan={2}>
                TP={cl.tp} TN={cl.tn} FP={cl.fp} FN={cl.fn}
              </td>
            </tr>
            <tr className="border-b border-gray-50">
              <td className="px-3 py-1.5 text-gray-600">Sensitivity / Specificity</td>
              <td className="px-3 py-1.5 text-right font-mono text-gray-800">{(cl.sensitivity * 100).toFixed(1)}% / {(cl.specificity * 100).toFixed(1)}%</td>
              <td className="px-3 py-1.5 text-gray-400 text-[10px]" colSpan={2}>
                PPV={(cl.ppv * 100).toFixed(1)}% NPV={(cl.npv * 100).toFixed(1)}%
              </td>
            </tr>
          </>)}
        </tbody>
      </table>
    </div>
  );
}

