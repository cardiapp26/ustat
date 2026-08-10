import { useMemo, useRef, useState } from "react";
import { usePlotLayout } from "../plotStyle";
import Plot from "../PlotComponent";
import { analysisCols, isNumericKind, isCategoricalKind, type Session } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { runIntervalCensored } from "../api";
import { Tip } from "./Tip";
import ResultExporter from "./ResultExporter";
import { fmtPubP } from "../lib/format";
import type { Data, Layout } from "plotly.js";
import type { PlotData, PlotLayout, PlotCaptureHandle } from "../lib/plotTypes";

interface CurvePoint { time: number; survival: number; lower: number; upper: number }
interface RegRow {
  variable: string;
  time_ratio: number; tr_ci_low: number; tr_ci_high: number;
  hazard_ratio: number; hr_ci_low: number; hr_ci_high: number;
  p: number;
}
interface ICResult {
  n: number; n_exact: number; n_interval_censored: number; n_right_censored: number;
  median_survival_time: number | null;
  npmle_curve: CurvePoint[];
  groups: { level: string; n: number; curve: CurvePoint[] }[] | null;
  regression: { shape: number; coefficients: RegRow[]; aic: number; note?: string; error?: string } | null;
  result_text: string;
}

const PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"];

export default function IntervalCensoredPanel({ session }: { session: Session }) {
  const themedBase = usePlotLayout();
  const numCols = useMemo(() => analysisCols(session.columns).filter((c) => isNumericKind(c.kind)).map((c) => c.name), [session.columns]);
  const catCols = useMemo(() => analysisCols(session.columns).filter((c) => isCategoricalKind(c.kind)).map((c) => c.name), [session.columns]);
  const allCols = useMemo(() => analysisCols(session.columns).map((c) => c.name), [session.columns]);

  const sid = session.session_id;
  const [lowerCol, setLowerCol] = usePersistedPanelState<string>("survival", "icLower", numCols[0] ?? "");
  const [upperCol, setUpperCol] = usePersistedPanelState<string>("survival", "icUpper", numCols[1] ?? "");
  const [groupCol, setGroupCol] = usePersistedPanelState<string>("survival", "icGroup", "");
  const [covariates, setCovariates] = usePersistedPanelState<string[]>("survival", "icCovs", []);

  const [result, setResult] = useState<ICResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const plotRef = useRef<PlotCaptureHandle | null>(null);

  const toggleCov = (c: string) =>
    setCovariates((prev) => (prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]));

  const run = async () => {
    if (!lowerCol || !upperCol) { setError("Select both the lower and upper bound columns."); return; }
    if (lowerCol === upperCol) { setError("Lower and upper bound must be different columns."); return; }
    setLoading(true); setError(null);
    try {
      const res = await runIntervalCensored({
        session_id: sid, lower_col: lowerCol, upper_col: upperCol,
        covariates, group_col: groupCol || undefined,
      });
      setResult(res.data);
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(typeof msg === "string" ? msg : (e instanceof Error ? e.message : "Analysis failed"));
      setResult(null);
    } finally { setLoading(false); }
  };

  // NPMLE step curves → Plotly traces.
  const traces: PlotData[] = useMemo(() => {
    if (!result) return [];
    const mk = (curve: CurvePoint[], name: string, color: string): PlotData => ({
      x: curve.map((p) => p.time),
      y: curve.map((p) => p.survival),
      type: "scatter", mode: "lines", name,
      line: { shape: "hv", color, width: 2 },
      hovertemplate: `${name}<br>t=%{x}<br>S(t)=%{y:.3f}<extra></extra>`,
    });
    if (result.groups && result.groups.length) {
      return result.groups.map((g, i) => mk(g.curve, `${groupCol}=${g.level} (<i>n</i>=${g.n})`, PALETTE[i % PALETTE.length]));
    }
    return result.npmle_curve.length ? [mk(result.npmle_curve, "Overall", PALETTE[0])] : [];
  }, [result, groupCol]);

  const layout: PlotLayout = {
    ...themedBase,
    margin: { t: 16, r: 20, b: 48, l: 56 },
    xaxis: { title: { text: "Time" }, gridcolor: "#eef2f7", zeroline: false },
    yaxis: { title: { text: "Survival probability S(t)" }, gridcolor: "#eef2f7", range: [0, 1.02], zeroline: false },
    legend: { orientation: "h", y: -0.2 },
    showlegend: !!(result?.groups && result.groups.length),
  };

  return (
    <div className="space-y-4">
      <div className="panel">
        <div className="flex items-center gap-2 mb-2">
          <h3 className="font-semibold text-gray-900">Interval-censored survival</h3>
          <Tip wide text="Use when the event time is known only to fall inside a bracket [lower, upper] — e.g. a recurrence first seen on a scheduled scan, or seroconversion detected at a clinic visit. Plain Kaplan-Meier/Cox assume an exact event time and are biased here. Reports the Turnbull NPMLE survival curve and a Weibull regression for covariate effects. Leave the upper bound blank (or ∞) for participants still event-free at last contact (right-censored)." />
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <label className="space-y-1">
            <span className="text-xs font-medium text-gray-500">Lower bound (L)</span>
            <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
              value={lowerCol} onChange={(e) => setLowerCol(e.target.value)}>
              <option value="">— select —</option>
              {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-xs font-medium text-gray-500">Upper bound (R)</span>
            <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
              value={upperCol} onChange={(e) => setUpperCol(e.target.value)}>
              <option value="">— select —</option>
              {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-xs font-medium text-gray-500 flex items-center gap-1">
              Group (optional) <Tip text="Splits the NPMLE curve by a categorical variable for visual comparison." />
            </span>
            <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
              value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
              <option value="">— none —</option>
              {catCols.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <div className="flex items-end">
            <button onClick={run} disabled={loading}
              className="w-full rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:bg-gray-300">
              {loading ? "Running…" : "Run analysis"}
            </button>
          </div>
        </div>

        <div className="mt-3">
          <span className="text-xs font-medium text-gray-500 flex items-center gap-1">
            Covariates (Weibull regression) <Tip wide text="Optional. Adds a parametric Weibull accelerated-failure-time model, reporting each covariate's time ratio and the equivalent hazard ratio." />
          </span>
          <div className="flex flex-wrap gap-1.5 mt-1">
            {allCols.filter((c) => c !== lowerCol && c !== upperCol).map((c) => (
              <button key={c} onClick={() => toggleCov(c)}
                className={`px-2 py-1 text-xs rounded-md border ${covariates.includes(c) ? "bg-indigo-50 border-indigo-300 text-indigo-700" : "bg-white border-gray-200 text-gray-600 hover:border-gray-300"}`}>
                {c}
              </button>
            ))}
          </div>
        </div>

        {error && <div className="mt-3 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">{error}</div>}
      </div>

      {result && (
        <>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {([
              ["N", result.n],
              ["Interval-censored", result.n_interval_censored],
              ["Right-censored", result.n_right_censored],
              ["Median survival", result.median_survival_time ?? "—"],
            ] as const).map(([lbl, val]) => (
              <div key={lbl} className="rounded-xl bg-white border border-gray-200 px-3 py-2">
                <div className="text-[10px] uppercase tracking-wide text-gray-400">{lbl}</div>
                <div className="text-lg font-semibold text-gray-800 tabular-nums">{val}</div>
              </div>
            ))}
          </div>

          <div className="panel">
            <div className="flex items-center justify-between mb-1">
              <h4 className="font-semibold text-gray-900 flex items-center gap-1">
                Turnbull NPMLE survival curve
                <Tip wide text="The nonparametric maximum-likelihood survival estimate for interval-censored data — the unbiased analogue of Kaplan-Meier. Steps fall only where the data identify a drop." />
              </h4>
              <ResultExporter title="Interval-censored survival" plotRef={plotRef} />
            </div>
            <Plot data={traces as unknown as Data[]} layout={layout as unknown as Partial<Layout>}
              config={{ responsive: true, displaylogo: false, displayModeBar: false }}
              onInitialized={(_: object, gd: HTMLElement) => { plotRef.current = gd as unknown as PlotCaptureHandle; }}
              onUpdate={(_: object, gd: HTMLElement) => { plotRef.current = gd as unknown as PlotCaptureHandle; }}
              style={{ width: "100%", height: "420px" }} useResizeHandler />
          </div>

          {result.regression?.coefficients && result.regression.coefficients.length > 0 && (
            <div className="panel">
              <h4 className="font-semibold text-gray-900 mb-1 flex items-center gap-1">
                Weibull regression
                <Tip wide text="Accelerated-failure-time fit. Time ratio > 1 = longer survival; the hazard ratio is the Weibull proportional-hazards equivalent (HR = exp(−β·shape)). CI excluding 1 = significant." />
              </h4>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-200 text-xs">
                    <th className="text-left py-1.5 font-medium">Predictor</th>
                    <th className="text-right py-1.5 font-medium">Time ratio (95% CI)</th>
                    <th className="text-right py-1.5 font-medium">Hazard ratio (95% CI)</th>
                    <th className="text-right py-1.5 font-medium"><i>p</i></th>
                  </tr>
                </thead>
                <tbody>
                  {result.regression.coefficients.map((r) => {
                    const sig = r.p < 0.05;
                    return (
                      <tr key={r.variable} className={`border-b border-gray-100 ${sig ? "bg-indigo-50/40" : ""}`}>
                        <td className="py-1.5 font-mono text-gray-700">{r.variable}</td>
                        <td className="py-1.5 text-right tabular-nums">{r.time_ratio.toFixed(2)} ({r.tr_ci_low.toFixed(2)}–{r.tr_ci_high.toFixed(2)})</td>
                        <td className={`py-1.5 text-right tabular-nums ${sig ? "font-semibold text-indigo-700" : ""}`}>{r.hazard_ratio.toFixed(2)} ({r.hr_ci_low.toFixed(2)}–{r.hr_ci_high.toFixed(2)})</td>
                        <td className={`py-1.5 text-right tabular-nums ${sig ? "font-semibold text-indigo-700" : "text-gray-600"}`}>{r.p == null ? "—" : <><i>p</i>{fmtPubP(r.p).slice(1)}</>}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="text-[11px] text-gray-400 mt-1">Weibull shape = {result.regression.shape}, AIC = {result.regression.aic}.</p>
            </div>
          )}

          {result.regression?.error && (
            <div className="rounded-lg bg-amber-50 border border-amber-200 px-3 py-2 text-sm text-amber-700">{result.regression.error}</div>
          )}

          {result.result_text && (
            <div className="rounded-xl border border-indigo-100 bg-white px-4 py-3 flex gap-3">
              <span className="text-indigo-400 text-xl flex-shrink-0 mt-0.5">💬</span>
              <p className="text-sm text-gray-700 leading-relaxed">{result.result_text}</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
