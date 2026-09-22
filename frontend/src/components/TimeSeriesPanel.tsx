import { useState, useRef, type ReactNode } from "react";
import { useStore, isNumericKind } from "../store";
import { usePlotLayout, usePalette } from "../plotStyle";
import { runArima, runDecompose, runStationarity } from "../api";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { useStampedResult } from "../hooks/useStampedResult";
import { describeStale } from "../lib/resultStamp";
import { Tip } from "./Tip";
import TitledPlot from "./TitledPlot";
import { fmtP } from "../lib/format";
import type { PlotCaptureHandle } from "../lib/plotTypes";
import ResultExporter from "./ResultExporter";
import ThreeCol from "./ThreeCol";
import StaleResultNotice from "./StaleResultNotice";
import StaleGuard from "./StaleGuard";

type Mode = "arima" | "decompose" | "stationarity";

const asNumber = (v: unknown): number | undefined => typeof v === "number" ? v : undefined;
const asString = (v: unknown): string | undefined => typeof v === "string" ? v : undefined;

/** A point in an ACF/PACF stem plot returned by the stationarity endpoint. */
interface StemPoint {
  lag: number;
  value: number;
  ci_low: number;
  ci_high: number;
}

/** A single ARIMA coefficient row returned by the arima endpoint. */
interface Coefficient {
  term: string;
  estimate?: number;
  se?: number;
  p?: number;
}

export default function TimeSeriesPanel() {
  const session = useStore((s) => s.session);
  const showGrid = useStore((s) => s.showGrid);
  const baseLayout = usePlotLayout();
  const pal = usePalette();
  const mainRef = useRef<PlotCaptureHandle | null>(null);
  const acfRef = useRef<PlotCaptureHandle | null>(null);
  const pacfRef = useRef<PlotCaptureHandle | null>(null);
  const decompRef = useRef<PlotCaptureHandle | null>(null);

  const columns = session?.columns ?? [];
  const sid = session?.session_id ?? "";
  const numCols = columns.filter((c) => isNumericKind(c.kind)).map((c) => c.name);

  // Persisted with the result. `mode` above all: it picks how the result is
  // drawn, and a restored decomposition read under the default ARIMA layout
  // would throw on its missing coefficient table.
  const [mode, setMode] = usePersistedPanelState<Mode>("timeseries", "mode", "arima");
  const [valueCol, setValueCol] = usePersistedPanelState("timeseries", "valueCol", numCols[0] ?? "");
  const [timeCol, setTimeCol] = usePersistedPanelState("timeseries", "timeCol", "");
  // ARIMA params
  const [p, setP] = usePersistedPanelState("timeseries", "p", 1);
  const [d, setD] = usePersistedPanelState("timeseries", "d", 1);
  const [q, setQ] = usePersistedPanelState("timeseries", "q", 1);
  const [P, setPP] = usePersistedPanelState("timeseries", "P", 0);
  const [D, setDD] = usePersistedPanelState("timeseries", "D", 0);
  const [Q, setQQ] = usePersistedPanelState("timeseries", "Q", 0);
  const [s, setS] = usePersistedPanelState("timeseries", "s", 0);
  const [auto, setAuto] = usePersistedPanelState("timeseries", "auto", false);
  const [steps, setSteps] = usePersistedPanelState("timeseries", "steps", 12);
  // decompose
  const [period, setPeriod] = usePersistedPanelState("timeseries", "period", 12);
  const [method, setMethod] = usePersistedPanelState<"stl" | "classical">("timeseries", "method", "stl");
  // stationarity
  const [nLags, setNLags] = usePersistedPanelState("timeseries", "nLags", 24);

  // One result slot shared by the three analyses (switching mode clears it),
  // so the stamp carries the mode and only the options that mode sends.
  const series = { mode, valueCol, timeCol };
  const runParams = mode === "arima" ? { ...series, p, d, q, P, D, Q, s, auto, steps }
    : mode === "decompose" ? { ...series, period, method }
    : { ...series, nLags };
  const {
    result, setResult, stale, staleReasons: staleWhy,
  } = useStampedResult<Record<string, unknown>>("timeseries", runParams);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!valueCol) { setError("Select a value column."); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      const base = { session_id: sid, value_col: valueCol, time_col: timeCol || undefined };
      let res;
      if (mode === "arima") {
        res = await runArima({ ...base, p, d, q, P, D, Q, s, auto, forecast_steps: steps });
      } else if (mode === "decompose") {
        res = await runDecompose({ ...base, period, method });
      } else {
        res = await runStationarity({ ...base, n_lags: nLags });
      }
      setResult(res.data);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
      const fallback = e instanceof Error ? e.message : "Run failed";
      setError(Array.isArray(detail)
        ? detail.map((m: { msg?: string }) => m.msg ?? String(m)).join(", ")
        : (typeof detail === "string" ? detail : fallback));
    } finally { setLoading(false); }
  };

  const switchMode = (m: Mode) => { setMode(m); setResult(null); setError(null); };

  // ── ARIMA plot: observed + fitted + forecast band ──
  const arimaPlot = () => {
    if (!result?.fitted) return null;
    const fitted = result.fitted as Array<Record<string, unknown>>;
    const forecast = result.forecast as Array<Record<string, unknown>>;
    const obsX = fitted.map((r) => r.x);
    const obsY = fitted.map((r) => r.observed);
    const fitY = fitted.map((r) => r.fitted);
    const fcX = forecast.map((r) => r.x);
    const fcY = forecast.map((r) => r.forecast);
    const fcLo = forecast.map((r) => r.ci_low);
    const fcHi = forecast.map((r) => r.ci_high);
    return (
      <TitledPlot
        plotRefOut={mainRef}
        storageKey={`ts:arima:${asString(result.value_col) ?? ""}`}
        data={[
          { type: "scatter", mode: "lines", x: obsX, y: obsY, line: { color: "#374151", width: 1.5 }, name: "Observed" },
          { type: "scatter", mode: "lines", x: obsX, y: fitY, line: { color: pal[0], width: 1.5, dash: "dot" as const }, name: "Fitted" },
          { type: "scatter", mode: "lines", x: [...fcX, ...fcX.slice().reverse()], y: [...fcHi, ...fcLo.slice().reverse()], fill: "toself", fillcolor: `${pal[1] ?? "#6366f1"}22`, line: { color: "transparent" }, name: "95% CI", hoverinfo: "skip" as const },
          { type: "scatter", mode: "lines+markers", x: fcX, y: fcY, line: { color: pal[1] ?? "#6366f1", width: 2 }, marker: { size: 4 }, name: "Forecast" },
        ]}
        layout={{
          ...baseLayout,
          xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid },
          yaxis: { ...(baseLayout.yaxis as object), showgrid: showGrid },
          legend: { orientation: "h", y: -0.2, font: { size: 10 } },
          margin: { t: 36, r: 20, b: 50, l: 60 },
        }}
        config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        defaultTitle={`ARIMA${JSON.stringify(result.order)}×${JSON.stringify(result.seasonal_order)} — ${asString(result.value_col) ?? ""}`}
        defaultSubtitle=""
        defaultXAxis={timeCol || "t"}
        defaultYAxis={asString(result.value_col)} />
    );
  };

  // ── Decompose plot: 4 stacked panels ──
  const decompPlot = () => {
    if (!result?.observed) return null;
    const x = result.x;
    const rows: [string, number[], string][] = [
      ["Observed", result.observed as number[], "#374151"],
      ["Trend", result.trend as number[], pal[0]],
      ["Seasonal", result.seasonal as number[], pal[1] ?? "#6366f1"],
      ["Residual", result.resid as number[], "#9ca3af"],
    ];
    return (
      <TitledPlot
        plotRefOut={decompRef}
        storageKey={`ts:decompose:${result.value_col ?? ""}`}
        data={rows.map(([name, y, color], i) => ({
          type: "scatter", mode: i === 3 ? "markers" : "lines",
          x, y, name, line: { color, width: 1.4 }, marker: { color, size: 3 },
          xaxis: i === 0 ? "x" : `x${i + 1}`, yaxis: i === 0 ? "y" : `y${i + 1}`,
        }))}
        layout={{
          ...baseLayout, grid: { rows: 4, columns: 1, pattern: "independent" as const },
          showlegend: false,
          height: 560, margin: { t: 36, r: 20, b: 40, l: 60 },
          ...Object.fromEntries(rows.flatMap((r, i) => {
            const suf = i === 0 ? "" : String(i + 1);
            return [
              [`xaxis${suf}`, { showgrid: showGrid, gridcolor: "#eef2f7" }],
              [`yaxis${suf}`, { showgrid: showGrid, gridcolor: "#eef2f7", title: { text: r[0], font: { size: 9 } } }],
            ];
          })),
        }}
        config={{ responsive: true, displaylogo: false, displayModeBar: false }}
        defaultTitle={`${asString(result.method)?.toUpperCase() ?? ""} decomposition (period ${result.period})`}
        defaultSubtitle=""
        defaultXAxis=""
        defaultYAxis="" />
    );
  };

  // ── Stationarity: ACF + PACF stem plots ──
  const stem = (data: StemPoint[], title: string, ref: React.MutableRefObject<PlotCaptureHandle | null>, exportName: string) => (
    <TitledPlot
      plotRefOut={ref}
      storageKey={`ts:stem:${exportName}`}
      data={[
        // CI band
        { type: "scatter", mode: "lines", x: data.map((d) => d.lag), y: data.map((d) => d.ci_high), line: { color: "transparent" }, hoverinfo: "skip" as const, showlegend: false },
        { type: "scatter", mode: "lines", x: data.map((d) => d.lag), y: data.map((d) => d.ci_low), fill: "tonexty", fillcolor: "rgba(148,163,184,0.18)", line: { color: "transparent" }, hoverinfo: "skip" as const, showlegend: false },
        { type: "bar", x: data.map((d) => d.lag), y: data.map((d) => d.value), marker: { color: pal[0] }, width: 0.15, name: title, hovertemplate: "lag %{x}<br>%{y:.3f}<extra></extra>" },
      ]}
      layout={{
        ...baseLayout,
        xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid },
        yaxis: { ...(baseLayout.yaxis as object), showgrid: showGrid, range: [-1.05, 1.05] },
        showlegend: false, margin: { t: 36, r: 20, b: 40, l: 50 },
      }}
      config={{ responsive: true, displaylogo: false, displayModeBar: false }}
      defaultTitle={title}
      defaultSubtitle=""
      defaultXAxis="Lag"
      defaultYAxis="" />
  );

  return (
    <div className="p-4 space-y-3">
      <ThreeCol
        storageKey="TimeSeriesPanel"
        left={
          <div className="panel space-y-2">
            <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
              Time Series
              <Tip wide text="ARIMA/SARIMA model with forecast, classical/STL seasonal decomposition, and stationarity testing (ADF + KPSS) with ACF/PACF. Order the series with an optional time column (datetime or numeric)." />
            </h3>
            <div className="flex rounded-lg overflow-hidden border border-gray-300">
              {(["arima", "decompose", "stationarity"] as const).map((m) => (
                <button key={m} onClick={() => switchMode(m)}
                  className={`flex-1 px-1.5 py-1.5 text-[11px] font-medium transition-colors ${mode === m ? "bg-indigo-600 text-white" : "text-gray-500 hover:bg-gray-50"}`}>
                  {m === "arima" ? "ARIMA" : m === "decompose" ? "Decompose" : "Stationarity"}
                </button>
              ))}
            </div>

            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-500 font-medium">Value (series)</span>
              <select value={valueCol} onChange={(e) => { setValueCol(e.target.value); setResult(null); }}
                className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400">
                <option value="">— select —</option>
                {numCols.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-500 font-medium">Time / order (optional)</span>
              <select value={timeCol} onChange={(e) => setTimeCol(e.target.value)}
                className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:border-indigo-400">
                <option value="">— row order —</option>
                {columns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
            </label>

            {mode === "arima" && (
              <>
                <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
                  <input type="checkbox" className="accent-indigo-500" checked={auto} onChange={(e) => setAuto(e.target.checked)} />
                  Auto order (AIC grid)
                  <Tip text="Searches p,d,q ∈ {0,1,2}×{0,1}×{0,1,2}, picks minimum AIC. Seasonal order is taken from the s field below." />
                </label>
                {!auto && (
                  <div className="grid grid-cols-3 gap-1.5">
                    {([["p", p, setP], ["d", d, setD], ["q", q, setQ]] as const).map(([lbl, val, set]) => (
                      <label key={lbl} className="flex flex-col gap-0.5">
                        <span className="text-[10px] text-gray-500 text-center">{lbl}</span>
                        <input type="number" min={0} max={5} value={val} onChange={(e) => set(Number(e.target.value))}
                          className="text-xs border border-gray-300 rounded px-2 py-1 text-center focus:outline-none focus:border-indigo-400" />
                      </label>
                    ))}
                  </div>
                )}
                <p className="text-[10px] text-gray-400">Seasonal (set s &gt; 1 to enable)</p>
                <div className="grid grid-cols-4 gap-1.5">
                  {([["P", P, setPP], ["D", D, setDD], ["Q", Q, setQQ], ["s", s, setS]] as const).map(([lbl, val, set]) => (
                    <label key={lbl} className="flex flex-col gap-0.5">
                      <span className="text-[10px] text-gray-500 text-center">{lbl}</span>
                      <input type="number" min={0} max={lbl === "s" ? 60 : 3} value={val} onChange={(e) => set(Number(e.target.value))}
                        className="text-xs border border-gray-300 rounded px-2 py-1 text-center focus:outline-none focus:border-indigo-400" />
                    </label>
                  ))}
                </div>
                <label className="flex flex-col gap-0.5">
                  <span className="text-[10px] text-gray-500">Forecast steps</span>
                  <input type="number" min={1} max={120} value={steps} onChange={(e) => setSteps(Number(e.target.value))}
                    className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" />
                </label>
              </>
            )}

            {mode === "decompose" && (
              <>
                <label className="flex flex-col gap-0.5">
                  <span className="text-[10px] text-gray-500">Period (obs / cycle)</span>
                  <input type="number" min={2} max={365} value={period} onChange={(e) => setPeriod(Number(e.target.value))}
                    className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" />
                </label>
                <div className="flex rounded-lg overflow-hidden border border-gray-300">
                  {(["stl", "classical"] as const).map((m) => (
                    <button key={m} onClick={() => setMethod(m)}
                      className={`flex-1 px-2 py-1 text-[11px] ${method === m ? "bg-indigo-600 text-white" : "text-gray-500 hover:bg-gray-50"}`}>
                      {m === "stl" ? "STL" : "Classical"}
                    </button>
                  ))}
                </div>
              </>
            )}

            {mode === "stationarity" && (
              <label className="flex flex-col gap-0.5">
                <span className="text-[10px] text-gray-500">ACF/PACF lags</span>
                <input type="number" min={1} max={60} value={nLags} onChange={(e) => setNLags(Number(e.target.value))}
                  className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-indigo-400" />
              </label>
            )}

            <button onClick={run} disabled={loading}
              className="w-full px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors">
              {loading ? "Running…" : mode === "arima" ? "Fit ARIMA" : mode === "decompose" ? "Decompose" : "Test stationarity"}
            </button>
            {error && <p className="text-xs text-red-500">{error}</p>}
          </div>
        }
        middle={
          result ? (
            <div className="space-y-3">
              {stale && (
                <StaleResultNotice
                  reasons={staleWhy}
                  onRecompute={run}
                  busy={loading}
                  what="This time-series analysis"
                />
              )}
              <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
              {mode === "arima" && arimaPlot()}
              {mode === "decompose" && decompPlot()}
              {mode === "stationarity" && (
                <>
                  {stem((result.acf as StemPoint[]) ?? [], "ACF (autocorrelation)", acfRef, "ACF")}
                  {stem((result.pacf as StemPoint[]) ?? [], "PACF (partial autocorrelation)", pacfRef, "PACF")}
                </>
              )}
              </StaleGuard>
            </div>
          ) : (
            <div className="flex items-center justify-center h-[360px] border border-dashed border-gray-200 rounded-lg text-xs text-gray-400">
              Configure and run a time-series analysis
            </div>
          )
        }
        right={
          result ? (
            <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
              {mode === "arima" && (
                <>
                  <div className="panel space-y-2">
                    <h4 className="text-sm font-semibold text-gray-800">Fit</h4>
                    <div className="grid grid-cols-2 gap-1.5">
                      {([["Order", JSON.stringify(result.order)], ["Seasonal", JSON.stringify(result.seasonal_order)],
                        ["AIC", result.aic], ["BIC", result.bic],
                        ["Ljung-Box p", result.ljung_box_p ?? "—"], [<i>n</i>, result.n]] as [ReactNode, ReactNode][]).map(([k, v], i) => (
                        <div key={i} className="bg-gray-50 border border-gray-200 rounded p-1.5 text-center">
                          <p className="text-[9px] text-gray-400">{k}</p>
                          <p className="font-semibold text-gray-800 text-xs font-mono">{v}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="panel space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-semibold text-gray-700">Coefficients</h4>
                      <ResultExporter title={`ARIMA_${String(result.value_col)}`} headers={["Term", "Estimate", "SE", "p"]}
                        rows={(result.coefficients as Coefficient[]).map((c) => [c.term, c.estimate, c.se, c.p])} />
                    </div>
                    <div className="overflow-auto rounded-lg border border-gray-200 max-h-60">
                      <table className="w-full text-[11px] border-collapse">
                        <thead className="sticky top-0 bg-gray-50 border-b border-gray-200 text-gray-500">
                          <tr><th className="text-left px-1.5 py-1">Term</th><th className="text-right px-1.5 py-1">Est</th><th className="text-right px-1.5 py-1">SE</th><th className="text-right px-1.5 py-1"><i>p</i></th></tr>
                        </thead>
                        <tbody>
                          {(result.coefficients as Coefficient[]).map((c) => (
                            <tr key={c.term} className="border-b border-gray-100">
                              <td className="px-1.5 py-1 font-mono text-gray-700">{c.term}</td>
                              <td className="px-1.5 py-1 font-mono text-right">{c.estimate?.toFixed(3)}</td>
                              <td className="px-1.5 py-1 font-mono text-right text-gray-500">{c.se != null ? c.se.toFixed(3) : "—"}</td>
                              <td className="px-1.5 py-1 font-mono text-right">{fmtP(c.p)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}

              {mode === "decompose" && (
                <div className="panel space-y-2">
                  <h4 className="text-sm font-semibold text-gray-800">Strength</h4>
                  <div className="grid grid-cols-2 gap-1.5">
                    <div className="bg-gray-50 border border-gray-200 rounded p-2 text-center">
                      <p className="text-[10px] text-gray-400">Trend</p>
                      <p className="font-semibold text-indigo-700 text-sm font-mono">{asNumber(result.strength_trend)?.toFixed(2) ?? "—"}</p>
                    </div>
                    <div className="bg-gray-50 border border-gray-200 rounded p-2 text-center">
                      <p className="text-[10px] text-gray-400">Seasonal</p>
                      <p className="font-semibold text-indigo-700 text-sm font-mono">{asNumber(result.strength_seasonal)?.toFixed(2) ?? "—"}</p>
                    </div>
                  </div>
                </div>
              )}

              {mode === "stationarity" && (
                <div className="panel space-y-2">
                  <h4 className="text-sm font-semibold text-gray-800">Stationarity</h4>
                  <div className="space-y-1 text-xs">
                    <div className={`flex justify-between px-2 py-1 rounded ${result.adf_stationary ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
                      <span>ADF (H₀: unit root)</span>
                      <span className="font-mono"><i>p</i> = {fmtP(asNumber(result.adf_p))}</span>
                    </div>
                    <div className={`flex justify-between px-2 py-1 rounded ${result.kpss_stationary ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
                      <span>KPSS (H₀: stationary)</span>
                      <span className="font-mono">p = {(() => {
                        const kpssP = asNumber(result.kpss_p);
                        return kpssP == null ? "—" : kpssP < 0.01 ? "<0.01" : kpssP.toFixed(3);
                      })()}</span>
                    </div>
                  </div>
                </div>
              )}

              {typeof result.interpretation === "string" && (
                <div className="bg-indigo-50 border border-indigo-200 rounded-xl px-3 py-2 text-xs text-indigo-900 leading-relaxed">
                  {result.interpretation}
                </div>
              )}
            </StaleGuard>
          ) : null
        }
      />
    </div>
  );
}
