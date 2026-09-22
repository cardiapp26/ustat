import { useState, useRef } from "react";
import { useStore, isNumericKind, isCategoricalKind, type Session } from "../store";
import { usePlotLayout, usePalette } from "../plotStyle";
import { runBayesian } from "../api";
import TitledPlot from "./TitledPlot";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { useStampedResult } from "../hooks/useStampedResult";
import { describeStale } from "../lib/resultStamp";
import StaleResultNotice from "./StaleResultNotice";
import StaleGuard from "./StaleGuard";
import type { PlotData, PlotCaptureHandle } from "../lib/plotTypes";

type AnalysisType = "ttest_one" | "ttest_ind" | "ttest_paired" | "correlation" | "regression";

interface PlotCoord {
  x: number;
  prior: number;
  posterior: number;
}
interface BayesianResult {
  analysis: string;
  n: number;
  bf10: number;
  bf01: number;
  interpretation: string;
  statistic_label: string;
  statistic_value: number;
  df?: number;
  effect_size_label: string;
  effect_size_value: number;
  plot_coords?: PlotCoord[];
  r_code?: string;
}

export default function BayesianPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <BayesianPanelBody session={session} />;
}

function BayesianPanelBody({ session }: { session: Session }) {
  const showGrid = useStore((s) => s.showGrid);
  const baseLayout = usePlotLayout();
  const pal = usePalette();
  const plotRef = useRef<PlotCaptureHandle | null>(null);

  const numCols = session.columns.filter((c) => isNumericKind(c.kind) && !c.analysis_excluded).map((c) => c.name);
  const catCols = session.columns.filter((c) => isCategoricalKind(c.kind) && !c.analysis_excluded).map((c) => c.name);


  // States
  const [analysisType, setAnalysisType] = usePersistedPanelState<AnalysisType>("bayesian", "analysisType", "ttest_one");
  const [outcome, setOutcome] = usePersistedPanelState<string>("bayesian", "outcome", numCols[0] ?? "");
  const [predictor, setPredictor] = usePersistedPanelState<string>("bayesian", "predictor", numCols[1] ?? numCols[0] ?? "");
  const [predictors, setPredictors] = usePersistedPanelState<string[]>("bayesian", "predictors", []);
  // Persisted with the rest: the result now survives a tab switch, and a test
  // value reset to 0 beside it would date a one-sample result for nothing.
  const [mu, setMu] = usePersistedPanelState<number>("bayesian", "mu", 0.0);
  const [imputation, setImputation] = usePersistedPanelState<string>("bayesian", "imputation", "listwise");

  // The request less the session id, which is also what the result is stamped
  // with: the predictor fields an analysis type does not send cannot date it.
  const runParams: Record<string, unknown> = {
    analysis_type: analysisType,
    outcome,
    imputation,
    ...(analysisType === "ttest_one" ? { mu } : {}),
    ...(analysisType === "ttest_ind" || analysisType === "ttest_paired" || analysisType === "correlation"
      ? { predictor } : {}),
    ...(analysisType === "regression" ? { predictors } : {}),
  };
  const {
    result, setResult, stale, staleReasons: staleWhy,
  } = useStampedResult<BayesianResult>("bayesian", runParams);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await runBayesian({ session_id: session.session_id, ...runParams });
      setResult(res.data as BayesianResult);
      
      // Log session action
      useStore.getState().logAction("bayesian_stats", {
        analysis_type: analysisType,
        outcome,
        predictor: analysisType !== "regression" ? predictor : undefined,
        predictors: analysisType === "regression" ? predictors : undefined,
        mu: analysisType === "ttest_one" ? mu : undefined
      });
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Bayesian calculation failed.");
    } finally {
      setLoading(false);
    }
  };

  // Interpretation styling helper
  const interpretationStyle = (bf10: number) => {
    if (bf10 >= 10.0) return "bg-emerald-50 border-emerald-200 text-emerald-800";
    if (bf10 <= 0.1) return "bg-blue-50 border-blue-200 text-blue-800";
    return "bg-amber-50 border-amber-200 text-amber-800";
  };

  // Render Prior vs Posterior plot
  const priorPosteriorPlot = () => {
    if (!result?.plot_coords || result.plot_coords.length === 0) return null;
    const coords: PlotCoord[] = result.plot_coords;

    const data: PlotData[] = [
      {
        type: "scatter",
        mode: "lines",
        x: coords.map((c) => c.x),
        y: coords.map((c) => c.prior),
        line: { color: "#94a3b8", width: 1.8, dash: "dash" },
        name: "Prior (Cauchy)"
      },
      {
        type: "scatter",
        mode: "lines",
        x: coords.map((c) => c.x),
        y: coords.map((c) => c.posterior),
        line: { color: pal[0], width: 2.2 },
        name: "Posterior"
      }
    ];

    // JASP Savage-Dickey visual: find density at effect size = 0
    const zeroIndex = coords.reduce((bestIdx: number, curr: PlotCoord, currIdx: number) => {
      return Math.abs(curr.x) < Math.abs(coords[bestIdx].x) ? currIdx : bestIdx;
    }, 0);

    const zeroCoord = coords[zeroIndex];
    if (zeroCoord && Math.abs(zeroCoord.x) < 0.05) {
      data.push({
        type: "scatter",
        mode: "markers",
        x: [0, 0],
        y: [zeroCoord.prior, zeroCoord.posterior],
        marker: {
          color: ["#475569", "#dc2626"],
          size: [8, 8],
          symbol: ["circle", "circle"]
        },
        showlegend: false,
        hoverinfo: "none"
      });
    }

    return (
      <div className="panel w-full">
        <TitledPlot
          plotRefOut={plotRef}
          storageKey="bayesian:prior-posterior"
          data={data}
          layout={{
            ...baseLayout,
            xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid },
            yaxis: { ...(baseLayout.yaxis as object), showgrid: showGrid },
            margin: { t: 36, r: 24, b: 44, l: 60 }
          }}
          config={{ responsive: true, displaylogo: false, displayModeBar: false }}
          defaultTitle="Prior and Posterior Distribution of Effect Size"
          defaultSubtitle=""
          defaultXAxis="Effect Size (δ / ρ)"
          defaultYAxis="Probability Density"
        />
      </div>
    );
  };

  return (
    <div className="flex gap-4 p-4 max-w-7xl mx-auto">
      {/* Sidebar Controls */}
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
            ⚖️ Bayesian Statistics
          </h3>
          <p className="text-xs text-gray-400">
            Compare hypotheses by calculating the Bayes Factor directly using JZS priors.
          </p>

          <div className="space-y-1.5 pt-1">
            <label className="text-xs font-medium text-gray-600 block">Analysis Type</label>
            <select
              className="select text-xs w-full"
              value={analysisType}
              onChange={(e) => {
                setAnalysisType(e.target.value as AnalysisType);
                setResult(null);
              }}
            >
              <option value="ttest_one">Bayesian One-Sample t-test</option>
              <option value="ttest_ind">Bayesian Independent t-test</option>
              <option value="ttest_paired">Bayesian Paired t-test</option>
              <option value="correlation">Bayesian Correlation (Pearson)</option>
              <option value="regression">Bayesian Multiple Regression</option>
            </select>
          </div>

          <div className="space-y-3 border-t border-gray-100 pt-3">
            {/* Outcome Selection */}
            <div>
              <label className="text-xs font-medium text-gray-600 block mb-1">
                {analysisType === "regression" ? "Outcome (numeric)" : "Variable / Outcome"}
              </label>
              <select
                className="select text-xs w-full"
                value={outcome}
                onChange={(e) => setOutcome(e.target.value)}
              >
                {numCols.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </div>

            {/* Test value for one-sample */}
            {analysisType === "ttest_one" && (
              <div>
                <label className="text-xs font-medium text-gray-600 block mb-1">Test Value (mu)</label>
                <input
                  type="number"
                  step="any"
                  className="select text-xs w-full"
                  value={mu}
                  onChange={(e) => setMu(parseFloat(e.target.value) || 0)}
                />
              </div>
            )}

            {/* Predictor for independent t-test (categorical) */}
            {analysisType === "ttest_ind" && (
              <div>
                <label className="text-xs font-medium text-gray-600 block mb-1">Grouping Variable</label>
                <select
                  className="select text-xs w-full"
                  value={predictor}
                  onChange={(e) => setPredictor(e.target.value)}
                >
                  {[...catCols, ...numCols].map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </div>
            )}

            {/* Predictor for paired t-test (numeric) */}
            {analysisType === "ttest_paired" && (
              <div>
                <label className="text-xs font-medium text-gray-600 block mb-1">Paired Variable (Time 2)</label>
                <select
                  className="select text-xs w-full"
                  value={predictor}
                  onChange={(e) => setPredictor(e.target.value)}
                >
                  {numCols.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </div>
            )}

            {/* Predictor for correlation (numeric) */}
            {analysisType === "correlation" && (
              <div>
                <label className="text-xs font-medium text-gray-600 block mb-1">Second Variable</label>
                <select
                  className="select text-xs w-full"
                  value={predictor}
                  onChange={(e) => setPredictor(e.target.value)}
                >
                  {numCols.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </div>
            )}

            {/* Predictors for multiple regression */}
            {analysisType === "regression" && (
              <div>
                <label className="text-xs font-medium text-gray-600 block mb-1">
                  Predictors (numeric)
                </label>
                <select
                  multiple
                  className="select text-xs w-full h-24 font-mono"
                  value={predictors}
                  onChange={(e) => setPredictors(Array.from(e.target.selectedOptions, (o) => o.value))}
                >
                  {numCols.filter((c) => c !== outcome).map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
                <p className="text-[9px] text-gray-400 mt-0.5">Hold Ctrl/Cmd to select multiple.</p>
              </div>
            )}
          </div>

          <div className="space-y-1.5 pt-1">
            <label className="text-xs font-medium text-gray-600 block">Missing Values</label>
            <select
              className="select text-xs w-full"
              value={imputation}
              onChange={(e) => setImputation(e.target.value)}
            >
              <option value="listwise">Listwise deletion</option>
              <option value="mean">Mean substitution</option>
              <option value="median">Median substitution</option>
            </select>
          </div>

          <button
            className="btn-primary w-full mt-3 py-1.5"
            onClick={run}
            disabled={loading || (analysisType === "regression" && predictors.length === 0)}
          >
            {loading ? "Computing..." : "Compute Bayes Factor"}
          </button>
          {error && <p className="text-red-400 text-xs mt-1">{error}</p>}
        </div>

        <div className="panel bg-indigo-50 border-indigo-200 space-y-2">
          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">How to Read</p>
          <p className="text-xs text-indigo-800 leading-relaxed font-mono">
            <strong>BF₁₀</strong>: Relative likelihood of Alternative (H₁) vs Null (H₀). E.g. BF₁₀ = 10 means H₁ is 10 times more likely than H₀.
          </p>
          <p className="text-xs text-indigo-800 leading-relaxed font-mono">
            <strong>BF₀₁</strong>: Relative likelihood of Null (H₀) vs Alternative (H₁). E.g. BF₀₁ = 5 means H₀ is 5 times more likely than H₁.
          </p>
        </div>
      </div>

      {/* Main Results Panel */}
      <div className="flex-1 min-w-0 space-y-4">
        {result && stale && (
          <StaleResultNotice reasons={staleWhy} onRecompute={run} busy={loading} what={`This ${result.analysis}`} />
        )}
        {result ? (
          // The prior/posterior plot's exporter closes while the result is out
          // of date.
          <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
          <div className="space-y-4">
            {/* Bayes Factor primary card */}
            <div className="panel flex flex-col md:flex-row gap-6 items-center justify-between">
              <div className="space-y-3 flex-1">
                <div className="flex items-baseline gap-2">
                  <h4 className="font-bold text-gray-900 text-base">{result.analysis}</h4>
                  <span className="text-xs text-gray-400 font-mono"><i>n</i> = {result.n}</span>
                </div>

                <div className="grid grid-cols-2 gap-4 max-w-sm">
                  <div className="bg-indigo-50 border border-indigo-100 rounded-xl p-3 text-center">
                    <p className="text-[10px] font-semibold text-indigo-900 uppercase tracking-wide">
                      Bayes Factor BF₁₀
                    </p>
                    <p className="text-2xl font-bold font-mono text-indigo-700 mt-1">
                      {typeof result.bf10 === "number" ? result.bf10.toFixed(4) : result.bf10}
                    </p>
                  </div>
                  <div className="bg-gray-50 border border-gray-200 rounded-xl p-3 text-center">
                    <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">
                      Bayes Factor BF₀₁
                    </p>
                    <p className="text-2xl font-bold font-mono text-gray-600 mt-1">
                      {typeof result.bf01 === "number" ? result.bf01.toFixed(4) : result.bf01}
                    </p>
                  </div>
                </div>

                <div className={`border rounded-xl p-3 text-xs font-semibold ${interpretationStyle(result.bf10)}`}>
                  💡 Result: {result.interpretation}
                </div>
              </div>

              {/* Statistics details */}
              <div className="w-full md:w-56 p-4 bg-gray-50 border rounded-2xl space-y-2 text-xs">
                <h5 className="font-bold text-gray-700 mb-2 border-b pb-1">Frequentist Equivalents</h5>
                <div className="flex justify-between py-1 border-b border-dashed">
                  <span className="text-gray-400">{result.statistic_label} value</span>
                  <span className="font-mono font-semibold text-gray-700">{result.statistic_value.toFixed(4)}</span>
                </div>
                {result.df !== undefined && (
                  <div className="flex justify-between py-1 border-b border-dashed">
                    <span className="text-gray-400">Degrees of freedom</span>
                    <span className="font-mono font-semibold text-gray-700">{result.df}</span>
                  </div>
                )}
                <div className="flex justify-between py-1 border-b border-dashed">
                  <span className="text-gray-400">Sample size (n)</span>
                  <span className="font-mono font-semibold text-gray-700">{result.n}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-gray-400">Effect size ({result.effect_size_label})</span>
                  <span className="font-mono font-semibold text-gray-700">{result.effect_size_value.toFixed(4)}</span>
                </div>
              </div>
            </div>

            {/* Prior vs Posterior chart */}
            {result.plot_coords && result.plot_coords.length > 0 && priorPosteriorPlot()}

            {/* R code replication */}
            {result.r_code && (
              <details className="panel text-xs cursor-pointer">
                <summary className="text-gray-400 font-semibold hover:text-indigo-600">
                  Equivalent R Replication Code
                </summary>
                <pre className="mt-2 p-3 bg-gray-50 rounded-lg font-mono text-[10px] text-gray-600 whitespace-pre-wrap select-all cursor-text leading-snug">
                  {result.r_code}
                </pre>
              </details>
            )}
          </div>
          </StaleGuard>
        ) : (
          <div className="panel text-center text-gray-400 py-24 bg-white border border-dashed border-gray-200">
            <p className="text-3xl mb-3">⚖️</p>
            <p className="font-semibold text-gray-600">Run Bayesian Analysis</p>
            <p className="text-xs mt-1 max-w-sm mx-auto">
              Select an analysis type and select variables to compute the JZS Bayes Factor. Unlike frequentist p-values, Bayesian statistics allow you to quantify evidence for *both* the alternative hypothesis (H₁) and the null hypothesis (H₀).
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
