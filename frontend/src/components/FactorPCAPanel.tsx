import { useState, useRef } from "react";
import { useStore, isNumericKind, type Session } from "../store";
import { usePlotLayout, usePalette } from "../plotStyle";
import { runFactorPCA } from "../api";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { useStampedResult } from "../hooks/useStampedResult";
import { describeStale } from "../lib/resultStamp";
import TitledPlot from "./TitledPlot";
import ResultExporter from "./ResultExporter";
import StaleResultNotice from "./StaleResultNotice";
import StaleGuard from "./StaleGuard";
import { fmtP } from "../lib/format";
import type { PlotData, PlotCaptureHandle } from "../lib/plotTypes";

type ActiveResultTab = "suitability" | "loadings" | "scree" | "biplot";

type LoadingRow = Record<string, number | string> & {
  variable: string;
  h2: number;
  u2: number;
};

interface ScreeCoord {
  component: number;
  eigenvalue: number;
}

interface BiplotPoint {
  variable: string;
  x: number;
  y: number;
}

interface VarianceRow {
  component: number;
  eigenvalue: number;
  pct_variance: number;
  cum_variance: number;
}

interface FactorPCAResult {
  factors: string[];
  rotation_method: string;
  n_factors: number;
  loadings: LoadingRow[];
  scree_coords: ScreeCoord[];
  biplot: BiplotPoint[];
  variance_explained: VarianceRow[];
  export_rows?: string[][];
  r_code?: string;
  suitability: {
    bartlett_chi2: number;
    bartlett_df: number;
    bartlett_p: number;
    overall_kmo: number;
    kmo_rating: string;
  };
}

export default function FactorPCAPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <FactorPCAPanelBody session={session} />;
}

function FactorPCAPanelBody({ session }: { session: Session }) {
  const showGrid = useStore((s) => s.showGrid);
  const baseLayout = usePlotLayout();
  const pal = usePalette();
  const screeRef = useRef<PlotCaptureHandle | null>(null);
  const biplotRef = useRef<PlotCaptureHandle | null>(null);

  const numCols = session.columns.filter((c) => isNumericKind(c.kind)).map((c) => c.name);

  // Persisted with the result: a fit restored after a tab switch has to come
  // back with the settings it was run under, or it reads as stale and its
  // Recompute button would fit whatever the defaults happen to be.
  const [items, setItems] = usePersistedPanelState<string[]>("factor_pca", "items", []);
  const [extraction, setExtraction] = usePersistedPanelState<"pca" | "efa">("factor_pca", "extraction", "pca");
  const [rotation, setRotation] = usePersistedPanelState<"none" | "varimax" | "promax">("factor_pca", "rotation", "varimax");
  const [nFactorsMode, setNFactorsMode] = usePersistedPanelState<"auto" | "manual">("factor_pca", "nFactorsMode", "auto");
  const [nFactors, setNFactors] = usePersistedPanelState<number>("factor_pca", "nFactors", 1);
  const [imputation, setImputation] = usePersistedPanelState<"listwise" | "mean" | "median">("factor_pca", "imputation", "listwise");

  // What the request is built from. The loading cutoff, sort order and open
  // tab are display-only and stay out of it.
  const runParams = {
    items, extraction, rotation, imputation,
    n_factors: nFactorsMode === "manual" ? nFactors : null,
  };
  const {
    result, setResult, stale, staleReasons: staleWhy,
  } = useStampedResult<FactorPCAResult>("factor_pca", runParams);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Custom display settings
  const [resultTab, setResultTab] = useState<ActiveResultTab>("suitability");
  const [cutoff, setCutoff] = useState<number>(0.3);
  const [sortLoadings, setSortLoadings] = useState(false);

  const run = async () => {
    if (items.length < 3) {
      setError("Please select at least 3 numeric variables.");
      return;
    }
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await runFactorPCA({
        session_id: session.session_id,
        items,
        extraction,
        rotation,
        n_factors: nFactorsMode === "manual" ? nFactors : null,
        imputation
      });
      setResult(res.data as FactorPCAResult);
      // Track this action for R replication
      useStore.getState().logAction("factor_pca", {
        variables: items,
        extraction,
        rotation,
        n_factors: nFactorsMode === "manual" ? nFactors : "auto",
        imputation
      });
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Factor analysis computation failed.");
    } finally {
      setLoading(false);
    }
  };

  // Helper to color KMO values
  const kmoColor = (val: number) => {
    if (val >= 0.8) return "text-emerald-700 bg-emerald-50";
    if (val >= 0.6) return "text-indigo-700 bg-indigo-50";
    if (val >= 0.5) return "text-amber-700 bg-amber-50";
    return "text-red-700 bg-red-50";
  };

  // Format table loading values
  const fmtLoading = (val: number) => {
    if (Math.abs(val) < cutoff) return "";
    return val.toFixed(3);
  };

  // Return sorted loadings if toggled
  const getSortedLoadings = () => {
    if (!result?.loadings) return [];
    if (!sortLoadings) return result.loadings;
    
    // Sort by primary loading factor (first column factor with absolute max loading)
    return [...result.loadings].sort((a, b) => {
      const maxA = Math.max(...result.factors.map((f) => Math.abs(Number(a[f]))));
      const maxB = Math.max(...result.factors.map((f) => Math.abs(Number(b[f]))));
      return maxB - maxA;
    });
  };

  // Scree Plot JSX
  const screePlot = () => {
    if (!result?.scree_coords) return null;
    const coords = result.scree_coords;
    const data: PlotData[] = [
      {
        type: "scatter",
        mode: "lines+markers",
        x: coords.map((c) => c.component),
        y: coords.map((c) => c.eigenvalue),
        marker: { color: pal[0], size: 8 },
        line: { color: pal[0], width: 2 },
        name: "Eigenvalue"
      },
      // Draw standard line at eigenvalue = 1 (Kaiser Criterion)
      {
        type: "scatter",
        mode: "lines",
        x: [0.5, coords.length + 0.5],
        y: [1.0, 1.0],
        line: { color: "#dc2626", width: 1, dash: "dash" },
        name: "Kaiser limit (1.0)",
        showlegend: false
      }
    ];

    return (
      <div className="panel w-full">
        <TitledPlot
          plotRefOut={screeRef}
          storageKey="factorpca:scree"
          data={data}
          layout={{
            ...baseLayout,
            xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid, title: { text: "Component / Factor" }, dtick: 1 },
            yaxis: { ...(baseLayout.yaxis as object), showgrid: showGrid, title: { text: "Eigenvalue" } },
            margin: { t: 36, r: 24, b: 44, l: 60 }
          }}
          config={{ responsive: true, displaylogo: false, displayModeBar: false }}
          defaultTitle="Scree Plot — Component Eigenvalues"
          defaultSubtitle=""
          defaultXAxis="Component / Factor"
          defaultYAxis="Eigenvalue"
        />
      </div>
    );
  };

  // Loadings Biplot JSX
  const loadingsBiplot = () => {
    if (!result?.biplot) return null;
    const points = result.biplot;
    
    // Draw scatter points for variable coordinates
    const scatterData: PlotData = {
      type: "scatter",
      mode: "markers+text",
      x: points.map((p) => p.x),
      y: points.map((p) => p.y),
      text: points.map((p) => p.variable),
      textposition: "top right",
      marker: { color: "#4f46e5", size: 8 },
      name: "Variables"
    };

    // Draw origin vectors/arrows using plotly layout shapes
    const shapes: PlotData[] = points.map((p) => ({
      type: "line",
      x0: 0,
      y0: 0,
      x1: p.x,
      y1: p.y,
      line: {
        color: "#818cf8",
        width: 1.5
      }
    }));

    // Add dashed lines at origin (0,0) axes
    shapes.push({
      type: "line", x0: -1.1, y0: 0, x1: 1.1, y1: 0,
      line: { color: "#cbd5e1", width: 1, dash: "dash" }
    });
    shapes.push({
      type: "line", x0: 0, y0: -1.1, x1: 0, y1: 1.1,
      line: { color: "#cbd5e1", width: 1, dash: "dash" }
    });

    return (
      <div className="panel w-full">
        <TitledPlot
          plotRefOut={biplotRef}
          storageKey="factorpca:biplot"
          data={[scatterData]}
          layout={{
            ...baseLayout,
            xaxis: { ...(baseLayout.xaxis as object), showgrid: showGrid, title: { text: result.factors[0] }, range: [-1.1, 1.1] },
            yaxis: { ...(baseLayout.yaxis as object), showgrid: showGrid, title: { text: result.factors[1] ?? "PC2" }, range: [-1.1, 1.1] },
            shapes,
            margin: { t: 36, r: 24, b: 44, l: 60 }
          }}
          config={{ responsive: true, displaylogo: false, displayModeBar: false }}
          defaultTitle={`Loadings Plot (${result.factors[0]} vs ${result.factors[1] ?? "PC2"})`}
          defaultSubtitle=""
          defaultXAxis={result.factors[0]}
          defaultYAxis={result.factors[1] ?? "PC2"}
        />
      </div>
    );
  };

  return (
    <div className="flex gap-4 p-4 max-w-7xl mx-auto">
      {/* Left controls panel */}
      <div className="w-72 flex-shrink-0 space-y-4">
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1">
            📊 PCA & Factor Analysis
          </h3>
          <p className="text-xs text-gray-400">
            Extract latent factors or reduce dimensionality. Select 3+ numeric columns.
          </p>

          <select
            multiple
            className="select w-full h-44 font-mono text-xs"
            value={items}
            onChange={(e) => setItems(Array.from(e.target.selectedOptions, (o) => o.value))}
          >
            {numCols.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
          <div className="flex items-center justify-between mt-1 text-[10px]">
            <span className="text-gray-400">Hold Ctrl/Cmd to multi-select.</span>
            <span className={`font-semibold ${items.length < 3 ? "text-amber-600 bg-amber-50" : "text-emerald-600 bg-emerald-50"} px-1.5 py-0.5 rounded`}>
              Selected: {items.length} / 3 min
            </span>
          </div>

          <div className="space-y-1.5 pt-2">
            <label className="text-xs font-medium text-gray-600 block">Extraction</label>
            <select
              className="select text-xs w-full"
              value={extraction}
              onChange={(e) => setExtraction(e.target.value as "pca" | "efa")}
            >
              <option value="pca">PCA (Principal Components)</option>
              <option value="efa">EFA (Principal Axis Factoring)</option>
            </select>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-medium text-gray-600 block">Rotation</label>
            <select
              className="select text-xs w-full"
              value={rotation}
              onChange={(e) => setRotation(e.target.value as "none" | "varimax" | "promax")}
            >
              <option value="none">None (Unrotated)</option>
              <option value="varimax">Varimax (Orthogonal)</option>
              <option value="promax">Promax (Oblique)</option>
            </select>
          </div>

          <div className="space-y-1.5 border-t border-gray-100 pt-2">
            <label className="text-xs font-medium text-gray-600 block">Number of Factors</label>
            <div className="flex gap-2">
              <label className="flex items-center gap-1 text-xs cursor-pointer text-gray-700">
                <input
                  type="radio"
                  name="n_fac_mode"
                  checked={nFactorsMode === "auto"}
                  onChange={() => setNFactorsMode("auto")}
                  className="accent-indigo-600"
                />
                Auto (Eig &gt; 1)
              </label>
              <label className="flex items-center gap-1 text-xs cursor-pointer text-gray-700">
                <input
                  type="radio"
                  name="n_fac_mode"
                  checked={nFactorsMode === "manual"}
                  onChange={() => setNFactorsMode("manual")}
                  className="accent-indigo-600"
                />
                Manual
              </label>
            </div>
            {nFactorsMode === "manual" && (
              <input
                type="number"
                min={1}
                max={items.length || 1}
                className="select text-xs w-full mt-1"
                value={nFactors}
                onChange={(e) => setNFactors(Math.max(1, parseInt(e.target.value, 10)))}
              />
            )}
          </div>

          <div className="space-y-1.5 pt-1">
            <label className="text-xs font-medium text-gray-600 block">Missing Values</label>
            <select
              className="select text-xs w-full"
              value={imputation}
              onChange={(e) => setImputation(e.target.value as "listwise" | "mean" | "median")}
            >
              <option value="listwise">Listwise deletion</option>
              <option value="mean">Mean substitution</option>
              <option value="median">Median substitution</option>
            </select>
          </div>

          <button
            className="btn-primary w-full mt-3 py-1.5"
            onClick={run}
            disabled={loading}
          >
            {loading ? "Computing..." : "Run Factor Analysis"}
          </button>
          {error && <p className="text-red-400 text-xs mt-1">{error}</p>}
        </div>

        <div className="panel bg-indigo-50 border-indigo-200 space-y-2">
          <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">Guide & Tips</p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            <strong>Bartlett's test</strong> should be significant (p &lt; 0.05) to justify factor analysis.
          </p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            <strong>KMO</strong> should be &gt; 0.60 (ideally &gt; 0.80) to indicate sufficient sampling adequacy.
          </p>
          <p className="text-xs text-indigo-800 leading-relaxed">
            <strong>Promax</strong> oblique rotation allows factors to correlate (realistic in psychology/medicine). <strong>Varimax</strong> keeps them perfectly independent.
          </p>
        </div>
      </div>

      {/* Right results display */}
      <div className="flex-1 min-w-0 space-y-4">
        {result && stale && (
          <StaleResultNotice
            reasons={staleWhy}
            onRecompute={run}
            busy={loading}
            what="This factor analysis"
          />
        )}
        {result ? (
          <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
          <div className="space-y-4">
            {/* Top custom sliders/switches */}
            <div className="panel py-2 px-4 flex flex-wrap items-center gap-4 text-xs bg-white border border-gray-200 rounded-xl">
              <label className="flex items-center gap-2 font-medium text-gray-600">
                Suppress loadings below
                <input
                  type="range"
                  min={0.0}
                  max={0.8}
                  step={0.05}
                  value={cutoff}
                  onChange={(e) => setCutoff(parseFloat(e.target.value))}
                  className="w-24 accent-indigo-600"
                />
                <span className="font-mono bg-gray-50 border px-1.5 py-0.5 rounded text-[10px] text-gray-700">
                  {cutoff.toFixed(2)}
                </span>
              </label>

              <label className="flex items-center gap-1.5 font-medium text-gray-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={sortLoadings}
                  onChange={(e) => setSortLoadings(e.target.checked)}
                  className="accent-indigo-600 rounded"
                />
                Sort by loading size
              </label>
              
              <div className="ml-auto flex items-center gap-1.5">
                <ResultExporter title="Factor_Analysis" headers={result.export_rows?.[0]} rows={result.export_rows?.slice(1)} />
              </div>
            </div>

            {/* Results tabs */}
            <div className="border-b border-gray-200">
              <nav className="flex gap-1">
                {[
                  ["suitability", "Suitability & Variance"],
                  ["loadings", "Loadings Matrix"],
                  ["scree", "Scree Plot"],
                  ["biplot", "Loadings Plot (2D)"]
                ].map(([tabId, label]) => (
                  <button
                    key={tabId}
                    onClick={() => setResultTab(tabId as ActiveResultTab)}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-t-lg transition-colors border-b-2
                      ${resultTab === tabId
                        ? "border-indigo-600 text-indigo-700"
                        : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-200"}`}
                  >
                    {label}
                  </button>
                ))}
              </nav>
            </div>

            {/* Tab contents */}
            {resultTab === "suitability" && (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Sphericity Sphericity card */}
                  <div className="panel space-y-2.5">
                    <h4 className="font-semibold text-gray-900 text-sm">Bartlett's Test of Sphericity</h4>
                    <div className="grid grid-cols-3 gap-2 text-center text-xs">
                      <div className="bg-gray-50 p-2 rounded-lg border">
                        <p className="text-[10px] text-gray-400 font-medium">Chi-Square</p>
                        <p className="font-mono font-semibold text-gray-700 text-sm mt-0.5">
                          {result.suitability.bartlett_chi2.toFixed(3)}
                        </p>
                      </div>
                      <div className="bg-gray-50 p-2 rounded-lg border">
                        <p className="text-[10px] text-gray-400 font-medium">df</p>
                        <p className="font-mono font-semibold text-gray-700 text-sm mt-0.5">
                          {result.suitability.bartlett_df}
                        </p>
                      </div>
                      <div className="bg-gray-50 p-2 rounded-lg border">
                        <p className="text-[10px] text-gray-400 font-medium"><i>p</i>-value</p>
                        <p className={`font-mono font-semibold text-sm mt-0.5 ${result.suitability.bartlett_p < 0.05 ? "text-indigo-600" : "text-gray-500"}`}>
                          {fmtP(result.suitability.bartlett_p)}
                        </p>
                      </div>
                    </div>
                    <p className="text-[11px] text-gray-500 italic mt-1 leading-relaxed">
                      {result.suitability.bartlett_p < 0.05
                        ? "Significant: Variables are sufficiently correlated to extract factors."
                        : "Warning: Variables do not show significant correlation. Factor analysis may not be appropriate."}
                    </p>
                  </div>

                  {/* KMO Card */}
                  <div className="panel space-y-2.5">
                    <h4 className="font-semibold text-gray-900 text-sm">Kaiser-Meyer-Olkin (KMO)</h4>
                    <div className="flex items-center gap-4">
                      <div className={`text-2xl font-bold px-3 py-1.5 rounded-xl ${kmoColor(result.suitability.overall_kmo)}`}>
                        {result.suitability.overall_kmo.toFixed(3)}
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-gray-700">Sampling Adequacy</p>
                        <p className="text-[11px] text-gray-400">Rating: {result.suitability.kmo_rating}</p>
                      </div>
                    </div>
                    <p className="text-[11px] text-gray-500 italic leading-relaxed">
                      Measures proportion of variance in items that might be caused by underlying factors. Values &gt; 0.80 are considered meritorious/excellent.
                    </p>
                  </div>
                </div>

                {/* Variance Table */}
                <div className="panel space-y-2">
                  <h4 className="font-semibold text-gray-900 text-sm">Eigenvalues & Variance Explained</h4>
                  <div className="overflow-auto rounded-lg border border-gray-200">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="bg-gray-50 text-gray-500 font-medium">
                          <th className="px-3 py-2 text-left">Component / Factor</th>
                          <th className="px-3 py-2 text-right">Eigenvalue</th>
                          <th className="px-3 py-2 text-right">% of Variance</th>
                          <th className="px-3 py-2 text-right">Cumulative %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.variance_explained.map((row) => {
                          const isRetained = row.component <= result.n_factors;
                          return (
                            <tr
                              key={row.component}
                              className={`border-t border-gray-100 ${isRetained ? "bg-indigo-50/40 text-indigo-900 font-medium" : "text-gray-500"}`}
                            >
                              <td className="px-3 py-2">
                                {row.component} {isRetained && <span className="text-[9px] bg-indigo-100 text-indigo-700 rounded px-1.5 py-0.25 ml-2">Retained</span>}
                              </td>
                              <td className="px-3 py-2 text-right font-mono">{row.eigenvalue.toFixed(3)}</td>
                              <td className="px-3 py-2 text-right font-mono">{row.pct_variance.toFixed(2)}%</td>
                              <td className="px-3 py-2 text-right font-mono">{row.cum_variance.toFixed(2)}%</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {resultTab === "loadings" && (
              <div className="panel space-y-2">
                <h4 className="font-semibold text-gray-900 text-sm">
                  Factor Loadings Matrix ({result.rotation_method})
                </h4>
                <p className="text-[11px] text-gray-400">
                  Cutoff active: Loadings with absolute values &lt; {cutoff} are hidden.
                </p>
                <div className="overflow-auto rounded-lg border border-gray-200">
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr className="bg-gray-50 text-gray-500 font-medium border-b border-gray-200">
                        <th className="px-4 py-2 text-left">Variable</th>
                        {result.factors.map((f: string) => (
                          <th key={f} className="px-4 py-2 text-right">{f}</th>
                        ))}
                        <th className="px-4 py-2 text-right border-l">h² (Communality)</th>
                        <th className="px-4 py-2 text-right">u² (Uniqueness)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {getSortedLoadings().map((row) => (
                        <tr key={row.variable} className="border-t border-gray-100 hover:bg-gray-50 transition-colors">
                          <td className="px-4 py-2 font-mono text-gray-700">{row.variable}</td>
                          {result.factors.map((f: string) => {
                            const val = Number(row[f]);
                            const isStrong = Math.abs(val) >= 0.4;
                            return (
                              <td
                                key={f}
                                className={`px-4 py-2 text-right font-mono ${isStrong ? "text-indigo-600 font-semibold" : "text-gray-600"}`}
                              >
                                {fmtLoading(val)}
                              </td>
                            );
                          })}
                          <td className="px-4 py-2 text-right font-mono text-gray-400 border-l">{row.h2.toFixed(3)}</td>
                          <td className="px-4 py-2 text-right font-mono text-gray-400">{row.u2.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {resultTab === "scree" && screePlot()}

            {resultTab === "biplot" && (
              <div className="space-y-4">
                {loadingsBiplot()}
                {result.n_factors < 2 && (
                  <p className="text-xs text-amber-600 bg-amber-50 rounded-lg p-2.5">
                    ⚠️ Loadings plot requires at least 2 retained factors/components.
                  </p>
                )}
              </div>
            )}

            {/* Equivalent R Replication Code details */}
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
            <p className="text-3xl mb-3">📊</p>
            <p className="font-semibold text-gray-600">Select Variables to begin</p>
            <p className="text-xs mt-1 max-w-sm mx-auto">
              Principal Component Analysis (PCA) and Factor Analysis (EFA) allow you to identify hidden dimensions and reduce the complexity of multi-variable measurement scales.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
