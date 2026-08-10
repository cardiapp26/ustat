import { useState, useEffect, useRef, useMemo } from "react";
import Plot from "../PlotComponent";
import TitledPlot from "./TitledPlot";
import PlotExporter from "./PlotExporter";
import { useStore, paletteOf, isNumericKind, type Session } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import ResultExporter from "./ResultExporter";
import ThreeCol from "./ThreeCol";
import { Tip, LabelTip, InfoBanner } from "./Tip";
import {
  runCorrelationPair,
  runCorrelationMatrix,
  runICC,
  runCohensKappa,
  getRawColumns,
} from "../api";
import { fmtP } from "../lib/format";
import type { PlotData, PlotCaptureHandle } from "../lib/plotTypes";

// Reads the live theme, so a custom palette — whose colours live on the theme
// rather than in the PALETTES constant — is honoured here too.
const _pal = () => paletteOf(useStore.getState().plotTheme);

/** Background, font and colourway from the global chart theme.
 *
 *  This panel hard-coded all three, so changing the theme did nothing to it.
 *  A hook rather than a plain read: subscribing is what makes the chart
 *  repaint when the palette changes, instead of at the next unrelated render.
 */
function usePlotBg(): Record<string, unknown> {
  const t = useStore((s) => s.plotTheme);
  return {
    paper_bgcolor: "transparent",
    plot_bgcolor: t.plotBg,
    font: { family: t.fontFamily, color: "#374151", size: t.fontSize },
    colorway: paletteOf(t),
  };
}

const TABS = ["Pairwise", "Matrix", "ICC", "Cohen's κ"] as const;
type Tab = (typeof TABS)[number];

function sig(p: number) {
  return p < 0.05 ? "text-indigo-600 font-semibold" : "text-gray-400";
}
function starsFor(p: number | null): string {
  if (p == null) return "";
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "";
}

function getErrorDetail(e: unknown, fallback = "Error"): string {
  const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  return typeof detail === "string" ? detail : fallback;
}

function downloadCSV(filename: string, rows: string[][]): void {
  const csv = rows.map((r) => r.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

interface PairResult {
  var1: string;
  var2: string;
  r: number;
  p: number;
  n: number;
  ci_low: number;
  ci_high: number;
  method: string;
  label: string;
  normality_test: string;
  normality: Record<string, {
    p: number | null;
    statistic: number | null;
    normal: boolean;
    skewness: number;
    test: string;
    bypass: string | null;
  }>;
  scatter: { x: number[]; y: number[] };
  regression_line: { x: number[]; y: number[] };
  ci_band: { x: number[]; y_upper: number[]; y_lower: number[] };
  autoSwitched: boolean;
  result_text?: string;
}

// ── PairwiseTab ───────────────────────────────────────────────────────────────
function PairwiseTab({ sessionId, columns }: { sessionId: string; columns: string[] }) {
  const plotBg = usePlotBg();
  const showGrid = useStore((s) => s.showGrid);
  const corrScatterRef = useRef<PlotCaptureHandle | null>(null);
  const [vars, setVars] = usePersistedPanelState<string[]>("correlation_pairwise", "vars", columns.slice(0, Math.min(4, columns.length)));
  const [varFilter, setVarFilter] = useState("");
  const [method, setMethod] = usePersistedPanelState<string>("correlation_pairwise", "method", "auto");
  // Detect ordinal variables among the selection → Spearman is the right
  // measure for ordered data (Pearson assumes interval scale).
  const sessionCols = useStore((s) => s.session?.columns);
  const ordinalNames = useMemo(
    () => new Set((sessionCols ?? []).filter((c) => c.kind === "ordinal").map((c) => c.name)),
    [sessionCols],
  );
  const hasOrdinalSelected = vars.some((v) => ordinalNames.has(v));
  const [results, setResults] = useState<PairResult[]>([]);
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const toggle = (c: string) =>
    setVars(vars.includes(c) ? vars.filter((x) => x !== c) : [...vars, c]);

  const nPairs = Math.max(0, vars.length * (vars.length - 1) / 2);

  const run = async () => {
    if (vars.length < 2) { setError("Select at least 2 variables"); return; }
    setError(""); setResults([]); setActiveIdx(null); setLoading(true);

    const pairs: [string, string][] = [];
    for (let i = 0; i < vars.length; i++)
      for (let j = i + 1; j < vars.length; j++)
        pairs.push([vars[i], vars[j]]);

    try {
      const settled = await Promise.allSettled(
        pairs.map(([v1, v2]) =>
          runCorrelationPair({ session_id: sessionId, var1: v1, var2: v2, method, imputation: "listwise" })
        )
      );
      const parsed: PairResult[] = [];
      settled.forEach((s, i) => {
        if (s.status !== "fulfilled") return;
        const d = s.value.data;
        const autoSwitched =
          method === "auto" &&
          d.method === "spearman" &&
          Object.values(d.normality as Record<string, { normal: boolean }>).some((n) => !n.normal);
        parsed.push({ ...d, var1: pairs[i][0], var2: pairs[i][1], autoSwitched });
      });
      parsed.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
      setResults(parsed);
      if (parsed.length > 0) setActiveIdx(0);
    } catch {
      setError("Computation failed");
    } finally {
      setLoading(false);
    }
  };

  const pairHeaders = ["Variable 1", "Variable 2", "r / ρ", "95% CI Low", "95% CI High", "p", "n", "Method", "Stars"];
  const pairRows = results.map((res) => [
    res.var1, res.var2,
    res.r.toFixed(4),
    res.ci_low.toFixed(4),
    res.ci_high.toFixed(4),
    fmtP(res.p),
    String(res.n),
    res.method,
    starsFor(res.p),
  ]);

  const active = activeIdx != null ? results[activeIdx] : null;

  const leftCol = (
    <div className="panel space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">Variables</h3>
        <div className="flex gap-1">
          <button onClick={() => setVars([...columns])} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">All</button>
          <button onClick={() => setVars([])} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">None</button>
        </div>
      </div>
      <input
        type="text"
        placeholder="Filter variables…"
        value={varFilter}
        onChange={(e) => setVarFilter(e.target.value)}
        className="select w-full text-xs py-1"
      />
      <div className="max-h-40 overflow-y-auto space-y-0.5 border border-gray-200 rounded p-1 bg-white">
        {columns
          .filter((c) => c.toLowerCase().includes(varFilter.toLowerCase()))
          .map((c) => (
            <label key={c} className="flex items-center gap-2 px-1 py-0.5 rounded hover:bg-gray-50 cursor-pointer">
              <input type="checkbox" checked={vars.includes(c)} onChange={() => toggle(c)} className="accent-indigo-500" />
              <span className="text-xs text-gray-700 truncate">{c}</span>
            </label>
          ))}
      </div>
      <p className="text-[10px] text-gray-400">{vars.length} selected · {nPairs} pair{nPairs !== 1 ? "s" : ""}</p>

      <h3 className="text-sm font-semibold text-gray-700 pt-1 border-t border-gray-100">
        Method
        <Tip text="Auto: tests each variable's normality (Shapiro-Wilk for n<50, Lilliefors-corrected KS for 50–2000, skewness/CLT for n>2000) and picks Pearson when both are normal (p ≥ 0.05), or Spearman if either is not. Prevents the common mistake of running Pearson on skewed data." wide />
      </h3>
      <div className="space-y-1.5">
        {(["auto", "pearson", "spearman"] as const).map((m) => (
          <label key={m} className="flex items-center gap-2 cursor-pointer">
            <input type="radio" name="pw-method" value={m} checked={method === m}
              onChange={() => setMethod(m)} className="accent-indigo-500" />
            <span className="text-xs text-gray-700">{m === "auto" ? "Auto (by normality)" : m === "pearson" ? "Pearson r" : "Spearman ρ"}</span>
          </label>
        ))}
      </div>

      {hasOrdinalSelected && method !== "spearman" && (
        <div className="text-[10px] text-teal-700 bg-teal-50 border border-teal-200 rounded px-2 py-1.5 leading-snug flex items-start gap-1.5">
          <span className="flex-1">
            Ordinal (ordered categorical) variable selected — <strong>Spearman ρ</strong> is
            recommended for ordered data; Pearson assumes an interval scale.
          </span>
          <button onClick={() => setMethod("spearman")} className="flex-shrink-0 underline hover:text-teal-900">
            Use Spearman
          </button>
        </div>
      )}

      <button className="btn-primary w-full mt-2" onClick={run} disabled={loading || vars.length < 2}>
        {loading ? "Computing…" : `Compute ${nPairs > 1 ? `${nPairs} Pairs` : "Pair"}`}
      </button>
      {error && <p className="text-xs text-red-500">{error}</p>}

      {active && (
        <div className="border-t border-gray-100 pt-3 space-y-2 text-[11px]">
          <p className="text-gray-500 font-semibold flex items-center gap-1">
            Normality Assessment
            <Tip wide text={
              active.n <= 2000
                ? `Shapiro-Wilk (n = ${active.n}): gold-standard test for normality in small-to-medium samples. ✓ = normal (p ≥ 0.05). If either variable is non-normal, Spearman ρ is preferred.`
                : `Large sample (n = ${active.n} > 2000). Two-step approach: (1) Skewness check — if |skewness| ≤ 1.5, the Central Limit Theorem ensures Pearson r remains valid regardless of distribution shape. (2) If |skewness| > 1.5, Lilliefors-corrected KS test is used (standard KS adjusted for estimated parameters). Standard KS without Lilliefors correction would give anti-conservative p-values.`
            } />
          </p>
          {Object.entries(active.normality).map(([v, nm]) => (
            <div key={v} className="space-y-0.5 bg-gray-50 p-1.5 rounded border border-gray-100">
              <div className="flex justify-between items-center">
                <span className="text-gray-600 truncate max-w-[120px] font-medium" title={v}>{v}</span>
                <div className="flex items-center gap-1.5">
                  {nm.bypass === "clt_skew" ? (
                    <span className="text-green-600 font-semibold" title="CLT bypass: mild skewness at large n — Pearson is robust">
                      CLT ✓
                    </span>
                  ) : (
                    <span className={nm.normal ? "text-green-600 font-semibold" : "text-red-500 font-semibold"}>
                      {nm.p != null ? <><i>p</i>={fmtP(nm.p)}</> : ""} {nm.normal ? "✓" : "✗"}
                    </span>
                  )}
                </div>
              </div>
              <div className="flex justify-between text-[9px] text-gray-400 pl-0.5">
                <span>{nm.test}</span>
                <span title="Skewness: 0 = symmetric. |skew| < 1 = mild, 1–2 = moderate, > 2 = severe">
                  skew={nm.skewness.toFixed(2)}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const middleCol = active ? (
    <div className="panel">
      <TitledPlot
        plotRefOut={corrScatterRef}
        storageKey={`corr:pairwise:scatter:${active.var1}:${active.var2}`}
        defaultTitle=""
        defaultSubtitle=""
        defaultXAxis={active.var1}
        defaultYAxis={active.var2}
        data={[
          {
            type: "scatter",
            mode: "markers",
            x: active.scatter.x,
            y: active.scatter.y,
            marker: { color: _pal()[0], opacity: 0.65, size: 6 },
            name: "Data",
            hovertemplate: `${active.var1}: %{x:.3f}<br>${active.var2}: %{y:.3f}<extra></extra>`,
          },
          {
            type: "scatter",
            mode: "lines",
            x: active.regression_line.x,
            y: active.regression_line.y,
            line: { color: "#f59e0b", width: 2 },
            name: "Regression line",
            hoverinfo: "skip",
          },
          {
            type: "scatter",
            mode: "lines",
            x: [...active.ci_band.x, ...[...active.ci_band.x].reverse()],
            y: [...active.ci_band.y_upper, ...[...active.ci_band.y_lower].reverse()],
            fill: "toself",
            fillcolor: "rgba(245,158,11,0.12)",
            line: { width: 0 },
            name: "95% CI band",
            hoverinfo: "skip",
            showlegend: true,
          },
        ]}
        layout={{
          ...plotBg,
          autosize: true,
          xaxis: { title: active.var1, gridcolor: "#e5e7eb", showgrid: showGrid, zeroline: false },
          yaxis: { title: active.var2, gridcolor: "#e5e7eb", showgrid: showGrid, zeroline: false },
          legend: { orientation: "h", y: -0.18, font: { size: 10, color: "#374151" } },
          annotations: [
            {
              xref: "paper" as const,
              yref: "paper" as const,
              x: 0.98,
              y: 0.98,
              xanchor: "right" as const,
              yanchor: "top" as const,
              text: `${active.label} = ${active.r.toFixed(3)}${starsFor(active.p)}, <i>p</i> = ${fmtP(active.p)}`,
              showarrow: false,
              font: { color: "#374151", size: 12 },
              bgcolor: "rgba(249,250,251,0.9)",
              bordercolor: "#e5e7eb",
              borderpad: 4,
              borderwidth: 1,
            },
          ],
          margin: { t: 20, r: 20, b: 60, l: 60 },
        }}
        config={{ responsive: true, displayModeBar: false }}
      />
    </div>
  ) : (
    <div className="panel h-[480px] flex items-center justify-center text-gray-400 text-xs text-center border-dashed border-2">
      <div>
        <p className="text-2xl mb-2">📊</p>
        <p className="font-semibold text-gray-500">Correlation Chart</p>
        <p className="text-[11px] text-gray-400 mt-1 max-w-xs">Select variables on the left and click Compute to view the interactive scatter plot and regression line.</p>
      </div>
    </div>
  );

  const rightCol = results.length > 0 ? (
    <div className="space-y-3">
      {/* Results table */}
      <div className="panel">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-gray-500">Results ({results.length} pairs)</span>
          <ResultExporter title="correlation_pairwise" headers={pairHeaders} rows={pairRows} />
        </div>
        <div className="overflow-auto max-h-48 border border-gray-100 rounded-lg">
          <table className="w-full text-xs border-collapse">
            <thead className="sticky top-0 bg-gray-50 border-b border-gray-200 z-10">
              <tr className="text-left text-gray-400">
                <th className="px-2 py-1.5 font-medium">Variable 1</th>
                <th className="px-2 py-1.5 font-medium">Variable 2</th>
                <th className="px-2 py-1.5 font-medium">r / ρ</th>
                <th className="px-2 py-1.5 font-medium">95% CI</th>
                <th className="px-2 py-1.5 font-medium"><i>p</i></th>
              </tr>
            </thead>
            <tbody>
              {results.map((res, i) => {
                const highCorr = Math.abs(res.r) >= 0.70;
                const isActive = i === activeIdx;
                return (
                  <tr
                    key={`${res.var1}-${res.var2}`}
                    onClick={() => setActiveIdx(i)}
                    className={`cursor-pointer border-b border-gray-100 transition-colors ${
                      isActive
                        ? "bg-indigo-50/80 font-medium"
                        : highCorr
                        ? "bg-amber-50 hover:bg-amber-100/80"
                        : "hover:bg-gray-50"
                    }`}
                  >
                    <td className="px-2 py-1 font-mono text-gray-700 truncate max-w-[100px]">{res.var1}</td>
                    <td className="px-2 py-1 font-mono text-gray-700 truncate max-w-[100px]">{res.var2}</td>
                    <td className="px-2 py-1 font-mono">
                      <span className={Math.abs(res.r) >= 0.5 ? "text-indigo-600 font-bold" : "text-gray-700"}>
                        {res.r.toFixed(3)}
                      </span>
                      <span className="text-amber-500 ml-0.5">{starsFor(res.p)}</span>
                      {highCorr && <span className="ml-1 text-amber-500" title="High collinearity (|r| ≥ 0.70)">⚠</span>}
                    </td>
                    <td className="px-2 py-1 text-gray-400 font-mono text-[10px] whitespace-nowrap">
                      [{res.ci_low.toFixed(2)}, {res.ci_high.toFixed(2)}]
                    </td>
                    <td className={`px-2 py-1 font-mono ${res.p < 0.05 ? "text-indigo-600 font-semibold" : "text-gray-400"}`}>
                      {fmtP(res.p)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {active?.autoSwitched && (
        <div className="flex items-start gap-1.5 bg-blue-50 border border-blue-200 rounded-lg p-2 text-[11px] text-blue-700">
          <span>⚡</span>
          <span>
            <strong>Switched to Spearman:</strong>{" "}
            {active.normality_test?.includes("Lilliefors")
              ? <>Lilliefors-corrected KS <i>p</i> &lt; 0.05 (<i>n</i> = {active.n}). Distribution is non-normal.</>
              : <>Skewed data detected ({active.normality_test ?? "Shapiro-Wilk"} <i>p</i> &lt; 0.05). Spearman ρ used for robust estimation.</>}
          </span>
        </div>
      )}

      {active && Math.abs(active.r) >= 0.70 && (
        <div className="flex items-start gap-1.5 bg-amber-50 border border-amber-200 rounded-lg p-2 text-[11px] text-amber-700">
          <span>⚠</span>
          <span>
            <strong>High Collinearity (|r| = {Math.abs(active.r).toFixed(3)} ≥ 0.70).</strong>{" "}
            Avoid putting both <em>{active.var1}</em> and <em>{active.var2}</em> into the same multivariable regression model.
          </span>
        </div>
      )}

      {active && (
        <div className="panel space-y-2">
          <div className="flex gap-4 text-xs flex-wrap items-center bg-gray-50 p-2 rounded-xl border border-gray-100">
            <span className="text-gray-500">
              <LabelTip tip="Correlation coefficient. |r| < 0.1 negligible · 0.1–0.3 weak · 0.3–0.5 moderate · > 0.5 strong." wide>
                {active.label}
              </LabelTip>
              {" = "}
              <span className={`font-bold ${Math.abs(active.r) >= 0.5 ? "text-indigo-600" : "text-gray-700"}`}>
                {active.r.toFixed(4)}
              </span>
              <span className="ml-1 text-amber-500 font-semibold">{starsFor(active.p)}</span>
            </span>
            <span className="text-gray-500">
              95% CI: <span className="text-gray-700 font-mono font-semibold">[{active.ci_low.toFixed(3)}, {active.ci_high.toFixed(3)}]</span>
            </span>
            <span className="text-gray-500">
              <i>p</i> = <span className={sig(active.p)}>{fmtP(active.p)}</span>
            </span>
            <span className="text-gray-400 font-mono"><i>n</i> = {active.n}</span>
          </div>
          <InfoBanner>
            {active.p < 0.05
              ? <>Significant {active.r > 0 ? "positive" : "negative"} correlation between {active.var1} and {active.var2} ({active.label} = {active.r.toFixed(3)}, <i>p</i> {fmtP(active.p)}).</>
              : <>No statistically significant correlation found between {active.var1} and {active.var2} (<i>p</i> = {fmtP(active.p)}).</>}
          </InfoBanner>
          {active.result_text && (
            <div className="bg-gray-50 border border-gray-200 rounded-xl px-3 py-2.5 mt-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] font-semibold text-gray-400 uppercase">Results Paragraph</span>
                <button onClick={() => navigator.clipboard.writeText(active.result_text!)} className="text-[10px] px-2 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 transition-colors">Copy</button>
              </div>
              <p className="text-xs text-gray-700 leading-relaxed font-sans">{active.result_text}</p>
            </div>
          )}
        </div>
      )}
    </div>
  ) : (
    <div className="panel p-6 flex items-center justify-center text-gray-400 text-xs text-center border-dashed border-2">
      <div>
        <p className="text-2xl mb-2">📝</p>
        <p className="font-semibold text-gray-500">Written Results</p>
        <p className="text-[11px] text-gray-400 mt-1 max-w-xs">Computation results, APA-style paragraphs, and correlation tables will be displayed here.</p>
      </div>
    </div>
  );

  return (
    <ThreeCol
      storageKey="CorrelationPanel.Pairwise"
      left={leftCol}
      middle={middleCol}
      right={rightCol}
    />
  );
}

interface MulticollinearityWarning {
  var1: string;
  var2: string;
  r: number;
}
interface MatrixResult {
  variables: string[];
  matrix: Record<string, Record<string, number | null>>;
  p_matrix?: Record<string, Record<string, number | null>>;
  multicollinearity_warnings: MulticollinearityWarning[];
}

// ── MatrixTab ─────────────────────────────────────────────────────────────────
function MatrixTab({ sessionId, columns }: { sessionId: string; columns: string[] }) {
  const plotBg = usePlotBg();
  const showGrid = useStore((s) => s.showGrid);
  const corrHeatmapRef = useRef<PlotCaptureHandle | null>(null);
  const corrSplomRef = useRef<PlotCaptureHandle | null>(null);
  const [selected, setSelected] = usePersistedPanelState<string[]>("correlation_matrix", "selected", columns.slice(0, Math.min(8, columns.length)));
  const [colFilter, setColFilter] = useState("");
  const [method, setMethod] = usePersistedPanelState<string>("correlation_matrix", "method", "pearson");
  const [data, setData] = useState<MatrixResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [displayMode, setDisplayMode] = usePersistedPanelState<"heatmap" | "splom">("correlation_matrix", "displayMode", "heatmap");
  const [rawData, setRawData] = useState<Record<string, (number | null)[]> | null>(null);
  const [rawLoading, setRawLoading] = useState(false);
  const [selectedVar, setSelectedVar] = useState<string | null>(null);

  const toggle = (c: string) =>
    setSelected(selected.includes(c) ? selected.filter((x) => x !== c) : [...selected, c]);

  const fetchRaw = (cols: string[]) => {
    setRawLoading(true);
    getRawColumns(sessionId, cols)
      .then((r) => setRawData(r.data))
      .catch(() => {})
      .finally(() => setRawLoading(false));
  };

  const run = async () => {
    if (selected.length < 2) { setError("Select at least 2 variables"); return; }
    setError("");
    setLoading(true);
    try {
      const res = await runCorrelationMatrix({ session_id: sessionId, variables: selected, method, imputation: "listwise" });
      setData(res.data as MatrixResult);
      if (displayMode === "splom") fetchRaw(selected);
    } catch (e: unknown) {
      setError(getErrorDetail(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (displayMode !== "splom" || selected.length < 2) return;
    fetchRaw(selected);
    // Use a primitive join of `selected` so we re-fetch only when the column
    // set actually changes — not on every array identity tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [displayMode, selected.join(","), sessionId]);

  const exportMatrix = () => {
    if (!data) return;
    const vars: string[] = data.variables;
    const header = ["", ...vars];
    const rows = vars.map((r: string) => [
      r,
      ...vars.map((c: string) => {
        if (r === c) return "1";
        const v = data.matrix[r][c];
        return v != null ? v.toFixed(4) : "";
      }),
    ]);
    downloadCSV("correlation_matrix.csv", [header, ...rows]);
  };

  const renderVarDetail = (v: string) => {
    if (!rawData || !rawData[v]) return null;
    const vals = rawData[v].filter((x): x is number => x != null);
    if (vals.length === 0) return null;
    const minV = Math.min(...vals), maxV = Math.max(...vals);
    const nBins = Math.min(20, Math.max(6, Math.round(vals.length ** 0.38)));
    const binW = (maxV - minV) / nBins;
    const counts = Array(nBins).fill(0);
    vals.forEach((x) => { const b = Math.min(Math.floor((x - minV) / binW), nBins - 1); counts[b]++; });
    const xs = counts.map((_, i) => +(minV + (i + 0.5) * binW).toFixed(4));
    return (
      <div className="panel flex-shrink-0 flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <p className="text-xs font-semibold text-gray-700 truncate">{v}</p>
          <button onClick={() => setSelectedVar(null)} className="text-gray-300 hover:text-gray-600 text-xs">✕</button>
        </div>
        <p className="text-[10px] text-gray-400 font-mono"><i>n</i> = {vals.length}</p>
        <Plot
          data={[{ type: "bar" as const, x: xs, y: counts,
            marker: { color: _pal()[0], opacity: 0.8 },
            hovertemplate: "%{x:.2f}: %{y}<extra></extra>" }]}
          layout={{
            ...plotBg,
            // A 130px strip needs smaller type than the global setting.
            font: { ...(plotBg.font as object), size: 9 },
            height: 130, margin: { t: 8, r: 4, b: 28, l: 28 },
            xaxis: { gridcolor: "#f3f4f6", zeroline: false, title: { text: v, font: { size: 8 } } },
            yaxis: { gridcolor: "#f3f4f6", zeroline: false },
            showlegend: false,
          }}
          style={{ width: "100%", height: 130 }}
          useResizeHandler
          config={{ displayModeBar: false, responsive: true }}
        />
        {data?.matrix?.[v] && (
          <div className="space-y-1">
            <p className="text-[10px] text-gray-400 font-semibold">Correlations</p>
            {Object.entries(data.matrix[v] as Record<string, number>)
              .filter(([k]) => k !== v)
              .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
              .slice(0, 6)
              .map(([k, r]) => (
                <div key={k} className="flex items-center justify-between gap-1 bg-gray-50 px-2 py-1 rounded border border-gray-100">
                  <span className="text-[9px] text-gray-500 truncate flex-1 font-mono">{k}</span>
                  <span className={`text-[9px] font-mono font-semibold ${Math.abs(r) >= 0.5 ? "text-indigo-600" : "text-gray-500"}`}>
                    {r.toFixed(3)}
                  </span>
                </div>
              ))}
          </div>
        )}
      </div>
    );
  };

  const leftCol = (
    <div className="panel space-y-3">
      <h3 className="text-sm font-semibold text-gray-700">
        Method
        <Tip text="Pearson: linear relationships, normally distributed data. Spearman: ranked/non-normal. Kendall: small samples or many tied ranks. Missing values handled with pairwise deletion." wide />
      </h3>
      <div className="space-y-1">
        {["pearson", "spearman", "kendall"].map((m) => (
          <label key={m} className="flex items-center gap-2 cursor-pointer">
            <input type="radio" name="mx-method" value={m} checked={method === m}
              onChange={() => setMethod(m)} className="accent-indigo-500" />
            <span className="text-xs text-gray-700 capitalize">{m}</span>
          </label>
        ))}
      </div>

      <div className="pt-2 border-t border-gray-100">
        <p className="text-xs font-semibold text-gray-700 mb-2">Display</p>
        <div className="flex rounded overflow-hidden border border-gray-200">
          {(["heatmap", "splom"] as const).map((mode) => (
            <button key={mode} onClick={() => setDisplayMode(mode)}
              className={`flex-1 text-[11px] py-1 transition-colors ${displayMode === mode ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`}>
              {mode === "heatmap" ? "Heatmap" : "Scatter"}
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between pt-2 border-t border-gray-100">
        <h3 className="text-sm font-semibold text-gray-700">Variables</h3>
        <div className="flex gap-1">
          <button onClick={() => setSelected([...columns])} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">All</button>
          <button onClick={() => setSelected([])} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50">None</button>
        </div>
      </div>
      <input
        type="text"
        placeholder="Filter variables…"
        value={colFilter}
        onChange={(e) => setColFilter(e.target.value)}
        className="select w-full text-xs py-1"
      />
      <div className="space-y-0.5 max-h-40 overflow-y-auto border border-gray-200 rounded p-1 bg-white">
        {columns
          .filter((c) => c.toLowerCase().includes(colFilter.toLowerCase()))
          .map((c) => (
            <label key={c} className="flex items-center gap-2 px-1 py-0.5 rounded hover:bg-gray-50 cursor-pointer">
              <input type="checkbox" checked={selected.includes(c)} onChange={() => toggle(c)}
                className="accent-indigo-500" />
              <span className="text-xs text-gray-700 truncate">{c}</span>
            </label>
          ))}
      </div>
      <p className="text-[10px] text-gray-400">{selected.length} selected</p>

      <button className="btn-primary w-full mt-1" onClick={run} disabled={loading}>
        {loading ? "Computing…" : "Compute Matrix"}
      </button>
      {error && <p className="text-xs text-red-500">{error}</p>}
    </div>
  );

  const middleCol = data ? (
    displayMode === "heatmap" ? (
      <div className="panel flex flex-col gap-2">
        <div className="flex items-center justify-between flex-shrink-0">
          <span className="text-xs font-semibold text-gray-500">Correlation Matrix</span>
          <button
            onClick={exportMatrix}
            className="flex items-center gap-1 text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-gray-50 hover:text-indigo-600 hover:border-indigo-300 transition-colors"
          >
            ↓ Export CSV
          </button>
        </div>
        <TitledPlot
          plotRefOut={corrHeatmapRef}
          storageKey="corr:matrix:heatmap"
          defaultTitle=""
          defaultSubtitle=""
          defaultXAxis=""
          defaultYAxis=""
          data={[{
            type: "heatmap",
            z: data.variables.map((c1: string) =>
              data.variables.map((c2: string) => data.matrix[c1][c2])
            ),
            x: data.variables,
            y: data.variables,
            colorscale: [
              [0,   "#2563eb"],
              [0.25,"#93c5fd"],
              [0.5, "#f9fafb"],
              [0.75,"#fca5a5"],
              [1,   "#dc2626"],
            ],
            zmid: 0, zmin: -1, zmax: 1,
            text: data.variables.map((c1: string) =>
              data.variables.map((c2: string) => {
                if (c1 === c2) return "1";
                const v = data.matrix[c1][c2];
                const p = data.p_matrix?.[c1]?.[c2];
                if (v == null) return "";
                return `${v.toFixed(2)}${starsFor(p ?? null)}`;
              })
            ),
            texttemplate: "%{text}",
            textfont: { size: 11, color: "#111827" },
            hovertemplate: "%{x} vs %{y}: %{z:.4f}<extra></extra>",
          }]}
          layout={{
            ...plotBg,
            autosize: true,
            xaxis: { showgrid: showGrid, gridcolor: "#e5e7eb", zeroline: false },
            yaxis: { showgrid: showGrid, gridcolor: "#e5e7eb", zeroline: false },
            margin: { t: 20, r: 20, b: 100, l: 100 },
          }}
          config={{ responsive: true, displayModeBar: false }}
        />
        <div className="flex gap-4 text-[10px] text-gray-400 flex-shrink-0 border-t pt-1 border-gray-100">
          <span>* p &lt; 0.05 · ** p &lt; 0.01 · *** p &lt; 0.001</span>
          <span className="text-blue-500 font-semibold">■ Neg</span>
          <span className="text-red-500 font-semibold">■ Pos</span>
        </div>
      </div>
    ) : (
      <div className="panel h-[480px] flex flex-col gap-2 relative">
        <PlotExporter plotRef={corrSplomRef} title="Scatter_Matrix" />
        <div className="flex items-center justify-between flex-shrink-0">
          <span className="text-xs font-semibold text-gray-500">
            Scatter Matrix
          </span>
          <div className="flex items-center gap-2">
            {rawLoading && <span className="text-[10px] text-gray-400 animate-pulse">Loading data…</span>}
            <button onClick={exportMatrix}
              className="flex items-center gap-1 text-[10px] px-2 py-1 rounded border border-gray-300 text-gray-500 hover:bg-gray-50 hover:text-indigo-600 hover:border-indigo-300 transition-colors">
              ↓ Export CSV
            </button>
          </div>
        </div>
        {rawData && Object.keys(rawData).length >= 2 ? (
          <Plot
            data={[{
              type: "splom" as const,
              dimensions: Object.keys(rawData).map((col) => ({
                label: col,
                values: rawData[col],
              })),
              marker: {
                color: _pal()[0],
                size: 3,
                opacity: 0.45,
                line: { color: "#a5b4fc", width: 0.5 },
              },
              diagonal: { visible: true },
              showupperhalf: false,
              text: Object.keys(rawData).join(", "),
              hovertemplate: "%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<extra></extra>",
            } as PlotData]}
            layout={{
              ...plotBg,
              font: { ...(plotBg.font as object), size: 10 },
              autosize: true,
              margin: { t: 20, r: 20, b: 20, l: 20 },
              dragmode: "select" as const,
            }}
            onInitialized={(_: object, gd: HTMLElement) => { corrSplomRef.current = gd as unknown as PlotCaptureHandle; }}
            onUpdate={(_: object, gd: HTMLElement)      => { corrSplomRef.current = gd as unknown as PlotCaptureHandle; }}
            style={{ width: "100%", flex: 1 }}
            useResizeHandler
            config={{ responsive: true, displayModeBar: false }}
            onClickAnnotation={(e: Readonly<{ annotation?: { text?: string } }>) => setSelectedVar(e?.annotation?.text ?? null)}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
            {rawLoading ? "Loading scatter data…" : "Matrix computed. Scatter view active."}
          </div>
        )}
        <p className="text-[9px] text-gray-400 flex-shrink-0">
          Showing up to 3,000 rows · click diagonal label to explore variable detail
        </p>
      </div>
    )
  ) : (
    <div className="panel h-[480px] flex items-center justify-center text-gray-400 text-xs text-center border-dashed border-2">
      <div>
        <p className="text-2xl mb-2">🧮</p>
        <p className="font-semibold text-gray-500">Correlation Matrix Visual</p>
        <p className="text-[11px] text-gray-400 mt-1 max-w-xs">Select 2+ variables on the left and click Compute Matrix to view the interactive Heatmap or Scatter Matrix (SPLOM) plot.</p>
      </div>
    </div>
  );

  const rightCol = data ? (
    <div className="space-y-3">
      {data.multicollinearity_warnings.length > 0 && (
        <div className="panel space-y-1.5 border-amber-200 bg-amber-50">
          <p className="text-xs font-semibold text-amber-700 flex items-center gap-1">
            ⚠ High Collinearity Detected (|r| ≥ 0.70)
            <Tip text="Do not include strongly correlated variables in the same regression model to prevent standard error inflation." wide />
          </p>
          {data.multicollinearity_warnings.map((w: MulticollinearityWarning, i: number) => (
            <div key={i} className="text-xs text-gray-700 flex items-center justify-between bg-white/60 px-2 py-1 rounded border border-amber-100">
              <span className="font-mono truncate max-w-[150px]">{w.var1} ↔ {w.var2}</span>
              <span className="text-amber-600 font-bold font-mono">r = {w.r.toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}
      {displayMode === "splom" && selectedVar ? (
        renderVarDetail(selectedVar)
      ) : (
        <div className="panel text-gray-400 text-xs p-6 text-center border-dashed border-2">
          <div>
            <p className="text-xl mb-1">📋</p>
            <p className="font-semibold">Variable Distribution Detail</p>
            <p className="text-[10px] text-gray-400 mt-1">
              {displayMode === "splom" 
                ? "Click a variable's label in the scatter matrix to view its histogram and top correlations here." 
                : "Heatmap mode displays global coefficients. Switch to Scatter mode to enable individual variable exploration."}
            </p>
          </div>
        </div>
      )}
    </div>
  ) : (
    <div className="panel flex items-center justify-center text-gray-400 text-xs text-center border-dashed border-2 p-6">
      <div>
        <p className="text-2xl mb-2">💡</p>
        <p className="font-semibold text-gray-500">Collinearity & Detailed Info</p>
        <p className="text-[11px] text-gray-400 mt-1 max-w-xs">Highly collinear variables and distribution histograms will be displayed in this panel after computing.</p>
      </div>
    </div>
  );

  return (
    <ThreeCol
      storageKey="CorrelationPanel.Matrix"
      left={leftCol}
      middle={middleCol}
      right={rightCol}
    />
  );
}

interface ICCResult {
  icc: number;
  ci_low: number;
  ci_high: number;
  f_stat: number;
  f_p: number;
  n: number;
  interpretation: string;
  bland_altman: {
    means: number[];
    diffs: number[];
    mean_diff: number;
    loa_upper: number;
    loa_lower: number;
  };
}

// ── ICCTab ────────────────────────────────────────────────────────────────────
function ICCTab({ sessionId, columns }: { sessionId: string; columns: string[] }) {
  const plotBg = usePlotBg();
  const showGrid = useStore((s) => s.showGrid);
  const blandAltmanRef = useRef<PlotCaptureHandle | null>(null);
  const [rater1, setRater1] = useState(columns[0] ?? "");
  const [rater2, setRater2] = useState(columns[1] ?? "");
  const [data, setData] = useState<ICCResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    if (!rater1 || !rater2 || rater1 === rater2) { setError("Select two different columns"); return; }
    setError("");
    setLoading(true);
    try {
      const res = await runICC({ session_id: sessionId, rater1_col: rater1, rater2_col: rater2 });
      setData(res.data as ICCResult);
    } catch (e: unknown) {
      setError(getErrorDetail(e));
    } finally {
      setLoading(false);
    }
  };

  const interpColor = (i: string) =>
    i === "Excellent" ? "text-green-600 font-bold" : i === "Good" ? "text-emerald-600 font-semibold" :
    i === "Moderate" ? "text-amber-600 font-semibold" : "text-red-500 font-bold";

  const exportICC = () => {
    if (!data) return;
    const header = ["ICC(2,1)", "95% CI Low", "95% CI High", "F stat", "p", "n", "Interpretation"];
    const row = [
      data.icc.toFixed(4),
      data.ci_low.toFixed(4),
      data.ci_high.toFixed(4),
      data.f_stat.toFixed(4),
      fmtP(data.f_p),
      String(data.n),
      data.interpretation,
    ];
    downloadCSV("icc_result.csv", [header, row]);
  };

  const leftCol = (
    <div className="panel space-y-3">
      <h3 className="text-sm font-semibold text-gray-700">
        ICC(2,1) — Absolute Agreement
        <Tip text="Intraclass Correlation Coefficient measures how consistently two raters measure the same subjects. ICC(2,1) is a two-way mixed model that tests both whether raters agree AND whether their absolute values match — stricter than consistency." wide />
      </h3>
      <p className="text-[11px] text-gray-400 leading-normal bg-indigo-50/50 p-2 rounded border border-indigo-100/50">
        Two-way mixed model for continuous inter-observer agreement (Shrout &amp; Fleiss, 1979).
      </p>
      <div className="space-y-1">
        <label className="text-xs text-gray-500 font-medium">Rater 1 Column</label>
        <select className="select w-full text-xs" value={rater1} onChange={(e) => setRater1(e.target.value)}>
          {columns.map((c) => <option key={c}>{c}</option>)}
        </select>
      </div>
      <div className="space-y-1">
        <label className="text-xs text-gray-500 font-medium">Rater 2 Column</label>
        <select className="select w-full text-xs" value={rater2} onChange={(e) => setRater2(e.target.value)}>
          {columns.map((c) => <option key={c}>{c}</option>)}
        </select>
      </div>
      <button className="btn-primary w-full mt-2" onClick={run} disabled={loading}>
        {loading ? "Computing…" : "Compute"}
      </button>
      {error && <p className="text-xs text-red-500">{error}</p>}
    </div>
  );

  const middleCol = data ? (
    <div className="panel">
      <TitledPlot
        plotRefOut={blandAltmanRef}
        storageKey="corr:icc:bland-altman"
        defaultTitle="Bland-Altman Plot"
        defaultSubtitle=""
        defaultXAxis={`Mean of ${rater1} & ${rater2}`}
        defaultYAxis={`${rater1} − ${rater2}`}
        data={[{
          type: "scatter", mode: "markers",
          x: data.bland_altman.means, y: data.bland_altman.diffs,
          marker: { color: _pal()[0], opacity: 0.7, size: 6 },
          name: "Subjects",
          hovertemplate: "Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>",
        }]}
        layout={{
          ...plotBg,
          autosize: true,
          xaxis: { title: `Mean of ${rater1} & ${rater2}`, gridcolor: "#e5e7eb", showgrid: showGrid, zeroline: false },
          yaxis: { title: `${rater1} − ${rater2}`, gridcolor: "#e5e7eb", showgrid: showGrid, zeroline: true, zerolinecolor: "#d1d5db" },
          shapes: [
            { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: data.bland_altman.mean_diff, y1: data.bland_altman.mean_diff, line: { color: "#f59e0b", width: 2 } },
            { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: data.bland_altman.loa_upper, y1: data.bland_altman.loa_upper, line: { color: "#ef4444", width: 1.5, dash: "dash" } },
            { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: data.bland_altman.loa_lower, y1: data.bland_altman.loa_lower, line: { color: "#ef4444", width: 1.5, dash: "dash" } },
          ],
          annotations: [
            { xref: "paper", yref: "y", x: 1.01, y: data.bland_altman.mean_diff, text: `Bias: ${data.bland_altman.mean_diff.toFixed(3)}`, showarrow: false, font: { color: "#b45309", size: 10 }, xanchor: "left" },
            { xref: "paper", yref: "y", x: 1.01, y: data.bland_altman.loa_upper, text: `+1.96 SD: ${data.bland_altman.loa_upper.toFixed(3)}`, showarrow: false, font: { color: "#dc2626", size: 10 }, xanchor: "left" },
            { xref: "paper", yref: "y", x: 1.01, y: data.bland_altman.loa_lower, text: `−1.96 SD: ${data.bland_altman.loa_lower.toFixed(3)}`, showarrow: false, font: { color: "#dc2626", size: 10 }, xanchor: "left" },
          ],
          margin: { t: 20, r: 130, b: 60, l: 60 },
          title: { text: "Bland-Altman Plot", font: { color: "#374151", size: 12 } },
        }}
        config={{ responsive: true, displayModeBar: false }}
      />
    </div>
  ) : (
    <div className="panel h-[480px] flex items-center justify-center text-gray-400 text-xs text-center border-dashed border-2">
      <div>
        <p className="text-2xl mb-2">📈</p>
        <p className="font-semibold text-gray-500">Bland-Altman Plot</p>
        <p className="text-[11px] text-gray-400 mt-1 max-w-xs">Select two different continuous rater columns and click Compute to view the Bland-Altman agreement plot with bias and limits of agreement (±1.96 SD).</p>
      </div>
    </div>
  );

  const rightCol = data ? (
    <div className="panel space-y-3 text-xs">
      <div className="flex items-center justify-between">
        <p className="text-gray-400 text-[10px] font-semibold uppercase tracking-wide">ICC(2,1) Result</p>
        <button onClick={exportICC} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50 hover:text-indigo-600 hover:border-indigo-300 transition-colors">↓ CSV</button>
      </div>
      <div className="bg-indigo-50 border border-indigo-100 rounded-xl p-3.5 text-center">
        <p className="text-[10px] font-semibold text-indigo-900 uppercase">Intraclass Correlation (ICC)</p>
        <p className="text-3xl font-bold font-mono text-indigo-700 mt-1">{data.icc.toFixed(3)}</p>
      </div>
      <div className="space-y-1.5 bg-gray-50 p-3 rounded-xl border border-gray-100">
        <p className="text-gray-500 flex justify-between">
          <span>95% Confidence Interval:</span>
          <span className="font-semibold font-mono text-gray-700">[{data.ci_low.toFixed(3)}, {data.ci_high.toFixed(3)}]</span>
        </p>
        <p className="text-gray-500 flex justify-between">
          <span>F-Statistic:</span>
          <span className="font-semibold font-mono text-gray-700">F({data.n - 1}, {data.n - 1}) = {data.f_stat.toFixed(2)}</span>
        </p>
        <p className="text-gray-500 flex justify-between">
          <span>Significance (<i>p</i>-value):</span>
          <span className="font-semibold font-mono text-gray-700">{fmtP(data.f_p)}</span>
        </p>
        <p className="text-gray-500 flex justify-between border-t border-gray-200/60 pt-1 mt-1">
          <span>Agreement Level:</span>
          <span className={interpColor(data.interpretation)}>{data.interpretation}</span>
        </p>
        <p className="text-gray-400 text-[10px] font-mono text-right mt-1"><i>n</i> = {data.n} subjects</p>
      </div>
      <InfoBanner>
        ICC = {data.icc.toFixed(3)} — {data.interpretation} agreement.{" "}
        {data.icc >= 0.75 ? "These two raters can be used interchangeably." : data.icc >= 0.5 ? "Agreement is acceptable but consider rater training." : "Poor agreement — do not treat the two raters as equivalent."}
      </InfoBanner>
      <div className="text-[10px] text-gray-400 pt-2 leading-tight border-t border-gray-100">
        <p className="font-semibold uppercase mb-1">Standard Criteria</p>
        <p>≥ 0.90 Excellent · ≥ 0.75 Good · ≥ 0.50 Moderate · &lt; 0.50 Poor</p>
      </div>
    </div>
  ) : (
    <div className="panel p-6 flex items-center justify-center text-gray-400 text-xs text-center border-dashed border-2">
      <div>
        <p className="text-2xl mb-2">📋</p>
        <p className="font-semibold text-gray-500">ICC Results Card</p>
        <p className="text-[11px] text-gray-400 mt-1 max-w-xs">Detailed F-statistics, confidence intervals, sample size, and clinical interpretation will be rendered here.</p>
      </div>
    </div>
  );

  return (
    <ThreeCol
      storageKey="CorrelationPanel.ICC"
      left={leftCol}
      middle={middleCol}
      right={rightCol}
    />
  );
}

interface KappaResult {
  kappa: number;
  ci_low: number;
  ci_high: number;
  se: number;
  n: number;
  interpretation: string;
  confusion_matrix: number[][];
  labels: string[];
}

// ── KappaTab ──────────────────────────────────────────────────────────────────
function KappaTab({ sessionId, columns }: { sessionId: string; columns: string[] }) {
  const plotBg = usePlotBg();
  const showGrid = useStore((s) => s.showGrid);
  const kappaMatrixRef = useRef<PlotCaptureHandle | null>(null);
  const [rater1, setRater1] = useState(columns[0] ?? "");
  const [rater2, setRater2] = useState(columns[1] ?? "");
  const [data, setData] = useState<KappaResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    if (!rater1 || !rater2 || rater1 === rater2) { setError("Select two different columns"); return; }
    setError("");
    setLoading(true);
    try {
      const res = await runCohensKappa({ session_id: sessionId, rater1_col: rater1, rater2_col: rater2 });
      setData(res.data as KappaResult);
    } catch (e: unknown) {
      setError(getErrorDetail(e));
    } finally {
      setLoading(false);
    }
  };

  const interpColor = (i: string) =>
    i === "Almost Perfect" ? "text-green-600 font-bold" : i === "Substantial" ? "text-emerald-600 font-semibold" :
    i === "Moderate" ? "text-amber-600 font-semibold" : i === "Fair" ? "text-orange-500 font-semibold" : "text-red-500 font-bold";

  const exportKappa = () => {
    if (!data) return;
    const header = ["κ", "95% CI Low", "95% CI High", "SE", "n", "Interpretation"];
    const row = [
      data.kappa.toFixed(4),
      data.ci_low.toFixed(4),
      data.ci_high.toFixed(4),
      data.se.toFixed(4),
      String(data.n),
      data.interpretation,
    ];
    downloadCSV("cohens_kappa_result.csv", [header, row]);
  };

  const leftCol = (
    <div className="panel space-y-3">
      <h3 className="text-sm font-semibold text-gray-700">
        Cohen's Kappa
        <Tip text="Cohen's κ measures agreement between two raters on categorical labels, correcting for agreement that would occur purely by chance. A κ of 0 means agreement no better than chance; 1 means perfect agreement." wide />
      </h3>
      <p className="text-[11px] text-gray-400 leading-normal bg-indigo-50/50 p-2 rounded border border-indigo-100/50">
        Categorical inter-observer agreement (Landis &amp; Koch, 1977).
      </p>
      <div className="space-y-1">
        <label className="text-xs text-gray-500 font-medium">Rater 1 Column</label>
        <select className="select w-full text-xs" value={rater1} onChange={(e) => setRater1(e.target.value)}>
          {columns.map((c) => <option key={c}>{c}</option>)}
        </select>
      </div>
      <div className="space-y-1">
        <label className="text-xs text-gray-500 font-medium">Rater 2 Column</label>
        <select className="select w-full text-xs" value={rater2} onChange={(e) => setRater2(e.target.value)}>
          {columns.map((c) => <option key={c}>{c}</option>)}
        </select>
      </div>
      <button className="btn-primary w-full mt-2" onClick={run} disabled={loading}>
        {loading ? "Computing…" : "Compute"}
      </button>
      {error && <p className="text-xs text-red-500">{error}</p>}
    </div>
  );

  const middleCol = data ? (
    <div className="panel">
      <TitledPlot
        plotRefOut={kappaMatrixRef}
        storageKey="corr:kappa:confusion"
        defaultTitle="Confusion Matrix"
        defaultSubtitle=""
        defaultXAxis=""
        defaultYAxis=""
        data={[{
          type: "heatmap",
          z: data.confusion_matrix,
          x: data.labels.map((l: string) => `Rater2: ${l}`),
          y: data.labels.map((l: string) => `Rater1: ${l}`),
          colorscale: [[0, "#f9fafb"], [1, _pal()[0]]],
          showscale: false,
          text: data.confusion_matrix.map((row: number[]) => row.map((v: number) => String(v))),
          texttemplate: "%{text}",
          hovertemplate: "Rater1=%{y}<br>Rater2=%{x}<br>Count=%{z}<extra></extra>",
        }]}
        layout={{
          ...plotBg,
          autosize: true,
          title: { text: "Confusion Matrix", font: { color: "#374151", size: 13 } },
          xaxis: { side: "bottom", showgrid: showGrid, gridcolor: "#e5e7eb", zeroline: false },
          yaxis: { showgrid: showGrid, gridcolor: "#e5e7eb", zeroline: false },
          margin: { t: 50, r: 20, b: 80, l: 100 },
        }}
        config={{ responsive: true, displayModeBar: false }}
      />
    </div>
  ) : (
    <div className="panel h-[480px] flex items-center justify-center text-gray-400 text-xs text-center border-dashed border-2">
      <div>
        <p className="text-2xl mb-2">🔲</p>
        <p className="font-semibold text-gray-500">Confusion Matrix Heatmap</p>
        <p className="text-[11px] text-gray-400 mt-1 max-w-xs">Select two different categorical/group columns and click Compute to view the confusion matrix agreement heatmap.</p>
      </div>
    </div>
  );

  const rightCol = data ? (
    <div className="panel space-y-3 text-xs">
      <div className="flex items-center justify-between">
        <p className="text-gray-400 text-[10px] font-semibold uppercase tracking-wide">κ Result</p>
        <button onClick={exportKappa} className="text-[10px] px-1.5 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-gray-50 hover:text-indigo-600 hover:border-indigo-300 transition-colors">↓ CSV</button>
      </div>
      <div className="bg-indigo-50 border border-indigo-100 rounded-xl p-3.5 text-center">
        <p className="text-[10px] font-semibold text-indigo-900 uppercase">Cohen's Kappa (κ)</p>
        <p className="text-3xl font-bold font-mono text-indigo-700 mt-1">{data.kappa.toFixed(3)}</p>
      </div>
      <div className="space-y-1.5 bg-gray-50 p-3 rounded-xl border border-gray-100">
        <p className="text-gray-500 flex justify-between">
          <span>95% Confidence Interval:</span>
          <span className="font-semibold font-mono text-gray-700">[{data.ci_low.toFixed(3)}, {data.ci_high.toFixed(3)}]</span>
        </p>
        <p className="text-gray-500 flex justify-between">
          <span>Standard Error (SE):</span>
          <span className="font-semibold font-mono text-gray-700">{data.se.toFixed(4)}</span>
        </p>
        <p className="text-gray-500 flex justify-between border-t border-gray-200/60 pt-1 mt-1">
          <span>Agreement Strength:</span>
          <span className={interpColor(data.interpretation)}>{data.interpretation}</span>
        </p>
        <p className="text-gray-400 text-[10px] font-mono text-right mt-1"><i>n</i> = {data.n} subjects</p>
      </div>
      <InfoBanner>
        κ = {data.kappa.toFixed(3)} ({data.interpretation}).{" "}
        {data.kappa > 0.8 ? "Excellent inter-rater reliability." : data.kappa > 0.6 ? "Good reliability — raters agree beyond chance most of the time." : data.kappa > 0.4 ? "Moderate reliability — training may help." : "Low reliability — review classification criteria."}
      </InfoBanner>
      <div className="text-[10px] text-gray-400 pt-2 leading-tight border-t border-gray-100">
        <p className="font-semibold uppercase mb-1">Standard Criteria</p>
        <p>&gt; 0.81 Almost Perfect · 0.61–0.80 Substantial · 0.41–0.60 Moderate · 0.21–0.40 Fair</p>
      </div>
    </div>
  ) : (
    <div className="panel p-6 flex items-center justify-center text-gray-400 text-xs text-center border-dashed border-2">
      <div>
        <p className="text-2xl mb-2">📋</p>
        <p className="font-semibold text-gray-500">Kappa Results Card</p>
        <p className="text-[11px] text-gray-400 mt-1 max-w-xs">Detailed standard error calculations, confidence intervals, sample size, and Landis-Koch interpretation will be rendered here.</p>
      </div>
    </div>
  );

  return (
    <ThreeCol
      storageKey="CorrelationPanel.Kappa"
      left={leftCol}
      middle={middleCol}
      right={rightCol}
    />
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function CorrelationPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <CorrelationPanelBody session={session} />;
}

function CorrelationPanelBody({ session }: { session: Session }) {
  const numColumns = session.columns
    .filter((c) => isNumericKind(c.kind) && !c.analysis_excluded)
    .map((c) => c.name);
  const allColumns = session.columns
    .filter((c) => !c.analysis_excluded)
    .map((c) => c.name);

  const [activeTab, setActiveTab] = useState<Tab>("Pairwise");

  return (
    <div className="flex flex-col h-full gap-3 p-4">
      <div className="flex gap-1 flex-shrink-0">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setActiveTab(t)}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              activeTab === t
                ? "bg-indigo-600 text-white shadow-sm"
                : "bg-white border border-gray-200 text-gray-500 hover:bg-gray-50 hover:text-gray-700"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="flex-1 min-h-0">
        {activeTab === "Pairwise"  && <PairwiseTab sessionId={session.session_id} columns={numColumns} />}
        {activeTab === "Matrix"    && <MatrixTab   sessionId={session.session_id} columns={numColumns} />}
        {activeTab === "ICC"       && <ICCTab      sessionId={session.session_id} columns={numColumns} />}
        {activeTab === "Cohen's κ" && <KappaTab    sessionId={session.session_id} columns={allColumns} />}
      </div>
    </div>
  );
}
