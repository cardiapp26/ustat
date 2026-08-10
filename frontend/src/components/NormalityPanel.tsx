/**
 * NormalityPanel — is this variable normally distributed, in the cohort and
 * within each group?
 *
 * Deliberately shows three kinds of evidence side by side rather than a single
 * verdict, because that is what the guidance actually says to do (Ghasemi &
 * Zahediasl, Int J Endocrinol Metab 2012; Kim, Restor Dent Endod 2013): a
 * formal test, the shape statistics with their z-scores, and the Q-Q plot. The
 * formal test alone is misleading at both ends of the sample-size range — under
 * ~20 it has no power, over ~300 it rejects on departures nobody would notice.
 *
 * The pooled sample is always reported next to the groups. What a t-test or
 * ANOVA assumes is normality *within* each group; the pooled sample of two
 * groups that differ is a mixture and can fail for that reason alone.
 */
import { useMemo, useRef, useState } from "react";
import Plot from "../PlotComponent";
import { usePlotLayout, usePalette } from "../plotStyle";
import { analysisCols, isNumericKind, isCategoricalKind, useStore, type Session } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { runNormality } from "../api";
import { Tip } from "./Tip";
import ThreeCol from "./ThreeCol";
import ResultExporter from "./ResultExporter";
import { fmtP, pCellTitle } from "../lib/format";
import type { Data, Layout } from "plotly.js";
import type { PlotData, PlotLayout, PlotCaptureHandle } from "../lib/plotTypes";

interface TestRow {
  id: string;
  name: string;
  stat: number | null;
  p: number | null;
  applicable: boolean;
  note: string;
}
interface Shape {
  n: number;
  skewness: number | null;
  skew_se: number | null;
  skew_z: number | null;
  kurtosis: number | null;
  kurt_se: number | null;
  kurt_z: number | null;
}
interface Verdict {
  code: "normal" | "borderline" | "non_normal" | "undetermined";
  label: string;
  reason: string;
  notes: string[];
}
export interface NormalityBlock {
  label: string;
  n: number;
  n_missing: number;
  n_total: number;
  constant: boolean;
  mean?: number | null;
  sd?: number | null;
  median?: number | null;
  q1?: number | null;
  q3?: number | null;
  min?: number | null;
  max?: number | null;
  shape: Shape | Record<string, never>;
  shape_flag: boolean | null;
  tests: TestRow[];
  primary: TestRow | null;
  verdict: Verdict;
  qq: { theoretical?: (number | null)[]; sample?: (number | null)[]; line?: { slope: number; intercept: number }; thinned?: boolean };
  histogram: { bin_edges?: (number | null)[]; counts?: number[]; curve_x?: (number | null)[]; curve_y?: (number | null)[] };
  sentence: string;
}
interface VariableResult {
  variable: string;
  overall: NormalityBlock;
  groups: NormalityBlock[];
  group_summary?: string;
}
interface NormalityResult {
  alpha: number;
  group_column: string | null;
  group_levels: string[];
  variables: VariableResult[];
  warnings: string[];
  guidance: string;
}

const VERDICT_STYLE: Record<Verdict["code"], string> = {
  normal: "bg-green-50 text-green-700 border-green-200",
  borderline: "bg-amber-50 text-amber-700 border-amber-200",
  non_normal: "bg-red-50 text-red-700 border-red-200",
  undetermined: "bg-gray-50 text-gray-500 border-gray-200",
};

/** Short forms for the table, which has to fit beside the plots and the
 *  right-hand column. The full name is in the tooltip and in "All tests". */
const TEST_ABBR: Record<string, string> = {
  shapiro: "S-W", anderson: "A-D", lilliefors: "KS-L",
  dagostino: "K²", jarque_bera: "JB",
};
const VERDICT_SHORT: Record<Verdict["code"], string> = {
  normal: "Normal", borderline: "Borderline",
  non_normal: "Departs", undetermined: "—",
};

const num = (v: number | null | undefined, d = 3) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(d);

/** Every (variable, sample) pair as one flat row — the table and the export
 *  read the same list, so what is on screen is what leaves the app. */
function flatten(result: NormalityResult): { variable: string; block: NormalityBlock }[] {
  return result.variables.flatMap((v) => [
    { variable: v.variable, block: v.overall },
    ...v.groups.map((g) => ({ variable: v.variable, block: g })),
  ]);
}

export default function NormalityPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <NormalityPanelBody session={session} />;
}

function NormalityPanelBody({ session }: { session: Session }) {
  const themedBase = usePlotLayout();
  const palette = usePalette();
  const cols = useMemo(() => analysisCols(session.columns), [session.columns]);
  const numCols = useMemo(() => cols.filter((c) => isNumericKind(c.kind)).map((c) => c.name), [cols]);
  const catCols = useMemo(() => cols.filter((c) => isCategoricalKind(c.kind)).map((c) => c.name), [cols]);

  const [variables, setVariables] = usePersistedPanelState<string[]>("normality", "vars", []);
  const [groupCol, setGroupCol] = usePersistedPanelState<string>("normality", "group", "");
  const [alpha, setAlpha] = usePersistedPanelState<number>("normality", "alpha", 0.05);

  const [result, setResult] = useState<NormalityResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [focusKey, setFocusKey] = useState<string>("");
  const qqRef = useRef<PlotCaptureHandle | null>(null);
  const histRef = useRef<PlotCaptureHandle | null>(null);

  const toggleVar = (c: string) =>
    setVariables((prev) => (prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]));

  const run = async () => {
    if (!variables.length) { setError("Select at least one numeric variable."); return; }
    setLoading(true); setError(null);
    try {
      const res = await runNormality({
        session_id: session.session_id,
        variables,
        group_column: groupCol || undefined,
        alpha,
      });
      const data = res.data as NormalityResult;
      setResult(data);
      const first = flatten(data)[0];
      setFocusKey(first ? `${first.variable}||${first.block.label}` : "");
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      setError(typeof msg === "string" ? msg : e instanceof Error ? e.message : "Analysis failed");
      setResult(null);
    } finally { setLoading(false); }
  };

  const rows = result ? flatten(result) : [];
  const focus = rows.find((r) => `${r.variable}||${r.block.label}` === focusKey) ?? rows[0];

  const qqTraces: PlotData[] = useMemo(() => {
    const qq = focus?.block.qq;
    if (!qq?.sample?.length || !qq.theoretical?.length) return [];
    const traces: PlotData[] = [{
      x: qq.theoretical as number[],
      y: qq.sample as number[],
      type: "scatter", mode: "markers", name: "Observed",
      marker: { size: 6, color: palette[0], opacity: 0.75 },
      hovertemplate: "theoretical %{x:.3f}<br>observed %{y:.4g}<extra></extra>",
    }];
    if (qq.line) {
      // R's qqline: through the two quartiles, not mean ± SD. A line fitted to
      // the ends would bend toward the outliers it is meant to expose.
      const xs = (qq.theoretical as number[]).filter((v) => Number.isFinite(v));
      const lo = Math.min(...xs), hi = Math.max(...xs);
      traces.push({
        x: [lo, hi],
        y: [qq.line.slope * lo + qq.line.intercept, qq.line.slope * hi + qq.line.intercept],
        type: "scatter", mode: "lines", name: "Normal reference",
        line: { color: palette[1] ?? "#dc2626", width: 2, dash: "dash" },
        hoverinfo: "skip",
      });
    }
    return traces;
  }, [focus, palette]);

  const histTraces: PlotData[] = useMemo(() => {
    const h = focus?.block.histogram;
    if (!h?.counts?.length || !h.bin_edges?.length) return [];
    const edges = h.bin_edges as number[];
    const centres = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
    const width = edges.length > 1 ? edges[1] - edges[0] : 1;
    return [
      {
        x: centres, y: h.counts, type: "bar", name: "Observed",
        width: Array(centres.length).fill(width * 0.96),
        marker: { color: palette[0], opacity: 0.65 },
        hovertemplate: "%{y} obs<extra></extra>",
      },
      {
        x: h.curve_x as number[], y: h.curve_y as number[],
        type: "scatter", mode: "lines", name: "Normal curve",
        line: { color: palette[1] ?? "#dc2626", width: 2 },
        hoverinfo: "skip",
      },
    ];
  }, [focus, palette]);

  const qqLayout: PlotLayout = {
    ...themedBase,
    height: 300,
    margin: { t: 10, r: 12, b: 44, l: 56 },
    xaxis: { title: { text: "Theoretical quantiles" }, zeroline: false },
    yaxis: { title: { text: "Sample quantiles" }, zeroline: false },
    showlegend: false,
  };
  const histLayout: PlotLayout = {
    ...themedBase,
    height: 300,
    margin: { t: 10, r: 12, b: 44, l: 48 },
    xaxis: { title: { text: focus?.variable ?? "" }, zeroline: false },
    yaxis: { title: { text: "Count" }, zeroline: false },
    bargap: 0.02,
    showlegend: false,
  };

  const exportHeaders = [
    "Variable", "Sample", "n", "Missing", "Mean", "SD", "Median",
    "Skewness", "z(skew)", "Excess kurtosis", "z(kurt)",
    "Test", "Statistic", "p", "Verdict",
  ];
  const exportRows = rows.map(({ variable, block }) => {
    const s = block.shape as Shape;
    return [
      variable, block.label, String(block.n), String(block.n_missing),
      num(block.mean), num(block.sd), num(block.median),
      num(s?.skewness), num(s?.skew_z, 2), num(s?.kurtosis), num(s?.kurt_z, 2),
      block.primary?.name ?? "—", num(block.primary?.stat), fmtP(block.primary?.p ?? null),
      block.verdict.label,
    ];
  });

  return (
    <ThreeCol
      storageKey="NormalityPanel"
      left={
        <div className="panel space-y-3">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-gray-900">Normality</h3>
            <Tip wide text="Checks whether a continuous variable is normally distributed — for the whole cohort and, if you pick a grouping variable, within each group separately. Report the test, the shape statistics and the Q-Q plot together: no single one of them settles it." />
          </div>

          <div className="space-y-1">
            <span className="text-xs font-medium text-gray-500">Variables</span>
            <div className="max-h-56 overflow-y-auto rounded-lg border border-gray-200 p-1.5 space-y-0.5">
              {numCols.length === 0 && <p className="text-xs text-gray-400 px-1 py-2">No numeric columns.</p>}
              {numCols.map((c) => (
                <label key={c} className="flex items-center gap-2 px-1 py-0.5 text-xs text-gray-700 hover:bg-gray-50 rounded cursor-pointer">
                  <input type="checkbox" checked={variables.includes(c)} onChange={() => toggleVar(c)} />
                  <span className="truncate">{c}</span>
                </label>
              ))}
            </div>
          </div>

          <label className="block space-y-1">
            <span className="text-xs font-medium text-gray-500 flex items-center gap-1">
              Group by (optional)
              <Tip text="The t-test and ANOVA assume normality WITHIN each group. A pooled sample of two groups that differ is a mixture and can fail normality for that reason alone — so check the groups when you are about to compare them." />
            </span>
            <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
              value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
              <option value="">— whole cohort only —</option>
              {catCols.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>

          <label className="block space-y-1">
            <span className="text-xs font-medium text-gray-500">Significance level (α)</span>
            <select className="w-full rounded-lg border border-gray-300 px-2 py-1.5 text-sm bg-white"
              value={alpha} onChange={(e) => setAlpha(Number(e.target.value))}>
              {[0.01, 0.05, 0.10].map((a) => <option key={a} value={a}>{a.toFixed(2)}</option>)}
            </select>
          </label>

          <button className="btn-primary w-full" onClick={run} disabled={loading || !variables.length}>
            {loading ? "Testing…" : "Assess normality"}
          </button>
          {error && <p className="text-xs text-red-500">{error}</p>}

          <div className="rounded-lg border border-indigo-100 bg-indigo-50/60 p-2.5 space-y-1.5">
            <p className="text-[10px] font-bold uppercase text-indigo-900">How this is judged</p>
            <p className="text-[11px] leading-relaxed text-indigo-800">
              Shapiro-Wilk is the primary test (most powerful across the shapes met in
              clinical data). Anderson-Darling, Lilliefors KS, D'Agostino-Pearson K² and
              Jarque-Bera are shown alongside it.
            </p>
            <p className="text-[11px] leading-relaxed text-indigo-800">
              Shape: |z| &gt; 1.96 flags a departure under n = 50, &gt; 3.29 up to n = 300;
              beyond that the absolute values are used (|skew| &gt; 2, |excess kurtosis| &gt; 7).
            </p>
          </div>
        </div>
      }
      middle={
        result ? (
          <div className="space-y-3">
            <div className="panel">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold text-gray-700">
                  Normality assessment{result.group_column ? ` by ${result.group_column}` : ""}
                </h4>
                <ResultExporter title="Normality" headers={exportHeaders} rows={exportRows} />
              </div>
              <div className="overflow-auto rounded border border-gray-200">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-gray-50 text-gray-500">
                      {/* One column, not two: the variable name only ever appears on
                          its pooled row, so a separate "Sample" column was an empty
                          cell on every group row and 56px the verdict badge needed. */}
                      <th className="px-1.5 py-1.5 text-left">Variable / group</th>
                      <th className="px-1.5 py-1.5 text-right" title="Non-missing count — every statistic in the row uses it. Missing follows in grey.">
                        <i>n</i> <span className="font-normal text-gray-400">(miss)</span>
                      </th>
                      <th className="px-1.5 py-1.5 text-right">Skew (z)</th>
                      <th className="px-1.5 py-1.5 text-right">Kurt (z)</th>
                      <th className="px-1.5 py-1.5 text-left">Test</th>
                      <th className="px-1.5 py-1.5 text-right"><i>p</i></th>
                      <th className="px-1.5 py-1.5 text-left">Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(({ variable, block }) => {
                      const key = `${variable}||${block.label}`;
                      const s = block.shape as Shape;
                      const pooled = block.label === "All (pooled)";
                      return (
                        <tr key={key}
                          onClick={() => setFocusKey(key)}
                          className={`border-t border-gray-100 cursor-pointer hover:bg-indigo-50/50 ${
                            focusKey === key ? "bg-indigo-50" : ""}`}>
                          <td title={pooled ? `${variable} — ${block.label}` : `${variable} — group ${block.label}`}
                            className={`px-1.5 py-1.5 whitespace-nowrap ${
                              pooled ? "font-medium text-gray-700" : "pl-5 text-gray-500"}`}>
                            {pooled ? variable : block.label}
                          </td>
                          <td className="px-1.5 py-1.5 text-right font-mono whitespace-nowrap">
                            {block.n}
                            {block.n_missing > 0 && <span className="text-gray-400"> ({block.n_missing})</span>}
                          </td>
                          <td className="px-1.5 py-1.5 text-right font-mono">
                            {num(s?.skewness, 2)}<span className="text-gray-400"> ({num(s?.skew_z, 1)})</span>
                          </td>
                          <td className="px-1.5 py-1.5 text-right font-mono">
                            {num(s?.kurtosis, 2)}<span className="text-gray-400"> ({num(s?.kurt_z, 1)})</span>
                          </td>
                          <td className="px-1.5 py-1.5 text-gray-600"
                            title={block.primary ? `${block.primary.name}, statistic = ${num(block.primary.stat)}` : "no applicable test"}>
                            {block.primary ? TEST_ABBR[block.primary.id] ?? block.primary.name : "—"}
                          </td>
                          <td className="px-1.5 py-1.5 text-right font-mono" title={pCellTitle(block.primary?.p ?? null)}>
                            {fmtP(block.primary?.p ?? null)}
                          </td>
                          <td className="px-1.5 py-1.5">
                            <span title={block.verdict.label}
                              className={`inline-block whitespace-nowrap rounded border px-1.5 py-0.5 text-[10px] ${VERDICT_STYLE[block.verdict.code]}`}>
                              {VERDICT_SHORT[block.verdict.code]}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="mt-1.5 text-[10px] text-gray-400">
                Click a row to plot it. Skewness and excess kurtosis are the SPSS/G1-G2
                estimators, with their z-scores in brackets. The test statistic is in the
                cell tooltip and in the panel on the right.
              </p>
            </div>

            {focus && (
              <div className="panel">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-semibold text-gray-700">
                    {focus.variable} — {focus.block.label}
                  </h4>
                  <span className={`rounded border px-2 py-0.5 text-[11px] ${VERDICT_STYLE[focus.block.verdict.code]}`}>
                    {focus.block.verdict.label}
                  </span>
                </div>
                <div className="grid gap-3 lg:grid-cols-2">
                  <div>
                    {/* Each figure carries its own export control: two toolbars in
                        the card header crowded the title and gave no clue which
                        one saved which plot. */}
                    <div className="mb-1 flex items-center justify-between gap-2">
                      <p className="text-[11px] font-medium text-gray-500">Normal Q-Q plot</p>
                      <ResultExporter title={`QQ_${focus.variable}_${focus.block.label}`} plotRef={qqRef} />
                    </div>
                    <Plot data={qqTraces as unknown as Data[]} layout={qqLayout as unknown as Partial<Layout>}
                      config={{ displaylogo: false, responsive: true, displayModeBar: false }}
                      onInitialized={(_: object, gd: HTMLElement) => { qqRef.current = gd as unknown as PlotCaptureHandle; }}
                      onUpdate={(_: object, gd: HTMLElement) => { qqRef.current = gd as unknown as PlotCaptureHandle; }}
                      style={{ width: "100%" }} useResizeHandler />
                    <p className="mt-1 text-[10px] text-gray-400">
                      Points on the dashed line = normal. Curvature at the ends is a tail
                      departure; an S-shape is skew.
                      {focus.block.qq?.thinned && " Thinned for display."}
                    </p>
                  </div>
                  <div>
                    <div className="mb-1 flex items-center justify-between gap-2">
                      <p className="text-[11px] font-medium text-gray-500">Histogram with fitted normal</p>
                      <ResultExporter title={`Histogram_${focus.variable}_${focus.block.label}`} plotRef={histRef} />
                    </div>
                    <Plot data={histTraces as unknown as Data[]} layout={histLayout as unknown as Partial<Layout>}
                      config={{ displaylogo: false, responsive: true, displayModeBar: false }}
                      onInitialized={(_: object, gd: HTMLElement) => { histRef.current = gd as unknown as PlotCaptureHandle; }}
                      onUpdate={(_: object, gd: HTMLElement) => { histRef.current = gd as unknown as PlotCaptureHandle; }}
                      style={{ width: "100%" }} useResizeHandler />
                    <p className="mt-1 text-[10px] text-gray-400">
                      Curve is the normal density with this sample's mean and SD, scaled to
                      the bin counts.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="panel py-12 text-center text-gray-400">
            <p className="mb-2 text-lg">📈</p>
            <p>Pick the variables to check on the left.</p>
            <p className="mt-2 text-xs">
              Add a grouping variable to check normality within each group — that, not the
              pooled sample, is what a t-test or ANOVA assumes.
            </p>
          </div>
        )
      }
      right={
        result && focus ? (
          <div className="space-y-3">
            <div className="panel space-y-2">
              <h4 className="text-sm font-semibold text-gray-700">All tests — {focus.block.label}</h4>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-400">
                    <th className="py-1 text-left font-medium">Test</th>
                    <th className="py-1 text-right font-medium">Statistic</th>
                    <th className="py-1 text-right font-medium"><i>p</i></th>
                  </tr>
                </thead>
                <tbody>
                  {focus.block.tests.map((t) => (
                    <tr key={t.id} className={`border-t border-gray-100 ${t.id === focus.block.primary?.id ? "font-semibold" : ""}`}>
                      <td className="py-1 text-gray-600">
                        {t.name}
                        {!t.applicable && <span className="ml-1 text-[10px] text-gray-400">({t.note})</span>}
                      </td>
                      <td className="py-1 text-right font-mono">{num(t.stat)}</td>
                      <td className="py-1 text-right font-mono" title={pCellTitle(t.p)}>{fmtP(t.p)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-[10px] text-gray-400">
                The primary test is in bold. The others are shown so a departure that only
                one test catches is visible rather than hidden.
              </p>
            </div>

            <div className="panel space-y-2">
              <h4 className="text-sm font-semibold text-gray-700">Reading it</h4>
              <p className="text-xs text-gray-600">{focus.block.verdict.reason}</p>
              {focus.block.verdict.notes.map((n) => (
                <p key={n} className="rounded bg-amber-50 px-2 py-1.5 text-[11px] leading-relaxed text-amber-800">{n}</p>
              ))}
              <div className="grid grid-cols-3 gap-1.5 text-center">
                {[["Mean", focus.block.mean], ["SD", focus.block.sd], ["Median", focus.block.median],
                  ["Q1", focus.block.q1], ["Q3", focus.block.q3], ["Range", null]].map(([label, v], i) => (
                  <div key={label as string} className="rounded bg-gray-50 px-1.5 py-1">
                    <p className="text-[10px] text-gray-400">{label}</p>
                    <p className="font-mono text-[11px] text-gray-700">
                      {i === 5 ? `${num(focus.block.min, 1)}–${num(focus.block.max, 1)}` : num(v as number | null)}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div className="panel space-y-2">
              <h4 className="text-sm font-semibold text-gray-700">For the manuscript</h4>
              {rows.map(({ variable, block }) => (
                <p key={`${variable}||${block.label}`} className="rounded border border-indigo-100 bg-white px-2 py-1.5 text-[11px] leading-relaxed text-gray-600">
                  {block.sentence}
                </p>
              ))}
              <p className="text-[11px] leading-relaxed text-gray-500">{result.guidance}</p>
              {result.warnings.map((w) => (
                <p key={w} className="rounded bg-amber-50 px-2 py-1.5 text-[11px] text-amber-800">{w}</p>
              ))}
            </div>
          </div>
        ) : (
          <div className="panel text-xs text-gray-400">
            Results, all five tests and a paste-ready sentence appear here.
          </div>
        )
      }
    />
  );
}
