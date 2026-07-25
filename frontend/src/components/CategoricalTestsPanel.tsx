import { useState } from "react";
import { useStore, isNumericKind, isCategoricalKind, type Session } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { runBinomial, runOneProportion, runTwoProportions, runMcNemar, runCochranQ, runMantelHaenszel, runCochranArmitage } from "../api";
import { fmtP } from "../lib/format";
// ResultExporter available via ResultCard pattern

/** True when a stat-grid key holds a p-value (route through the canonical fmtP). */
function isPKey(k: string): boolean {
  return /^p$|^p_?value$|_p$|_p_?value$/.test(k);
}

const TESTS = [
  { id: "binomial",       label: "Binomial test",         group: "One-sample" },
  { id: "one_prop",       label: "One proportion z-test", group: "One-sample" },
  { id: "two_prop",       label: "Two proportions z-test", group: "Two-sample" },
  { id: "mcnemar",        label: "McNemar test",          group: "Paired" },
  { id: "cochran_q",      label: "Cochran's Q",           group: "Paired" },
  { id: "mantel_haenszel", label: "Mantel-Haenszel",      group: "Stratified" },
  { id: "cochran_armitage", label: "Cochran-Armitage trend", group: "Trend" },
] as const;

const GUIDANCE: Record<string, { when: string; reading: string }> = {
  binomial:  { when: "Test whether the observed proportion differs from an expected proportion (e.g. 50% heads).", reading: "p < 0.05 means the observed proportion significantly differs from the expected." },
  one_prop:  { when: "z-test version of the binomial test, using normal approximation. Better for larger samples (n > 30).", reading: "Report: z, p, observed proportion, and 95% CI." },
  two_prop:  { when: "Compare proportions between two independent groups (e.g. treatment vs control event rates).", reading: "Cohen's h measures the effect size. Report proportions, z, p, and h." },
  mcnemar:   { when: "Test change in a binary outcome for paired data (e.g. before/after intervention on the same patients).", reading: "Tests whether discordant pairs (changed responses) are symmetric. OR of discordant pairs is the effect size." },
  cochran_q: { when: "Extension of McNemar for 3+ related binary measures. Tests whether proportions differ across conditions.", reading: "Significant Q means at least one proportion differs. Follow up with pairwise McNemar (Holm-corrected)." },
  mantel_haenszel: { when: "Test association between two binary variables while controlling for a stratifying variable (e.g. hospital site).", reading: "Common OR summarises the overall effect across strata. Homogeneity test checks whether the OR is consistent." },
  cochran_armitage: { when: "Test for a monotone linear trend in the proportion of a binary outcome across 3+ ordered groups (e.g. dose levels 0/1/2/3 vs adverse event).", reading: "Significant Z = the proportion changes linearly across the ordered groups. Sign of Z indicates direction (positive = increasing, negative = decreasing)." },
};

interface EffectSize {
  name?: string;
  value?: number;
  ci_low?: number;
  ci_high?: number;
  magnitude?: string;
}

interface PostHocRow {
  group1?: string;
  group2?: string;
  p_adj?: number;
  significant?: boolean;
}

interface CategoricalResult {
  test?: string;
  interpretation?: string;
  result_text?: string;
  significant?: boolean;
  effect_sizes?: EffectSize[];
  posthoc?: PostHocRow[];
  posthoc_method?: string;
  r_code?: string;
  warnings?: string[];
  [key: string]: unknown;
}

function ResultCard({ result }: { result: CategoricalResult }) {
  const fmt = (v: unknown) => typeof v !== "number" ? String(v ?? "") : Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(3) : v.toFixed(4);
  const skip = new Set(["test","interpretation","result_text","significant","effect_sizes","assumptions","warnings","summary","posthoc","posthoc_method","export_rows","r_code","effects","table","row_labels","col_labels","plot_data","crosstab","strata_tables"]);
  const stats = Object.entries(result).filter(([k, v]) => !skip.has(k) && typeof v !== "object");
  return (
    <div className="panel space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-gray-900">{result.test}</h4>
        {"significant" in result && <span className={result.significant ? "badge-sig" : "badge-ns"}>{result.significant ? "Significant" : "Not significant"}</span>}
      </div>
      <p className="text-sm text-gray-500 italic">{result.interpretation}</p>
      {/* Warnings can invert the reading of a result (e.g. an assumed level
          ordering flips the trend direction), so they sit above the numbers. */}
      {(result.warnings?.length ?? 0) > 0 && (
        <div className="space-y-1">
          {(result.warnings ?? []).map((w, i) => (
            <p key={i} className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-xs text-amber-800 leading-relaxed">
              ⚠ {w}
            </p>
          ))}
        </div>
      )}
      <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm">
        {stats.map(([k, v]) => (
          <div key={k} className="flex justify-between border-b border-gray-100 py-1">
            <span className="text-gray-400">{k}</span><span className="text-gray-700 font-mono">{isPKey(k) ? fmtP(v as number | null | undefined) : fmt(v)}</span>
          </div>
        ))}
      </div>
      {(result.effect_sizes?.length ?? 0) > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-semibold text-gray-600">Effect Sizes</p>
          {(result.effect_sizes ?? []).map((es: EffectSize, i: number) => (
            <div key={i} className="flex items-center gap-3 bg-indigo-50 rounded-lg px-3 py-1.5 text-xs">
              <span className="font-semibold text-indigo-800">{es.name?.replace(/_/g, " ")}</span>
              <span className="font-mono text-indigo-700">{es.value?.toFixed(3)}</span>
              {es.ci_low != null && <span className="text-indigo-500">95% CI: [{es.ci_low?.toFixed(3)}, {es.ci_high?.toFixed(3)}]</span>}
              {es.magnitude && <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${es.magnitude==="large"?"bg-red-100 text-red-700":es.magnitude==="medium"?"bg-amber-100 text-amber-700":"bg-blue-100 text-blue-700"}`}>{es.magnitude}</span>}
            </div>
          ))}
        </div>
      )}
      {(result.posthoc?.length ?? 0) > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-600 mb-1">Post-hoc: {result.posthoc_method ?? "Pairwise"}</p>
          <div className="overflow-auto rounded border border-gray-200">
            <table className="w-full text-xs"><thead><tr className="bg-gray-50">
              <th className="px-2 py-1 text-left">Comparison</th><th className="px-2 py-1 text-right"><i>p</i> (adj)</th><th className="px-2 py-1 text-center">Sig</th>
            </tr></thead><tbody>
              {(result.posthoc ?? []).map((ph: PostHocRow, i: number) => (
                <tr key={i} className={`border-t border-gray-100 ${ph.significant?"":"text-gray-400"}`}>
                  <td className="px-2 py-1">{ph.group1} vs {ph.group2}</td>
                  <td className="px-2 py-1 text-right font-mono">{fmtP(ph.p_adj)}</td>
                  <td className="px-2 py-1 text-center">{ph.significant?"\u2713":"\u2014"}</td>
                </tr>
              ))}
            </tbody></table>
          </div>
        </div>
      )}
      {result.result_text && (
        <div className="rounded-lg border border-indigo-100 bg-white px-3 py-2 text-xs text-gray-600 leading-relaxed">
          <span className="text-indigo-400 mr-1">\uD83D\uDCAC</span> {result.result_text}
        </div>
      )}
      {result.r_code && (
        <details className="text-xs"><summary className="text-gray-400 cursor-pointer hover:text-indigo-600">R code</summary>
          <pre className="mt-1 p-2 bg-gray-50 rounded-lg text-gray-600 font-mono text-[10px] whitespace-pre-wrap">{result.r_code}</pre>
        </details>
      )}
    </div>
  );
}

export default function CategoricalTestsPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <CategoricalTestsPanelBody session={session} />;
}

function CategoricalTestsPanelBody({ session }: { session: Session }) {
  const numCols = session.columns.filter((c) => isNumericKind(c.kind) && !c.analysis_excluded).map((c) => c.name);
  const catCols = session.columns.filter((c) => isCategoricalKind(c.kind) && !c.analysis_excluded).map((c) => c.name);
  const binCols = [...catCols, ...numCols]; // binary cols could be either

  const [test, setTest] = usePersistedPanelState<string>("categorical_tests", "test", "binomial");
  const [col, setCol] = usePersistedPanelState<string>("categorical_tests", "col", binCols[0] ?? "");
  const [col2, setCol2] = usePersistedPanelState<string>("categorical_tests", "col2", binCols[1] ?? binCols[0] ?? "");
  const [groupCol, setGroupCol] = usePersistedPanelState<string>("categorical_tests", "groupCol", catCols[0] ?? "");
  const [strataCol, setStrataCol] = usePersistedPanelState<string>("categorical_tests", "strataCol", catCols[1] ?? catCols[0] ?? "");
  const [nullProp, setNullProp] = usePersistedPanelState<string>("categorical_tests", "nullProp", "0.5");
  // Low→high ordering for word-labelled exposures. Left blank the backend sorts
  // alphabetically, which reverses e.g. Low/Medium/High and flips the trend.
  const [levelOrder, setLevelOrder] = usePersistedPanelState<string>("categorical_tests", "levelOrder", "");
  const [friedmanCols, setFriedmanCols] = usePersistedPanelState<string[]>("categorical_tests", "friedmanCols", []);
  const [result, setResult] = useState<CategoricalResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const isPaired = test === "mcnemar";
  const isCochran = test === "cochran_q";
  const isMH = test === "mantel_haenszel";
  const isTwoProp = test === "two_prop";
  const isCA = test === "cochran_armitage";
  const needsNull = test === "binomial" || test === "one_prop";
  // Ordinal exposure + binary outcome → Cochran-Armitage tests a dose-response
  // trend that a plain proportion/χ² test ignores.
  const ordinalNames = new Set(session.columns.filter((c) => c.kind === "ordinal").map((c) => c.name));
  const suggestTrend = !isCA && [col, col2, groupCol].some((v) => ordinalNames.has(v));

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    const sid = session.session_id;
    try {
      let res: { data?: CategoricalResult } | null = null;
      if (test === "binomial") res = await runBinomial({ session_id: sid, column: col, expected_proportion: +nullProp });
      else if (test === "one_prop") res = await runOneProportion({ session_id: sid, column: col, null_proportion: +nullProp });
      else if (test === "two_prop") res = await runTwoProportions({ session_id: sid, column: col, group_column: groupCol });
      else if (test === "mcnemar") res = await runMcNemar({ session_id: sid, col1: col, col2: col2 });
      else if (test === "cochran_q") res = await runCochranQ({ session_id: sid, columns: friedmanCols });
      else if (test === "mantel_haenszel") res = await runMantelHaenszel({ session_id: sid, row_col: col, col_col: col2, strata_col: strataCol });
      else if (test === "cochran_armitage") {
        const order = levelOrder.split(",").map((s) => s.trim()).filter(Boolean);
        res = await runCochranArmitage({
          session_id: sid, ordinal_col: groupCol, event_col: col,
          ...(order.length > 0 ? { level_order: order } : {}),
        });
      }
      setResult(res?.data ?? null);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail ?? "Error");
    }
    finally { setLoading(false); }
  };

  const g = GUIDANCE[test];
  return (
    <div className="flex gap-4">
      <div className="w-64 flex-shrink-0 space-y-4">
        <div className="panel space-y-1">
          {["One-sample", "Two-sample", "Paired", "Stratified", "Trend"].map((grp) => (
            <div key={grp}>
              <p className="text-xs text-gray-400 uppercase tracking-wider mt-3 mb-1 first:mt-0">{grp}</p>
              {TESTS.filter((t) => t.group === grp).map(({ id, label }) => (
                <label key={id} className="flex items-center gap-2 cursor-pointer py-0.5">
                  <input type="radio" name="cat_test" value={id} checked={test === id}
                    onChange={() => { setTest(id); setResult(null); }} className="accent-indigo-500" />
                  <span className="text-sm text-gray-700">{label}</span>
                </label>
              ))}
            </div>
          ))}
        </div>
        {suggestTrend && (
          <div className="text-[10px] text-teal-700 bg-teal-50 border border-teal-200 rounded px-2 py-1.5 leading-snug flex items-start gap-1.5">
            <span className="flex-1">
              Ordinal variable selected — for an ordered exposure vs a binary outcome,
              <strong> Cochran-Armitage trend</strong> tests dose-response that χ²/proportion tests miss.
            </span>
            <button onClick={() => { setTest("cochran_armitage"); setResult(null); }} className="flex-shrink-0 underline hover:text-teal-900">
              Use trend
            </button>
          </div>
        )}
        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Variables</h3>
          <div>
            <label className="text-xs text-gray-400 block mb-1">{isMH ? "Row variable" : isCA ? "Binary outcome (event)" : "Binary column"}</label>
            <select className="select w-full" value={col} onChange={(e) => setCol(e.target.value)}>
              {binCols.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          {needsNull && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Expected proportion</label>
              <input className="select w-full" type="number" step="0.01" min="0" max="1" value={nullProp} onChange={(e) => setNullProp(e.target.value)} />
            </div>
          )}
          {isTwoProp && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Group column</label>
              <select className="select w-full" value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
                {catCols.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}
          {(isPaired || isMH) && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">{isMH ? "Column variable" : "Second measurement"}</label>
              <select className="select w-full" value={col2} onChange={(e) => setCol2(e.target.value)}>
                {binCols.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}
          {isMH && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Stratifying variable</label>
              <select className="select w-full" value={strataCol} onChange={(e) => setStrataCol(e.target.value)}>
                {catCols.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}
          {isCochran && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Binary columns (3+)</label>
              <select multiple className="select w-full h-28" value={friedmanCols}
                onChange={(e) => setFriedmanCols(Array.from(e.target.selectedOptions, o => o.value))}>
                {binCols.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}
          {isCA && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Ordered exposure (3+ levels)</label>
              <select className="select w-full" value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
                {[...catCols, ...numCols].map((c) => <option key={c}>{c}</option>)}
              </select>
              <p className="text-[10px] text-gray-400 mt-1">Scores default to 0,1,2,… (rank order). Custom scores not exposed in UI for v1.</p>
              <label className="text-xs text-gray-400 block mt-2 mb-1">Level order, low → high (optional)</label>
              <input
                className="input w-full text-xs"
                placeholder="e.g. Low, Medium, High"
                value={levelOrder}
                onChange={(e) => setLevelOrder(e.target.value)}
              />
              <p className="text-[10px] text-gray-400 mt-1">
                Numeric levels sort themselves. For word labels, leaving this blank sorts
                alphabetically — which reverses e.g. Low/Medium/High and flips the trend.
              </p>
            </div>
          )}
          <button className="btn-primary w-full" onClick={run} disabled={loading || (isCochran && friedmanCols.length < 3)}>
            {loading ? "Running\u2026" : "Run Test"}
          </button>
          {error && <p className="text-red-400 text-xs">{error}</p>}
        </div>
      </div>
      <div className="flex-1 space-y-3">
        {g && (
          <div className="panel bg-indigo-50 border-indigo-200 space-y-2">
            <p className="text-[10px] font-bold text-indigo-900 uppercase">When to use</p>
            <p className="text-xs text-indigo-800">{g.when}</p>
            <p className="text-[10px] font-bold text-indigo-900 uppercase mt-2">How to read</p>
            <p className="text-xs text-indigo-800">{g.reading}</p>
          </div>
        )}
        {result ? <ResultCard result={result} /> : (
          <div className="panel text-center text-gray-400 py-12">Select a test and configure variables</div>
        )}
      </div>
    </div>
  );
}
