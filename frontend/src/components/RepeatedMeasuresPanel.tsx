import { useState } from "react";
import { useStore, isNumericKind, type Session } from "../store";
import { runPairedTTest, runWilcoxonSR, runFriedman, runRMAnova, runMixedAnova } from "../api";
import ResultExporter from "./ResultExporter";
import { fmtP, warningText } from "../lib/format";

const RM_TESTS = [
  { id: "paired_ttest",   label: "Paired t-test",        group: "Parametric" },
  { id: "rm_anova",       label: "RM ANOVA",             group: "Parametric" },
  { id: "mixed_anova",    label: "Mixed ANOVA",          group: "Parametric" },
  { id: "wilcoxon_sr",    label: "Wilcoxon signed-rank", group: "Non-parametric" },
  { id: "friedman",       label: "Friedman test",        group: "Non-parametric" },
] as const;

const RM_GUIDANCE: Record<string, { when: string; assumptions: string; reading: string }> = {
  paired_ttest: {
    when: "Compare two related measurements (e.g. before/after treatment, left/right eye) from the same subjects.",
    assumptions: "Differences should be approximately normally distributed (robust if n > 30). Use Wilcoxon signed-rank if violated.",
    reading: "p < 0.05 means the mean difference is significantly different from zero. Report: t(df) = X, p = Y, Cohen's d_z = Z.",
  },
  wilcoxon_sr: {
    when: "Non-parametric alternative to paired t-test. Use when differences are non-normal or ordinal.",
    assumptions: "Symmetric distribution of differences (less strict than normality). No minimum sample size, but n >= 6 recommended.",
    reading: "p < 0.05 means the median difference is significantly different from zero. Report: W = X, p = Y, rank-biserial r = Z.",
  },
  friedman: {
    when: "Compare 3+ related measurements from the same subjects (non-parametric RM ANOVA alternative). Data in wide format.",
    assumptions: "Each subject measured under all conditions. Ranks are used, so no normality assumption.",
    reading: "p < 0.05 means at least one condition differs. Kendall's W measures concordance. Follow up with pairwise Wilcoxon tests.",
  },
  rm_anova: {
    when: "Compare 3+ related measurements from the same subjects (parametric). Data must be in long format.",
    assumptions: "Normality within each condition. Sphericity (equal variances of differences) — checked via Mauchly's test. Greenhouse-Geisser correction applied if violated.",
    reading: "Significant F means at least one timepoint differs. Report: F(df1,df2) = X, p = Y, partial eta-squared = Z. Follow up with pairwise paired t-tests.",
  },
  mixed_anova: {
    when: "Combine within-subjects (repeated measures) and between-subjects (group) factors. E.g. pre/post scores across treatment and control groups.",
    assumptions: "Same as RM ANOVA plus homogeneity of variances across between-subjects groups.",
    reading: "Check the interaction term first. Significant interaction means the within-subjects effect differs across groups. Main effects should be interpreted cautiously if interaction is significant.",
  },
};

interface EffectSize {
  name?: string;
  value?: number;
  ci_low?: number;
  ci_high?: number;
  magnitude?: string;
}

interface AssumptionCheck {
  name?: string;
  detail?: string;
  met?: boolean;
}

interface AnovaEffect {
  term?: string;
  F?: number;
  df_num?: number;
  df_den?: number;
  p?: number;
  effect_size?: { value?: number };
  significant?: boolean;
}

interface PostHocRow {
  group1?: string;
  group2?: string;
  statistic?: number;
  p_adj?: number;
  significant?: boolean;
}

interface RMResult {
  test?: string;
  interpretation?: string;
  result_text?: string;
  significant?: boolean;
  effect_sizes?: EffectSize[];
  assumptions?: AssumptionCheck[];
  warnings?: unknown[];
  posthoc?: PostHocRow[];
  posthoc_method?: string;
  export_rows?: (string | number | null | undefined)[][];
  r_code?: string;
  effects?: AnovaEffect[];
  [key: string]: unknown;
}

function ResultCard({ result }: { result: RMResult }) {
  const fmt = (v: unknown) => {
    if (typeof v !== "number") return String(v ?? "");
    if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(3);
    return v.toFixed(4);
  };
  const skip = new Set(["test", "interpretation", "result_text", "significant", "effect_sizes",
    "assumptions", "warnings", "summary", "posthoc", "posthoc_method", "export_rows",
    "r_code", "effects"]);

  const statEntries = Object.entries(result).filter(([k, v]) => !skip.has(k) && typeof v !== "object");
  const exportHeaders = (result.export_rows?.[0] ?? ["Statistic", "Value"]).map((h) => String(h ?? ""));
  const exportRows = result.export_rows?.slice(1) ?? statEntries.map(([k, v]) => [k, fmt(v)]);

  return (
    <div className="panel space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-gray-900">{result.test}</h4>
        <div className="flex items-center gap-2">
          <ResultExporter title={result.test ?? "repeated_test"} headers={exportHeaders} rows={exportRows} />
          {"significant" in result && (
            <span className={result.significant ? "badge-sig" : "badge-ns"}>
              {result.significant ? "Significant" : "Not significant"}
            </span>
          )}
        </div>
      </div>
      <p className="text-sm text-gray-500 italic">{result.interpretation}</p>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm">
        {statEntries.map(([k, v]) => (
          <div key={k} className="flex justify-between border-b border-gray-100 py-1">
            <span className="text-gray-400">{k}</span>
            <span className="text-gray-700 font-mono">{fmt(v)}</span>
          </div>
        ))}
      </div>

      {/* Effect Sizes */}
      {(result.effect_sizes?.length ?? 0) > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-semibold text-gray-600">Effect Sizes</p>
          {(result.effect_sizes ?? []).map((es: EffectSize, i: number) => (
            <div key={i} className="flex items-center gap-3 bg-indigo-50 rounded-lg px-3 py-1.5 text-xs">
              <span className="font-semibold text-indigo-800">{es.name?.replace(/_/g, " ")}</span>
              <span className="font-mono text-indigo-700">{es.value?.toFixed(3)}</span>
              {es.ci_low != null && es.ci_high != null && (
                <span className="text-indigo-500">95% CI: [{es.ci_low?.toFixed(3)}, {es.ci_high?.toFixed(3)}]</span>
              )}
              {es.magnitude && (
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                  es.magnitude === "large" ? "bg-red-100 text-red-700" :
                  es.magnitude === "medium" ? "bg-amber-100 text-amber-700" :
                  es.magnitude === "small" ? "bg-blue-100 text-blue-700" :
                  "bg-gray-100 text-gray-500"}`}>{es.magnitude}</span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Assumptions */}
      {(result.assumptions?.length ?? 0) > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-semibold text-gray-600">Assumption Checks</p>
          {(result.assumptions ?? []).map((a: AssumptionCheck, i: number) => (
            <div key={i} className={`flex items-center gap-2 text-xs px-3 py-1 rounded-lg ${a.met ? "bg-green-50 text-green-800" : "bg-amber-50 text-amber-800"}`}>
              <span>{a.met ? "\u2713" : "\u26A0"}</span>
              <span className="font-medium">{a.name}</span>
              <span className="text-gray-500">\u2014 {a.detail}</span>
            </div>
          ))}
        </div>
      )}

      {/* Warnings */}
      {(result.warnings?.length ?? 0) > 0 && (result.warnings ?? []).map((w, i) => (
        <div key={i} className="text-xs px-3 py-1 rounded-lg bg-amber-50 text-amber-800">\u26A0 {warningText(w)}</div>
      ))}

      {/* Result text */}
      {result.result_text && (
        <div className="rounded-lg border border-indigo-100 bg-white px-3 py-2 text-xs text-gray-600 leading-relaxed">
          <span className="text-indigo-400 mr-1">\uD83D\uDCAC</span> {result.result_text}
        </div>
      )}

      {/* Multi-effect table (mixed ANOVA) */}
      {(result.effects?.length ?? 0) > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-600 mb-1">ANOVA Effects</p>
          <div className="overflow-auto rounded border border-gray-200">
            <table className="w-full text-xs">
              <thead><tr className="bg-gray-50">
                <th className="px-2 py-1 text-left">Term</th>
                <th className="px-2 py-1 text-right">F</th>
                <th className="px-2 py-1 text-right">df</th>
                <th className="px-2 py-1 text-right"><i>p</i></th>
                <th className="px-2 py-1 text-right">Partial \u03B7\u00B2</th>
                <th className="px-2 py-1 text-center">Sig</th>
              </tr></thead>
              <tbody>
                {(result.effects ?? []).map((e: AnovaEffect, i: number) => (
                  <tr key={i} className={`border-t border-gray-100 ${e.significant ? "" : "text-gray-400"}`}>
                    <td className="px-2 py-1 font-medium">{e.term}</td>
                    <td className="px-2 py-1 text-right font-mono">{e.F?.toFixed(3)}</td>
                    <td className="px-2 py-1 text-right">{e.df_num},{e.df_den}</td>
                    <td className="px-2 py-1 text-right font-mono">{fmtP(e.p)}</td>
                    <td className="px-2 py-1 text-right font-mono">{e.effect_size?.value?.toFixed(3)}</td>
                    <td className="px-2 py-1 text-center">{e.significant ? "\u2713" : "\u2014"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Post-hoc */}
      {(result.posthoc?.length ?? 0) > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-600 mb-1">Post-hoc: {result.posthoc_method ?? "Pairwise"}</p>
          <div className="overflow-auto rounded border border-gray-200">
            <table className="w-full text-xs">
              <thead><tr className="bg-gray-50">
                <th className="px-2 py-1 text-left">Comparison</th>
                <th className="px-2 py-1 text-right">Statistic</th>
                <th className="px-2 py-1 text-right"><i>p</i> (adj)</th>
                <th className="px-2 py-1 text-center">Sig</th>
              </tr></thead>
              <tbody>
                {(result.posthoc ?? []).map((ph: PostHocRow, i: number) => (
                  <tr key={i} className={`border-t border-gray-100 ${ph.significant ? "" : "text-gray-400"}`}>
                    <td className="px-2 py-1">{ph.group1} vs {ph.group2}</td>
                    <td className="px-2 py-1 text-right font-mono">{ph.statistic?.toFixed(3)}</td>
                    <td className="px-2 py-1 text-right font-mono">{fmtP(ph.p_adj)}</td>
                    <td className="px-2 py-1 text-center">{ph.significant ? "\u2713" : "\u2014"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* R code */}
      {result.r_code && (
        <details className="text-xs">
          <summary className="text-gray-400 cursor-pointer hover:text-indigo-600">R code</summary>
          <pre className="mt-1 p-2 bg-gray-50 rounded-lg text-gray-600 font-mono text-[10px] whitespace-pre-wrap">{result.r_code}</pre>
        </details>
      )}
    </div>
  );
}

export default function RepeatedMeasuresPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <RepeatedMeasuresPanelBody session={session} />;
}

function RepeatedMeasuresPanelBody({ session }: { session: Session }) {
  const numCols = session.columns.filter((c) => isNumericKind(c.kind)).map((c) => c.name);
  const allCols = session.columns.map((c) => c.name);

  const [test, setTest] = useState<string>("paired_ttest");
  const [col1, setCol1] = useState(numCols[0] ?? "");
  const [col2, setCol2] = useState(numCols[1] ?? numCols[0] ?? "");
  const [friedmanCols, setFriedmanCols] = useState<string[]>([]);
  const [subjectCol, setSubjectCol] = useState(allCols[0] ?? "");
  const [withinCol, setWithinCol] = useState(allCols[1] ?? "");
  const [betweenCol, setBetweenCol] = useState(allCols[2] ?? "");
  const [valueCol, setValueCol] = useState(numCols[0] ?? "");
  const [result, setResult] = useState<RMResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const isPaired = test === "paired_ttest" || test === "wilcoxon_sr";
  const isFriedman = test === "friedman";
  const isLong = test === "rm_anova" || test === "mixed_anova";
  const isMixed = test === "mixed_anova";

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    const sid = session.session_id;
    try {
      let res: { data?: RMResult } | null = null;
      if (test === "paired_ttest") res = await runPairedTTest({ session_id: sid, col1, col2 });
      else if (test === "wilcoxon_sr") res = await runWilcoxonSR({ session_id: sid, col1, col2 });
      else if (test === "friedman") res = await runFriedman({ session_id: sid, columns: friedmanCols });
      else if (test === "rm_anova") res = await runRMAnova({ session_id: sid, subject_col: subjectCol, within_col: withinCol, value_col: valueCol });
      else if (test === "mixed_anova") res = await runMixedAnova({ session_id: sid, subject_col: subjectCol, within_col: withinCol, between_col: betweenCol, value_col: valueCol });
      setResult(res?.data ?? null);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail ?? "Error");
    } finally { setLoading(false); }
  };

  const guidance = RM_GUIDANCE[test];

  return (
    <div className="flex gap-4">
      {/* Left sidebar */}
      <div className="w-64 flex-shrink-0 space-y-4">
        <div className="panel space-y-1">
          {["Parametric", "Non-parametric"].map((grp) => (
            <div key={grp}>
              <p className="text-xs text-gray-400 uppercase tracking-wider mt-3 mb-1 first:mt-0">{grp}</p>
              {RM_TESTS.filter((t) => t.group === grp).map(({ id, label }) => (
                <label key={id} className="flex items-center gap-2 cursor-pointer py-0.5">
                  <input type="radio" name="rm_test" value={id} checked={test === id}
                    onChange={() => { setTest(id); setResult(null); }} className="accent-indigo-500" />
                  <span className="text-sm text-gray-700">{label}</span>
                </label>
              ))}
            </div>
          ))}
        </div>

        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Variables</h3>

          {/* Paired: two columns */}
          {isPaired && (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Measurement 1</label>
                <select className="select w-full" value={col1} onChange={(e) => setCol1(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Measurement 2</label>
                <select className="select w-full" value={col2} onChange={(e) => setCol2(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
            </>
          )}

          {/* Friedman: multi-select */}
          {isFriedman && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Conditions (3+ columns, wide format)</label>
              <select multiple className="select w-full h-32" value={friedmanCols}
                onChange={(e) => setFriedmanCols(Array.from(e.target.selectedOptions, o => o.value))}>
                {numCols.map((c) => <option key={c}>{c}</option>)}
              </select>
              <p className="text-[10px] text-gray-400 mt-1">Hold Ctrl/Cmd to select multiple</p>
            </div>
          )}

          {/* Long-format: subject, within, value, (between) */}
          {isLong && (
            <>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Subject ID column</label>
                <select className="select w-full" value={subjectCol} onChange={(e) => setSubjectCol(e.target.value)}>
                  {allCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Within-subjects factor</label>
                <select className="select w-full" value={withinCol} onChange={(e) => setWithinCol(e.target.value)}>
                  {allCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              {isMixed && (
                <div>
                  <label className="text-xs text-gray-400 block mb-1">Between-subjects factor</label>
                  <select className="select w-full" value={betweenCol} onChange={(e) => setBetweenCol(e.target.value)}>
                    {allCols.map((c) => <option key={c}>{c}</option>)}
                  </select>
                </div>
              )}
              <div>
                <label className="text-xs text-gray-400 block mb-1">Outcome (numeric)</label>
                <select className="select w-full" value={valueCol} onChange={(e) => setValueCol(e.target.value)}>
                  {numCols.map((c) => <option key={c}>{c}</option>)}
                </select>
              </div>
              <p className="text-[10px] text-amber-600 bg-amber-50 rounded px-2 py-1">
                Data must be in long format. Use Compute \u2192 Melt to reshape wide data first.
              </p>
            </>
          )}

          <button className="btn-primary w-full" onClick={run} disabled={loading || (isFriedman && friedmanCols.length < 3)}>
            {loading ? "Running\u2026" : "Run Test"}
          </button>
          {error && <p className="text-red-400 text-xs">{error}</p>}
        </div>
      </div>

      {/* Right: results */}
      <div className="flex-1 space-y-3">
        {guidance && (
          <div className="panel bg-indigo-50 border-indigo-200 space-y-2">
            <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">When to use</p>
            <p className="text-xs text-indigo-800 leading-relaxed">{guidance.when}</p>
            <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider mt-2">Assumptions</p>
            <p className="text-xs text-indigo-800 leading-relaxed">{guidance.assumptions}</p>
            <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider mt-2">How to read</p>
            <p className="text-xs text-indigo-800 leading-relaxed">{guidance.reading}</p>
          </div>
        )}
        {result ? <ResultCard result={result} /> : (
          <div className="panel text-center text-gray-400 py-12">
            Select a test and configure variables to begin
          </div>
        )}
      </div>
    </div>
  );
}
