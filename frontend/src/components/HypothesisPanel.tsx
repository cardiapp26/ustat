import { useState, useEffect } from "react";
import { useStore, isNumericKind, isCategoricalKind, type Session } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { runTTest, runChiSquare, runAnova, runMannWhitney, runFisher, runKruskal, runAncova, runTwoWayAnova, runJonckheereTerpstra, runMancova } from "../api";
import ResultExporter from "./ResultExporter";
import { fmtP, warningText } from "../lib/format";

/** True when a stat-grid key holds a p-value (route through the canonical fmtP). */
function isPKey(k: string): boolean {
  return /^p$|^p_?value$|_p$|_p_?value$/.test(k);
}

const TESTS = [
  { id: "ttest_1sample",  label: "One-sample t-test",     group: "Parametric" },
  { id: "ttest_2sample",  label: "Independent t-test",    group: "Parametric" },
  { id: "anova",          label: "One-way ANOVA",         group: "Parametric" },
  { id: "mannwhitney",    label: "Mann-Whitney U",        group: "Non-parametric" },
  { id: "kruskal",        label: "Kruskal-Wallis",        group: "Non-parametric" },
  { id: "jonckheere",     label: "Jonckheere-Terpstra trend", group: "Non-parametric" },
  { id: "ancova",          label: "ANCOVA",                group: "Parametric" },
  { id: "mancova",        label: "MANCOVA",               group: "Parametric" },
  { id: "two_way",        label: "Two-way ANOVA",         group: "Parametric" },
  { id: "chisquare",      label: "Chi-square",            group: "Categorical" },
  { id: "fisher",         label: "Fisher's exact",        group: "Categorical" },
];

const TEST_GUIDANCE: Record<string, { when: string; assumptions: string; reading: string }> = {
  ttest_1sample: {
    when: "Compare one group's mean to a known reference value (e.g. population norm, clinical threshold).",
    assumptions: "Outcome is continuous and approximately normally distributed (check Q-Q plot in Summary tab). Robust to mild non-normality if n > 30 (CLT).",
    reading: "If p < 0.05, the sample mean is significantly different from the test value. Report: t(df) = X.XX, p = Y.YYY, mean difference = Z.ZZ.",
  },
  ttest_2sample: {
    when: "Compare means between two independent groups (e.g. treatment vs. control, male vs. female).",
    assumptions: "Both groups approximately normal (or n > 30 each). Levene's test checks equal variances — if violated, Welch's t-test is used automatically.",
    reading: "p < 0.05 means the groups differ significantly. Report: t(df) = X.XX, p = Y.YYY. Add Hedges' g for effect size (small 0.2, medium 0.5, large 0.8).",
  },
  anova: {
    when: "Compare means across 3+ groups simultaneously (e.g. drug A vs. B vs. C). Avoids multiple-comparison inflation from running many t-tests.",
    assumptions: "Each group approximately normal, roughly equal variances (Levene's test). Robust if groups are balanced and n > 20 per group.",
    reading: "A significant F-test (p < 0.05) means at least one group differs. Use post-hoc tests (Tukey, Bonferroni) to identify which pairs differ.",
  },
  mannwhitney: {
    when: "Non-parametric alternative to the independent t-test. Use when data are ordinal, heavily skewed, or n < 20 per group.",
    assumptions: "Both samples are independent. Tests whether one distribution is stochastically greater than the other (rank-based).",
    reading: "p < 0.05 means the groups' rank distributions differ significantly. Report U statistic and p-value. Effect size: r = Z / sqrt(N).",
  },
  kruskal: {
    when: "Non-parametric alternative to one-way ANOVA. Use when comparing 3+ groups with non-normal or ordinal data.",
    assumptions: "Groups are independent. Tests whether at least one group's distribution differs from the others (rank-based).",
    reading: "p < 0.05 means at least one group differs. Post-hoc Dunn's test with Holm/Bonferroni/FDR correction identifies pairwise differences.",
  },
  jonckheere: {
    when: "Trend test for a continuous outcome across 3+ ordered groups (e.g. tertiles/quartiles of a biomarker vs. a downstream measure). Non-parametric analogue of Cochran-Armitage.",
    assumptions: "Groups are ordered, observations independent. Tests monotone trend in medians (rank-based).",
    reading: "p < 0.05 means a monotone increasing or decreasing trend across the ordered groups. Sign of Z indicates direction.",
  },
  chisquare: {
    when: "Test association between two categorical variables (e.g. treatment group vs. outcome category). Use when all expected cell counts are >= 5.",
    assumptions: "Each observation is independent. Expected frequency in each cell >= 5. If any cell < 5, use Fisher's exact test instead.",
    reading: "p < 0.05 means the variables are significantly associated. Report: \u03C7\u00B2(df) = X.XX, p = Y.YYY. Add Cramer's V for effect size.",
  },
  fisher: {
    when: "Exact test for 2\u00D72 tables, especially when sample is small or any expected cell count < 5. Preferred over chi-square for small samples.",
    assumptions: "Fixed marginals. No minimum sample size requirement — valid even for very small tables.",
    reading: "p < 0.05 means the row and column variables are significantly associated. Also report the odds ratio and its 95% CI.",
  },
  ancova: {
    when: "Compare group means on an outcome after controlling for one or more continuous covariates. Essential when groups differ on a baseline variable (e.g. age, BMI).",
    assumptions: "Normality of residuals. Homogeneity of regression slopes (covariate effect is the same in all groups). Linear relationship between covariate and outcome.",
    reading: "The F-test for the group factor shows whether groups differ AFTER adjustment. EMMs (estimated marginal means) show the adjusted group means. Report: F, p, partial \u03B7\u00B2, and EMMs.",
  },
  two_way: {
    when: "Examine the effects of two categorical factors (and their interaction) on a continuous outcome. E.g. drug type \u00D7 dose level on blood pressure.",
    assumptions: "Normality of residuals. Homogeneity of variances across all factor-level combinations.",
    reading: "Check the interaction first. If significant, main effects are qualified by the interaction. Report F, p, and partial \u03B7\u00B2 for each term. Use EMMs to understand cell means.",
  },
  mancova: {
    when: "Test a group effect on several correlated continuous outcomes at once (e.g. BDNF, GDNF, NTF3, NGF), while controlling for covariates (age, sex, anxiety, BMI, smoking). Use it as the omnibus step before per-outcome ANCOVAs to guard against multiple-comparison inflation.",
    assumptions: "Multivariate normality of residuals, homogeneity of covariance matrices (Box's M), linear covariate\u2013outcome relationships, and no severe multicollinearity among outcomes. Log-transform skewed outcomes first (Compute \u2192 LOG).",
    reading: "Read Pillai's Trace first \u2014 it is the most robust to assumption violations. p < 0.05 means the groups differ on the combined outcomes after adjustment. Report Pillai's Trace, F(num,den), p, and partial \u03B7\u00B2 (0.01 small / 0.06 medium / 0.14 large). If significant, follow up with an ANCOVA per outcome.",
  },
};

interface EffectSize {
  name?: string;
  value?: number;
  ci_low?: number | null;
  ci_high?: number | null;
  magnitude?: string;
}
interface AssumptionCheck {
  name?: string;
  detail?: string;
  met?: boolean;
}
interface PostHocRow {
  group1?: string;
  group2?: string;
  statistic?: number;
  p_adj?: number | null;
  rank_diff?: number;
  effect_size?: { value?: number; magnitude?: string };
  significant?: boolean;
}
interface TestResult {
  test?: string;
  interpretation?: string;
  significant?: boolean;
  result_text?: string;
  effect_sizes?: EffectSize[];
  assumptions?: AssumptionCheck[];
  warnings?: unknown[];
  posthoc?: PostHocRow[];
  posthoc_method?: string;
  groups?: Record<string, unknown>[];
  table?: number[][];
  row_labels?: string[];
  col_labels?: string[];
  [key: string]: unknown;
}

function ResultCard({ result }: { result: TestResult }) {
  const fmt = (v: unknown) => {
    if (typeof v !== "number") return String(v);
    if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(3);
    return v.toFixed(4);
  };

  const skip = ["test", "interpretation", "significant", "crosstab", "groups",
                "table", "row_labels", "col_labels", "curve", "effect_sizes",
                "assumptions", "warnings", "summary", "posthoc", "posthoc_method",
                "result_text", "export_rows"];

  const statEntries = Object.entries(result).filter(([k]) => !skip.includes(k) && typeof result[k] !== "object");
  const exportHeaders = ["Statistic", "Value"];
  const exportRows = statEntries.map(([k, v]) => [k, isPKey(k) ? fmtP(v as number | null | undefined) : fmt(v)]);

  return (
    <div className="panel space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-gray-900">{result.test}</h4>
        <div className="flex items-center gap-2">
          <ResultExporter title={result.test ?? "hypothesis_test"} headers={exportHeaders} rows={exportRows} />
          {"significant" in result && (
            <span className={result.significant ? "badge-sig" : "badge-ns"}>
              {result.significant ? "Significant" : "Not significant"}
            </span>
          )}
        </div>
      </div>
      <p className="text-sm text-gray-500 italic">{result.interpretation}</p>

      <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm">
        {Object.entries(result)
          .filter(([k]) => !skip.includes(k))
          .map(([k, v]) => (
            <div key={k} className="flex justify-between border-b border-gray-100 py-1">
              <span className="text-gray-400">{k}</span>
              <span className="text-gray-700 font-mono">{isPKey(k) ? fmtP(v as number | null | undefined) : fmt(v)}</span>
            </div>
          ))}
      </div>

      {/* Effect Sizes */}
      {(result.effect_sizes?.length ?? 0) > 0 && (
        <div className="mt-2 space-y-1">
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
                  "bg-gray-100 text-gray-500"}`}>
                  {es.magnitude}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Assumptions */}
      {(result.assumptions?.length ?? 0) > 0 && (
        <div className="mt-2 space-y-1">
          <p className="text-xs font-semibold text-gray-600">Assumption Checks</p>
          {(result.assumptions ?? []).map((a: AssumptionCheck, i: number) => (
            <div key={i} className={`flex items-center gap-2 text-xs px-3 py-1 rounded-lg ${a.met ? "bg-green-50 text-green-800" : "bg-amber-50 text-amber-800"}`}>
              <span>{a.met ? "✓" : "⚠"}</span>
              <span className="font-medium">{a.name}</span>
              <span className="text-gray-500">— {a.detail}</span>
            </div>
          ))}
        </div>
      )}

      {/* Warnings */}
      {(result.warnings?.length ?? 0) > 0 && (
        <div className="mt-2 space-y-1">
          {(result.warnings ?? []).map((w, i) => (
            <div key={i} className="flex items-center gap-2 text-xs px-3 py-1 rounded-lg bg-amber-50 text-amber-800">
              <span>⚠</span> {warningText(w)}
            </div>
          ))}
        </div>
      )}

      {/* Results Paragraph */}
      {result.result_text && (
        <div className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 mt-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] font-semibold text-gray-400 uppercase">Results Paragraph</span>
            <button onClick={() => navigator.clipboard.writeText(result.result_text ?? "")} className="text-[10px] px-2 py-0.5 rounded border border-gray-300 text-gray-500 hover:bg-indigo-50 hover:text-indigo-600 transition-colors">Copy</button>
          </div>
          <p className="text-sm text-gray-700 leading-relaxed">{result.result_text}</p>
        </div>
      )}

      {/* Post-hoc results */}
      {(result.posthoc?.length ?? 0) > 0 && (
        <div className="mt-3">
          <p className="text-xs font-semibold text-gray-600 mb-1">
            Post-hoc: {result.posthoc_method ?? "Pairwise comparisons"}
          </p>
          <div className="overflow-auto rounded border border-gray-200">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-2 py-1 text-left">Comparison</th>
                  <th className="px-2 py-1 text-right">Statistic</th>
                  <th className="px-2 py-1 text-right"><i>p</i> (adj)</th>
                  <th className="px-2 py-1 text-right">Effect size</th>
                  <th className="px-2 py-1 text-center">Sig</th>
                </tr>
              </thead>
              <tbody>
                {(result.posthoc ?? []).map((ph: PostHocRow, i: number) => (
                  <tr key={i} className={`border-t border-gray-100 ${ph.significant ? "" : "text-gray-400"}`}>
                    <td className="px-2 py-1 font-medium">{ph.group1} vs {ph.group2}</td>
                    <td className="px-2 py-1 text-right font-mono">{ph.statistic?.toFixed(3)}</td>
                    <td className="px-2 py-1 text-right font-mono">{fmtP(ph.p_adj)}</td>
                    <td className="px-2 py-1 text-right font-mono">
                      {ph.effect_size ? `${ph.effect_size.value?.toFixed(3)} (${ph.effect_size.magnitude})` : ph.rank_diff?.toFixed(2) ?? "—"}
                    </td>
                    <td className="px-2 py-1 text-center">{ph.significant ? "✓" : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {result.groups && (
        <div className="overflow-auto rounded border border-gray-200 mt-2">
          <table>
            <thead>
              <tr>{Object.keys(result.groups[0]).map((k) => <th key={k}>{k}</th>)}</tr>
            </thead>
            <tbody>
              {result.groups.map((g: Record<string, unknown>, i: number) => (
                <tr key={i}>{Object.values(g).map((v: unknown, j) => (
                  <td key={j}>{typeof v === "number" ? fmt(v) : String(v)}</td>
                ))}</tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {result.table && (
        <div className="overflow-auto rounded border border-gray-200 mt-2">
          <table>
            <thead>
              <tr>
                <th></th>
                {(result.col_labels ?? []).map((l: string) => <th key={l}>{l}</th>)}
              </tr>
            </thead>
            <tbody>
              {result.table.map((row: number[], i: number) => (
                <tr key={i}>
                  <td className="font-medium text-gray-900">{(result.row_labels ?? [])[i] ?? ""}</td>
                  {row.map((v, j) => <td key={j}>{v}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function HypothesisPanel() {
  const session = useStore((s) => s.session);
  if (!session) return null;
  return <HypothesisPanelBody session={session} />;
}

function HypothesisPanelBody({ session }: { session: Session }) {
  const numCols = session.columns.filter((c) => isNumericKind(c.kind) && !c.analysis_excluded).map((c) => c.name);
  const catCols = session.columns.filter((c) => isCategoricalKind(c.kind) && !c.analysis_excluded).map((c) => c.name);

  const [test, setTest] = usePersistedPanelState<string>("hypothesis", "test", "ttest_1sample");
  const [col, setCol] = usePersistedPanelState<string>("hypothesis", "col", numCols[0] ?? "");
  const [col2, setCol2] = usePersistedPanelState<string>("hypothesis", "col2", catCols[1] ?? catCols[0] ?? "");
  const [groupCol, setGroupCol] = usePersistedPanelState<string>("hypothesis", "groupCol", catCols[0] ?? "");
  const [mu, setMu] = useState("0");
  const [covariates, setCovariates] = usePersistedPanelState<string[]>("hypothesis", "covariates", []);
  const [outcomes, setOutcomes] = usePersistedPanelState<string[]>("hypothesis", "outcomes", []);  // MANCOVA: ≥2 dependent vars
  const [factor2, setFactor2] = usePersistedPanelState<string>("hypothesis", "factor2", catCols[1] ?? catCols[0] ?? "");
  // Dunn's post-hoc correction for Kruskal-Wallis. Bonferroni is the
  // conventional reporting choice in clinical journals; Holm is the
  // default because it strictly dominates Bonferroni while controlling
  // the same family-wise error rate.
  const [posthocCorrection, setPosthocCorrection] = usePersistedPanelState<"holm" | "bonferroni" | "fdr" | "none">("hypothesis", "correction", "holm");
  const cached = useStore((s) => s.panelCache.hypothesis);
  const setCache = useStore((s) => s.setPanelCache);
  const [result, _setResult] = useState<TestResult | null>(((cached as { result?: TestResult | null } | undefined)?.result) ?? null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  // Merge into the existing cache object so the persisted selection keys
  // (test, col, groupCol, …) written by usePersistedPanelState survive.
  const setResult = (r: TestResult | null) => {
    _setResult(r);
    setCache("hypothesis", { ...(useStore.getState().panelCache.hypothesis ?? {}), result: r });
  };

  const isCat = test === "chisquare" || test === "fisher";
  // "two_way" belongs here: the request sends factor1: groupCol, but the test
  // was missing from this list, so the Group column selector never rendered
  // for it. Whatever groupCol happened to hold from the previous test was
  // sent as factor1, unseen.
  const needsGroup = ["ttest_2sample", "anova", "mannwhitney", "kruskal", "jonckheere", "ancova", "mancova", "two_way"].includes(test);
  const isKruskal = test === "kruskal";
  const needsSecondCat = isCat;
  const isAncova = test === "ancova";
  const isMancova = test === "mancova";
  const isTwoWay = test === "two_way";
  // Ordinal grouping → an ordered-alternative trend test (Jonckheere) is more
  // powerful than Kruskal/ANOVA which ignore the group ordering.
  const groupIsOrdinal = session.columns.some((c) => c.name === groupCol && c.kind === "ordinal");
  const suggestJonckheere = groupIsOrdinal && (test === "kruskal" || test === "anova");

  // Switching test type changes which list each selector draws from — the
  // primary column comes from the categorical list for chi-square/Fisher and
  // from the numeric list for everything else. The selections used to carry
  // over untouched, so a numeric column left over from a t-test stayed in
  // `col` while the dropdown showed the categorical options: the value was
  // not visible anywhere on screen and was still what got sent. Point every
  // selection back at a member of its own list.
  useEffect(() => {
    const primary = isCat ? catCols : numCols;
    if (primary.length && !primary.includes(col)) setCol(primary[0]);
    if (catCols.length && !catCols.includes(groupCol)) setGroupCol(catCols[0]);
    if (catCols.length && !catCols.includes(col2)) setCol2(catCols[1] ?? catCols[0]);
    if (catCols.length && !catCols.includes(factor2)) setFactor2(catCols[1] ?? catCols[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [test, isCat, numCols.join("|"), catCols.join("|")]);

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    const sid = session.session_id;
    try {
      let res: { data: unknown } | undefined;
      if (test === "ttest_1sample")  res = await runTTest({ session_id: sid, column: col, mu: +mu });
      else if (test === "ttest_2sample") res = await runTTest({ session_id: sid, column: col, group_column: groupCol });
      else if (test === "anova")     res = await runAnova({ session_id: sid, column: col, group_column: groupCol });
      else if (test === "mannwhitney") res = await runMannWhitney({ session_id: sid, column: col, group_column: groupCol });
      else if (test === "kruskal")   res = await runKruskal({ session_id: sid, column: col, group_column: groupCol, posthoc_correction: posthocCorrection });
      else if (test === "jonckheere") res = await runJonckheereTerpstra({ session_id: sid, column: col, group_column: groupCol });
      else if (test === "chisquare") res = await runChiSquare({ session_id: sid, row_column: col, col_column: col2 });
      else if (test === "fisher")    res = await runFisher({ session_id: sid, row_column: col, col_column: col2 });
      else if (test === "ancova")    res = await runAncova({ session_id: sid, outcome: col, group_col: groupCol, covariates });
      else if (test === "mancova")   res = await runMancova({ session_id: sid, outcomes, group_col: groupCol, covariates });
      else if (test === "two_way")   res = await runTwoWayAnova({ session_id: sid, outcome: col, factor1: groupCol, factor2 });
      setResult((res?.data as TestResult | undefined) ?? null);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Error");
    } finally { setLoading(false); }
  };

  const groups = ["Parametric", "Non-parametric", "Categorical"];

  return (
    <div className="flex gap-4">
      <div className="w-64 flex-shrink-0 space-y-4">
        <div className="panel space-y-1">
          {groups.map((grp) => (
            <div key={grp}>
              <p className="text-xs text-gray-400 uppercase tracking-wider mt-3 mb-1 first:mt-0">{grp}</p>
              {TESTS.filter((t) => t.group === grp).map(({ id, label }) => (
                <label key={id} className="flex items-center gap-2 cursor-pointer py-0.5">
                  <input type="radio" name="test" value={id} checked={test === id}
                    onChange={() => { setTest(id); setResult(null); }} className="accent-indigo-500" />
                  <span className="text-sm text-gray-700">{label}</span>
                </label>
              ))}
            </div>
          ))}
        </div>

        <div className="panel space-y-3">
          <h3 className="text-sm font-semibold text-gray-700">Variables</h3>
          {!isMancova && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">
                {isCat ? "Row column" : "Outcome column"}
              </label>
              <select className="select w-full" value={col} onChange={(e) => setCol(e.target.value)}>
                {(isCat ? catCols : numCols).map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}

          {/* MANCOVA: ≥2 dependent variables */}
          {isMancova && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Outcomes (≥2 continuous)</label>
              <select multiple className="select w-full h-24" value={outcomes}
                onChange={(e) => setOutcomes(Array.from(e.target.selectedOptions, o => o.value))}>
                {numCols.map((c) => <option key={c}>{c}</option>)}
              </select>
              <p className="text-[10px] text-gray-400 mt-1">{outcomes.length} selected · hold Ctrl/Cmd to pick multiple</p>
            </div>
          )}

          {test === "ttest_1sample" && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Test value (μ₀)</label>
              <input className="select w-full" type="number" value={mu} onChange={(e) => setMu(e.target.value)} />
            </div>
          )}

          {needsGroup && catCols.length > 0 && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Group column</label>
              <select className="select w-full" value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
                {catCols.map((c) => <option key={c}>{c}</option>)}
              </select>
              {suggestJonckheere && (
                <div className="mt-1 text-[10px] text-teal-700 bg-teal-50 border border-teal-200 rounded px-2 py-1.5 leading-snug flex items-start gap-1.5">
                  <span className="flex-1">
                    Ordinal group — for ordered groups, <strong>Jonckheere-Terpstra</strong> tests a
                    monotonic trend (Kruskal-Wallis only tests "any difference").
                  </span>
                  <button onClick={() => { setTest("jonckheere"); setResult(null); }} className="flex-shrink-0 underline hover:text-teal-900">
                    Use Jonckheere
                  </button>
                </div>
              )}
            </div>
          )}

          {isKruskal && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Post-hoc Dunn correction</label>
              <div className="flex gap-0 rounded overflow-hidden border border-gray-200">
                {(["holm", "bonferroni", "fdr", "none"] as const).map((m) => (
                  <button key={m}
                    onClick={() => setPosthocCorrection(m)}
                    className={`flex-1 text-[10px] py-1 transition-colors ${
                      posthocCorrection === m ? "bg-indigo-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50"
                    }`}>
                    {m === "holm" ? "Holm" : m === "bonferroni" ? "Bonf." : m === "fdr" ? "FDR" : "None"}
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-gray-400 mt-1">Bonferroni = most conservative; Holm dominates it.</p>
            </div>
          )}

          {needsSecondCat && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Column variable</label>
              <select className="select w-full" value={col2} onChange={(e) => setCol2(e.target.value)}>
                {catCols.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}

          {/* ANCOVA / MANCOVA: covariates multi-select */}
          {(isAncova || isMancova) && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Covariates (continuous)</label>
              <select multiple className="select w-full h-24" value={covariates}
                onChange={(e) => setCovariates(Array.from(e.target.selectedOptions, o => o.value))}>
                {numCols.filter(c => isMancova ? !outcomes.includes(c) : c !== col).map((c) => <option key={c}>{c}</option>)}
              </select>
              <p className="text-[10px] text-gray-400 mt-1">Hold Ctrl/Cmd to select multiple</p>
            </div>
          )}

          {/* Two-way ANOVA: factor 2 */}
          {isTwoWay && (
            <div>
              <label className="text-xs text-gray-400 block mb-1">Factor 2</label>
              <select className="select w-full" value={factor2} onChange={(e) => setFactor2(e.target.value)}>
                {catCols.filter(c => c !== groupCol).map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}

          <button className="btn-primary w-full" onClick={run} disabled={loading || (isAncova && covariates.length === 0) || (isMancova && outcomes.length < 2)}>
            {loading ? "Running…" : "Run Test"}
          </button>
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>
      </div>

      <div className="flex-1 space-y-3">
        {TEST_GUIDANCE[test] && (
          <div className="panel bg-indigo-50 border-indigo-200 space-y-2">
            <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider">When to use</p>
            <p className="text-xs text-indigo-800 leading-relaxed">{TEST_GUIDANCE[test].when}</p>
            <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider mt-2">Assumptions</p>
            <p className="text-xs text-indigo-800 leading-relaxed">{TEST_GUIDANCE[test].assumptions}</p>
            <p className="text-[10px] font-bold text-indigo-900 uppercase tracking-wider mt-2">How to read</p>
            <p className="text-xs text-indigo-800 leading-relaxed">{TEST_GUIDANCE[test].reading}</p>
          </div>
        )}
        {result ? <ResultCard result={result} /> : (
          <div className="panel h-64 flex items-center justify-center text-gray-400">
            Configure and run a hypothesis test
          </div>
        )}
      </div>
    </div>
  );
}
