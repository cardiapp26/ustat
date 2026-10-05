import { useEffect, useState } from "react";
import { useStore, isNumericKind, isCategoricalKind, type Session } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { useStampedResult } from "../hooks/useStampedResult";
import { runBinomial, runChisquareGof, runOneProportion, runTwoProportions, runMcNemar, runCochranQ, runMantelHaenszel, runCochranArmitage, runPairedCategorical, getFrequency } from "../api";
import { fmtP, warningText } from "../lib/format";
import { describeStale } from "../lib/resultStamp";
import StaleResultNotice from "./StaleResultNotice";
import StaleGuard from "./StaleGuard";
import ResultExporter from "./ResultExporter";
import ResultProvenanceLine from "./ResultProvenanceLine";
import RiskMeasuresTable, { type RiskMeasures } from "./RiskMeasuresTable";
import GofExpectedInputs from "./GofExpectedInputs";
import {
  MAX_GOF_CATEGORIES, gofCategoriesFrom, gofKey, parseGofWeight, type GofCategory,
} from "../lib/gofProportions";
import type { Provenance } from "../lib/engine/provenance";

/** True when a stat-grid key holds a p-value (route through the canonical fmtP). */
function isPKey(k: string): boolean {
  return /^p$|^p_?value$|_p$|_p_?value$/.test(k);
}

const TESTS = [
  { id: "binomial",       label: "Binomial test",         group: "One-sample" },
  { id: "chisquare_gof",  label: "Chi-square goodness of fit (one sample)", group: "One-sample" },
  { id: "one_prop",       label: "One proportion z-test", group: "One-sample" },
  { id: "two_prop",       label: "Two proportions z-test", group: "Two-sample" },
  { id: "mcnemar",        label: "McNemar test",          group: "Paired" },
  { id: "bowker", label: "Bowker symmetry", group: "Paired" },
  { id: "stuart_maxwell", label: "Stuart-Maxwell homogeneity", group: "Paired" },
  { id: "cochran_q",      label: "Cochran's Q",           group: "Paired" },
  { id: "mantel_haenszel", label: "Mantel-Haenszel",      group: "Stratified" },
  { id: "cochran_armitage", label: "Cochran-Armitage trend", group: "Trend" },
] as const;

const GUIDANCE: Record<string, { when: string; reading: string }> = {
  binomial:  { when: "Test whether the observed proportion differs from an expected proportion (e.g. 50% heads).", reading: "p < 0.05 means the observed proportion significantly differs from the expected." },
  chisquare_gof: { when: "Test whether the observed distribution across the categories of one variable matches an expected distribution (equal shares, or a ratio such as 9:3:3:1).", reading: "p < 0.05 means the distribution departs from the expected one. Cohen's w is the effect size; adjusted residuals beyond about 2 show which categories drive the departure. With small expected counts (< 5) prefer the exact multinomial p." },
  one_prop:  { when: "z-test version of the binomial test, using normal approximation. Better for larger samples (n > 30).", reading: "Report: z, p, observed proportion, and 95% CI." },
  two_prop:  { when: "Compare proportions between two independent groups (e.g. treatment vs control event rates).", reading: "Cohen's h measures the effect size. Report proportions, z, p, and h." },
  mcnemar:   { when: "Test change in a binary outcome for paired data (e.g. before/after intervention on the same patients).", reading: "Tests whether discordant pairs (changed responses) are symmetric. OR of discordant pairs is the effect size." },
  bowker: { when: "Test symmetry of paired categorical responses across two conditions.", reading: "Significant chi-square indicates asymmetric changes between categories." },
  stuart_maxwell: { when: "Compare marginal category distributions across paired measurements.", reading: "Significant chi-square indicates changed marginal distributions." },
  cochran_q: { when: "Extension of McNemar for 3+ related binary measures. Tests whether proportions differ across conditions.", reading: "Significant Q means at least one proportion differs. Follow up with pairwise McNemar (Holm-corrected)." },
  mantel_haenszel: { when: "Test association between two binary variables while controlling for a stratifying variable (e.g. hospital site).", reading: "Common OR summarises the overall effect across strata. Homogeneity test checks whether the OR is consistent." },
  cochran_armitage: { when: "Test for a monotone linear trend in the proportion of a binary outcome across 3+ ordered groups (e.g. dose levels 0/1/2/3 vs adverse event).", reading: "Significant Z = the proportion changes linearly across the ordered groups. Sign of Z indicates direction (positive = increasing, negative = decreasing)." },
};

/** "Low, Medium, High" -> ["Low", "Medium", "High"]; blanks dropped. */
function parseLevelOrder(raw: string): string[] {
  return raw.split(",").map((s) => s.trim()).filter(Boolean);
}

interface RunFields {
  col: string;
  col2: string;
  groupCol: string;
  strataCol: string;
  nullProp: string;
  friedmanCols: string[];
  levelOrder: string;
  gofMode: "equal" | "custom";
  /** Typed weights for the selected column (defaults of 1 are not stored). */
  gofWeights: Record<string, string>;
}

/**
 * The inputs the selected test's request is built from, and only those. Every
 * picker persists across tests, so stamping them all would flag a binomial
 * result stale for a stratifying variable it never read. Mirrors the branches
 * in `run`. `nullProp` is stamped as typed, as HypothesisPanel does with `mu`.
 */
function runParamsFor(test: string, f: RunFields): Record<string, unknown> {
  switch (test) {
    case "binomial":
    case "one_prop": return { test, col: f.col, nullProp: f.nullProp };
    case "chisquare_gof": return { test, col: f.col, mode: f.gofMode, weights: f.gofMode === "custom" ? f.gofWeights : null };
    case "two_prop": return { test, col: f.col, groupCol: f.groupCol };
    case "bowker":
    case "stuart_maxwell":
    case "mcnemar": return { test, col: f.col, col2: f.col2 };
    case "cochran_q": return { test, friedmanCols: f.friedmanCols };
    case "mantel_haenszel": return { test, col: f.col, col2: f.col2, strataCol: f.strataCol };
    case "cochran_armitage": return { test, groupCol: f.groupCol, col: f.col, levelOrder: parseLevelOrder(f.levelOrder) };
    default: return { test };
  }
}

interface EffectSize {
  name?: string;
  value?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  ci_level?: number;
  magnitude?: string;
}

/** Breslow-Day homogeneity of the stratum odds ratios (Mantel-Haenszel). */
interface HomogeneityTest {
  name?: string;
  statistic?: number;
  df?: number;
  p?: number | null;
  adjusted?: boolean;
  homogeneous?: boolean;
}

/** McNemar: Newcombe interval for the paired difference in proportions. */
interface PairedDifference {
  estimate?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  method?: string;
  confidence_level?: number;
  positive_level?: string;
  note?: string;
}

/** One row of the goodness-of-fit table. */
interface GofRow {
  category: string;
  observed: number;
  expected_count: number;
  expected_proportion: number;
  observed_proportion: number;
  pearson_residual: number | null;
  adjusted_residual: number | null;
}

interface ExactMultinomial {
  p?: number | null;
  n_outcomes?: number;
  note?: string | null;
}

type ExportCell = string | number | boolean | null | undefined;

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
  warnings?: unknown[];
  table?: number[][];
  row_labels?: string[];
  col_labels?: string[];
  ci_proportion?: { low: number; high: number; method?: string; confidence_level?: number };
  /** Two proportions: ARD, RR, RRR and NNT/NNH. */
  risk_measures?: RiskMeasures;
  homogeneity_test?: HomogeneityTest | null;
  homogeneity_note?: string | null;
  paired_difference?: PairedDifference;
  /** Chi-square goodness of fit. */
  categories?: GofRow[];
  exact_multinomial?: ExactMultinomial;
  methods_text?: string;
  export_rows?: ExportCell[][];
  [key: string]: unknown;
}

/** 0.95 -> "95%". */
function levelPct(level: number | undefined): string {
  const pct = typeof level === "number" && level > 0 && level < 1 ? level * 100 : 95;
  return `${Number(pct.toPrecision(6))}%`;
}

const isFiniteNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const fixed = (v: number | null | undefined, d = 3) => (isFiniteNum(v) ? v.toFixed(d) : "n/a");
const pctText = (v: number) => `${(v * 100).toFixed(1)}%`;

function ResultCard({ result, stale = false, staleReason, provenance }: {
  result: CategoricalResult;
  stale?: boolean;
  staleReason?: string;
  provenance?: Provenance | null;
}) {
  const fmt = (v: unknown) => typeof v !== "number" ? String(v ?? "") : Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(3) : v.toFixed(4);
  const skip = new Set(["test","interpretation","result_text","significant","effect_sizes","assumptions","warnings","summary","posthoc","posthoc_method","export_rows","r_code","effects","table","row_labels","col_labels","plot_data","crosstab","strata_tables","methods_text","homogeneity_note"]);
  const stats = Object.entries(result).filter(([k, v]) => !skip.has(k) && typeof v !== "object");
  // The backend's export_rows lead with their own header row; booleans (e.g.
  // "Exact test") become text because the exporter takes strings and numbers.
  const exportTable = Array.isArray(result.export_rows) && result.export_rows.length > 1
    ? {
        headers: result.export_rows[0].map((h) => String(h ?? "")),
        rows: result.export_rows.slice(1).map((r) => r.map((c) => (typeof c === "boolean" ? String(c) : c))),
      }
    : null;
  const homogeneity = result.homogeneity_test ?? null;
  const heterogeneous = homogeneity != null && homogeneity.homogeneous === false;
  const exact = result.exact_multinomial;
  return (
    <div className="panel space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-gray-900">{result.test}</h4>
        <div className="flex items-center gap-2">
          {exportTable && (
            <ResultExporter title={result.test ?? "categorical_test"} headers={exportTable.headers} rows={exportTable.rows}
              stale={stale} staleReason={staleReason} provenance={provenance} />
          )}
          {"significant" in result && <span className={result.significant ? "badge-sig" : "badge-ns"}>{result.significant ? "Significant" : "Not significant"}</span>}
        </div>
      </div>
      <p className="text-sm text-gray-500 italic">{result.interpretation}</p>
      {result.table && result.row_labels && result.col_labels && <table className="w-full text-sm"><thead><tr><th>First / second</th>{result.col_labels.map(c => <th key={c}>{c}</th>)}</tr></thead><tbody>{result.table.map((row, i) => <tr key={i}><th>{result.row_labels?.[i]}</th>{row.map((v,j) => <td key={j}>{v}</td>)}</tr>)}</tbody></table>}
      {result.ci_proportion && <p className="text-sm">{result.ci_proportion.method ?? "Exact"} proportion CI: [{result.ci_proportion.low.toFixed(4)}, {result.ci_proportion.high.toFixed(4)}]</p>}
      {/* Warnings can invert the reading of a result (e.g. an assumed level
          ordering flips the trend direction), so they sit above the numbers. */}
      {heterogeneous && homogeneity && (
        <p role="alert" className="bg-amber-50 border border-amber-300 rounded-lg px-3 py-2 text-xs text-amber-900 leading-relaxed">
          ⚠ Caution: the odds ratios are not homogeneous across strata (Breslow-Day test, Tarone adjusted, <i>p</i> = {fmtP(homogeneity.p)}).
          A single pooled OR may be misleading; report the stratum-specific odds ratios as well.
        </p>
      )}
      {(result.warnings?.length ?? 0) > 0 && (
        <div className="space-y-1">
          {(result.warnings ?? []).map((w, i) => (
            <p key={i} className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-xs text-amber-800 leading-relaxed">
              ⚠ {warningText(w)}
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
      {result.risk_measures && <RiskMeasuresTable measures={result.risk_measures} />}
      {(homogeneity || result.homogeneity_note) && (
        <div className="rounded border border-gray-200 px-3 py-2 text-xs space-y-0.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-semibold text-gray-600">Homogeneity of odds ratios (Breslow-Day, Tarone)</span>
            {homogeneity ? (
              <span className="flex items-center gap-2">
                <span className="font-mono text-gray-700">
                  chi-square({homogeneity.df}) = {fixed(homogeneity.statistic)}, <i>p</i> = {fmtP(homogeneity.p)}
                </span>
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${heterogeneous ? "bg-amber-100 text-amber-800" : "bg-green-100 text-green-700"}`}>
                  {heterogeneous ? "heterogeneous" : "homogeneous"}
                </span>
              </span>
            ) : (
              <span className="text-gray-400">not computed</span>
            )}
          </div>
          {!homogeneity && result.homogeneity_note && <p className="text-[11px] text-gray-500">{result.homogeneity_note}</p>}
        </div>
      )}
      {result.paired_difference && (
        <div className="rounded border border-gray-200 px-3 py-2 text-xs space-y-0.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-semibold text-gray-600">
              Paired proportion difference (Newcombe {levelPct(result.paired_difference.confidence_level)} CI)
            </span>
            <span className="font-mono text-gray-700">
              {fixed(result.paired_difference.estimate)}{" "}
              {isFiniteNum(result.paired_difference.ci_low) && isFiniteNum(result.paired_difference.ci_high)
                ? `[${result.paired_difference.ci_low.toFixed(3)}, ${result.paired_difference.ci_high.toFixed(3)}]`
                : "[CI n/a]"}
            </span>
          </div>
          {result.paired_difference.note && <p className="text-[11px] text-gray-500">{result.paired_difference.note}</p>}
        </div>
      )}
      {result.categories && result.categories.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-600 mb-1">Observed and expected counts</p>
          <div className="overflow-auto rounded border border-gray-200">
            <table className="w-full text-xs">
              <thead><tr className="bg-gray-50">
                <th className="px-2 py-1 text-left">Category</th>
                <th className="px-2 py-1 text-right">Observed</th>
                <th className="px-2 py-1 text-right">Expected</th>
                <th className="px-2 py-1 text-right">Observed %</th>
                <th className="px-2 py-1 text-right">Expected %</th>
                <th className="px-2 py-1 text-right">Pearson residual</th>
                <th className="px-2 py-1 text-right">Adjusted residual</th>
              </tr></thead>
              <tbody>
                {result.categories.map((c) => (
                  <tr key={c.category} className="border-t border-gray-100">
                    <td className="px-2 py-1 font-medium">{c.category}</td>
                    <td className="px-2 py-1 text-right font-mono">{c.observed}</td>
                    <td className={`px-2 py-1 text-right font-mono ${c.expected_count < 5 ? "text-amber-700" : ""}`}>{c.expected_count.toFixed(2)}</td>
                    <td className="px-2 py-1 text-right font-mono">{pctText(c.observed_proportion)}</td>
                    <td className="px-2 py-1 text-right font-mono">{pctText(c.expected_proportion)}</td>
                    <td className="px-2 py-1 text-right font-mono">{fixed(c.pearson_residual, 2)}</td>
                    <td className={`px-2 py-1 text-right font-mono ${isFiniteNum(c.adjusted_residual) && Math.abs(c.adjusted_residual) >= 1.96 ? "font-bold" : ""}`}>{fixed(c.adjusted_residual, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-gray-400 mt-1">Expected counts below 5 are shown in amber; bold adjusted residuals exceed 1.96 in absolute value (exploratory, unadjusted for multiplicity).</p>
        </div>
      )}
      {exact && (isFiniteNum(exact.p) ? (
        <p className="text-sm">
          Exact multinomial <i>p</i> = <span className="font-mono">{fmtP(exact.p)}</span>
          {isFiniteNum(exact.n_outcomes) && <span className="text-xs text-gray-400"> ({exact.n_outcomes.toLocaleString()} outcomes enumerated)</span>}
        </p>
      ) : exact.note ? (
        <p className="text-[11px] text-gray-500">{exact.note}</p>
      ) : null)}
      {(result.effect_sizes?.length ?? 0) > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-semibold text-gray-600">Effect Sizes</p>
          {(result.effect_sizes ?? []).map((es: EffectSize, i: number) => (
            <div key={i} className="flex items-center gap-3 bg-indigo-50 rounded-lg px-3 py-1.5 text-xs">
              <span className="font-semibold text-indigo-800">{es.name?.replace(/_/g, " ")}</span>
              <span className="font-mono text-indigo-700">{es.value?.toFixed(3) ?? "n/a"}</span>
              {es.ci_low != null && es.ci_high != null && (
                <span className="text-indigo-500">
                  {levelPct(es.ci_level ?? result.paired_difference?.confidence_level)} CI: [{es.ci_low.toFixed(3)}, {es.ci_high.toFixed(3)}]
                </span>
              )}
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
      {result.methods_text && (
        <details className="text-xs"><summary className="text-gray-400 cursor-pointer hover:text-indigo-600">Methods text</summary>
          <p className="mt-1 p-2 bg-gray-50 rounded-lg text-gray-600 leading-relaxed">{result.methods_text}</p>
        </details>
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
  // Goodness of fit: equal shares, or per-category weights typed as numbers,
  // percentages or ratio parts (the server rescales them to sum to 1).
  const [gofMode, setGofMode] = usePersistedPanelState<"equal" | "custom">("categorical_tests", "gofMode", "equal");
  const [gofWeightsAll, setGofWeightsAll] = usePersistedPanelState<Record<string, string>>("categorical_tests", "gofWeights", {});
  const gofPrefix = gofKey(col, "");
  const gofWeights = Object.fromEntries(
    Object.entries(gofWeightsAll).filter(([k]) => k.startsWith(gofPrefix)).map(([k, v]) => [k.slice(gofPrefix.length), v]),
  );
  const runParams = runParamsFor(test, { col, col2, groupCol, strataCol, nullProp, friedmanCols, levelOrder, gofMode, gofWeights });
  const {
    result, setResult, stale, staleReasons: staleWhy, stamp,
  } = useStampedResult<CategoricalResult>("categorical_tests", runParams);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const isPaired = ["mcnemar", "bowker", "stuart_maxwell"].includes(test);
  const isCochran = test === "cochran_q";
  const isMH = test === "mantel_haenszel";
  const isTwoProp = test === "two_prop";
  const isCA = test === "cochran_armitage";
  const isGof = test === "chisquare_gof";
  const needsNull = test === "binomial" || test === "one_prop";
  // Ordinal exposure + binary outcome → Cochran-Armitage tests a dose-response
  // trend that a plain proportion/χ² test ignores.
  const ordinalNames = new Set(session.columns.filter((c) => c.kind === "ordinal").map((c) => c.name));
  const suggestTrend = !isCA && [col, col2, groupCol].some((v) => ordinalNames.has(v));

  // Observed categories of the chosen column, fetched from the frequency
  // endpoint so a weight can be typed against each one. Keyed by column so a
  // slow answer for the previous column is never shown under the new one.
  const dataVersion = useStore((s) => s.dataVersion);
  const [gofLoad, setGofLoad] = useState<{ col: string; cats: GofCategory[] | null; error: string | null } | null>(null);
  const sessionId = session.session_id;
  const floatColumn = (session.columns.find((c) => c.name === col)?.dtype ?? "").startsWith("float");
  useEffect(() => {
    if (!isGof || !col) return;
    let cancelled = false;
    getFrequency(sessionId, col)
      .then((res) => {
        if (cancelled) return;
        const table = res.data?.[col];
        setGofLoad({ col, cats: table ? gofCategoriesFrom(table, floatColumn) : [], error: null });
      })
      .catch(() => {
        if (!cancelled) setGofLoad({ col, cats: null, error: "Could not load the categories of this column." });
      });
    return () => { cancelled = true; };
  }, [isGof, col, sessionId, dataVersion, floatColumn]);
  const gofCats = gofLoad?.col === col ? gofLoad.cats : null;
  const gofLoadError = gofLoad?.col === col ? gofLoad.error : null;
  const gofTooMany = (gofCats?.length ?? 0) > MAX_GOF_CATEGORIES;
  // Typed weight per category (unedited rows read as 1), parsed for the request.
  const gofParsed = (gofCats ?? []).map((c) => ({ value: c.value, weight: parseGofWeight(gofWeights[c.value] ?? "1") }));
  const gofCustomValid = gofCats != null && gofCats.length >= 2 && !gofTooMany && gofParsed.every((c) => c.weight !== null);
  const setGofWeight = (category: string, text: string) =>
    setGofWeightsAll((prev) => {
      const key = gofKey(col, category);
      const rest = Object.fromEntries(Object.entries(prev).filter(([k]) => k !== key));
      return text.trim() === "1" ? rest : { ...rest, [key]: text };
    });
  const applyGofRatio = (weights: number[]) =>
    setGofWeightsAll((prev) => {
      const kept = Object.fromEntries(Object.entries(prev).filter(([k]) => !k.startsWith(gofPrefix)));
      const next: Record<string, string> = { ...kept };
      (gofCats ?? []).forEach((c, i) => {
        if (weights[i] !== 1) next[gofKey(col, c.value)] = String(weights[i]);
      });
      return next;
    });

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    const sid = session.session_id;
    try {
      let res: { data?: CategoricalResult } | null = null;
      if (test === "binomial") res = await runBinomial({ session_id: sid, column: col, expected_proportion: +nullProp });
      else if (test === "chisquare_gof") {
        res = await runChisquareGof({
          session_id: sid, column: col, alpha: 0.05,
          ...(gofMode === "custom"
            ? { expected_proportions: Object.fromEntries(gofParsed.map((c) => [c.value, c.weight])) }
            : {}),
        });
      }
      else if (test === "one_prop") res = await runOneProportion({ session_id: sid, column: col, null_proportion: +nullProp });
      else if (test === "two_prop") res = await runTwoProportions({ session_id: sid, column: col, group_column: groupCol });
      else if (test === "bowker" || test === "stuart_maxwell") res = await runPairedCategorical({ session_id: sid, col1: col, col2, method: test });
      else if (test === "mcnemar") res = await runMcNemar({ session_id: sid, col1: col, col2: col2 });
      else if (test === "cochran_q") res = await runCochranQ({ session_id: sid, columns: friedmanCols });
      else if (test === "mantel_haenszel") res = await runMantelHaenszel({ session_id: sid, row_col: col, col_col: col2, strata_col: strataCol });
      else if (test === "cochran_armitage") {
        const order = parseLevelOrder(levelOrder);
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
            <label className="text-xs text-gray-400 block mb-1">{isMH ? "Row variable" : isCA ? "Binary outcome (event)" : isGof ? "Categorical column" : (test === "bowker" || test === "stuart_maxwell") ? "First categorical measurement" : "Binary column"}</label>
            <select className="select w-full" value={col} onChange={(e) => setCol(e.target.value)}>
              {binCols.map((c) => <option key={c}>{c}</option>)}
            </select>
          </div>
          {isGof && (
            <GofExpectedInputs
              mode={gofMode}
              onModeChange={(m) => { setGofMode(m); }}
              cats={gofCats}
              loadError={gofLoadError}
              tooMany={gofTooMany}
              weights={gofWeights}
              parsed={gofParsed}
              onWeight={setGofWeight}
              onApplyRatio={applyGofRatio}
            />
          )}
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
          <button className="btn-primary w-full" onClick={run} disabled={loading || (isCochran && friedmanCols.length < 3) || (isGof && gofMode === "custom" && !gofCustomValid)}>
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
        {result && <ResultProvenanceLine provenance={stamp?.provenance} />}
        {result && stale && (
          <StaleResultNotice
            reasons={staleWhy}
            onRecompute={run}
            busy={loading}
            what={`This ${result.test ?? "test"}`}
          />
        )}
        {result ? (
          <StaleGuard stale={stale} reason={describeStale(staleWhy)}>
            <ResultCard result={result} stale={stale} staleReason={describeStale(staleWhy)} provenance={stamp?.provenance} />
          </StaleGuard>
        ) : (
          <div className="panel text-center text-gray-400 py-12">Select a test and configure variables</div>
        )}
      </div>
    </div>
  );
}
