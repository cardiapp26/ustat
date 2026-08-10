import { useState, useRef } from "react";
import TitledPlot from "./TitledPlot";
import { runPower, parseArticle } from "../api";
import { useStore, paletteOf } from "../store";
import { usePersistedPanelState } from "../hooks/usePersistedPanelState";
import { Tip } from "./Tip";
import type { PlotCaptureHandle, PlotData, PlotLayout } from "../lib/plotTypes";

// Article-parser finding shape: only the fields this panel reads from it.
interface ArticleFinding {
  type?: string;
  test_label?: string;
  power_test?: TestId;
  effect_size?: number;
  effect_label?: string;
  statistic?: string;
  p?: number;
  n_approx?: number;
  k_groups?: number;
  p1?: number;
  p2?: number;
  source?: string;
}

const _pal = () => paletteOf(useStore.getState().plotTheme);

// ── Types ────────────────────────────────────────────────────────────────────

type TestId   = "t_two" | "t_one" | "anova" | "correlation" | "proportion" | "chi2" | "logistic" | "survival_cox";
type SolveFor = "n" | "power" | "effect_size";

interface CurvePoint { n: number; power: number }
interface PowerResult { result: number | null; label: string; curve: CurvePoint[]; result_text?: string }

// ── Constants ─────────────────────────────────────────────────────────────────

const TESTS: {
  id: TestId; label: string; short: string; desc: string; effectLabel: string;
  hasRatio: boolean; hasGroups: boolean; hasTails: boolean; isProportions: boolean;
  isRegression?: "logistic" | "cox";
}[] = [
  { id: "t_two",       label: "Independent t-test",        short: "t-test (2-grp)",
    desc: "Compare means of two independent groups (e.g. treatment vs. control).",
    effectLabel: "Cohen's d", hasRatio: true,  hasGroups: false, hasTails: true,  isProportions: false },
  { id: "t_one",       label: "One-sample / Paired t-test", short: "t-test (1-grp)",
    desc: "One group vs. a fixed value, or paired measurements (before/after).",
    effectLabel: "Cohen's d", hasRatio: false, hasGroups: false, hasTails: true,  isProportions: false },
  { id: "anova",       label: "One-way ANOVA",              short: "ANOVA",
    desc: "Compare means across three or more groups simultaneously.",
    effectLabel: "Cohen's f", hasRatio: false, hasGroups: true,  hasTails: false, isProportions: false },
  { id: "correlation", label: "Pearson correlation",        short: "Correlation",
    desc: "Detect a linear relationship between two continuous variables.",
    effectLabel: "Pearson r", hasRatio: false, hasGroups: false, hasTails: true,  isProportions: false },
  { id: "proportion",  label: "Two proportions (z-test)",   short: "Proportions",
    desc: "Compare event rates or percentages between two groups.",
    effectLabel: "Cohen's h", hasRatio: true,  hasGroups: false, hasTails: true,  isProportions: true  },
  { id: "chi2",        label: "Chi-square test",            short: "Chi-square",
    desc: "Test association between two categorical variables.",
    effectLabel: "Cohen's w", hasRatio: false, hasGroups: true,  hasTails: false, isProportions: false },
  { id: "logistic",    label: "Logistic regression",        short: "Logistic",
    desc: "Detect the effect (odds ratio) of one predictor on a binary outcome, adjusting for others.",
    effectLabel: "Odds ratio (OR)", hasRatio: false, hasGroups: false, hasTails: true, isProportions: false, isRegression: "logistic" },
  { id: "survival_cox", label: "Cox regression (survival)", short: "Cox PH",
    desc: "Detect the effect (hazard ratio) of one predictor on time-to-event, given the expected event rate.",
    effectLabel: "Hazard ratio (HR)", hasRatio: false, hasGroups: false, hasTails: true, isProportions: false, isRegression: "cox" },
];

const SOLVE_OPTS: { id: SolveFor; label: string; icon: string; desc: string }[] = [
  { id: "n",           icon: "👥", label: "Sample size (n)",   desc: "How many participants do I need?" },
  { id: "power",       icon: "⚡", label: "Power (1−β)",        desc: "What are my chances of detecting a real effect?" },
  { id: "effect_size", icon: "📏", label: "Effect size",         desc: "What is the smallest detectable effect?" },
];

const PRESETS: Record<string, [number, number, number]> = {
  t_two: [0.20, 0.50, 0.80], t_one: [0.20, 0.50, 0.80],
  anova: [0.10, 0.25, 0.40], correlation: [0.10, 0.30, 0.50], chi2: [0.10, 0.30, 0.50],
};

const COHEN_TABLE = [
  { effect: "Cohen's d  (t-tests)",  small: 0.20, medium: 0.50, large: 0.80 },
  { effect: "Cohen's f  (ANOVA)",    small: 0.10, medium: 0.25, large: 0.40 },
  { effect: "Pearson r  (corr.)",    small: 0.10, medium: 0.30, large: 0.50 },
  { effect: "Cohen's w  (χ²)",       small: 0.10, medium: 0.30, large: 0.50 },
  { effect: "Cohen's h  (props.)",   small: 0.20, medium: 0.50, large: 0.80 },
];

const TIPS = {
  alpha:  "The false-positive rate — probability of a significant result when no true effect exists. α = 0.05 (5%) is the worldwide standard. Stricter (0.01) reduces false alarms but needs more participants.",
  power:  "The probability of detecting a real effect. 80% is the minimum standard (you'd still miss 1 in 5 real effects). 90% is preferred for confirmatory or clinical trials.",
  d:      "Cohen's d = difference between group means ÷ pooled SD. d = 0.5 means means are half a SD apart. Use Small/Medium/Large if you have no prior estimate.",
  f:      "Cohen's f measures spread of group means in ANOVA relative to within-group variability. f = 0.25 is a medium effect — detectable but not dramatic.",
  r:      "Pearson r is the correlation coefficient. r = 0.3 means ~9% shared variance. r = 0.5 is a large, clinically meaningful correlation.",
  w:      "Cohen's w measures departure from expected frequencies in a chi-square table. Same Small/Medium/Large cutoffs as most other measures.",
  p1p2:   "Enter expected event rates in each group. Larger difference between p₁ and p₂ = smaller sample needed. Example: 30% controls vs 50% treated → enter 0.30 and 0.50.",
  or:     "The odds ratio you want to detect for a one-unit increase in the predictor (or exposed vs. unexposed). OR = 2.0 means the odds of the outcome double. Values nearer 1.0 are harder to detect and need a larger sample.",
  hr:     "The hazard ratio you want to detect. HR = 1.5 means a 50% higher instantaneous event rate in the exposed group. Cox power depends on the number of EVENTS, not just total n.",
  pEvent: "Overall probability of the binary outcome (event prevalence) in your sample. Power is greatest near 0.5 and falls for rare outcomes — a 5% outcome needs many more participants than a 50% one.",
  eventRate: "Proportion of participants expected to experience the event by end of follow-up. Cox power is driven by event count = n × event rate, so low event rates demand large cohorts.",
  pExposed: "Fraction of the cohort in the exposed / treated group (the predictor = 1). Balanced exposure (0.5) is most efficient; rare exposures need more participants.",
  n:      "Participants per group. Total N = n × (1 + ratio) for two-group designs. Equal groups (ratio = 1) give the best statistical efficiency.",
  tails:  "Two-tailed tests for effects in either direction (A > B or A < B) — use by default. One-tailed tests assume a direction in advance and need fewer participants, but require strong justification.",
  ratio:  "Size of group 2 relative to group 1. Ratio = 1 means equal groups (most efficient). Ratio = 2 means group 2 is twice as large.",
  groups: "Number of groups compared. More groups → more total participants needed to maintain 80% power.",
  cats:   "Number of categories in the chi-square table. A 2×2 table → 2 categories (df = 1). A 3-level variable → 3 categories.",
};

// Geometry only. Background, font and colourway follow the global chart theme
// at render; hard-coded here they made the theme picker a no-op on this panel.
const BASE_LAYOUT = {
  margin: { t: 12, r: 20, b: 48, l: 52 },
  xaxis:  { gridcolor: "#e5e7eb" },
  yaxis:  { gridcolor: "#e5e7eb", range: [0, 1.05], title: { text: "Power (1−β)" } },
};

// ── Plain-English result text ──────────────────────────────────────────────────

function plainEnglish(
  test: TestId, solveFor: SolveFor, resultVal: number,
  { alpha, power, effectSize, n, p1, p2, testInfo }: {
    alpha: string; power: string; effectSize: string; n: string;
    p1: string; p2: string; testInfo: typeof TESTS[0];
  }
): string {
  const pct   = (x: string | number) => `${(parseFloat(String(x)) * 100).toFixed(0)}%`;
  const es    = parseFloat(effectSize);
  const presets = PRESETS[test];
  const size  = presets
    ? (es <= presets[0] ? "small" : es <= presets[1] ? "small-to-medium" : es < presets[2] ? "medium" : "large")
    : "specified";
  const perGrp = testInfo.hasRatio || testInfo.hasGroups;
  const effectDesc = testInfo.isProportions
    ? `a difference of ${parseFloat(p1) * 100}% vs ${parseFloat(p2) * 100}%`
    : `a ${size} effect (${testInfo.effectLabel} = ${es})`;

  if (solveFor === "n") {
    const nCeil = Math.ceil(resultVal);
    return `To detect ${effectDesc} with ${pct(power)} power at α = ${alpha}, you need ${nCeil}${perGrp ? " per group" : ""}. If the true effect is this size or larger, your study has a ${pct(power)} chance of reaching statistical significance.`;
  }
  if (solveFor === "power") {
    const pctVal = (resultVal * 100).toFixed(1);
    if (resultVal >= 0.80)
      return `With ${n}${perGrp ? " per group" : ""}, you have a ${pctVal}% chance of detecting ${effectDesc}. This exceeds the 80% threshold — the study is adequately powered.`;
    if (resultVal >= 0.50)
      return `With ${n}${perGrp ? " per group" : ""}, your power is only ${pctVal}% for ${effectDesc}. This is below the 80% minimum — you would miss this effect about 1 in ${Math.round(1/(1-resultVal))} times.`;
    return `With only ${n}${perGrp ? " per group" : ""}, power is ${pctVal}% — severely underpowered for ${effectDesc}. You would miss this effect more than half the time. Increase sample size.`;
  }
  return `With ${n}${perGrp ? " per group" : ""} and ${pct(power)} power at α = ${alpha}, effects smaller than ${resultVal.toFixed(3)} (${testInfo.effectLabel}) will likely go undetected. If you expect a smaller true effect, recruit more participants.`;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function PowerPanel() {
  const showGrid = useStore((s) => s.showGrid);
  const plotTheme = useStore((s) => s.plotTheme);
  const themedLayout: Record<string, unknown> = {
    ...BASE_LAYOUT,
    paper_bgcolor: "transparent",
    plot_bgcolor: plotTheme.plotBg,
    font: { family: plotTheme.fontFamily, color: "#374151", size: plotTheme.fontSize },
    colorway: paletteOf(plotTheme),
  };
  const powerRef = useRef<PlotCaptureHandle | null>(null);

  const [test,       setTest]       = usePersistedPanelState<TestId>("power_sel", "test", "t_two");
  const [solveFor,   setSolveFor]   = usePersistedPanelState<SolveFor>("power_sel", "solveFor", "n");
  const [alpha,      setAlpha]      = usePersistedPanelState<string>("power_sel", "alpha", "0.05");
  const [power,      setPower]      = usePersistedPanelState<string>("power_sel", "power", "0.80");
  const [effectSize, setEffectSize] = usePersistedPanelState<string>("power_sel", "effectSize", "0.50");
  const [n,          setN]          = usePersistedPanelState<string>("power_sel", "n", "64");
  const [tails,      setTails]      = usePersistedPanelState<string>("power_sel", "tails", "2");
  const [ratio,      setRatio]      = usePersistedPanelState<string>("power_sel", "ratio", "1.0");
  const [kGroups,    setKGroups]    = usePersistedPanelState<string>("power_sel", "kGroups", "3");
  const [p1,         setP1]         = usePersistedPanelState<string>("power_sel", "p1", "0.50");
  const [p2,         setP2]         = usePersistedPanelState<string>("power_sel", "p2", "0.30");
  // Regression-power inputs (logistic + Cox)
  const [pEvent,     setPEvent]     = usePersistedPanelState<string>("power_sel", "pEvent", "0.30");
  const [eventRate,  setEventRate]  = usePersistedPanelState<string>("power_sel", "eventRate", "0.40");
  const [pExposed,   setPExposed]   = usePersistedPanelState<string>("power_sel", "pExposed", "0.50");

  const cachedPower = useStore((s) => s.panelCache.power);
  const setCachePower = useStore((s) => s.setPanelCache);
  const [result,  _setResultPower]  = useState<PowerResult | null>(((cachedPower as { result?: PowerResult | null } | undefined)?.result) ?? null);
  const setResult = (r: PowerResult | null) => { _setResultPower(r); setCachePower("power", { result: r }); };
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  // Article import state
  const [articleFindings, setArticleFindings] = useState<ArticleFinding[] | null>(null);
  const [articleLoading, setArticleLoading] = useState(false);
  const [articleError, setArticleError] = useState<string | null>(null);
  const [articleFile, setArticleFile] = useState<string | null>(null);
  const [showArticle, setShowArticle] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleArticleUpload = async (file: File) => {
    setArticleLoading(true); setArticleError(null);
    try {
      const res = await parseArticle(file);
      setArticleFindings(res.data.findings);
      setArticleFile(res.data.filename);
      if (res.data.findings.length === 0) {
        const preview = res.data.text_preview ? `\nPreview: "${res.data.text_preview.slice(0, 200)}…"` : "";
        setArticleError(`No statistical results found (${res.data.n_chars} chars extracted).${preview}`);
      }
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setArticleError(detail ?? "Failed to parse article");
    }
    finally { setArticleLoading(false); }
  };

  const applyFinding = (f: ArticleFinding) => {
    if (f.power_test) setTest(f.power_test);
    if (f.effect_size != null) setEffectSize(String(f.effect_size));
    if (f.n_approx) setN(String(f.n_approx));
    if (f.k_groups) setKGroups(String(f.k_groups));
    if (f.p1 != null) setP1(String(f.p1));
    if (f.p2 != null) setP2(String(f.p2));
    setSolveFor("n");
    setResult(null);
    setShowArticle(false);
  };

  const testInfo = TESTS.find((t) => t.id === test)!;
  const presets  = PRESETS[test];

  const effectTip: Record<TestId, string> = {
    t_two: TIPS.d, t_one: TIPS.d, anova: TIPS.f,
    correlation: TIPS.r, proportion: TIPS.p1p2, chi2: TIPS.w,
    logistic: TIPS.or, survival_cox: TIPS.hr,
  };

  // ── Calculate ────────────────────────────────────────────────────────────────

  const calculate = async () => {
    setLoading(true); setError(null);
    try {
      const payload: Record<string, unknown> = {
        test, solve_for: solveFor,
        alpha:    parseFloat(alpha)  || 0.05,
        tails:    parseInt(tails)    || 2,
        k_groups: parseInt(kGroups)  || 3,
        ratio:    parseFloat(ratio)  || 1.0,
        p1: parseFloat(p1), p2: parseFloat(p2),
      };
      if (solveFor !== "n")            payload.n           = parseInt(n);
      if (solveFor !== "power")        payload.power       = parseFloat(power);
      if (testInfo.isRegression === "logistic") {
        // effectSize holds the OR; backend logs it (OR>0 ⇒ log). Always sent so
        // the request validates even in solve-for-effect-size mode.
        payload.log_or  = parseFloat(effectSize);
        payload.p_event = parseFloat(pEvent);
      } else if (testInfo.isRegression === "cox") {
        payload.hr         = parseFloat(effectSize);  // effectSize holds the HR
        payload.event_rate = parseFloat(eventRate);
        payload.p_exposed  = parseFloat(pExposed);
      } else if (solveFor !== "effect_size" && !testInfo.isProportions) {
        payload.effect_size = parseFloat(effectSize);
      }
      const res = await runPower(payload);
      setResult(res.data);
      if (res.data.result != null) {
        if (solveFor === "n")          setN(String(Math.ceil(res.data.result)));
        else if (solveFor === "power") setPower(res.data.result.toFixed(4));
        else                           setEffectSize(res.data.result.toFixed(4));
      }
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      const fallback = e instanceof Error ? e.message : String(e);
      setError(typeof msg === "string" ? msg : (fallback ?? "Calculation failed"));
    } finally { setLoading(false); }
  };

  const switchTest  = (id: TestId)  => {
    // OR/HR live on a different scale than the Cohen-style effects, so reset the
    // effect field to a sensible default when crossing into/out of a regression
    // test (otherwise a Cohen's d of 0.5 would be read as OR=0.5, etc.).
    const fromReg = !!testInfo.isRegression;
    const toReg   = TESTS.find((t) => t.id === id)?.isRegression;
    if (toReg === "logistic") setEffectSize("2.0");
    else if (toReg === "cox") setEffectSize("1.5");
    else if (fromReg)         setEffectSize("0.50");
    setTest(id); setResult(null); setError(null);
  };
  const switchSolve = (s: SolveFor) => { setSolveFor(s); setResult(null); setError(null); };

  // ── Plot data ────────────────────────────────────────────────────────────────

  const currentN = solveFor === "n" && result?.result
    ? Math.ceil(result.result) : parseInt(n) || 0;
  const xLabel = (testInfo.hasRatio || testInfo.hasGroups) ? "n per group" : "Sample size (n)";

  const plotTraces: object[] = result?.curve.length ? [
    {
      type: "scatter", mode: "lines",
      x: result.curve.map((p) => p.n),
      y: result.curve.map((p) => p.power),
      line: { color: _pal()[0], width: 2.5 }, name: "Power",
      hovertemplate: "n = %{x}<br>Power = %{y:.3f}<extra></extra>",
    },
    {
      type: "scatter", mode: "lines",
      x: [result.curve[0]?.n ?? 4, result.curve[result.curve.length - 1]?.n ?? 200],
      y: [0.80, 0.80],
      line: { color: "#dc2626", width: 1.5, dash: "dash" },
      name: "80% threshold", hoverinfo: "skip",
    },
  ] : [];
  if (result?.curve.length && currentN) {
    const pt = [...result.curve].reverse().find((p) => p.n <= currentN) ?? result.curve[0];
    if (pt) plotTraces.push({
      type: "scatter", mode: "markers",
      x: [pt.n], y: [pt.power],
      marker: { color: _pal()[0], size: 10, line: { color: "#fff", width: 2 } },
      name: `<i>n</i> = ${pt.n}`,
      hovertemplate: `<i>n</i> = ${pt.n}<br>Power = ${pt.power.toFixed(3)}<extra></extra>`,
    });
  }

  // ── Style helpers ────────────────────────────────────────────────────────────

  const inputCls = (disabled: boolean) =>
    `w-full rounded-lg border px-3 py-2 text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-indigo-400 ${
      disabled
        ? "bg-indigo-50 border-indigo-300 text-indigo-700 font-semibold cursor-not-allowed"
        : "bg-white border-gray-300 text-gray-900"}`;

  const chipCls = (active: boolean) =>
    `flex-1 text-xs py-1.5 rounded-lg border transition-colors select-none cursor-pointer text-center ${
      active ? "bg-indigo-600 text-white border-indigo-600 font-medium shadow-sm"
             : "text-gray-500 border-gray-200 hover:border-indigo-300 hover:text-indigo-600 bg-white"}`;

  const powerColor = (pwr: number) =>
    pwr >= 0.80 ? "text-emerald-600" : pwr >= 0.50 ? "text-amber-600" : "text-red-500";
  const powerBg = (pwr: number) =>
    pwr >= 0.80 ? "bg-emerald-50 border-emerald-200" : pwr >= 0.50 ? "bg-amber-50 border-amber-200" : "bg-red-50 border-red-200";

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="max-w-5xl mx-auto space-y-4">

      {/* ── Test selector ── */}
      <div className="panel space-y-3">
        <div className="flex items-center gap-2 mb-1">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Statistical Test</h3>
          <Tip wide text="Choose the test that matches your study design. Your choice determines which formula is used to compute the sample size or power." />
        </div>
        <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
          {TESTS.map((t) => (
            <button key={t.id} onClick={() => switchTest(t.id)}
              title={t.desc}
              className={`px-2 py-2 rounded-xl border text-xs font-medium transition-all text-center leading-tight ${
                test === t.id
                  ? "bg-indigo-600 text-white border-indigo-600 shadow-md"
                  : "bg-white border-gray-200 text-gray-600 hover:border-indigo-300 hover:text-indigo-700"
              }`}>
              {t.short}
            </button>
          ))}
        </div>
        {/* Active test description */}
        <p className="text-xs text-gray-500 bg-gray-50 rounded-lg px-3 py-2 border border-gray-100">
          <span className="font-medium text-gray-700">{testInfo.label}:</span> {testInfo.desc}
        </p>
      </div>

      {/* ── Import from Article ── */}
      <div className="panel">
        <input ref={fileInputRef} type="file" accept=".pdf,.docx,.doc,.txt" className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleArticleUpload(f); e.target.value = ""; }} />
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button onClick={() => setShowArticle(!showArticle)}
              className="text-xs font-semibold text-gray-500 uppercase tracking-wider hover:text-indigo-600 transition-colors flex items-center gap-1">
              {showArticle ? "−" : "+"} Import from Article
            </button>
            <Tip wide text="Upload a published article (PDF/DOCX) to automatically extract effect sizes, sample sizes, and test statistics. The extracted values can be used directly for power analysis to replicate or extend the study." />
          </div>
          {articleFile && <span className="text-[10px] text-gray-400">{articleFile} — {articleFindings?.length ?? 0} findings</span>}
        </div>

        {showArticle && (
          <div className="mt-3 space-y-3">
            <div className="flex items-center gap-3">
              <button onClick={() => fileInputRef.current?.click()} disabled={articleLoading}
                className="px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors">
                {articleLoading ? "Parsing…" : "Upload PDF / DOCX"}
              </button>
              {articleError && <p className="text-xs text-red-500">{articleError}</p>}
            </div>

            {articleFindings && articleFindings.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs text-gray-500 font-medium">{articleFindings.length} statistical results extracted — click to use:</p>
                <div className="max-h-64 overflow-y-auto space-y-1.5">
                  {articleFindings.map((f, i: number) => (
                    <button key={i} onClick={() => applyFinding(f)}
                      className="w-full text-left bg-white border border-gray-200 rounded-lg px-3 py-2 hover:border-indigo-300 hover:bg-indigo-50 transition-colors group">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                            f.type === "t_test" ? "bg-blue-100 text-blue-700" :
                            f.type === "anova" ? "bg-purple-100 text-purple-700" :
                            f.type === "correlation" ? "bg-emerald-100 text-emerald-700" :
                            f.type === "odds_ratio" ? "bg-amber-100 text-amber-700" :
                            f.type === "hazard_ratio" ? "bg-red-100 text-red-700" :
                            f.type === "chi_square" ? "bg-orange-100 text-orange-700" :
                            f.type === "proportions" ? "bg-pink-100 text-pink-700" :
                            "bg-gray-100 text-gray-600"
                          }`}>{f.test_label ?? f.type}</span>
                          {f.effect_size != null && (
                            <span className="text-xs text-gray-700 font-mono">{f.effect_label} = {f.effect_size}</span>
                          )}
                          {f.statistic != null && !f.effect_size && (
                            <span className="text-xs text-gray-700 font-mono">{f.statistic}</span>
                          )}
                          {f.p != null && (
                            <span className={`text-[10px] ${f.p < 0.05 ? "text-indigo-600" : "text-gray-400"}`}><i>p</i> = {f.p < 0.001 ? "<0.001" : f.p}</span>
                          )}
                        </div>
                        <span className="text-[10px] text-gray-300 group-hover:text-indigo-500">Use →</span>
                      </div>
                      <p className="text-[10px] text-gray-400 mt-0.5 truncate">{f.source}</p>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Solve-for + Parameters + Calculate ── */}
      <div className="panel space-y-4">

        {/* Solve for */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Solve For</h3>
            <Tip text="Choose what you want to calculate. 'Sample size' is most common when planning a new study." />
          </div>
          <div className="flex gap-2">
            {SOLVE_OPTS.map(({ id, icon, label, desc }) => (
              <button key={id} onClick={() => switchSolve(id)}
                className={`flex-1 flex flex-col items-center gap-0.5 py-2.5 rounded-xl border transition-all ${
                  solveFor === id
                    ? "bg-indigo-600 text-white border-indigo-600 shadow-md"
                    : "bg-white border-gray-200 text-gray-600 hover:border-indigo-300 hover:text-indigo-600"
                }`}
                title={desc}>
                <span className="text-base">{icon}</span>
                <span className="text-xs font-medium">{label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Parameters grid */}
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">

          {/* Effect size */}
          {testInfo.isProportions ? (
            <div className="col-span-2 space-y-1">
              <label className="text-xs font-medium text-gray-500 flex items-center gap-1">
                Proportions (p₁ and p₂) <Tip text={TIPS.p1p2} wide />
              </label>
              <div className="flex gap-2">
                {([["p₁", p1, setP1], ["p₂", p2, setP2]] as const).map(([lbl, val, setter]) => (
                  <div key={lbl} className="flex-1">
                    <p className="text-[10px] text-gray-400 mb-1">{lbl}</p>
                    <input type="number" min="0.01" max="0.99" step="0.01"
                      className={inputCls(solveFor === "effect_size")}
                      value={val} disabled={solveFor === "effect_size"}
                      onChange={(e) => setter(e.target.value)} />
                  </div>
                ))}
              </div>
              {solveFor !== "effect_size" && (
                <p className="text-[10px] text-gray-400">
                  → Cohen's h = {Math.abs(2 * Math.asin(Math.sqrt(parseFloat(p1) || 0)) - 2 * Math.asin(Math.sqrt(parseFloat(p2) || 0))).toFixed(3)}
                </p>
              )}
            </div>
          ) : (
            <div className="space-y-1">
              <label className="text-xs font-medium text-gray-500 flex items-center gap-1">
                Effect size ({testInfo.effectLabel}) <Tip text={effectTip[test]} wide />
              </label>
              {solveFor === "effect_size" ? (
                <div className={`${inputCls(true)} flex items-center justify-center text-indigo-600 font-bold`}>
                  {result ? result.result!.toFixed(4) : "← will be calculated"}
                </div>
              ) : (
                <>
                  <input type="number" min="0.001" step="0.01"
                    className={inputCls(false)}
                    value={effectSize}
                    onChange={(e) => setEffectSize(e.target.value)} />
                  {presets && (
                    <div className="flex gap-1 mt-1">
                      {(["Small", "Medium", "Large"] as const).map((s, i) => (
                        <button key={s} onClick={() => setEffectSize(String(presets[i]))}
                          className={chipCls(parseFloat(effectSize) === presets[i])}>
                          {s} <span className="opacity-60 text-[9px]">({presets[i]})</span>
                        </button>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* Sample size */}
          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-500 flex items-center gap-1">
              {`n${testInfo.hasRatio || testInfo.hasGroups ? " per group" : ""}`} <Tip text={TIPS.n} wide />
            </label>
            {solveFor === "n" ? (
              <div className={`${inputCls(true)} flex items-center justify-center text-indigo-600 font-bold`}>
                {result ? Math.ceil(result.result!) : "← will be calculated"}
              </div>
            ) : (
              <input type="number" min="4" step="1"
                className={inputCls(false)}
                value={n}
                onChange={(e) => setN(e.target.value)} />
            )}
          </div>

          {/* Power */}
          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-500 flex items-center gap-1">
              Power (1−β) <Tip text={TIPS.power} wide />
            </label>
            {solveFor === "power" ? (
              <div className={`${inputCls(true)} flex items-center justify-center text-indigo-600 font-bold`}>
                {result ? `${(result.result! * 100).toFixed(1)}%` : "← will be calculated"}
              </div>
            ) : (
              <>
                <input type="number" min="0.01" max="0.999" step="0.01"
                  className={inputCls(false)}
                  value={power}
                  onChange={(e) => setPower(e.target.value)} />
                <div className="flex gap-1 mt-1">
                  {[["80%","0.80"],["90%","0.90"],["95%","0.95"]].map(([lbl, val]) => (
                    <button key={val} onClick={() => setPower(val)}
                      className={chipCls(power === val)}>{lbl}</button>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Alpha */}
          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-500 flex items-center gap-1">
              Alpha (α) <Tip text={TIPS.alpha} wide />
            </label>
            <div className="flex gap-1">
              {["0.01","0.05","0.10"].map((v) => (
                <button key={v} onClick={() => setAlpha(v)} className={chipCls(alpha === v)}>{v}</button>
              ))}
            </div>
          </div>
        </div>

        {/* Secondary params row */}
        {(testInfo.hasTails || testInfo.hasRatio || testInfo.hasGroups) && (
          <div className="flex flex-wrap gap-4 pt-2 border-t border-gray-100">
            {testInfo.hasTails && (
              <div className="space-y-1 min-w-[160px]">
                <label className="text-xs font-medium text-gray-500 flex items-center gap-1">
                  Tails <Tip text={TIPS.tails} wide />
                </label>
                <div className="flex gap-1">
                  {[["2","Two-tailed"],["1","One-tailed"]].map(([v,l]) => (
                    <button key={v} onClick={() => setTails(v)} className={chipCls(tails === v)}>{l}</button>
                  ))}
                </div>
              </div>
            )}
            {testInfo.hasRatio && (
              <div className="space-y-1 min-w-[120px]">
                <label className="text-xs font-medium text-gray-500 flex items-center gap-1">
                  Allocation ratio <Tip text={TIPS.ratio} wide />
                </label>
                <input type="number" min="0.1" step="0.1"
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  value={ratio} onChange={(e) => setRatio(e.target.value)} />
              </div>
            )}
            {testInfo.hasGroups && (
              <div className="space-y-1 min-w-[120px]">
                <label className="text-xs font-medium text-gray-500 flex items-center gap-1">
                  {test === "anova" ? "Groups (k)" : "Categories"} <Tip text={test === "anova" ? TIPS.groups : TIPS.cats} wide />
                </label>
                <input type="number" min="2" step="1"
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  value={kGroups} onChange={(e) => setKGroups(e.target.value)} />
              </div>
            )}
            {testInfo.isRegression === "logistic" && (
              <div className="space-y-1 min-w-[140px]">
                <label className="text-xs font-medium text-gray-500 flex items-center gap-1">
                  Event prevalence <Tip text={TIPS.pEvent} wide />
                </label>
                <input type="number" min="0.01" max="0.99" step="0.01"
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  value={pEvent} onChange={(e) => setPEvent(e.target.value)} />
              </div>
            )}
            {testInfo.isRegression === "cox" && (
              <>
                <div className="space-y-1 min-w-[140px]">
                  <label className="text-xs font-medium text-gray-500 flex items-center gap-1">
                    Event rate <Tip text={TIPS.eventRate} wide />
                  </label>
                  <input type="number" min="0.01" max="0.99" step="0.01"
                    className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    value={eventRate} onChange={(e) => setEventRate(e.target.value)} />
                </div>
                <div className="space-y-1 min-w-[140px]">
                  <label className="text-xs font-medium text-gray-500 flex items-center gap-1">
                    Exposed fraction <Tip text={TIPS.pExposed} wide />
                  </label>
                  <input type="number" min="0.01" max="0.99" step="0.01"
                    className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-300"
                    value={pExposed} onChange={(e) => setPExposed(e.target.value)} />
                </div>
              </>
            )}
          </div>
        )}

        {/* Calculate button */}
        <button
          className="w-full py-3 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm transition-colors shadow-sm flex items-center justify-center gap-2"
          onClick={calculate} disabled={loading}>
          {loading
            ? <><span className="animate-spin">⏳</span> Calculating…</>
            : <><span>⚡</span> Calculate {SOLVE_OPTS.find(s => s.id === solveFor)?.label}</>}
        </button>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-2.5 text-xs text-red-600">{error}</div>
        )}
      </div>

      {/* ── Result ── */}
      {result ? (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">

          {/* Result card + interpretation — 2 cols */}
          <div className="lg:col-span-2 space-y-3">

            {/* Big result number */}
            <div className={`panel border ${
              solveFor === "power" && result.result != null
                ? powerBg(result.result)
                : "border-indigo-200 bg-indigo-50"
            }`}>
              <div className="flex items-start justify-between mb-1">
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                  {SOLVE_OPTS.find(s => s.id === solveFor)?.label}
                </p>
                <span className="text-[10px] bg-white/60 border border-gray-200 rounded px-1.5 py-0.5 text-gray-500">
                  {testInfo.short}
                </span>
              </div>
              <p className={`text-5xl font-black leading-none mb-1 ${
                solveFor === "power" && result.result != null
                  ? powerColor(result.result)
                  : "text-indigo-700"
              }`}>
                {solveFor === "n"
                  ? Math.ceil(result.result ?? 0).toLocaleString()
                  : solveFor === "power"
                    ? `${((result.result ?? 0) * 100).toFixed(1)}%`
                    : result.result?.toFixed(4)}
              </p>
              <p className="text-sm text-gray-600">{result.label}</p>

              {/* Power bar (only when solving for power) */}
              {solveFor === "power" && result.result != null && (
                <div className="mt-3">
                  <div className="h-2.5 bg-white/60 rounded-full overflow-hidden border border-gray-200">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${
                        result.result >= 0.80 ? "bg-emerald-500" : result.result >= 0.50 ? "bg-amber-400" : "bg-red-400"
                      }`}
                      style={{ width: `${Math.min(100, (result.result ?? 0) * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[9px] text-gray-400 mt-0.5">
                    <span>0%</span>
                    <span className="text-red-400">80% target</span>
                    <span>100%</span>
                  </div>
                </div>
              )}

              {/* Input summary */}
              <div className="mt-3 pt-3 border-t border-white/40 grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px]">
                <span className="text-gray-500">α = <span className="text-gray-700 font-medium">{alpha}</span></span>
                {solveFor !== "power"       && <span className="text-gray-500">Power = <span className="text-gray-700 font-medium">{(parseFloat(power)*100).toFixed(0)}%</span></span>}
                {solveFor !== "n"           && <span className="text-gray-500"><i>n</i> = <span className="text-gray-700 font-medium">{n}</span></span>}
                {!testInfo.isProportions && solveFor !== "effect_size" &&
                  <span className="text-gray-500">ES = <span className="text-gray-700 font-medium">{effectSize}</span></span>}
                {testInfo.isProportions     && <><span className="text-gray-500">p₁ = {p1}</span><span className="text-gray-500">p₂ = {p2}</span></>}
                {testInfo.hasTails          && <span className="text-gray-500">{tails === "2" ? "Two-tailed" : "One-tailed"}</span>}
              </div>
            </div>

            {/* Plain-English interpretation */}
            {result.result != null && (
              <div className="rounded-xl border border-indigo-100 bg-white px-4 py-3 flex gap-3">
                <span className="text-indigo-400 text-xl flex-shrink-0 mt-0.5">💬</span>
                <p className="text-sm text-gray-700 leading-relaxed">
                  {result.result_text || plainEnglish(test, solveFor, result.result, {
                    alpha, power, effectSize, n, p1, p2, testInfo,
                  })}
                </p>
              </div>
            )}

            {/* Cohen conventions mini-table */}
            <div className="panel space-y-2">
              <p className="text-xs font-semibold text-gray-600 flex items-center gap-1">
                Cohen's Effect-Size Conventions
                <Tip text="Use these benchmarks when you have no prior data. Medium effects are the most common default in clinical research. Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.)." wide />
              </p>
              <table className="w-full text-[10px] border-collapse">
                <thead>
                  <tr className="border-b border-gray-200 text-gray-400">
                    <th className="text-left py-1 font-medium">Measure</th>
                    <th className="text-right py-1 font-medium text-amber-500">Small</th>
                    <th className="text-right py-1 font-medium text-indigo-500">Medium ★</th>
                    <th className="text-right py-1 font-medium text-emerald-600">Large</th>
                  </tr>
                </thead>
                <tbody>
                  {COHEN_TABLE.map((row) => (
                    <tr key={row.effect} className={`border-b border-gray-50 ${row.effect.includes(testInfo.effectLabel.split(" ")[0]) ? "bg-indigo-50/50" : ""}`}>
                      <td className="py-0.5 text-gray-600 font-mono">{row.effect}</td>
                      <td className="text-right py-0.5 font-mono text-gray-600">{row.small}</td>
                      <td className="text-right py-0.5 font-mono font-semibold text-indigo-600">{row.medium}</td>
                      <td className="text-right py-0.5 font-mono text-gray-600">{row.large}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Power curve — 3 cols */}
          {result.curve.length > 0 && (
            <div className="lg:col-span-3 panel space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-semibold text-gray-700">Power Curve</h4>
                <p className="text-[10px] text-gray-400">
                  Each point = power you'd have at that sample size. <span className="text-red-400">Red dashed</span> = 80% target. <span className="text-indigo-500">●</span> = your n.
                </p>
              </div>
              <TitledPlot
                plotRefOut={powerRef}
                storageKey="power:curve"
                data={plotTraces as PlotData[]}
                layout={{
                  ...themedLayout,
                  autosize: true, height: 320,
                  xaxis: { ...BASE_LAYOUT.xaxis, showgrid: showGrid, title: { text: xLabel } },
                  yaxis: { ...BASE_LAYOUT.yaxis, showgrid: showGrid },
                  legend: { orientation: "h" as const, y: -0.22, font: { size: 11 } },
                  shapes: currentN ? [{
                    type: "line", xref: "x", yref: "y",
                    x0: currentN, x1: currentN, y0: 0, y1: 1,
                    line: { color: _pal()[0], width: 1.5, dash: "dot" },
                  }] : [],
                } as PlotLayout}
                config={{ responsive: true, displaylogo: false, displayModeBar: false }}
                defaultTitle=""
                defaultSubtitle=""
                defaultXAxis={xLabel}
                defaultYAxis="Power (1−β)"
              />

              {/* Zone legend */}
              <div className="grid grid-cols-3 gap-2 pt-1 border-t border-gray-100">
                {[
                  { range: "< 50%",    color: "bg-red-100 text-red-700",     label: "Severely underpowered — likely to miss the effect" },
                  { range: "50–80%",   color: "bg-amber-100 text-amber-700", label: "Underpowered — below the accepted minimum" },
                  { range: "≥ 80%",    color: "bg-emerald-100 text-emerald-700", label: "Adequately powered — meets the standard" },
                ].map(z => (
                  <div key={z.range} className={`rounded-lg px-2 py-1.5 text-[9px] leading-snug ${z.color}`}>
                    <p className="font-bold text-[10px]">{z.range}</p>
                    <p className="opacity-80">{z.label}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        /* ── Empty state ── */
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-3">
            {[
              { icon: "🎯", color: "indigo", title: "Statistical Power",
                tip: "The probability of detecting a real effect. Aim for ≥ 80%.",
                body: "A study with 50% power is like flipping a coin — you'd miss half of all real effects. 80% is the minimum; 90% for confirmatory trials." },
              { icon: "📏", color: "violet", title: "Effect Size",
                tip: "How large the true difference is in standardized units.",
                body: "Cohen's presets (Small/Medium/Large) are safe defaults when you have no prior data. Medium is the most common assumption for planning." },
              { icon: "🚦", color: "rose",   title: "Significance (α)",
                tip: "The false-positive rate — probability of a result being a fluke.",
                body: "α = 0.05 means a 5% chance of a false positive. Stricter thresholds (0.01) demand more participants but reduce spurious findings." },
            ].map(({ icon, title, tip, body }) => (
              <div key={title} className="panel flex flex-col gap-2 border-t-4 border-indigo-200">
                <div className="flex items-center gap-2">
                  <span className="text-2xl">{icon}</span>
                  <p className="text-sm font-semibold text-gray-700 flex items-center gap-1">
                    {title} <Tip text={tip} />
                  </p>
                </div>
                <p className="text-xs text-gray-500 leading-relaxed">{body}</p>
              </div>
            ))}
          </div>

          <div className="panel bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-100">
            <p className="text-xs font-semibold text-indigo-700 mb-2">Quick start guide</p>
            <ol className="text-xs text-gray-600 space-y-1.5">
              {[
                ["1", "Choose your test", "e.g. Independent t-test for two-group comparisons"],
                ["2", "Select Solve For → Sample size", "most common when planning a new study"],
                ["3", "Set effect size", "use Medium (0.50 for d) if you have no prior estimate"],
                ["4", "Keep Power = 0.80 and α = 0.05", "the accepted standards for most journals"],
                ["5", "Click Calculate", "you'll see the required n, a plain-language interpretation, and a power curve"],
              ].map(([num, bold, rest]) => (
                <li key={num} className="flex gap-2.5">
                  <span className="w-5 h-5 rounded-full bg-indigo-600 text-white text-[10px] font-bold flex items-center justify-center flex-shrink-0">{num}</span>
                  <span><strong className="text-gray-700">{bold}</strong> — {rest}</span>
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}
    </div>
  );
}
