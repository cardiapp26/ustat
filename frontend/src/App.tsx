import "./index.css";
import { Component, useState, useRef, useEffect, useMemo, type ReactNode } from "react";
import { BarChart2, Table2, FlaskConical, GitMerge, Brain, X, TrendingUp, ClipboardList, Calculator, Grid3x3, Grid2x2, Shapes, FolderOpen, Target, Filter, Info, Save, Search, Layers, Scale, HelpCircle, Command } from "lucide-react";
import { clearCases, saveSession as saveSessionApi } from "./api";
import AboutModal from "./components/AboutModal";
import HelpModal from "./components/HelpModal";
import { useAutoSession } from "./hooks/useAutoSession";
import { exportDataset, downloadSessionJson, type ExportFmt } from "./lib/exportDataset";

class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(error: Error) {
    // Surface the failure for debugging without taking down the whole app.
    console.error("Panel render error:", error);
  }
  private reset = () => this.setState({ error: null });
  render() {
    const { error } = this.state;
    if (error) return (
      <div className="flex-1 flex items-center justify-center p-8 overflow-y-auto">
        <div className="max-w-lg w-full bg-white border border-red-200 rounded-2xl shadow-sm p-6 text-center">
          <div className="text-red-500 text-3xl mb-2">⚠️</div>
          <h2 className="text-lg font-semibold text-gray-800">This panel hit an error</h2>
          <p className="text-sm text-gray-500 mt-1">
            The rest of the app is fine — switch tabs, or try again. Your data is unaffected.
          </p>
          <div className="mt-4 flex items-center justify-center gap-3">
            <button
              onClick={this.reset}
              className="px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
            >
              Try again
            </button>
            <RefreshAppButton variant="inline" label="Update app" confirmBeforeReload />
          </div>
          <p className="mt-2 text-xs text-gray-400">
            Still broken? Update the app to fetch the latest version.
          </p>
          <details className="mt-4 text-left">
            <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-600">Technical details</summary>
            <pre className="mt-2 p-3 bg-gray-50 rounded-lg text-[11px] text-red-600 whitespace-pre-wrap overflow-x-auto max-h-60">
              {error.message}{error.stack ? "\n\n" + error.stack : ""}
            </pre>
          </details>
        </div>
      </div>
    );
    return this.props.children;
  }
}
import { useStore } from "./store";
import UploadZone from "./components/UploadZone";
import DataTable from "./components/DataTable";
import DescriptivePanel from "./components/DescriptivePanel";
import ChartsPanel from "./components/ChartsPanel";
import SubgroupBarPanel from "./components/SubgroupBarPanel";
import ScoreCompositePanel from "./components/ScoreCompositePanel";
import KMCompositePanel from "./components/KMCompositePanel";
import ForestBuilderPanel from "./components/ForestBuilderPanel";
import HypothesisPanel from "./components/HypothesisPanel";
import CorrelationPanel from "./components/CorrelationPanel";
import ModelsPanel from "./components/ModelsPanel";
import VisualModelPanel from "./components/VisualModelPanel";
import AddedValuePanel from "./components/AddedValuePanel";
import InternalValidationPanel from "./components/InternalValidationPanel";
import CausalPanel from "./components/CausalPanel";
import ROCPanel from "./components/ROCPanel";
import Table1Panel from "./components/Table1Panel";
import PowerPanel from "./components/PowerPanel";
import { usePersistedPanelState } from "./hooks/usePersistedPanelState";
import ComputePanel from "./components/ComputePanel";
import PSMPanel from "./components/PSMPanel";
import IPTWPanel from "./components/IPTWPanel";
import RepeatedMeasuresPanel from "./components/RepeatedMeasuresPanel";
import CategoricalTestsPanel from "./components/CategoricalTestsPanel";
import ReliabilityPanel from "./components/ReliabilityPanel";
import GatekeepingPanel from "./components/GatekeepingPanel";
import NonInferiorityPanel from "./components/NonInferiorityPanel";
import PlotThemeBar from "./components/PlotThemeBar";
import RefreshAppButton from "./components/RefreshAppButton";
import SurvivalAdvancedPanel from "./components/SurvivalAdvancedPanel";
import RCSPanel from "./components/RCSPanel";
import DecisionCurvePanel from "./components/DecisionCurvePanel";
import MLPanel from "./components/MLPanel";
import TimeSeriesPanel from "./components/TimeSeriesPanel";
import MetaPanel from "./components/MetaPanel";
import WeightedStatsPanel from "./components/WeightedStatsPanel";
import MissingDataPanel from "./components/MissingDataPanel";
import FactorPCAPanel from "./components/FactorPCAPanel";
import BayesianPanel from "./components/BayesianPanel";
import CloudSyncBar from "./components/CloudSyncBar";
import CommandPalette from "./components/CommandPalette";
import { cloudSync } from "./lib/cloudSync";
import { purgeExpiredTrash, notifySessionsChanged, TRASH_PURGE_INTERVAL_MS } from "./lib/sessionDb";

const TABS = [
  { id: "data",        label: "Data",        icon: Table2 },
  { id: "summary",     label: "Summary",     icon: BarChart2 },
  { id: "table1",      label: "Table",       icon: ClipboardList },
  { id: "tests",       label: "Tests",       icon: FlaskConical },
  { id: "correlation", label: "Correlation", icon: GitMerge },
  { id: "roc",         label: "ROC",         icon: TrendingUp },
  { id: "models",      label: "Models",      icon: Brain },
  { id: "psm",         label: "PSM",         icon: Target },
  { id: "iptw",        label: "IPTW",        icon: Scale },
  { id: "causal",      label: "Causal+",     icon: GitMerge }, // IV / mediation / target-trial
  { id: "dca",         label: "DCA",         icon: Target }, // Decision Curve Analysis (Phase 13)
  { id: "meta",        label: "Meta",        icon: Layers },
  { id: "missing",     label: "Missing",     icon: Filter },
  { id: "visual",      label: "Visual",      icon: Shapes },
  { id: "compute",     label: "Compute",     icon: Calculator },
];

// Searchable catalog of every test / model / analysis exposed by the app,
// keyed by the tab id where the test lives. The header search box matches
// keywords case-insensitively across `name`, `aliases`, and `tab`; clicking
// a result jumps to that tab. Aliases include short codes, alternative
// names, and common typos so the user can find a test by whatever they
// remember (e.g. "M-W", "U test" → Mann-Whitney; "lojistik" → Logistic).
interface TestEntry {
  name: string;
  tab: string;
  aliases?: string[];
  group?: string;
}

const TEST_CATALOG: TestEntry[] = [
  // Hypothesis (Tests tab)
  { name: "One-sample t-test", tab: "tests", group: "Parametric", aliases: ["t test", "tek örneklem"] },
  { name: "Independent t-test", tab: "tests", group: "Parametric", aliases: ["two sample t", "bağımsız t"] },
  { name: "One-way ANOVA", tab: "tests", group: "Parametric", aliases: ["anova", "tek yönlü varyans"] },
  { name: "ANCOVA", tab: "tests", group: "Parametric", aliases: ["covariance analysis"] },
  { name: "MANCOVA", tab: "tests", group: "Parametric", aliases: ["multivariate ancova", "manova", "pillai", "wilks", "multivariate analysis of covariance", "çok değişkenli kovaryans"] },
  { name: "Added Predictive Value", tab: "visual", group: "Models", aliases: ["incremental value", "delta auc", "nri", "idi", "reclassification", "added value", "predictor improves model", "discrimination calibration", "eklenen değer"] },
  { name: "Internal Validation", tab: "models", group: "Models", aliases: ["bootstrap", "optimism", "optimism correction", "harrell", "cross-validation", "cross validation", "k-fold", "10-fold", "overfitting", "optimism-corrected auc", "calibration slope", "shrinkage", "internal validation", "dahili doğrulama"] },
  { name: "External Validation", tab: "models", group: "Models", aliases: ["external validation", "validation cohort", "transportability", "calibration in the large", "delong", "hosmer lemeshow", "o/e ratio", "calibration plot", "val.prob", "dış doğrulama", "harici doğrulama"] },
  { name: "Instrumental Variable (2SLS)", tab: "causal", group: "Causal", aliases: ["iv", "instrumental variable", "2sls", "two stage least squares", "endogeneity", "wu-hausman", "sargan", "enstrümantal değişken"] },
  { name: "Causal Mediation", tab: "causal", group: "Causal", aliases: ["mediation", "mediator", "acme", "ade", "indirect effect", "sobel", "proportion mediated", "baron kenny", "aracılık"] },
  { name: "Target Trial Emulation", tab: "causal", group: "Causal", aliases: ["target trial", "emulation", "iptw ate", "eligibility", "hernan", "emulated trial", "hedef çalışma"] },
  { name: "Difference-in-Differences", tab: "causal", group: "Causal", aliases: ["did", "diff in diff", "parallel trends", "pre post", "fark farkı"] },
  { name: "Regression Discontinuity", tab: "causal", group: "Causal", aliases: ["rdd", "rd design", "cutoff", "running variable", "local linear", "süreksizlik"] },
  { name: "DAG Backdoor Analysis", tab: "causal", group: "Causal", aliases: ["dag", "directed acyclic graph", "backdoor", "adjustment set", "confounder", "mediator", "collider", "dagitty"] },
  { name: "Two-way ANOVA", tab: "tests", group: "Parametric", aliases: ["iki yönlü anova", "factorial"] },
  { name: "Mann-Whitney U", tab: "tests", group: "Non-parametric", aliases: ["m-w", "u test", "wilcoxon rank"] },
  { name: "Kruskal-Wallis", tab: "tests", group: "Non-parametric", aliases: ["kw", "nonparametric anova"] },
  { name: "Jonckheere-Terpstra trend", tab: "tests", group: "Non-parametric", aliases: ["jt", "trend test", "tertil trend", "ordered"] },
  { name: "Chi-square", tab: "tests", group: "Categorical", aliases: ["chi2", "ki kare"] },
  { name: "Fisher's exact", tab: "tests", group: "Categorical", aliases: ["fisher exact"] },

  // Categorical (Tests tab — under CategoricalTestsPanel; same hub)
  { name: "Binomial test", tab: "tests", group: "Categorical" },
  { name: "One-proportion z-test", tab: "tests", group: "Categorical" },
  { name: "Two-proportions z-test", tab: "tests", group: "Categorical", aliases: ["two prop"] },
  { name: "McNemar test", tab: "tests", group: "Paired", aliases: ["paired binary"] },
  { name: "Cochran's Q", tab: "tests", group: "Paired", aliases: ["cochran q"] },
  { name: "Mantel-Haenszel", tab: "tests", group: "Stratified", aliases: ["cmh", "common or"] },
  { name: "Cochran-Armitage trend", tab: "tests", group: "Trend", aliases: ["ca trend", "doz cevap"] },
  { name: "Gatekeeping (truncated Hochberg / Holm)", tab: "tests", group: "Multiplicity", aliases: ["gatekeeping", "hochberg", "holm", "multiplicity", "hierarchical", "çoklu test", "fwer", "endpoint hierarchy"] },
  { name: "Non-inferiority test (RR / RD / OR / mean, margin)", tab: "tests", group: "Trial design", aliases: ["non-inferiority", "noninferiority", "non inferiority", "margin", "equivalence", "üstünlük dışılık", "itt", "intention to treat", "90% ci"] },

  // Correlation
  { name: "Pearson correlation", tab: "correlation", aliases: ["pearson r"] },
  { name: "Spearman correlation", tab: "correlation", group: "Non-parametric", aliases: ["spearman rho"] },
  { name: "Kendall tau", tab: "correlation" },

  // ROC
  { name: "ROC curve", tab: "roc", aliases: ["roc analysis", "auc"] },
  { name: "DeLong AUC comparison", tab: "roc", aliases: ["delong", "auc compare"] },
  { name: "Multi-curve ROC", tab: "roc", aliases: ["multi roc"] },
  { name: "Combined ROC model", tab: "roc", aliases: ["combined model"] },

  // Models
  { name: "Linear Regression", tab: "models", aliases: ["lineer regresyon", "ols", "lm"] },
  { name: "Logistic Regression", tab: "models", aliases: ["lojistik regresyon", "logit"] },
  { name: "Firth Logistic (penalized)", tab: "models", aliases: ["firth", "penalized logistic", "rare events"] },
  { name: "OR Table (Uni + Multi)", tab: "models", aliases: ["ortable", "or table"] },
  { name: "Firth OR Table", tab: "models", aliases: ["firth ortable"] },
  { name: "Poisson Regression", tab: "models", aliases: ["count regression", "irr"] },
  { name: "Kaplan-Meier", tab: "models", aliases: ["km", "survival"] },
  { name: "Cox Proportional Hazards", tab: "models", aliases: ["cox model", "hr", "survival regression"] },
  { name: "Cox time-varying", tab: "models", aliases: ["cox tv", "time varying"] },
  { name: "RCS Dose-Response", tab: "models", aliases: ["restricted cubic spline", "spline"] },
  { name: "Cox-RCS (multivariable)", tab: "models", aliases: ["cox rcs", "multivariable spline"] },
  { name: "Negative Binomial", tab: "models", aliases: ["nb regression"] },
  { name: "Gamma GLM", tab: "models" },
  { name: "Polynomial", tab: "models", aliases: ["polynomial regression"] },
  { name: "Ordinal Logistic", tab: "models", aliases: ["proportional odds"] },
  { name: "Mixed-effects (LMM)", tab: "models", aliases: ["lmm", "linear mixed"] },
  { name: "GEE", tab: "models", aliases: ["generalized estimating equations"] },
  { name: "Stepwise selection", tab: "models", aliases: ["forward backward stepwise"] },
  { name: "Random Forest", tab: "models", group: "Machine Learning", aliases: ["rf", "random orman", "ensemble", "ml", "makine öğrenmesi", "predictive"] },
  { name: "Gradient Boosting", tab: "models", group: "Machine Learning", aliases: ["gbm", "boosting", "ml", "predictive", "xgboost"] },
  { name: "Feature importance", tab: "models", group: "Machine Learning", aliases: ["permutation importance", "değişken önemi", "shap"] },
  { name: "ARIMA / SARIMA forecast", tab: "models", group: "Time Series", aliases: ["arima", "sarima", "zaman serisi", "time series", "forecast", "tahmin"] },
  { name: "Seasonal decomposition (STL)", tab: "models", group: "Time Series", aliases: ["stl", "decompose", "seasonal", "trend", "mevsimsellik"] },
  { name: "Stationarity (ADF / KPSS) + ACF/PACF", tab: "models", group: "Time Series", aliases: ["adf", "kpss", "stationarity", "acf", "pacf", "durağanlık"] },

  // Visual
  { name: "Polynomial fit (Visual)", tab: "visual", aliases: ["polynomial visual"] },
  { name: "Random Forest", tab: "visual", aliases: ["rf", "random forest"] },

  // PSM
  { name: "Propensity Score Matching", tab: "psm", aliases: ["psm matching"] },
  { name: "IPTW", tab: "iptw", aliases: ["inverse probability weighting", "weighted", "iptw weighting"] },
  // Phase 13
  { name: "Decision Curve Analysis", tab: "dca", aliases: ["dca", "decision curve", "net benefit", "vickers", "clinical utility"] },

  // Weighted / survey
  { name: "Weighted descriptives (survey weights)", tab: "summary", group: "Weighted", aliases: ["weighted mean", "survey", "sampling weights", "ağırlıklı", "kish", "horvitz"] },

  // Meta-analysis
  { name: "Meta-analysis (random / fixed effects)", tab: "meta", group: "Meta-analysis", aliases: ["meta analiz", "pooled", "forest", "dersimonian", "random effects"] },
  { name: "Subgroup meta-analysis", tab: "meta", group: "Meta-analysis", aliases: ["subgroup", "alt grup"] },
  { name: "Meta-regression", tab: "meta", group: "Meta-analysis", aliases: ["meta regression", "moderator", "bubble"] },
  { name: "Publication bias (Egger / Begg / funnel)", tab: "meta", group: "Meta-analysis", aliases: ["egger", "begg", "funnel", "trim and fill", "yayın yanlılığı"] },

  // Survival Advanced (lives in Models tab as a sub-section)
  { name: "Fine-Gray competing risks", tab: "models", aliases: ["competing risks", "shr", "subdistribution"] },
  { name: "RMST", tab: "models", aliases: ["restricted mean survival time"] },
  { name: "Recurrent events (LWYY)", tab: "models", group: "Survival", aliases: ["lwyy", "lin wei yang ying", "andersen gill", "recurrent", "tekrarlayan", "mean cumulative function", "mcf", "rate ratio"] },
  { name: "E-value", tab: "models", aliases: ["sensitivity unmeasured confounding"] },
  { name: "Log-rank test", tab: "models", aliases: ["logrank", "km test"] },
  { name: "Schoenfeld residuals", tab: "models", aliases: ["proportional hazards check", "ph"] },

  // Summary / Descriptive
  { name: "Descriptive statistics", tab: "summary", aliases: ["betimsel", "tanımlayıcı"] },
  { name: "Histogram", tab: "summary" },
  { name: "Boxplot", tab: "summary", aliases: ["kutu grafiği"] },
  { name: "Violin plot", tab: "summary" },
  { name: "Q-Q plot", tab: "summary", aliases: ["qq normality"] },

  // Table 1
  { name: "Table 1 (clinical baseline)", tab: "table1", aliases: ["baseline table", "tablo 1"] },

  // Compute
  { name: "Recode", tab: "compute", aliases: ["recoding"] },
  { name: "Formula", tab: "compute" },
  { name: "BMI / eGFR / CHA2DS2-VASc", tab: "compute", aliases: ["clinical calc"] },

  // Missing
  { name: "Missing data audit", tab: "missing", aliases: ["eksik veri", "mice", "imputation"] },

  // Reliability / repeated (lives under tests too)
  { name: "Cronbach’s alpha", tab: "tests", aliases: ["reliability"] },
  { name: "Fleiss’ kappa", tab: "tests", aliases: ["agreement"] },
  { name: "ICC", tab: "tests", aliases: ["intraclass correlation"] },

  // Factor Analysis / PCA
  { name: "Principal Component Analysis (PCA)", tab: "tests", aliases: ["pca", "faktör analizi", "scree plot", "loadings", "varimax", "promax"] },
  { name: "Exploratory Factor Analysis (EFA)", tab: "tests", aliases: ["efa", "factor analysis", "kmo", "bartlett"] },

  // Bayesian
  { name: "Bayesian T-test", tab: "tests", aliases: ["bayes", "bayesian t test", "jzs", "bf10"] },
  { name: "Bayesian Correlation", tab: "tests", aliases: ["bayesian pearson", "rho prior"] },
  { name: "Bayesian Regression", tab: "tests", aliases: ["bayesian multiple regression", "bic bf"] },
];

/** Download via fetch + blob + anchor click. Iframe-based downloads swallow
 *  backend errors silently (the iframe loads the error HTML and nothing
 *  reaches the user) — this surfaces failures via the catch path. */
async function downloadViaFetch(url: string, filename: string, mime: string) {
  const res = await import("./api").then(m => m.default.get(url, { responseType: "blob" }));
  const ct = (res.headers["content-type"] || "").toString();
  if (ct.includes("application/json")) {
    const txt = await (res.data as Blob).text();
    throw new Error(`Server returned JSON instead of file: ${txt.slice(0, 200)}`);
  }
  const blob = new Blob([res.data], { type: mime });
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objectUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}

async function triggerDownload(sessionId: string, format: "csv" | "xlsx", originalFilename: string) {
  const outName = originalFilename.replace(/\.(csv|xlsx|sav|xls|sas7bdat|dta)$/i, "") + `_export.${format}`;
  const mime = format === "xlsx"
    ? "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    : "text/csv";
  await downloadViaFetch(`/api/sessions/${sessionId}/export/${format}?filename=${encodeURIComponent(outName)}`, outName, mime);
}

/** Save Session → fetch JSON as Blob via axios (same-origin) and trigger
 *  an anchor-click download. Resolves when the download has been initiated
 *  so callers can sequence post-download cleanup. */
async function triggerSessionDownload(sessionId: string, filename?: string) {
  const res = await saveSessionApi(sessionId);
  const blob = new Blob([res.data], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = (filename ?? `session_${sessionId.slice(0, 8)}`).replace(/\.[^.]+$/, "") + ".json";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Modal asking user to save before opening a new file */
function SaveBeforeOpenModal({
  session,
  onSave,
  onSkip,
  onCancel,
}: {
  session: { filename: string; columns: { name: string }[]; preview: Record<string, unknown>[] };
  onSave: (fmt: "csv" | "xlsx" | "json") => void;
  onSkip: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-sm p-6 space-y-4">
        <h3 className="font-semibold text-gray-900 text-base">Save current dataset?</h3>
        <p className="text-sm text-gray-500">
          Do you want to save <strong>{session.filename}</strong> before opening a new file?
        </p>
        <div className="flex flex-col gap-2">
          <div className="flex gap-3">
            <button
              onClick={() => onSave("csv")}
              className="flex-1 btn-primary text-sm py-2"
            >
              Save as CSV
            </button>
            <button
              onClick={() => onSave("xlsx")}
              className="flex-1 btn-primary text-sm py-2"
            >
              Save as XLSX
            </button>
            <button
              onClick={() => onSave("json")}
              className="flex-1 btn-primary text-sm py-2"
              title="Download .json then open a new file"
            >
              Session + Open New
            </button>
          </div>
          <button
            onClick={onSkip}
            className="w-full text-sm text-gray-600 border border-gray-200 rounded-lg py-2 hover:bg-gray-50 transition-colors"
          >
            Don't save, open new file
          </button>
          <button
            onClick={onCancel}
            className="w-full text-xs text-gray-400 hover:text-gray-700 py-1"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

function TestsCombo() {
  const [sub, setSub] = usePersistedPanelState<"hypothesis" | "repeated" | "categorical" | "reliability" | "noninferiority" | "gatekeeping" | "factor" | "bayesian">("combo_tests", "sub", "hypothesis");
  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="flex gap-1 px-4 pt-2 pb-1 bg-tint border-b border-line flex-shrink-0">
        {([["hypothesis", "Hypothesis"], ["repeated", "Repeated Measures"], ["categorical", "Categorical"], ["reliability", "Reliability"], ["noninferiority", "Non-Inferiority"], ["gatekeeping", "Gatekeeping"], ["factor", "Factor Analysis"], ["bayesian", "Bayesian Statistics"]] as const).map(([id, label]) => (
          <button key={id} onClick={() => setSub(id)}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
              sub === id ? "bg-surface text-ink-600 shadow-card border border-line" : "text-slate-500 hover:text-slate-700 hover:bg-chip"
            }`}>
            {label}
          </button>
        ))}
      </div>
      <div className="flex-1 p-4 overflow-y-auto">
        {sub === "hypothesis" && <HypothesisPanel />}
        {sub === "repeated" && <RepeatedMeasuresPanel />}
        {sub === "categorical" && <CategoricalTestsPanel />}
        {sub === "reliability" && <ReliabilityPanel />}
        {sub === "noninferiority" && <NonInferiorityPanel />}
        {sub === "gatekeeping" && <GatekeepingPanel />}
        {sub === "factor" && <div className="flex-1 overflow-y-auto"><FactorPCAPanel /></div>}
        {sub === "bayesian" && <div className="flex-1 overflow-y-auto"><BayesianPanel /></div>}
      </div>
    </div>
  );
}

function ComputeCombo() {
  // Dictionary moved to the Data tab toolbar as a modal. Compute is now a
  // single panel.
  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="flex-1 p-4 overflow-y-auto">
        <ComputePanel />
      </div>
    </div>
  );
}

function SummaryCombo() {
  const [sub, setSub] = usePersistedPanelState<"descriptive" | "weighted">("combo_summary", "sub", "descriptive");
  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="flex gap-1 px-4 pt-2 pb-1 bg-tint border-b border-line flex-shrink-0">
        {([["descriptive", "Descriptive"], ["weighted", "Weighted (survey)"]] as const).map(([id, label]) => (
          <button key={id} onClick={() => setSub(id)}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
              sub === id ? "bg-surface text-ink-600 shadow-card border border-line" : "text-slate-500 hover:text-slate-700 hover:bg-chip"
            }`}>
            {label}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto">
        {sub === "descriptive" ? <DescriptivePanel /> : <WeightedStatsPanel />}
      </div>
    </div>
  );
}

function ModelsCombo() {
  const [sub, setSub] = usePersistedPanelState<"regression" | "survival" | "rcs" | "ml" | "timeseries" | "validation">("combo_models", "sub", "regression");
  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="flex gap-1 px-4 pt-2 pb-1 bg-tint border-b border-line flex-shrink-0">
        {([["regression", "Regression"], ["survival", "Survival Advanced"], ["rcs", "Restricted Cubic Spline"], ["ml", "Machine Learning", "dev"], ["timeseries", "Time Series"], ["validation", "Validation (internal / external)"]] as const).map(([id, label, badge]) => (
          <button key={id} onClick={() => setSub(id)}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-colors inline-flex items-center gap-1.5 ${
              sub === id ? "bg-surface text-ink-600 shadow-card border border-line" : "text-slate-500 hover:text-slate-700 hover:bg-chip"
            }`}>
            {label}
            {badge && (
              <span className="text-[8px] font-bold uppercase px-1 py-0.5 rounded bg-amber-100 text-amber-700 border border-amber-300"
                title="Under development — not yet tested to the standard of the other panels">
                {badge}
              </span>
            )}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto">
        {sub === "regression" ? <div className="p-4"><ModelsPanel /></div>
          : sub === "rcs" ? <div className="p-4"><RCSPanel /></div>
          : sub === "ml" ? <div className="p-4"><MLPanel /></div>
          : sub === "timeseries" ? <TimeSeriesPanel />
          : sub === "validation" ? <div className="p-4"><InternalValidationPanel /></div>
          : <div className="p-4"><SurvivalAdvancedPanel /></div>}
      </div>
    </div>
  );
}

function VisualChartsCombo() {
  const [sub, setSub] = usePersistedPanelState<"models" | "charts" | "subgroup" | "score" | "kmcomposite" | "forest" | "addedvalue">("combo_visual", "sub", "models");
  // Deep-link: another panel can request a specific inner sub-tab (e.g.
  // the Cox time-horizon panel jumps straight to "forest"). Consume once.
  const visualSubTab = useStore((s) => s.visualSubTab);
  const setVisualSubTab = useStore((s) => s.setVisualSubTab);
  useEffect(() => {
    if (visualSubTab) {
      setSub(visualSubTab as typeof sub);
      setVisualSubTab(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visualSubTab]);
  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="flex gap-1 px-4 pt-2 pb-1 bg-tint border-b border-line flex-shrink-0">
        {([
          ["models", "Models & Diagnostics"],
          ["addedvalue", "Added Predictive Value"],
          ["charts", "Charts"],
          ["subgroup", "Subgroup Bar Chart"],
          ["score", "Score Figure"],
          ["kmcomposite", "KM Composite (endpoints)"],
          ["forest", "Forest plot (sensitivity / multi-endpoint)"],
        ] as const).map(([id, label]) => (
          <button key={id} onClick={() => setSub(id)}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
              sub === id ? "bg-surface text-ink-600 shadow-card border border-line" : "text-slate-500 hover:text-slate-700 hover:bg-chip"
            }`}>
            {label}
          </button>
        ))}
      </div>
      <div className="flex-1 p-4 overflow-y-auto">
        {sub === "models" ? <VisualModelPanel />
          : sub === "addedvalue" ? <AddedValuePanel />
          : sub === "charts" ? <ChartsPanel />
          : sub === "subgroup" ? <SubgroupBarPanel />
          : sub === "score" ? <ScoreCompositePanel />
          : sub === "kmcomposite" ? <KMCompositePanel />
          : <ForestBuilderPanel />}
      </div>
    </div>
  );
}

/**
 * Header pill that displays the session filename and lets the user
 * rename it in place. Click → input opens with the current name
 * selected. Enter / blur commits via store.renameSession (which also
 * fires POST /api/sessions/{sid}/rename so save_session round-trips
 * the new value). Escape cancels.
 */
function SessionNamePill() {
  const session = useStore((s) => s.session);
  const renameSession = useStore((s) => s.renameSession);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  if (!session) return null;

  const commit = () => {
    const next = draft.trim();
    if (next && next !== session.filename) renameSession(next);
    setEditing(false);
  };

  return (
    <div className="flex items-center gap-1 bg-gray-50 border border-gray-200 rounded-lg px-2 py-1 min-w-0 max-w-xs">
      {editing ? (
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") { e.preventDefault(); commit(); }
            if (e.key === "Escape") { e.preventDefault(); setEditing(false); }
          }}
          maxLength={200}
          className="text-xs text-gray-800 bg-white border border-indigo-300 rounded px-1 py-0 outline-none focus:border-indigo-500 min-w-0 w-44"
          aria-label="Rename this session"
        />
      ) : (
        <button
          onClick={() => { setDraft(session.filename); setEditing(true); }}
          title="Click to rename this session"
          className="text-xs text-gray-600 truncate hover:text-indigo-700 hover:underline decoration-dotted underline-offset-2 cursor-pointer text-left"
        >
          {session.filename}
        </button>
      )}
      <span className="text-xs text-gray-400 ml-2 flex-shrink-0">
        {session.rows.toLocaleString()} × {session.columns.length}
      </span>
    </div>
  );
}

export default function App() {
  const { session, activeTab, setActiveTab, clearSession, showGrid, toggleGrid, caseFilter, setCaseFilter, originalSession, setOriginalSession, setSession } = useStore();
  const caseFilterKey = caseFilter
    ? JSON.stringify([caseFilter.conditions, caseFilter.selected, caseFilter.total])
    : "all";
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  // ⌘K / Ctrl+K command palette. Toggles a rule-based NL command box that
  // resolves a phrase like "roc group age" to the ROC tab with the outcome
  // and score columns pre-filled. Pure client-side — no LLM, no network.
  const [paletteOpen, setPaletteOpen] = useState(false);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  // Header auto-save status indicator.
  const [autoSaveStatus, setAutoSaveStatus] = useState<{
    state: "idle" | "saving" | "saved" | "error";
    at?: number;
  }>({ state: "idle" });
  // Stream every session change to IndexedDB (debounced + periodic +
  // beforeunload flush). Surfaces "Autosaved · …" next to the
  // header Save menu when the user has an open session.
  useAutoSession({
    onStatus: (state, at) => setAutoSaveStatus({ state, at }),
  });
  // Initialise Google Drive cloud sync once on mount. Restores a cached
  // token (silent re-auth if needed) and kicks off the background pull so
  // sessions saved on another device appear in Recent Sessions. Safe to
  // call before sign-in — it just warms up the GIS client. (Idempotent.)
  useEffect(() => {
    void cloudSync.init().catch((e) => console.warn("[cloud] init", e));
    // Trash auto-expiry: purge trashed records older than 30 days on startup
    // and every 5 minutes while the app is open. Runs before any Drive push
    // so expired tombstones are removed locally AND remotely on the next sync.
    void purgeExpiredTrash()
      .then((n) => { if (n > 0) notifySessionsChanged(); })
      .catch((e) => console.warn("[trash] purge failed", e));
    const t = setInterval(() => {
      void purgeExpiredTrash()
        .then((n) => { if (n > 0) notifySessionsChanged(); })
        .catch((e) => console.warn("[trash] purge failed", e));
    }, TRASH_PURGE_INTERVAL_MS);
    return () => clearInterval(t);
  }, []);
  // Header Save-As dropdown (consolidates dataset export + session JSON
  // — supersedes the dropdown previously hidden inside the DataTable
  // toolbar). Closes on outside-click.
  const [showHeaderSaveMenu, setShowHeaderSaveMenu] = useState(false);
  const headerSaveMenuRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!showHeaderSaveMenu) return;
    const handler = (e: MouseEvent) => {
      if (headerSaveMenuRef.current && !headerSaveMenuRef.current.contains(e.target as Node)) {
        setShowHeaderSaveMenu(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showHeaderSaveMenu]);
  // Code tab is always visible; the panel itself handles the
  // "ENABLE_CODE_RUNNER not set" case with an in-page disabled banner.
  const visibleTabs = TABS;

  // Header test-search box state. The dropdown lists tests matching the
  // current query against `name`, `aliases`, and `group` (case-insensitive,
  // word-substring). Picking a match jumps to that tab. Closes on
  // outside-click and on Escape so it never traps focus.
  const [testQuery, setTestQuery] = useState("");
  const [showTestSearch, setShowTestSearch] = useState(false);
  const testSearchRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!showTestSearch) return;
    const handler = (e: MouseEvent) => {
      if (testSearchRef.current && !testSearchRef.current.contains(e.target as Node)) {
        setShowTestSearch(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showTestSearch]);

  const testMatches = useMemo(() => {
    const q = testQuery.trim().toLowerCase();
    if (!q) return [];
    return TEST_CATALOG
      .filter((t) => {
        const hay = [t.name, t.group ?? "", ...(t.aliases ?? [])].join(" ").toLowerCase();
        return q.split(/\s+/).every((tok) => hay.includes(tok));
      })
      .slice(0, 12);
  }, [testQuery]);

  const tabLabel = (id: string) => TABS.find((t) => t.id === id)?.label ?? id;

  const handleOpenNew = () => setShowSaveModal(true);

  const handleSave = async (fmt: "csv" | "xlsx" | "json") => {
    if (!session) return;
    try {
      if (fmt === "json") {
        await triggerSessionDownload(session.session_id, session.filename);
        // JSON download is fully complete (blob fetched + anchor clicked).
        // Safe to clear immediately.
        setShowSaveModal(false);
        clearSession();
      } else {
        // CSV/XLSX go via fetch+blob now (was iframe). Await so failures
        // surface in the catch block and the React tree stays mounted on
        // error — clearing the session only after the download completes.
        await triggerDownload(session.session_id, fmt, session.filename);
        setShowSaveModal(false);
        clearSession();
      }
    } catch (e) {
      console.error("Save failed:", e);
      const msg = e instanceof Error ? e.message : String(e);
      alert(`Save failed: ${msg}`);
      // Keep the modal and session intact so the user can retry.
    }
  };

  const handleSkip = () => {
    setShowSaveModal(false);
    clearSession();
  };

  const handleCancel = () => setShowSaveModal(false);

  if (!session) return <UploadZone />;

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-page">
      {showAbout && <AboutModal onClose={() => setShowAbout(false)} />}
      {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
      {showSaveModal && (
        <SaveBeforeOpenModal
          session={session}
          onSave={handleSave}
          onSkip={handleSkip}
          onCancel={handleCancel}
        />
      )}

      {/* Header — two rows so tabs always have full width */}
      <header className="border-b border-line bg-surface flex-shrink-0 shadow-card">
        {/* Row 1: logo · filename · actions */}
        <div className="flex items-center gap-3 px-4 pt-2 pb-1.5">
          {/* Logo + wordmark — clickable. Returns to the landing screen
              (drop zone + Power Analysis + Recent Sessions). The autosave
              hook has already mirrored the current session to IndexedDB,
              so the user can resume from the Recent list. We still
              confirm in case they haven't seen the auto-save indicator
              yet. */}
          <button
            type="button"
            onClick={() => {
              const ok = window.confirm(
                "Back to the home screen? The current session was saved automatically — resume it from the Recent work list."
              );
              if (!ok) return;
              clearSession();
              setActiveTab("data");
            }}
            title="Back to the home screen (drop zone, Power Analysis, Recent work)"
            className="flex items-center gap-2 flex-shrink-0 hover:opacity-80 transition-opacity cursor-pointer"
          >
            <img src="/logo.png" alt="uSTAT logo" className="w-7 h-7 rounded-lg" />
            <span className="font-bold text-gray-900 text-sm tracking-tight">uSTAT</span>
          </button>

          <SessionNamePill />


          {caseFilter && caseFilter.conditions && caseFilter.conditions.length > 0 && (
            <div className="flex items-center gap-1 bg-amber-50 border border-amber-300 text-amber-700 font-semibold rounded-lg px-2 py-1 text-[10px] animate-pulse flex-shrink-0" title="All active analyses are automatically filtered by this subset.">
              🧹 Select Cases Active ({caseFilter.selected.toLocaleString()})
            </div>
          )}

          {originalSession && (
            <button
              onClick={() => {
                setSession(originalSession);
                setOriginalSession(null);
                setActiveTab("data");
              }}
              className="flex items-center gap-1 bg-indigo-50 border border-indigo-200 text-indigo-700 hover:bg-indigo-100 font-bold rounded-lg px-2.5 py-1 text-[10px] flex-shrink-0 transition-colors shadow-sm cursor-pointer"
              title={`Return to original dataset: ${originalSession.filename}`}
            >
              ↩️ Return to Original Dataset ({originalSession.filename})
            </button>
          )}

          {/* Test / analysis search. Type a name (e.g. "ROC", "Cox", "Firth",
              "Jonckheere") to find which tab it lives in; click a result
              to jump there. Aliases include short codes and Turkish names
              so "lojistik" → Logistic, "ki kare" → Chi-square. */}
          <div className="relative w-72 max-w-xs flex-shrink-0" ref={testSearchRef}>
            <div className="relative">
              <Search size={13} className="absolute left-2 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
              <input
                value={testQuery}
                onChange={(e) => { setTestQuery(e.target.value); setShowTestSearch(true); }}
                onFocus={() => setShowTestSearch(true)}
                onKeyDown={(e) => {
                  if (e.key === "Escape") { setShowTestSearch(false); (e.target as HTMLInputElement).blur(); }
                  if (e.key === "Enter" && testMatches.length > 0) {
                    const top = testMatches[0];
                    setActiveTab(top.tab);
                    setShowTestSearch(false);
                    setTestQuery("");
                  }
                }}
                placeholder="Search tests / models…  (ROC, Cox, Firth, Jonckheere)"
                className="w-full pl-7 pr-2 py-1 text-xs border border-gray-200 rounded-lg bg-white focus:outline-none focus:border-indigo-400 placeholder-gray-400"
              />
            </div>
            {showTestSearch && testQuery.trim() && (
              <div className="absolute left-0 right-0 top-full mt-1 bg-white border border-gray-200 rounded-xl shadow-xl z-50 max-h-96 overflow-y-auto">
                {testMatches.length === 0 ? (
                  <p className="text-xs text-gray-400 px-3 py-2">No match for "{testQuery}".</p>
                ) : (
                  testMatches.map((t, i) => (
                    <button
                      key={`${t.name}-${i}`}
                      onClick={() => { setActiveTab(t.tab); setShowTestSearch(false); setTestQuery(""); }}
                      className="w-full flex items-center justify-between gap-2 px-3 py-1.5 text-left hover:bg-indigo-50 transition-colors border-b border-gray-100 last:border-b-0"
                    >
                      <span className="flex flex-col min-w-0">
                        <span className="text-xs text-gray-800 truncate">{t.name}</span>
                        {t.group && <span className="text-[10px] text-gray-400">{t.group}</span>}
                      </span>
                      <span className="text-[10px] text-indigo-600 bg-indigo-50 rounded px-1.5 py-0.5 flex-shrink-0">
                        {tabLabel(t.tab)}
                      </span>
                    </button>
                  ))
                )}
              </div>
            )}
          </div>

          {/* ⌘K command palette trigger. Opens a rule-based NL command box:
              "roc group age" → ROC tab, form pre-filled. Keyboard equivalent:
              Cmd/Ctrl+K (handled by the window listener above). */}
          <button
            onClick={() => setPaletteOpen(true)}
            className="flex items-center gap-1.5 px-2 py-1 rounded-lg border border-gray-200 bg-white text-gray-500 hover:text-indigo-600 hover:border-indigo-300 transition-colors flex-shrink-0"
            title="Command palette — run analyses by name (Cmd/Ctrl+K)"
          >
            <Command size={13} />
            <kbd className="text-[10px] font-mono">⌘K</kbd>
          </button>

          <div className="ml-auto flex items-center gap-1.5">
            <PlotThemeBar />
            <button
              onClick={toggleGrid}
              className={`p-1.5 rounded-lg transition-colors ${showGrid ? "text-indigo-500 bg-indigo-50 hover:bg-indigo-100" : "text-gray-400 hover:text-gray-700 hover:bg-gray-100"}`}
              title={showGrid ? "Hide chart grid lines" : "Show chart grid lines"}
            >
              {showGrid ? <Grid3x3 size={16} /> : <Grid2x2 size={16} />}
            </button>
            <RefreshAppButton confirmBeforeReload />
            <button
              onClick={() => setShowHelp(true)}
              className="p-1.5 rounded-lg text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 transition-colors"
              title="Help & Interactive Tutorials"
            >
              <HelpCircle size={16} />
            </button>
            <button
              onClick={() => setShowAbout(true)}
              className="p-1.5 rounded-lg text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 transition-colors"
              title="About uSTAT — packages & methods"
            >
              <Info size={16} />
            </button>
            {/* Auto-save status pill — sits to the LEFT of the Save menu
                so the user always sees that work is being mirrored to
                local IndexedDB (and can hit Save explicitly if not). */}
            {session && (
              <div
                className="flex items-center gap-1 text-[10px] text-gray-400 px-2"
                title={
                  autoSaveStatus.state === "saving" ? "Autosaving…"
                  : autoSaveStatus.state === "saved" && autoSaveStatus.at
                    ? `Autosaved ${new Date(autoSaveStatus.at).toLocaleTimeString()} — stored in your browser (never sent to the server)`
                  : autoSaveStatus.state === "error" ? "Autosave failed — use the Save button"
                  : "Autosave is on (every 60 s, and 5 s after a change)"
                }
              >
                <span
                  className={`inline-block w-1.5 h-1.5 rounded-full ${
                    autoSaveStatus.state === "saving"  ? "bg-amber-400 animate-pulse"
                    : autoSaveStatus.state === "saved" ? "bg-emerald-500"
                    : autoSaveStatus.state === "error" ? "bg-red-500"
                    : "bg-gray-300"
                  }`}
                />
                <span className="hidden md:inline">
                  {autoSaveStatus.state === "saving"  ? "Kaydediliyor…"
                   : autoSaveStatus.state === "saved" && autoSaveStatus.at
                     ? `Auto-saved · ${new Date(autoSaveStatus.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`
                   : autoSaveStatus.state === "error" ? "Auto-save hata"
                   : "Auto-save"}
                </span>
              </div>
            )}
            <CloudSyncBar />
            <div className="relative" ref={headerSaveMenuRef}>
              <button
                onClick={() => setShowHeaderSaveMenu(v => !v)}
                className={`p-1.5 rounded-lg transition-colors ${showHeaderSaveMenu ? "text-emerald-600 bg-emerald-50" : "text-gray-400 hover:text-emerald-600 hover:bg-emerald-50"}`}
                title="Save / Export dataset"
              >
                <Save size={16} />
              </button>
              {showHeaderSaveMenu && session && (
                <div className="absolute right-0 top-full mt-1 w-48 bg-white border border-gray-200 rounded-xl shadow-xl z-50 overflow-hidden">
                  <p className="px-3 pt-2.5 pb-1 text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
                    Export dataset
                  </p>
                  {([
                    { fmt: "csv" as ExportFmt,  label: "CSV",          desc: "Comma-separated" },
                    { fmt: "xlsx" as ExportFmt, label: "Excel (.xlsx)", desc: "With value labels sheet" },
                    { fmt: "sav" as ExportFmt,  label: "SPSS (.sav)",  desc: "Native value labels" },
                    { fmt: "tsv" as ExportFmt,  label: "TSV",          desc: "Tab-separated" },
                  ]).map(({ fmt, label, desc }) => (
                    <button
                      key={fmt}
                      onClick={() => { setShowHeaderSaveMenu(false); exportDataset(session, session.columns, fmt); }}
                      className="w-full flex items-center gap-2.5 px-3 py-2 text-left hover:bg-gray-50 transition-colors"
                    >
                      <div>
                        <p className="text-xs text-gray-700 font-medium">{label}</p>
                        <p className="text-[10px] text-gray-400">{desc}</p>
                      </div>
                    </button>
                  ))}
                  <div className="border-t border-gray-100" />
                  <p className="px-3 pt-2 pb-1 text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
                    Session
                  </p>
                  <button
                    onClick={() => { setShowHeaderSaveMenu(false); downloadSessionJson(session); }}
                    className="w-full flex items-center gap-2.5 px-3 py-2 text-left hover:bg-gray-50 transition-colors"
                  >
                    <div>
                      <p className="text-xs text-gray-700 font-medium">Session (.json)</p>
                      <p className="text-[10px] text-gray-400">Data + labels + filters + audit</p>
                    </div>
                  </button>
                </div>
              )}
            </div>
            <button
              onClick={handleOpenNew}
              className="p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100 transition-colors"
              title="Open new file"
            >
              <FolderOpen size={16} />
            </button>
            <button
              onClick={handleOpenNew}
              className="p-1.5 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50 transition-colors"
              title="Close dataset"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Case filter banner */}
        {caseFilter && (
          <div className="flex items-center gap-2 px-4 py-1 bg-violet-50 border-t border-violet-200 text-xs text-violet-700">
            <Filter size={12} className="flex-shrink-0" />
            <span className="font-semibold">{caseFilter.selected.toLocaleString()} of {caseFilter.total.toLocaleString()} cases selected</span>
            <span className="text-violet-400">— all analyses use this subset</span>
            <button
              onClick={async () => {
                if (!session) return;
                await clearCases(session.session_id);
                setCaseFilter(null);
              }}
              className="ml-auto flex items-center gap-1 px-2 py-0.5 rounded bg-violet-200 hover:bg-violet-300 text-violet-800 font-medium transition-colors"
            >
              <X size={10} /> Clear filter
            </button>
          </div>
        )}

        {/* Row 2: tab strip — scrollable so tabs are never clipped */}
        <nav className="flex gap-0.5 px-3 pb-1.5 overflow-x-auto"
          style={{ scrollbarWidth: "none" }}>
          {visibleTabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex-shrink-0
                ${activeTab === id
                  ? "bg-ink-500 text-surface"
                  : "text-slate-600 hover:text-slate-900 hover:bg-chip"}`}
            >
              <Icon size={14} />
              {label}
            </button>
          ))}
        </nav>
      </header>

      {/* Content */}
      <main className="flex-1 overflow-hidden flex flex-col">
        <ErrorBoundary key={`${activeTab}:${caseFilterKey}`}>
          {activeTab === "data"        && <div className="flex-1 p-4 overflow-hidden flex flex-col" style={{minHeight:0}}><DataTable /></div>}
          {activeTab === "summary"     && <SummaryCombo />}
          {activeTab === "table1"      && <Table1Panel />}
          {activeTab === "tests"       && <TestsCombo />}
          {activeTab === "correlation" && <div className="flex-1 p-4 overflow-y-auto"><CorrelationPanel /></div>}
          {activeTab === "roc"         && <ROCPanel />}
          {activeTab === "models"      && <ModelsCombo />}
          {activeTab === "visual"      && <VisualChartsCombo />}
          {activeTab === "power"       && <div className="flex-1 p-4 overflow-y-auto"><PowerPanel /></div>}
          {activeTab === "psm"         && <div className="flex-1 p-4 overflow-y-auto"><PSMPanel /></div>}
          {activeTab === "iptw"        && <div className="flex-1 p-4 overflow-y-auto"><IPTWPanel /></div>}
          {activeTab === "causal"      && <div className="flex-1 p-4 overflow-y-auto"><CausalPanel /></div>}
          {activeTab === "meta"        && <div className="flex-1 overflow-y-auto"><MetaPanel /></div>}
          {activeTab === "missing"     && <div className="flex-1 overflow-y-auto"><MissingDataPanel /></div>}
          {activeTab === "dca"         && <div className="flex-1 p-4 overflow-y-auto"><DecisionCurvePanel /></div>}
          {activeTab === "compute"     && <ComputeCombo />}
        </ErrorBoundary>
      </main>
    </div>
  );
}
