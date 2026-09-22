/**
 * Where a cached result lives in the interface: a readable name, the header
 * tab, and the sub-tab of the combo that hosts it.
 *
 * A saved analysis used to record whichever tab was open when it was kept,
 * so keeping a Kaplan-Meier fit from the Tests tab restored it to Tests, and
 * the menu listed raw cache keys ("survival_km"). Results are keyed per
 * panel (see useStampedResult), so the key alone says where they belong.
 *
 * Rules match a key exactly or as a prefix followed by "_" ("survival" covers
 * "survival_km"); the first match wins, so the more specific rules come first.
 */

export interface PanelLocation {
  label: string;
  tab: string;
  /** The combo's panelCache key and the sub-tab value to select. */
  combo?: { key: string; sub: string };
}

interface Rule {
  prefix: string;
  label: string;
  tab: string;
  combo?: { key: string; sub: string };
}

const tests = (sub: string) => ({ key: "combo_tests", sub });
const models = (sub: string) => ({ key: "combo_models", sub });
const visual = (sub: string) => ({ key: "combo_visual", sub });
const summary = (sub: string) => ({ key: "combo_summary", sub });

const RULES: Rule[] = [
  { prefix: "hypothesis", label: "Hypothesis test", tab: "tests", combo: tests("hypothesis") },
  { prefix: "normality", label: "Normality", tab: "tests", combo: tests("normality") },
  { prefix: "repeated_measures", label: "Repeated measures", tab: "tests", combo: tests("repeated") },
  { prefix: "categorical_tests", label: "Categorical test", tab: "tests", combo: tests("categorical") },
  { prefix: "reliability", label: "Reliability", tab: "tests", combo: tests("reliability") },
  { prefix: "noninferiority", label: "Non-inferiority", tab: "tests", combo: tests("noninferiority") },
  { prefix: "gatekeeping", label: "Gatekeeping", tab: "tests", combo: tests("gatekeeping") },
  { prefix: "factor_pca", label: "Factor analysis / PCA", tab: "tests", combo: tests("factor") },
  { prefix: "bayesian", label: "Bayesian test", tab: "tests", combo: tests("bayesian") },
  { prefix: "descriptive", label: "Descriptive", tab: "summary", combo: summary("descriptive") },
  { prefix: "weighted_stats", label: "Weighted statistics", tab: "summary", combo: summary("weighted") },
  { prefix: "table1", label: "Table 1", tab: "table1" },
  { prefix: "correlation", label: "Correlation", tab: "correlation" },
  { prefix: "roc", label: "ROC", tab: "roc" },
  { prefix: "models", label: "Regression model", tab: "models", combo: models("regression") },
  { prefix: "interval_censored", label: "Interval-censored survival", tab: "models", combo: models("survival") },
  { prefix: "survival", label: "Survival", tab: "models", combo: models("survival") },
  { prefix: "rcs", label: "Restricted cubic spline", tab: "models", combo: models("rcs") },
  { prefix: "threshold", label: "Threshold", tab: "models", combo: models("threshold") },
  { prefix: "multimodel", label: "Progressive adjustment", tab: "models", combo: models("multimodel") },
  { prefix: "subgroup_bar", label: "Subgroup bar chart", tab: "visual", combo: visual("subgroup") },
  { prefix: "subgroup", label: "Subgroup analysis", tab: "models", combo: models("subgroup") },
  { prefix: "ml", label: "Machine learning", tab: "models", combo: models("ml") },
  { prefix: "timeseries", label: "Time series", tab: "models", combo: models("timeseries") },
  { prefix: "internal_validation", label: "Model validation", tab: "models", combo: models("validation") },
  { prefix: "validation", label: "Model validation", tab: "models", combo: models("validation") },
  { prefix: "visual_model", label: "Visual model", tab: "visual", combo: visual("models") },
  { prefix: "charts", label: "Chart", tab: "visual", combo: visual("charts") },
  { prefix: "score_composite", label: "Score composite", tab: "visual", combo: visual("score") },
  { prefix: "km_composite", label: "KM composite", tab: "visual", combo: visual("kmcomposite") },
  { prefix: "added_value", label: "Added predictive value", tab: "visual", combo: visual("addedvalue") },
  { prefix: "power", label: "Power analysis", tab: "power" },
  { prefix: "psm", label: "Propensity score matching", tab: "psm" },
  { prefix: "iptw", label: "IPTW", tab: "iptw" },
  { prefix: "causal", label: "Causal analysis", tab: "causal" },
  { prefix: "meta", label: "Meta-analysis", tab: "meta" },
  { prefix: "missing", label: "Missing data", tab: "missing" },
  { prefix: "dca", label: "Decision curve", tab: "dca" },
];

function humanize(rest: string): string {
  return rest.replace(/_/g, " ").trim();
}

/** Where `key`'s result lives, or null for a key no rule covers. */
export function locatePanel(key: string): PanelLocation | null {
  for (const r of RULES) {
    if (key === r.prefix || key.startsWith(`${r.prefix}_`)) {
      const rest = key.slice(r.prefix.length + 1);
      return { label: rest ? `${r.label}: ${humanize(rest)}` : r.label, tab: r.tab, combo: r.combo };
    }
  }
  return null;
}

/** The name to show for a cached result's key. */
export function panelLabel(key: string): string {
  return locatePanel(key)?.label ?? key;
}
