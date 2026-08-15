/**
 * Which Pyodide packages an analysis needs.
 *
 * Loading everything would be simpler and wrong: scikit-learn alone adds tens
 * of megabytes and around 45 MB of wasm heap, and a t-test has no use for it.
 * The browser pays this cost once per package per session, so the plan is per
 * analysis family rather than global.
 *
 * The authoritative answer lives on each AnalysisSpec in the engine
 * (`spec.deps`). This table is the copy the main thread can consult BEFORE
 * booting Pyodide, which is when the decision has to be made -- asking the
 * engine would mean loading the engine first.
 */

const BASE = ["numpy"] as const;

const PLANS: Record<string, readonly string[]> = {
  "stats.power": ["numpy", "scipy", "statsmodels"],
  "meta.analyze": ["numpy", "scipy", "statsmodels"],
  "meta.subgroup": ["numpy", "scipy", "statsmodels"],
  "meta.regression": ["numpy", "scipy", "statsmodels"],
  "meta.bias": ["numpy", "scipy", "statsmodels"],
};

/**
 * Analyses the browser is currently allowed to run.
 *
 * Deliberately NOT `Object.keys(PLANS)`. Knowing which packages an analysis
 * needs is not the same as having watched it produce the right numbers in a
 * browser, and only the second earns a place here. The meta-analysis family
 * has fixtures harvested but not yet executed under Pyodide, so it keeps its
 * load plan and stays off this list until they run green.
 */
export const LOCAL_ALLOW_LIST = new Set(["stats.power"]);

export function packagesFor(analysisId: string): string[] {
  return [...(PLANS[analysisId] ?? BASE)];
}

/**
 * The union of packages for everything already loaded plus the new analysis.
 * Pyodide keeps packages once loaded, so the worker is told the cumulative
 * set and skips what it already has.
 */
export function mergePackages(loaded: Iterable<string>, analysisId: string): string[] {
  return [...new Set([...loaded, ...packagesFor(analysisId)])];
}
