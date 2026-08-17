/**
 * Which analyses the R engine is allowed to answer, and what it must install
 * first.
 *
 * The R counterpart of `../loadPlan.ts`, and it splits the same way: the
 * PACKAGES an analysis needs are a fact about the engine, while the ALLOW-LIST
 * is a claim about what has been watched producing the right numbers in a real
 * browser. The first is generated; the second is written by hand and shrinks to
 * nothing on its own if someone deletes an analysis.
 */
import { R_MANIFEST_ANALYSES, type RManifestAnalysis } from "./manifestAnalyses.generated";

export { R_MANIFEST_ANALYSES };
export type { RManifestAnalysis };

/**
 * Packages the BUNDLE itself needs, regardless of which analysis runs.
 *
 * jsonlite is not an analysis dependency and does not appear in any spec's
 * `packages`: it is how `ustat_to_json`/`ustat_from_json` exist at all, so the
 * frame envelope cannot be parsed and no result can be serialised without it.
 * webR 0.6.0 does not bundle it (see `webr/runtime/vendored.json`
 * `bundled_r_library`), so it is fetched from the mirror like the rest.
 */
export const R_BUNDLE_PACKAGES: readonly string[] = ["jsonlite"];

/**
 * Analyses the browser is currently allowed to run under R.
 *
 * Deliberately NOT every id in the manifest. Knowing an analysis is registered
 * is not the same as having watched it produce the right numbers under webR,
 * and only the second earns a place here. `stats.ttest` is here because R1b
 * ran it in a browser against the server and against `qa/tests_audit/
 * reference.json` (R 4.5.2, native) and compared every field; nothing else has,
 * so nothing else is listed.
 *
 * Intersected with the generated manifest copy rather than written out flat, so
 * that an id removed from the R sources leaves the allow-list automatically
 * instead of lingering as a name that boots webR and then 404s inside it.
 */
const PROVEN_IN_BROWSER: readonly string[] = ["stats.ttest"];

const DECLARED_IDS = new Set(R_MANIFEST_ANALYSES.map((a) => a.id));

export const R_ALLOW_LIST: ReadonlySet<string> = new Set(
  PROVEN_IN_BROWSER.filter((id) => DECLARED_IDS.has(id)),
);

/** The manifest declaration for `analysisId`, or undefined if it declares none. */
export function rAnalysis(analysisId: string): RManifestAnalysis | undefined {
  return R_MANIFEST_ANALYSES.find((a) => a.id === analysisId);
}

/** Whether `analysisId` reads a dataset. Unknown ids are treated as needing one. */
export function rNeedsFrame(analysisId: string): boolean {
  return rAnalysis(analysisId)?.needsFrame ?? true;
}

/**
 * The union of packages for everything already installed plus the new analysis.
 *
 * webR keeps a package once installed, so the worker is handed the cumulative
 * set and skips what it already has -- the same contract as
 * `mergePackages` on the Pyodide side.
 */
export function mergeRPackages(installed: Iterable<string>, packages: Iterable<string>): string[] {
  return [...new Set([...installed, ...R_BUNDLE_PACKAGES, ...packages])];
}

/**
 * Why this build's copy of the manifest disagrees with the one just fetched, or
 * null when they agree on everything the allow-list rests on.
 *
 * The generated copy exists so the allow-list can be decided before a fetch;
 * this is the check that the decision was made on current information. A
 * disagreement means the served bundle is not the bundle this app was built
 * against -- a half-finished deploy, a stale service-worker entry -- and the
 * honest response is to refuse rather than to run whichever copy happens to be
 * newer.
 */
export function rManifestDisagreement(
  fetched: ReadonlyArray<{ id: string; needs_frame?: boolean; r_packages?: readonly string[] }>,
): string | null {
  const byId = new Map(fetched.map((a) => [a.id, a]));
  for (const id of R_ALLOW_LIST) {
    const live = byId.get(id);
    if (!live) return `${id} is allow-listed here but the fetched manifest does not declare it`;
    const expected = rAnalysis(id);
    if (!expected) return `${id} is allow-listed but this build has no declaration for it`;
    if (live.needs_frame !== expected.needsFrame) {
      return `${id}: needs_frame is ${String(live.needs_frame)} in the fetched manifest, ` +
        `${String(expected.needsFrame)} in this build`;
    }
    const livePackages = [...(live.r_packages ?? [])].sort().join(",");
    const builtPackages = [...expected.rPackages].sort().join(",");
    if (livePackages !== builtPackages) {
      return `${id}: r_packages are [${livePackages}] in the fetched manifest, [${builtPackages}] in this build`;
    }
  }
  return null;
}
