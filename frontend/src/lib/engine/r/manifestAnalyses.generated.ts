/**
 * GENERATED FILE -- DO NOT EDIT.
 *
 * Written by frontend/scripts/gen_r_manifest.mjs from
 * frontend/public/webr/bundle/manifest.json (itself built by
 * scripts/build_r_bundle.py). Re-run `npm run r:bundle` after changing
 * backend/ustat_engine_r/.
 *
 * This is the build-time COPY of what the R bundle declares, so the allow-list
 * in ./loadPlan.ts can be decided -- and tested -- without a network fetch. The
 * fetched manifest remains the authority: r/client.ts cross-checks this copy
 * against it at boot and refuses to run locally if they disagree.
 */

/** One analysis the R bundle registers. */
export interface RManifestAnalysis {
  readonly id: string;
  /** True when the analysis reads a dataset, i.e. a frame must be pushed first. */
  readonly needsFrame: boolean;
  /** CRAN packages the host installs from the local mirror before running it. */
  readonly rPackages: readonly string[];
}

export const R_MANIFEST_ANALYSES: readonly RManifestAnalysis[] = [
  {
    id: "stats.ttest",
    needsFrame: true,
    rPackages: ["moments", "nortest"],
  },
] as const;
