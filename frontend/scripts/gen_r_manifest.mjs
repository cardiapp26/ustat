#!/usr/bin/env node
/**
 * Freeze the R bundle's analysis declarations into a committed TypeScript file.
 *
 * `frontend/public/webr/bundle/manifest.json` is a build artifact and is
 * gitignored, so nothing that has to work in a fresh checkout -- the allow-list,
 * and the unit tests over it -- may read it. Those need an answer before any
 * fetch has happened anyway: the allow-list decides whether to boot webR at all,
 * and asking the manifest would mean downloading the engine to find out whether
 * to download the engine.
 *
 * So the manifest's *declarations* (ids, needs_frame, r_packages) are copied
 * here at build time and committed. Deliberately NOT copied: `bundle_sha256`
 * and `source_fingerprint`, which change on every edit to the R sources. Those
 * are identity, they are checked at runtime against the real manifest and the
 * server, and freezing them into source would produce a diff on every build and
 * a second place to be wrong.
 *
 * The generated copy is not trusted on its own either -- `r/client.ts`
 * cross-checks it against the manifest it actually fetched, and refuses to
 * compute if the two disagree.
 *
 * Run by `npm run r:bundle`, immediately after the Python script that writes the
 * manifest. Rewrites nothing when the content is unchanged, so a dev server does
 * not see a modified file (and reload) on every start.
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST = resolve(HERE, "../public/webr/bundle/manifest.json");
const OUT = resolve(HERE, "../src/lib/engine/r/manifestAnalyses.generated.ts");

const HEADER = `/**
 * GENERATED FILE -- DO NOT EDIT.
 *
 * Written by frontend/scripts/gen_r_manifest.mjs from
 * frontend/public/webr/bundle/manifest.json (itself built by
 * scripts/build_r_bundle.py). Re-run \`npm run r:bundle\` after changing
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

export const R_MANIFEST_ANALYSES: readonly RManifestAnalysis[] = `;

function main() {
  if (!existsSync(MANIFEST)) {
    console.error(
      `[gen_r_manifest] ${MANIFEST} not found.\n` +
        `Run \`npm run r:bundle\` (which builds it) rather than this script alone.`,
    );
    process.exit(1);
  }

  const manifest = JSON.parse(readFileSync(MANIFEST, "utf8"));
  const declared = Array.isArray(manifest.analyses) ? manifest.analyses : [];
  if (!declared.length) {
    console.error("[gen_r_manifest] the manifest declares no analyses; refusing to write an empty allow-list source");
    process.exit(1);
  }

  const entries = declared
    .map((a) => {
      const packages = Array.isArray(a.r_packages) ? a.r_packages : [];
      return (
        `  {\n` +
        `    id: ${JSON.stringify(a.id)},\n` +
        `    needsFrame: ${a.needs_frame === true},\n` +
        `    rPackages: [${packages.map((p) => JSON.stringify(p)).join(", ")}],\n` +
        `  },`
      );
    })
    .join("\n");

  const body = `${HEADER}[\n${entries}\n] as const;\n`;

  const previous = existsSync(OUT) ? readFileSync(OUT, "utf8") : null;
  if (previous === body) {
    console.log(`[gen_r_manifest] ${OUT} already current (${declared.length} analyses)`);
    return;
  }
  mkdirSync(dirname(OUT), { recursive: true });
  writeFileSync(OUT, body);
  console.log(`[gen_r_manifest] wrote ${OUT} (${declared.length} analyses)`);
}

main();
