/**
 * Nothing that looks like a dataset may sit in `frontend/public/`.
 *
 * Vite copies that directory into `dist/` verbatim, so a file dropped there
 * during debugging is published at a guessable URL the moment the app deploys.
 * This is not hypothetical: a 233-row CSV with `AD`, `SOYAD`, `TC NO` and
 * `DOGUM TARIHI` columns was found in `public/` and in the built `dist/`,
 * unreferenced by any code, left over from working on the column-alignment
 * feature. It had been there for weeks. Nothing in the build, the linter or
 * the test suite had any reason to mention it.
 *
 * For an app whose entire premise is that clinical data stays private, the
 * cost of that mistake is not proportional to how easy it is to make. So the
 * suite objects on the app's behalf.
 */
/// <reference types="node" />
// Referenced for this file only. tsconfig.app.json deliberately limits `types`
// to vite/client so browser code cannot reach for node globals by accident;
// this check has to read the filesystem, and says so locally rather than
// opening the door for everything in src/.
import { describe, expect, it } from "vitest";
import { existsSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";

// Under the jsdom environment `import.meta.url` is an http URL served by Vite,
// not a file path, so the directory is resolved from the working directory
// vitest runs in (the `frontend` package root).
const PUBLIC_DIR = resolve(process.cwd(), "public");

// Extensions that carry rows of data rather than app assets. Deliberately not
// including .json: manifests and fixtures legitimately live here, and banning
// the extension outright would be ignored rather than obeyed.
const DATA_EXTENSIONS = [
  ".csv",
  ".tsv",
  ".xls",
  ".xlsx",
  ".xlsm",
  ".sav",
  ".zsav",
  ".por",
  ".dta",
  ".sas7bdat",
  ".xpt",
  ".rdata",
  ".rds",
  ".parquet",
  ".sqlite",
  ".db",
];

function walk(dir: string): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      // The vendored Pyodide runtime is a build artifact of known provenance
      // fetched by scripts/vendor_pyodide.py, not somewhere data gets dropped.
      if (entry === "pyodide") continue;
      // Same for the vendored webR runtime, its CRAN mirror and the generated
      // R engine bundle (scripts/vendor_webr.py, scripts/build_r_bundle.py).
      // Worth naming explicitly: the mirror holds .tgz archives of R packages,
      // and R's own datasets package ships .rda files inside the runtime's
      // virtual filesystem image — provenance is the vendoring script's
      // manifest, not this walk.
      if (entry === "webr") continue;
      found.push(...walk(full));
    } else {
      found.push(full);
    }
  }
  return found;
}

describe("frontend/public", () => {
  it("is where this test thinks it is", () => {
    // Without this, a wrong path would make the check below scan nothing and
    // pass forever -- a guard that cannot fail is worse than no guard, because
    // it is believed.
    expect(existsSync(PUBLIC_DIR), `expected a directory at ${PUBLIC_DIR}`).toBe(true);
    expect(readdirSync(PUBLIC_DIR).length).toBeGreaterThan(0);
  });

  it("contains no dataset files, because everything here is published", () => {
    const offenders = walk(PUBLIC_DIR)
      .filter((path) => DATA_EXTENSIONS.some((ext) => path.toLowerCase().endsWith(ext)))
      .map((path) => relative(PUBLIC_DIR, path));

    expect(
      offenders,
      offenders.length
        ? `These files are served publicly at https://<host>/<path> once deployed:\n` +
            offenders.map((f) => `  public/${f}`).join("\n") +
            `\nMove them out of public/. If a dataset genuinely has to ship, ` +
            `add it to DATA_EXTENSIONS' exceptions with a comment saying why ` +
            `it is safe to publish.`
        : undefined,
    ).toEqual([]);
  });
});
