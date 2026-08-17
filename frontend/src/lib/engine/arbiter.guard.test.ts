/**
 * The one-resident-runtime rule, enforced by reading the source.
 *
 * In the spirit of `backend/tests/test_engine_isolation.py`: the real
 * invariant ("no two local-compute runtimes are ever resident at once") is
 * enforced at runtime by `arbiter.ts` itself, but the mistake this guards
 * against happens earlier than runtime -- it is adding a third local
 * runtime and forgetting to route its worker construction through the
 * arbiter at all. A runtime test can only catch that if someone remembers
 * to exercise the new code path; this one catches it the moment the file is
 * written.
 *
 * This is a textual check over source text, not an AST, and it says so
 * honestly rather than implying more than it proves: it looks for a runtime
 * allocation (`new Worker(` or `new WebR(`) preceded somewhere earlier in
 * the same file by the literal substring `acquire(`. It cannot see through
 * an indirection -- a helper function elsewhere that wraps
 * `new Worker(...)`, or a `Worker` reference reassigned to a renamed local
 * -- and does not try to. It exists to catch the realistic failure (a new
 * client that never calls the arbiter), not to prove the property in
 * general.
 *
 * It recurses into subdirectories, because that is where the second engine
 * actually landed (`engine/r/`), and a check that stopped at the top level
 * would have passed the whole of R1b without looking at it.
 */
/// <reference types="node" />
// This check needs the filesystem; under the jsdom test environment
// `import.meta.url` resolves to an http URL served by Vite, not a file path,
// so paths are derived from process.cwd() the way
// frontend/src/lib/publicAssets.test.ts does.
import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { extname, join, relative, resolve } from "node:path";

const ENGINE_DIR = resolve(process.cwd(), "src/lib/engine");

/**
 * Runtime allocations that must be preceded by an arbiter claim.
 *
 * `new WebR(` covers a module that reaches for webR directly. It does NOT
 * match `r/worker.ts`'s `new mod.WebR(`, and should not: that construction is
 * already downstream of `r/client.ts`, which acquires before it creates the
 * worker that would run it.
 */
const ALLOCATIONS = ["new Worker(", "new WebR("];

/** Every engine source, as a path relative to ENGINE_DIR, subdirectories included. */
function engineSources(dir: string = ENGINE_DIR): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      found.push(...engineSources(full));
    } else if (extname(entry.name) === ".ts" && !entry.name.endsWith(".test.ts")) {
      found.push(relative(ENGINE_DIR, full));
    }
  }
  return found.sort();
}

describe("frontend/src/lib/engine: worker construction goes through the arbiter", () => {
  it("is where this test thinks it is", () => {
    // A wrong path would make the check below scan nothing and pass
    // forever. That is worse than no guard, because it would be believed.
    const sources = engineSources();
    expect(sources.length).toBeGreaterThan(0);
    expect(sources).toContain("arbiter.ts");
    expect(sources).toContain("client.ts");
    // The R client lives one level down; without the recursion above this
    // whole check would silently stop covering the engine it was written for.
    expect(sources).toContain(join("r", "client.ts"));
  });

  it("never allocates a runtime without acquiring the arbiter first, outside arbiter.ts itself", () => {
    const offenders: string[] = [];

    for (const file of engineSources()) {
      if (file === "arbiter.ts") continue; // the arbiter is what "acquire(" is defined in

      const source = readFileSync(join(ENGINE_DIR, file), "utf8");
      const allocIndex = Math.min(
        ...ALLOCATIONS.map((token) => source.indexOf(token)).filter((i) => i !== -1),
        Number.POSITIVE_INFINITY,
      );
      if (!Number.isFinite(allocIndex)) continue; // this file allocates no runtime at all

      const acquireIndex = source.indexOf("acquire(");
      if (acquireIndex === -1 || acquireIndex > allocIndex) {
        offenders.push(file);
      }
    }

    expect(
      offenders,
      offenders.length
        ? `These files allocate a local-compute runtime without calling arbiter.acquire() first:\n` +
            offenders.map((f) => `  ${f}`).join("\n") +
            `\nCall acquire(<engine kind>) before constructing it -- the way ` +
            `client.ts does in ensureWorker() -- so the arbiter can tear down a ` +
            `different resident runtime before this one allocates its wasm heap.`
        : undefined,
    ).toEqual([]);
  });
});
