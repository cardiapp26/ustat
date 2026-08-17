/**
 * One resident local-compute runtime per tab, enforced structurally rather
 * than by convention.
 *
 * uSTAT is gaining a second in-browser statistics engine: R via webR,
 * alongside the existing Python one via Pyodide. Both are heavy. Measured:
 * the Pyodide runtime holds roughly 103 MB of wasm heap with
 * numpy+scipy+statsmodels loaded, climbing toward ~150 MB once scikit-learn
 * is pulled in; a webR instance adds another ~55 MB of wasm heap on top of an
 * 18 MB `R.wasm` download. Two of those resident in one tab at once is not a
 * "slow" tab -- it is how a clinical session dies mid-analysis on a modest
 * laptop, and the failure mode is the tab getting killed outright, i.e. the
 * user loses their work, not a graceful error they can read and act on.
 *
 * The engine is chosen once per session, at the welcome screen, so
 * co-residency should never be *needed*. This module exists to make it
 * impossible rather than merely unlikely: whichever client is about to
 * construct its worker/webR instance calls `acquire` first, and if a
 * different engine is already resident, that engine is torn down before the
 * new one is allowed to start up. There is no code path that gets to have
 * both.
 *
 * Registration is kept separate from the engine kind so this module never
 * has to import either client directly. `client.ts` must call `acquire` on
 * its own way in, and this module must be able to tear `client.ts` down on
 * its way out -- importing `client.ts` from here to do that would be a
 * cycle. Instead each client registers its own teardown hook, once, at
 * module scope, and this module calls it by name.
 */
import type { EngineKind } from "./types";

interface RuntimeHooks {
  teardown: () => void;
}

/** Registered per engine kind, once, by each client at module scope. */
const registry = new Map<EngineKind, RuntimeHooks>();

/** The engine currently resident in this tab, or null if none has booted. */
let residentEngine: EngineKind | null = null;

/**
 * Register `engine`'s teardown hook.
 *
 * Called once, at module scope, by each client -- `client.ts` for Python,
 * and the webR client for R when it lands. Re-registering the same kind
 * (as happens if a test re-imports a mocked module) simply replaces the
 * hooks; the arbiter does not care who called last, only who is current.
 */
export function registerRuntime(engine: EngineKind, hooks: RuntimeHooks): void {
  registry.set(engine, hooks);
}

/**
 * Make `engine` the resident runtime in this tab, tearing down whatever else
 * was resident first.
 *
 * A no-op if `engine` is already resident: re-entering an already-running
 * engine (a second analysis, a retried boot) must not tear itself down out
 * from under itself. Tearing down an engine that was never registered is
 * also fine and silent -- nothing was holding memory for it, so there is
 * nothing to release; that covers a client that has not finished loading
 * yet, which is a normal state, not a bug.
 */
export function acquire(engine: EngineKind): void {
  if (residentEngine === engine) return;

  if (residentEngine !== null) {
    const outgoing = residentEngine;
    const hooks = registry.get(outgoing);
    if (hooks) {
      try {
        hooks.teardown();
      } catch (err) {
        // Log-and-continue, deliberately. A broken outgoing engine must not
        // block the incoming one from getting the tab: that would turn "R's
        // teardown threw" into "the user can no longer run Python either,"
        // which is strictly worse than the leaked memory this module exists
        // to prevent. console.error keeps the failure visible in devtools
        // without escalating it into a blocked acquisition.
        console.error(
          `arbiter: teardown for "${outgoing}" threw while switching to "${engine}"`,
          err,
        );
      }
    }
  }

  residentEngine = engine;
}

/** The engine currently resident in this tab, or null if none is. */
export function current(): EngineKind | null {
  return residentEngine;
}

/**
 * Record that `engine` is no longer resident -- but only if it was.
 *
 * A no-op for any other engine. A stale or duplicate teardown call from an
 * engine that already lost the tab (its own `worker.onerror` firing after
 * `acquire` already switched residency away from it, for instance) must not
 * clobber whichever engine is current now.
 */
export function release(engine: EngineKind): void {
  if (residentEngine === engine) residentEngine = null;
}

/**
 * Test seam: forget who is resident.
 *
 * Deliberately does not clear `registry`. Registrations are made once, at
 * module scope, by each real client; wiping them here would unregister that
 * client for the rest of the test file rather than merely resetting state
 * between tests, since nothing re-runs the module-scope `registerRuntime`
 * call afterwards.
 */
export function resetArbiter(): void {
  residentEngine = null;
}
