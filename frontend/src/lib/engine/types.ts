/**
 * Shared vocabulary between the main thread and the Pyodide worker.
 */

/** Where a result was actually computed. Travels with every result. */
export type Runtime = "local" | "server";

/**
 * Which in-browser statistics engine. Only one may be resident in a tab at
 * once -- see `arbiter.ts` for why: Pyodide (numpy/scipy/statsmodels) and
 * webR are each tens to well over a hundred MB of wasm heap, and a tab
 * holding both is a killed tab, not a slow one.
 */
export type EngineKind = "python" | "r";

export interface EngineIdentity {
  version: string;
  /** sha256 over the engine package's own sources. */
  fingerprint: string | null;
  modules: number;
  analyses?: string[];
}

/** Why local compute was not used. Recorded so the fallback is never silent. */
export type LocalUnavailableReason =
  | "disabled-by-user"
  | "not-allow-listed"
  | "no-worker-support"
  | "runtime-load-failed"
  | "wheel-missing"
  | "fingerprint-mismatch"
  | "server-unreachable"
  | "engine-error"
  /** A run named a frame the worker is not holding — it was never pushed, or an LRU eviction took it. */
  | "frame-missing"
  | "crashed"
  /** webR itself failed to load -- the R-side counterpart of `runtime-load-failed`. */
  | "r-runtime-load-failed"
  /** The analysis has no R implementation to fall back to. */
  | "no-r-implementation"
  /**
   * The R engine raised a condition it could not attribute to the request
   * (status_hint 500). Distinct from `engine-error` on purpose: a refusal is an
   * answer and is not retried on the server, whereas this is a BUG in the R
   * code, so the question still gets asked server-side -- and the fact that R
   * broke has to be legible rather than filed as "the user asked for something
   * impossible".
   */
  | "r-engine-bug";

export interface LocalRunFailure {
  reason: LocalUnavailableReason;
  detail: string;
}

export type WorkerRequest =
  | { id: number; cmd: "init"; packages: string[]; runtimeBase: string; wheelBase: string }
  /**
   * Hand the worker a `ustat.frame/1` envelope to keep resident under `frameKey`.
   *
   * Separate from `run` because the same dataset serves many analyses: sending
   * it with each one would re-pay the transfer and the rebuild every time a
   * user changes a variable in a panel. The key is the caller's name for one
   * (dataset, filter, column-set) combination — change any of those and it is
   * a different key, never a mutation of an existing one.
   */
  | { id: number; cmd: "frame"; frameKey: string; envelope: unknown }
  | { id: number; cmd: "run"; analysisId: string; params: unknown; frameKey?: string };

export type WorkerResponse =
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; reason: LocalUnavailableReason; detail: string }
  | { id: number; ok: true; identity: EngineIdentity; kind: "init" };

/**
 * `Omit` applied to a union collapses it to the keys every member shares,
 * which would quietly drop `packages` and `analysisId`. Distributing over the
 * union first keeps each variant whole.
 */
export type DistributiveOmit<T, K extends PropertyKey> = T extends unknown
  ? Omit<T, K>
  : never;

export type WorkerRequestBody = DistributiveOmit<WorkerRequest, "id">;
