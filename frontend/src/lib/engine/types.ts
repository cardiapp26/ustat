/**
 * Shared vocabulary between the main thread and the Pyodide worker.
 */

/** Where a result was actually computed. Travels with every result. */
export type Runtime = "local" | "server";

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
  | "crashed";

export interface LocalRunFailure {
  reason: LocalUnavailableReason;
  detail: string;
}

export type WorkerRequest =
  | { id: number; cmd: "init"; packages: string[]; runtimeBase: string; wheelBase: string }
  | { id: number; cmd: "run"; analysisId: string; params: unknown };

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
