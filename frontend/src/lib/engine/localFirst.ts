/**
 * Try the browser first, fall back to the server, and never lose track of which
 * one answered.
 *
 * Shaped to be a drop-in for the axios calls in api.ts: the returned object has
 * `.data`, which is all the panels read, plus a `runtime` field saying where the
 * number came from. That lets an endpoint move to local compute by changing one
 * line in api.ts rather than by touching the panel that calls it -- and panels
 * in this codebase destructure backend fields without optional chaining, so a
 * change in response shape is a runtime crash, not a type error.
 */
import type { Runtime } from "./types";
import {
  LocalComputeUnavailable,
  ensureEngineBooted,
  localRunBlockedBecause,
  runLocal,
} from "./client";

export interface AnalysisResponse<T> {
  data: T;
  status: number;
  /** Where this was computed. Worth surfacing: it is the privacy claim. */
  runtime: Runtime;
  /** Present when it ran on the server despite local being possible. */
  fellBackBecause?: string;
}

/** The last few routing decisions, for a diagnostics view and for tests. */
const decisions: Array<{ analysisId: string; runtime: Runtime; reason?: string }> = [];
const MAX_DECISIONS = 50;

export function recentDecisions(): ReadonlyArray<{
  analysisId: string;
  runtime: Runtime;
  reason?: string;
}> {
  return decisions;
}

function record(analysisId: string, runtime: Runtime, reason?: string): void {
  decisions.push({ analysisId, runtime, reason });
  if (decisions.length > MAX_DECISIONS) decisions.shift();
}

export interface LocalFirstOptions {
  /**
   * The columns this analysis reads. Present means "this one needs a dataset":
   * the frame for exactly these columns is fetched and handed to the worker
   * before the run, and nothing else of the patient's data moves.
   *
   * Deliberately supplied by the caller rather than derived from `params`
   * here. The engine's `AnalysisSpec.required_columns` is the authority, and it
   * lives in Python — reading it would mean booting the engine to find out
   * whether to boot the engine.
   */
  frameColumns?: string[];
  /** Which session the frame belongs to. Read off `params.session_id` when omitted. */
  sessionId?: string;
}

/**
 * Push the frame this run needs, and state which filter it was cut under.
 *
 * The fingerprint travels as `__filter_fingerprint` in the params so the engine
 * can refuse a frame that no longer matches the active Select Cases. That check
 * is worth paying for: a worker keeps a frame between runs, the user's filter
 * does not have to stay still while it does, and the failure without it is a
 * perfectly ordinary-looking result computed over the wrong patients.
 */
async function withFrame(
  params: unknown,
  options: LocalFirstOptions,
): Promise<{ frameKey?: string; params: unknown }> {
  const columns = options.frameColumns;
  if (!columns || !columns.length) return { params };

  const sessionId =
    options.sessionId ??
    (params && typeof params === "object"
      ? (params as { session_id?: string }).session_id
      : undefined);
  if (!sessionId) return { params };

  const { ensureFrame, filterFingerprintFor } = await import("./frame");
  const frameKey = await ensureFrame(sessionId, columns);
  const fingerprint = filterFingerprintFor(frameKey);
  if (!fingerprint || typeof params !== "object" || params === null) {
    return { frameKey, params };
  }
  return { frameKey, params: { ...(params as object), __filter_fingerprint: fingerprint } };
}

export async function localFirst<T>(
  analysisId: string,
  params: unknown,
  server: () => Promise<{ data: T; status: number }>,
  options: LocalFirstOptions = {},
): Promise<AnalysisResponse<T>> {
  try {
    // The frame is fetched only once local compute is otherwise on the table.
    // Doing it first would download a dataset for a user who has the feature
    // switched off, which is a worse trade than one extra round trip.
    const blocked = localRunBlockedBecause(analysisId);
    if (blocked) throw new LocalComputeUnavailable(blocked);
    // Before the frame, not after: the worker rebuilds an envelope into a
    // DataFrame the instant it arrives, so there has to be a worker.
    if (options.frameColumns?.length) await ensureEngineBooted(analysisId);
    const prepared = await withFrame(params, options);
    const data = await runLocal<T>(analysisId, prepared.params, {
      frameKey: prepared.frameKey,
    });
    record(analysisId, "local");
    return { data, status: 200, runtime: "local" };
  } catch (err) {
    if (!(err instanceof LocalComputeUnavailable)) throw err;

    // An analysis that rejected its own input will reject it identically on
    // the server. Asking twice would turn one clear message into a round trip
    // and the same message.
    if (err.reason === "engine-error") throw err;

    const response = await server();
    record(analysisId, "server", err.reason);
    return { ...response, runtime: "server", fellBackBecause: err.reason };
  }
}
