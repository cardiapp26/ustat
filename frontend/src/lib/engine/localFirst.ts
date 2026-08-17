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
 *
 * There are two local engines now, and the session says which one is in play.
 * `runtime` keeps its existing meaning ("here" vs "the server"); `engine` is the
 * new, orthogonal question of WHICH engine produced the number, and the two
 * together are what the badge bar reads.
 */
import type { EngineKind, Runtime } from "./types";
import { useStore } from "../../store";
import { PYTHON_ENGINE_DETAIL } from "./engineDetail";
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
  /** Which engine produced it. Orthogonal to `runtime`: the server is Python too. */
  engine: EngineKind;
  /** How to name that engine to a reader, e.g. "R 4.6.0 · webR 0.6.0". */
  engineDetail?: string;
  /** Present when it ran on the server despite local being possible. */
  fellBackBecause?: string;
}

/** The last few routing decisions, for a diagnostics view and for tests. */
const decisions: Array<{
  analysisId: string;
  runtime: Runtime;
  engine: EngineKind;
  reason?: string;
}> = [];
const MAX_DECISIONS = 50;

export function recentDecisions(): ReadonlyArray<{
  analysisId: string;
  runtime: Runtime;
  engine: EngineKind;
  reason?: string;
}> {
  return decisions;
}

function record(
  analysisId: string,
  runtime: Runtime,
  engine: EngineKind,
  engineDetail?: string,
  reason?: string,
): void {
  decisions.push({ analysisId, runtime, engine, reason });
  if (decisions.length > MAX_DECISIONS) decisions.shift();
  // The store slice the badge bar reads. Kept here rather than in the panels so
  // that provenance cannot be reported by some analyses and not others.
  try {
    useStore.getState().noteEngineRun(analysisId, {
      engine,
      engineDetail,
      fellBackBecause: reason,
      at: Date.now(),
    });
  } catch {
    /* the store is not available in the worker-protocol tests; routing still works */
  }
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
 *
 * Engine-agnostic on purpose: both engines read the same `ustat.frame/1`
 * envelope and both enforce the same 409, with a byte-identical message.
 */
async function withFrame(
  params: unknown,
  options: LocalFirstOptions,
  engine: EngineKind,
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
  const frameKey = await ensureFrame(sessionId, columns, engine);
  const fingerprint = filterFingerprintFor(frameKey);
  if (!fingerprint || typeof params !== "object" || params === null) {
    return { frameKey, params };
  }
  return { frameKey, params: { ...(params as object), __filter_fingerprint: fingerprint } };
}

/** The engine this session was opened under. Defaults to Python outside a store. */
function sessionEngine(): EngineKind {
  try {
    return useStore.getState().engine;
  } catch {
    return "python";
  }
}

async function serverAnswer<T>(
  analysisId: string,
  server: () => Promise<{ data: T; status: number }>,
  fellBackBecause: string,
): Promise<AnalysisResponse<T>> {
  const response = await server();
  record(analysisId, "server", "python", PYTHON_ENGINE_DETAIL, fellBackBecause);
  return {
    ...response,
    runtime: "server",
    // The server is the Python engine. Saying "r" here because the session is
    // an R one would be the single most misleading thing this module could do.
    engine: "python",
    engineDetail: PYTHON_ENGINE_DETAIL,
    fellBackBecause,
  };
}

/**
 * R first, then the server. Never local Python.
 *
 * THE LADDER IS DELIBERATELY TWO RUNGS. The obvious third rung -- "R could not,
 * so try Pyodide before giving up" -- is wrong here, and not marginally.
 * Booting Pyodide makes the arbiter tear webR down (one resident runtime per
 * tab, see arbiter.ts), so the next analysis that IS in R_ALLOW_LIST re-downloads
 * ~22 MB of R runtime and rebuilds its frames. A session that alternates between
 * an R-capable and an R-incapable analysis would pay that on every switch, and
 * would do it to save a round trip to a server that answers in milliseconds.
 * So an R session has exactly one local engine, and everything else is the
 * server's.
 */
async function rFirst<T>(
  analysisId: string,
  params: unknown,
  server: () => Promise<{ data: T; status: number }>,
  options: LocalFirstOptions,
): Promise<AnalysisResponse<T>> {
  const { ensureREngineBooted, localRRunBlockedBecause, rEngineDetail, runLocalR } = await import(
    "./r/client"
  );
  try {
    // The frame is fetched only once R is otherwise on the table: downloading a
    // patient dataset for an analysis R was never going to run moves data for
    // no reason.
    const blocked = localRRunBlockedBecause(analysisId);
    if (blocked) throw new LocalComputeUnavailable(blocked);
    // Before the frame, not after: the worker rebuilds an envelope into a
    // data.frame the instant it arrives, so there has to be a worker.
    if (options.frameColumns?.length) await ensureREngineBooted(analysisId);
    const prepared = await withFrame(params, options, "r");
    const data = await runLocalR<T>(analysisId, prepared.params, { frameKey: prepared.frameKey });
    const detail = rEngineDetail();
    record(analysisId, "local", "r", detail);
    return { data, status: 200, runtime: "local", engine: "r", engineDetail: detail };
  } catch (err) {
    if (!(err instanceof LocalComputeUnavailable)) throw err;

    // An analysis that rejected its own input will reject it identically on the
    // server -- the 409 filter guard says the same sentence in both engines.
    // Asking twice would turn one clear message into a round trip and the same
    // message.
    if (err.reason === "engine-error") throw err;

    return serverAnswer(analysisId, server, err.reason);
  }
}

async function pythonFirst<T>(
  analysisId: string,
  params: unknown,
  server: () => Promise<{ data: T; status: number }>,
  options: LocalFirstOptions,
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
    const prepared = await withFrame(params, options, "python");
    const data = await runLocal<T>(analysisId, prepared.params, {
      frameKey: prepared.frameKey,
    });
    record(analysisId, "local", "python", PYTHON_ENGINE_DETAIL);
    return {
      data,
      status: 200,
      runtime: "local",
      engine: "python",
      engineDetail: PYTHON_ENGINE_DETAIL,
    };
  } catch (err) {
    if (!(err instanceof LocalComputeUnavailable)) throw err;

    // An analysis that rejected its own input will reject it identically on
    // the server. Asking twice would turn one clear message into a round trip
    // and the same message.
    if (err.reason === "engine-error") throw err;

    return serverAnswer(analysisId, server, err.reason);
  }
}

export async function localFirst<T>(
  analysisId: string,
  params: unknown,
  server: () => Promise<{ data: T; status: number }>,
  options: LocalFirstOptions = {},
): Promise<AnalysisResponse<T>> {
  if (sessionEngine() === "r") return rFirst(analysisId, params, server, options);
  return pythonFirst(analysisId, params, server, options);
}
