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
import { LocalComputeUnavailable, runLocal } from "./client";

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

export async function localFirst<T>(
  analysisId: string,
  params: unknown,
  server: () => Promise<{ data: T; status: number }>,
): Promise<AnalysisResponse<T>> {
  try {
    const data = await runLocal<T>(analysisId, params);
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
