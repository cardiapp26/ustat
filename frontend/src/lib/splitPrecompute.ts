/**
 * Split File, every level at once: when an analysis returns under a split,
 * its recorded request is sent again for each other level (and for the
 * unsplit view) in the background, and each answer is kept beside the panel's
 * result. Switching levels then shows that level's result at once instead of
 * an out-of-date notice, which is what SPSS's SPLIT FILE gives: every group's
 * output from one run.
 *
 * Only requests whose recorded form can be replayed (lib/requestLog) take
 * part, and only read-only analyses: an endpoint with side effects (PSM saves
 * a matched cohort as a session) or a heavy one (ML) is not repeated behind
 * the user's back.
 */
import type { AxiosInstance } from "axios";
import { forSession, type RecordedRequest } from "./requestLog";
import { makeStamp, type ResultStamp, type StampInputs } from "./resultStamp";
import { SPLIT_HEADER, splitHeaderValue } from "./splitHeader";

/** Key a level's result is kept under; the unsplit view has its own. */
export const ALL_LEVELS_KEY = "__all__";
export const levelKeyOf = (level: string | null): string => level ?? ALL_LEVELS_KEY;

const NOT_REPEATED = [
  "/api/models/psm", "/api/models/iptw", "/api/ml/", "/api/code/", "/api/agent/",
  "/api/sessions/", "/api/compute/", "/api/merge/", "/api/project/", "/api/upload/",
  "/api/pub_export/", "/api/engine/",
];

export function repeatable(request: RecordedRequest | null | undefined): request is RecordedRequest {
  if (!request) return false;
  const path = request.url.split("?")[0];
  return path.startsWith("/api/") && !NOT_REPEATED.some((p) => path.startsWith(p));
}

export interface LevelResult<T> {
  result: T;
  stamp: ResultStamp;
}

export type SplitResults<T> = Record<string, LevelResult<T>>;

interface PrecomputeArgs<T> {
  api: AxiosInstance;
  request: RecordedRequest;
  sessionId: string;
  column: string;
  /** Every level, plus the unsplit view, except the one already computed. */
  targets: Array<string | null>;
  /** Stamp inputs of the run being expanded, minus the split part of scope. */
  stampFor: (level: string | null) => Omit<StampInputs, "params"> & { params: unknown };
  /** Stop when a newer run superseded this one. */
  cancelled: () => boolean;
  onLevel: (key: string, value: LevelResult<T>) => void;
}

export async function precomputeLevels<T>(args: PrecomputeArgs<T>): Promise<void> {
  // The recorded request names the session as "{sid}" in its URL and body.
  const call = forSession(args.request, args.sessionId);
  for (const level of args.targets) {
    if (args.cancelled()) return;
    try {
      const res = await args.api.request<T>({
        method: call.method,
        url: call.url,
        data: call.body ?? undefined,
        // An empty value asks for the unsplit view: the interceptor leaves a
        // header that is already set alone, and the server ignores an empty one.
        headers: { [SPLIT_HEADER]: level == null ? "" : splitHeaderValue(args.column, level) },
      });
      if (args.cancelled()) return;
      args.onLevel(levelKeyOf(level), {
        result: res.data,
        stamp: makeStamp({ ...args.stampFor(level), request: args.request }),
      });
    } catch {
      // A level the analysis cannot run on (too few cases) is simply not
      // kept: switching to it shows the out-of-date notice, and recomputing
      // there shows the analysis's own error.
    }
  }
}
