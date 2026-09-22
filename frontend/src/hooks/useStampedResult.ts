import { useCallback, useMemo, useState } from "react";
import { useStore } from "../store";
import {
  makeStamp,
  staleReasons,
  type ResultStamp,
  type StaleOptions,
  type StaleReason,
} from "../lib/resultStamp";
import { requestFor, sentContextFor } from "../lib/requestLog";

interface StampedCache<T> {
  result?: T | null;
  stamp?: ResultStamp | null;
  [key: string]: unknown;
}

export interface StampedResult<T> {
  result: T | null;
  setResult: (r: T | null) => void;
  /** The conditions the result on screen was computed under. */
  stamp: ResultStamp | null;
  stale: boolean;
  staleReasons: StaleReason[];
}

/**
 * A panel result that knows what it was computed from.
 *
 * Drop-in for the `useState` + `setPanelCache` pair every result panel wrote by
 * hand. Two things change beyond persistence:
 *
 * 1. **The result is stamped** with the data version, case filter, params and
 *    engine at the moment of the run, so `stale` can answer "is this still an
 *    answer about the data on screen?" See `lib/resultStamp`.
 * 2. **The cache entry is merged, not replaced.** The hand-written versions did
 *    `setPanelCache("models", { result })`, which overwrote the sibling keys
 *    `usePersistedPanelState` keeps there -- so running a model silently reset
 *    that panel's own variable selections on the next remount.
 *
 * `params` must contain everything the run depends on and nothing else: a
 * post-hoc display toggle in there would flag the result stale for a change
 * that cannot alter it.
 */
export function useStampedResult<T>(
  panel: string,
  params: unknown,
  opts: StaleOptions = {},
): StampedResult<T> {
  const cached = useStore((s) => s.panelCache[panel]) as StampedCache<T> | undefined;
  const setPanelCache = useStore((s) => s.setPanelCache);
  const dataVersion = useStore((s) => s.dataVersion);
  const caseFilter = useStore((s) => s.caseFilter);
  const engine = useStore((s) => s.engine);
  const sessionId = useStore((s) => s.session?.session_id ?? null);

  // Read the cache once, on mount, for the same reason usePersistedPanelState
  // does: this hook owns the value afterwards and re-reading would fight it.
  const [result, setLocalResult] = useState<T | null>(() => cached?.result ?? null);
  const [stamp, setLocalStamp] = useState<ResultStamp | null>(() => {
    if (cached?.stamp) return cached.stamp;
    // A cached result with no stamp predates stamping, or came through a
    // cache that dropped it. Nothing says what it was computed from, so it
    // cannot be shown as current: stamp it with a data version no session
    // reaches, which reads as "the data changed" until it is recomputed.
    if (cached?.result != null) {
      const s = useStore.getState();
      return makeStamp({
        dataVersion: -1, caseFilter: s.caseFilter, engine: s.engine, params, provenance: null,
        sessionId: s.session?.session_id ?? null,
      });
    }
    return null;
  });

  const current = useMemo(
    () => makeStamp({ dataVersion, caseFilter, engine, params, sessionId }),
    [dataVersion, caseFilter, engine, params, sessionId],
  );

  const setResult = useCallback((r: T | null) => {
    const next = r == null
      ? null
      : (() => {
          const now = useStore.getState();
          // The state the request was SENT under, when the result is the
          // response itself: a fit sent before an edit and landing after it
          // is a fit of the old data, and must read as out of date.
          const sent = sentContextFor(r);
          return makeStamp({
            dataVersion: sent?.dataVersion ?? now.dataVersion,
            caseFilter: sent ? (sent.caseFilter as typeof now.caseFilter) : now.caseFilter,
            engine: now.engine,
            params,
            sessionId: sent?.sessionId ?? now.session?.session_id ?? null,
            // The request that returned exactly this object, if any: what a
            // saved analysis re-runs and the replay script calls.
            // Substitute the session the request was sent to: a dataset
            // switched while it was in flight must not leave a dead id in a
            // request that re-runs are aimed with.
            request: requestFor(r, sent?.sessionId ?? now.session?.session_id),
          });
        })();
    setLocalResult(r);
    setLocalStamp(next);
    const existing = useStore.getState().panelCache[panel];
    const base = existing && typeof existing === "object" ? existing as Record<string, unknown> : {};
    setPanelCache(panel, { ...base, result: r, stamp: next });
    // `params` belongs in the deps: the handler that calls this closes over
    // the render it was created in, which is the render whose params were sent
    // to the backend. Pinning them any other way would stamp a result with
    // settings it was not computed under.
  }, [panel, params, setPanelCache]);

  const reasons = result == null ? [] : staleReasons(stamp, current, opts);

  return { result, setResult, stamp, stale: reasons.length > 0, staleReasons: reasons };
}
