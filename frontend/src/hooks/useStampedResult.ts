import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { analysisScope, useStore } from "../store";
import {
  makeStamp,
  staleReasons,
  type ResultStamp,
  type StaleOptions,
  type StaleReason,
} from "../lib/resultStamp";
import { requestFor, sentContextFor, type RecordedRequest } from "../lib/requestLog";
import { levelKeyOf, precomputeLevels, repeatable, type SplitResults } from "../lib/splitPrecompute";

interface StampedCache<T> {
  result?: T | null;
  stamp?: ResultStamp | null;
  /** Split File: the result for each level, keyed by lib/splitPrecompute levelKeyOf. */
  splitResults?: SplitResults<T>;
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
  const scope = useStore(analysisScope);
  const engine = useStore((s) => s.engine);
  const sessionId = useStore((s) => s.session?.session_id ?? null);
  const splitLevel = useStore((s) => s.splitFile?.level ?? null);
  // Bumped by every run, so a precompute for an older run stops writing.
  const splitGeneration = useRef(0);

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
        dataVersion: -1, caseFilter: s.caseFilter, scope: analysisScope(s),
        engine: s.engine, params, provenance: null,
        sessionId: s.session?.session_id ?? null,
      });
    }
    return null;
  });

  const current = useMemo(
    () => makeStamp({ dataVersion, caseFilter, scope, engine, params, sessionId }),
    [dataVersion, caseFilter, scope, engine, params, sessionId],
  );

  // Split File: send this run's request again for every other level (and the
  // unsplit view) and keep each answer, so switching levels is instant.
  const startSplitPrecompute = useCallback((request: RecordedRequest, sentLevel: string | null, generation: number) => {
    const s = useStore.getState();
    const split = s.splitFile;
    const sid = s.session?.session_id;
    if (!split || !sid) return;
    const targets = [null, ...split.levels.map((l) => l.level)].filter((l) => l !== sentLevel);
    void import("../api").then(({ default: api }) => precomputeLevels<T>({
      api, request, sessionId: sid, column: split.column, targets,
      // Each level's request is sent now, so it is stamped with the state now.
      stampFor: (level) => {
        const now = useStore.getState();
        return {
          dataVersion: now.dataVersion,
          caseFilter: now.caseFilter,
          scope: analysisScope({ caseWeight: now.caseWeight, splitFile: now.splitFile ? { ...now.splitFile, level } : null }),
          engine: now.engine,
          params,
          sessionId: now.session?.session_id ?? null,
        };
      },
      cancelled: () => splitGeneration.current !== generation,
      onLevel: (key, value) => {
        const existing = (useStore.getState().panelCache[panel] ?? {}) as StampedCache<T>;
        setPanelCache(panel, { ...existing, splitResults: { ...(existing.splitResults ?? {}), [key]: value } });
      },
    }));
  }, [panel, params, setPanelCache]);

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
            scope: sent ? sent.scope : analysisScope(now),
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
    const sentLevel = r == null ? null : sentContextFor(r)?.splitLevel ?? useStore.getState().splitFile?.level ?? null;
    // A new run starts a new set of per-level results (Split File).
    const splitResults: SplitResults<T> = next ? { [levelKeyOf(sentLevel)]: { result: r as T, stamp: next } } : {};
    setPanelCache(panel, { ...base, result: r, stamp: next, splitResults });
    const generation = ++splitGeneration.current;
    if (next && repeatable(next.request)) startSplitPrecompute(next.request, sentLevel, generation);
    // `params` belongs in the deps: the handler that calls this closes over
    // the render it was created in, which is the render whose params were sent
    // to the backend. Pinning them any other way would stamp a result with
    // settings it was not computed under.
  }, [panel, params, setPanelCache, startSplitPrecompute]);

  // Split File: when the level on view changes, show that level's result if
  // the background precompute already has a current one.
  useEffect(() => {
    if (result == null || staleReasons(stamp, current, opts).length === 0) return;
    const entry = cached?.splitResults?.[levelKeyOf(splitLevel)];
    if (!entry || entry.result === result || staleReasons(entry.stamp, current, opts).length > 0) return;
    setLocalResult(entry.result);
    setLocalStamp(entry.stamp);
    const existing = useStore.getState().panelCache[panel];
    const base = existing && typeof existing === "object" ? existing as Record<string, unknown> : {};
    setPanelCache(panel, { ...base, result: entry.result, stamp: entry.stamp });
  }, [current, splitLevel, cached, result, stamp, opts, panel, setPanelCache]);

  const reasons = result == null ? [] : staleReasons(stamp, current, opts);

  return { result, setResult, stamp, stale: reasons.length > 0, staleReasons: reasons };
}
