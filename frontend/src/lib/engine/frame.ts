/**
 * Getting the dataset to the worker, once, and naming what was sent.
 *
 * The worker holds frames by key and the engine refuses one that no longer
 * matches the active Select Cases, so this module has exactly two jobs: mint a
 * key that changes whenever the data behind it changes, and make sure the
 * worker is holding whatever that key names before an analysis asks for it.
 *
 * The key covers four things, and each of them can change the numbers:
 *
 *   - the session id, because a different upload is a different dataset;
 *   - `dataVersion`, the store's counter for every cell edit, recode, paste,
 *     column add and undo -- none of which change the session id, and all of
 *     which change what a mean is;
 *   - the Select Cases conditions, which decide the denominator;
 *   - the column list, because a frame transferred for `sbp ~ arm` does not
 *     contain the column the next panel is about, and reusing it would fail
 *     inside Python with a KeyError instead of fetching what was needed.
 *
 * Miss any of those and the tab answers a question about data the user has
 * already changed, with no error anywhere -- which is the failure this whole
 * feature has to avoid, not merely the one it has to report.
 */
import api from "../../api";
import { useStore } from "../../store";
import type { EngineKind } from "./types";
import { onLocalEngineReset, pushFrame } from "./client";
import { onREngineReset, pushFrameR } from "./r/client";

/**
 * Keys the live workers have been handed, qualified by which engine holds them.
 *
 * Qualified because "the Pyodide worker is holding this frame" says nothing
 * about webR, and an unqualified set would make `ensureFrame` skip the transfer
 * for an engine that never received it -- surfacing as `frame-missing` from a
 * code path the user never asked about. Cleared whenever the worker it names is.
 */
const pushed = new Set<string>();

function pushedKey(engine: EngineKind, frameKey: string): string {
  return `${engine}:${frameKey}`;
}

/**
 * The envelope's `filter.fingerprint` for the frame last pushed under a key.
 *
 * Sent back to the engine as `__filter_fingerprint` on the run, which makes the
 * engine's 409 guard a real check rather than a dormant one: the frontend
 * states which filter it believes the frame was cut under, and Python compares
 * that against what the frame actually carries.
 *
 * Note what that does and does not catch. It catches the worker answering with
 * a frame other than the one named -- an eviction, a key collision, a `run`
 * that raced a `frame`. It does NOT catch the key itself being wrong, because
 * this value came from the same envelope the frame was built from. Catching
 * that would mean hashing the store's conditions here with the same canonical
 * sha256 `ustat_engine.frame.envelope.filter_fingerprint` uses, and a second
 * implementation of a hash is its own way to be wrong -- any serialisation
 * difference turns into a permanent 409 and a feature that silently never runs
 * locally. So the key's correctness rests on the store agreeing with the
 * server's applied filter, which is what the Select Cases panel maintains.
 */
const fingerprints = new Map<string, string>();

/** Stable JSON: object keys sorted at every depth, arrays left in order. */
function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, v]) => v !== undefined)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${canonical(v)}`).join(",")}}`;
}

/**
 * djb2 rather than sha256: this key is a cache label, not a security boundary,
 * and `crypto.subtle.digest` is async, which would make every call site that
 * merely wants to name a frame await one. Two independently seeded passes are
 * concatenated so the result is 64 bits — enough that two frames alive in a
 * two-entry cache colliding is not a thing that happens.
 */
function djb2(input: string, seed: number): string {
  let h = seed;
  for (let i = 0; i < input.length; i += 1) {
    h = (h * 33) ^ input.charCodeAt(i);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

export function frameKeyFor(
  sessionId: string,
  dataVersion: number,
  filterConditions: unknown,
  columns: readonly string[],
): string {
  // Columns are sorted so that `[sbp, arm]` and `[arm, sbp]` name one frame:
  // the transfer is identical and the engine reads by name, so treating them
  // as two would re-fetch the same bytes on every panel that reorders them.
  const payload = canonical({
    session: sessionId,
    dataVersion,
    filter: filterConditions ?? [],
    columns: [...columns].sort(),
  });
  return `${djb2(payload, 5381)}${djb2(payload, 52711)}`;
}

interface FrameEnvelope {
  schema?: string;
  filter?: { fingerprint?: string };
}

/**
 * Ensure the worker holds the frame for `columns`, and return its key.
 *
 * Fetched through the shared axios instance, not `fetch`: that instance has the
 * session-recovery interceptor on it, so a session the backend has forgotten
 * (its datasets live in memory) is restored and retried rather than surfacing
 * as "Session not found" from a code path the user never asked about.
 *
 * A `LocalComputeUnavailable` from `pushFrame` propagates untouched so
 * `localFirst` can fall back to the server, which is the correct outcome: the
 * question still gets answered, just not here.
 */
export async function ensureFrame(
  sessionId: string,
  columns: readonly string[],
  engine: EngineKind = "python",
): Promise<string> {
  const { dataVersion, caseFilter } = useStore.getState();
  const conditions = caseFilter?.conditions ?? [];
  const key = frameKeyFor(sessionId, dataVersion, conditions, columns);
  if (pushed.has(pushedKey(engine, key))) return key;

  const response = await api.get(`/api/sessions/${sessionId}/frame`, {
    params: { columns: [...columns].join(",") },
  });
  const envelope = response.data as FrameEnvelope;

  // The envelope is engine-agnostic on purpose: `ustat.frame/1` is one wire
  // format with two readers, `frame_from_envelope` in Python and
  // `ustat_frame_from_envelope` in R, so the same bytes serve either.
  if (engine === "r") await pushFrameR(key, envelope);
  else await pushFrame(key, envelope);
  pushed.add(pushedKey(engine, key));
  const fingerprint = envelope?.filter?.fingerprint;
  if (typeof fingerprint === "string") fingerprints.set(key, fingerprint);
  return key;
}

/** The filter fingerprint the frame under `key` arrived with, if it is known. */
export function filterFingerprintFor(key: string): string | undefined {
  return fingerprints.get(key);
}

/**
 * Forget which frames are resident. Called from `resetLocalEngine`, because a
 * terminated worker holds nothing: a key remembered across that would make
 * `ensureFrame` skip a fetch for a frame the new worker never received, and the
 * run would come back `frame-missing` for no visible reason.
 */
export function resetPushedFrames(): void {
  pushed.clear();
  fingerprints.clear();
}

onLocalEngineReset(resetPushedFrames);
onREngineReset(resetPushedFrames);
