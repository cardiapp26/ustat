/**
 * Putting a session back after the server has forgotten it.
 *
 * Datasets live in the backend's memory and are deliberately not written to
 * disk — clinical data stays off the filesystem. That means a restart, a
 * redeploy, or an app update leaves every open session id pointing at nothing,
 * and the app answered every click with "Session not found" while the browser
 * was holding a complete copy of the same dataset the whole time.
 *
 * So a 404 on the live session is treated as a recoverable condition rather
 * than an error: the autosaved snapshot is re-uploaded, the store is pointed at
 * the new server id, and the request that failed is retried once.
 *
 * Concurrent 404s share one recovery. A grid refresh can fire several requests
 * at once and each would otherwise upload its own copy, leaving a pile of
 * duplicate sessions and a race over which id the store ends up on.
 */
import type { AxiosInstance } from "axios";
import { getRecentSessionByServerId } from "./sessionDb";
import { useStore } from "../store";

let inFlight: Promise<string | null> | null = null;
/** Ids already tried and failed — never attempt them twice. */
const hopeless = new Set<string>();
/**
 * Dead id → the id it was restored as.
 *
 * A recovery moves the store to the new id, but requests already built (or
 * already in flight) still name the old one: a mutation is recovered and
 * retried, and the refresh that follows it carries the id captured before the
 * swap. Without this map that refresh is a plain 404 — the mutation lands on
 * the server while the grid keeps showing the state from before it, which
 * reads as "it errored and my column vanished".
 */
const redirected = new Map<string, string>();

/** Re-upload the saved snapshot for `deadId`; returns the new server id. */
export async function recoverSession(deadId: string, api: AxiosInstance): Promise<string | null> {
  if (!deadId || hopeless.has(deadId)) return null;
  if (inFlight) return inFlight;

  inFlight = (async () => {
    try {
      const state = useStore.getState();
      // Prefer the row the session was resumed from; a duplicate can share a
      // server id with its original until its first autosave.
      const rec = (state.localSessionId
        ? await import("./sessionDb").then((m) => m.getRecentSession(state.localSessionId as string))
        : undefined) ?? await getRecentSessionByServerId(deadId);
      if (!rec?.payload) { hopeless.add(deadId); return null; }

      const form = new FormData();
      form.append("file", new Blob([rec.payload], { type: "application/json" }),
                  `${rec.name || "session"}.json`);
      const res = await api.post("/api/sessions/load_session", form);
      const restored = res.data as { session_id?: string };
      if (!restored?.session_id) { hopeless.add(deadId); return null; }

      // Keep whatever the user was looking at: setSession resets per-session UI
      // when the id changes, so the row pin has to be reapplied after it.
      const current = useStore.getState();
      if (current.session?.session_id === deadId || !current.session) {
        current.setSession(restored as never);
        useStore.getState().setLocalSessionId(rec.id);
      }
      // The restore is a rollback to the snapshot's moment: work done between
      // that snapshot and the crash is gone, and the request that triggered
      // all this may now fail against a dataset that no longer has the column
      // it names. Saying so is the difference between a recovery and the app
      // appearing to undo the user's work by itself.
      useStore.getState().setSessionRecovery({
        restoredAt: Date.now(),
        snapshotAt: rec.savedAt,
        name: rec.name || "session",
      });
      redirected.set(deadId, restored.session_id);
      return restored.session_id;
    } catch {
      hopeless.add(deadId);
      return null;
    } finally {
      inFlight = null;
    }
  })();

  return inFlight;
}

/**
 * Retry a request once against a restored session.
 *
 * Only the session that is actually open is recovered. A 404 for some other id
 * — a stale panel, a deleted matched-cohort session — is a genuine 404 and is
 * left alone.
 */
export function installSessionRecovery(api: AxiosInstance): void {
  api.interceptors.response.use(
    (r) => r,
    async (error) => {
      const cfg = error?.config as (typeof error.config & { _sessionRecovered?: boolean }) | undefined;
      const status = error?.response?.status;
      const url: string = cfg?.url ?? "";
      const openId = useStore.getState().session?.session_id;

      if (status !== 404 || !cfg || cfg._sessionRecovered) {
        return Promise.reject(error);
      }

      // A request left over from before a recovery: the session it names is
      // already back under a new id, so point it there rather than uploading
      // the snapshot a second time and forking the session in two.
      for (const [deadId, newId] of redirected) {
        if (url.includes(deadId)) {
          cfg._sessionRecovered = true;
          cfg.url = url.split(deadId).join(newId);
          return api.request(cfg);
        }
      }

      // Only the session that is actually open gets restored. A 404 for some
      // other id — a stale panel, a deleted matched-cohort session — is a
      // genuine 404.
      if (!openId || !url.includes(openId)) return Promise.reject(error);

      const newId = await recoverSession(openId, api);
      if (!newId) return Promise.reject(error);

      cfg._sessionRecovered = true;
      cfg.url = url.split(openId).join(newId);
      return api.request(cfg);
    },
  );
}

/** Test seam: forget which ids have already been written off. */
export function resetSessionRecovery(): void {
  inFlight = null;
  hopeless.clear();
  redirected.clear();
}
