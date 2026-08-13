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

      if (status !== 404 || !cfg || cfg._sessionRecovered || !openId || !url.includes(openId)) {
        return Promise.reject(error);
      }
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
}
