/**
 * useAutoSession — autosaves the live session to IndexedDB.
 *
 * Strategy:
 *   1. **Change-triggered, debounced 5 s.** Whenever session_id, the
 *      active tab, the case filter, or any tracked dirty marker
 *      changes, schedule a snapshot 5 s later. Subsequent changes
 *      restart the timer so we don't fetch on every keystroke.
 *   2. **Periodic 60 s.** A fallback interval saves once a minute even
 *      if nothing tracked has changed (analyses ran, filters applied
 *      server-side, etc.).
 *   3. **`beforeunload` flush.** When the user closes the tab, fire one
 *      last best-effort save. The browser may kill the page mid-fetch
 *      so this is a `keepalive: true` request — small risk of data
 *      loss vs. zero, by design.
 *
 *   The snapshot itself is the same blob the manual "Save Session JSON"
 *   button has always produced. Resume = re-upload it via
 *   POST /api/sessions/load_session.
 */

import { useEffect, useRef } from "react";
import api from "../api";
import { useStore } from "../store";
import { upsertRecentSession, notifySessionsChanged } from "../lib/sessionDb";
import { cloudSync } from "../lib/cloudSync";

const DEBOUNCE_MS = 5_000;
const PERIODIC_MS = 60_000;

/** Cheap, full-payload fingerprint (djb2). Catches in-place edits that keep the
 *  JSON length constant, which a length+prefix hash missed. */
function djb2(s: string): string {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return `${s.length}:${h >>> 0}`;
}

interface AutoSaveDeps {
  /** Optional setter for a status indicator in the header. */
  onStatus?: (status: "idle" | "saving" | "saved" | "error", at?: number) => void;
}

export function useAutoSession({ onStatus }: AutoSaveDeps = {}): void {
  // Pull only the bits that should *retrigger* the debounce. Anything
  // that mutates per-render (e.g. functions) would defeat the timer.
  const sessionId = useStore((s) => s.session?.session_id ?? null);
  // The saved row this session was resumed from, when there is one. Neither
  // the server id (fresh on every restore) nor the filename (a duplicate's
  // blob still names the original) identifies the row to write back to.
  const localId   = useStore((s) => s.localSessionId);
  const filename  = useStore((s) => s.session?.filename ?? null);
  const nRows     = useStore((s) => s.session?.rows ?? null);
  const nCols     = useStore((s) => s.session?.columns.length ?? null);
  const activeTab = useStore((s) => s.activeTab);
  const caseFilter = useStore((s) => s.caseFilter);
  // Bumped on EVERY data mutation (incl. in-place edits that leave row/column
  // counts unchanged: cell edits, recode-in-place, find-replace, date parse).
  // Without this, such edits only got picked up by the 60 s periodic snapshot,
  // so a quick reload after editing could resume from a stale blob.
  const dataVersion = useStore((s) => s.dataVersion);
  // Value-labels / decimals changes don't alter row or column counts, so they
  // wouldn't retrigger the debounce on their own — a quick reload before the
  // 60 s periodic snapshot would resume from a stale blob and lose the labels.
  // Fold a compact signature of per-column value_labels into the deps so any
  // label edit schedules a fresh snapshot within the debounce window.
  const valueLabelSig = useStore((s) =>
    (s.session?.columns ?? [])
      .map((c) => (c.value_labels ? `${c.name}:${Object.entries(c.value_labels).map(([k, v]) => `${k}=${v}`).join(",")}` : ""))
      .filter(Boolean)
      .join(";")
  );

  const lastSavedHashRef = useRef<string | null>(null);
  const inFlightRef = useRef<boolean>(false);

  // Capture a stable status setter so the snapshot function can be
  // memoised without re-running on every render.
  const onStatusRef = useRef(onStatus);
  onStatusRef.current = onStatus;

  useEffect(() => {
    if (!sessionId) return;

    let cancelled = false;

    const snapshot = async (source: "auto" | "manual" = "auto"): Promise<void> => {
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      try {
        onStatusRef.current?.("saving");
        // Server already knows how to serialise; we just need the JSON.
        const res = await api.get(`/api/sessions/${sessionId}/save_session`);
        if (cancelled) return;
        const payload = typeof res.data === "string"
          ? res.data
          : JSON.stringify(res.data);
        // Skip the write if the blob hasn't changed since the last
        // snapshot — cuts down on IndexedDB churn for read-only sessions.
        // djb2 over the whole payload (not just length + prefix, which was
        // blind to in-place edits that keep the JSON length constant).
        const hash = djb2(payload);
        if (hash === lastSavedHashRef.current) {
          onStatusRef.current?.("saved", Date.now());
          return;
        }
        lastSavedHashRef.current = hash;
        await upsertRecentSession({
          localId: localId ?? undefined,
          serverSessionId: sessionId,
          name: filename ?? "Untitled",
          payload,
          nRows: nRows ?? undefined,
          nCols: nCols ?? undefined,
          activeTab: activeTab ?? undefined,
          source,
        });
        notifySessionsChanged();
        // Mirror the snapshot to Google Drive too — schedules a debounced
        // push (no-op when cloud sync isn't connected). Lives alongside the
        // IndexedDB autosave so Drive is always a backup of the same blob.
        cloudSync.markDirty();
        onStatusRef.current?.("saved", Date.now());
      } catch {
        // Network blip, server restart, 404 on session_id — try again
        // next tick rather than surfacing an error. The user can still
        // hit the explicit Save button if they want certainty.
        onStatusRef.current?.("error");
      } finally {
        inFlightRef.current = false;
      }
    };

    // 1) Debounced change snapshot.
    const debounceTimer = setTimeout(snapshot, DEBOUNCE_MS);

    // 2) Periodic snapshot — independent of the debounce.
    const interval = setInterval(snapshot, PERIODIC_MS);

    // 3) Best-effort beforeunload flush.
    const onBeforeUnload = () => {
      // Don't await — the page may already be tearing down.
      void snapshot();
    };
    window.addEventListener("beforeunload", onBeforeUnload);

    return () => {
      cancelled = true;
      clearTimeout(debounceTimer);
      clearInterval(interval);
      window.removeEventListener("beforeunload", onBeforeUnload);
    };
  }, [sessionId, localId, filename, nRows, nCols, activeTab, caseFilter, valueLabelSig, dataVersion]);
}
