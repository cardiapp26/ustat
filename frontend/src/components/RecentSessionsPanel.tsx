/**
 * RecentSessionsPanel — lists previously-saved sessions on the upload
 * landing page so the user can pick up where they left off without
 * re-uploading their dataset.
 *
 * Each card carries: dataset name, n × m, last-visited tab, save
 * timestamp, an indicator of auto vs. manual save, and two actions —
 * "Devam et" (re-upload to the backend → setSession + restore tab) and
 * "Sil" (purge the local snapshot).
 *
 * Everything renders from IndexedDB; the backend is contacted only on
 * Devam et and on the explicit Save button elsewhere in the app.
 */

import { useCallback, useEffect, useState } from "react";
import { Clock, Database, RotateCcw, Trash2, Copy, Pencil, Sparkles, FileText, HardDrive, Cloud, CloudDownload, Download } from "lucide-react";
import api from "../api";
import { useStore } from "../store";
import {
  listRecentSessions,
  listTrashedSessions,
  trashSession,
  restoreSession,
  purgeSession,
  emptyTrash,
  getRecentSession,
  subscribeSessions,
  getStorageEstimate,
  clearAllRecentSessions,
  duplicateRecentSession,
  renameRecentSession,
  upsertRecentSession,
  TRASH_TTL_MS,
  type RecentSessionMeta,
} from "../lib/sessionDb";
import { cloudSync } from "../lib/cloudSync";
import { exportSnapshot, SNAPSHOT_FORMATS, type SnapshotFmt } from "../lib/exportSnapshot";

function fmtBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

function fmtAgo(epochMs: number): string {
  const diff = Date.now() - epochMs;
  const sec = Math.floor(diff / 1000);
  if (sec < 60) return "just now";
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} min ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr} h ago`;
  const day = Math.floor(hr / 24);
  if (day < 7) return `${day} d ago`;
  const d = new Date(epochMs);
  return d.toLocaleDateString();
}

const TAB_LABELS: Record<string, string> = {
  data: "Data",
  summary: "Summary",
  table: "Table 1",
  tests: "Tests",
  correlation: "Correlation",
  roc: "ROC",
  models: "Models",
  psm: "PSM",
  iptw: "IPTW",
  dca: "DCA",
  meta: "Meta",
  missing: "Missing",
  visual: "Visual",
  compute: "Compute",
  causal: "Causal",
  code: "Code",
};

export default function RecentSessionsPanel() {
  const setSession = useStore((s) => s.setSession);
  const setLocalSessionId = useStore((s) => s.setLocalSessionId);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const activeSessionId = useStore((s) => s.session?.session_id ?? null);
  const [items, setItems] = useState<RecentSessionMeta[]>([]);
  const [trashedItems, setTrashedItems] = useState<RecentSessionMeta[]>([]);
  const [trashOpen, setTrashOpen] = useState(false);
  const [estimate, setEstimate] = useState<{ count: number; bytes: number; capCount: number; capBytes: number } | null>(null);
  const [restoring, setRestoring] = useState<string | null>(null);
  const [duplicating, setDuplicating] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [saveAsOpen, setSaveAsOpen] = useState<string | null>(null);
  const [exportingId, setExportingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  // Cloud sync state — refreshes when the sync motor emits (sign-in/out,
  // manual sync). Drives the "☁ Drive" badge + the "Import from Drive" link.
  const [cloudOn, setCloudOn] = useState(cloudSync.isSignedIn());
  const [cloudBusy, setCloudBusy] = useState(false);
  useEffect(() => {
    const unsub = cloudSync.subscribe((s) => setCloudOn(s.signedIn));
    return unsub;
  }, []);

  const reload = useCallback(async () => {
    try {
      const [list, trash, est] = await Promise.all([
        listRecentSessions(),
        listTrashedSessions(),
        getStorageEstimate(),
      ]);
      setItems(list);
      setTrashedItems(trash);
      setEstimate(est);
    } catch {
      // IndexedDB unavailable (Safari private mode etc.) — silently
      // degrade; the upload zone still works.
      setItems([]);
      setTrashedItems([]);
      setEstimate(null);
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    void reload();
    const unsub = subscribeSessions(() => { void reload(); });
    return unsub;
  }, [reload]);

  // Hide the panel entirely when there's nothing to show AND cloud sync isn't
  // connected. When connected, keep it visible even with zero local records so
  // the "Import from Drive" entry point stays reachable (a brand-new device
  // has no IndexedDB snapshots yet, but Drive may have them).
  if (loaded && items.length === 0 && !cloudOn) return null;

  const onRestore = async (id: string) => {
    setRestoring(id);
    setError(null);
    try {
      const rec = await getRecentSession(id);
      if (!rec) throw new Error("Snapshot not found");
      // POST /api/sessions/load_session expects multipart File; wrap
      // the stored payload as a Blob so the existing endpoint accepts it.
      const blob = new Blob([rec.payload], { type: "application/json" });
      const form = new FormData();
      form.append("file", blob, `${rec.name || "session"}.json`);
      const res = await api.post("/api/sessions/load_session", form);
      setSession(res.data);
      // Pin the row this came from. setSession has just cleared it, so this
      // has to follow. Without it autosave has only the fresh server id and
      // the filename inside the blob to go on — and for a duplicate that
      // filename is still the original's, so the edits landed on the
      // original and the copy never moved past the state it was copied in.
      setLocalSessionId(rec.id);
      // Restore the user's last tab, falling back to Data.
      if (rec.activeTab) setActiveTab(rec.activeTab);
      // Re-hydrate column-decimal overrides the same way UploadZone
      // does on a fresh load — keeps the data table formatting stable.
      try {
        const dres = await api.get(`/api/sessions/${res.data.session_id}/decimals`);
        if (dres.data && Object.keys(dres.data).length > 0) {
          const { useStore: store } = await import("../store");
          store.setState({ columnDecimals: dres.data });
        }
      } catch { /* non-fatal */ }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load the snapshot");
    } finally {
      setRestoring(null);
    }
  };

  // Save as… builds the file from the stored snapshot rather than going
  // through the backend's export endpoint, which would need this session
  // uploaded there first. See lib/exportSnapshot.
  const onSaveAs = async (id: string, fmt: SnapshotFmt) => {
    setSaveAsOpen(null);
    setExportingId(id);
    setError(null);
    try {
      const rec = await getRecentSession(id);
      if (!rec) throw new Error("Snapshot not found");
      await exportSnapshot({ name: rec.name, payload: rec.payload }, fmt);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save the snapshot");
    } finally {
      setExportingId(null);
    }
  };

  const onDuplicate = async (id: string) => {
    setDuplicating(id);
    setError(null);
    try {
      // A card holds the last autosaved snapshot, which for the session the
      // user currently has open can trail their edits by up to the autosave
      // interval — long enough that rows they just deleted reappear in the
      // copy. Refresh that one row from the server first so the copy is of
      // what they are actually looking at. Cards for other sessions have no
      // live state to refresh; their stored snapshot is all there is.
      const row = items.find((r) => r.id === id);
      if (row?.serverSessionId && row.serverSessionId === activeSessionId) {
        const res = await api.get(`/api/sessions/${row.serverSessionId}/save_session`);
        const payload = typeof res.data === "string" ? res.data : JSON.stringify(res.data);
        await upsertRecentSession({
          serverSessionId: row.serverSessionId,
          name: row.name,
          payload,
          nRows: row.nRows,
          nCols: row.nCols,
          activeTab: row.activeTab,
          source: "manual",
        });
      }
      await duplicateRecentSession(id);
      await reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not duplicate the session");
    } finally {
      setDuplicating(null);
    }
  };

  const startRename = (id: string, current: string) => {
    setRenamingId(id);
    setRenameDraft(current);
    setError(null);
  };

  const commitRename = async () => {
    if (!renamingId) return;
    const id = renamingId;
    const next = renameDraft.trim();
    const current = items.find((r) => r.id === id)?.name;
    setRenamingId(null);
    if (!next || next === current) return;
    try {
      await renameRecentSession(id, next);
      await reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Rename failed");
    }
  };

  const onDelete = async (id: string) => {
    // Soft-delete: move to Trash. Restoreable for 30 days, then permanently
    // purged by purgeExpiredTrash() (local + Drive).
    await trashSession(id);
    void reload();
  };

  const onRestoreFromTrash = async (id: string) => {
    setRestoring(id);
    try {
      await restoreSession(id);
      await reload();
    } finally {
      setRestoring(null);
    }
  };

  const onPurgeOne = async (id: string) => {
    if (!window.confirm("Delete this session permanently? This cannot be undone.")) return;
    await purgeSession(id);
    void reload();
  };

  const onEmptyTrash = async () => {
    if (trashedItems.length === 0) return;
    if (!window.confirm(`Permanently delete ${trashedItems.length} session(s) from the trash? This cannot be undone.`)) return;
    await emptyTrash();
    void reload();
  };

  const onClearAll = async () => {
    if (!window.confirm("Delete every locally saved session? This cannot be undone.")) return;
    await clearAllRecentSessions();
    void reload();
  };

  // Pull remote session snapshots from Drive into IndexedDB, then refresh
  // the card list. Only offered when cloud sync is connected.
  const onImportFromDrive = async () => {
    setCloudBusy(true);
    setError(null);
    try {
      await cloudSync.syncNow(true);
      await reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Import from Drive failed");
    } finally {
      setCloudBusy(false);
    }
  };

  if (!loaded) return null;

  return (
    <div className="w-full max-w-2xl mt-2">
      <div className="flex items-center justify-between mb-2 px-1">
        <div className="flex items-center gap-1.5">
          <Clock size={14} className="text-indigo-500" />
          <h3 className="text-xs font-semibold text-gray-700">Recent work</h3>
          {cloudOn && (
            <span
              className="inline-flex items-center gap-0.5 text-[9px] font-bold uppercase tracking-wide bg-sky-50 text-sky-600 px-1.5 py-0.5 rounded border border-sky-200"
              title="Google Drive sync is connected — sessions follow you across devices"
            >
              <Cloud size={9} /> Drive
            </span>
          )}
          <span className="text-[10px] text-gray-400 font-normal">
            (saved automatically in your browser — never sent to the server)
          </span>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-gray-400">
          {cloudOn && (
            <button
              onClick={onImportFromDrive}
              disabled={cloudBusy}
              className="inline-flex items-center gap-1 text-sky-600 hover:text-sky-700 hover:underline underline-offset-2 disabled:opacity-50"
              title="Pull session snapshots from Google Drive"
            >
              <CloudDownload size={11} />
              {cloudBusy ? "Importing…" : "Import from Drive"}
            </button>
          )}
          {estimate && (
            <>
              <HardDrive size={11} />
              <span>{estimate.count}/{estimate.capCount} · {fmtBytes(estimate.bytes)}</span>
              <button
                onClick={onClearAll}
                className="text-gray-400 hover:text-red-500 underline-offset-2 hover:underline"
              >
                Clear all
              </button>
            </>
          )}
        </div>
      </div>

      {error && (
        <p className="text-[11px] text-red-500 mb-2 px-1">{error}</p>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {items.map((it) => (
          <div
            key={it.id}
            className="bg-white border border-gray-200 rounded-xl p-3 shadow-sm hover:shadow-md hover:border-indigo-300 transition-all group"
          >
            <div className="flex items-start justify-between mb-2 gap-2">
              <div className="flex items-center gap-1.5 min-w-0 flex-1">
                <FileText size={13} className="text-indigo-500 flex-shrink-0" />
                {renamingId === it.id ? (
                  <input
                    autoFocus
                    className="text-xs font-semibold text-gray-800 border border-indigo-400 rounded px-1 py-0.5 w-full focus:outline-none"
                    value={renameDraft}
                    onChange={(e) => setRenameDraft(e.target.value)}
                    onBlur={commitRename}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") void commitRename();
                      if (e.key === "Escape") setRenamingId(null);
                    }}
                  />
                ) : (
                  <span
                    className="text-xs font-semibold text-gray-800 truncate cursor-text"
                    title={`${it.name}\nDouble-click to rename`}
                    onDoubleClick={() => startRename(it.id, it.name)}
                  >
                    {it.name}
                  </span>
                )}
              </div>
              {it.source === "auto" && (
                <span
                  className="text-[8px] uppercase tracking-wide bg-indigo-50 text-indigo-600 px-1.5 py-0.5 rounded font-bold flex-shrink-0"
                  title="Autosaved"
                >
                  <Sparkles size={9} className="inline mr-0.5" />Auto
                </span>
              )}
            </div>

            <div className="flex items-center gap-2 text-[10px] text-gray-500 mb-2">
              {(it.nRows != null && it.nCols != null) && (
                <span className="flex items-center gap-0.5">
                  <Database size={10} />
                  {it.nRows.toLocaleString()} × {it.nCols}
                </span>
              )}
              <span className="text-gray-300">·</span>
              <span>{fmtBytes(it.sizeBytes)}</span>
              <span className="text-gray-300">·</span>
              <span title={new Date(it.savedAt).toLocaleString()}>{fmtAgo(it.savedAt)}</span>
            </div>

            {it.activeTab && (
              <p className="text-[10px] text-gray-400 mb-2.5">
                Last seen:{" "}
                <span className="font-semibold text-gray-600">
                  {TAB_LABELS[it.activeTab] ?? it.activeTab}
                </span>
              </p>
            )}

            <div className="flex items-center gap-1.5">
              <button
                onClick={() => onRestore(it.id)}
                disabled={restoring === it.id}
                className="flex-1 flex items-center justify-center gap-1 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-[11px] font-semibold px-2.5 py-1.5 rounded-lg transition-colors"
              >
                <RotateCcw size={11} />
                {restoring === it.id ? "Loading…" : "Resume"}
              </button>
              <div className="relative">
                <button
                  onClick={() => setSaveAsOpen((cur) => (cur === it.id ? null : it.id))}
                  disabled={exportingId === it.id}
                  aria-haspopup="menu"
                  aria-expanded={saveAsOpen === it.id}
                  className="text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 disabled:opacity-50 px-2 py-1.5 rounded-lg transition-colors"
                  title="Save as — download this dataset without opening it"
                >
                  <Download size={12} />
                </button>
                {saveAsOpen === it.id && (
                  <>
                    {/* Click-away catcher; the menu is small enough that a
                        full popover library would cost more than it saves. */}
                    <div
                      className="fixed inset-0 z-10"
                      onClick={() => setSaveAsOpen(null)}
                    />
                    <div
                      role="menu"
                      className="absolute right-0 bottom-full mb-1 z-20 w-36 bg-white border border-gray-200 rounded-lg shadow-lg py-1"
                    >
                      {SNAPSHOT_FORMATS.map((f) => (
                        <button
                          key={f.fmt}
                          role="menuitem"
                          onClick={() => void onSaveAs(it.id, f.fmt)}
                          className="w-full text-left text-[11px] text-gray-700 hover:bg-indigo-50 hover:text-indigo-700 px-2.5 py-1.5 transition-colors"
                        >
                          {f.label}
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>
              <button
                onClick={() => startRename(it.id, it.name)}
                className="text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 px-2 py-1.5 rounded-lg transition-colors"
                title="Rename"
              >
                <Pencil size={12} />
              </button>
              <button
                onClick={() => onDuplicate(it.id)}
                disabled={duplicating === it.id}
                className="text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 disabled:opacity-50 px-2 py-1.5 rounded-lg transition-colors"
                title="Duplicate — makes an independent copy you can edit without touching this one"
              >
                <Copy size={12} />
              </button>
              <button
                onClick={() => onDelete(it.id)}
                className="text-gray-400 hover:text-red-600 hover:bg-red-50 px-2 py-1.5 rounded-lg transition-colors"
                title="Move to trash (permanently deleted after 30 days)"
              >
                <Trash2 size={12} />
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Trash bin — collapsed by default; only renders when there are
          trashed records. Each item shows when it was deleted and a
          countdown to permanent purge. Restore / permanently-delete /
          empty-trash actions mirror a typical recycle-bin UX. */}
      {trashedItems.length > 0 && (
        <div className="mt-3 border border-gray-200 rounded-xl overflow-hidden">
          <button
            onClick={() => setTrashOpen((v) => !v)}
            className="w-full flex items-center gap-2 px-3 py-2 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
          >
            <Trash2 size={14} className="text-gray-500 flex-shrink-0" />
            <span className="text-xs font-semibold text-gray-600">
              Trash
            </span>
            <span className="text-[10px] text-gray-400 bg-white border border-gray-200 rounded-full px-1.5 py-0.5">
              {trashedItems.length}
            </span>
            <span className="ml-auto text-[10px] text-gray-400">
              {trashOpen ? "Hide ▲" : "Show ▼"}
            </span>
          </button>

          {trashOpen && (
            <div className="p-2 space-y-1.5">
              {trashedItems.map((it) => {
                const deletedAt = it.deletedAt ?? Date.now();
                const daysLeft = Math.max(
                  0,
                  Math.ceil((deletedAt + TRASH_TTL_MS - Date.now()) / (24 * 60 * 60 * 1000)),
                );
                return (
                  <div
                    key={it.id}
                    className="flex items-center gap-2 bg-white border border-gray-200 rounded-lg px-2.5 py-2"
                  >
                    <FileText size={12} className="text-gray-400 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-gray-600 truncate" title={it.name}>
                        {it.name}
                      </p>
                      <p className="text-[10px] text-gray-400">
                        Silindi: {fmtAgo(deletedAt)} ·
                        <span className={daysLeft <= 3 ? "text-amber-600 font-semibold" : ""}>
                          {" "}deleted permanently in {daysLeft} d
                        </span>
                      </p>
                    </div>
                    <button
                      onClick={() => onRestoreFromTrash(it.id)}
                      disabled={restoring === it.id}
                      className="inline-flex items-center gap-1 text-[10px] font-semibold text-indigo-600 hover:text-indigo-700 bg-indigo-50 hover:bg-indigo-100 px-2 py-1 rounded transition-colors disabled:opacity-50"
                      title="Restore"
                    >
                      <RotateCcw size={10} />
                      Restore
                    </button>
                    <button
                      onClick={() => onPurgeOne(it.id)}
                      className="inline-flex items-center text-[10px] font-semibold text-red-600 hover:text-red-700 bg-red-50 hover:bg-red-100 px-2 py-1 rounded transition-colors"
                      title="Delete permanently"
                    >
                      <Trash2 size={10} />
                    </button>
                  </div>
                );
              })}
              <button
                onClick={onEmptyTrash}
                className="w-full mt-1 text-[10px] font-semibold text-red-600 hover:text-red-700 hover:bg-red-50 py-1.5 rounded transition-colors"
              >
                Empty trash ({trashedItems.length})
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
