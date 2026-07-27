/**
 * Local-only autosave for uSTAT sessions.
 *
 * Mirrors the endnotere editor pattern (Dexie / IndexedDB store + 600 ms
 * debounce + cross-tab BroadcastChannel) but adapted for uSTAT's
 * server-side session model:
 *
 *   - The FastAPI backend keeps the live DataFrame in memory with a 30 min
 *     TTL. Snapshotting it for "resume where I left off" requires fetching
 *     the existing JSON exporter (`GET /api/sessions/{sid}/save_session`)
 *     and storing the blob in IndexedDB.
 *   - Resuming reuploads the stored JSON via `POST
 *     /api/sessions/load_session` (multipart File), receives a fresh
 *     session_id, and hands the resulting Session object back to the
 *     Zustand store.
 *
 * Nothing leaves the user's browser — the snapshot lives in IndexedDB,
 * scoped to this origin only.
 */

import Dexie, { type EntityTable } from "dexie";

// ── Types ─────────────────────────────────────────────────────────────

/** Lightweight metadata kept alongside each saved blob. Mirrored to the
 *  card grid on the upload zone without having to deserialise the full
 *  session every render. */
export interface RecentSessionMeta {
  id: string;             // local UUID — not the server session_id
  serverSessionId?: string; // last-known server id (helps dedupe)
  name: string;             // dataset filename or user-chosen label
  savedAt: number;          // epoch ms — used for LRU ordering
  sizeBytes: number;        // JSON blob length, for the storage cap
  nRows?: number;
  nCols?: number;
  activeTab?: string;       // header tab the user was on
  source: "auto" | "manual";
  // Soft-delete (trash) timestamp. null/undefined = active record; a value =
  // moved to the Trash bin at that epoch ms. After TRASH_TTL_MS it is purged
  // permanently (local + Drive) by purgeExpiredTrash().
  deletedAt?: number | null;
  // Set on rows the user deliberately created with Duplicate. A copy is
  // byte-identical to its source, so it matches the rows x cols x bytes
  // fingerprint dedupeByName() uses and would otherwise delete the original
  // the moment the list was next read.
  userCopy?: boolean;
}

/** Full record stored in IndexedDB — extends the metadata with the
 *  serialised session JSON. */
export interface RecentSessionRecord extends RecentSessionMeta {
  // Stringified JSON returned by the backend's save_session endpoint.
  // Keep as a string (not parsed) so re-uploading is a one-liner and
  // there is no schema-version coupling here.
  payload: string;
}

// ── DB schema ─────────────────────────────────────────────────────────

export interface SessionDB extends Dexie {
  sessions: EntityTable<RecentSessionRecord, "id">;
}

let _db: SessionDB | null = null;

function getDb(): SessionDB {
  if (typeof window === "undefined") {
    throw new Error("sessionDb is browser-only");
  }
  if (_db) return _db;
  const db = new Dexie("wiz3-sessions-v1") as SessionDB;
  db.version(1).stores({
    // Indexes: id (primary) + savedAt (LRU ordering) + serverSessionId
    // (dedup lookups on autosave).
    sessions: "id, savedAt, serverSessionId",
  });
  // v2 adds a `name` index for filename-based dedup — the server
  // session_id isn't stable across reloads, so the filename is the
  // identity that collapses duplicate rows.
  db.version(2).stores({
    sessions: "id, savedAt, serverSessionId, name",
  });
  // v3 adds a `deletedAt` index for the Trash bin — records moved to trash
  // are soft-deleted (deletedAt = timestamp) rather than hard-removed, so
  // they can be restored and are aged out after TRASH_TTL_MS.
  db.version(3).stores({
    sessions: "id, savedAt, serverSessionId, name, deletedAt",
  });
  _db = db;
  return db;
}

// ── Capacity & Trash policy ──────────────────────────────────────────

const MAX_SESSIONS = 20;
const MAX_TOTAL_BYTES = 200 * 1024 * 1024; // 200 MB hard cap

/** How long a trashed record survives before permanent deletion. */
export const TRASH_TTL_MS = 30 * 24 * 60 * 60 * 1000; // 30 days
/** How often purgeExpiredTrash runs while the app is open. */
export const TRASH_PURGE_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

/** Drop the oldest ACTIVE records until the store fits within the cap.
 *  Trashed records are ignored here — they age out via purgeExpiredTrash(). */
async function pruneToCap(): Promise<void> {
  const db = getDb();
  const all = await db.sessions
    .orderBy("savedAt")
    .filter((r) => !r.deletedAt)
    .toArray();
  if (all.length <= MAX_SESSIONS) {
    const total = all.reduce((s, r) => s + r.sizeBytes, 0);
    if (total <= MAX_TOTAL_BYTES) return;
  }
  let total = all.reduce((s, r) => s + r.sizeBytes, 0);
  let i = 0;
  // Oldest-first deletion until both caps satisfied.
  while (
    (all.length - i > MAX_SESSIONS || total > MAX_TOTAL_BYTES) &&
    i < all.length - 1 // never delete the very newest
  ) {
    total -= all[i].sizeBytes;
    await db.sessions.delete(all[i].id);
    i++;
  }
}

// ── Public API ────────────────────────────────────────────────────────

/** Generate a local id. Avoids importing a uuid library — Dexie just
 *  needs uniqueness within this store. */
function newLocalId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

/** Collapse ACTIVE records that look like the same logical dataset down to
 *  the newest one. Trashed records are excluded — a name can legitimately
 *  appear once in the active list and once in the Trash.
 *
 *  The server session_id changes on every upload / reload / restore, so the
 *  same logical file accumulates a row per browser session. Two identity
 *  passes collapse them:
 *    1. by name — covers renames and re-uploads where the filename is stable
 *    2. by content fingerprint (rows × cols × byte size) — catches the
 *       "session_XXX.json" duplicate that save_session used to mint when the
 *       backend hadn't persisted the uploaded filename (now fixed in
 *       upload.py). Two snapshots of the exact same data land side by side
 *       under different names; we keep the newest and drop the stale alias. */
async function dedupeByName(): Promise<void> {
  const db = getDb();
  const rows = await db.sessions
    .orderBy("savedAt")
    .reverse()
    .filter((r) => !r.deletedAt)
    .toArray();
  const seenNames = new Set<string>();
  const seenFp = new Set<string>();
  const stale: string[] = [];
  for (const r of rows) {
    // A user-made copy is exempt in both directions: it is never deleted as
    // stale, and it never claims the fingerprint slot — otherwise, being the
    // newer row, it would shadow the original it was copied from and the
    // dedupe would silently delete the source.
    if (r.userCopy) continue;
    const nameKey = r.name || r.id;
    const fpKey = `${r.nRows ?? "?"}x${r.nCols ?? "?"}:${r.sizeBytes}`;
    // Name is the primary identity. A real fingerprint collision (two
    // genuinely different files that happen to share rows×cols×bytes) is
    // vanishingly rare, and even then we'd just keep the newest snapshot —
    // acceptable for a local recents list.
    const dupeByName = seenNames.has(nameKey);
    const dupeByFp = r.nRows != null && r.nCols != null && seenFp.has(fpKey);
    if (dupeByName || dupeByFp) {
      stale.push(r.id);   // older (rows already sorted newest-first)
    } else {
      seenNames.add(nameKey);
      if (r.nRows != null && r.nCols != null) seenFp.add(fpKey);
    }
  }
  if (stale.length) {
    await db.sessions.bulkDelete(stale);
  }
}

/** List ACTIVE records (deletedAt null/undefined), newest first. */
export async function listRecentSessions(): Promise<RecentSessionMeta[]> {
  const db = getDb();
  await dedupeByName();
  const rows = await db.sessions
    .orderBy("savedAt")
    .reverse()
    .filter((r) => !r.deletedAt)
    .toArray();
  return rows.map((row) => {
    const { payload: _ignored, ...meta } = row;
    void _ignored;
    return meta;
  });
}

/** List TRASHED records (deletedAt set), most-recently-trashed first. */
export async function listTrashedSessions(): Promise<RecentSessionMeta[]> {
  const db = getDb();
  const rows = await db.sessions
    .orderBy("deletedAt")
    .reverse()
    .filter((r) => !!r.deletedAt)
    .toArray();
  return rows.map((row) => {
    const { payload: _ignored, ...meta } = row;
    void _ignored;
    return meta;
  });
}

export async function getRecentSession(id: string): Promise<RecentSessionRecord | undefined> {
  return getDb().sessions.get(id);
}

/** Get a trashed record including its payload (for restore). */
export async function getTrashedSession(id: string): Promise<RecentSessionRecord | undefined> {
  const rec = await getDb().sessions.get(id);
  return rec && rec.deletedAt ? rec : undefined;
}

// ── Trash: soft-delete / restore / purge ─────────────────────────────

/** Move a record to the Trash (soft delete). The record stays in IndexedDB
 *  (and is mirrored to Drive as a tombstone) so it can be restored within
 *  TRASH_TTL_MS; after that purgeExpiredTrash() removes it permanently. */
export async function trashSession(id: string): Promise<void> {
  await getDb().sessions.update(id, { deletedAt: Date.now() });
}

/** Restore a trashed record back to active. Resets savedAt to now so the
 *  card reappears at the top of the Recent list and is re-pushed to Drive
 *  as active (tombstone cleared). */
export async function restoreSession(id: string): Promise<void> {
  await getDb().sessions.update(id, { deletedAt: null, savedAt: Date.now() });
}

/** Permanently delete a single record from IndexedDB (hard delete). Used
 *  by "Delete permanently" in the Trash bin and by purgeExpiredTrash(). */
export async function purgeSession(id: string): Promise<void> {
  await getDb().sessions.delete(id);
}

/** Permanently delete ALL trashed records. */
export async function emptyTrash(): Promise<void> {
  const db = getDb();
  const ids = await db.sessions
    .where("deletedAt")
    .above(0)
    .primaryKeys();
  await db.sessions.bulkDelete(ids);
}

/**
 * Permanently delete trashed records older than TRASH_TTL_MS. Idempotent —
 * safe to call on app open and periodically. Returns the number purged.
 * Mirrors notepad's auto-expiry: items stay in Trash for 30 days, then are
 * gone for good (local AND Drive, via the tombstone sync in cloudSync.ts).
 */
export async function purgeExpiredTrash(
  ttlMs: number = TRASH_TTL_MS,
  now: number = Date.now(),
): Promise<number> {
  const db = getDb();
  const cutoff = now - ttlMs;
  // deletedAt index gives us all trashed records; filter to expired only.
  const expiredIds = await db.sessions
    .where("deletedAt")
    .belowOrEqual(cutoff)
    .primaryKeys();
  if (expiredIds.length) {
    await db.sessions.bulkDelete(expiredIds);
  }
  return expiredIds.length;
}

/** Remove all ACTIVE records (leaves the Trash untouched). */
export async function clearAllRecentSessions(): Promise<void> {
  const db = getDb();
  const ids = await db.sessions
    .filter((r) => !r.deletedAt)
    .primaryKeys();
  await db.sessions.bulkDelete(ids);
}

/** Permanently delete a single ACTIVE record (legacy hard-delete). Kept for
 *  any callers that still want immediate removal rather than trash. */
export async function deleteRecentSession(id: string): Promise<void> {
  await getDb().sessions.delete(id);
}

/** Copy a saved session into a new, independent row.
 *
 *  Deliberately does NOT go through upsertRecentSession: that function
 *  dedupes by serverSessionId and then by name, so a copy made through it
 *  would match the original and overwrite the very row being duplicated.
 *  The copy is written directly with a fresh id, a free name, and no
 *  serverSessionId — it is a snapshot on disk, not a second handle on the
 *  live server session, which would otherwise let edits in one leak into
 *  the other's autosave.
 */
export async function duplicateRecentSession(
  id: string,
): Promise<RecentSessionMeta | undefined> {
  const db = getDb();
  const source = await db.sessions.get(id);
  if (!source) return undefined;

  const taken = new Set((await db.sessions.toArray()).map((s) => s.name));
  const base = `${source.name} (copy)`;
  let name = base;
  for (let i = 2; taken.has(name); i++) name = `${source.name} (copy ${i})`;

  const rec: RecentSessionRecord = {
    ...source,
    id: newLocalId(),
    serverSessionId: undefined,
    name,
    savedAt: Date.now(),
    source: "manual",
    userCopy: true,
  };
  await db.sessions.put(rec);
  await pruneToCap();
  const { payload: _ignored, ...meta } = rec;
  void _ignored;
  return meta;
}

/** Upsert a session blob, deduping so the same logical file occupies a
 *  single row.
 *
 *  The server session_id is NOT stable — every upload, reload, and
 *  restore mints a fresh one — so keying only on it spawns a new row per
 *  browser session (the "why are there 3 copies of my file" bug). We
 *  match in two passes:
 *    1. by serverSessionId  → same in-progress session (covers renames,
 *       where the id is stable but the name just changed)
 *    2. by name             → same file across reloads / re-uploads /
 *       restores (the id differs but the filename is the user's stable
 *       identity)
 */
export async function upsertRecentSession(input: {
  serverSessionId: string;
  name: string;
  payload: string;
  nRows?: number;
  nCols?: number;
  activeTab?: string;
  source: "auto" | "manual";
}): Promise<RecentSessionMeta> {
  const db = getDb();
  let existing =
    input.serverSessionId
      ? await db.sessions.where("serverSessionId").equals(input.serverSessionId).first()
      : undefined;
  if (!existing && input.name) {
    existing = await db.sessions.where("name").equals(input.name).first();
  }
  const id = existing?.id ?? newLocalId();
  const rec: RecentSessionRecord = {
    id,
    serverSessionId: input.serverSessionId,
    name: input.name,
    payload: input.payload,
    sizeBytes: input.payload.length,
    nRows: input.nRows,
    nCols: input.nCols,
    activeTab: input.activeTab,
    savedAt: Date.now(),
    source: input.source,
  };
  await db.sessions.put(rec);
  await pruneToCap();
  const { payload: _ignored, ...meta } = rec;
  void _ignored;
  return meta;
}

/**
 * Raw upsert used by the Google Drive cloud-sync pull path. Identical to
 * {@link upsertRecentSession} EXCEPT it preserves the original `savedAt`
 * timestamp instead of stamping `Date.now()`. This is essential for the
 * last-write-wins clock: when a remote snapshot is pulled back locally, the
 * record's `savedAt` must reflect when the snapshot was *taken* (so a
 * subsequent push does not mark it newer than the remote and bounce it
 * back), not when it landed in IndexedDB.
 */
export async function upsertRecentSessionRaw(input: {
  serverSessionId?: string;
  name: string;
  payload: string;
  savedAt: number;
  nRows?: number;
  nCols?: number;
  activeTab?: string;
  source: "auto" | "manual";
}): Promise<RecentSessionMeta> {
  const db = getDb();
  let existing: RecentSessionRecord | undefined =
    input.serverSessionId
      ? await db.sessions.where("serverSessionId").equals(input.serverSessionId).first()
      : undefined;
  if (!existing && input.name) {
    existing = await db.sessions.where("name").equals(input.name).first();
  }
  const id = existing?.id ?? newLocalId();
  const rec: RecentSessionRecord = {
    id,
    serverSessionId: input.serverSessionId,
    name: input.name,
    payload: input.payload,
    sizeBytes: input.payload.length,
    nRows: input.nRows,
    nCols: input.nCols,
    activeTab: input.activeTab,
    savedAt: input.savedAt,
    source: input.source,
  };
  await db.sessions.put(rec);
  await pruneToCap();
  const { payload: _ignored, ...meta } = rec;
  void _ignored;
  return meta;
}

// ── Cross-tab notifications ───────────────────────────────────────────

const CHANNEL = "wiz3-sessions";

let _bc: BroadcastChannel | null = null;
const _listeners = new Set<() => void>();

function getChannel(): BroadcastChannel | null {
  if (typeof window === "undefined") return null;
  if (typeof BroadcastChannel === "undefined") return null;
  if (_bc) return _bc;
  _bc = new BroadcastChannel(CHANNEL);
  return _bc;
}

export function notifySessionsChanged(): void {
  _listeners.forEach((listener) => {
    try {
      listener();
    } catch {
      /* listener failures must not block sync/save notifications */
    }
  });
  const bc = getChannel();
  if (bc) bc.postMessage({ type: "changed", at: Date.now() });
}

export function subscribeSessions(onChange: () => void): () => void {
  _listeners.add(onChange);
  const bc = getChannel();
  if (!bc) return () => { _listeners.delete(onChange); };
  const handler = () => onChange();
  bc.addEventListener("message", handler);
  return () => {
    _listeners.delete(onChange);
    bc.removeEventListener("message", handler);
  };
}

// ── Storage estimate (for diagnostics / UI) ───────────────────────────

export async function getStorageEstimate(): Promise<{
  count: number;
  bytes: number;
  capCount: number;
  capBytes: number;
}> {
  const db = getDb();
  const all = await db.sessions.toArray();
  const bytes = all.reduce((s, r) => s + r.sizeBytes, 0);
  return {
    count: all.length,
    bytes,
    capCount: MAX_SESSIONS,
    capBytes: MAX_TOTAL_BYTES,
  };
}
