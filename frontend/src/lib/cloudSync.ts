/**
 * Google Drive Cloud Sync for uSTAT session snapshots (appDataFolder).
 *
 * Architecture (adapted from notepad's `js/cloud-sync.js` to uSTAT's
 * server-side session model):
 *   - Auth: Google Identity Services (GIS) implicit token flow, with a
 *     full-page redirect fallback for environments where popups are
 *     unreliable (iOS standalone, embedded WebViews, blocked 3rd-party
 *     cookies).
 *   - Storage: Drive REST v3, scope=drive.appdata (hidden per-app folder).
 *   - Layout:
 *       ustat-index.json        → { version, lastSync, sessions: [{id,name,updated,bytes,rev}] }
 *       ustat-session-<id>.json → { meta: RecentSessionMeta, payload: save_session JSON string }
 *   - Sync: last-write-wins by `updated` timestamp, name-based dedup
 *     (matches sessionDb.ts's upsertRecentSession identity rules — the
 *     server session_id is not stable across reloads, so the filename is
 *     the durable identity).
 *   - Triggers: pull on token acquired, push on dirty (debounced),
 *     background pull every PULL_INTERVAL_MS.
 *
 * Integration: lives ALONGSIDE the existing IndexedDB autosave layer
 * (sessionDb.ts). IndexedDB remains the primary local store; Drive is a
 * mirror/backup that the module reads from and writes to via the same
 * sessionDb functions (listRecentSessions / getRecentSession /
 * upsertRecentSession / deleteRecentSession).
 */

import {
  CLOUD_CONFIG,
  INDEX_FILE_NAME,
  sessionFileName,
} from "./cloudConfig";
import {
  listRecentSessions,
  listTrashedSessions,
  getRecentSession,
  upsertRecentSessionRaw,
  trashSession,
  purgeExpiredTrash,
  purgeSession,
  notifySessionsChanged,
  TRASH_TTL_MS,
  type RecentSessionMeta,
  type RecentSessionRecord,
} from "./sessionDb";

// ── Types ────────────────────────────────────────────────────────────

export type CloudStatus =
  | "idle"
  | "syncing"
  | "ok"
  | "error"
  | "setupNeeded";

export interface CloudUserInfo {
  email?: string;
  name?: string;
  picture?: string;
}

export interface CloudStatusInfo {
  signedIn: boolean;
  status: CloudStatus;
  message: string;
  user: CloudUserInfo | null;
  lastSync: number | null;
}

type Listener = (status: CloudStatusInfo) => void;

// ── localStorage keys ────────────────────────────────────────────────

const LS_TOKEN = "ustat_cloud_token";
const LS_USER = "ustat_cloud_user";
const LS_LAST_SYNC = "ustat_cloud_last_sync";
const LS_MODE = "ustat_cloud_mode"; // 'popup' | 'redirect'
const SS_STATE = "ustat_oauth_state"; // CSRF state (sessionStorage)
const AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth";

// ── GIS type shims (minimal) ─────────────────────────────────────────

// Minimal GIS type shim — only `revoke` is still used (in signOut) now that
// sign-in/refresh use the redirect flow. Keeping the global so TS knows about
// window.google.accounts.oauth2.revoke.
interface GoogleAccounts {
  oauth2: {
    revoke: (token: string, cb: () => void) => void;
  };
}

declare global {
  interface Window {
    google?: { accounts: GoogleAccounts };
  }
}

// ── State ────────────────────────────────────────────────────────────

let accessToken: string | null = null;
let tokenExpiresAt = 0;
let userInfo: CloudUserInfo | null = null;
let signedIn = false;
let authMode: "popup" | "redirect" =
  (localStorage.getItem(LS_MODE) as "popup" | "redirect") || "popup";
let status: CloudStatus = "idle";
let statusMsg = "";
let pushTimer: ReturnType<typeof setTimeout> | null = null;
let pullTimer: ReturnType<typeof setInterval> | null = null;
let initialized = false;
let listeners: Listener[] = [];
let inFlight = false;

// ── Platform detection ───────────────────────────────────────────────
// Sign-in always uses the full-page redirect flow (see note above), so no
// popup/platform-specific routing is needed here.

function redirectUri(): string {
  // Must EXACTLY match an "Authorized redirect URI" in the OAuth client config.
  return location.origin + "/";
}

function randomState(): string {
  try {
    const a = new Uint8Array(16);
    crypto.getRandomValues(a);
    return Array.from(a)
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  } catch {
    return Date.now().toString(36) + Math.random().toString(36).slice(2);
  }
}

// ── Listeners / status ───────────────────────────────────────────────

function emit(): void {
  const snapshot = getStatus();
  for (const fn of listeners) {
    try {
      fn(snapshot);
    } catch (e) {
      console.warn("[cloud] listener", e);
    }
  }
}

function setStatus(s: CloudStatus, msg = ""): void {
  status = s;
  statusMsg = msg;
  emit();
}

export function getStatus(): CloudStatusInfo {
  return {
    signedIn,
    status,
    message: statusMsg,
    user: userInfo,
    lastSync: parseInt(localStorage.getItem(LS_LAST_SYNC) || "0", 10) || null,
  };
}

export function subscribe(fn: Listener): () => void {
  listeners.push(fn);
  return () => {
    listeners = listeners.filter((l) => l !== fn);
  };
}

// ── Token persistence ────────────────────────────────────────────────

function persistToken(): void {
  if (accessToken && tokenExpiresAt > Date.now()) {
    localStorage.setItem(
      LS_TOKEN,
      JSON.stringify({ t: accessToken, e: tokenExpiresAt }),
    );
    localStorage.setItem(LS_MODE, authMode);
  } else {
    localStorage.removeItem(LS_TOKEN);
  }
}

function restoreToken(): boolean {
  try {
    const raw = localStorage.getItem(LS_TOKEN);
    if (!raw) return false;
    const { t, e } = JSON.parse(raw) as { t?: string; e?: number };
    if (!t || !e || e <= Date.now() + 60_000) return false; // expired or expiring soon
    accessToken = t;
    tokenExpiresAt = e;
    return true;
  } catch {
    return false;
  }
}

function restoreUser(): CloudUserInfo | null {
  try {
    const raw = localStorage.getItem(LS_USER);
    if (!raw) return null;
    return JSON.parse(raw) as CloudUserInfo;
  } catch {
    return null;
  }
}

// NOTE: sign-in and refresh both use the full-page redirect flow. The GIS
// popup token flow was removed because it is unreliable under third-party
// cookie blocking (Safari ITP / Firefox ETP / incognito / embedded
// WebViews): Google's GIS script opens the popup then immediately closes it
// without showing consent, firing { type: 'popup_closed' }. The redirect
// flow navigates the whole tab and works everywhere.

async function fetchUserInfo(): Promise<void> {
  try {
    const r = await fetch("https://www.googleapis.com/oauth2/v3/userinfo", {
      headers: { Authorization: "Bearer " + accessToken },
    });
    if (r.ok) {
      userInfo = (await r.json()) as CloudUserInfo;
      localStorage.setItem(LS_USER, JSON.stringify(userInfo));
      return;
    }
  } catch (e) {
    console.warn("[cloud] userinfo fetch failed, trying Drive about", e);
  }

  try {
    // Fallback: use Drive API's about endpoint (authorized by drive.appdata scope)
    const r = await driveFetch("/about?fields=user");
    const data = (await r.json()) as {
      user?: { emailAddress?: string; displayName?: string; photoLink?: string };
    };
    if (data?.user) {
      userInfo = {
        email: data.user.emailAddress,
        name: data.user.displayName,
        picture: data.user.photoLink,
      };
      localStorage.setItem(LS_USER, JSON.stringify(userInfo));
    }
  } catch (e) {
    console.warn("[cloud] userinfo fallback failed", e);
  }
}

// ── Redirect (implicit) flow — universal, works on iOS standalone & TWA ─
// Full-page navigation to Google, returns with #access_token=... in URL fragment.

function buildAuthUrl(silent: boolean): string {
  const state = randomState();
  try {
    sessionStorage.setItem(SS_STATE, state);
  } catch {
    /* ignore */
  }
  const params = new URLSearchParams({
    client_id: CLOUD_CONFIG.GOOGLE_CLIENT_ID,
    redirect_uri: redirectUri(),
    response_type: "token",
    scope: CLOUD_CONFIG.SCOPE,
    include_granted_scopes: "true",
    state,
  });
  if (silent) params.set("prompt", "none");
  if (userInfo?.email) params.set("login_hint", userInfo.email);
  return AUTH_ENDPOINT + "?" + params.toString();
}

async function startRedirectAuth(silent: boolean): Promise<void> {
  if (!CLOUD_CONFIG.GOOGLE_CLIENT_ID) {
    setStatus("setupNeeded", "");
    return;
  }
  authMode = "redirect";
  localStorage.setItem(LS_MODE, "redirect");
  location.href = buildAuthUrl(silent);
}

// Parse #access_token / #error from the URL after a redirect return.
// Returns 'ok' | 'error' | null (not a callback).
function handleRedirectCallback(): "ok" | "error" | null {
  const hash = location.hash || "";
  if (hash.indexOf("access_token") === -1 && hash.indexOf("error=") === -1)
    return null;
  const frag = new URLSearchParams(hash.replace(/^#/, ""));
  const token = frag.get("access_token");
  const err = frag.get("error");
  const state = frag.get("state");
  let savedState: string | null = null;
  try {
    savedState = sessionStorage.getItem(SS_STATE);
    sessionStorage.removeItem(SS_STATE);
  } catch {
    /* ignore */
  }
  // Clean the URL (drop fragment + any query) regardless of outcome
  try {
    history.replaceState(null, "", location.pathname + location.search);
  } catch {
    /* ignore */
  }
  if (err) {
    console.warn("[cloud] redirect auth error:", err);
    return "error";
  }
  if (!token) return "error";
  if (savedState && state !== savedState) {
    console.warn("[cloud] OAuth state mismatch — possible CSRF, ignoring token");
    return "error";
  }
  const expiresIn = parseInt(frag.get("expires_in") || "3600", 10);
  accessToken = token;
  tokenExpiresAt = Date.now() + (expiresIn - 60) * 1000;
  signedIn = true;
  authMode = "redirect";
  persistToken();
  return "ok";
}

// ── Sign in / out ────────────────────────────────────────────────────

export async function signIn(): Promise<void> {
  if (!CLOUD_CONFIG.GOOGLE_CLIENT_ID) {
    setStatus("setupNeeded", "OAuth client ID not configured in cloudConfig.ts");
    return;
  }
  // Already signed in with a live token — don't re-trigger auth. Just
  // surface the current status so the UI is in sync.
  if (signedIn && accessToken && tokenExpiresAt > Date.now()) {
    emit();
    return;
  }
  // Use the full-page REDIRECT flow as the default. The GIS popup token
  // flow is unreliable in modern browsers that block third-party cookies
  // (Safari ITP, Firefox ETP, Chrome incognito, embedded WebViews): Google's
  // GIS script opens the popup then immediately closes it WITHOUT showing
  // consent, firing { type: 'popup_closed' } and leaving the user stuck.
  // The redirect flow navigates the whole tab to Google and back, which
  // works regardless of cookie/popup-blocker settings.
  await startRedirectAuth(false);
}

export async function signOut(): Promise<void> {
  if (
    accessToken &&
    window.google?.accounts?.oauth2
  ) {
    try {
      window.google.accounts.oauth2.revoke(accessToken, () => {});
    } catch {
      /* ignore */
    }
  }
  accessToken = null;
  tokenExpiresAt = 0;
  signedIn = false;
  userInfo = null;
  localStorage.removeItem(LS_TOKEN);
  localStorage.removeItem(LS_USER);
  localStorage.removeItem(LS_MODE);
  if (pushTimer) clearTimeout(pushTimer);
  if (pullTimer) clearInterval(pullTimer);
  pullTimer = null;
  setStatus("idle", "");
}

// Refresh the access token before it expires.
// Sign-in is always via the full-page redirect flow, so refresh also uses a
// silent redirect (prompt=none): Google immediately bounces back with a
// fresh token if consent is still valid, otherwise it asks for consent.
// `allowRedirect=false` means the caller can't tolerate a navigation (e.g.
// mid-sync); in that case we throw and the caller surfaces "re-sign-in".
async function refreshToken(allowRedirect = true): Promise<void> {
  if (!allowRedirect) {
    throw new Error("Token expired, re-sign-in required");
  }
  // prompt=none → page navigates to Google and back; resumes via init() on
  // return (the page is unloading, so nothing after this runs).
  await startRedirectAuth(true);
}

// ── Drive REST ───────────────────────────────────────────────────────

async function driveFetch(
  path: string,
  init?: RequestInit,
  allowRedirect = false,
): Promise<Response> {
  if (!accessToken) throw new Error("No access token");
  if (tokenExpiresAt && tokenExpiresAt <= Date.now()) {
    await refreshToken(allowRedirect);
  }
  const opts = init ?? {};
  opts.headers = {
    ...(opts.headers as Record<string, string> | undefined),
    Authorization: "Bearer " + accessToken,
  };
  const url = path.startsWith("http")
    ? path
    : CLOUD_CONFIG.DRIVE_API + path;
  let r = await fetch(url, opts);
  // 401 → token rejected server-side (revoked/expired early). Refresh once, retry.
  if (r.status === 401) {
    try {
      await refreshToken(allowRedirect);
      (opts.headers as Record<string, string>).Authorization =
        "Bearer " + accessToken;
      r = await fetch(url, opts);
    } catch {
      signedIn = false;
      localStorage.removeItem(LS_TOKEN);
      setStatus("idle", "Re-sign-in required");
      throw new Error("Token rejected (401), re-sign-in required");
    }
  }
  if (!r.ok) {
    const errText = await r.text().catch(() => "");
    throw new Error(`Drive ${r.status}: ${errText.slice(0, 200)}`);
  }
  return r;
}

interface DriveFile {
  id: string;
  name: string;
  modifiedTime?: string;
  size?: string;
}

async function listAppData(allowRedirect = false): Promise<DriveFile[]> {
  const r = await driveFetch(
    "/files?spaces=appDataFolder&fields=files(id,name,modifiedTime,size)&pageSize=1000",
    undefined,
    allowRedirect,
  );
  const data = (await r.json()) as { files?: DriveFile[] };
  return data.files ?? [];
}

async function downloadJson<T>(fileId: string, allowRedirect = false): Promise<T> {
  const r = await driveFetch(`/files/${fileId}?alt=media`, undefined, allowRedirect);
  return (await r.json()) as T;
}

/** Permanently delete a Drive file (used to purge expired tombstones). */
async function deleteFile(fileId: string, allowRedirect = false): Promise<void> {
  try {
    await driveFetch(`/files/${fileId}`, { method: "DELETE" }, allowRedirect);
  } catch (e) {
    console.warn("[cloud] deleteFile failed (non-fatal)", fileId, e);
  }
}

async function uploadJson<T>(
  name: string,
  json: unknown,
  existingFileId: string | null,
  allowRedirect = false,
): Promise<{ id: string } & T> {
  const meta = existingFileId
    ? { name }
    : { name, parents: ["appDataFolder"], mimeType: "application/json" };
  const boundary = "-------USTATCloud" + Math.random().toString(36).slice(2);
  const body =
    `--${boundary}\r\n` +
    `Content-Type: application/json; charset=UTF-8\r\n\r\n` +
    JSON.stringify(meta) +
    `\r\n` +
    `--${boundary}\r\n` +
    `Content-Type: application/json\r\n\r\n` +
    JSON.stringify(json) +
    `\r\n` +
    `--${boundary}--`;
  const path = existingFileId
    ? `${CLOUD_CONFIG.DRIVE_UPLOAD}/files/${existingFileId}?uploadType=multipart`
    : `${CLOUD_CONFIG.DRIVE_UPLOAD}/files?uploadType=multipart`;
  const r = await driveFetch(
    path,
    {
      method: existingFileId ? "PATCH" : "POST",
      headers: { "Content-Type": `multipart/related; boundary=${boundary}` },
      body,
    },
    allowRedirect,
  );
  return (await r.json()) as { id: string } & T;
}

// ── Remote index shape ───────────────────────────────────────────────

interface RemoteIndexEntry {
  id: string; // local uSTAT id (matches sessionDb RecentSessionRecord.id)
  name: string;
  updated: number; // epoch ms (LWW clock — matches sessionDb savedAt)
  bytes: number;
  rev: string | null; // Drive file id of ustat-session-<id>.json
  // Tombstone: null/omitted = active; a timestamp = moved to Trash at that
  // time. After TRASH_TTL_MS the tombstone (and its session file) are
  // permanently deleted from Drive during push().
  deletedAt?: number | null;
}

interface RemoteIndex {
  version: number;
  lastSync: number;
  sessions: RemoteIndexEntry[];
}

// On-Drive per-session file shape. `meta` mirrors sessionDb's metadata
// (without payload); `payload` is the save_session JSON string, kept as a
// string so re-upload is a one-liner with no schema coupling.
interface RemoteSessionFile {
  meta: RecentSessionMeta;
  payload: string;
}

// ── Sync ─────────────────────────────────────────────────────────────

function findFile(files: DriveFile[], name: string): DriveFile | undefined {
  return files.find((f) => f.name === name);
}

/**
 * Pull remote session snapshots + tombstones into IndexedDB.
 *
 * Tombstone handling: if a remote entry carries `deletedAt`, the matching
 * local record is soft-deleted (moved to Trash) with that timestamp — so a
 * delete on device A propagates to device B. If the remote tombstone is
 * already older than TRASH_TTL_MS, the record is purged permanently on both
 * sides during push() (Drive file + index entry) rather than here.
 */
async function pull(allowRedirect = false): Promise<{
  pulled: number;
  failures: string[];
}> {
  const files = await listAppData(allowRedirect);
  const indexFile = findFile(files, INDEX_FILE_NAME);
  if (!indexFile) return { pulled: 0, failures: [] }; // first sync, nothing remote
  const remoteIndex = await downloadJson<RemoteIndex>(
    indexFile.id,
    allowRedirect,
  );
  const remoteSessions = remoteIndex.sessions ?? [];
  const fileByName = new Map(files.map((f) => [f.name, f]));

  // Build the current local index (active + trashed) once so we can compare.
  const localActive = await listRecentSessions();
  const localTrashed = await listTrashedSessions();
  const localList = [...localActive, ...localTrashed];
  const localById = new Map(localList.map((m) => [m.id, m]));

  let pulled = 0;
  const failures: string[] = [];
  for (const r of remoteSessions) {
    // Identity by local id first, then by name (sessionDb dedup rules).
    const loc = localById.get(r.id);
    const localByName = localList.find((m) => m.name === r.name);
    const candidate = loc ?? localByName;

    // ── Tombstone (remote trashed) ───────────────────────────────────
    if (r.deletedAt) {
      // Already expired remotely? push() will have removed it from the index;
      // if we still see it (e.g. mid-sync), purge locally too.
      if (Date.now() - r.deletedAt > TRASH_TTL_MS) {
        if (candidate) await purgeSession(candidate.id);
        continue;
      }
      // Propagate the tombstone: only move-to-trash if the local record is
      // currently active AND the remote delete is newer than the local
      // snapshot (LWW — a more recent local edit wins over a stale delete).
      if (candidate && !candidate.deletedAt && r.deletedAt > (candidate.savedAt ?? 0)) {
        await trashSession(candidate.id);
        pulled++; // count as a pulled change so the UI refreshes
      }
      continue;
    }

    // ── Active session ───────────────────────────────────────────────
    const remoteNewer =
      !candidate || (r.updated ?? 0) > (candidate.savedAt ?? 0);
    if (!remoteNewer) continue;

    const noteFile = fileByName.get(sessionFileName(r.id));
    if (!noteFile) {
      failures.push(r.id);
      continue;
    }
    try {
      const downloaded = await downloadJson<RemoteSessionFile>(
        noteFile.id,
        allowRedirect,
      );
      // Upsert into IndexedDB using the raw path (preserves the original
      // savedAt timestamp so last-write-wins is stable across pull/push
      // cycles). Dedup + capacity pruning stay consistent with the autosave
      // hook, and the Recent Sessions cards refresh via notifySessionsChanged().
      await upsertRecentSessionRaw({
        // Prefer the remote's server session id; fall back to the local id
        // so we update in place rather than creating a duplicate row.
        serverSessionId: downloaded.meta.serverSessionId ?? r.id,
        name: downloaded.meta.name || r.name,
        payload: downloaded.payload,
        savedAt: downloaded.meta.savedAt || r.updated,
        nRows: downloaded.meta.nRows,
        nCols: downloaded.meta.nCols,
        activeTab: downloaded.meta.activeTab,
        source: downloaded.meta.source ?? "auto",
      });
      pulled++;
    } catch (e) {
      console.warn("[cloud] pull session failed", r.id, e);
      failures.push(r.id);
    }
  }
  if (pulled > 0) notifySessionsChanged();
  return { pulled, failures };
}

/**
 * Push local session snapshots + tombstones to Drive.
 *
 * - Active records: upload payload + index entry (deletedAt: null).
 * - Trashed records: keep index entry with deletedAt = timestamp (payload is
 *   still written so the record is restorable). After TRASH_TTL_MS the
 *   tombstone — and its Drive session file — are permanently deleted here.
 * - Records that exist only remotely (deleted on this device) are dropped
 *   from the index; their session files are left for Drive's own cleanup
 *   (orphaned appDataFolder files cost negligible quota).
 */
async function push(allowRedirect = false): Promise<{
  pushed: number;
  failures: string[];
}> {
  const files = await listAppData(allowRedirect);
  const indexFile = findFile(files, INDEX_FILE_NAME);
  const remoteIndex: RemoteIndex = indexFile
    ? await downloadJson<RemoteIndex>(indexFile.id, allowRedirect)
    : { version: 1, lastSync: 0, sessions: [] };
  const remoteMap = new Map(
    (remoteIndex.sessions ?? []).map((r) => [r.id, r]),
  );
  const fileMap = new Map(files.map((f) => [f.name, f]));

  // All local records: active + trashed. Both must be reflected in the index
  // so deletes/restore propagate to other devices.
  const localActive = await listRecentSessions();
  const localTrashed = await listTrashedSessions();
  const localAll = [...localActive, ...localTrashed];
  const now = Date.now();

  let pushed = 0;
  const failures: string[] = [];
  const newEntries: RemoteIndexEntry[] = [];

  for (const meta of localAll) {
    const isTrashed = !!meta.deletedAt;

    // ── Expired tombstone: purge from Drive (index + session file) ─────
    if (isTrashed && meta.deletedAt! && now - meta.deletedAt! > TRASH_TTL_MS) {
      const r = remoteMap.get(meta.id);
      const fname = sessionFileName(meta.id);
      const existing = fileMap.get(fname);
      if (existing) await deleteFile(existing.id, allowRedirect);
      if (r?.rev) await deleteFile(r.rev, allowRedirect);
      // Not added to newEntries → drops out of the remote index.
      pushed++;
      continue;
    }

    let rec: RecentSessionRecord | undefined;
    try {
      rec = await getRecentSession(meta.id);
    } catch {
      rec = undefined;
    }
    // For trashed records we may not need the payload, but we keep it so the
    // record stays restorable on other devices within the TTL window.
    const payload = rec?.payload ?? "";

    const r = remoteMap.get(meta.id);
    // For trashed records, "newer" is judged on the deletedAt tombstone vs
    // the remote deletedAt; for active records, on savedAt vs remote updated.
    const localClock = isTrashed ? meta.deletedAt! : meta.savedAt;
    const remoteClock = isTrashed ? r?.deletedAt ?? 0 : r?.updated ?? 0;
    const localNewer = !r || localClock > remoteClock;
    if (!localNewer) {
      // Keep the existing remote entry as-is (carry over its deletedAt too).
      newEntries.push({ ...r, deletedAt: isTrashed ? r!.deletedAt : r?.deletedAt });
      continue;
    }

    if (payload.length > CLOUD_CONFIG.MAX_SESSION_BYTES) {
      console.warn(
        "[cloud] session too large, skipping",
        meta.id,
        payload.length,
      );
      failures.push(meta.id);
      newEntries.push(
        r ?? {
          id: meta.id,
          name: meta.name,
          updated: meta.savedAt,
          bytes: payload.length,
          rev: null,
          deletedAt: meta.deletedAt ?? null,
        },
      );
      continue;
    }

    const fname = sessionFileName(meta.id);
    const existing = fileMap.get(fname);
    const fileObj: RemoteSessionFile = {
      meta: {
        id: meta.id,
        serverSessionId: meta.serverSessionId,
        name: meta.name,
        savedAt: meta.savedAt,
        sizeBytes: payload.length,
        nRows: meta.nRows,
        nCols: meta.nCols,
        activeTab: meta.activeTab,
        source: meta.source,
        deletedAt: meta.deletedAt ?? null,
      },
      payload,
    };
    try {
      // Only upload the session file if it actually changed (active records
      // always need their payload; trashed records only need it once so they
      // remain restorable — skip re-upload if the file already exists).
      let rev = r?.rev ?? null;
      if (!isTrashed || !existing) {
        const uploaded = await uploadJson<Record<string, never>>(
          fname,
          fileObj,
          existing ? existing.id : null,
          allowRedirect,
        );
        rev = uploaded.id;
      }
      newEntries.push({
        id: meta.id,
        name: meta.name,
        updated: meta.savedAt,
        bytes: payload.length,
        rev,
        deletedAt: meta.deletedAt ?? null,
      });
      pushed++;
    } catch (e) {
      console.warn("[cloud] push session failed", meta.id, e);
      failures.push(meta.id);
      newEntries.push(
        r ?? {
          id: meta.id,
          name: meta.name,
          updated: meta.savedAt,
          bytes: payload.length,
          rev: null,
          deletedAt: meta.deletedAt ?? null,
        },
      );
    }
  }

  // Always rebuild + push index (even if 0 sessions changed) so lastSync
  // advances and tombstones/removed locals fall out of the remote list.
  const newIndex: RemoteIndex = {
    version: 1,
    lastSync: Date.now(),
    sessions: newEntries,
  };
  await uploadJson(
    INDEX_FILE_NAME,
    newIndex,
    indexFile ? indexFile.id : null,
    allowRedirect,
  );
  return { pushed, failures };
}

export async function syncNow(allowRedirect = false): Promise<void> {
  if (!signedIn) return;
  if (inFlight) return;
  inFlight = true;
  setStatus("syncing", "");
  try {
    // Age out expired tombstones locally BEFORE syncing, so they are purged
    // from Drive (not re-pushed as live tombstones) on this same pass.
    await purgeExpiredTrash();
    const p1 = await pull(allowRedirect);
    const p2 = await push(allowRedirect);
    const failures = [...p1.failures, ...p2.failures];
    if (failures.length) {
      throw new Error(`${failures.length} session(s) could not be synced`);
    }
    localStorage.setItem(LS_LAST_SYNC, String(Date.now()));
    setStatus("ok", `Pulled ${p1.pulled}, pushed ${p2.pushed}`);
  } catch (e) {
    console.error("[cloud] sync error", e);
    setStatus("error", e instanceof Error ? e.message : String(e));
  } finally {
    inFlight = false;
  }
}

/**
 * Mark the local session set dirty — schedules a debounced Drive push.
 * Called by useAutoSession after each successful IndexedDB snapshot.
 */
export function markDirty(): void {
  if (!signedIn) return;
  if (pushTimer) clearTimeout(pushTimer);
  pushTimer = setTimeout(() => {
    void syncNow().catch((e) =>
      console.warn("[cloud] debounced sync", e),
    );
  }, CLOUD_CONFIG.PUSH_DEBOUNCE_MS);
}

function startBackgroundPull(): void {
  if (pullTimer) clearInterval(pullTimer);
  pullTimer = setInterval(() => {
    if (!signedIn || !navigator.onLine || inFlight) return;
    void syncNow().catch((e) => console.warn("[cloud] bg pull", e));
  }, CLOUD_CONFIG.PULL_INTERVAL_MS);
}

async function afterSignedIn(): Promise<void> {
  // Signed-in state is already persisted before this runs; treat userinfo +
  // sync as best-effort enrichment. A failure here must never revert to a
  // non-connected UI (that was the root cause of the "stuck not-connected"
  // bug after the redirect return).
  setStatus("syncing", "");
  try {
    await fetchUserInfo();
  } catch (e) {
    console.warn("[cloud] userinfo after sign-in failed (non-fatal)", e);
  } finally {
    emit();
  }
  try {
    await syncNow();
  } catch (e) {
    console.warn("[cloud] initial sync after sign-in failed (non-fatal)", e);
  }
  startBackgroundPull();
}

// ── Init ─────────────────────────────────────────────────────────────

export interface InitOptions {
  /** Whether init may leave the page to restore an expired token (the silent
   *  prompt=none redirect). False when the page holds something a navigation
   *  would destroy, such as a file the OS launched it with; the user then
   *  reconnects from the Drive strip instead. */
  allowRedirect?: boolean;
}

export async function init({ allowRedirect = true }: InitOptions = {}): Promise<void> {
  if (initialized) return;
  initialized = true;
  userInfo = restoreUser();
  if (!CLOUD_CONFIG.GOOGLE_CLIENT_ID) {
    setStatus("setupNeeded", "OAuth client ID not configured");
    return;
  }

  window.addEventListener("online", () => {
    if (!signedIn) return;
    void syncNow();
  });

  // 1. Returning from a redirect sign-in? Token is in the URL fragment.
  const cb = handleRedirectCallback();
  if (cb === "ok") {
    try {
      await afterSignedIn();
    } catch (e) {
      setStatus("error", e instanceof Error ? e.message : String(e));
    }
    return;
  }
  if (cb === "error") {
    // Silent (prompt=none) refresh failed → user must re-consent interactively
    setStatus("idle", "Re-sign-in required");
    // fall through to allow popup client init for browsers
  }

  // 2. Restore a cached token or try silent refresh if expired but signed-in.
  let tokenRestored = restoreToken();
  if (!tokenRestored && cb !== "error" && userInfo && navigator.onLine) {
    if (!allowRedirect) {
      setStatus("idle", "Re-sign-in required");
      return;
    }
    try {
      setStatus("syncing", "Restoring session...");
      await refreshToken(true); // allow redirect since it is page load
      tokenRestored = true;
    } catch (err) {
      console.warn("[cloud] silent token restore failed:", err);
      // If silent refresh failed and we are not in redirect mode, attempt silent redirect
      if (authMode !== "redirect") {
        console.log("[cloud] falling back to redirect silent auth...");
        await startRedirectAuth(true);
        return;
      }
    }
  }

  if (tokenRestored) {
    signedIn = true;
    setStatus("ok", "");
    try {
      await driveFetch("/about?fields=user", undefined, allowRedirect); // validates token
      await afterSignedIn();
      return;
    } catch {
      // refreshToken in redirect mode navigates away; only reaches here in
      // popup mode on failure.
      accessToken = null;
      tokenExpiresAt = 0;
      signedIn = false;
      localStorage.removeItem(LS_TOKEN);
      setStatus("idle", "Re-sign-in required");
    }
  } else {
    // Ensure status doesn't get stuck in 'syncing'.
    if (status === "syncing") {
      setStatus("idle", cb === "error" ? "Re-sign-in required" : "");
    }
  }

  // 3. Redirect-mode users with an expired token: attempt a silent re-auth
  //    (prompt=none) on a clean load. Loop-safe: only when there was no
  //    callback fragment this load, so an error return falls through to the
  //    sign-in button.
  if (
    allowRedirect &&
    cb === null &&
    !signedIn &&
    authMode === "redirect" &&
    userInfo &&
    navigator.onLine
  ) {
    await startRedirectAuth(true); // navigates away; returns via handleRedirectCallback
    return;
  }
}

export function isSignedIn(): boolean {
  return signedIn;
}

export const cloudSync = {
  init,
  signIn,
  signOut,
  syncNow,
  markDirty,
  isSignedIn,
  getStatus,
  subscribe,
};
