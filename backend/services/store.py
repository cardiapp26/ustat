"""In-memory dataframe store keyed by session id with automatic cleanup.

Also periodically snapshots dirty sessions to disk (SESSION_CACHE_DIR) so a
container restart/redeploy doesn't silently wipe a user's in-progress edits —
without that, every backend restart lost all unsaved work. Snapshots are
overwritten in place (one file pair per session, not versioned) so the cache
never grows unbounded; the same TTL/MAX_SESSIONS eviction that prunes memory
also deletes the matching disk files.
"""
import json
import os
import tempfile
import threading
import pandas as pd
from typing import Dict, List, Optional
import time
from threading import Lock
from fastapi import HTTPException

# Per-dataset size ceiling (rows × columns). Guards the in-memory store against
# a single oversized frame — from upload or from a runaway compute/merge.
# ~20M cells ≈ 200k rows × 100 cols. Override via env.
MAX_SESSION_CELLS = int(os.environ.get("MAX_SESSION_CELLS", str(20_000_000)))

_store: Dict[str, dict] = {}  # {session_id: {"df": DataFrame, "timestamp": float}}
_filters: Dict[str, List[dict]] = {}
_audit: Dict[str, list] = {}
_metadata: Dict[str, dict] = {}
_kinds: Dict[str, Dict[str, str]] = {}  # {session_id: {col: "numeric"|"categorical"|...}}
_decimals: Dict[str, Dict[str, int]] = {}  # {session_id: {col: decimal_places}} for cell-format overrides
_filenames: Dict[str, str] = {}  # {session_id: user-chosen display name}
_undo: Dict[str, list] = {}   # {session_id: [DataFrame snapshots]}
_redo: Dict[str, list] = {}
_lock = Lock()
MAX_UNDO = 30
VALID_FILTER_OPERATORS = {"eq", "ne", "gt", "lt", "gte", "lte", "missing", "not_missing", "contains"}

# Every per-session map, so cleanup/delete can drop a session completely
# (a partial pop leaks the user's kinds/decimals/filename/filters after TTL).
_SESSION_MAPS: tuple = (_store, _filters, _audit, _metadata, _kinds, _decimals, _filenames, _undo, _redo)


def _purge_locked(session_id: str) -> None:
    """Remove a session from EVERY per-session map. Caller must hold _lock."""
    for m in _SESSION_MAPS:
        m.pop(session_id, None)
    _dirty.discard(session_id)
    _delete_disk_snapshot(session_id)


# ── Disk autosave (opt-in — OFF by default) ────────────────────────────────
# Privacy default: uploaded data lives ONLY in server RAM and is never written
# to disk, matching the guarantee in README/privacy policy. This snapshot-to-
# disk feature (which pickles each session's DataFrame so in-progress edits
# survive a backend restart/redeploy) is therefore OFF unless the operator
# explicitly opts in with SESSION_DISK_CACHE=1. When off: no session data ever
# touches the disk. Resume-after-restart is still covered client-side by the
# browser's IndexedDB autosave, so the default costs little resilience.
#
# ⚠️ If you enable it, the pickle files are UNENCRYPTED clinical data at rest
# on the host volume — add disk encryption, restrictive file permissions, and a
# guaranteed-delete policy, and update your privacy disclosure accordingly.
DISK_CACHE_ENABLED = os.environ.get("SESSION_DISK_CACHE", "0") == "1"
SESSION_CACHE_DIR = os.environ.get(
    "SESSION_CACHE_DIR",
    "/app/backend/session_cache" if os.path.isdir("/app/backend") else
    os.path.join(tempfile.gettempdir(), "ustat_session_cache"),
)
AUTOSAVE_INTERVAL_SECONDS = int(os.environ.get("AUTOSAVE_INTERVAL_SECONDS", "20"))
_dirty: set = set()  # session_ids changed since the last disk flush


def _mark_dirty(session_id: str) -> None:
    """Flag a session as changed so the next disk-flush writes its snapshot.

    Meta-only mutations (filename, kinds, decimals, metadata, filter) used to
    skip this, so a backend restart reverted them to the last DataFrame-mutating
    save — e.g. a header rename was lost and the welcome screen kept listing the
    session under the old (upload) filename. Any function that touches a
    persisted meta map must mark dirty. Lock-guarded so it can't race the
    flush worker's read-and-clear under `_lock`."""
    with _lock:
        _dirty.add(session_id)


def _cache_paths(session_id: str) -> tuple:
    base = os.path.join(SESSION_CACHE_DIR, session_id)
    return base + ".pkl", base + ".meta.json"


def _delete_disk_snapshot(session_id: str) -> None:
    df_path, meta_path = _cache_paths(session_id)
    for p in (df_path, meta_path):
        try:
            os.remove(p)
        except OSError:
            pass


def _atomic_write_pickle(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_pickle(tmp)
    os.replace(tmp, path)


def _flush_dirty_to_disk() -> None:
    """Snapshot every session touched since the last flush. Called from the
    autosave thread — overwrites each session's single file pair in place so
    the cache directory never accumulates history."""
    if not DISK_CACHE_ENABLED:
        return  # Opt-in only — never write session data to disk by default.
    with _lock:
        pending = list(_dirty)
        _dirty.clear()
        snapshot = {}
        for sid in pending:
            entry = _store.get(sid)
            if entry is not None:
                snapshot[sid] = (entry["df"], entry["timestamp"])

    if not snapshot:
        return
    try:
        os.makedirs(SESSION_CACHE_DIR, exist_ok=True)
    except OSError:
        return  # No writable/mounted cache dir — degrade to memory-only silently.

    for sid, (df, ts) in snapshot.items():
        df_path, meta_path = _cache_paths(sid)
        try:
            _atomic_write_pickle(df, df_path)
            meta = {
                "timestamp": ts,
                "kinds": _kinds.get(sid, {}),
                "decimals": _decimals.get(sid, {}),
                "filename": _filenames.get(sid),
                "metadata": _metadata.get(sid, {}),
            }
            with open(meta_path + ".tmp", "w") as f:
                json.dump(meta, f)
            os.replace(meta_path + ".tmp", meta_path)
        except OSError:
            continue  # Best-effort — a write failure just skips this session's snapshot.


def _autosave_worker() -> None:
    while True:
        time.sleep(AUTOSAVE_INTERVAL_SECONDS)
        try:
            _flush_dirty_to_disk()
        except Exception:
            pass  # Autosave must never crash the request-handling thread.


def _reaper_worker() -> None:
    """Expire sessions on a timer, independent of request traffic.

    _cleanup_old_sessions() is otherwise only reached from save(), so on a
    server nobody is currently uploading to, an expired session stayed in RAM
    indefinitely — the Privacy page promises it is cleared 30 minutes after
    the user's last activity, and without this thread that promise held only
    if some other user happened to upload a file.
    """
    while True:
        time.sleep(REAPER_INTERVAL_SECONDS)
        try:
            _cleanup_old_sessions(force=True)
        except Exception:
            pass  # Reaping must never take down the process.


# Only spin up the disk-flush thread when the operator has opted in. Off by
# default → the thread never runs and no session data is written to disk.
if DISK_CACHE_ENABLED:
    _autosave_thread = threading.Thread(target=_autosave_worker, daemon=True)
    _autosave_thread.start()

# The reaper thread is started at the bottom of this module, once
# _cleanup_old_sessions and the TTL constants it reads are defined.


def load_persisted_sessions() -> None:
    """Rehydrate the in-memory store from disk snapshots on backend startup.
    Skips (and deletes) anything past the normal session TTL. Called once
    from main.py's startup hook. No-op unless disk cache is opted in."""
    if not DISK_CACHE_ENABLED:
        return
    if not os.path.isdir(SESSION_CACHE_DIR):
        return
    now = time.time()
    try:
        names = [f[:-4] for f in os.listdir(SESSION_CACHE_DIR) if f.endswith(".pkl")]
    except OSError:
        return

    for sid in names:
        df_path, meta_path = _cache_paths(sid)
        try:
            mtime = os.path.getmtime(df_path)
        except OSError:
            continue
        if now - mtime > SESSION_TTL_SECONDS:
            _delete_disk_snapshot(sid)
            continue
        try:
            df = pd.read_pickle(df_path)
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
        except Exception:
            _delete_disk_snapshot(sid)
            continue

        with _lock:
            _store[sid] = {"df": df, "timestamp": meta.get("timestamp", mtime)}
            if meta.get("kinds"):
                _kinds[sid] = meta["kinds"]
            if meta.get("decimals"):
                _decimals[sid] = meta["decimals"]
            if meta.get("filename"):
                _filenames[sid] = meta["filename"]
            if meta.get("metadata"):
                _metadata[sid] = meta["metadata"]


def purge_session(session_id: str) -> None:
    """Public: fully remove a session and all its metadata."""
    with _lock:
        _purge_locked(session_id)

# Session configuration. The TTL is a privacy guarantee stated on the public
# Privacy and Security pages, so it is read from the environment rather than
# being a constant those pages describe as "configurable" without it being so.
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", str(30 * 60)))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "20"))  # concurrent sessions
_last_cleanup = time.time()

# How often the reaper thread below wakes. Must stay well under the TTL, and
# under the 60s interval at which an open browser tab refreshes its session,
# or a live session could be reaped between two heartbeats.
REAPER_INTERVAL_SECONDS = 30


def _cleanup_old_sessions(force: bool = False) -> None:
    """Remove sessions older than TTL, keeping only the most recent MAX_SESSIONS.

    force=True skips the once-a-minute throttle. The throttle exists to keep
    this cheap on the request path (save() calls it); the reaper thread has no
    such concern and must not be silently skipped because a request happened
    to run the cleanup moments earlier.
    """
    global _last_cleanup
    now = time.time()
    if not force and now - _last_cleanup < 60:  # Cleanup every 60 seconds max
        return

    _last_cleanup = now
    with _lock:
        # Remove expired sessions
        expired = [sid for sid, entry in _store.items() if now - entry["timestamp"] > SESSION_TTL_SECONDS]
        for sid in expired:
            _purge_locked(sid)

        # If still over limit, remove oldest sessions
        if len(_store) > MAX_SESSIONS:
            sorted_sids = sorted(_store.items(), key=lambda x: x[1]["timestamp"])
            to_remove = len(_store) - MAX_SESSIONS
            for sid, _ in sorted_sids[:to_remove]:
                _purge_locked(sid)
                _metadata.pop(sid, None)


def save(session_id: str, df: pd.DataFrame, track_undo: bool = True) -> None:
    """Save dataframe with timestamp for TTL tracking.
    If track_undo=True, pushes the previous state onto the undo stack.
    """
    _cleanup_old_sessions()
    n_cells = int(df.shape[0]) * int(df.shape[1])
    if n_cells > MAX_SESSION_CELLS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Dataset too large: {df.shape[0]:,} rows × {df.shape[1]:,} cols "
                f"= {n_cells:,} cells (limit {MAX_SESSION_CELLS:,})."
            ),
        )
    with _lock:
        # Push current state to undo stack before overwriting
        if track_undo and session_id in _store:
            old_df = _store[session_id]["df"]
            _undo.setdefault(session_id, []).append(old_df.copy())
            if len(_undo[session_id]) > MAX_UNDO:
                _undo[session_id] = _undo[session_id][-MAX_UNDO:]
            # Clear redo stack on new action
            _redo.pop(session_id, None)
        _store[session_id] = {"df": df, "timestamp": time.time()}
        _dirty.add(session_id)
    log_action(session_id, "data_updated")


def delete_row(session_id: str, row_index: int) -> bool:
    """Delete a row by its 0-based POSITION (matches the frontend's positional
    row index) and save to trigger undo tracking. Resets the index afterwards so
    the stored frame keeps a contiguous RangeIndex — otherwise a later
    positional ``df.at[pos]`` edit would target a missing label and silently
    append a phantom row."""
    with _lock:
        entry = _store.get(session_id)
        if entry is None:
            return False
        df = entry["df"]

        # row_index is a 0-based position, not a pandas label.
        if row_index < 0 or row_index >= len(df):
            return False

    # Drop outside the lock then save through the standard pipeline (undo).
    new_df = df.drop(df.index[row_index]).reset_index(drop=True)
    save(session_id, new_df)
    log_action(session_id, "row_deleted", {"row_index": row_index})
    return True



def get(session_id: str) -> Optional[pd.DataFrame]:
    """Get dataframe and update access timestamp."""
    with _lock:
        entry = _store.get(session_id)
        if entry is None:
            return None
        # Update timestamp on access to keep active sessions alive
        entry["timestamp"] = time.time()
        return entry["df"]


def save_filter(session_id: str, conditions: List[dict]) -> None:
    _filters[session_id] = conditions


def get_filter(session_id: str) -> List[dict]:
    return _filters.get(session_id, [])


def clear_filter(session_id: str) -> None:
    _filters.pop(session_id, None)


def validate_conditions(df: pd.DataFrame, conditions: List[dict]) -> None:
    for i, cond in enumerate(conditions or [], start=1):
        col = cond.get("column", "")
        if col not in df.columns:
            raise HTTPException(status_code=404, detail=f"Condition {i}: column '{col}' not found")
        op = cond.get("operator", "eq")
        if op not in VALID_FILTER_OPERATORS:
            allowed = ", ".join(sorted(VALID_FILTER_OPERATORS))
            raise HTTPException(
                status_code=422,
                detail=f"Condition {i}: unsupported operator '{op}'. Use one of: {allowed}.",
            )


def _apply_conditions(df: pd.DataFrame, conditions: List[dict]) -> pd.DataFrame:
    if not conditions:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    for i, cond in enumerate(conditions):
        col = cond.get("column", "")
        if col not in df.columns:
            continue
        op = cond.get("operator", "eq")
        val = cond.get("value", "")
        join = cond.get("join", "AND")

        if op == "missing":
            cond_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        elif op == "not_missing":
            cond_mask = df[col].notna() & (df[col].astype(str).str.strip() != "")
        elif op == "contains":
            cond_mask = df[col].astype(str).str.contains(str(val), case=False, na=False)
        else:
            # Try numeric comparison first, fall back to string
            try:
                num_val = float(val)
                s = pd.to_numeric(df[col], errors="coerce")
                if op == "eq":
                    cond_mask = s == num_val
                elif op == "ne":
                    cond_mask = s != num_val
                elif op == "gt":
                    cond_mask = s > num_val
                elif op == "lt":
                    cond_mask = s < num_val
                elif op == "gte":
                    cond_mask = s >= num_val
                elif op == "lte":
                    cond_mask = s <= num_val
                else:
                    cond_mask = pd.Series([True] * len(df), index=df.index)
            except (ValueError, TypeError):
                s = df[col].astype(str)
                if op == "eq":
                    cond_mask = s == str(val)
                elif op == "ne":
                    cond_mask = s != str(val)
                else:
                    cond_mask = pd.Series([True] * len(df), index=df.index)

        if i == 0 or join == "AND":
            mask = mask & cond_mask
        else:
            mask = mask | cond_mask

    return df[mask]


def get_filtered(session_id: str) -> Optional[pd.DataFrame]:
    """Return the session dataframe with any active case filter applied."""
    with _lock:
        entry = _store.get(session_id)
        if entry is None:
            return None
        df = entry["df"]
        # Update access timestamp
        entry["timestamp"] = time.time()
        conditions = _filters.get(session_id, [])
    return _apply_conditions(df, conditions)


def fill_values_by_index(session_id: str, column: str, values: Dict[int, object]) -> bool:
    """Fill selected dataframe labels in one column and persist through save().

    Used by workflows that analyze the active filtered view but need mutation to
    land in the backing dataset. Keys are dataframe index labels preserved by
    get_filtered().
    """
    with _lock:
        entry = _store.get(session_id)
        if entry is None:
            return False
        df = entry["df"]
        if column not in df.columns:
            return False
    updated = df.copy()
    for idx, value in values.items():
        if idx in updated.index:
            updated.at[idx, column] = value
    save(session_id, updated)
    return True


def delete(session_id: str) -> None:
    """Fully remove a session and every per-session map (kinds/decimals/filename
    included — they used to leak here)."""
    with _lock:
        _purge_locked(session_id)


def list_sessions() -> list[str]:
    return list(_store.keys())


def undo(session_id: str) -> Optional[pd.DataFrame]:
    """Pop the last undo snapshot and restore it. Returns the restored DataFrame or None."""
    with _lock:
        stack = _undo.get(session_id, [])
        if not stack:
            return None
        prev_df = stack.pop()
        # Push current state to redo
        if session_id in _store:
            _redo.setdefault(session_id, []).append(_store[session_id]["df"].copy())
            if len(_redo[session_id]) > MAX_UNDO:
                _redo[session_id] = _redo[session_id][-MAX_UNDO:]
        _store[session_id] = {"df": prev_df, "timestamp": time.time()}
    return prev_df


def redo(session_id: str) -> Optional[pd.DataFrame]:
    """Pop the last redo snapshot and restore it. Returns the restored DataFrame or None."""
    with _lock:
        stack = _redo.get(session_id, [])
        if not stack:
            return None
        next_df = stack.pop()
        # Push current state to undo
        if session_id in _store:
            _undo.setdefault(session_id, []).append(_store[session_id]["df"].copy())
            if len(_undo[session_id]) > MAX_UNDO:
                _undo[session_id] = _undo[session_id][-MAX_UNDO:]
        _store[session_id] = {"df": next_df, "timestamp": time.time()}
    return next_df


def undo_depth(session_id: str) -> int:
    return len(_undo.get(session_id, []))


def redo_depth(session_id: str) -> int:
    return len(_redo.get(session_id, []))


# ── Audit trail ───────────────────────────────────────────────────────────────

def log_action(session_id: str, action: str, params: Optional[dict] = None) -> None:
    """Append an audit entry for the given session."""
    entry = {"action": action, "params": params, "timestamp": time.time()}
    _audit.setdefault(session_id, []).append(entry)


def get_audit(session_id: str) -> list:
    """Return the audit trail for a session."""
    return _audit.get(session_id, [])


# ── Column metadata ──────────────────────────────────────────────────────────

def save_metadata(session_id: str, meta: dict) -> None:
    """Merge column-level metadata for a session.

    Merges per-column so a partial update (e.g. just ``analysis_excluded`` for
    one column) never wipes other columns' metadata or other fields of the same
    column (e.g. a previously-saved ``value_labels`` map). A full map still
    works — every supplied key overwrites its prior value.
    """
    cur = dict(_metadata.get(session_id, {}))
    for col, m in (meta or {}).items():
        if isinstance(m, dict):
            prev = dict(cur.get(col, {}) or {})
            prev.update(m)
            cur[col] = prev
        else:
            cur[col] = m
    _metadata[session_id] = cur
    _mark_dirty(session_id)


def get_metadata(session_id: str) -> dict:
    """Return column-level metadata for a session."""
    return _metadata.get(session_id, {})


# ── Column kind overrides ────────────────────────────────────────────────────
# User-driven `numeric` ↔ `categorical` (etc.) flips made through the data-tab
# badge / dictionary. Persists alongside the dataframe so save_session can
# round-trip them — otherwise the next load re-runs auto-detection and the
# user's classification choices are silently discarded.

def save_kind_overrides(session_id: str, overrides: Dict[str, str]) -> None:
    """Merge a dict of {column: kind} into the per-session override map."""
    current = _kinds.get(session_id, {})
    current.update({k: v for k, v in overrides.items() if v})
    _kinds[session_id] = current
    _mark_dirty(session_id)


def set_kind_overrides(session_id: str, overrides: Dict[str, str]) -> None:
    """Replace the override map wholesale (used on load_session restore)."""
    _kinds[session_id] = dict(overrides or {})
    _mark_dirty(session_id)


def get_kind_overrides(session_id: str) -> Dict[str, str]:
    return _kinds.get(session_id, {})


def clear_kind_override(session_id: str, column: str) -> None:
    if session_id in _kinds:
        _kinds[session_id].pop(column, None)


# ── Column decimal-places overrides ──────────────────────────────────────────
# Per-cell display precision the user picks via the right-click context menu
# in the data table. Persisted server-side so save_session round-trips the
# choice — the previous frontend-only implementation forgot decimals on every
# JSON export.

def save_decimals(session_id: str, decimals: Dict[str, int]) -> None:
    """Replace the decimal-places map wholesale."""
    _decimals[session_id] = {k: int(v) for k, v in (decimals or {}).items()}
    _mark_dirty(session_id)


def get_decimals(session_id: str) -> Dict[str, int]:
    return _decimals.get(session_id, {})


def set_decimal(session_id: str, column: str, decimals: int) -> None:
    """Set the decimal-places override for a single column."""
    cur = _decimals.get(session_id, {})
    cur[column] = int(decimals)
    _decimals[session_id] = cur
    _mark_dirty(session_id)


def clear_decimal(session_id: str, column: str) -> None:
    if session_id in _decimals:
        _decimals[session_id].pop(column, None)
        _mark_dirty(session_id)


def rename_decimal_key(session_id: str, old: str, new: str) -> None:
    """Move the decimal entry to a new column name (called on rename)."""
    if session_id in _decimals and old in _decimals[session_id]:
        _decimals[session_id][new] = _decimals[session_id].pop(old)
        _mark_dirty(session_id)


def rename_column_key(session_id: str, old: str, new: str) -> None:
    """Move ALL per-column state to a new column name on rename so flags such
    as value_labels, analysis_excluded, display_name and the kind override are
    not orphaned under the old name."""
    if old == new:
        return
    meta = _metadata.get(session_id)
    if meta and old in meta:
        meta[new] = meta.pop(old)
    kinds = _kinds.get(session_id)
    if kinds and old in kinds:
        kinds[new] = kinds.pop(old)
    if session_id in _decimals and old in _decimals[session_id]:
        _decimals[session_id][new] = _decimals[session_id].pop(old)
    _mark_dirty(session_id)


# ── Session display name (user-facing rename) ────────────────────────────────

def set_filename(session_id: str, name: str) -> None:
    """Store a user-chosen display name for the session.

    Independent of the original upload filename. Used by save_session so
    the round-tripped JSON carries the renamed value, and by the
    /sessions/{sid}/filename endpoint so the React store can sync on
    rehydrate.
    """
    if name:
        _filenames[session_id] = str(name)
        _mark_dirty(session_id)


def get_filename(session_id: str) -> Optional[str]:
    return _filenames.get(session_id)


def clear_filename(session_id: str) -> None:
    _filenames.pop(session_id, None)


def get_df(session_id: str) -> "pd.DataFrame":
    """
    Convenience helper: return the (filtered) dataframe for a session.
    Raises 404 if the session does not exist.
    This used to live in routers/models.py as _get_df.
    """
    df = get_filtered(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return df


# Started last, so _cleanup_old_sessions and the TTL constants it reads are
# already bound by the time the thread's first tick fires.
_reaper_thread = threading.Thread(target=_reaper_worker, daemon=True, name="session-reaper")
_reaper_thread.start()
