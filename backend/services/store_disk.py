"""Disk persistence for the in-memory session store (opt-in, OFF by default).

Split out of services/store.py: this module owns everything that touches the
filesystem — snapshot paths, atomic writes, the autosave flush loop, and
startup rehydration. The session maps and their lock stay in store.py; this
module reads them through the module object (``store._store`` etc.) at call
time, and store.py in turn imports this module only lazily inside functions,
which is what keeps the two-way dependency loadable in either order.

Privacy default: uploaded data lives ONLY in server RAM and is never written
to disk, matching the guarantee in README/privacy policy. Snapshotting each
session's DataFrame so in-progress edits survive a backend restart/redeploy is
therefore OFF unless the operator explicitly opts in with SESSION_DISK_CACHE=1.
When off: no session data ever touches the disk. Resume-after-restart is still
covered client-side by the browser's IndexedDB autosave, so the default costs
little resilience.

⚠️ If you enable it, the pickle files are UNENCRYPTED clinical data at rest on
the host volume — add disk encryption, restrictive file permissions, and a
guaranteed-delete policy, and update your privacy disclosure accordingly.
"""
import json
import logging
import os
import threading
import time
from copy import deepcopy

import pandas as pd

# The config constants (DISK_CACHE_ENABLED, SESSION_CACHE_DIR,
# AUTOSAVE_INTERVAL_SECONDS) stay in store.py with the rest of the session
# state and are read through the module object at call time, so tests that
# monkeypatch store.SESSION_CACHE_DIR keep working.
from services import store

_logger = logging.getLogger(__name__)


def _cache_paths(session_id: str) -> tuple:
    base = os.path.join(store.SESSION_CACHE_DIR, session_id)
    return base + ".pkl", base + ".meta.json"


def delete_disk_snapshot(session_id: str) -> None:
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


def _requeue_dirty(session_ids) -> None:
    """Put sessions whose snapshot write failed back on the dirty set so the
    next autosave tick retries them, instead of silently dropping the retry.
    Sessions purged in the meantime are skipped — _purge_locked already removed
    their disk files, so resurrecting them in _dirty would be a stale entry."""
    with store._lock:
        for sid in session_ids:
            if sid in store._store:
                store._dirty.add(sid)


def flush_dirty_to_disk() -> None:
    """Snapshot every session touched since the last flush. Called from the
    autosave thread — overwrites each session's single file pair in place so
    the cache directory never accumulates history."""
    if not store.DISK_CACHE_ENABLED:
        return  # Opt-in only — never write session data to disk by default.
    with store._lock:
        pending = list(store._dirty)
        store._dirty.clear()
        snapshot = {}
        for sid in pending:
            entry = store._store.get(sid)
            if entry is not None:
                snapshot[sid] = {
                    "df": entry["df"].copy(),
                    "timestamp": entry["timestamp"],
                    "kinds": deepcopy(store._kinds.get(sid, {})),
                    "decimals": deepcopy(store._decimals.get(sid, {})),
                    "filename": store._filenames.get(sid),
                    "metadata": deepcopy(store._metadata.get(sid, {})),
                    "filters": deepcopy(store._filters.get(sid, [])),
                }

    if not snapshot:
        return
    try:
        os.makedirs(store.SESSION_CACHE_DIR, exist_ok=True)
    except OSError as exc:
        _requeue_dirty(snapshot.keys())
        _logger.warning(
            "Session autosave: cache dir %s not writable (%s); "
            "%d session(s) re-queued for the next flush.",
            store.SESSION_CACHE_DIR, exc, len(snapshot),
        )
        return

    failed: list = []
    for sid, state in snapshot.items():
        df_path, meta_path = _cache_paths(sid)
        try:
            _atomic_write_pickle(state["df"], df_path)
            meta = {
                "timestamp": state["timestamp"],
                "kinds": state["kinds"],
                "decimals": state["decimals"],
                "filename": state["filename"],
                "metadata": state["metadata"],
                "filters": state["filters"],
            }
            with open(meta_path + ".tmp", "w") as f:
                json.dump(meta, f)
            os.replace(meta_path + ".tmp", meta_path)
        except OSError as exc:
            failed.append(sid)
            _logger.warning(
                "Session autosave: snapshot write failed for session %s (%s); "
                "re-queued for the next flush.", sid, exc,
            )

    if failed:
        _requeue_dirty(failed)


def _autosave_worker() -> None:
    while True:
        time.sleep(store.AUTOSAVE_INTERVAL_SECONDS)
        try:
            flush_dirty_to_disk()
        except Exception:
            pass  # Autosave must never crash the request-handling thread.


def load_persisted_sessions() -> None:
    """Rehydrate the in-memory store from disk snapshots on backend startup.
    Skips (and deletes) anything past the normal session TTL. Called once
    from main.py's startup hook. No-op unless disk cache is opted in."""
    if not store.DISK_CACHE_ENABLED:
        return
    if not os.path.isdir(store.SESSION_CACHE_DIR):
        return
    now = time.time()
    try:
        names = [f[:-4] for f in os.listdir(store.SESSION_CACHE_DIR) if f.endswith(".pkl")]
    except OSError:
        return

    for sid in names:
        df_path, meta_path = _cache_paths(sid)
        try:
            mtime = os.path.getmtime(df_path)
        except OSError:
            continue
        if now - mtime > store.SESSION_TTL_SECONDS:
            delete_disk_snapshot(sid)
            continue
        try:
            # nosec B301 - this pickle is not untrusted input. It is written by
            # _atomic_write_pickle above, into a cache directory this process
            # owns, and read back only when the session id matches a file this
            # process wrote. Anyone able to plant a file there can already run
            # code as this user. The cache is off by default (SESSION_DISK_CACHE=0).
            df = pd.read_pickle(df_path)  # nosec B301
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
        except Exception:
            delete_disk_snapshot(sid)
            continue

        with store._lock:
            store._store[sid] = {"df": df, "timestamp": meta.get("timestamp", mtime)}
            if meta.get("kinds"):
                store._kinds[sid] = meta["kinds"]
            if meta.get("decimals"):
                store._decimals[sid] = meta["decimals"]
            if meta.get("filename"):
                store._filenames[sid] = meta["filename"]
            if meta.get("metadata"):
                store._metadata[sid] = meta["metadata"]
            if meta.get("filters"):
                store._filters[sid] = meta["filters"]


# Only spin up the disk-flush thread when the operator has opted in. Off by
# default → the thread never runs and no session data is written to disk.
if store.DISK_CACHE_ENABLED:
    _autosave_thread = threading.Thread(target=_autosave_worker, daemon=True)
    _autosave_thread.start()
