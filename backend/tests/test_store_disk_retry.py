"""Disk-autosave retry semantics: a failed snapshot write must re-queue the
session in _dirty (so the next flush retries it) instead of silently dropping
it, and a successful flush must leave _dirty clean."""
import pandas as pd
import pytest

from services import store


@pytest.fixture()
def disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DISK_CACHE_ENABLED", True)
    monkeypatch.setattr(store, "SESSION_CACHE_DIR", str(tmp_path / "cache"))
    sid = "retry-test-session"
    with store._lock:
        store._store[sid] = {"df": pd.DataFrame({"a": [1, 2]}), "timestamp": 0.0}
        store._dirty.add(sid)
    yield sid
    with store._lock:
        store._purge_locked(sid)


def test_write_failure_requeues_session(disk_cache, monkeypatch, caplog):
    sid = disk_cache

    def boom(df, path):
        raise OSError("disk full")

    monkeypatch.setattr(store, "_atomic_write_pickle", boom)
    with caplog.at_level("WARNING", logger="services.store"):
        store._flush_dirty_to_disk()

    assert sid in store._dirty, "failed session must be re-queued for retry"
    assert any("re-queued" in r.message for r in caplog.records)


def test_unwritable_cache_dir_requeues_all(disk_cache, monkeypatch, caplog):
    sid = disk_cache

    def no_dir(path, exist_ok=False):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(store.os, "makedirs", no_dir)
    with caplog.at_level("WARNING", logger="services.store"):
        store._flush_dirty_to_disk()

    assert sid in store._dirty
    assert any("not writable" in r.message for r in caplog.records)


def test_successful_flush_clears_dirty(disk_cache):
    sid = disk_cache
    store._flush_dirty_to_disk()
    assert sid not in store._dirty
    df_path, meta_path = store._cache_paths(sid)
    import os
    assert os.path.exists(df_path)
    assert os.path.exists(meta_path)


def test_purged_session_not_resurrected(disk_cache):
    sid = disk_cache
    with store._lock:
        store._store.pop(sid)
        store._dirty.discard(sid)
    store._requeue_dirty([sid])
    assert sid not in store._dirty
