"""The stated session TTL has to hold on an idle server.

_cleanup_old_sessions() used to be reachable only from save(), so a session
whose TTL had elapsed stayed in RAM until some *other* user uploaded a file.
On a server with no traffic it never expired at all, while the public Privacy
page promised it was cleared 30 minutes after the user's last activity.

These tests pin the two halves of that promise: expiry happens without any
request touching the store, and an open browser tab (which polls once a
minute) keeps its session alive.
"""
import threading
import time

import pandas as pd
import pytest

from services import store


@pytest.fixture(autouse=True)
def _clean_store():
    """Each test starts from an empty store and restores the real TTL after."""
    original_ttl = store.SESSION_TTL_SECONDS
    with store._lock:
        store._store.clear()
    yield
    store.SESSION_TTL_SECONDS = original_ttl
    with store._lock:
        store._store.clear()


def _put(session_id: str, age_seconds: float = 0.0) -> None:
    """Insert a session directly, optionally backdating its timestamp."""
    with store._lock:
        store._store[session_id] = {
            "df": pd.DataFrame({"a": [1, 2, 3]}),
            "timestamp": time.time() - age_seconds,
        }


def test_reaper_thread_is_running_by_default():
    """The reaper must run in the default configuration, disk cache off."""
    assert store.DISK_CACHE_ENABLED is False, "default config should not touch disk"
    names = [t.name for t in threading.enumerate()]
    assert "session-reaper" in names
    assert store._reaper_thread.daemon, "must not block interpreter shutdown"


def test_reaper_interval_is_shorter_than_the_browser_heartbeat():
    """A live tab refreshes its session every 60s; reaping must not race that."""
    assert store.REAPER_INTERVAL_SECONDS < 60
    assert store.REAPER_INTERVAL_SECONDS < store.SESSION_TTL_SECONDS


def test_expired_session_is_dropped_without_any_request():
    """The whole point: no save(), no get(), no traffic — it still goes."""
    store.SESSION_TTL_SECONDS = 60
    _put("stale", age_seconds=120)
    _put("fresh", age_seconds=5)

    store._cleanup_old_sessions(force=True)

    assert "stale" not in store._store
    assert "fresh" in store._store


def test_force_bypasses_the_once_a_minute_throttle():
    """A request-path cleanup moments earlier must not skip the reaper's turn."""
    store.SESSION_TTL_SECONDS = 60
    store._last_cleanup = time.time()  # as if save() just ran
    _put("stale", age_seconds=120)

    store._cleanup_old_sessions()  # throttled — nothing happens
    assert "stale" in store._store

    store._cleanup_old_sessions(force=True)
    assert "stale" not in store._store


def test_reading_a_session_resets_its_ttl():
    """An open tab polls save_session, which calls get() — that must keep it."""
    store.SESSION_TTL_SECONDS = 60
    _put("active", age_seconds=59)

    assert store.get("active") is not None  # the heartbeat
    store._cleanup_old_sessions(force=True)

    assert "active" in store._store


def test_ttl_is_configurable_from_the_environment(monkeypatch):
    """The Privacy page calls the TTL configurable; it has to actually be."""
    import importlib

    monkeypatch.setenv("SESSION_TTL_SECONDS", "900")
    reloaded = importlib.reload(store)
    try:
        assert reloaded.SESSION_TTL_SECONDS == 900
    finally:
        monkeypatch.delenv("SESSION_TTL_SECONDS", raising=False)
        importlib.reload(store)
