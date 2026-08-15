"""Translating an engine call into an HTTP one.

The engine knows nothing about HTTP: it raises EngineError carrying a status
hint. This is the one place that turns that into the HTTPException FastAPI
expects, so no router has to remember to.

Deliberately thin. Anything richer here -- retries, caching, coercion -- would
be behaviour the browser's caller does not have, and the two runtimes are only
equivalent for as long as everything meaningful happens inside the engine.
"""
from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException

from ustat_engine import EngineError


def adapt(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call `fn`, reporting an EngineError as the HTTP status it asked for."""
    try:
        return fn(*args, **kwargs)
    except EngineError as exc:
        raise HTTPException(status_code=exc.status_hint, detail=exc.message) from exc
