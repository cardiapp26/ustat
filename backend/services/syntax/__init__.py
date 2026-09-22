"""Endpoint templates for the syntax view, one module per family.

Each module exports ENDPOINTS: API path (session segments as {sid}) to a
handler taking the request body and returning {title, python, r}. The
public entry points stay in services.syntax_templates, which merges these
and adds the shared header; nothing here imports it back.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from . import association, causal, group_tests, meta, models, roc, survival, tables

Handler = Callable[[dict], Optional[dict]]

ENDPOINTS: Dict[str, Handler] = {}
for _module in (group_tests, tables, association, models, survival, roc, causal, meta):
    _overlap = ENDPOINTS.keys() & _module.ENDPOINTS.keys()
    if _overlap:  # two templates for one path would silently shadow each other
        raise RuntimeError(f"Duplicate syntax templates for {sorted(_overlap)}")
    ENDPOINTS.update(_module.ENDPOINTS)

__all__ = ["ENDPOINTS", "Handler"]
