"""One implementation of "make this safe to serialise".

There were several near-identical copies of this before (`_sanitize` in
routers/stats/inferential.py, `_safe` in routers/ml.py, meta.py, charts.py and
services/survival_advanced_service.py), each handling a slightly different
subset -- one recursed but left numpy scalars alone, another converted numpy
scalars but only at the top level. Which one an endpoint happened to use
decided whether a NaN reached the client as `null` or as invalid JSON.

This is the union of them: recursive, and numpy-aware at every depth.

It matters more here than it did before. On the server, FastAPI's encoder was a
second net under this one; in the browser there is no such net -- whatever this
returns is handed to `json.dumps` and then straight across the worker boundary.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def sanitize(obj: Any) -> Any:
    """Return `obj` with numpy scalars unwrapped and non-finite floats as None.

    NaN and infinity are not representable in JSON. Emitting them produces a
    document that a strict parser rejects outright, so they become null: the
    reader sees a missing value, which is what a NaN in a result table means.
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set, frozenset)):
        return [sanitize(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]

    # numpy's bool_ is not a Python bool and is not an integer either, so it
    # has to be caught before the integer branch below.
    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        obj = float(obj)

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    return obj
