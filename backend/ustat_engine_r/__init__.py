"""The R engine's identity module, in Python.

`backend/ustat_engine_r/` is a tree of .R sources -- there is no Python to
import and run here. What this package provides is the *server's* view of that
tree: its version, the sha256 over its sources, and the analyses it declares,
so `GET /api/engine/r/identity` can tell a browser what bundle it should be
holding before that browser computes anything from patient data.

Everything lives in `fingerprint`; this module only re-exports it, so
`from ustat_engine_r import identity` reads the same as it does for the Python
engine.
"""
from __future__ import annotations

from .fingerprint import (
    __version__,
    analyses,
    identity,
    source_fingerprint,
)

__all__ = ["__version__", "analyses", "identity", "source_fingerprint"]
