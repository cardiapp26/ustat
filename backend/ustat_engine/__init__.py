"""Runtime-agnostic statistics core (distributed as the `ustat_engine` wheel).

Everything in this package must import and run identically under CPython on the
server and under CPython-compiled-to-WebAssembly in the browser. That rules out
more than it sounds like:

  - no `fastapi`, `starlette` or `pydantic` -- the HTTP layer is a caller, not
    a dependency, and pydantic-core is a Rust extension besides;
  - no `services.store` -- it starts a background thread at import time;
  - no `async def`, `threading` or `subprocess` -- the browser build is
    single-threaded, so an offload here is a crash there. Offloading is the
    server's problem and belongs in its adapter;
  - no filesystem paths outside what the caller passes in.

tests/test_engine_isolation.py enforces this by reading the source, because a
convention that is only written down gets broken by the first person who has
not read it.
"""
from __future__ import annotations

from .errors import EngineError
from .fingerprint import __version__, identity, source_fingerprint
from .jsonsafe import sanitize
from .registry import ANALYSES, get, register, run
from .spec import AnalysisSpec

# Importing the package registers every analysis it can run. Leaving that to
# whichever caller happens to import a submodule first would make the answer to
# "can this run in the browser?" depend on import order -- the registry would
# look complete on the server, where a router imported everything, and short in
# the browser, where nothing had.
from . import meta as _meta  # noqa: E402,F401  (imported for its registrations)
from .stats import power as _power  # noqa: E402,F401
from .stats import ttest as _ttest  # noqa: E402,F401

__all__ = [
    "__version__",
    "ANALYSES",
    "AnalysisSpec",
    "EngineError",
    "get",
    "identity",
    "register",
    "run",
    "sanitize",
    "source_fingerprint",
]
