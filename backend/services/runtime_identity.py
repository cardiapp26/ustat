"""
What actually computed a number, in a form a result can carry.

WHY THIS IS NOT A COSMETIC DETAIL. The session header says "R-based
statistics" because the user picked R at the welcome gate. That is a statement
about a *choice*, and it was being read as a statement about a *result* -- but
only two analyses are routed through the local engines at all, so a regression,
a Cox model or a random forest was computed by this server's Python while the
header said R. A reader reproducing the work in R would then fail to match
numbers that were never R's.

Provenance therefore has to be a property of each result, and it has to be
reported by whatever produced it rather than inferred by whoever displays it.
This module is the server's half: it says which engine ran (Python, here), on
what runtime, and at exactly which library versions -- because "computed with
Python" is not reproducible and "scipy 1.14.1" is.

The values travel as response headers rather than as fields in every response
body. Response contracts in this codebase are load-bearing (panels destructure
backend fields without optional chaining), and threading a provenance object
through two hundred endpoints would be two hundred chances to forget one -- the
same failure mode this is meant to fix.
"""

from __future__ import annotations

import platform
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, Optional

#: The libraries that actually produce numbers here. Every one of them is
#: pinned in requirements.txt, and a result that does not name their versions
#: cannot be reproduced from the paper it ends up in.
REPORTED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "statsmodels",
    "scikit-learn",
    "lifelines",
    "patsy",
)

#: Header names, in one place so the client and the tests agree with the server.
HEADER_RUNTIME = "X-uStat-Runtime"
HEADER_ENGINE = "X-uStat-Engine"
HEADER_ENGINE_VERSION = "X-uStat-Engine-Version"
HEADER_ENGINE_FINGERPRINT = "X-uStat-Engine-Fingerprint"
HEADER_PYTHON = "X-uStat-Python"
HEADER_PACKAGES = "X-uStat-Packages"

PROVENANCE_HEADERS = (
    HEADER_RUNTIME,
    HEADER_ENGINE,
    HEADER_ENGINE_VERSION,
    HEADER_ENGINE_FINGERPRINT,
    HEADER_PYTHON,
    HEADER_PACKAGES,
)


def _version_of(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def package_versions() -> Dict[str, str]:
    """Installed versions of the libraries that compute the statistics."""
    return {name: v for name in REPORTED_PACKAGES if (v := _version_of(name))}


@lru_cache(maxsize=1)
def runtime_identity() -> dict:
    """The full provenance record for this process.

    Cached: nothing here can change without a restart, and it is read on every
    request.
    """
    try:
        from ustat_engine import identity as engine_identity

        engine = engine_identity()
    except Exception:  # pragma: no cover - the engine is always importable here
        engine = {}
    return {
        "runtime": "server",
        "engine": "python",
        "engine_version": engine.get("version"),
        # Truncated: this identifies the engine build, and the full digest is
        # already served by /api/engine/identity for anyone verifying it.
        "engine_fingerprint": (engine.get("fingerprint") or "")[:16] or None,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        # Coarse on purpose. "Linux x86_64" is what a methods section needs;
        # a kernel build string is only useful to someone fingerprinting the
        # host.
        "os": f"{platform.system()} {platform.machine()}".strip(),
        "packages": package_versions(),
    }


@lru_cache(maxsize=1)
def packages_header() -> str:
    """``numpy=2.0.2,scipy=1.14.1,...`` -- compact enough for a header."""
    return ",".join(f"{name}={v}" for name, v in package_versions().items())


@lru_cache(maxsize=1)
def provenance_headers() -> Dict[str, str]:
    """The header block stamped onto every API response."""
    info = runtime_identity()
    headers = {
        HEADER_RUNTIME: "server",
        HEADER_ENGINE: "python",
        HEADER_PYTHON: info["python"],
        HEADER_PACKAGES: packages_header(),
    }
    if info.get("engine_version"):
        headers[HEADER_ENGINE_VERSION] = str(info["engine_version"])
    if info.get("engine_fingerprint"):
        headers[HEADER_ENGINE_FINGERPRINT] = str(info["engine_fingerprint"])
    return headers
