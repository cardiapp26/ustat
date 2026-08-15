"""Proof that both runtimes are running the same code.

A version label only records what someone meant to ship. This hashes what is
actually there: every .py file in the package, by content. The server computes
it from its checkout, the browser computes it from the wheel micropip
installed, and if the two agree the code is byte-identical -- not "the same
release", the same bytes.

That distinction is the whole safety property. A stale wheel in a service
worker cache, a half-applied deploy, a developer running an edited engine
against a colleague's browser build: each produces two different answers to
the same analysis, and each is invisible to a version string that was bumped
by hand.

When the fingerprints disagree the caller must refuse to compute locally and
fall back to the server, rather than quietly returning a number from code the
server has never seen.
"""
from __future__ import annotations

import hashlib
import pathlib

# Bumped by hand, and only meaningful alongside the fingerprint: it names the
# shape of the API, while the fingerprint identifies the exact code.
__version__ = "0.1.0"

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent


def _source_files() -> list[pathlib.Path]:
    return sorted(
        p for p in _PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts
    )


def source_fingerprint() -> str | None:
    """sha256 over the package's own sources, or None if they are unreadable.

    Returns None rather than a made-up value when the sources are missing --
    a runtime serving only bytecode, say. None means "cannot prove equality",
    which the caller must treat as a mismatch, not as a pass.
    """
    files = _source_files()
    if not files:
        return None

    digest = hashlib.sha256()
    for path in files:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        # The relative path is hashed too: moving code between modules changes
        # what runs even when the bytes are collectively identical.
        digest.update(str(path.relative_to(_PACKAGE_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def identity() -> dict:
    """Everything a caller needs to decide whether to trust a local run."""
    return {
        "version": __version__,
        "fingerprint": source_fingerprint(),
        "modules": len(_source_files()),
    }
