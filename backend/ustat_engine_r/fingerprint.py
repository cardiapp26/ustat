"""Proof that the server and the browser are running the same R engine.

The exact analogue of `ustat_engine/fingerprint.py`, over .R files instead of
.py files, and deliberately the same algorithm byte for byte:

    sha256( for each source, in sorted relative-path order:
              relative_path (utf-8) + NUL + file bytes + NUL )

The relative path is hashed alongside the bytes because moving a function
between modules changes what runs even when the tree's total bytes do not.

Why a *second* implementation of the same eight lines rather than an import: the
two engines are separately shippable artifacts with separately versioned
sources, and the Python engine's `fingerprint.py` is itself inside the tree it
hashes. Importing it here would put a change to the R engine's identity code
inside the Python engine's fingerprint, and vice versa -- so "did the Python
engine change?" would start answering yes when only R had. The duplication is
about 15 lines and `test_r_bundle_and_identity.py` pins that the two algorithms
agree on identical input.

This module also parses each `analyses/*.R` for what it registers, which is how
the server can answer "which analyses can this browser run, and what does it
have to install first?" without evaluating any R.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

# Bumped by hand, and only meaningful alongside the fingerprint: it names the
# shape of the API, while the fingerprint identifies the exact code.
__version__ = "0.1.0"

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent.parent
_VENDORED_WEBR = _REPO_ROOT / "frontend" / "public" / "webr" / "runtime" / "vendored.json"

# The declaration each analyses/*.R file ends with, e.g.
#
#     ustat_register(list(
#       id = "stats.ttest",
#       needs_frame = TRUE,
#       packages = c("moments", "nortest"),
#       columns_for = function(params) ...,
#       fn = ustat_ttest
#     ))
#
# Parsed with regexes rather than an R parser, because the server has no R. The
# format that buys is narrow and enforced by test_r_engine_isolation.py: the id
# must be a plain double-quoted literal, needs_frame a bare TRUE/FALSE, and
# packages either `character(0)` or a `c("a", "b")` of double-quoted literals --
# no variables, no concatenation, nothing that would need evaluating to read.
_REGISTER_CALL = "ustat_register("
_ID_RE = re.compile(r'\bid\s*=\s*"([^"]+)"')
_NEEDS_FRAME_RE = re.compile(r"\bneeds_frame\s*=\s*(TRUE|FALSE)\b")
_PACKAGES_RE = re.compile(r"\bpackages\s*=\s*(character\(0\)|c\(([^)]*)\))")
_QUOTED_RE = re.compile(r'"([^"]*)"')


def _source_files() -> list[pathlib.Path]:
    return sorted(_PACKAGE_ROOT.rglob("*.R"))


def _analysis_files() -> list[pathlib.Path]:
    return sorted((_PACKAGE_ROOT / "analyses").glob("*.R"))


def source_fingerprint() -> str | None:
    """sha256 over the R tree's own sources, or None if they are unreadable.

    None rather than a made-up value: it means "cannot prove equality", which
    the caller must treat as a mismatch and not as a pass.
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
        digest.update(str(path.relative_to(_PACKAGE_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def analyses() -> list[dict]:
    """What each `analyses/*.R` file registers, in file-name order.

    Returns dicts of ``{id, needs_frame, packages, source}``. Raises ValueError
    on a file that has no parsable registration, because a silently-skipped
    analysis would show up as an endpoint that says the browser cannot run
    something it can.
    """
    out: list[dict] = []
    for path in _analysis_files():
        text = path.read_text(encoding="utf-8")
        start = text.rfind(_REGISTER_CALL)
        if start == -1:
            raise ValueError(f"{path.name}: no {_REGISTER_CALL} call found")
        call = text[start:]

        id_match = _ID_RE.search(call)
        if not id_match:
            raise ValueError(f'{path.name}: {_REGISTER_CALL} has no id = "..."')

        needs_frame_match = _NEEDS_FRAME_RE.search(call)
        packages_match = _PACKAGES_RE.search(call)
        packages: list[str] = []
        if packages_match and packages_match.group(2) is not None:
            packages = _QUOTED_RE.findall(packages_match.group(2))

        out.append(
            {
                "id": id_match.group(1),
                # The R default when the key is absent, mirrored here rather
                # than assumed by the reader.
                "needs_frame": (
                    True if needs_frame_match is None
                    else needs_frame_match.group(1) == "TRUE"
                ),
                "packages": packages,
                "source": path.name,
            }
        )

    ids = [a["id"] for a in out]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate analysis id(s): {', '.join(duplicates)}")
    return out


def _vendored_webr() -> dict:
    """webR and R versions from the vendored runtime, or {} if it is absent.

    The mirror is gitignored (see .gitignore), so a fresh checkout that has not
    run `npm run webr:vendor` legitimately has no answer here. Callers get None
    for both fields rather than a guess -- the R version decides which package
    ABI the browser can install, and guessing it wrong fails only in the tab.
    """
    if not _VENDORED_WEBR.is_file():
        return {}
    try:
        return json.loads(_VENDORED_WEBR.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def identity() -> dict:
    """Everything a caller needs to decide whether to trust a local R run."""
    vendored = _vendored_webr()
    return {
        "version": __version__,
        "fingerprint": source_fingerprint(),
        "modules": len(_source_files()),
        "analyses": [a["id"] for a in analyses()],
        "webr_version": vendored.get("webr_version"),
        "r_version": vendored.get("r_version"),
    }
