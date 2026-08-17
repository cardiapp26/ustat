#!/usr/bin/env python3
"""Concatenate backend/ustat_engine_r into one .R file the browser can eval.

The R sibling of scripts/build_engine_wheel.py, and it exists for the same
reason: the server and the browser have to be running byte-identical code, and
that is only checkable if the build is reproducible. Same sources in, same bytes
out.

Reproducibility is nearly free here, because unlike a wheel there is no archive
to timestamp -- the artifact is a text file. What it takes is a deterministic
concatenation order, so RUNTIME_ORDER below is an explicit list rather than a
glob, and the build FAILS on a runtime module that is not in it. A new file
silently landing at whatever position sorted() put it in is exactly the sort of
change that would alter the bundle without altering anything anyone meant to
change.

Order is not cosmetic. R has no imports: the bundle is one flat script, so a
function has to be defined before the file-scope code that calls it runs.
registry.R is last of the runtime modules because every analyses/ file ends with
a ustat_register() call at ITS file scope, and analyses/ follows in sorted
order.

Usage:
    python3 scripts/build_r_bundle.py [--out DIR]

Writes ustat_engine_r.R and a manifest.json next to it. The manifest carries the
source fingerprint the server computes from the same tree; a browser holding a
bundle whose manifest disagrees with GET /api/engine/r/identity is holding code
this server has never seen, and must not compute anything locally with it.

Standard library only, so `python3 scripts/build_r_bundle.py` works the way
frontend/package.json invokes its sibling scripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "backend" / "ustat_engine_r"
RUNTIME_DIR = PACKAGE_DIR / "runtime"
ANALYSES_DIR = PACKAGE_DIR / "analyses"
DEFAULT_OUT = REPO_ROOT / "frontend" / "public" / "webr" / "bundle"

BUNDLE_NAME = "ustat_engine_r.R"

#: Runtime modules, in evaluation order. Not sorted -- see the module docstring.
RUNTIME_ORDER = ("errors", "jsonsafe", "frame", "stats", "text", "registry")

HEADER = """# GENERATED FILE -- DO NOT EDIT.
#
# Built by scripts/build_r_bundle.py from backend/ustat_engine_r/. Edit the
# sources there and re-run the build (npm run r:bundle); anything changed here
# is overwritten on the next dev start and would in any case fail the
# fingerprint check the browser makes against GET /api/engine/r/identity.
#
# This file is pure ASCII by construction; characters above U+007F are written
# as \\u escapes in the sources.
"""

SEPARATOR = "# ===========================================================================\n# {path}\n# ===========================================================================\n"


def _module_paths() -> list[pathlib.Path]:
    """Every source, in the order it is concatenated.

    Raises if a runtime module is present on disk but missing from
    RUNTIME_ORDER, or named in RUNTIME_ORDER but missing from disk. A build that
    quietly dropped a module would produce a bundle whose fingerprint still
    matched the server's -- the fingerprint is over the tree, not the bundle --
    and would fail at the first call into whatever went missing.
    """
    on_disk = {p.stem for p in RUNTIME_DIR.glob("*.R")}
    declared = set(RUNTIME_ORDER)
    if on_disk != declared:
        missing = sorted(declared - on_disk)
        extra = sorted(on_disk - declared)
        problems = []
        if missing:
            problems.append(f"declared in RUNTIME_ORDER but absent: {', '.join(missing)}")
        if extra:
            problems.append(f"present but not in RUNTIME_ORDER: {', '.join(extra)}")
        raise SystemExit(f"{RUNTIME_DIR}: {'; '.join(problems)}")

    paths = [RUNTIME_DIR / f"{name}.R" for name in RUNTIME_ORDER]
    paths.extend(sorted(ANALYSES_DIR.glob("*.R")))
    return paths


def _identity():
    """Ask the tree's own identity module, so the manifest cannot disagree."""
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    try:
        from ustat_engine_r.fingerprint import __version__, analyses, source_fingerprint
    finally:
        sys.path.pop(0)

    fingerprint = source_fingerprint()
    if not fingerprint:
        raise SystemExit("R engine sources are unreadable; refusing to build a bundle")
    return __version__, fingerprint, analyses()


def render(paths: list[pathlib.Path]) -> bytes:
    chunks = [HEADER.encode("utf-8")]
    for path in paths:
        rel = path.relative_to(PACKAGE_DIR).as_posix()
        chunks.append(b"\n")
        chunks.append(SEPARATOR.format(path=rel).encode("utf-8"))
        data = path.read_bytes()
        chunks.append(data)
        # One trailing newline per module regardless of how the file ended, so a
        # missing final newline cannot glue two statements together.
        if not data.endswith(b"\n"):
            chunks.append(b"\n")
    return b"".join(chunks)


def build(out_dir: pathlib.Path) -> dict:
    version, fingerprint, declared = _identity()
    paths = _module_paths()
    if not paths:
        raise SystemExit(f"no R sources found under {PACKAGE_DIR}")

    bundle = render(paths)
    non_ascii = [b for b in set(bundle) if b > 127]
    if non_ascii:
        raise SystemExit(
            "R sources must be pure ASCII (write \\u escapes instead); found "
            f"{len(non_ascii)} distinct byte(s) above 0x7f"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / BUNDLE_NAME).write_bytes(bundle)

    manifest = {
        "version": version,
        "bundle": BUNDLE_NAME,
        "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "bundle_bytes": len(bundle),
        # What the SERVER computes from backend/ustat_engine_r/. The browser
        # cannot recompute this from the bundle -- the bundle is a
        # concatenation, and the fingerprint is per-file -- so it checks the
        # bundle against bundle_sha256 and this manifest against the endpoint.
        "source_fingerprint": fingerprint,
        "modules": len(paths),
        "analyses": [
            {
                "id": a["id"],
                "needs_frame": a["needs_frame"],
                "packages": a["packages"],
                # The same list under the name a webR host will recognise: these
                # are CRAN package names to be installed with
                # webr::install(..., repos = "/webr/repo") before the analysis
                # runs. Mirrored rather than aliased so a future spec can
                # declare a package that is not on the mirror.
                "r_packages": a["packages"],
            }
            for a in declared
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = build(args.out)
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out / manifest['bundle']}")


if __name__ == "__main__":
    main()
