#!/usr/bin/env python3
"""Package backend/ustat_engine as a wheel the browser can install.

Built with the standard library rather than setuptools, for one reason: the
wheel has to be reproducible. This whole design rests on the server and the
browser running byte-identical code, and a build that stamps the current time
into the archive produces a different artifact every run -- which makes "did
the engine change?" unanswerable from the artifact alone.

Same sources in, same bytes out. Every zip entry gets a fixed timestamp and
the entries are written in sorted order.

Usage:
    python scripts/build_engine_wheel.py [--out DIR]

Writes the wheel and a manifest.json next to it. The manifest carries the
fingerprint the browser must see after installing; if what it computes from
the installed package differs, the wheel it got is not the wheel we built and
it must not compute anything locally.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import pathlib
import zipfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "backend" / "ustat_engine"
DEFAULT_OUT = REPO_ROOT / "frontend" / "public" / "pyodide" / "wheels"

# A fixed DOS timestamp (1980-01-01), so the archive does not change just
# because the clock did.
FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)

METADATA_TEMPLATE = """Metadata-Version: 2.1
Name: ustat-engine
Version: {version}
Summary: uSTAT's runtime-agnostic statistics core
Requires-Python: >=3.10
Requires-Dist: numpy
Requires-Dist: scipy
Requires-Dist: statsmodels

The statistics uSTAT runs, packaged so the identical code can execute on the
server and in the browser under Pyodide. Not published anywhere; built from
the repository and served to the browser from the application's own origin.
"""

WHEEL_TEMPLATE = """Wheel-Version: 1.0
Generator: ustat build_engine_wheel
Root-Is-Purelib: true
Tag: py3-none-any
"""


def _urlsafe_b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _package_files() -> list[pathlib.Path]:
    return sorted(
        p for p in PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts
    )


def _read_version_and_fingerprint() -> tuple[str, str]:
    """Ask the package itself, so the manifest cannot disagree with the code.

    fingerprint.py is loaded by file path, NOT by importing ustat_engine:
    the package __init__ deliberately imports jsonsafe (numpy) and registers
    every analysis, and this script runs in CI's frontend job under a bare
    Python that has none of the backend's dependencies. The wheel only needs
    the version and the hash, and fingerprint.py is stdlib-only by design --
    the same self-containment the browser's boot check relies on. Loading it
    in place changes nothing about the answer: source_fingerprint() hashes
    files relative to __file__, which a file-path import preserves.
    """
    module_path = PACKAGE_DIR / "fingerprint.py"
    spec = importlib.util.spec_from_file_location("ustat_engine_fingerprint", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fingerprint = module.source_fingerprint()
    if not fingerprint:
        raise SystemExit("engine sources are unreadable; refusing to build a wheel")
    return module.__version__, fingerprint


def build(out_dir: pathlib.Path) -> dict:
    version, fingerprint = _read_version_and_fingerprint()
    files = _package_files()
    if not files:
        raise SystemExit(f"no python sources found under {PACKAGE_DIR}")

    dist_info = f"ustat_engine-{version}.dist-info"
    wheel_name = f"ustat_engine-{version}-py3-none-any.whl"
    out_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = out_dir / wheel_name

    records: list[tuple[str, str, int]] = []

    def add(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
        info = zipfile.ZipInfo(arcname, date_time=FIXED_DATE_TIME)
        info.external_attr = 0o644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, data)
        digest = _urlsafe_b64(hashlib.sha256(data).digest())
        records.append((arcname, f"sha256={digest}", len(data)))

    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arcname = f"ustat_engine/{path.relative_to(PACKAGE_DIR).as_posix()}"
            add(zf, arcname, path.read_bytes())

        add(zf, f"{dist_info}/METADATA", METADATA_TEMPLATE.format(version=version).encode())
        add(zf, f"{dist_info}/WHEEL", WHEEL_TEMPLATE.encode())
        add(zf, f"{dist_info}/top_level.txt", b"ustat_engine\n")

        # RECORD lists every other entry and, by convention, itself with no
        # hash -- it cannot contain its own digest.
        lines = [f"{name},{h},{size}" for name, h, size in records]
        lines.append(f"{dist_info}/RECORD,,")
        add_record = zipfile.ZipInfo(f"{dist_info}/RECORD", date_time=FIXED_DATE_TIME)
        add_record.external_attr = 0o644 << 16
        add_record.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(add_record, ("\n".join(lines) + "\n").encode())

    wheel_bytes = wheel_path.read_bytes()
    manifest = {
        "version": version,
        "wheel": wheel_name,
        "wheel_sha256": hashlib.sha256(wheel_bytes).hexdigest(),
        "wheel_bytes": len(wheel_bytes),
        # What the browser must compute from the INSTALLED package. Anything
        # else means it is running code this build did not produce.
        "source_fingerprint": fingerprint,
        "modules": len(files),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = build(args.out)
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out / manifest['wheel']}")


if __name__ == "__main__":
    main()
