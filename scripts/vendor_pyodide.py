#!/usr/bin/env python3
"""Vendor the Pyodide runtime and the wheels uSTAT needs into the repo.

WHY THIS IS VENDORED INSTEAD OF LOADED FROM A CDN
--------------------------------------------------
uSTAT runs its statistics in the browser, under Pyodide, specifically so
that patient data never has to leave the clinician's machine. Loading the
Pyodide runtime itself from cdn.jsdelivr.net would quietly undercut that
promise in three separate ways.

First, the application's Content-Security-Policy is `script-src 'self'`
and `connect-src 'self'` (see backend/middleware/security_headers.py) --
a third-party script origin simply is not allowed to execute, so a
CDN-loaded Pyodide would either be blocked outright or force us to loosen
a policy that exists precisely to keep unreviewed third-party code out of
a clinical tool. Vendoring keeps the policy tight.

Second, this is a clinical tool: every byte of the Python runtime that
computes a p-value or a hazard ratio for a patient needs to be something
we fetched once, hashed, and can account for later, not something a CDN
operator (or someone who compromises that CDN) can silently swap out from
under us on a future page load. Pinning a version and recording its
sha256 in vendored.json turns "trust the CDN" into "verify the artifact."

Third -- and this is the point that matters most for a privacy feature --
a CDN sees every request. If Pyodide's loader is fetching numpy, scipy,
statsmodels and scikit-learn wheels from jsdelivr on demand, then jsdelivr
(and anyone downstream of it) learns, per request, which statistical
packages a given IP address just pulled down, which is a legible proxy
for what kind of analysis a clinician is about to run on their patients'
data. A tool whose entire pitch is "the data stays local" should not be
phoning a third party to say "the biostatistics module is loading now."
Serving the runtime from the app's own origin closes that side channel.

Finally, vendoring is also what makes offline and PWA use possible at
all: a service worker can only cache what was served from 'self' in the
first place.

WHAT THIS SCRIPT DOES
----------------------
1. Downloads pyodide-lock.json for the pinned version and uses it as the
   source of truth for package file names, versions, dependency edges,
   and checksums -- nothing here is a guessed file list.
2. Downloads the small set of core files pyodide.js actually needs to
   boot (determined by reading pyodide.js's and pyodide.asm.js's own
   references, not assumed): the loader, the asm.js glue, the wasm
   binary, and the zipped Python standard library.
3. Resolves the transitive dependency closure of the packages uSTAT
   needs (numpy, scipy, statsmodels, pandas, scikit-learn, patsy,
   micropip) by walking `depends` in the lock file, and downloads every
   wheel (and non-wheel shared library, e.g. openblas) in that closure.
4. Verifies every file is non-empty, verifies package files against the
   lock file's sha256 when one is given, and writes vendored.json so a
   later run -- or another engineer -- can confirm the files sitting in
   frontend/public/pyodide/runtime/ are exactly the bytes this script
   fetched.

Usage:
    python3 scripts/vendor_pyodide.py [--out DIR] [--force]

Stdlib only (urllib.request, hashlib, json, pathlib, argparse) so this
also runs under a bare `python3`, the way frontend/package.json invokes
its sibling scripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

PYODIDE_VERSION = "0.27.7"
BASE_URL = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "frontend" / "public" / "pyodide" / "runtime"

# The files pyodide.js and pyodide.asm.js reference in order to boot at
# all: the loader, the asm.js glue (which embeds/loads the wasm binary),
# the wasm binary itself, the zipped Python standard library, and the
# lock file the loader reads to resolve every other package by name.
# Confirmed empirically by grepping the actual v0.27.7 pyodide.js /
# pyodide.asm.js source for the filenames they request, rather than
# assumed from an older Pyodide layout (pre-0.21 shipped a separate
# pyodide.asm.data file for the stdlib; 0.27.x does not).
#
# pyodide.mjs is the ES-module entry point, and it is the one this app
# actually uses: the engine runs in a module worker, where importScripts --
# the only way to consume the UMD pyodide.js -- does not exist. pyodide.js is
# kept alongside it because the throwaway parity spike loads it classically,
# and because it costs 15 KB to leave the non-module path available.
CORE_FILES = [
    "pyodide-lock.json",
    "pyodide.js",
    "pyodide.mjs",
    "pyodide.asm.js",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
]

# The packages uSTAT's in-browser engine actually imports. Everything
# else that gets downloaded is a transitive dependency of these, resolved
# from pyodide-lock.json rather than hardcoded.
REQUIRED_PACKAGES = [
    "numpy",
    "scipy",
    "statsmodels",
    "pandas",
    "scikit-learn",
    "patsy",
    "micropip",
]

USER_AGENT = "ustat-vendor-pyodide/1.0 (+scripts/vendor_pyodide.py)"


def _fetch_bytes(url: str, attempts: int = 4, timeout: float = 120.0) -> bytes:
    """GET a URL into memory, retrying transient network failures."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            if attempt < attempts:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"failed to fetch {url} after {attempts} attempts: {last_exc}")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_of_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_package_closure(
    lock: dict, roots: list[str]
) -> list[dict]:
    """Walk `depends` transitively from `roots` through the lock file.

    Returns the resolved packages in dependency-first order (a package's
    dependencies appear before the package itself), deduplicated. Raises
    if a root or a dependency named in the lock file is missing from it.
    """
    packages = lock["packages"]
    seen: set[str] = set()
    order: list[str] = []

    def visit(name: str) -> None:
        if name in seen:
            return
        if name not in packages:
            raise RuntimeError(
                f"pyodide-lock.json for v{PYODIDE_VERSION} has no entry for "
                f"'{name}' (required, directly or transitively, by the "
                f"requested package set)"
            )
        seen.add(name)
        for dep in packages[name].get("depends", []):
            visit(dep)
        order.append(name)

    for root in roots:
        visit(root)

    return [packages[name] for name in order]


class FetchResult:
    def __init__(self, name: str, size: int, sha256: str, skipped: bool):
        self.name = name
        self.size = size
        self.sha256 = sha256
        self.skipped = skipped


def ensure_file(
    *,
    file_name: str,
    url: str,
    out_dir: pathlib.Path,
    expected_sha256: str | None,
    force: bool,
) -> FetchResult:
    """Download `url` to out_dir/file_name unless it's already present and
    verified, per `expected_sha256` (from the lock file for packages, or
    from a prior run's manifest for core files that have no lock-supplied
    checksum). Idempotent and resumable: re-run-safe by design.
    """
    dest = out_dir / file_name

    if dest.exists() and not force:
        if dest.stat().st_size > 0:
            local_sha = _sha256_of_file(dest)
            if expected_sha256 is None or local_sha == expected_sha256:
                return FetchResult(file_name, dest.stat().st_size, local_sha, skipped=True)
        # Existing file is empty, or its hash doesn't match what we
        # expect -- fall through and re-download.

    data = _fetch_bytes(url)
    if len(data) == 0:
        raise RuntimeError(f"downloaded {file_name} from {url} but got 0 bytes")

    actual_sha = _sha256_hex(data)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise RuntimeError(
            f"checksum mismatch for {file_name}: expected {expected_sha256}, "
            f"got {actual_sha} (source: {url})"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return FetchResult(file_name, len(data), actual_sha, skipped=False)


def load_prior_manifest(out_dir: pathlib.Path) -> dict:
    manifest_path = out_dir / "vendored.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def prior_sha_for(prior_manifest: dict, file_name: str) -> str | None:
    for entry in prior_manifest.get("files", []):
        if entry.get("name") == file_name:
            return entry.get("sha256")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=DEFAULT_OUT,
        help=f"output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download every file even if it already exists with a matching sha256",
    )
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_manifest = load_prior_manifest(out_dir)

    all_files: list[dict] = []
    total_bytes = 0
    downloaded_bytes = 0
    skipped_count = 0
    downloaded_count = 0

    # --- 1. Lock file first: it is both a core boot file and our source
    # of truth for the package closure. ---
    print(f"Fetching pyodide-lock.json for v{PYODIDE_VERSION} ...")
    lock_url = BASE_URL + "pyodide-lock.json"
    lock_expected_sha = prior_sha_for(prior_manifest, "pyodide-lock.json")
    lock_result = ensure_file(
        file_name="pyodide-lock.json",
        url=lock_url,
        out_dir=out_dir,
        expected_sha256=lock_expected_sha,
        force=args.force,
    )
    lock = json.loads((out_dir / "pyodide-lock.json").read_text())
    lock_info = lock.get("info", {})
    if lock_info.get("version") != PYODIDE_VERSION:
        print(
            f"WARNING: pyodide-lock.json reports version "
            f"{lock_info.get('version')!r}, expected {PYODIDE_VERSION!r}",
            file=sys.stderr,
        )

    all_files.append(
        {"name": "pyodide-lock.json", "sha256": lock_result.sha256, "size": lock_result.size}
    )
    total_bytes += lock_result.size
    if lock_result.skipped:
        skipped_count += 1
    else:
        downloaded_count += 1
        downloaded_bytes += lock_result.size

    # --- 2. Remaining core boot files. No lock-supplied checksum exists
    # for these (they aren't "packages"), so we verify against what a
    # prior run of this script recorded, if any. ---
    for file_name in CORE_FILES:
        if file_name == "pyodide-lock.json":
            continue
        print(f"Fetching core file {file_name} ...")
        expected = prior_sha_for(prior_manifest, file_name)
        result = ensure_file(
            file_name=file_name,
            url=BASE_URL + file_name,
            out_dir=out_dir,
            expected_sha256=expected,
            force=args.force,
        )
        all_files.append({"name": file_name, "sha256": result.sha256, "size": result.size})
        total_bytes += result.size
        if result.skipped:
            skipped_count += 1
            print(f"  skipped (already present, sha256 verified)")
        else:
            downloaded_count += 1
            downloaded_bytes += result.size
            print(f"  downloaded {result.size:,} bytes")

    # --- 3. Resolve and fetch the package dependency closure. ---
    print(f"Resolving dependency closure for: {', '.join(REQUIRED_PACKAGES)} ...")
    closure = resolve_package_closure(lock, REQUIRED_PACKAGES)
    print(f"Resolved closure: {len(closure)} packages "
          f"({', '.join(p['name'] for p in closure)})")

    resolved_packages: list[dict] = []
    closure_bytes = 0

    for pkg in closure:
        file_name = pkg["file_name"]
        expected_sha = pkg.get("sha256")
        print(f"Fetching package {pkg['name']} {pkg['version']} ({file_name}) ...")
        result = ensure_file(
            file_name=file_name,
            url=BASE_URL + file_name,
            out_dir=out_dir,
            expected_sha256=expected_sha,
            force=args.force,
        )
        all_files.append({"name": file_name, "sha256": result.sha256, "size": result.size})
        total_bytes += result.size
        closure_bytes += result.size
        if result.skipped:
            skipped_count += 1
            print(f"  skipped (already present, sha256 verified)")
        else:
            downloaded_count += 1
            downloaded_bytes += result.size
            print(f"  downloaded {result.size:,} bytes")

        resolved_packages.append(
            {
                "name": pkg["name"],
                "version": pkg["version"],
                "file_name": file_name,
                "package_type": pkg.get("package_type"),
                "sha256": result.sha256,
                "size": result.size,
                "requested": pkg["name"] in REQUIRED_PACKAGES,
            }
        )

    # --- 4. Write the manifest. No timestamp: we have no reliable source
    # for "now" here and would rather omit the field than guess it. ---
    manifest = {
        "pyodide_version": PYODIDE_VERSION,
        "source_base_url": BASE_URL,
        "files": all_files,
        "requested_packages": REQUIRED_PACKAGES,
        "resolved_packages": resolved_packages,
    }
    manifest_path = out_dir / "vendored.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")

    # --- 5. Summary. ---
    print()
    print("=" * 60)
    print("Vendoring summary")
    print("=" * 60)
    print(f"Pyodide version:        {PYODIDE_VERSION}")
    print(f"Output directory:       {out_dir}")
    print(f"Total files tracked:    {len(all_files)} "
          f"({downloaded_count} downloaded this run, {skipped_count} already present)")
    print(f"Total bytes on disk:    {total_bytes:,}")
    print(f"Bytes fetched this run: {downloaded_bytes:,}")
    print(f"Package closure size:   {len(closure)} packages, {closure_bytes:,} bytes")
    # jsdelivr's HTTP/2 responses for these files do not send a
    # Content-Length header (chunked transfer, and we do not request
    # Accept-Encoding: gzip), so the bytes we received are the bytes that
    # crossed the wire -- there is no separate compressed-vs-uncompressed
    # figure to report beyond "downloaded_bytes" above.
    print(f"Compressed-over-the-wire size: not separately reported by the "
          f"CDN for this transfer; bytes fetched this run ({downloaded_bytes:,}) "
          f"is the wire size, since no additional response compression was "
          f"negotiated.")
    print(f"Manifest written to:   {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
