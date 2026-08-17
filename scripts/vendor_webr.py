#!/usr/bin/env python3
"""Vendor the webR runtime and an R package mirror into the repo.

FEASIBILITY SPIKE. This is the R-side sibling of scripts/vendor_pyodide.py,
written to find out what a second in-browser engine would cost. Nothing in
frontend/src or backend depends on it yet.

WHY THIS IS VENDORED INSTEAD OF LOADED FROM A CDN
--------------------------------------------------
The reasoning is identical to the Pyodide case, and if anything sharper,
because webR's package story is *more* chatty than Pyodide's.

First, the application's Content-Security-Policy is `script-src 'self'`
and `connect-src 'self'` (see backend/middleware/security_headers.py).
webR's own documentation points people at https://webr.r-wasm.org/v0.6.0/
for the runtime and https://repo.r-wasm.org/ for packages -- two separate
third-party origins, neither of which the policy allows. Vendoring both
keeps the policy tight instead of punching two holes in it.

Second, this is a clinical tool. The bytes that compute a p-value need to
be bytes we fetched once, hashed, and can account for later -- not bytes a
CDN operator can silently swap on a future page load. R.wasm is 18 MB of
compiled C and Fortran; "trust the CDN" is not a posture we want for it.
Pinning a version and recording sha256 per file turns that into "verify
the artifact".

Third -- the point that matters most for a privacy feature -- webR resolves
packages at *runtime*: `webr::install("survival")` issues an HTTP request to
repo.r-wasm.org the moment a clinician asks for a survival analysis. That
request tells a third party, per IP address, exactly which statistical
method is about to be run, which is a legible proxy for what kind of
analysis is happening to whose data. A tool whose entire pitch is "the data
stays local" should not be phoning anyone to announce "the survival module
is loading now". Serving a mirror from our own origin closes that channel:
`webr::install(..., repos = "/webr/repo")` resolves entirely from 'self'.

Finally, vendoring is what makes offline and PWA use possible at all: a
service worker can only cache what was served from 'self' in the first
place.

WHAT THIS SCRIPT DOES
----------------------
1. Downloads the pinned webR release tarball from GitHub and extracts it
   into frontend/public/webr/runtime/, dropping the developer-REPL cruft
   (repl.*, assets/, *.map, index.html) that no embedding application
   loads -- roughly 8 MB of the archive.
2. Downloads the emscripten-contrib PACKAGES index from repo.r-wasm.org
   and resolves the transitive Depends/Imports/LinkingTo closure of the
   packages we want, dropping anything that ships inside R itself. What
   counts as "inside R itself" is not assumed: it is read off the
   vfs/usr/lib/R/library/ tree in the tarball we just extracted, so the
   list cannot drift from the runtime it describes.
3. Downloads each .tgz in the closure into
   frontend/public/webr/repo/bin/emscripten/contrib/<R version>/ and
   regenerates a
   LOCAL PACKAGES file containing only that closure, so a runtime install
   against repos="/webr/repo" never has to reach the network.
4. Writes frontend/public/webr/runtime/vendored.json recording the version,
   every file with its sha256 and size, and the resolved package closure
   with versions -- so a later run, or another engineer, can confirm the
   bytes on disk are exactly the bytes this script fetched.

Re-running is cheap and safe: every file is skipped when it is already
present with a matching sha256.

Usage:
    python3 scripts/vendor_webr.py [--out DIR] [--force]

Stdlib only (urllib.request, hashlib, json, pathlib, argparse, tarfile) so
this also runs under a bare `python3`, the way frontend/package.json invokes
its sibling scripts.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import pathlib
import re
import tarfile
import time
import urllib.error
import urllib.request

WEBR_VERSION = "0.6.0"
RELEASE_URL = (
    f"https://github.com/r-wasm/webr/releases/download/"
    f"v{WEBR_VERSION}/webr-{WEBR_VERSION}.tar.gz"
)

# The R version inside the runtime selects the contrib path on
# repo.r-wasm.org, and it is NOT the webR version: webR 0.6.0 ships R 4.6.0.
# It is read out of the extracted webr.mjs (see r_version_from_runtime)
# rather than hardcoded, because guessing it wrong does not fail loudly --
# repo.r-wasm.org serves a perfectly valid PACKAGES index for every R
# version it has ever built, so a wrong guess mirrors real packages built
# for the wrong ABI and only fails at install time inside the browser.
REPO_BASE = "https://repo.r-wasm.org"


def contrib_path(r_minor: str) -> str:
    return f"bin/emscripten/contrib/{r_minor}"


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "frontend" / "public" / "webr"

# Developer-REPL cruft. webR ships its own interactive REPL page in the same
# tarball as the runtime; an embedding application loads none of it, and it
# is ~8 MB (repl.js.map alone is 6 MB). Excluding it is not an optimisation
# so much as declining to serve a second, unreviewed application from the
# clinical tool's own origin.
#
# The .map files go for the same reason: they are debug artifacts for
# webR's own source, useless without webR's source tree, and 7 MB in total.
EXCLUDE_EXACT = {"index.html"}
EXCLUDE_PREFIXES = ("repl.", "assets/")
EXCLUDE_SUFFIXES = (".map",)

# Packages the spike installs at runtime. Base R already carries t.test;
# jsonlite is only here to get numbers back out at full precision.
REQUIRED_PACKAGES = ["jsonlite"]

USER_AGENT = "ustat-vendor-webr/1.0 (+scripts/vendor_webr.py)"


def _fetch_bytes(url: str, attempts: int = 4, timeout: float = 300.0) -> bytes:
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


def _md5_hex(data: bytes) -> str:
    # Not a security property: repo.r-wasm.org's PACKAGES index publishes
    # MD5sum and nothing stronger, so this is only a transport-integrity
    # check against a truncated or corrupted download. The sha256 we record
    # in vendored.json is what a later run actually verifies against.
    return hashlib.md5(data).hexdigest()


def is_excluded(rel_path: str) -> bool:
    """True for release files no embedding application ever loads."""
    if rel_path in EXCLUDE_EXACT:
        return True
    if rel_path.startswith(EXCLUDE_PREFIXES):
        return True
    if rel_path.endswith(EXCLUDE_SUFFIXES):
        return True
    return False


# ── manifest helpers ────────────────────────────────────────────────────────

def load_prior_manifest(runtime_dir: pathlib.Path) -> dict:
    manifest_path = runtime_dir / "vendored.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def prior_shas(prior_manifest: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in prior_manifest.get("files", []):
        name = entry.get("name")
        sha = entry.get("sha256")
        if name and sha:
            out[name] = sha
    return out


class Written:
    def __init__(self, name: str, size: int, sha256: str, skipped: bool):
        self.name = name
        self.size = size
        self.sha256 = sha256
        self.skipped = skipped


def write_if_changed(
    *, dest: pathlib.Path, name: str, data: bytes, expected_sha: str | None, force: bool
) -> Written:
    """Write `data` to `dest` unless the identical bytes are already there.

    Idempotence is decided by hashing what is on disk, not by mtime or size:
    a truncated or half-written file from an interrupted run must re-download,
    and an unchanged one must not.
    """
    sha = _sha256_hex(data)
    if dest.exists() and not force and dest.stat().st_size > 0:
        local = _sha256_of_file(dest)
        if local == sha and (expected_sha is None or local == expected_sha):
            return Written(name, dest.stat().st_size, local, skipped=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return Written(name, len(data), sha, skipped=False)


# ── PACKAGES parsing ────────────────────────────────────────────────────────

def parse_packages_index(text: str) -> dict[str, dict]:
    """Parse a Debian-control-style PACKAGES file into {name: fields}.

    Handles the continuation lines the index uses for long Depends/Imports
    fields (a leading space or tab continues the previous field).
    """
    entries: dict[str, dict] = {}
    current: dict[str, str] = {}
    last_key: str | None = None

    def flush() -> None:
        nonlocal current, last_key
        if current.get("Package"):
            entries[current["Package"]] = current
        current = {}
        last_key = None

    for raw in text.splitlines():
        if not raw.strip():
            flush()
            continue
        if raw[0] in " \t" and last_key is not None:
            current[last_key] = current[last_key] + " " + raw.strip()
            continue
        if ":" not in raw:
            continue
        key, _, value = raw.partition(":")
        key = key.strip()
        current[key] = value.strip()
        last_key = key
    flush()
    return entries


def parse_dep_field(value: str) -> list[str]:
    """Split a Depends/Imports/LinkingTo field into bare package names.

    Entries look like `methods`, `R (>= 3.5.0)`, `Rcpp (>= 1.0)`. The version
    constraint is dropped: the mirror carries exactly one version of each
    package, so there is nothing to choose between, and a constraint we
    cannot satisfy would be a hard error at install time rather than
    something this script could resolve differently.
    """
    names: list[str] = []
    for part in value.split(","):
        name = part.split("(")[0].strip()
        if name and name != "R":
            names.append(name)
    return names


def r_version_from_runtime(runtime_dir: pathlib.Path) -> str:
    """Read the R version the extracted runtime actually carries.

    webr.mjs is minified, so the version lives in a one- or two-character
    identifier; what is stable across builds is the `R_VERSION:<ident>` entry
    in the default R environment table, and the `<ident>="x.y.z"` assignment
    it points at. Both are matched with a leading non-identifier guard so
    `WEBR_VERSION` (which is a different value entirely) cannot be mistaken
    for `R_VERSION`.
    """
    src = (runtime_dir / "webr.mjs").read_text(errors="replace")
    ref = re.search(r"(?<![A-Za-z_$])R_VERSION\s*:\s*([A-Za-z_$][\w$]*)", src)
    if not ref:
        raise RuntimeError("could not find R_VERSION in webr.mjs")
    ident = re.escape(ref.group(1))
    val = re.search(rf'(?<![A-Za-z_$]){ident}\s*=\s*"([^"]+)"', src)
    if not val:
        raise RuntimeError(f"found R_VERSION -> {ref.group(1)} in webr.mjs but no assignment")
    full = val.group(1)
    parts = full.split(".")
    if len(parts) < 2:
        raise RuntimeError(f"unexpected R version string in webr.mjs: {full!r}")
    return f"{parts[0]}.{parts[1]}"


def bundled_packages(runtime_dir: pathlib.Path) -> set[str]:
    """Packages that ship inside the webR runtime itself.

    Read off the extracted vfs/usr/lib/R/library/ tree rather than hardcoded,
    so this cannot drift from the runtime it describes. webR's tarball stores
    each library subtree either as a directory or as an Emscripten
    `<name>.data.gz` / `<name>.js.metadata` file-package pair, so both
    spellings are collapsed to the package name.
    """
    lib = runtime_dir / "vfs" / "usr" / "lib" / "R" / "library"
    found: set[str] = set()
    if not lib.exists():
        return found
    for child in lib.iterdir():
        name = child.name
        for suffix in (".data.gz", ".js.metadata"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        found.add(name)
    # 'translations' and 'webr' are not R packages a user would install, but
    # they live in the same tree; leaving them in the set is harmless since
    # nothing depends on them by those names.
    return found


def resolve_closure(
    index: dict[str, dict], roots: list[str], bundled: set[str]
) -> list[dict]:
    """Walk Depends/Imports/LinkingTo transitively from `roots`.

    Returns entries in dependency-first order, deduplicated, with anything
    already inside the runtime dropped. Raises if a package that is neither
    bundled nor in the index is required.
    """
    seen: set[str] = set()
    order: list[str] = []

    def visit(name: str, required_by: str | None) -> None:
        if name in seen or name in bundled:
            return
        if name not in index:
            origin = f" (required by {required_by})" if required_by else ""
            raise RuntimeError(
                f"repo.r-wasm.org's PACKAGES index "
                f"has no entry for '{name}'{origin}, and it is not bundled "
                f"inside webR {WEBR_VERSION}"
            )
        seen.add(name)
        fields = index[name]
        for field in ("Depends", "Imports", "LinkingTo"):
            for dep in parse_dep_field(fields.get(field, "")):
                visit(dep, name)
        order.append(name)

    for root in roots:
        visit(root, None)
    return [index[name] for name in order]


def render_packages_index(entries: list[dict]) -> str:
    """Regenerate a PACKAGES file carrying only the vendored closure.

    Only the fields R's install machinery reads to resolve and verify a
    binary package are emitted. Suggests is deliberately dropped: it names
    packages we did not mirror, and leaving it in makes a local resolver
    look for files that are not there.
    """
    keep = ("Package", "Version", "Depends", "Imports", "LinkingTo", "License", "MD5sum")
    blocks: list[str] = []
    for entry in entries:
        lines = [f"{k}: {entry[k]}" for k in keep if entry.get(k)]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


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
        help="rewrite every file even if it already exists with a matching sha256",
    )
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out
    runtime_dir = out_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    prior = load_prior_manifest(runtime_dir)
    prior_sha = prior_shas(prior)

    all_files: list[dict] = []
    total_bytes = 0
    written_count = 0
    written_bytes = 0
    skipped_count = 0

    def record(w: Written, label: str) -> None:
        nonlocal total_bytes, written_count, written_bytes, skipped_count
        all_files.append({"name": w.name, "sha256": w.sha256, "size": w.size})
        total_bytes += w.size
        if w.skipped:
            skipped_count += 1
        else:
            written_count += 1
            written_bytes += w.size
            print(f"  wrote {label} ({w.size:,} bytes)")

    # --- 1. Runtime tarball. -------------------------------------------------
    # Every file in the archive is needed to decide what is on disk, so the
    # cheap path is still a full download; what idempotence buys is not
    # rewriting 26 MB of unchanged files (and not invalidating anyone's
    # HTTP cache or a service worker's precache for no reason).
    print(f"Fetching webR {WEBR_VERSION} release tarball ...")
    tar_bytes = _fetch_bytes(RELEASE_URL)
    print(f"  {len(tar_bytes):,} bytes; extracting (excluding repl.*, assets/, *.map, index.html)")

    excluded_count = 0
    excluded_bytes = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            # Strip the single `webr-<version>/` top-level directory.
            parts = member.name.split("/", 1)
            if len(parts) != 2:
                continue
            rel = parts[1]
            if not rel:
                continue
            if is_excluded(rel):
                excluded_count += 1
                excluded_bytes += member.size
                continue
            # Refuse anything that would escape the output directory. A
            # release tarball from a project we trust is still an archive
            # from the network.
            dest = (runtime_dir / rel).resolve()
            if not str(dest).startswith(str(runtime_dir.resolve()) + "/"):
                raise RuntimeError(f"refusing to extract outside out dir: {member.name}")
            fh = tf.extractfile(member)
            if fh is None:
                continue
            data = fh.read()
            record(
                write_if_changed(
                    dest=dest,
                    name=f"runtime/{rel}",
                    data=data,
                    expected_sha=prior_sha.get(f"runtime/{rel}"),
                    force=args.force,
                ),
                f"runtime/{rel}",
            )

    print(f"  excluded {excluded_count} files ({excluded_bytes:,} bytes uncompressed)")

    r_minor = r_version_from_runtime(runtime_dir)
    contrib = contrib_path(r_minor)
    packages_url = f"{REPO_BASE}/{contrib}/PACKAGES"
    contrib_dir = out_dir / "repo" / contrib
    contrib_dir.mkdir(parents=True, exist_ok=True)
    print(f"  runtime carries R {r_minor}.x -> mirroring {contrib}")

    bundled = bundled_packages(runtime_dir)
    print(f"  webR bundles {len(bundled)} R library entries: {', '.join(sorted(bundled))}")

    # --- 2. Upstream package index and closure. ------------------------------
    print(f"Fetching {packages_url} ...")
    index_bytes = _fetch_bytes(packages_url)
    index = parse_packages_index(index_bytes.decode("utf-8", errors="replace"))
    print(f"  upstream index lists {len(index):,} packages")

    print(f"Resolving closure for: {', '.join(REQUIRED_PACKAGES)} ...")
    closure = resolve_closure(index, REQUIRED_PACKAGES, bundled)
    print(
        f"  closure: {len(closure)} package(s) to mirror "
        f"({', '.join(p['Package'] for p in closure) or 'none'})"
    )

    resolved: list[dict] = []
    closure_bytes = 0
    for entry in closure:
        name = entry["Package"]
        version = entry["Version"]
        file_name = f"{name}_{version}.tgz"
        url = f"{REPO_BASE}/{contrib}/{file_name}"
        manifest_name = f"repo/{contrib}/{file_name}"
        dest = contrib_dir / file_name

        expected_sha = prior_sha.get(manifest_name)
        if dest.exists() and not args.force and expected_sha:
            local = _sha256_of_file(dest)
            if local == expected_sha:
                size = dest.stat().st_size
                all_files.append({"name": manifest_name, "sha256": local, "size": size})
                total_bytes += size
                closure_bytes += size
                skipped_count += 1
                resolved.append(
                    {
                        "name": name,
                        "version": version,
                        "file_name": file_name,
                        "sha256": local,
                        "size": size,
                        "md5_from_index": entry.get("MD5sum"),
                        "requested": name in REQUIRED_PACKAGES,
                    }
                )
                print(f"  {file_name}: already present, sha256 verified")
                continue

        print(f"Fetching {file_name} ...")
        data = _fetch_bytes(url)
        expected_md5 = entry.get("MD5sum")
        if expected_md5:
            actual_md5 = _md5_hex(data)
            if actual_md5 != expected_md5:
                raise RuntimeError(
                    f"MD5 mismatch for {file_name}: PACKAGES says {expected_md5}, "
                    f"got {actual_md5} (source: {url})"
                )
        w = write_if_changed(
            dest=dest,
            name=manifest_name,
            data=data,
            expected_sha=expected_sha,
            force=args.force,
        )
        record(w, manifest_name)
        closure_bytes += w.size
        resolved.append(
            {
                "name": name,
                "version": version,
                "file_name": file_name,
                "sha256": w.sha256,
                "size": w.size,
                "md5_from_index": expected_md5,
                "requested": name in REQUIRED_PACKAGES,
            }
        )

    # --- 3. Local PACKAGES index, plain and gzipped. --------------------------
    # Both, because R's available.packages() tries PACKAGES.rds, then
    # PACKAGES.gz, then PACKAGES, and takes the first response it can read.
    # A dev server with an SPA history fallback answers the two it does not
    # have with index.html at HTTP 200 rather than 404, and R then parses the
    # HTML as a package index and fails with "Line starting '<!doctype html>'
    # is malformed" -- an error that names neither the repository nor the
    # real cause. Shipping a valid PACKAGES.gz means the first readable
    # response is the right one. (.rds is left out: writing R's serialisation
    # format from stdlib Python is not worth it, and readRDS rejects an HTML
    # body cleanly on the magic bytes.)
    local_index = render_packages_index(closure).encode("utf-8")
    record(
        write_if_changed(
            dest=contrib_dir / "PACKAGES",
            name=f"repo/{contrib}/PACKAGES",
            data=local_index,
            expected_sha=prior_sha.get(f"repo/{contrib}/PACKAGES"),
            force=args.force,
        ),
        f"repo/{contrib}/PACKAGES",
    )
    # mtime=0 so the gzip header is byte-identical run to run; otherwise the
    # sha256 changes on every run and idempotence is a lie.
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(local_index)
    record(
        write_if_changed(
            dest=contrib_dir / "PACKAGES.gz",
            name=f"repo/{contrib}/PACKAGES.gz",
            data=buf.getvalue(),
            expected_sha=prior_sha.get(f"repo/{contrib}/PACKAGES.gz"),
            force=args.force,
        ),
        f"repo/{contrib}/PACKAGES.gz",
    )

    # --- 4. Manifest. No timestamp: we have no reliable source for "now"
    # here and would rather omit the field than guess it. ---------------------
    manifest = {
        "webr_version": WEBR_VERSION,
        "r_version": r_minor,
        "r_contrib_path": contrib,
        "runtime_source_url": RELEASE_URL,
        "repo_source_url": f"{REPO_BASE}/{contrib}/",
        "excluded_from_runtime": {
            "exact": sorted(EXCLUDE_EXACT),
            "prefixes": sorted(EXCLUDE_PREFIXES),
            "suffixes": sorted(EXCLUDE_SUFFIXES),
            "file_count": excluded_count,
            "uncompressed_bytes": excluded_bytes,
        },
        "bundled_r_library": sorted(bundled),
        "requested_packages": REQUIRED_PACKAGES,
        "resolved_packages": resolved,
        "files": all_files,
    }
    manifest_path = runtime_dir / "vendored.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")

    # --- 5. Summary. ---------------------------------------------------------
    print()
    print("=" * 60)
    print("Vendoring summary")
    print("=" * 60)
    print(f"webR version:           {WEBR_VERSION}")
    print(f"Output directory:       {out_dir}")
    print(
        f"Total files tracked:    {len(all_files)} "
        f"({written_count} written this run, {skipped_count} already present)"
    )
    print(f"Total bytes on disk:    {total_bytes:,}")
    print(f"Bytes written this run: {written_bytes:,}")
    print(f"Package mirror:         {len(resolved)} package(s), {closure_bytes:,} bytes")
    print(f"Manifest written to:    {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
