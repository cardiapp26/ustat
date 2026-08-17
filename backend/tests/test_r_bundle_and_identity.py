"""Proof that the server, the R bundle, and /api/engine/r/identity agree on
what R code the browser is about to run.

The mirror of test_engine_wheel_and_identity.py, and it rests on the same
claim: a sha256 over the tree's own .R sources, by relative path and content,
identifies the code byte-for-byte -- not "the same release", the same bytes.
The claim is checked in both directions:

  - the fingerprint moves when the source moves, on content AND on a rename
    (it is not a constant dressed up as a hash);
  - the bundle builder reproduces that exact fingerprint and emits
    byte-identical text across runs;
  - the endpoint serves what the tree computes in-process.

There is one structural difference from the Python engine worth stating. The
Python browser gets a wheel and can recompute source_fingerprint from the
package it installed. The R browser gets a CONCATENATION, from which a per-file
fingerprint cannot be recovered -- so the chain is: browser hashes the bundle
and compares to manifest.bundle_sha256; manifest.source_fingerprint is compared
to this endpoint. Two links instead of one, and both are checked here.

Everything that mutates works on a copy under tmp_path. Nothing touches
backend/ustat_engine_r on disk.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import re
import pathlib
import shutil
import sys
from pathlib import Path

import pytest

from ustat_engine_r.fingerprint import __version__, analyses, source_fingerprint

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_DIR = REPO_ROOT / "backend" / "ustat_engine_r"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_r_bundle.py"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _load_build_module():
    spec = importlib.util.spec_from_file_location("_build_r_bundle_under_test", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _copy_package(dest_root: Path, package_name: str) -> Path:
    dest = dest_root / package_name
    shutil.copytree(PACKAGE_DIR, dest)
    return dest


def _import_copy(dest_root: Path, package_name: str):
    sys.path.insert(0, str(dest_root))
    try:
        return importlib.import_module(f"{package_name}.fingerprint")
    except BaseException:
        sys.path.remove(str(dest_root))
        raise


def _unimport_copy(dest_root: Path, package_name: str) -> None:
    path_str = str(dest_root)
    if path_str in sys.path:
        sys.path.remove(path_str)
    for name in list(sys.modules):
        if name == package_name or name.startswith(package_name + "."):
            del sys.modules[name]


# ---------------------------------------------------------------------------
# 1. The endpoints
# ---------------------------------------------------------------------------


def test_r_identity_endpoint_shape(client):
    response = client.get("/api/engine/r/identity")
    assert response.status_code == 200

    body = response.json()
    assert {"version", "fingerprint", "modules", "analyses", "webr_version", "r_version"} <= set(body)

    assert isinstance(body["fingerprint"], str)
    assert _HEX64.match(body["fingerprint"]), body["fingerprint"]
    assert body["fingerprint"] == source_fingerprint()
    assert body["version"] == __version__

    assert isinstance(body["modules"], int) and body["modules"] > 0
    assert isinstance(body["analyses"], list)
    assert "stats.ttest" in body["analyses"]


def test_r_identity_reports_the_vendored_webr_or_says_it_does_not_know(client):
    """The mirror is gitignored, so a fresh checkout has no answer here. null is
    the honest one -- the R version decides which package ABI the browser can
    install, and a guess fails only in the tab."""
    body = client.get("/api/engine/r/identity").json()
    vendored = REPO_ROOT / "frontend" / "public" / "webr" / "runtime" / "vendored.json"
    if vendored.is_file():
        expected = json.loads(vendored.read_text())
        assert body["webr_version"] == expected["webr_version"]
        assert body["r_version"] == expected["r_version"]
    else:
        assert body["webr_version"] is None
        assert body["r_version"] is None


def test_combined_identity_keeps_its_flat_python_keys_and_gains_both_engines(client):
    """The flat keys are the Python engine's and predate the R one; client.ts
    reads them positionally. Adding a second engine must not move them."""
    body = client.get("/api/engine/identity").json()

    import ustat_engine
    from ustat_engine.fingerprint import source_fingerprint as py_fingerprint

    assert body["fingerprint"] == py_fingerprint()
    assert body["version"] == ustat_engine.__version__
    assert "stats.power" in body["analyses"]

    assert body["python"]["fingerprint"] == py_fingerprint()
    assert body["python"]["analyses"] == body["analyses"]
    assert body["r"]["fingerprint"] == source_fingerprint()
    assert "stats.ttest" in body["r"]["analyses"]


# ---------------------------------------------------------------------------
# 2. The fingerprint responds to source changes (on a throwaway copy)
# ---------------------------------------------------------------------------


def test_fingerprint_changes_when_source_bytes_change(tmp_path):
    dest_root = tmp_path / "root_a"
    dest_root.mkdir()
    package_name = "ustat_engine_r_copy_a"
    _copy_package(dest_root, package_name)

    module = _import_copy(dest_root, package_name)
    try:
        original = module.source_fingerprint()
        assert original is not None
        assert original == source_fingerprint(), (
            "an unmodified copy must fingerprint identically to the original tree"
        )

        target = dest_root / package_name / "runtime" / "registry.R"
        with target.open("ab") as fh:
            fh.write(b"\n")

        mutated = module.source_fingerprint()
        assert mutated is not None
        assert mutated != original
    finally:
        _unimport_copy(dest_root, package_name)

    assert package_name not in sys.modules
    assert str(dest_root) not in sys.path


def test_fingerprint_changes_when_a_file_is_renamed(tmp_path):
    """The relative path is hashed alongside the bytes, so moving code between
    modules changes what runs even when the tree's total bytes do not."""
    dest_root = tmp_path / "root_b"
    dest_root.mkdir()
    package_name = "ustat_engine_r_copy_b"
    package_path = _copy_package(dest_root, package_name)

    module = _import_copy(dest_root, package_name)
    try:
        original = module.source_fingerprint()
        assert original is not None

        (package_path / "runtime" / "text.R").rename(
            package_path / "runtime" / "text_renamed.R"
        )
        renamed = module.source_fingerprint()
    finally:
        _unimport_copy(dest_root, package_name)

    assert renamed is not None
    assert renamed != original


def test_the_two_engines_run_the_same_hashing_algorithm(tmp_path):
    """fingerprint.py is duplicated between the two engines on purpose (see its
    docstring). This is the check that the duplication has not drifted.

    They cannot be pointed at one tree and compared -- one walks .py and the
    other .R, and the relative path is part of the digest, so identical content
    under different names is *supposed* to hash differently. So both are checked
    against the algorithm written out longhand instead, which pins what the
    algorithm IS rather than that two copies of it coincide.
    """
    from ustat_engine import fingerprint as py_fp
    from ustat_engine_r import fingerprint as r_fp

    def expected(root: pathlib.Path, suffix: str) -> str:
        digest = hashlib.sha256()
        for path in sorted(root.rglob(f"*{suffix}")):
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    py_tree = tmp_path / "py"
    (py_tree / "sub").mkdir(parents=True)
    (py_tree / "a.py").write_bytes(b"one\n")
    (py_tree / "sub" / "b.py").write_bytes(b"two\n")

    r_tree = tmp_path / "r"
    (r_tree / "sub").mkdir(parents=True)
    (r_tree / "a.R").write_bytes(b"one\n")
    (r_tree / "sub" / "b.R").write_bytes(b"two\n")

    py_root, r_root = py_fp._PACKAGE_ROOT, r_fp._PACKAGE_ROOT
    try:
        py_fp._PACKAGE_ROOT = py_tree
        r_fp._PACKAGE_ROOT = r_tree
        assert py_fp.source_fingerprint() == expected(py_tree, ".py")
        assert r_fp.source_fingerprint() == expected(r_tree, ".R")
    finally:
        py_fp._PACKAGE_ROOT = py_root
        r_fp._PACKAGE_ROOT = r_root


# ---------------------------------------------------------------------------
# 3. The bundle builder
# ---------------------------------------------------------------------------


def test_bundle_build_is_byte_reproducible_and_manifest_matches_source(tmp_path):
    build_module = _load_build_module()

    manifest_1 = build_module.build(tmp_path / "out1")
    manifest_2 = build_module.build(tmp_path / "out2")

    bundle_1 = (tmp_path / "out1" / manifest_1["bundle"]).read_bytes()
    bundle_2 = (tmp_path / "out2" / manifest_2["bundle"]).read_bytes()

    assert bundle_1 == bundle_2
    assert hashlib.sha256(bundle_1).hexdigest() == manifest_1["bundle_sha256"]
    assert manifest_1 == manifest_2

    assert manifest_1["source_fingerprint"] == source_fingerprint()
    assert manifest_1["version"] == __version__
    assert manifest_1["bundle_bytes"] == len(bundle_1)


def test_bundle_carries_every_source_in_dependency_order(tmp_path):
    build_module = _load_build_module()
    manifest = build_module.build(tmp_path / "out")
    text = (tmp_path / "out" / manifest["bundle"]).read_text(encoding="ascii")

    # Match the builder's own separator line, not a bare filename: several
    # modules mention each other by path in a comment.
    def at(rel: str) -> int:
        marker = f"\n# {rel}\n"
        assert marker in text, f"{rel} missing from the bundle"
        return text.index(marker)

    # R has no imports: the bundle is one flat script, so registry.R must be
    # evaluated before any analyses/ file calls ustat_register at its own file
    # scope, and errors.R before anything that calls ustat_stop.
    assert at("runtime/errors.R") < at("runtime/registry.R")
    assert at("runtime/registry.R") < at("analyses/ttest.R")

    sources = sorted(PACKAGE_DIR.rglob("*.R"))
    positions = [at(p.relative_to(PACKAGE_DIR).as_posix()) for p in sources]
    assert len(set(positions)) == len(sources)
    assert manifest["modules"] == len(sources)


def test_bundle_build_refuses_a_runtime_module_it_was_not_told_about(tmp_path, monkeypatch):
    """A new runtime file landing at whatever position sorted() put it in would
    change the bundle without anyone choosing to. The builder has to notice."""
    build_module = _load_build_module()
    monkeypatch.setattr(build_module, "RUNTIME_ORDER", ("errors", "jsonsafe"))
    with pytest.raises(SystemExit) as exc:
        build_module.build(tmp_path / "out")
    assert "RUNTIME_ORDER" in str(exc.value)


def test_manifest_analyses_match_the_sources(tmp_path):
    build_module = _load_build_module()
    manifest = build_module.build(tmp_path / "out")

    declared = {a["id"]: a for a in analyses()}
    assert set(declared) == {a["id"] for a in manifest["analyses"]}

    ttest = next(a for a in manifest["analyses"] if a["id"] == "stats.ttest")
    assert ttest["needs_frame"] is True
    # The host must install these before calling in; the browser's boot plan is
    # built from this field, so a wrong answer is a crash at n >= 50.
    assert ttest["packages"] == ["moments", "nortest"]
    assert ttest["r_packages"] == ttest["packages"]


def test_the_committed_bundle_if_present_matches_the_sources():
    """frontend/public/webr/bundle/ is gitignored and rebuilt by `npm run
    r:bundle` on every dev start. If one is lying around, it has to be current
    -- a stale bundle is exactly what the fingerprint exists to catch."""
    out_dir = REPO_ROOT / "frontend" / "public" / "webr" / "bundle"
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("no bundle built yet (npm run r:bundle)")

    manifest = json.loads(manifest_path.read_text())
    bundle = (out_dir / manifest["bundle"]).read_bytes()
    assert hashlib.sha256(bundle).hexdigest() == manifest["bundle_sha256"]
    assert manifest["source_fingerprint"] == source_fingerprint(), (
        "the built bundle predates the current R sources -- run npm run r:bundle"
    )
