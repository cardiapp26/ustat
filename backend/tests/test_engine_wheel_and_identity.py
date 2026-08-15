"""Proof that the server, the wheel, and the /api/engine/identity endpoint
all agree on what code the engine actually is.

The design (see backend/ustat_engine/fingerprint.py and
scripts/build_engine_wheel.py) rests on a single claim: a sha256 over the
engine's own .py sources, by relative path and content, identifies the code
byte-for-byte -- not "the same release", the same bytes. These tests check
that the claim holds in both directions:

  - the fingerprint actually moves when the source moves (it is not a
    constant dressed up as a hash);
  - the wheel builder reproduces that exact fingerprint and produces
    byte-identical archives across runs;
  - the archive it produces is a real, installable, pure-Python wheel.

Everything here works on copies under tmp_path. Nothing touches the real
backend/ustat_engine package on disk.
"""
from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

import ustat_engine
from ustat_engine.fingerprint import source_fingerprint

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_DIR = REPO_ROOT / "backend" / "ustat_engine"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_engine_wheel.py"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _load_build_module():
    """Import scripts/build_engine_wheel.py as a module, without touching sys.path."""
    spec = importlib.util.spec_from_file_location("_build_engine_wheel_under_test", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _copy_package(dest_root: Path, package_name: str) -> Path:
    """Copy backend/ustat_engine into dest_root/<package_name> and return that path."""
    dest = dest_root / package_name
    shutil.copytree(PACKAGE_DIR, dest)
    return dest


def _import_copy(dest_root: Path, package_name: str):
    """Import the copied package under `package_name`, with dest_root on sys.path.

    Returns the imported module. Caller is responsible for cleanup via
    `_unimport_copy`.
    """
    sys.path.insert(0, str(dest_root))
    try:
        module = importlib.import_module(package_name)
    except BaseException:
        sys.path.remove(str(dest_root))
        raise
    return module


def _unimport_copy(dest_root: Path, package_name: str) -> None:
    """Undo `_import_copy`: drop the path entry and every submodule from sys.modules."""
    path_str = str(dest_root)
    if path_str in sys.path:
        sys.path.remove(path_str)
    for name in list(sys.modules):
        if name == package_name or name.startswith(package_name + "."):
            del sys.modules[name]


# ---------------------------------------------------------------------------
# 1. GET /api/engine/identity
# ---------------------------------------------------------------------------


def test_engine_identity_endpoint_shape(client):
    response = client.get("/api/engine/identity")
    assert response.status_code == 200

    body = response.json()
    assert set(["version", "fingerprint", "modules", "analyses"]).issubset(body.keys())

    assert isinstance(body["fingerprint"], str)
    assert _HEX64.match(body["fingerprint"]), body["fingerprint"]

    assert isinstance(body["analyses"], list)
    assert "stats.power" in body["analyses"]

    assert isinstance(body["modules"], int)
    assert body["modules"] > 0


def test_engine_identity_fingerprint_matches_in_process_computation(client):
    response = client.get("/api/engine/identity")
    body = response.json()

    assert body["fingerprint"] == source_fingerprint()
    assert body["version"] == ustat_engine.__version__


# ---------------------------------------------------------------------------
# 2. The fingerprint responds to source changes (on a throwaway copy)
# ---------------------------------------------------------------------------


def test_fingerprint_changes_when_source_bytes_change(tmp_path):
    dest_root = tmp_path / "root_a"
    dest_root.mkdir()
    package_name = "ustat_engine_copy_a"
    _copy_package(dest_root, package_name)

    module = _import_copy(dest_root, package_name)
    try:
        original_fingerprint = module.source_fingerprint()
        assert original_fingerprint is not None
        assert original_fingerprint == source_fingerprint(), (
            "an unmodified copy must fingerprint identically to the original package"
        )

        # Mutate one .py file in the copy -- append a single byte. The real
        # backend/ustat_engine package on disk is never touched.
        target = dest_root / package_name / "registry.py"
        with target.open("ab") as fh:
            fh.write(b"\n")

        mutated_fingerprint = module.source_fingerprint()
        assert mutated_fingerprint is not None
        assert mutated_fingerprint != original_fingerprint
    finally:
        _unimport_copy(dest_root, package_name)

    assert package_name not in sys.modules
    assert str(dest_root) not in sys.path


# ---------------------------------------------------------------------------
# 3. The fingerprint is path-sensitive, not just content-sensitive
# ---------------------------------------------------------------------------


def test_fingerprint_changes_when_a_file_is_renamed(tmp_path):
    """fingerprint.py hashes each file's relative path along with its bytes,
    so renaming a file changes the fingerprint even though the package's
    total bytes are unchanged.
    """
    dest_root = tmp_path / "root_b"
    dest_root.mkdir()
    package_name = "ustat_engine_copy_b"
    package_path = _copy_package(dest_root, package_name)

    # Import once. `source_fingerprint()` only walks the directory tree and
    # reads bytes -- it does not import the engine's submodules -- so we can
    # call it again on the same module object after renaming a file on disk,
    # even though renaming registry.py would otherwise break `__init__.py`'s
    # `from .registry import ...` if the package were re-imported.
    module = _import_copy(dest_root, package_name)
    try:
        original_fingerprint = module.source_fingerprint()
        assert original_fingerprint is not None

        # Rename a file inside the copy -- same bytes, different relative path.
        renamed_from = package_path / "registry.py"
        renamed_to = package_path / "registry_renamed.py"
        renamed_from.rename(renamed_to)

        renamed_fingerprint = module.source_fingerprint()
    finally:
        _unimport_copy(dest_root, package_name)

    assert renamed_fingerprint is not None
    assert renamed_fingerprint != original_fingerprint

    assert package_name not in sys.modules
    assert str(dest_root) not in sys.path


# ---------------------------------------------------------------------------
# 4. The wheel builder is reproducible
# ---------------------------------------------------------------------------


def test_build_is_byte_reproducible_and_manifest_matches_source(tmp_path):
    build_module = _load_build_module()

    out_dir_1 = tmp_path / "out1"
    out_dir_2 = tmp_path / "out2"

    manifest_1 = build_module.build(out_dir_1)
    manifest_2 = build_module.build(out_dir_2)

    wheel_path_1 = out_dir_1 / manifest_1["wheel"]
    wheel_path_2 = out_dir_2 / manifest_2["wheel"]
    assert wheel_path_1.is_file()
    assert wheel_path_2.is_file()

    sha_1 = hashlib.sha256(wheel_path_1.read_bytes()).hexdigest()
    sha_2 = hashlib.sha256(wheel_path_2.read_bytes()).hexdigest()
    assert sha_1 == sha_2
    assert manifest_1["wheel_sha256"] == sha_2

    assert manifest_1["source_fingerprint"] == source_fingerprint()
    assert manifest_2["source_fingerprint"] == source_fingerprint()


# ---------------------------------------------------------------------------
# 5. The wheel is a valid, installable, pure-Python wheel
# ---------------------------------------------------------------------------


def test_built_wheel_is_a_valid_pure_python_wheel(tmp_path):
    build_module = _load_build_module()
    out_dir = tmp_path / "out"
    manifest = build_module.build(out_dir)

    wheel_path = out_dir / manifest["wheel"]
    version = manifest["version"]
    dist_info = f"ustat_engine-{version}.dist-info"

    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())

        assert "ustat_engine/__init__.py" in names
        assert f"{dist_info}/METADATA" in names
        assert f"{dist_info}/WHEEL" in names
        assert f"{dist_info}/RECORD" in names

        wheel_metadata = zf.read(f"{dist_info}/WHEEL").decode("utf-8")
        assert "Root-Is-Purelib: true" in wheel_metadata
        assert "Tag: py3-none-any" in wheel_metadata

        for name in names:
            for ext in (".so", ".pyd", ".dylib"):
                assert not name.endswith(ext), f"native extension found in wheel: {name}"

        record_text = zf.read(f"{dist_info}/RECORD").decode("utf-8")
        record_path = f"{dist_info}/RECORD"

        checked_any = False
        for line in record_text.splitlines():
            if not line:
                continue
            name, hash_field, size_field = line.rsplit(",", 2)
            if name == record_path:
                # RECORD lists itself with an empty hash and size, by convention.
                assert hash_field == ""
                assert size_field == ""
                continue

            assert name in names, f"RECORD references missing entry: {name}"
            data = zf.read(name)

            assert size_field == str(len(data))

            assert hash_field.startswith("sha256=")
            encoded = hash_field[len("sha256="):]
            expected_digest = hashlib.sha256(data).digest()
            # urlsafe-base64 of the sha256 digest with '=' padding stripped.
            expected_encoded = base64.urlsafe_b64encode(expected_digest).rstrip(b"=").decode("ascii")
            assert encoded == expected_encoded

            checked_any = True

        assert checked_any, "RECORD had no non-self entries to verify"
