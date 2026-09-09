"""Round-trip tests for the .ustat project file (Phase 1 of the design).

The contract under test, from docs/DESIGN_project_file.md:

- A project saves as a zip with named JSON parts (manifest, data, dictionary,
  prep steps, audit) and loads back with no loss of meaning: kinds, labels,
  value labels, decimals, case filter, audit trail, ingest originals.
- The manifest carries schema_version, engine identity and a sha256 of the
  dataset part; a dataset that does not match its recorded hash is refused.
- A file written by a newer major schema is refused with a clear message, not
  half-loaded.
- Legacy v1.x save_session JSON files load through the same endpoint forever.
"""
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store
from services.project_file import SCHEMA_VERSION, build_project

client = TestClient(app)


def _make_session(sid: str = "proj_sess") -> str:
    df = pd.DataFrame(
        {
            "age": [30.5, 41.0, np.nan, 55.25],
            "group": ["a", "b", "a", "b"],
            "score": [1, 2, 3, 4],
        }
    )
    store.save(sid, df)
    store.set_filename(sid, "trial dataset")
    store.set_kind_overrides(sid, {"score": "categorical"})
    store.save_metadata(
        sid,
        {
            "group": {"label": "Treatment group", "value_labels": {"a": "Arm A", "b": "Arm B"}},
            "age": {"unit": "years", "missing_codes": [999]},
        },
    )
    store.save_decimals(sid, {"age": 1})
    store.save_filter(sid, [{"column": "group", "op": "==", "value": "a"}])
    store.log_action(sid, "upload", {"filename": "trial.csv"})
    store.save_ingest_report(
        sid,
        {"n_columns_examined": 3, "columns": []},
        {"age": {2: "<0.1"}},
    )
    return sid


def _zip_parts(content: bytes) -> dict:
    zf = zipfile.ZipFile(io.BytesIO(content))
    return {name: zf.read(name) for name in zf.namelist()}


# ── Container shape ──────────────────────────────────────────────────────────

def test_build_produces_expected_parts():
    sid = _make_session("proj_parts")
    parts = _zip_parts(build_project(sid))
    for required in ("manifest.json", "data/dataset.json", "dictionary.json", "audit.json"):
        assert required in parts, f"missing part {required}"
    # Filter exists on this session, so the prep recipe must carry it.
    assert "prep/steps.json" in parts
    # Ingest originals exist, so they must survive.
    assert "data/originals.json" in parts


def test_manifest_identity_and_hash():
    sid = _make_session("proj_manifest")
    parts = _zip_parts(build_project(sid))
    manifest = json.loads(parts["manifest.json"])
    assert manifest["kind"] == "ustat-project"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["filename"] == "trial dataset"
    # Engine identity comes from runtime_identity(): the environment at save.
    assert manifest["engine"]["engine"] == "python"
    assert manifest["engine"]["packages"]
    digest = hashlib.sha256(parts["data/dataset.json"]).hexdigest()
    assert manifest["data_hash"] == f"sha256:{digest}"


def test_dictionary_carries_data_semantics():
    sid = _make_session("proj_dict")
    parts = _zip_parts(build_project(sid))
    dictionary = json.loads(parts["dictionary.json"])
    cols = {c["name"]: c for c in dictionary["columns"]}
    assert cols["score"]["kind"] == "categorical"  # user override wins
    assert cols["group"]["metadata"]["value_labels"] == {"a": "Arm A", "b": "Arm B"}
    assert cols["age"]["metadata"]["missing_codes"] == [999]
    assert cols["age"]["decimals"] == 1


# ── Round trip ───────────────────────────────────────────────────────────────

def test_save_load_round_trip_preserves_meaning():
    sid = _make_session("proj_rt")
    r = client.get(f"/api/project/{sid}/save")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert ".ustat" in r.headers["content-disposition"]

    r2 = client.post(
        "/api/project/load",
        files={"file": ("trial.ustat", r.content, "application/zip")},
    )
    assert r2.status_code == 200
    body = r2.json()
    new_sid = body["session_id"]
    assert new_sid != sid
    assert body["filename"] == "trial dataset"
    assert body["rows"] == 4

    df = store.get(new_sid)
    assert list(df.columns) == ["age", "group", "score"]
    assert df["age"].dtype.kind == "f"

    assert store.get_kind_overrides(new_sid) == {"score": "categorical"}
    meta = store.get_metadata(new_sid)
    assert meta["group"]["value_labels"] == {"a": "Arm A", "b": "Arm B"}
    assert meta["age"]["missing_codes"] == [999]
    assert store.get_decimals(new_sid) == {"age": 1}
    assert store.get_filter(new_sid) == [{"column": "group", "op": "==", "value": "a"}]
    assert store.get_filename(new_sid) == "trial dataset"
    audit = store.get_audit(new_sid)
    assert any(e["action"] == "upload" for e in audit)
    preserved = store.get_preserved_cells(new_sid)
    assert preserved["age"][2] == "<0.1"


def test_ui_state_round_trip_via_post_save():
    sid = _make_session("proj_ui")
    ui_state = {
        "dataVersion": 7,
        "panelCache": {
            "models": {
                "result": {"or": 1.4},
                "stamp": {"dataVersion": 7, "filterKey": "none", "paramsKey": "{}", "engine": "python"},
            }
        },
    }
    r = client.post(f"/api/project/{sid}/save", json={"ui_state": ui_state})
    assert r.status_code == 200
    parts = _zip_parts(r.content)
    assert "ui/state.json" in parts
    manifest = json.loads(parts["manifest.json"])
    assert "ui/state.json" in manifest["parts"]

    r2 = client.post(
        "/api/project/load",
        files={"file": ("trial.ustat", r.content, "application/zip")},
    )
    assert r2.status_code == 200
    assert r2.json()["ui_state"] == ui_state


def test_get_save_has_no_ui_state_part():
    sid = _make_session("proj_no_ui")
    parts = _zip_parts(client.get(f"/api/project/{sid}/save").content)
    assert "ui/state.json" not in parts


def test_legacy_v13_json_returns_ui_state():
    payload = {
        "version": "1.3",
        "filename": "with ui",
        "data": [{"x": 1}, {"x": 2}],
        "ui_state": {"dataVersion": 3, "panelCache": {"roc": {"result": {"auc": 0.8}}}},
    }
    r = client.post(
        "/api/project/load",
        files={"file": ("s.json", json.dumps(payload).encode(), "application/json")},
    )
    assert r.status_code == 200
    assert r.json()["ui_state"]["panelCache"]["roc"]["result"]["auc"] == 0.8


# ── Integrity and version gates ──────────────────────────────────────────────

def _tampered(content: bytes, part: str, new_bytes: bytes) -> bytes:
    parts = _zip_parts(content)
    parts[part] = new_bytes
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in parts.items():
            zf.writestr(name, data)
    return out.getvalue()


def test_dataset_hash_mismatch_refused():
    sid = _make_session("proj_tamper")
    content = build_project(sid)
    bad = _tampered(content, "data/dataset.json", b"[{\"age\": 1}]")
    r = client.post("/api/project/load", files={"file": ("x.ustat", bad, "application/zip")})
    assert r.status_code == 400
    assert "hash" in r.json()["detail"].lower()


def test_newer_major_schema_refused():
    sid = _make_session("proj_newer")
    content = build_project(sid)
    parts = _zip_parts(content)
    manifest = json.loads(parts["manifest.json"])
    manifest["schema_version"] = "99.0.0"
    bad = _tampered(content, "manifest.json", json.dumps(manifest).encode())
    r = client.post("/api/project/load", files={"file": ("x.ustat", bad, "application/zip")})
    assert r.status_code == 400
    assert "newer" in r.json()["detail"].lower()


def test_not_a_zip_and_not_json_refused():
    r = client.post("/api/project/load", files={"file": ("x.ustat", b"\x00\x01garbage", "application/zip")})
    assert r.status_code == 400


# ── Legacy v1.x import ───────────────────────────────────────────────────────

def test_legacy_v12_json_loads_as_project():
    with open("tests/fixtures/projects/legacy_v12.json", "rb") as fh:
        raw = fh.read()
    r = client.post("/api/project/load", files={"file": ("old.json", raw, "application/json")})
    assert r.status_code == 200
    body = r.json()
    sid = body["session_id"]
    df = store.get(sid)
    assert list(df.columns) == ["age", "sex"]
    # Numeric restored despite JSON stringification, as load_session does.
    assert df["age"].dtype.kind in ("i", "f")
    assert store.get_kind_overrides(sid) == {"sex": "categorical"}
    assert store.get_decimals(sid) == {"age": 2}
    assert store.get_filter(sid) == [{"column": "sex", "op": "==", "value": "f"}]
    assert store.get_filename(sid) == "legacy study"
