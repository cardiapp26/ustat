"""The prep recipe: middleware/prep_steps.py + store.log_step/get_steps.

Contract: every successful data-mutating request lands in the session's step
list with its FULL request params; failed requests, read endpoints and
analysis endpoints leave no step; the recipe rides the project file and
seeds the restored session.
"""
import json

import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store

client = TestClient(app)


def _session(sid: str) -> str:
    store.save(sid, pd.DataFrame({"age": [30.0, 41.0, 55.0], "group": ["a", "b", "a"]}))
    return sid


def test_cell_edit_is_recorded_with_full_params():
    sid = _session("steps_cell")
    r = client.patch(
        f"/api/sessions/{sid}/cell",
        json={"session_id": sid, "row_index": 1, "column": "age", "value": 42},
    )
    assert r.status_code == 200
    steps = store.get_steps(sid)
    assert len(steps) == 1
    assert steps[0]["op"] == "sessions/edit_cell"
    assert steps[0]["params"]["row_index"] == 1
    assert steps[0]["params"]["column"] == "age"
    assert steps[0]["params"]["value"] == 42
    assert "session_id" not in steps[0]["params"]


def test_compute_op_takes_route_slug_and_full_body():
    sid = _session("steps_formula")
    r = client.post(
        f"/api/compute/{sid}/formula",
        json={"new_col": "age2", "formula": "age * 2"},
    )
    assert r.status_code == 200
    steps = store.get_steps(sid)
    assert steps[-1]["op"] == "compute/formula"
    assert steps[-1]["params"]["formula"] == "age * 2"


def test_failed_mutation_records_nothing():
    sid = _session("steps_fail")
    client.post(f"/api/compute/{sid}/formula", json={"new_col": "x", "formula": "no_such * 2"})
    assert store.get_steps(sid) == []


def test_analysis_and_read_endpoints_record_nothing():
    sid = _session("steps_reads")
    client.get(f"/api/stats/{sid}/descriptive")
    client.post(f"/api/compute/{sid}/missing_diagnostics", json={"columns": ["age"]})
    assert store.get_steps(sid) == []


def test_delete_column_path_param_lands_in_step():
    sid = _session("steps_delcol")
    r = client.delete(f"/api/compute/{sid}/column/group")
    assert r.status_code == 200
    steps = store.get_steps(sid)
    assert steps[-1]["op"] == "compute/delete_column"
    assert steps[-1]["params"]["column"] == "group"


def test_recipe_rides_the_project_file_and_seeds_the_restore():
    sid = _session("steps_roundtrip")
    client.patch(
        f"/api/sessions/{sid}/cell",
        json={"session_id": sid, "row_index": 0, "column": "age", "value": 31},
    )
    client.post(
        f"/api/sessions/{sid}/select_cases",
        json={"conditions": [{"column": "group", "operator": "eq", "value": "a"}], "apply": True},
    )

    saved = client.get(f"/api/project/{sid}/save")
    import io, zipfile
    zf = zipfile.ZipFile(io.BytesIO(saved.content))
    steps = json.loads(zf.read("prep/steps.json"))
    ops = [s["op"] for s in steps]
    assert ops == ["sessions/edit_cell", "sessions/select_cases"]
    # The middleware recorded the filter itself: no synthetic fallback step.
    assert "filter" not in ops

    r2 = client.post(
        "/api/project/load",
        files={"file": ("p.ustat", saved.content, "application/zip")},
    )
    new_sid = r2.json()["session_id"]
    assert [s["op"] for s in store.get_steps(new_sid)] == ops
    assert store.get_filter(new_sid) == [{"column": "group", "operator": "eq", "value": "a"}]
