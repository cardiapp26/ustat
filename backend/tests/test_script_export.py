"""The generated replay script: services/script_export.py + its endpoint.

The strongest single check is ast.parse: whatever the recipe contained, the
generated text must be valid Python. Content checks then pin the structure:
import first, steps in order, export at the end, analyses as definitions.
"""
import ast

import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services import store
from services.script_export import generate_python_script

client = TestClient(app)


def _steps():
    return [
        {"op": "import", "params": {"filename": "trial.csv"}, "t": 1.0},
        {"op": "compute/formula", "params": {"new_col": "bmi", "formula": "kg / (m*m)"}, "t": 2.0},
        {"op": "sessions/edit_cell", "params": {"row_index": 3, "column": "age", "value": 42}, "t": 3.0},
        {"op": "compute/delete_column", "params": {"column": "scratch col"}, "t": 4.0},
        {"op": "sessions/select_cases", "params": {"conditions": [{"column": "g", "operator": "eq", "value": "a"}], "apply": True}, "t": 5.0},
    ]


def test_script_is_valid_python_and_ordered():
    script = generate_python_script(_steps(), project_name="trial")
    ast.parse(script)  # must compile, whatever the recipe held
    assert 'RAW_FILE = "trial.csv"' in script
    upload = script.index("/api/upload/")
    formula = script.index("/api/compute/{sid}/formula")
    cell = script.index("/api/sessions/{sid}/cell")
    delete = script.index("/api/compute/{sid}/column/scratch%20col")
    select = script.index("/api/sessions/{sid}/select_cases")
    export = script.index("/api/sessions/{sid}/export/csv")
    assert upload < formula < cell < delete < select < export
    assert '"formula": "kg / (m*m)"' in script


def test_unknown_op_becomes_manual_note_not_broken_code():
    steps = [{"op": "someday/new_op", "params": {"x": 1}, "t": 1.0}]
    script = generate_python_script(steps)
    ast.parse(script)
    assert "no replay route known" in script
    assert "someday/new_op" in script
    assert "had no replay route" in script


def test_analyses_listed_as_definitions():
    analyses = [
        {
            "id": "a1", "name": "Model 1", "panel": "models", "tab": "models", "createdAt": 1,
            "snapshot": {"result": {"or": 2.0}, "stamp": {"dataVersion": 0, "paramsKey": '{"y":"DM"}'}},
        }
    ]
    script = generate_python_script([], analyses=analyses)
    ast.parse(script)
    assert "Model 1 (panel: models)" in script
    assert '{"y":"DM"}' in script
    # Definitions, not guessed calls: no request is generated for an analysis.
    assert "models/logistic" not in script


def test_endpoint_serves_script_from_recorded_steps():
    sid = "script_ep"
    store.save(sid, pd.DataFrame({"age": [30.0, 41.0]}))
    store.set_filename(sid, "trial dataset")
    client.post(f"/api/compute/{sid}/formula", json={"new_col": "age2", "formula": "age * 2"})

    r = client.post(f"/api/project/{sid}/script", json={"ui_state": None})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/x-python")
    ast.parse(r.text)
    assert "Project: trial dataset" in r.text
    assert "/api/compute/{sid}/formula" in r.text

    r2 = client.post(f"/api/project/{sid}/script?lang=r", json={"ui_state": None})
    assert r2.status_code == 400

    r3 = client.post("/api/project/nope/script", json={"ui_state": None})
    assert r3.status_code == 404
