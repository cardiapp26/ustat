"""The missingness dependence scan must adjust for its ~30 tests per variable.

Unadjusted, crp, hdl and ldl (missing completely at random by construction in
the walkthrough dataset) were each reported as "MAR".
"""

from pathlib import Path

import numpy as np
import pandas as pd
from conftest import make_session
from fastapi.testclient import TestClient

from main import app

DATASET = Path(__file__).resolve().parents[2] / "qa" / "full_walkthrough_100x40" / "dataset.csv"


def _diag(df: pd.DataFrame, cols: list[str], name: str) -> dict:
    sid = make_session(df, name)
    r = TestClient(app).post(f"/api/compute/{sid}/missing_diagnostics", json={"columns": cols})
    assert r.status_code == 200, r.text
    return {c["name"]: c for c in r.json()["columns"]} | {"_body": r.json()}


def test_mcar_columns_are_not_classified_as_mar():
    out = _diag(pd.read_csv(DATASET), ["crp", "hdl", "ldl"], "holm_mcar")
    for col in ("crp", "hdl", "ldl"):
        assert out[col]["depends_on"] == []
        assert out[col]["likely"] in ("MCAR-consistent", "possible MAR")
        assert out[col]["n_tests"] > 20
    assert out["_body"]["any_mar"] is False


def test_strong_mar_dependence_survives_the_adjustment():
    rng = np.random.default_rng(4)
    n = 300
    age = rng.normal(60, 10, n)
    other = rng.normal(0, 1, (n, 25))
    y = rng.normal(0, 1, n)
    y[age > 66] = np.nan  # missing whenever older: unmistakable MAR
    df = pd.DataFrame(other, columns=[f"x{i}" for i in range(25)]).assign(age=age, y=y)
    out = _diag(df, ["y"], "holm_mar")
    assert out["y"]["depends_on"] == ["age"]
    assert out["y"]["likely"] == "MAR"
