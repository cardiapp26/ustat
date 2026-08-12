"""Select Cases must restrict every dataset-backed analysis."""

from pathlib import Path

import numpy as np
import pandas as pd

from conftest import make_session
from services import store


def _case_filter_session(suffix: str) -> tuple[str, pd.DataFrame]:
    x = np.arange(1, 13, dtype=float)
    df = pd.DataFrame({
        "KEEP": [1] * 6 + [0] * 6,
        "GROUP": [0, 1] * 6,
        "X": x,
        "Y": 3.0 + 1.75 * x + np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3] * 2),
        "MISS": x.copy(),
    })
    df.loc[[1, 4, 8], "MISS"] = np.nan
    sid = make_session(df.copy(), f"tcase_{suffix}")
    return sid, df


def _selection() -> dict:
    return {
        "conditions": [
            {"column": "KEEP", "operator": "eq", "value": "1", "join": "AND"},
        ],
    }


def test_select_cases_preview_does_not_apply_filter(client):
    sid, df = _case_filter_session("preview")
    response = client.post(
        f"/api/sessions/{sid}/select_cases",
        json={**_selection(), "apply": False},
    )
    assert response.status_code == 200, response.text
    assert response.json()["selected"] == 6
    assert response.json()["applied"] is False
    assert store.get_filter(sid) == []

    descriptive = client.get(f"/api/stats/{sid}/descriptive", params={"column": "X"})
    assert descriptive.status_code == 200, descriptive.text
    assert descriptive.json()["X"]["n"] == len(df)


def test_select_cases_restricts_analysis_families(client):
    sid, _ = _case_filter_session("scope")
    applied = client.post(
        f"/api/sessions/{sid}/select_cases",
        json={**_selection(), "apply": True},
    )
    assert applied.status_code == 200, applied.text
    assert applied.json()["selected"] == 6
    assert applied.json()["applied"] is True

    descriptive = client.get(f"/api/stats/{sid}/descriptive", params={"column": "X"})
    assert descriptive.status_code == 200, descriptive.text
    assert descriptive.json()["X"]["n"] == 6

    ttest = client.post("/api/stats/ttest", json={
        "session_id": sid,
        "column": "Y",
        "group_column": "GROUP",
    })
    assert ttest.status_code == 200, ttest.text
    assert ttest.json()["n1"] + ttest.json()["n2"] == 6

    linear = client.post("/api/models/linear", json={
        "session_id": sid,
        "outcome": "Y",
        "predictors": ["X"],
    })
    assert linear.status_code == 200, linear.text
    assert linear.json()["n"] == 6

    scatter = client.post("/api/charts/scatter", json={
        "session_id": sid,
        "x": "X",
        "y": "Y",
    })
    assert scatter.status_code == 200, scatter.text
    assert len(scatter.json()["points"]) == 6

    missingness = client.post(f"/api/compute/{sid}/missing_diagnostics", json={
        "columns": ["MISS"],
    })
    assert missingness.status_code == 200, missingness.text
    assert missingness.json()["columns"][0]["n_missing"] == 2
    assert missingness.json()["columns"][0]["pct"] == 33.3


def test_analysis_routers_do_not_use_unfiltered_store_access():
    routers = Path(__file__).resolve().parents[1] / "routers"
    # These modules intentionally manage or mutate the complete dataset.
    # merge.py joins onto the whole sheet on purpose: reading the filtered
    # view and saving the result would delete every excluded row, turning a
    # display filter into permanent data loss.
    allowed_unfiltered = {"session.py", "compute.py", "pub_export.py", "merge.py"}
    violations = []
    for path in routers.rglob("*.py"):
        if path.name in allowed_unfiltered:
            continue
        if "store.get(" in path.read_text(encoding="utf-8"):
            violations.append(str(path.relative_to(routers)))
    assert violations == [], f"Analysis routers bypass Select Cases: {violations}"
