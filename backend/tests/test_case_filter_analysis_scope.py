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


def test_select_cases_reports_which_rows_it_drops(client):
    """The grid marks excluded rows rather than hiding them: cell edits address
    rows by POSITION in the unfiltered frame, so a grid that hid rows would
    write edits to the wrong ones. It cannot work out which rows those are
    without a second copy of the condition semantics, so the server says."""
    sid, _ = _case_filter_session("excluded")
    r = client.post(f"/api/sessions/{sid}/select_cases",
                    json={**_selection(), "apply": True})
    assert r.status_code == 200, r.text
    body = r.json()
    # KEEP is 1 for the first six rows and 0 for the last six.
    assert body["excluded_rows"] == [6, 7, 8, 9, 10, 11]
    assert body["excluded_beyond_preview"] == 0


def test_a_preview_run_still_reports_the_rows_it_would_drop(client):
    sid, _ = _case_filter_session("excluded_preview")
    r = client.post(f"/api/sessions/{sid}/select_cases",
                    json={**_selection(), "apply": False})
    assert r.json()["excluded_rows"] == [6, 7, 8, 9, 10, 11]
    # …without actually applying it.
    assert store.get_filter(sid) == []


def test_the_charts_a_figure_is_built_from_respect_the_filter(client):
    """A figure drawn from all the rows while the tests beside it used six is
    the kind of disagreement a reader cannot see."""
    sid, _ = _case_filter_session("charts")
    client.post(f"/api/sessions/{sid}/select_cases", json={**_selection(), "apply": True})

    box = client.post("/api/charts/boxplot", json={
        "session_id": sid, "x": "X", "color": "GROUP",
    })
    assert box.status_code == 200, box.text
    assert sum(len(g["values"]) for g in box.json()["groups"]) == 6

    bar = client.post("/api/charts/bar", json={"session_id": sid, "x": "GROUP"})
    assert sum(d["value"] for d in bar.json()["data"]) == 6

    grouped = client.post("/api/charts/bar", json={
        "session_id": sid, "x": "GROUP", "color": "KEEP",
    })
    assert sum(d["value"] for s in grouped.json()["series"] for d in s["data"]) == 6


def test_deleting_rows_while_a_filter_is_active_hits_the_rows_the_grid_showed(client):
    """The grid shows every row while a filter is active — excluded ones are
    marked, not hidden — so a tick is a position in the UNFILTERED frame, and
    that is what delete_rows takes. If it took a filtered position instead,
    ticking the third visible row would delete some other patient.
    """
    df = pd.DataFrame({"id": list(range(1, 11)), "keep": [1, 0] * 5})
    sid = make_session(df.copy(), "tcase_delete")
    client.post(f"/api/sessions/{sid}/select_cases", json={
        "conditions": [{"column": "keep", "operator": "eq", "value": "1", "join": "AND"}],
        "apply": True,
    })
    assert store.get_filtered(sid)["id"].tolist() == [1, 3, 5, 7, 9]

    # Tick the rows holding id 3 and id 4 — positions 2 and 3 as displayed.
    r = client.post(f"/api/compute/{sid}/delete_rows", json={"row_indices": [2, 3]})
    assert r.status_code == 200, r.text
    assert store.get(sid)["id"].tolist() == [1, 2, 5, 6, 7, 8, 9, 10]
    # The filter re-evaluates on values, so it simply no longer sees id 3.
    assert store.get_filtered(sid)["id"].tolist() == [1, 5, 7, 9]


def test_excluded_positions_are_restated_after_rows_move(client):
    """Excluded rows are POSITIONS, so a deletion shifts every one below it.
    The client re-asks rather than trying to adjust them itself; this pins that
    the server's answer is correct for the new frame."""
    df = pd.DataFrame({"id": list(range(1, 11)), "keep": [1, 0] * 5})
    sid = make_session(df.copy(), "tcase_restate")
    conditions = {"conditions": [
        {"column": "keep", "operator": "eq", "value": "1", "join": "AND"}]}
    first = client.post(f"/api/sessions/{sid}/select_cases",
                        json={**conditions, "apply": True}).json()
    assert first["excluded_rows"] == [1, 3, 5, 7, 9]

    client.post(f"/api/compute/{sid}/delete_rows", json={"row_indices": [0, 1]})
    again = client.post(f"/api/sessions/{sid}/select_cases",
                        json={**conditions, "apply": True}).json()
    # Two rows gone from the top: the excluded positions all shift down by two.
    assert again["excluded_rows"] == [1, 3, 5, 7]
    assert again["total"] == 8


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
