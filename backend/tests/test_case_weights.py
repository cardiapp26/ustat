"""Weight Cases (SPSS frequency weights) applied once for every analysis.

The proof that matters: a weighted session answers exactly as the same data
written out row by row does.
"""
import numpy as np
import pandas as pd
import pytest

from services import case_weights, store

AGG = pd.DataFrame({
    "group": ["A", "A", "A", "B", "B", "B"],
    "sbp": [120.0, 135.0, 150.0, 128.0, 142.0, 160.0],
    "count": [3, 1, 2, 2, 4, 1],
})


def _long(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.index.repeat(df["count"])].reset_index(drop=True)


def _sid(name: str, df: pd.DataFrame = AGG) -> str:
    sid = f"weights_{name}"
    store.save(sid, df.copy())
    store.clear_filter(sid)
    store.clear_weight(sid)
    return sid


# ── the rule ────────────────────────────────────────────────────────────────

def test_rows_are_replicated_and_bad_weights_dropped():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "w": [2, 0, np.nan, 3]})
    out = case_weights.apply_weights(df, "w")
    assert out["x"].tolist() == [1, 1, 4, 4, 4]
    # Labels survive, so a row still names the stored row it came from.
    assert out.index.tolist() == [0, 0, 3, 3, 3]


@pytest.mark.parametrize("values,fragment", [
    ([1.5, 2], "fractional"),
    (["a", 2], "non-numeric"),
    ([0, -1], "no positive"),
])
def test_unusable_weights_are_refused(values, fragment):
    with pytest.raises(case_weights.CaseWeightError, match=fragment):
        case_weights.summarize(pd.DataFrame({"w": values}), "w")


def test_replication_is_capped(monkeypatch):
    monkeypatch.setattr(case_weights, "MAX_WEIGHTED_ROWS", 10)
    with pytest.raises(case_weights.CaseWeightError, match="add up to 11"):
        case_weights.summarize(pd.DataFrame({"w": [5, 6]}), "w")


# ── through the API ────────────────────────────────────────────────────────

def test_weighted_ttest_equals_the_long_data(client):
    sid = _sid("ttest")
    long_sid = _sid("ttest_long", _long(AGG))
    r = client.post(f"/api/sessions/{sid}/weight_cases", json={"column": "count"})
    assert r.status_code == 200, r.text
    assert r.json() == {"column": "count", "n_rows": 6, "n_excluded": 0, "sum_weights": 13}

    body = {"column": "sbp", "group_column": "group"}
    weighted = client.post("/api/stats/ttest", json={"session_id": sid, **body}).json()
    long = client.post("/api/stats/ttest", json={"session_id": long_sid, **body}).json()
    assert weighted["p"] == pytest.approx(long["p"], rel=1e-12)
    assert weighted["t"] == pytest.approx(long["t"], rel=1e-12)
    assert weighted["n1"] + weighted["n2"] == 13


def test_select_cases_applies_before_weights(client):
    sid = _sid("order")
    client.post(f"/api/sessions/{sid}/select_cases", json={
        "conditions": [{"column": "group", "operator": "eq", "value": "B", "join": "AND"}]})
    r = client.post(f"/api/sessions/{sid}/weight_cases", json={"column": "count"})
    assert r.json()["sum_weights"] == 7
    assert len(store.get_filtered(sid)) == 7
    assert len(store.get_filtered(sid, weighted=False)) == 3


def test_fractional_column_is_refused_at_the_endpoint(client):
    sid = _sid("frac", AGG.assign(count=[1.5, 1, 1, 1, 1, 1]))
    r = client.post(f"/api/sessions/{sid}/weight_cases", json={"column": "count"})
    assert r.status_code == 422
    assert "survey weights" in r.json()["detail"]
    assert store.get_weight(sid) is None


def test_clearing_returns_to_unweighted(client):
    sid = _sid("clear")
    client.post(f"/api/sessions/{sid}/weight_cases", json={"column": "count"})
    assert client.delete(f"/api/sessions/{sid}/weight_cases").status_code == 200
    assert len(store.get_filtered(sid)) == 6


def test_weights_that_turn_invalid_block_analyses_but_not_the_session(client):
    sid = _sid("invalid")
    client.post(f"/api/sessions/{sid}/weight_cases", json={"column": "count"})
    store.save(sid, AGG.assign(count=[1.5, 1, 1, 1, 1, 1]))
    r = client.post("/api/stats/ttest", json={"session_id": sid, "column": "sbp", "group_column": "group"})
    assert r.status_code == 422
    assert "Weight Cases is on" in r.json()["detail"]
    info = client.get(f"/api/sessions/{sid}").json()
    assert info["case_weight"]["column"] == "count"
    assert "fractional" in info["case_weight"]["error"]


def test_rename_and_drop_follow_the_weight_column(client):
    sid = _sid("rename")
    client.post(f"/api/sessions/{sid}/weight_cases", json={"column": "count"})
    r = client.post(f"/api/compute/{sid}/rename", json={"old_name": "count", "new_name": "n"})
    assert r.status_code == 200, r.text
    assert store.get_weight(sid) == "n"
    assert client.delete(f"/api/compute/{sid}/column/n").status_code == 200
    assert store.get_weight(sid) is None


def test_in_browser_frames_are_refused_while_weighted(client):
    sid = _sid("frame")
    client.post(f"/api/sessions/{sid}/weight_cases", json={"column": "count"})
    r = client.get(f"/api/sessions/{sid}/frame", params={"columns": "sbp,group"})
    assert r.status_code == 409


def test_project_file_round_trip_keeps_the_weight(client):
    sid = _sid("project")
    client.post(f"/api/sessions/{sid}/weight_cases", json={"column": "count"})
    saved = client.get(f"/api/project/{sid}/save")
    assert saved.status_code == 200
    loaded = client.post("/api/project/load", files={"file": ("w.ustat", saved.content, "application/zip")})
    assert loaded.status_code == 200, loaded.text
    body = loaded.json()
    assert body["case_weight"]["sum_weights"] == 13
    assert store.get_weight(body["session_id"]) == "count"


def test_method_appendix_states_the_weighting(client):
    sid = _sid("appendix")
    client.post(f"/api/sessions/{sid}/weight_cases", json={"column": "count"})
    client.post("/api/stats/ttest", json={"session_id": sid, "column": "sbp", "group_column": "group"})
    r = client.post("/api/pub_export/method_appendix", json={"session_id": sid})
    assert r.status_code == 200, r.text
    import io
    import docx
    text = " ".join(p.text for p in docx.Document(io.BytesIO(r.content)).paragraphs)
    assert "weighted by 'count' as frequency weights" in text
