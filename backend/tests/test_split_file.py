"""Split File: the level a request states scopes the rows its analysis sees."""
import json
from urllib.parse import quote

import pandas as pd
import pytest
from scipy import stats

from services import store

DF = pd.DataFrame({
    "sex": ["F", "F", "F", "F", "M", "M", "M", "M", "M", None],
    "arm": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
    "sbp": [120.0, 131, 118, 140, 150, 128, 155, 133, 149, 170],
    "w": [1, 2, 1, 1, 3, 1, 1, 2, 1, 1],
})


def _sid(name: str) -> str:
    sid = f"split_{name}"
    store.save(sid, DF.copy())
    store.clear_filter(sid)
    store.clear_weight(sid)
    return sid


def _hdr(column: str, level: str) -> dict:
    return {"X-Ustat-Split": quote(json.dumps({"column": column, "level": level}))}


def test_an_analysis_sees_only_the_stated_level(client):
    sid = _sid("ttest")
    r = client.post("/api/stats/ttest", json={"session_id": sid, "column": "sbp", "group_column": "arm"},
                    headers=_hdr("sex", "M"))
    assert r.status_code == 200, r.text
    out = r.json()
    m = DF[DF.sex == "M"]
    welch = out["variance_assumption"] == "welch"
    expected = stats.ttest_ind(m[m.arm == "a"].sbp, m[m.arm == "b"].sbp, equal_var=not welch)
    assert out["n1"] + out["n2"] == 5
    assert out["p"] == pytest.approx(expected.pvalue, rel=1e-9)


def test_without_the_header_nothing_changes(client):
    sid = _sid("none")
    assert len(store.get_filtered(sid)) == 10


def test_order_is_filter_then_split_then_weights(client):
    sid = _sid("order")
    store.save_filter(sid, [{"column": "arm", "operator": "eq", "value": "a", "join": "AND"}])
    store.save_weight(sid, "w")
    r = client.post("/api/stats/ttest", json={"session_id": sid, "column": "sbp"},
                    headers=_hdr("sex", "M"))
    assert r.status_code == 200, r.text
    # Males in arm a: weights 3 + 1 + 1.
    assert r.json()["n"] == 5


def test_select_cases_counts_ignore_the_split(client):
    sid = _sid("counts")
    r = client.post(f"/api/sessions/{sid}/select_cases",
                    json={"conditions": [{"column": "arm", "operator": "eq", "value": "a", "join": "AND"}]},
                    headers=_hdr("sex", "F"))
    assert r.json()["selected"] == 5


@pytest.mark.parametrize("header,fragment", [
    ({"X-Ustat-Split": "%7Bnot json"}, "not valid JSON"),
    ({"X-Ustat-Split": quote(json.dumps({"column": "sex"}))}, "needs a column and a level"),
    (_hdr("nope", "x"), "not found"),
])
def test_bad_split_headers_are_422(client, header, fragment):
    sid = _sid("bad")
    r = client.post("/api/stats/ttest", json={"session_id": sid, "column": "sbp"}, headers=header)
    assert r.status_code == 422
    assert fragment in r.json()["detail"]


def test_split_levels_lists_groups_with_counts(client):
    sid = _sid("levels")
    r = client.get(f"/api/sessions/{sid}/split_levels", params={"column": "sex"})
    assert r.status_code == 200, r.text
    assert r.json() == {"column": "sex", "levels": [{"level": "F", "n": 4}, {"level": "M", "n": 5}], "n_missing": 1}


def test_too_many_levels_is_refused(client):
    sid = _sid("many")
    store.save(sid, pd.DataFrame({"id": range(30), "y": range(30)}))
    r = client.get(f"/api/sessions/{sid}/split_levels", params={"column": "id"})
    assert r.status_code == 422 and "at most 20" in r.json()["detail"]


def test_in_browser_frames_are_refused_under_a_split(client):
    sid = _sid("frame")
    r = client.get(f"/api/sessions/{sid}/frame", params={"columns": "sbp"}, headers=_hdr("sex", "F"))
    assert r.status_code == 409


def test_unicode_column_and_level_names(client):
    sid = "split_unicode"
    store.save(sid, pd.DataFrame({"cinsiyet": ["Kadın", "Kadın", "Erkek", "Erkek"], "y": [1.0, 2, 3, 5]}))
    store.clear_filter(sid)
    r = client.post("/api/stats/ttest", json={"session_id": sid, "column": "y", "mu": 0},
                    headers=_hdr("cinsiyet", "Kadın"))
    assert r.status_code == 200, r.text
    assert r.json()["n"] == 2
