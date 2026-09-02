"""Icon array (waffle) and the oncology waterfall plot.

The waffle's contract is that its cells are whole, sum to the grid exactly,
and each stands for a real share of real people. The waterfall's is that the
bars are sorted, every subject with a value is drawn, the ones without are
counted, and a threshold is counted on the side it points to.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture()
def cohort() -> str:
    rng = np.random.default_rng(12)
    n = 391
    df = pd.DataFrame(
        {
            "histology": rng.choice(["Benign", "Malignant", "Uncertain"], n, p=[0.494, 0.274, 0.232]),
            "change": np.concatenate([rng.normal(-40, 25, 200), rng.normal(15, 20, 180), [np.nan] * 11]),
            "response": ["PR"] * 200 + ["SD"] * 100 + ["PD"] * 80 + [None] * 11,
            "pid": [f"P{i:03d}" for i in range(n)],
        }
    )
    return make_session(df, "waffle_wf")


def test_waffle_cells_are_whole_and_fill_the_grid_exactly(client, cohort):
    r = client.post("/api/charts/waffle", json={"session_id": cohort, "category": "histology"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["units"] == 100 and d["n"] == 391
    assert sum(lv["cells"] for lv in d["levels"]) == 100
    for lv in d["levels"]:
        assert isinstance(lv["cells"], int)
        # No level is more than one cell away from its exact share.
        assert abs(lv["cells"] - lv["percent"]) < 1.0
        assert lv["count"] > 0


def test_waffle_largest_remainder_does_not_lose_a_person(client):
    # 49.4 / 27.4 / 23.2 % round to 49 + 27 + 23 = 99 — one cell short.
    df = pd.DataFrame({"g": ["a"] * 494 + ["b"] * 274 + ["c"] * 232})
    sid = make_session(df, "waffle_lr")
    d = client.post("/api/charts/waffle", json={"session_id": sid, "category": "g"}).json()
    assert [lv["cells"] for lv in d["levels"]] == [50, 27, 23]


def test_waffle_folds_a_long_tail_and_bounds_the_grid(client):
    df = pd.DataFrame({"g": [f"L{i}" for i in range(12)] * 10})
    sid = make_session(df, "waffle_tail")
    d = client.post("/api/charts/waffle", json={"session_id": sid, "category": "g", "max_levels": 4}).json()
    assert [lv["label"] for lv in d["levels"]][-1] == "Other"
    assert len(d["levels"]) == 4 and d["n_folded_into_other"] == 9
    assert client.post("/api/charts/waffle", json={"session_id": sid, "category": "g", "units": 5}).status_code == 400


def test_waterfall_is_sorted_and_counts_the_missing(client, cohort):
    r = client.post(
        "/api/charts/waterfall",
        json={"session_id": cohort, "y": "change", "group": "response", "label": "pid",
              "thresholds": [20, -30]},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n"] == 380 and d["n_missing"] == 11
    values = [row["value"] for row in d["rows"]]
    assert values == sorted(values, reverse=True)
    assert [row["rank"] for row in d["rows"]][:3] == [1, 2, 3]
    assert all(row["label"].startswith("P") for row in d["rows"])
    assert {row["group"] for row in d["rows"]} == {"PR", "SD", "PD"}


def test_waterfall_thresholds_count_on_the_side_they_point_to(client, cohort):
    d = client.post(
        "/api/charts/waterfall",
        json={"session_id": cohort, "y": "change", "thresholds": [20, -30]},
    ).json()
    values = np.array([row["value"] for row in d["rows"]])
    up, down = d["thresholds"]
    assert up == {"value": 20.0, "side": "at_or_above", "n": int((values >= 20).sum())}
    assert down == {"value": -30.0, "side": "at_or_below", "n": int((values <= -30).sum())}
    assert up["n"] > 0 and down["n"] > 0


def test_waterfall_names_a_missing_column(client, cohort):
    r = client.post("/api/charts/waterfall", json={"session_id": cohort, "y": "nope"})
    assert r.status_code == 400 and "nope" in r.json()["detail"]
