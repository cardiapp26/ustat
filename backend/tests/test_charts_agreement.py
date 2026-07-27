"""Agreement-plot support: log axes, a y = x reference, and the dumbbell chart.

Both shapes exist to compare two numbers that ought to match — a reported
value against a recomputed one. The scatter carries the pairs; the dumbbell
carries the gap per variable, ranked. What is tested here is mostly the
things that quietly go wrong: points dropped by a log axis, a fit drawn in
the wrong space, and duplicate categories collapsing without a word.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_session


@pytest.fixture()
def agreement_session() -> str:
    """Reported vs correct p-values, spanning three orders of magnitude."""
    import pandas as pd

    rng = np.random.default_rng(11)
    correct = np.array([0.001, 0.015, 0.06, 0.08, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 0.95, 0.99])
    # Reported values sit systematically below the correct ones.
    reported = correct * rng.uniform(0.55, 0.95, size=correct.size)
    df = pd.DataFrame(
        {
            "correct_p": correct,
            "reported_p": reported,
            "variable": [f"var_{i}" for i in range(correct.size)],
            "verdict": ["one-sided"] * 6 + ["irreproducible"] * 6,
        }
    )
    return make_session(df, "agreement_sess")


def _scatter(client, sid, **kw):
    body = {"session_id": sid, "x": "correct_p", "y": "reported_p"}
    body.update(kw)
    return client.post("/api/charts/scatter", json=body)


def test_scatter_still_works_without_the_new_options(client, agreement_session):
    r = _scatter(client, agreement_session)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["log_x"] is False and d["log_y"] is False
    assert d["identity"] == {}
    assert d["regression"]["space"] == "linear"
    assert d["warnings"] == []


def test_identity_line_spans_both_axes_and_counts_which_side(client, agreement_session):
    r = _scatter(client, agreement_session, identity_line=True)
    assert r.status_code == 200, r.text
    ident = r.json()["identity"]
    assert ident["line_x"] == ident["line_y"], "y = x must have unit slope"
    # Every reported value was built below its correct value.
    assert ident["n_below"] == 12
    assert ident["n_above"] == 0


def test_log_fit_is_computed_in_log_space_not_raw(client, agreement_session):
    """A fit from raw values renders as a curve on a log axis; it must not."""
    linear = _scatter(client, agreement_session).json()["regression"]
    logged = _scatter(
        client, agreement_session, log_x=True, log_y=True
    ).json()["regression"]

    assert linear["space"] == "linear"
    assert logged["space"] == "log10-log10"
    assert logged["slope"] != pytest.approx(linear["slope"])

    # The returned endpoints are in data space, so they must be back-transformed
    # already: a straight line in log space through these points.
    lx, ly = logged["line_x"], logged["line_y"]
    assert all(v > 0 for v in lx + ly), "log-space line must return positive data coords"
    implied = (np.log10(ly[1]) - np.log10(ly[0])) / (np.log10(lx[1]) - np.log10(lx[0]))
    assert implied == pytest.approx(logged["slope"], rel=1e-9)


def test_nonpositive_values_are_dropped_loudly_on_a_log_axis(client):
    import pandas as pd

    df = pd.DataFrame({"a": [1.0, 2.0, 0.0, -3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0]})
    sid = make_session(df, "tmp_sess_%d" % abs(hash(str(df.shape)+str(df.columns.tolist()))) )
    r = client.post(
        "/api/charts/scatter",
        json={"session_id": sid, "x": "a", "y": "b", "log_x": True},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["points"]) == 3
    assert len(d["warnings"]) == 1
    w = d["warnings"][0]
    assert w["type"] == "log_axis_nonpositive"
    assert w["n_dropped"] == 2
    assert "2 of 5" in w["message"]


def test_log_axis_with_nothing_positive_left_is_a_400(client):
    import pandas as pd

    df = pd.DataFrame({"a": [0.0, -1.0, -2.0], "b": [1.0, 2.0, 3.0]})
    sid = make_session(df, "tmp_sess_%d" % abs(hash(str(df.shape)+str(df.columns.tolist()))) )
    r = client.post(
        "/api/charts/scatter",
        json={"session_id": sid, "x": "a", "y": "b", "log_x": True},
    )
    assert r.status_code == 400
    assert "log axis" in r.json()["detail"]


def test_log_axis_on_a_text_column_is_a_400_not_a_crash(client, agreement_session):
    r = client.post(
        "/api/charts/scatter",
        json={
            "session_id": agreement_session,
            "x": "variable",
            "y": "reported_p",
            "log_x": True,
        },
    )
    assert r.status_code == 400
    assert "numeric" in r.json()["detail"]


def test_label_column_rides_along_with_each_point(client, agreement_session):
    r = _scatter(client, agreement_session, label="variable")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["label"] == "variable"
    assert d["points"][0]["variable"].startswith("var_")


# ── dumbbell ────────────────────────────────────────────────────────────────

@pytest.fixture()
def effect_session() -> str:
    import pandas as pd

    df = pd.DataFrame(
        {
            "variable": ["Uric acid", "MPV", "Age", "SBP", "Platelet"],
            "d_implied": [0.40, 0.40, 0.36, 0.25, 0.02],
            "d_printed": [1.31, 1.06, 0.39, 0.25, 0.02],
            "verdict": ["irreconcilable", "irreconcilable", "consistent", "consistent", "consistent"],
        }
    )
    return make_session(df, "effect_sess")


def _dumbbell(client, sid, **kw):
    body = {
        "session_id": sid,
        "category": "variable",
        "start": "d_implied",
        "end": "d_printed",
    }
    body.update(kw)
    return client.post("/api/charts/dumbbell", json=body)


def test_dumbbell_ranks_by_gap_and_reports_the_worst(client, effect_session):
    r = _dumbbell(client, effect_session)
    assert r.status_code == 200, r.text
    d = r.json()
    gaps = [abs(row["gap"]) for row in d["rows"]]
    assert gaps == sorted(gaps, reverse=True), "worst disagreement must come first"
    assert d["rows"][0]["category"] == "Uric acid"
    assert d["summary"]["largest_gap_category"] == "Uric acid"
    assert d["summary"]["max_abs_gap"] == pytest.approx(0.91)
    assert d["summary"]["n_end_above_start"] == 3
    assert d["summary"]["n_end_below_start"] == 0


def test_dumbbell_carries_the_group_band(client, effect_session):
    d = _dumbbell(client, effect_session, group="verdict").json()
    assert {row["group"] for row in d["rows"]} == {"irreconcilable", "consistent"}


def test_dumbbell_sort_by_category_is_alphabetical(client, effect_session):
    d = _dumbbell(client, effect_session, sort="category").json()
    names = [row["category"] for row in d["rows"]]
    assert names == sorted(names)


def test_duplicate_categories_are_refused_by_name(client):
    """Silently averaging them would draw a believable, wrong chart."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "variable": ["Age", "Age", "SBP", "SBP"],
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [1.5, 2.5, 3.5, 4.5],
        }
    )
    sid = make_session(df, "tmp_sess_%d" % abs(hash(str(df.shape)+str(df.columns.tolist()))) )
    r = client.post(
        "/api/charts/dumbbell",
        json={"session_id": sid, "category": "variable", "start": "a", "end": "b"},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "Age" in detail and "SBP" in detail


def test_dumbbell_rejects_an_unknown_sort_key(client, effect_session):
    r = _dumbbell(client, effect_session, sort="whatever")
    assert r.status_code == 400
    assert "sort must be one of" in r.json()["detail"]


def test_dumbbell_missing_column_is_a_400(client, effect_session):
    r = _dumbbell(client, effect_session, start="not_a_column")
    assert r.status_code == 400
    assert "not_a_column" in r.json()["detail"]
