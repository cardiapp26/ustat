"""Bar chart of means with a whisker — geom_col + stat_summary(errorbar).

The whisker is the difference between a dynamite plot and a figure: a bare
bar says nothing about how firm the mean is. What is pinned here is that the
three spreads are the ones they claim to be, that a bare request still draws
bare bars, and that the grouped chart carries the same fields.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture()
def arms() -> str:
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            "ldl": np.concatenate([rng.normal(120, 20, 30), rng.normal(100, 20, 30)]),
            "arm": ["Control"] * 30 + ["Drug"] * 30,
            "sex": (["F", "M"] * 30),
        }
    )
    return make_session(df, "bar_err")


def _bar(client, sid, **kw):
    body = {"session_id": sid, "x": "arm", "y": "ldl", "y_mode": "mean"}
    body.update(kw)
    return client.post("/api/charts/bar", json=body)


def test_bare_request_draws_bare_bars(client, arms):
    d = _bar(client, arms).json()
    assert d["error"] is None
    for row in d["data"]:
        assert "lower" not in row
        assert row["n"] == 30


def test_sd_se_ci_are_the_spreads_they_claim(client, arms):
    sd = _bar(client, arms, error="sd").json()["data"][0]
    se = _bar(client, arms, error="se").json()["data"][0]
    ci = _bar(client, arms, error="ci").json()["data"][0]
    assert sd["upper"] - sd["value"] == pytest.approx(sd["sd"])
    assert se["upper"] - se["value"] == pytest.approx(sd["sd"] / np.sqrt(30))
    # t(0.975, 29) = 2.045..., so the CI half-width is wider than one SE and
    # narrower than one SD on a group this size.
    assert se["upper"] - se["value"] < ci["upper"] - ci["value"] < sd["upper"] - sd["value"]
    assert ci["upper"] - ci["value"] == pytest.approx(2.0452 * se["se"], rel=1e-3)


def test_error_label_names_the_whisker(client, arms):
    assert _bar(client, arms, error="ci").json()["error_label"] == "mean with 95% CI"
    assert _bar(client, arms, error="sd").json()["error_label"] == "mean ± SD"


def test_unknown_spread_is_rejected(client, arms):
    assert _bar(client, arms, error="iqr").status_code == 400


def test_grouped_bars_carry_the_whisker_too(client, arms):
    d = _bar(client, arms, color="sex", error="se").json()
    assert d["color"] == "sex"
    assert len(d["series"]) == 2
    for series in d["series"]:
        for row in series["data"]:
            assert row["lower"] < row["value"] < row["upper"]
            assert row["n"] == 15


def test_single_observation_gets_a_zero_whisker_not_a_nan(client):
    df = pd.DataFrame({"v": [1.0, 5.0, 6.0], "g": ["a", "b", "b"]})
    sid = make_session(df, "bar_err_one")
    r = client.post(
        "/api/charts/bar",
        json={"session_id": sid, "x": "g", "y": "v", "y_mode": "mean", "error": "ci"},
    )
    assert r.status_code == 200, r.text
    lone = next(row for row in r.json()["data"] if row["label"] == "a")
    assert lone["lower"] == lone["upper"] == lone["value"]
