"""Three things a published multi-panel figure needs that the charts API had no
way to draw: a confidence band around a fitted line, Spearman beside Pearson,
and one panel per VARIABLE rather than per level of a column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from services import store


@pytest.fixture()
def sid(client) -> str:
    rng = np.random.default_rng(4)
    n = 80
    esr = rng.gamma(3, 5, n)
    store.save("fig", pd.DataFrame({
        "esr": esr,
        "iceb": 3.5 + 0.02 * esr + rng.normal(0, 0.4, n),
        "qt": rng.normal(380, 20, n),
        "qrs": rng.normal(95, 10, n),
        "sex": rng.choice(["Female", "Male"], n),
    }))
    return "fig"


def _scatter(client, sid, **extra):
    r = client.post("/api/charts/scatter", json={
        "session_id": sid, "x": "esr", "y": "iceb", **extra})
    assert r.status_code == 200, r.text
    return r.json()["data"] if "data" in r.json() else r.json()


# ── confidence band ──────────────────────────────────────────────────────────

def test_the_fitted_line_carries_a_confidence_band(client, sid):
    band = _scatter(client, sid)["regression"]["band"]
    assert band["level"] == 0.95
    assert len(band["x"]) == len(band["lo"]) == len(band["hi"]) == 60
    assert all(lo < hi for lo, hi in zip(band["lo"], band["hi"]))


def test_the_band_is_narrowest_at_the_mean_of_x(client, sid):
    """The line is pinned down best in the middle and worst at the ends —
    which is exactly where a reader extrapolates."""
    band = _scatter(client, sid)["regression"]["band"]
    width = [hi - lo for lo, hi in zip(band["lo"], band["hi"])]
    middle = width[len(width) // 2]
    assert middle < width[0]
    assert middle < width[-1]


def test_the_band_matches_the_textbook_formula(client, sid):
    reg = _scatter(client, sid)["regression"]
    df = store.get(sid)
    x, y = df["esr"].to_numpy(), df["iceb"].to_numpy()
    n = len(x)
    slope, intercept, *_ = scipy_stats.linregress(x, y)
    resid = y - (slope * x + intercept)
    s_err = np.sqrt((resid ** 2).sum() / (n - 2))
    sxx = ((x - x.mean()) ** 2).sum()
    t_crit = scipy_stats.t.ppf(0.975, n - 2)
    x0 = reg["band"]["x"][0]
    expected = t_crit * s_err * np.sqrt(1 / n + (x0 - x.mean()) ** 2 / sxx)
    got = (reg["band"]["hi"][0] - reg["band"]["lo"][0]) / 2
    assert got == pytest.approx(float(expected), rel=1e-6)


def test_the_band_follows_a_log_axis(client, sid):
    """Fitted in log space, reported in data space — the band is evenly spaced
    where it was computed, not where it is drawn, so it sits on the line."""
    plain = _scatter(client, sid)["regression"]["band"]["x"]
    logged = _scatter(client, sid, log_x=True)["regression"]["band"]["x"]
    assert all(v > 0 for v in logged)
    # Linear grid: equal differences. Log grid: equal RATIOS.
    lin_steps = np.diff(plain)
    log_steps = np.diff(np.log10(logged))
    assert lin_steps.std() == pytest.approx(0, abs=1e-9)
    assert log_steps.std() == pytest.approx(0, abs=1e-9)
    assert np.diff(logged).std() > 1e-6


def test_a_degenerate_fit_still_answers_with_the_same_shape(client):
    """A constant x has no slope to band. The failure path must carry the same
    keys as the success path, or a client reading reg.band crashes on exactly
    the data it was meant to survive."""
    store.save("flat", pd.DataFrame({"a": [2.0] * 6, "b": [1.0, 3.0, 2.0, 5.0, 4.0, 2.5]}))
    r = client.post("/api/charts/scatter",
                    json={"session_id": "flat", "x": "a", "y": "b"})
    assert r.status_code == 200, r.text
    reg = r.json()["regression"]
    assert reg["band"] == {}
    assert reg["spearman"] == {"rho": None, "p": None}


# ── Spearman ─────────────────────────────────────────────────────────────────

def test_spearman_travels_with_the_scatter(client, sid):
    reg = _scatter(client, sid)["regression"]
    df = store.get(sid)
    rho, p = scipy_stats.spearmanr(df["esr"], df["iceb"])
    assert reg["spearman"]["rho"] == pytest.approx(float(rho))
    assert reg["spearman"]["p"] == pytest.approx(float(p))
    assert reg["spearman"]["rho"] != reg["r"]


def test_a_two_point_group_does_not_cost_the_whole_chart(client):
    """scipy answers n = 2 with rho = 1 and p = nan, and the endpoint rejects
    any response carrying a nan — so an unguarded Spearman turned a small
    group into a 400 for the entire scatter."""
    store.save("pair", pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]}))
    r = client.post("/api/charts/scatter",
                    json={"session_id": "pair", "x": "x", "y": "y"})
    assert r.status_code == 200, r.text
    assert r.json()["regression"]["spearman"]["p"] is None


# ── one panel per variable ───────────────────────────────────────────────────

def _panels(client, sid, **extra):
    r = client.post("/api/charts/facet", json={
        "session_id": sid, "kind": "boxplot", "color": "sex",
        "variables": ["qt", "qrs", "iceb"], **extra})
    assert r.status_code == 200, r.text
    return r.json()


def test_one_panel_per_variable_split_by_the_group(client, sid):
    body = _panels(client, sid)
    assert body["facet_by"] == "variable"
    assert [p["panel"] for p in body["panels"]] == ["qt", "qrs", "iceb"]
    for panel in body["panels"]:
        assert [g["group"] for g in panel["groups"]] == ["Female", "Male"]


def test_each_panel_keeps_its_own_scale(client, sid):
    """QT in milliseconds beside a unitless index: forced onto one axis the
    index collapses to a line at the bottom."""
    body = _panels(client, sid)
    assert body["shared_range"] == {}
    ranges = {p["panel"]: p["range"] for p in body["panels"]}
    assert ranges["qt"][0] > ranges["iceb"][1]


def test_variable_panels_work_without_a_group(client, sid):
    r = client.post("/api/charts/facet", json={
        "session_id": sid, "kind": "boxplot", "variables": ["qt", "qrs"]})
    assert r.status_code == 200, r.text
    assert [g["group"] for g in r.json()["panels"][0]["groups"]] == ["All"]


def test_level_faceting_still_shares_its_axis(client, sid):
    """Same measurement in every panel — there the shared range is the point."""
    r = client.post("/api/charts/facet", json={
        "session_id": sid, "kind": "boxplot", "x": "qt", "facet": "sex"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["facet_by"] == "level"
    assert body["shared_range"]["x"]


def test_a_missing_variable_is_named(client, sid):
    r = client.post("/api/charts/facet", json={
        "session_id": sid, "kind": "boxplot", "variables": ["qt", "nope"]})
    assert r.status_code == 400
    assert "nope" in r.json()["detail"]


def test_variable_panels_refuse_the_scatter_kind(client, sid):
    r = client.post("/api/charts/facet", json={
        "session_id": sid, "kind": "scatter", "variables": ["qt", "qrs"]})
    assert r.status_code == 400


def test_neither_variables_nor_a_facet_column_is_a_400(client, sid):
    r = client.post("/api/charts/facet", json={"session_id": sid, "kind": "boxplot"})
    assert r.status_code == 400


def test_an_all_missing_variable_is_reported_not_drawn(client, sid):
    df = store.get(sid).copy()
    df["blank"] = np.nan
    store.save("fig_blank", df)
    body = _panels(client, "fig_blank", variables=["qt", "blank"])
    assert [p["panel"] for p in body["panels"]] == ["qt"]
    assert any("blank" in w["message"] for w in body["warnings"])
