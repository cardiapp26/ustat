"""Reference handling, nadir and distribution for RCS curves.

R reference: rms::lrm(event_binary ~ rcs(age, 4)) on
qa/models_audit/dataset.csv, contrast(age = grid[50] vs age = 60).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_session

from services.rcs_curve import distribution, find_nadir

_AUDIT_CSV = Path(__file__).resolve().parents[2] / "qa" / "models_audit" / "dataset.csv"


@pytest.fixture(scope="module")
def audit_df():
    return pd.read_csv(_AUDIT_CSV)


def _post(client, df, name, **payload):
    sid = make_session(df, name)
    body = {"session_id": sid, "predictor": "age", "n_knots": 4, "model_type": "logistic",
            "outcome": "event_binary", **payload}
    return client.post("/api/models/rcs", json=body)


def test_curve_is_relative_to_exact_reference_and_matches_rms_contrast(client, audit_df):
    r = _post(client, audit_df, "curve_exact_ref", ref_value=60)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ref_value"] == 60 and d["ref_strategy_used"] == "value"
    assert d["x_values"][50] == pytest.approx(49.80256281407035)
    assert d["or_values"][50] == pytest.approx(0.7176882902, rel=1e-4)
    assert d["ci_low"][50] == pytest.approx(0.4960788753, rel=1e-4)
    assert d["ci_high"][50] == pytest.approx(1.0382955363, rel=1e-4)
    # The curve passes through OR = 1 at the reference, between grid points.
    assert np.interp(60, d["x_values"], d["or_values"]) == pytest.approx(1.0, abs=2e-3)


def _u_shaped(seed: int = 3, n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(20, 80, n)
    p = 1 / (1 + np.exp(-(-1.5 + 0.004 * (x - 50) ** 2)))
    return pd.DataFrame({"age": x, "event_binary": rng.binomial(1, p)})


def test_nadir_found_inside_u_shaped_curve(client):
    d = _post(client, _u_shaped(), "curve_nadir").json()
    assert d["nadir"]["at_boundary"] is False
    assert d["nadir"]["x"] == pytest.approx(50, abs=4)
    assert d["ref_strategy_used"] == "median"


def test_nadir_reference_puts_curve_at_or_above_one(client):
    d = _post(client, _u_shaped(), "curve_nadir_ref", ref_strategy="nadir").json()
    assert d["ref_strategy_used"] == "nadir" and d["ref_fallback"] is False
    assert d["ref_value"] == pytest.approx(d["nadir"]["x"], abs=1e-4)
    assert min(d["or_values"]) >= 1 - 1e-6


def test_nadir_request_falls_back_to_median_without_interior_minimum(client, audit_df):
    """age vs event_binary rises over the whole range: no interior nadir."""
    d = _post(client, audit_df, "curve_nadir_fallback", ref_strategy="nadir").json()
    assert d["nadir"]["at_boundary"] is True
    assert d["ref_strategy_used"] == "median" and d["ref_fallback"] is True
    assert d["ref_value"] == pytest.approx(float(np.median(audit_df["age"])))


def test_explicit_reference_wins_over_strategy(client, audit_df):
    d = _post(client, audit_df, "curve_ref_wins", ref_value=55, ref_strategy="nadir").json()
    assert d["ref_strategy_used"] == "value" and d["ref_value"] == 55


def test_monotone_curve_reports_boundary_nadir():
    grid = np.linspace(0, 10, 50)
    nadir = find_nadir(lambda xs: 2.0 * xs, grid)
    assert nadir.at_boundary is True and nadir.x == 0.0


def test_find_nadir_refines_between_grid_points():
    grid = np.linspace(0, 10, 11)
    nadir = find_nadir(lambda xs: (xs - 3.37) ** 2, grid)
    assert nadir.at_boundary is False
    assert nadir.x == pytest.approx(3.37, abs=1e-4)


def test_unknown_ref_strategy_rejected(client, audit_df):
    r = _post(client, audit_df, "curve_bad_strategy", ref_strategy="mode")
    assert r.status_code == 422


def test_distribution_counts_every_observation(client, audit_df):
    d = _post(client, audit_df, "curve_dist").json()
    dist = d["x_distribution"]
    assert len(dist["edges"]) == len(dist["counts"]) + 1
    assert sum(dist["counts"]) + dist["n_outside"] == d["n"]
    assert dist["edges"][0] == pytest.approx(d["x_values"][0])
    assert dist["edges"][-1] == pytest.approx(d["x_values"][-1])


def test_distribution_bins_are_bounded():
    x = np.linspace(0, 1, 100_000)
    assert len(distribution(x, 0, 1)["counts"]) == 40
    assert len(distribution(x[:20], 0, 1)["counts"]) == 10


def test_cox_rcs_curve_uses_exact_reference(client, audit_df):
    sid = make_session(audit_df, "coxrcs_exact_ref")
    r = client.post("/api/models/survival/cox_rcs", json={
        "session_id": sid, "duration_col": "time", "event_col": "status",
        "spline_terms": [{"column": "age", "n_knots": 4, "ref_value": 60}],
    })
    assert r.status_code == 200, r.text
    curve = r.json()["curves_1d"][0]
    assert curve["ref"] == 60 and len(curve["x"]) == 200
    assert np.interp(60, curve["x"], curve["hr"]) == pytest.approx(1.0, abs=2e-3)
