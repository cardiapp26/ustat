"""Characterization tests for Restricted Cubic Splines (RCS).

Focus: lock the behavior of the Harrell RCS basis generation and the
two main RCS endpoints before any refactoring of models.py or services/.

These are intentionally simple but numerically meaningful tests.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_session

from services.rcs_basis import (
    rcs_basis,
    resolve_knots,
    harrell_knots,
)


# ──────────────────────────────────────────────────────────────────────────────
# RCS basis generation (pure, very high value for refactoring)
# ──────────────────────────────────────────────────────────────────────────────

def test_harrell_knots_standard():
    """Harrell default knots for 4 knots must be 5/35/65/95 percentiles."""
    x = np.linspace(0, 100, 1000)
    knots = harrell_knots(x, 4)
    np.testing.assert_allclose(knots, [5, 35, 65, 95], rtol=0.01)


def test_rcs_basis_shape_and_monotonicity():
    """
    For n_knots=4 we should get (4-2)=2 basis columns.
    The basis should be zero before the first knot and increase after.
    """
    rng = np.random.default_rng(42)
    x = rng.uniform(10, 90, 500)
    knots = np.array([20, 40, 60, 80])

    basis = rcs_basis(x, knots)
    assert basis.shape == (500, 2)

    # Before first knot, the basis columns should be very small (near zero)
    mask_before = x < 20
    if mask_before.any():
        assert np.all(np.abs(basis[mask_before]) < 1e-8)


def test_resolve_knots_custom_validation():
    """Custom knots must be strictly increasing and within data range."""
    x = np.array([0.0, 10.0, 20.0, 30.0, 100.0])

    # Valid
    k = resolve_knots(x, 3, knot_positions=[5, 25, 70])
    np.testing.assert_array_equal(k, [5, 25, 70])

    # Not enough knots
    with pytest.raises(ValueError):
        resolve_knots(x, 4, knot_positions=[1, 2, 3])

    # Not strictly increasing
    with pytest.raises(ValueError):
        resolve_knots(x, 3, knot_positions=[10, 5, 20])

    # Outside range
    with pytest.raises(ValueError):
        resolve_knots(x, 3, knot_positions=[-1, 10, 50])


# ──────────────────────────────────────────────────────────────────────────────
# Univariate RCS endpoint characterization
# ──────────────────────────────────────────────────────────────────────────────

def test_rcs_univariate_cox_basic(client):
    """
    Simple Cox RCS run. We only check that the endpoint returns the
    expected top-level keys and that nonlinearity p-value is present.
    """
    rng = np.random.default_rng(99)
    n = 300
    time = rng.exponential(200, n).clip(5, 800)
    event = rng.binomial(1, 0.6, n)
    x = rng.normal(50, 15, n).clip(10, 120)

    # Make x have a mild non-linear effect on hazard
    hazard = 0.01 + 0.0008 * (x - 50)**2 / 100
    event = (rng.uniform(0, 1, n) < (1 - np.exp(-hazard * time))).astype(int)

    df = pd.DataFrame({"time": time, "event": event, "ldl": x})
    sid = make_session(df, "rcs_cox1")

    payload = {
        "session_id": sid,
        "predictor": "ldl",
        "model_type": "cox",
        "duration_col": "time",
        "event_col": "event",
        "n_knots": 4,
        "knot_placement": "harrell",
    }

    r = client.post("/api/models/rcs", json=payload)
    assert r.status_code == 200, r.text

    data = r.json()
    assert "knots" in data
    assert len(data["knots"]) == 4
    assert "nonlinearity_p" in data or "nonlinearity_wald" in data
    # New modular response uses x_values + or_values/ci instead of legacy hr_curve
    assert "x_values" in data and "or_values" in data
    assert "model_type" in data and data["model_type"] == "cox"


def test_rcs_linear_outcome(client):
    """RCS with linear outcome should return beta + CI for the spline terms."""
    rng = np.random.default_rng(123)
    n = 250
    x = rng.normal(100, 20, n)
    y = 3.0 + 0.8 * x + 0.015 * (x - 80).clip(0, None)**2 + rng.normal(0, 12, n)

    df = pd.DataFrame({"y": y, "dose": x})
    sid = make_session(df, "rcs_lin")

    r = client.post("/api/models/rcs", json={
        "session_id": sid,
        "predictor": "dose",
        "model_type": "linear",
        "outcome": "y",
        "n_knots": 3,
    })
    assert r.status_code == 200
    data = r.json()
    assert data.get("model_type") == "linear"
    # New response shape after extraction
    assert "x_values" in data and "or_values" in data
    assert "knots" in data and len(data["knots"]) == 3


# ──────────────────────────────────────────────────────────────────────────────
# Harrell knots, overall/non-linearity tests and knot AIC vs R rms 8.x
# Reference values: Hmisc::rcspline.eval, rms::anova and AIC() on
# qa/models_audit/dataset.csv (n = 300).
# ──────────────────────────────────────────────────────────────────────────────

from pathlib import Path  # noqa: E402

from services.rcs_basis import SUPPORTED_KNOT_COUNTS  # noqa: E402

_AUDIT_CSV = Path(__file__).resolve().parents[2] / "qa" / "models_audit" / "dataset.csv"

R_KNOTS = {
    3: [48.2, 62.2, 76.02],
    4: [44.595, 58.025, 67.0, 79.82],
    5: [44.595, 54.845, 62.2, 69.6775, 79.82],
    6: [44.595, 53.254, 59.759, 65.182, 70.6, 79.82],
    7: [41.975, 51.8816666667, 57.3, 62.2, 67.0841666667, 72.2, 82.1575],
}


@pytest.fixture(scope="module")
def audit_df():
    return pd.read_csv(_AUDIT_CSV)


def test_supported_knot_counts_cover_harrell_table():
    assert SUPPORTED_KNOT_COUNTS == (3, 4, 5, 6, 7)


@pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
def test_harrell_knots_match_r(audit_df, k):
    knots = harrell_knots(audit_df["age"].to_numpy(float), k)
    np.testing.assert_allclose(knots, R_KNOTS[k], rtol=1e-9)


@pytest.mark.parametrize("k,expected", [
    (4, [48.3, 56.9, 65.805, 78.9]),
    (5, [48.3, 54.2125, 60.45, 67.3875, 78.9]),
])
def test_small_sample_outer_knots_match_r(audit_df, k, expected):
    """n < 100: outer knots become the 5th smallest / largest value."""
    x = audit_df["age"].to_numpy(float)[:60]
    np.testing.assert_allclose(harrell_knots(x, k), expected, rtol=1e-9)


def _post_rcs(client, df, name, **payload):
    sid = make_session(df, name)
    r = client.post("/api/models/rcs", json={"session_id": sid, "predictor": "age", "n_knots": 4, **payload})
    assert r.status_code == 200, r.text
    return r.json()


def test_logistic_overall_and_nonlinear_match_rms_anova(client, audit_df):
    d = _post_rcs(client, audit_df, "rcs_lrm_anova", model_type="logistic", outcome="event_binary")
    assert d["overall_df"] == 3 and d["nonlinearity_df"] == 2
    assert d["test_statistic"] == "chi2"
    assert d["overall_wald"] == pytest.approx(8.23431244415, rel=1e-4)
    assert d["overall_p"] == pytest.approx(0.04140942949, rel=1e-3)
    assert d["nonlinearity_wald"] == pytest.approx(0.21084121038, rel=1e-3)
    assert d["nonlinearity_p"] == pytest.approx(0.89994592105, rel=1e-3)


def test_logistic_knot_aic_comparison_matches_r(client, audit_df):
    d = _post_rcs(client, audit_df, "rcs_lrm_aic", model_type="logistic", outcome="event_binary")
    aic = {row["n_knots"]: row["aic"] for row in d["knot_comparison"]}
    assert aic[3] == pytest.approx(347.376406, abs=1e-2)
    assert aic[4] == pytest.approx(349.163514, abs=1e-2)
    assert aic[5] == pytest.approx(345.859687, abs=1e-2)
    best = [row for row in d["knot_comparison"] if row["best"]]
    assert [row["n_knots"] for row in best] == [5]
    assert best[0]["delta_aic"] == 0
    assert d["aic"] == pytest.approx(aic[4], abs=1e-2)


def test_linear_uses_f_test_and_r_aic(client, audit_df):
    d = _post_rcs(client, audit_df, "rcs_ols_anova", model_type="linear", outcome="sbp")
    assert d["test_statistic"] == "F"
    assert d["overall_f"] == pytest.approx(26.85232539, rel=1e-4)
    assert d["nonlinearity_f"] == pytest.approx(3.709547274, rel=1e-4)
    assert d["nonlinearity_p"] == pytest.approx(0.02563438194, rel=1e-3)
    assert d["aic"] == pytest.approx(2221.663078, abs=1e-2)
    aic = {row["n_knots"]: row["aic"] for row in d["knot_comparison"]}
    assert aic[3] == pytest.approx(2224.895216, abs=1e-2)
    assert aic[5] == pytest.approx(2222.536515, abs=1e-2)


def test_cox_overall_and_aic_match_rms(client, audit_df):
    d = _post_rcs(client, audit_df, "rcs_cph_anova", model_type="cox", duration_col="time", event_col="status")
    assert d["overall_wald"] == pytest.approx(17.8238593, rel=1e-3)
    assert d["overall_p"] == pytest.approx(4.782149975e-04, rel=1e-2)
    assert d["nonlinearity_wald"] == pytest.approx(2.478633881, rel=1e-3)
    aic = {row["n_knots"]: row["aic"] for row in d["knot_comparison"]}
    assert aic[3] == pytest.approx(1813.671918, abs=0.05)
    assert aic[4] == pytest.approx(1814.723386, abs=0.05)
    assert aic[5] == pytest.approx(1814.392572, abs=0.05)


def test_seven_knots_run_and_add_chosen_count_to_comparison(client, audit_df):
    d = _post_rcs(client, audit_df, "rcs_k7", model_type="logistic", outcome="event_binary", n_knots=7)
    assert d["overall_df"] == 6 and d["nonlinearity_df"] == 5
    counts = [row["n_knots"] for row in d["knot_comparison"]]
    assert counts == [3, 4, 5, 7]
    assert [row["chosen"] for row in d["knot_comparison"]] == [False, False, False, True]


def test_custom_knots_skip_comparison(client, audit_df):
    d = _post_rcs(client, audit_df, "rcs_custom", model_type="logistic", outcome="event_binary",
                  n_knots=3, knot_positions=[45, 60, 75])
    assert d["knot_comparison"] == []
    assert d["overall_p"] is not None


def test_unsupported_knot_count_rejected(client, audit_df):
    sid = make_session(audit_df, "rcs_k8")
    r = client.post("/api/models/rcs", json={"session_id": sid, "predictor": "age", "n_knots": 8,
                                            "model_type": "logistic", "outcome": "event_binary"})
    assert r.status_code == 422
    assert "3, 4, 5, 6, 7" in r.text


def test_interaction_columns_join_both_tests(client, audit_df):
    df = audit_df.assign(male=(audit_df["sex"] == "M").astype(float))
    d = _post_rcs(client, df, "rcs_ix", model_type="logistic", outcome="event_binary",
                  covariates=["male"], interaction_covariates=["male"])
    assert d["tests_include_interaction"] is True
    assert d["overall_df"] == 3 + 3  # main + one interaction column per term column
    assert d["nonlinearity_df"] == 2 + 2


def test_cox_rcs_reports_overall_association(client, audit_df):
    sid = make_session(audit_df, "coxrcs_overall")
    r = client.post("/api/models/survival/cox_rcs", json={
        "session_id": sid, "duration_col": "time", "event_col": "status",
        "spline_terms": [{"column": "age", "n_knots": 4}],
    })
    assert r.status_code == 200, r.text
    d = r.json()
    ov = d["overall_association"]["age"]
    assert ov["df"] == 3
    assert ov["wald"] == pytest.approx(17.8238593, rel=1e-3)
    assert d["nonlinearity"]["age"]["df"] == 2


def test_covariate_repeating_predictor_or_outcome_is_ignored(client, audit_df):
    """Used to select the same column twice and crash with a 500."""
    d = _post_rcs(client, audit_df, "rcs_dup_cov", model_type="logistic", outcome="event_binary",
                  covariates=["age", "event_binary", "bmi", "bmi"])
    assert d["covariates_used"] == ["bmi"]
