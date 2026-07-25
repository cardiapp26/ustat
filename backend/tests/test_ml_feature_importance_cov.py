"""Coverage tests for POST /api/ml/feature_importance (routers/ml.py).

The endpoint had no test reference. The fixture is built so the ranking is
known in advance — one strong predictor, one weak one, two pure-noise columns —
so the tests fail if the importance ordering ever stops tracking the signal.

Hyper-parameters are dialled down from the defaults (300 trees, 10 permutation
repeats) to keep the file's runtime around a second.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session

SEED = 20260725
PREFIX = "/api/ml"

# Small-but-honest model settings; the defaults are far too slow for a test.
FAST = {"n_estimators": 60, "cv_folds": 3, "n_permutation_repeats": 5,
        "min_samples_leaf": 5}


@pytest.fixture(scope="module")
def ml_df():
    """`signal` drives both outcomes, `weak` contributes a little, the rest is noise."""
    rng = np.random.default_rng(SEED)
    n = 220
    signal = rng.normal(0, 1, n)
    weak = rng.normal(0, 1, n)
    noise1 = rng.normal(0, 1, n)
    noise2 = rng.normal(0, 1, n)

    lin = 2.2 * signal + 0.7 * weak
    y = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-lin))).astype(int)
    yreg = 3.0 * signal + 1.0 * weak + rng.normal(0, 1.0, n)

    return pd.DataFrame({
        "signal": signal,
        "weak": weak,
        "noise1": noise1,
        "noise2": noise2,
        "y": y,
        "yreg": yreg,
        "cat": np.where(rng.uniform(0, 1, n) < 0.5, "a", "b"),
        "ymulti": rng.integers(0, 4, n),
    })


@pytest.fixture(scope="module")
def sid(ml_df):
    return make_session(ml_df, "ml_fi_main")


def _post(client, **payload):
    return client.post(f"{PREFIX}/feature_importance", json={**FAST, **payload})


def _by_feature(result):
    return {row["feature"]: row for row in result["importance"]}


# ── Ranking correctness ──────────────────────────────────────────────────────


def test_classification_ranks_signal_above_noise(client, sid):
    r = _post(client, session_id=sid, outcome="y",
              predictors=["signal", "weak", "noise1", "noise2"])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["task"] == "classification"
    assert d["model"] == "Random Forest"
    assert d["outcome"] == "y"
    assert d["n"] == 220

    imp = _by_feature(d)
    assert set(imp) == {"signal", "weak", "noise1", "noise2"}

    # The informative feature must top the ranking, by a wide margin.
    assert d["importance"][0]["feature"] == "signal"
    for noise in ("noise1", "noise2"):
        assert imp["signal"]["permutation"] > 3 * imp[noise]["permutation"]
        assert imp["signal"]["impurity"] > imp[noise]["impurity"]
    # The weak-but-real predictor must sit between signal and noise.
    assert imp["signal"]["permutation"] > imp["weak"]["permutation"]
    assert imp["weak"]["permutation"] > imp["noise1"]["permutation"]
    assert imp["weak"]["permutation"] > imp["noise2"]["permutation"]

    assert "signal" in d["interpretation"]


def test_importance_list_is_sorted_by_permutation_descending(client, sid):
    d = _post(client, session_id=sid, outcome="y",
              predictors=["noise1", "signal", "noise2", "weak"]).json()
    perms = [row["permutation"] for row in d["importance"]]
    assert perms == sorted(perms, reverse=True)
    # Ordering must come from the data, not from the request order.
    assert d["importance"][0]["feature"] == "signal"
    for row in d["importance"]:
        assert isinstance(row["permutation_sd"], float)
        assert row["permutation_sd"] >= 0.0


def test_regression_task_ranks_signal_above_noise(client, sid):
    r = _post(client, session_id=sid, outcome="yreg",
              predictors=["signal", "noise1", "noise2"])
    assert r.status_code == 200, r.text
    d = r.json()
    # A continuous outcome with many distinct values resolves to regression.
    assert d["task"] == "regression"
    imp = _by_feature(d)
    assert d["importance"][0]["feature"] == "signal"
    # Permutation importance is scored on R², so a real predictor costs a lot.
    assert imp["signal"]["permutation"] > 0.5
    for noise in ("noise1", "noise2"):
        assert imp[noise]["permutation"] < 0.2
        assert imp["signal"]["permutation"] > 10 * imp[noise]["permutation"]


def test_explicit_task_overrides_the_auto_resolution(client, sid):
    """yreg is continuous; forcing classification must be refused, not guessed."""
    auto = _post(client, session_id=sid, outcome="yreg",
                 predictors=["signal", "noise1"]).json()
    assert auto["task"] == "regression"
    forced = _post(client, session_id=sid, outcome="yreg",
                   predictors=["signal", "noise1"], task="classification")
    assert forced.status_code == 422, forced.text
    assert "binary" in forced.json()["detail"]

    # The binary outcome can be forced the other way and stays classification.
    r = _post(client, session_id=sid, outcome="y",
              predictors=["signal", "noise1"], task="classification")
    assert r.status_code == 200, r.text
    assert r.json()["task"] == "classification"


def test_categorical_predictor_is_one_hot_encoded(client, sid):
    r = _post(client, session_id=sid, outcome="yreg",
              predictors=["signal", "cat"])
    assert r.status_code == 200, r.text
    features = {row["feature"] for row in r.json()["importance"]}
    # drop_first=True, so the two-level column yields exactly one dummy.
    assert features == {"signal", "cat_b"}
    imp = _by_feature(r.json())
    assert imp["signal"]["permutation"] > imp["cat_b"]["permutation"]


def test_results_are_deterministic_for_a_fixed_random_state(client, sid):
    """random_state is part of the request; the same request must replay exactly."""
    payload = dict(session_id=sid, outcome="y",
                   predictors=["signal", "weak", "noise1"], random_state=7)
    first = _post(client, **payload).json()
    second = _post(client, **payload).json()
    assert first["importance"] == second["importance"]
    assert first["interpretation"] == second["interpretation"]


def test_response_is_the_trimmed_screening_payload(client, sid):
    """This route deliberately drops the curves/metrics that /random_forest returns."""
    d = _post(client, session_id=sid, outcome="y",
              predictors=["signal", "noise1"]).json()
    assert set(d) == {"model", "task", "n", "outcome", "importance",
                      "interpretation"}
    for heavy in ("roc_curve", "calibration", "confusion", "auc", "scatter"):
        assert heavy not in d
    for row in d["importance"]:
        assert set(row) == {"feature", "impurity", "permutation",
                            "permutation_sd"}


def test_importance_agrees_with_the_random_forest_endpoint(client, sid):
    """Same estimator underneath, so the ranking must match /random_forest."""
    payload = dict(session_id=sid, outcome="y",
                   predictors=["signal", "weak", "noise1", "noise2"])
    fi = _post(client, **payload).json()
    rf = client.post(f"{PREFIX}/random_forest", json={**FAST, **payload}).json()
    assert fi["importance"] == rf["importance"]
    assert fi["n"] == rf["n"]


# ── Error paths ──────────────────────────────────────────────────────────────


def test_unknown_predictor_400(client, sid):
    r = _post(client, session_id=sid, outcome="y", predictors=["signal", "ghost"])
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "ghost" in detail


def test_unknown_outcome_400(client, sid):
    r = _post(client, session_id=sid, outcome="ghost", predictors=["signal"])
    assert r.status_code == 400, r.text
    assert "ghost" in r.json()["detail"]


def test_no_predictors_422(client, sid):
    r = _post(client, session_id=sid, outcome="y", predictors=[])
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "at least one predictor" in detail


def test_outcome_used_as_its_own_predictor_422(client, sid):
    """Leaking the outcome into X would give a perfect, meaningless model."""
    r = _post(client, session_id=sid, outcome="y", predictors=["y", "signal"])
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "Outcome cannot also be a predictor" in detail


def test_multiclass_outcome_422(client, sid):
    r = _post(client, session_id=sid, outcome="ymulti",
              predictors=["signal", "noise1"], task="classification")
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "binary" in detail


def test_non_numeric_regression_outcome_422(client, sid):
    r = _post(client, session_id=sid, outcome="cat",
              predictors=["signal", "noise1"], task="regression")
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "numeric" in detail


def test_too_few_rows_400(client, ml_df):
    tiny = make_session(ml_df.head(15), "ml_fi_tiny")
    r = _post(client, session_id=tiny, outcome="y", predictors=["signal"])
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "20" in detail


def test_unknown_session_404(client):
    r = _post(client, session_id="no_such_session", outcome="y",
              predictors=["signal"])
    assert r.status_code == 404, r.text
    assert r.json()["detail"].strip()
