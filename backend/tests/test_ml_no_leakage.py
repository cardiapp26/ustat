"""The ML router must not learn anything from the rows it reports on.

Three separate leaks were possible before, and each gets a guard here:

1. **Outcome imputation.** ``apply_imputation`` ran over the outcome column
   too, so a binary outcome with a gap came back with a third class (an
   imputed 0.5128) and n grew past the number of observed outcomes.
2. **Preprocessing fitted on the whole dataset.** Imputer and encoder were fit
   once over every row, then cross-validated, so each validation fold's own
   values had already shaped the numbers reported as out-of-sample.
3. **Permutation importance on training data.** Permuting a column on the rows
   a deep forest memorised measures recall, not prediction, and hands pure
   noise a respectable-looking importance.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session

SEED = 20260905
PREFIX = "/api/ml"

FAST = {"n_estimators": 60, "cv_folds": 3, "n_permutation_repeats": 5,
        "min_samples_leaf": 5}


@pytest.fixture(scope="module")
def gappy():
    """One real predictor, three noise columns, gaps in both y and X."""
    rng = np.random.default_rng(SEED)
    n = 200
    signal = rng.normal(0, 1, n)
    lin = 2.0 * signal
    y = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-lin))).astype(float)
    df = pd.DataFrame({
        "signal": signal,
        "noise1": rng.normal(0, 1, n),
        "noise2": rng.normal(0, 1, n),
        "noise3": rng.normal(0, 1, n),
        "y": y,
        "yreg": 3.0 * signal + rng.normal(0, 1, n),
    })
    # 20 missing outcomes and 25 missing predictor values, disjoint enough to
    # tell the two exclusion reasons apart.
    df.loc[rng.choice(n, 20, replace=False), "y"] = np.nan
    df.loc[rng.choice(n, 25, replace=False), "noise1"] = np.nan
    return df


@pytest.fixture(scope="module")
def sid(gappy):
    return make_session(gappy, "ml_leak_main")


def _rf(client, **payload):
    return client.post(f"{PREFIX}/random_forest", json={**FAST, **payload})


# -- 1. The outcome is never imputed ----------------------------------------


@pytest.mark.parametrize("strategy", ["listwise", "median", "mean", "mice"])
def test_missing_outcomes_are_excluded_not_filled(client, sid, gappy, strategy):
    """n counts observed outcomes, whatever the imputation strategy."""
    r = _rf(client, session_id=sid, outcome="y",
            predictors=["signal", "noise2", "noise3"], imputation=strategy)
    assert r.status_code == 200, r.text
    d = r.json()
    observed = int(gappy["y"].notna().sum())
    assert d["n"] == observed
    assert d["n_events"] + d["n_non_events"] == observed
    assert d["imputation"]["outcome_imputed"] is False
    assert d["imputation"]["n_excluded_missing_outcome"] == len(gappy) - observed


def test_imputed_outcome_can_no_longer_invent_a_third_class(client, sid, gappy):
    """A filled binary outcome used to produce a non-binary class and a 422."""
    r = _rf(client, session_id=sid, outcome="y",
            predictors=["signal", "noise2"], imputation="mice",
            task="classification")
    assert r.status_code == 200, r.text
    d = r.json()
    # Exactly two classes, and both counts are whole subjects.
    assert d["n_events"] > 0 and d["n_non_events"] > 0
    assert d["n_events"] + d["n_non_events"] == int(gappy["y"].notna().sum())


# -- 2. Preprocessing is refitted inside every fold -------------------------


def test_report_says_the_imputer_is_refitted_per_fold(client, sid):
    fitted = _rf(client, session_id=sid, outcome="y",
                 predictors=["signal", "noise1"], imputation="median").json()
    assert fitted["imputation"]["fitted_per_fold"] is True
    assert fitted["imputation"]["applied"] == "median"

    listwise = _rf(client, session_id=sid, outcome="y",
                   predictors=["signal", "noise1"], imputation="listwise").json()
    # Nothing is estimated under complete-case deletion, so there is nothing
    # to refit -- and the rows with a gap are gone from n instead.
    assert listwise["imputation"]["fitted_per_fold"] is False
    assert listwise["n"] < fitted["n"]


def test_mice_is_reported_as_a_single_completion(client, sid):
    d = _rf(client, session_id=sid, outcome="y",
            predictors=["signal", "noise1"], imputation="mice").json()
    imp = d["imputation"]
    assert imp["requested"] == "mice"
    # Never plain "mice": that word is Rubin-pooled multiple imputation to a
    # reader, and this is one completed dataset.
    assert imp["applied"] == "mice_single"
    assert imp["n_imputations"] == 1
    assert imp["pooled"] is False
    assert "no Rubin pooling" in imp["label"]


# -- 3. Importance is measured where the model cannot have memorised --------


def test_pure_noise_importance_stays_near_zero(client, sid):
    """On held-out rows, permuting noise cannot help; on training rows it did."""
    d = _rf(client, session_id=sid, outcome="y",
            predictors=["signal", "noise1", "noise2", "noise3"]).json()
    assert d["importance_scope"] == "held-out folds"
    imp = {row["feature"]: row for row in d["importance"]}
    assert d["importance"][0]["feature"] == "signal"
    for noise in ("noise1", "noise2", "noise3"):
        # Held-out permutation importance of a noise column is ~0 and may be
        # slightly negative; what it must not be is a solid positive number.
        assert imp[noise]["permutation"] < 0.02
    assert imp["signal"]["permutation"] > 0.05


def test_regression_noise_importance_stays_near_zero(client, sid):
    d = _rf(client, session_id=sid, outcome="yreg",
            predictors=["signal", "noise2", "noise3"]).json()
    assert d["task"] == "regression"
    assert d["importance_scope"] == "held-out folds"
    imp = {row["feature"]: row for row in d["importance"]}
    assert imp["signal"]["permutation"] > 0.3
    for noise in ("noise2", "noise3"):
        assert imp[noise]["permutation"] < 0.05


def test_every_predictor_gets_exactly_one_importance_row(client, sid):
    preds = ["signal", "noise1", "noise2", "noise3"]
    d = _rf(client, session_id=sid, outcome="y", predictors=preds).json()
    assert [row["feature"] for row in d["importance"]] != preds  # sorted by value
    assert sorted(row["feature"] for row in d["importance"]) == sorted(preds)
    assert d["n_features"] == len(preds)
