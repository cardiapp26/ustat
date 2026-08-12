"""Progressive adjustment: one exposure, several models, one table.

Every number below is R's, computed on the same complete cases. The logistic
comparisons use glm.control(epsilon = 1e-12): at R's default of 1e-8 the IRLS
loop stops slightly early and its coefficients differ from a fully converged
fit in the seventh decimal, so the looser default was the disagreement rather
than either implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services import store


@pytest.fixture()
def cohort(client) -> str:
    rng = np.random.default_rng(77)
    n = 500
    age = rng.normal(62, 11, n)
    sex = rng.integers(0, 2, n)
    bmi = rng.normal(27, 4.5, n)
    smoke = rng.choice(["never", "former", "current"], n)
    crp = np.exp(rng.normal(0.4, 0.9, n))
    sbp = (120 + 1.8 * np.log(crp) + 0.35 * age + 4.0 * sex + 0.9 * bmi
           + np.where(smoke == "current", 5.0, 0.0) + rng.normal(0, 12, n))
    event = rng.binomial(1, 1 / (1 + np.exp(-(-3 + 0.35 * np.log(crp) + 0.03 * (age - 62) + 0.5 * sex))))
    df = pd.DataFrame({"crp": crp, "sbp": sbp, "event": event, "age": age,
                       "sex": sex, "bmi": bmi, "smoke": smoke})
    df["crp_q"] = pd.qcut(df["crp"], 4, labels=[1, 2, 3, 4]).astype(int)
    for col, idx in [("sbp", [3, 40]), ("bmi", [7]), ("age", [11, 12])]:
        df.loc[idx, col] = np.nan
    store.save("mm", df)
    return "mm"


MODELS = [
    {"label": "Crude", "covariates": []},
    {"label": "Model 1", "covariates": ["age", "sex"]},
    {"label": "Model 2", "covariates": ["age", "sex", "bmi", "smoke"]},
]


def _run(client, sid, **extra):
    body = {"session_id": sid, "outcome": "sbp", "exposure": "crp_q",
            "outcome_kind": "continuous", "exposure_categorical": True,
            "models": MODELS, "categorical": ["smoke"], **extra}
    r = client.post("/api/multimodel/analyze", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _q4(model: dict) -> dict:
    return next(e for e in model["effects"] if e["level"] == "4")


# ── agreement with R ───────────────────────────────────────────────────────────


def test_the_quartile_effect_matches_r_in_every_model(client, cohort):
    # R: lm(sbp ~ factor(crp_q) [+ adjustment]) on the shared complete cases.
    out = _run(client, cohort)
    expected = {"Crude": (2.5490231, 1.7113936, 0.13701281),
                "Model 1": (2.8448346, 1.6072669, 0.077353094),
                "Model 2": (3.0969018, 1.4952959, 0.038877025)}
    for m in out["models"]:
        beta, se, p = expected[m["label"]]
        assert _q4(m)["beta"] == pytest.approx(beta, rel=1e-6)
        assert _q4(m)["se"] == pytest.approx(se, rel=1e-6)
        assert _q4(m)["p"] == pytest.approx(p, rel=1e-6)


def test_p_for_trend_matches_r(client, cohort):
    # R: the same models with crp_q entered as a number instead of a factor.
    out = _run(client, cohort)
    expected = {"Crude": 0.12933502, "Model 1": 0.10783198, "Model 2": 0.044024347}
    for m in out["models"]:
        assert m["trend"]["p"] == pytest.approx(expected[m["label"]], rel=1e-6)
    assert out["trend_basis"] == "level value"


def test_a_continuous_exposure_matches_r_per_unit(client, cohort):
    out = _run(client, cohort, exposure="crp", exposure_categorical=False)
    expected = {"Crude": 0.046259188, "Model 1": 0.1393435, "Model 2": 0.23138764}
    for m in out["models"]:
        assert m["effects"][0]["beta"] == pytest.approx(expected[m["label"]], rel=1e-6)
        assert m["effects"][0]["level"] == "per unit"


def test_a_binary_outcome_matches_r(client, cohort):
    # R: glm(event ~ factor(crp_q) …, binomial, epsilon = 1e-12)
    out = _run(client, cohort, outcome="event", outcome_kind="binary")
    expected = {"Crude": 1.876556005888, "Model 1": 1.9006413994, "Model 2": 1.9361748142}
    for m in out["models"]:
        assert _q4(m)["beta"] == pytest.approx(expected[m["label"]], rel=1e-8)
        assert _q4(m)["ratio"] == pytest.approx(np.exp(expected[m["label"]]), rel=1e-8)


def test_a_linear_model_uses_t_for_its_interval(client, cohort):
    # R's confint() on an lm is t-based; confint.default() forces the normal
    # one. With 490-odd residual df the two differ in the third decimal, and
    # the gap widens as n falls.
    out = _run(client, cohort)
    crude = _q4(out["models"][0])
    assert crude["ci_low"] == pytest.approx(-0.81353545, rel=1e-5)
    assert crude["ci_high"] == pytest.approx(5.9115817, rel=1e-5)


# ── the property the table is read for ─────────────────────────────────────────


def test_every_model_is_fitted_on_the_same_rows(client, cohort):
    """The whole point: a change across the row is adjustment, not attrition.

    Fitted on their own complete cases the crude model would use more rows
    than the adjusted ones, and the reader would have no way to tell whether
    the estimate moved because of the covariates or because the sample did.
    """
    out = _run(client, cohort)
    df = store.get("mm")
    expected = int(df[["sbp", "crp_q", "age", "sex", "bmi", "smoke"]].notna().all(axis=1).sum())
    assert out["n_used"] == expected
    assert out["n_dropped"] == len(df) - expected
    assert any("same complete cases" in w for w in out["warnings"])


def test_the_crude_model_does_not_get_its_own_larger_sample(client, cohort):
    # Crude uses no covariates, so on its own it could be fitted on more rows.
    # It is not, and this pins that: the union mask is what counts.
    out = _run(client, cohort)
    df = store.get("mm")
    crude_alone = int(df[["sbp", "crp_q"]].notna().all(axis=1).sum())
    assert crude_alone > out["n_used"]


def test_the_lowest_level_is_the_reference(client, cohort):
    out = _run(client, cohort)
    first = out["models"][0]["effects"][0]
    assert first["level"] == "1" and first["reference"] is True
    assert out["levels"] == ["1", "2", "3", "4"]


def test_text_levels_score_by_rank_and_say_so(client, cohort):
    df = store.get("mm").copy()
    df["band"] = pd.Categorical.from_codes(df["crp_q"] - 1, ["low", "mid", "high", "top"]).astype(str)
    store.save("mm_txt", df)
    out = _run(client, "mm_txt", exposure="band")
    assert out["trend_basis"] == "rank"
    assert any("rank" in w and "equally spaced" in w for w in out["warnings"])


# ── refusals ───────────────────────────────────────────────────────────────────


def test_the_exposure_cannot_also_be_an_adjustment_variable(client, cohort):
    r = client.post("/api/multimodel/analyze", json={
        "session_id": cohort, "outcome": "sbp", "exposure": "age",
        "models": [{"label": "M", "covariates": ["age", "sex"]}]})
    assert r.status_code == 400
    assert "cannot also be" in r.json()["detail"]


def test_no_models_is_refused(client, cohort):
    r = client.post("/api/multimodel/analyze", json={
        "session_id": cohort, "outcome": "sbp", "exposure": "crp", "models": []})
    assert r.status_code == 400


def test_an_unknown_column_is_refused(client, cohort):
    r = client.post("/api/multimodel/analyze", json={
        "session_id": cohort, "outcome": "sbp", "exposure": "crp",
        "models": [{"label": "M", "covariates": ["nope"]}]})
    assert r.status_code == 400
    assert "nope" in r.json()["detail"]


def test_a_non_binary_outcome_is_refused_for_logistic(client, cohort):
    r = client.post("/api/multimodel/analyze", json={
        "session_id": cohort, "outcome": "age", "exposure": "crp",
        "outcome_kind": "binary", "models": MODELS})
    assert r.status_code == 400


def test_the_sentence_reports_the_fully_adjusted_model(client, cohort):
    out = _run(client, cohort)
    assert "Model 2" in out["result_text"]
    assert "P for trend" in out["result_text"]
    assert f"same {out['n_used']} complete cases" in out["result_text"]
