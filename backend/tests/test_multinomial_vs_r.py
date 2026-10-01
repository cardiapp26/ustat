"""Multinomial logistic regression against R's nnet::multinom.

Reference: R 4.5.2, nnet 7.3, on qa/models_audit/dataset.csv:

    multinom(factor(stage) ~ age + bmi + arm + sex, reltol = 1e-14, maxit = 2000)

with SEs from summary(), Wald p = 2 pnorm(-|b/se|), and each predictor's LR
test from the fit without it (update(fit, . ~ . - term)).
"""
import pandas as pd
import pytest

from services import store

DATA = "../qa/models_audit/dataset.csv"

# (category, term) -> (coef, se, p)
R_COEF = {
    ("II", "Intercept"): (-2.352658896287, 1.207671506003, 0.0514035896754),
    ("II", "age"): (0.003960558242, 0.012279930000, 0.747056616417),
    ("II", "bmi"): (0.056774758314, 0.033692371416, 0.0919707497148),
    ("II", "arm_treat"): (0.022454843558, 0.270767737363, 0.933906979108),
    ("III", "Intercept"): (-2.504509499695, 1.357615892400, 0.0650688132531),
    ("III", "age"): (0.009842928816, 0.013902616121, 0.478950769418),
    ("III", "bmi"): (0.043071759106, 0.037883942622, 0.255563486166),
    ("III", "arm_treat"): (0.036430551899, 0.303227481627, 0.904370150628),
}
R_LOGLIK, R_AIC = -311.7229188210, 643.4458376421
R_LR = {"age": (0.5094760038, 2), "bmi": (3.2168978649, 2), "arm": (0.0163075895, 2), "sex": (1.6947681553, 2)}


@pytest.fixture()
def sid():
    store.save("multinomial_vs_r", pd.read_csv(DATA))
    return "multinomial_vs_r"


def _fit(client, sid, **kw):
    body = {"session_id": sid, "outcome": "stage", "predictors": ["age", "bmi", "arm", "sex"], **kw}
    return client.post("/api/models/multinomial", json=body)


def test_coefficients_ses_and_p_match_nnet(client, sid):
    r = _fit(client, sid)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["categories"] == ["I", "II", "III"] and out["reference"] == "I"
    got = {(eq["category"], c["variable"]): c for eq in out["equations"] for c in eq["coefficients"]}
    for key, (b, se, p) in R_COEF.items():
        assert got[key]["log_rrr"] == pytest.approx(b, rel=1e-5, abs=1e-7), key
        assert got[key]["se"] == pytest.approx(se, rel=1e-6), key
        assert got[key]["p"] == pytest.approx(p, rel=1e-5), key
    assert out["log_likelihood"] == pytest.approx(R_LOGLIK, rel=1e-9)
    assert out["aic"] == pytest.approx(R_AIC, rel=1e-9)


def test_per_predictor_lr_tests_match_nested_nnet_fits(client, sid):
    out = _fit(client, sid).json()
    got = {t["variable"]: t for t in out["lr_tests"]}
    for term, (lr, df) in R_LR.items():
        assert got[term]["lr_chi2"] == pytest.approx(lr, rel=1e-6, abs=1e-8), term
        assert got[term]["df"] == df


def test_reference_category_can_be_chosen(client, sid):
    out = _fit(client, sid, reference="III").json()
    assert out["reference"] == "III"
    assert [eq["category"] for eq in out["equations"]] == ["I", "II"]
    # Re-basing changes the equations, not the fit.
    assert out["log_likelihood"] == pytest.approx(R_LOGLIK, rel=1e-9)


def test_unknown_reference_is_rejected(client, sid):
    r = _fit(client, sid, reference="IV")
    assert r.status_code == 422
    assert "not a category" in r.json()["detail"]


def test_two_categories_point_to_logistic(client, sid):
    r = client.post("/api/models/multinomial", json={
        "session_id": sid, "outcome": "event_binary", "predictors": ["age"]})
    assert r.status_code == 422
    assert "logistic regression" in r.json()["detail"]


def test_classification_table_counts_every_case(client, sid):
    out = _fit(client, sid).json()
    table = out["classification_table"]["table"]
    assert sum(map(sum, table)) == out["n"] == 300
    assert [sum(row) for row in table] == [out["category_counts"][c] for c in out["categories"]]


def test_complete_separation_is_refused_not_served_as_blanks(client):
    # x orders the three categories perfectly: every estimate diverges and
    # used to come back as a table of nulls with no warning.
    df = pd.DataFrame({
        "y": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
        "x": list(range(20)) + list(range(100, 120)) + list(range(50, 70)),
    })
    store.save("multinomial_separation", df)
    r = client.post("/api/models/multinomial", json={
        "session_id": "multinomial_separation", "outcome": "y", "predictors": ["x"]})
    assert r.status_code == 422
    assert "separate" in r.json()["detail"]


def test_quasi_separation_in_one_equation_is_flagged(client):
    # Category 'c' never occurs with g = 1: its g coefficient runs off to
    # -infinity while the other equation stays estimable.
    rows = (
        [("a", 0, x) for x in range(25)] + [("a", 1, x) for x in range(25)]
        + [("b", 0, x) for x in range(25)] + [("b", 1, x) for x in range(25)]
        + [("c", 0, x) for x in range(25)]
    )
    df = pd.DataFrame(rows, columns=["y", "g", "x"])
    store.save("multinomial_quasi", df)
    r = client.post("/api/models/multinomial", json={
        "session_id": "multinomial_quasi", "outcome": "y", "predictors": ["g", "x"]})
    assert r.status_code == 200, r.text
    assert any("separation" in w for w in r.json()["warnings"])
