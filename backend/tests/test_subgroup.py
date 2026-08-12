"""Subgroup analysis: per-stratum effects and P for interaction.

Reference values are R's, on the same complete cases. The per-stratum figures
come from lm/glm fitted inside each level; the interaction p comes from a
likelihood-ratio test between `y ~ treat * subgroup + covariates` and the same
model without the product term, on the whole sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services import store


@pytest.fixture()
def trial(client) -> str:
    """A treatment that works in women and not in men — a real interaction."""
    rng = np.random.default_rng(913)
    n = 600
    treat = rng.integers(0, 2, n).astype(float)
    sex = rng.choice(["F", "M"], n)
    agegrp = rng.choice(["<65", "65-74", ">=75"], n)
    age = rng.normal(68, 9, n)
    bmi = rng.normal(27, 4, n)
    sbp = (140 + np.where(sex == "F", -6.0, -0.5) * treat + 0.25 * age + 0.6 * bmi
           + rng.normal(0, 10, n))
    lp = -1.2 + np.where(sex == "F", -1.1, -0.1) * treat + 0.02 * (age - 68)
    event = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    df = pd.DataFrame({"treat": treat, "sbp": sbp, "event": event, "sex": sex,
                       "agegrp": agegrp, "age": age, "bmi": bmi})
    for col, idx in [("sbp", [4, 9]), ("bmi", [17]), ("sex", [200])]:
        df.loc[idx, col] = np.nan
    store.save("sg", df)
    return "sg"


def _run(client, sid, **extra):
    body = {"session_id": sid, "outcome": "sbp", "exposure": "treat",
            "subgroups": ["sex", "agegrp"], "covariates": ["age", "bmi"],
            "outcome_kind": "continuous", **extra}
    r = client.post("/api/subgroup/analyze", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _block(out, name):
    return next(b for b in out["subgroups"] if b["variable"] == name)


def _row(block, level):
    return next(r for r in block["rows"] if r["level"] == level)


# ── agreement with R ───────────────────────────────────────────────────────────


def test_per_stratum_effects_match_r(client, trial):
    # R: lm(sbp ~ treat + age + bmi) inside each level of sex.
    out = _run(client, trial)
    sex = _block(out, "sex")
    assert _row(sex, "F")["beta"] == pytest.approx(-5.75856218107, rel=1e-9)
    assert _row(sex, "F")["se"] == pytest.approx(1.11385672964, rel=1e-9)
    assert _row(sex, "F")["p"] == pytest.approx(4.35254943111e-07, rel=1e-8)
    assert _row(sex, "F")["ci_low"] == pytest.approx(-7.95076745502, rel=1e-9)
    assert _row(sex, "M")["beta"] == pytest.approx(0.505136927746, rel=1e-9)
    assert _row(sex, "F")["n"] == 296 and _row(sex, "M")["n"] == 300


def test_p_for_interaction_matches_r(client, trial):
    # R: 2 * (logLik(sbp ~ treat * sex + age + bmi) - logLik(sbp ~ treat + sex
    # + age + bmi)) on 1 df, over the rows complete for all of them.
    out = _run(client, trial)
    assert _block(out, "sex")["p_interaction"] == pytest.approx(0.000136982065557, rel=1e-8)
    assert _block(out, "agegrp")["p_interaction"] == pytest.approx(0.495342860452, rel=1e-8)
    assert "1 df" in _block(out, "sex")["interaction_note"]
    assert "2 df" in _block(out, "agegrp")["interaction_note"]


def test_a_binary_outcome_matches_r(client, trial):
    out = _run(client, trial, outcome="event", outcome_kind="binary")
    sex = _block(out, "sex")
    # R agrees on the coefficient; its standard error is the one that differs,
    # for the reason the next test pins down.
    assert _row(sex, "F")["beta"] == pytest.approx(-0.547914372322, rel=1e-9)
    assert _row(sex, "F")["ratio"] == pytest.approx(np.exp(-0.547914372322), rel=1e-9)
    assert _row(sex, "F")["n"] == 297


def test_the_standard_error_is_the_analytic_one(client, trial):
    """statsmodels' default IRLS tolerance leaves the SE one iterate stale.

    At the default the standard error comes out 1.1e-7 away from the analytic
    sqrt(diag((X'WX)^-1)) evaluated at the fitted coefficients; the endpoint
    tightens the tolerance so it is exact to ~1e-13. R's glm keeps the stale
    value even at epsilon = 1e-12, which is the last small disagreement
    between the two.
    """
    import statsmodels.api as sm

    out = _run(client, trial, outcome="event", outcome_kind="binary")
    df = store.get("sg")
    sub = df[df["sex"] == "F"][["event", "treat", "age", "bmi"]].dropna()
    X = sm.add_constant(sub[["treat", "age", "bmi"]], has_constant="add").to_numpy(float)
    beta = np.array([sm.GLM(sub["event"], sm.add_constant(sub[["treat", "age", "bmi"]], has_constant="add"),
                            family=sm.families.Binomial()).fit(maxiter=200, tol=1e-12).params[c]
                     for c in ["const", "treat", "age", "bmi"]])
    p = 1 / (1 + np.exp(-(X @ beta)))
    exact = float(np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))))[1])
    assert _row(_block(out, "sex"), "F")["se"] == pytest.approx(exact, rel=1e-10)


# ── the distinction the panel exists to preserve ───────────────────────────────


def test_the_interaction_comes_from_the_whole_sample_not_the_strata(client, trial):
    """Two strata either side of the null, and no interaction to speak of.

    Age group is generated with no interaction at all, yet its strata carry
    different point estimates and one can look 'significant' while another
    does not. The interaction p is what says they do not really differ, and it
    cannot be recovered from the stratum rows.
    """
    out = _run(client, trial)
    age = _block(out, "agegrp")
    assert age["p_interaction"] > 0.4
    betas = [r["beta"] for r in age["rows"]]
    assert max(betas) - min(betas) > 2.0  # they look different; they are not


def test_the_caveat_ships_with_the_result(client, trial):
    out = _run(client, trial)
    assert "difference in significance" not in out["caveat"]
    assert "not evidence that the effect differs" in out["caveat"]


def test_each_subgroup_gets_its_own_complete_cases(client, trial):
    # sex has one missing value and agegrp none, so the two must not be forced
    # onto a shared mask that would drop the row from both.
    out = _run(client, trial)
    assert _block(out, "agegrp")["n_used"] > _block(out, "sex")["n_used"]


def test_a_constant_covariate_inside_a_stratum_is_dropped(client, trial):
    # Adjusting for sex inside "F" is a column of ones: no information, and a
    # singular design if kept.
    out = _run(client, trial, subgroups=["sex"], covariates=["age", "bmi", "agegrp"],
               categorical=["agegrp"])
    assert _row(_block(out, "sex"), "F")["beta"] is not None


def test_a_thin_stratum_is_reported_and_flagged(client):
    rng = np.random.default_rng(2)
    n = 220
    grp = np.array(["big"] * 205 + ["tiny"] * 15)
    x = rng.integers(0, 2, n).astype(float)
    store.save("sg_thin", pd.DataFrame({"y": 10 + 2 * x + rng.normal(0, 3, n), "x": x, "g": grp}))
    r = client.post("/api/subgroup/analyze", json={
        "session_id": "sg_thin", "outcome": "y", "exposure": "x", "subgroups": ["g"]})
    assert r.status_code == 200
    out = r.json()
    tiny = _row(_block(out, "g"), "tiny")
    assert tiny["thin"] is True and tiny["n"] == 15
    assert any("fewer than 20" in w for w in out["warnings"])


def test_an_identifier_is_refused_as_a_subgroup(client, trial):
    df = store.get("sg").copy()
    df["pid"] = [f"P{i}" for i in range(len(df))]
    store.save("sg_ids", df)
    out = _run(client, "sg_ids", subgroups=["sex", "pid"])
    assert [b["variable"] for b in out["subgroups"]] == ["sex"]
    assert any("identifier" in w for w in out["warnings"])


def test_the_exposure_cannot_also_be_a_subgroup(client, trial):
    r = client.post("/api/subgroup/analyze", json={
        "session_id": trial, "outcome": "sbp", "exposure": "treat", "subgroups": ["treat"]})
    assert r.status_code == 400
    assert "cannot also be" in r.json()["detail"]


def test_no_subgroup_variable_is_refused(client, trial):
    r = client.post("/api/subgroup/analyze", json={
        "session_id": trial, "outcome": "sbp", "exposure": "treat", "subgroups": []})
    assert r.status_code == 400


def test_the_overall_row_is_the_reference(client, trial):
    out = _run(client, trial)
    assert out["overall"]["n"] == 597
    assert out["overall"]["beta"] is not None
    assert out["null_value"] == 0.0


def test_a_ratio_outcome_gets_a_null_of_one(client, trial):
    out = _run(client, trial, outcome="event", outcome_kind="binary")
    assert out["null_value"] == 1.0
    assert out["effect_label"] == "Odds ratio"
