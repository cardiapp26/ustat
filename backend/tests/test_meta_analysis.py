"""
Statistical correctness tests for the meta-analysis router (routers/meta.py).

Existing structural coverage lives in test_package8_grabbag.py,
test_package9_polish.py, and test_v2_endpoints.py — this file focuses on
verifying the *statistics* are actually correct against synthetic data with
known ground truth (pooled effect, heterogeneity direction, regression slope
sign, funnel asymmetry direction), plus error paths.

Studies use the "generic" measure (linear scale, effect + se) so pooled
means are directly comparable to the constructed true effect without log
transforms complicating the arithmetic.
"""

import math

import numpy as np
import pytest


# ── 1. Homogeneous studies clustered around a known true effect ─────────────


def _homogeneous_studies(true_effect=0.5, n=9, seed=0):
    rng = np.random.default_rng(seed)
    studies = []
    for i in range(n):
        se = 0.05 + 0.01 * (i % 3)
        eff = true_effect + rng.normal(0, se * 0.3)  # small noise, well within SE
        studies.append({"label": f"S{i+1}", "effect": round(float(eff), 4), "se": round(se, 4)})
    return studies


def _heterogeneous_studies(n=9, seed=1):
    rng = np.random.default_rng(seed)
    studies = []
    # Wildly different effects with small SEs -> large between-study variance
    # relative to within-study variance -> high I^2.
    true_effects = np.linspace(-2.0, 2.0, n)
    for i, te in enumerate(true_effects):
        se = 0.05
        eff = te + rng.normal(0, se * 0.2)
        studies.append({"label": f"H{i+1}", "effect": round(float(eff), 4), "se": round(se, 4)})
    return studies


def test_analyze_homogeneous_pooled_estimate_close_to_truth(client):
    true_effect = 0.5
    studies = _homogeneous_studies(true_effect=true_effect)
    r = client.post("/api/meta/analyze", json={
        "studies": studies, "measure": "generic", "tau2_method": "DL",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["k"] == len(studies)
    assert abs(d["random"]["effect"] - true_effect) < 0.1
    assert abs(d["fixed"]["effect"] - true_effect) < 0.1
    # Q >= 0, I2 in [0, 100]
    assert d["Q"] >= 0
    assert 0 <= d["I2_pct"] <= 100
    # Homogeneous data -> low heterogeneity
    assert d["I2_pct"] < 50


def test_analyze_homogeneous_fixed_and_random_agree_closely(client):
    studies = _homogeneous_studies(true_effect=0.5)
    r = client.post("/api/meta/analyze", json={
        "studies": studies, "measure": "generic", "tau2_method": "PM",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    # With near-zero tau2, fixed and random estimates should nearly coincide
    assert abs(d["fixed"]["effect"] - d["random"]["effect"]) < 0.05
    assert d["tau2_method"] == "PM"


def test_analyze_heterogeneous_gives_high_i2(client):
    studies = _heterogeneous_studies()
    r = client.post("/api/meta/analyze", json={
        "studies": studies, "measure": "generic", "tau2_method": "DL",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["I2_pct"] > 50
    assert d["Q"] > 0
    assert d["tau2"] > 0


def test_analyze_heterogeneous_i2_exceeds_homogeneous_i2(client):
    homo = client.post("/api/meta/analyze", json={
        "studies": _homogeneous_studies(), "measure": "generic",
    }).json()
    hetero = client.post("/api/meta/analyze", json={
        "studies": _heterogeneous_studies(), "measure": "generic",
    }).json()
    assert hetero["I2_pct"] > homo["I2_pct"]


def test_analyze_or_measure_pools_near_true_odds_ratio(client):
    # log-scale measure: build studies whose effects cluster around OR = 2.0
    rng = np.random.default_rng(2)
    studies = []
    for i in range(8):
        se = 0.1
        log_eff = math.log(2.0) + rng.normal(0, 0.02)
        studies.append({"label": f"OR{i+1}", "effect": round(math.exp(log_eff), 4), "se": se})
    r = client.post("/api/meta/analyze", json={"studies": studies, "measure": "OR"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["log_scale"] is True
    assert abs(d["random"]["effect"] - 2.0) < 0.3
    assert d["null_line"] == 1.0


# ── 2. Subgroup analysis ─────────────────────────────────────────────────────


def test_subgroup_pooled_estimates_differ_in_expected_direction(client):
    rng = np.random.default_rng(3)
    studies = []
    for i in range(5):
        eff = 0.2 + rng.normal(0, 0.02)
        studies.append({"label": f"LOW{i+1}", "effect": round(float(eff), 4), "se": 0.05, "subgroup": "low"})
    for i in range(5):
        eff = 1.5 + rng.normal(0, 0.02)
        studies.append({"label": f"HIGH{i+1}", "effect": round(float(eff), 4), "se": 0.05, "subgroup": "high"})

    r = client.post("/api/meta/subgroup", json={"studies": studies, "measure": "generic"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["subgroups"]) == 2
    by_name = {s["subgroup"]: s for s in d["subgroups"]}
    assert by_name["low"]["effect"] < by_name["high"]["effect"]
    assert abs(by_name["low"]["effect"] - 0.2) < 0.1
    assert abs(by_name["high"]["effect"] - 1.5) < 0.1
    # Between-subgroup heterogeneity should be significant given the huge gap
    assert d["q_between"] is not None
    assert d["q_between_p"] < 0.05


def test_subgroup_requires_subgroup_field(client):
    studies = _homogeneous_studies()  # no subgroup key
    r = client.post("/api/meta/subgroup", json={"studies": studies, "measure": "generic"})
    assert r.status_code == 422


# ── 3. Meta-regression ───────────────────────────────────────────────────────


def test_regression_recovers_positive_slope_with_low_p(client):
    rng = np.random.default_rng(4)
    studies = []
    true_slope = 0.3
    for i in range(10):
        mod = float(i)
        eff = 0.1 + true_slope * mod + rng.normal(0, 0.02)
        studies.append({
            "label": f"R{i+1}", "effect": round(float(eff), 4), "se": 0.05, "moderator": mod,
        })
    r = client.post("/api/meta/regression", json={"studies": studies, "measure": "generic"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["slope"] > 0
    assert abs(d["slope"] - true_slope) < 0.05
    assert d["slope_p"] < 0.05


def test_regression_recovers_negative_slope(client):
    rng = np.random.default_rng(5)
    studies = []
    true_slope = -0.4
    for i in range(10):
        mod = float(i)
        eff = 5.0 + true_slope * mod + rng.normal(0, 0.02)
        studies.append({
            "label": f"NR{i+1}", "effect": round(float(eff), 4), "se": 0.05, "moderator": mod,
        })
    r = client.post("/api/meta/regression", json={"studies": studies, "measure": "generic"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["slope"] < 0
    assert d["slope_p"] < 0.05


def test_regression_requires_moderator_on_every_study(client):
    studies = _homogeneous_studies(n=5)  # no moderator
    r = client.post("/api/meta/regression", json={"studies": studies, "measure": "generic"})
    assert r.status_code == 422


def test_regression_requires_at_least_three_studies(client):
    studies = [
        {"label": "A", "effect": 0.5, "se": 0.1, "moderator": 1.0},
        {"label": "B", "effect": 0.6, "se": 0.1, "moderator": 2.0},
    ]
    r = client.post("/api/meta/regression", json={"studies": studies, "measure": "generic"})
    assert r.status_code == 422


# ── 4. Publication-bias diagnostics ──────────────────────────────────────────


def _symmetric_funnel_studies(n=12, seed=6):
    """Symmetric funnel: effect noise independent of precision."""
    rng = np.random.default_rng(seed)
    studies = []
    for i in range(n):
        se = 0.05 + 0.4 * (i / n)  # ranges from small to large SE
        eff = 0.5 + rng.normal(0, se)  # noise scales with se -> symmetric
        studies.append({"label": f"SYM{i+1}", "effect": round(float(eff), 4), "se": round(se, 4)})
    return studies


def _asymmetric_funnel_studies(n=12, seed=7):
    """Small (imprecise) studies are systematically shifted to larger effects,
    while large (precise) studies stay near the true effect -- classic
    small-study / publication-bias funnel asymmetry."""
    rng = np.random.default_rng(seed)
    studies = []
    for i in range(n):
        se = 0.05 + 0.4 * (i / n)
        # bias term grows with se -> small studies (large se) inflated upward
        bias = 1.5 * se
        eff = 0.5 + bias + rng.normal(0, se * 0.3)
        studies.append({"label": f"ASYM{i+1}", "effect": round(float(eff), 4), "se": round(se, 4)})
    return studies


def test_bias_asymmetric_funnel_has_lower_egger_p_than_symmetric(client):
    sym = client.post("/api/meta/bias", json={
        "studies": _symmetric_funnel_studies(), "measure": "generic",
    })
    asym = client.post("/api/meta/bias", json={
        "studies": _asymmetric_funnel_studies(), "measure": "generic",
    })
    assert sym.status_code == 200, sym.text
    assert asym.status_code == 200, asym.text
    sym_d, asym_d = sym.json(), asym.json()
    assert asym_d["egger_p"] < sym_d["egger_p"]
    assert asym_d["egger_p"] < 0.05
    # Symmetric case should not show strong asymmetry
    assert sym_d["egger_p"] > 0.05


def test_bias_funnel_and_pooled_effect_present(client):
    r = client.post("/api/meta/bias", json={
        "studies": _asymmetric_funnel_studies(), "measure": "generic",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["funnel"]) == 12
    assert "pooled_effect" in d
    assert "trim_fill_missing" in d
    assert d["trim_fill_missing"] >= 0


def test_bias_requires_at_least_three_studies(client):
    studies = [
        {"label": "A", "effect": 0.5, "se": 0.1},
        {"label": "B", "effect": 0.6, "se": 0.1},
    ]
    r = client.post("/api/meta/bias", json={"studies": studies, "measure": "generic"})
    assert r.status_code == 422


# ── 5. Error paths ───────────────────────────────────────────────────────────


def test_analyze_rejects_single_study(client):
    r = client.post("/api/meta/analyze", json={
        "studies": [{"label": "Solo", "effect": 0.5, "se": 0.1}],
        "measure": "generic",
    })
    assert r.status_code == 422
    assert r.status_code < 500


def test_analyze_rejects_empty_studies(client):
    r = client.post("/api/meta/analyze", json={"studies": [], "measure": "generic"})
    assert r.status_code == 422


def test_analyze_rejects_study_with_no_usable_inputs(client):
    studies = [
        {"label": "A", "effect": 0.5, "se": 0.1},
        {"label": "Bad"},  # no effect/CI/SE/2x2 at all
    ]
    r = client.post("/api/meta/analyze", json={"studies": studies, "measure": "generic"})
    assert r.status_code == 422
    assert "Bad" in r.json()["detail"]


def test_analyze_rejects_nonpositive_effect_on_log_scale(client):
    studies = [
        {"label": "A", "effect": 1.2, "se": 0.1},
        {"label": "B", "effect": -0.5, "se": 0.1},
    ]
    r = client.post("/api/meta/analyze", json={"studies": studies, "measure": "OR"})
    assert r.status_code == 422


def test_analyze_rejects_nonpositive_se(client):
    studies = [
        {"label": "A", "effect": 0.5, "se": 0.1},
        {"label": "B", "effect": 0.6, "se": 0.0},
    ]
    r = client.post("/api/meta/analyze", json={"studies": studies, "measure": "generic"})
    assert r.status_code == 422


def test_analyze_accepts_2x2_table_input(client):
    studies = [
        {"label": "A", "e1": 10, "n1": 100, "e2": 30, "n2": 100},
        {"label": "B", "e1": 12, "n1": 100, "e2": 28, "n2": 100},
        {"label": "C", "e1": 8, "n1": 100, "e2": 32, "n2": 100},
    ]
    r = client.post("/api/meta/analyze", json={"studies": studies, "measure": "OR"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["k"] == 3
    assert d["random"]["effect"] < 1.0  # treated arm has fewer events -> OR < 1


def test_analyze_rejects_unsupported_measure_for_2x2(client):
    studies = [
        {"label": "A", "e1": 10, "n1": 100, "e2": 30, "n2": 100},
        {"label": "B", "e1": 12, "n1": 100, "e2": 28, "n2": 100},
    ]
    r = client.post("/api/meta/analyze", json={"studies": studies, "measure": "SMD"})
    assert r.status_code == 422


# ── REML τ², Hartung-Knapp CI, tau2_method validation ───────────────────────


def test_analyze_reml_tau2_between_estimates(client):
    # Unequal SEs matter: with equal within-study variances DL, PM and REML
    # coincide exactly, and the test would pass against a silent fallback.
    rng = np.random.default_rng(5)
    studies = [
        {"label": f"R{i}", "effect": round(float(te + rng.normal(0, 0.05)), 4),
         "se": round(0.05 + 0.06 * (i % 4), 4)}
        for i, te in enumerate(np.linspace(-1.5, 1.5, 9))
    ]
    out = {}
    for m in ("DL", "PM", "REML"):
        r = client.post("/api/meta/analyze", json={
            "studies": studies, "measure": "generic", "tau2_method": m,
        })
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["tau2_method"] == m
        out[m] = d["tau2"]
    # All three see the same large heterogeneity; REML must be a real,
    # distinct estimate, not a silent DL fallback.
    assert out["REML"] > 0
    assert out["REML"] != out["DL"]


def test_analyze_reml_matches_restricted_likelihood_argmax(client):
    studies = [
        {"label": "A", "effect": 0.5, "se": 0.10},
        {"label": "B", "effect": 0.3, "se": 0.12},
        {"label": "C", "effect": 0.8, "se": 0.09},
        {"label": "D", "effect": 0.4, "se": 0.15},
        {"label": "E", "effect": 0.6, "se": 0.11},
    ]
    r = client.post("/api/meta/analyze", json={
        "studies": studies, "measure": "generic", "tau2_method": "REML",
    })
    d = r.json()
    y = np.array([s["effect"] for s in studies])
    v = np.array([s["se"] for s in studies]) ** 2

    def restll(t2):
        w = 1 / (v + t2)
        mu = (w * y).sum() / w.sum()
        return -0.5 * (np.log(v + t2).sum() + (w * (y - mu) ** 2).sum() + np.log(w.sum()))

    grid = np.linspace(0, 0.2, 20001)
    argmax = grid[int(np.argmax([restll(t) for t in grid]))]
    assert abs(d["tau2"] - argmax) < 1e-4


def test_analyze_hartung_knapp_widens_ci_and_labels_it(client):
    studies = _heterogeneous_studies()
    base = client.post("/api/meta/analyze", json={
        "studies": studies, "measure": "generic", "tau2_method": "REML",
    }).json()
    hk = client.post("/api/meta/analyze", json={
        "studies": studies, "measure": "generic", "tau2_method": "REML",
        "hartung_knapp": True,
    }).json()
    assert base["ci_method"] == "Wald (z)"
    assert hk["ci_method"].startswith("Hartung-Knapp")
    assert hk["hartung_knapp"] is True
    # Same point estimate, different interval machinery.
    assert hk["random"]["effect"] == base["random"]["effect"]
    width_base = base["random"]["ci_high"] - base["random"]["ci_low"]
    width_hk = hk["random"]["ci_high"] - hk["random"]["ci_low"]
    assert width_hk != width_base


def test_analyze_unknown_tau2_method_is_rejected_not_mislabelled(client):
    """'HK' used to come back 200, labelled HK, computed as DL."""
    studies = _homogeneous_studies()
    r = client.post("/api/meta/analyze", json={
        "studies": studies, "measure": "generic", "tau2_method": "HK",
    })
    assert r.status_code == 422
    assert "Unknown tau2_method" in r.text
    assert "hartung_knapp" in r.text  # points at the correct knob


def test_subgroup_accepts_reml_and_hk(client):
    studies = _heterogeneous_studies()
    for i, s in enumerate(studies):
        s["subgroup"] = "early" if i < 4 else "late"
    r = client.post("/api/meta/subgroup", json={
        "studies": studies, "measure": "generic", "tau2_method": "REML",
        "hartung_knapp": True,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["subgroups"]) == 2
    for g in d["subgroups"]:
        assert g["ci_low"] < g["effect"] < g["ci_high"]


# ── I² and H² follow the τ² estimator ───────────────────────────────────────
#
# Reference: base R 4.5.2, independent of uSTAT's algorithms. DL in closed
# form, PM by uniroot on the generalised Q equation, REML by optimize on the
# restricted log-likelihood; I² = 100 τ²/(τ² + ṽ) and H² = (τ² + ṽ)/ṽ with
# metafor's typical within-study variance ṽ. I² used to be (Q - df)/Q under
# every estimator, so PM and REML printed DL's 85.59%.
_I2_STUDIES = [
    {"label": f"s{i}", "effect": e, "se": s}
    for i, (e, s) in enumerate(zip(
        [0.2, 0.8, -0.1, 0.5, 0.05], [0.10, 0.15, 0.12, 0.20, 0.08]))
]
_I2_R = {
    "DL": (0.08213115, 85.589350, 6.939312),
    "PM": (0.11095793, 88.918341, 9.023920),
    "REML": (0.10936465, 88.775021, 8.908702),
}


def test_i2_and_h2_follow_the_tau2_estimator(client):
    for method, (tau2, i2, h2) in _I2_R.items():
        r = client.post("/api/meta/analyze", json={
            "studies": _I2_STUDIES, "measure": "generic", "tau2_method": method,
        })
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["tau2"] == pytest.approx(tau2, abs=1e-5), method
        assert d["I2_pct"] == pytest.approx(i2, abs=0.01), method
        assert d["H2"] == pytest.approx(h2, abs=1e-3), method


def test_dl_i2_equals_the_q_based_formula(client):
    # Under DL the τ²-based I² is algebraically (Q - df)/Q.
    r = client.post("/api/meta/analyze", json={
        "studies": _I2_STUDIES, "measure": "generic", "tau2_method": "DL",
    })
    d = r.json()
    assert d["I2_pct"] == pytest.approx(100 * (d["Q"] - d["Q_df"]) / d["Q"], abs=0.01)
