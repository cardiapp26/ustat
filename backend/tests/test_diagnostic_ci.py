"""Diagnostic accuracy CIs (Wilson, Simel LR) and their ROC endpoint wiring."""

import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from statsmodels.stats.proportion import proportion_confint

from main import app
from services import store
from services.diagnostic_ci import fagan_post_test, simel_lr_ci, wilson_ci

client = TestClient(app)


def test_wilson_matches_hand_value():
    lo, hi = wilson_ci(90, 100)
    assert lo == pytest.approx(0.8256, abs=5e-4)
    assert hi == pytest.approx(0.9448, abs=5e-4)


@pytest.mark.parametrize("k,n", [(0, 10), (10, 10), (1, 7), (45, 100), (3, 3000)])
def test_wilson_matches_statsmodels(k, n):
    lo, hi = wilson_ci(k, n)
    slo, shi = proportion_confint(k, n, alpha=0.05, method="wilson")
    assert lo == pytest.approx(slo, abs=1e-9)
    assert hi == pytest.approx(shi, abs=1e-9)


def test_wilson_undefined_for_empty_denominator():
    assert wilson_ci(0, 0) is None


def test_simel_lr_pos_hand_value():
    # tp=90 fn=10 fp=20 tn=80: LR+ = 0.9 / 0.2 = 4.5
    lo, hi = simel_lr_ci(90, 10, 20, 80, "pos")
    se = math.sqrt(1 / 90 - 1 / 100 + 1 / 20 - 1 / 100)
    assert lo == pytest.approx(math.exp(math.log(4.5) - 1.959964 * se), rel=1e-4)
    assert hi == pytest.approx(math.exp(math.log(4.5) + 1.959964 * se), rel=1e-4)
    assert lo == pytest.approx(3.024, abs=2e-3)
    assert hi == pytest.approx(6.696, abs=2e-3)
    assert lo < 4.5 < hi


def test_simel_lr_neg_hand_value():
    # LR- = 0.1 / 0.8 = 0.125
    lo, hi = simel_lr_ci(90, 10, 20, 80, "neg")
    assert lo == pytest.approx(0.0689, abs=5e-4)
    assert hi == pytest.approx(0.2269, abs=5e-4)
    assert lo < 0.125 < hi


def test_simel_continuity_correction_when_a_cell_is_zero():
    ci = simel_lr_ci(50, 0, 10, 40, "neg")
    assert ci is not None
    assert all(math.isfinite(v) and v > 0 for v in ci)
    ci_pos = simel_lr_ci(50, 10, 0, 40, "pos")
    assert ci_pos is not None and ci_pos[0] < ci_pos[1]


def test_simel_undefined_for_empty_group():
    assert simel_lr_ci(0, 0, 5, 5, "pos") is None
    assert simel_lr_ci(5, 5, 0, 0, "neg") is None


def test_fagan_post_test():
    out = fagan_post_test(0.2, 0.9, 0.8)
    # odds 0.25 * LR+ 4.5 = 1.125 -> 0.5294 ; odds 0.25 * LR- 0.125 = 0.03125 -> 0.0303
    assert out["post_positive"] == pytest.approx(1.125 / 2.125, abs=1e-9)
    assert out["post_negative"] == pytest.approx(0.03125 / 1.03125, abs=1e-9)
    assert fagan_post_test(0.0, 0.9, 0.8) is None
    assert fagan_post_test(1.0, 0.9, 0.8) is None


# ── Endpoint wiring ────────────────────────────────────────────────────────────


def _roc_df(n=200, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    score = rng.normal(0, 1, n) + 1.2 * y
    return pd.DataFrame({"score": score, "y": y})


def test_roc_endpoint_surfaces_cis_but_not_on_curve():
    store.save("diag_ci_roc", _roc_df())
    r = client.post(
        "/api/stats/roc",
        json={
            "session_id": "diag_ci_roc",
            "score_column": "score",
            "outcome_column": "y",
            "manual_cutoff": 0.5,
            "prevalence": 0.2,
        },
    )
    assert r.status_code == 200, r.text
    d = r.json()
    for block in (d["optimal"], d["manual"]):
        for key in ("sensitivity", "specificity", "ppv", "npv"):
            lo, hi = block[f"{key}_ci"]
            assert lo <= block[key] <= hi
        for key in ("lr_pos", "lr_neg"):
            ci = block[f"{key}_ci"]
            assert ci is None or ci[0] < ci[1]
    o = d["optimal"]
    slo, shi = proportion_confint(o["tp"], o["tp"] + o["fn"], method="wilson")
    assert o["sensitivity_ci"] == pytest.approx([slo, shi], abs=1e-4)
    assert "95% CI" in d["result_text"]
    assert all("sensitivity_ci" not in pt for pt in d["curve"])
    pt = d["post_test"]
    assert pt["pretest"] == 0.2
    assert pt["post_negative"] < 0.2 < pt["post_positive"]


def test_roc_endpoint_without_prevalence_has_null_post_test_and_validates():
    store.save("diag_ci_roc2", _roc_df())
    base = {"session_id": "diag_ci_roc2", "score_column": "score", "outcome_column": "y"}
    r = client.post("/api/stats/roc", json=base)
    assert r.status_code == 200
    assert r.json()["post_test"] is None
    assert client.post("/api/stats/roc", json={**base, "prevalence": 1.5}).status_code == 422


def test_roc_combined_optimal_has_cis():
    rng = np.random.default_rng(5)
    n = 200
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {"a": rng.normal(0, 1, n) + y, "b": rng.normal(0, 1, n) + 0.5 * y, "y": y}
    )
    store.save("diag_ci_comb", df)
    r = client.post(
        "/api/stats/roc_combined",
        json={"session_id": "diag_ci_comb", "predictor_columns": ["a", "b"], "outcome_column": "y"},
    )
    assert r.status_code == 200, r.text
    o = r.json()["optimal"]
    assert o["sensitivity_ci"][0] <= o["sensitivity"] <= o["sensitivity_ci"][1]
    assert o["ppv_ci"] is not None
