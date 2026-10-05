"""Weighted Cohen's kappa: Fleiss-Cohen-Everitt (1969) SE, CI and null z test."""
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from conftest import make_session
from services.weighted_kappa import agreement_weights, weighted_kappa_stats

HAVE_R = shutil.which("Rscript") is not None

CM = np.array([[20, 5, 1], [4, 30, 6], [2, 3, 19]])


def _frame(cm):
    rows = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            rows.extend([{"x": i + 1, "y": j + 1}] * int(cm[i, j]))
    return pd.DataFrame(rows)


def _r_has(pkg: str) -> bool:
    if not HAVE_R:
        return False
    out = subprocess.run(
        ["Rscript", "-e", f"cat(requireNamespace('{pkg}', quietly=TRUE))"],
        capture_output=True, text=True, timeout=60,
    )
    return out.stdout.strip() == "TRUE"


# --------------------------------------------------------------------------
# weights and kappa value
# --------------------------------------------------------------------------

def test_agreement_weight_matrices():
    assert np.allclose(agreement_weights(3, "linear"), [[1, .5, 0], [.5, 1, .5], [0, .5, 1]])
    assert np.allclose(agreement_weights(3, "quadratic"), [[1, .75, 0], [.75, 1, .75], [0, .75, 1]])
    assert np.allclose(agreement_weights(4, None), np.eye(4))
    # with two categories every weighting collapses to identity
    for kind in ("linear", "quadratic"):
        assert np.allclose(agreement_weights(2, kind), np.eye(2))
    with pytest.raises(ValueError):
        agreement_weights(3, "cubic")


@pytest.mark.parametrize("kind", [None, "linear", "quadratic"])
def test_kappa_value_equals_sklearn(kind):
    frame = _frame(CM)
    expected = cohen_kappa_score(frame["x"], frame["y"], weights=kind)
    assert weighted_kappa_stats(CM, kind)["kappa"] == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------
# variances
# --------------------------------------------------------------------------

def test_identity_weights_reproduce_nominal_null_se_and_z(client):
    """Null variance with identity weights is exactly the Fleiss (1981) variance
    the unweighted endpoint already reports."""
    sid = make_session(_frame(CM), "wk_identity")
    body = client.post("/api/stats/cohens_kappa", json={"session_id": sid, "rater1_col": "x", "rater2_col": "y"}).json()
    res = weighted_kappa_stats(CM, None)
    assert res["se_null"] == pytest.approx(body["se_null"], rel=1e-12)
    assert res["z"] == pytest.approx(body["z"], rel=1e-12)
    assert res["p"] == pytest.approx(body["p"], rel=1e-10)
    assert res["po"] == pytest.approx(body["po"]) and res["pe"] == pytest.approx(body["pe"])
    # The legacy unweighted SE is Cohen's simplified po(1-po)/(n(1-pe)^2); the full FCE
    # variance is within about a percent of it (R psych: 0.0687246 vs 0.0683946 here).
    assert res["se"] == pytest.approx(body["se"], rel=0.01)


def test_hand_computed_values_3x3():
    # reference values from R psych::cohen.kappa / DescTools::CohenKappa on CM
    lin = weighted_kappa_stats(CM, "linear")
    assert lin["kappa"] == pytest.approx(0.6717325228, abs=1e-9)
    assert lin["se"] == pytest.approx(0.0673080692, abs=1e-9)
    assert (lin["ci_low"], lin["ci_high"]) == pytest.approx((0.5398111313, 0.8036539143), abs=1e-8)
    quad = weighted_kappa_stats(CM, "quadratic")
    assert quad["kappa"] == pytest.approx(0.7058823529, abs=1e-9)
    assert quad["se"] == pytest.approx(0.0742965714, abs=1e-9)
    assert (quad["ci_low"], quad["ci_high"]) == pytest.approx((0.5602637488, 0.8515009572), abs=1e-8)
    unw = weighted_kappa_stats(CM, None)
    assert unw["se"] == pytest.approx(0.0687246103, abs=1e-9)


def test_invariants_perfect_agreement_and_symmetry():
    perfect = np.diag([10, 12, 8])
    res = weighted_kappa_stats(perfect, "quadratic")
    assert res["kappa"] == pytest.approx(1.0)
    assert res["se"] == pytest.approx(0.0, abs=1e-6)  # cancellation noise only
    assert (res["ci_low"], res["ci_high"]) == pytest.approx((1.0, 1.0), abs=1e-6)
    # transposing swaps the raters: kappa and both variances are unchanged
    a, b = weighted_kappa_stats(CM, "linear"), weighted_kappa_stats(CM.T, "linear")
    for key in ("kappa", "se", "se_null", "z"):
        assert a[key] == pytest.approx(b[key], rel=1e-12)
    # CI is clipped into [-1, 1] and contains the estimate
    small = weighted_kappa_stats(np.array([[9, 1, 0], [0, 1, 0], [0, 0, 9]]), "linear")
    assert -1.0 <= small["ci_low"] <= small["kappa"] <= small["ci_high"] <= 1.0
    # more data with the same proportions shrinks the SE like 1 / sqrt(n)
    assert weighted_kappa_stats(CM * 4, "linear")["se"] == pytest.approx(
        weighted_kappa_stats(CM, "linear")["se"] / 2, rel=1e-12)
    # CI half-width equals 1.96 * se when unclipped
    lin = weighted_kappa_stats(CM, "linear")
    assert lin["ci_high"] - lin["kappa"] == pytest.approx(stats.norm.ppf(0.975) * lin["se"])


def test_degenerate_chance_agreement_returns_none_fields():
    # everyone in one category on both raters: pe_w = 1, kappa undefined
    res = weighted_kappa_stats(np.array([[10, 0], [0, 0]]), "linear")
    assert res["kappa"] is None and res["se"] is None and res["z"] is None


def test_input_validation():
    with pytest.raises(ValueError):
        weighted_kappa_stats(np.ones((2, 3)), "linear")
    with pytest.raises(ValueError):
        weighted_kappa_stats(np.zeros((3, 3)), "linear")


@pytest.mark.skipif(not _r_has("psych"), reason="R package psych not available")
@pytest.mark.parametrize("kind", ["linear", "quadratic"])
def test_matches_psych_cohen_kappa(kind):
    code = (
        "suppressMessages(library(psych));cm<-matrix(c(20,4,2,5,30,3,1,6,19),3);"
        "d<-abs(row(cm)-col(cm))/2;"
        f"w<-{'1-d' if kind == 'linear' else '1-d^2'};"
        "r<-cohen.kappa(cm,w=w);cat(r$weighted.kappa,sqrt(r$var.weighted),r$confid[2,1],r$confid[2,3])"
    )
    out = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        pytest.skip(out.stderr[-200:])
    kappa, se, lo, hi = map(float, out.stdout.split())
    mine = weighted_kappa_stats(CM, kind)
    assert mine["kappa"] == pytest.approx(kappa, abs=1e-6)
    assert mine["se"] == pytest.approx(se, abs=1e-6)
    assert mine["ci_low"] == pytest.approx(lo, abs=1e-6)
    assert mine["ci_high"] == pytest.approx(hi, abs=1e-6)


@pytest.mark.skipif(not _r_has("DescTools"), reason="R package DescTools not available")
@pytest.mark.parametrize("kind,r_weights", [("linear", "Equal-Spacing"), ("quadratic", "Fleiss-Cohen")])
def test_matches_desctools_cohenkappa_ci(kind, r_weights):
    code = (
        "suppressMessages(library(DescTools));cm<-matrix(c(20,4,2,5,30,3,1,6,19),3);"
        f"cat(CohenKappa(cm,weights='{r_weights}',conf.level=.95))"
    )
    out = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        pytest.skip(out.stderr[-200:])
    kappa, lo, hi = map(float, out.stdout.split())
    mine = weighted_kappa_stats(CM, kind)
    assert mine["kappa"] == pytest.approx(kappa, abs=1e-6)
    assert mine["ci_low"] == pytest.approx(lo, abs=1e-6)
    assert mine["ci_high"] == pytest.approx(hi, abs=1e-6)


# --------------------------------------------------------------------------
# endpoint
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["linear", "quadratic"])
def test_endpoint_reports_weighted_se_ci_and_test(client, kind):
    frame = _frame(CM)
    sid = make_session(frame, f"wk_endpoint_{kind}")
    r = client.post("/api/stats/cohens_kappa", json={
        "session_id": sid, "rater1_col": "x", "rater2_col": "y", "weights": kind})
    assert r.status_code == 200, r.text
    body = r.json()
    ref = weighted_kappa_stats(CM, kind)
    assert body["kappa"] == pytest.approx(cohen_kappa_score(frame["x"], frame["y"], weights=kind))
    assert body["se"] == pytest.approx(ref["se"]) and body["se"] > 0
    assert body["ci_low"] == pytest.approx(ref["ci_low"]) and body["ci_high"] == pytest.approx(ref["ci_high"])
    assert body["se_null"] == pytest.approx(ref["se_null"])
    assert body["z"] == pytest.approx(ref["z"])
    assert body["p"] == pytest.approx(ref["p"]) and body["p"] < 0.001
    assert body["po"] == pytest.approx(ref["po"]) and body["pe"] == pytest.approx(ref["pe"])
    assert body["po"] > body["pe"]
    assert "Fleiss" in body["uncertainty_note"]
    assert body["weights"] == kind
    assert body["ci_low"] <= body["kappa"] <= body["ci_high"] <= 1.0


def test_endpoint_unweighted_unchanged(client):
    sid = make_session(_frame(CM), "wk_unweighted")
    body = client.post("/api/stats/cohens_kappa", json={"session_id": sid, "rater1_col": "x", "rater2_col": "y"}).json()
    assert body["weights"] == "none" and body["uncertainty_note"] is None
    n = CM.sum()
    po = np.trace(CM) / n
    pe = float((CM.sum(0) * CM.sum(1)).sum() / n ** 2)
    assert body["po"] == pytest.approx(po) and body["pe"] == pytest.approx(pe)
    assert body["se"] == pytest.approx(np.sqrt(po * (1 - po) / (n * (1 - pe) ** 2)))


def test_endpoint_linear_weights_on_two_categories_equal_identity(client):
    cm = np.array([[25, 5], [8, 22]])
    sid = make_session(_frame(cm), "wk_2x2")
    base = {"session_id": sid, "rater1_col": "x", "rater2_col": "y"}
    unweighted = client.post("/api/stats/cohens_kappa", json=base).json()
    weighted = client.post("/api/stats/cohens_kappa", json={**base, "weights": "linear"}).json()
    assert weighted["kappa"] == pytest.approx(unweighted["kappa"])
    assert weighted["se_null"] == pytest.approx(unweighted["se_null"])
    assert weighted["z"] == pytest.approx(unweighted["z"])
    assert weighted["po"] == pytest.approx(unweighted["po"])
