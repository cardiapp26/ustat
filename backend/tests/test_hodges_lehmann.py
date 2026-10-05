"""Hodges-Lehmann estimate + distribution-free CI (services.hodges_lehmann).

Hand-computed values, structural invariants, a cross-check against R's
wilcox.test(conf.int = TRUE) when Rscript is available, and endpoint wiring.
"""
import itertools
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from conftest import make_session
from services.hodges_lehmann import (
    hl_one_sample,
    hl_paired,
    hl_two_sample,
    qsignrank_lower,
    qwilcox_lower,
)

HAVE_R = shutil.which("Rscript") is not None


def _rscript(code: str) -> str:
    out = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        pytest.skip(f"Rscript failed: {out.stderr[-200:]}")
    return out.stdout.strip()


def _rvec(a) -> str:
    return "c(" + ",".join(repr(float(v)) for v in a) + ")"


# --------------------------------------------------------------------------
# exact null distributions (hand-checkable)
# --------------------------------------------------------------------------

def test_qwilcox_small_cases():
    # m = n = 2: U in {0,1,2,3,4} with counts 1,1,2,1,1 out of 6
    assert qwilcox_lower(0.1, 2, 2) == 0   # P(U<=0) = 1/6 >= 0.1
    assert qwilcox_lower(0.2, 2, 2) == 1   # P(U<=1) = 2/6 >= 0.2
    assert qwilcox_lower(0.4, 2, 2) == 2   # P(U<=2) = 4/6


def test_qwilcox_matches_brute_force():
    m, n = 4, 5
    counts = np.zeros(m * n + 1, dtype=int)
    for comb in itertools.combinations(range(m + n), m):
        u = sum(rank - i for i, rank in enumerate(comb))
        counts[u] += 1
    cdf = np.cumsum(counts) / counts.sum()
    for p in (0.005, 0.025, 0.05, 0.2, 0.5):
        assert qwilcox_lower(p, m, n) == int(np.argmax(cdf >= p - 1e-12))


def test_qsignrank_matches_brute_force():
    n = 7
    counts = np.zeros(n * (n + 1) // 2 + 1, dtype=int)
    for signs in itertools.product([0, 1], repeat=n):
        counts[sum(r + 1 for r, s in enumerate(signs) if s)] += 1
    cdf = np.cumsum(counts) / counts.sum()
    for p in (0.01, 0.025, 0.05, 0.3):
        assert qsignrank_lower(p, n) == int(np.argmax(cdf >= p - 1e-12))


# --------------------------------------------------------------------------
# two samples
# --------------------------------------------------------------------------

def test_two_sample_hand_computed_and_r_example():
    x = [1.1, 2.2, 3.3, 4.4]
    y = [0.5, 1.7, 2.9, 3.1, 5.0]
    r = hl_two_sample(x, y)
    # median of the 20 differences x_i - y_j (R: 0.3), qwilcox(.025, 4, 5) = 2
    diffs = np.sort(np.subtract.outer(x, y).ravel())
    assert r["estimate"] == pytest.approx(float(np.median(diffs)))
    assert r["estimate"] == pytest.approx(0.3)
    assert r["ci_low"] == pytest.approx(diffs[1])
    assert r["ci_high"] == pytest.approx(diffs[20 - 2])
    assert (r["ci_low"], r["ci_high"]) == pytest.approx((-2.8, 2.8))
    assert r["method"].startswith("exact")
    assert r["confidence_level"] == pytest.approx(0.95)
    # achieved alpha = 2 * P(U <= qu - 1) = 2 * P(U <= 1) = 2 * 2/126
    assert r["achieved_confidence_level"] == pytest.approx(1 - 4 / 126)
    assert set(r) >= {"estimate", "ci_low", "ci_high", "confidence_level", "method", "note"}


def test_two_sample_invariants():
    rng = np.random.default_rng(3)
    x, y = rng.normal(1.0, 1, 18), rng.normal(0.0, 1, 22)
    r = hl_two_sample(x, y)
    assert r["ci_low"] <= r["estimate"] <= r["ci_high"]
    # location equivariance and sign symmetry
    shifted = hl_two_sample(x + 10, y)
    assert shifted["estimate"] == pytest.approx(r["estimate"] + 10)
    assert shifted["ci_low"] == pytest.approx(r["ci_low"] + 10)
    flipped = hl_two_sample(y, x)
    assert flipped["estimate"] == pytest.approx(-r["estimate"])
    assert flipped["ci_low"] == pytest.approx(-r["ci_high"])
    # scale equivariance
    scaled = hl_two_sample(3 * x, 3 * y)
    assert scaled["ci_high"] == pytest.approx(3 * r["ci_high"])
    # higher confidence gives a wider interval
    wide = hl_two_sample(x, y, conf_level=0.99)
    assert wide["ci_low"] <= r["ci_low"] and wide["ci_high"] >= r["ci_high"]


def test_two_sample_ties_use_normal_approximation_and_contain_estimate():
    x = [1, 2, 2, 3, 3, 3, 4, 5, 5, 6]
    y = [0, 1, 1, 2, 2, 3, 3, 4]
    r = hl_two_sample(x, y)
    assert r["method"].startswith("normal approximation")
    assert r["ci_low"] <= r["estimate"] <= r["ci_high"]


def test_two_sample_pair_cap_returns_none_with_note():
    r = hl_two_sample(np.arange(100.0), np.arange(100.0) + 0.5, max_pairs=5000)
    assert r["estimate"] is None and r["ci_low"] is None and r["ci_high"] is None
    assert "cap" in r["note"]


def test_two_sample_degenerate_constant_data_does_not_crash():
    r = hl_two_sample([1.0] * 5, [1.0] * 6)
    assert r["ci_low"] is None and r["ci_high"] is None
    assert "undefined" in r["note"]


def test_two_sample_rejects_bad_input():
    with pytest.raises(ValueError):
        hl_two_sample([1, 2], [3, 4], conf_level=1.5)
    with pytest.raises(ValueError):
        hl_two_sample([], [1.0])


# --------------------------------------------------------------------------
# one sample / paired
# --------------------------------------------------------------------------

def test_paired_hand_computed_walsh():
    d = np.array([1.0, 2.0, 4.0, 7.0])
    first = d + 10
    second = np.full(4, 10.0)
    r = hl_paired(first, second)
    walsh = np.sort([(d[i] + d[j]) / 2 for i in range(4) for j in range(i, 4)])
    assert r["estimate"] == pytest.approx(float(np.median(walsh)))
    # n = 4 cannot reach 95%: qsignrank(.025, 4) = 0 -> qu = 1, so the CI is the full range
    assert r["ci_low"] == pytest.approx(walsh[0])
    assert r["ci_high"] == pytest.approx(walsh[-1])
    assert r["achieved_confidence_level"] == pytest.approx(1 - 2 * 1 / 16)
    assert "not attainable" in r["note"]


def test_paired_invariants_and_zero_handling():
    rng = np.random.default_rng(11)
    d = rng.normal(0.7, 1, 25)
    r = hl_one_sample(d)
    assert r["ci_low"] <= r["estimate"] <= r["ci_high"]
    neg = hl_one_sample(-d)
    assert neg["estimate"] == pytest.approx(-r["estimate"])
    assert neg["ci_low"] == pytest.approx(-r["ci_high"])
    with_zeros = hl_one_sample(np.concatenate([d, [0.0, 0.0]]))
    assert with_zeros["method"].startswith("normal approximation")  # zeros => no exact CI, as in R
    assert with_zeros["ci_low"] <= with_zeros["estimate"] <= with_zeros["ci_high"]
    assert hl_one_sample([0.0, 0.0])["estimate"] is None


def test_paired_length_mismatch():
    with pytest.raises(ValueError):
        hl_paired([1, 2, 3], [1, 2])


@pytest.mark.skipif(not HAVE_R, reason="Rscript not installed")
@pytest.mark.parametrize("n1,n2,decimals", [(4, 5, None), (12, 15, None), (30, 40, None),
                                            (60, 70, None), (12, 15, 0), (30, 25, 0), (60, 55, 0)])
def test_two_sample_matches_r(n1, n2, decimals):
    rng = np.random.default_rng(n1 * 100 + n2)
    x, y = rng.normal(1, 1, n1), rng.normal(0, 1.3, n2)
    if decimals is not None:
        x, y = np.round(x, decimals), np.round(y, decimals)
    out = _rscript(
        f"r<-wilcox.test({_rvec(x)},{_rvec(y)},conf.int=TRUE);"
        "cat(r$conf.int[1],r$conf.int[2],r$estimate)"
    ).split()
    lo, hi, est = map(float, out)
    mine = hl_two_sample(x, y)
    assert mine["ci_low"] == pytest.approx(lo, abs=2e-4)
    assert mine["ci_high"] == pytest.approx(hi, abs=2e-4)
    # R's uniroot estimate for the approximate branch sits anywhere in the flat
    # central segment; the exact branch is the median of the differences.
    assert mine["estimate"] == pytest.approx(est, abs=2e-4 if decimals is None and n1 < 50 else 0.51)


@pytest.mark.skipif(not HAVE_R, reason="Rscript not installed")
@pytest.mark.parametrize("n,decimals", [(8, None), (30, None), (60, None), (15, 0), (40, 0), (70, 0)])
def test_paired_matches_r(n, decimals):
    rng = np.random.default_rng(n + 7)
    x, y = rng.normal(0.4, 1, n), rng.normal(0, 1, n)
    if decimals is not None:
        x, y = np.round(x, decimals), np.round(y, decimals)
    out = _rscript(
        f"r<-wilcox.test({_rvec(x)},{_rvec(y)},paired=TRUE,conf.int=TRUE);"
        "cat(r$conf.int[1],r$conf.int[2],r$estimate)"
    ).split()
    lo, hi, est = map(float, out)
    mine = hl_paired(x, y)
    assert mine["ci_low"] == pytest.approx(lo, abs=2e-4)
    assert mine["ci_high"] == pytest.approx(hi, abs=2e-4)
    assert mine["estimate"] == pytest.approx(est, abs=2e-4 if decimals is None and n < 50 else 0.51)


# --------------------------------------------------------------------------
# endpoint wiring
# --------------------------------------------------------------------------

def test_mannwhitney_endpoint_reports_hodges_lehmann(client):
    x = [1.1, 2.2, 3.3, 4.4]
    y = [0.5, 1.7, 2.9, 3.1, 5.0]
    df = pd.DataFrame({"v": x + y, "g": ["A"] * 4 + ["B"] * 5})
    sid = make_session(df, "hl_mw")
    r = client.post("/api/stats/mannwhitney", json={"session_id": sid, "column": "v", "group_column": "g"})
    assert r.status_code == 200, r.text
    body = r.json()
    hl = body["hodges_lehmann"]
    assert hl["estimate"] == pytest.approx(0.3)
    assert (hl["ci_low"], hl["ci_high"]) == pytest.approx((-2.8, 2.8))
    assert hl["confidence_level"] == pytest.approx(0.95)
    assert "Hodges-Lehmann" in body["result_text"]
    assert any("Hodges-Lehmann" in str(row[0]) for row in body["export_rows"])
    assert "conf.int = TRUE" in body["r_code"]
    # existing keys untouched
    assert {"U", "p", "effect_sizes", "median1", "methods_text"} <= set(body)


def test_wilcoxon_signed_rank_endpoint_reports_hodges_lehmann_and_alpha(client):
    rng = np.random.default_rng(5)
    pre = rng.normal(10, 2, 20)
    post = pre + rng.normal(1, 1, 20)
    sid = make_session(pd.DataFrame({"pre": pre, "post": post}), "hl_wsr")
    base = {"session_id": sid, "col1": "post", "col2": "pre"}
    body = client.post("/api/repeated/wilcoxon_signed_rank", json=base).json()
    hl = body["hodges_lehmann"]
    expected = hl_paired(post, pre, 0.95)
    assert hl["estimate"] == pytest.approx(expected["estimate"])
    assert hl["ci_low"] == pytest.approx(expected["ci_low"])
    assert "Hodges-Lehmann" in body["result_text"]
    assert "conf.int = TRUE" in body["r_code"]
    assert any("Hodges-Lehmann" in str(row[0]) for row in body["export_rows"])
    narrow = client.post("/api/repeated/wilcoxon_signed_rank", json={**base, "alpha": 0.2}).json()["hodges_lehmann"]
    assert narrow["confidence_level"] == pytest.approx(0.8)
    assert narrow["ci_high"] - narrow["ci_low"] < hl["ci_high"] - hl["ci_low"]


def test_wilcoxon_onesample_endpoint_reports_pseudomedian(client):
    vals = [2.5, 3.1, 4.7, 1.9, 5.2, 3.8, 4.4, 2.2, 6.1]
    sid = make_session(pd.DataFrame({"x": vals}), "hl_one")
    body = client.post("/api/stats/wilcoxon_onesample", json={"session_id": sid, "column": "x", "mu": 3}).json()
    expected = hl_one_sample(vals, 3.0)
    assert body["hodges_lehmann"]["estimate"] == pytest.approx(expected["estimate"])
    assert body["hodges_lehmann"]["ci_low"] == pytest.approx(expected["ci_low"])
