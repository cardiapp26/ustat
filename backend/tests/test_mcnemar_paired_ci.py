"""McNemar endpoint: paired proportion difference (Newcombe 1998 method 10) and
the exact conditional CI for the discordant odds ratio."""
import math
import shutil
import subprocess

import pandas as pd
import pytest
from scipy import stats

from conftest import make_session
from routers.categorical import conditional_or_ci, newcombe_paired_ci

HAVE_R = shutil.which("Rscript") is not None


def _wilson(k, n, z=1.959963984540054):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return centre - half, centre + half


def _newcombe_reference(a, b, c, d):
    """Independent re-implementation straight from Newcombe (1998), method 10."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = _wilson(a + b, n)
    l2, u2 = _wilson(a + c, n)
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(denom) if denom else 0.0
    lower = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2))
    upper = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2))
    return p1 - p2, lower, upper


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------

def test_newcombe_hand_computed_example():
    # a = both +, b = col1 + / col2 -, c = col1 - / col2 +, d = both -
    r = newcombe_paired_ci(30, 5, 20, 45)
    assert r["estimate"] == pytest.approx(-0.15)          # (b - c) / n
    assert r["proportion_col1"] == pytest.approx(0.35)    # (a + b) / n
    assert r["proportion_col2"] == pytest.approx(0.50)    # (a + c) / n
    assert r["ci_low"] == pytest.approx(-0.2394434222, abs=1e-9)
    assert r["ci_high"] == pytest.approx(-0.0555474908, abs=1e-9)
    assert (r["estimate"], r["ci_low"], r["ci_high"]) == pytest.approx(_newcombe_reference(30, 5, 20, 45))


@pytest.mark.parametrize("cells", [(30, 5, 20, 45), (10, 12, 3, 5), (50, 1, 0, 49), (7, 7, 7, 7), (0, 6, 2, 12)])
def test_newcombe_invariants(cells):
    a, b, c, d = cells
    r = newcombe_paired_ci(a, b, c, d)
    assert -1.0 <= r["ci_low"] <= r["estimate"] <= r["ci_high"] <= 1.0
    # swapping the two columns negates the difference and mirrors the interval
    swapped = newcombe_paired_ci(a, c, b, d)
    assert swapped["estimate"] == pytest.approx(-r["estimate"])
    assert swapped["ci_low"] == pytest.approx(-r["ci_high"])
    assert swapped["ci_high"] == pytest.approx(-r["ci_low"])
    # a higher confidence level widens the interval
    wide = newcombe_paired_ci(a, b, c, d, alpha=0.01)
    assert wide["ci_low"] <= r["ci_low"] and wide["ci_high"] >= r["ci_high"]
    assert (r["estimate"], r["ci_low"], r["ci_high"]) == pytest.approx(_newcombe_reference(a, b, c, d))


def test_newcombe_empty_margin_uses_phi_zero():
    # column 2 never positive: a + c = 0, phi is defined as 0
    r = newcombe_paired_ci(0, 4, 0, 16)
    assert r["phi"] == 0.0
    assert r["estimate"] == pytest.approx(0.2)
    assert (r["ci_low"], r["ci_high"]) == pytest.approx(_newcombe_reference(0, 4, 0, 16)[1:])


def test_conditional_or_ci_matches_exact_binomial_transform():
    b, c = 5, 20
    lo, hi = conditional_or_ci(b, c)
    bt = stats.binomtest(b, b + c).proportion_ci(confidence_level=0.95, method="exact")
    assert lo == pytest.approx(bt.low / (1 - bt.low))
    assert hi == pytest.approx(bt.high / (1 - bt.high))
    assert lo < b / c < hi
    assert lo == pytest.approx(0.07332006499, abs=1e-9)
    assert hi == pytest.approx(0.68644709538, abs=1e-9)


def test_conditional_or_ci_edge_cases():
    assert conditional_or_ci(0, 0) is None
    assert conditional_or_ci(5, 0) is None     # OR infinite
    lo, hi = conditional_or_ci(0, 5)           # OR = 0
    assert lo == 0.0 and hi > 0


@pytest.mark.skipif(not HAVE_R, reason="Rscript not installed")
def test_newcombe_matches_r_prop_test_construction():
    code = (
        "a<-30;b<-5;c<-20;d<-45;n<-a+b+c+d;"
        "w1<-prop.test(a+b,n,correct=FALSE)$conf.int;w2<-prop.test(a+c,n,correct=FALSE)$conf.int;"
        "p1<-(a+b)/n;p2<-(a+c)/n;phi<-(a*d-b*c)/sqrt((a+b)*(c+d)*(a+c)*(b+d));"
        "lo<-(p1-p2)-sqrt((p1-w1[1])^2+(w2[2]-p2)^2-2*phi*(p1-w1[1])*(w2[2]-p2));"
        "hi<-(p1-p2)+sqrt((w1[2]-p1)^2+(p2-w2[1])^2-2*phi*(w1[2]-p1)*(p2-w2[1]));"
        "bt<-binom.test(b,b+c)$conf.int;"
        "cat(p1-p2,lo,hi,bt[1]/(1-bt[1]),bt[2]/(1-bt[2]))"
    )
    out = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        pytest.skip(out.stderr[-200:])
    est, lo, hi, or_lo, or_hi = map(float, out.stdout.split())
    mine = newcombe_paired_ci(30, 5, 20, 45)
    assert (mine["estimate"], mine["ci_low"], mine["ci_high"]) == pytest.approx((est, lo, hi), abs=1e-6)
    assert conditional_or_ci(5, 20) == pytest.approx((or_lo, or_hi), abs=1e-6)


# --------------------------------------------------------------------------
# endpoint
# --------------------------------------------------------------------------

def _frame():
    # same fixture as test_categorical.test_mcnemar_known: a=30, c=20 (0 -> 1), b=5 (1 -> 0), d=45
    return pd.DataFrame({
        "before": [1] * 30 + [0] * 20 + [1] * 5 + [0] * 45,
        "after": [1] * 30 + [1] * 20 + [0] * 5 + [0] * 45,
    })


def test_mcnemar_endpoint_paired_difference(client):
    sid = make_session(_frame(), "mc_paired_ci")
    r = client.post("/api/categorical/mcnemar", json={"session_id": sid, "col1": "before", "col2": "after"})
    assert r.status_code == 200, r.text
    body = r.json()
    pdiff = body["paired_difference"]
    assert pdiff["estimate"] == pytest.approx(-0.15)
    assert pdiff["ci_low"] == pytest.approx(-0.2394434222, abs=1e-8)
    assert pdiff["ci_high"] == pytest.approx(-0.0555474908, abs=1e-8)
    assert pdiff["confidence_level"] == pytest.approx(0.95)
    assert "Newcombe" in pdiff["method"]
    assert pdiff["proportion_col1"] == pytest.approx(0.35) and pdiff["proportion_col2"] == pytest.approx(0.5)
    assert "before" in pdiff["note"] and "after" in pdiff["note"]
    # discordant OR b / c = 5 / 20 now carries its exact conditional CI
    es = body["effect_sizes"][0]
    assert es["name"] == "odds_ratio_discordant"
    assert es["value"] == pytest.approx(0.25)
    assert es["ci_low"] == pytest.approx(0.0733, abs=1e-4)
    assert es["ci_high"] == pytest.approx(0.6864, abs=1e-4)
    # text, export rows and R code mention the new quantities
    assert "Newcombe" in body["result_text"] and "exact conditional CI" in body["result_text"]
    labels = " ".join(str(row[0]) for row in body["export_rows"])
    assert "Paired difference" in labels and "OR (discordant) 95% CI lower" in labels
    assert "mcnemar.test(table)" in body["r_code"] and "Newcombe" in body["r_code"]
    assert "BinomDiffCI" in body["r_code"]
    # existing keys keep their meaning
    assert body["cells"]["positive_to_negative"] == 5 and body["cells"]["negative_to_positive"] == 20


def test_mcnemar_endpoint_respects_alpha(client):
    sid = make_session(_frame(), "mc_paired_alpha")
    base = {"session_id": sid, "col1": "before", "col2": "after"}
    wide = client.post("/api/categorical/mcnemar", json=base).json()
    narrow = client.post("/api/categorical/mcnemar", json={**base, "alpha": 0.2}).json()
    assert narrow["paired_difference"]["confidence_level"] == pytest.approx(0.8)
    assert narrow["paired_difference"]["ci_high"] - narrow["paired_difference"]["ci_low"] < (
        wide["paired_difference"]["ci_high"] - wide["paired_difference"]["ci_low"]
    )
    assert narrow["effect_sizes"][0]["ci_high"] < wide["effect_sizes"][0]["ci_high"]


def test_mcnemar_endpoint_infinite_or_has_no_ci(client):
    # c = 0: nobody moves 0 -> 1, so the discordant OR b / c (b = 1 -> 0 moves) is infinite
    df = pd.DataFrame({
        "x": [1] * 8 + [1] * 12 + [0] * 10,
        "y": [0] * 8 + [1] * 12 + [0] * 10,
    })
    sid = make_session(df, "mc_paired_inf")
    body = client.post("/api/categorical/mcnemar", json={"session_id": sid, "col1": "x", "col2": "y"}).json()
    es = body["effect_sizes"][0]
    assert es["value"] is None and es["ci_low"] is None and es["ci_high"] is None
    assert body["paired_difference"]["estimate"] == pytest.approx(8 / 30)
    assert body["paired_difference"]["ci_low"] > 0.0


def test_mcnemar_endpoint_zero_or_has_finite_upper_limit(client):
    # b = 0 (nobody moves 1 -> 0): OR = 0, the exact interval starts at 0
    df = pd.DataFrame({
        "x": [0] * 8 + [1] * 12 + [0] * 10,
        "y": [1] * 8 + [1] * 12 + [0] * 10,
    })
    sid = make_session(df, "mc_paired_zero")
    body = client.post("/api/categorical/mcnemar", json={"session_id": sid, "col1": "x", "col2": "y"}).json()
    es = body["effect_sizes"][0]
    assert es["value"] == 0.0
    assert es["ci_low"] == 0.0 and es["ci_high"] > 0.0
    assert body["paired_difference"]["estimate"] == pytest.approx(-8 / 30)
    assert body["paired_difference"]["ci_high"] < 0.0
