"""Power analysis against the worked examples in Zhang & Hartmann (2023).

    Zhang X, Hartmann P. How to calculate sample size in animal and human
    studies. Front Med. 2023;10:1215927. doi:10.3389/fmed.2023.1215927

The paper publishes G*Power screenshots for six calculations plus three
worked totals in prose. Those are a published reference implementation of the
same calculations this panel performs, so agreement with them is the test —
and the noncentrality parameters printed in the screenshots pin the method,
not just the rounded answer: matching n could be luck, matching delta to
seven figures means the same noncentral-t formulation.
"""
from __future__ import annotations

import numpy as np
import pytest


def _power(client, **body):
    r = client.post("/api/stats/power", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ── two-sample t-test, Figures 1A-1D ──────────────────────────────────────────

# (label, d, alpha, power, n per group, G*Power's "actual power")
T_CASES = [
    ("Fig 1A hepatic triglyceride", 0.9575908, 0.05, 0.80, 19, 0.8191075),
    ("Fig 1B TNF expression",       1.349154,  0.05, 0.80, 10, 0.8139794),
    ("Fig 1C alpha 0.01",           1.349154,  0.01, 0.80, 15, 0.8152953),
    ("Fig 1D power 0.95",           1.349154,  0.05, 0.95, 16, 0.9582959),
]


@pytest.mark.parametrize("label,d,alpha,power,n_exp,actual_exp", T_CASES)
def test_two_sample_t_matches_gpower(client, label, d, alpha, power, n_exp, actual_exp):
    got = _power(client, test="t_two", solve_for="n", effect_size=d,
                 alpha=alpha, power=power, tails=2, ratio=1.0)
    assert int(got["result"]) == n_exp, label

    # The achieved power at that n — G*Power reports it because the required n
    # is rounded up, so the study is slightly better powered than requested.
    back = _power(client, test="t_two", solve_for="power", effect_size=d,
                  alpha=alpha, n=n_exp, tails=2, ratio=1.0)
    assert back["result"] == pytest.approx(actual_exp, abs=1e-6), label


def test_the_effect_size_formula_matches_the_paper():
    """Fig 1A: 143.26 (SD 54.50) vs 192.84 (SD 48.90) -> d = 0.9575908.

    Pooled as sqrt((SD1^2 + SD2^2) / 2), which is the equal-n form the paper
    states. Getting this wrong would shift every n downstream of it.
    """
    d = abs(192.84 - 143.26) / np.sqrt((48.90 ** 2 + 54.50 ** 2) / 2)
    assert d == pytest.approx(0.9575908, abs=1e-7)


@pytest.mark.parametrize("label,d,n,delta_exp", [
    ("1A", 0.9575908, 19, 2.9514931),
    ("1B", 1.349154,  10, 3.0168001),
    ("1C", 1.349154,  15, 3.6948104),
    ("1D", 1.349154,  16, 3.8159838),
])
def test_noncentrality_matches_the_screenshots(label, d, n, delta_exp):
    """delta = d * sqrt(n1*n2/(n1+n2)). Seven-figure agreement means the same
    noncentral-t formulation, not merely the same rounded sample size."""
    assert d * np.sqrt(n * n / (n + n)) == pytest.approx(delta_exp, abs=2e-5), label


# ── two proportions, Figure 2 ─────────────────────────────────────────────────

@pytest.mark.parametrize("label,p1,p2,alpha,power,n_exp", [
    ("Fig 2A rifaximin",       0.55, 0.40, 0.05, 0.95, 286),
    ("Fig 2B dexmedetomidine", 0.54, 0.27, 0.05, 0.80, 51),
])
def test_two_proportions_match_gpower(client, label, p1, p2, alpha, power, n_exp):
    got = _power(client, test="proportion", solve_for="n", p1=p1, p2=p2,
                 alpha=alpha, power=power, tails=2, ratio=1.0)
    assert int(got["result"]) == n_exp, label


# ── worked totals stated in the prose ─────────────────────────────────────────

def test_human_pilot_example(client):
    """Faecal calprotectin, d = 0.63, 95% power -> 67 per group."""
    got = _power(client, test="t_two", solve_for="n", effect_size=0.63,
                 alpha=0.05, power=0.95, tails=2, ratio=1.0)
    assert int(got["result"]) == 67


def test_one_tailed_needs_fewer_than_two_tailed(client):
    """Pilot tumour experiment, d = 1.07: 15 per group two-tailed, 12 one-tailed.

    The paper makes the point that a one-sided design is cheaper; if the tails
    flag were ignored both would come back the same.
    """
    two = _power(client, test="t_two", solve_for="n", effect_size=1.07,
                 alpha=0.05, power=0.80, tails=2, ratio=1.0)
    one = _power(client, test="t_two", solve_for="n", effect_size=1.07,
                 alpha=0.05, power=0.80, tails=1, ratio=1.0)
    assert int(two["result"]) == 15
    assert int(one["result"]) == 12


def test_unequal_allocation_costs_more_subjects(client):
    """The paper: 2:1 allocation needs ~12% more subjects than 1:1, 3:1 ~33%."""
    base = _power(client, test="t_two", solve_for="n", effect_size=0.5,
                  alpha=0.05, power=0.80, tails=2, ratio=1.0)
    total_1_1 = int(base["result"]) * 2
    for ratio, lo, hi in ((2.0, 0.08, 0.16), (3.0, 0.28, 0.38)):
        got = _power(client, test="t_two", solve_for="n", effect_size=0.5,
                     alpha=0.05, power=0.80, tails=2, ratio=ratio)
        n1 = int(got["result"])
        total = n1 + int(np.ceil(n1 * ratio))
        excess = total / total_1_1 - 1
        assert lo <= excess <= hi, f"ratio {ratio}:1 needed {excess:.0%} more"


# ── expected attrition ────────────────────────────────────────────────────────

def test_attrition_reports_the_recruitment_target(client):
    """Paper: 19 per group with 10% expected attrition -> enrol 22 (19/0.9).

    The computed n is the number that must COMPLETE the study; enrolling
    exactly that many leaves the trial underpowered the first time anyone
    withdraws.
    """
    got = _power(client, test="t_two", solve_for="n", effect_size=0.9575908,
                 alpha=0.05, power=0.80, tails=2, ratio=1.0, attrition=0.10)
    assert int(got["result"]) == 19          # unchanged: the statistical requirement
    assert got["n_corrected"] == 22          # what to recruit
    assert "22" in got["label"]


@pytest.mark.parametrize("attrition,expected", [(0.05, 302), (0.10, 318)])
def test_attrition_on_the_proportions_example(client, attrition, expected):
    """Paper: 286 per group corrected to 286/0.95 = 301.05 and 286/0.9 = 317.78.

    Rounded UP — a fractional participant recruited down is a participant
    short.
    """
    got = _power(client, test="proportion", solve_for="n", p1=0.55, p2=0.40,
                 alpha=0.05, power=0.95, tails=2, ratio=1.0, attrition=attrition)
    assert int(got["result"]) == 286
    assert got["n_corrected"] == expected


def test_no_attrition_leaves_the_answer_alone(client):
    got = _power(client, test="t_two", solve_for="n", effect_size=0.5,
                 alpha=0.05, power=0.80, tails=2, ratio=1.0)
    assert got["n_corrected"] is None
    assert got["attrition"] is None


def test_attrition_is_ignored_when_not_solving_for_n(client):
    """There is no recruitment target to correct when n is the input."""
    got = _power(client, test="t_two", solve_for="power", effect_size=0.5,
                 alpha=0.05, n=64, tails=2, ratio=1.0, attrition=0.2)
    assert got["n_corrected"] is None


@pytest.mark.parametrize("bad", [1.0, 1.5, -0.1])
def test_an_impossible_attrition_rate_is_refused(client, bad):
    """1.0 would divide by zero and produce an infinite recruitment target."""
    r = client.post("/api/stats/power", json={
        "test": "t_two", "solve_for": "n", "effect_size": 0.5,
        "alpha": 0.05, "power": 0.8, "tails": 2, "ratio": 1.0, "attrition": bad,
    })
    assert r.status_code == 422, r.text
