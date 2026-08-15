"""Power analysis against R's `pwr`.

Every literal here was produced by R 4.5.2 with the pwr package, on the cases
in qa/power_audit/cases.json; the script that produced them is committed next
to it. The panel is a sample-size calculator — its output is the whole
deliverable, so agreement with the reference implementation is the test.
"""
from __future__ import annotations

import pytest


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# Cases that go through scipy's noncentral t or F distribution hold to a looser
# tolerance than the rest of this file. scipy 1.15 improved the accuracy of
# those two distributions; uSTAT pins 1.14.1 because that is what Pyodide ships
# (see requirements.txt), so the server and the browser agree with each other to
# the last bits rather than each landing somewhere different near R.
#
# The residual against R is a few parts per million -- 0.8637155 where R says
# 0.8637149, both of which a paper reports as 0.864. It is loose enough to
# absorb that and still tight enough to catch a real regression, which would
# move the value by orders of magnitude more. Everything computed in closed
# form, and the chi-square family, stays at 1e-9 below.
NONCENTRAL_REL = 1e-5


def _power(client, **body):
    r = client.post("/api/stats/power", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ── one-way ANOVA: n is per group, and used to be read as the total ─────────

def test_anova_power_reads_n_as_per_group(client):
    """R: pwr.anova.test(k = 4, n = 52, f = 0.25) -> power 0.8637148576836544.

    statsmodels' FTestAnovaPower takes `nobs` as the TOTAL sample size while
    the panel asks for participants per group, and the two were never
    translated. With four groups this reported 0.275 — a properly powered
    study called hopeless.
    """
    j = _power(client, test="anova", solve_for="power",
               effect_size=0.25, n=52, alpha=0.05, k_groups=4)
    assert _rel(j["result"], 0.8637148576836544) < NONCENTRAL_REL


def test_anova_sample_size_is_per_group(client):
    """R: pwr.anova.test(k = 4, f = 0.25, power = 0.8) -> 44.6 per group.

    The solver returns the total; it was reported as the per-group figure and
    then multiplied by k again, so a study needing 179 participants was told
    to recruit 716.
    """
    j = _power(client, test="anova", solve_for="n",
               effect_size=0.25, power=0.8, alpha=0.05, k_groups=4)
    assert j["result"] == 45
    assert "n/group = 45" in j["label"]
    assert "total N = 180" in j["label"]


def test_anova_three_groups(client):
    """R: pwr.anova.test(k = 3, n = 20, f = 0.4) -> 0.7757304353075136."""
    j = _power(client, test="anova", solve_for="power",
               effect_size=0.4, n=20, alpha=0.05, k_groups=3)
    assert _rel(j["result"], 0.7757304353075136) < NONCENTRAL_REL


# ── correlation: t critical value and the Fisher-z bias term ────────────────

@pytest.mark.parametrize("n,r_es,tails,expected", [
    (85, 0.3, 2, 0.8043956918386735),
    (85, 0.3, 1, 0.8799369356905099),
    (12, 0.5, 2, 0.4001744512820906),
])
def test_correlation_power_matches_pwr_r_test(client, n, r_es, tails, expected):
    """R: pwr.r.test(n, r, sig.level = 0.05).

    The plain Fisher-z version took its critical value from a normal rather
    than the t on n-2 df that is actually tested, and left out the
    r/(2(n-1)) bias of the transform. At n = 85, r = 0.3 that landed on
    exactly 0.800 — the conventional threshold a study is judged against —
    where the correct value is 0.804.
    """
    j = _power(client, test="correlation", solve_for="power",
               effect_size=r_es, n=n, alpha=0.05, tails=tails)
    assert _rel(j["result"], expected) < 1e-8


def test_correlation_sample_size(client):
    """R: pwr.r.test(r = 0.3, power = 0.8) -> 84.07, so 85."""
    j = _power(client, test="correlation", solve_for="n",
               effect_size=0.3, power=0.8, alpha=0.05, tails=2)
    assert j["result"] == 85


# ── the families that already agreed, kept honest ───────────────────────────

def test_two_sample_t_matches_pwr(client):
    """R: pwr.t.test(n = 64, d = 0.5, type = "two.sample")."""
    j = _power(client, test="t_two", solve_for="power",
               effect_size=0.5, n=64, alpha=0.05, tails=2)
    assert _rel(j["result"], 0.8014595579222848) < NONCENTRAL_REL


def test_two_proportions_matches_pwr_2p(client):
    """R: pwr.2p.test(h = ES.h(0.5, 0.3), n = 100)."""
    j = _power(client, test="proportion", solve_for="power",
               p1=0.5, p2=0.3, n=100, alpha=0.05, tails=2)
    assert _rel(j["result"], 0.8289189081102789) < 1e-9


def test_chi_square_matches_pwr_chisq(client):
    """R: pwr.chisq.test(w = 0.3, N = 150, df = 3) — df = categories - 1."""
    j = _power(client, test="chi2", solve_for="power",
               effect_size=0.3, n=150, alpha=0.05, k_groups=4)
    assert _rel(j["result"], 0.884032726811765) < 1e-9


def test_cox_matches_schoenfeld(client):
    """Schoenfeld: events = (z_a + z_b)^2 / (p(1-p) log(HR)^2), n = events /
    event rate. HR 1.8, 40% events, balanced exposure, 80% power."""
    j = _power(client, test="survival_cox", solve_for="n",
               hr=1.8, event_rate=0.4, power=0.8, alpha=0.05, tails=2,
               p_exposed=0.5)
    assert j["result"] == 228


def test_logistic_treats_its_effect_field_as_an_odds_ratio(client):
    """The panel labels the field "Odds ratio (OR)" and posts the OR in
    `log_or`, so 1.5 is an odds ratio and the coefficient is log(1.5).
    Hsieh: n = (z_a + z_b)^2 / (p(1-p) B^2) = 228."""
    j = _power(client, test="logistic", solve_for="n",
               log_or=1.5, p_event=0.3, power=0.8, alpha=0.05, tails=2)
    assert j["result"] == 228
