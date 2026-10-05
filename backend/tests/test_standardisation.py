"""Rate standardisation and rate ratio (services/standardisation.py, /api/epidemiology).

Reference values are computed by hand in the tests (numpy / closed forms) and
compared with the implementation. R parity against epitools runs only when
Rscript and epitools are installed.
"""
import json
import shutil
import subprocess

import numpy as np
import pytest
from scipy import stats

from services import standardisation as sd

# Four age strata: events, person-years, standard population.
EVENTS = [4, 15, 60, 130]
PT = [20000.0, 15000.0, 9000.0, 4000.0]
STD = [40000.0, 30000.0, 20000.0, 10000.0]
BASE = "/api/epidemiology"


def _strata():
    return [
        {"label": f"age{i}", "events": e, "person_time": t, "standard_population": s}
        for i, (e, t, s) in enumerate(zip(EVENTS, PT, STD))
    ]


# ── exact Poisson building block ─────────────────────────────────────────────


def test_poisson_exact_known_values():
    # Garwood limits for a count of 10 and of 0 (classic table values).
    ci = sd.poisson_exact_ci(10, 1.0, 0.05)
    assert ci["low"] == pytest.approx(4.795389, abs=1e-5)
    assert ci["high"] == pytest.approx(18.390356, abs=1e-5)
    ci0 = sd.poisson_exact_ci(0, 1.0, 0.05)
    assert ci0["low"] == 0.0
    assert ci0["high"] == pytest.approx(3.688879, abs=1e-5)


# ── direct standardisation ───────────────────────────────────────────────────


def test_direct_matches_hand_computation():
    res = sd.direct_standardisation(_strata(), multiplier=100000, alpha=0.05)

    ev, pt, std = (np.array(x, dtype=float) for x in (EVENTS, PT, STD))
    w = std / std.sum()
    rate = ev / pt
    dsr = (w * rate).sum()
    var = (w ** 2 * ev / pt ** 2).sum()
    wm = (w / pt).max()
    lo = stats.gamma.ppf(0.025, dsr ** 2 / var, scale=var / dsr)
    hi = stats.gamma.ppf(0.975, (dsr + wm) ** 2 / (var + wm ** 2), scale=(var + wm ** 2) / (dsr + wm))

    assert res["standardised_rate"] == pytest.approx(dsr * 1e5)
    assert res["standardised_ci_low"] == pytest.approx(lo * 1e5)
    assert res["standardised_ci_high"] == pytest.approx(hi * 1e5)
    assert res["crude_rate"] == pytest.approx(ev.sum() / pt.sum() * 1e5)
    assert res["standardised_ci_low"] < res["standardised_rate"] < res["standardised_ci_high"]
    assert res["crude_ci_low"] < res["crude_rate"] < res["crude_ci_high"]

    rows = res["strata"]
    assert [r["weight"] for r in rows] == pytest.approx(list(w))
    assert sum(r["weight"] for r in rows) == pytest.approx(1.0)
    assert sum(r["contribution"] for r in rows) == pytest.approx(res["standardised_rate"])
    assert sum(r["contribution_pct"] for r in rows) == pytest.approx(100.0)
    assert [r["rate"] for r in rows] == pytest.approx(list(rate * 1e5))
    assert res["warnings"] == []
    assert "directly standardised" in res["result_text"]
    assert "ageadjust.direct" in res["r_code"]


def test_direct_constant_rate_is_trivial():
    # Equal stratum rates (0.01): the DSR must equal that rate whatever the weights.
    strata = [
        {"events": 10, "person_time": 1000, "standard_population": 100},
        {"events": 20, "person_time": 2000, "standard_population": 200},
        {"events": 30, "person_time": 3000, "standard_population": 300},
    ]
    res = sd.direct_standardisation(strata, multiplier=1000)
    assert res["standardised_rate"] == pytest.approx(10.0)
    assert res["crude_rate"] == pytest.approx(10.0)


def test_direct_single_stratum_gamma_ci_equals_exact_poisson():
    # With one stratum the Fay-Feuer interval collapses to the Garwood interval.
    res = sd.direct_standardisation(
        [{"events": 12, "person_time": 800, "standard_population": 500}], multiplier=1000)
    exact = sd.poisson_exact_ci(12, 800, 0.05)
    assert res["standardised_ci_low"] == pytest.approx(exact["low"] * 1000, rel=1e-9)
    assert res["standardised_ci_high"] == pytest.approx(exact["high"] * 1000, rel=1e-9)


def test_direct_zero_events_everywhere():
    res = sd.direct_standardisation(
        [{"events": 0, "person_time": 100, "standard_population": 1},
         {"events": 0, "person_time": 200, "standard_population": 3}])
    assert res["standardised_rate"] == 0.0
    assert res["standardised_ci_low"] == 0.0
    assert res["standardised_ci_high"] > 0.0


def test_direct_population_alias_and_multiplier():
    a = sd.direct_standardisation(_strata(), multiplier=1000)
    s2 = [{"label": s["label"], "events": s["events"], "population": s["person_time"],
           "standard_population": s["standard_population"]} for s in _strata()]
    b = sd.direct_standardisation(s2, multiplier=1000)
    assert a["standardised_rate"] == pytest.approx(b["standardised_rate"])
    c = sd.direct_standardisation(_strata(), multiplier=100000)
    assert c["standardised_rate"] == pytest.approx(a["standardised_rate"] * 100)


def test_direct_zero_person_time_stratum_is_excluded_with_warning():
    strata = _strata()
    strata.append({"label": "empty", "events": 0, "person_time": 0, "standard_population": 5000})
    res = sd.direct_standardisation(strata)
    base = sd.direct_standardisation(_strata())
    assert res["standardised_rate"] == pytest.approx(base["standardised_rate"])
    assert any("empty" in w and "zero person-time" in w for w in res["warnings"])
    assert res["strata"][-1]["excluded"] is True
    assert res["strata"][-1]["rate"] is None


def test_direct_zero_standard_population_gets_zero_weight():
    strata = _strata()
    strata.append({"label": "nostd", "events": 3, "person_time": 100, "standard_population": 0})
    res = sd.direct_standardisation(strata)
    base = sd.direct_standardisation(_strata())
    assert res["standardised_rate"] == pytest.approx(base["standardised_rate"])
    assert res["strata"][-1]["weight"] == 0.0
    assert any("nostd" in w and "zero standard population" in w for w in res["warnings"])
    # the crude rate still counts that stratum
    assert res["total_events"] == sum(EVENTS) + 3


@pytest.mark.parametrize("mutate,msg", [
    (lambda s: s.clear(), "at least one stratum"),
    (lambda s: s[0].update(events=-1), "negative"),
    (lambda s: s[0].update(events=2.5), "whole number"),
    (lambda s: s[0].update(person_time=float("nan")), "finite"),
    (lambda s: s[0].update(standard_population=-5), "negative"),
    (lambda s: s[0].update(person_time=0), "person_time is 0"),
])
def test_direct_validation_errors(mutate, msg):
    strata = _strata()
    mutate(strata)
    with pytest.raises(ValueError, match=msg):
        sd.direct_standardisation(strata)


def test_direct_all_unusable_raises():
    with pytest.raises(ValueError, match="No usable stratum"):
        sd.direct_standardisation([{"events": 1, "person_time": 10, "standard_population": 0}])


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
def test_alpha_validation(alpha):
    with pytest.raises(ValueError, match="alpha"):
        sd.direct_standardisation(_strata(), alpha=alpha)


# ── indirect standardisation ─────────────────────────────────────────────────


def _indirect_strata():
    # Study group person-years and observed events; reference rates per 1000.
    return [
        {"label": "a", "observed": 5, "person_time": 2000, "reference_rate": 1.5},
        {"label": "b", "observed": 12, "person_time": 1500, "reference_rate": 6.0},
        {"label": "c", "observed": 40, "person_time": 1000, "reference_rate": 25.0},
    ]


def test_indirect_hand_computation():
    res = sd.indirect_standardisation(_indirect_strata(), reference_multiplier=1000)
    expected = 2000 * 1.5 / 1000 + 1500 * 6.0 / 1000 + 1000 * 25.0 / 1000   # 3 + 9 + 25 = 37
    assert res["expected"] == pytest.approx(37.0)
    assert res["expected"] == pytest.approx(expected)
    assert res["observed"] == 57
    assert res["smr"] == pytest.approx(57 / 37)
    assert res["smr_x100"] == pytest.approx(57 / 37 * 100)

    z = stats.norm.ppf(0.975)
    o = 57.0
    byar_lo = o * (1 - 1 / (9 * o) - z / (3 * np.sqrt(o))) ** 3 / 37
    byar_hi = (o + 1) * (1 - 1 / (9 * (o + 1)) + z / (3 * np.sqrt(o + 1))) ** 3 / 37
    assert res["byar_ci_low"] == pytest.approx(byar_lo)
    assert res["byar_ci_high"] == pytest.approx(byar_hi)
    assert res["exact_ci_low"] == pytest.approx(stats.chi2.ppf(0.025, 114) / 2 / 37)
    assert res["exact_ci_high"] == pytest.approx(stats.chi2.ppf(0.975, 116) / 2 / 37)

    p = 2 * min(stats.poisson.cdf(57, 37), stats.poisson.sf(56, 37))
    assert res["p_value"] == pytest.approx(p)
    assert res["p_value"] < 0.01

    assert [r["expected"] for r in res["strata"]] == pytest.approx([3.0, 9.0, 25.0])
    assert [r["ratio"] for r in res["strata"]] == pytest.approx([5 / 3, 12 / 9, 40 / 25])
    assert "SMR" in res["result_text"]
    assert "pois.exact" in res["r_code"]


def test_indirect_reference_events_equivalent_to_rate():
    a = sd.indirect_standardisation(_indirect_strata(), reference_multiplier=1000)
    alt = [
        {"label": "a", "observed": 5, "person_time": 2000, "reference_events": 15, "reference_person_time": 10000},
        {"label": "b", "observed": 12, "person_time": 1500, "reference_events": 60, "reference_person_time": 10000},
        {"label": "c", "observed": 40, "person_time": 1000, "reference_events": 250, "reference_person_time": 10000},
    ]
    b = sd.indirect_standardisation(alt)
    assert b["expected"] == pytest.approx(a["expected"])
    assert b["smr"] == pytest.approx(a["smr"])


def test_indirect_byar_close_to_exact_for_large_counts():
    res = sd.indirect_standardisation(
        [{"observed": 400, "person_time": 1000, "reference_rate": 0.5}])   # E = 500
    assert res["smr"] == pytest.approx(0.8)
    assert res["byar_ci_low"] == pytest.approx(res["exact_ci_low"], abs=2e-3)
    assert res["byar_ci_high"] == pytest.approx(res["exact_ci_high"], abs=2e-3)


def test_indirect_zero_observed():
    res = sd.indirect_standardisation(
        [{"observed": 0, "person_time": 100, "reference_rate": 0.04}])   # E = 4
    assert res["smr"] == 0.0
    assert res["byar_ci_low"] == 0.0
    assert res["exact_ci_low"] == 0.0
    assert res["exact_ci_high"] == pytest.approx(3.688879 / 4, abs=1e-5)
    assert res["byar_ci_high"] > 0
    assert res["p_value"] == pytest.approx(2 * np.exp(-4))
    assert any("Fewer than 10" in w for w in res["warnings"])


def test_indirect_smr_one_gives_p_one():
    res = sd.indirect_standardisation([{"observed": 20, "person_time": 100, "reference_rate": 0.2}])
    assert res["smr"] == pytest.approx(1.0)
    assert res["p_value"] > 0.9


@pytest.mark.parametrize("mutate,msg", [
    (lambda s: s.clear(), "at least one stratum"),
    (lambda s: s[0].update(observed=1.5), "whole number"),
    (lambda s: s[0].update(observed=-1), "negative"),
    (lambda s: s[0].update(reference_rate=-1), "negative"),
    (lambda s: s[0].update(person_time=-1), "negative"),
    (lambda s: (s[0].pop("reference_rate")), "reference_rate"),
    (lambda s: s[0].update(reference_rate=float("inf")), "finite"),
])
def test_indirect_validation_errors(mutate, msg):
    strata = _indirect_strata()
    mutate(strata)
    with pytest.raises(ValueError, match=msg):
        sd.indirect_standardisation(strata, reference_multiplier=1000)


def test_indirect_zero_expected_raises():
    with pytest.raises(ValueError, match="Expected count is 0"):
        sd.indirect_standardisation([{"observed": 0, "person_time": 100, "reference_rate": 0}])


def test_indirect_bad_multiplier():
    with pytest.raises(ValueError, match="reference_multiplier"):
        sd.indirect_standardisation(_indirect_strata(), reference_multiplier=0)


# ── rate ratio ───────────────────────────────────────────────────────────────


def test_rate_ratio_hand_computation():
    res = sd.rate_ratio(20, 1000, 10, 2000)
    assert res["group1"]["rate"] == pytest.approx(0.02)
    assert res["group2"]["rate"] == pytest.approx(0.005)
    assert res["rate_ratio"] == pytest.approx(4.0)
    se = np.sqrt(1 / 20 + 1 / 10)
    z = stats.norm.ppf(0.975)
    assert res["rr_ci_low"] == pytest.approx(np.exp(np.log(4) - z * se))
    assert res["rr_ci_high"] == pytest.approx(np.exp(np.log(4) + z * se))
    assert res["rate_difference"] == pytest.approx(0.015)
    sed = np.sqrt(20 / 1000 ** 2 + 10 / 2000 ** 2)
    assert res["rd_ci_low"] == pytest.approx(0.015 - z * sed)
    assert res["rd_ci_high"] == pytest.approx(0.015 + z * sed)
    g1 = sd.poisson_exact_ci(20, 1000)
    assert res["group1"]["ci_low"] == pytest.approx(g1["low"])
    assert res["group1"]["ci_high"] == pytest.approx(g1["high"])

    # conditional exact test: events1 | 30 ~ Binom(30, 1/3) under H0
    p_exact = stats.binomtest(20, 30, 1 / 3).pvalue
    assert res["p_value"] == pytest.approx(p_exact)
    assert res["p_value_midp"] <= res["p_value"] + 1e-12
    assert res["p_value"] < 0.001
    assert res["rr_exact_ci_low"] < 4.0 < res["rr_exact_ci_high"]
    # the conditional exact CI is wider than or similar to the log-method CI
    assert res["rr_exact_ci_low"] == pytest.approx(res["rr_ci_low"], rel=0.25)
    assert "rate ratio" in res["result_text"].lower()
    assert "poisson.test" in res["r_code"]


def test_rate_ratio_equal_rates_not_significant():
    res = sd.rate_ratio(10, 1000, 20, 2000)
    assert res["rate_ratio"] == pytest.approx(1.0)
    assert res["rate_difference"] == pytest.approx(0.0)
    assert res["p_value"] == pytest.approx(1.0)
    assert res["p_value_wald"] == pytest.approx(1.0)


def test_rate_ratio_zero_cell_uses_exact_ci():
    res = sd.rate_ratio(0, 500, 8, 600)
    assert res["rate_ratio"] == 0.0
    assert res["rr_ci_low"] is None and res["rr_ci_high"] is None
    assert res["rr_exact_ci_low"] == 0.0
    assert res["rr_exact_ci_high"] is not None and res["rr_exact_ci_high"] > 0
    assert res["p_value"] is not None and res["p_value"] < 0.05
    assert res["warnings"]
    res2 = sd.rate_ratio(5, 500, 0, 600)
    assert res2["rate_ratio"] is None
    assert res2["rr_exact_ci_low"] > 0
    assert res2["rr_exact_ci_high"] is None


def test_rate_ratio_no_events_at_all():
    res = sd.rate_ratio(0, 500, 0, 600)
    assert res["rate_ratio"] is None
    assert res["p_value"] is None
    assert any("undefined" in w for w in res["warnings"])
    assert res["result_text"]


@pytest.mark.parametrize("args,msg", [
    ((-1, 10, 1, 10), "negative"),
    ((1.5, 10, 1, 10), "whole number"),
    ((1, 0, 1, 10), "greater than 0"),
    ((1, 10, 1, -3), "negative"),
    ((1, float("inf"), 1, 10), "finite"),
    ((1, 10, None, 10), "required"),
])
def test_rate_ratio_validation(args, msg):
    with pytest.raises(ValueError, match=msg):
        sd.rate_ratio(*args)


# ── HTTP layer ───────────────────────────────────────────────────────────────

DIRECT_KEYS = {
    "test", "alpha", "conf_level", "multiplier", "n_strata", "total_events",
    "total_person_time", "crude_rate", "crude_ci_low", "crude_ci_high",
    "standardised_rate", "standardised_ci_low", "standardised_ci_high",
    "standardised_se", "ci_method", "strata", "warnings", "result_text", "r_code",
}
INDIRECT_KEYS = {
    "test", "alpha", "conf_level", "reference_multiplier", "n_strata", "observed",
    "expected", "smr", "smr_x100", "byar_ci_low", "byar_ci_high", "byar_ci_low_x100",
    "byar_ci_high_x100", "exact_ci_low", "exact_ci_high", "exact_ci_low_x100",
    "exact_ci_high_x100", "p_value", "p_method", "strata", "warnings",
    "result_text", "r_code",
}
RATE_RATIO_KEYS = {
    "test", "alpha", "conf_level", "group1", "group2", "rate_ratio", "rr_ci_low",
    "rr_ci_high", "rr_se_log", "rr_ci_method", "rr_exact_ci_low", "rr_exact_ci_high",
    "rate_difference", "rd_ci_low", "rd_ci_high", "rd_se", "p_value", "p_value_midp",
    "p_value_wald", "p_method", "warnings", "result_text", "r_code",
}


def test_api_direct(client):
    r = client.post(f"{BASE}/direct_standardisation", json={
        "strata": _strata(), "multiplier": 100000, "alpha": 0.05})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == DIRECT_KEYS
    assert body["standardised_rate"] == pytest.approx(
        sd.direct_standardisation(_strata())["standardised_rate"])


def test_api_direct_defaults_and_population_alias(client):
    strata = [{"label": "x", "events": 5, "population": 1000, "standard_population": 10},
              {"label": "y", "events": 9, "population": 800, "standard_population": 30}]
    r = client.post(f"{BASE}/direct_standardisation", json={"strata": strata})
    assert r.status_code == 200
    assert r.json()["multiplier"] == 100000


def test_api_indirect(client):
    r = client.post(f"{BASE}/indirect_standardisation", json={
        "strata": _indirect_strata(), "reference_multiplier": 1000})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == INDIRECT_KEYS
    assert body["expected"] == pytest.approx(37.0)


def test_api_rate_ratio(client):
    r = client.post(f"{BASE}/rate_ratio", json={
        "events1": 20, "person_time1": 1000, "events2": 10, "person_time2": 2000})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == RATE_RATIO_KEYS
    assert body["rate_ratio"] == pytest.approx(4.0)


def test_api_validation_returns_422_with_message(client):
    r = client.post(f"{BASE}/direct_standardisation", json={"strata": []})
    assert r.status_code == 422
    assert "at least one stratum" in r.json()["detail"]

    bad = _strata()
    bad[0]["events"] = 1.5
    r = client.post(f"{BASE}/direct_standardisation", json={"strata": bad})
    assert r.status_code == 422
    assert "whole number" in r.json()["detail"]

    r = client.post(f"{BASE}/indirect_standardisation", json={
        "strata": [{"observed": 1, "person_time": 10}]})
    assert r.status_code == 422
    assert "reference_rate" in r.json()["detail"]

    r = client.post(f"{BASE}/rate_ratio", json={
        "events1": 1, "person_time1": 0, "events2": 1, "person_time2": 5})
    assert r.status_code == 422

    # schema-level failure (missing field) is a 422 as well
    r = client.post(f"{BASE}/rate_ratio", json={"events1": 1})
    assert r.status_code == 422


def test_routes_registered(client):
    from main import app
    paths = {getattr(route, "path", None) for route in app.routes}
    for name in ("direct_standardisation", "indirect_standardisation", "rate_ratio"):
        assert f"{BASE}/{name}" in paths


# ── optional R parity (epitools) ─────────────────────────────────────────────


def _epitools_available() -> bool:
    if not shutil.which("Rscript"):
        return False
    probe = subprocess.run(
        ["Rscript", "-e", "quit(status = ifelse(requireNamespace('epitools', quietly = TRUE), 0, 1))"],
        capture_output=True, text=True, timeout=60)
    return probe.returncode == 0


def test_r_parity_ageadjust_direct():
    if not _epitools_available():
        pytest.skip("Rscript with epitools not installed")
    code = (
        "library(epitools);library(jsonlite);"
        f"r <- ageadjust.direct(count=c({','.join(map(str, EVENTS))}),"
        f"pop=c({','.join(map(str, PT))}),stdpop=c({','.join(map(str, STD))}));"
        "cat(toJSON(as.list(r), digits=15))"
    )
    out = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    ref = json.loads(out.stdout)
    res = sd.direct_standardisation(_strata(), multiplier=1)
    assert res["standardised_rate"] == pytest.approx(ref["adj.rate"][0] if "adj.rate" in ref else ref["adj.rate"], rel=1e-8)
