"""Evidence-based-medicine risk measures (ARD, RR, RRR, NNT/NNH) and the
Breslow-Day homogeneity test next to the Mantel-Haenszel common odds ratio.

Hand-computed reference: the 2x2 table [[20, 80], [40, 60]] has
    risk_exposed = 0.2, risk_reference = 0.4
    ARD = -0.2, RR = 0.5, RRR = 0.5, NNT = 1 / 0.2 = 5
    Katz SE(ln RR) = sqrt(1/20 - 1/100 + 1/40 - 1/100) = sqrt(0.055)
"""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.proportion import confint_proportions_2indep

from conftest import make_session
from services.risk_measures import (
    compute_risk_measures,
    number_needed,
    risk_measures_export_rows,
    risk_measures_text,
    risk_ratio_katz,
)


# ── pure helpers ──────────────────────────────────────────────────────────────

def test_reference_table_hand_computed():
    rm = compute_risk_measures(20, 100, 40, 100, alpha=0.05,
                               event="yes", exposed="A", reference="B")
    assert rm["risk_exposed"] == pytest.approx(0.2)
    assert rm["risk_reference"] == pytest.approx(0.4)
    assert rm["ard"] == pytest.approx(-0.2)
    assert rm["arr"] == pytest.approx(0.2)
    assert rm["rr"] == pytest.approx(0.5)
    assert rm["rrr"] == pytest.approx(0.5)
    se = math.sqrt(0.055)
    z = sp.norm.ppf(0.975)
    assert rm["rr_ci"][0] == pytest.approx(math.exp(math.log(0.5) - z * se))
    assert rm["rr_ci"][1] == pytest.approx(math.exp(math.log(0.5) + z * se))
    # RRR CI is 1 - RR CI with the limits swapped
    assert rm["rrr_ci"][0] == pytest.approx(1 - rm["rr_ci"][1])
    assert rm["rrr_ci"][1] == pytest.approx(1 - rm["rr_ci"][0])
    assert rm["event"] == "yes" and rm["exposed"] == "A" and rm["reference"] == "B"

    nn = rm["nnt_nnh"]
    assert nn["kind"] == "NNT"
    assert nn["value"] == 5
    assert nn["ci_spans_zero"] is False

    lo, hi = confint_proportions_2indep(20, 100, 40, 100, method="newcomb", alpha=0.05)
    assert rm["ard_ci"] == pytest.approx([lo, hi])
    # Altman: invert the ARD CI limits, round up
    assert nn["low"] == math.ceil(1 / abs(lo) - 1e-9)
    assert nn["high"] == math.ceil(1 / abs(hi) - 1e-9)
    assert nn["low"] <= nn["value"] <= nn["high"]


def test_harm_direction_is_nnh():
    rm = compute_risk_measures(40, 100, 20, 100)
    assert rm["ard"] == pytest.approx(0.2)
    assert rm["rr"] == pytest.approx(2.0)
    assert rm["rrr"] == pytest.approx(-1.0)
    assert rm["nnt_nnh"]["kind"] == "NNH"
    assert rm["nnt_nnh"]["value"] == 5


def test_nnt_rounds_up():
    # ARD = 0.15 -> 1 / 0.15 = 6.67 -> 7
    rm = compute_risk_measures(15, 100, 30, 100)
    assert rm["nnt_nnh"]["kind"] == "NNT"
    assert rm["nnt_nnh"]["value"] == 7


def test_ci_text_in_altman_two_part_form_when_ard_ci_spans_zero():
    # Exposed worse: ARD +0.10, CI -0.04 to 0.125 -> NNH 8 to infinity to NNT 25
    nn = number_needed(0.10, (-0.04, 0.125))
    assert nn["kind"] == "NNH"
    assert nn["value"] == 10
    assert nn["ci_spans_zero"] is True
    assert nn["low"] == 8
    assert nn["high"] is None
    assert nn["other_kind"] == "NNT"
    assert nn["other_low"] == 25
    assert nn["ci_text"] == "NNH 8 to infinity to NNT 25"

    # Mirror image: exposed protective
    nn = number_needed(-0.10, (-0.125, 0.04))
    assert nn["kind"] == "NNT"
    assert nn["ci_text"] == "NNT 8 to infinity to NNH 25"


def test_spanning_zero_from_real_counts():
    rm = compute_risk_measures(12, 100, 8, 100)
    nn = rm["nnt_nnh"]
    assert nn["kind"] == "NNH"
    assert nn["ci_spans_zero"] is True
    assert nn["high"] is None
    assert nn["ci_text"].startswith("NNH ") and " to infinity to NNT " in nn["ci_text"]


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2])
def test_rr_ci_uses_alpha(alpha):
    got = risk_ratio_katz(20, 100, 40, 100, alpha)
    z = sp.norm.ppf(1 - alpha / 2)
    se = math.sqrt(0.055)
    assert got["ci_low"] == pytest.approx(math.exp(math.log(0.5) - z * se))
    assert got["ci_high"] == pytest.approx(math.exp(math.log(0.5) + z * se))


def test_narrower_alpha_widens_everything():
    a = compute_risk_measures(20, 100, 40, 100, alpha=0.05)
    b = compute_risk_measures(20, 100, 40, 100, alpha=0.01)
    assert b["ard_ci"][0] < a["ard_ci"][0] and b["ard_ci"][1] > a["ard_ci"][1]
    assert b["rr_ci"][0] < a["rr_ci"][0] and b["rr_ci"][1] > a["rr_ci"][1]
    assert b["ci_level"] == pytest.approx(0.99)


def test_zero_ard_gives_infinite_number_needed():
    rm = compute_risk_measures(10, 50, 20, 100)
    assert rm["ard"] == pytest.approx(0.0)
    assert rm["nnt_nnh"]["value"] is None
    assert rm["nnt_nnh"]["kind"] is None
    assert "infinite" in rm["nnt_nnh"]["ci_text"].lower()
    assert rm["rr"] == pytest.approx(1.0)
    assert rm["rrr"] == pytest.approx(0.0)
    rows = dict(risk_measures_export_rows(rm))
    assert rows["NNT/NNH"] == "infinite"


def test_zero_cells_do_not_crash():
    # no events in the exposed group: RR = 0, Katz CI undefined
    rm = compute_risk_measures(0, 50, 10, 50)
    assert rm["rr"] == pytest.approx(0.0)
    assert rm["rr_ci"] == [None, None]
    assert rm["rrr"] == pytest.approx(1.0)
    assert rm["nnt_nnh"]["kind"] == "NNT"
    assert rm["nnt_nnh"]["value"] == 5

    # no events in the reference group: RR undefined
    rm = compute_risk_measures(10, 50, 0, 50)
    assert rm["rr"] is None and rm["rrr"] is None
    assert rm["nnt_nnh"]["kind"] == "NNH"

    # no events anywhere
    rm = compute_risk_measures(0, 50, 0, 50)
    assert rm["ard"] == 0.0
    assert rm["nnt_nnh"]["value"] is None
    assert isinstance(risk_measures_text(rm), str)


def test_text_and_rows_mention_the_measures():
    rm = compute_risk_measures(20, 100, 40, 100, event="yes", exposed="A", reference="B")
    text = risk_measures_text(rm)
    assert "NNT = 5" in text and "risk ratio = 0.500" in text
    rows = dict(risk_measures_export_rows(rm))
    assert rows["NNT"] == 5
    assert rows["Relative risk reduction"] == pytest.approx(0.5)


# ── /two_proportions ──────────────────────────────────────────────────────────

def _two_prop_frame(k_a, n_a, k_b, n_b):
    return pd.DataFrame({
        "outcome": [1] * k_a + [0] * (n_a - k_a) + [1] * k_b + [0] * (n_b - k_b),
        "group": ["A"] * n_a + ["B"] * n_b,
    })


def test_two_proportions_endpoint_risk_measures(client):
    sid = make_session(_two_prop_frame(20, 100, 40, 100), "ebm_tp1")
    r = client.post("/api/categorical/two_proportions", json={
        "session_id": sid, "column": "outcome", "group_column": "group",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    # existing keys are untouched
    assert d["diff_prop"] == pytest.approx(-0.2)
    assert d["ci_diff_low"] < d["diff_prop"] < d["ci_diff_high"]
    rm = d["risk_measures"]
    assert rm["exposed"] == "A" and rm["reference"] == "B"
    assert rm["event"] == "1"
    assert rm["ard"] == pytest.approx(d["diff_prop"])
    assert rm["ard_ci"] == pytest.approx([d["ci_diff_low"], d["ci_diff_high"]])
    assert d["arr"] == pytest.approx(0.2)
    assert rm["rr"] == pytest.approx(0.5)
    assert rm["rrr"] == pytest.approx(0.5)
    assert rm["nnt_nnh"]["kind"] == "NNT" and rm["nnt_nnh"]["value"] == 5
    assert "NNT = 5" in d["result_text"]
    assert any(row[0] == "NNT" and row[1] == 5 for row in d["export_rows"])
    assert "RelRisk" in d["r_code"]


def test_two_proportions_endpoint_respects_alpha(client):
    sid = make_session(_two_prop_frame(20, 100, 40, 100), "ebm_tp_alpha")
    base = client.post("/api/categorical/two_proportions", json={
        "session_id": sid, "column": "outcome", "group_column": "group",
    }).json()["risk_measures"]
    wide = client.post("/api/categorical/two_proportions", json={
        "session_id": sid, "column": "outcome", "group_column": "group", "alpha": 0.01,
    }).json()["risk_measures"]
    assert wide["ci_level"] == pytest.approx(0.99)
    assert wide["rr_ci"][0] < base["rr_ci"][0]
    assert wide["rr_ci"][1] > base["rr_ci"][1]


def test_two_proportions_endpoint_zero_cell(client):
    sid = make_session(_two_prop_frame(0, 50, 10, 50), "ebm_tp_zero")
    r = client.post("/api/categorical/two_proportions", json={
        "session_id": sid, "column": "outcome", "group_column": "group",
    })
    assert r.status_code == 200, r.text
    rm = r.json()["risk_measures"]
    assert rm["rr"] == pytest.approx(0.0)
    assert rm["rr_ci"] == [None, None]
    assert rm["nnt_nnh"]["value"] == 5


def test_two_proportions_endpoint_zero_ard(client):
    sid = make_session(_two_prop_frame(10, 50, 20, 100), "ebm_tp_zero_ard")
    r = client.post("/api/categorical/two_proportions", json={
        "session_id": sid, "column": "outcome", "group_column": "group",
    })
    assert r.status_code == 200, r.text
    assert r.json()["risk_measures"]["nnt_nnh"]["value"] is None


# ── /chisquare 2x2 ────────────────────────────────────────────────────────────

def test_chisquare_2x2_risk_measures(client):
    # crosstab sorts levels, so row 0 = "A", column 0 = "ev" (event).
    df = pd.DataFrame({
        "arm": ["A"] * 100 + ["B"] * 100,
        "result": (["ev"] * 20 + ["no"] * 80) + (["ev"] * 40 + ["no"] * 60),
    })
    sid = make_session(df, "ebm_chi1")
    r = client.post("/api/stats/chisquare", json={
        "session_id": sid, "row_column": "arm", "col_column": "result",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["relative_risk"] == pytest.approx(0.5)
    rm = d["risk_measures"]
    assert rm["event"] == d["risk_event"] == "ev"
    assert rm["exposed"] == d["risk_exposed"] == "A"
    assert rm["reference"] == d["risk_reference"] == "B"
    assert rm["ard"] == pytest.approx(-0.2)
    assert rm["rr"] == pytest.approx(0.5)
    assert rm["rrr"] == pytest.approx(0.5)
    assert rm["nnt_nnh"]["kind"] == "NNT" and rm["nnt_nnh"]["value"] == 5


def test_chisquare_larger_table_has_no_risk_measures(client):
    df = pd.DataFrame({
        "arm": ["A", "B", "C"] * 40,
        "result": (["x", "y"] * 60),
    })
    sid = make_session(df, "ebm_chi_3x2")
    r = client.post("/api/stats/chisquare", json={
        "session_id": sid, "row_column": "arm", "col_column": "result",
    })
    assert r.status_code == 200, r.text
    assert "risk_measures" not in r.json()


# ── /mantel_haenszel homogeneity ──────────────────────────────────────────────

def _strat_frame(tables):
    rows = []
    for stratum, table in enumerate(tables):
        for i in range(2):
            for j in range(2):
                rows.extend({"row": i, "col": j, "stratum": stratum}
                            for _ in range(table[i][j]))
    return pd.DataFrame(rows)


def _post_mh(client, tables, sid, **extra):
    sid = make_session(_strat_frame(tables), sid)
    return client.post("/api/categorical/mantel_haenszel", json={
        "session_id": sid, "row_col": "row", "col_col": "col",
        "strata_col": "stratum", **extra,
    })


def test_cmh_homogeneity_matches_statsmodels(client):
    tables = [[[12, 8], [6, 14]], [[15, 5], [10, 10]], [[9, 11], [7, 13]]]
    r = _post_mh(client, tables, "ebm_mh1")
    assert r.status_code == 200, r.text
    d = r.json()
    ref = StratifiedTable([np.array(t, dtype=float) for t in tables]).test_equal_odds(adjust=True)
    h = d["homogeneity_test"]
    assert h["adjusted"] is True
    assert h["statistic"] == pytest.approx(float(ref.statistic), abs=1e-4)
    assert h["p"] == pytest.approx(float(ref.pvalue))
    assert h["df"] == len(tables) - 1
    assert h["homogeneous"] is True
    assert "Breslow-Day" in h["name"]
    assert not any("pooled odds ratio" in w for w in d["warnings"])
    assert "Breslow-Day" in d["result_text"]
    assert any("Breslow-Day" in str(row[0]) for row in d["export_rows"])
    assert "BreslowDayTest" in d["r_code"]


def test_cmh_heterogeneous_strata_warn(client):
    # very different odds ratios: OR = 36 in one stratum, 1.0 in the other
    tables = [[[30, 5], [10, 60]], [[20, 20], [20, 20]]]
    r = _post_mh(client, tables, "ebm_mh_het")
    assert r.status_code == 200, r.text
    d = r.json()
    ref = StratifiedTable([np.array(t, dtype=float) for t in tables]).test_equal_odds(adjust=True)
    h = d["homogeneity_test"]
    assert h["p"] == pytest.approx(float(ref.pvalue))
    assert h["p"] < 0.05
    assert h["homogeneous"] is False
    assert any("pooled odds ratio may be misleading" in w for w in d["warnings"])
    assert "heterogeneous" in d["result_text"]


def test_cmh_homogeneity_threshold_follows_alpha(client):
    tables = [[[12, 8], [6, 14]], [[15, 5], [10, 10]], [[9, 11], [7, 13]]]
    p = _post_mh(client, tables, "ebm_mh_a1").json()["homogeneity_test"]["p"]
    strict = _post_mh(client, tables, "ebm_mh_a2", alpha=min(0.99, p + 0.01)).json()
    assert strict["homogeneity_test"]["homogeneous"] is False


def test_cmh_degenerate_strata_do_not_fail(client):
    # the second stratum has an empty column margin (singular table)
    tables = [[[12, 8], [6, 14]], [[5, 0], [3, 0]], [[15, 5], [10, 10]]]
    r = _post_mh(client, tables, "ebm_mh_sing")
    assert r.status_code == 200, r.text
    d = r.json()
    h = d["homogeneity_test"]
    if h is None:
        assert d["homogeneity_note"]
        assert any("Breslow-Day" in w for w in d["warnings"])
    else:
        assert np.isfinite(h["p"])
    # the CMH result itself is still reported
    assert d["statistic"] is not None
