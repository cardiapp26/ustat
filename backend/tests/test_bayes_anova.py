"""Bayesian one-way ANOVA against BayesFactor::anovaBF.

Reference: R 4.5.2, BayesFactor 0.9.12-4.8,

    anovaBF(y ~ g, data = d, rscaleFixed = r)   # r = 0.5, sqrt(2)/2, 1

on two simulated unbalanced designs (4 and 3 groups). BayesFactor reports a
numerical error of at most about 1e-4 on these; uSTAT agrees to 1e-6.
"""
import math

import numpy as np
import pandas as pd
import pytest

from conftest import make_session
from services.bayes_anova import jzs_oneway_bf

_CASE3_Y = [-0.9619334159, -0.2925257229, 0.2587882162, -1.1521318859, 0.1957828263, 0.0301239446, 0.0854177316, 1.1166102127, -1.2188574156, 1.2673687221, 0.0552184039, -0.3312185708, 0.08364151, 1.0526523696, 0.9520457067, 0.4923435703, -0.1530173309, 0.1517571886, 2.0243136243, 0.999811608, 0.2215162781, -0.1423007335, -0.0037281797, -1.46647484, -0.2844551092, -0.5410726607, 1.3606157792, 1.2120671249, 0.1279215259, -0.9367822981, 1.100624729, 1.3517704471, 1.2277151742, 1.2365021457, 0.1478703831, 1.2055155135, 1.8003579887, 0.5382520141, -0.47928377, 1.2937612309, 1.286506872, 0.189536869, 2.1988848456, -0.2945937085, 0.8484377162]
_CASE3_G = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
_CASE3_BF = {'medium': 2.42233294028, 'wide': 1.96354615871, 'ultrawide': 1.36038780329}

_CASE5_Y = [-0.8408554808, 1.3843593435, -1.2554918626, 0.0701427664, 1.7114408727, -0.6029079815, -0.4721663852, -0.6353713125, 0.9142263651, 1.3381082248, 2.4276303439, 0.3982205453, 0.1196074, 1.0424656439, 0.1282399601, 1.0610138595, 0.6026869053, -0.9839667601, 1.4408172559, 0.9406445933, 2.1005119453, 2.1418693939, 2.6679619034, 1.9067610896, 2.0190089303, 0.9065181513, 2.6185890725, 2.6987738274, 0.5429179055, 0.34720456, 1.5159150384, 2.3096941677, 3.4154605717, 2.417103639, 2.6792217866, 2.1515738324, 0.190467354, -0.8004727386, -1.3621858725, 0.257391874, 1.9500603695, -0.4024231817, 0.3254210801, 2.2956679547, -0.0565689409, 0.9622233626, -0.4870085115, -0.0602445762, -0.3243284861, 0.3307888442]
_CASE5_G = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
_CASE5_BF = {'medium': 44.0344791378, 'wide': 43.7082098154, 'ultrawide': 37.9059637748}


def _post(client, sid, **kw):
    return client.post("/api/bayesian", json={
        "session_id": sid, "analysis_type": "anova", "outcome": "y", "predictor": "g", **kw})


@pytest.mark.parametrize("scale", ["medium", "wide", "ultrawide"])
@pytest.mark.parametrize("y,g,bf", [(_CASE3_Y, _CASE3_G, _CASE3_BF), (_CASE5_Y, _CASE5_G, _CASE5_BF)])
def test_bf10_matches_anovabf(client, y, g, bf, scale):
    sid = make_session(pd.DataFrame({"y": y, "g": g}), f"bayes_anova_{len(y)}_{scale}")
    r = _post(client, sid, rscale=scale)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["bf10"] == pytest.approx(bf[scale], rel=1e-5)
    assert out["bf10"] * out["bf01"] == pytest.approx(1.0, rel=1e-4)
    assert "anovaBF" in out["r_code"]


def test_default_prior_is_bayesfactor_medium(client):
    sid = make_session(pd.DataFrame({"y": _CASE3_Y, "g": _CASE3_G}), "bayes_anova_default")
    out = _post(client, sid).json()
    assert out["prior_scale"] == 0.5
    assert out["bf10"] == pytest.approx(_CASE3_BF["medium"], rel=1e-5)


def test_f_eta_squared_and_group_summaries(client):
    sid = make_session(pd.DataFrame({"y": _CASE5_Y, "g": _CASE5_G}), "bayes_anova_desc")
    out = _post(client, sid).json()
    assert out["df"] == "2, 47"
    assert [grp["n"] for grp in out["groups"]] == [8, 30, 12]
    assert 0 < out["effect_size_value"] < 1


def test_overwhelming_evidence_reports_the_log_not_infinity():
    # log10 BF ~ 219.7 here (checked against anovaBF: 219.7231207762).
    rng = np.random.default_rng(0)
    y = np.r_[rng.normal(0, 1, 2000), rng.normal(0.5, 1, 2000), rng.normal(1, 1, 2000)]
    g = np.repeat([1, 2, 3], 2000)
    bf = jzs_oneway_bf(y, g)
    assert bf.log_bf10 / math.log(10) == pytest.approx(219.7231207762, rel=1e-9)


def test_unknown_prior_scale_is_rejected(client):
    sid = make_session(pd.DataFrame({"y": _CASE3_Y, "g": _CASE3_G}), "bayes_anova_bad_scale")
    r = _post(client, sid, rscale="huge")
    assert r.status_code == 422


def test_single_group_is_rejected(client):
    sid = make_session(pd.DataFrame({"y": [1.0, 2.0, 3.0], "g": ["a", "a", "a"]}), "bayes_anova_one_group")
    assert _post(client, sid).status_code == 400
