"""Two defects that reversed a clinical reading, and their invariants.

K1 — mixed ANOVA fitted a plain factorial OLS with no subject term at all,
so repeated measurements of the same person counted as independent cases and
the between-subjects effect was judged against within-subject noise.

K2 — the non-inferiority p-value was built from the wrong tail, so a call
that demonstrated non-inferiority on the confidence-interval rule printed
p = 0.99999 beside it. The decision and its p contradicted each other on
every call, in both directions of the test.

Reference values come from R: aov(y ~ arm*time + Error(id/time)) for the
split plot, and a one-sided Welch t for the continuous non-inferiority case.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp

from conftest import make_session


# ── K1: mixed ANOVA error strata ──────────────────────────────────────────────


@pytest.fixture()
def split_plot() -> pd.DataFrame:
    """9 subjects, 3 arms, 3 time points; a strong time effect, no arm effect.

    Every subject carries their own offset, which is exactly the structure a
    pooled residual ignores.
    """
    rows = []
    offsets = {1: 0.0, 2: 1.0, 3: 2.0, 4: 0.5, 5: 1.5, 6: 2.5, 7: 1.0, 8: 2.0, 9: 3.0}
    arms = {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B", 7: "C", 8: "C", 9: "C"}
    time_effect = {"t1": 0.0, "t2": 1.0, "t3": 2.0}
    for sid, off in offsets.items():
        for t, te in time_effect.items():
            rows.append({"id": sid, "arm": arms[sid], "time": t, "score": off + te})
    return pd.DataFrame(rows)


def _mixed(client, sid):
    r = client.post(
        "/api/repeated/mixed_anova",
        json={
            "session_id": sid,
            "subject_col": "id",
            "within_col": "time",
            "between_col": "arm",
            "value_col": "score",
        },
    )
    assert r.status_code == 200, r.text
    return {e["term"]: e for e in r.json()["effects"]}, r.json()


def test_each_effect_uses_its_own_error_stratum(client, split_plot):
    sid = make_session(split_plot, "audit_mixed_strata")
    eff, _ = _mixed(client, sid)

    # Between-subjects: 3 arms, 9 subjects -> F(2, 6).
    assert eff["arm"]["df_num"] == 2
    assert eff["arm"]["df_den"] == 6, "arm must be judged against subject variation"
    # Within-subjects: 3 times -> F(2, 12) on the subject x time residual.
    assert eff["time"]["df_num"] == 2
    assert eff["time"]["df_den"] == 12
    assert eff["arm"]["error_stratum"] != eff["time"]["error_stratum"]


def test_interaction_keeps_its_own_degrees_of_freedom(client, split_plot):
    """Dropping the between main effect from the within model let the
    interaction absorb it and gain a degree of freedom."""
    sid = make_session(split_plot, "audit_mixed_inter")
    eff, _ = _mixed(client, sid)
    key = next(k for k in eff if "interaction" in k)
    assert eff[key]["df_num"] == 4, "(3-1) x (3-1)"


def test_matches_a_hand_computed_split_plot(client, split_plot):
    """The additive fixture has an exact answer: the arms differ only by the
    subject offsets, so there is no arm effect and no interaction."""
    sid = make_session(split_plot, "audit_mixed_exact")
    eff, _ = _mixed(client, sid)
    key = next(k for k in eff if "interaction" in k)

    # Time is perfectly reproduced in every subject -> zero residual, F -> inf.
    assert eff["time"]["p"] < 1e-9
    assert eff["arm"]["p"] > 0.05, "the arms differ only by subject offsets"
    assert eff[key]["p"] > 0.05


def test_rejects_a_subject_that_spans_two_arms(client, split_plot):
    bad = split_plot.copy()
    bad.loc[bad.index[0], "arm"] = "B"  # subject 1 now appears in A and B
    sid = make_session(bad, "audit_mixed_crossed")
    r = client.post(
        "/api/repeated/mixed_anova",
        json={
            "session_id": sid, "subject_col": "id", "within_col": "time",
            "between_col": "arm", "value_col": "score",
        },
    )
    assert r.status_code == 400
    assert "between-subjects" in r.json()["detail"]


def test_incomplete_subjects_are_dropped_and_said_so(client, split_plot):
    thin = split_plot[~((split_plot["id"] == 9) & (split_plot["time"] == "t3"))]
    sid = make_session(thin, "audit_mixed_incomplete")
    _, payload = _mixed(client, sid)
    assert any("do not" in w or "lack" in w for w in payload["warnings"])
    assert payload["summary"]["n_subjects"] == 8


def test_a_strongly_significant_within_effect_keeps_its_p(client, split_plot):
    """round(p, 6) printed a p of 3e-12 as 0.0."""
    noisy = split_plot.copy()
    rng = np.random.default_rng(1)
    noisy["score"] = noisy["score"] + rng.normal(0, 0.05, len(noisy))
    sid = make_session(noisy, "audit_mixed_smallp")
    eff, _ = _mixed(client, sid)
    assert 0 < eff["time"]["p"] < 1e-6


# ── K2: non-inferiority tail ──────────────────────────────────────────────────


@pytest.fixture()
def ni_continuous() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "arm": ["ref"] * 20 + ["test"] * 20,
        "y": np.r_[rng.normal(100, 8, 20), rng.normal(109, 8, 20)],
    })


def _ni(client, sid, **kw):
    body = {
        "session_id": sid, "outcome_col": "y", "group_col": "arm",
        "test_group": "test", "ref_group": "ref",
        "outcome_type": "continuous", "margin": 20.0, "bound": "upper",
        "alpha": 0.05,
    }
    body.update(kw)
    r = client.post("/api/stats/noninferiority", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_the_decision_and_the_p_value_agree(client, ni_continuous):
    """The whole defect in one assertion: the CI rule said non-inferior and
    the p-value said the opposite, on the same call."""
    sid = make_session(ni_continuous, "audit_ni_agree")
    out = _ni(client, sid)
    assert out["non_inferior"] is True
    assert out["p_noninferiority"] < out["alpha_one_sided"], (
        f"declared non-inferior at alpha={out['alpha_one_sided']} but "
        f"p={out['p_noninferiority']}"
    )


def test_the_invariant_holds_for_the_lower_bound_too(client, ni_continuous):
    """The report named the upper bound; the lower bound was reversed the
    same way."""
    sid = make_session(ni_continuous, "audit_ni_lower")
    for margin, expect in ((0.0, True), (30.0, False)):
        out = _ni(client, sid, bound="lower", margin=margin)
        assert out["non_inferior"] is expect, (margin, out["ci_low"])
        assert (out["p_noninferiority"] < out["alpha_one_sided"]) is expect


def test_a_failing_margin_gives_a_large_p(client, ni_continuous):
    sid = make_session(ni_continuous, "audit_ni_fail")
    out = _ni(client, sid, margin=2.0)
    assert out["non_inferior"] is False
    assert out["p_noninferiority"] > 0.05


def test_continuous_p_matches_a_one_sided_welch_t(client, ni_continuous):
    """The SE used to be recovered from a t-based interval by dividing by a
    normal quantile, mixing the two scales."""
    sid = make_session(ni_continuous, "audit_ni_welch")
    out = _ni(client, sid, margin=20.0)
    t = ni_continuous[ni_continuous["arm"] == "test"]["y"].to_numpy()
    r = ni_continuous[ni_continuous["arm"] == "ref"]["y"].to_numpy()
    se = np.sqrt(t.var(ddof=1) / len(t) + r.var(ddof=1) / len(r))
    num = (t.mean() - r.mean()) - 20.0
    dof = se ** 4 / (
        (t.var(ddof=1) / len(t)) ** 2 / (len(t) - 1)
        + (r.var(ddof=1) / len(r)) ** 2 / (len(r) - 1)
    )
    assert out["p_noninferiority"] == pytest.approx(
        float(sp.t.cdf(num / se, dof)), abs=1e-6
    )
    assert out["statistic_name"] == "t"


def test_binary_arm_keeps_the_invariant(client):
    df = pd.DataFrame({
        "arm": ["ref"] * 60 + ["test"] * 60,
        "y": [1] * 30 + [0] * 30 + [1] * 31 + [0] * 29,
    })
    sid = make_session(df, "audit_ni_binary")
    r = client.post("/api/stats/noninferiority", json={
        "session_id": sid, "outcome_col": "y", "group_col": "arm",
        "test_group": "test", "ref_group": "ref",
        "outcome_type": "binary", "effect": "RR", "margin": 3.0,
        "bound": "upper", "alpha": 0.05,
    })
    assert r.status_code == 200, r.text
    out = r.json()
    assert (out["p_noninferiority"] < out["alpha_one_sided"]) is out["non_inferior"]
    assert out["statistic_name"] == "z"
