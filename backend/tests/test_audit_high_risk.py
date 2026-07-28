"""High-risk findings from the 10x20 statistical audit.

Y1 — RM ANOVA read `res.epsilon` off an AnovaRM result that has no such
attribute, so no sphericity was ever reported and no correction ever applied.

Y2 / Y5 — one non-finite number made the whole response unserialisable and
the global handler turned that into a 400, discarding a valid omnibus ANOVA
or a valid CMH statistic along with it.

Y3 — Jonckheere-Terpstra placed non-numeric levels in alphabetical order
without saying so, and the test measures a trend across exactly that order.

Y4 — the proportion tests accepted a variable with three or more levels and
silently redefined "success" as its commonest value.

Y6 — the Wilcoxon effect size was built from SciPy's two-sided statistic,
min(W+, W-), which carries no direction.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


# ── Y1: sphericity ────────────────────────────────────────────────────────────


def _long(matrix: np.ndarray) -> pd.DataFrame:
    rows = []
    for s, subject in enumerate(matrix):
        for k, val in enumerate(subject):
            rows.append({"id": f"s{s}", "time": f"t{k + 1}", "y": float(val)})
    return pd.DataFrame(rows)


def _rm(client, sid):
    r = client.post(
        "/api/repeated/rm_anova",
        json={"session_id": sid, "subject_col": "id", "within_col": "time", "value_col": "y"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_sphericity_is_actually_computed(client):
    rng = np.random.default_rng(0)
    sid = make_session(_long(rng.normal(0, 1, (30, 4))), "audit_sph_ok")
    out = _rm(client, sid)
    assert out["sphericity"] is not None
    assert out["sphericity"]["applicable"] is True
    assert 0 < out["sphericity"]["gg"] <= 1
    assert any("Sphericity" in a["name"] for a in out["assumptions"])


def test_a_violation_is_reported_and_corrected(client):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (30, 4))
    x[:, 3] = x[:, 0] * 4 + rng.normal(0, 0.1, 30)  # wrecks sphericity
    x += np.array([0.0, 1.5, 3.0, 4.5])             # and a real time effect
    sid = make_session(_long(x), "audit_sph_bad")
    out = _rm(client, sid)

    assert out["sphericity"]["gg"] < 0.75
    assert out["sphericity"]["mauchly_p"] < 0.05
    sph = next(a for a in out["assumptions"] if "Sphericity" in a["name"])
    assert sph["met"] is False

    labels = {c["correction"] for c in out["corrected"]}
    assert labels == {"Greenhouse-Geisser", "Huynh-Feldt"}
    gg = next(c for c in out["corrected"] if c["correction"] == "Greenhouse-Geisser")
    # A correction shrinks the degrees of freedom, so for a real effect
    # (F > 1) the corrected p can only be larger — that is the whole point.
    assert out["F"] > 1
    assert gg["p"] >= out["p"]
    assert gg["df_num"] < 3


def test_two_levels_need_no_correction(client):
    rng = np.random.default_rng(2)
    sid = make_session(_long(rng.normal(0, 1, (20, 2))), "audit_sph_two")
    out = _rm(client, sid)
    assert out["sphericity"] is None
    assert out["corrected"] == []


# ── Y2: a constant post-hoc difference must not sink the response ─────────────


def test_a_constant_pair_keeps_the_omnibus_result(client):
    """Subjects all change by exactly the same amount between t1 and t2, so
    SciPy returns t = inf for that pair. The whole response used to 400."""
    rows = []
    for s in range(10):
        base = float(s)
        for k, bump in enumerate((0.0, 5.0, 5.0 + (s % 3))):
            rows.append({"id": f"s{s}", "time": f"t{k + 1}", "y": base + bump})
    sid = make_session(pd.DataFrame(rows), "audit_posthoc_inf")
    out = _rm(client, sid)

    assert out["significant"] is True
    assert np.isfinite(out["F"])
    frozen = [ph for ph in out["posthoc"] if ph["statistic"] is None]
    assert frozen, "the constant pair should be reported, not omitted"
    assert "no variance" in frozen[0]["note"]
    # Every other pair still carries a corrected p.
    live = [ph for ph in out["posthoc"] if ph["statistic"] is not None]
    assert live and all(ph.get("p_adj") is not None for ph in live)


# ── Y5: an infinite common odds ratio must not sink the CMH test ──────────────


def test_cmh_survives_a_zero_cell_stratum(client):
    df = pd.DataFrame({
        # Stratum B has a zero cell, which sends the pooled OR to infinity.
        "stratum": ["A"] * 40 + ["B"] * 20,
        "exposure": (["yes"] * 20 + ["no"] * 20) + (["yes"] * 10 + ["no"] * 10),
        "outcome": (
            ["event"] * 12 + ["none"] * 8 + ["event"] * 6 + ["none"] * 14
        ) + (["event"] * 10 + ["none"] * 10),
    })
    sid = make_session(df, "audit_cmh_inf")
    r = client.post("/api/categorical/mantel_haenszel", json={
        "session_id": sid, "row_col": "exposure", "col_col": "outcome",
        "strata_col": "stratum",
    })
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["p"] is not None and np.isfinite(out["p"])
    assert out["statistic"] is not None
    if out["effect_sizes"][0]["value"] is None:
        assert any("odds ratio is not finite" in str(w) for w in out["warnings"])


# ── Y3: the order the trend is measured across ────────────────────────────────


def _jt(client, sid, **kw):
    body = {"session_id": sid, "column": "y", "group_column": "dose"}
    body.update(kw)
    return client.post("/api/stats/jonckheere_terpstra", json=body)


@pytest.fixture()
def dose_frame() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows = []
    for level, centre in (("Low", 10.0), ("Medium", 14.0), ("High", 18.0)):
        for _ in range(12):
            rows.append({"dose": level, "y": float(rng.normal(centre, 1.5))})
    return pd.DataFrame(rows)


def test_known_ordinal_labels_are_ordered_by_meaning(client, dose_frame):
    """Alphabetically Low/Medium/High sorts to High < Low < Medium, which
    reverses the trend the test is meant to detect."""
    sid = make_session(dose_frame, "audit_jt_words")
    out = _jt(client, sid).json()
    assert out["level_order"] == ["Low", "Medium", "High"]
    assert out["summary"]["direction"] == "increasing"


def test_an_assumed_order_is_stated(client):
    rng = np.random.default_rng(5)
    rows = []
    for level, centre in (("alpha", 10.0), ("beta", 14.0), ("gamma", 18.0)):
        for _ in range(12):
            rows.append({"dose": level, "y": float(rng.normal(centre, 1.5))})
    sid = make_session(pd.DataFrame(rows), "audit_jt_assumed")
    out = _jt(client, sid).json()
    assert out["level_order_source"].startswith("alphabetical")
    assert any("alphabetical order" in str(w) for w in out["warnings"])
    assert " < ".join(out["level_order"]) in out["interpretation"]


def test_explicit_scores_still_win(client, dose_frame):
    sid = make_session(dose_frame, "audit_jt_scores")
    out = _jt(client, sid, scores=[3.0, 2.0, 1.0]).json()
    # Levels sort alphabetically first (High, Low, Medium), then by score.
    assert out["level_order"][0] != out["level_order"][-1]
    assert out["p"] is not None


def test_numeric_levels_are_not_warned_about(client):
    rng = np.random.default_rng(6)
    rows = []
    for level, centre in ((1, 10.0), (2, 14.0), (3, 18.0)):
        for _ in range(12):
            rows.append({"dose": level, "y": float(rng.normal(centre, 1.5))})
    sid = make_session(pd.DataFrame(rows), "audit_jt_numeric")
    out = _jt(client, sid).json()
    assert out["level_order_source"] == "numeric value"
    assert out["warnings"] == []


# ── Y4: these tests need a binary variable ────────────────────────────────────


def test_binomial_refuses_a_three_level_column(client):
    df = pd.DataFrame({"grade": ["I"] * 20 + ["II"] * 15 + ["III"] * 5})
    sid = make_session(df, "audit_bin_three")
    r = client.post("/api/categorical/binomial", json={"session_id": sid, "column": "grade"})
    assert r.status_code == 422
    assert "binary" in r.json()["detail"]


def test_one_proportion_refuses_a_three_level_column(client):
    df = pd.DataFrame({"grade": ["I"] * 20 + ["II"] * 15 + ["III"] * 5})
    sid = make_session(df, "audit_one_prop_three")
    r = client.post("/api/categorical/one_proportion",
                    json={"session_id": sid, "column": "grade"})
    assert r.status_code == 422


def test_a_genuine_two_level_column_still_works_and_names_the_event(client):
    df = pd.DataFrame({"resp": ["yes"] * 30 + ["no"] * 20})
    sid = make_session(df, "audit_bin_two")
    r = client.post("/api/categorical/binomial", json={"session_id": sid, "column": "resp"})
    assert r.status_code == 200, r.text
    assert r.json()["p"] is not None


def test_cochran_q_refuses_continuous_columns(client):
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "a": rng.normal(0, 1, 20), "b": rng.normal(0, 1, 20), "c": rng.normal(0, 1, 20),
    })
    sid = make_session(df, "audit_q_continuous")
    r = client.post("/api/categorical/cochran_q",
                    json={"session_id": sid, "columns": ["a", "b", "c"]})
    assert r.status_code == 422
    assert "binary" in r.json()["detail"]


def test_cochran_q_accepts_zero_one_columns(client):
    rng = np.random.default_rng(8)
    df = pd.DataFrame({c: rng.integers(0, 2, 20).astype(float) for c in "abc"})
    sid = make_session(df, "audit_q_binary")
    r = client.post("/api/categorical/cochran_q",
                    json={"session_id": sid, "columns": ["a", "b", "c"]})
    assert r.status_code == 200, r.text


# ── Y6: the effect size has to know which way the change went ─────────────────


@pytest.mark.parametrize("sign,expected", [(1, 1.0), (-1, -1.0)])
def test_wilcoxon_effect_size_carries_direction(client, sign, expected):
    """All-positive and all-negative differences both used to give r = -1."""
    n = 8
    pre = np.arange(1.0, n + 1)
    post = pre + sign * np.arange(1.0, n + 1)
    df = pd.DataFrame({"pre": pre, "post": post})
    sid = make_session(df, f"audit_wsr_{sign}")
    r = client.post("/api/repeated/wilcoxon_signed_rank",
                    json={"session_id": sid, "col1": "post", "col2": "pre"})
    assert r.status_code == 200, r.text
    assert r.json()["effect_sizes"][0]["value"] == pytest.approx(expected)


def test_mcnemar_cells_are_named_by_what_they_are(client):
    """crosstab sorts levels ascending, so cell [0][0] is BOTH NEGATIVE — it
    was labelled "a (both +)" and every other label followed it, inverting the
    reported direction and the discordant odds ratio."""
    # 12 subjects go No -> Yes, 3 go Yes -> No.
    df = pd.DataFrame({
        "before": ["No"] * 12 + ["Yes"] * 3 + ["Yes"] * 10 + ["No"] * 5,
        "after": ["Yes"] * 12 + ["No"] * 3 + ["Yes"] * 10 + ["No"] * 5,
    })
    sid = make_session(df, "audit_mcnemar_dir")
    r = client.post("/api/categorical/mcnemar",
                    json={"session_id": sid, "col1": "before", "col2": "after"})
    assert r.status_code == 200, r.text
    cells = r.json()["cells"]
    assert cells["negative_to_positive"] == 12
    assert cells["positive_to_negative"] == 3
    assert cells["both_positive"] == 10
    assert cells["both_negative"] == 5
    # The odds ratio must follow the same direction: more went up than down.
    assert r.json()["effect_sizes"][0]["value"] == pytest.approx(3 / 12, abs=1e-6)
