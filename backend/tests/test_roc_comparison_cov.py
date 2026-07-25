"""Coverage tests for the ROC comparison endpoints in routers/stats/nonparametric.py.

Covers POST /api/stats/roc_compare, /roc_multi_compare and /roc_combined, none
of which had any test reference. The assertions target statistical correctness
(DeLong invariants, AUC identities, direction handling) rather than HTTP 200 —
a wrong answer, not just a crash, has to fail these.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session

SEED = 20260725
PREFIX = "/api/stats"


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def roc_df():
    """Binary outcome plus predictors whose AUCs are known by construction.

    perfect     : ranges never overlap between classes  -> AUC exactly 1.0
    noise       : independent of the outcome            -> AUC ~ 0.5
    good        : shifted normal, clear but imperfect   -> AUC ~ 0.85
    good_copy   : bit-identical copy of `good`          -> DeLong ΔAUC == 0
    good_neg    : exact negation of `good`              -> AUC == 1 - AUC(good)
    weak        : small shift                           -> AUC ~ 0.65
    """
    rng = np.random.default_rng(SEED)
    n_pos, n_neg = 120, 120
    n = n_pos + n_neg
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])

    perfect = np.where(y == 1, rng.uniform(1.0, 2.0, n), rng.uniform(-2.0, -1.0, n))
    noise = rng.normal(0.0, 1.0, n)
    good = rng.normal(0.0, 1.0, n) + 1.5 * y
    weak = rng.normal(0.0, 1.0, n) + 0.55 * y

    return pd.DataFrame(
        {
            "outcome": y,
            "perfect": perfect,
            "noise": noise,
            "good": good,
            "good_copy": good.copy(),
            "good_neg": -good,
            "weak": weak,
            # Non-binary and degenerate columns for the error paths.
            "three_level": np.tile([0, 1, 2], n // 3),
            "constant": np.ones(n),
        }
    )


@pytest.fixture(scope="module")
def sid(roc_df):
    return make_session(roc_df, "roc_cmp_main")


@pytest.fixture(scope="module")
def combined_df():
    """y is a deterministic threshold on x1 + x2.

    Neither predictor alone can separate the classes (each carries about
    AUC 0.76), but their sum separates them perfectly, so a logistic fusion
    that actually uses both columns must land near AUC 1.0.
    """
    rng = np.random.default_rng(SEED + 7)
    n = 250
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    y = (x1 + x2 > 0).astype(int)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "y": y,
            "z1": rng.normal(0.0, 1.0, n),
            "z2": rng.normal(0.0, 1.0, n),
            "grp": np.where(rng.uniform(0, 1, n) < 0.5, "A", "B"),
            "y3": np.tile([0, 1, 2], (n // 3) + 1)[:n],
        }
    )


@pytest.fixture(scope="module")
def sid_comb(combined_df):
    return make_session(combined_df, "roc_cmp_combined")


def _post(client, path, **payload):
    return client.post(f"{PREFIX}/{path}", json=payload)


# ── /roc_compare — DeLong invariants ─────────────────────────────────────────


def test_roc_compare_identical_predictors_is_a_null_result(client, sid):
    """The strongest DeLong invariant: comparing a score with its own copy.

    ΔAUC must be exactly 0 and the two-sided p must be 1. Any sign error or
    mis-paired covariance term in _delong_compare breaks this immediately.
    """
    r = _post(client, "roc_compare", session_id=sid,
              score_column_1="good", score_column_2="good_copy",
              outcome_column="outcome")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["auc_1"] == d["auc_2"]
    assert d["difference"] == pytest.approx(0.0, abs=1e-9)
    assert d["z"] == pytest.approx(0.0, abs=1e-6)
    assert d["p"] == pytest.approx(1.0, abs=1e-6)
    assert d["significant"] is False
    # A null CI must straddle zero.
    assert d["ci_diff_low"] <= 0.0 <= d["ci_diff_high"]
    assert "No significant difference" in d["interpretation"]
    assert d["n"] == 240


def test_roc_compare_perfect_vs_noise(client, sid):
    """A perfectly separating score against pure noise must be a clear win."""
    r = _post(client, "roc_compare", session_id=sid,
              score_column_1="perfect", score_column_2="noise",
              outcome_column="outcome", direction_2="higher")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["auc_1"] == pytest.approx(1.0, abs=1e-9)
    assert 0.40 <= d["auc_2"] <= 0.60
    assert d["difference"] > 0.35
    assert d["p"] < 1e-6
    assert d["significant"] is True
    assert d["ci_diff_low"] > 0.0
    # The winner named in the prose must be the better score.
    assert d["interpretation"].startswith("perfect ")
    # A perfect AUC's CI cannot exceed 1.
    assert d["ci_1_high"] <= 1.0


def test_roc_compare_negated_score_gives_complementary_auc(client, sid):
    """AUC(-s) == 1 - AUC(s) when the auto-flip is disabled on both sides."""
    r = _post(client, "roc_compare", session_id=sid,
              score_column_1="good", score_column_2="good_neg",
              outcome_column="outcome",
              direction_1="higher", direction_2="higher")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["direction_1_flipped"] is False
    assert d["direction_2_flipped"] is False
    assert d["auc_1"] + d["auc_2"] == pytest.approx(1.0, abs=2e-4)
    assert d["auc_1"] > 0.8
    assert d["difference"] == pytest.approx(d["auc_1"] - d["auc_2"], abs=2e-4)


def test_roc_compare_auto_direction_flips_the_negated_score(client, sid):
    """With direction='auto' the negated score is flipped back to the original."""
    r = _post(client, "roc_compare", session_id=sid,
              score_column_1="good", score_column_2="good_neg",
              outcome_column="outcome")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["direction_1_flipped"] is False
    assert d["direction_1_used"] == "higher"
    assert d["direction_2_flipped"] is True
    assert d["direction_2_used"] == "lower"
    # After flipping, -(-good) == good, so the two curves coincide again.
    assert d["auc_1"] == pytest.approx(d["auc_2"], abs=1e-9)
    assert d["p"] == pytest.approx(1.0, abs=1e-6)


def test_roc_compare_ordering_only_changes_the_sign(client, sid):
    """Swapping the two scores must negate ΔAUC and leave p/|z| untouched."""
    a = _post(client, "roc_compare", session_id=sid, score_column_1="good",
              score_column_2="weak", outcome_column="outcome").json()
    b = _post(client, "roc_compare", session_id=sid, score_column_1="weak",
              score_column_2="good", outcome_column="outcome").json()
    assert a["difference"] == pytest.approx(-b["difference"], abs=1e-9)
    assert a["p"] == pytest.approx(b["p"], abs=1e-9)
    assert abs(a["z"]) == pytest.approx(abs(b["z"]), abs=1e-4)
    assert a["auc_1"] == b["auc_2"]


def test_roc_compare_curves_are_monotone_and_anchored(client, sid):
    r = _post(client, "roc_compare", session_id=sid,
              score_column_1="good", score_column_2="weak",
              outcome_column="outcome")
    assert r.status_code == 200, r.text
    d = r.json()
    for key in ("curve_1", "curve_2"):
        pts = d[key]
        assert len(pts) >= 2
        fpr = [p["fpr"] for p in pts]
        tpr = [p["tpr"] for p in pts]
        assert fpr == sorted(fpr)
        assert tpr == sorted(tpr)
        assert fpr[0] == pytest.approx(0.0)
        assert tpr[0] == pytest.approx(0.0)
        assert fpr[-1] == pytest.approx(1.0)
        assert tpr[-1] == pytest.approx(1.0)


# ── /roc_compare — error paths ───────────────────────────────────────────────


def test_roc_compare_unknown_column_400(client, sid):
    r = _post(client, "roc_compare", session_id=sid, score_column_1="nope",
              score_column_2="good", outcome_column="outcome")
    assert r.status_code == 400, r.text
    assert isinstance(r.json()["detail"], str) and r.json()["detail"].strip()
    assert "nope" in r.json()["detail"]


def test_roc_compare_non_binary_outcome_400(client, sid):
    r = _post(client, "roc_compare", session_id=sid, score_column_1="good",
              score_column_2="weak", outcome_column="three_level")
    assert r.status_code == 400, r.text
    assert "2 unique values" in r.json()["detail"]


def test_roc_compare_constant_score_400(client, sid):
    r = _post(client, "roc_compare", session_id=sid, score_column_1="constant",
              score_column_2="good", outcome_column="outcome")
    assert r.status_code == 400, r.text
    assert "constant" in r.json()["detail"]


def test_roc_compare_bad_direction_422(client, sid):
    r = _post(client, "roc_compare", session_id=sid, score_column_1="good",
              score_column_2="weak", outcome_column="outcome",
              direction_1="sideways")
    assert r.status_code == 422, r.text
    assert isinstance(r.json()["detail"], str) and r.json()["detail"].strip()


def test_roc_compare_unknown_session_404(client):
    r = _post(client, "roc_compare", session_id="no_such_session",
              score_column_1="good", score_column_2="weak",
              outcome_column="outcome")
    assert r.status_code == 404, r.text
    assert r.json()["detail"].strip()


# ── /roc_multi_compare ───────────────────────────────────────────────────────


def test_roc_multi_compare_pairwise_structure_and_null_pair(client, sid):
    r = _post(client, "roc_multi_compare", session_id=sid,
              score_columns=["good", "good_copy", "noise"],
              outcome_column="outcome",
              directions=["higher", "higher", "higher"])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["test"] == "ROC Multi-Curve DeLong"
    assert d["n"] == 240 and d["n_positive"] == 120 and d["n_negative"] == 120
    assert d["n_pairs"] == 3 and len(d["pairs"]) == 3
    assert [s["name"] for s in d["scores"]] == ["good", "good_copy", "noise"]

    by_name = {s["name"]: s for s in d["scores"]}
    # Identical columns must produce identical AUC and identical SE.
    assert by_name["good"]["auc"] == by_name["good_copy"]["auc"]
    assert by_name["good"]["se"] == by_name["good_copy"]["se"]
    for s in d["scores"]:
        assert s["ci_low"] <= s["auc"] <= s["ci_high"]
        assert 0.0 <= s["ci_low"] and s["ci_high"] <= 1.0

    pairs = {(p["a"], p["b"]): p for p in d["pairs"]}
    null_pair = pairs[("good", "good_copy")]
    assert null_pair["delta_auc"] == pytest.approx(0.0, abs=1e-9)
    assert null_pair["p_raw"] == pytest.approx(1.0, abs=1e-6)
    assert null_pair["p_adj"] == pytest.approx(1.0, abs=1e-6)
    assert null_pair["significant"] is False

    real_pair = pairs[("good", "noise")]
    assert real_pair["delta_auc"] > 0.25
    assert real_pair["p_raw"] < 1e-6
    assert real_pair["significant"] is True
    assert real_pair["ci_low"] > 0.0


def test_roc_multi_compare_auc_matches_roc_compare(client, sid):
    """Cross-endpoint consistency: the same DeLong AUC from both routes."""
    multi = _post(client, "roc_multi_compare", session_id=sid,
                  score_columns=["good", "weak"], outcome_column="outcome",
                  directions=["higher", "higher"]).json()
    pair = _post(client, "roc_compare", session_id=sid, score_column_1="good",
                 score_column_2="weak", outcome_column="outcome",
                 direction_1="higher", direction_2="higher").json()
    by_name = {s["name"]: s for s in multi["scores"]}
    assert by_name["good"]["auc"] == pytest.approx(pair["auc_1"], abs=1e-4)
    assert by_name["weak"]["auc"] == pytest.approx(pair["auc_2"], abs=1e-4)
    assert multi["pairs"][0]["p_raw"] == pytest.approx(pair["p"], abs=1e-5)


def test_roc_multi_compare_holm_and_bonferroni_adjustments(client, sid):
    payload = dict(session_id=sid,
                   score_columns=["good", "weak", "noise"],
                   outcome_column="outcome",
                   directions=["higher", "higher", "higher"])
    holm = _post(client, "roc_multi_compare", **payload, p_adjust="holm").json()
    bonf = _post(client, "roc_multi_compare", **payload, p_adjust="bonferroni").json()
    none = _post(client, "roc_multi_compare", **payload, p_adjust="none").json()

    assert holm["p_adjust"] == "holm"
    for h, b, n in zip(holm["pairs"], bonf["pairs"], none["pairs"]):
        # Same raw p regardless of adjustment method.
        assert h["p_raw"] == b["p_raw"] == n["p_raw"]
        # An adjustment never makes a p-value smaller, and Holm is never
        # more conservative than Bonferroni.
        assert h["p_adj"] >= h["p_raw"] - 1e-9
        assert b["p_adj"] == pytest.approx(min(1.0, b["p_raw"] * 3), abs=1e-5)
        assert h["p_adj"] <= b["p_adj"] + 1e-9
        assert n["p_adj"] == pytest.approx(n["p_raw"], abs=1e-9)


def test_roc_multi_compare_auto_flip_reported_per_score(client, sid):
    r = _post(client, "roc_multi_compare", session_id=sid,
              score_columns=["good", "good_neg"], outcome_column="outcome")
    assert r.status_code == 200, r.text
    d = r.json()
    by_name = {s["name"]: s for s in d["scores"]}
    assert by_name["good"]["direction_flipped"] is False
    assert by_name["good_neg"]["direction_flipped"] is True
    assert by_name["good_neg"]["direction_used"] == "lower"
    # After the flip the two are the same score, so the pair is null.
    assert d["pairs"][0]["delta_auc"] == pytest.approx(0.0, abs=1e-9)
    assert d["pairs"][0]["p_raw"] == pytest.approx(1.0, abs=1e-6)


def test_roc_multi_compare_too_few_scores_422(client, sid):
    r = _post(client, "roc_multi_compare", session_id=sid,
              score_columns=["good"], outcome_column="outcome")
    assert r.status_code == 422, r.text
    assert "at least 2" in r.json()["detail"]


def test_roc_multi_compare_duplicate_scores_422(client, sid):
    r = _post(client, "roc_multi_compare", session_id=sid,
              score_columns=["good", "good"], outcome_column="outcome")
    assert r.status_code == 422, r.text
    assert "Duplicate" in r.json()["detail"]


def test_roc_multi_compare_unknown_column_400(client, sid):
    r = _post(client, "roc_multi_compare", session_id=sid,
              score_columns=["good", "ghost"], outcome_column="outcome")
    assert r.status_code == 400, r.text
    assert "ghost" in r.json()["detail"]


def test_roc_multi_compare_non_binary_outcome_422(client, sid):
    r = _post(client, "roc_multi_compare", session_id=sid,
              score_columns=["good", "weak"], outcome_column="three_level")
    assert r.status_code == 422, r.text
    assert "binary" in r.json()["detail"]


def test_roc_multi_compare_bad_p_adjust_422(client, sid):
    r = _post(client, "roc_multi_compare", session_id=sid,
              score_columns=["good", "weak"], outcome_column="outcome",
              p_adjust="fdr")
    assert r.status_code == 422, r.text
    assert isinstance(r.json()["detail"], str) and r.json()["detail"].strip()


# ── /roc_combined ────────────────────────────────────────────────────────────


def test_roc_combined_fuses_both_predictors(client, sid_comb):
    """y = 1[x1 + x2 > 0]: the pair must beat either predictor on its own."""
    both = _post(client, "roc_combined", session_id=sid_comb,
                 predictor_columns=["x1", "x2"], outcome_column="y",
                 model_name="Two-marker model")
    assert both.status_code == 200, both.text
    d = both.json()
    assert d["test"] == "ROC Analysis (Combined Model)"
    assert d["model_name"] == "Two-marker model"
    assert d["predictors"] == ["x1", "x2"]
    assert d["n"] == 250
    assert d["n_positive"] + d["n_negative"] == 250
    # Cross-validated, so not exactly 1.0, but the fusion is near-perfect.
    assert d["auc"] > 0.95
    assert d["direction_flipped"] is False

    solo = _post(client, "roc_combined", session_id=sid_comb,
                 predictor_columns=["x1"], outcome_column="y").json()
    assert solo["auc"] < 0.85
    assert d["auc"] > solo["auc"] + 0.10


def test_roc_combined_optimal_cutoff_is_internally_consistent(client, sid_comb):
    r = _post(client, "roc_combined", session_id=sid_comb,
              predictor_columns=["x1", "x2"], outcome_column="y")
    assert r.status_code == 200, r.text
    d = r.json()
    o = d["optimal"]
    assert {"cutoff", "tp", "tn", "fp", "fn", "sensitivity", "specificity",
            "ppv", "npv", "accuracy", "youden_j"} <= set(o)
    assert o["tp"] + o["fn"] == d["n_positive"]
    assert o["tn"] + o["fp"] == d["n_negative"]
    assert o["sensitivity"] == pytest.approx(o["tp"] / (o["tp"] + o["fn"]), abs=1e-3)
    assert o["specificity"] == pytest.approx(o["tn"] / (o["tn"] + o["fp"]), abs=1e-3)
    assert o["youden_j"] == pytest.approx(
        o["sensitivity"] + o["specificity"] - 1, abs=2e-4)
    # Youden's J is maximised at the optimal cutoff, so it must beat chance here.
    assert o["youden_j"] > 0.8
    fpr = [p["fpr"] for p in d["curve"]]
    assert fpr == sorted(fpr)


def test_roc_combined_noise_predictors_stay_near_chance(client, sid_comb):
    """Pure noise must not produce discrimination — this catches label leakage.

    The endpoint auto-flips a sub-0.5 AUC, so the value is bounded below by 0.5
    by construction; only the upper bound is informative.
    """
    r = _post(client, "roc_combined", session_id=sid_comb,
              predictor_columns=["z1", "z2"], outcome_column="y")
    assert r.status_code == 200, r.text
    d = r.json()
    assert 0.5 <= d["auc"] <= 0.72


def test_roc_combined_accepts_a_categorical_predictor(client, sid_comb):
    r = _post(client, "roc_combined", session_id=sid_comb,
              predictor_columns=["x1", "x2", "grp"], outcome_column="y")
    assert r.status_code == 200, r.text
    assert r.json()["auc"] > 0.95


def test_roc_combined_unknown_outcome_400(client, sid_comb):
    r = _post(client, "roc_combined", session_id=sid_comb,
              predictor_columns=["x1"], outcome_column="nope")
    assert r.status_code == 400, r.text
    assert "nope" in r.json()["detail"]


def test_roc_combined_unknown_predictor_400(client, sid_comb):
    r = _post(client, "roc_combined", session_id=sid_comb,
              predictor_columns=["x1", "ghost"], outcome_column="y")
    assert r.status_code == 400, r.text
    assert "ghost" in r.json()["detail"]


def test_roc_combined_non_binary_outcome_400(client, sid_comb):
    r = _post(client, "roc_combined", session_id=sid_comb,
              predictor_columns=["x1", "x2"], outcome_column="y3")
    assert r.status_code == 400, r.text
    assert "exactly 0 and 1" in r.json()["detail"]


def test_roc_combined_too_few_rows_400(client, combined_df):
    tiny = make_session(combined_df.head(15), "roc_cmp_tiny")
    r = _post(client, "roc_combined", session_id=tiny,
              predictor_columns=["x1", "x2"], outcome_column="y")
    assert r.status_code == 400, r.text
    assert "20" in r.json()["detail"]
