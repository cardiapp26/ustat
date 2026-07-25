"""Coverage tests for POST /api/categorical/cochran_armitage (routers/categorical.py).

The endpoint had no test reference. The fixtures are built with *exact* cell
counts, so the Cochran-Armitage Z is a closed-form number that can be asserted
to four decimal places rather than merely bounded:

    dose 0..3, n_k = 80 each, successes = 8 / 20 / 36 / 52
    N = 320, total successes = 116, p̂ = 0.3625, scores w = 0,1,2,3, w̄ = 1.5
    numerator = Σ w_k s_k − p̂ Σ w_k n_k = 248 − 174           = 74
    variance  = p̂(1−p̂) Σ n_k (w_k − w̄)² = 0.23109375 × 400   = 92.4375
    Z         = 74 / √92.4375                                  = 7.69702…

Any sign flip, wrong pooled proportion, or wrong variance term moves that
number, so the assertion is a real regression net rather than a smoke test.
"""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session

PREFIX = "/api/categorical"

EXPECTED_Z = 7.697  # see module docstring
GROUP_N = 80
SUCCESSES_INCREASING = [8, 20, 36, 52]


def _dose_frame(successes, n_per_group=GROUP_N, doses=(0, 1, 2, 3)):
    """Exact-count long frame: `successes[k]` events among `n_per_group` at dose k."""
    dose, event = [], []
    for d, s in zip(doses, successes):
        dose.extend([d] * n_per_group)
        event.extend([1] * s + [0] * (n_per_group - s))
    return pd.DataFrame({"dose": dose, "event": event})


@pytest.fixture(scope="module")
def trend_df():
    df = _dose_frame(SUCCESSES_INCREASING)
    # Extra columns for the error/variant paths, aligned to the same 320 rows.
    rng = np.random.default_rng(20260725)
    df["flat_event"] = ([1] * 28 + [0] * (GROUP_N - 28)) * 4
    df["all_event"] = 1
    df["binary_label"] = np.where(df["event"] == 1, "yes", "no")
    df["two_level_dose"] = np.where(df["dose"] <= 1, 0, 1)
    df["noise"] = rng.normal(0, 1, len(df))
    return df


@pytest.fixture(scope="module")
def sid(trend_df):
    return make_session(trend_df, "ca_trend_main")


def _post(client, **payload):
    return client.post(f"{PREFIX}/cochran_armitage", json=payload)


# ── Monotone increasing trend ────────────────────────────────────────────────


def test_increasing_trend_matches_the_closed_form_statistic(client, sid):
    r = _post(client, session_id=sid, ordinal_col="dose", event_col="event")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["test"] == "Cochran-Armitage trend test"

    # Closed-form Z for the exact cell counts (see module docstring).
    assert d["z"] == pytest.approx(EXPECTED_Z, abs=5e-4)
    assert d["statistic"] == d["z"]
    assert d["z"] > 0, "increasing proportions must give a positive Z"
    assert d["p"] < 1e-10
    assert d["significant"] is True
    assert d["summary"]["direction"] == "increasing"


def test_increasing_trend_summary_counts(client, sid):
    d = _post(client, session_id=sid, ordinal_col="dose",
              event_col="event").json()
    s = d["summary"]
    assert s["n"] == 320
    assert s["n_successes"] == sum(SUCCESSES_INCREASING)
    assert s["pooled_proportion"] == pytest.approx(116 / 320, abs=1e-4)
    assert s["n_levels"] == 4
    assert s["scores"] == [0.0, 1.0, 2.0, 3.0]

    levels = s["levels"]
    assert [lv["level"] for lv in levels] == ["0", "1", "2", "3"]
    assert [lv["successes"] for lv in levels] == SUCCESSES_INCREASING
    assert all(lv["n"] == GROUP_N for lv in levels)
    # Ordered levels must come back with monotonically rising proportions.
    props = [lv["proportion"] for lv in levels]
    assert props == sorted(props)
    for lv in levels:
        assert lv["proportion"] == pytest.approx(lv["successes"] / lv["n"],
                                                 abs=1e-4)
    assert "increasing" in d["result_text"]


# ── Direction, flatness, and score invariances ───────────────────────────────


def test_reversed_proportions_flip_the_sign_only(client, trend_df):
    """Reversing the dose ordering must negate Z and leave |Z| and p intact."""
    up = _post(client, session_id=make_session(trend_df, "ca_trend_main"),
               ordinal_col="dose", event_col="event").json()

    down_sid = make_session(_dose_frame(SUCCESSES_INCREASING[::-1]),
                            "ca_trend_down")
    down = _post(client, session_id=down_sid, ordinal_col="dose",
                 event_col="event").json()

    assert down["z"] == pytest.approx(-up["z"], abs=1e-6)
    assert down["z"] == pytest.approx(-EXPECTED_Z, abs=5e-4)
    assert down["p"] == pytest.approx(up["p"], abs=1e-12)
    assert down["summary"]["direction"] == "decreasing"
    assert down["significant"] is True


def test_flat_proportions_give_no_trend(client, sid):
    """Identical proportions at every dose: Z is exactly 0 and p is 1."""
    r = _post(client, session_id=sid, ordinal_col="dose",
              event_col="flat_event")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["z"] == pytest.approx(0.0, abs=1e-9)
    assert d["p"] == pytest.approx(1.0, abs=1e-9)
    assert d["significant"] is False
    assert d["summary"]["direction"] == "flat"
    assert "No significant" in d["interpretation"]


def test_z_is_invariant_to_shifting_and_scaling_the_scores(client, sid):
    """w -> a·w + b (a > 0) is a re-parameterisation, not a different test.

    This pins down the centring (w − w̄) in the variance and the use of the
    pooled proportion in the numerator; dropping either breaks the invariance.
    """
    base = _post(client, session_id=sid, ordinal_col="dose",
                 event_col="event").json()
    shifted = _post(client, session_id=sid, ordinal_col="dose",
                    event_col="event", scores=[10, 11, 12, 13]).json()
    scaled = _post(client, session_id=sid, ordinal_col="dose",
                   event_col="event", scores=[0, 2, 4, 6]).json()
    explicit = _post(client, session_id=sid, ordinal_col="dose",
                     event_col="event", scores=[0, 1, 2, 3]).json()

    assert shifted["z"] == pytest.approx(base["z"], abs=1e-6)
    assert scaled["z"] == pytest.approx(base["z"], abs=1e-6)
    assert explicit["z"] == pytest.approx(base["z"], abs=1e-9)
    assert shifted["summary"]["scores"] == [10.0, 11.0, 12.0, 13.0]


def test_unequally_spaced_scores_change_the_statistic(client, sid):
    """Spacing must actually enter the statistic, not just be echoed back."""
    base = _post(client, session_id=sid, ordinal_col="dose",
                 event_col="event").json()
    spaced = _post(client, session_id=sid, ordinal_col="dose",
                   event_col="event", scores=[0, 1, 2, 20]).json()
    assert spaced["z"] != pytest.approx(base["z"], abs=1e-3)
    assert spaced["z"] > 0
    assert spaced["summary"]["scores"] == [0.0, 1.0, 2.0, 20.0]


def test_alpha_controls_the_significance_flag_only(client, sid):
    """A tiny alpha must flip `significant` without touching Z or p."""
    strict = _post(client, session_id=sid, ordinal_col="dose",
                   event_col="flat_event", alpha=0.99).json()
    lenient = _post(client, session_id=sid, ordinal_col="dose",
                    event_col="flat_event", alpha=0.05).json()
    assert strict["z"] == lenient["z"]
    assert strict["p"] == lenient["p"]
    # p == 1.0 exactly, so even alpha = 0.99 must not call it significant.
    assert strict["significant"] is False and lenient["significant"] is False

    sig = _post(client, session_id=sid, ordinal_col="dose",
                event_col="event", alpha=1e-12).json()
    assert sig["p"] < 1e-12
    assert sig["significant"] is True


def test_explicit_success_value_on_a_labelled_outcome(client, sid):
    """Choosing the success label must reproduce the numeric-coded result."""
    numeric = _post(client, session_id=sid, ordinal_col="dose",
                    event_col="event").json()
    labelled = _post(client, session_id=sid, ordinal_col="dose",
                     event_col="binary_label", success_value="yes").json()
    assert labelled["z"] == pytest.approx(numeric["z"], abs=1e-9)
    assert labelled["summary"]["n_successes"] == numeric["summary"]["n_successes"]

    # Choosing the other label must give the mirror-image trend.
    inverted = _post(client, session_id=sid, ordinal_col="dose",
                     event_col="binary_label", success_value="no").json()
    assert inverted["z"] == pytest.approx(-numeric["z"], abs=1e-9)
    assert inverted["summary"]["direction"] == "decreasing"


def test_r_code_carries_the_observed_counts(client, sid):
    d = _post(client, session_id=sid, ordinal_col="dose",
              event_col="event").json()
    assert "prop.trend.test" in d["r_code"]
    assert "(8, 20, 36, 52)" in d["r_code"]
    assert "(80, 80, 80, 80)" in d["r_code"]
    export = {row[0]: row[1] for row in d["export_rows"][1:]}
    assert export["Z"] == d["z"]
    assert export["Levels"] == 4
    assert export["Total n"] == 320
    assert export["Direction"] == "increasing"


# ── Error paths ──────────────────────────────────────────────────────────────


def test_unknown_column_400(client, sid):
    r = _post(client, session_id=sid, ordinal_col="ghost", event_col="event")
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "ghost" in detail


def test_unknown_event_column_400(client, sid):
    r = _post(client, session_id=sid, ordinal_col="dose", event_col="ghost")
    assert r.status_code == 400, r.text
    assert "ghost" in r.json()["detail"]


def test_two_groups_is_rejected_422(client, sid):
    r = _post(client, session_id=sid, ordinal_col="two_level_dose",
              event_col="event")
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "at least 3 ordered groups" in detail


def test_constant_outcome_422(client, sid):
    r = _post(client, session_id=sid, ordinal_col="dose", event_col="all_event")
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "constant" in detail


def test_non_binary_outcome_422(client, sid):
    """A >2-level outcome with no success_value cannot be dichotomised."""
    r = _post(client, session_id=sid, ordinal_col="dose", event_col="noise")
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "binary" in detail


def test_score_length_mismatch_422(client, sid):
    r = _post(client, session_id=sid, ordinal_col="dose", event_col="event",
              scores=[0, 1])
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "match the number of levels" in detail


def test_too_few_rows_422(client, trend_df):
    tiny = make_session(trend_df.head(4), "ca_trend_tiny")
    r = _post(client, session_id=tiny, ordinal_col="dose", event_col="event")
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.strip()
    assert "at least 5" in detail


def test_unknown_session_404(client):
    r = _post(client, session_id="no_such_session", ordinal_col="dose",
              event_col="event")
    assert r.status_code == 404, r.text
    assert r.json()["detail"].strip()


# ── Level ordering for non-numeric labels ────────────────────────────────────


def _labelled_frame():
    """Real dose-response with word labels whose alphabetical order is wrong.

    Low 10% → Medium 30% → High 60%. Alphabetically that reads High, Low,
    Medium, which reverses the exposure and flips the sign of the trend.
    """
    rows = []
    for level, k in (("Low", 8), ("Medium", 24), ("High", 48)):
        rows += [{"dose": level, "evt": 1}] * k
        rows += [{"dose": level, "evt": 0}] * (GROUP_N - k)
    return pd.DataFrame(rows)


def test_alphabetical_fallback_is_flagged_not_silent(client):
    sid_lbl = make_session(_labelled_frame(), "ca_labels_warn")
    r = _post(client, session_id=sid_lbl, ordinal_col="dose", event_col="evt")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["level_order_source"] == "alphabetical (assumed)"
    assert body["warnings"], "an assumed ordering must be surfaced to the caller"
    assert "level_order" in " ".join(body["warnings"])
    # The whole point: the assumed order genuinely inverts this dataset.
    assert body["summary"]["direction"] == "decreasing"


def test_explicit_level_order_recovers_the_true_trend(client):
    sid_lbl = make_session(_labelled_frame(), "ca_labels_ordered")
    r = _post(client, session_id=sid_lbl, ordinal_col="dose", event_col="evt",
              level_order=["Low", "Medium", "High"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["level_order_source"] == "caller-supplied level_order"
    assert body["warnings"] == []
    assert [row["level"] for row in body["summary"]["levels"]] == ["Low", "Medium", "High"]
    assert body["z"] > 0
    assert body["summary"]["direction"] == "increasing"


def test_alphabetical_and_explicit_orders_disagree_in_sign(client):
    """Guard the premise of the two tests above: the orderings really conflict."""
    sid_lbl = make_session(_labelled_frame(), "ca_labels_sign")
    assumed = _post(client, session_id=sid_lbl, ordinal_col="dose", event_col="evt").json()
    stated = _post(client, session_id=sid_lbl, ordinal_col="dose", event_col="evt",
                   level_order=["Low", "Medium", "High"]).json()
    assert assumed["z"] < 0 < stated["z"], (
        f"expected opposite signs, got {assumed['z']} and {stated['z']}"
    )
    # Both are 'significant', which is exactly why the silent version was unsafe.
    assert assumed["significant"] and stated["significant"]


def test_numeric_levels_need_no_warning(client, sid):
    r = _post(client, session_id=sid, ordinal_col="dose", event_col="event")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["level_order_source"] == "numeric value"
    assert body["warnings"] == []


def test_level_order_must_match_the_data_exactly(client):
    sid_lbl = make_session(_labelled_frame(), "ca_labels_bad")
    r = _post(client, session_id=sid_lbl, ordinal_col="dose", event_col="evt",
              level_order=["Low", "Medium"])
    assert r.status_code == 422, r.text
    assert "level_order" in r.json()["detail"]
