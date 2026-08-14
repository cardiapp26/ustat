from pathlib import Path

import pandas as pd

from services import store
from conftest import make_session
from services.dirty_value_guard import coerce_numeric, flag_sentinels, mask_sentinels


COHORT = Path(__file__).resolve().parents[2] / "qa" / "cohort_test.csv"


def _upload_cohort(client):
    with COHORT.open("rb") as f:
        r = client.post("/api/upload/", files={"file": ("cohort_test.csv", f, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def test_dirty_guard_coerces_comma_decimal_and_flags_bmi_999():
    s = pd.Series(["25,9", "30.6", "999", None])
    numeric = coerce_numeric(s)
    assert numeric.tolist()[:3] == [25.9, 30.6, 999.0]

    mask = flag_sentinels(s, max_plausible=100)
    assert mask.tolist() == [False, False, True, False]
    assert pd.isna(mask_sentinels(s, max_plausible=100).iloc[2])


def test_a_sparse_binary_column_is_not_flagged_sentinel():
    """Reported: a 'Hematom' column of mostly 0 with two genuine 1s had both
    1s counted as missing in the header badge. With fewer than 1% of rows
    nonzero, the 99th percentile of the body is itself 0, so the old
    robust_high = q99 * 1.5 evaluated to 0 and every real 1 cleared that
    'threshold'."""
    s = pd.Series([0.0] * 380 + [1.0, 1.0])
    mask = flag_sentinels(s, max_plausible=None)
    assert mask.sum() == 0
    assert not mask.iloc[-1] and not mask.iloc[-2]


def test_an_all_zero_column_is_not_flagged():
    mask = flag_sentinels(pd.Series([0.0] * 50), max_plausible=None)
    assert mask.sum() == 0


def test_a_rare_positive_minority_of_any_size_survives():
    """The same shape at a few different minority fractions — the fix should
    not depend on exactly two positives."""
    for n_pos in (1, 2, 5, 10):
        s = pd.Series([0.0] * 500 + [1.0] * n_pos)
        mask = flag_sentinels(s, max_plausible=None)
        assert mask.sum() == 0, f"n_pos={n_pos}"


def test_a_real_sentinel_is_still_caught_when_the_body_has_spread():
    """The fix only withholds judgment when the body has no spread to compare
    against. With real variation in the healthy values, a 999 among them is
    still flagged."""
    s = pd.Series([70.0, 72.0, 75.0, 71.0, 73.0, 68.0, 999.0])
    mask = flag_sentinels(s, max_plausible=None)
    assert mask.tolist() == [False, False, False, False, False, False, True]


def test_the_column_badges_endpoint_does_not_count_a_real_value_as_missing(client):
    df = pd.DataFrame({"Hematom": [0.0] * 380 + [1.0, 1.0]})
    sid = make_session(df, "sentinel_badge_probe")
    r = client.get(f"/api/stats/{sid}/column_badges")
    assert r.status_code == 200, r.text
    badge = r.json()["columns"]["Hematom"]
    assert badge["n_missing"] == 0
    assert badge["pct_missing"] == 0.0


def test_missing_diagnostics_reports_implausible_bmi(client):
    sid = _upload_cohort(client)
    r = client.post(f"/api/compute/{sid}/missing_diagnostics", json={"columns": ["bmi"]})
    assert r.status_code == 200, r.text
    col = r.json()["columns"][0]
    assert col["name"] == "bmi"
    assert col["n_implausible"] >= 1
    assert 999.0 in col["implausible_values"]
    assert col["review_flag"] == "implausible (review)"


def test_fill_blanks_mean_excludes_bmi_sentinel(client):
    sid = _upload_cohort(client)
    r = client.post(f"/api/compute/{sid}/fill_blanks", json={
        "column": "bmi",
        "value": "__mean__",
        "new_column": "bmi_mean_guarded",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_implausible"] == 1
    assert d["fill_value"].startswith("mean (27.")
    filled = store.get(sid)["bmi_mean_guarded"]
    assert filled.max() < 100


def test_h2fpef_masks_implausible_bmi_and_warns(client):
    sid = _upload_cohort(client)
    r = client.post(f"/api/compute/{sid}/clinical/h2fpef", json={
        "column_map": {"bmi": "bmi", "age": "age"},
        "new_col": "h2_guarded",
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert any("implausible" in w for w in d.get("warnings", []))
    out = store.get(sid)["h2_guarded"]
    assert out.iloc[19] == 3  # P020: comma decimal BMI 34.3 + elderly
    assert out.iloc[22] == 0  # P023: BMI 999 is not counted as obese


def test_mice_counts_and_imputes_bmi_sentinel(client):
    sid = _upload_cohort(client)
    r = client.post("/api/survival_advanced/mice", json={
        "session_id": sid,
        "columns": ["bmi"],
        "n_imputations": 1,
        "max_iter": 3,
        "new_columns": False,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["total_imputed"] == 4
    assert d["columns"][0]["n_imputed"] == 4
    assert store.get(sid)["bmi"].iloc[22] < 100
