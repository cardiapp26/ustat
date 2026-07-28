"""Missing values must not become a category, and a category must not
silently become missing.

Both directions were live: `astype(str)` turned NaN into the level "nan", and
`clean_two_level` deleted every row whose value read as a missing token —
including "none", which in clinical data usually means the finding is absent,
not that nobody recorded it.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session
from services.category_health import clean_two_level
from services.external_validation import transportability_analysis
from services.stat_utils import (
    MAX_TEST_CATEGORIES,
    _categorical_p_with_rule,
    looks_continuous,
)


# ── clean_two_level ───────────────────────────────────────────────────────────


def test_none_is_a_level_not_a_missing_marker():
    s = pd.Series(["typical", "atypical", "none", "none", "typical"], name="cp")
    out = clean_two_level(s)
    assert out.series.notna().all(), "'none' rows were deleted"
    assert set(out.series.dropna()) == {"typical", "atypical", "none"}


def test_recognised_missing_tokens_are_reported_not_swallowed():
    s = pd.Series(["yes", "no", "unknown", "N/A", "yes"], name="diabetes")
    out = clean_two_level(s)
    assert out.series.isna().sum() == 2
    assert out.warnings, "rows vanished with nothing said"
    dropped = {d["level"] for w in out.warnings for d in w["dropped_levels"]}
    assert dropped == {"unknown", "n/a"}


def test_blanks_stay_quiet():
    """A blank cell is unambiguous; warning about it would be noise."""
    s = pd.Series(["M", "F", "", "  ", "M"], name="sex")
    out = clean_two_level(s)
    assert out.series.isna().sum() == 2
    assert out.warnings == []


def test_table1_surfaces_the_dropped_rows(client):
    df = pd.DataFrame(
        {
            "arm": ["A", "B"] * 20,
            "status": (["mild", "severe", "none", "unknown"] * 10),
        }
    )
    sid = make_session(df, "mnc_tokens")
    r = client.post(
        "/api/stats/table1",
        json={"session_id": sid, "variables": ["status"], "group_column": "arm"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    row = payload["rows"][0]
    printed = {sub["category"] for sub in row["sub_rows"]}
    assert "none" in printed, "a real level was deleted"
    assert "unknown" not in printed
    assert any("unknown" in str(w) for w in payload["warnings"])


# ── the small-cell / degenerate rule ──────────────────────────────────────────


def test_single_row_table_has_no_p():
    p, reason = _categorical_p_with_rule(np.array([[10.0, 12.0]]))
    assert p is None
    assert "one category" in reason


def test_single_column_table_has_no_p():
    p, reason = _categorical_p_with_rule(np.array([[10.0], [12.0]]))
    assert p is None
    assert "one group" in reason


def test_empty_table_returns_a_reason_instead_of_raising():
    p, reason = _categorical_p_with_rule(np.zeros((0, 0)))
    assert p is None
    assert reason


def test_too_many_categories_is_refused():
    n = MAX_TEST_CATEGORIES + 1
    obs = np.ones((n, 2))
    p, reason = _categorical_p_with_rule(obs)
    assert p is None
    assert str(n) in reason


def test_a_real_table_still_gets_a_p():
    p, reason = _categorical_p_with_rule(np.array([[30.0, 10.0], [10.0, 30.0]]))
    assert p is not None and p < 0.05
    assert reason == "Chi-square"


# ── numeric detection ─────────────────────────────────────────────────────────


def test_fractional_floats_are_continuous_even_with_few_rows():
    assert looks_continuous(pd.Series([1.5, 2.25, 3.75, 4.5]))


def test_integer_flags_stay_categorical():
    assert not looks_continuous(pd.Series([0.0, 1.0] * 25))
    assert not looks_continuous(pd.Series([1, 2, 3, 4, 5] * 10))


def test_many_distinct_integers_are_still_continuous():
    assert looks_continuous(pd.Series(range(50)))


def test_text_is_never_continuous():
    assert not looks_continuous(pd.Series(["a", "b", "c"]))


# ── transportability ──────────────────────────────────────────────────────────


def _shift_row(report, name):
    return next(r for r in report["covariate_shift"] if r["covariate"] == name)


def test_missingness_is_not_a_covariate_level():
    """astype(str) made NaN the level "nan", so a cohort that merely recorded
    a covariate less often looked like it had a different case mix."""
    dev = pd.DataFrame({"sex": ["M", "F"] * 50})
    val = pd.DataFrame({"sex": ["M", "F"] * 50})
    val.loc[:59, "sex"] = np.nan  # same mix, 60% of it simply not recorded

    report = transportability_analysis(dev, val, ["sex"])
    row = _shift_row(report, "sex")
    assert row["max_absolute_level_shift"] == pytest.approx(0.0, abs=1e-9)
    assert not row["flag_large_shift"]
    # It is still a transport problem, so it has to be visible somewhere.
    assert row["flag_missingness_shift"]
    assert row["val_missing_rate"] == pytest.approx(0.6, abs=0.01)
    assert row["n_val"] == 40
    assert report["n_missingness_shifts"] == 1
    assert any("different rates" in w for w in report["warnings"])


def test_a_genuine_level_shift_is_still_caught():
    dev = pd.DataFrame({"sex": ["M"] * 80 + ["F"] * 20})
    val = pd.DataFrame({"sex": ["M"] * 20 + ["F"] * 80})
    row = _shift_row(transportability_analysis(dev, val, ["sex"]), "sex")
    assert row["max_absolute_level_shift"] == pytest.approx(0.6, abs=1e-9)
    assert row["flag_large_shift"]
    assert not row["flag_missingness_shift"]


def test_numeric_covariates_report_the_rows_behind_the_mean():
    dev = pd.DataFrame({"age": [60.0] * 100})
    val = pd.DataFrame({"age": [60.0] * 50 + [np.nan] * 50})
    row = _shift_row(transportability_analysis(dev, val, ["age"]), "age")
    assert row["type"] == "numeric"
    assert row["n_dev"] == 100 and row["n_val"] == 50
    assert row["val_missing_rate"] == pytest.approx(0.5)
    assert row["flag_missingness_shift"]
