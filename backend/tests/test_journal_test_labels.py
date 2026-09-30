"""Journal Table 1 footnotes name the test that produced each p.

Reported on a real manuscript table: age was compared with Welch's t-test
(Levene p<0.001) but footnoted as Student's; smoking carried a
Yates-corrected chi-square p (0.829) under a "Pearson chi-square" footnote,
whose uncorrected p is 0.662; and the table never said which normality test
chose mean ± SD over median [IQR]. Chi-square is now Pearson's without the
continuity correction everywhere, so the footnote and the number agree.
"""
import numpy as np
import pandas as pd
from scipy import stats

from services import store
from services.journal_formatter import (
    _assign_test_symbol,
    _TEST_DISPLAY_NAMES,
    format_table1_for_journal,
)
from services.stat_utils import _categorical_p_with_rule

DAGGERS = ("†", "‡")

# The smoking table from the manuscript: rows smoker yes/no, columns
# control/myocarditis. Smallest expected count is about 13.
SMOKING = np.array([[12, 25], [21, 36]])


def test_welch_is_not_footnoted_as_student():
    welch = _assign_test_symbol("t-test (Welch)")
    student = _assign_test_symbol("t-test")
    assert welch != student
    assert "Welch" in _TEST_DISPLAY_NAMES[welch]
    assert "Student" in _TEST_DISPLAY_NAMES[student]


def test_ffh_and_fisher_get_distinct_symbols():
    ffh = _assign_test_symbol("Fisher-Freeman-Halton (MC, 5000 resamples)")
    fisher = _assign_test_symbol("Fisher")
    assert ffh and fisher and ffh != fisher


def test_no_dagger_symbols_are_used():
    for sym, name in _TEST_DISPLAY_NAMES.items():
        assert not any(d in sym for d in DAGGERS), (sym, name)


def test_2x2_rule_reports_uncorrected_pearson():
    p, name = _categorical_p_with_rule(SMOKING)
    assert name == "Chi-square"
    pearson = stats.chi2_contingency(SMOKING, correction=False)[1]
    assert p == pearson
    assert round(p, 3) == 0.662


def _manuscript_like_session() -> str:
    rng = np.random.default_rng(7)
    n0, n1 = 33, 61
    df = pd.DataFrame({
        "grp": [0] * n0 + [1] * n1,
        # Very different spreads so Levene rejects and Welch runs.
        "age": np.r_[rng.normal(27, 2.4, n0), rng.normal(26, 5.6, n1)].round(),
        "smoker": [1] * 12 + [0] * 21 + [1] * 25 + [0] * 36,
    })
    sid = "journal_test_labels"
    store.save(sid, df)
    return sid


def test_exported_table_names_welch_pearson_and_normality_rule(client):
    sid = _manuscript_like_session()
    r = client.post("/api/stats/table1", json={
        "session_id": sid, "group_column": "grp",
        "variables": ["age", "smoker"],
        "variable_kinds": {"age": "numeric", "smoker": "categorical"},
    })
    assert r.status_code == 200, r.text
    result = r.json()
    rows = {row["variable"]: row for row in result["rows"]}
    assert rows["age"]["test"] == "t-test (Welch)"
    assert rows["smoker"]["test"] == "Chi-square"
    assert rows["smoker"]["p_value"] == "0.662"

    journal = format_table1_for_journal(result)
    notes = " ".join(journal["footnotes"])
    assert "Welch’s t-test" in notes
    assert "Pearson chi-square test (no continuity correction)" in notes
    assert "Yates" not in notes
    # Groups of 33 and 61 straddle the n=50 switch, so both tests are named.
    assert "Shapiro-Wilk" in notes and "Lilliefors" in notes
    assert "within each group" in notes
    assert not any(d in notes for d in DAGGERS)
    assert not any(d in row["test_symbol"] for row in journal["rows"] for d in DAGGERS)


def test_chisquare_endpoint_reports_pearson_on_2x2(client):
    sid = "journal_test_labels_ep"
    store.save(sid, pd.DataFrame({
        "g": [0] * 33 + [1] * 61,
        "s": [1] * 12 + [0] * 21 + [1] * 25 + [0] * 36,
    }))
    r = client.post("/api/stats/chisquare", json={
        "session_id": sid, "row_column": "s", "col_column": "g",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert round(body["p"], 3) == 0.662
    assert "Pearson" in body["methods_text"]
    assert "correct = FALSE" in body["r_code"]
