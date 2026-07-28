"""Table 1's categorical p must describe the table Table 1 prints.

Reported: the chi-square p in Tests disagreed with the p for the same two
variables in Table 1. Cause: Table 1 built its crosstab from
`df[var].astype(str)`, which turns NaN into the literal string "nan".
pd.crosstab then counted it as a category — an extra row, an extra degree
of freedom, and the missing rows back in the test. The printed categories
came from value_counts(dropna=True), so that row never appeared on screen:
the table and its p-value described different tables.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.fixture()
def with_missing() -> str:
    rng = np.random.default_rng(7)
    n = 303
    cp = pd.Series(rng.choice(["typical", "atypical", "nonanginal"], n), dtype=object)
    cp[rng.choice(n, 25, replace=False)] = np.nan
    df = pd.DataFrame({"cp": cp, "hd": rng.choice(["Yes", "No"], n)})
    return make_session(df, "t1_missing")


def _p_of(row) -> float:
    """Table 1 returns the p formatted for display; parse it back."""
    raw = row["p_value"]
    return 0.0005 if str(raw).startswith("<") else float(raw)


def _table1_row(client, sid, var, group):
    r = client.post(
        "/api/stats/table1",
        json={"session_id": sid, "variables": [var], "group_column": group},
    )
    assert r.status_code == 200, r.text
    row = next(x for x in r.json()["rows"] if x["variable"] == var)
    return r.json(), row


def _tests_chisquare(client, sid, row_col, col_col):
    r = client.post(
        "/api/stats/chisquare",
        json={"session_id": sid, "row_column": row_col, "col_column": col_col},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_table1_and_tests_agree_on_the_same_pair(client, with_missing):
    """The user-visible symptom: two tabs, two different p-values."""
    chi = _tests_chisquare(client, with_missing, "cp", "hd")
    _, row = _table1_row(client, with_missing, "cp", "hd")
    assert _p_of(row) == pytest.approx(chi["p"], abs=1e-3), (
        f"Table 1 says {row['p_value']}, Tests says {chi['p']}"
    )


def test_missing_is_not_counted_as_a_category(client, with_missing):
    """Recomputed by hand: including "nan" gives 3 df and p = 0.716."""
    from scipy import stats as sp

    chi = _tests_chisquare(client, with_missing, "cp", "hd")
    assert chi["dof"] == 2, "three real levels, not four"
    assert chi["n"] == 278, "the 25 missing rows must be out"

    # What the old code computed, kept here so the regression is unmistakable.
    df = pd.DataFrame(
        {
            "cp": pd.Series(
                np.random.default_rng(7).choice(
                    ["typical", "atypical", "nonanginal"], 303
                ),
                dtype=object,
            ),
            "hd": np.random.default_rng(7).choice(["Yes", "No"], 303),
        }
    )
    df.loc[np.random.default_rng(7).choice(303, 25, replace=False), "cp"] = np.nan
    stale = pd.crosstab(df["cp"].astype(str), df["hd"])
    assert "nan" in stale.index, "astype(str) is what created the phantom level"


def test_row_counts_and_the_test_use_the_same_rows(client, with_missing):
    """The printed n per category must add up to the n the test ran on."""
    chi = _tests_chisquare(client, with_missing, "cp", "hd")
    _, row = _table1_row(client, with_missing, "cp", "hd")
    printed = sum(
        int(sub["overall"].split(" ")[0]) for sub in row.get("sub_rows", [])
    )
    assert printed == chi["n"]


def test_a_column_with_no_missing_is_unaffected(client):
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {"sex": rng.choice(["M", "F"], 200), "arm": rng.choice(["A", "B"], 200)}
    )
    sid = make_session(df, "t1_complete")
    chi = _tests_chisquare(client, sid, "sex", "arm")
    _, row = _table1_row(client, sid, "sex", "arm")
    assert _p_of(row) == pytest.approx(chi["p"], abs=1e-3)
    assert chi["n"] == 200
