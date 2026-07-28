"""Table 1 reports missingness without adding it to inferential categories."""

import numpy as np
import pandas as pd

from conftest import make_session
from routers.stats.descriptive import WeightedDescriptiveRequest


def _table1(client, sid: str, *, selected_stats=None):
    payload = {
        "session_id": sid,
        "group_column": "arm",
        "variables": ["age", "status", "complete"],
        "variable_kinds": {
            "age": "numeric",
            "status": "categorical",
            "complete": "numeric",
        },
    }
    if selected_stats is not None:
        payload["selected_stats"] = selected_stats
    response = client.post("/api/stats/table1", json=payload)
    assert response.status_code == 200, response.text
    return {row["variable"]: row for row in response.json()["rows"]}


def _session() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "arm": ["A"] * 5 + ["B"] * 5,
            "age": [
                10.0,
                11.0,
                np.nan,
                13.0,
                14.0,
                20.0,
                np.nan,
                22.0,
                23.0,
                24.0,
            ],
            "status": [
                "yes",
                "no",
                None,
                "yes",
                "no",
                "yes",
                None,
                "no",
                None,
                "yes",
            ],
            "complete": list(range(10)),
        }
    )


def test_missing_rows_are_in_the_default_api_contract(client):
    sid = make_session(_session(), "table1_missing_rows_default")
    rows = _table1(client, sid)

    assert rows["age"]["missing_row"] == {
        "label": "Missing n (%)",
        "overall": "2 (20.0%)",
        "group_stats": {"A": "1 (20.0%)", "B": "1 (20.0%)"},
    }
    assert rows["status"]["missing_row"] == {
        "label": "Missing n (%)",
        "overall": "3 (30.0%)",
        "group_stats": {"A": "1 (20.0%)", "B": "2 (40.0%)"},
    }
    assert rows["complete"]["missing_row"] is None

    # Missingness is display metadata, not a categorical level.
    printed_categories = {sub["category"] for sub in rows["status"]["sub_rows"]}
    assert len(printed_categories) == 2
    assert printed_categories.isdisjoint({"Missing", "nan", "None"})


def test_explicit_missing_stat_does_not_duplicate_numeric_missing_row(client):
    sid = make_session(_session(), "table1_missing_rows_explicit")
    rows = _table1(client, sid, selected_stats=["mean_sd", "missing"])
    age = rows["age"]

    assert age["missing_row"] is None
    missing_stats = [
        stat for stat in age["stat_rows"] if stat["label"] == "Missing n (%)"
    ]
    assert missing_stats == [
        {
            "label": "Missing n (%)",
            "overall": "2 (20.0%)",
            "group_stats": {"A": "1 (20.0%)", "B": "1 (20.0%)"},
        }
    ]


def test_categorical_missingness_does_not_change_p_value_or_smd(client):
    source = _session()
    with_missing = make_session(source, "table1_missing_rows_inference")
    complete_cases = make_session(
        source[source["status"].notna()].reset_index(drop=True),
        "table1_missing_rows_complete_cases",
    )

    missing_status = _table1(client, with_missing)["status"]
    complete_status = _table1(client, complete_cases)["status"]

    assert missing_status["p_raw"] == complete_status["p_raw"]
    assert missing_status["smd"] == complete_status["smd"]


def test_journal_formatter_preserves_missing_rows(client):
    sid = make_session(_session(), "table1_missing_rows_journal")
    response = client.post(
        "/api/stats/table1",
        json={
            "session_id": sid,
            "group_column": "arm",
            "variables": ["age", "status"],
            "variable_kinds": {"age": "numeric", "status": "categorical"},
        },
    )
    assert response.status_code == 200, response.text

    formatted = client.post(
        "/api/pub_tables/format",
        json={"table1_result": response.json()},
    )
    assert formatted.status_code == 200, formatted.text
    missing_rows = [
        row
        for row in formatted.json()["rows"]
        if row["label"] == "Missing n (%)"
    ]
    assert [row["values"] for row in missing_rows] == [
        ["1 (20.0%)", "1 (20.0%)"],
        ["1 (20.0%)", "2 (40.0%)"],
    ]
    assert all(row["indent"] == 1 for row in missing_rows)


def test_weighted_descriptive_request_accepts_long_form_column_aliases():
    request = WeightedDescriptiveRequest.model_validate(
        {
            "session_id": "alias-test",
            "value_columns": ["age", "bmi"],
            "weight_column": "weight",
            "group_column": "arm",
        }
    )
    assert request.value_cols == ["age", "bmi"]
    assert request.weight_col == "weight"
    assert request.group_col == "arm"

    canonical_wins = WeightedDescriptiveRequest.model_validate(
        {
            "session_id": "canonical-test",
            "value_cols": ["canonical"],
            "value_columns": ["alias"],
            "weight_col": "canonical_weight",
            "weight_column": "alias_weight",
        }
    )
    assert canonical_wins.value_cols == ["canonical"]
    assert canonical_wins.weight_col == "canonical_weight"
