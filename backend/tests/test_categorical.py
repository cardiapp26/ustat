"""Tests for categorical endpoints."""
import numpy as np
import pandas as pd
import pytest
from conftest import make_session
from routers.categorical import (
    BinomialRequest,
    CochranArmitageRequest,
    MantelHaenszelRequest,
    McnemarRequest,
    OneProportionRequest,
    TwoProportionsRequest,
)


@pytest.mark.parametrize(
    ("model", "payload", "expected"),
    [
        (
            BinomialRequest,
            {"session_id": "s", "column": "x", "p": 0.4},
            {"expected_proportion": 0.4},
        ),
        (
            OneProportionRequest,
            {"session_id": "s", "column": "x", "p0": 0.45},
            {"null_proportion": 0.45},
        ),
        (
            TwoProportionsRequest,
            {"session_id": "s", "column": "x", "group_col": "group"},
            {"group_column": "group"},
        ),
        (
            McnemarRequest,
            {"session_id": "s", "column1": "before", "column2": "after"},
            {"col1": "before", "col2": "after"},
        ),
        (
            MantelHaenszelRequest,
            {
                "session_id": "s",
                "row_column": "row",
                "col_column": "col",
                "strata_column": "stratum",
            },
            {"row_col": "row", "col_col": "col", "strata_col": "stratum"},
        ),
        (
            CochranArmitageRequest,
            {
                "session_id": "s",
                "ordinal_column": "dose",
                "event_column": "event",
            },
            {"ordinal_col": "dose", "event_col": "event"},
        ),
    ],
)
def test_categorical_request_models_accept_specific_legacy_aliases(
    model, payload, expected
):
    request = model.model_validate(payload)
    for field, value in expected.items():
        assert getattr(request, field) == value


@pytest.mark.parametrize(
    ("model", "payload", "field", "canonical_value"),
    [
        (
            BinomialRequest,
            {
                "session_id": "s",
                "column": "x",
                "expected_proportion": 0.3,
                "p": 0.8,
            },
            "expected_proportion",
            0.3,
        ),
        (
            OneProportionRequest,
            {
                "session_id": "s",
                "column": "x",
                "null_proportion": 0.35,
                "p0": 0.75,
            },
            "null_proportion",
            0.35,
        ),
        (
            TwoProportionsRequest,
            {
                "session_id": "s",
                "column": "x",
                "group_column": "canonical_group",
                "group_col": "legacy_group",
            },
            "group_column",
            "canonical_group",
        ),
        (
            McnemarRequest,
            {
                "session_id": "s",
                "col1": "canonical_before",
                "column1": "legacy_before",
                "col2": "after",
            },
            "col1",
            "canonical_before",
        ),
        (
            MantelHaenszelRequest,
            {
                "session_id": "s",
                "row_col": "canonical_row",
                "row_column": "legacy_row",
                "col_col": "col",
                "strata_col": "stratum",
            },
            "row_col",
            "canonical_row",
        ),
        (
            CochranArmitageRequest,
            {
                "session_id": "s",
                "ordinal_col": "canonical_dose",
                "ordinal_column": "legacy_dose",
                "event_col": "event",
            },
            "ordinal_col",
            "canonical_dose",
        ),
    ],
)
def test_categorical_request_canonical_fields_win_over_legacy_aliases(
    model, payload, field, canonical_value
):
    assert getattr(model.model_validate(payload), field) == canonical_value


def test_binomial_known(client):
    df = pd.DataFrame({"outcome": [1]*60 + [0]*40})
    sid = make_session(df, "bin1")
    r = client.post("/api/categorical/binomial", json={"session_id": sid, "column": "outcome"})
    assert r.status_code == 200
    d = r.json()
    assert d["k"] == 60
    assert d["n"] == 100
    assert "r_code" in d
    assert "result_text" in d


def test_one_proportion(client):
    df = pd.DataFrame({"x": [1]*70 + [0]*30})
    sid = make_session(df, "op1")
    r = client.post("/api/categorical/one_proportion", json={"session_id": sid, "column": "x", "null_proportion": 0.5})
    assert r.status_code == 200
    d = r.json()
    assert d["significant"] is True
    assert "r_code" in d


def test_two_proportions(client):
    df = pd.DataFrame({
        "outcome": [1]*30 + [0]*20 + [1]*15 + [0]*35,
        "group": ["A"]*50 + ["B"]*50,
    })
    sid = make_session(df, "tp1")
    r = client.post("/api/categorical/two_proportions", json={"session_id": sid, "column": "outcome", "group_column": "group"})
    assert r.status_code == 200
    d = r.json()
    assert "effect_sizes" in d
    assert d["effect_sizes"][0]["name"] == "cohens_h"
    assert "r_code" in d


def test_mcnemar_known(client):
    # Classic McNemar: discordant pairs b=20, c=5
    df = pd.DataFrame({
        "before": [1]*30 + [0]*20 + [1]*5 + [0]*45,
        "after":  [1]*30 + [1]*20 + [0]*5 + [0]*45,
    })
    sid = make_session(df, "mc1")
    r = client.post("/api/categorical/mcnemar", json={"session_id": sid, "col1": "before", "col2": "after"})
    assert r.status_code == 200
    d = r.json()
    assert "r_code" in d
    assert "result_text" in d


def test_cochran_q(client):
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "t1": np.random.binomial(1, 0.3, n),
        "t2": np.random.binomial(1, 0.5, n),
        "t3": np.random.binomial(1, 0.7, n),
    })
    sid = make_session(df, "cq1")
    r = client.post("/api/categorical/cochran_q", json={"session_id": sid, "columns": ["t1", "t2", "t3"]})
    assert r.status_code == 200
    d = r.json()
    assert "Q" in d or "chi2" in d or "test" in d
    assert "r_code" in d


def test_mantel_haenszel(client):
    data = []
    for stratum in ["Hospital_A", "Hospital_B"]:
        base_or = 2.0 if stratum == "Hospital_A" else 1.5
        for _ in range(50):
            treat = np.random.binomial(1, 0.5)
            p_event = 0.3 * base_or if treat else 0.3
            event = np.random.binomial(1, min(p_event, 0.9))
            data.append({"treatment": treat, "event": event, "hospital": stratum})
    df = pd.DataFrame(data)
    sid = make_session(df, "mh1")
    r = client.post("/api/categorical/mantel_haenszel", json={
        "session_id": sid, "row_col": "treatment", "col_col": "event", "strata_col": "hospital"
    })
    assert r.status_code == 200
    d = r.json()
    assert "r_code" in d
    assert "result_text" in d


def _stratified_frame(tables):
    rows = []
    for stratum, table in enumerate(tables):
        for row_level in range(2):
            for col_level in range(2):
                rows.extend(
                    {
                        "row": row_level,
                        "col": col_level,
                        "stratum": stratum,
                    }
                    for _ in range(table[row_level][col_level])
                )
    return pd.DataFrame(rows)


def test_mantel_haenszel_common_or_ci_respects_alpha(client):
    tables = [
        [[12, 8], [6, 14]],
        [[15, 5], [10, 10]],
    ]
    sid = make_session(_stratified_frame(tables), "mh_ci_alpha")
    r = client.post("/api/categorical/mantel_haenszel", json={
        "session_id": sid,
        "row_col": "row",
        "col_col": "col",
        "strata_col": "stratum",
        "alpha": 0.1,
    })

    assert r.status_code == 200, r.text
    d = r.json()
    effect = d["effect_sizes"][0]
    assert effect["value"] == pytest.approx(3.2449, abs=1e-4)
    assert effect["ci_low"] == pytest.approx(1.4792, abs=1e-4)
    assert effect["ci_high"] == pytest.approx(7.1181, abs=1e-4)
    assert effect["ci_level"] == pytest.approx(0.9)
    assert "90% CI [1.479–7.118]" in d["interpretation"]
    exported = dict(d["export_rows"][1:])
    assert exported["Common OR 90% CI lower"] == pytest.approx(1.4792, abs=1e-4)
    assert exported["Common OR 90% CI upper"] == pytest.approx(7.1181, abs=1e-4)


@pytest.mark.parametrize("alpha", [-0.1, 0, 1, 1.1])
def test_mantel_haenszel_rejects_invalid_alpha(client, alpha):
    tables = [
        [[12, 8], [6, 14]],
        [[15, 5], [10, 10]],
    ]
    sid = make_session(_stratified_frame(tables), f"mh_invalid_alpha_{alpha}")
    response = client.post(
        "/api/categorical/mantel_haenszel",
        json={
            "session_id": sid,
            "row_col": "row",
            "col_col": "col",
            "strata_col": "stratum",
            "alpha": alpha,
        },
    )
    assert response.status_code == 422


def test_mantel_haenszel_rejects_nonfinite_alpha():
    with pytest.raises(ValueError):
        MantelHaenszelRequest.model_validate(
            {
                "session_id": "s",
                "row_col": "row",
                "col_col": "col",
                "strata_col": "stratum",
                "alpha": float("nan"),
            }
        )


def test_mantel_haenszel_common_or_ci_handles_zero_cells(client):
    tables = [
        [[12, 8], [6, 14]],
        [[10, 0], [10, 10]],
    ]
    sid = make_session(_stratified_frame(tables), "mh_ci_zero")
    r = client.post("/api/categorical/mantel_haenszel", json={
        "session_id": sid,
        "row_col": "row",
        "col_col": "col",
        "strata_col": "stratum",
    })

    assert r.status_code == 200, r.text
    effect = r.json()["effect_sizes"][0]
    assert effect["value"] == pytest.approx(6.2778, abs=1e-4)
    assert effect["ci_low"] == pytest.approx(1.8893, abs=1e-4)
    assert effect["ci_high"] == pytest.approx(20.8599, abs=1e-4)


def test_mantel_haenszel_nonfinite_common_or_and_ci_are_null(client):
    tables = [
        [[10, 0], [0, 10]],
        [[10, 0], [0, 10]],
    ]
    sid = make_session(_stratified_frame(tables), "mh_ci_nonfinite")
    r = client.post("/api/categorical/mantel_haenszel", json={
        "session_id": sid,
        "row_col": "row",
        "col_col": "col",
        "strata_col": "stratum",
    })

    assert r.status_code == 200, r.text
    d = r.json()
    effect = d["effect_sizes"][0]
    assert effect["value"] is None
    assert effect["ci_low"] is None
    assert effect["ci_high"] is None
    assert any("odds ratio is not finite" in warning for warning in d["warnings"])
