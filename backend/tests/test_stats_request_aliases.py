from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from conftest import make_session
from routers.stats.correlation import FleissKappaRequest, ICCRequest, KappaRequest
from routers.stats.inferential import (
    AnovaRequest,
    ChiSqRequest,
    FisherRequest,
    NonInferiorityRequest,
    TOSTRequest,
    TTestRequest,
)
from routers.stats.nonparametric import (
    JonckheereRequest,
    KruskalRequest,
    MannWhitneyRequest,
    ROCCombinedRequest,
    ROCCompareRequest,
    ROCMultiCompareRequest,
    ROCRequest,
)


def _alias_case(
    model_type: type[BaseModel],
    base: dict[str, Any],
    canonical: str,
    alias: str,
    *,
    plural: bool = False,
) -> pytest.ParameterSet:
    canonical_value: Any = ["canonical"] if plural else "canonical"
    alias_value: Any = ["alias"] if plural else "alias"
    return pytest.param(
        model_type,
        base,
        canonical,
        alias,
        canonical_value,
        alias_value,
        id=f"{model_type.__name__}.{canonical}",
    )


ALIAS_CASES = [
    _alias_case(
        TTestRequest,
        {"session_id": "s", "column": "value"},
        "group_column",
        "group_col",
    ),
    _alias_case(
        ChiSqRequest,
        {"session_id": "s", "row_column": "row", "col_column": "col"},
        "row_column",
        "row_col",
    ),
    _alias_case(
        ChiSqRequest,
        {"session_id": "s", "row_column": "row", "col_column": "col"},
        "col_column",
        "col_col",
    ),
    _alias_case(
        FisherRequest,
        {"session_id": "s", "row_column": "row", "col_column": "col"},
        "row_column",
        "row_col",
    ),
    _alias_case(
        FisherRequest,
        {"session_id": "s", "row_column": "row", "col_column": "col"},
        "col_column",
        "col_col",
    ),
    _alias_case(
        AnovaRequest,
        {"session_id": "s", "column": "value", "group_column": "group"},
        "group_column",
        "group_col",
    ),
    _alias_case(
        TOSTRequest,
        {"session_id": "s", "column": "value", "low": -1, "high": 1},
        "group_column",
        "group_col",
    ),
    _alias_case(
        TOSTRequest,
        {"session_id": "s", "column": "value", "low": -1, "high": 1},
        "paired_column",
        "paired_col",
    ),
    _alias_case(
        NonInferiorityRequest,
        {"session_id": "s", "outcome_col": "outcome", "group_col": "group"},
        "outcome_col",
        "outcome_column",
    ),
    _alias_case(
        NonInferiorityRequest,
        {"session_id": "s", "outcome_col": "outcome", "group_col": "group"},
        "group_col",
        "group_column",
    ),
    _alias_case(
        MannWhitneyRequest,
        {"session_id": "s", "column": "value", "group_column": "group"},
        "group_column",
        "group_col",
    ),
    _alias_case(
        KruskalRequest,
        {"session_id": "s", "column": "value", "group_column": "group"},
        "group_column",
        "group_col",
    ),
    _alias_case(
        JonckheereRequest,
        {"session_id": "s", "column": "value", "group_column": "group"},
        "group_column",
        "group_col",
    ),
    _alias_case(
        ROCRequest,
        {"session_id": "s", "score_column": "score", "outcome_column": "outcome"},
        "score_column",
        "score_col",
    ),
    _alias_case(
        ROCRequest,
        {"session_id": "s", "score_column": "score", "outcome_column": "outcome"},
        "outcome_column",
        "outcome_col",
    ),
    _alias_case(
        ROCCompareRequest,
        {
            "session_id": "s",
            "score_column_1": "score1",
            "score_column_2": "score2",
            "outcome_column": "outcome",
        },
        "score_column_1",
        "score_col_1",
    ),
    _alias_case(
        ROCCompareRequest,
        {
            "session_id": "s",
            "score_column_1": "score1",
            "score_column_2": "score2",
            "outcome_column": "outcome",
        },
        "score_column_2",
        "score_col_2",
    ),
    _alias_case(
        ROCCompareRequest,
        {
            "session_id": "s",
            "score_column_1": "score1",
            "score_column_2": "score2",
            "outcome_column": "outcome",
        },
        "outcome_column",
        "outcome_col",
    ),
    _alias_case(
        ROCMultiCompareRequest,
        {
            "session_id": "s",
            "score_columns": ["score1", "score2"],
            "outcome_column": "outcome",
        },
        "score_columns",
        "score_cols",
        plural=True,
    ),
    _alias_case(
        ROCMultiCompareRequest,
        {
            "session_id": "s",
            "score_columns": ["score1", "score2"],
            "outcome_column": "outcome",
        },
        "outcome_column",
        "outcome_col",
    ),
    _alias_case(
        ROCCombinedRequest,
        {
            "session_id": "s",
            "predictor_columns": ["score1", "score2"],
            "outcome_column": "outcome",
        },
        "predictor_columns",
        "predictor_cols",
        plural=True,
    ),
    _alias_case(
        ROCCombinedRequest,
        {
            "session_id": "s",
            "predictor_columns": ["score1", "score2"],
            "outcome_column": "outcome",
        },
        "outcome_column",
        "outcome_col",
    ),
    _alias_case(
        ICCRequest,
        {"session_id": "s", "rater1_col": "rater1", "rater2_col": "rater2"},
        "rater1_col",
        "rater1_column",
    ),
    _alias_case(
        ICCRequest,
        {"session_id": "s", "rater1_col": "rater1", "rater2_col": "rater2"},
        "rater2_col",
        "rater2_column",
    ),
    _alias_case(
        KappaRequest,
        {"session_id": "s", "rater1_col": "rater1", "rater2_col": "rater2"},
        "rater1_col",
        "rater1_column",
    ),
    _alias_case(
        KappaRequest,
        {"session_id": "s", "rater1_col": "rater1", "rater2_col": "rater2"},
        "rater2_col",
        "rater2_column",
    ),
    _alias_case(
        FleissKappaRequest,
        {"session_id": "s", "rater_cols": ["rater1", "rater2", "rater3"]},
        "rater_cols",
        "rater_columns",
        plural=True,
    ),
]


@pytest.mark.parametrize(
    (
        "model_type",
        "base",
        "canonical",
        "alias",
        "canonical_value",
        "alias_value",
    ),
    ALIAS_CASES,
)
def test_request_models_accept_safe_aliases(
    model_type: type[BaseModel],
    base: dict[str, Any],
    canonical: str,
    alias: str,
    canonical_value: Any,
    alias_value: Any,
) -> None:
    payload = {key: value for key, value in base.items() if key != canonical}
    payload[alias] = alias_value

    model = model_type.model_validate(payload)

    assert getattr(model, canonical) == alias_value
    dumped = model.model_dump()
    assert dumped[canonical] == alias_value
    assert alias not in dumped


@pytest.mark.parametrize(
    (
        "model_type",
        "base",
        "canonical",
        "alias",
        "canonical_value",
        "alias_value",
    ),
    ALIAS_CASES,
)
def test_canonical_request_fields_take_precedence(
    model_type: type[BaseModel],
    base: dict[str, Any],
    canonical: str,
    alias: str,
    canonical_value: Any,
    alias_value: Any,
) -> None:
    payload = {**base, canonical: canonical_value, alias: alias_value}

    model = model_type.model_validate(payload)

    assert getattr(model, canonical) == canonical_value
    assert model.model_dump()[canonical] == canonical_value


@pytest.fixture
def alias_session() -> str:
    outcome = np.tile([0, 1], 20)
    group = np.repeat(["control", "test"], 20)
    score = outcome + np.linspace(0.0, 0.2, 40)
    rater1 = np.linspace(10.0, 50.0, 40)
    df = pd.DataFrame(
        {
            "value": np.concatenate(
                [np.linspace(1.0, 3.0, 20), np.linspace(2.0, 4.5, 20)]
            ),
            "group": group,
            "outcome": outcome,
            "score": score,
            "rater1": rater1,
            "rater2": rater1 + np.tile([-0.2, 0.2], 20),
        }
    )
    return make_session(df, "request_alias_integration")


@pytest.mark.parametrize(
    ("endpoint", "payload", "result_key"),
    [
        pytest.param(
            "/api/stats/ttest",
            {"column": "value", "group_col": "group"},
            "t",
            id="inferential",
        ),
        pytest.param(
            "/api/stats/noninferiority",
            {"outcome_column": "outcome", "group_column": "group"},
            "p_noninferiority",
            id="noninferiority",
        ),
        pytest.param(
            "/api/stats/mannwhitney",
            {"column": "value", "group_col": "group"},
            "U",
            id="nonparametric",
        ),
        pytest.param(
            "/api/stats/roc",
            {"score_col": "score", "outcome_col": "outcome"},
            "auc",
            id="roc",
        ),
        pytest.param(
            "/api/stats/icc",
            {"rater1_column": "rater1", "rater2_column": "rater2"},
            "icc",
            id="correlation",
        ),
    ],
)
def test_aliases_reach_stats_endpoints(
    client,
    alias_session: str,
    endpoint: str,
    payload: dict[str, Any],
    result_key: str,
) -> None:
    response = client.post(
        endpoint,
        json={"session_id": alias_session, **payload},
    )

    assert response.status_code == 200, response.text
    assert result_key in response.json()
