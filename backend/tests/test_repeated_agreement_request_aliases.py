"""Backward-compatible request aliases for repeated and agreement endpoints."""

import pandas as pd
import pytest

from conftest import make_session
from routers.agreement import BlandAltmanRequest
from routers.repeated import PairedTTestRequest


def test_repeated_request_accepts_long_column_names_and_canonical_wins():
    aliased = PairedTTestRequest.model_validate(
        {"session_id": "s", "column1": "pre", "column2": "post"}
    )
    assert (aliased.col1, aliased.col2) == ("pre", "post")

    canonical = PairedTTestRequest.model_validate(
        {
            "session_id": "s",
            "col1": "pre",
            "column1": "wrong",
            "col2": "post",
            "column2": "wrong",
        }
    )
    assert (canonical.col1, canonical.col2) == ("pre", "post")


def test_agreement_request_accepts_generic_columns_and_canonical_wins():
    aliased = BlandAltmanRequest.model_validate(
        {"session_id": "s", "column1": "first", "column2": "second"}
    )
    assert (aliased.method1, aliased.method2) == ("first", "second")

    canonical = BlandAltmanRequest.model_validate(
        {
            "session_id": "s",
            "method1": "first",
            "column1": "wrong",
            "method2": "second",
            "column2": "wrong",
        }
    )
    assert (canonical.method1, canonical.method2) == ("first", "second")


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/api/repeated/paired_ttest",
            {"column1": "first", "column2": "second"},
        ),
        (
            "/api/agreement/concordance",
            {"column1": "first", "column2": "second"},
        ),
    ],
)
def test_aliases_reach_representative_endpoints(client, path, payload):
    sid = make_session(
        pd.DataFrame(
            {
                "first": [1.0, 2.0, 4.0, 5.0, 7.0, 8.0],
                "second": [1.2, 2.1, 3.8, 5.3, 6.9, 8.2],
            }
        ),
        f"alias-{path.rsplit('/', 1)[-1]}",
    )
    response = client.post(path, json={"session_id": sid, **payload})
    assert response.status_code == 200, response.text
