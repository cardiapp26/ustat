"""The 'consider MICE' warning must not appear under a result that used MICE."""

from services.missing_data import add_missing_data_diagnostics

INFO = {"pct_affected": 15.0}


def _warnings(result: dict) -> list:
    return add_missing_data_diagnostics(result, INFO).get("warnings", [])


def test_complete_case_result_still_gets_the_recommendation():
    assert any("multiple imputation" in w for w in _warnings({"imputation": "listwise"}))


def test_pooled_result_does_not_recommend_mice():
    assert _warnings({"imputation": "mice (pooled)", "pooled_from_imputations": True}) == []
    assert _warnings({"pooled_from_imputations": True}) == []
