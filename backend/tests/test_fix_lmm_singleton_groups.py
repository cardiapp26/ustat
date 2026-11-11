"""LMM must refuse a grouping variable with one row per group.

Grouping wide data by the patient ID gave ICC = 0.5 and an age standard error
four times too small (p < 0.001 where OLS gave p = 0.11).
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_session
from fastapi import HTTPException

from routers.models.linear import LMMRequest, linear_mixed_model

N = 120


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    site = rng.choice(["s1", "s2", "s3", "s4"], N)
    age = rng.normal(60, 10, N)
    y = 70 - 0.15 * (age - 60) + pd.Series(site).map({"s1": -2, "s2": 0, "s3": 1, "s4": 3}).to_numpy() + rng.normal(0, 8, N)
    return pd.DataFrame({"pid": np.arange(1, N + 1), "site": site, "age": age, "y": y})


def test_lmm_rejects_one_row_per_group():
    sid = make_session(_frame(), "lmm_singleton")
    with pytest.raises(HTTPException) as err:
        linear_mixed_model(LMMRequest(session_id=sid, outcome="y", group_col="pid", fixed_effects=["age"]))
    assert err.value.status_code == 422
    assert "single observation" in err.value.detail


def test_lmm_still_fits_with_real_clusters():
    sid = make_session(_frame(), "lmm_clusters")
    res = linear_mixed_model(LMMRequest(session_id=sid, outcome="y", group_col="site", fixed_effects=["age"]))
    assert res["n_groups"] == 4
