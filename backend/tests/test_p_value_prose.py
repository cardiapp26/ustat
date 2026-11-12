import numpy as np
import pandas as pd
import pytest

from conftest import make_session


@pytest.mark.parametrize(("endpoint", "parameters", "p_field"), [
    ("/api/stats/ttest", {"column": "after"}, "p"),
    ("/api/stats/ttest", {"column": "after", "group_column": "group"}, "p"),
    ("/api/repeated/paired_ttest", {"col1": "before", "col2": "after"}, "p"),
    ("/api/repeated/wilcoxon_signed_rank", {"col1": "before", "col2": "after"}, "p"),
    (
        "/api/categorical/mantel_haenszel",
        {"row_col": "group", "col_col": "event", "strata_col": "stratum"},
        "p",
    ),
    (
        "/api/stats/noninferiority",
        {"outcome_col": "after", "group_col": "group", "outcome_type": "continuous",
         "effect": "MD", "margin": -1, "bound": "lower"},
        "p_noninferiority",
    ),
])
def test_small_p_value_reads_as_inequality(client, endpoint, parameters, p_field):
    rng = np.random.default_rng(144)
    group = np.repeat([0, 1], 50)
    before = rng.normal(size=100)
    frame = pd.DataFrame({
        "before": before,
        "after": before + 5 + 6 * group + rng.normal(size=100),
        "group": group,
        "stratum": np.tile(np.repeat(["A", "B"], 25), 2),
        "event": np.tile([0] * 22 + [1] * 3, 4) ^ group,
    })
    sid = make_session(frame, "p_value_prose")
    r = client.post(endpoint, json={"session_id": sid, **parameters})
    assert r.status_code == 200, r.text
    result = r.json()
    assert result[p_field] < 0.001
    assert "p < 0.001" in result["interpretation"]
    if "result_text" in result:
        assert "p < 0.001" in result["result_text"]
