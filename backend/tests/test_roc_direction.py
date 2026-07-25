"""ROC direction semantics.

Verify that ``direction='lower'`` returns ``1 - AUC`` with swapped
confidence-interval bounds compared with ``direction='higher'`` for the
same data.  This matters for "protective" scores (higher score = lower
event risk), e.g. higher eGFR predicting lower renal-failure risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_session


BASE = "/api/stats/roc"


def _make_roc_df(rng: np.random.Generator, n: int = 200) -> pd.DataFrame:
    """Higher scores predict higher event probability (positive association)."""
    score = rng.normal(0, 1, n)
    prob = 1.0 / (1.0 + np.exp(-(-0.5 + 2.0 * score)))
    event = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({"score": score, "event": event})


def test_roc_lower_returns_one_minus_auc_and_swapped_ci_bounds(client) -> None:
    """direction='lower' reports 1-AUC with swapped CI vs. direction='higher'."""
    rng = np.random.default_rng(20260709)
    df = _make_roc_df(rng)
    sid = make_session(df, "roc_dir_compare")

    base = {
        "session_id": sid,
        "score_column": "score",
        "outcome_column": "event",
    }

    hi = client.post(BASE, json={**base, "direction": "higher"})
    lo = client.post(BASE, json={**base, "direction": "lower"})

    assert hi.status_code == 200, hi.text
    assert lo.status_code == 200, lo.text

    h = hi.json()
    lo_d = lo.json()

    assert h["direction_used"] == "higher"
    assert lo_d["direction_used"] == "lower"
    assert h["direction_flipped"] is False
    assert lo_d["direction_flipped"] is True
    assert h["auc"] > 0.5

    # direction='lower' reports the complement AUC.
    assert lo_d["auc"] == pytest.approx(1.0 - h["auc"], abs=1e-6)

    # Confidence-interval bounds are swapped.
    assert lo_d["ci_lower"] == pytest.approx(1.0 - h["ci_upper"], abs=1e-6)
    assert lo_d["ci_upper"] == pytest.approx(1.0 - h["ci_lower"], abs=1e-6)


def test_roc_direction_invalid_rejected(client) -> None:
    """Unknown direction values are rejected with a clear error."""
    rng = np.random.default_rng(20260709)
    df = _make_roc_df(rng)
    sid = make_session(df, "roc_dir_invalid")

    r = client.post(
        BASE,
        json={
            "session_id": sid,
            "score_column": "score",
            "outcome_column": "event",
            "direction": "sideways",
        },
    )
    assert r.status_code == 422
    assert "direction" in r.text.lower()
