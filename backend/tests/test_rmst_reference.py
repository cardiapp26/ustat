"""Reference checks for restricted mean survival time."""

import math

import numpy as np

from scipy.stats import norm

from services.survival_advanced_service import (
    _rmst_one_group,
    _two_sided_normal_p,
)


def test_rmst_greenwood_se_matches_r_survival():
    """Match survival::survfit(..., rmean=6) for a fixed example."""
    duration = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    event = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=int)

    result = _rmst_one_group(duration, event, tau=6)

    assert result == {
        "n": 8,
        "n_events": 5,
        "rmst": 4.6458,
        "se": 0.6521,
        "ci_low": 3.3678,
        "ci_high": 5.9239,
    }


def test_two_sided_normal_p_preserves_extreme_tail():
    """Avoid cancellation from 1 - CDF for large absolute z statistics."""
    expected = float(2 * norm.sf(9.0))

    assert expected > 0
    assert _two_sided_normal_p(9.0) == expected
    assert _two_sided_normal_p(-9.0) == expected


def test_two_sided_normal_p_preserves_subnormal_tail():
    """Use log survival probability after scipy.stats.norm.sf underflows."""
    expected = 5.7708567e-316
    actual = _two_sided_normal_p(38.0)

    assert actual > 0
    assert math.isclose(actual, expected, rel_tol=1e-8, abs_tol=0)
