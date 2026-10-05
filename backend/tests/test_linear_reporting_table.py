import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from conftest import make_session


@pytest.mark.parametrize('robust', [False, True])
def test_linear_anova_and_tolerance(client, robust):
    rng = np.random.default_rng(481)
    x = rng.normal(size=40)
    z = 0.6 * x + rng.normal(size=40)
    y = 2 + 1.4 * x - 0.5 * z + rng.normal(size=40)
    frame = pd.DataFrame({'x': x, 'z': z, 'y': y})
    sid = make_session(frame, f'linear_reporting_{robust}')
    response = client.post('/api/models/linear', json={
        'session_id': sid, 'outcome': 'y', 'predictors': ['x', 'z'], 'robust_se': robust,
    })
    assert response.status_code == 200, response.text
    result = response.json()
    fitted = sm.OLS(y, sm.add_constant(frame[['x', 'z']])).fit()
    rows = result['anova_table']
    assert [row['df'] for row in rows] == [2, 37, 39]
    assert rows[0]['ss'] == pytest.approx(fitted.ess)
    assert rows[1]['ss'] == pytest.approx(fitted.ssr)
    assert rows[2]['ss'] == pytest.approx(fitted.centered_tss)
    assert rows[0]['ss'] + rows[1]['ss'] == pytest.approx(rows[2]['ss'])
    assert rows[0]['ms'] / rows[1]['ms'] == pytest.approx(fitted.fvalue)
    coefficients = {c['variable']: c for c in result['coefficients']}
    assert coefficients['const']['tolerance'] is None
    # With two predictors tolerance equals 1 minus their squared correlation.
    expected = 1 - np.corrcoef(x, z)[0, 1] ** 2
    for name in ['x', 'z']:
        assert coefficients[name]['tolerance'] == pytest.approx(expected)
        assert coefficients[name]['tolerance'] * coefficients[name]['vif'] == pytest.approx(1)
    assert bool(result['anova_note']) == robust
