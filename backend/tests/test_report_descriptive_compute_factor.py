import numpy as np
import pandas as pd
import pytest
from scipy import stats
from conftest import make_session
from services import store


def test_descriptive_additional_measures(client):
    sid = make_session(pd.DataFrame({'x': [1., 2., 4., 8., 10.], 'zero': [-2., -1., 0., 1., 2.]}), 'report_descriptive')
    result = client.get(f'/api/stats/{sid}/descriptive').json()
    x = result['x']
    assert x['cv'] == pytest.approx(stats.variation([1, 2, 4, 8, 10], ddof=1))
    assert x['cv_percent'] == pytest.approx(100 * x['cv'])
    assert x['quartile_deviation'] == pytest.approx((x['q3'] - x['q1']) / 2)
    assert x['harmonic_mean'] == pytest.approx(stats.hmean([1, 2, 4, 8, 10]))
    assert x['skewness'] == pytest.approx(stats.skew([1, 2, 4, 8, 10], bias=False))
    assert result['zero']['cv'] is None
    assert result['zero']['harmonic_mean'] is None


@pytest.mark.parametrize('transform,values,expected', [
    ('tscore', [1., 2., 3.], [40., 50., 60.]),
    ('reciprocal', [1., 2., -4.], [1., .5, -.25]),
    ('arcsin_sqrt', [0., .5, 1.], [0., np.pi / 4, np.pi / 2]),
])
def test_additional_transforms(client, transform, values, expected):
    sid = make_session(pd.DataFrame({'x': values}), 'report_transform')
    response = client.post(f'/api/compute/{sid}/transform', json={'source_col': 'x', 'new_col': 'out', 'transform': transform})
    assert response.status_code == 200, response.text
    assert store.get(sid)['out'].tolist() == pytest.approx(expected)


@pytest.mark.parametrize('transform,values', [('reciprocal', [0., 1., 2.]), ('arcsin_sqrt', [-.1, .5, 1.]), ('arcsin_sqrt', [0., .5, 100.]), ('tscore', [1., 1., 1.])])
def test_transform_domain_rejection(client, transform, values):
    sid = make_session(pd.DataFrame({'x': values}), 'report_invalid')
    response = client.post(f'/api/compute/{sid}/transform', json={'source_col': 'x', 'new_col': 'out', 'transform': transform})
    assert response.status_code == 422
    assert 'out' not in store.get(sid)


def test_factor_determinant_and_ml_label(client):
    rng = np.random.default_rng(42)
    latent = rng.normal(size=100)
    data = pd.DataFrame({f'x{i}': latent + rng.normal(size=100) for i in range(3)})
    sid = make_session(data, 'report_factor')
    response = client.post('/api/factor/factor_pca', json={'session_id': sid, 'items': list(data), 'extraction': 'efa', 'rotation': 'none', 'n_factors': 1})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['suitability']['correlation_determinant'] == pytest.approx(np.linalg.det(data.corr()))
    assert 'Maximum Likelihood' in result['extraction_method']
    assert 'fm = "ml"' in result['r_code']
    assert 'different optimization' in result['replication_note']


def test_descriptive_shape_standard_errors_match_normality(client):
    from routers.stats.normality import _shape
    from services.distribution_shape import distribution_shape
    values = np.array([1., 2., 4., 8., 10.])
    sid = make_session(pd.DataFrame({'x': values}), 'report_shape_se')
    result = client.get(f'/api/stats/{sid}/descriptive').json()['x']
    summary = client.get(f'/api/stats/{sid}/column_summary', params={'column': 'x', 'kind': 'numeric'}).json()
    expected = distribution_shape(values)
    assert _shape is distribution_shape
    for key in ['skewness', 'kurtosis', 'skew_se', 'kurt_se', 'skew_z', 'kurt_z']:
        assert result[key] == pytest.approx(expected[key])
        assert summary[key] == pytest.approx(expected[key])
    assert expected['skew_se'] == pytest.approx(0.9128709291752769)
    assert expected['kurt_se'] == pytest.approx(2.)


def test_descriptive_shape_standard_errors_undefined_for_constant(client):
    sid = make_session(pd.DataFrame({'x': [2., 2., 2., 2.]}), 'report_constant_shape')
    result = client.get(f'/api/stats/{sid}/descriptive').json()['x']
    assert result['skew_se'] is None
    assert result['kurt_se'] is None
