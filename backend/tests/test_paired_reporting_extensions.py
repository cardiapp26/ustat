import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.contingency_tables import SquareTable, cochrans_q
from statsmodels.stats.proportion import proportion_confint
from conftest import make_session


def test_rank_reporting(client):
    d = np.array([0,1,-2,2,-3,4,5,-6])
    sid=make_session(pd.DataFrame({'a':d,'b':np.zeros(len(d))}))
    r=client.post('/api/repeated/wilcoxon_signed_rank',json={'session_id':sid,'col1':'a','col2':'b'})
    assert r.status_code == 200
    data=r.json()
    ranks=stats.rankdata(abs(d[d!=0]))
    assert data['W_plus'] == ranks[d[d!=0]>0].sum()
    assert data['W_minus'] == ranks[d[d!=0]<0].sum()
    assert data['z_asymptotic'] == pytest.approx(stats.wilcoxon(d,method='approx').zstatistic)
    sid=make_session(pd.DataFrame({'a':[1]*5,'b':[1]*5,'c':[3]*5}))
    data=client.post('/api/repeated/friedman',json={'session_id':sid,'columns':['a','b','c']}).json()
    assert [r['mean_rank'] for r in data['ranks']] == [1.5,1.5,3]


@pytest.mark.parametrize('method',['bowker','stuart_maxwell'])
def test_paired_categories(client,method):
    table=np.array([[5,3,1],[2,4,3],[1,2,6]])
    pairs=[(str(i),str(j)) for i in range(3) for j in range(3) for _ in range(table[i,j])]
    sid=make_session(pd.DataFrame(pairs+[(None,'2')],columns=['a','b']))
    r=client.post('/api/categorical/paired_categorical',json={'session_id':sid,'col1':'a','col2':'b','method':method})
    assert r.status_code == 200
    data=r.json(); square=SquareTable(table,shift_zeros=False)
    ref=square.symmetry() if method=='bowker' else square.homogeneity()
    assert data['p'] == pytest.approx(ref.pvalue)
    assert data['table'] == table.tolist()
    assert data['n'] == int(table.sum())


def test_wilson_and_cochran_constant_column(client):
    sid=make_session(pd.DataFrame({'a':[1]*10}))
    r=client.post('/api/categorical/one_proportion',json={'session_id':sid,'column':'a','alpha':.1})
    assert r.status_code == 200
    data=r.json()
    assert data['ci_proportion']['low'] == pytest.approx(proportion_confint(10,10,alpha=.1,method='wilson')[0])
    assert np.isfinite(data['z'])
    assert 'correct = FALSE' in data['r_code']
    mat=np.array([[0,1,1]]*20+[[0,0,1]]*5)
    sid=make_session(pd.DataFrame(mat,columns=['a','b','c']))
    r=client.post('/api/categorical/cochran_q',json={'session_id':sid,'columns':['a','b','c']})
    assert r.status_code == 200
    data=r.json()
    assert data['Q'] == pytest.approx(cochrans_q(mat).statistic,abs=1e-4)
    assert len(data['posthoc']) == 3
    assert data['posthoc'][0]['p_adj'] < .05


def test_paired_numeric_category_keys(client):
    sid=make_session(pd.DataFrame({'a':pd.Series([1,1,2,2],dtype='int64'), 'b':[1.,2.,1.,2.]}))
    r=client.post('/api/categorical/paired_categorical',json={'session_id':sid,'col1':'a','col2':'b'})
    assert r.status_code == 200
    assert r.json()['row_labels'] == ['1','2']
    assert r.json()['table'] == [[1,1],[1,1]]
