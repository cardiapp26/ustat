import ast
import shutil
import subprocess
import pytest
from services.syntax.association import icc, cohens_kappa, ordinal_association, paired_categorical, cronbach, correlation_pair


@pytest.mark.parametrize('fn,body',[(icc,{'rater_cols':['a','b','c'],'agreement':'consistency','unit':'average'}),(cohens_kappa,{'weights':'quadratic','level_order':['Low','High']}),(ordinal_association,{}),(paired_categorical,{}),(paired_categorical,{'method':'stuart_maxwell'}),(cronbach,{}),(correlation_pair,{'method':'pointbiserial'})])
def test_new_syntax_parses(fn,body):
    out=fn(body)
    ast.parse(out['python'])
    rscript=shutil.which('Rscript')
    if rscript is None:
        pytest.skip('Rscript not available')
    result=subprocess.run([rscript,'-e','invisible(parse(text=commandArgs(TRUE)[1]))',out['r']],capture_output=True,text=True)
    assert result.returncode == 0,result.stderr


def test_syntax_parameters_and_honest_replication():
    r=icc({'rater_cols':['a','b','c'],'agreement':'consistency','unit':'average'})['r']
    assert 'consistency' in r and 'average' in r and '"c"' in r
    assert 'check.keys = FALSE' in cronbach({})['r']
    assert 'not exact replication' in cronbach({})['r']
    assert 'weights=\'quadratic\'' in cohens_kappa({'weights':'quadratic'})['python']


def test_pointbiserial_binary_second_and_numeric_kappa_labels():
    import pandas as pd
    from scipy.stats import pointbiserialr
    out=correlation_pair({'method':'pointbiserial','var1':'score','var2':'group'})
    scope={'df':pd.DataFrame({'score':[4,8,5,9], 'group':[2,5,2,5]})}
    exec(out['python'],scope)  # nosec B102
    assert scope['binary_col'] == 'group'
    assert scope['stats'].pointbiserialr(scope['binary'],scope['d']['score']).statistic == pointbiserialr([False,True,False,True],[4,8,5,9]).statistic
    scope={'df':pd.DataFrame({'a':[1,2,1,2],'b':[1.,2.,2.,2.]})}
    exec(cohens_kappa({'rater1_col':'a','rater2_col':'b','weights':'linear','level_order':['1','2']})['python'],scope)  # nosec B102
    assert scope['d']['a'].tolist() == ['1','2','1','2']
    assert scope['d']['b'].tolist() == ['1','2','2','2']
