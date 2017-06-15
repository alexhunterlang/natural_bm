#%%
import pytest
import numpy as np
from numpy.testing import assert_allclose

import natural_bm.backend as B
from natural_bm import estimators
from natural_bm.initializers import init_standard
from natural_bm.datasets.random import Random
from natural_bm.utils_testing import nnet_for_testing


#%%
def test_exact():
    nnet = nnet_for_testing('rbm')
    logZ = estimators.exact_logZ(nnet)
    assert logZ.size == 1


#%%
def test_exact_error():
    nnet = nnet_for_testing('dbm')
    with pytest.raises(Exception) as e_info:
        fail = estimators.exact_logZ(nnet)


#%%
@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
                         ids=['rbm', 'dbm', 'dbm_complex'])
@pytest.mark.parametrize('betamode', ['n_betas', 'betas'], ids=['n_betas', 'betas'])
def test_ais_init(nnet_type, betamode):

    n_runs = 5
    
    nnet = nnet_for_testing(nnet_type)
    dataset = Random('probability')
    data = B.get_value(dataset.train.data)

    if betamode == 'n_betas':
        n_betas = 100
        betas = None
    elif betamode == 'betas':
        n_betas = None
        n_betas_init = 25
        betas = np.linspace(0, 1, n_betas_init)
    else:
        raise NotImplementedError

    ais = estimators.AIS(nnet, data, n_runs, n_betas=n_betas, betas=betas)

    assert hasattr(ais, 'dbm_a')
    assert hasattr(ais, 'dbm_b')
    assert hasattr(ais, 'dbm_b_sampler')
    assert hasattr(ais, 'logZa')
    assert hasattr(ais, 'init_sample_ls')
    assert ais.n_runs == n_runs

    ais_betas = B.get_value(ais.betas)
    if betamode == 'n_betas':
        assert ais.n_betas == n_betas
        assert_allclose(ais_betas, np.linspace(0, 1, n_betas))
    elif betamode == 'betas':
        assert ais.n_betas == n_betas_init
        assert_allclose(ais_betas, betas)


#%%
@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
                         ids=['rbm', 'dbm', 'dbm_complex'])
def test_ais_run(nnet_type):

    n_runs = 5
    n_betas = 5
    
    nnet = nnet_for_testing(nnet_type)
    dataset = Random('probability')
    data = B.get_value(dataset.train.data)

    ais = estimators.AIS(nnet, data, n_runs, n_betas=n_betas)

    ais.run_logZ()
    logZ_out, logZ_low_out, logZ_high_out = ais.estimate_log_error_Z()

    assert logZ_high_out >= logZ_out
    assert logZ_low_out <= logZ_out


#%%
def test_ais_vs_exact():

    n_runs = 5
    n_betas = 5
    
    nnet = nnet_for_testing('rbm')
    dataset = Random('probability')
    data = B.get_value(dataset.train.data)
    
    nnet = init_standard(nnet, dataset)

    ais = estimators.AIS(nnet, data, n_runs, n_betas=n_betas)

    ais.run_logZ()
    logZ_out, logZ_low_out, logZ_high_out = ais.estimate_log_error_Z()

    logZ = estimators.exact_logZ(nnet)

    assert logZ >= logZ_low_out
    assert logZ <= logZ_high_out


#%%
if __name__ == '__main__':
    pytest.main([__file__])
