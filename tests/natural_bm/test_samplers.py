#%%
import pytest
import numpy as np
from numpy.testing import assert_allclose

from natural_bm import samplers
import natural_bm.backend as B
from natural_bm.utils_testing import nnet_for_testing


#%%
@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
                         ids=['rbm', 'dbm', 'dbm_complex'])
@pytest.mark.parametrize('constant', [None, 0, 1], ids=['None', '0', '1'])
def test_Sampler_init(nnet_type, constant):
    beta = 1.0
    nnet = nnet_for_testing(nnet_type)

    constant_ls = []
    if constant is not None:
        constant_ls = [constant]

    sampler = samplers.Sampler(nnet)
    sampler.set_param(beta=beta, constant=constant_ls)

    assert sampler.beta == beta
    assert sampler.constant == constant_ls


#%%
@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
                         ids=['rbm', 'dbm', 'dbm_complex'])
@pytest.mark.parametrize('constant', [None, 0, 1], ids=['None', '0', '1'])
@pytest.mark.parametrize('sampler_type', ['probability', 'sample', 'sample_inputs'],
                         ids=['probability', 'sample', 'sample_inputs'])
def test_base_Sampler(nnet_type, constant, sampler_type):
    beta = 1.0
    nnet = nnet_for_testing(nnet_type)
    batch_size = 30

    constant_ls = []
    if constant is not None:
        constant_ls = [constant]

    sampler = samplers.Sampler(nnet)

    input_ls = [B.variable(np.ones((batch_size, size))) for size in nnet.layer_size_list]

    sampler.set_param(beta=beta, constant=constant_ls)

    if sampler_type == 'probability':
        prob_ls = sampler.probability(*input_ls)
    elif sampler_type == 'sample':
        prob_ls = sampler.sample(*input_ls)
    elif sampler_type == 'sample_inputs':
        prob_ls = sampler.sample_inputs(*input_ls)
    else:
        raise NotImplementedError

    assert len(prob_ls) == len(input_ls)
    for i, p in enumerate(prob_ls):
        if i in constant_ls:
            assert p == input_ls[i]
        else:
            m = np.ones((batch_size, nnet.layer_size_list[i]))
            pp = B.eval(p)
            if sampler_type == 'sample':
                assert_allclose((pp+np.logical_not(pp)), m)
            else:
                assert_allclose(pp, 0.5*m)


#%%
@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
                         ids=['rbm', 'dbm', 'dbm_complex'])
@pytest.mark.parametrize('constant', [None, 0, 1], ids=['None', '0', '1'])
@pytest.mark.parametrize('sampler_type', ['meanfield', 'gibbs', 'gibbs_prob'],
                         ids=['meanfield', 'gibbs', 'gibbs_prob'])
def test_Sampler(nnet_type, constant, sampler_type):
    beta = 1.0
    nnet = nnet_for_testing(nnet_type)
    batch_size = 30

    constant_ls = []
    if constant is not None:
        constant_ls = [constant]

    if sampler_type == 'meanfield':
        sampler = samplers.Meanfield(nnet)
    elif sampler_type == 'gibbs':
        sampler = samplers.Gibbs(nnet)
    elif sampler_type == 'gibbs_prob':
        sampler = samplers.GibbsProb(nnet)
    else:
        raise NotImplementedError

    input_ls = [np.ones((batch_size, size)) for size in nnet.layer_size_list]
    input_ls_placeholder = [B.placeholder(in_np.shape) for in_np in input_ls]

    sampler.set_param(beta=beta, constant=constant_ls)

    prob_ls, updates = sampler.run_chain(input_ls_placeholder, beta=beta,
                                         constant=constant_ls)
    fn = B.function(input_ls_placeholder, prob_ls, updates=updates)
    prob_ls = fn(*input_ls)

    assert len(prob_ls) == len(input_ls)
    for i, p in enumerate(prob_ls):
        if i in constant_ls:
            assert_allclose(p, input_ls[i])
        else:
            m = np.ones((batch_size, nnet.layer_size_list[i]))
            if sampler_type == 'gibbs':
                assert_allclose((p+np.logical_not(p)), m)
            else:
                assert_allclose(p, 0.5*m)


#%% Main
if __name__ == '__main__':
    pytest.main([__file__])
