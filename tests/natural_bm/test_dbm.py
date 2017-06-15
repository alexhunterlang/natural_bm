#%%
import pytest
import numpy as np

from natural_bm import dbm
import natural_bm.backend as B
from natural_bm.utils_testing import nnet_for_testing


#%%
def test_prep_topology():
    topology_dict = {0: {1}}
    pairs, topology_input_dict = dbm.prep_topology(topology_dict)
    assert pairs == [(0, 1)]
    assert topology_input_dict == {1: {0}}

    topology_dict = {0: {1}, 1: {2}}
    pairs, topology_input_dict = dbm.prep_topology(topology_dict)
    assert pairs == [(0, 1), (1, 2)]
    assert topology_input_dict == {1: {0}, 2: {1}}
    
    topology_dict = {0: {1, 3}, 1: {2}, 2: {3}}
    pairs, topology_input_dict = dbm.prep_topology(topology_dict)
    assert pairs == [(0, 1), (0, 3), (1, 2), (2, 3)]
    assert topology_input_dict == {1: {0}, 2: {1}, 3: {0, 2}}


#%%
def test_prep_topology_fail():
    topology_dict = {0: {1, 2}, 1: {2}}
    with pytest.raises(Exception) as e_info:
        pairs, topology_input_dict = dbm.prep_topology(topology_dict)


#%%
@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
                         ids=['rbm', 'dbm', 'dbm_complex'])
def test_dbm_init(nnet_type):
    nnet = nnet_for_testing(nnet_type)

    layers = nnet.layers
    synapses = nnet.synapses
    parts = layers + synapses
    weights = []
    for lay in layers:
        weights.append(lay.b)
    for syn in synapses:
        weights.append(syn.W)

    assert nnet.parts == parts
    assert nnet.trainable_weights == weights
    assert nnet.non_trainable_weights == []


#%%
def _dbm_prep(nnet_type):
    mbs = 6
    nnet = nnet_for_testing(nnet_type)
    x = B.variable(np.zeros((mbs, nnet.layer_size_list[0])))
    
    return nnet, x


#%%
@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
                         ids=['rbm', 'dbm', 'dbm_complex'])
@pytest.mark.parametrize('beta', [None, 0.5, 1.0], ids=['None', '0.5', '1.0'])
def test_dbm_prop(nnet_type, beta):
   
    nnet, x = _dbm_prep(nnet_type)

    if beta is None:
        prob_ls = nnet.propup(x)
    else:
        prob_ls = nnet.propup(x, beta=beta)

    assert len(prob_ls) == len(nnet.layer_size_list)


#%%
@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
                         ids=['rbm', 'dbm', 'dbm_complex'])
@pytest.mark.parametrize('beta', [0.0, 1.0], ids=['0.0', '1.0'])
@pytest.mark.parametrize('constant', [None, 0, 1], ids=['None', '0', '1'])
def test_dbm_probability(nnet_type, beta, constant):
    
    nnet, x = _dbm_prep(nnet_type)

    constant_ls = []
    if constant is not None:
        constant_ls += [constant]

    # tests
    input_ls = nnet.propup(x, beta=beta)
    output_even_ls = nnet.prob_even_given_odd(input_ls, beta=beta, constant=constant_ls)
    output_odd_ls = nnet.prob_odd_given_even(input_ls, beta=beta, constant=constant_ls)
    assert len(output_even_ls) == len(input_ls)
    assert len(output_odd_ls) == len(input_ls)


#%%
@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
                         ids=['rbm', 'dbm', 'dbm_complex'])
@pytest.mark.parametrize('beta', [0.0, 1.0], ids=['0.0', '1.0'])
@pytest.mark.parametrize('propup', [True, False], ids=['True', 'False'])
@pytest.mark.parametrize('fe_type', ['fe', 'fe_odd', 'fe_even'],
                         ids=['fe', 'fe_odd', 'fe_even'])
def test_free_energy(nnet_type, beta, propup, fe_type):

    nnet, x = _dbm_prep(nnet_type)

    if propup:
        input_ls = nnet.propup(x, beta=beta)
    else:
        input_ls = x

    if fe_type == 'fe':
        fe = nnet.free_energy(input_ls, beta=beta)
    elif fe_type == 'fe_odd':
        fe = nnet.free_energy_sumover_odd(input_ls, beta=beta)
    elif fe_type == 'fe_even':
        fe = nnet.free_energy_sumover_even(input_ls, beta=beta)

    assert B.eval(fe).shape == B.eval(x.shape)[0]


#%% Main
if __name__ == '__main__':
    pytest.main([__file__])
