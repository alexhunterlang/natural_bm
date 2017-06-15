#%%
import pytest
import numpy as np
from numpy.testing import assert_allclose

from natural_bm import initializers
from natural_bm.datasets import random
import natural_bm.backend as B
from natural_bm.utils_testing import nnet_for_testing


#%%
@pytest.mark.parametrize('shape', [(10, 10), (10, 15)], ids=['10_10', '10_15'])
@pytest.mark.parametrize('gain', [None, 0.1, 1.0, 2.0], ids=['None','0.1', '1.0', '2.0'])
def test_orthogonal(shape, gain):
    if gain is not None:
        W = initializers.orthogonal(shape, gain)
    else:
        W = initializers.orthogonal(shape)
    _, S, _ = np.linalg.svd(W)

    assert W.shape == shape

    if gain is not None:
        assert_allclose(S, gain)
    else:
        assert_allclose(S, 1.0)


#%%
@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
                         ids=['rbm', 'dbm', 'dbm_complex'])
def test_init_standard(nnet_type):

    nnet = nnet_for_testing(nnet_type)
    dataset = random.Random('probability')

    b_ls = []
    for layer in nnet.layers:
        b_ls.append(B.eval(layer.b.shape))

    W_ls = []
    for synapse in nnet.synapses:
        W_ls.append(B.eval(synapse.W.shape))

    nnet = initializers.init_standard(nnet, dataset)

    for size, layer in zip(b_ls, nnet.layers):
        assert size == B.eval(layer.b.shape)
    for size, synapse in zip(W_ls, nnet.synapses):
        assert_allclose(size, B.eval(synapse.W.shape))


#%% Main
if __name__ == '__main__':
    pytest.main([__file__])
