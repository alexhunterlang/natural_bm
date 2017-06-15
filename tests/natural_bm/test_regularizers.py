#%%
import pytest

from natural_bm import regularizers
from natural_bm import initializers, optimizers, training
from natural_bm.models import Model
from natural_bm.datasets import random
from natural_bm.callbacks import History
from natural_bm.utils_testing import nnet_for_testing


#%%
@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm'], ids=['rbm', 'dbm'])
@pytest.mark.parametrize('W_reg_type', [None, 'l1', 'l2', 'l1_l2'],
                         ids=['None', 'l1', 'l2', 'l1_l2'])
@pytest.mark.parametrize('b_reg_type', [None, 'l1', 'l2', 'l1_l2'],
                         ids=['None', 'l1', 'l2', 'l1_l2'])
def test_regularization_init(nnet_type, W_reg_type, b_reg_type):
    
    if W_reg_type is None:
        W_regularizer = None
    else:
        W_regularizer = regularizers.get(W_reg_type)
    
    if b_reg_type is None:
        b_regularizer = None
    else:
        b_regularizer = regularizers.get(b_reg_type)
    
    nnet = nnet_for_testing(nnet_type, W_regularizer, b_regularizer)
    for synapse in nnet.synapses:
        assert synapse.regularizer == W_regularizer
    for layer in nnet.layers:
        assert layer.regularizer == b_regularizer
    
    
#%%
# NOTE: the dbm tests are slow, so I am leaving it out for now
#@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
#                         ids=['rbm', 'dbm', 'dbm_complex'])
@pytest.mark.parametrize('nnet_type', ['rbm'], ids=['rbm'])
def test_regularization_fit(nnet_type):
    batch_size = 100
    n_epoch = 1
    W_reg_type = 'l1_l2'
    b_reg_type = 'l1_l2'

    data = random.Random('probability')

    nnet = nnet_for_testing(nnet_type, W_reg_type, b_reg_type)

    nnet = initializers.init_standard(nnet, data)
    optimizer = optimizers.SGD()
    trainer = training.CD(nnet)
    model = Model(nnet, optimizer, trainer)

    # test fit
    out = model.fit(data.train.data, n_epoch=n_epoch, batch_size=batch_size)
    assert isinstance(out, History)


#%%
if __name__ == '__main__':
    pytest.main([__file__])
