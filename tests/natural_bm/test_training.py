#%%
import pytest
import numpy as np

from natural_bm import training
import natural_bm.backend as B
from natural_bm.utils_testing import nnet_for_testing


#%%
def _init_training(training_type, nnet, nb_pos_steps, nb_neg_steps, batch_size):
    if training_type == 'cd':
        train = training.CD(nnet, nb_pos_steps=nb_pos_steps, nb_neg_steps=nb_neg_steps)
    elif training_type == 'pcd':
        train = training.PCD(nnet, nb_pos_steps=nb_pos_steps, nb_neg_steps=nb_neg_steps,
                             batch_size=batch_size)
    else:
        raise NotImplementedError

    return train


#%%
def _init_data(batch_size):

    shape = (batch_size, 10)
    inputs = B.placeholder(shape=shape)
    data = np.random.uniform(size=shape)

    return inputs, data


#%%
@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
                         ids=['rbm', 'dbm', 'dbm_complex'])
@pytest.mark.parametrize('training_type', ['cd', 'pcd'], ids=['cd', 'pcd'])
def test_training_init(nnet_type, training_type):
    nb_pos_steps = 2
    nb_neg_steps = 2
    batch_size = 6

    nnet = nnet_for_testing(nnet_type)
    train = _init_training(training_type, nnet, nb_pos_steps, nb_neg_steps, batch_size)

    assert hasattr(train, 'pos_sampler')
    assert hasattr(train, 'neg_sampler')

    if nnet_type == 'rbm':
        nb_pos_steps = 1

    assert train.nb_pos_steps == nb_pos_steps
    assert train.nb_neg_steps == nb_neg_steps


#%%
@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
                         ids=['rbm', 'dbm', 'dbm_complex'])
@pytest.mark.parametrize('training_type', ['cd', 'pcd'], ids=['cd', 'pcd'])
@pytest.mark.parametrize('pos_neg', ['pos', 'neg'], ids=['pos', 'neg'])
def test_training_prob(nnet_type, training_type, pos_neg):
    nb_pos_steps = 2
    nb_neg_steps = 2
    batch_size = 6

    nnet = nnet_for_testing(nnet_type)
    train = _init_training(training_type, nnet, nb_pos_steps, nb_neg_steps, batch_size)
    inputs, data = _init_data(batch_size)
    
    prob = train.pos_stats(inputs)
    if pos_neg == 'neg':
        prob = train.neg_stats(prob)
    
    fn = B.function([inputs], prob, updates=train.updates)

    output = fn(data)
    assert len(output) == len(nnet.layers)
    for out, size in zip(output, nnet.layer_size_list):
        assert out.shape == (batch_size, size)


#%%
@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
                         ids=['rbm', 'dbm', 'dbm_complex'])
@pytest.mark.parametrize('training_type', ['cd', 'pcd'], ids=['cd', 'pcd'])
def test_training_loss(nnet_type, training_type):
    nb_pos_steps = 2
    nb_neg_steps = 2
    batch_size = 6

    nnet = nnet_for_testing(nnet_type)
    train = _init_training(training_type, nnet, nb_pos_steps, nb_neg_steps, batch_size)
    inputs, data = _init_data(batch_size)

    loss = train.loss_fn()
    fn = B.function([inputs], loss(inputs), updates=train.updates)

    output = fn(data)
    assert output.size == 1


#%%
if __name__ == '__main__':
    pytest.main([__file__])
