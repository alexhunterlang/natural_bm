#%%
import pytest
import numpy as np
from numpy.testing import assert_allclose

from natural_bm import components
import natural_bm.backend as B


#%%
def test_NeuralNetParts():

    # initialize an abstract neural net part
    name = 'test'
    nnet = components.NeuralNetParts(name)

    # add some weights to it
    train_weights = [0, 1, 2]
    non_train_weights = [3, 4, 5, 6]
    weights = train_weights + non_train_weights

    # test two ways to add weights: item and lists
    nnet.trainable_weights = train_weights[0]
    nnet.trainable_weights = train_weights[1:]
    nnet.non_trainable_weights = non_train_weights[0]
    nnet.non_trainable_weights = non_train_weights[1:]

    # verify
    assert name == nnet.name
    assert train_weights == nnet.trainable_weights
    assert non_train_weights == nnet.non_trainable_weights
    assert weights == nnet.weights


#%%
def test_Synapse():

    # initialize
    name = 'test'
    shape = (10, 100)
    init_W = np.zeros(shape)

    synapse = components.Synapse(name, init_W)

    # verify
    assert shape == synapse.shape
    assert_allclose(init_W, synapse.W.eval())
    assert_allclose(init_W, synapse.trainable_weights[0].eval())


#%%
def test_Layer_init():

    # initialize
    name = 'test'
    shape = (10, )
    init_b = np.zeros(shape)
    W = B.variable(np.zeros((10, 10)))
    up_dict = {0: W}
    down_dict = {1: W}

    layer = components.Layer(name, init_b, up_dict, down_dict)

    # verify
    assert shape[0] == layer.dim
    assert_allclose(init_b, layer.b.eval())
    assert_allclose(init_b, layer.trainable_weights[0].eval())


#%%
def _make_layer(up_dict, down_dict):

    # initialize
    name = 'test'
    shape = (10, )
    init_b = np.zeros(shape)
    layer = components.Layer(name, init_b, up_dict, down_dict)

    return layer


#%%
@pytest.mark.parametrize('direction', ['up', 'down', 'both'],
                         ids=['up', 'down', 'both'])
def test_Layer_direction(direction):

    W = B.variable(np.zeros((10, 10)))
    
    up_dict, down_dict = {}, {}
    if direction in ['up', 'both']:   
        up_dict = {0: W, 1: W, 2: W}
    if direction in ['down', 'both']:
        down_dict = {0: W, 1: W}

    layer = _make_layer(up_dict, down_dict)

    assert direction == layer.direction
    if direction == 'both':
        assert (len(up_dict)+len(down_dict))/len(up_dict) == layer._z_up_adj
        assert (len(up_dict)+len(down_dict))/len(down_dict) == layer._z_down_adj
    else:
        assert not hasattr(layer, '_z_up_adj')
        assert not hasattr(layer, '_z_down_adj')


#%%
def test_Layer_direction_fail():
    with pytest.raises(Exception) as e_info:
        layer = _make_layer({}, {})


#%%
def test_Layer_input_z():

    mbs = 25
    pixel = 15
    
    layer_size_list = [10, pixel, 20]
    
    # initialize
    name = 's0'
    s0_shape = (layer_size_list[0], pixel)
    init_W = np.zeros(s0_shape)
    synapse = components.Synapse(name, init_W)
    up_dict = {0: synapse.W}

    name = 's2'
    s2_shape = (pixel, layer_size_list[2])
    init_W = np.zeros(s2_shape)
    synapse = components.Synapse(name, init_W)
    down_dict = {2: synapse.W}

    name = 'layer1'
    shape = (pixel, )
    init_b = np.zeros(shape)
    layer = components.Layer(name, init_b, up_dict, down_dict)

    input_ls = [np.zeros((mbs, size)) for size in layer_size_list]

    # tests
    for direction in ['both', 'up', 'down']:
        z = layer.input_z(input_ls, direction=direction)
        assert_allclose(np.zeros((mbs, pixel)), z.eval())


#%% Main
if __name__ == '__main__':
    pytest.main([__file__])
