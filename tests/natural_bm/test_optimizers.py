#%%
import pytest
import numpy as np

import natural_bm.backend as B
from natural_bm import optimizers
from natural_bm.datasets import random


#%%
def _test_optimizer(optimizer):
    
    mbs = 10

    dataset = random.Random('probability')
    data = B.eval(dataset.train.data[0:mbs])
    pixels = data.shape[1]

    W0 = B.variable(np.random.normal(size=(pixels,)), dtype=B.floatx(), name='W0')
    W1 = B.variable(np.random.normal(size=(pixels,)), dtype=B.floatx(), name='W1')
    params = [W0, W1]
    inputs = B.placeholder((mbs, pixels), dtype=B.floatx())
    loss = B.sum(B.dot(inputs, B.square(W0)+B.square(W1)))

    updates = optimizer.get_updates(params, loss)

    f = B.function([inputs], [loss], updates=updates)

    output = f(data)
    assert len(output) == 1
    assert output[0].size == 1


#%%
def test_sgd():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    _test_optimizer(sgd)


#%%
def test_adam():
    _test_optimizer(optimizers.Adam())
    _test_optimizer(optimizers.Adam(decay=1e-3))


#%%
def test_nadam():
    _test_optimizer(optimizers.Nadam())


#%%
def test_clipnorm():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=0.5)
    _test_optimizer(sgd)


#%%
def test_clipvalue():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, clipvalue=0.5)
    _test_optimizer(sgd)


#%%
if __name__ == '__main__':
    pytest.main([__file__])
