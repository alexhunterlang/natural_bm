#%%
import pytest

from natural_bm import initializers, optimizers, training
from natural_bm.models import Model
from natural_bm.datasets import random
from natural_bm.callbacks import History
from natural_bm.utils_testing import nnet_for_testing


#%%
# NOTE: the dbm tests are slow, so I am leaving it out for now
#@pytest.mark.parametrize('nnet_type', ['rbm', 'dbm', 'dbm_complex'],
#                         ids=['rbm', 'dbm', 'dbm_complex'])
@pytest.mark.parametrize('nnet_type', ['rbm'], ids=['rbm'])
def test_models(nnet_type):
    batch_size = 6
    n_epoch = 1

    data = random.Random('probability')

    nnet = nnet_for_testing(nnet_type)

    nnet = initializers.init_standard(nnet, data)
    optimizer = optimizers.SGD()
    trainer = training.CD(nnet, nb_pos_steps=2, nb_neg_steps=2)
    model = Model(nnet, optimizer, trainer)

    # test train_on_batch
    out = model.train_on_batch(data.train.data)
    assert out.size == 1

    # predict_on_batch
    out = model.predict_on_batch(data.valid.data)
    assert out.size == 1

    # test fit
    out = model.fit(data.train.data, n_epoch=n_epoch, batch_size=batch_size)
    assert isinstance(out, History)

    # test validation data
    out = model.fit(data.train.data, n_epoch=n_epoch, batch_size=batch_size,
                    validation_data=data.valid.data)
    assert isinstance(out, History)


#%%
if __name__ == '__main__':
    pytest.main([__file__])
