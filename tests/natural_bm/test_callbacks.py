#%%
import pytest
import os
from csv import Sniffer

from natural_bm import callbacks
from natural_bm import optimizers
from natural_bm import training
from natural_bm.models import Model
from natural_bm.datasets import random
from natural_bm.utils_testing import nnet_for_testing


#%%
@pytest.mark.parametrize('sep', [',', '\t'], ids=['csv', 'tsv'])
def test_CSVLogger(sep):
    """
    This test is a slight modification of test_CSVLogger from
    https://github.com/fchollet/keras/blob/master/tests/keras/test_callbacks.py
    """
    
    nnet = nnet_for_testing('rbm')
    data = random.Random('probability')
    
    batch_size = 6
    n_epoch = 1

    if sep == '\t':
        filepath = 'log.tsv'
    elif sep == ',':
        filepath = 'log.csv'

    def make_model(dbm, data):

        optimizer = optimizers.SGD()

        trainer = training.CD(dbm)

        model = Model(dbm, optimizer, trainer)

        return model

    # case 1, create new file with defined separator
    model = make_model(nnet, data)
    cbks = [callbacks.CSVLogger(filepath, separator=sep)]
    history = model.fit(data.train.data,
                        batch_size=batch_size,
                        n_epoch=n_epoch,
                        callbacks=cbks,
                        validation_data=data.valid.data)

    assert os.path.exists(filepath)
    with open(filepath) as csvfile:
        dialect = Sniffer().sniff(csvfile.read())
    assert dialect.delimiter == sep
    del model
    del cbks

    # case 2, append data to existing file, skip header
    model = make_model(nnet, data)
    cbks = [callbacks.CSVLogger(filepath, separator=sep, append=True)]
    history = model.fit(data.train.data,
                        batch_size=batch_size,
                        n_epoch=n_epoch,
                        callbacks=cbks,
                        validation_data=data.valid.data)

    # case 3, reuse of CSVLogger object
    history = model.fit(data.train.data,
                        batch_size=batch_size,
                        n_epoch=n_epoch,
                        callbacks=cbks,
                        validation_data=data.valid.data)

    import re
    with open(filepath) as csvfile:
        output = " ".join(csvfile.readlines())
        assert len(re.findall('epoch', output)) == 1

    os.remove(filepath)


#%% Main
if __name__ == '__main__':
    pytest.main([__file__])
