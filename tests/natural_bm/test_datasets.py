#%%
import os
import numpy as np
import pytest
from natural_bm.datasets.common import threshold_data
from natural_bm.datasets import mnist, svhn, fast
import natural_bm.backend as B


#%%
def test_treshold_data():

    datasets = {'train.data': 0.6*np.ones((100, 10))}
    datasets = threshold_data(datasets, threshold=None)
    assert np.all(datasets['train.data'] == 1.0)

    datasets = {'train.data': 0.6*np.ones((100, 10))}
    datasets = threshold_data(datasets, threshold=0.7)
    assert np.all(datasets['train.data'] == 0.0)

    datasets = {'train.data': 0.6*np.ones((100, 10))}
    threshold = np.concatenate((0.7*np.ones((5,)), 0.5*np.ones((5,))))
    datasets = threshold_data(datasets, threshold=threshold)
    verify = np.concatenate((np.zeros((100, 5)), np.ones((100, 5))), axis=1)
    assert np.all(datasets['train.data'] == verify)


#%% 
def test_mnist():

    name = 'mnist'
    datatype_ls = ['probability', 'sampled', 'threshold']

    # delete files if they exist
    filepath = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.abspath(os.path.join(filepath, '..', '..', 'data'))
    print(folder)
    for datatype in datatype_ls:
        filename = os.path.join(folder, name + '_' + datatype + '.npz')
        try:
            os.remove(filename)
        except OSError:
            pass

    # this checks on creating and loading datasets
    for datatype in datatype_ls:
        data = mnist.MNIST(datatype)

    # this checks on loading existing
    for datatype in datatype_ls:
        data = mnist.MNIST(datatype)


#%% 
def test_fast():

    name = 'fast'
    datatype_ls = ['probability', 'sampled', 'threshold']

    # delete files if they exist
    filepath = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.abspath(os.path.join(filepath, '..', '..', 'data'))
    for datatype in datatype_ls:
        filename = os.path.join(folder, name + '_' + datatype + '.npz')
        try:
            os.remove(filename)
        except OSError:
            pass

    train_samples = 1000
    other_samples = 100

    # this checks on creating and loading datasets
    for datatype in datatype_ls:
        data = fast.Fast(datatype)
        assert B.eval(data.train.data).shape[0] == train_samples
        assert B.eval(data.valid.data).shape[0] == other_samples
        assert B.eval(data.test.data).shape[0] == other_samples
        assert B.eval(data.train.lbl).shape[0] == train_samples
        assert B.eval(data.valid.lbl).shape[0] == other_samples
        assert B.eval(data.test.lbl).shape[0] == other_samples

    # this checks on loading existing
    for datatype in datatype_ls:
        data = fast.Fast(datatype)
        assert B.eval(data.train.data).shape[0] == train_samples
        assert B.eval(data.valid.data).shape[0] == other_samples
        assert B.eval(data.test.data).shape[0] == other_samples
        assert B.eval(data.train.lbl).shape[0] == train_samples
        assert B.eval(data.valid.lbl).shape[0] == other_samples
        assert B.eval(data.test.lbl).shape[0] == other_samples


#%%
def longtest_svhn(__file__):
    """
    This test is internet dependent and requires a large downloand.
    Since it is slow, I did not include it in auto pytesting.
    """
    
    name = 'svhn'
    datatype_ls = ['probability',  'threshold']

    # delete files if they exist
    filepath = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.abspath(os.path.join(filepath, '..', '..', 'data'))
    for datatype in datatype_ls:
        filename = os.path.join(folder, name + '_' + datatype + '.npz')
        try:
            os.remove(filename)
        except OSError:
            pass

    # this checks on creating and loading datasets
    for datatype in datatype_ls:
        data = svhn.SVHN(datatype)

    # this checks on loading existing
    for datatype in datatype_ls:
        data = svhn.SVHN(datatype)


#%%
if __name__ == '__main__':
    # This test will take a couple of minutes depending on your internet speed
    # longtest_svhn(__file__)

    pytest.main([__file__])
