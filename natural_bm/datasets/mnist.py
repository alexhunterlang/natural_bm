"""Standard MNIST digit dataset """

#%%
import os
import urllib
import gzip
import pickle
import numpy as np

from natural_bm.datasets.common import Dataset, sample_data, threshold_data, convert2uint8, convert2prob


#%%
class MNIST(Dataset):
    def __init__(self, datatype):
        super().__init__('mnist', datatype)

    def _process_load_probability(self, datasets):
        # These are stored as uint8
        datasets = convert2prob(datasets)
        return datasets

    def _create_probability(self):

        # Download MNIST probabilities
        # This will also be used to create theshold dataset
        filename = self.folder + 'mnist.pkl.gz'
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading MNIST from %s' % origin)
        _ = urllib.request.urlretrieve(origin, filename)

        # Load the dataset
        with gzip.open(filename, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)

        # Convert the dataset to a dict
        datasets = {'train.data': train_set[0],
                    'train.lbl': train_set[1],
                    'valid.data': valid_set[0],
                    'valid.lbl': valid_set[1],
                    'test.data': test_set[0],
                    'test.lbl': test_set[1]}

        # convert datasets back to uint8
        for dset in ['train', 'valid', 'test']:
            datasets[dset+'.data'] = datasets[dset+'.data']*256.0

        # To save space, will store as uint8
        datasets = convert2uint8(datasets)

        # Save the dataset
        np.savez_compressed(self.savename, **datasets)

        # cleanup the download
        os.remove(filename)

    def _create_sampled(self):
        # Start from the probabilities
        prob = MNIST('probability')
        datasets = prob.dataset_dict

        # do the sampling
        datasets = sample_data(datasets)

        # reduce precision, only need uint8
        datasets = convert2uint8(datasets)

        # Save the dataset
        np.savez_compressed(self.savename, **datasets)

    def _create_threshold(self):
        # Start from the probabilities
        prob = MNIST('probability')
        datasets = prob.dataset_dict

        # threshold the data
        datasets = threshold_data(datasets)

        # reduce precision, only need uint8
        datasets = convert2uint8(datasets)

        # Save the dataset
        np.savez_compressed(self.savename, **datasets)
