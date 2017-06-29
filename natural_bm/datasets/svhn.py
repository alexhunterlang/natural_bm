"""Google Street View House Number Dataset

These are originally color images that I converted to grayscale to make
it simpler to identify as probabilities. I also created a threshold version
of the probabilities, but I have not made a sampled version.
"""

#%%
import os
import urllib
import numpy as np
import scipy.io as io
import skimage.filters as filters

from natural_bm.datasets.common import Dataset, threshold_data, convert2uint8, convert2prob
from natural_bm.utils import scale_to_unit_interval


#%%
class SVHN(Dataset):

    def __init__(self, datatype):
        super().__init__('svhn', datatype)

    def _process_load_probability(self, datasets):
        # These are stored as uint8
        datasets = convert2prob(datasets)
        return datasets

    def _create_probability(self):
        # Prep for the downloads
        dataset_ls = ['train_32x32.mat', 'test_32x32.mat']
        filename_ls = [os.path.join(self.folder, d) for d in dataset_ls]
        url_base = 'http://ufldl.stanford.edu/housenumbers/'
        url_ls = [url_base + d for d in dataset_ls]

        # Download
        for u, f in zip(url_ls, filename_ls):
            print('Downloading SVHN from %s' % u)
            urllib.request.urlretrieve(u, f)

        # load up the data
        datasets = {}
        for i, dset in enumerate(['train', 'test']):
            temp = io.loadmat(filename_ls[i])
            datasets[dset+'.data'] = temp['X']
            datasets[dset+'.lbl'] = temp['y']

        # make the validation set
        # nothing super scientific about this choice
        # about half the size of the test data, leaves train as round number
        num_valid = 13257
        num_train = datasets['train.data'].shape[3]
        # need random split since original order is correlated
        np_rng = np.random.RandomState(0)
        valid_index = np_rng.choice(np.arange(num_train), size=num_valid, replace=False)    
        train_set = set(list(range(num_train)))
        valid_set = set(valid_index)
        train_index = np.array(list(train_set.difference(valid_set)))
        datasets['valid.data'] = datasets['train.data'][:, :, :, valid_index]
        datasets['valid.lbl'] = datasets['train.lbl'][valid_index]
        datasets['train.data'] = datasets['train.data'][:, :, :, train_index]
        datasets['train.lbl'] = datasets['train.lbl'][train_index]

        # standard labels to numpy 0 indexing
        for dset in ['train', 'valid', 'test']:
            datasets[dset+'.lbl'] = (datasets[dset+'.lbl']-1)

        # standardize the scale
        for dset in ['train', 'valid', 'test']:
            datasets[dset+'.data'] = scale_to_unit_interval(datasets[dset+'.data'])

        # convert to samples, features, channels order
        for dset in ['train', 'valid', 'test']:
            data = datasets[dset+'.data']
            data = np.rollaxis(data, 3)
            datasets[dset+'.data'] = data.reshape((-1, 32**2, 3))

        # convert to grayscale
        # https://en.wikipedia.org/wiki/Grayscale
        gray = [0.299, 0.587, 0.114]
        for dset in ['train', 'valid', 'test']:
            data = datasets[dset+'.data']
            datasets[dset+'.data'] = data.dot(gray)

        # To save space, will store as uint8
        for dset in ['train', 'valid', 'test']:
            data = np.floor(255*datasets[dset+'.data'])
            datasets[dset+'.data'] = data.astype('uint8')

        # Save the dataset
        np.savez_compressed(self.savename, **datasets)

        # cleanup the downloads
        for f in filename_ls:
            os.remove(f)


    def _create_threshold(self):
        # https://en.wikipedia.org/wiki/Otsu%27s_method
        # just 0.5 threshold leads to terrible images

        # Start from the probabilities
        prob = SVHN('probability')
        datasets = prob.dataset_dict

        # calculate the thresholds
        thresh = {}
        for dset in ['train', 'valid', 'test']:
            data = datasets[dset+'.data']
            temp = np.zeros((data.shape[0], 1))
            for i in range(data.shape[0]):
                img = data[i]
                temp[i, 0] = filters.threshold_otsu(img)
            thresh[dset+'.data'] = temp

        # threshold the data
        datasets = threshold_data(datasets)

        # reduce precision, only need uint8
        datasets = convert2uint8(datasets)

        # Save the dataset
        np.savez_compressed(self.savename, **datasets)
