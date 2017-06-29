"""Classes for storing datasets """

#%%
import os
import numpy as np

import natural_bm.backend as B
from natural_bm.utils import scale_to_unit_interval


#%% 
def sample_data(datasets, seed=0):
    """Randomly samples probabilities to binary variables """
    
    np_rng = np.random.RandomState(seed)
    
    for k, v in datasets.items():
        if k.endswith('.data'):
            datasets[k] = 1.0*(v > np_rng.uniform(size=v.shape))

    return datasets


#%%
def threshold_data(datasets, threshold=None):
    """Deterministically thresholds probabilities to binary variables """

    for k, v in datasets.items():
        if k.endswith('.data'):
            if threshold is None:
                thresh = 0.5
            elif isinstance(threshold, dict):
                thresh = threshold[k]
            else:
                thresh = threshold
            datasets[k] = 1.0*(v > thresh)

    return datasets


#%%
def convert2uint8(datasets):
    """Converst to unit8 """
    for k, v in datasets.items():
        datasets[k] = v.astype('uint8')

    return datasets

#%%
def convert2prob(datasets):
    """Converts to probabilities """
    for k, v in datasets.items():
        if k.endswith('.data'):
            datasets[k] = scale_to_unit_interval(v)

    return datasets

#%%
class Dataset:
    """Class that organizes a given dataset """
    def __init__(self, name, datatype):
        self.name = name
        self.datatype = datatype

        filepath = os.path.dirname(os.path.abspath(__file__))  
        self.folder = os.path.abspath(os.path.join(filepath, '..', '..','data'))
        self.savename = os.path.join(self.folder, self.name + '_' + self.datatype)
        self.filename = self.savename + '.npz'

        self.dataset_dict = self._load()
        self.train = Dataslice('train', self.dataset_dict['train.data'], self.dataset_dict['train.lbl'])
        self.valid = Dataslice('valid', self.dataset_dict['valid.data'], self.dataset_dict['valid.lbl'])
        self.test = Dataslice('test', self.dataset_dict['test.data'], self.dataset_dict['test.lbl'])

    def _load(self):
        if not os.path.isfile(self.filename):
            if self.datatype == 'probability':
                self._create_probability()
            elif self.datatype == 'sampled':
                self._create_sampled()
            elif self.datatype == 'threshold':
                self._create_threshold()
            else:
                msg = 'Datatype, {}, is not a recognized keyword'.format(self.datatype)
                raise ValueError(msg)

        # load up the data
        temp = np.load(self.filename)
        dataset = {}
        for k, v in temp.items():
            dataset[k] = v

        # perform standard processing
        if self.datatype == 'probability':
            dataset = self._process_load_probability(dataset)
        elif self.datatype == 'sampled':
            dataset = self._process_load_sampled(dataset)
        elif self.datatype == 'threshold':
            dataset = self._process_load_threshold(dataset)

        return dataset

    def _process_load_probability(self, dataset):
        return dataset

    def _process_load_sampled(self, dataset):
        return dataset

    def _process_load_threshold(self, dataset):
        return dataset

    def _create_probability(self):
        msg = 'The dataset, {}, has no method for datatype: {}'.format(self.name, self.datatype)
        raise NotImplementedError(msg)

    def _create_sampled(self):
        msg = 'The dataset, {}, has no method for datatype: {}'.format(self.name, self.datatype)
        raise NotImplementedError(msg)

    def _create_threshold(self):
        msg = 'The dataset, {}, has no method for datatype: {}'.format(self.name, self.datatype)
        raise NotImplementedError(msg)


#%%
class Dataslice:
    """Specific dataslice of dataset such as training, validation, or testing. """
    def __init__(self, slicetype, data, lbl):
        self.slicetype = slicetype
        self.data = B.variable(data, name=self.slicetype+'.data')
        self.lbl = B.variable(lbl, name=self.slicetype+'.lbl')

        self.num_samples = data.shape[0]
        self.num_pixels = data.shape[1]
        self.num_classes = np.unique(lbl).size

    def get_index_examples(self, num_each):
        # Returns examples of each label
        index = []
        for i in range(self.num_cat):
            index += list(np.where(i == self.lbl)[0][0:num_each])

        return np.array(index, dtype=B.intx())
