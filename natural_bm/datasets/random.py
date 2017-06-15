"""Tiny dataset that is just used for tests """

#%%

import numpy as np

from natural_bm.datasets.common import Dataset


#%%
def _make_random(data_type):
    
    num_pixels = 10
        
    dataset = {}
    for dset in ['train', 'valid', 'test']:
        if dset == 'train':
            num_samples = 12
        else:
            num_samples = 6
        
        if data_type == 'probability':
            dataset[dset+'.data'] = np.random.uniform(size=(num_samples, num_pixels))
        else:
            dataset[dset+'.data'] = np.random.randint(2, size=(num_samples, num_pixels))
        
        dataset[dset+'.lbl'] =  np.random.randint(2, size=(num_samples,))
            
    return dataset


#%%
class Random(Dataset):
    def __init__(self, datatype):
        super().__init__('random', datatype)

    def _create_probability(self):
        
        dataset = _make_random('probability')

        # save the dataset
        np.savez_compressed(self.savename, **dataset)

    def _create_sampled(self):
        dataset = _make_random('sampled')

        # save the dataset
        np.savez_compressed(self.savename, **dataset)

    def _create_threshold(self):
        dataset = _make_random('threshold')

        # save the dataset
        np.savez_compressed(self.savename, **dataset)
