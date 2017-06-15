"""Simplified version of MNIST that is useful for demos and testing """

#%%
import numpy as np

try:
    import PIL.Image as Image
except ImportError:
    import Image

from natural_bm.datasets.common import Dataset, sample_data, threshold_data, convert2uint8
from natural_bm.datasets import mnist


#%%
class Fast(Dataset):
    def __init__(self, datatype):
        super().__init__('fast', datatype)

    def _create_probability(self):
        # Start from the MNIST probabilities
        prob = mnist.MNIST('probability')
        mnist_dataset = prob.dataset_dict

        def shrink_data(data, lbl, n_sample):
            # only keep 0's and 1's
            # subsample to 14 by 14
            # then just drop first 2, last 2 rows/cols since mainly zero

            new_data = np.zeros((2*n_sample, 10**2), dtype='float32')
            new_lbl = np.concatenate((np.zeros((n_sample, )),
                                      np.ones((n_sample, )))).astype('int32')

            index0 = np.where(lbl == 0)[0][0:n_sample]
            index1 = np.where(lbl == 1)[0][0:n_sample]
            index = np.concatenate((index0, index1))

            for i in range(new_data.shape[0]):
                img = Image.fromarray(data[index[i]].reshape((28, 28)))
                img_down = img.resize((14, 14))
                temp = np.asarray(img_down)
                temp = temp[:, 2:-2]
                temp = temp[2:-2]
                new_data[i] = temp.flatten()

            return new_data, new_lbl

        dataset = {}
        for dset in ['train', 'valid', 'test']:
            if dset == 'train':
                num_samples = 500
            else:
                num_samples = 50
            data, lbl = shrink_data(mnist_dataset[dset+'.data'],
                                    mnist_dataset[dset+'.lbl'],
                                    num_samples)
            dataset[dset+'.data'] = data
            dataset[dset+'.lbl'] = lbl

        # save the dataset
        np.savez_compressed(self.savename, **dataset)

    def _create_sampled(self):
        # Start from the probabilities
        prob = Fast('probability')
        datasets = prob.dataset_dict

        # do the sampling
        datasets = sample_data(datasets)

        # reduce precision, only need uint8
        datasets = convert2uint8(datasets)

        # Save the dataset
        np.savez_compressed(self.savename, **datasets)

    def _create_threshold(self):
        # Start from the probabilities
        prob = Fast('probability')
        datasets = prob.dataset_dict

        # threshold the data
        datasets = threshold_data(datasets)

        # reduce precision, only need uint8
        datasets = convert2uint8(datasets)

        # Save the dataset
        np.savez_compressed(self.savename, **datasets)
