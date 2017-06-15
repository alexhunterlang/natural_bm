"""Functions to intialize weights and neural networks """

#%%
import numpy as np
import natural_bm.backend as B


#%%
def orthogonal(shape, gain=1.0):
    """Creates an orthogonal matrix """

    W = np.random.normal(size=shape)
    U, _, V = np.linalg.svd(W)
    S = np.zeros(shape)
    np.fill_diagonal(S, gain)
    W = U.dot(S).dot(V)

    return W


#%%
def init_standard(nnet, dataset):
    """My standard initialization for a neural network """

    # use orthogonal initialization of all W's
    for synapse in nnet.synapses:
        W_start = synapse.W
        shape = B.eval(W_start.shape)
        W_final = orthogonal(shape)
        B.set_value(W_start, W_final)

    # set visible bias based on data
    # Reference: A Practical Guide to Training Restricted Boltzmann Machines by Geoffrey Hinton
    pixel_mean = B.mean(dataset.train.data, axis=0)
    p = B.clip(pixel_mean, B.epsilon(), 1-B.epsilon())
    b_start = nnet.layers[0].b
    b_final = B.eval(B.log(p/(1-p)))
    B.set_value(b_start, b_final)

    return nnet
