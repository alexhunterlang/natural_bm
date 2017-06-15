"""Utility functions only called by tests """

#%%
from natural_bm import dbm
from natural_bm import regularizers


#%%
def nnet_for_testing(nnet_type, W_reg_type=None, b_reg_type=None):
    """
    This makes some small neural networks that are useful for testing.
    
    # Arguments
        nnet_type: Str; neural network identifier.
        W_reg_type: Str or Regularizer; weight regularization.
        b_reg_type: Str or Regularizer; bias regularization.
    """

    if nnet_type == 'rbm':
        layer_size_list = [10, 9]
        topology_dict = {0: {1}}
    elif nnet_type == 'dbm':
        layer_size_list = [10, 9, 8]
        topology_dict = {0: {1}, 1: {2}}
    elif nnet_type == 'dbm_complex':
        layer_size_list = [10, 9, 8, 7]
        topology_dict = {0: {1, 3}, 1: {2}, 2: {3}}
    else:
        raise ValueError('Cannot recognize nnet_type input: {}'.format(nnet_type))
        
    if W_reg_type is None:
        W_regularizer = None
    else:
        W_regularizer = regularizers.get(W_reg_type)
    
    if b_reg_type is None:
        b_regularizer = None
    else:
        b_regularizer = regularizers.get(b_reg_type)

    nnet = dbm.DBM(layer_size_list, topology_dict,
                   W_regularizer=W_regularizer, b_regularizer=b_regularizer)

    return nnet
