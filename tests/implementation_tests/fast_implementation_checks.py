"""Implementation test that is fast enough to run on the CPU. """

#%%       
import os
import numpy as np
 
from natural_bm import callbacks
from natural_bm import regularizers
from natural_bm import optimizers
from natural_bm import training
from natural_bm.models import Model
from natural_bm.dbm import DBM
from natural_bm.datasets import fast
from natural_bm.initializers import init_standard
from natural_bm.utils import standard_save_folders


#%%
def train_check():
    """
    This is a full implementation test. On the fast dataset with the training
    setup below, the final log likelihood on the validation data should be
    around -34.5 nats.
    
    I left this out of the automatic pytests since this takes around 
    2 minutes to run on a CPU and really should be run only if all the pytest
    tests were successful.
    """

    # needed datasets, standard in field is probability for train, sampled for test
    data_prob = fast.Fast('probability')
    data_sampled = fast.Fast('sampled')
    
    # rbm setup
    layer_size_list = [data_prob.train.num_pixels, 16]
    topology_dict = {0: {1}}
    W_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-5)

    # training parameters
    batch_size = 100
    n_epoch = 100   
    n_batch = np.floor(data_prob.train.num_samples/batch_size)
    optimizer_kwargs = {'lr' : 0.01,
                        'momentum' : 0.9,
                        'nesterov' : True,
                        'decay' : 1.8e-5,
                        'schedule_decay': 0.004,
                        'mom_iter_max': n_batch*(n_epoch-50)}
   
    # initialize the model
    dbm = DBM(layer_size_list, topology_dict, W_regularizer=W_regularizer)
    dbm = init_standard(dbm, data_prob)
    optimizer = optimizers.SGD(**optimizer_kwargs)
    trainer = training.PCD(dbm, batch_size=batch_size)
    model = Model(dbm, optimizer, trainer)
    
    # list of epochs to check on performance, want a few at start/end
    temp = [0, 1]
    fixed_ls = temp + [n_epoch-t for t in temp] + [n_epoch-2]
    n_runs = 100
    n_betas = 10000
    
    # prep the callbacks
    filepath = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.abspath(os.path.join(filepath, '..', '..', 'results', 'fast_checks'))
    save_dict = standard_save_folders(save_folder, overwrite=True)
    cb_csv = callbacks.CSVLogger(save_dict['csv'], separator='\t')
    cb_ais = callbacks.AISCallback(dbm, n_runs, n_betas, epoch_ls=fixed_ls)
    callbacks_ls = [cb_csv, cb_ais]

    # do the actual training
    history = model.fit(data_prob.train.data,
                        batch_size=batch_size,
                        n_epoch=n_epoch,
                        callbacks=callbacks_ls,
                        validation_data=data_sampled.valid.data)
    
    # check how training went
    val_prob = history.history['val_prob']
    best_val_prob = np.nanmax(val_prob)
    
    # Bare minimum goal
    assert best_val_prob >= -40.0
    
    # Ultimate goal
    assert best_val_prob >= -35.0


#%%
if __name__ == '__main__':
    train_check()
