"""Long demo that should only by run on the GPU.

Relevant literature comparison is:
On the Quantitative Analysis of Deep Belief Networks by Salakhutdinov and Murray.
Specifically, this code should beat their resuilts for CD25(500)
which achieved a test log likelihood of -86.34 on MNIST.

If the code is ran for 1000 epochs, should definitely have validation
log likelihood > -85. If the code is only ran for 250 epochs, expect a log
likelihood of > 86.5.
"""

#%%
import os
import numpy as np
 
from natural_bm.models import Model
from natural_bm.dbm import DBM
from natural_bm.datasets import mnist
from natural_bm import optimizers
from natural_bm import training
from natural_bm.initializers import init_standard
from natural_bm import callbacks
from natural_bm import regularizers
from natural_bm.utils import standard_save_folders


#%%
def run_rbm():

    # parameters    
    batch_size = 100
    n_epoch = 1000 # 1000 for true literature comparison, but 250 should be good
    results_folder = 'long_demo'
    
    # needed datasets, standard in field is probability for train, sampled for test    
    data_train = mnist.MNIST('probability')
    data_valid = mnist.MNIST('sampled')
    
    # this is a large neural network
    layer_size_list = [784, 500]
    topology_dict = {0: {1}}
    
    # training parameters 
    # lr rate decays from 0.01 to 0.001 over training epochs
    # momentum is ramped up (controlled by schedule_decay) to max
    # and then ramped back down to a smaller value for the last 50 epochs
    n_batch = np.floor(data_train.train.num_samples/batch_size)
    decay = (1/0.1-1)/(n_epoch*data_train.train.num_samples/batch_size)
    optimizer_kwargs = {'lr' : 0.01,
                        'momentum' : 0.9,
                        'nesterov' : True,
                        'decay' : decay,
                        'schedule_decay': 0.004,
                        'mom_iter_max': n_batch*(n_epoch-50)}
   
    # create and initialize the rbm
    W_regularizer = regularizers.l2(l=2e-4)
    dbm = DBM(layer_size_list, topology_dict, W_regularizer=W_regularizer)
    dbm = init_standard(dbm, data_train)
    
    # make the model    
    optimizer = optimizers.SGD(**optimizer_kwargs)
    trainer = training.PCD(dbm, nb_pos_steps=25, nb_neg_steps=25, batch_size=batch_size)
    model = Model(dbm, optimizer, trainer)
    
    # prepare output paths
    filepath = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.abspath(os.path.join(filepath, '..', 'results', results_folder))
    save_dict = standard_save_folders(save_folder, overwrite=True)
  
    # epochs to monitor
    temp = [0, 1, 5, 10]
    fixed_ls = temp + [n_epoch-t for t in temp] + [n_epoch-2]
    epoch_ls = list(set(list(range(0, n_epoch, 25)) + fixed_ls))
    
    # these callbacks monitor progress
    cb_csv = callbacks.CSVLogger(save_dict['csv'], separator='\t')
    cb_ais = callbacks.AISCallback(dbm, 1000, 30000, epoch_ls=epoch_ls)
    cb_period = callbacks.PeriodicSave(save_dict['weights'], epoch_ls,
                                       opt_path=save_dict['opt_weights'])
    cb_opt = callbacks.OptimizerSpy()
    cb_plot = callbacks.PlotCallback(save_dict['plots'], save_dict['csv'])
    cb_summary = callbacks.SummaryCallback(save_folder, save_dict['csv'])
    callbacks_ls = [cb_csv, cb_ais, cb_period, cb_opt, cb_plot, cb_summary]

    # do the actual training
    history = model.fit(data_train.train.data,
                        batch_size=batch_size,
                        n_epoch=n_epoch,
                        callbacks=callbacks_ls,
                        validation_data=data_valid.valid.data)
    

#%%
if __name__ == '__main__':
    run_rbm()
