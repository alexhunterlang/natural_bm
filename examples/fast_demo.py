"""Fast demo that runs very fast even on a CPU """

#%%
import os
 
from natural_bm.models import Model
from natural_bm.dbm import DBM
from natural_bm.datasets import fast
from natural_bm import optimizers
from natural_bm import training
from natural_bm.initializers import init_standard
from natural_bm import callbacks
from natural_bm.utils import standard_save_folders


#%%
def run_rbm(nnet_type='rbm'):

    # parameters
    batch_size = 100
    n_epoch = 2
    results_folder = 'fast_demo/'
    
    # just makes a tiny neural network
    if nnet_type == 'rbm':
        layer_size_list = [100, 10]
        topology_dict = {0: {1}}
    elif nnet_type == 'dbm':
        layer_size_list = [100, 10, 5]
        topology_dict = {0: {1}, 1: {2}}
    elif nnet_type == 'dbm_complex':
        layer_size_list = [100, 10, 5, 2]
        topology_dict = {0: {1, 3}, 1: {2}, 2: {3}}
    else:
        raise NotImplementedError
    
    # this is a small dataset useful for demos        
    data = fast.Fast('probability')
   
    # create and initialize the rbm
    dbm = DBM(layer_size_list, topology_dict)
    dbm = init_standard(dbm, data)
    
    # make the model
    optimizer = optimizers.SGD()
    trainer = training.CD(dbm)
    model = Model(dbm, optimizer, trainer)
    
    # prepare output paths
    path = os.path.abspath(__file__)
    save_folder = '/'.join(path.split('/')[:-2])+'/results/'+results_folder
    save_dict = standard_save_folders(save_folder, overwrite=True)
  
    # these callbacks monitor progress
    cb_csv = callbacks.CSVLogger(save_dict['csv'], separator='\t')
    cb_ais = callbacks.AISCallback(dbm, 100, 1000, epoch_ls=[0, 1])
    cb_period = callbacks.PeriodicSave(save_dict['weights'], [0, 1],
                                       opt_path=save_dict['opt_weights'])
    cb_plot = callbacks.PlotCallback(save_dict['plots'], save_dict['csv'])
    cb_summary = callbacks.SummaryCallback(save_folder, save_dict['csv'])
    callbacks_ls = [cb_csv, cb_ais, cb_period, cb_plot, cb_summary]

    # do the actual training
    history = model.fit(data.train.data,
                        batch_size=batch_size,
                        n_epoch=n_epoch,
                        callbacks=callbacks_ls,
                        validation_data=data.valid.data)
    

#%%
if __name__ == '__main__':
    run_rbm()
