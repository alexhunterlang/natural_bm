"""Callbacks to monitor and enhance training of neural networks """

#%%
import os
import csv
import numpy as np
import time
import warnings
from collections import deque, OrderedDict, Iterable
import pandas as pd

try:
    import PIL.Image as Image
except ImportError:
    import Image

from natural_bm.estimators import AIS, exact_logZ
import natural_bm.backend as B
from natural_bm import samplers, utils

# Force matplotlib to not use any Xwindows backend.
# Otherwise will get error on linux servers
import matplotlib
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


#%%
class CallbackList:
    """Container abstracting a list of callbacks.
    # Arguments
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    """

    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length
        self.cb_time = 0.0

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        start_time = time.time()

        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
        self._delta_t_batch = 0.
        self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
        self._delta_ts_batch_end = deque([], maxlen=self.queue_length)

        self.cb_time += time.time() - start_time

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        start_time = time.time()

        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

        self.cb_time += time.time() - start_time

    def on_batch_begin(self, batch, logs=None):
        """Called right before processing a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        start_time = time.time()

        logs = logs or {}
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
        self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_begin)
        if (self._delta_t_batch > 0. and
           delta_t_median > 0.95 * self._delta_t_batch and
           delta_t_median > 0.1):
            warnings.warn('Method on_batch_begin() is slow compared '
                          'to the batch update (%f). Check your callbacks.'
                          % delta_t_median)
        self._t_enter_batch = time.time()

        self.cb_time += time.time() - start_time

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        start_time = time.time()

        logs = logs or {}
        if not hasattr(self, '_t_enter_batch'):
            self._t_enter_batch = time.time()
        self._delta_t_batch = time.time() - self._t_enter_batch
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
        self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_end)
        if (self._delta_t_batch > 0. and
           (delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1)):
            warnings.warn('Method on_batch_end() is slow compared '
                          'to the batch update (%f). Check your callbacks.'
                          % delta_t_median)

        self.cb_time += time.time() - start_time

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        # Arguments
            logs: dictionary of logs.
        """
        start_time = time.time()

        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

        self.cb_time += time.time() - start_time

    def on_train_end(self, logs=None):
        """Called at the end of training.
        # Arguments
            logs: dictionary of logs.
        """
        start_time = time.time()

        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

        self.cb_time += time.time() - start_time


#%%
class Callback:
    """Abstract base class used to build new callbacks.
    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `natural_bm.models.Model`.
            Reference of the model being trained.
    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.
    Currently, the `.fit()` method of the model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:
        on_epoch_end: logs include `cost` and
            optionally include `val_cost` (if validation is enabled in `fit`).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `cost`.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


#%%
class BaseLogger(Callback):
    """Callback that accumulates epoch averages of metrics.
    This callback is automatically applied to every Natural BM model.
    """

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.params['metrics']:
                if k in self.totals:
                    # Make value available to next callbacks.
                    logs[k] = self.totals[k] / self.seen


#%%
class History(Callback):
    """Callback that records events into a `History` object.
    This callback is automatically applied to
    every Natural BM model. The `History` object
    gets returned by the `fit` method of models.
    """

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


#%%
class CSVLogger(Callback):
    """Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.
    # Example
        ```python
            csv_logger = CSVLogger('training.log')
            model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```
    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        super(CSVLogger, self).__init__()

        self.sep = str(separator)
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a')
        else:
            self.csv_file = open(self.filename, 'w')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] + self.keys,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


#%%
class PeriodicSave(Callback):
    """Periodically saves the weights of the neural network.
    
    # Arguments:
        weight_path: filepath; where weights will be saved
        epoch_ls: list, optional; which epochs should be saved, if empty will save all    
    """
    def __init__(self, weight_path, epoch_ls=[], opt_path=None):
        super(PeriodicSave, self).__init__()
        
        self.weight_path = weight_path
        self.opt_path = opt_path
        self.epoch_ls = epoch_ls

    def on_epoch_end(self, epoch, logs=None):        

        if (len(self.epoch_ls) == 0) or (epoch in self.epoch_ls):
            filepath = self.weight_path.format(epoch=epoch, **logs)
            self.model.nnet.save_weights(filepath, overwrite=True)
            if self.opt_path is not None:
                filepath = self.opt_path.format(epoch=epoch, **logs)
                self.model.optimizer.save_weights(filepath, overwrite=True)


#%%
class LearningRateScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """

    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
            
        lr = self.schedule(epoch)
        
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
            
        lr = B.eval(B.cast(lr, dtype=B.floatx()))
        self.model.optimizer.lr.set_value(lr)


#%%
class MomentumRateScheduler(Callback):
    """Momentum rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            momentum rate as output (float).
    """

    def __init__(self, schedule):
        super(MomentumRateScheduler, self).__init__()
        
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        
        has_mom = hasattr(self.model.optimizer, 'momentum')
        has_beta = hasattr(self.model.optimizer, 'beta_1')
        
        if (not has_mom) and (not has_beta):
            raise ValueError('Optimizer must have a "momentum" or "beta_1" attribute.')
        
        momentum = self.schedule(epoch)
        
        if momentum is not None:
        
            if not isinstance(momentum, (float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function '
                                 'should be float.')
            
            momentum = B.eval(B.cast(momentum, dtype=B.floatx()))
            if hasattr(self.model.optimizer, 'momentum_goal'):
                self.model.optimizer.momentum_goal.set_value(momentum)
            elif hasattr(self.model.optimizer, 'beta_1'):
                self.model.optimizer.beta_1.set_value(momentum)


#%%
class OptimizerSpy(Callback):
    """Collects logs of Optimizer parameters per epoch """
    def __init__(self):
        super(OptimizerSpy, self).__init__()
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        config = self.model.optimizer.get_config()
        
        kwargs = ['lr', 'momentum', 'beta_1', 'beta_2']
        for k in kwargs:
            if k in config.keys():
                logs[k] = config[k]


#%%
class AISCallback(Callback):
    """Annealed Importance Sampling (AIS) to estimate partition functions.
    
    # Arguments:
        dbm: DBM object
        n_runs: int; number of AIS chains to run in parallel
        n_betas: int, number of intermediate betas (inverse temperatures)
        epoch_ls: list, optional; when to run callback, if empty will run every epoch
        name: str, optional
        exact: Boolean, optional; whether to exactly calcualte partition function in addition to AIS estimate
    
    """
    def __init__(self, dbm, n_runs, n_betas,
                 epoch_ls=[], name='', exact=False):
        super(AISCallback, self).__init__()

        self.dbm = dbm
        self.n_runs = n_runs
        self.n_betas = n_betas
        self.epoch_ls = epoch_ls
        self.exact = exact

        if len(name) > 0:
            self.name = '_'+name
        else:
            self.name = ''

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if (len(self.epoch_ls) == 0) or (epoch in self.epoch_ls):

            train_data = B.get_value(self.model.train_data)
            valid_data = B.get_value(self.model.validation_data)

            ais = AIS(dbm=self.dbm,
                      data=train_data,
                      n_runs=self.n_runs,
                      n_betas=self.n_betas)

            ais.run_logZ()
            logz, logz_high, logz_low = ais.estimate_log_error_Z()

            x = B.placeholder(shape=(None, train_data.shape[1]))
            fe_fxn = B.function([x], B.mean(self.dbm.free_energy(x)))

            fe = fe_fxn(train_data)
            logs['free_energy'] = fe

            val_fe = fe_fxn(valid_data)
            logs['val_free_energy'] = val_fe

            logs['logz'+self.name] = logz
            logs['logz_high'+self.name] = logz_high
            logs['logz_low'+self.name] = logz_low

            logs['prob'+self.name] = -fe - logz
            logs['prob_high'+self.name] = -fe - logz_low
            logs['prob_low'+self.name] = -fe - logz_high

            logs['val_prob'+self.name] = -val_fe - logz
            logs['val_prob_high'+self.name] = -val_fe - logz_low
            logs['val_prob_low'+self.name] = -val_fe - logz_high

            if self.exact:
                logs['logz_exact'+self.name] = exact_logZ(self.dbm)

        else:

            keys = ['free_energy', 'val_free_energy']

            for w1 in ['logz', 'prob', 'val_prob']:
                for w2 in ['', '_high', '_low']:
                    keys.append(w1+w2+self.name)

            if self.exact:
                keys.append('logz_exact'+self.name)

            for k in keys:
                logs[k] = np.nan


#%%
class SampleCallback(Callback):
    """Generates samples of visible layer and saves as images 
    
    # Arguments:
        dbm: DBM object
        savename: filepath; where to save samples
        n_chain: int; number of sampling chains to run in parallel
        n_samples: int; number of images to generate
        plot_every: int; number of chain updates between images    
        epoch_ls: list, optional; when to run callback, if empty will run every epoch
    """
    def __init__(self, dbm, savename,
                 n_chains=20, n_samples=10, plot_every=2000, epoch_ls=[]):

        super(SampleCallback, self).__init__()

        self.dbm = dbm
        self.savename = savename
        assert np.mod(n_chains, 5) == 0
        self.n_chains = n_chains
        self.n_samples = n_samples
        self.plot_every = plot_every
        self.epoch_ls = epoch_ls
        self.sampler = samplers.GibbsProb(self.dbm, nb_gibbs=plot_every)
 
    def on_epoch_end(self, epoch, logs=None):
        
        if (len(self.epoch_ls) == 0) or (epoch in self.epoch_ls):
            
            filepath = self.savename + '.{:04d}'.format(epoch)
            
            data = B.get_value(self.model.train_data)
            n_features = B.eval(data.shape[1])
            beta = 1.0

            # some assumptions I made about data
            x = int(np.sqrt(n_features))
            assert x**2 == n_features
            y = x
            assert np.mod(x, 2) == 0        
                         
            init_v = np.zeros((self.n_chains, n_features))
            
            num_each = int(self.n_chains/5)
        
            # make some starting chains
            # these illustrate different types of reconstructions
        
            # white noise examples
            init_v[0:num_each] = np.random.uniform(size=(num_each, n_features))
            
            # actual examples
            index = np.random.choice(np.arange(data.shape[0]), size=num_each,
                                     replace=False) 
            init_v[num_each:2*num_each] = data[index]
            
            # bit flips
            index = np.random.choice(np.arange(data.shape[0]), size=num_each,
                                     replace=False)
            d = data[index]
            f = np.random.uniform(size=d.shape) > 0.5
            init_v[2*num_each:3*num_each] = d*(1-f)+(1-d)*f
    
            # additive white noise
            index = np.random.choice(np.arange(data.shape[0]), size=num_each,
                                     replace=False)
            d = data[index]
            g = 0.1*np.random.normal(size=d.shape)
            init_v[3*num_each:4*num_each] = np.clip(d+g, 0, 1)
    
            # masking   
            index = np.random.choice(np.arange(data.shape[0]), size=num_each,
                                     replace=False)
            d = data[index]  
            for i, dd in enumerate(d):
                j = np.random.randint(0, 4, 1)
                if j == 0:
                    mask = np.concatenate((np.ones(int(n_features/2)),
                                           np.zeros(int(n_features/2))))
                elif j == 1:
                    mask = np.concatenate((np.zeros(int(n_features/2)),
                                           np.ones(int(n_features/2))))
                elif j == 2:
                    mask = np.tile(np.concatenate((np.ones((int(x/2))),
                                                   np.zeros(int(x/2)))),
                                                    x)
                elif j == 3:
                    mask = np.tile(np.concatenate((np.zeros((int(x/2))),
                                                   np.ones(int(x/2)))),
                                                    x)
                init_v[4*num_each+i] = dd*mask
            
            init_v = init_v.astype(B.floatx())
            
            init_v_var = B.variable(init_v, name='init_v')
            
            prob_ls = self.dbm.propup(init_v_var, beta)

            output_ls, updates = self.sampler.run_chain(prob_ls, beta)           
            
            prob_v = output_ls[0]        
            updates[init_v_var] = prob_v[-1]
        
            # construct the function that implements our persistent chain.
            sample_fxn = B.function([],
                                    prob_v[-1],
                                    updates=updates,
                                    name='sample_fxn')
            
            # create a space to store the image for plotting ( we need to leave
            # room for the tile_spacing as well)
            ts = 1 # tile spacing
            xx = x+ts
            yy = y+ts    
            image_data = np.zeros((xx*(self.n_samples+2)+1, yy*self.n_chains-1), dtype='uint8')
            npz_data = np.zeros((self.n_samples+1, self.n_chains, n_features))
            npz_data[0] = init_v              
                        
            image_data[0:x, :] = utils.tile_raster_images(
                                        X=init_v, img_shape=(x, y),
                                        tile_shape=(1, self.n_chains), tile_spacing=(ts, ts))
            for idx in range(self.n_samples):
                # generate `plot_every` intermediate samples that we discard,
                # because successive samples in the chain are too correlated
                # I left a blank row between original images and gibbs samples
                vis_prob = sample_fxn(0)
                image_data[xx*(idx+2) : xx*(idx+2) + x, :] =\
                    utils.tile_raster_images(X=vis_prob, img_shape=(x, y),
                        tile_shape=(1, self.n_chains), tile_spacing=(ts, ts))
                npz_data[idx+1] = vis_prob
            
            # save image
            image = Image.fromarray(image_data)
            image.save(filepath+'.pdf')    
            kwargs = {'samples':npz_data}
            np.savez_compressed(filepath, **kwargs)

#%%
class PlotCallback(Callback):
    """Generates plots that summarize training.
    
    # Arguments:
        save_folder: folder path; where to save plots
        csv_filepath: filepath; where CSVLogger saved results
    """
    def __init__(self, save_folder, csv_filepath):
        super(PlotCallback, self).__init__()

        self.save_folder = save_folder
        self.csv_filepath = csv_filepath

    def on_train_end(self, logs=None):

        df = pd.read_csv(self.csv_filepath, sep='\t')

        data_dict = {}
        for c in df.columns.values:
            data_dict[c] = df[c].values

        def convert2list(ls):
            if not isinstance(ls, list):
                ls = [ls]
            return ls

        def plot_multi_cost(x, y_ls, color_ls, lbl_ls, title, savename):
            fig = plt.figure()

            y_ls = convert2list(y_ls)
            color_ls = convert2list(color_ls)
            lbl_ls = convert2list(lbl_ls)

            for y, color, lbl in zip(y_ls, color_ls, lbl_ls):
                plt.plot(x, y, color=color, label=lbl)

            plt.xlabel('Epoch')
            plt.title(title)
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            fig.savefig(savename, bbox_inches='tight')
            plt.close()

        def plot_errorbars(x, y, high, low, color, lbl, title, savename):
            fig = plt.figure()

            low_err = np.array(y)-np.array(low)
            high_err = np.array(high)-np.array(y)
            data_err = np.array([low_err, high_err])

            plt.errorbar(x, y, yerr=data_err, color=color,
                         fmt='o', ls='', label=lbl, capsize=10)

            plt.xlim([-5, x[-1]+5])  # so you can see the start/end error bars

            plt.xlabel('Epoch')
            plt.title(title)
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            fig.savefig(savename, bbox_inches='tight')
            plt.close()

        x = data_dict['epoch']

        # These plots are similar, 2 lines:
        # Loss, pslike, and recon
        color_ls = ['red', 'blue']
        for t in ['cost', 'pslike', 'recon']:
            if t in data_dict.keys():
                lbl_ls = [t, 'val_'+t]
                y_ls = [data_dict[lbl_ls[0]], data_dict[lbl_ls[1]]]
                title = t
                savename = self.save_folder+t+'.pdf'
                plot_multi_cost(x, y_ls, color_ls, lbl_ls, title, savename)

        # These plots are single lines:
        # lr, momentum, beta
        if 'lr' in data_dict.keys():
            lbl_ls = ['lr']
            y_ls = [data_dict['lr']]
            title = 'Learning Rate'
            savename = self.save_folder+'learning_rate.pdf'
            plot_multi_cost(x, y_ls, color_ls, lbl_ls, title, savename)
        if 'momentum' in data_dict.keys():
            lbl_ls = ['momentum']
            y_ls = [data_dict['momentum']]
            title = 'Momentum'
            savename = self.save_folder+'momentum.pdf'
            plot_multi_cost(x, y_ls, color_ls, lbl_ls, title, savename)
        if 'beta' in data_dict.keys():
            lbl_ls = ['beta']
            y_ls = [data_dict['beta']]
            title = 'Beta'
            savename = self.save_folder+'beta.pdf'
            plot_multi_cost(x, y_ls, color_ls, lbl_ls, title, savename)


        if 'logz' in data_dict.keys():

            IS_good = np.logical_not(np.isnan(data_dict['logz']))
            x_d = x[IS_good]

            # Free energy difference
            fe = 'free_energy'
            y = data_dict[fe][IS_good]-data_dict['val_'+fe][IS_good]
            color = 'black'
            lbl = 'free_energy_diff'
            title = 'Free energy difference, train minus valid'
            savename = self.save_folder+'free_energy.pdf'
            plot_multi_cost(x_d, y, color, lbl, title, savename)


            # logz plots
            y = data_dict['logz'][IS_good]
            high = data_dict['logz_high'][IS_good]
            low = data_dict['logz_low'][IS_good]

            color = 'red'
            lbl = 'Mean'
            title = 'Log Z. Mean estimate'
            savename = self.save_folder+'logz.pdf'
            plot_multi_cost(x_d, y, color, lbl, title, savename)

            title = 'Log Z. Error bars are 3 std.'
            savename = self.save_folder+'logz_errorbars.pdf'
            plot_errorbars(x_d, y, high, low, 'black', 'logz',
                           title, savename)


            # probability plots
            yt_d = data_dict['prob'][IS_good]
            y_d = data_dict['val_prob'][IS_good]
            high = data_dict['val_prob_high'][IS_good]
            low = data_dict['val_prob_low'][IS_good]

            y_ls = [yt_d, y_d]
            color_ls = ['blue', 'red']
            lbl_ls = ['train', 'valid']
            title = 'Probability of train vs valid. Mean estimate.'
            savename = self.save_folder+'prob.pdf'
            plot_multi_cost(x_d, y_ls, color_ls, lbl_ls, title, savename)

            title = 'Probability of valid. Error bars are 3 std.'
            savename = self.save_folder+'prob_errorbars.pdf'
            plot_errorbars(x_d, y_d, high, low, 'black', 'val_prob',
                           title, savename)


#%%
class SummaryCallback(Callback):
    """Generates text file that summarizes training.
    
    # Arguments:
        save_folder: filepath; where to save summary
        csv_filepath: filepath; where CSVLogger saved results
    """
    def __init__(self, save_folder, csv_filepath):
        super(SummaryCallback, self).__init__()

        self.save_folder = save_folder
        self.csv_filepath = csv_filepath

    def on_train_end(self, logs=None):

        history = self.model.history.history
        n_epoch = np.max(self.model.history.epoch)
        try:
            val_prob = history['val_prob']
            HAS_best = True
            index = np.where(np.logical_not(np.isnan(val_prob)))[0]
            vp = np.array(val_prob)[index]
            best_val_prob = np.max(vp)
            best_epoch = np.where(best_val_prob == val_prob)[0].item()
        except KeyError:
            HAS_best = False
        
        with open(self.save_folder+'summary.txt', 'w') as f:
            total = self.model.history.fit_train_time
            f.write('Number of epochs {}.\n'.format(n_epoch))
            if HAS_best:
                f.write('Best epoch is {}.\n'.format(best_epoch))
                f.write('Best log val prob is {}.\n'.format(best_val_prob))
            f.write('Total fit time was {} minutes.\n'.format(total/60.0))
            f.write('Per epoch total fit time was {} seconds.\n'.format(total/n_epoch))
            f.write('Fit time was {}% callbacks.\n'.format(self.model.history.fit_callback_time/total*100)) 
