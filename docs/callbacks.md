# Documentation for Callbacks (callbacks.py)

Callbacks to monitor and enhance training of neural networks 
## class LearningRateScheduler
Learning rate scheduler.<br /># Arguments<br />    schedule: a function that takes an epoch index as input<br />        (integer, indexed from 0) and returns a new<br />        learning rate as output (float).
### \_\_init\_\_
```py

def __init__(self, schedule)

```



Initialize self.  See help(type(self)) for accurate signature.


### on\_batch\_begin
```py

def on_batch_begin(self, batch, logs=None)

```



### on\_batch\_end
```py

def on_batch_end(self, batch, logs=None)

```



### on\_epoch\_begin
```py

def on_epoch_begin(self, epoch, logs=None)

```



### on\_epoch\_end
```py

def on_epoch_end(self, epoch, logs=None)

```



### on\_train\_begin
```py

def on_train_begin(self, logs=None)

```



### on\_train\_end
```py

def on_train_end(self, logs=None)

```



### set\_model
```py

def set_model(self, model)

```



### set\_params
```py

def set_params(self, params)

```





## class MomentumRateScheduler
Momentum rate scheduler.<br /># Arguments<br />    schedule: a function that takes an epoch index as input<br />        (integer, indexed from 0) and returns a new<br />        momentum rate as output (float).
### \_\_init\_\_
```py

def __init__(self, schedule)

```



Initialize self.  See help(type(self)) for accurate signature.


### on\_batch\_begin
```py

def on_batch_begin(self, batch, logs=None)

```



### on\_batch\_end
```py

def on_batch_end(self, batch, logs=None)

```



### on\_epoch\_begin
```py

def on_epoch_begin(self, epoch, logs=None)

```



### on\_epoch\_end
```py

def on_epoch_end(self, epoch, logs=None)

```



### on\_train\_begin
```py

def on_train_begin(self, logs=None)

```



### on\_train\_end
```py

def on_train_end(self, logs=None)

```



### set\_model
```py

def set_model(self, model)

```



### set\_params
```py

def set_params(self, params)

```





## class SummaryCallback
Generates text file that summarizes training.<br /><br /># Arguments:<br />    save_folder: filepath; where to save summary<br />    csv_filepath: filepath; where CSVLogger saved results
### \_\_init\_\_
```py

def __init__(self, save_folder, csv_filepath)

```



Initialize self.  See help(type(self)) for accurate signature.


### on\_batch\_begin
```py

def on_batch_begin(self, batch, logs=None)

```



### on\_batch\_end
```py

def on_batch_end(self, batch, logs=None)

```



### on\_epoch\_begin
```py

def on_epoch_begin(self, epoch, logs=None)

```



### on\_epoch\_end
```py

def on_epoch_end(self, epoch, logs=None)

```



### on\_train\_begin
```py

def on_train_begin(self, logs=None)

```



### on\_train\_end
```py

def on_train_end(self, logs=None)

```



### set\_model
```py

def set_model(self, model)

```



### set\_params
```py

def set_params(self, params)

```





## class SampleCallback
Generates samples of visible layer and saves as images <br /><br /># Arguments:<br />    dbm: DBM object<br />    savename: filepath; where to save samples<br />    n_chain: int; number of sampling chains to run in parallel<br />    n_samples: int; number of images to generate<br />    plot_every: int; number of chain updates between images    <br />    epoch_ls: list, optional; when to run callback, if empty will run every epoch
### \_\_init\_\_
```py

def __init__(self, dbm, savename, n_chains=20, n_samples=10, plot_every=2000, epoch_ls=[])

```



Initialize self.  See help(type(self)) for accurate signature.


### on\_batch\_begin
```py

def on_batch_begin(self, batch, logs=None)

```



### on\_batch\_end
```py

def on_batch_end(self, batch, logs=None)

```



### on\_epoch\_begin
```py

def on_epoch_begin(self, epoch, logs=None)

```



### on\_epoch\_end
```py

def on_epoch_end(self, epoch, logs=None)

```



### on\_train\_begin
```py

def on_train_begin(self, logs=None)

```



### on\_train\_end
```py

def on_train_end(self, logs=None)

```



### set\_model
```py

def set_model(self, model)

```



### set\_params
```py

def set_params(self, params)

```





## class CallbackList
Container abstracting a list of callbacks.<br /># Arguments<br />    callbacks: List of `Callback` instances.<br />    queue_length: Queue length for keeping<br />        running statistics over callback execution time.
### \_\_init\_\_
```py

def __init__(self, callbacks=None, queue_length=10)

```



Initialize self.  See help(type(self)) for accurate signature.


### append
```py

def append(self, callback)

```



### on\_batch\_begin
```py

def on_batch_begin(self, batch, logs=None)

```



Called right before processing a batch.<br /># Arguments<br /> ~ batch: integer, index of batch within the current epoch.<br /> ~ logs: dictionary of logs.


### on\_batch\_end
```py

def on_batch_end(self, batch, logs=None)

```



Called at the end of a batch.<br /># Arguments<br /> ~ batch: integer, index of batch within the current epoch.<br /> ~ logs: dictionary of logs.


### on\_epoch\_begin
```py

def on_epoch_begin(self, epoch, logs=None)

```



Called at the start of an epoch.<br /># Arguments<br /> ~ epoch: integer, index of epoch.<br /> ~ logs: dictionary of logs.


### on\_epoch\_end
```py

def on_epoch_end(self, epoch, logs=None)

```



Called at the end of an epoch.<br /># Arguments<br /> ~ epoch: integer, index of epoch.<br /> ~ logs: dictionary of logs.


### on\_train\_begin
```py

def on_train_begin(self, logs=None)

```



Called at the beginning of training.<br /># Arguments<br /> ~ logs: dictionary of logs.


### on\_train\_end
```py

def on_train_end(self, logs=None)

```



Called at the end of training.<br /># Arguments<br /> ~ logs: dictionary of logs.


### set\_model
```py

def set_model(self, model)

```



### set\_params
```py

def set_params(self, params)

```





## class OptimizerSpy
Collects logs of Optimizer parameters per epoch 
### \_\_init\_\_
```py

def __init__(self)

```



Initialize self.  See help(type(self)) for accurate signature.


### on\_batch\_begin
```py

def on_batch_begin(self, batch, logs=None)

```



### on\_batch\_end
```py

def on_batch_end(self, batch, logs=None)

```



### on\_epoch\_begin
```py

def on_epoch_begin(self, epoch, logs=None)

```



### on\_epoch\_end
```py

def on_epoch_end(self, epoch, logs=None)

```



### on\_train\_begin
```py

def on_train_begin(self, logs=None)

```



### on\_train\_end
```py

def on_train_end(self, logs=None)

```



### set\_model
```py

def set_model(self, model)

```



### set\_params
```py

def set_params(self, params)

```





## class PeriodicSave
Periodically saves the weights of the neural network.<br /><br /># Arguments:<br />    weight_path: filepath; where weights will be saved<br />    epoch_ls: list, optional; which epochs should be saved, if empty will save all    
### \_\_init\_\_
```py

def __init__(self, weight_path, epoch_ls=[], opt_path=None)

```



Initialize self.  See help(type(self)) for accurate signature.


### on\_batch\_begin
```py

def on_batch_begin(self, batch, logs=None)

```



### on\_batch\_end
```py

def on_batch_end(self, batch, logs=None)

```



### on\_epoch\_begin
```py

def on_epoch_begin(self, epoch, logs=None)

```



### on\_epoch\_end
```py

def on_epoch_end(self, epoch, logs=None)

```



### on\_train\_begin
```py

def on_train_begin(self, logs=None)

```



### on\_train\_end
```py

def on_train_end(self, logs=None)

```



### set\_model
```py

def set_model(self, model)

```



### set\_params
```py

def set_params(self, params)

```





## class PlotCallback
Generates plots that summarize training.<br /><br /># Arguments:<br />    save_folder: folder path; where to save plots<br />    csv_filepath: filepath; where CSVLogger saved results
### \_\_init\_\_
```py

def __init__(self, save_folder, csv_filepath)

```



Initialize self.  See help(type(self)) for accurate signature.


### on\_batch\_begin
```py

def on_batch_begin(self, batch, logs=None)

```



### on\_batch\_end
```py

def on_batch_end(self, batch, logs=None)

```



### on\_epoch\_begin
```py

def on_epoch_begin(self, epoch, logs=None)

```



### on\_epoch\_end
```py

def on_epoch_end(self, epoch, logs=None)

```



### on\_train\_begin
```py

def on_train_begin(self, logs=None)

```



### on\_train\_end
```py

def on_train_end(self, logs=None)

```



### set\_model
```py

def set_model(self, model)

```



### set\_params
```py

def set_params(self, params)

```





## class AISCallback
Annealed Importance Sampling (AIS) to estimate partition functions.<br /><br /># Arguments:<br />    dbm: DBM object<br />    n_runs: int; number of AIS chains to run in parallel<br />    n_betas: int, number of intermediate betas (inverse temperatures)<br />    epoch_ls: list, optional; when to run callback, if empty will run every epoch<br />    name: str, optional<br />    exact: Boolean, optional; whether to exactly calcualte partition function in addition to AIS estimate
### \_\_init\_\_
```py

def __init__(self, dbm, n_runs, n_betas, epoch_ls=[], name='', exact=False)

```



Initialize self.  See help(type(self)) for accurate signature.


### on\_batch\_begin
```py

def on_batch_begin(self, batch, logs=None)

```



### on\_batch\_end
```py

def on_batch_end(self, batch, logs=None)

```



### on\_epoch\_begin
```py

def on_epoch_begin(self, epoch, logs=None)

```



### on\_epoch\_end
```py

def on_epoch_end(self, epoch, logs=None)

```



### on\_train\_begin
```py

def on_train_begin(self, logs=None)

```



### on\_train\_end
```py

def on_train_end(self, logs=None)

```



### set\_model
```py

def set_model(self, model)

```



### set\_params
```py

def set_params(self, params)

```





## class OrderedDict
Dictionary that remembers insertion order


## class BaseLogger
Callback that accumulates epoch averages of metrics.<br />This callback is automatically applied to every Natural BM model.
### \_\_init\_\_
```py

def __init__(self)

```



Initialize self.  See help(type(self)) for accurate signature.


### on\_batch\_begin
```py

def on_batch_begin(self, batch, logs=None)

```



### on\_batch\_end
```py

def on_batch_end(self, batch, logs=None)

```



### on\_epoch\_begin
```py

def on_epoch_begin(self, epoch, logs=None)

```



### on\_epoch\_end
```py

def on_epoch_end(self, epoch, logs=None)

```



### on\_train\_begin
```py

def on_train_begin(self, logs=None)

```



### on\_train\_end
```py

def on_train_end(self, logs=None)

```



### set\_model
```py

def set_model(self, model)

```



### set\_params
```py

def set_params(self, params)

```





## class CSVLogger
Callback that streams epoch results to a csv file.<br />Supports all values that can be represented as a string,<br />including 1D iterables such as np.ndarray.<br /># Example<br />    ```python<br />        csv_logger = CSVLogger('training.log')<br />        model.fit(X_train, Y_train, callbacks=[csv_logger])<br />    ```<br /># Arguments<br />    filename: filename of the csv file, e.g. 'run/log.csv'.<br />    separator: string used to separate elements in the csv file.<br />    append: True: append if file exists (useful for continuing<br />        training). False: overwrite existing file,
### \_\_init\_\_
```py

def __init__(self, filename, separator=',', append=False)

```



Initialize self.  See help(type(self)) for accurate signature.


### on\_batch\_begin
```py

def on_batch_begin(self, batch, logs=None)

```



### on\_batch\_end
```py

def on_batch_end(self, batch, logs=None)

```



### on\_epoch\_begin
```py

def on_epoch_begin(self, epoch, logs=None)

```



### on\_epoch\_end
```py

def on_epoch_end(self, epoch, logs=None)

```



### on\_train\_begin
```py

def on_train_begin(self, logs=None)

```



### on\_train\_end
```py

def on_train_end(self, logs=None)

```



### set\_model
```py

def set_model(self, model)

```



### set\_params
```py

def set_params(self, params)

```





## class Callback
Abstract base class used to build new callbacks.<br /># Properties<br />    params: dict. Training parameters<br />        (eg. verbosity, batch size, number of epochs...).<br />    model: instance of `natural_bm.models.Model`.<br />        Reference of the model being trained.<br />The `logs` dictionary that callback methods<br />take as argument will contain keys for quantities relevant to<br />the current batch or epoch.<br />Currently, the `.fit()` method of the model class<br />will include the following quantities in the `logs` that<br />it passes to its callbacks:<br />    on_epoch_end: logs include `cost` and<br />        optionally include `val_cost` (if validation is enabled in `fit`).<br />    on_batch_begin: logs include `size`,<br />        the number of samples in the current batch.<br />    on_batch_end: logs include `cost`.
### \_\_init\_\_
```py

def __init__(self)

```



Initialize self.  See help(type(self)) for accurate signature.


### on\_batch\_begin
```py

def on_batch_begin(self, batch, logs=None)

```



### on\_batch\_end
```py

def on_batch_end(self, batch, logs=None)

```



### on\_epoch\_begin
```py

def on_epoch_begin(self, epoch, logs=None)

```



### on\_epoch\_end
```py

def on_epoch_end(self, epoch, logs=None)

```



### on\_train\_begin
```py

def on_train_begin(self, logs=None)

```



### on\_train\_end
```py

def on_train_end(self, logs=None)

```



### set\_model
```py

def set_model(self, model)

```



### set\_params
```py

def set_params(self, params)

```





## class Iterable


## class History
Callback that records events into a `History` object.<br />This callback is automatically applied to<br />every Natural BM model. The `History` object<br />gets returned by the `fit` method of models.
### \_\_init\_\_
```py

def __init__(self)

```



Initialize self.  See help(type(self)) for accurate signature.


### on\_batch\_begin
```py

def on_batch_begin(self, batch, logs=None)

```



### on\_batch\_end
```py

def on_batch_end(self, batch, logs=None)

```



### on\_epoch\_begin
```py

def on_epoch_begin(self, epoch, logs=None)

```



### on\_epoch\_end
```py

def on_epoch_end(self, epoch, logs=None)

```



### on\_train\_begin
```py

def on_train_begin(self, logs=None)

```



### on\_train\_end
```py

def on_train_end(self, logs=None)

```



### set\_model
```py

def set_model(self, model)

```



### set\_params
```py

def set_params(self, params)

```





## class deque
deque([iterable[, maxlen]]) --> deque object<br /><br />A list-like sequence optimized for data accesses near its endpoints.


## class AIS
AIS<br /><br />In general need to do the following:<br /><br />    1. s1, s2, ... sn (samples)<br />    2. w = p1(s1)/p0(s1) * p2(s2)/p1(s2) * ... * pn(sn)/pn-1(sn) where p is unnormalized<br />    3. Zb/Za ~ 1/M sum(w) = r_{AIS}<br />    4. Zb ~ Za*r_{AIS}<br /><br />This means that we need a model that has the following properties for AIS to work:<br />    1. easy to generate samples<br />    2. easy to estimate unnormalized probabilities<br />    3. easy to exactly calculate Za<br /><br />The Za model is called the data base rate (DBR) model. All weights and biases<br />are zero except for the visible bias. The visibile bias is an estimate <br />based on the mean of the data but biased to guarantee p not equal to zero.<br /><br />I will sum over the even states since this simplifies the intermediate sampling.
### \_\_init\_\_
```py

def __init__(self, dbm, data, n_runs, n_betas=None, betas=None)

```



Initialize an object to perform AIS.<br /> ~  ~ <br /># Arguments:<br /> ~ dbm: DBM object<br /> ~ data: numpy array, needed for data base rate model<br /> ~ n_runs: int, number of parallel AIS estimates to run<br /> ~ n_betas: int, optional. Will create evenly spaced betas. Need either n_betas or betas.<br /> ~ betas: numpy array, optional. Betas for intermediate distributions. Need either n_betas or betas.<br /> ~ <br /> ~ <br /># References:<br /> ~ 1. On the quantitative analysis of deep belief networks by R Salakhutdinov and I Murray. ACM 2008.<br /> ~ 2. Deep boltzmann machines by R Salakhutdinov and G Hinton. AIS, 2009.


### estimate\_log\_error\_Z
```py

def estimate_log_error_Z(self)

```



Error bars on estimate of partition function.<br /><br />The output is the mean and +- 3 standard deviations of the true<br />(ie not logged) partition function. This is why the standard deviations<br />are not symmetric about the mean.<br /><br />Returns:<br /> ~ * mean logZ: float<br /> ~ * -3 std logZ: float<br /> ~ * +3 std logZ: float


### run\_logZ
```py

def run_logZ(self)

```



Performs calculatations of AIS runs.<br /><br />Must be called before estimates.




## functions

### exact\_logZ
```py

def exact_logZ(dbm)

```



Exactly calculate the partition function for a RBM.<br /><br /># Arguments:<br /> ~ dbm: DBM object; must be a RBM.<br /> ~ <br /># Returns:<br /> ~ logZ: float; log of the exact partition function

