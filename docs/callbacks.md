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




