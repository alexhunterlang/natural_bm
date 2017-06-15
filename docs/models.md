# Documentation for Models (models.py)

A model organizes the training of a neural network.

The general structure, and especial the fit method, are similar to the keras
Model class.

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





## class Model
Class that handles the training of a neural network 
### \_\_init\_\_
```py

def __init__(self, nnet, optimizer, trainer)

```



Initialize self.  See help(type(self)) for accurate signature.


### fit
```py

def fit(self, x, batch_size=100, n_epoch=10, callbacks=None, validation_data=None, shuffle=True, initial_epoch=0)

```



Trains the model for a fixed number of epochs (iterations on a dataset).<br /><br /># Arguments<br /> ~ x: Theano shared array of training data<br /> ~ batch_size: integer. Number of samples per gradient update.<br /> ~ n_epoch: integer, the number of times to iterate<br /> ~  ~ over the training data arrays.<br /> ~ callbacks: list of callbacks to be called during training.<br /> ~ validation_data: Theano shared array of data on which to evaluate<br /> ~  ~ the loss and any model metrics at the end of each epoch.<br /> ~  ~ The model will not be trained on this data.<br /> ~ shuffle: boolean, whether to shuffle the training data<br /> ~  ~ before each epoch.<br /> ~ initial_epoch: epoch at which to start training<br /> ~  ~ (useful for resuming a previous training run)<br /><br /># Returns<br /> ~ A `History` instance. Its `history` attribute contains<br /> ~ all information collected during training.


### predict\_on\_batch
```py

def predict_on_batch(self, x)

```



Runs a single gradient update on a single batch of data.<br /># Arguments<br /> ~ x: Numpy array of training data,<br /> ~  ~ or list of Numpy arrays if the model has multiple inputs.<br /> ~  ~ If all inputs in the model are named,<br /> ~  ~ you can also pass a dictionary<br /> ~  ~ mapping input names to Numpy arrays.<br /># Returns<br /> ~ Scalar training loss<br /> ~ (if the model has a single output and no metrics)<br /> ~ or list of scalars (if the model has multiple outputs<br /> ~ and/or metrics).


### train\_on\_batch
```py

def train_on_batch(self, x)

```



Runs a single gradient update on a single batch of data.<br /># Arguments<br /> ~ x: Numpy array of training data,<br /> ~  ~ or list of Numpy arrays if the model has multiple inputs.<br /> ~  ~ If all inputs in the model are named,<br /> ~  ~ you can also pass a dictionary<br /> ~  ~ mapping input names to Numpy arrays.<br /># Returns<br /> ~ Scalar training loss<br /> ~ (if the model has a single output and no metrics)<br /> ~ or list of scalars (if the model has multiple outputs<br /> ~ and/or metrics).




## functions

### check\_batches
```py

def check_batches(size, batch_size)

```



Checks batches on the first epoch to see if any data is missed 


### make\_batches
```py

def make_batches(size, batch_size, epoch=None)

```



Returns a list of batch indices (tuples of indices). 


### merge\_OrderedDicts
```py

def merge_OrderedDicts(d1, d2)

```



Merge two OrderedDicts into a new one 

