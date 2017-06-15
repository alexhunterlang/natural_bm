# Documentation for Optimizers (optimizers.py)

Optimizer class for calculating weight updates.

In general, this code closely matches keras.optimizers. There is a small
change in SGD.

## class Optimizer
Abstract optimizer base class.<br />Note: this is the parent class of all optimizers, not an actual optimizer<br />that can be used for training models.
### \_\_init\_\_
```py

def __init__(self, **kwargs)

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_config
```py

def get_config(self)

```



### get\_gradients
```py

def get_gradients(self, loss, params)

```



### get\_updates
```py

def get_updates(self, params, loss)

```



### get\_weights
```py

def get_weights(self)

```



Returns the current value of the weights of the optimizer.<br /><br /># Returns<br /> ~ A list of numpy arrays.


### save\_weights
```py

def save_weights(self, filepath, overwrite=True)

```



Dumps all layer weights to a HDF5 file. 


### save\_weights\_to\_hdf5\_group
```py

def save_weights_to_hdf5_group(self, f)

```



### set\_weights
```py

def set_weights(self, weights)

```



Sets the weights of the optimizer, from Numpy arrays.<br />Should only be called after computing the gradients<br />(otherwise the optimizer has no weights).<br /><br /># Arguments<br /> ~ weights: a list of Numpy arrays. The number<br /> ~  ~ of arrays and their shape must match<br /> ~  ~ number of the dimensions of the weights<br /> ~  ~ of the optimizer (i.e. it should match the<br /> ~  ~ output of `get_weights`).<br /><br /># Raises<br /> ~ ValueError: in case of incompatible weight shapes.




## class Nadam
Nesterov Adam optimizer.<br />Much like Adam is essentially RMSprop with momentum,<br />Nadam is Adam RMSprop with Nesterov momentum.<br />Default parameters follow those provided in the paper.<br />It is recommended to leave the parameters of this optimizer<br />at their default values.<br /># Arguments<br />    lr: float >= 0. Learning rate.<br />    beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.<br />    epsilon: float >= 0. Fuzz factor.<br /># References<br />    - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)<br />    - [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
### \_\_init\_\_
```py

def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, **kwargs)

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_config
```py

def get_config(self)

```



### get\_gradients
```py

def get_gradients(self, loss, params)

```



### get\_updates
```py

def get_updates(self, params, loss)

```



### get\_weights
```py

def get_weights(self)

```



Returns the current value of the weights of the optimizer.<br /><br /># Returns<br /> ~ A list of numpy arrays.


### save\_weights
```py

def save_weights(self, filepath, overwrite=True)

```



Dumps all layer weights to a HDF5 file. 


### save\_weights\_to\_hdf5\_group
```py

def save_weights_to_hdf5_group(self, f)

```



### set\_weights
```py

def set_weights(self, weights)

```



Sets the weights of the optimizer, from Numpy arrays.<br />Should only be called after computing the gradients<br />(otherwise the optimizer has no weights).<br /><br /># Arguments<br /> ~ weights: a list of Numpy arrays. The number<br /> ~  ~ of arrays and their shape must match<br /> ~  ~ number of the dimensions of the weights<br /> ~  ~ of the optimizer (i.e. it should match the<br /> ~  ~ output of `get_weights`).<br /><br /># Raises<br /> ~ ValueError: in case of incompatible weight shapes.




## class Adam
Adam optimizer.<br />Default parameters follow those provided in the original paper.<br /># Arguments<br />    lr: float >= 0. Learning rate.<br />    beta_1: float, 0 < beta < 1. Generally close to 1.<br />    beta_2: float, 0 < beta < 1. Generally close to 1.<br />    epsilon: float >= 0. Fuzz factor.<br />    decay: float >= 0. Learning rate decay over each update.<br /># References<br />    - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
### \_\_init\_\_
```py

def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, **kwargs)

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_config
```py

def get_config(self)

```



### get\_gradients
```py

def get_gradients(self, loss, params)

```



### get\_updates
```py

def get_updates(self, params, loss)

```



### get\_weights
```py

def get_weights(self)

```



Returns the current value of the weights of the optimizer.<br /><br /># Returns<br /> ~ A list of numpy arrays.


### save\_weights
```py

def save_weights(self, filepath, overwrite=True)

```



Dumps all layer weights to a HDF5 file. 


### save\_weights\_to\_hdf5\_group
```py

def save_weights_to_hdf5_group(self, f)

```



### set\_weights
```py

def set_weights(self, weights)

```



Sets the weights of the optimizer, from Numpy arrays.<br />Should only be called after computing the gradients<br />(otherwise the optimizer has no weights).<br /><br /># Arguments<br /> ~ weights: a list of Numpy arrays. The number<br /> ~  ~ of arrays and their shape must match<br /> ~  ~ number of the dimensions of the weights<br /> ~  ~ of the optimizer (i.e. it should match the<br /> ~  ~ output of `get_weights`).<br /><br /># Raises<br /> ~ ValueError: in case of incompatible weight shapes.




## class SGD
Stochastic gradient descent optimizer.<br />Includes support for momentum,<br />learning rate decay, and Nesterov momentum.<br /><br />Additional support for ramping up and down momentum to adhere to advice<br />in paper cited below.<br /><br /># Arguments<br />    lr: float >= 0. Learning rate.<br />    momentum: float >= 0. Parameter updates momentum.<br />    decay: float >= 0. Learning rate decay over each update.<br />    nesterov: boolean. Whether to apply Nesterov momentum.<br />    schedule_decay: float >= 0. Controls how fast to ramp up momentum.<br />    mom_iter_max: int >= 0. Batch iteration at which to hit final, smaller momentum.<br />    <br /># References<br />    - [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
### \_\_init\_\_
```py

def __init__(self, lr=0.01, momentum=0.0, decay=0.0, nesterov=False, schedule_decay=0.004, mom_iter_max=475000, **kwargs)

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_config
```py

def get_config(self)

```



### get\_gradients
```py

def get_gradients(self, loss, params)

```



### get\_updates
```py

def get_updates(self, params, loss)

```



### get\_weights
```py

def get_weights(self)

```



Returns the current value of the weights of the optimizer.<br /><br /># Returns<br /> ~ A list of numpy arrays.


### save\_weights
```py

def save_weights(self, filepath, overwrite=True)

```



Dumps all layer weights to a HDF5 file. 


### save\_weights\_to\_hdf5\_group
```py

def save_weights_to_hdf5_group(self, f)

```



### set\_weights
```py

def set_weights(self, weights)

```



Sets the weights of the optimizer, from Numpy arrays.<br />Should only be called after computing the gradients<br />(otherwise the optimizer has no weights).<br /><br /># Arguments<br /> ~ weights: a list of Numpy arrays. The number<br /> ~  ~ of arrays and their shape must match<br /> ~  ~ number of the dimensions of the weights<br /> ~  ~ of the optimizer (i.e. it should match the<br /> ~  ~ output of `get_weights`).<br /><br /># Raises<br /> ~ ValueError: in case of incompatible weight shapes.




## functions

### clip\_norm
```py

def clip_norm(g, c, n)

```



Clips gradient 

