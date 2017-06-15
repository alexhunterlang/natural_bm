# Documentation for Components (components.py)

Components needed to assemble a neural network 
## class NeuralNetParts
Generic object for neural network components 
### \_\_init\_\_
```py

def __init__(self, name)

```



Requries a name to initialize parts 




## class Synapse
Connects together neural network layers with a weight 
### \_\_init\_\_
```py

def __init__(self, name, init_W, regularizer=None)

```



Creates the synapse<br /><br /># Arguments:<br /> ~ name: str<br /> ~ init_W: numpy array, initial weights of synapse<br /> ~ regularizer: Regularizer object




## class Layer
Layer of neurons 
### \_\_init\_\_
```py

def __init__(self, name, init_b, up_dict=None, down_dict=None, activation=<function sigmoid at 0x1107a39d8>, regularizer=None)

```



Creates the layer of neurons.<br /><br /># Arguments:<br /> ~ name: str<br /> ~ init_b: numpy array, initial bias of neurons<br /> ~ up_dict: dict, contains weights that connect lower numbered layers to this layer<br /> ~ down_dict: dict, contains weights that connect higher numbered layers to this layer<br /> ~ activation: function, internal neuron activity to final activity<br /> ~ regularizer: Regularizer object


### input\_z
```py

def input_z(self, input_ls, direction=None)

```



Weighted inputs of neurons<br /><br /># Arguments:<br /> ~ input_ls: list; activities of other layers in neural network <br /> ~ direction: str, optional; controls types of inputs to layer<br /> ~  ~ <br /># Returns:<br /> ~ z: weighted inputs of neurons


### output
```py

def output(self, input_ls, beta=1.0, direction=None)

```



Activity of neurons<br /><br /># Arguments:<br /> ~ input_ls: list; activities of other layers in neural network <br /> ~ beta: float; inverse temperature <br /> ~ direction: str, optional; controls types of inputs to layer<br /> ~  ~ <br /># Returns:<br /> ~ activity: final output activity of neurons




## functions

### dict\_2\_OrderedDict
```py

def dict_2_OrderedDict(d)

```



Convert a standard dict to a sorted, OrderedDict 

