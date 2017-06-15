# Documentation for Dbm (dbm.py)

Deep Boltzmann Machines (DBM).

This can create DBMs with almost any type of topology connecting layers.
The only restriction is that when the layers are in topological order, 
only connections between even to odd (and odd to even) layers are allowed.
This restriction allows for efficient free energy calculations and
Gibbs sampling since the layers can be updated in only two steps
(even then odd or odd then even).

Technically these are deep restricted boltzmann machines (DRBM) since each
layer of neurons has only external connections and no within layer connections.
However, the literature convention is that DBM actually means DRBM.

Can also create a standard restricted boltzmann machine (RBM) since this
is just a DBM with only two layers.


## class DBM
Deep Boltzmann Machines (DBM) 
### \_\_init\_\_
```py

def __init__(self, layer_size_list, topology_dict, rng=None, W_regularizer=None, b_regularizer=None)

```



Creates a Deep Boltzmann Machine.<br /><br /># Arguments:<br /> ~ layer_size_list: list; size of each Layer<br /> ~ topology_dict: dict; controls which Layers are connected<br /> ~ rng: Random Number Generator, optional<br /> ~ W_regularizer: Regularizer object, optional<br /> ~ b_regularizer: Regularizer object, optional<br /> ~ <br /># References:<br /> ~ 1. Deep boltzmann machines by R Salakhutdinov and G Hinton. AIS, 2009.


### free\_energy
```py

def free_energy(self, x, beta=1.0)

```



Calcualtes free energy of a DBM.<br /><br /># Arguments:<br /> ~ x: tensor; visible inputs<br /> ~ beta: float, optional; inverse temperature<br /><br /># Returns:<br /> ~ fe: tensor; free energy of DBM


### free\_energy\_sumover\_even
```py

def free_energy_sumover_even(self, x, beta)

```



Calcualtes free energy of a DBM by analyically summing out even layers.<br /><br /># Arguments:<br /> ~ x: tensor; visible inputs<br /> ~ beta: float, optional; inverse temperature<br /><br /># Returns:<br /> ~ fe: tensor; free energy of DBM


### free\_energy\_sumover\_odd
```py

def free_energy_sumover_odd(self, x, beta)

```



Calcualtes free energy of a DBM by analyically summing out odd layers.<br /><br /># Arguments:<br /> ~ x: tensor; visible inputs<br /> ~ beta: float, optional; inverse temperature<br /><br /># Returns:<br /> ~ fe: tensor; free energy of DBM


### get\_weights
```py

def get_weights(self)

```



Returns the weights of the model,<br />as a flat list of Numpy arrays.


### prob\_even\_given\_odd
```py

def prob_even_given_odd(self, input_ls, beta=1.0, constant=[])

```



Conditional probabilies of even layers given odd layers as input.<br /><br /># Arguments:<br /> ~ input_ls: list; input activations of layers<br /> ~ beta: float, optional; inverse temperature<br /> ~ constant: list, optional; list of layers to clamp to input values<br /> ~ <br /># Returns:<br /> ~ probabilities: list; probabilities of all layers


### prob\_odd\_given\_even
```py

def prob_odd_given_even(self, input_ls, beta=1.0, constant=[])

```



Conditional probabilies of odd layers given even layers as input.<br /><br /># Arguments:<br /> ~ input_ls: list; input activations of layers<br /> ~ beta: float, optional; inverse temperature<br /> ~ constant: list, optional; list of layers to clamp to input values<br /> ~ <br /># Returns:<br /> ~ probabilities: list; probabilities of all layers


### propup
```py

def propup(self, x, beta=1.0)

```



Propagates visible layer inputs to deeper layers in the network.<br /><br /># Arguments:<br /> ~ x: tensor; visible inputs<br /> ~ beta: float, optional; inverse temperature<br /><br /># Returns:<br /> ~ prob_ls: list of probabilies of activity for each layer


### save\_weights
```py

def save_weights(self, filepath, overwrite=True)

```



Dumps all layer weights to a HDF5 file.<br />The weight file has:<br /> ~ - `layer_names` (attribute), a list of strings<br /> ~  ~ (ordered names of model layers).<br /> ~ - For every layer, a `group` named `layer.name`<br /> ~  ~ - For every such layer group, a group attribute `weight_names`,<br /> ~  ~  ~ a list of strings<br /> ~  ~  ~ (ordered names of weights tensor of the layer).<br /> ~  ~ - For every weight in the layer, a dataset<br /> ~  ~  ~ storing the weight value, named after the weight tensor.


### save\_weights\_to\_hdf5\_group
```py

def save_weights_to_hdf5_group(self, f)

```



Saves the weights 




## functions

### prep\_topology
```py

def prep_topology(topology_dict)

```



Prepares for connecting together Layers and Synapses to create a neural netowrk.<br /><br /># Arguments:<br /> ~ topology_dict: dict; key is the lower layer, value is the set of higher layers connected to lower layer <br /> ~ <br /># Returns:<br /> ~ pairs: list; tuples of layers that need a Synapse<br /> ~ topology_input_dict: dict; reversed topology dict (keys are higher layers)

