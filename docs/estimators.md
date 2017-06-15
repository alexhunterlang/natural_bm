# Documentation for Estimators (estimators.py)

Methods to evaluate partition functions. 
## class Sampler
Generic class that creates samples from a neural network 
### \_\_init\_\_
```py

def __init__(self, nnet, beta=1.0, constant=[])

```



General sampler.<br /><br /># Arguments:<br /> ~ nnet: DBM object<br /> ~ beta: inverse temperature for sampling<br /> ~ constant: list of layers with activity clamped to inputs


### probability
```py

def probability(self, *args)

```



Calculates probabilies without intermediate sampling. 


### run\_chain
```py

def run_chain(self, input_ls, beta=1.0, constant=[])

```



This is where the general logic of a sampler lives 


### sample
```py

def sample(self, *args)

```



Calculates samples with intermediate sampling. 


### sample\_inputs
```py

def sample_inputs(self, *args)

```



Calculates probabilities with intermediate sampling. 


### set\_param
```py

def set_param(self, beta=1.0, constant=[])

```



Sets parameters of a Sampler.<br /><br />These parameters may change over time and hence the need for setting<br />parameters after intialization. For example, beta changes during AIS.




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

### exact\_logZ
```py

def exact_logZ(dbm)

```



Exactly calculate the partition function for a RBM.<br /><br /># Arguments:<br /> ~ dbm: DBM object; must be a RBM.<br /> ~ <br /># Returns:<br /> ~ logZ: float; log of the exact partition function

