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



