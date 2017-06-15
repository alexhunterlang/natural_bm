# Documentation for Samplers (samplers.py)

Samplers feed a neural network inputs and generates new output samples 
## class GibbsProb
Generic class that creates samples from a neural network 
### \_\_init\_\_
```py

def __init__(self, nnet, nb_gibbs=5)

```



Performs Gibbs probability sampling.<br /><br />This means intermediate and activities are samples but final output<br />is a probability.<br /><br /># Arguments:<br /> ~ nnet: DBM object<br /> ~ nb_gibbs: int, optional; number of Gibb updates to do


### probability
```py

def probability(self, *args)

```



Calculates probabilies without intermediate sampling. 


### run\_chain
```py

def run_chain(self, args, beta=1.0, constant=[])

```



Generates chains of Gibbs sampling but final output is probability.<br /><br /># Arguments:<br /> ~ args: input probabilities<br /> ~ beta: inverse temperature for sampling<br /> ~ constant: list of layers with activity clamped to inputs


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




## class Meanfield
Meanfield calculations of probabilities 
### \_\_init\_\_
```py

def __init__(self, nnet, max_steps=25, rtol=1e-05, atol=1e-06)

```



Performs meanfield (only probabilities, no samples) updates.<br /><br /># Arguments:<br /> ~ nnet: DBM object<br /> ~ max_steps: int, optional; maximum number of chain updates<br /> ~ rtol: float, optional; relative tolerance for stopping chain updates<br /> ~ atol: float, optional; absolute tolerance for stopping chain updates


### probability
```py

def probability(self, *args)

```



Calculates probabilies without intermediate sampling. 


### run\_chain
```py

def run_chain(self, args, beta=1.0, constant=[])

```



Generates chains of meanfield updates.<br /><br /># Arguments:<br /> ~ args: input probabilities<br /> ~ beta: inverse temperature for sampling<br /> ~ constant: list of layers with activity clamped to inputs


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




## class Gibbs
Generic class that creates samples from a neural network 
### \_\_init\_\_
```py

def __init__(self, nnet, nb_gibbs=5)

```



Performs Gibbs sampling.<br /><br />This means intermediate and final activities are samples.<br /><br /># Arguments:<br /> ~ nnet: DBM object<br /> ~ nb_gibbs: int, optional; number of Gibb updates to do


### probability
```py

def probability(self, *args)

```



Calculates probabilies without intermediate sampling. 


### run\_chain
```py

def run_chain(self, args, beta=1.0, constant=[])

```



Generates chains of Gibbs sampling.<br /><br /># Arguments:<br /> ~ args: input probabilities<br /> ~ beta: inverse temperature for sampling<br /> ~ constant: list of layers with activity clamped to inputs


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



