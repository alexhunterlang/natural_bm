# Documentation for Svhn (svhn.py)

Google Street View House Number Dataset

These are originally color images that I converted to grayscale to make
it simpler to identify as probabilities. I also created a threshold version
of the probabilities, but I have not made a sampled version.

## class Dataset
Class that organizes a given dataset 
### \_\_init\_\_
```py

def __init__(self, name, datatype)

```



Initialize self.  See help(type(self)) for accurate signature.




## class SVHN
Class that organizes a given dataset 
### \_\_init\_\_
```py

def __init__(self, datatype)

```



Initialize self.  See help(type(self)) for accurate signature.




## functions

### convert2prob
```py

def convert2prob(datasets)

```



Converts to probabilities 


### convert2uint8
```py

def convert2uint8(datasets)

```



Converst to unit8 


### scale\_to\_unit\_interval
```py

def scale_to_unit_interval(ndar, eps=1.1920929e-07)

```



Scales all values in the ndarray ndar to be between 0 and 1 


### threshold\_data
```py

def threshold_data(datasets, threshold=None)

```



Deterministically thresholds probabilities to binary variables 

