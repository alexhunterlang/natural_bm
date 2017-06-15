# Documentation for Common (common.py)

Classes for storing datasets 
## class Dataslice
Specific dataslice of dataset such as training, validation, or testing. 
### \_\_init\_\_
```py

def __init__(self, slicetype, data, lbl)

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_index\_examples
```py

def get_index_examples(self, num_each)

```





## class Dataset
Class that organizes a given dataset 
### \_\_init\_\_
```py

def __init__(self, name, datatype)

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


### sample\_data
```py

def sample_data(datasets, seed=0)

```



Randomly samples probabilities to binary variables 


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

