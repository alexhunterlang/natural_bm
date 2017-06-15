# Documentation for Fast (fast.py)

Simplified version of MNIST that is useful for demos and testing 
## class Dataset
Class that organizes a given dataset 
### \_\_init\_\_
```py

def __init__(self, name, datatype)

```



Initialize self.  See help(type(self)) for accurate signature.




## class Fast
Class that organizes a given dataset 
### \_\_init\_\_
```py

def __init__(self, datatype)

```



Initialize self.  See help(type(self)) for accurate signature.




## functions

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


### threshold\_data
```py

def threshold_data(datasets, threshold=None)

```



Deterministically thresholds probabilities to binary variables 

