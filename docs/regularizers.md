# Documentation for Regularizers (regularizers.py)

Regularizers for weights. Simplification of keras.regularizers 
## class Regularizer
Regularizer base class. 


## class L1L2
Regularizer for L1 and L2 regularization.<br /><br /># Arguments<br />    l1: Float; L1 regularization factor.<br />    l2: Float; L2 regularization factor.
### \_\_init\_\_
```py

def __init__(self, l1=0.0, l2=0.0)

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_config
```py

def get_config(self)

```





## functions

### get
```py

def get(identifier)

```



### l1
```py

def l1(l=0.01)

```



### l1\_l2
```py

def l1_l2(l1=0.01, l2=0.01)

```



### l2
```py

def l2(l=0.01)

```


