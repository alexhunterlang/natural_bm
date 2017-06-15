# Documentation for Training (training.py)

Generates statistics needed for training DBMs 
## class OrderedDict
Dictionary that remembers insertion order


## class Trainer
### \_\_init\_\_
```py

def __init__(self, nnet, nb_pos_steps=25, nb_neg_steps=5)

```



General Trainer.<br /><br /># Arguments:<br /> ~ nnet: DBM object<br /> ~ nb_pos_steps: int, optional. Number of updates in the positive, or data phase.<br /> ~ nb_neg_steps: int, optional. Number of updates in the negative, or model phase.


### loss\_fn
```py

def loss_fn(self)

```



Result is a Theano expression with the form loss = f(x).


### neg\_stats
```py

def neg_stats(self, prob_data)

```



Logic that generates model (negative stage) driven statistics 


### pos\_stats
```py

def pos_stats(self, x)

```



Logic that generates data (positive stage) driven statistics 




## class PCD
Implements Persistent Contrastive Divergence (PCD) training.<br /><br />This is very similar to CD training. The only difference is in the<br />initialization of the negative (model) statistics stage. In CD, the <br />positive statistics initialize the model. In PCD, the probabilites from<br />the last time the negative statistics were generated is used as the new<br />start. This works because weight changes are relatively small and hence<br />the last chain from the the old model is close to a good chain for the <br />new model.
### \_\_init\_\_
```py

def __init__(self, nnet, nb_pos_steps=25, nb_neg_steps=5, init_chain=None, batch_size=None)

```



PCD Trainer.<br /><br />Need to provide either the init_chain or the batch_size.<br /><br /># Arguments:<br /> ~ nnet: DBM object<br /> ~ nb_pos_steps: int, optional. Number of updates in the positive, or data phase.<br /> ~ nb_neg_steps: int, optional. Number of updates in the negative, or model phase.<br /> ~ init_chain: list of tensors, optional. Initial starting point of model persistent chains.<br /> ~ batch_size: int, optional. Need batch size to generate appropriate number of persistent chains.


### loss\_fn
```py

def loss_fn(self)

```



Compute contrastive divergence loss with k steps of Gibbs sampling (CD-k).<br /><br />Result is a Theano expression with the form loss = f(x).


### neg\_stats
```py

def neg_stats(self, prob_data=None)

```



Generates negative (model dependent) statistics.<br /><br /># Arguments:<br /> ~ prob_data: list of tensors; completely ignored, just here so that same function signature as CD<br /> ~ <br /># Returns:<br /> ~ prob_model: list; model dependent probabilities


### pos\_stats
```py

def pos_stats(self, x)

```



Generates positive (data dependent) statistics.<br /><br /># Arguments:<br /> ~ x: tensor; input data<br /> ~ <br /># Returns:<br /> ~ prob_data: list; data dependent probabilities




## class CD
Implements Contrastive Divergence (CD) training.<br /><br />The proper training should run the positive and negative stages to<br />convergence. However, it was realized that just a few updates of the stages<br />actually led to good training. Therefore CD is just a few steps of the updates<br />and stops before the neural network converges to the true equilibrium<br />distribution.<br /><br />This is often shown in the literature as CD-k where k is the number of<br />negative steps. The positive stage is still run to convergence since this<br />is exact in one step for an RBM and usually converges fast for a DBM.
### \_\_init\_\_
```py

def __init__(self, nnet, nb_pos_steps=25, nb_neg_steps=5)

```



CD Trainer.<br /><br /># Arguments:<br /> ~ nnet: DBM object<br /> ~ nb_pos_steps: int, optional. Number of updates in the positive, or data phase.<br /> ~ nb_neg_steps: int, optional. Number of updates in the negative, or model phase.


### loss\_fn
```py

def loss_fn(self)

```



Compute contrastive divergence loss with k steps of Gibbs sampling (CD-k).<br /><br />Result is a Theano expression with the form loss = f(x).


### neg\_stats
```py

def neg_stats(self, prob_data)

```



Generates negative (model dependent) statistics.<br /><br /># Arguments:<br /> ~ prob_data: list of tensors; input data from pos_stats<br /> ~ <br /># Returns:<br /> ~ prob_model: list; model dependent probabilities


### pos\_stats
```py

def pos_stats(self, x)

```



Generates positive (data dependent) statistics.<br /><br /># Arguments:<br /> ~ x: tensor; input data<br /> ~ <br /># Returns:<br /> ~ prob_data: list; data dependent probabilities




## functions

### merge\_OrderedDicts
```py

def merge_OrderedDicts(d1, d2)

```



Merge two OrderedDicts into a new one 

