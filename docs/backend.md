# Documentation for Backend (backend.py)


This follows the init used by keras

## class RandomStreams
Module component with similar interface to numpy.random<br />(numpy.random.RandomState).<br /><br />Parameters<br />----------<br />seed : int or list of 6 int<br />    A default seed to initialize the random state.<br />    If a single int is given, it will be replicated 6 times.<br />    The first 3 values of the seed must all be less than M1 = 2147483647,<br />    and not all 0; and the last 3 values must all be less than<br />    M2 = 2147462579, and not all 0.
### \_\_init\_\_
```py

def __init__(self, seed=12345, use_cuda=None)

```



Initialize self.  See help(type(self)) for accurate signature.


### binomial
```py

def binomial(self, size=None, n=1, p=0.5, ndim=None, dtype='int64', nstreams=None)

```



### choice
```py

def choice(self, size=1, a=None, replace=True, p=None, ndim=None, dtype='int64', nstreams=None)

```



Sample `size` times from a multinomial distribution defined by<br />probabilities `p`, and returns the indices of the sampled elements.<br />Sampled values are between 0 and `p.shape[1]-1`.<br />Only sampling without replacement is implemented for now.<br /><br />Parameters<br />----------<br />size: integer or integer tensor (default 1)<br /> ~ The number of samples. It should be between 1 and `p.shape[1]-1`.<br />a: int or None (default None)<br /> ~ For now, a should be None. This function will sample<br /> ~ values between 0 and `p.shape[1]-1`. When a != None will be<br /> ~ implemented, if `a` is a scalar, the samples are drawn from the<br /> ~ range 0,...,a-1. We default to 2 as to have the same interface as<br /> ~ RandomStream.<br />replace: bool (default True)<br /> ~ Whether the sample is with or without replacement.<br /> ~ Only replace=False is implemented for now.<br />p: 2d numpy array or theano tensor<br /> ~ the probabilities of the distribution, corresponding to values<br /> ~ 0 to `p.shape[1]-1`.<br /><br />Example : p = [[.98, .01, .01], [.01, .49, .50]] and size=1 will<br />probably result in [[0],[2]]. When setting size=2, this<br />will probably result in [[0,1],[2,1]].<br /><br />Notes<br />-----<br />-`ndim` is only there keep the same signature as other<br />uniform, binomial, normal, etc.<br /><br />-Does not do any value checking on pvals, i.e. there is no<br />check that the elements are non-negative, less than 1, or<br />sum to 1. passing pvals = [[-2., 2.]] will result in<br />sampling [[0, 0]]<br /><br />-Only replace=False is implemented for now.


### get\_substream\_rstates
```py

def get_substream_rstates(*args, **kwargs)

```



Initialize a matrix in which each row is a MRG stream state,<br />and they are spaced by 2**72 samples.


### inc\_rstate
```py

def inc_rstate(self)

```



Update self.rstate to be skipped 2^134 steps forward to the next stream<br />start.


### multinomial
```py

def multinomial(self, size=None, n=1, pvals=None, ndim=None, dtype='int64', nstreams=None)

```



Sample `n` (`n` needs to be >= 1, default 1) times from a multinomial<br />distribution defined by probabilities pvals.<br /><br />Example : pvals = [[.98, .01, .01], [.01, .49, .50]] and n=1 will<br />probably result in [[1,0,0],[0,0,1]]. When setting n=2, this<br />will probably result in [[2,0,0],[0,1,1]].<br /><br />Notes<br />-----<br />-`size` and `ndim` are only there keep the same signature as other<br />uniform, binomial, normal, etc.<br />TODO : adapt multinomial to take that into account<br /><br />-Does not do any value checking on pvals, i.e. there is no<br />check that the elements are non-negative, less than 1, or<br />sum to 1. passing pvals = [[-2., 2.]] will result in<br />sampling [[0, 0]]


### multinomial\_wo\_replacement
```py

def multinomial_wo_replacement(self, size=None, n=1, pvals=None, ndim=None, dtype='int64', nstreams=None)

```



### n\_streams
```py

def n_streams(self, size)

```



### normal
```py

def normal(self, size, avg=0.0, std=1.0, ndim=None, dtype=None, nstreams=None)

```



Parameters<br />----------<br />size<br /> ~ Can be a list of integers or Theano variables (ex: the shape<br /> ~ of another Theano Variable).<br />dtype<br /> ~ The output data type. If dtype is not specified, it will be<br /> ~ inferred from the dtype of low and high, but will be at<br /> ~ least as precise as floatX.<br />nstreams<br /> ~ Number of streams.


### pretty\_return
```py

def pretty_return(self, node_rstate, new_rstate, sample, size, nstreams)

```



### seed
```py

def seed(self, seed=None)

```



Re-initialize each random stream.<br /><br />Parameters<br />----------<br />seed : None or integer in range 0 to 2**30<br /> ~ Each random stream will be assigned a unique state that depends<br /> ~ deterministically on this value.<br /><br />Returns<br />-------<br />None


### set\_rstate
```py

def set_rstate(self, seed)

```



### uniform
```py

def uniform(self, size, low=0.0, high=1.0, ndim=None, dtype=None, nstreams=None)

```



Sample a tensor of given size whose element from a uniform<br />distribution between low and high.<br /><br />If the size argument is ambiguous on the number of dimensions,<br />ndim may be a plain integer to supplement the missing information.<br /><br />Parameters<br />----------<br />low<br /> ~ Lower bound of the interval on which values are sampled.<br /> ~ If the ``dtype`` arg is provided, ``low`` will be cast into<br /> ~ dtype. This bound is excluded.<br />high<br /> ~ Higher bound of the interval on which values are sampled.<br /> ~ If the ``dtype`` arg is provided, ``high`` will be cast into<br /> ~ dtype. This bound is excluded.<br />size<br />  Can be a list of integer or Theano variable (ex: the shape<br />  of other Theano Variable).<br />dtype<br /> ~ The output data type. If dtype is not specified, it will be<br /> ~ inferred from the dtype of low and high, but will be at<br /> ~ least as precise as floatX.


### updates
```py

def updates(self)

```





## functions

### abs
```py

def abs(x)

```



Absolute value of a tensor. 


### all
```py

def all(x, axis=None, keepdims=False)

```



Bitwise reduction (logical AND). 


### allclose
```py

def allclose(x, y, rtol=1e-05, atol=1e-08)

```



Elementwise comparison of tensor x and y 


### any
```py

def any(x, axis=None, keepdims=False)

```



Bitwise reduction (logical OR). 


### arange
```py

def arange(start, stop=None, step=1, dtype='int32')

```



Creates a 1-D tensor containing a sequence of integers.<br />The function arguments use the same convention as<br />Theano's arange: if only one argument is provided,<br />it is in fact the "stop" argument.


### argmax
```py

def argmax(x, axis=-1)

```



Index of the maximum of the values in a tensor, alongside the specified axis. 


### argmin
```py

def argmin(x, axis=-1)

```



Index of the maximum of the values in a tensor, alongside the specified axis. 


### backend
```py

def backend()

```



Publicly accessible method<br />for determining the current backend.<br /># Returns<br /> ~ String, the name of the backend Natural_BM is currently using.<br /># Example<br />```python<br /> ~ >>> natural_bm.backend.backend()<br /> ~ 'theano'<br />```


### batch\_get\_value
```py

def batch_get_value(xs)

```



Returns the value of more than one tensor variable,<br />as a list of Numpy arrays.


### batch\_set\_value
```py

def batch_set_value(tuples)

```



Sets the values of more than one tensor, numpy array pair. 


### cast
```py

def cast(x, dtype)

```



Casts x to dtype. 


### cast\_to\_floatx
```py

def cast_to_floatx(x)

```



Cast a Numpy array to the default Keras float type.<br /><br /># Arguments<br /> ~ x: Numpy array.<br /><br /># Returns<br /> ~ The same Numpy array, cast to its new type.


### clip
```py

def clip(x, min_value, max_value)

```



Clips tensor x to be between min_value and max_value 


### concatenate
```py

def concatenate(tensors, axis=-1)

```



Concatenates list of tensors along given axis 


### cos
```py

def cos(x)

```



Elementwise cosine 


### cumprod
```py

def cumprod(x, axis=0)

```



Cumulative product of the values in a tensor, alongside the specified axis.<br /><br /># Arguments<br /> ~ x: A tensor or variable.<br /> ~ axis: An integer, the axis to compute the product.<br /><br /># Returns<br /> ~ A tensor of the cumulative product of values of `x` along `axis`.


### cumsum
```py

def cumsum(x, axis=0)

```



Cumulative sum of the values in a tensor, alongside the specified axis.<br /><br /># Arguments<br /> ~ x: A tensor or variable.<br /> ~ axis: An integer, the axis to compute the sum.<br /><br /># Returns<br /> ~ A tensor of the cumulative sum of values of `x` along `axis`.


### diag
```py

def diag(x)

```



Extracts diagonal of a tensor. 


### dot
```py

def dot(x, y)

```



Dot product of x and y 


### dtype
```py

def dtype(x)

```



Returns the dtype of a tensor as a string. 


### epsilon
```py

def epsilon()

```



Returns the value of the fuzz<br />factor used in numeric expressions.<br /><br /># Returns<br /> ~ A float.


### equal
```py

def equal(x, y)

```



Elementwise x == y 


### eval
```py

def eval(x)

```



Returns the value of a tensor. 


### exp
```py

def exp(x)

```



Exponential of a tensor. 


### eye
```py

def eye(size, dtype=None, name=None)

```



Instantiates an identity matrix. 


### fill\_diagonal
```py

def fill_diagonal(x, val)

```



Fills in the diagonal of a tensor. 


### flatten
```py

def flatten(x)

```



Collapses a tensor to a single dimension 


### floatx
```py

def floatx()

```



Returns the default float type, as a string.<br />(e.g. 'float16', 'float32', 'float64').<br /><br /># Returns<br /> ~ String, the current default float type.


### function
```py

def function(inputs, outputs, updates=[], name=None, **kwargs)

```



Creates function for computational graphs 


### get\_value
```py

def get_value(x)

```



Returns value of tensor as numpy array 


### get\_variable\_shape
```py

def get_variable_shape(x)

```



Returns the shape of a tensor 


### gradients
```py

def gradients(loss, variables)

```



Calcuates the gradients of loss with respect to variables 


### greater
```py

def greater(x, y)

```



Elementwise x > y 


### greater\_equal
```py

def greater_equal(x, y)

```



Elementwise x >= y 


### ifelse
```py

def ifelse(condition, then_expression, else_expression)

```



Controls if else logic flow inside a computational graph 


### intx
```py

def intx()

```



Returns the default int type, as a string.<br />(e.g. 'int16', 'int32', 'int64').<br /><br /># Returns<br /> ~ String, the current default int type.


### less
```py

def less(x, y)

```



Elementwise x < y 


### less\_equal
```py

def less_equal(x, y)

```



Elementwise x <= y 


### log
```py

def log(x)

```



Natural logarithm of a tensor. 


### logdiffexp
```py

def logdiffexp(x, axis=None, keepdims=False)

```



Computes the log(diff(exp(elements across dimensions of a tensor))).<br />This function is more numerically stable than log(diff(exp(x))).


### logsumexp
```py

def logsumexp(x, axis=None, keepdims=False)

```



Computes log(sum(exp(elements across dimensions of a tensor))).<br />This function is more numerically stable than log(sum(exp(x))).<br />It avoids overflows caused by taking the exp of large inputs and<br />underflows caused by taking the log of small inputs.<br /><br /># Arguments<br /> ~ x: A tensor or variable.<br /> ~ axis: An integer, the axis to reduce over.<br /> ~ keepdims: A boolean, whether to keep the dimensions or not.<br /> ~  ~ If `keepdims` is `False`, the rank of the tensor is reduced<br /> ~  ~ by 1. If `keepdims` is `True`, the reduced dimension is<br /> ~  ~ retained with length 1.<br /><br /># Returns<br /> ~ The reduced tensor.


### make\_rng
```py

def make_rng(seed=None)

```



Creates a Random Number Generator (RNG) 


### max
```py

def max(x, axis=None, keepdims=False)

```



Max of the values in a tensor, alongside the specified axis. 


### maximum
```py

def maximum(x, y)

```



Elementwise maximum 


### mean
```py

def mean(x, axis=None, keepdims=False)

```



Mean of a tensor, alongside the specified axis. 


### min
```py

def min(x, axis=None, keepdims=False)

```



Min of the values in a tensor, alongside the specified axis. 


### minimum
```py

def minimum(x, y)

```



Elementwise minimum 


### ndim
```py

def ndim(x)

```



Returns the dimension of a tensor. 


### not\_equal
```py

def not_equal(x, y)

```



Elementwise x != y 


### one\_hot
```py

def one_hot(indices, num_classes)

```



Input: nD integer tensor of shape (batch_size, dim1, dim2, ... dim(n-1))<br />Output: (n + 1)D one hot representation of the input<br />with shape (batch_size, dim1, dim2, ... dim(n-1), num_classes)


### ones
```py

def ones(shape, dtype=None, name=None)

```



Instantiates an all-ones variable. 


### ones\_like
```py

def ones_like(x, dtype=None, name=None)

```



Instantiates an all-ones variable with the same shape as x. 


### permute\_dimensions
```py

def permute_dimensions(x, pattern)

```



Transpose dimensions.<br />pattern should be a tuple or list of<br />dimension indices, e.g. [0, 2, 1].


### placeholder
```py

def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None)

```



Instantiate an input data placeholder variable. 


### pow
```py

def pow(x, a)

```



Elementwise power of a tensor. 


### prod
```py

def prod(x, axis=None, keepdims=False)

```



Multiply the values in a tensor, alongside the specified axis. 


### random\_binomial
```py

def random_binomial(shape, p=0.0, dtype=None, rng=None)

```



Random samples from the binomial distribution 


### random\_normal
```py

def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, rng=None)

```



Random samples from the normal distribution 


### random\_uniform
```py

def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, rng=None)

```



Random samples from the uniform distribution 


### repeat
```py

def repeat(x, n)

```



Repeat a 2D tensor.<br />If x has shape (samples, dim) and n=2,<br />the output will have shape (samples, 2, dim).


### repeat\_elements
```py

def repeat_elements(x, rep, axis)

```



Repeat the elements of a tensor along an axis, like np.repeat.<br />If x has shape (s1, s2, s3) and axis=1, the output<br />will have shape (s1, s2 * rep, s3).


### reshape
```py

def reshape(x, shape)

```



Reshapes tensor x to given shape 


### reverse
```py

def reverse(x, axes)

```



Reverse a tensor along the specified axes 


### round
```py

def round(x)

```



Round tensor to nearest integer. Rounds half to even. 


### scan
```py

def scan(function, outputs_info=None, n_steps=None, name=None, **kwargs)

```



Scan is the for loop equivalent of Theano 


### set\_epsilon
```py

def set_epsilon(e)

```



Sets the value of the fuzz<br />factor used in numeric expressions.<br /><br /># Arguments<br /> ~ e: float. New value of epsilon.


### set\_floatx
```py

def set_floatx(floatx)

```



Sets the default float type.<br /><br /># Arguments<br /> ~ String: 'float16', 'float32', or 'float64'.


### set\_intx
```py

def set_intx(intx)

```



Sets the default int type.<br /><br /># Arguments<br /> ~ String: 'int16', 'int32', or 'int64'.


### set\_value
```py

def set_value(x, value)

```



Sets value of tensor with a numpy array 


### shape
```py

def shape(x)

```



Returns the shape of a tensor. 


### sigmoid
```py

def sigmoid(x)

```



Sigmoid of a tensor 


### sign
```py

def sign(x)

```



Sign of a tensor 


### sin
```py

def sin(x)

```



Elementwise sine 


### solve
```py

def solve(a, b)

```



Solves the equation ax=b for x. 


### sqrt
```py

def sqrt(x)

```



Square root of a tensor after clipping to positive definite. 


### square
```py

def square(x)

```



Elementwise square of a tensor. 


### squeeze
```py

def squeeze(x, axis)

```



Remove a 1-dimension from the tensor at index "axis". 


### stack
```py

def stack(x, axis=0)

```



Join a sequence of tensors along a new axis 


### std
```py

def std(x, axis=None, keepdims=False)

```



Standard deviation of a tensor, alongside the specified axis. 


### stop\_gradient
```py

def stop_gradient(variables)

```



Returns `variables` but with zero gradient with respect to every other<br />variables.


### sum
```py

def sum(x, axis=None, keepdims=False)

```



Sum of the values in a tensor, alongside the specified axis. 


### svd
```py

def svd(x)

```



Singular value decomposition (SVD) of x. Returns U, S, V. 


### tile
```py

def tile(x, n)

```



Repeats a tensor n times along each axis 


### transpose
```py

def transpose(x)

```



Tensor transpose 


### until
```py

def until(condition)

```



Allows a scan to be a while loop 


### var
```py

def var(x, axis=None, keepdims=False)

```



Variance of a tensor, alongside the specified axis. 


### variable
```py

def variable(value, dtype=None, name=None)

```



Instantiates a variable and returns it.<br /><br /># Arguments<br /> ~ value: Numpy array, initial value of the tensor.<br /> ~ dtype: Tensor type.<br /> ~ name: Optional name string for the tensor.<br /><br /># Returns<br /> ~ A variable instance.


### zeros
```py

def zeros(shape, dtype=None, name=None)

```



Instantiates an all-zeros variable. 


### zeros\_like
```py

def zeros_like(x, dtype=None, name=None)

```



Instantiates an all-zeros variable with the same shape as x. 

