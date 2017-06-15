# Documentation for Theano_Backend (theano_backend.py)

Backend based on Theano.

This backend is a simplified version of the Keras backend.


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

