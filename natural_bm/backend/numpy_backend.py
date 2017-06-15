"""
Backend based on Numpy.

This code is used to test the theano backend and is a possible option in
the preprocessing module. It is too slow to actually train neural networks.
"""

#%%
import numpy as np
from scipy.special import expit
from scipy.misc import logsumexp as scipy_logsumexp

from natural_bm.backend.common import epsilon, set_epsilon, floatx, set_floatx, cast_to_floatx, intx, set_intx


#%% Variables
def variable(value, dtype=None, name=None):
    """Instantiates a variable and returns it.

    # Arguments
        value: Numpy array, initial value of the tensor.
        dtype: Tensor type.
        name: Optional name string for the tensor.

    # Returns
        A variable instance .
    """
    if dtype is None:
        dtype = floatx()

    if hasattr(value, 'eval'):
        value = value.eval()

    return np.asarray(value, dtype=dtype)


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    """Instantiate an input data placeholder variable. """
    raise NotImplementedError('This function is not implemented for the numpy backend.')
    
    
def shape(x):
    """Returns the shape of a tensor. """
    return x.shape


def ndim(x):
    """Returns the dimension of a tensor. """
    return x.ndim


def dtype(x):
    """Returns the dtype of a tensor as a string. """
    return x.dtype.name


def eval(x):
    """Returns the value of a tensor. """
    return x


def zeros(shape, dtype=None, name=None):
    """Instantiates an all-zeros variable. """
    if dtype is None:
        dtype = floatx()
    return variable(np.zeros(shape), dtype, name)


def ones(shape, dtype=None, name=None):
    """Instantiates an all-ones variable. """
    if dtype is None:
        dtype = floatx()
    return variable(np.ones(shape), dtype, name)


def eye(size, dtype=None, name=None):
    """Instantiates an identity matrix. """
    if dtype is None:
        dtype = floatx()
    return variable(np.eye(size), dtype, name)


def ones_like(x, dtype=None, name=None):
    """Instantiates an all-ones variable with the same shape as x. """
    return np.ones_like(x, dtype=dtype)


def zeros_like(x, dtype=None, name=None):
    """Instantiates an all-zeros variable with the same shape as x. """
    return np.zeros_like(x, dtype=dtype)


def cast(x, dtype):
    """Casts x to dtype. """
    if isinstance(x, np.ndarray):
        x = x.astype(dtype)
    else:
        x = np.asarray(x, dtype).item()
    return x

#%% LINEAR ALGEBRA
"""
Assumed overridden:
+, -, /, *, +=, -=, *=, /=
"""
def dot(x, y):
    """Dot product of x and y """
    return np.dot(x, y)


def transpose(x):
    """Tensor transpose """
    return np.transpose(x)


def svd(x):
    """Singular value decomposition (SVD) of x. Returns U, S, V. """
    return np.linalg.svd(x)


def diag(x):
    """Extracts diagonal of a tensor. """
    return np.diag(x)


def fill_diagonal(x, val):
    """Fills in the diagonal of a tensor. """
    n = x.shape[0]
    x[range(n), range(n)] = val
    return x


def solve(a, b):
    """Solves the equation ax=b for x. """
    return np.linalg.solve(a, b)


#%% ELEMENT-WISE OPERATIONS
def _keras_axis(x, axis):
    """This is what keras expects axis to do for things like mean, std, etc
    """
    if isinstance(axis, list):
        assert len(axis) == 2 and axis[1] == -1, 'Trying to match behavior from keras backend tests'
        x = np.reshape(x, (x.shape[0], -1))
        axis = axis[0]
    return x, axis


def max(x, axis=None, keepdims=False):
    """Max of the values in a tensor, alongside the specified axis. """
    x, axis = _keras_axis(x, axis)
    return np.max(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    """Min of the values in a tensor, alongside the specified axis. """
    x, axis = _keras_axis(x, axis)
    return np.min(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    """Sum of the values in a tensor, alongside the specified axis. """
    x, axis = _keras_axis(x, axis)
    return np.sum(x, axis=axis, keepdims=keepdims)


def prod(x, axis=None, keepdims=False):
    """Multiply the values in a tensor, alongside the specified axis. """
    x, axis = _keras_axis(x, axis)
    return np.prod(x, axis=axis, keepdims=keepdims)


def cumsum(x, axis=0):
    """Cumulative sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the sum.
    
    # Returns
        A tensor of the cumulative sum of values of `x` along `axis`.
    """
    return np.cumsum(x, axis=axis)


def cumprod(x, axis=0):
    """Cumulative product of the values in a tensor, alongside the specified axis.
    
    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.
    
    # Returns
        A tensor of the cumulative product of values of `x` along `axis`.
    """
    return np.cumprod(x, axis=axis)


def mean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis. """
    x, axis = _keras_axis(x, axis)
    return np.mean(x, axis=axis, keepdims=keepdims)


def std(x, axis=None, keepdims=False):
    """Standard deviation of the values in a tensor, alongside the specified axis. """
    x, axis = _keras_axis(x, axis)
    return np.std(x, axis=axis, keepdims=keepdims)


def var(x, axis=None, keepdims=False):
    """Variance of the values in a tensor, alongside the specified axis. """
    x, axis = _keras_axis(x, axis)
    return np.var(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    """Bitwise reduction (logical OR). """
    x, axis = _keras_axis(x, axis)
    return np.bitwise_or(x, axis=axis, keepdims=keepdims)


def all(x, axis=None, keepdims=False):
    """Bitwise reduction (logical AND). """
    x, axis = _keras_axis(x, axis)
    return np.bitwise_and(x, axis=axis, keepdims=keepdims)


def argmax(x, axis=-1):
    """Index of the maximum of the values in a tensor, alongside the specified axis. """
    return np.argmax(x, axis=axis)


def argmin(x, axis=-1):
    """Index of the maximum of the values in a tensor, alongside the specified axis. """
    return np.argmin(x, axis=axis)


def square(x):
    """Elementwise square of a tensor. """
    return np.square(x)


def abs(x):
    """Absolute value of a tensor. """
    return np.abs(x)


def sqrt(x):
    """Square root of a tensor after clipping to positive definite. """
    x = np.clip(x, 0., np.inf)
    return np.sqrt(x)


def exp(x):
    """Exponential of a tensor. """
    return np.exp(x)


def log(x):
    """Natural logarithm of a tensor. """
    return np.log(x)


def logsumexp(x, axis=None, keepdims=False):
    """Computes log(sum(exp(elements across dimensions of a tensor))).
    This function is more numerically stable than log(sum(exp(x))).
    It avoids overflows caused by taking the exp of large inputs and
    underflows caused by taking the log of small inputs.
    
    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to reduce over.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`, the reduced dimension is
            retained with length 1.
    
    # Returns
        The reduced tensor.
    """
    return scipy_logsumexp(x, axis=axis, keepdims=keepdims)


def logdiffexp(x, axis=None, keepdims=False):
    """Computes the log(diff(exp(elements across dimensions of a tensor))).
    This function is more numerically stable than log(diff(exp(x))).
    """
    assert x.shape[0] == 2
    a = np.max(x)
    logdiff = a + np.log(np.diff(np.exp(x-a)))
    logdiff = logdiff.item()

    return logdiff


def round(x):
    """Round tensor to nearest integer. Rounds half to even. """
    return np.round(x)


def sign(x):
    """Sign of tensor. """
    return np.sign(x)


def pow(x, a):
    """Elementwise power of a tensor. """
    return np.power(x, a)


def clip(x, min_value, max_value):
    """Clips tensor x to be between min_value and max_value """
    if max_value is not None and max_value < min_value:
        max_value = min_value
    if max_value is None:
        max_value = np.inf
    return np.clip(x, min_value, max_value)


def equal(x, y):
    """Elementwise x == y """
    return np.equal(x, y)


def not_equal(x, y):
    """Elementwise x != y """
    return np.not_equal(x, y)


def greater(x, y):
    """Elementwise x > y """
    return np.greater(x, y)


def greater_equal(x, y):
    """Elementwise x >= y """
    return np.greater_equal(x, y)


def less(x, y):
    """Elementwise x < y """
    return np.less(x, y)


def less_equal(x, y):
    """Elementwise x <= y """
    return np.less_equal(x, y)


def maximum(x, y):
    """Elementwise maximum """
    return np.maximum(x, y)


def minimum(x, y):
    """Elementwise minimum """
    return np.minimum(x, y)


def sin(x):
    """Elementwise sine """
    return np.sin(x)


def cos(x):
    """Elementwise cosine """
    return np.cos(x)


#%% SHAPE OPERATIONS
def concatenate(tensors, axis=-1):
    """Concatenates list of tensors along given axis """
    return np.concatenate([x for x in tensors], axis=axis)


def reshape(x, shape):
    """Reshapes tensor x to given shape """
    return np.reshape(x, shape)


def permute_dimensions(x, pattern):
    """Transpose dimensions.
    pattern should be a tuple or list of
    dimension indices, e.g. [0, 2, 1].
    """
    pattern = tuple(pattern)
    return x.transpose(pattern)


def repeat_elements(x, rep, axis):
    """Repeat the elements of a tensor along an axis, like np.repeat.
    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3).
    """
    return np.repeat(x, rep, axis=axis)


def repeat(x, n):
    """Repeat a 2D tensor.
    If x has shape (samples, dim) and n=2,
    the output will have shape (samples, 2, dim).
    """
    assert x.ndim == 2
    y = x.reshape((x.shape[0], 1, x.shape[1]))
    y = np.repeat(y, n, axis=1)
    return y


def arange(start, stop=None, step=1, dtype='int32'):
    """Creates a 1-D tensor containing a sequence of integers.
    The function arguments use the same convention as
    Theano's arange: if only one argument is provided,
    it is in fact the "stop" argument.
    """
    return np.arange(start, stop=stop, step=step).astype(dtype)


def tile(x, n):
    """Repeats a tensor n times along each axis """
    return np.tile(x, n)


def flatten(x):
    """Collapses a tensor to a single dimension """
    return x.flatten()


def squeeze(x, axis):
    """Remove a 1-dimension from the tensor at index "axis". """
    shape = list(x.shape)
    shape.pop(axis)
    return np.reshape(x, tuple(shape))


def stack(x, axis=0):
    """Join a sequence of tensors along a new axis """
    return np.stack(x, axis=axis)


def one_hot(indices, num_classes):
    """Input: nD integer tensor of shape (batch_size, dim1, dim2, ... dim(n-1))
    Output: (n + 1)D one hot representation of the input
    with shape (batch_size, dim1, dim2, ... dim(n-1), num_classes)
    """
    return np.eye(num_classes)[indices]


def reverse(x, axes):
    """Reverse a tensor along the specified axes """
    if isinstance(axes, int):
        axes = [axes]
    slices = [slice(None, None, -1) if i in axes else slice(None, None, None) for i in range(x.ndim)]
    return x[slices]


#%% VALUE MANIPULATION
def get_value(x):
    """Returns value of tensor as numpy array """
    return x


def batch_get_value(xs):
    """
    Returns the value of more than one tensor variable,
    as a list of Numpy arrays.
    """
    return [get_value(x) for x in xs]

def set_value(x, value):
    """Sets value of tensor with a numpy array """
    raise NotImplementedError('This function is not implemented for the numpy backend.')
    

def batch_set_value(tuples):
    """Sets the values of more than one tensor, numpy array pair. """
    raise NotImplementedError('This function is not implemented for the numpy backend.')


def get_variable_shape(x):
    """Returns the shape of a tensor """
    return x.shape


#%% GRAPH MANIPULATION
def function(inputs, outputs, updates=[], name=None, **kwargs):
    """Creates function for computational graphs """
    raise NotImplementedError('This function is not implemented for the numpy backend.')
    

def scan(function, outputs_info=None, n_steps=None, name=None, **kwargs):
    """Scan is the for loop equivalent of Theano """
    raise NotImplementedError('This function is not implemented for the numpy backend.')
    

def gradients(loss, variables):
    """Calcuates the gradients of loss with respect to variables """
    raise NotImplementedError('This function is not implemented for the numpy backend.')
    

def stop_gradient(variables):
    """Returns `variables` but with zero gradient with respect to every other
    variables.
    """
    raise NotImplementedError('This function is not implemented for the numpy backend.')


def sigmoid(x):
    """Sigmoid of a tensor """
    return expit(x)


# %% RANDOMNESS
def _random_prep(dtype=None, rng=None):
    """Helper function for random functions """
    if dtype is None:
        dtype = floatx()
    if rng is None:
        rng = make_rng()
    return dtype, rng


def make_rng(seed=None):
    """Creates a Random Number Generator (RNG) """
    if seed is None:
        seed = 1234
    rng = np.random.RandomState(seed)
    return rng


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, rng=None):
    """Random samples from the normal distribution """
    dtype, rng = _random_prep(dtype, rng)
    return rng.normal(size=shape, loc=mean, scale=stddev).astype(dtype)


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, rng=None):
    """Random samples from the uniform distribution """
    dtype, rng = _random_prep(dtype, rng)
    return rng.uniform(size=shape, low=minval, high=maxval).astype(dtype)


def random_binomial(shape, p=0.0, dtype=None, rng=None):
    """Random samples from the binomial distribution """
    dtype, rng = _random_prep(dtype, rng)
    return rng.binomial(size=shape, n=1, p=p).astype(dtype)


#%% CONTROL FLOW
def ifelse(condition, then_expression, else_expression):
    """Controls if else logic flow inside a computational graph """
    if condition:
        output = then_expression
    else:
        output = else_expression
    return output


def until(condition):
    """Allows a scan to be a while loop """
    raise NotImplementedError('This function is not implemented for the numpy backend.')


#%% Misc
def allclose(x, y, rtol=1e-5, atol=1e-8):
    """Elementwise comparison of tensor x and y """
    return np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=False)
