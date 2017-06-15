"""Backend based on Theano.

This backend is a simplified version of the Keras backend.
"""

#%%
import numpy as np

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from natural_bm.backend.common import epsilon, set_epsilon, floatx, set_floatx, cast_to_floatx, intx, set_intx

#%% Variables
def variable(value, dtype=None, name=None):
    """Instantiates a variable and returns it.
    
    # Arguments
        value: Numpy array, initial value of the tensor.
        dtype: Tensor type.
        name: Optional name string for the tensor.
    
    # Returns
        A variable instance.
    """
    if dtype is None:
        dtype = floatx()

    if isinstance(value, (theano.tensor.TensorVariable,
                          theano.tensor.sharedvar.TensorSharedVariable,
                          theano.tensor.TensorConstant)):
        value = value.eval()
    value = np.asarray(value, dtype=dtype)
    variable = theano.shared(value=value,
                             name=name,
                             strict=False)
    return variable


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    """Instantiate an input data placeholder variable. """
    if dtype is None:
        dtype = floatx()
    if shape is None and ndim is None:
        raise ValueError('Specify either a shape or ndim value.')
    if shape is not None:
        ndim = len(shape)

    broadcast = (False,) * ndim
    x = T.TensorType(dtype, broadcast)(name)
    return x


def shape(x):
    """Returns the shape of a tensor. """
    return x.shape


def ndim(x):
    """Returns the dimension of a tensor. """
    return x.ndim


def dtype(x):
    """Returns the dtype of a tensor as a string. """
    return x.dtype


def eval(x):
    """Returns the value of a tensor. """
    return x.eval()


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
    return T.ones_like(x, dtype=dtype)


def zeros_like(x, dtype=None, name=None):
    """Instantiates an all-zeros variable with the same shape as x. """
    return T.zeros_like(x, dtype=dtype)


def cast(x, dtype):
    """Casts x to dtype. """
    return T.cast(x, dtype)


#%% LINEAR ALGEBRA
"""
Assumed overridden:
+, -, /, *, +=, -=, *=, /=
"""
def dot(x, y):
    """Dot product of x and y """
    return T.dot(x, y)


def transpose(x):
    """Tensor transpose """
    return T.transpose(x)


def svd(x):
    """Singular value decomposition (SVD) of x. Returns U, S, V. """
    return T.nlinalg.SVD()(x)


def diag(x):
    """Extracts diagonal of a tensor. """
    return T.nlinalg.diag(x)


def fill_diagonal(x, val):
    """Fills in the diagonal of a tensor. """

    if val.size.eval() == 1:
        val = T.extra_ops.repeat(val, x.eval().shape[0])

    # adapted from following theano help topic: https://groups.google.com/forum/#!topic/theano-users/zYD-gsddIYs
    orig_diag = T.diag(T.diagonal(x))
    new_diag = T.diag(val)
    y = x - orig_diag + new_diag

    return y


def solve(a, b):
    """Solves the equation ax=b for x. """
    return T.slinalg.solve(a, b)


#%% ELEMENT-WISE OPERATIONS
def max(x, axis=None, keepdims=False):
    """Max of the values in a tensor, alongside the specified axis. """
    return T.max(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    """Min of the values in a tensor, alongside the specified axis. """
    return T.min(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    """Sum of the values in a tensor, alongside the specified axis. """
    return T.sum(x, axis=axis, keepdims=keepdims)


def prod(x, axis=None, keepdims=False):
    """Multiply the values in a tensor, alongside the specified axis. """
    return T.prod(x, axis=axis, keepdims=keepdims)


def cumsum(x, axis=0):
    """Cumulative sum of the values in a tensor, alongside the specified axis.
    
    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the sum.
    
    # Returns
        A tensor of the cumulative sum of values of `x` along `axis`.
    """
    return T.extra_ops.cumsum(x, axis=axis)


def cumprod(x, axis=0):
    """Cumulative product of the values in a tensor, alongside the specified axis.
    
    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.
    
    # Returns
        A tensor of the cumulative product of values of `x` along `axis`.
    """
    return T.extra_ops.cumprod(x, axis=axis)


def mean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis. """
    dtype = None
    # bool is available since theano v0.9dev
    if 'int' in x.dtype or x.dtype == 'bool':
        dtype = floatx()
    return T.mean(x, axis=axis, keepdims=keepdims, dtype=dtype)


def std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis. """
    return T.std(x, axis=axis, keepdims=keepdims)


def var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis. """
    return T.var(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    """Bitwise reduction (logical OR). """
    return T.any(x, axis=axis, keepdims=keepdims)


def all(x, axis=None, keepdims=False):
    """Bitwise reduction (logical AND). """
    return T.all(x, axis=axis, keepdims=keepdims)


def argmax(x, axis=-1):
    """Index of the maximum of the values in a tensor, alongside the specified axis. """
    return T.argmax(x, axis=axis, keepdims=False)


def argmin(x, axis=-1):
    """Index of the maximum of the values in a tensor, alongside the specified axis. """
    return T.argmin(x, axis=axis, keepdims=False)


def square(x):
    """Elementwise square of a tensor. """
    return T.sqr(x)


def abs(x):
    """Absolute value of a tensor. """
    return T.abs_(x)


def sqrt(x):
    """Square root of a tensor after clipping to positive definite. """
    x = T.clip(x, 0., np.inf)
    return T.sqrt(x)


def exp(x):
    """Exponential of a tensor. """
    return T.exp(x)


def log(x):
    """Natural logarithm of a tensor. """
    return T.log(x)


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
    # Theano has a built-in optimization for logsumexp (see https://github.com/Theano/Theano/pull/4736)
    # so we can just write the expression directly:
    return T.log(T.sum(T.exp(x), axis=axis, keepdims=keepdims))


def logdiffexp(x, axis=None, keepdims=False):
    """Computes the log(diff(exp(elements across dimensions of a tensor))).
    This function is more numerically stable than log(diff(exp(x))).
    """
    a = T.max(x)
    logdiff = a + T.log(T.extra_ops.diff(T.exp(x-a)))

    return logdiff


def round(x):
    """Round tensor to nearest integer. Rounds half to even. """
    return T.round(x, mode='half_to_even')


def sign(x):
    """Sign of a tensor """
    return T.sgn(x)


def pow(x, a):
    """Elementwise power of a tensor. """
    return T.pow(x, a)


def clip(x, min_value, max_value):
    """Clips tensor x to be between min_value and max_value """
    if max_value is not None and max_value < min_value:
        max_value = min_value
    if max_value is None:
        max_value = np.inf
    return T.clip(x, min_value, max_value)


def equal(x, y):
    """Elementwise x == y """
    return T.eq(x, y)


def not_equal(x, y):
    """Elementwise x != y """
    return T.neq(x, y)


def greater(x, y):
    """Elementwise x > y """
    return T.gt(x, y)


def greater_equal(x, y):
    """Elementwise x >= y """
    return T.ge(x, y)


def less(x, y):
    """Elementwise x < y """
    return T.lt(x, y)


def less_equal(x, y):
    """Elementwise x <= y """
    return T.le(x, y)


def maximum(x, y):
    """Elementwise maximum """
    return T.maximum(x, y)


def minimum(x, y):
    """Elementwise minimum """
    return T.minimum(x, y)


def sin(x):
    """Elementwise sine """
    return T.sin(x)


def cos(x):
    """Elementwise cosine """
    return T.cos(x)


#%% SHAPE OPERATIONS
def concatenate(tensors, axis=-1):
    """Concatenates list of tensors along given axis """
    return T.concatenate([x for x in tensors], axis=axis)


def reshape(x, shape):
    """Reshapes tensor x to given shape """
    return T.reshape(x, shape)


def permute_dimensions(x, pattern):
    """Transpose dimensions.
    pattern should be a tuple or list of
    dimension indices, e.g. [0, 2, 1].
    """
    pattern = tuple(pattern)
    return x.dimshuffle(pattern)


def repeat_elements(x, rep, axis):
    """Repeat the elements of a tensor along an axis, like np.repeat.
    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3).
    """
    return T.repeat(x, rep, axis=axis)


def repeat(x, n):
    """Repeat a 2D tensor.
    If x has shape (samples, dim) and n=2,
    the output will have shape (samples, 2, dim).
    """
    assert x.ndim == 2
    y = x.dimshuffle((0, 'x', 1))
    y = T.extra_ops.repeat(y, n, axis=1)
    return y


def arange(start, stop=None, step=1, dtype='int32'):
    """Creates a 1-D tensor containing a sequence of integers.
    The function arguments use the same convention as
    Theano's arange: if only one argument is provided,
    it is in fact the "stop" argument.
    """
    return T.arange(start, stop=stop, step=step, dtype=dtype)


def tile(x, n):
    """Repeats a tensor n times along each axis """
    return T.tile(x, n)


def flatten(x):
    """Collapses a tensor to a single dimension """
    return T.flatten(x)


def squeeze(x, axis):
    """Remove a 1-dimension from the tensor at index "axis". """
    shape = list(x.shape)
    shape.pop(axis)
    return T.reshape(x, tuple(shape))


def stack(x, axis=0):
    """Join a sequence of tensors along a new axis """
    return T.stack(x, axis=axis)


def one_hot(indices, num_classes):
    """Input: nD integer tensor of shape (batch_size, dim1, dim2, ... dim(n-1))
    Output: (n + 1)D one hot representation of the input
    with shape (batch_size, dim1, dim2, ... dim(n-1), num_classes)
    """
    input_shape = tuple((indices.shape[i] for i in range(indices.ndim)))
    indices = T.flatten(indices)
    oh = T.extra_ops.to_one_hot(indices, num_classes)
    oh = T.reshape(oh, input_shape + (num_classes,))
    return oh


def reverse(x, axes):
    """Reverse a tensor along the specified axes """
    if isinstance(axes, int):
        axes = [axes]
    slices = [slice(None, None, -1) if i in axes else slice(None, None, None) for i in range(x.ndim)]
    return x[slices]


#%% VALUE MANIPULATION
def get_value(x):
    """Returns value of tensor as numpy array """
    if not hasattr(x, 'get_value'):
        raise TypeError('get_value() can only be called on a variable. '
                        'If you have an expression instead, use eval().')
    return x.get_value()


def batch_get_value(xs):
    """Returns the value of more than one tensor variable,
    as a list of Numpy arrays.
    """
    return [get_value(x) for x in xs]


def set_value(x, value):
    """Sets value of tensor with a numpy array """
    x.set_value(np.asarray(value, dtype=x.dtype))


def batch_set_value(tuples):
    """Sets the values of more than one tensor, numpy array pair. """
    for x, value in tuples:
        x.set_value(np.asarray(value, dtype=x.dtype))


def get_variable_shape(x):
    """Returns the shape of a tensor """
    return x.get_value(borrow=True, return_internal_type=True).shape


#%% GRAPH MANIPULATION
def function(inputs, outputs, updates=[], name=None, **kwargs):
    """Creates function for computational graphs """
    function = theano.function(inputs,
                               outputs,
                               updates=updates,
                               allow_input_downcast=True,
                               on_unused_input='ignore',
                               name=name,
                               **kwargs)
    return function


def scan(function, outputs_info=None, n_steps=None, name=None, **kwargs):
    """Scan is the for loop equivalent of Theano """
    output, updates = theano.scan(function,
                                  outputs_info=outputs_info,
                                  n_steps=n_steps,
                                  name=name,
                                  **kwargs)

    return output, updates


def gradients(loss, variables):
    """Calcuates the gradients of loss with respect to variables """
    return T.grad(loss, variables)


def stop_gradient(variables):
    """Returns `variables` but with zero gradient with respect to every other
    variables.
    """
    return theano.gradient.disconnected_grad(variables)


def sigmoid(x):
    """Sigmoid of a tensor """
    return T.nnet.sigmoid(x)


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
        seed = np.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)
    return rng


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, rng=None):
    """Random samples from the normal distribution """
    dtype, rng = _random_prep(dtype, rng)
    return rng.normal(size=shape, avg=mean, std=stddev, dtype=dtype)


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, rng=None):
    """Random samples from the uniform distribution """
    dtype, rng = _random_prep(dtype, rng)
    return rng.uniform(shape, low=minval, high=maxval, dtype=dtype)


def random_binomial(shape, p=0.0, dtype=None, rng=None):
    """Random samples from the binomial distribution """
    dtype, rng = _random_prep(dtype, rng)
    return rng.binomial(shape, p=p, dtype=dtype)


#%% CONTROL FLOW
def ifelse(condition, then_expression, else_expression):
    """Controls if else logic flow inside a computational graph """
    return theano.ifelse.ifelse(condition, then_expression, else_expression)


def until(condition):
    """Allows a scan to be a while loop """
    return theano.scan_module.until(condition)


#%% Misc
def allclose(x, y, rtol=1e-5, atol=1e-8):
    """Elementwise comparison of tensor x and y """
    return theano.tensor.allclose(x, y, rtol=rtol, atol=atol, equal_nan=False)
