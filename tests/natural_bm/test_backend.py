#%%
import pytest
from numpy.testing import assert_allclose
import numpy as np

import natural_bm.backend.theano_backend as BTH
import natural_bm.backend.numpy_backend as BNP
from natural_bm.backend.common import floatx, set_floatx


#%% Define checks
def check_dtype(var, dtype):
    assert var.dtype == dtype


def check_single_tensor_operation(function_name, input_shape, **kwargs):
    val = np.random.random(input_shape) - 0.5
    xth = BTH.variable(val)
    xnp = BNP.variable(val)

    _zth = getattr(BTH, function_name)(xth, **kwargs)
    zth = BTH.eval(_zth)
    znp = BNP.eval(getattr(BNP, function_name)(xnp, **kwargs))

    assert zth.shape == znp.shape
    assert_allclose(zth, znp, atol=1e-05)


def check_two_tensor_operation(function_name, x_input_shape,
                               y_input_shape, **kwargs):
    xval = np.random.random(x_input_shape) - 0.5

    xth = BTH.variable(xval)
    xnp = BNP.variable(xval)

    yval = np.random.random(y_input_shape) - 0.5

    yth = BTH.variable(yval)
    ynp = BNP.variable(yval)

    _zth = getattr(BTH, function_name)(xth, yth, **kwargs)
    zth = BTH.eval(_zth)
    znp = BNP.eval(getattr(BNP, function_name)(xnp, ynp, **kwargs))

    assert zth.shape == znp.shape
    assert_allclose(zth, znp, atol=1e-05)


def check_composed_tensor_operations(first_function_name, first_function_args,
                                     second_function_name, second_function_args,
                                     input_shape):
    ''' Creates a random tensor t0 with shape input_shape and compute
                 t1 = first_function_name(t0, **first_function_args)
                 t2 = second_function_name(t1, **second_function_args)
        with both Theano and TensorFlow backends and ensures the answers match.
    '''
    val = np.random.random(input_shape) - 0.5
    xth = BTH.variable(val)
    xnp = BNP.variable(val)

    yth = getattr(BTH, first_function_name)(xth, **first_function_args)
    ynp = getattr(BNP, first_function_name)(xnp, **first_function_args)

    zth = BTH.eval(getattr(BTH, second_function_name)(yth, **second_function_args))
    znp = BNP.eval(getattr(BNP, second_function_name)(ynp, **second_function_args))

    assert zth.shape == znp.shape
    assert_allclose(zth, znp, atol=1e-05)


#%%
def test_linear_operations():
    check_two_tensor_operation('dot', (4, 2), (2, 4))
    check_two_tensor_operation('dot', (4, 2), (5, 2, 3))

    check_single_tensor_operation('transpose', (4, 2))
    check_single_tensor_operation('reverse', (4, 3, 2), axes=1)
    check_single_tensor_operation('reverse', (4, 3, 2), axes=(1, 2))


#%%
def test_linear_algebra_operations():
    check_single_tensor_operation('diag', (4, 4))

    check_two_tensor_operation('solve', (4, 4), (4,))
    check_two_tensor_operation('fill_diagonal', (4, 4), (1,))
    check_two_tensor_operation('fill_diagonal', (4, 4), (4,))


#%%
def test_svd():
    input_shape = (4, 4)

    val = np.random.random(input_shape) - 0.5
    xth = BTH.variable(val)
    xnp = BNP.variable(val)

    Uth, Sth, Vth = BTH.svd(xth)
    Unp, Snp, Vnp = BNP.svd(xnp)

    Uth = Uth.eval()
    Sth = Sth.eval()
    Vth = Vth.eval()

    assert Uth.shape == Unp.shape
    assert Sth.shape == Snp.shape
    assert Vth.shape == Vnp.shape
    assert_allclose(Uth, Unp, atol=1e-05)
    assert_allclose(Sth, Snp, atol=1e-05)
    assert_allclose(Vth, Vnp, atol=1e-05)


#%%
def test_shape_operations():
    # concatenate
    xval = np.random.random((4, 3))
    xth = BTH.variable(xval)
    xnp = BNP.variable(xval)
    yval = np.random.random((4, 2))
    yth = BTH.variable(yval)
    ynp = BNP.variable(yval)
    zth = BTH.eval(BTH.concatenate([xth, yth], axis=-1))
    znp = BNP.eval(BNP.concatenate([xnp, ynp], axis=-1))
    assert zth.shape == znp.shape
    assert_allclose(zth, znp, atol=1e-05)

    check_single_tensor_operation('reshape', (4, 2), shape=(8, 1))
    check_single_tensor_operation('permute_dimensions', (4, 2, 3),
                                  pattern=(2, 0, 1))
    check_single_tensor_operation('repeat', (4, 1), n=3)
    check_single_tensor_operation('flatten', (4, 1))
    check_single_tensor_operation('squeeze', (4, 3, 1), axis=2)
    check_single_tensor_operation('squeeze', (4, 1, 1), axis=1)
    check_composed_tensor_operations('reshape', {'shape': (4, 3, 1, 1)},
                                     'squeeze', {'axis': 2},
                                     (4, 3, 1, 1))


#%%
def test_repeat_elements():
    reps = 3
    for ndims in [1, 2, 3]:
        shape = np.arange(2, 2 + ndims)
        arr = np.arange(np.prod(shape)).reshape(shape)
        arr_th = BTH.variable(arr)
        arr_np = BNP.variable(arr)

        for rep_axis in range(ndims):
            np_rep = np.repeat(arr, reps, axis=rep_axis)
            th_z = BTH.repeat_elements(arr_th, reps, axis=rep_axis)
            th_rep = BTH.eval(th_z)
            bnp_rep = BNP.eval(
                BNP.repeat_elements(arr_np, reps, axis=rep_axis))

            assert th_rep.shape == np_rep.shape
            assert bnp_rep.shape == np_rep.shape
            assert_allclose(np_rep, th_rep, atol=1e-05)
            assert_allclose(np_rep, bnp_rep, atol=1e-05)


#%%
def test_tile():
    shape = (3, 4)
    arr = np.arange(np.prod(shape)).reshape(shape)
    arr_th = BTH.variable(arr)
    arr_np = BNP.variable(arr)

    n = (2, 1)
    th_z = BTH.tile(arr_th, n)
    th_rep = BTH.eval(th_z)
    np_rep = BNP.eval(BNP.tile(arr_np, n))
    assert_allclose(np_rep, th_rep, atol=1e-05)


#%%
def test_value_manipulation():
    val = np.random.random((4, 2))
    xth = BTH.variable(val)
    xnp = BNP.variable(val)

    # get_value
    valth = BTH.get_value(xth)
    valnp = BNP.get_value(xnp)
    assert valnp.shape == valth.shape
    assert_allclose(valth, valnp, atol=1e-05)

    # set_value
    BTH.set_value(xth, val)

    valth = BTH.get_value(xth)
    assert valnp.shape == val.shape
    assert_allclose(valth, val, atol=1e-05)


#%%
def test_elementwise_operations():
    check_single_tensor_operation('max', (4, 2))
    check_single_tensor_operation('max', (4, 2), axis=1, keepdims=True)

    check_single_tensor_operation('min', (4, 2))
    check_single_tensor_operation('min', (4, 2), axis=1, keepdims=True)
    check_single_tensor_operation('min', (4, 2, 3), axis=[1, -1])

    check_single_tensor_operation('mean', (4, 2))
    check_single_tensor_operation('mean', (4, 2), axis=1, keepdims=True)
    check_single_tensor_operation('mean', (4, 2, 3), axis=-1, keepdims=True)
    check_single_tensor_operation('mean', (4, 2, 3), axis=[1, -1])

    check_single_tensor_operation('std', (4, 2))
    check_single_tensor_operation('std', (4, 2), axis=1, keepdims=True)
    check_single_tensor_operation('std', (4, 2, 3), axis=[1, -1])

    check_single_tensor_operation('prod', (4, 2))
    check_single_tensor_operation('prod', (4, 2), axis=1, keepdims=True)
    check_single_tensor_operation('prod', (4, 2, 3), axis=[1, -1])

    check_single_tensor_operation('cumsum', (4, 2))
    check_single_tensor_operation('cumsum', (4, 2), axis=1)

    check_single_tensor_operation('cumprod', (4, 2))
    check_single_tensor_operation('cumprod', (4, 2), axis=1)

    check_single_tensor_operation('argmax', (4, 2))
    check_single_tensor_operation('argmax', (4, 2), axis=1)

    check_single_tensor_operation('argmin', (4, 2))
    check_single_tensor_operation('argmin', (4, 2), axis=1)

    check_single_tensor_operation('square', (4, 2))
    check_single_tensor_operation('abs', (4, 2))
    check_single_tensor_operation('sqrt', (4, 2))
    check_single_tensor_operation('exp', (4, 2))
    check_single_tensor_operation('log', (4, 2))
    check_single_tensor_operation('round', (4, 2))
    check_single_tensor_operation('sign', (4, 2))
    check_single_tensor_operation('pow', (4, 2), a=3)
    check_single_tensor_operation('clip', (4, 2), min_value=0.4,
                                  max_value=0.6)

    # two-tensor ops
    check_two_tensor_operation('equal', (4, 2), (4, 2))
    check_two_tensor_operation('not_equal', (4, 2), (4, 2))
    check_two_tensor_operation('greater', (4, 2), (4, 2))
    check_two_tensor_operation('greater_equal', (4, 2), (4, 2))
    check_two_tensor_operation('less', (4, 2), (4, 2))
    check_two_tensor_operation('less_equal', (4, 2), (4, 2))
    check_two_tensor_operation('maximum', (4, 2), (4, 2))
    check_two_tensor_operation('minimum', (4, 2), (4, 2))


#%%
@pytest.mark.parametrize('x_np,axis,keepdims', [
    (np.array([1.1, 0.8, 0.9]), 0, False),
    (np.array([[1.1, 0.8, 0.9]]), 0, False),
    (np.array([[1.1, 0.8, 0.9]]), 1, False),
    (np.array([[1.1, 0.8, 0.9]]), -1, False),
    (np.array([[1.1, 0.8, 0.9]]), 1, True),
    (np.array([[1.1], [1.2]]), 0, False),
    (np.array([[1.1], [1.2]]), 1, False),
    (np.array([[1.1], [1.2]]), -1, False),
    (np.array([[1.1], [1.2]]), -1, True),
    (np.array([[1.1, 1.2, 1.3], [0.9, 0.7, 1.4]]), None, False),
    (np.array([[1.1, 1.2, 1.3], [0.9, 0.7, 1.4]]), 0, False),
    (np.array([[1.1, 1.2, 1.3], [0.9, 0.7, 1.4]]), 1, False),
    (np.array([[1.1, 1.2, 1.3], [0.9, 0.7, 1.4]]), -1, False),
])
@pytest.mark.parametrize('B', [BTH, BNP], ids=["BTH", "BNP"])
def test_logsumexp(x_np, axis, keepdims, B):
    '''
    Check if K.logsumexp works properly for values close to one.
    '''
    x = B.variable(x_np)
    assert_allclose(B.eval(B.logsumexp(x, axis=axis, keepdims=keepdims)),
                    np.log(np.sum(np.exp(x_np), axis=axis, keepdims=keepdims)),
                    rtol=1e-5)


#%%
@pytest.mark.parametrize('B', [BTH, BNP], ids=["BTH", "BNP"])
def test_logsumexp_optim(B):
    '''
    Check if optimization works.
    '''
    x_np = np.array([1e+4, 1e-4])
    assert_allclose(B.eval(B.logsumexp(B.variable(x_np), axis=0)),
                    1e4,
                    rtol=1e-5)


#%%
def test_switch():
    val = np.random.random()
    xth = BTH.variable(val)
    xth = BTH.ifelse(xth >= 0.5, xth * 0.1, xth * 0.2)

    xnp = BNP.variable(val)
    xnp = BNP.ifelse(xnp >= 0.5, xnp * 0.1, xnp * 0.2)

    zth = BTH.eval(xth)
    znp = BNP.eval(xnp)

    assert zth.shape == znp.shape
    assert_allclose(zth, znp, atol=1e-05)


#%%
def test_nn_operations():
    check_single_tensor_operation('sigmoid', (4, 2))


#%%
def test_random_normal():
    mean = 0.
    std = 1.
    rand = BNP.eval(BNP.random_normal((1000, 1000), mean=mean, stddev=std))
    assert rand.shape == (1000, 1000)
    assert np.abs(np.mean(rand) - mean) < 0.01
    assert np.abs(np.std(rand) - std) < 0.01

    rand = BTH.eval(BTH.random_normal((1000, 1000), mean=mean, stddev=std))
    assert rand.shape == (1000, 1000)
    assert np.abs(np.mean(rand) - mean) < 0.01
    assert np.abs(np.std(rand) - std) < 0.01


#%%
def test_random_uniform():
    min_val = -1.
    max_val = 1.
    rand = BNP.eval(BNP.random_uniform((1000, 1000), min_val, max_val))
    assert rand.shape == (1000, 1000)
    assert np.abs(np.mean(rand)) < 0.01
    assert np.max(rand) <= max_val
    assert np.min(rand) >= min_val

    rand = BTH.eval(BTH.random_uniform((1000, 1000), min_val, max_val))
    assert rand.shape == (1000, 1000)
    assert np.abs(np.mean(rand)) < 0.01
    assert np.max(rand) <= max_val
    assert np.min(rand) >= min_val


#%%
def test_random_binomial():
    p = 0.5
    rand = BNP.eval(BNP.random_binomial((1000, 1000), p))
    assert rand.shape == (1000, 1000)
    assert np.abs(np.mean(rand) - p) < 0.01
    assert np.max(rand) == 1
    assert np.min(rand) == 0

    rand = BTH.eval(BTH.random_binomial((1000, 1000), p))
    assert rand.shape == (1000, 1000)
    assert np.abs(np.mean(rand) - p) < 0.01
    assert np.max(rand) == 1
    assert np.min(rand) == 0


#%%
def test_one_hot():
    input_length = 10
    num_classes = 20
    batch_size = 30
    indices = np.random.randint(0, num_classes, size=(batch_size, input_length))
    oh = np.eye(num_classes)[indices]
    for B in [BTH, BNP]:
        koh = B.eval(B.one_hot(B.variable(indices, dtype='int32'), num_classes))
        assert np.all(koh == oh)


#%%
def test_arange():
    for test_value in (-20, 0, 1, 10):
        t_a = BNP.arange(test_value)
        a = BNP.eval(t_a)
        assert np.array_equal(a, np.arange(test_value))
        t_b = BTH.arange(test_value)
        b = BTH.eval(t_b)
        assert np.array_equal(b, np.arange(test_value))
        assert np.array_equal(a, b)
        assert BNP.dtype(t_a) == BTH.dtype(t_b)
    for start, stop, step in ((0, 5, 1), (-5, 5, 2), (0, 1, 2)):
        a = BNP.eval(BNP.arange(start, stop, step))
        assert np.array_equal(a, np.arange(start, stop, step))
        b = BTH.eval(BTH.arange(start, stop, step))
        assert np.array_equal(b, np.arange(start, stop, step))
        assert np.array_equal(a, b)
    for dtype in ('int32', 'int64', 'float32', 'float64'):
        for backend in (BNP, BTH):
            t = backend.arange(10, dtype=dtype)
            assert backend.dtype(t) == dtype


#%%
def test_setfloatx_incorrect_values():
    # Keep track of the old value
    old_floatx = floatx()
    # Try some incorrect values
    initial = floatx()
    for value in ['', 'beerfloat', 123]:
        with pytest.raises(Exception):
            set_floatx(value)
    assert floatx() == initial
    # Restore old value
    set_floatx(old_floatx)


#%%
def test_setfloatx_correct_values():
    # Keep track of the old value
    old_floatx = floatx()
    # Check correct values
    for value in ['float16', 'float32', 'float64']:
        set_floatx(value)
        assert floatx() == value
    # Restore old value
    set_floatx(old_floatx)


#%%
def test_set_floatx():
    """
    Make sure that changes to the global floatx are effectively
    taken into account by the backend.
    """
    # Keep track of the old value
    old_floatx = floatx()

    set_floatx('float16')
    var = BTH.variable([10])
    check_dtype(var, 'float16')

    set_floatx('float64')
    var = BTH.variable([10])
    check_dtype(var, 'float64')

    # Restore old value
    set_floatx(old_floatx)


#%%
if __name__ == '__main__':
    pytest.main([__file__])
