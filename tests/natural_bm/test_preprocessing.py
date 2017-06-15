#%%
import numpy as np
import pytest

from natural_bm.preprocessing import make_PCA_matrix, make_ZCA_matrix
import natural_bm.backend.theano_backend as BTH
import natural_bm.backend.numpy_backend as BNP


#%% Test prep for tests
def _diag_non_diag(x):
    diag = np.copy(np.diag(x))
    index = np.where(~np.eye(x.shape[0], dtype=bool))
    non_diag = x[index]
    return diag, non_diag


def setup_data():
    n = 10
    data = np.random.normal(size=(n, n))
    cov = np.cov(data.T)

    return data, cov


def setup_datatype(B, data, cov):

    data = B.variable(data)
    cov = B.variable(cov)

    return data, cov


def setup_white(whitetype, cov, eps):
    if whitetype == 'PCA':
        white = make_PCA_matrix(cov, eps)
    elif whitetype == 'ZCA':
        white = make_ZCA_matrix(cov, eps)
    else:
        raise NotImplementedError

    return white


def verify(whitetype, eps, cov, new_cov):

    # break into diag and non-diagonal
    diag, non_diag = _diag_non_diag(new_cov)

    if whitetype == 'PCA':
        atol = 2e-2

        # Non-diagonal elements should all be zero
        assert np.allclose(non_diag, 0.0,  atol=atol)

        if eps == 1e-2:
            # first element is one
            assert np.isclose(diag[0], 1.0, atol=atol)
            # other elements, besides last, should be greater than zero
            assert np.all(diag[1:-1] > 0.0)
        elif eps == 1e-5:
            # last element is zero, but everyone else should be one
            assert np.allclose(diag[:-1], 1.0, atol=atol)
        else:
            raise NotImplementedError

    elif whitetype == 'ZCA':
        # break old cov into diag and non-diagonal
        diag_old, non_diag_old = _diag_non_diag(cov)

        # checks on diagonal
        assert np.max(diag) <= 1.0
        assert np.min(diag) >= 0.0

        # checks on non-diagonal, just a statistical argument
        assert np.std(non_diag) < 0.75*np.std(non_diag_old)

    else:
        raise NotImplementedError


#%%
@pytest.mark.parametrize('whitetype', ['PCA', 'ZCA'], ids=['PCA', 'ZCA'])
@pytest.mark.parametrize('B', [BTH, BNP], ids=["BTH", "BNP"])
@pytest.mark.parametrize('eps', [1e-2, 1e-5], ids=['1e-2', '1e-5'])
def test_white(whitetype, B, eps):

    data, cov = setup_data()
    data, cov = setup_datatype(B, data, cov)
    white = setup_white(whitetype, cov, eps)
    new_data = data.dot(white)
    if B == BTH:
        cov = B.get_value(cov)
        new_data =  B.eval(new_data)
    new_cov = np.cov(new_data.T)
    verify(whitetype, eps, cov, new_cov)


#%% Main
if __name__ == '__main__':
    pytest.main([__file__])
