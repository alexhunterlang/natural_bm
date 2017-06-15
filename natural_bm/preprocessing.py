"""Provides functions for preprocessing of images. """

#%%
import numpy as np

from natural_bm.backend import theano_backend as BTH
from natural_bm.backend import numpy_backend as BNP


#%% helper functions
def _set_backend(x):
    """Whitening matrices could be made based on numpy or theano variables """
    if isinstance(x, np.ndarray):
        B = BNP
    else:
        B = BTH

    return B


def _variance_stabilization(cov, eps):
    """Increase numerical stability by clipping diagonal of covariance """

    B = _set_backend(cov)

    var = B.diag(cov)
    var = B.maximum(var, eps)
    cov = B.fill_diagonal(cov, var)

    return cov


#%%
def make_PCA_matrix(cov, eps=1e-3):
    """PCA whitening """

    B = _set_backend(cov)

    eps_inv = B.cast(eps, B.floatx())  # user supplied eps for taking inverse
    eps_machine = B.epsilon()  # machine epsilon
    eps_var = 100*eps_machine  # want extra safety for variance epsilon

    cov = _variance_stabilization(cov, eps_var)

    # take the inverse
    # minor difference between PCA and ZCA for numerical stability
    U, S, V = B.svd(cov)
    S = B.maximum(S, eps_machine)
    S_inv = B.diag(1. / B.sqrt(S + eps_inv))

    # create whitening matrix
    # main difference between PCA and ZCA
    PCA = S_inv.dot(U.T).T

    PCA = B.cast(PCA, B.floatx())

    return PCA


#%%
def make_ZCA_matrix(cov, eps=1e-3):
    """ZCA whitening """

    B = _set_backend(cov)

    eps_inv = B.cast(eps, B.floatx())  # user supplied eps for taking inverse
    eps_machine = B.epsilon()  # machine epsilon
    eps_var = 100*eps_machine  # want extra safety for variance epsilon

    cov = _variance_stabilization(cov, eps_var)

    n = cov.shape[0]
    if hasattr(n, 'eval'):
        n = n.eval()

    # take the inverse
    # minor difference between PCA and ZCA for numerical stability
    # Need to implement stochastic robust approximation
    # For details see https://arxiv.org/abs/1205.1828, section 2.6, eqn 35
    A = cov.T.dot(cov) + eps_inv*B.eye(n)
    X = B.solve(A, cov.T)
    U, S, V = B.svd(X)
    S_inv = B.diag(B.sqrt(B.maximum(S, eps_machine)))

    # create whitening matrix
    # main difference between PCA and ZCA
    ZCA = U.dot(S_inv).dot(U.T).T

    ZCA = B.cast(ZCA, B.floatx())

    return ZCA
