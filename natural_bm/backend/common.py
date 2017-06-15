"""Common operations for all backends.

These functions control global constants.
"""

#%%
import numpy as np

_FLOATX = 'float32'
_INTX = 'int32'
_EPSILON = np.finfo(_FLOATX).eps


#%% 
def epsilon():
    """Returns the value of the fuzz
    factor used in numeric expressions.
    
    # Returns
        A float.
    """
    return _EPSILON


def set_epsilon(e):
    """Sets the value of the fuzz
    factor used in numeric expressions.
    
    # Arguments
        e: float. New value of epsilon.
    """
    global _EPSILON
    _EPSILON = e


def floatx():
    """Returns the default float type, as a string.
    (e.g. 'float16', 'float32', 'float64').
    
    # Returns
        String, the current default float type.
    """
    return _FLOATX


def set_floatx(floatx):
    """Sets the default float type.
    
    # Arguments
        String: 'float16', 'float32', or 'float64'.
    """
    global _FLOATX
    if floatx not in {'float16', 'float32', 'float64'}:
        raise ValueError('Unknown floatx type: ' + str(floatx))
    _FLOATX = str(floatx)


def cast_to_floatx(x):
    """Cast a Numpy array to the default Keras float type.
    
    # Arguments
        x: Numpy array.
    
    # Returns
        The same Numpy array, cast to its new type.
    """
    return np.asarray(x, dtype=_FLOATX)


def intx():
    """Returns the default int type, as a string.
    (e.g. 'int16', 'int32', 'int64').
    
    # Returns
        String, the current default int type.
    """
    return _INTX


def set_intx(intx):
    """Sets the default int type.
    
    # Arguments
        String: 'int16', 'int32', or 'int64'.
    """
    global _INTX
    if intx not in {'int16', 'int32', 'int64'}:
        raise ValueError('Unknown intx type: ' + str(intx))
    _INTX = str(intx)
