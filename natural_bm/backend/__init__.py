"""
This follows the init used by keras
"""
import os
import json
import sys
from .common import epsilon
from .common import floatx
from .common import intx
from .common import set_epsilon
from .common import set_floatx
from .common import set_intx
from .common import cast_to_floatx

# Obtain base dir path: either ~/.natural_bm or /tmp.
_naturalbm_base_dir = os.path.expanduser('~')
if not os.access(_naturalbm_base_dir, os.W_OK):
    _naturalbm_base_dir = '/tmp'
_naturalbm_dir = os.path.join(_naturalbm_base_dir, '.natural_bm')

# Default backend: Theano.
_BACKEND = 'theano'

# Attempt to read config file.
_config_path = os.path.expanduser(os.path.join(_naturalbm_dir, 'natural_bm.json'))
if os.path.exists(_config_path):
    try:
        _config = json.load(open(_config_path))
    except ValueError:
        _config = {}
    _floatx = _config.get('floatx', floatx())
    assert _floatx in {'float16', 'float32', 'float64'}
    _epsilon = _config.get('epsilon', epsilon())
    assert _floatx in {'float16', 'float32', 'float64'}
    _backend = _config.get('backend', _BACKEND)
    assert _backend in {'theano', 'numpy'}

    set_floatx(_floatx)
    set_epsilon(_epsilon)
    _BACKEND = _backend

# Save config file, if possible.
if not os.path.exists(_naturalbm_dir):
    try:
        os.makedirs(_naturalbm_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

if not os.path.exists(_config_path):
    _config = {
        'floatx': floatx(),
        'epsilon': epsilon(),
        'backend': _BACKEND,
    }
    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass

# Set backend based on NATURAL_BM_BACKEND flag, if applicable.
if 'NATURAL_BM_BACKEND' in os.environ:
    _backend = os.environ['NATURAL_BM_BACKEND']
    assert _backend in {'theano', 'numpy'}
    _BACKEND = _backend

# Import backend functions.
if _BACKEND == 'theano':
    sys.stderr.write('Using Theano backend.\n')
    from .theano_backend import *
elif _BACKEND == 'numpy':
    sys.stderr.write('Using Numpy backend.\n')
    from .numpy_backend import *
else:
    raise ValueError('Unknown backend: ' + str(_BACKEND))


def backend():
    """Publicly accessible method
    for determining the current backend.
    # Returns
        String, the name of the backend Natural_BM is currently using.
    # Example
    ```python
        >>> natural_bm.backend.backend()
        'theano'
    ```
    """
    return _BACKEND
