"""Regularizers for weights. Simplification of keras.regularizers """

#%%
import natural_bm.backend as B


#%%
class Regularizer(object):
    """Regularizer base class. """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)


#%%
class L1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.
    
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = B.cast_to_floatx(l1)
        self.l2 = B.cast_to_floatx(l2)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += B.sum(self.l1 * B.abs(x))
        if self.l2:
            regularization += B.sum(self.l2 * B.square(x))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}


#%%
# Aliases.


def l1(l=0.01):
    return L1L2(l1=l)


def l2(l=0.01):
    return L1L2(l2=l)


def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)


#%%
def get(identifier):
    if identifier is None:
        return None
    elif isinstance(identifier, str):
        identifier = identifier.lower().strip()
        if identifier == 'l1':
            return l1()
        elif identifier == 'l2':
            return l2()
        elif identifier == 'l1_l2':
            return l1_l2()
        else:
            raise NotImplementedError
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret regularizer identifier:',
                         identifier)
