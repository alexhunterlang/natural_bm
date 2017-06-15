"""Optimizer class for calculating weight updates.

In general, this code closely matches keras.optimizers. There is a small
change in SGD.
"""

#%%
import os
import h5py
from collections import OrderedDict

import natural_bm.backend as B


#%%
def clip_norm(g, c, n):
    """Clips gradient """
    if c > 0:
        g = B.ifelse(n >= c, g * c / n, g)
    return g


#%%
class Optimizer:
    """Abstract optimizer base class.
    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)
        self.updates = OrderedDict()
        self.weights = []

    def get_updates(self, params, loss):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        grads = B.gradients(loss, params)
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = B.sqrt(sum([B.sum(B.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [B.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads

    def set_weights(self, weights):
        """Sets the weights of the optimizer, from Numpy arrays.
        Should only be called after computing the gradients
        (otherwise the optimizer has no weights).
        
        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the optimizer (i.e. it should match the
                output of `get_weights`).
        
        # Raises
            ValueError: in case of incompatible weight shapes.
        """
        params = self.weights
        weight_value_tuples = []
        param_values = B.batch_get_value(params)
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise ValueError('Optimizer weight shape ' +
                                 str(pv.shape) +
                                 ' not compatible with '
                                 'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        B.batch_set_value(weight_value_tuples)

    def get_weights(self):
        """Returns the current value of the weights of the optimizer.
        
        # Returns
            A list of numpy arrays.
        """
        return B.batch_get_value(self.weights)

    def save_weights(self, filepath, overwrite=True):
        """ Dumps all layer weights to a HDF5 file. """
        
        # If file exists and should not be overwritten:
        if not overwrite and os.path.isfile(filepath):
            raise NotImplementedError
        f = h5py.File(filepath, 'w')
        self.save_weights_to_hdf5_group(f)
        f.flush()
        f.close()

    def save_weights_to_hdf5_group(self, f):
        
        weights = self.weights
        f.attrs['weights'] = [w.name.encode('utf8') for w in weights]

        for w in weights:
            g = f.create_group(w.name)
            weight_values = [w.get_value()]
            weight_names = [w.name.encode('utf8')]
            g.attrs['weight_names'] = weight_names
            for name, val in zip(weight_names, weight_values):
                param_dset = g.create_dataset(name, val.shape,
                                              dtype=val.dtype)
                if not val.shape:
                    # scalar
                    param_dset[()] = val
                else:
                    param_dset[:] = val

    def get_config(self):
        config = {}
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


#%%
class SGD(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    
    Additional support for ramping up and down momentum to adhere to advice
    in paper cited below.
    
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        schedule_decay: float >= 0. Controls how fast to ramp up momentum.
        mom_iter_max: int >= 0. Batch iteration at which to hit final, smaller momentum.
        
    # References
        - [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    """

    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False,
                 schedule_decay=0.004, mom_iter_max=500*950, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.iterations = B.variable(0., name='iterations')
        self.lr = B.variable(lr, name='lr')
        self.momentum = B.variable(momentum, name='momentum')
        self.momentum_goal = B.variable(momentum, name='momentum_goal')
        self.decay = B.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.schedule_decay = B.variable(schedule_decay, name='schedule_decay')
        self.mom_iter_max = B.cast(mom_iter_max, B.floatx())

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        self.updates = OrderedDict()

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates[self.iterations] = self.iterations + 1
            
        momentum = B.ifelse(B.less(self.iterations, self.mom_iter_max),
                         (self.momentum_goal
                         *(1. - 0.5 * (B.pow(0.96, self.iterations * self.schedule_decay)))
                         *(1. - 0.5 * (B.pow(0.96, (self.mom_iter_max-self.iterations) * self.schedule_decay)))),
                         0.5*self.momentum_goal)
        self.updates[self.momentum] = momentum

        # momentum
        moments = []
        for p in params:
            shape = B.get_variable_shape(p)
            name = p.name + '_moment'
            moment =B.zeros(shape, name=name)
            moments.append(moment)
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates[m] = v

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            self.updates[p] = new_p
        return self.updates

    def get_config(self):
        config = {'lr': float(B.get_value(self.lr)),
                  'momentum': float(B.get_value(self.momentum)),
                  'decay': float(B.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#%%
class Adam(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        super(Adam, self).__init__(**kwargs)
        self.iterations = B.variable(0, name='iterations')
        self.lr = B.variable(lr, name='lr')
        self.beta_1 = B.variable(beta_1, name='beta_1')
        self.beta_2 = B.variable(beta_2, name='beta_2')
        self.epsilon = epsilon
        self.decay = B.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        self.updates = OrderedDict()
        self.updates[self.iterations] = self.iterations+1

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (B.sqrt(1. - B.pow(self.beta_2, t)) /
                     (1. - B.pow(self.beta_1, t)))

        ms = []
        vs = []
        for p in params:
            shape = B.get_variable_shape(p)
            name = p.name + '_ms'
            ms.append(B.zeros(shape, name=name))
            name = p.name + '_vs'
            vs.append(B.zeros(shape, name=name))
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * B.square(g)
            p_t = p - lr_t * m_t / (B.sqrt(v_t) + self.epsilon)

            self.updates[m] = m_t
            self.updates[v] = v_t

            new_p = p_t
            self.updates[p] = new_p
        return self.updates

    def get_config(self):
        config = {'lr': float(B.get_value(self.lr)),
                  'beta_1': float(B.get_value(self.beta_1)),
                  'beta_2': float(B.get_value(self.beta_2)),
                  'decay': float(B.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#%%
class Nadam(Optimizer):
    """Nesterov Adam optimizer.
    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.
    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    """

    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, schedule_decay=0.004, **kwargs):
        super(Nadam, self).__init__(**kwargs)
        self.iterations = B.variable(0., name='iterations')
        self.m_schedule = B.variable(1., name='m_schedule')
        self.lr = B.variable(lr, name='lr')
        self.beta_1 = B.variable(beta_1, name='beta_1')
        self.beta_2 = B.variable(beta_2, name='beta_2')
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        self.updates = OrderedDict()
        self.updates[self.iterations] = self.iterations + 1

        t = self.iterations + 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (B.pow(0.96, t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (B.pow(0.96, (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates[self.m_schedule] = m_schedule_new

        ms = []
        vs = []
        for p in params:
            shape = B.get_variable_shape(p)
            name = p.name + '_ms'
            ms.append(B.zeros(shape, name=name))
            name = p.name + '_vs'
            vs.append(B.zeros(shape, name=name))

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * B.square(g)
            v_t_prime = v_t / (1. - B.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

            self.updates[m] = m_t
            self.updates[v] = v_t

            p_t = p - self.lr * m_t_bar / (B.sqrt(v_t_prime) + self.epsilon)
            new_p = p_t

            self.updates[p] = new_p
        return self.updates

    def get_config(self):
        config = {'lr': float(B.get_value(self.lr)),
                  'beta_1': float(B.get_value(self.beta_1)),
                  'beta_2': float(B.get_value(self.beta_2)),
                  'epsilon': self.epsilon,
                  'schedule_decay': self.schedule_decay}
        base_config = super(Nadam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
