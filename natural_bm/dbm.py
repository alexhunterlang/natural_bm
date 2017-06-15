"""Deep Boltzmann Machines (DBM).

This can create DBMs with almost any type of topology connecting layers.
The only restriction is that when the layers are in topological order, 
only connections between even to odd (and odd to even) layers are allowed.
This restriction allows for efficient free energy calculations and
Gibbs sampling since the layers can be updated in only two steps
(even then odd or odd then even).

Technically these are deep restricted boltzmann machines (DRBM) since each
layer of neurons has only external connections and no within layer connections.
However, the literature convention is that DBM actually means DRBM.

Can also create a standard restricted boltzmann machine (RBM) since this
is just a DBM with only two layers.
"""

#%%
import os
from collections import OrderedDict
import numpy as np
import h5py

import natural_bm.backend as B
from natural_bm.components import Synapse, Layer
from natural_bm import samplers


#%%
def prep_topology(topology_dict):
    """Prepares for connecting together Layers and Synapses to create a neural netowrk.
    
    # Arguments:
        topology_dict: dict; key is the lower layer, value is the set of higher layers connected to lower layer 
        
    # Returns:
        pairs: list; tuples of layers that need a Synapse
        topology_input_dict: dict; reversed topology dict (keys are higher layers)
    """

    # First check that parity is consistent with current Gibbs sampler
    parity_violation = 0
    for key, value in topology_dict.items():
        parity = key % 2
        IS_diff = np.array([parity != (v % 2) for v in value])
        if not np.all(IS_diff):
            parity_violation += 1

    if parity_violation:
        raise ValueError('Current DBM Gibbs updates assumes odd / even states are conditionally independent.')

    # This finds the inputs to each layer
    topology_input_dict = {}
    for key, value in topology_dict.items():
        for i in list(value):
            if i not in topology_input_dict.keys():
                topology_input_dict[i] = [key]
            else:
                topology_input_dict[i] += [key]
    # convert to a set to have same format as topology_dict
    for key, value in topology_input_dict.items():
        topology_input_dict[key] = set(value)

    pairs = []
    for k, v in topology_dict.items():
        for i in list(v):
            pairs.append((k, i))

    return pairs, topology_input_dict


#%%
class DBM:
    """Deep Boltzmann Machines (DBM) """
    def __init__(self,
                 layer_size_list,
                 topology_dict,
                 rng=None,
                 W_regularizer=None,
                 b_regularizer=None):
        """Creates a Deep Boltzmann Machine.
        
        # Arguments:
            layer_size_list: list; size of each Layer
            topology_dict: dict; controls which Layers are connected
            rng: Random Number Generator, optional
            W_regularizer: Regularizer object, optional
            b_regularizer: Regularizer object, optional
            
        # References:
            1. Deep boltzmann machines by R Salakhutdinov and G Hinton. AIS, 2009.
        """

        self.layer_size_list = layer_size_list
        self.topology_dict = topology_dict

        self.IS_rbm = (len(self.layer_size_list) == 2)

        self.layers = []
        self.synapses = []
        self._updates = OrderedDict()

        if rng is None:
            rng = B.make_rng()
        self.rng = rng

        pairs, topology_input_dict = prep_topology(self.topology_dict)
        self.synapse_pairs = pairs
        self.topology_input_dict = topology_input_dict

        # first prepare all the weights connecting layers
        self.synapses = []
        for sp in self.synapse_pairs:
            name = 'h{}-{}'.format(sp[0], sp[1])
            shape = (self.layer_size_list[sp[0]], self.layer_size_list[sp[1]])
            init_W = np.zeros(shape=shape, dtype=B.floatx())

            synapse = Synapse(name, init_W, regularizer=W_regularizer)
            self.synapses.append(synapse)

        # now make the layers and feed in the given weights
        for i, output_dim in enumerate(self.layer_size_list):
            name = 'h'+str(i)
            init_b = np.zeros(shape=output_dim, dtype=B.floatx())

            up_dict = {}
            down_dict = {}
            for j, sp in enumerate(self.synapse_pairs):
                if i == sp[0]:
                    down_dict[sp[1]] = self.synapses[j].W
                elif i == sp[1]:
                    up_dict[sp[0]] = self.synapses[j].W

            layer = Layer(name, init_b, up_dict, down_dict, regularizer=b_regularizer)

            self.layers.append(layer)

        if not self.IS_rbm:
            self.meanfield_sampler = samplers.Meanfield(self)


    @property
    def parts(self):
        """All neural network parts, layers and synapses """
        return self.layers + self.synapses

    @property
    def trainable_weights(self):
        """Weights that are optimized by gradients """
        weights = []
        for part in self.parts:
            weights += part.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        """Weights that are not optimized by gradients """
        weights = []
        for part in self.parts:
            weights += part.non_trainable_weights
        return weights

    def get_weights(self):
        """Returns the weights of the model,
        as a flat list of Numpy arrays.
        """
        weights = []
        for part in self.parts:
            weights += part.weights
        return [w.get_value() for w in weights]

    def save_weights(self, filepath, overwrite=True):
        """Dumps all layer weights to a HDF5 file.
        The weight file has:
            - `layer_names` (attribute), a list of strings
                (ordered names of model layers).
            - For every layer, a `group` named `layer.name`
                - For every such layer group, a group attribute `weight_names`,
                    a list of strings
                    (ordered names of weights tensor of the layer).
                - For every weight in the layer, a dataset
                    storing the weight value, named after the weight tensor.
        """

        # If file exists and should not be overwritten:
        if not overwrite and os.path.isfile(filepath):
            raise NotImplementedError
        f = h5py.File(filepath, 'w')
        self.save_weights_to_hdf5_group(f)
        f.flush()
        f.close()

    def save_weights_to_hdf5_group(self, f):
        """Saves the weights """
        parts = self.parts
        f.attrs['part_names'] = [p.name.encode('utf8') for p in parts]

        for part in parts:
            g = f.create_group(part.name)
            symbolic_weights = part.weights
            weight_values = [w.get_value() for w in symbolic_weights]
            weight_names = []
            for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
                if hasattr(w, 'name') and w.name:
                    name = str(w.name)
                else:
                    name = 'param_' + str(i)
                weight_names.append(name.encode('utf8'))
            g.attrs['weight_names'] = weight_names
            for name, val in zip(weight_names, weight_values):
                param_dset = g.create_dataset(name, val.shape,
                                              dtype=val.dtype)
                if not val.shape:
                    # scalar
                    param_dset[()] = val
                else:
                    param_dset[:] = val


    def propup(self, x, beta=1.0):
        """Propagates visible layer inputs to deeper layers in the network.
        
        # Arguments:
            x: tensor; visible inputs
            beta: float, optional; inverse temperature
        
        # Returns:
            prob_ls: list of probabilies of activity for each layer
        """

        prob_ls = [x]+[None]*(len(self.layers)-1)

        for i, layer in enumerate(self.layers[1:]):
            prob_ls[i+1] = layer.output(prob_ls, beta=beta, direction='up')

        return prob_ls

    def _free_energy_prep(self, x, beta):
        """Helper function to standardize free energy input """
        if isinstance(x, list):
            input_ls = x
        else:
            input_ls = self.propup(x, beta)
            
        return input_ls

    def free_energy(self, x, beta=1.0):
        """Calcualtes free energy of a DBM.
        
        # Arguments:
            x: tensor; visible inputs
            beta: float, optional; inverse temperature
        
        # Returns:
            fe: tensor; free energy of DBM
        """

        input_ls = self._free_energy_prep(x, beta)
        
        # For a RBM, the propup is actually exactly the true probabilities
        # For a DBM, the propup is only half the connections (up only) and
        # we need to perform mean field sampling to get true probabilities
        if not self.IS_rbm:
            input_ls, updates = self.meanfield_sampler.run_chain(input_ls, beta=beta, constant=[0])
            assert len(updates) == 0

        return self.free_energy_sumover_odd(input_ls, beta)

    def free_energy_sumover_odd(self, x, beta):
        """Calcualtes free energy of a DBM by analyically summing out odd layers.
        
        # Arguments:
            x: tensor; visible inputs
            beta: float, optional; inverse temperature
        
        # Returns:
            fe: tensor; free energy of DBM
        """

        input_ls = self._free_energy_prep(x, beta)
        
        return self._free_energy_sumover_parity(1, input_ls, beta)

    def free_energy_sumover_even(self, x, beta):
        """Calcualtes free energy of a DBM by analyically summing out even layers.
        
        # Arguments:
            x: tensor; visible inputs
            beta: float, optional; inverse temperature
        
        # Returns:
            fe: tensor; free energy of DBM
        """
        
        input_ls = self._free_energy_prep(x, beta)
        
        return self._free_energy_sumover_parity(0, input_ls, beta)

    def _free_energy_sumover_parity(self, parity, input_ls, beta):
        """Helper function that actually does the free energy analytical summation """
        
        beta = B.cast(beta, B.floatx())

        # Handle exactly summed out layers
        z_exact_ls = []
        for layer in self.layers[parity::2]:
            z_exact_ls.append(layer.input_z(input_ls))
        z = B.concatenate(z_exact_ls, axis=1)
        fe = -B.sum(B.log(1 + B.exp(beta*z)), axis=1)

        # Handle input dependent layers
        non_parity = int(not parity)
        for p, layer in zip(input_ls[non_parity::2], self.layers[non_parity::2]):
            fe -= beta*B.dot(p, layer.b)

        return fe

    def prob_even_given_odd(self, input_ls, beta=1.0, constant=[]):
        """Conditional probabilies of even layers given odd layers as input.
        
        # Arguments:
            input_ls: list; input activations of layers
            beta: float, optional; inverse temperature
            constant: list, optional; list of layers to clamp to input values
            
        # Returns:
            probabilities: list; probabilities of all layers
        
        """
        return self._parity_update(0, input_ls, beta, constant)

    def prob_odd_given_even(self, input_ls, beta=1.0, constant=[]):
        """Conditional probabilies of odd layers given even layers as input.
        
        # Arguments:
            input_ls: list; input activations of layers
            beta: float, optional; inverse temperature
            constant: list, optional; list of layers to clamp to input values
            
        # Returns:
            probabilities: list; probabilities of all layers
        
        """
        return self._parity_update(1, input_ls, beta, constant)

    def _parity_update(self, parity, input_ls, beta, constant):
        """Updates either even or odd layers """

        index = list(range(parity, len(input_ls), 2))
        index = [i for i in index if i not in constant]

        prob_ls = [None]*len(input_ls)

        for i in range(len(prob_ls)):
            if i in index:
                layer = self.layers[i]
                p = layer.output(input_ls, beta=beta)
                prob_ls[i] = p
            else:
                prob_ls[i] = input_ls[i]

        return prob_ls
