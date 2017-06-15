"""Components needed to assemble a neural network """

#%%
import natural_bm.backend as B
from natural_bm.utils import dict_2_OrderedDict


#%%
class NeuralNetParts:
    """Generic object for neural network components """

    def __init__(self, name):
        """Requries a name to initialize parts """
        self.name = name
        self._trainable_weights = []
        self._non_trainable_weights = []
        self._losses = []

    @property
    def weights(self):
        """All weights, with trainable weights before non trainable weights """
        return self._trainable_weights + self._non_trainable_weights

    @property
    def trainable_weights(self):
        """Weights that are optimized by gradients """
        return self._trainable_weights

    @trainable_weights.setter
    def trainable_weights(self, new_weights):
        """Setter for weights that are optimized by gradients.
        
        # Arguments:
            new_weights: tensor variable or list of tensor variables
        
        """
        if not isinstance(new_weights, list):
            new_weights = [new_weights]
        self._trainable_weights += new_weights

    @property
    def non_trainable_weights(self):
        """Weights that are not optimized by gradients """
        return self._non_trainable_weights

    @non_trainable_weights.setter
    def non_trainable_weights(self, new_weights):
        """Setter for weights that are not optimized by gradients.
        
        # Arguments:
            new_weights: tensor variable or list of tensor variables
        
        """
        if not isinstance(new_weights, list):
            new_weights = [new_weights]
        self._non_trainable_weights += new_weights
        
    @property
    def losses(self):
        """Losses that are specific to a given component, like a weight regularizer. """
        return self._losses
    
    @losses.setter
    def losses(self, new_losses):
        """Setter for losses that are specific to a given component, like a weight regularizer. 
         
        # Arguments:
            new_weights: tensor variable or list of tensor variables
        """
        if not isinstance(new_losses, list):
            new_losses = [new_losses]
        self._losses += new_losses


#%%
class Synapse(NeuralNetParts):
    """Connects together neural network layers with a weight """
    def __init__(self, name, init_W, regularizer=None):
        """Creates the synapse
        
        # Arguments:
            name: str
            init_W: numpy array, initial weights of synapse
            regularizer: Regularizer object
        """
        
        super().__init__(name)

        self.shape = init_W.shape
        self.W = B.variable(init_W, name=self.name+'_W')
        self.trainable_weights = self.W
        
        self.regularizer = regularizer        
        if regularizer is not None:
            self.losses = self.regularizer(self.W)


#%%
class Layer(NeuralNetParts):
    """Layer of neurons """
    def __init__(self,
                 name,
                 init_b,
                 up_dict=None,
                 down_dict=None,
                 activation=B.sigmoid,
                 regularizer=None):
        """Creates the layer of neurons.
        
        # Arguments:
            name: str
            init_b: numpy array, initial bias of neurons
            up_dict: dict, contains weights that connect lower numbered layers to this layer
            down_dict: dict, contains weights that connect higher numbered layers to this layer
            activation: function, internal neuron activity to final activity
            regularizer: Regularizer object
        """
        super().__init__(name)

        self.activation = activation

        self.dim = init_b.shape[0]
        self.b = B.variable(init_b, name=self.name+'_b')
        self.trainable_weights = self.b
        
        self.regularizer = regularizer
        if regularizer is not None:
            self.losses = self.regularizer(self.b)

        up_dict = dict_2_OrderedDict(up_dict)
        down_dict = dict_2_OrderedDict(down_dict)

        self.index_up = list(up_dict.keys())
        self.index_down = list(down_dict.keys())

        self.W_up = list(up_dict.values())
        W_down = list(down_dict.values())
        self.W_down = [W.T for W in W_down]

        # check that the weight shapes are sensible
        W_all = self.W_up + self.W_down
        for W in W_all:
            W_shape = B.eval(W).shape
            msg = 'Dimension mismatch. Expected {} but {} has shape {}'.format(self.dim, W.name, W_shape)
            assert W_shape[1] == self.dim, msg

        # assign layer properties
        n_up = len(up_dict)
        n_down = len(down_dict)
        if n_up > 0 and n_down > 0:
            self.direction = 'both'
            # need to compensate for reduced inputs when only going single direction
            # Reference: Deep boltzmann machines by R Salakhutdinov and G Hinton. AIS, 2009.
            self._z_up_adj = (n_up+n_down)/n_up
            self._z_down_adj = (n_up+n_down)/n_down
        elif n_up > 0 and n_down == 0:
            self.direction = 'up'
        elif n_up == 0 and n_down > 0:
            self.direction = 'down'
        else:
            raise ValueError('Both up_dict and down_dict cannot be empty.')


    def output(self, input_ls, beta=1.0, direction=None):
        """Activity of neurons
        
        # Arguments:
            input_ls: list; activities of other layers in neural network 
            beta: float; inverse temperature 
            direction: str, optional; controls types of inputs to layer
                
        # Returns:
            activity: final output activity of neurons
        """

        z = self.input_z(input_ls=input_ls, direction=direction)

        beta = B.cast(beta, B.floatx())

        activity = self.activation(beta*z)

        return activity


    def input_z(self, input_ls, direction=None):
        """Weighted inputs of neurons
        
        # Arguments:
            input_ls: list; activities of other layers in neural network 
            direction: str, optional; controls types of inputs to layer
                
        # Returns:
            z: weighted inputs of neurons
        """

        if direction is None:
            direction = self.direction
        else:
            direction = direction.lower().strip()

        if direction not in ['both', 'up', 'down']:
            raise ValueError('Not a valid direction keyword: {}'.format(direction))
        if (self.direction == 'up' and direction == 'down') or (self.direction == 'down' and direction == 'up'):
            raise ValueError('Layer has direction, {}, and hence cannot request direction: {}'.format(self.direction, direction))

        input_ls = self._prep_input(input_ls, direction)

        # collect needed weights
        W_ls = []
        if direction in ['both', 'up']:
            W_ls += self.W_up
        elif direction in ['both', 'down']:
            W_ls += self.W_down

        # Calculate activation input
        z = self.b
        for data, W in zip(input_ls, W_ls):
            z += B.dot(data, W)

        # compensates for reduced input
        if (self.direction == 'both') and (direction == 'up'):
            z *= self._z_up_adj
        elif (self.direction == 'both') and (direction == 'down'):
            z *= self._z_down_adj

        return z


    def _prep_input(self, full_input_ls, direction=None):
        """Helper function that selects only the needed inputs """

        index = []
        if direction in ['both', 'up']:
            index += self.index_up
        if direction in ['both', 'down']:
            index += self.index_down

        input_ls = [full_input_ls[i] for i in index]

        return input_ls
