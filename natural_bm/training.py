"""Generates statistics needed for training DBMs """

#%%
from collections import OrderedDict

import natural_bm.backend as B
from natural_bm.utils import merge_OrderedDicts
from natural_bm import samplers


#%%
class Trainer:

    def __init__(self, nnet, nb_pos_steps=25, nb_neg_steps=5):
        """General Trainer.
        
        # Arguments:
            nnet: DBM object
            nb_pos_steps: int, optional. Number of updates in the positive, or data phase.
            nb_neg_steps: int, optional. Number of updates in the negative, or model phase.
        """
        self.nnet = nnet
        self.nb_pos_steps = nb_pos_steps
        self.nb_neg_steps = nb_neg_steps

        self._updates = OrderedDict()

    @property
    def updates(self):
        """Non-gradient updates needed for training a neural network """
        return self._updates

    @updates.setter
    def updates(self, updates):
        """Setter for non-gradient updates needed for training a neural network
        
        # Arguments:
            updates: OrderedDict of updates
        """
        self._updates = merge_OrderedDicts(self._updates, updates)

    def pos_stats(self, x):
        """Logic that generates data (positive stage) driven statistics """
        raise NotImplementedError

    def neg_stats(self, prob_data):
        """Logic that generates model (negative stage) driven statistics """
        raise NotImplementedError

    def loss_fn(self):
        """
        Result is a Theano expression with the form loss = f(x).
        """
        raise NotImplementedError


#%%
class CD(Trainer):
    """Implements Contrastive Divergence (CD) training.
    
    The proper training should run the positive and negative stages to
    convergence. However, it was realized that just a few updates of the stages
    actually led to good training. Therefore CD is just a few steps of the updates
    and stops before the neural network converges to the true equilibrium
    distribution.
    
    This is often shown in the literature as CD-k where k is the number of
    negative steps. The positive stage is still run to convergence since this
    is exact in one step for an RBM and usually converges fast for a DBM.
    """

    def __init__(self, nnet, nb_pos_steps=25, nb_neg_steps=5):
        """CD Trainer.
        
        # Arguments:
            nnet: DBM object
            nb_pos_steps: int, optional. Number of updates in the positive, or data phase.
            nb_neg_steps: int, optional. Number of updates in the negative, or model phase.
        """
        
        super().__init__(nnet, nb_pos_steps, nb_neg_steps)

        self.pos_sampler = samplers.Meanfield(nnet, nb_pos_steps)
        self.neg_sampler = samplers.GibbsProb(nnet, nb_neg_steps)

        self.nb_pos_steps = self.pos_sampler.max_steps

    def pos_stats(self, x):
        """Generates positive (data dependent) statistics.
        
        # Arguments:
            x: tensor; input data
            
        # Returns:
            prob_data: list; data dependent probabilities
        """

        prob_data = self.nnet.propup(x)

        prob_data, updates = self.pos_sampler.run_chain(prob_data, constant=[0])

        self.updates = updates

        return prob_data

    def neg_stats(self, prob_data):
        """Generates negative (model dependent) statistics.
        
        # Arguments:
            prob_data: list of tensors; input data from pos_stats
            
        # Returns:
            prob_model: list; model dependent probabilities
        """

        prob_model, updates = self.neg_sampler.run_chain(prob_data)

        self.updates = updates
        
        # Recommended to always use probs for update statistics
        # Reference: A Practical Guide to Training Restricted Boltzmann Machines by Geoffrey Hinton
        return prob_model

    def loss_fn(self):
        """
        Compute contrastive divergence loss with k steps of Gibbs sampling (CD-k).

        Result is a Theano expression with the form loss = f(x).
        """
        def cd(x):
            prob_data = self.pos_stats(x)
            prob_model = self.neg_stats(prob_data)

            # do not take gradients through the markov chain
            # Reference: http://deeplearning.net/tutorial/rbm.html
            prob_data = [prob_data[0]]+[B.stop_gradient(p) for p in prob_data[1:]]
            prob_model = [B.stop_gradient(p) for p in prob_model]

            pos_mean = B.mean(self.nnet.free_energy(prob_data))
            neg_mean = B.mean(self.nnet.free_energy(prob_model))

            cd = pos_mean - neg_mean

            return cd

        return cd


#%%
class PCD(CD):
    """Implements Persistent Contrastive Divergence (PCD) training.
    
    This is very similar to CD training. The only difference is in the
    initialization of the negative (model) statistics stage. In CD, the 
    positive statistics initialize the model. In PCD, the probabilites from
    the last time the negative statistics were generated is used as the new
    start. This works because weight changes are relatively small and hence
    the last chain from the the old model is close to a good chain for the 
    new model.
    """

    def __init__(self,
                 nnet,
                 nb_pos_steps=25,
                 nb_neg_steps=5,
                 init_chain=None,
                 batch_size=None):
        """PCD Trainer.
        
        Need to provide either the init_chain or the batch_size.
        
        # Arguments:
            nnet: DBM object
            nb_pos_steps: int, optional. Number of updates in the positive, or data phase.
            nb_neg_steps: int, optional. Number of updates in the negative, or model phase.
            init_chain: list of tensors, optional. Initial starting point of model persistent chains.
            batch_size: int, optional. Need batch size to generate appropriate number of persistent chains.
        """
        super().__init__(nnet, nb_pos_steps, nb_neg_steps)

        assert (init_chain is not None) or (batch_size is not None)
        if init_chain is None:
            init_chain = []
            for size in self.nnet.layer_size_list:
                sample = B.eval(B.random_binomial(shape=(batch_size, size), p=0.5))
                init_chain.append(B.variable(sample))
        else:
            init_chain = [B.variable(ic) for ic in init_chain]

            batch_size = B.eval(init_chain[0].shape)[0]
            for ic, size in zip(init_chain, self.nnet.layer_size_list):
                assert B.eval(ic.shape)[0][0] == batch_size
                assert B.eval(ic.shape)[0][1] == size

        self.persist_chain = init_chain


    def neg_stats(self, prob_data=None):
        """Generates negative (model dependent) statistics.
        
        # Arguments:
            prob_data: list of tensors; completely ignored, just here so that same function signature as CD
            
        # Returns:
            prob_model: list; model dependent probabilities
        """

        prob_model = self.persist_chain

        prob_model, updates = self.neg_sampler.run_chain(prob_model)

        self.updates = updates

        # update persist chain
        persist_updates = OrderedDict()
        for per, p in zip(self.persist_chain, prob_model):
            persist_updates[per] = p
        self.updates = persist_updates

        # Recommended to always use probs for update statistics
        # Reference: A Practical Guide to Training Restricted Boltzmann Machines by Geoffrey Hinton
        return prob_model
