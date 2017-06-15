"""Samplers feed a neural network inputs and generates new output samples """

#%%
import warnings

import natural_bm.backend as B


#%%
class Sampler:
    """Generic class that creates samples from a neural network """

    def __init__(self, nnet, beta=1.0, constant=[]):
        """General sampler.
        
        # Arguments:
            nnet: DBM object
            beta: inverse temperature for sampling
            constant: list of layers with activity clamped to inputs
        
        """
        self.nnet = nnet
        self.set_param(beta=beta, constant=constant)

    def set_param(self, beta=1.0, constant=[]):
        """Sets parameters of a Sampler.
        
        These parameters may change over time and hence the need for setting
        parameters after intialization. For example, beta changes during AIS.
        """
        self.beta = beta
        self.constant = constant

    def probability(self, *args):
        """Calculates probabilies without intermediate sampling. """

        prob_ls = self.nnet.prob_odd_given_even(args, beta=self.beta,
                                                constant=self.constant)
        prob_ls = self.nnet.prob_even_given_odd(prob_ls, beta=self.beta,
                                                constant=self.constant)
        return prob_ls

    def _sample_parity(self, start, input_ls, constant):
        """Helper function to sample either even or odd layers """
        
        # convert inputs to samples
        index = list(range(start, len(input_ls), 2))
        index = [i for i in index if i not in constant]
        sample_ls = [None]*len(input_ls)

        for i in range(len(sample_ls)):
            p = input_ls[i]
            if i in index:
                sample_ls[i] = B.random_binomial(p.shape, p)
            else:
                sample_ls[i] = p

        return sample_ls

    def sample(self, *args):
        """Calculates samples with intermediate sampling. """

        # convert even inputs to samples
        output_ls = self._sample_parity(0, args, self.constant)

        # update odd probabilities
        output_ls = self.nnet.prob_odd_given_even(output_ls, beta=self.beta, constant=self.constant)

        # convert odd inputs to samples
        output_ls = self._sample_parity(1, output_ls, self.constant)

        # update even probabilities
        output_ls = self.nnet.prob_even_given_odd(output_ls, beta=self.beta, constant=self.constant)

        # convert even to samples
        output_ls = self._sample_parity(0, output_ls, self.constant)

        return output_ls


    def sample_inputs(self, *args):
        """Calculates probabilities with intermediate sampling. """

        prob_output_ls = [None]*len(args)

        # convert even inputs to samples
        output_ls = self._sample_parity(0, args, self.constant)

        # update odd probabilities
        output_ls = self.nnet.prob_odd_given_even(output_ls, beta=self.beta, constant=self.constant)

        # save odd probs for later output
        for i in range(1, len(prob_output_ls), 2):
            prob_output_ls[i] = output_ls[i]

        # convert odd inputs to samples
        output_ls = self._sample_parity(1, output_ls, self.constant)

        # update even probabilities
        output_ls = self.nnet.prob_even_given_odd(output_ls, beta=self.beta, constant=self.constant)

        # save even probs for later output
        for i in range(0, len(prob_output_ls), 2):
            prob_output_ls[i] = output_ls[i]

        return prob_output_ls

    def run_chain(self, input_ls, beta=1.0, constant=[]):
        """This is where the general logic of a sampler lives """
        raise NotImplementedError


#%%
class Meanfield(Sampler):
    """Meanfield calculations of probabilities """
    def __init__(self, nnet, max_steps=25, rtol=1e-5, atol=1e-6):
        """Performs meanfield (only probabilities, no samples) updates.
        
        # Arguments:
            nnet: DBM object
            max_steps: int, optional; maximum number of chain updates
            rtol: float, optional; relative tolerance for stopping chain updates
            atol: float, optional; absolute tolerance for stopping chain updates
        """
        super().__init__(nnet)

        if self.nnet.IS_rbm:
            max_steps = 1  # rbm only needs a single step
        self.max_steps = max_steps
        self.rtol = rtol
        self.atol = atol
        if self.atol < B.epsilon():
            # If atol < B.epsilon, will keep doing updates even if it converged
            warn = 'atol {:0.4} should be greater than B.epsilon() {:0.4}'.format(self.atol, B.epsilon())
            warnings.warn(warn)

    def _fn_chain_rbm(self):
        """RBM chains are just generic Sampler probability """
        return self.probability

    def _fn_chain_dbm(self):
        """DBM chains are generic Sampler probability and a convergence check """

        def prob_update(*input_prob):

            output_prob = self.probability(*input_prob)

            tensor_check = []
            for p_in, p_out in zip(input_prob, output_prob):
                tensor_check.append(B.allclose(p_in, p_out, self.rtol, self.atol))

            until = B.until(B.all(tensor_check))

            return output_prob, until

        return prob_update

    def run_chain(self, args, beta=1.0, constant=[]):
        """Generates chains of meanfield updates.
        
        # Arguments:
            args: input probabilities
            beta: inverse temperature for sampling
            constant: list of layers with activity clamped to inputs
        """

        self.set_param(beta, constant)

        if self.nnet.IS_rbm:
            fn = self._fn_chain_rbm()
        else:
            fn = self._fn_chain_dbm()

        scan_out, updates = B.scan(fn,
                                   outputs_info=args,
                                   n_steps=self.max_steps,
                                   name='scan_meanfield')

        prob_data = [s[-1] for s in scan_out]

        return prob_data, updates


#%%
class Gibbs(Sampler):

    def __init__(self, nnet, nb_gibbs=5):
        """Performs Gibbs sampling.
        
        This means intermediate and final activities are samples.
        
        # Arguments:
            nnet: DBM object
            nb_gibbs: int, optional; number of Gibb updates to do
        """
        super().__init__(nnet)

        self.nb_gibbs = nb_gibbs

    def run_chain(self, args, beta=1.0, constant=[]):
        """Generates chains of Gibbs sampling.
        
        # Arguments:
            args: input probabilities
            beta: inverse temperature for sampling
            constant: list of layers with activity clamped to inputs
        """

        self.set_param(beta, constant)

        scan_out, updates = B.scan(self.sample,
                                   outputs_info=args,
                                   n_steps=self.nb_gibbs,
                                   name='scan_gibbs')

        sample_data = [s[-1] for s in scan_out]

        return sample_data, updates


#%%
class GibbsProb(Sampler):

    def __init__(self, nnet, nb_gibbs=5):
        """Performs Gibbs probability sampling.
        
        This means intermediate and activities are samples but final output
        is a probability.
        
        # Arguments:
            nnet: DBM object
            nb_gibbs: int, optional; number of Gibb updates to do
        """
        super().__init__(nnet)

        self.nb_gibbs = nb_gibbs

    def run_chain(self, args, beta=1.0, constant=[]):
        """Generates chains of Gibbs sampling but final output is probability.
        
        # Arguments:
            args: input probabilities
            beta: inverse temperature for sampling
            constant: list of layers with activity clamped to inputs
        """

        self.set_param(beta, constant)

        scan_out, updates = B.scan(self.sample_inputs,
                                   outputs_info=args,
                                   n_steps=self.nb_gibbs,
                                   name='scan_gibbs_prob')

        prob_data = [s[-1] for s in scan_out]

        return prob_data, updates
