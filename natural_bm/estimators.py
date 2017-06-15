"""Methods to evaluate partition functions. """

#%%
import numpy as np

import natural_bm.backend as B
from natural_bm.samplers import Sampler
from natural_bm.dbm import DBM


#%%
def _pylearn2_allocation(width):
    """
    Creates data for exactly calculating partition function.
    
    Original source: https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/rbm_tools.py
    
    # Arugments:
        width: int; dimension of data
        
    Returns:
        logZ_data: int; 2**width possible data configurations
    """
    
    try:
        logZ_data_c = np.zeros((2**width, width), order='C', dtype=B.floatx())
    except MemoryError:
        print('Ran out of memory when using width of {}'.format(width))

    # fill in the first block_bits, which will remain fixed for all configs
    tensor_10D_idx = np.ndindex(*([2] * width))
    for i, j in enumerate(tensor_10D_idx):
        logZ_data_c[i, -width:] = j
    try:
        logZ_data = np.array(logZ_data_c, order='F', dtype=B.floatx())
    except MemoryError:
        print('Ran out of memory when using width of {}'.format(width))

    return logZ_data


#%%
def exact_logZ(dbm):
    """
    Exactly calculate the partition function for a RBM.
    
    # Arguments:
        dbm: DBM object; must be a RBM.
        
    # Returns:
        logZ: float; log of the exact partition function
    """

    if len(dbm.layers) != 2:
        raise ValueError('Exact log partition assumes a RBM')

    n0 = dbm.layers[0].dim
    n1 = dbm.layers[1].dim
    b0 = dbm.layers[0].b
    b1 = dbm.layers[1].b
    W = dbm.synapses[0].W

    # Pick whether to iterate over visible or hidden states.
    if n0 < n1:
        width = n0
        b_in = b0
        b_z = b1
    else:
        width = n1
        b_in = b1
        b_z = b0
        W = W.T

    inputs = B.placeholder(shape=(width**2, width), name='input')

    b_logZ = B.dot(inputs, b_in)
    z = b_z + B.dot(inputs, W)

    logZ_data = _pylearn2_allocation(width)

    logZ_all = b_logZ + B.sum(B.log(1+B.exp(z)), axis=1)

    logZ_output = B.logsumexp(logZ_all)

    fn = B.function([inputs], logZ_output)

    logZ = fn(logZ_data)

    return logZ


#%%
class AIS:
    """    
    AIS
    
    In general need to do the following:
    
        1. s1, s2, ... sn (samples)
        2. w = p1(s1)/p0(s1) * p2(s2)/p1(s2) * ... * pn(sn)/pn-1(sn) where p is unnormalized
        3. Zb/Za ~ 1/M sum(w) = r_{AIS}
        4. Zb ~ Za*r_{AIS}
    
    This means that we need a model that has the following properties for AIS to work:
        1. easy to generate samples
        2. easy to estimate unnormalized probabilities
        3. easy to exactly calculate Za
    
    The Za model is called the data base rate (DBR) model. All weights and biases
    are zero except for the visible bias. The visibile bias is an estimate 
    based on the mean of the data but biased to guarantee p not equal to zero.
    
    I will sum over the even states since this simplifies the intermediate sampling.
    
    """
    def __init__(self, dbm, data, n_runs, n_betas=None, betas=None):
        """
        Initialize an object to perform AIS.
                
        # Arguments:
            dbm: DBM object
            data: numpy array, needed for data base rate model
            n_runs: int, number of parallel AIS estimates to run
            n_betas: int, optional. Will create evenly spaced betas. Need either n_betas or betas.
            betas: numpy array, optional. Betas for intermediate distributions. Need either n_betas or betas.
            
            
        # References:
            1. On the quantitative analysis of deep belief networks by R Salakhutdinov and I Murray. ACM 2008.
            2. Deep boltzmann machines by R Salakhutdinov and G Hinton. AIS, 2009.

        """
        
        self.dbm_b = dbm
        self.n_runs = n_runs

        if (n_betas is not None) and (betas is None):
            self.n_betas = n_betas
            betas = np.linspace(0, 1, n_betas)
        elif (n_betas is None) and (betas is not None):
            self.n_betas = betas.shape[0]
        else:
            raise ValueError('Need to provide at least one of n_betas or betas')

        self.betas = B.variable(betas, name='betas')

        # this is the data base rate model of reference 1
        # The ais estimate is very sensitive to this base rate model, so the 
        # standard practice in the literature is to use this setup
        vis_mean = np.clip(np.mean(data, axis=0), B.epsilon(), 1-B.epsilon())
        p_ruslan = (vis_mean + 0.05)/1.05
        b0_a = np.log(p_ruslan/(1-p_ruslan)).astype(B.floatx())
        self.dbm_a = DBM(layer_size_list=self.dbm_b.layer_size_list,
                         topology_dict=self.dbm_b.topology_dict)
        B.set_value(self.dbm_a.layers[0].b, b0_a)

        # make the initial sample
        # visible layer depends on data base rate bias
        p0 = np.tile(1. / (1 + np.exp(-b0_a)), (n_runs, 1))
        s0 = np.array(p0 > np.random.random_sample(p0.shape), dtype=B.floatx())
        sample_ls = [s0]
        # rest of layers are uniform sample
        for n in self.dbm_b.layer_size_list[1:]:
            s = B.random_binomial((self.n_runs, n), p=0.5)
            sample_ls.append(B.variable(B.eval(s)))
        self.init_sample_ls = sample_ls

        # this is the exact partition function of the base rate model
        self.logZa = np.sum(self.dbm_b.layer_size_list[1:])*np.log(2)
        self.logZa += np.sum(np.log(1 + np.exp(b0_a)))

        # This is the sampler for the final model
        self.dbm_b_sampler = Sampler(self.dbm_b)

    def _update(self, *args):
        """AIS update called by scan function """

        log_ais_w, index, *sample_ls = args
        beta = self.betas[index]

        # use input sample to estimate log_w denominator
        log_ais_w -= self.dbm_b.free_energy_sumover_even(sample_ls, beta)

        # generate a new sample
        self.dbm_b_sampler.set_param(beta=beta)
        sample_ls = self.dbm_b_sampler.sample(*sample_ls)

        # use new sample to estimate log_w numerator
        log_ais_w += self.dbm_b.free_energy_sumover_even(sample_ls, beta)

        index += 1

        return [log_ais_w, index] + sample_ls

    def run_logZ(self):
        """Performs calculatations of AIS runs.
        
        Must be called before estimates.
        """
        
        # initial sample
        sample_ls = self.init_sample_ls

        # this is the inital beta=0 case
        log_ais_w = B.eval(self.dbm_a.free_energy_sumover_even(sample_ls, 1.0))

        log_ais_w = B.variable(log_ais_w, name='log_ais_w')
        index = B.variable(1, name='index', dtype=B.intx())

        scan_out, updates = B.scan(self._update,
                                   outputs_info=[log_ais_w, index]+sample_ls,
                                   n_steps=self.n_betas - 2,
                                   name='scan_ais')

        log_ais_w = scan_out[0][-1]
        sample_ls = [s[-1] for s in scan_out[2:]]

        # this is the final beta=1 case
        log_ais_w -= self.dbm_b.free_energy_sumover_even(sample_ls, 1.0)

        logZ_fn = B.function([], [log_ais_w], updates=updates)

        self.logZ = self.logZa + logZ_fn()


    def estimate_log_error_Z(self):
        """Error bars on estimate of partition function.
        
        The output is the mean and +- 3 standard deviations of the true
        (ie not logged) partition function. This is why the standard deviations
        are not symmetric about the mean.
        
        Returns:
            * mean logZ: float
            * -3 std logZ: float
            * +3 std logZ: float
        """

        if not hasattr(self, 'logZ'):
            raise RuntimeError('You must run_logZ before calculating the final estimates.') 

        logZ = B.placeholder(shape=self.logZ.shape)

        # this is the mean factor from the weights
        logZ_mean = B.logsumexp(logZ) - B.log(self.n_runs)

        # this is the standard deviation
        m = B.max(logZ)
        logstd_AIS = B.log(B.std(B.exp(logZ - m))) + m - B.log(self.n_runs)/2.0

        # find +- 3 std
        l_input = B.stack([B.log(3)+logstd_AIS, logZ_mean])
        logZ_high = B.logsumexp(l_input)
        logZ_low = B.logdiffexp(l_input)

        # actually calculate the estimates
        logZ_est_fn = B.function([logZ], [logZ_mean, logZ_low, logZ_high])
        logZ_out, logZ_low_out, logZ_high_out = logZ_est_fn(self.logZ)
        
        # convert to floats
        logZ_out = logZ_out.item()
        logZ_low_out = logZ_low_out.item()
        logZ_high_out = logZ_high_out.item()

        # fix any nans
        if np.isnan(logZ_low_out):
            logZ_low_out = 0.0

        return logZ_out, logZ_low_out, logZ_high_out
