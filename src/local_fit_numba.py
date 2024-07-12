#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 22:17:11 2022

@author: hill103

this script implement the optimization of theta and sigma^2 using Poisson log-normal distribution + heavy-tail

use Numba parallel + nested Numba cache dict (finally NOT used) for best performance

we calculate the likelihood for each gene sequentially, utilizing Numba's parallel capabilities to accelerate the computation within each gene's calculation.

we also cached the calculated likelihoods instead of repreated calculation, which depends on the combination of one nUMI, sigma^2, and mu value.

but we failed implement the caching, we tried rounding MU, rounding w, even directly copying MU values as dict keys, using uneven initial w, and we checked the cached likelihoods which match the calculated values, but the optimization only yields solution with all elements being the same value. And the reason is still unknown. One guess is caching affects the searching of direction to take the next step in optimization, even we provide the gradient of w.

NOTE Numba assumes global variables are constants, and does not allow their values to be changed within the JIT-compiled function. In order to modify the cache dict inpalce, we create a wrapper function outside the Numba function, then assign the global cache dict to a local variable when calling Numba function

and print not supported in Numba function

functions for cache Numba dict:
    
    initialization:
        
        global Numba dicts: empty_mu_dict, empty_y_dict, likelihood_dict
        
        global list: unique_nUMI_values (first empty then updated by update_unique_nUMI function)
        
    
    save and retrieve value in likelihood_dict:
        
        likelihood_dict indexed by sigma^2 (rounded) -> nUMI value (y; int) -> mu (rounded)
        
        Numba functions: retrieve_value, insert_value
        
    
    manage dict sigma^2 keys:
    
        insert_key_sigma2_wrapper -> insert_key_sigma2 (Numba function)
        
        purge_keys


functions for likelihood calculation:
    
    calc_hv_numba: Numba function so parallel computing by Numba; calculate likelihoods given an array of y and mu then return; NOT related with cache dict
    
    hv_comb: return sum of negative log-likelihoods + gradient of w; used for theta optimization; directly call hv_numba
    
    hv_wrapper: return sum of negative log-likelihoods; used for sigma^2 optimization; directly call hv_numba
    
    hv_numba: call calc_hv_numba and return array of likelihoods; support caching
    
    
sequence of function calls for parameter optimization:
    
    update_theta -> objective_loss_theta -> hv_comb -> hv_numba
    
    update_sigma2 -> objective_loss_sigma2 -> hv_wrapper -> hv_numba
    
    calcHVBaseModelLoss -> hv_wrapper in hyper parameter tunning
"""



import numpy as np
from scipy.optimize import minimize, basinhopping
import scipy.stats
import numba as na
import math
from config import min_val, min_theta, min_sigma2, N_z, gamma, print, mu_digits, sigma2_digits, theta_eps, sigma2_eps



################################# code related to global optimization #################################
class RandomDisplacementBounds(object):
    """A class to perform random displacement with bounds on parameters during optimization.

    This step function is designed to modify the basinhopping algorithm's step-taking behavior by 
    ensuring that new positions remain within specified bounds. This implementation uses a direct 
    calculation to determine the minimum and maximum allowable steps rather than using
    acceptance-rejection sampling, which can be found in the more general approach discussed on 
    StackOverflow (https://stackoverflow.com/a/21967888/2320035). The approach has been modified 
    for enhanced performance with specific bounds.
    
    ref: https://stackoverflow.com/questions/47055970/how-can-i-write-bounds-of-parameters-by-using-basinhopping

    Attributes:
        xmin (np.ndarray or list): The lower bounds of the parameters. None indicates unbounded.
        xmax (np.ndarray or list): The upper bounds of the parameters. None indicates unbounded.
        stepsize (float): The maximum step size to take in any parameter direction.

    """
    def __init__(self, xmin, xmax, stepsize=0.5):
        """
        Initializes the RandomDisplacementBounds with specified bounds and step size.

        Args:
            xmin (np.ndarray or list): The lower bounds of the parameters.
            xmax (np.ndarray or list): The upper bounds of the parameters.
            stepsize (float, optional): The maximum step size to take in any parameter direction. Defaults to 0.5.
        """
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """
        Calculates a new position for the optimization algorithm by taking a random step
        within the defined bounds.

        This method ensures that the new position does not violate the specified bounds.

        Args:
            x (np.ndarray): The current position of the parameters in the optimization algorithm.

        Returns:
            np.ndarray: The new position after taking a random step within the bounds.
        """
        # define a custom function to consider None in bound
        def calcMinStep(xmin, x, stepsize):
            if xmin is None:
                return -stepsize
            else:
                return np.maximum(xmin - x, -stepsize)
            
        def calcMaxStep(xmax, x, stepsize):
            if xmax is None:
                return stepsize
            else:
                return np.minimum(xmax - x, stepsize)
        
        min_step = np.array([calcMinStep(this_xmin, this_x, self.stepsize) for (this_xmin, this_x) in zip(self.xmin, x)])
        max_step = np.array([calcMaxStep(this_xmax, this_x, self.stepsize) for (this_xmax, this_x) in zip(self.xmax, x)])
        
        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        xnew = x + random_step
        
        return xnew



################################# code related to cache calculated Poisson heavy-tail density values ###############################

# start with a empty nested Numba dict
empty_mu_dict = na.typed.Dict.empty(key_type=na.types.float64, value_type=na.types.float64)
empty_y_dict = na.typed.Dict.empty(key_type=na.types.int64, value_type=na.typeof(empty_mu_dict))
likelihood_dict = na.typed.Dict.empty(key_type=na.types.float64, value_type=na.typeof(empty_y_dict))


unique_nUMI_values = []


def update_unique_nUMI(Y):
    '''
    update global variable unique_nUMI_values based on the observed gene raw nUMI count in all spots across all included genes
    
    unique_nUMI_values is an array of integers, and is fixed once the spots and genes in regression are determined
    
    UPDATE: we add so add Y+1 into dict since calculation of gradient of w needs it

    Parameters
    ----------
    Y : 2-D numpy matrix
        spatial gene expression (spots * genes).

    Returns
    -------
    None.

    '''
    global unique_nUMI_values
    
    tmp_values = set(Y.flatten())
    print(f'total {len(tmp_values)} unique nUMIs, min: {min(tmp_values)}, max: {max(tmp_values)}')
    
    unique_values = sorted(set(np.concatenate((Y.flatten(), Y.flatten()+1))))
    #print(f'total {len(unique_values)} unique nUMIs for caching, min: {min(unique_values)}, max: {max(unique_values)}')
    unique_nUMI_values = np.array(unique_values, dtype=int)


@na.jit(nopython=True, parallel=False, fastmath=False, cache=False)
def insert_key_sigma2(local_dict, sigma2, unique_nUMI_values, empty_y_dict, empty_mu_dict):
    '''
    insert a new key of sigma2 to the dict of cached calculated likelihood values
    
    also pre-set the keys for the nested dict
    
    sigma2 has already been rounded to 2 digits
    
    directly modify the local dict inplace

    Parameters
    ----------
    local_dict : Numba dict
        cached dict containing already calculated likelihood values
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
    unique_nUMI_values : 1-D numpy array
        observed gene raw nUMI count values
    empty_y_dict : Numba dict
        a empty Numba dict for nested dict initialization
    empty_mu_dict : Numba dict
        a empty Numba dict for nested dict initialization
    
    Returns
    -------
    None.
    '''
    
    if sigma2 in local_dict:
        #print(f'sigma2 {sigma2} already in cache dict!')
        return
    
    #print(f'generate new key {sigma2} for cache dict')
    
    local_dict[sigma2] = empty_y_dict
    
    for k in unique_nUMI_values:
        local_dict[sigma2][k] = empty_mu_dict
    


def insert_key_sigma2_wrapper(sigma2):
    '''
    a wrapper to call jit function to insert new keys into cache dict
    
    sigma2 has already been rounded to X digits
    
    Parameters
    ----------
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
        
    Returns
    -------
    None.
    '''
    global likelihood_dict, unique_nUMI_values, empty_y_dict, empty_mu_dict
    insert_key_sigma2(likelihood_dict, sigma2, unique_nUMI_values, empty_y_dict, empty_mu_dict)


def purge_keys(keep_sigma2):
    '''
    after sigma2 optimization, delete other sigma2 keys
    
    sigma2 has already been rounded to X digits

    Parameters
    ----------
    local_dict : Numba dict
        cached dict containing already calculated likelihood values
    keep_sigma2 : float
        optimized sigma2 for current dataset.

    Returns
    -------
    None.

    '''
    global likelihood_dict
    
    print(f'total {len(likelihood_dict)} sigma2 keys, only keep key {keep_sigma2} and delete others')
    
    for k in list(likelihood_dict.keys()):
        if not k == keep_sigma2:
            del likelihood_dict[k]


################################# code related to Poisson heavy-tail density calculation in Python #################################

# parameters related to heavy tail, just for standard N(0, 1)
# a and c are chosen so that density is continuously differentiable at the boundary (x=3)
a = 4/9 * np.exp(-3**2/2) / np.sqrt(2*np.pi)
c = 7/3
# C is a normalizing constant making the density integrate to 1
C = 1 / ((a/(3-c) - scipy.stats.norm(0, 1).cdf(-3))*2 + 1)

# generate data points to calculate integral
z_hv = np.array(range(-N_z, N_z+1)) * gamma


def generate_log_heavytail_array(x, sigma):
    '''
    generate a array of log density values of standard normal distribution + heavy-tail given a specific sigma
    
    we assume the normal distribution is N(0, sigma^2), and transform to N(0, 1) by divided by sigma, and calculate the heavy-tail density values, then transform back to N(0, sigma^2) and take log. We can re-use these pre-calculated values for all genes and spots
    
    after divided by sigma, the normal + heavy-tail integrate to 1, i.e. sum(p*gamma)=1, gamma is the interval

    Parameters
    ----------
    x : 1-D numpy array
        data points served as x for calculation of probability density values
    sigma : float
        SD of normal distribution, corresponding to the dispertion in GLRM. All genes and all spots share the same SD

    Returns
    -------
    1-D numpy array
        log density values of normal distribution N(0, sigma^2) + heavy-tail.

    '''
    
    if not math.isfinite(sigma):
        raise Exception('Error: sigma2 optimization in local model of all spots returns a non-finite value!')
    
    # change to N(0, 1)
    tmp_x = x / sigma
    p = np.empty(tmp_x.shape)
    tmp_idx = abs(tmp_x) < 3
    not_idx = np.logical_not(tmp_idx)
    # normal part
    p[tmp_idx] = C/np.sqrt(2*np.pi) * np.exp(-0.5*(tmp_x[tmp_idx]**2))
    # heavy tail part
    p[not_idx] = C*a / ((abs(tmp_x[not_idx])-c) ** 2)
    
    assert((p>=0).all())
    
    # change back to N(0, sigma^2) and take log
    return np.log(p / sigma + min_val)



############ code related to Numba functions to calculate negative log-likelihood values  ############

# use Numba to speed up Python and NumPy code
# check in Reference Manual to make sure all Python and Numpy features in the code are supported
# based on test, Numba parallel on one gene DO NOT improve speed much
# see the ref https://stackoverflow.com/a/64613366/13752320 for advices of using Numba efficiently
@na.jit(nopython=False, parallel=True, fastmath=True, error_model="numpy", cache=False)
def calc_hv_numba(MU, y_vec, hv_x, hv_log_p, output, gamma):
    '''
    calculate the likelihood value given the distribution parameters considering heavy-tail and assign them to the output variable
    
    note that add a small value to avoid log(0)
    
    Parameters
    ----------
    MU : 1-D numpy array
        mean value of log-normal distribution for all genes.
    y_vec : 1-D numpy array
        observed gene counts of one spot.
    hv_x : 1-D numpy array, optional
        data points served as x for calculation of probability density values. Only used for heavy-tail.
    hv_log_p : 1-D numpy array, optional
        log density values of normal distribution N(0, sigma^2) + heavy-tail. Only used for heavy-tail.
    output : 1-D numpy array
        empty array used to store the calculated values.
    gamma : float
        increment to calculate the heavy-tail probabilities.
    
    Returns
    -------
    1-D numpy array
        likelihood of input genes in one spot.
    '''
    
    for i in na.prange(MU.shape[0]):
        for j in na.prange(hv_x.shape[0]):
            tmp = MU[i] + hv_x[j]
            output[i] += math.exp(tmp*y_vec[i] + hv_log_p[j] - math.exp(tmp) - math.lgamma(y_vec[i]+1))
    return output * gamma



@na.jit(nopython=True, parallel=False, fastmath=False, cache=False)
def retrieve_value(local_dict, sigma2, y_vec, MU, output):
    '''
    retrieve values from cache dict
    
    sigma2 and MU are already rounded
    
    note that the returned array has a numpy type 'object' since it may contains None
    
    Update: since Numba not support object array, use numpy.nan to replace None, then the returned array is a float array

    Parameters
    ----------
    local_dict : Numba dict
        cached dict containing already calculated likelihood values
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
    y_vec : 1-D numpy array
        observed gene counts of one spot.
    MU : 1-D numpy array
        mean value of log-normal distribution for all genes.
    output : 1-D numpy array
        an empty array used to store the returned values.

    Returns
    -------
    1-D numpy array
        pre-calculated likelihood values or np.nan
    '''
    
    for i in na.prange(y_vec.shape[0]):
        output[i] = local_dict[sigma2][y_vec[i]].get(MU[i], np.nan)
    return output



@na.jit(nopython=True, parallel=False, fastmath=False, cache=False)
def insert_value(local_dict, sigma2, y_vec, MU, values):
    '''
    insert calculated values into cache dict
    
    sigma2 and MU are already rounded

    Parameters
    ----------
    local_dict : Numba dict
        cached dict containing already calculated likelihood values
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
    y_vec : 1-D numpy array
        observed gene counts of one spot.
    MU : 1-D numpy array
        mean value of log-normal distribution for all genes.
    values : 1-D numpy array
        just calculated likelihoods of input genes in one spot.

    Returns
    -------
    None.
    '''
    
    for i in na.prange(y_vec.shape[0]):
        local_dict[sigma2][y_vec[i]][MU[i]] = values[i]



def hv_comb(w_vec, y_vec, mu, gamma_g, sigma2, hv_x, hv_log_p, N, use_cache):
    '''
    a function to calculate the negative likelihood value given the distribution parameters considering heavy-tail plus corresponding gradient vector
    
    also return 1st order derivative vector of w
    
    this function will be used as target function for theta optimization
    
    we assume:
        
        y ~ Poisson(N*lambda)

        ln(lambda) follow epsilon's distribution ~ N(mu, sigma2)
        
    then ln(N*lambda) ~ N(mu+ln(N), sigma2)
    
    N is sequencing depth for this spot
    mu is alpha + log(theta*marker gene) + gamma
    
    Now we also consider the heavy-tail instead of just normal distribution
    
    Parameters
    ----------
    w_vec : 1-D numpy array
        e_alpha (spot-specific effect) * theta (celltype proportion) of one spot.
    y_vec : 1-D numpy array
        observed gene counts of one spot.
    mu : 2-D numpy matrix
        celltype specific marker genes (celltypes * genes)
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
    hv_x : 1-D numpy array, optional
        data points served as x for calculation of probability density values. Only used for heavy-tail.
    hv_log_p : 1-D numpy array, optional
        log density values of normal distribution N(0, sigma^2) + heavy-tail. Only used for heavy-tail.
    N : float
        sequencing depth for this spot. If is None, use sum(y_vec) instead.
    use_cache : bool, optional
        if True, use the cached dict of calculated likelihood values.

    Returns
    -------
    tuple of (float, 1-D numpy array)
        sum of negative log-likelihood across all genes in one spot + gradient vector of w.
    '''
    # return both negative log-likelihoods and 1st order derivative
    # The @ operator performs matrix multiplication, w_vec@mu produces a 1-D numpy array with length as number of genes
    MU = np.log(w_vec@mu+min_val) + gamma_g + np.log(N)
    this_lambda = (w_vec@mu) * np.exp(gamma_g) * N
    
    likelihoods = hv_numba(MU, y_vec, sigma2, hv_x, hv_log_p, use_cache)
    
    loss = -np.sum(np.log(likelihoods + min_val))
    
    # for gradient of w
    likelihoods_y_add_1 = hv_numba(MU, y_vec+1, sigma2, hv_x, hv_log_p, use_cache)
    
    # element-wise operation
    tmp = (y_vec*likelihoods - (y_vec+1)*likelihoods_y_add_1 + min_val) / (this_lambda*likelihoods + min_val)
    der = []
    # sum over genes, get result related to cell-types
    for i in range(mu.shape[0]):
        der.append(-np.sum(tmp * np.exp(gamma_g) * N * mu[i, :].flatten()))
    
    return loss, np.array(der)



# change it to a normal non-parallel function for optimization
def hv_numba(MU, y_vec, sigma2, hv_x, hv_log_p, use_cache):
    '''
    calculate the likelihood value given the distribution parameters considering heavy-tail
    
    MU need to be rounded here, and sigma2 has already been rounded if use_cache is true
    
    if use_cache is true, update the cache dict inplace at the same time
    
    NOTE since we already define the nUMI in cache dict is int64 type, we need to manually cast the array to int64 before passing to the function, otherwise the Numba function will raise type error
    
    Parameters
    ----------
    MU : 1-D numpy array
        mean value of log-normal distribution for all genes.
    y_vec : 1-D numpy array
        observed gene counts of one spot.
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
    hv_x : 1-D numpy array, optional
        data points served as x for calculation of probability density values. Only used for heavy-tail.
    hv_log_p : 1-D numpy array, optional
        log density values of normal distribution N(0, sigma^2) + heavy-tail. Only used for heavy-tail.
    use_cache : bool, optional
        if True, use the cached dict of calculated likelihood values.

    Returns
    -------
    1-D numpy array
        likelihoods across all genes in one spot.
    '''
    
    global likelihood_dict, gamma
    
    if use_cache:
        
        MU_rounded = np.round(MU, mu_digits)
        
        # first retrieve dict to see how many entries need to be calculate
        # note that it's a numpy array with object type, as it contains None. Update: use numpy.nan instead
        retri_values = retrieve_value(likelihood_dict, sigma2, y_vec.astype(np.int64), MU_rounded, np.empty(MU_rounded.shape))
        
        # numpy.nan in the array means this entry need to be calculated
        n_calc = np.sum(np.isnan(retri_values))  # sum over bool array
        #print(f'total {retri_values.shape[0]} entries, {n_calc} need to be calculated')
        
        if n_calc == 0:
            # all needed likelihoods can be retrieved, no new likelihood need to be calculated
            return retri_values
        
        else:
            # calculate the values for those entries
            # dimension will be kept by array slicing
            # for safe, manually select them since we need to put the newly calculated likelihoods back to the corresponding position
            sub_MU = []
            sub_y_vec = []
            sub_index = []
            
            for i in range(retri_values.shape[0]):
                if np.isnan(retri_values[i]):
                    sub_MU.append(MU_rounded[i])
                    sub_y_vec.append(y_vec[i])
                    sub_index.append(i)
            
            sub_MU = np.array(sub_MU)
            sub_y_vec = np.array(sub_y_vec)
        
            calc_values = calc_hv_numba(sub_MU, sub_y_vec, hv_x, hv_log_p, np.zeros((n_calc,)), gamma)
        
            # update cache dict
            insert_value(likelihood_dict, sigma2, sub_y_vec.astype(np.int64), sub_MU, calc_values)
            
            # put calculated values back to the array
            for i, one_index in enumerate(sub_index):
                assert np.isnan(retri_values[one_index])
                retri_values[one_index] = calc_values[i]
            
            return retri_values
        
    else:
        # do not use cache
        return calc_hv_numba(MU, y_vec, hv_x, hv_log_p, np.zeros(MU.shape), gamma)
    


def hv_wrapper(w_vec, y_vec, mu, gamma_g, sigma2, hv_x, hv_log_p, N, use_cache):
    '''
    a wrapper to calculate the negative log-likelihood value given the distribution parameters considering heavy-tail
    
    this wrapper will be used as target function for sigma^2 optimization
    
    we assume:
        
        y ~ Poisson(N*lambda)
    
        ln(lambda) follow epsilon's distribution ~ N(mu, sigma2)
    
    then ln(N*lambda) ~ N(mu+ln(N), sigma2)
    
    N is sequencing depth for this spot
    mu is alpha + log(theta*marker gene) + gamma
    
    Now we also consider the heavy-tail instead of just normal distribution
    
    Parameters
    ----------
    w_vec : 1-D numpy array
        e_alpha (spot-specific effect) * theta (celltype proportion) of one spot.
    y_vec : 1-D numpy array
        observed gene counts of one spot.
    mu : 2-D numpy matrix
        celltype specific marker genes (celltypes * genes)
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
    hv_x : 1-D numpy array, optional
        data points served as x for calculation of probability density values. Only used for heavy-tail.
    hv_log_p : 1-D numpy array, optional
        log density values of normal distribution N(0, sigma^2) + heavy-tail. Only used for heavy-tail.
    N : float
        sequencing depth for this spot. If is None, use sum(y_vec) instead.
    use_cache : bool, optional
        if True, use the cached dict of calculated likelihood values.

    Returns
    -------
    float
        sum of negative log-likelihood across all genes in one spot.
    '''
    
    MU = np.log(w_vec@mu+min_val) + gamma_g + np.log(N)
    likelihoods = hv_numba(MU, y_vec, sigma2, hv_x, hv_log_p, use_cache)
    loss = -np.sum(np.log(likelihoods + min_val))
    
    return loss



################################# code related to update theta #################################

def objective_loss_theta(w_vec, y_vec, mu, gamma_g, sigma2, nu_vec, rho, lambda_r, lasso_weight_vec, lambda_l2, hybrid_version, hv_x, hv_log_p, N, use_cache):
    '''
    calculate loss function for updating theta (celltype proportion)
    
    the loss function contains four parts (defined for each spot separately)
    
    also return gradient of w
    
    1. negative log-likelihood of the base model given observed data and initial parameter value. It sums across all genes
    
    2. a loss of ADMM to make theta equals theta_hat (used for regularization/penalty; optional, controlled by nu_vec and rho)
                    
        1/(2*rho) (||w/sum(w) - theta_hat + u||_2)^2
    
    u is the scaled dual variables to make theta = theta_hat and nu = theta_hat - u
    
    and we did re-parametrization w = e^alpha * theta, so theta = w / sum(w)
    
    3. a loss of Adaptive Lasso to shrink theta (optional, controlled by lambda_r and lasso_weight_vec)
    
        lambda_r * (inner product(w/sum(w), lasso_weight))
                        
    4. a loss of L2 penalty to shrink theta (optional, controlled by lambda_l2)
    
        lambda_l2 * (sum(squared(w/sum(w))))
    
    Parameters
    ----------
    w_vec : 1-D numpy array
        e_alpha (spot-specific effect) * theta (celltype proportion) of one spot.
    y_vec : 1-D numpy array
        observed gene counts of one spot.
    mu : 2-D numpy matrix
        celltype specific marker genes (celltypes * genes)
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
    nu_vec : 1-D numpy array
        variable for ADMM penalty of one spot
        in 3 part ADMM nu = theta_hat (used for regularization/penalty) - u (scaled dual variables to make theta = theta_hat)
    rho : float
        parameter for the strength of ADMM loss to make theta equals theta_hat
    lambda_r : float
        parameter for the strength of Adaptive Lasso loss to shrink theta
    lasso_weight_vec : 1-D numpy array
        weight of Adaptive Lasso, 1 ./ theta
    lambda_l2 : float
        parameter for the strength of L2 panealty to shrink theta
    hybrid_version : bool, optional
        if True, use the hybrid_version of GLRM, i.e. in ADMM local model loss function optimization for w but adaptive lasso constrain on theta. If False, local model loss function optimization and adaptive lasso will on the same w. The default is True.
    hv_x : 1-D numpy array, optional
        data points served as x for calculation of probability density values. Only used for heavy-tail.
    hv_log_p : 1-D numpy array, optional
        log density values of normal distribution N(0, sigma^2) + heavy-tail. Only used for heavy-tail.
    N : float
        sequencing depth for this spot. If is None, use sum(y_vec) instead.
    use_cache : bool, optional
        if True, use the cached dict of calculated likelihood values.
    
    Returns
    -------
    a tuple (float, 1-D numpy array)
        the loss function (base model loss + ADMM loss + Adaptive Lasso loss + L2 loss) of one spot to update w (e_alpha*theta) + gradient vector
    '''

    
    def admm_penalty_loss():
        '''
        calculate the loss for ADMM of making theta=theta_hat
        
                       1/(2*rho) (||w/sum(w) - theta_hat + u||_2)^2
                       
                       and nu = theta_hat - u

        Returns
        -------
        a tuple (float, 1-D numpy array)
            ADMM loss + gradient
        '''
        
        if rho is None or nu_vec is None:
            return (0, np.zeros(w_vec.shape))
        else:
            if hybrid_version:
                # ADMM loss on theta
                tmp_sum = np.sum(w_vec)
                return (np.sum(((w_vec/tmp_sum - nu_vec)**2))/(2*rho), 1/rho * ((w_vec*tmp_sum-np.sum(np.square(w_vec)))/(tmp_sum**3) - (nu_vec*tmp_sum-np.inner(nu_vec, w_vec))/(tmp_sum**2)))
            else:
                # ADMM loss on w
                return (np.sum(((w_vec-nu_vec)**2))/(2*rho), (w_vec-nu_vec)/rho)
        
    
    def adaptive_lasso_loss():
        '''
        calculate the loss for Adaptive Lasso
        
                       lambda_r * (inner product(w/sum(w), lasso_weight))

        Returns
        -------
        a tuple (float, 1-D numpy array)
            Adaptive Lasso loss + gradient
        '''
        
        if lambda_r is None or lasso_weight_vec is None:
            return (0, np.zeros(w_vec.shape))
        else:
            if hybrid_version:
                # Adaptive Lasso loss on theta
                tmp_sum = np.sum(w_vec)
                return (lambda_r * np.inner(w_vec/tmp_sum, lasso_weight_vec), lambda_r * (lasso_weight_vec*tmp_sum-np.inner(w_vec, lasso_weight_vec)) / (tmp_sum**2))
            else:
                # Adaptive Lasso loss on w
                return (lambda_r * np.inner(w_vec, lasso_weight_vec), lambda_r*lasso_weight_vec)
    
    
    def l2_loss():
        '''
        calculate the L2 penalty
        
                        lambda_l2 * (sum(squared(w/sum(w))))

        Returns
        -------
        a tuple (float, 1-D numpy array)
            L2 penalty + gradient

        '''
        
        if lambda_l2 is None:
            return (0, np.zeros(w_vec.shape))
        else:
            if hybrid_version:
                # L2 loss on theta
                tmp_sum = np.sum(w_vec)
                return (lambda_l2 * np.sum(np.square(w_vec/tmp_sum)), lambda_l2 * 2 * (w_vec*tmp_sum - np.sum(np.square(w_vec))) / (tmp_sum**3))
            else:
                # L2 loss on w
                return (lambda_l2 * np.inner(np.square(w_vec)), lambda_l2*2*w_vec)
    
    # combine loss and gradient on w separately
    return tuple([a+b+c+d for a,b,c,d in zip(hv_comb(w_vec, y_vec, mu, gamma_g, sigma2, hv_x, hv_log_p, N, use_cache), admm_penalty_loss(), adaptive_lasso_loss(), l2_loss())])



def update_theta(data, warm_start_theta, warm_start_e_alpha, gamma_g, sigma2, nu=None, rho=None, lambda_r=None, lasso_weight=None, lambda_l2=None, global_optimize=False, hybrid_version=True, opt_method='L-BFGS-B', hv_x=None, hv_log_p=None, theta_mask=None, verbose=False, use_cache=True):
    '''
    update theta (celltype proportion) and e_alpha (spot-specific effect) given sigma2 (variance paramter of the log-normal distribution) and gamma_g (gene-specific platform effect) by MLE
    
    we assume 
    
        ln(lambda) = alpha + gamma_g + ln(sum(theta*mu_X)) + epsilon
    
        subject to sum(theta)=1, theta>=0
    
    mu_X is marker genes from data['X']
    
    then the mean parameter of the lognormal distribution of ln(lambda) is alpha + gamma_g + ln(sum(theta*mu_X))
    
    we did re-parametrization w = e^alpha * theta, then
    
        ln(lambda) = gamma_g + ln(sum([e^alpha*theta]*mu_X)) + epsilon
    
        subject to w>=0, it will imply sum(theta)=1 and theta>=0
    
    the steps to update theta and e_alpha:
        1. dimension change of theta, theta_hat, u from 3-D (spots * celltypes * 1) to 1-D (celltypes), and do re-parametrization to get w
        2. solve w for each spot in parallel
        3. extract updated theta and e_alpha from w, and change the dimension of updated theta, theta_hat, u back

    Parameters
    ----------
    data : Dict
        a Dict contains all info need for modeling:
            X: a 2-D numpy matrix of celltype specific marker gene expression (celltypes * genes).\n
            Y: a 2-D numpy matrix of spatial gene expression (spots * genes).\n
            A: a 2-D numpy matrix of Adjacency matrix (spots * spots), or is None. Adjacency matrix of spatial sptots (1: connected / 0: disconnected). All 0 in diagonal.\n
            N: a 1-D numpy array of sequencing depth of all spots (length #spots). If it's None, use sum of observed marker gene expressions as sequencing depth.\n
            non_zero_mtx: If it's None, then do not filter zeros during regression. If it's a bool 2-D numpy matrix (spots * genes) as False means genes whose nUMI=0 while True means genes whose nUMI>0 in corresponding spots. The bool indicators can be calculated based on either observerd raw nUMI counts in spatial data, or CVAE transformed nUMI counts.\n
            spot_names: a list of string of spot barcodes. Only keep spots passed filtering.\n
            gene_names: a list of string of gene symbols. Only keep actually used marker genes.\n
            celltype_names: a list of string of celltype names.
    warm_start_theta : 3-D numpy array (spots * celltypes * 1)
        initial guess of theta (celltype proportion).
    warm_start_e_alpha : 1-D numpy array
        initial guess of e_alpha (spot-specific effect).
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
    nu : 3-D numpy array (spots * celltypes * 1), optional
        variable for ADMM penalty of all spots
        in 3 part ADMM nu = theta_hat (used for regularization/penalty) - u (scaled dual variables to make theta = theta_hat)
    rho : float, optional
        parameter for the strength of ADMM loss to make theta equals theta_hat
    lambda_r : float, optional
        parameter for the strength of Adaptive Lasso loss to shrink theta
    lasso_weight : 3-D numpy array (spots * celltypes * 1), optional
        weight of Adaptive Lasso, 1 ./ theta
    lambda_l2 : float
        parameter for the strength of L2 panealty to shrink theta
    global_optimize : bool, optional
        if is True, use basin-hopping algorithm to find the global minimum. The default is False.
    hybrid_version : bool, optional
        if True, use the hybrid_version of GLRM, i.e. in ADMM local model loss function optimization for w but adaptive lasso constrain on theta. If False, local model loss function optimization and adaptive lasso will on the same w. The default is True.
    opt_method : string, optional
        specify method used in scipy.optimize.minimize for local model fitting. The default is 'L-BFGS-B', a default method in scipy for optimization with bounds. Another choice would be 'SLSQP', a default method in scipy for optimization with constrains and bounds.
    hv_x : 1-D numpy array, optional
        data points served as x for calculation of probability density values. Only used for heavy-tail.
    hv_log_p : 1-D numpy array, optional
        log density values of normal distribution N(0, sigma^2) + heavy-tail. Only used for heavy-tail.
    theta_mask : 3-D numpy array (spots * celltypes * 1), optional
        mask for cell-type proportions (1: present, 0: not present). Only used for stage 2 theta optmization.
    verbose : bool, optional
        if True, print more information.
    use_cache : bool, optional
        if True, use the cached dict of calculated likelihood values.
        
    Returns
    -------
    theta_results : 3-D numpy array (spots * celltypes * 1)
        updated theta (celltype proportion).
    e_alpha_results : 1-D numpy array
        updated e_alpha (spot-specific effect).
    '''


    n_celltype = data["X"].shape[0]
    n_spot = data["Y"].shape[0]
    
    # skip optimization if the initial theta corresponding to only one cell-type
    skip_opt = True
    
    # prepare parameter tuples for parallel computing
    results = []
    for i in range(n_spot):
        
        if skip_opt:
            this_warm_start_theta = warm_start_theta[i, :, :].flatten()
            if (this_warm_start_theta==1).any():
                assert np.sum(this_warm_start_theta) == 1
                results.append(this_warm_start_theta)
                #print(f'skip optimization for spot {i} as only one celltype present')
                continue
        
        if theta_mask is None:
            this_present_celltype_index = None
        else:
            this_present_celltype_index = theta_mask[i,:,:]==1
        
            # only one cell-type present, we can directly determine the proportion as 1
            if np.sum(this_present_celltype_index) == 1:
                simple_sol = np.zeros((n_celltype,))
                simple_sol[this_present_celltype_index.flatten()] = 1
                results.append(simple_sol)
                continue
        
        y_vec = data["Y"][i, :]
        
        # extract only marker gene expressions for presented cell-types
        if theta_mask is None:
            mu = data["X"]
        else:
            mu = data["X"][this_present_celltype_index.flatten(), :]
        
        if nu is None:
            nu_vec = None
        else:
            if theta_mask is None:
                nu_vec = nu[i, :, :].flatten()
            else:
                nu_vec = nu[i, this_present_celltype_index].flatten()
        
        if theta_mask is None:
            this_warm_start_theta = warm_start_theta[i, :, :].flatten()
        else:
            this_warm_start_theta = warm_start_theta[i, this_present_celltype_index].flatten()
        
        this_warm_start_e_alpha = warm_start_e_alpha[i]
        
        # re-parametrization
        warm_start_w = this_warm_start_theta * this_warm_start_e_alpha
        
        if lasso_weight is None:
            lasso_weight_vec = None
        else:
            if theta_mask is None:
                lasso_weight_vec = lasso_weight[i, :, :].flatten()
            else:
                lasso_weight_vec = lasso_weight[i, this_present_celltype_index].flatten()
            
        # sequencing depth
        if data["N"] is None:
            N = None
        else:
            N = data["N"][i]
            
        # filter zero genes
        if data['non_zero_mtx'] is None:
            this_y_vec = y_vec
            this_gamma_g = gamma_g
            this_mu = mu
        else:
            non_zero_gene_ind = data['non_zero_mtx'][i, :]
            #print(f'total {np.sum(non_zero_gene_ind)} non-zero genes ({np.sum(non_zero_gene_ind)/len(non_zero_gene_ind):.2%}) for spot {i:d}')
            this_y_vec = y_vec[non_zero_gene_ind]
            this_gamma_g = gamma_g[non_zero_gene_ind]
            this_mu = mu[:, non_zero_gene_ind]
            
        # start optimization
        # call minimize function to solve w (e_alpha*theta)
        # bounds : tuple of tuples
        #    sequence of (min, max) pairs for each element in w_vec
        # min not set as 0 to avoid divided by 0 or log(0)
        # if jac is a Boolean and is True, fun is assumed to return a tuple (f, g) containing the objective function and the gradient
        
        #from time import time
        #start_time = time()
        
        bounds = (((min_theta, None),) * len(warm_start_w))
        
        if global_optimize:
            bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
            # use the basin-hopping algorithm to find the global minimum
            sol = basinhopping(objective_loss_theta, warm_start_w, niter=10, T=1.0, take_step=bounded_step,
                               minimizer_kwargs={'method': opt_method,
                                                 'args': (this_y_vec, this_mu, this_gamma_g, sigma2, nu_vec, rho, lambda_r, lasso_weight_vec, lambda_l2, hybrid_version, hv_x, hv_log_p, N, use_cache),
                                                 'bounds': bounds,
                                                 'options': {'maxiter': 250, 'eps': theta_eps},
                                                 'jac': True
                                                 },
                               disp=False)
        else:
            sol = minimize(objective_loss_theta, warm_start_w, args=(this_y_vec, this_mu, this_gamma_g, sigma2, nu_vec, rho, lambda_r, lasso_weight_vec, lambda_l2, hybrid_version, hv_x, hv_log_p, N, use_cache),
                       method=opt_method,
                       bounds=bounds, options={'disp': False, 'maxiter': 250, 'eps': theta_eps}, jac=True)
        
        #print(f'spot {i:d} optimization finished in {sol.nit} iterations. Elapsed time: {time()-start_time:.2f} seconds.')
        
        if not global_optimize:
            if sol.status != 0:
                if verbose:
                    # status 2: Maximum number of iterations has been exceeded
                    print(f'WARNING: w optimization in local model of spot {data["spot_names"][i]} not successful! Caused by: {sol.message}')
        
        solve_fail_flag = False
        
        if sum(sol.x) == 0:
            print(f'###### Error: w optimization in local model of spot {data["spot_names"][i]} returns all 0s! ######')
            solve_fail_flag = True
            
        if np.any(np.isnan(sol.x)):
            print(f'###### Error: w optimization in local model of spot {data["spot_names"][i]} returns NaN value! ######')
            solve_fail_flag = True
           
        if np.any(np.isinf(sol.x)):
            print(f'###### Error: w optimization in local model of spot {data["spot_names"][i]} returns Infinite value! ######')
            solve_fail_flag = True
            
        if solve_fail_flag:
            # replace NaN or Infinite as 0
            this_sol = np.nan_to_num(sol.x, nan=0.0, posinf=0.0, neginf=0.0)
            if sum(this_sol) == 0:
                # reset w to all elements equal
                tmp_len = this_sol.shape[0]
                this_sol = np.full((tmp_len,), 1.0/tmp_len) * this_warm_start_e_alpha
                print('reset w to all elements identical')
            else:
                # renormalize to sum to 1
                this_sol = this_sol / sum(this_sol)
                print('repalce non-numeric value as 0 then re-normalize to sum to 1')
        else:
            this_sol = sol.x
        
        # transform back theta, adding non-present values
        if theta_mask is None:
            results.append(this_sol)
        else:
            this_sol_full = np.zeros((n_celltype,))
            this_sol_full[this_present_celltype_index.flatten()] = this_sol
            results.append(this_sol_full)
    
    # collect results: theta and e_alpha
    theta_results = np.zeros((n_spot, n_celltype, 1))
    e_alpha_results = []
    
    for i, this_result in enumerate(results):
        # extract theta and e_alpha
        tmp_e_alpha = np.sum(this_result)
        tmp_theta = this_result / tmp_e_alpha
        # change dimension back
        theta_results[i, :, :] = np.reshape(tmp_theta, (n_celltype, 1))
        e_alpha_results.append(tmp_e_alpha)
    
    e_alpha_results = np.array(e_alpha_results)
    
    return theta_results, e_alpha_results



################################# code related to update sigma2 #################################

def update_sigma2(data, theta, e_alpha, gamma_g, sigma2, opt_method='L-BFGS-B', global_optimize=False, hv_x=None, verbose=False, use_cache=True):
    '''
    update sigma2 (variance paramter of the lognormal distribution) given theta (celltype proportion), e_alpha (spot-specific effect) and gamma_g (gene-specific platform effect) by MLE
    
    we assume
    
        ln(lambda) = alpha + gamma_g + ln(sum(theta*mu_X)) + epsilon
        
        subject to sum(theta)=1, theta>=0
    
    mu_X is marker genes from data['X']
    
    then the mean parameter of the log-normal distribution of ln(lambda) is alpha + gamma_g + ln(sum(theta*mu_X))
    
    we did re-parametrization w = e^alpha * theta, then
    
        ln(lambda) = gamma_g + ln(sum([e^alpha*theta]*mu_X)) + epsilon
        
        subject to w>=0, it will imply sum(theta)=1 and theta>=0
              
    currently optimization of sigma2 DO NOT support parallel computing

    Parameters
    ----------
    data : Dict
        a Dict contains all info need for modeling:
            X: a 2-D numpy matrix of celltype specific marker gene expression (celltypes * genes).\n
            Y: a 2-D numpy matrix of spatial gene expression (spots * genes).\n
            A: a 2-D numpy matrix of Adjacency matrix (spots * spots), or is None. Adjacency matrix of spatial sptots (1: connected / 0: disconnected). All 0 in diagonal.\n
            N: a 1-D numpy array of sequencing depth of all spots (length #spots). If it's None, use sum of observed marker gene expressions as sequencing depth.\n
            non_zero_mtx: If it's None, then do not filter zeros during regression. If it's a bool 2-D numpy matrix (spots * genes) as False means genes whose nUMI=0 while True means genes whose nUMI>0 in corresponding spots. The bool indicators can be calculated based on either observerd raw nUMI counts in spatial data, or CVAE transformed nUMI counts.\n
            spot_names: a list of string of spot barcodes. Only keep spots passed filtering.\n
            gene_names: a list of string of gene symbols. Only keep actually used marker genes.\n
            celltype_names: a list of string of celltype names.
    theta : 3-D numpy array (spots * celltypes * 1)
        theta (celltype proportion).
    e_alpha : 1-D numpy array
        e_alpha (spot-specific effect).
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    sigma2 : float
        initial guess of variance paramter of the lognormal distribution of ln(lambda). All genes and spots share the same variance.
    opt_method : string, optional
        specify method used in scipy.optimize.minimize for local model fitting. The default is 'L-BFGS-B', a default method in scipy for optimization with bounds. Another choice would be 'SLSQP', a default method in scipy for optimization with constrains and bounds.
    global_optimize : bool, optional
        if is True, use basin-hopping algorithm to find the global minimum. The default is False.
    hv_x : 1-D numpy array, optional
        data points served as x for calculation of probability density values. Only used for heavy-tail.
    verbose : bool, optional
        if True, print more information.
    use_cache : bool, optional
        if True, use the cached dict of calculated likelihood values.
        
    Returns
    -------
    float
        updated sigma2
    '''
  
    def objective_loss_sigma2(sigma2, data, theta, e_alpha, gamma_g, hv_x, use_cache):
        '''
        calculate loss function for updating sigma2 (variance paramter of the log-normal distribution of ln(lambda)). All spots and genes share the same variance.
        
        the loss function is a sum of negative log-likelihood of the base model of each spot, and the so-called basemodel loss is the same as that for updating theta (celltype proportion)
        
        and we did re-parametrization w = e^alpha * theta, so theta = w / sum(w)
        
        we calculate the basemodel loss of each spot in a parallel way by Numba
        
        WARNING: this input sigma2 inside the loss function is actually a numpy array
        
        Returns
        -------
        float
            the loss function (sum of base model loss across all spots) to update sigma2
        '''
    
        n_spot = data["Y"].shape[0]
        
        # round sigma2 for rough but quick analysis
        if use_cache:
            this_sigma2 = round(sigma2[0], sigma2_digits)
        else:
            this_sigma2 = sigma2[0]
        
        # insert new sigma2 key
        if use_cache:
            insert_key_sigma2_wrapper(this_sigma2)
        
        # update density values of heavy-tail with current sigma^2
        hv_log_p = generate_log_heavytail_array(hv_x, np.sqrt(this_sigma2))
        
        #from time import time
        #start_time = time()
        
        results = 0.0
        
        for i in range(n_spot):
            y_vec = data["Y"][i, :]
            this_theta = theta[i, :, :].flatten()
            this_e_alpha = e_alpha[i]
            # re-parametrization
            this_w = this_theta * this_e_alpha
            
            # sequencing depth
            if data["N"] is None:
                N = None
            else:
                N = data["N"][i]
                
            # filter zero genes
            if data['non_zero_mtx'] is None:
                this_y_vec = y_vec
                this_gamma_g = gamma_g
                this_mu = data["X"]
            else:
                non_zero_gene_ind = data['non_zero_mtx'][i, :]
                #print(f'total {np.sum(non_zero_gene_ind)} non-zero genes ({np.sum(non_zero_gene_ind)/len(non_zero_gene_ind):.2%}) for spot {i:d}')
                this_y_vec = y_vec[non_zero_gene_ind]
                this_gamma_g = gamma_g[non_zero_gene_ind]
                this_mu = data["X"][:, non_zero_gene_ind]
        
            results += hv_wrapper(this_w, this_y_vec, this_mu, this_gamma_g, this_sigma2, hv_x, hv_log_p, N, use_cache)
            
            #print(f'after summing up spot {i:d}, current loss is {results:.6f}')
            
        #print(f'One round of summing up loss across all {n_spot:d} spots for sigma^2 optimization. Elapsed time: {time()-start_time:.2f} seconds.')
  
        return results
    
    
    # the min value for clip sigma^2 should be larger
    bounds = ((min_sigma2, None),)
    
    if global_optimize:
        bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
        # use the basin-hopping algorithm to find the global minimum
        sol = basinhopping(objective_loss_sigma2, sigma2, niter=10, T=1.0, take_step=bounded_step,
                           minimizer_kwargs={'method': opt_method,
                                             'args': (data, theta, e_alpha, gamma_g, hv_x, use_cache),
                                             'bounds': bounds,
                                             'options': {'maxiter': 250, 'eps': sigma2_eps}
                                             },
                           disp=False)
    else:
        sol = minimize(objective_loss_sigma2, sigma2, args=(data, theta, e_alpha, gamma_g, hv_x, use_cache),
                   method=opt_method,
                   bounds=bounds, options={'disp': False, 'maxiter': 250, 'eps': sigma2_eps})
    
    
    if not global_optimize:
        if sol.status != 0:
            if verbose:
                print(f'WARNING: sigma2 optimization in local model of all spots not successful! Caused by: {sol.message}')
    
    if not math.isfinite(sol.x[0]):
        raise Exception('Error: sigma2 optimization in local model of all spots returns a non-finite value!')
        
    
    # the solution in x is a numpy array
    return sol.x[0]



################################# code related to update theta_tilde by Adaptive Lasso in ADMM #################################

def adaptive_lasso(nu, rho, lambda_r=1.0, lasso_weight=None):
    '''
    update theta_tilde by Adaptive Lasso and ADMM loss
    
    theta_tilde = argmin lambda_r*lasso_weight*theta_tilde + 1/(2*rho) * (||theta_tilde-theta_hat+u_tilde||_2)^2
    
    that is
    
        - if theta_tilde>=0, theta_tilde = theta_hat - u_tilde - lambda_r*rho*lasso_weight
        
        - if theta_tilde<0, theta_tilde = theta_hat - u_tilde + lambda_r*rho*lasso_weight
            
    change it to
    
        - if theta_hat-u_tilde-lambda_r*rho*lasso_weight>=0, theta_tilde = theta_hat - u_tilde - lambda_r*rho*lasso_weight
        
        - if theta_hat-u_tilde+lambda_r*rho*lasso_weight<=0, theta_tilde = theta_hat - u_tilde + lambda_r*rho*lasso_weight
        
        - if theta_hat-u_tilde-lambda_r*rho*lasso_weight<0 or theta_hat-u_tilde+lambda_r*rho*lasso_weight>0, not defined, let theta_tilde = 0
            
    and nu = theta_hat - u_tilde
    
    WANRING: negative theta tilde value observed, so also clip it to >= 0

    Parameters
    ----------
    nu : 3-D numpy array (spots * celltypes * 1)
        variable for ADMM penalty of all spots
        in 3 part ADMM nu = theta_hat (used for regularization/penalty) - u_tilde (scaled dual variables to make theta_tilde = theta_hat)
    rho : float
        parameter for the strength of ADMM loss to make theta_tilde equals theta_hat
    lambda_r : float, optional
        strength for Adaptive Lasso penalty. The default is 1.0.
    lasso_weight : 3-D numpy array (spots * celltypes * 1), optional
        calculated weight for adaptive lasso. The default is None.

    Returns
    -------
    3-D numpy array (spots * celltypes * 1)
        updated theta_tilde.
    '''

    if lasso_weight is None:
        lasso_weight = np.ones(nu.shape)
    
    result = np.maximum(nu - rho*lambda_r*lasso_weight, 0) - np.maximum(-nu - rho*lambda_r*lasso_weight, 0)
    
    # avoid negative values
    result[result<min_theta] = min_theta
    
    return result