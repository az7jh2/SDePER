#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 22:17:11 2022

@author: hill103

this script implement the optimization of theta and sigma^2 using Poisson log-normal distribution + heavy-tail

use Numba parallel for best performance

we calculate the likelihood for each gene sequentially, utilizing Numba's parallel capabilities to accelerate the computation within each gene's calculation.

NOTE: simplifying the loss-computation code and removing unneeded nested wrappers yielded a speedup

and print not supported in Numba function

functions for likelihood calculation:
    
    calc_hv_numba: Numba function so parallel computing by Numba; calculate likelihoods given an array of y and mu then return; NOT related with cache dict
    
    hv_comb: return sum of negative log-likelihoods + gradient of w; used for theta optimization; directly call hv_numba
    
    hv_wrapper: return sum of negative log-likelihoods; used for sigma^2 optimization; directly call hv_numba
    
    hv_numba: call calc_hv_numba and return array of likelihoods; support caching (DELETED)
    
    
sequence of function calls for parameter optimization:
    
    update_theta -> objective_loss_theta -> hv_comb -> calc_hv_numba
    
    update_sigma2 -> objective_loss_sigma2 -> hv_wrapper -> calc_hv_numba
    
    calcHVBaseModelLoss -> hv_wrapper in hyper parameter tunning
"""



import numpy as np
from scipy.optimize import minimize, basinhopping
import scipy.stats
import numba as na
import math
from config import min_val, min_theta, min_sigma2, N_z, gamma, print, theta_eps, sigma2_eps



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
# based on test, Numba parallel on one gene DO NOT improve speed much; on ALL genes DOES
# set nopython True/False, fastmath True/False DO NOT affect speed
# see the ref https://stackoverflow.com/a/64613366/13752320 for advices of using Numba efficiently
# UPDATE: we modified it for better Numba handling
@na.jit(nopython=True, parallel=True, fastmath=False, error_model="numpy", cache=False)
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
    
    G = MU.shape[0]
    Z = hv_x.shape[0]
    for i in na.prange(G):                # parallel ONLY here
        mu_i = MU[i]
        yi   = y_vec[i]
        lg   = math.lgamma(yi + 1.0)   # invariant over j
        acc  = 0.0
        for j in range(Z):             # <- plain range (not prange)
            tmp = mu_i + hv_x[j]
            # two exps is unavoidable here; still hoisted invariants help
            acc += math.exp(yi*tmp + hv_log_p[j] - math.exp(tmp) - lg)
        output[i] = acc * gamma        # single write
    return output



def hv_comb(w_vec, y_vec, mu, gamma_g, sigma2, hv_x, hv_log_p, N):
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

    Returns
    -------
    tuple of (float, 1-D numpy array)
        sum of negative log-likelihood across all genes in one spot + gradient vector of w.
    '''
    # return both negative log-likelihoods and 1st order derivative
    if N is None:
        N = np.sum(y_vec)
    
    # The @ operator performs matrix multiplication, w_vec@mu produces a 1-D numpy array with length as number of genes
    MU = np.log(w_vec@mu+min_val) + gamma_g + np.log(N)
    this_lambda = (w_vec@mu) * np.exp(gamma_g) * N
    
    likelihoods = calc_hv_numba(MU, y_vec, hv_x, hv_log_p, np.zeros(MU.shape), gamma)
    
    loss = -np.sum(np.log(likelihoods + min_val))
    
    # for gradient of w
    likelihoods_y_add_1 = calc_hv_numba(MU, y_vec+1, hv_x, hv_log_p, np.zeros(MU.shape), gamma)
    
    # element-wise operation
    tmp = (y_vec*likelihoods - (y_vec+1)*likelihoods_y_add_1 + min_val) / (this_lambda*likelihoods + min_val)
    der = []
    # sum over genes, get result related to cell-types
    for i in range(mu.shape[0]):
        der.append(-np.sum(tmp * np.exp(gamma_g) * N * mu[i, :].flatten()))
    
    return loss, np.array(der)



def hv_wrapper(w_vec, y_vec, mu, gamma_g, sigma2, hv_x, hv_log_p, N):
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

    Returns
    -------
    float
        sum of negative log-likelihood across all genes in one spot.
    '''
    
    if N is None:
        N = np.sum(y_vec)
    
    MU = np.log(w_vec@mu+min_val) + gamma_g + np.log(N)
    likelihoods = calc_hv_numba(MU, y_vec, hv_x, hv_log_p, np.zeros(MU.shape), gamma)
    loss = -np.sum(np.log(likelihoods + min_val))
    
    return loss



################################# code related to update theta #################################

def objective_loss_theta(w_vec, y_vec, mu, gamma_g, sigma2, nu_vec, rho, lambda_r, lasso_weight_vec, lambda_l2, hybrid_version, hv_x, hv_log_p, N):
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
    return tuple([a+b+c+d for a,b,c,d in zip(hv_comb(w_vec, y_vec, mu, gamma_g, sigma2, hv_x, hv_log_p, N), admm_penalty_loss(), adaptive_lasso_loss(), l2_loss())])



def optimize_one_theta(mu, y_vec, N, this_warm_start_theta, this_warm_start_e_alpha, gamma_g, sigma2, spot_name, nu_vec=None, rho=None, lambda_r=None, lasso_weight_vec=None, lambda_l2=None, global_optimize=False, hybrid_version=True, opt_method='L-BFGS-B', hv_x=None, hv_log_p=None, this_theta_mask=None, skip_opt=True, verbose=False):
    '''
    update theta (celltype proportion) sigma2 (variance paramter of the log-normal distribution) and gamma_g (gene-specific platform effect) by MLE in ONE SPOT

    Parameters
    ----------
    mu : 2-D numpy matrix
        matrix of celltype specific marker gene expression (celltypes * genes).
    y_vec : 1-D numpy array
        spatial gene expression (length #genes).
    N : int or None
        sequencing depth of this spot. If it's None, use sum of observed marker gene expressions as sequencing depth.
    spot_name : string
        name of this spot.
    this_warm_start_theta : 1-D numpy array (length #celltypes)
        initial guess of theta (celltype proportion).
    this_warm_start_e_alpha : float
        initial guess of e_alpha (spot-specific effect).
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
    nu_vec : 1-D numpy array (length #celltypes), optional
        variable for ADMM penalty of all spots
        in 3 part ADMM nu = theta_hat (used for regularization/penalty) - u (scaled dual variables to make theta = theta_hat)
    rho : float, optional
        parameter for the strength of ADMM loss to make theta equals theta_hat
    lambda_r : float, optional
        parameter for the strength of Adaptive Lasso loss to shrink theta
    lasso_weight_vec : 1-D numpy array (length #celltypes), optional
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
    this_theta_mask : 1-D numpy array (length #celltypes), optional
        mask for cell-type proportions (1: present, 0: not present).
    skip_opt : bool, optional
        if True, when only one cell type present, we skip optimization and directly return theta.
    verbose : bool, optional
        if True, print more information.
        
    Returns
    -------
    w_result : 1-D numpy array (length #celltypes)
        updated w, i.e. theta (celltype proportion) * e_alpha.
    '''


    n_celltype = len(this_warm_start_theta)
    
    if skip_opt:
        if np.isclose(this_warm_start_theta, 1.0, atol=1e-7).any():
            assert np.sum(this_warm_start_theta) == 1
            #print(f'skip optimization for spot {spot_name} as only one celltype present')
            return this_warm_start_theta
    
    # Prepare variables based on whether sparsity considered via theta mask
    if this_theta_mask is None:
        this_present_celltype_index = None
    else:
        this_present_celltype_index = this_theta_mask==1
    
        # only one cell-type present, we can directly determine the proportion as 1
        if skip_opt:
            if np.sum(this_present_celltype_index) == 1:
                simple_sol = np.zeros((n_celltype,))
                simple_sol[this_present_celltype_index] = 1
                #print(f'skip optimization for spot {spot_name} as only one celltype present in mask')
                return simple_sol

    if this_present_celltype_index is not None:
        # extract only marker gene expressions for presented cell-types
        mu = mu[this_present_celltype_index, :]
        
        if nu_vec is not None:
            nu_vec = nu_vec[this_present_celltype_index]
    
        this_warm_start_theta = this_warm_start_theta[this_present_celltype_index]
    
        if lasso_weight_vec is not None:
            lasso_weight_vec = lasso_weight_vec[this_present_celltype_index]
            
    # re-parametrization
    warm_start_w = this_warm_start_theta * this_warm_start_e_alpha
    
        
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
                                             'args': (y_vec, mu, gamma_g, sigma2, nu_vec, rho, lambda_r, lasso_weight_vec, lambda_l2, hybrid_version, hv_x, hv_log_p, N),
                                             'bounds': bounds,
                                             'options': {'maxiter': 250, 'eps': theta_eps},
                                             'jac': True
                                             },
                           disp=False)
    else:
        sol = minimize(objective_loss_theta, warm_start_w, args=(y_vec, mu, gamma_g, sigma2, nu_vec, rho, lambda_r, lasso_weight_vec, lambda_l2, hybrid_version, hv_x, hv_log_p, N),
                   method=opt_method,
                   bounds=bounds, options={'disp': False, 'maxiter': 250, 'eps': theta_eps}, jac=True)
    
    #print(f'spot {spot_name} optimization finished in {sol.nit} iterations. Elapsed time: {time()-start_time:.2f} seconds.')
    
    if not global_optimize:
        if not sol.success:
            if verbose:
                # Ref: status 2 - Maximum number of iterations has been exceeded
                print(f'[WARNING] w optimization in local model of spot {spot_name} not successful! Caused by: {sol.message}')
    
    solve_fail_flag = False
    
    if sum(sol.x) == 0:
        print(f'###### [Error] w optimization in local model of spot {spot_name} returns all 0s! ######')
        solve_fail_flag = True
        
    if np.any(np.isnan(sol.x)):
        print(f'###### [Error] w optimization in local model of spot {spot_name} returns NaN value! ######')
        solve_fail_flag = True
       
    if np.any(np.isinf(sol.x)):
        print(f'###### [Error] w optimization in local model of spot {spot_name} returns Infinite value! ######')
        solve_fail_flag = True
        
    if solve_fail_flag:
        # replace NaN or Infinite as 0
        this_sol = np.nan_to_num(sol.x, nan=0.0, posinf=0.0, neginf=0.0)
        # NOTE this is w, no need to sum to 1
        print('repalce non-numeric value as 0')
        
        if sum(this_sol) == 0:
            # reset w to all elements equal
            tmp_len = this_sol.shape[0]
            this_sol = np.full((tmp_len,), 1.0/tmp_len) * this_warm_start_e_alpha
            print('[WARNING] reset w to all elements identical!')

    else:
        this_sol = sol.x
        
    # set w values at boundary to 0; note to allow a tolerance
    this_sol[this_sol<=(min_theta*2)] = 0
    
    
    # transform back w to original dimension, adding non-present values
    if this_theta_mask is None:
        w_result = this_sol
    else:
        w_result = np.zeros((n_celltype,))
        w_result[this_present_celltype_index] = this_sol
    
    return w_result 


def update_theta(data, warm_start_theta, warm_start_e_alpha, gamma_g, sigma2, nu=None, rho=None, lambda_r=None, lasso_weight=None, lambda_l2=None, global_optimize=False, hybrid_version=True, opt_method='L-BFGS-B', hv_x=None, hv_log_p=None, theta_mask=None, verbose=False):
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
        
    Returns
    -------
    theta_results : 3-D numpy array (spots * celltypes * 1)
        updated theta (celltype proportion).
    e_alpha_results : 1-D numpy array
        updated e_alpha (spot-specific effect).
    '''


    n_celltype = data["X"].shape[0]
    n_spot = data["Y"].shape[0]
    
    # NOTE input warm_start_theta, warm_start_e_alpha will NOT be changed inside the function
    
    # skip optimization if the initial theta corresponding to only one cell-type
    skip_opt = True
    
    # prepare parameter tuples for parallel computing
    results = []
    for i in range(n_spot):
        
        this_spot_name = data["spot_names"][i]
        this_warm_start_theta = warm_start_theta[i, :, :].copy().flatten()
        this_warm_start_e_alpha = warm_start_e_alpha[i]

        if theta_mask is None:
            this_theta_mask = None
        else:
            this_theta_mask = theta_mask[i, :, :].copy().flatten()

        y_vec = data["Y"][i, :].copy()
        mu = data["X"].copy()

        if nu is None:
            nu_vec = None
        else:
            nu_vec = nu[i, :, :].copy().flatten()

        if lasso_weight is None:
            lasso_weight_vec = None
        else:
            lasso_weight_vec = lasso_weight[i, :, :].copy().flatten()
       
        if data["N"] is None:
            N = None
        else:
            N = data["N"][i]

        # filter zero genes
        if data['non_zero_mtx'] is None:
            this_y_vec = y_vec
            this_gamma_g = gamma_g.copy()
            this_mu = mu
        else:
            non_zero_gene_ind = data['non_zero_mtx'][i, :]
            #print(f'total {np.sum(non_zero_gene_ind)} non-zero genes ({np.sum(non_zero_gene_ind)/len(non_zero_gene_ind):.2%}) for spot {this_spot_name}')
            this_y_vec = y_vec[non_zero_gene_ind]
            this_gamma_g = gamma_g[non_zero_gene_ind].copy()
            this_mu = mu[:, non_zero_gene_ind]
            
        # call optimization function
        results.append(optimize_one_theta(this_mu, this_y_vec, N, this_warm_start_theta, this_warm_start_e_alpha, this_gamma_g, sigma2, this_spot_name, nu_vec=nu_vec, rho=rho, lambda_r=lambda_r, lasso_weight_vec=lasso_weight_vec, lambda_l2=lambda_l2, global_optimize=global_optimize, hybrid_version=hybrid_version, opt_method=opt_method, hv_x=hv_x, hv_log_p=hv_log_p, this_theta_mask=this_theta_mask, skip_opt=skip_opt, verbose=verbose))
        
    
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

def update_sigma2(data, theta, e_alpha, gamma_g, sigma2, opt_method='L-BFGS-B', global_optimize=False, hv_x=None, verbose=False):
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
        
    Returns
    -------
    float
        updated sigma2
    '''
  
    def objective_loss_sigma2(sigma2, data, theta, e_alpha, gamma_g, hv_x):
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
        
        this_sigma2 = sigma2[0]
        
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
        
            results += hv_wrapper(this_w, this_y_vec, this_mu, this_gamma_g, this_sigma2, hv_x, hv_log_p, N)
            
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
                                             'args': (data, theta, e_alpha, gamma_g, hv_x),
                                             'bounds': bounds,
                                             'options': {'maxiter': 250, 'eps': sigma2_eps}
                                             },
                           disp=False)
    else:
        sol = minimize(objective_loss_sigma2, sigma2, args=(data, theta, e_alpha, gamma_g, hv_x),
                   method=opt_method,
                   bounds=bounds, options={'disp': False, 'maxiter': 250, 'eps': sigma2_eps})
    
    
    if not global_optimize:
        if not sol.success:
            if verbose:
                print(f'[WARNING] sigma2 optimization in local model of all spots not successful! Caused by: {sol.message}')
    
    if not math.isfinite(sol.x[0]):
        raise Exception('[Error] sigma2 optimization in local model of all spots returns a non-finite value!')
        
    
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



################################# code related to update theta by in Stage Two without ADMM #################################

def fit_base_model_plus_laplacian(data, L, theta, e_alpha, gamma_g, sigma2,
                                  lambda_r=None, lasso_weight=None,
                                  hv_x=None, hv_log_p=None, theta_mask=None,
                                  opt_method='L-BFGS-B', global_optimize=False, hybrid_version=True,
                                  verbose=False):
    """
    fit base model plus Laplacian penalty, this function is to replace the whole ADMM framework in GLRM stage two
    
    Note: To keep consistent, the output of the fit result is still 3-Dimensional. The dimension transform will be performed outside this function if needed
    
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
            celltype_names: a list of string of celltype names.\n
            initial_guess: initial guess of cell-type proportions of spatial spots.
    L : scipy sparse matrix (spots * spots)
        Laplacian matrix. Note the strenth lambda_g is already absorbed in the L
    theta : 3-D numpy array (spots * celltypes * 1)
        initial guess of theta (celltype proportion).
    e_alpha : 1-D numpy array
        initial guess of e_alpha (spot-specific effect).
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    sigma2 : float
        initial guess of variance paramter of the lognormal distribution of ln(lambda). All genes and spots share the same variance.
        it may not be updated during this optimization, i.e. sigma2 is treated as an already optimized value.
    lambda_r : float, optional
        strength for Adaptive Lasso penalty. The default is None here.
    lasso_weight : 3-D numpy array (spots * celltypes * 1), optional
        calculated weight for adaptive lasso. The default is None.
    hv_x : 1-D numpy array, optional
        data points served as x for calculation of probability density values. Only used for heavy-tail.
    hv_log_p : 1-D numpy array, optional
        log density values of normal distribution N(0, sigma^2) + heavy-tail. Only used for heavy-tail.
    theta_mask : 3-D numpy array (spots * celltypes * 1), optional
        mask for cell-type proportions (1: present, 0: not present). Only used for stage 2 theta optmization.
    opt_method : string, optional
        specify method used in scipy.optimize.minimize for local model fitting. The default is 'L-BFGS-B', a default method in scipy for optimization with bounds. Another choice would be 'SLSQP', a default method in scipy for optimization with constrains and bounds.
    global_optimize : bool, optional
        if is True, use basin-hopping algorithm to find the global minimum. The default is False.
    hybrid_version : bool, optional
        if True, use the hybrid_version of GLRM, i.e. in ADMM local model loss function optimization for w but adaptive lasso constrain on theta. If False, local model loss function optimization and adaptive lasso will on the same w. The default is True.
    verbose : bool, optional
        if True, print more information.

    Returns
    -------
    Dict
        estimated model coefficients, including:
            theta : celltype proportions (#spots * #celltypes * 1)\n
            e_alpha : spot-specific effect (1-D array with length #spot)\n
            sigma2 : variance paramter of the lognormal distribution (float)\n
            gamma_g : gene-specific platform effect for all genes (1-D array with length #gene)
    """
    
    n_celltype = data['X'].shape[0]
    n_spot = data['Y'].shape[0]
    
    from time import time
    start_time = time()
    
    # NOTE input theta, e_alpha will NOT be changed inside the function
    
    # initialize x array for calculation of heavy-tail
    if hv_x is None:
        from local_fit_numba import z_hv
        hv_x = z_hv.copy()
    
    if hv_log_p is None:
        # initialize density values of heavy-tail with initial sigma^2
        hv_log_p = generate_log_heavytail_array(hv_x, np.sqrt(sigma2))

    assert theta_mask is not None
    if verbose:
        print('[HIGHLIGHT] set tighter bounds for proportion w values of NOT present cell types determined in Stage 1')
    
    assert lambda_r is None
    assert lasso_weight is None
    assert hybrid_version is True  # the optimization is on w, but the constrain is on theta
    
    
    # prepare w, and 3D -> 1D
    w0_list = []
    bounds = []
    
    for i in range(n_spot):
        # take the i-th spot
        this_warm_start_theta = theta[i, :, 0]  # shape (n_celltype,)
        this_theta_mask = theta_mask[i, :, 0]
        
        assert np.isclose(np.sum(this_warm_start_theta), 1.0, atol=1e-7), f"Spot {i}: sum={np.sum(this_warm_start_theta):.4g}"
        assert len(this_warm_start_theta) == n_celltype
        assert len(this_theta_mask) == n_celltype
        
        # per-cell-type-proportion bounds
        # avoid [[min_theta, None]] * n_celltype, which will creates shared inner lists, when change one, all will changed simultaneously
        # None means “no upper bound”
        this_bounds = [[min_theta, None] for _ in range(n_celltype)]
        
        # for cell type not presented, we set a very tight bounds
        for j in range(n_celltype):
            if this_theta_mask[j] == 0:
                # this cell type not present
                this_bounds[j][1] = min_theta * 2
        bounds.extend(this_bounds)
        
        # re-parametrization: multiply θ_i by e_alpha[i] to get w_i
        w0_list.append(this_warm_start_theta * e_alpha[i])
    
    w0 = np.concatenate(w0_list, axis=0)  # shape (n_spot * n_celltype,)
    bounds = tuple([tuple(x) for x in bounds])
    
    # Ensure symmetric Laplacian without densifying
    if (L - L.T).nnz != 0:
        raise ValueError("Laplacian L must be symmetric!")
    
    
    # loss function
    def loss_theta_L(w_vec, Y, mu, gamma_g, sigma2, hybrid_version, hv_x, hv_log_p, Ns, L):
        '''
        calculate loss function for updating theta (celltype proportion)
        
        the loss function contains two parts, sum of base model, defined for each spot separately; and the graph Laplacian penalty
        
        also return gradient respect to w
        
        1. for each spot, negative log-likelihood of the base model given observed data and initial parameter value. It sums across all genes. Then we sum it over all spots
        
        2. graph Laplacian penalty
        
        Parameters
        ----------
        w_vec : 1-D numpy array
            e_alpha (spot-specific effect) * theta (celltype proportion) of all spots, flattened to 1D array.
        Y : 2-D numpy array
            observed gene counts of all spots (spots * genes).
        mu : 2-D numpy matrix
            celltype specific marker genes (celltypes * genes)
        gamma_g : 1-D numpy array
            gene-specific platform effect for all genes.
        sigma2 : float
            variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
        hybrid_version : bool, optional
            if True, use the hybrid_version of GLRM, i.e. in ADMM local model loss function optimization for w but adaptive lasso constrain on theta. If False, local model loss function optimization and adaptive lasso will on the same w. The default is True.
        hv_x : 1-D numpy array, optional
            data points served as x for calculation of probability density values. Only used for heavy-tail.
        hv_log_p : 1-D numpy array, optional
            log density values of normal distribution N(0, sigma^2) + heavy-tail. Only used for heavy-tail.
        Ns : 1-D numpy array
            sequencing depth of all spots (length #spots). If is None, use sum(y_vec) instead.
        
        Returns
        -------
        a tuple (float, 1-D numpy array)
            the loss function (base model loss + graph Laplacian loss) over all spots to update w (e_alpha*theta) + gradient vector
        '''
        
        # first revert the w to spot * cell type
        n_spot = Y.shape[0]
        n_celltype = mu.shape[0]
        
        assert len(w_vec) == n_spot * n_celltype
        
        W = w_vec.reshape((n_spot, n_celltype)) # row-wise reshape
        
        base_model_loss = 0
        grad_W = np.zeros((n_spot, n_celltype))   # preallocate gradient matrix
        # re-construct theta from w, theta = w / sum(w)
        Theta = np.zeros((n_spot, n_celltype))
        
        for i in range(n_spot):
            this_w = W[i, :]
            this_y = Y[i, :]
                
            # sequencing depth
            if Ns is None:
                N = None
            else:
                N = Ns[i]
            
            # return the loss function value and gradient for w in this spot i
            cur_loss, cur_gradient = objective_loss_theta(this_w, this_y, mu, gamma_g, sigma2, nu_vec=None, rho=None, lambda_r=None, lasso_weight_vec=None, lambda_l2=None, hybrid_version=hybrid_version, hv_x=hv_x, hv_log_p=hv_log_p, N=N)
            
            # total loss is sum over all spots
            base_model_loss += cur_loss
            
            # since each spot is independent in base model, the the gradient for this spot is only related to this spot, we can directly use it
            grad_W[i, :] = cur_gradient
            
            # re-construct theta, we are sure sum will >0, so no need to add a smalle value to avoid divided by zero; rather add a small value will break the invariance in gradient calculation, leading to incorrect results
            tmp_sum = np.sum(this_w)
            Theta[i, :] = this_w / tmp_sum
        
        
        assert hybrid_version is True  # constrain on theta

        # Note L is a SciPy sparse matrix, and lambda_g is already absorbed in the L
        # @ is matrix multiply, * is Element-wise Multiplication
        # Theta: n_spot * n_celltype
        # loss is tr(Θ^⊤*L*Θ)
        graph_loss = np.sum(Theta * (L @ Theta))  # == tr(Theta.T @ L @ Theta)
        
        # we need to return gradient respect to w, so use chain rule df/dw = df/dθ * dθ/dw
        # df/dθ = 2LΘ, dθ/dw = 1/sum(w) - w/(sum(w)^2)
        # for each row i:
        # ∇_{w_i} f = g_i / s_i  -  ( (w_i · g_i) / s_i^2 ) * 1
        g = 2 * L @ Theta # (n,k)
        s = np.sum(W, axis=1, keepdims=True)  # (n,1)
        # rowwise corrections
        wg = np.sum(W * g, axis=1, keepdims=True)  # (n,1), row-wise inner product w·g
        graph_grad_W = (g / s) - (wg / (s ** 2))  # broadcast-safe, (n,k)
        
        loss = base_model_loss + graph_loss
        grad = grad_W + graph_grad_W
        # reshape grad to vector
        grad_vec = grad.flatten(order='C') # row-wise
        
        return loss, grad_vec

    
    # start optimization
    # call minimize function to solve w (e_alpha*theta)
    # bounds : tuple of tuples
    #    sequence of (min, max) pairs for each element in w_vec
    # min not set as 0 to avoid divided by 0 or log(0)
    # if jac is a Boolean and is True, fun is assumed to return a tuple (f, g) containing the objective function and the gradient
    sol = minimize(loss_theta_L, w0, args=(data["Y"], data["X"], gamma_g, sigma2, hybrid_version, hv_x, hv_log_p, data["N"], L),
               method=opt_method,
               bounds=bounds,
               options={'disp': False, 'maxiter': 250, 'eps': theta_eps}, jac=True)
    
    
    # sol would be (nk,) array, change to n*k
    if sol.success:
        if verbose:
            print(f'w optimization in Stage 2 successful in {sol.nit} iterations')
            # Ref: status 2 - Maximum number of iterations has been exceeded
    else:
        if verbose:
            print(f'[WARNING] w optimization in Stage 2 not successful! Caused by: {sol.message}')
    
    sol_W = np.reshape(sol.x, (n_spot, n_celltype)) # row-wise reshape
    sol_W_cor = np.zeros((n_spot, n_celltype))
    
    for i in range(n_spot):
        solve_fail_flag = False
        this_sol = sol_W[i, ].copy()
    
        if sum(this_sol) == 0:
            if verbose:
                print(f'###### [Error] w optimization in STAGE TWO of spot {data["spot_names"][i]} returns all 0s! ######')
            solve_fail_flag = True
        
        if np.any(np.isnan(this_sol)):
            if verbose:
                print(f'###### [Error] w optimization in STAGE TWO of spot {data["spot_names"][i]} returns NaN value! ######')
            solve_fail_flag = True
       
        if np.any(np.isinf(this_sol)):
            if verbose:
                print(f'###### [Error] w optimization in STAGE TWO of spot {data["spot_names"][i]} returns Infinite value! ######')
            solve_fail_flag = True
        
        if solve_fail_flag:
            # replace NaN or Infinite as 0
            this_sol = np.nan_to_num(this_sol, nan=0.0, posinf=0.0, neginf=0.0)
            if verbose:
                print('repalce non-numeric value as 0')
            # NOTE this is w, no need to sum to 1
            
            if sum(this_sol) == 0:
                # reset w to all elements equal
                tmp_len = this_sol.shape[0]
                this_sol = np.full((tmp_len,), 1.0/tmp_len) * e_alpha[i]
                if verbose:
                    print('[WARNING] reset w to all elements identical!')
    
        # set w values at boundary to 0; note to allow a tolerance
        tmp_ind = (this_sol > 0) & (this_sol <= min_theta * 5)
        if tmp_ind.any():
            this_sol[tmp_ind] = 0
            # NOTE this is w, no need to sum to 1
    
        sol_W_cor[i, ] = this_sol


    # collect results: theta and e_alpha
    theta_results = np.zeros((n_spot, n_celltype, 1))
    e_alpha_results = []

    for i in range(n_spot):
        # extract theta and e_alpha
        this_w_result = sol_W_cor[i, ]
        tmp_e_alpha = np.sum(this_w_result)
        tmp_theta = this_w_result / tmp_e_alpha
        # change dimension back
        theta_results[i, :, :] = np.reshape(tmp_theta, (n_celltype, 1))
        e_alpha_results.append(tmp_e_alpha)

    e_alpha_results = np.array(e_alpha_results)
    
    
    # construct result, DO NOT change theta back to 2-D array
    # the dimension transforming is performed outside this function is needed
    result = {
            'theta': theta_results,
            'e_alpha': e_alpha_results,
            'sigma2': sigma2,
            'gamma_g': gamma_g
            }
    
    if verbose:
        print(f'Optimization with graph Laplacian penalty finished. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.\n')
    
    return result