#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 23:18:26 2022

@author: hill103

this script stores functions related to ADMM framework for fitting model

ADMM framework based on https://github.com/cvxgrp/strat_models
"""



import numpy as np
from time import time
from local_fit_numba import update_theta, adaptive_lasso
from scipy.sparse.linalg import cg
from scipy import sparse
from utils import reparameterTheta, reportRMSE
from config import min_theta, print, sigma2_digits
from local_fit_numba import generate_log_heavytail_array



def one_admm_fit(data, L, theta, e_alpha, gamma_g, sigma2, lambda_r=1.0, lasso_weight=None,
                 hv_x=None, hv_log_p=None, theta_mask=None,
                 abs_tol=1e-3, rel_tol=1e-3,
                 rho=1, mu=10, tau_incr=2, tau_decr=2, max_rho=1e1, min_rho=1e-1,
                 maxiter=100, max_cg_iterations=10,
                 dynamic_rho=True, queue_len=3, diff_threshold=0.05, rho_incr=2, rho_decr=2, diff_scale=5, diff_stop=5e-5,
                 opt_method='L-BFGS-B', global_optimize=False, hybrid_version=True, verbose=False, use_cache=True):
    """
    perform a whole ADMM iterations once as one fitting procedure in GLRM
    
    Note: the output of the fit result is 3-Dimensional. The dimension transform will be performed outside this function if needed
    
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
        Laplacian matrix
    theta : 3-D numpy array (spots * celltypes * 1)
        initial guess of theta (celltype proportion).
    e_alpha : 1-D numpy array
        initial guess of e_alpha (spot-specific effect).
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    sigma2 : float
        initial guess of variance paramter of the lognormal distribution of ln(lambda). All genes and spots share the same variance.
        it may not be updated during the ADMM iterations, i.e. sigma2 is treated as an already optimized value.
    lambda_r : float, optional
        strength for Adaptive Lasso penalty. The default is 1.0.
    lasso_weight : 3-D numpy array (spots * celltypes * 1), optional
        calculated weight for adaptive lasso. The default is None.
    hv_x : 1-D numpy array, optional
        data points served as x for calculation of probability density values. Only used for heavy-tail.
    hv_log_p : 1-D numpy array, optional
        log density values of normal distribution N(0, sigma^2) + heavy-tail. Only used for heavy-tail.
    theta_mask : 3-D numpy array (spots * celltypes * 1), optional
        mask for cell-type proportions (1: present, 0: not present). Only used for stage 2 theta optmization.
    abs_tol : float, optional
        Absolute tolerance. The default is 1e-3.
    rel_tol : float, optional
        Relative tolerance. The default is 1e-3.
    rho : float, optional
        Initial penalty parameter. The default is 1. Actually used in code is 1/rho, and turned out to fasten the converge than using rho
    mu : float, optional
        Adaptive penalty parameters. The default is 10.
    tau_incr : float, optional
        Adaptive penalty parameters. The default is 2.
    tau_decr : float, optional
        Adaptive penalty parameters. The default is 2.
    max_rho : float, optional
        Adaptive penalty parameters. The default is 1e1.
    min_rho : float, optional
        Adaptive penalty parameters. The default is 1e-1.
    maxiter : int, optional
        Maximum number of ADMM iterations. The default is 100.
    max_cg_iterations : int, optional
        Max number of CG iterations for graph Laplacian contrain per ADMM iteration. The default is 10.
    dynamic_rho : bool, optional
        if True, dynamically increasing min_rho and max_rho. The default is True.
    queue_len : int, optional
        the length of queue to record the mean of theta-theta_tilde 'RMSE' and theta-theta_hat 'RMSE'. The default is 3.
    diff_threshold : float, optional
        the threshold of 'RMSE change' for rho adjustment. If all 'RMSE change' values in queue are less than or equal the threshold, then increasing min_rho and max_rho. The default is 0.05.
    rho_incr : float, optional
        the multiplier of min_rho and max_rho increasing. The default is 2.
    rho_decr : float, optional
        the divider of min_rho and max_rho decreasing. The default is 2.
    diff_scale : float, optional
        if current theta tilde and hat 'RMSE' > diff_scale * previous 'RMSE', which means current rho value is too large and cause unexpected oscillation, then decreasing min_rho and max_rho. The default is 5.
    diff_stop : float, optional
        if average of theta tilde and hat 'RMSE' <= diff_stop, stop ADMM iteration. The default is 5e-5.
    opt_method : string, optional
        specify method used in scipy.optimize.minimize for local model fitting. The default is 'L-BFGS-B', a default method in scipy for optimization with bounds. Another choice would be 'SLSQP', a default method in scipy for optimization with constrains and bounds.
    global_optimize : bool, optional
        if is True, use basin-hopping algorithm to find the global minimum. The default is False.
    hybrid_version : bool, optional
        if True, use the hybrid_version of GLRM, i.e. in ADMM local model loss function optimization for w but adaptive lasso constrain on theta. If False, local model loss function optimization and adaptive lasso will on the same w. The default is True.
    verbose : bool, optional
        if True, print information in each ADMM loop
    use_cache : bool, optional
        if True, use the cached dict of calculated likelihood values.

    Returns
    -------
    Dict
        estimated model coefficients, including:
            theta : celltype proportions (#spots * #celltypes * 1)\n
            e_alpha : spot-specific effect (1-D array with length #spot)\n
            sigma2 : variance paramter of the lognormal distribution (float)\n
            gamma_g : gene-specific platform effect for all genes (1-D array with length #gene)\n
            theta_tilde : celltype proportions for Adaptive Lasso (#spots * #celltypes * 1)\n
            theta_hat : celltype proportions for Graph Laplacian constrain (#spots * #celltypes * 1)
    """
    
    n_celltype = data['X'].shape[0]
    n_spot = data['Y'].shape[0]
    one_theta_shape = (n_celltype, 1)
    
    start_time = time()
    optimal = False
    
    # initialize x array for calculation of heavy-tail
    if hv_x is None:
        from local_fit_numba import z_hv
        hv_x = z_hv.copy()
    
    if hv_log_p is None:
        # initialize density values of heavy-tail with initial sigma^2
        if use_cache:
            hv_log_p = generate_log_heavytail_array(z_hv, np.sqrt(round(sigma2, sigma2_digits)))
        else:
            hv_log_p = generate_log_heavytail_array(z_hv, np.sqrt(sigma2))
        
    # if theta_mask is not None, then the theta and e_alpha are already pre-processed with theta_mask
    
    # theta_tilde for adaptive lasso
    theta_tilde = theta.copy()
    # theta_hat for graph laplacian constrain
    theta_hat = theta.copy()
        
    u = np.zeros(theta.shape)
    u_tilde = np.zeros(theta.shape)
        
    res_pri = np.zeros(theta.shape)
    res_pri_tilde = np.zeros(theta.shape)
    res_dual = np.zeros(theta.shape)
    res_dual_tilde = np.zeros(theta.shape)
        
    if dynamic_rho:
        #print('CAUTION: dynamic rho trick is turned on!')
        rmse_queue = []
        pre_tilde_rmse = 0
        pre_hat_rmse = 0
    
    if verbose:
        print('\nstart ADMM iteration...')
         
    if verbose:
        print(f'{"iter" : >6} | {"res_pri_n": >10} | {"res_dual_n": >10} | {"eps_pri": >10} | {"eps_dual": >10} | {"rho": >10} | {"new_rho": >10} | {"time_opt": >8} | {"time_reg": >8} | {"time_lap": >8} | {"tilde_RMSE": >10} | {"hat_RMSE": >10}')
        
    
    # Main ADMM loop
    for t in range(maxiter):
        
        # theta update
        tmp_start = time()
        # use the previous theta as the warm start for next iteration
        if hybrid_version:
            # ADMM loss function on theta
            theta, e_alpha = update_theta(data, theta, e_alpha, gamma_g, sigma2, theta_hat-u, 1./rho, global_optimize=global_optimize, hybrid_version=hybrid_version, opt_method=opt_method, hv_x=hv_x, hv_log_p=hv_log_p, theta_mask=theta_mask, verbose=False, use_cache=use_cache)
        else:
            # ADMM loss function on w
            theta, e_alpha = update_theta(data, theta, e_alpha, gamma_g, sigma2,
                                          reparameterTheta(theta_hat-u, e_alpha), 1./rho,
                                          global_optimize=global_optimize, hybrid_version=hybrid_version, opt_method=opt_method, hv_x=hv_x, hv_log_p=hv_log_p, theta_mask=theta_mask, verbose=False, use_cache=use_cache)
        
        # theta has already been masked inside update_theta function
        
        time_local_opt = time() - tmp_start
        
        # in 2-stage implement, NO sigma2 update step in ADMM iterations
        
        # update theta_tilde
        tmp_start = time()
            
        if hybrid_version:
            # ADMM loss function on theta
            theta_tilde = adaptive_lasso(theta_hat-u_tilde, 1./rho, lambda_r, lasso_weight)
        else:
            # ADMM loss function on w
            theta_tilde = reparameterTheta(adaptive_lasso(reparameterTheta(theta_hat-u_tilde, e_alpha), 1./rho, lambda_r, lasso_weight),
                                               1.0/e_alpha)
            
        # mask theta tilde
        if not theta_mask is None:
            theta_tilde[theta_mask==0] = 0
        
        time_reg = time() - tmp_start
            
        # theta_hat update by Laplacian constrain
        tmp_start = time()
            
        # put the constrain for each element in the theta_hat
        # each iteration deal with one element but across all spots
        sys = L + 2 * rho * sparse.eye(n_spot)
        M = sparse.diags(1. / sys.diagonal())
        indices = np.ndindex(one_theta_shape)
        rhs = rho * (theta.T + u.T + theta_tilde.T + u_tilde.T)
        for i, ind in enumerate(indices):
            index = ind[::-1]
            # Use Conjugate Gradient iteration to solve Ax = b.
            # M: Preconditioner for A. The preconditioner should approximate the inverse of A.
            sol = cg(sys, rhs[index], M=M,
                     x0=theta_hat.T[index], maxiter=max_cg_iterations)[0]
            res_dual.T[index] = -rho * (sol - theta_hat.T[index])
            res_dual_tilde.T[index] = res_dual.T[index]
            theta_hat.T[index] = sol
            
        # avoid negative values
        theta_hat[theta_hat<min_theta] = min_theta
            
        # mask theta hat
        if not theta_mask is None:
            theta_hat[theta_mask==0] = 0
            
        time_graph = time() - tmp_start
            
            
        # difference between theta, theta_tilde, theta_hat
        tilde_rmse = reportRMSE(np.squeeze(theta), np.squeeze(theta_tilde))
        hat_rmse = reportRMSE(np.squeeze(theta), np.squeeze(theta_hat))
            
        # u and u_tilde update
        res_pri = theta - theta_hat
        res_pri_tilde = theta_tilde - theta_hat
        u += theta - theta_hat
        u_tilde += theta_tilde - theta_hat
    
        # calculate residual norms
        # np.append by default will combine two input and flatten the output as array
        # np.linalg.norm by default calculate 2-norm for array
        res_pri_norm = np.linalg.norm(np.append(res_pri, res_pri_tilde))
        res_dual_norm = np.linalg.norm(np.append(res_dual, res_dual_tilde))
    
        eps_pri = np.sqrt(2 * n_spot * np.prod(one_theta_shape)) * abs_tol + \
                rel_tol * max(res_pri_norm, res_dual_norm)
        eps_dual = np.sqrt(2 * n_spot * np.prod(one_theta_shape)) * abs_tol + \
                rel_tol * np.linalg.norm(rho * np.append(u, u_tilde))
                
        if dynamic_rho:
            # use theta_tilde+theta_hat RMSE or res_pri_norm
            rmse_queue.append((abs(tilde_rmse-pre_tilde_rmse) + abs(hat_rmse-pre_hat_rmse))/2.0)
            if len(rmse_queue) > queue_len:
                rmse_queue.pop(0)
                
        # check stopping condition
        if res_pri_norm <= eps_pri and res_dual_norm <= eps_dual:
            optimal = True
            if verbose:
                print(f'{t : >6} | {res_pri_norm:10.3f} | {res_dual_norm:10.3f} | {eps_pri:10.3f} | {eps_dual:10.3f} | {rho:10.2f} | {"/" : >10} | {time_local_opt:8.3f} | {time_reg:8.3f} | {time_graph:8.3f} | {tilde_rmse:10.6f} | {hat_rmse:10.6f}')
            break
            
        # if the change of theta_tilde and theta_hat "RMSE" are small, early stop iteration
        if (tilde_rmse+hat_rmse)/2 <= diff_stop:
            optimal = True
            if verbose:
                print(f'{t : >6} | {res_pri_norm:10.3f} | {res_dual_norm:10.3f} | {eps_pri:10.3f} | {eps_dual:10.3f} | {rho:10.2f} | {"/" : >10} | {time_local_opt:8.3f} | {time_reg:8.3f} | {time_graph:8.3f} | {tilde_rmse:10.6f} | {hat_rmse:10.6f}')
                print('early stop!')
            break
            
        # dynamically adjust min_rho and max_rho
        # insert a large value into the queue after each adjustment of rho to force several iterations with adjusted rho and without immediate further rho adjustment
        if dynamic_rho:
            # first check whether rho is too large and cause oscillation after first several iterations
            if (tilde_rmse+hat_rmse)/2 > diff_scale * (pre_tilde_rmse+pre_hat_rmse)/2 and t>=2:
                min_rho /= rho_decr
                max_rho /= rho_decr
                rmse_queue.append(1)
                if len(rmse_queue) > queue_len:
                    rmse_queue.pop(0)
                #if verbose:
                    #print('dynamic rho trick: decreasing rho in next ADMM iteration!')
            # then check whether need to increase rho
            elif all(num <= diff_threshold for num in rmse_queue) and t>=1:
                if min_rho < rho:
                    min_rho = rho
                min_rho *= rho_incr
                max_rho *= rho_incr
                rmse_queue.append(1)
                if len(rmse_queue) > queue_len:
                    rmse_queue.pop(0)
                #if verbose:
                    #print('dynamic rho trick: increasing rho in next ADMM iteration!')
                
            pre_tilde_rmse = tilde_rmse
            pre_hat_rmse = hat_rmse
            
        # penalty parameter update
        new_rho = rho
        if res_pri_norm > mu * res_dual_norm:
            new_rho = tau_incr * rho
        elif res_dual_norm > mu * res_pri_norm:
            new_rho = rho / tau_decr
        new_rho = np.clip(new_rho, min_rho, max_rho)
            
        if verbose:
            print(f'{t : >6} | {res_pri_norm:10.3f} | {res_dual_norm:10.3f} | {eps_pri:10.3f} | {eps_dual:10.3f} | {rho:10.2f} | {new_rho:10.2f} | {time_local_opt:8.3f} | {time_reg:8.3f} | {time_graph:8.3f} | {tilde_rmse:10.6f} | {hat_rmse:10.6f}')
            
        u *= rho / new_rho
        u_tilde *= rho / new_rho
        rho = new_rho
    
        
    # ADMM loop finished
    if verbose:
        if optimal:
            print(f"Terminated (optimal) in {t+1} iterations.")
        else:
            print("Terminated (reached max iterations).")
        
    # construct result, DO NOT change theta back to 2-D array
    # the dimension transforming is performed outside this function is needed
    result = {
            'theta': theta,
            'theta_tilde': theta_tilde,
            'theta_hat': theta_hat,
            'e_alpha': e_alpha,
            'sigma2': sigma2,
            'gamma_g': gamma_g
            }
    
    if verbose:
        print(f'One optimization by ADMM finished. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.\n')
    
    return result