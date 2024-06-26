#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 22:36:00 2022

@author: hill103

this script stores functions of model fitting
currently we use a two-stage implement to fit GLRM
"""



import numpy as np
import networkx as nx
from time import time
from local_fit_numba import update_theta, update_sigma2
from admm_fit import one_admm_fit
import scipy.sparse as sparse
from utils import reparameterTheta
from hyper_parameter_optimization import cv_find_lambda_r, cv_find_lambda_g
from config import min_val, print, sigma2_digits



def fit_base_model(data, gamma_g=None, global_optimize=False, hybrid_version=True, opt_method='L-BFGS-B', verbose=False, use_cache=True, use_initial_guess=False):
    '''
    fit local or base model without any Adaptive Lasso constrain or Graph Laplacian constrain
    
    this fitting is only used for gamma_g estimation and MLE theta estimation in GLRM stage 1
    
    when fitting the base model, we will perform fitting iteratively until the sigma^2 change <= 0.001
    
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
            gene_names: a list of string of gene symbols. Only keep actually used marker gene symbols.\n
            celltype_names: a list of string of celltype names.\n
            initial_guess: initial guess of cell-type proportions of spatial spots.
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    global_optimize : bool, optional
        if is True, use basin-hopping algorithm to find the global minimum. The default is False.
    hybrid_version : bool, optional
        if True, use the hybrid_version of GLRM, i.e. in ADMM local model loss function optimization for w but adaptive lasso constrain on theta. If False, local model loss function optimization and adaptive lasso will on the same w. The default is True.
    opt_method : string, optional
        specify method used in scipy.optimize.minimize for local model fitting. The default is 'L-BFGS-B', a default method in scipy for optimization with bounds. Another choice would be 'SLSQP', a default method in scipy for optimization with constrains and bounds.
    verbose : bool, optional
        if True, print more information in program running.
    use_cache : bool, optional
        if True, use the cached dict of calculated likelihood values.
    use_initial_guess : bool, optional
        if True, use initial guess instead of uniform distribution for theta initialization.

    Returns
    -------
    tuple
        a tuple of estimated theta, e_alpha and sigma2:
            theta : celltype proportions (3-D numpy array (spots * celltypes * 1))\n
            e_alpha : spot-specific effect (1-D array with length #spot)\n
            sigma2 : variance paramter of the lognormal distribution (float)
    '''

    start_time = time()
    
    n_celltype, n_gene = data['X'].shape
    n_spot = data['Y'].shape[0]
    
    if verbose:
        print('\nGLRM model initialization...')
    
    # Initialization
    if use_initial_guess and (data['initial_guess'] is not None):
        print('HIGHLIGHT: use initial guess derived from CVAE rather than uniform distribution for theta initialization')
        # NOTE: some spots may be excluded in filtering
        assert data['initial_guess'].columns.to_list() == data['celltype_names']
        
        assert data['initial_guess'].shape[0] >= len(data['spot_names'])
        if data['initial_guess'].shape[0] == len(data['spot_names']):
            assert data['initial_guess'].index.to_list() == data['spot_names']
            theta = data['initial_guess'].values
        else:
            # use included spots only
            theta = data['initial_guess'].loc[data['spot_names']].values
            
        # note the shape is (#spots, #cell-types, 1)
        # use np.newaxis to add a new third dimension
        theta = theta[:, :, np.newaxis]
    else:
        theta = np.full((n_spot, n_celltype, 1), 1.0/n_celltype)
        
    e_alpha = np.full((n_spot,), 1.0)
    
    if use_cache:
        sigma2 = round(1.0, sigma2_digits)
    else:
        sigma2 = 1.0
   
    if gamma_g is None:
        gamma_g = np.zeros((n_gene,))
    
    # initialize x array for calculation of heavy-tail
    from local_fit_numba import z_hv, generate_log_heavytail_array
    # initialize density values of heavy-tail with initial sigma^2
    log_p_hv = generate_log_heavytail_array(z_hv, np.sqrt(sigma2))
    
    if use_cache:
        from local_fit_numba import insert_key_sigma2_wrapper
        insert_key_sigma2_wrapper(sigma2)
    
    if global_optimize:
        if verbose:
            print('CAUTION: global optimization turned on!')
    
    if verbose:
        print('calculate MLE theta and sigma^2 iteratively...')
    
    if verbose:
        print(f'{"iter" : >6} | {"time_opt": >8} | {"time_sig": >8} | {"sigma2": >6}')
    
    t = 0
    pre_sigma2 = sigma2
    
    while True:
        # update theta and e_alpha
        tmp_start_in_loop = time()
        theta, e_alpha = update_theta(data, theta, e_alpha, gamma_g, sigma2,
                                      global_optimize=global_optimize, hybrid_version=hybrid_version,  opt_method=opt_method,
                                      hv_x=z_hv, hv_log_p=log_p_hv, use_cache=use_cache)
        time_local_opt = time() - tmp_start_in_loop
        
        # update sigma^2
        tmp_start_in_loop = time()
        sigma2 = update_sigma2(data, theta, e_alpha, gamma_g, sigma2,
                                opt_method=opt_method, global_optimize=global_optimize,
                                hv_x=z_hv, use_cache=use_cache)
        # round sigma2 for rough but quick analysis
        if use_cache:
            sigma2 = round(sigma2, sigma2_digits)
        # update density values of heavy-tail with current sigma^2
        log_p_hv = generate_log_heavytail_array(z_hv, np.sqrt(sigma2))
        time_sigma2 = time() - tmp_start_in_loop
    
        if verbose:
            print(f'{t : >6} | {time_local_opt:8.3f} | {time_sigma2:8.3f} | {sigma2:6.3f}')
        
        if abs(sigma2 - pre_sigma2) <= 1e-3:
            break
        else:
            t += 1
            pre_sigma2 = sigma2
    
    if verbose:
        print(f'MLE theta and sigma^2 calculation finished. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.')
        
    return theta, e_alpha, sigma2



def estimating_gamma_g(data, hybrid_version=True, opt_method='L-BFGS-B', verbose=False):
    """
    estimate platform effect gamma_g by fit model on pseudo-bulk measurement

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
            gene_names: a list of string of gene symbols. Only keep actually used marker gene symbols.\n
            celltype_names: a list of string of celltype names.\n
            initial_guess: initial guess of cell-type proportions of spatial spots.
    hybrid_version : bool, optional
        if True, use the hybrid_version of GLRM, i.e. in ADMM local model loss function optimization for w but adaptive lasso constrain on theta. If False, local model loss function optimization and adaptive lasso will on the same w. The default is True.
    opt_method : string, optional
        specify method used in scipy.optimize.minimize for local model fitting. The default is 'L-BFGS-B', a default method in scipy for optimization with bounds. Another choice would be 'SLSQP', a default method in scipy for optimization with constrains and bounds.
    verbose : bool, optional
        if True, print more information in program running
    
    Returns
    -------
    1-D numpy array (#genes)
        estimated gamma_g
    """
    
    start_time = time()
    
    if verbose:
        print(f'estimate platform effect gammg_g by {opt_method} and basinhopping')
    
    # combine all spots into one spots then build a new data dict
    tmp_data = {'X': data['X'],
                'Y': np.sum(data['Y'], axis=0, keepdims=True)}
    # sequencing depth is sum across all spots and genes
    tmp_data['N'] = np.array([np.sum(data['N'])])
    # genes with nUMI>0
    if data['non_zero_mtx'] is None:
        tmp_data['non_zero_mtx'] = None
    else:
        tmp_data['non_zero_mtx'] = np.sum(data['non_zero_mtx'], axis=0, keepdims=True) > 0
    
    # fit base model
    theta, e_alpha, _ = fit_base_model(tmp_data, global_optimize=True, hybrid_version=hybrid_version, opt_method=opt_method, verbose=verbose, use_cache=False, use_initial_guess=False)
    
    
    assert(len(e_alpha)==1)
    # calculate gamma_g
    # note theta is 3-Dimensional
    w = np.squeeze(theta) * e_alpha[0]
    
    gamma_g = np.log(np.squeeze(tmp_data['Y'])) - np.log(np.sum(tmp_data['Y']) * w@tmp_data['X'] + min_val)
    
    print(f'gamma_g estimation finished. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.')
    
    return gamma_g



def fit_model_two_stage(data, G, gamma_g=None, lambda_r=None, weight_threshold=1e-3, lambda_g=None, global_optimize=False, hybrid_version=True, opt_method='L-BFGS-B', verbose=False, use_cache=True, diagnosis=False):
    """
    fit Graph Laplacian Regularized Stratified Model (GLRM) in a two-stage way
    
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
    G : built graph object from networks module
        used for constructing Laplacian Matrix
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    lambda_r : float
        strength for Adaptive Lasso penalty. If None, use cross-validation to determine optimal lambda_r
    weight_threshold : float, optional
        threshold for Adaptive Lasso weight. Theta values smaller than threshold value will be re-set. The default is 1e-3.
    lambda_g : float
        edge weight for graph, and will affect the strength of Graph Laplacian constrain. If None, use cross-validation to determine optimal graph_weight.
    global_optimize : bool, optional
        if is True, use basin-hopping algorithm to find the global minimum. The default is False.
    hybrid_version : bool, optional
        if True, use the hybrid_version of GLRM, i.e. in ADMM local model loss function optimization for w but adaptive lasso constrain on theta. If False, local model loss function optimization and adaptive lasso will on the same w. The default is True.
    opt_method : string, optional
        specify method used in scipy.optimize.minimize for local model fitting. The default is '', a default method in scipy for optimization with bounds. Another choice would be 'SLSQP', a default method in scipy for optimization with constrains and bounds.
    verbose : bool, optional
        if True, print more information in program running.
    use_cache : bool, optional
        if True, use the cached dict of calculated likelihood values.
    diagnosis : bool
        if True save more information to files for diagnosis CVAE and hyper-parameter selection

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
    
    print('\n\nStart GLRM fitting...\n')
    
    start_time = time()
    
    n_celltype, n_gene = data['X'].shape
    n_spot = data['Y'].shape[0]
    
    
    print('first estimate MLE theta and corresponding e^alpha and sigma^2...')
    
    tmp_start_time = time()
    
    theta, e_alpha, sigma2 = fit_base_model(data, gamma_g=gamma_g, global_optimize=global_optimize, hybrid_version=hybrid_version, opt_method=opt_method, verbose=True, use_cache=use_cache, use_initial_guess=True)
    
    print(f'MLE theta estimation finished. Elapsed time: {(time()-tmp_start_time)/60.0:.2f} minutes.')
    
    # initialize x array for calculation of heavy-tail
    from local_fit_numba import z_hv, generate_log_heavytail_array
    # initialize density values of heavy-tail with initial sigma^2
    log_p_hv = generate_log_heavytail_array(z_hv, np.sqrt(sigma2))
    
    # calculate the weight for adaptive lasso based on MLE theta
    print('\ncalculate weights of Adaptive Lasso...')
    
    # optimize cache dict
    # sigma2 already round to X digits
    if use_cache:
        from local_fit_numba import purge_keys
        purge_keys(sigma2)
    
    tmp_theta = theta.copy()
    if hybrid_version:
        # adaptive lasso constrain on theta
        tmp_theta[tmp_theta<weight_threshold] = weight_threshold
        lasso_weight = 1.0 / tmp_theta
    else:
        # adaptive lasso constrain on w=theta*e^alpha
        tmp_w = reparameterTheta(tmp_theta, e_alpha)
        tmp_w[tmp_w<weight_threshold] = weight_threshold
        lasso_weight = 1.0 / tmp_w
    
    del tmp_theta

    # stage 1
    tmp_start_time = time()
    
    print('\nStage 1: variable selection using Adaptive Lasso starts with the MLE theta and e^alpha, using already estimated sigma^2 and gamma_g...')
    
    print(f'specified hyper-parameter for Adaptive Lasso is: {lambda_r}')
    
    if isinstance(lambda_r, list):
        print(f'hyper-parameter for Adaptive Lasso: use cross-validation to find the optimal value from {len(lambda_r)} candidates...')
        lambda_r = cv_find_lambda_r(data, theta, e_alpha, gamma_g, sigma2, lasso_weight, lambda_r, hybrid_version=hybrid_version, opt_method=opt_method, hv_x=z_hv, hv_log_p=log_p_hv, use_admm=False, use_likelihood=True, use_cache=use_cache, diagnosis=diagnosis) 
        
    # stage 1 with determined lambda_r, but use ADMM framework
    # Laplacian Matrix is all 0 in stage 1
    # transform it to a scipy sparse matrix to be consistent with Laplacian Matrix derived from graph object
    L = sparse.csr_matrix(np.zeros((n_spot, n_spot)))
    stage1_result = one_admm_fit(data, L, theta, e_alpha, gamma_g, sigma2, lambda_r=lambda_r, lasso_weight=lasso_weight,
                                 hv_x=z_hv, hv_log_p=log_p_hv, theta_mask=None,
                                 opt_method=opt_method, global_optimize=global_optimize, hybrid_version=hybrid_version,
                                 verbose=verbose, use_cache=use_cache)
    print(f'Stage 1 variable selection finished. Elapsed time: {(time()-tmp_start_time)/60.0:.2f} minutes.')
        
    
    theta = stage1_result['theta']
    # binary theta by threshold to get a mask (1: present, 0: not present)
    theta_mask = np.zeros(theta.shape, dtype='int')
    theta_mask[theta >= weight_threshold] = 1
    
    
    # stage 2
    tmp_start_time = time()
    
    print('\nStage 2: final theta estimation with Graph Laplacian Constrain using already estimated sigma^2 and gamma_g')
    
    # re-initialize theta and e_alpha for only present cell-types
    # update: reuse the result from stage 1
    '''
    theta = np.zeros(theta.shape)
    for i in range(n_spot):
        theta[i, theta_mask[i,:,:]==1] = 1.0/np.sum(theta_mask[i,:,:])
    
    e_alpha = np.full((n_spot,), 1.0)
    '''
    
    # note theta shape is (#spots * #celltypes * 1)
    theta[theta < weight_threshold] = 0
    tmp_row_sums = theta.sum(axis=1)  # Sum along the second axis, got matrix (#spots * 1)
    theta = theta / tmp_row_sums[:, np.newaxis] # broadcasting row sum to (#spots * 1 * 1) then performs element-wise division
    assert theta.shape == theta_mask.shape
    for i in range(theta.shape[0]):
        assert abs(np.sum(theta[i,:,:]) - 1) < 1e-8
    
    # reuse e_alpha, as it related to spot not cell-types
    e_alpha = stage1_result['e_alpha']
    
    print('HIGHLIGHT: reuse estimated theta and e^alpha in stage 1 as initial value')
    
    
    if data['A'] is None:
        
        print('No Adjacency Matrix of spots specified! Ignore Graph Laplacian Constrain in stage 2')
        # Laplacian Matrix is all 0
        # transform it to a scipy sparse matrix to be consistent with Laplacian Matrix derived from graph object
        L = sparse.csr_matrix(np.zeros((n_spot, n_spot)))
        # stage 2 ADMM iterations
        result = one_admm_fit(data, L, theta, e_alpha, gamma_g, sigma2,
                          lambda_r=0, lasso_weight=None,
                          hv_x=z_hv, hv_log_p=log_p_hv, theta_mask=theta_mask,
                          opt_method=opt_method, global_optimize=global_optimize,
                          hybrid_version=hybrid_version, verbose=verbose, use_cache=use_cache)
    
    else:
        
        # considering Laplacian Constrain
        print(f'specified hyper-parameter for Graph Laplacian Constrain is: {lambda_g}')
    
        if isinstance(lambda_g, list):
            print(f'hyper-parameter for Graph Laplacian Constrain: use cross-validation to find the optimal value from {len(lambda_g)} candidates...')
            lambda_g = cv_find_lambda_g(data, G, theta, e_alpha, theta_mask, gamma_g, sigma2, lambda_g, hybrid_version=hybrid_version, opt_method=opt_method, hv_x=z_hv, hv_log_p=log_p_hv, use_admm=True, use_likelihood=True, use_cache=use_cache, diagnosis=diagnosis)
    
        # update edge weight in Graph, otherwise edge will have weight 1, then calculate the Laplacian Matrix
        for _, _, e in G.edges(data=True):
            e["weight"] = lambda_g
        # calculate Laplacian, result is a SciPy sparse matrix
        L = nx.laplacian_matrix(G)
        
        
        # stage 2 ADMM iterations
        result = one_admm_fit(data, L, theta, e_alpha, gamma_g, sigma2,
                              lambda_r=0, lasso_weight=None,
                              hv_x=z_hv, hv_log_p=log_p_hv, theta_mask=theta_mask,
                              opt_method=opt_method, global_optimize=global_optimize,
                              hybrid_version=hybrid_version, verbose=verbose, use_cache=use_cache)
    
    print(f'\nstage 2 finished. Elapsed time: {(time()-tmp_start_time)/60.0:.2f} minutes.\n')
    
    print(f'GLRM fitting finished. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.\n')
    
    return result