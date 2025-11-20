#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 22:36:00 2022

@author: hill103

this script stores functions of model fitting
currently we use a two-stage implement to fit GLRM
"""



import numpy as np
from time import time
from local_fit_numba import update_theta, update_sigma2, fit_base_model_plus_laplacian, optimize_one_theta
from admm_fit import one_admm_fit
import scipy.sparse as sparse
from utils import reparameterTheta
from hyper_parameter_optimization import cv_find_lambda_r, cv_find_lambda_g, BIC_find_lambda_r_one_spot
from config import min_val, print



def fit_base_model(data, gamma_g=None, global_optimize=False, hybrid_version=True, opt_method='L-BFGS-B', verbose=False, use_initial_guess=False):
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
        print('[HIGHLIGHT] use initial guess derived from CVAE rather than uniform distribution for theta initialization')
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
    
    sigma2 = 1.0
   
    if gamma_g is None:
        gamma_g = np.zeros((n_gene,))
    
    # initialize x array for calculation of heavy-tail
    from local_fit_numba import z_hv, generate_log_heavytail_array
    # initialize density values of heavy-tail with initial sigma^2
    log_p_hv = generate_log_heavytail_array(z_hv, np.sqrt(sigma2))
    
    if global_optimize:
        if verbose:
            print('[CAUTION] global optimization turned on!')
    
    if verbose:
        print('calculate MLE theta and sigma^2 iteratively...')
    
    if verbose:
        print(f'{"iter" : >6} | {"time_opt": >8} | {"time_sig": >8} | {"sigma2": >6}')
    
    t = 0
    pre_sigma2 = sigma2
    
    while True:
        # update theta and e_alpha
        tmp_start_in_loop = time()
        theta, e_alpha = update_theta(data, theta, e_alpha, gamma_g.copy(), sigma2,
                                      global_optimize=global_optimize, hybrid_version=hybrid_version,  opt_method=opt_method,
                                      hv_x=z_hv, hv_log_p=log_p_hv)
        time_local_opt = time() - tmp_start_in_loop
        
        # update sigma^2
        tmp_start_in_loop = time()
        sigma2 = update_sigma2(data, theta, e_alpha, gamma_g.copy(), sigma2,
                                opt_method=opt_method, global_optimize=global_optimize,
                                hv_x=z_hv)
       
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
    tmp_data = {'X': data['X'].copy(),
                'Y': np.sum(data['Y'], axis=0, keepdims=True)}
    # sequencing depth is sum across all spots and genes
    tmp_data['N'] = np.array([np.sum(data['N'])])
    # genes with nUMI>0
    if data['non_zero_mtx'] is None:
        tmp_data['non_zero_mtx'] = None
    else:
        tmp_data['non_zero_mtx'] = np.sum(data['non_zero_mtx'], axis=0, keepdims=True) > 0
        
    tmp_data['spot_names'] = ['comb_spot']  # only one spot
    
    # fit base model
    theta, e_alpha, _ = fit_base_model(tmp_data, global_optimize=True, hybrid_version=hybrid_version, opt_method=opt_method, verbose=verbose, use_initial_guess=False)
    
    
    assert(len(e_alpha)==1)
    # calculate gamma_g
    # note theta is 3-Dimensional
    w = np.squeeze(theta) * e_alpha[0]
    
    gamma_g = np.log(np.squeeze(tmp_data['Y'])) - np.log(np.sum(tmp_data['Y']) * w@tmp_data['X'] + min_val)
    
    print(f'gamma_g estimation finished. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.')
    
    return gamma_g



def fit_stage1_spotwise_lambda_r(data, warm_start_theta, warm_start_e_alpha, gamma_g, sigma2, lambda_r, lasso_weight, global_optimize=False, hybrid_version=True, opt_method='L-BFGS-B', hv_x=None, hv_log_p=None, verbose=False, diagnosis=False):
    '''
    update theta (celltype proportion) and e_alpha (spot-specific effect) given sigma2 (variance paramter of the log-normal distribution) and gamma_g (gene-specific platform effect) by MLE
    
    UPDATE: here we assume the hyperparameter lambda_r for Adaptive Lasso is spot-wise, i.e. for each spot, we search for the optimal lambda_r, then fit the model
    
    we assume 
    
        ln(lambda) = alpha + gamma_g + ln(sum(theta*mu_X)) + epsilon
    
        subject to sum(theta)=1, theta>=0
    
    mu_X is marker genes from data['X']
    
    then the mean parameter of the lognormal distribution of ln(lambda) is alpha + gamma_g + ln(sum(theta*mu_X))
    
    we did re-parametrization w = e^alpha * theta, then
    
        ln(lambda) = gamma_g + ln(sum([e^alpha*theta]*mu_X)) + epsilon
    
        subject to w>=0, it will imply sum(theta)=1 and theta>=0
    
    the steps to update theta and e_alpha:
        1. dimension change of theta from 3-D (spots * celltypes * 1) to 1-D (celltypes), and do re-parametrization to get w
        2. solve w for each spot in parallel
        3. extract updated theta and e_alpha from w, and change the dimension of updated theta back

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
    lambda_r : float or 1-D numpy array
        parameter for the strength of Adaptive Lasso loss to shrink theta
    lasso_weight : 3-D numpy array (spots * celltypes * 1)
        weight of Adaptive Lasso, 1 ./ theta
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
    verbose : bool, optional
        if True, print more information.
    diagnosis : bool
        if True save more information to files for diagnosis CVAE and hyper-parameter selection
        
    Returns
    -------
    theta_results : 3-D numpy array (spots * celltypes * 1)
        updated theta (celltype proportion).
    e_alpha_results : 1-D numpy array
        updated e_alpha (spot-specific effect).
    '''
    
    assert lasso_weight is not None
    
    n_celltype = data["X"].shape[0]
    n_spot = data["Y"].shape[0]
    
    # NOTE input warm_start_theta, warm_start_e_alpha will NOT be changed inside the function
    
    # prepare parameter tuples for parallel computing
    results = []
    if isinstance(lambda_r, list):
        lambda_r_list = []
    
    prev_percent = -1
    
    for i in range(n_spot):
        
        progress = int(i / n_spot * 100)   # 0–100%
        # print only when hitting multiples of 10
        if progress // 10 > prev_percent:
            prev_percent = progress // 10
            print(f"{progress}%...", end='')
        
        this_spot_name = data["spot_names"][i]
        this_warm_start_theta = warm_start_theta[i, :, :].copy().flatten()
        this_warm_start_e_alpha = warm_start_e_alpha[i]

        y_vec = data["Y"][i, :].copy()
        mu = data["X"].copy()

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
        
        # first determine whether need to tune hyperparameter lambda_r
        if isinstance(lambda_r, list):
            if verbose:
                print(f'{i}th spot {this_spot_name} hyper-parameter for Adaptive Lasso: use BIC criteria to find the optimal value from {len(lambda_r)} candidates...')
            # NOTE returned theta NOT w
            best_lambda_r, tmp_theta = BIC_find_lambda_r_one_spot(this_mu.copy(), this_y_vec.copy(), N, this_spot_name, this_warm_start_theta.copy(), this_warm_start_e_alpha, this_gamma_g, sigma2, lasso_weight_vec.copy(), lambda_r, hybrid_version=hybrid_version, opt_method=opt_method, hv_x=hv_x, hv_log_p=hv_log_p, verbose=False)
            
            lambda_r_list.append({'spot': this_spot_name, 'optimal_lambda_r': best_lambda_r})
        
        else:
            # directly run optimization with specified lambda_r
            tmp_w = optimize_one_theta(this_mu.copy(), this_y_vec.copy(), N, this_warm_start_theta.copy(), this_warm_start_e_alpha, this_gamma_g, sigma2, this_spot_name, nu_vec=None, rho=None, lambda_r=lambda_r, lasso_weight_vec=lasso_weight_vec.copy(), lambda_l2=None, global_optimize=global_optimize, hybrid_version=hybrid_version, opt_method=opt_method, hv_x=hv_x, hv_log_p=hv_log_p, this_theta_mask=None, skip_opt=True, verbose=False)
            
            # get theta and e_alpha
            tmp_theta = tmp_w / np.sum(tmp_w)
            
        # get non-zero cell types then refit; NOTE we set a min_theta in optimization to avoid theta as 0
        # and before return result, we already set them back to 0
        # binary theta by threshold to get a mask (1: present, 0: not present)
        tmp_theta_mask = np.zeros(tmp_theta.shape, dtype='int')
        tmp_theta_mask[tmp_theta > 0] = 1
        
        this_w = optimize_one_theta(this_mu.copy(), this_y_vec.copy(), N, this_warm_start_theta.copy(), this_warm_start_e_alpha, this_gamma_g, sigma2, this_spot_name, nu_vec=None, rho=None, lambda_r=None, lasso_weight_vec=None, lambda_l2=None, global_optimize=global_optimize, hybrid_version=hybrid_version, opt_method=opt_method, hv_x=hv_x, hv_log_p=hv_log_p, this_theta_mask=tmp_theta_mask, skip_opt=True, verbose=False)
            
        results.append(this_w.copy())
       
    
    print('100%\n')
    
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
    
    if isinstance(lambda_r, list):
        if diagnosis:
            # output optimal lambda_r
            import pandas as pd
            import os
            from config import diagnosis_path
            
            # need to create subfolders first, otherwise got FileNotFoundError
            os.makedirs(os.path.join(diagnosis_path, 'GLRM_params_tuning'), exist_ok=True)
            
            tmp_df = pd.DataFrame(lambda_r_list)
            tmp_df.to_csv(os.path.join(diagnosis_path, 'GLRM_params_tuning', 'optimal_lambda_r.csv.gz'), index=False, compression='gzip')
            
            # save optimal lambda_r histogram
            from diagnosis_plots import diagnosisParamsSpotwiseLambdarTuning
            diagnosisParamsSpotwiseLambdarTuning(tmp_df['optimal_lambda_r'].to_list())
    
    return theta_results, e_alpha_results



def fit_model_two_stage(data, gamma_g=None, lambda_r=None, lambda_g=None, global_optimize=False, hybrid_version=True, opt_method='L-BFGS-B', verbose=False, diagnosis=False, use_admm=False):
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
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    lambda_r : float
        strength for Adaptive Lasso penalty. If None, use cross-validation to determine optimal lambda_r
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
    diagnosis : bool, optional
        if True save more information to files for diagnosis CVAE and hyper-parameter selection.
    use_admm : bool, optinal
        if True, use ADMM framework in optimization, default is False.

    Returns
    -------
    Dict
        estimated model coefficients, including:
            theta : celltype proportions (#spots * #celltypes * 1)\n
            e_alpha : spot-specific effect (1-D array with length #spot)\n
            sigma2 : variance paramter of the lognormal distribution (float)\n
            gamma_g : gene-specific platform effect for all genes (1-D array with length #gene)
    """
    
    print('\n\nStart GLRM fitting...\n')
    
    start_time = time()
    
    n_celltype, n_gene = data['X'].shape
    n_spot = data['Y'].shape[0]
    
    
    print('first estimate MLE theta and corresponding e^alpha and sigma^2...')
    
    tmp_start_time = time()
    
    theta, e_alpha, sigma2 = fit_base_model(data, gamma_g=gamma_g, global_optimize=global_optimize, hybrid_version=hybrid_version, opt_method=opt_method, verbose=True, use_initial_guess=True)
    
    print(f'MLE theta estimation finished. Elapsed time: {(time()-tmp_start_time)/60.0:.2f} minutes.')
    
    # initialize x array for calculation of heavy-tail
    from local_fit_numba import z_hv, generate_log_heavytail_array
    # initialize density values of heavy-tail with initial sigma^2
    log_p_hv = generate_log_heavytail_array(z_hv, np.sqrt(sigma2))
    
    # calculate the weight for adaptive lasso based on MLE theta
    print('\ncalculate weights of Adaptive Lasso...')
    weight_threshold = 1e-3
    print(f'clip max weight to {1/weight_threshold:.0f} to avoid huge weights derived from tiny MLE theta')
    
    tmp_theta = theta.copy()
    # also avoid to divided by 0
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
    
    spotwise_lambda_r = True
    if spotwise_lambda_r:
        print('\n[HIGHLIGHT] use spot-wise hyper-parameter lambda_r instead of an overall lambda_r value for all spots\n')
    
    print(f'specified hyper-parameter for Adaptive Lasso is: {lambda_r}\n')
    
    
    if lambda_r == 0:
        print('[WARNING] User specified lambda_r as 0, meaning disable Adaptive Lasso. Stage 1 will be skiped\n')
        # directly copy MLE estimates to Stage 1 results
        stage1_result = {}
        stage1_result['theta'] = theta.copy()
        stage1_result['e_alpha'] = e_alpha.copy()
    
    elif isinstance(lambda_r, list):
        # we need tune hpyerparameter lambda_r
        if spotwise_lambda_r:
            print('Start tuning spot-wise hyper-parameter for Adaptive Lasso for each spot separately: use BIC to find the optimal value from {len(lambda_r)} candidates...')
            stage1_result = {}
            stage1_result['theta'], stage1_result['e_alpha'] = fit_stage1_spotwise_lambda_r(data, theta.copy(), e_alpha.copy(), gamma_g.copy(), sigma2, lambda_r, lasso_weight.copy(), global_optimize=False, hybrid_version=hybrid_version, opt_method=opt_method, hv_x=z_hv, hv_log_p=log_p_hv, verbose=False, diagnosis=diagnosis)
        
        else:
        
            print(f'hyper-parameter for Adaptive Lasso: use cross-validation to find the an overall optimal value for all spots from {len(lambda_r)} candidates...')
            best_lambda_r = cv_find_lambda_r(data, theta.copy(), e_alpha.copy(), gamma_g.copy(), sigma2, lasso_weight, lambda_r, hybrid_version=hybrid_version, opt_method=opt_method, hv_x=z_hv, hv_log_p=log_p_hv, use_admm=False, use_likelihood=True, diagnosis=diagnosis)
    
            print(f'\nStart optimization with found optimal lambda_r: {best_lambda_r}\n')
    
            if use_admm:
                # stage 1 with determined lambda_r, but use ADMM framework
                # Laplacian Matrix is all 0 in stage 1
                # transform it to a scipy sparse matrix to be consistent with Laplacian Matrix derived from graph object
                L = sparse.csr_matrix(np.zeros((n_spot, n_spot)))
                stage1_result = one_admm_fit(data, L, theta.copy(), e_alpha.copy(), gamma_g.copy(),
                                sigma2, lambda_r=best_lambda_r, lasso_weight=lasso_weight.copy(),
                                hv_x=z_hv, hv_log_p=log_p_hv, theta_mask=None,
                                opt_method=opt_method, global_optimize=global_optimize,
                                hybrid_version=hybrid_version, verbose=verbose)
            else:
                # stage 1 with determined lambda_r, use base model loss + Adaptive Lasso loss
                stage1_result = {}
                stage1_result['theta'], stage1_result['e_alpha'] = update_theta(data, theta.copy(),
                                    e_alpha.copy(), gamma_g.copy(), sigma2,
                                    lambda_r=best_lambda_r, lasso_weight=lasso_weight.copy(),
                                    hv_x=z_hv, hv_log_p=log_p_hv, theta_mask=None,
                                    global_optimize=global_optimize, hybrid_version=hybrid_version,
                                    opt_method=opt_method, verbose=verbose)
    
    else:
        print(f'\nStart optimization with user specified optimal lambda_r: {lambda_r}\n')
        
        if spotwise_lambda_r:
            stage1_result = {}
            stage1_result['theta'], stage1_result['e_alpha'] = fit_stage1_spotwise_lambda_r(data, theta.copy(), e_alpha.copy(), gamma_g.copy(), sigma2, lambda_r, lasso_weight.copy(), global_optimize=False, hybrid_version=hybrid_version, opt_method=opt_method, hv_x=z_hv, hv_log_p=log_p_hv, verbose=False, diagnosis=diagnosis)
        
        else:
            
            if use_admm:
                # stage 1 with determined lambda_r, but use ADMM framework
                # Laplacian Matrix is all 0 in stage 1
                # transform it to a scipy sparse matrix to be consistent with Laplacian Matrix derived from graph object
                L = sparse.csr_matrix(np.zeros((n_spot, n_spot)))
                stage1_result = one_admm_fit(data, L, theta.copy(), e_alpha.copy(), gamma_g.copy(),
                                sigma2, lambda_r=lambda_r, lasso_weight=lasso_weight.copy(),
                                hv_x=z_hv, hv_log_p=log_p_hv, theta_mask=None,
                                opt_method=opt_method, global_optimize=global_optimize,
                                hybrid_version=hybrid_version, verbose=verbose)
            else:
                # stage 1 with determined lambda_r, use base model loss + Adaptive Lasso loss
                # compared with spotwise-related function, just without re-fit step
                stage1_result = {}
                stage1_result['theta'], stage1_result['e_alpha'] = update_theta(data, theta.copy(),
                                    e_alpha.copy(), gamma_g.copy(), sigma2,
                                    lambda_r=lambda_r, lasso_weight=lasso_weight.copy(),
                                    hv_x=z_hv, hv_log_p=log_p_hv, theta_mask=None,
                                    global_optimize=global_optimize, hybrid_version=hybrid_version,
                                    opt_method=opt_method, verbose=verbose)


    print(f'Stage 1 variable selection finished. Elapsed time: {(time()-tmp_start_time)/60.0:.2f} minutes.')
    
    theta = stage1_result['theta']
    # binary theta by threshold to get a mask (1: present, 0: not present)
    theta_mask = np.zeros(theta.shape, dtype='int')
    # NOTE we already re-set w values at boundary to 0; so small theta values will be directly 0
    theta_mask[theta > 0] = 1
    
    
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
    
    # NO need to do it
    '''
    # set small values to 0
    # note theta shape is (#spots * #celltypes * 1)
    theta[theta < weight_threshold] = 0
    tmp_row_sums = theta.sum(axis=1)  # Sum along the second axis, got matrix (#spots * 1)
    theta = theta / tmp_row_sums[:, np.newaxis] # broadcasting row sum to (#spots * 1 * 1) then performs element-wise division
    assert theta.shape == theta_mask.shape
    for i in range(theta.shape[0]):
        assert abs(np.sum(theta[i,:,:]) - 1) < 1e-8
    '''
    
    # reuse e_alpha, as it related to spot not cell-types
    e_alpha = stage1_result['e_alpha']
    
    print('[HIGHLIGHT] reuse estimated theta and e^alpha in stage 1 as initial value')
    
    print(f'specified hyper-parameter for Graph Laplacian Constrain is: {lambda_g}\n')
    
    
    if data['A'] is None:
        
        print('[WARNING] No Adjacency Matrix of spots specified! Skip stage 2')
        result = {}
        result['theta'] = theta.copy()
        result['e_alpha'] = e_alpha.copy()
    
    elif lambda_g == 0:
        
        print('[WARNING] User specified lambda_g as 0, meaning disable graph Laplacian. Stage 2 will be skiped')
        result = {}
        result['theta'] = theta.copy()
        result['e_alpha'] = e_alpha.copy()
    
    else:
        
        # considering Laplacian Constrain
        # manually calculate Laplacian matrix L = D - W
        def calcLaplacian(W):
            # consistent with nx.laplacian_matrix(G), which return a SciPy sparse matrix
            deg = W.sum(axis=1)  # degree (row-sum of weights)
            D   = np.diag(deg)
            return D - W  # dense Laplacian
        
        # transform it to a scipy sparse matrix to be consistent with Laplacian Matrix derived from graph object
        L = sparse.csr_matrix(calcLaplacian(data['A']))
    
        if isinstance(lambda_g, list):
            print(f'hyper-parameter for Graph Laplacian Constrain: use cross-validation to find the optimal value from {len(lambda_g)} candidates...')
            # NOTE to use deep copy of Laplacian matrix to avoid modify it unexpectedly
            # UPDATE: NOT use ADMM for hyperparameter tuning
            best_lambda_g = cv_find_lambda_g(data, L.copy(), theta.copy(), e_alpha.copy(), theta_mask, gamma_g.copy(), sigma2, lambda_g, hybrid_version=hybrid_version, opt_method=opt_method, hv_x=z_hv, hv_log_p=log_p_hv, use_admm=False, use_likelihood=True, diagnosis=diagnosis)
        
            print(f'\nStart optimization with found optimal lambda_g: {best_lambda_g}\n')
        
        else:
            best_lambda_g = lambda_g
            print(f'\nStart optimization with user specified optimal lambda_g: {best_lambda_g}\n')
        
        # UPDATE: multiple the hyperparameter with Laplacian matrix get 𝜆L
        lambda_gL = L * best_lambda_g
        
        if use_admm:
            # stage 2 ADMM iterations
            result = one_admm_fit(data, lambda_gL, theta.copy(), e_alpha.copy(), gamma_g.copy(), sigma2,
                                  lambda_r=0, lasso_weight=None,
                                  hv_x=z_hv, hv_log_p=log_p_hv, theta_mask=theta_mask,
                                  opt_method=opt_method, global_optimize=global_optimize,
                                  hybrid_version=hybrid_version, verbose=verbose)
        else:
            result = fit_base_model_plus_laplacian(data, lambda_gL, theta.copy(), e_alpha.copy(),
                                               gamma_g.copy(), sigma2,
                                               lambda_r=None, lasso_weight=None,
                                               hv_x=z_hv, hv_log_p=log_p_hv, theta_mask=theta_mask,
                                               opt_method=opt_method, global_optimize=global_optimize,
                                               hybrid_version=hybrid_version, verbose=verbose)
    
    print(f'\nstage 2 finished. Elapsed time: {(time()-tmp_start_time)/60.0:.2f} minutes.\n')
    
    print(f'GLRM fitting finished. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.\n')
    
    return result