#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 23:32:08 2022

@author: hill103

this script stores functions of cross-validation to find optimal values for hyper-parameters:
    1. lambda_r: for Adaptive Lasso
    2. lambda_g: for graph edge weight, will affect the Laplacian Matrix
    
in k-fold cross-validation, we random subset the GENEs, then predict the gene expression in validation fold
the performance metric is RMSE of predicted gene expressions in validation fold
"""



import networkx as nx
import numpy as np
from scipy import sparse
from math import floor
from time import time
from admm_fit import one_admm_fit
from local_fit_numba import update_theta, hv_wrapper
from utils import reportRMSE, calcRMSE
from config import print



def checkEarlyStop(x, num=3):
    '''
    check whether we can early stop hyper-parameter tuning without test on rest candidates
    
    if we observe num successive increasings, then do early stop

    Parameters
    ----------
    x : list
        the performance metric of already tested candidates.

    Returns
    -------
    either None (means continuing test next candidate) or index of the optimal candidate with smallest performance metric value.
    '''
    
    if len(x) < num+1:
        return None
    
    # find min value
    optimal_idx = x.index(min(x))
    
    if optimal_idx == len(x)-1:
        return None
    
    x_diff = np.diff(x)
    # whether there is num successive positive values after optimal_idx
    count = 0
    for i in range(optimal_idx, min(len(x_diff), optimal_idx+num)):
        if x_diff[i] >= 0:
            count += 1
    if count == num:
        return optimal_idx
    else:
        return None



def calcHVBaseModelLoss(theta, e_alpha, y, mu, gamma_g, sigma2, hv_x, hv_log_p, N, non_zero_mtx, use_cache):
    '''
    calculate the negative log-likelihood of Poisson log-normal distribution + heavy_tail for all spots
    
    Parameters
    ----------
    theta : 2-D numpy matrix
        estimated cell-type proportions (spots * celltypes).
    e_alpha : 1-D numpy array
        spot-specific effect for all spots.
    y : 2-D numpy matrix
        observed gene nUMI (spots * genes).
    mu : 2-D numpy matrix
        celltype specific marker genes (celltypes * genes).
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    sigma2 : float
        estimated variance paramter of the lognormal distribution of ln(lambda). All genes share the same variance.
    hv_x : 1-D numpy array
        data points served as x for calculation of probability density values. Only used for heavy-tail.
    hv_log_p : 1-D numpy array
        log density values of normal distribution N(0, sigma^2) + heavy-tail. Only used for heavy-tail.
    N : 1-D numpy array
        sequencing depth for all spots.
    non_zero_mtx : None or 2-D numpy matrix (spots * genes)
        If it's None, then do not filter zeros during regression. If it's a bool 2-D numpy matrix (spots * genes) as False means genes whose nUMI=0 while True means genes whose nUMI>0 in corresponding spots. The bool indicators can be calculated based on either observerd raw nUMI counts in spatial data, or CVAE transformed nUMI counts.
    use_cache : bool, optional
        if True, use the cached dict of calculated likelihood values.
        
    Returns
    -------
    float
        sum of negative log-likelihood across all spots.

    '''
    
    output = 0
    
    for i in range(theta.shape[0]):
        this_w = theta[i, :] * e_alpha[i]
        
        # filter zero genes
        y_vec = y[i, :]
        if non_zero_mtx is None:
            this_y_vec = y_vec
            this_gamma_g = gamma_g
            this_mu = mu
        else:
            non_zero_gene_ind = non_zero_mtx[i, :]
            #print(f'total {np.sum(non_zero_gene_ind)} non-zero genes ({np.sum(non_zero_gene_ind)/len(non_zero_gene_ind):.2%}) for spot {i:d}')
            this_y_vec = y_vec[non_zero_gene_ind]
            this_gamma_g = gamma_g[non_zero_gene_ind]
            this_mu = mu[:, non_zero_gene_ind]
            
        output += hv_wrapper(this_w, this_y_vec, this_mu, this_gamma_g, sigma2, hv_x, hv_log_p, N[i], use_cache)
    return output



def cv_find_lambda_r(data, mle_theta, mle_e_alpha, gamma_g, sigma2, lasso_weight, candidate_list, hybrid_version=True, opt_method='L-BFGS-B', hv_x=None, hv_log_p=None, use_admm=False, use_likelihood=True, k=5, use_cache=True, diagnosis=False):
    '''
    find optimal value for hyper-parameter lambda_r by k fold cross-validation
    
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
    mle_theta : 3-D numpy array (spots * celltypes * 1)
        estimated theta (celltype proportion) by MLE.
    mle_e_alpha : 1-D numpy array
        estimated e_alpha (spot-specific effect) by MLE.
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
    lasso_weight : 3-D numpy array (spots * celltypes * 1)
        weight of Adaptive Lasso, 1 ./ MLE theta
    candidate_list : list
        candidates for the hyper-parameter
    hybrid_version : bool, optional
        if True, use the hybrid_version of GLRM, i.e. in ADMM local model loss function optimization for w but adaptive lasso constrain on theta. If False, local model loss function optimization and adaptive lasso will on the same w. The default is True.
    opt_method : string, optional
        specify method used in scipy.optimize.minimize for local model fitting. The default is 'L-BFGS-B', a default method in scipy for optimization with bounds. Another choice would be 'SLSQP', a default method in scipy for optimization with constrains and bounds.
    hv_x : 1-D numpy array, optional
        data points served as x for calculation of probability density values. Only used for heavy-tail.
    hv_log_p : 1-D numpy array, optional
        log density values of normal distribution N(0, sigma^2) + heavy-tail. Only used for heavy-tail.
    use_admm : bool, optional
        whether use ADMM iteration or directly use Adative Lasso loss function. The default is False, i.e. NOT use ADMM in cross-validation as ADMM will cost more time.
    use_likelihood : bool, optional
        whether use negative log-likelihood as performance metric in cross-validation. The default is True, if False use RMSE of predicted gene expression.
    k : int, optional
        the number of folds in cross-validation, The default value is 5.
    use_cache : bool, optional
        if True, use the cached dict of calculated likelihood values.
    diagnosis : bool
        if True save more information to files for diagnosis CVAE and hyper-parameter selection
        
    Returns
    -------
    float
        optimal lambda_r value
    '''
    
    print('\nStart cross-validation for hyper-parameter lambda_r...')
    
    if use_admm:
        print('still use ADMM even NO Graph Laplacian constrain')
    else:
        print('directly estimate theta by Adaptive Lasso loss function as NO Graph Laplacian constrain!')
    
    start_time = time()
    
    # add 0 to candidate list
    candidate_list = [0] + candidate_list
     
    n_spot, n_gene = data['Y'].shape
    
    # Laplacian Matrix is all 0
    # transform it to a scipy sparse matrix to be consistent with Laplacian Matrix derived from graph object
    L = sparse.csr_matrix(np.zeros((n_spot, n_spot)))
    
    # random permute the genes
    gene_idx = np.arange(n_gene)
    # set seed for reproducibility
    np.random.seed(420)
    np.random.shuffle(gene_idx)
    # divided into k folds
    fold_size = floor(n_gene / float(k))
    idx_dict = dict()
    for i in range(k):
        if i < k-1:
            idx_dict[i] = list(gene_idx[i*fold_size : (i+1)*fold_size])
        else:
            idx_dict[i] = list(gene_idx[i*fold_size :])
    
    # for each candidate value, start cross-validation
    avg_rmse_list = []
    early_stop = False
    
    for t, lambda_r in enumerate(candidate_list):
        
        print(f'{t/len(candidate_list):.0%}...', end='')
        
        this_rmse_list = []
        
        for i in range(k):
            # current validation set
            valid_gene_idx = idx_dict[i]
            # combine other gene idx as training set
            train_gene_idx = []
            for j in range(k):
                if j != i:
                    train_gene_idx += idx_dict[j]
                    
            # subset training gene, note all slices even the empty slice, are shallow copies
            train_data = {}
            train_data['X'] = data['X'][:, train_gene_idx].copy()
            train_data['Y'] = data['Y'][:, train_gene_idx].copy()
            train_data['N'] = data['N'].copy()
            
            if data['non_zero_mtx'] is None:
                train_data['non_zero_mtx'] = None
                valid_non_zero_mtx = None
            else:
                train_data['non_zero_mtx'] = data['non_zero_mtx'][:, train_gene_idx].copy()
                valid_non_zero_mtx = data['non_zero_mtx'][:, valid_gene_idx]
            
            train_gamma_g = gamma_g[train_gene_idx].copy()
            
            # update theta
            if use_admm:
                this_result = one_admm_fit(train_data, L, mle_theta, mle_e_alpha, train_gamma_g, sigma2,
                                           lambda_r=lambda_r, lasso_weight=lasso_weight,
                                           hv_x=hv_x, hv_log_p=hv_log_p, theta_mask=None,
                                           opt_method=opt_method, global_optimize=False, hybrid_version=hybrid_version,
                                           verbose=False, use_cache=use_cache)
                theta = this_result['theta']
                e_alpha = this_result['e_alpha']
                
            else:
                theta, e_alpha = update_theta(train_data, mle_theta, mle_e_alpha, train_gamma_g, sigma2,
                                              lambda_r=lambda_r, lasso_weight=lasso_weight,
                                              hv_x=hv_x, hv_log_p=hv_log_p, theta_mask=None,
                                              global_optimize=False, hybrid_version=hybrid_version, opt_method=opt_method,
                                              verbose=False, use_cache=use_cache)
            
            # note fitting result theta is 3-Dimensional
            
            # evaluate in validation fold
            if use_likelihood:
                this_rmse_list.append(calcHVBaseModelLoss(np.squeeze(theta), e_alpha, data['Y'][:, valid_gene_idx], data['X'][:, valid_gene_idx], gamma_g[valid_gene_idx], sigma2, hv_x, hv_log_p, data['N'], valid_non_zero_mtx, use_cache))
                
            else:
                # use estimated theta and e_alpha predict all gene expression then subset validation fold
                pred_gene = np.zeros((n_spot, n_gene))
                for spot_idx in range(n_spot):
                    pred_gene[spot_idx, ] = data['N'][spot_idx] * e_alpha[spot_idx] * (theta[spot_idx, :, :].flatten() @ data['X']) * np.exp(gamma_g)
                # calculate RMSE for each spot, then further calculate the average across all spots
                if valid_non_zero_mtx is None:
                    this_rmse_list.append(reportRMSE(data['Y'][:, valid_gene_idx], pred_gene[:, valid_gene_idx]))
                else:
                    tmp_list = []
                    for spot_idx in range(n_spot):
                        this_true_y = data['Y'][spot_idx, valid_gene_idx]
                        this_pred_y = pred_gene[spot_idx, valid_gene_idx]
                        non_zero_gene_ind = valid_non_zero_mtx[spot_idx, ]
                        tmp_list.append(calcRMSE(this_true_y[non_zero_gene_ind], this_pred_y[non_zero_gene_ind]))
                    this_rmse_list.append(sum(tmp_list)/len(tmp_list))
            
        # calculate average across k folds
        avg_rmse_list.append(sum(this_rmse_list)/len(this_rmse_list))
        
        # check early stop
        es_flag = checkEarlyStop(avg_rmse_list)
        
        if es_flag is not None:
            early_stop = True
            break
    
    if early_stop:
        optimal_idx = es_flag
        print('early stop')
    else:
        print('100%')
        # find the optimal value with smallest average RMSE, only record the first occurence which corresponding to smaller lambda_r
        optimal_idx = avg_rmse_list.index(min(avg_rmse_list))
    
    if use_likelihood:
        print(f'find optimal lambda_r {candidate_list[optimal_idx]:.3f} with average negative log-likelihood {avg_rmse_list[optimal_idx]:.4f} by {k} fold cross-validation. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.\n')
    else:
        print(f'find optimal lambda_r {candidate_list[optimal_idx]:.3f} with average RMSE {avg_rmse_list[optimal_idx]:.4f} by {k} fold cross-validation. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.\n')
        
    # draw plot
    if diagnosis:
        
        if use_likelihood:
            y_label = 'average negative log-likelihood'
        else:
            y_label = 'average theta RMSE'
        
        from diagnosis_plots import diagnosisParamsTuning
        diagnosisParamsTuning(candidate_list, avg_rmse_list, optimal_idx, 'lambda_r', y_label)
        

    return candidate_list[optimal_idx]
    
    
    
def cv_find_lambda_g(data, G, stage1_theta, stage1_e_alpha, theta_mask, gamma_g, sigma2, candidate_list, hybrid_version=True, opt_method='L-BFGS-B', hv_x=None, hv_log_p=None, use_admm=True, use_likelihood=True, k=5, use_cache=True, diagnosis=False):
    '''
    find optimal value for hyper-parameter lambda_g by k fold cross-validation
    
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
        used for constructing Laplacian Matrix.
    stage1_theta : 3-D numpy array (spots * celltypes * 1)
        estimated theta (celltype proportion) in stage 1.
    stage1_e_alpha : 1-D numpy array
        estimated e_alpha (spot-specific effect) in stage 1.
    theta_mask : 3-D numpy array (spots * celltypes * 1)
        mask for cell-type proportions (1: present, 0: not present). Only used for stage 2 theta optmization.
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
    candidate_list : list
        candidates for the hyper-parameter
    hybrid_version : bool, optional
        if True, use the hybrid_version of GLRM, i.e. in ADMM local model loss function optimization for w but adaptive lasso constrain on theta. If False, local model loss function optimization and adaptive lasso will on the same w. The default is True.
    opt_method : string, optional
        specify method used in scipy.optimize.minimize for local model fitting. The default is 'L-BFGS-B', a default method in scipy for optimization with bounds. Another choice would be 'SLSQP', a default method in scipy for optimization with constrains and bounds.
    hv_x : 1-D numpy array, optional
        data points served as x for calculation of probability density values. Only used for heavy-tail.
    hv_log_p : 1-D numpy array, optional
        log density values of normal distribution N(0, sigma^2) + heavy-tail. Only used for heavy-tail.
    use_admm : bool, optional
        whether use ADMM iteration or directly use Adative Lasso loss function. The default is True, i.e. use ADMM in cross-validation.
    use_likelihood : bool, optional
        whether use negative log-likelihood as performance metric in cross-validation. The default is True, if False use RMSE of predicted gene expression.
    k : int, optional
        the number of folds in cross-validation, The default value is 5.
    use_cache : bool, optional
        if True, use the cached dict of calculated likelihood values.
    diagnosis : bool
        if True save more information to files for diagnosis CVAE and hyper-parameter selection
        
    Returns
    -------
    float
        optimal lambda_g value
    '''
    
    print('\nStart cross-validation for hyper-parameter lambda_g...')
    
    if use_admm:
        print('still use ADMM even NO Graph Laplacian constrain (lambda_g=0)')
    else:
        print('directly estimate theta by Adaptive Lasso loss function when NO Graph Laplacian constrain (lambda_g=0)!')
    
    
    start_time = time()
    
    # add 0 to candidate list
    candidate_list = [0] + candidate_list
    
    n_spot, n_gene = data['Y'].shape
    
    # re-initialize theta and e_alpha for only present cell-types
    # update: reuse the result from stage 1
    '''
    start_theta = np.zeros(theta_mask.shape)
    for i in range(n_spot):
        start_theta[i, theta_mask[i,:,:]==1] = 1.0/np.sum(theta_mask[i,:,:])
    
    start_e_alpha = np.full((n_spot,), 1.0)
    '''
    
    start_theta = stage1_theta.copy()
    start_e_alpha = stage1_e_alpha.copy()
    
    # random permute the genes
    gene_idx = np.arange(n_gene)
    # set seed for reproducibility
    np.random.seed(420)
    np.random.shuffle(gene_idx)
    # divided into k folds
    fold_size = floor(n_gene / float(k))
    idx_dict = dict()
    for i in range(k):
        if i < k-1:
            idx_dict[i] = list(gene_idx[i*fold_size : (i+1)*fold_size])
        else:
            idx_dict[i] = list(gene_idx[i*fold_size :])
    
    # for each candidate value, start cross-validation
    avg_rmse_list = []
    early_stop = False
    
    for t, lambda_g in enumerate(candidate_list):
        
        print(f'{t/len(candidate_list):.0%}...', end='')
        
        # update edge weight in Graph, otherwise edge will have weight 1, then calculate the Laplacian Matrix
        for _, _, e in G.edges(data=True):
            e["weight"] = lambda_g
        # calculate Laplacian, result is a SciPy sparse matrix
        L = nx.laplacian_matrix(G)
        
        this_rmse_list = []
        
        for i in range(k):
            # current validation set
            valid_gene_idx = idx_dict[i]
            # combine other gene idx as training set
            train_gene_idx = []
            for j in range(k):
                if j != i:
                    train_gene_idx += idx_dict[j]
                    
            # subset training gene, note all slices even the empty slice, are shallow copies
            train_data = {}
            train_data['X'] = data['X'][:, train_gene_idx].copy()
            train_data['Y'] = data['Y'][:, train_gene_idx].copy()
            train_data['N'] = data['N'].copy()
            
            if data['non_zero_mtx'] is None:
                train_data['non_zero_mtx'] = None
                valid_non_zero_mtx = None
            else:
                train_data['non_zero_mtx'] = data['non_zero_mtx'][:, train_gene_idx].copy()
                valid_non_zero_mtx = data['non_zero_mtx'][:, valid_gene_idx]
            
            train_gamma_g = gamma_g[train_gene_idx].copy()
            
            # update theta
            if lambda_g > 0:
                this_result = one_admm_fit(train_data, L, start_theta, start_e_alpha, train_gamma_g, sigma2,
                                           lambda_r=0, lasso_weight=None,
                                           hv_x=hv_x, hv_log_p=hv_log_p, theta_mask=theta_mask,
                                           opt_method=opt_method, global_optimize=False, hybrid_version=hybrid_version,
                                           verbose=False, use_cache=use_cache)
                theta = this_result['theta']
                e_alpha = this_result['e_alpha']
                
            elif lambda_g == 0:
                if use_admm:
                    this_result = one_admm_fit(train_data, L, start_theta, start_e_alpha, train_gamma_g, sigma2,
                                           lambda_r=0, lasso_weight=None,
                                           hv_x=hv_x, hv_log_p=hv_log_p, theta_mask=theta_mask,
                                           opt_method=opt_method, global_optimize=False, hybrid_version=hybrid_version,
                                           verbose=False, use_cache=use_cache)
                    theta = this_result['theta']
                    e_alpha = this_result['e_alpha']
                else:
                    theta, e_alpha = update_theta(train_data, start_theta, start_e_alpha, train_gamma_g, sigma2,
                                              hv_x=hv_x, hv_log_p=hv_log_p, theta_mask=theta_mask,
                                              global_optimize=False, hybrid_version=hybrid_version, opt_method=opt_method,
                                              verbose=False, use_cache=use_cache)
            else:
                raise Exception(f'lambda_g can not be negative! currently it is {lambda_g}')
            
            # note fitting result theta is 3-Dimensional
            
            # evaluate in validation fold
            if use_likelihood:
                this_rmse_list.append(calcHVBaseModelLoss(np.squeeze(theta), e_alpha, data['Y'][:, valid_gene_idx], data['X'][:, valid_gene_idx], gamma_g[valid_gene_idx], sigma2, hv_x, hv_log_p, data['N'], valid_non_zero_mtx, use_cache))
                
            else:
                # use estimated theta and e_alpha predict all gene expression then subset validation fold
                pred_gene = np.zeros((n_spot, n_gene))
                for spot_idx in range(n_spot):
                    pred_gene[spot_idx, ] = data['N'][spot_idx] * e_alpha[spot_idx] * (theta[spot_idx, :, :].flatten() @ data['X']) * np.exp(gamma_g)
                # calculate RMSE for each spot, then further calculate the average across all spots
                if valid_non_zero_mtx is None:
                    this_rmse_list.append(reportRMSE(data['Y'][:, valid_gene_idx], pred_gene[:, valid_gene_idx]))
                else:
                    tmp_list = []
                    for spot_idx in range(n_spot):
                        this_true_y = data['Y'][spot_idx, valid_gene_idx]
                        this_pred_y = pred_gene[spot_idx, valid_gene_idx]
                        non_zero_gene_ind = valid_non_zero_mtx[spot_idx, :]
                        tmp_list.append(calcRMSE(this_true_y[non_zero_gene_ind], this_pred_y[non_zero_gene_ind]))
                    this_rmse_list.append(sum(tmp_list)/len(tmp_list))
            
        # calculate average across k folds
        avg_rmse_list.append(sum(this_rmse_list)/len(this_rmse_list))
    
        # check early stop
        es_flag = checkEarlyStop(avg_rmse_list)
        
        if es_flag is not None:
            early_stop = True
            break
    
    if early_stop:
        optimal_idx = es_flag
        print('early stop')
    else:
        print('100%')
        # find the optimal value with smallest average RMSE, only record the first occurence which corresponding to smaller lambda_r
        optimal_idx = avg_rmse_list.index(min(avg_rmse_list))
    
    if use_likelihood:
        print(f'find optimal lambda_g {candidate_list[optimal_idx]:.3f} with average negative log-likelihood {avg_rmse_list[optimal_idx]:.4f} by {k} fold cross-validation. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.\n')
    else:
        print(f'find optimal lambda_g {candidate_list[optimal_idx]:.3f} with average RMSE {avg_rmse_list[optimal_idx]:.4f} by {k} fold cross-validation. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.\n')
        
    # draw plot
    if diagnosis:
        
        if use_likelihood:
            y_label = 'average negative log-likelihood'
        else:
            y_label = 'average theta RMSE'
        
        from diagnosis_plots import diagnosisParamsTuning
        diagnosisParamsTuning(candidate_list, avg_rmse_list, optimal_idx, 'lambda_g', y_label)
    
    
    return candidate_list[optimal_idx]