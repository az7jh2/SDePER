#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 23:32:08 2022

@author: hill103

this script stores functions of cross-validation to find optimal values for hyper-parameters:
    1. lambda_r: for Adaptive Lasso
    2. lambda_g: for graph edge weight, will affect the Laplacian Matrix
    
in k-fold cross-validation, we random subset the GENEs, then predict the gene expression in validation fold
the performance metric is RMSE or negative log-likelihood of predicted gene expressions in validation fold

UPDATE: we add BIC for lambda_r selection
"""



import numpy as np
from scipy import sparse
from math import floor
from time import time
from admm_fit import one_admm_fit
from local_fit_numba import update_theta, hv_wrapper, fit_base_model_plus_laplacian, optimize_one_theta
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



def calcHVBaseModelLoss(theta, e_alpha, y, mu, gamma_g, sigma2, hv_x, hv_log_p, N, non_zero_mtx):
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
            
        output += hv_wrapper(this_w, this_y_vec, this_mu, this_gamma_g, sigma2, hv_x, hv_log_p, N[i])
    return output



def cv_find_lambda_r(data, mle_theta, mle_e_alpha, gamma_g, sigma2, lasso_weight, candidate_list, hybrid_version=True, opt_method='L-BFGS-B', hv_x=None, hv_log_p=None, use_admm=False, use_likelihood=True, k=5, diagnosis=False, verbose=False):
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
    diagnosis : bool
        if True save more information to files for diagnosis CVAE and hyper-parameter selection.
    verbose : bool, optional
        if True, print more information.
        
    Returns
    -------
    float
        optimal lambda_r value
    '''
    
    if verbose:
        print('\nStart cross-validation for hyper-parameter lambda_r...')
    
    if verbose:
        if use_admm:
            print('still use ADMM even NO Graph Laplacian constrain')
        else:
            print('directly estimate theta by Adaptive Lasso loss function as NO Graph Laplacian constrain!')
    
    start_time = time()
    
    # add 0 to candidate list; original list NOT affected
    candidate_list = [0] + candidate_list[:]
     
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
        
        #print(f'{t/len(candidate_list):.0%}...', end='')
        
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
            train_data['spot_names'] = data['spot_names'][:]  # copy list for safe
            
            if data['non_zero_mtx'] is None:
                train_data['non_zero_mtx'] = None
                valid_non_zero_mtx = None
            else:
                train_data['non_zero_mtx'] = data['non_zero_mtx'][:, train_gene_idx].copy()
                valid_non_zero_mtx = data['non_zero_mtx'][:, valid_gene_idx]
            
            train_gamma_g = gamma_g[train_gene_idx].copy()
            
            # update theta
            if use_admm:
                this_result = one_admm_fit(train_data, L, mle_theta.copy(), mle_e_alpha.copy(),
                                           train_gamma_g, sigma2,
                                           lambda_r=lambda_r, lasso_weight=lasso_weight,
                                           hv_x=hv_x, hv_log_p=hv_log_p, theta_mask=None,
                                           opt_method=opt_method, global_optimize=False, hybrid_version=hybrid_version,
                                           verbose=False)
                theta = this_result['theta']
                e_alpha = this_result['e_alpha']
                
            else:
                theta, e_alpha = update_theta(train_data, mle_theta.copy(), mle_e_alpha.copy(),
                                              train_gamma_g, sigma2,
                                              lambda_r=lambda_r, lasso_weight=lasso_weight,
                                              hv_x=hv_x, hv_log_p=hv_log_p, theta_mask=None,
                                              global_optimize=False, hybrid_version=hybrid_version, opt_method=opt_method,
                                              verbose=False)
            
            # note fitting result theta is 3-Dimensional
            
            # evaluate in validation fold
            if use_likelihood:
                this_rmse_list.append(calcHVBaseModelLoss(theta.reshape(theta.shape[0], theta.shape[1]), e_alpha, data['Y'][:, valid_gene_idx], data['X'][:, valid_gene_idx], gamma_g[valid_gene_idx], sigma2, hv_x, hv_log_p, data['N'], valid_non_zero_mtx))
                
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
        #print('early stop')
    else:
        #print('100%')
        # find the optimal value with smallest average RMSE, only record the first occurence which corresponding to smaller lambda_r
        optimal_idx = avg_rmse_list.index(min(avg_rmse_list))
    
    if verbose:
        if use_likelihood:
            print(f'find optimal lambda_r {candidate_list[optimal_idx]:.3f} with average negative log-likelihood {avg_rmse_list[optimal_idx]:.4f} by {k} fold cross-validation. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.')
        else:
            print(f'find optimal lambda_r {candidate_list[optimal_idx]:.3f} with average RMSE {avg_rmse_list[optimal_idx]:.4f} by {k} fold cross-validation. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.')
        
    # draw plot
    if diagnosis:
        
        if use_likelihood:
            y_label = 'average negative log-likelihood'
        else:
            y_label = 'average theta RMSE'
        
        from diagnosis_plots import diagnosisParamsTuning
        diagnosisParamsTuning(candidate_list, avg_rmse_list, optimal_idx, 'lambda_r', y_label)
        

    return candidate_list[optimal_idx]
    
    
    
def cv_find_lambda_g(data, L, stage1_theta, stage1_e_alpha, theta_mask, gamma_g, sigma2, candidate_list, hybrid_version=True, opt_method='L-BFGS-B', hv_x=None, hv_log_p=None, use_admm=False, use_likelihood=True, k=5, diagnosis=False):
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
    L : scipy sparse matrix
        already calculated Laplacian Matrix (spots * spots).
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
        whether use ADMM iteration or directly use Laplacian loss function. The default is False, i.e. NOT use ADMM in cross-validation.
    use_likelihood : bool, optional
        whether use negative log-likelihood as performance metric in cross-validation. The default is True, if False use RMSE of predicted gene expression.
    k : int, optional
        the number of folds in cross-validation, The default value is 5.
    diagnosis : bool
        if True save more information to files for diagnosis CVAE and hyper-parameter selection
        
    Returns
    -------
    float
        optimal lambda_g value
    '''
    
    print('\nStart cross-validation for hyper-parameter lambda_g...')
    
    if use_admm:
        print('still use ADMM even low speed')
    else:
        print('directly estimate theta by graph Laplacian loss function!')
    
    
    start_time = time()
    
    # add 0 to candidate list; original list NOT affected
    candidate_list = [0] + candidate_list[:]
    
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
        
        # UPDATE: multiple the hyperparameter with Laplacian matrix get 𝜆L
        # note to use deep copy
        lambda_gL = L.copy() * lambda_g
        
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
            train_data['spot_names'] = data['spot_names'][:]  # copy list for safe
            
            if data['non_zero_mtx'] is None:
                train_data['non_zero_mtx'] = None
                valid_non_zero_mtx = None
            else:
                train_data['non_zero_mtx'] = data['non_zero_mtx'][:, train_gene_idx].copy()
                valid_non_zero_mtx = data['non_zero_mtx'][:, valid_gene_idx]
            
            train_gamma_g = gamma_g[train_gene_idx].copy()
            
            # update theta
            if lambda_g >= 0:
                if use_admm:
                    this_result = one_admm_fit(train_data, lambda_gL,
                                               start_theta.copy(), start_e_alpha.copy(),
                                               train_gamma_g, sigma2,
                                               lambda_r=0, lasso_weight=None,
                                               hv_x=hv_x, hv_log_p=hv_log_p, theta_mask=theta_mask,
                                               opt_method=opt_method, global_optimize=False,
                                               hybrid_version=hybrid_version,
                                               verbose=False)
                else:
                    this_result = fit_base_model_plus_laplacian(train_data, lambda_gL,
                                                                start_theta.copy(), start_e_alpha.copy(),
                                                                train_gamma_g, sigma2,
                                                                lambda_r=None, lasso_weight=None,
                                                                hv_x=hv_x, hv_log_p=hv_log_p,
                                                                theta_mask=theta_mask,
                                                                opt_method=opt_method,
                                                                global_optimize=False,
                                                                hybrid_version=hybrid_version,
                                                                verbose=False)
                theta = this_result['theta']
                e_alpha = this_result['e_alpha']

            else:
                raise Exception(f'lambda_g can not be negative! currently it is {lambda_g}')
            
            # note fitting result theta is 3-Dimensional
            
            # evaluate in validation fold
            if use_likelihood:
                this_rmse_list.append(calcHVBaseModelLoss(theta.reshape(theta.shape[0], theta.shape[1]), e_alpha, data['Y'][:, valid_gene_idx], data['X'][:, valid_gene_idx], gamma_g[valid_gene_idx], sigma2, hv_x, hv_log_p, data['N'], valid_non_zero_mtx))
                
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
        print(f'find optimal lambda_g {candidate_list[optimal_idx]:.3f} with average negative log-likelihood {avg_rmse_list[optimal_idx]:.4f} by {k} fold cross-validation. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.')
    else:
        print(f'find optimal lambda_g {candidate_list[optimal_idx]:.3f} with average RMSE {avg_rmse_list[optimal_idx]:.4f} by {k} fold cross-validation. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.')
        
    # draw plot
    if diagnosis:
        
        if use_likelihood:
            y_label = 'average negative log-likelihood'
        else:
            y_label = 'average theta RMSE'
        
        from diagnosis_plots import diagnosisParamsTuning
        diagnosisParamsTuning(candidate_list, avg_rmse_list, optimal_idx, 'lambda_g', y_label)
    
    
    return candidate_list[optimal_idx]



def calc_AIC_BIC(w, y_vec, mu, gamma_g, sigma2, hv_x, hv_log_p, N, use_AIC=False):
    '''
    calculate BIC or AIC

    Parameters
    ----------
    w : 1-D numpy array (length #celltypes)
        estimated theta (celltype proportion) multiply e_alpha.
    y_vec : 1-D numpy array
        spatial gene expression (length #genes).
    mu : 2-D numpy matrix
        matrix of celltype specific marker gene expression (celltypes * genes).
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
    hv_x : 1-D numpy array, optional
        data points served as x for calculation of probability density values. Only used for heavy-tail.
    hv_log_p : 1-D numpy array, optional
        log density values of normal distribution N(0, sigma^2) + heavy-tail. Only used for heavy-tail.
    N : int or None
        sequencing depth of this spot. If it's None, use sum of observed marker gene expressions as sequencing depth.
    use_AIC : bool, optional
        if True, calculate and return AIC.

    Returns
    -------
    value : float
        calculated BIC or AIC.
    likelihood_part : float
        negative log-likelihood.
    dof : float
        degree of freedom, number of non-zeros minus 1.
    '''
    
    # get theta and e_alpha; NOTE we already set small theta to 0 before return results
    tmp_e_alpha = np.sum(w)
    tmp_theta = w / tmp_e_alpha
    # negative log-likelihood
    likelihood_part = hv_wrapper(w, y_vec, mu, gamma_g, sigma2, hv_x, hv_log_p, N)
    # degree of freedom, non zeros minus 1, note there is at least one non-zero in theta
    dof = np.count_nonzero(tmp_theta) - 1
    
    if use_AIC:
        value = dof * 2 + 2 * likelihood_part
    else:
        # note here the sample size is number of genes
        value = dof * np.log(len(y_vec)) + 2 * likelihood_part
    return value, likelihood_part, dof



def BIC_find_lambda_r_one_spot(mu, y_vec, N, spot_name, mle_theta, mle_e_alpha, gamma_g, sigma2, lasso_weight, candidate_list, hybrid_version=True, opt_method='L-BFGS-B', hv_x=None, hv_log_p=None, use_AIC=False, reinitialize_theta=False, verbose=False):
    '''
    find optimal value for hyper-parameter lambda_r by BIC
    
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
    mle_theta : 1-D numpy array (length #celltypes)
        estimated theta (celltype proportion) by MLE.
    mle_e_alpha : float
        estimated e_alpha by MLE.
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
    lasso_weight : 1-D numpy array (length #celltypes)
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
    use_AIC : bool, optional
        if True, calculate and return AIC.
    reinitialize_theta : bool, optional
        if True, reinitialize theta as 1/#celltype and e_alpha as 1 for optimization.
    verbose : bool, optional
        if True, print more information.
        
    Returns
    -------
    tuple of (float, 1-D numpy array)
        optimal lambda_r value and corresponding optimized theta (for refit)
    '''
    
    start_time = time()
    
    n_celltype = mle_theta.shape[0]
    
    # add 0 to candidate list; original list NOT affected
    candidate_list = [0] + candidate_list[:]
    
    # for each candidate value, start BIC calculation
    bic_list = []
    solution_list = []
    
    if reinitialize_theta:
        init_theta = np.full((n_celltype, ), 1.0/n_celltype)
        init_e_alpha = 1.0
    else:
        init_theta = mle_theta.copy()
        init_e_alpha = mle_e_alpha
    
    for lambda_r in candidate_list:
        # update theta
        w_result = optimize_one_theta(mu.copy(), y_vec.copy(), N, init_theta.copy(), init_e_alpha, gamma_g, sigma2, spot_name, nu_vec=None, rho=None, lambda_r=lambda_r, lasso_weight_vec=lasso_weight, lambda_l2=None, global_optimize=False, hybrid_version=hybrid_version, opt_method=opt_method, hv_x=hv_x, hv_log_p=hv_log_p, this_theta_mask=None, skip_opt=True, verbose=False)
        
        # get theta and e_alpha; NOTE we already set small theta to 0 before return results
        e_alpha = np.sum(w_result)
        theta = w_result / e_alpha

        # evaluate BIC or AIC
        bic, _, _ = calc_AIC_BIC(w_result, y_vec, mu, gamma_g, sigma2, hv_x, hv_log_p, N, use_AIC=use_AIC)
        
        bic_list.append(bic)
        # NOTE here we return the theta NOT w
        solution_list.append(theta.copy())
        
        # NOTE we do not check early stop
        # as we only get a gain on total BIC when we forced addtional theta to 0
        # L1 shrinkage on theta but not to 0 will only cause negative likelihood larger (worse)

    # find the optimal value with smallest BIC, only record the first occurence which corresponding to smaller lambda_r
    optimal_idx = bic_list.index(min(bic_list))
    
    if verbose:
        if use_AIC:
            criteria_label = 'AIC'
        else:
            criteria_label = 'BIC'
        print(f'Spot {spot_name}: find optimal lambda_r {candidate_list[optimal_idx]:.3f} with {criteria_label} {bic_list[optimal_idx]:.4f} in {len(bic_list)} trials. Elapsed time: {(time()-start_time):.2f} seconds.')

    return candidate_list[optimal_idx], solution_list[optimal_idx]



def BIC_find_theta_subset_one_spot(mu, y_vec, N, spot_name, mle_theta, mle_e_alpha, gamma_g, sigma2, hybrid_version=True, opt_method='L-BFGS-B', hv_x=None, hv_log_p=None, use_AIC=False, reinitialize_theta=False, verbose=False):
    '''
    Select sparse theta for one spot by nested subset selection and BIC.

    Strategy:
    - Start from the MLE theta.
    - Cell types with mle_theta == 0 are fixed to zero and never considered.
    - Among cell types with mle_theta > 0, sort by mle_theta (ascending).
    - Construct nested active sets by sequentially setting the smallest MLE-positive thetas to zero.
    - For each active set, refit the (unpenalized) local model with the remaining theta entries free (sum-to-one constraint), compute BIC, and select the active set with minimal BIC.
    
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
    mle_theta : 1-D numpy array (length #celltypes)
        estimated theta (celltype proportion) by MLE.
    mle_e_alpha : float
        estimated e_alpha by MLE.
    gamma_g : 1-D numpy array
        gene-specific platform effect for all genes.
    sigma2 : float
        variance paramter of the lognormal distribution of ln(lambda). All gene share the same variance.
    hybrid_version : bool, optional
        if True, use the hybrid_version of GLRM, i.e. in ADMM local model loss function optimization for w but adaptive lasso constrain on theta. If False, local model loss function optimization and adaptive lasso will on the same w. The default is True.
    opt_method : string, optional
        specify method used in scipy.optimize.minimize for local model fitting. The default is 'L-BFGS-B', a default method in scipy for optimization with bounds. Another choice would be 'SLSQP', a default method in scipy for optimization with constrains and bounds.
    hv_x : 1-D numpy array, optional
        data points served as x for calculation of probability density values. Only used for heavy-tail.
    hv_log_p : 1-D numpy array, optional
        log density values of normal distribution N(0, sigma^2) + heavy-tail. Only used for heavy-tail.
    use_AIC : bool, optional
        if True, calculate and return AIC.
    reinitialize_theta : bool, optional
        if True, reinitialize theta as 1/#celltype and e_alpha as 1 for optimization.
    verbose : bool, optional
        if True, print more information.
        
    Returns
    -------
    optimal theta : 1-D numpy array (length #celltypes)
        updated theta (celltype proportion); already after re-fit.
    optimal e_alpha : float
        updated e_alpha (spot-specific effect); already after re-fit.
    '''
    
    start_time = time()
    
    if np.isclose(mle_theta, 1.0, atol=1e-7).any():
        assert np.sum(mle_theta) == 1
        if verbose:
            print(f'skip adative Lasso for spot {spot_name} as already only one celltype present')
        return mle_theta, mle_e_alpha
    
    
    n_celltype = mle_theta.shape[0]
    
    bic_list = []
    theta_list = []
    e_alpha_list = []
    nonzero_list = []
    

    # start subset models
    # indices of cell types with strictly positive MLE theta:
    # only these are candidates to be dropped
    positive_idx = np.where(mle_theta > 0)[0]
    q = len(positive_idx)
   
    #if verbose:
    #    print(f'total {q} non-zeros in MLE theta')
   
    if q == 0:
       raise Exception(f'[ERROR] all 0s for spot {spot_name}!')
    
    if q == 1:
        raise Exception(f'spot {spot_name} with only one present cell type should not be considered for further dropping cell types!')
    
    # sort these positive-theta indices by their MLE magnitude (ascending)
    order_within_pos = np.argsort(mle_theta[positive_idx])
    sorted_positive_idx = positive_idx[order_within_pos]  # j_(1),...,j_(q), sorted order in original array index, with smallest positive theta first
   
    # k = 0,...,q-1: drop first k smallest MLE-positive thetas; k=0 equals mle theta re-fit
    for k in range(q):
        # active indices among the positive-theta set
        active_idx_pos = sorted_positive_idx[k:]  # keep j_(k+1)...j_(q)
        #if verbose:
        #    print(f'manually select {len(active_idx_pos)} non-zeros')
        
        if k == 0:
            # current BIC of mle theta as baseline; will be calculated based on MLE re-fit results
            if len(active_idx_pos) == n_celltype:
                # NO sparsity, all cell types presented, directly use MLE results, which are derived from all cell type presented condition
                # re-parametrization
                mle_w = mle_theta * mle_e_alpha
                this_bic, _, _ = calc_AIC_BIC(mle_w, y_vec, mu, gamma_g, sigma2, hv_x, hv_log_p, N, use_AIC=use_AIC)
                bic_list.append(this_bic)
                theta_list.append(mle_theta.copy())
                e_alpha_list.append(mle_e_alpha)
                nonzero_list.append(len(active_idx_pos))  # note we record the non-zeros we want before optimization, NOT the actual non-zeros in the optimized result
                
                continue
        

        # build full-length mask: True = active (can be nonzero), False = forced zero
        this_theta_mask = np.zeros((n_celltype,))
        this_theta_mask[active_idx_pos] = 1
        
        if reinitialize_theta:
            init_theta = np.zeros((n_celltype,))
            init_theta[active_idx_pos] = 1.0 / len(active_idx_pos)
            init_e_alpha = 1.0
        else:
            init_theta = mle_theta.copy()
            init_e_alpha = mle_e_alpha
    
        # update theta
        w_result = optimize_one_theta(mu.copy(), y_vec.copy(), N, init_theta.copy(), init_e_alpha, gamma_g, sigma2, spot_name, nu_vec=None, rho=None, lambda_r=None, lasso_weight_vec=None, lambda_l2=None, global_optimize=False, hybrid_version=hybrid_version, opt_method=opt_method, hv_x=hv_x, hv_log_p=hv_log_p, this_theta_mask=this_theta_mask, skip_opt=True, verbose=False)
    
        
        # get theta and e_alpha; NOTE we already set small theta to 0 before return results
        this_e_alpha = np.sum(w_result)
        this_theta = w_result / this_e_alpha

        # evaluate BIC
        this_bic, _, _ = calc_AIC_BIC(w_result, y_vec, mu, gamma_g, sigma2, hv_x, hv_log_p, N, use_AIC=use_AIC)
        bic_list.append(this_bic)
        theta_list.append(this_theta.copy())
        e_alpha_list.append(this_e_alpha)
        nonzero_list.append(len(active_idx_pos))  # note we record the non-zeros we want before optimization, NOT the actual non-zeros in the optimized result
        
        
        # NOTE we do not check early stop
        # as we only get a gain on total BIC when we forced addtional theta to 0
        # L1 shrinkage on theta but not to 0 will only cause negative likelihood larger (worse)

    # find the optimal value with smallest BIC, only record the first occurence which corresponding to smaller lambda_r
    optimal_idx = bic_list.index(min(bic_list))
    
    if verbose:
        print(f'Spot {spot_name}: find optimal pre-set non-zeros {nonzero_list[optimal_idx]} with BIC {bic_list[optimal_idx]:.4f} in {len(bic_list)} trials. Elapsed time: {(time()-start_time):.2f} seconds.')

    return theta_list[optimal_idx], e_alpha_list[optimal_idx]