#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 04:52:02 2022

@author: hill103

this script stores function to define a Graph Laplacian Regularized Stratified Model (GLRM) and call functions to fit the coefficients

It receieve the input data, build graph, build GLRM and fit coefficients
"""



import numpy as np
import networkx as nx
from model_fit import fit_model_two_stage, estimating_gamma_g
from config import print



def run_GLRM(data, lambda_r=None, weight_threshold=1e-3, lambda_g=None, estimate_gamma_g=True, n_jobs=1, threshold=0, diagnosis=False, verbose=False):
    """
    run GLRM, and fit coefficients

    Parameters
    ----------
    data : a Dict contains all info need for modeling
        X: a 2-D numpy matrix of celltype specific marker gene expression (celltypes * genes).
        Y: a 2-D numpy matrix of spatial gene expression (spots * genes).
        A: a 2-D numpy matrix of Adjacency matrix (spots * spots), or is None. Adjacency matrix of spatial sptots (1: connected / 0: disconnected). All 0 in diagonal.
        N: a 1-D numpy array of sequencing depth of all spots (length #spots). If it's None, use sum of observed marker gene expressions as sequencing depth.
        non_zero_mtx: If it's None, then do not filter zeros during regression. If it's a bool 2-D numpy matrix (spots * genes) as False means genes whose nUMI=0 while True means genes whose nUMI>0 in corresponding spots. The bool indicators can be calculated based on either observerd raw nUMI counts in spatial data, or CVAE transformed nUMI counts.
        spot_names: a list of string of spot barcodes. Only keep spots passed filtering.
        gene_names: a list of string of gene symbols. Only keep actually used marker genes.
        celltype_names: a list of string of celltype names.
    lambda_r : float, optional
        strength for Adaptive Lasso penalty. The default is None, i.e. use cross-validation to determine optimal value
    weight_threshold : float, optional
        threshold for Adaptive Lasso weight. Theta values smaller than threshold value will be re-set. The default is 1e-3.
    lambda_g : float, optional
        edge weight for graph, and will affect the strength of Graph Laplacian constrain. The default is None, i.e. use cross-validation to determine optimal value
    estimate_gamma_g : bool, optional
        whether to estimate gamma_g (gene-specific platform effect). The default is True.
    n_jobs : int, optional
        number of jobs to spawn. The default is 1.
    threshold : float, optional
        threshold value for hard thresholding estimated cell-type proportion theta. The default is 0, i.e. no hard thresholding.
    diagnosis : bool
        if True save more information to files for diagnosis CVAE and hyper-parameter selection
    verbose : bool, optional
        if True, print variables in each ADMM loop. The default is True.

    Returns
    -------
    result : Dict
        estimated model coefficients, including (note theta dimension changed back to 2-D):
            theta : celltype proportions (#spots * #celltypes)
            e_alpha : spot-specific effect (1-D array with length #spot)
            sigma2 : variance paramter of the lognormal distribution (float)
            gamma_g : gene-specific platform effect for all genes (1-D array with length #gene)
            theta_tilde : celltype proportions for Adaptive Lasso (#spots * #celltypes)
            theta_hat : celltype proportions for Graph Laplacian constrain (#spots * #celltypes)
    """
    
    assert(data['Y'].shape[1] == data['X'].shape[1])
    if data['A'] is not None:
        assert(data['A'].shape[0] == data['Y'].shape[0])
        assert(data['A'].shape[1] == data['Y'].shape[0])
    
    n_celltype, n_gene = data['X'].shape
    n_spot = data['Y'].shape[0]
    
    print('\nStart GLRM modeling...\n')
    
    print('GLRM settings:')
    # specify some program running parameters
    opt_method = 'L-BFGS-B'
    print('use SciPy minimize method: ', opt_method)
    
    global_optimize = False
    if global_optimize:
        print('use basin-hopping algorithm to find the global minimum')
    else:
        print('global optimization turned off, local minimum will be used in GLRM')
    
    hybrid_version = True
    if hybrid_version:
        print('use hybrid version of GLRM')
    else:
        print('CAUTION: use w version in GLRM!')
    
    import numba as na
    na.set_num_threads(n_jobs)
    print(f'Numba detected total {na.config.NUMBA_NUM_THREADS:d} available CPU cores. Use {na.get_num_threads():d} CPU cores')
    
    from local_fit_numba import z_hv
    print(f'use {len(z_hv):d} points to calculate the heavy-tail density')
    
    print('use weight threshold for Adaptive Lasso: ', weight_threshold)
    
    from local_fit_numba import update_unique_nUMI
    update_unique_nUMI(data['Y'])
    
    use_cache = False
    if use_cache:
        print('use dict to cache likelihoods in regression')
    
    # define graph from adjacency matrix
    if data['A'] is None:
        # empty graph
        G = nx.empty_graph(n_spot)
    else:
        np.fill_diagonal(data['A'], 0)
        rows, cols = np.where(data['A'] == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        # first add nodes. If only add edges, some isolated nodes will not be added
        all_rows = range(0, data['A'].shape[0])
        for n in all_rows:
            G.add_node(n)
        G.add_edges_from(edges)
        # edge weight will be added later
        
    print('\nBuild graph: \n', nx.info(G))
    
   
    # estimate gamma_g
    if estimate_gamma_g:
        print('\nEstimate gene-specific platform effect gamma_g...')
        gamma_g = estimating_gamma_g(data, hybrid_version=hybrid_version, opt_method=opt_method, verbose=True)
    else:
        gamma_g = np.zeros((n_gene,))
        print('\nestimation of gene-specific platform effect gamma_g is skipped as already using CVAE to adjust platform effect')

    
    # start fitting model
    # use two-stage implement
    result = fit_model_two_stage(data, G, gamma_g=gamma_g, lambda_r=lambda_r, weight_threshold=weight_threshold, lambda_g=lambda_g,  global_optimize=global_optimize, hybrid_version=hybrid_version, opt_method=opt_method, verbose=verbose, use_cache=use_cache, diagnosis=diagnosis)
    
    
    # change dimension of theta back to 2-D
    result['theta'] = np.squeeze(result['theta'])
    result['theta_tilde'] = np.squeeze(result['theta_tilde'])
    result['theta_hat'] = np.squeeze(result['theta_hat'])
    

    # post-process theta to set theta<1e-3 as 0 then re-normalize remaining theta to sum to 1
    print('\nPost-processing estimated cell-type proportion theta...')
    print(f'hard thresholding small theta values with threshold {threshold}')
    for i in range(result['theta'].shape[0]):
        tmp_ind = (result['theta'][i,:]>0) & (result['theta'][i,:]<threshold)
        if tmp_ind.any():
            print(f'set small theta values<{threshold:.6f} to 0 and renorm remaining theta for spot {i:d}')
            result['theta'][i, tmp_ind] = 0
            result['theta'][i, :] = result['theta'][i, :] / np.sum(result['theta'][i, :])
    
    return result