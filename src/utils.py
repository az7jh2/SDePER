#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 23:28:03 2022

@author: hill103

this script stores utils functions
"""



import os
import numpy as np
import pandas as pd
from config import print, min_val, output_path
import scanpy as sc
sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)



def calcRMSE(truth, predicted):
    '''
    calculate RMSE

    Parameters
    ----------
    truth : 1-D numpy array
        true values.
    predicted : 1-D numpy array
        predictions.

    Returns
    -------
    float
        RMSE
    '''

    return np.sqrt(((predicted - truth) ** 2).mean())



def reportRMSE(true_theta, pred_theta):
    '''
    calculate the RMSE of theta (celltype proportion) across all spots
    
    we first calculate the RMSE of each spot, then calculate the MEAN of RMSE values across all spots

    Parameters
    ----------
    true_theta : 2-D numpy array (spots * celltypes)
        true values.
    pred_theta : 2-D numpy array (spots * celltypes)
        predictions.

    Returns
    -------
    float
        RMSE across all spots
    '''
    
    return np.array([calcRMSE(true_theta[i,], pred_theta[i,]) for i in range(true_theta.shape[0])]).mean()



def reparameterTheta(theta, e_alpha):
    '''
    re-parametrization w = e^alpha * theta

    Parameters
    ----------
    theta : 3-D numpy array (#spot * #celltype * 1)
        theta (celltype proportion).
    e_alpha : 1-D numpy array
        e^alpha (spot-specific effect).

    Returns
    -------
    w : 3-D numpy array (#spot * #celltype * 1)
        re-parametrization w = e^alpha * theta.

    '''
    
    return e_alpha[:, None, None] * theta



def read_spatial_data(spatial_file):
    '''
    read spatial data saved as a CSV file by Scanpy

    Parameters
    ----------
    spatial_file : string
        full path of input csv file of raw nUMI counts in spatial transcriptomic data (spots * genes).
        
    Returns
    -------
    a AnnData object
    '''
    
    # Read spatial spot-level data
    spatial_spot_obj = sc.read_csv(spatial_file)
    spatial_spot_obj.layers['raw_nUMI'] = spatial_spot_obj.X.copy()
    print(f'read spatial data from file {spatial_file}')
    print(f'total {spatial_spot_obj.n_obs} spots; {spatial_spot_obj.n_vars} genes\n')
    
    # check whether cell name and gene name are unique
    if len(set(spatial_spot_obj.obs_names.tolist())) < spatial_spot_obj.n_obs:
        raise Exception('spot barcodes in spatial data are not unique!')
        
    if len(set(spatial_spot_obj.var_names.tolist())) < spatial_spot_obj.n_vars:
        raise Exception('gene names in spatial data are not unique!')
    
    # Normalize each cell by total counts over ALL genes
    sc.pp.normalize_total(spatial_spot_obj, target_sum=1, inplace=True)
    
    return spatial_spot_obj



def read_scRNA_data(ref_file, ref_anno_file):
    '''
    read scRNA-seq data saved as a CSV file by Scanpy, also read cell-type annotation, then subset cells with cell-type annotation

    Parameters
    ----------
    ref_file : string
        full path of input csv file of raw nUMI counts in scRNA-seq data (cells * genes).
    ref_anno_file : string
        full path of input csv file of cell-type annotations for all cells in scRNA-seq data.
        
    Returns
    -------
    a AnnData object
    '''
    
    # Read scRNA cell-level data and cell-type annotation
    scrna_obj = sc.read_csv(ref_file)
    print(f'read scRNA-seq data from file {ref_file}')
    print(f'total {scrna_obj.n_obs} cells; {scrna_obj.n_vars} genes')
    
    # check whether cell name and gene name are unique
    if len(set(scrna_obj.obs_names.tolist())) < scrna_obj.n_obs:
        raise Exception('spot barcodes in spatial data are not unique!')
        
    if len(set(scrna_obj.var_names.tolist())) < scrna_obj.n_vars:
        raise Exception('gene names in spatial data are not unique!')

    scrna_celltype = pd.read_csv(ref_anno_file, index_col=0)
    print(f'read scRNA-seq cell-type annotation from file {ref_anno_file}')
    print(f'total {len(set(scrna_celltype.iloc[:,0]))} cell-types')
    
    # check whether cell name are unique
    if len(set(scrna_celltype.index.to_list())) < scrna_celltype.shape[0]:
        raise Exception('cell barcodes in scRNA-seq cell-type annotation are not unique!')
        
    # check overlap of cells in gene expression and cell-type annotation
    overlap_cells = sorted(list(set(scrna_celltype.index.to_list()) & set(scrna_obj.obs_names)))
    if len(overlap_cells) < scrna_celltype.shape[0]:
        print(f'WARNING: {scrna_celltype.shape[0]-len(overlap_cells)} cells in cell-type annotation but not found in nUMI matrix')
    
    # only keep cells with cell-type annotations
    scrna_obj = scrna_obj[overlap_cells, ].copy()
    assert((scrna_obj.obs_names == overlap_cells).all())
    print(f'subset cells with cell-type annotation, finally keep {scrna_obj.n_obs} cells; {scrna_obj.n_vars} genes\n')

    # add cell-type annotation to metadata
    scrna_celltype = scrna_celltype.loc[overlap_cells, :]
    assert((scrna_obj.obs_names == scrna_celltype.index).all())
    scrna_obj.obs['celltype'] = pd.Categorical(scrna_celltype.iloc[:,0])  # Categoricals are preferred for efficiency
    # make a DEEP COPY of raw nUMI count
    scrna_obj.layers['raw_nUMI'] = scrna_obj.X.copy()
    # Normalize each cell by total counts over ALL genes
    sc.pp.normalize_total(scrna_obj, target_sum=1, inplace=True)
    
    return scrna_obj



def run_DE(sc_obj, n_marker_per_cmp, save_result=False, save_file_name=None):
    '''
    differential on cell-types in scRNA-seq data.
    
    we compare each cell-type with anotherr one cell-type at a time.
    
    only choose TOP X marker genes for one comparison with one cell-type vs another one cell-type, with the FDR adjusted p value < 0.05 + fold change > 1.2, and a combined rank of log fold change and pct.1/pct.2. Then combine all marker genes from all comparisons.
    
    Note: the genes in object are overlapped genes with spatial data only.
    
    cell-type annotation is saved in object metadata <celltype>
    
    gene expression in AnnData object has already been normalized by sequencing depth

    Parameters
    ----------
    sc_obj : AnnData object
        scRNA-seq data object.
    n_marker_per_cmp : int
        number of TOP marker genes for each comparison in DE
    save_result : bool
        if true, save dataframe of DE result to csv file
    save_file_name : string
        file name (without path) for saving DE result
        
    Returns
    -------
    marker_gene_list : list
        identified cell-type specific marker gene list
    '''
    
    def calc_pct(sc_obj, celltype):
        '''
        calculate pct of genes expressed in cells within one celltype.
        
        raw nUMI count saved in layer <raw_nUMI>

        Parameters
        ----------
        sc_obj : AnnData object
            scRNA-seq data object.
        celltype : string
            name of one cell-type.

        Returns
        -------
        pct_df : Series with genes as index
            A Series including genes and corresponding pcts.
        '''
        
        sub_obj = sc_obj[sc_obj.obs['celltype'] == celltype]
        # get raw nUMI count (cells * genes)
        sub_df = sc.get.obs_df(sub_obj, layer='raw_nUMI', keys=sub_obj.var_names.to_list())
        # column sum divided by number of rows
        return ((sub_df>0).sum(axis=0) + min_val) / (sub_df.shape[0] + 1e-6)
        
        
    
    print('Differential analysis across cell-types on scRNA-seq data...')
    celltypes = sorted(list(set(sc_obj.obs['celltype'])))
    
    scrna_marker_genes = list()
    de_result_list = []
    
    # first calculate pct for each cell-type
    pct_dict = {}
    for this_celltype in celltypes:
        pct_dict[this_celltype] = calc_pct(sc_obj, this_celltype)
    
    # perform test
    for this_celltype in celltypes:
        
        for other_celltype in celltypes:
            if this_celltype == other_celltype:
                continue
            
            # compare one cell-type against another cell-type
            sc.tl.rank_genes_groups(sc_obj, groupby='celltype', use_raw=False, corr_method='benjamini-hochberg',
                                    method='wilcoxon', groups=[this_celltype], reference=other_celltype)
            tmp_df = sc.get.rank_genes_groups_df(sc_obj, group=None)
            
            # add pcts
            tmp_df = tmp_df.merge(pct_dict[this_celltype].rename('pct1'), left_on='names', right_index=True, validate='one_to_one')
            tmp_df = tmp_df.merge(pct_dict[other_celltype].rename('pct2'), left_on='names', right_index=True, validate='one_to_one')
            
            # get genes with pvals_adj < 0.05 and log fold change > 1.2
            tmp_df = tmp_df.loc[(tmp_df['pvals_adj']<0.05) & (tmp_df['logfoldchanges']>np.log2(1.2))]
            
            # add cell-types
            tmp_df['celltype1'] = this_celltype
            tmp_df['celltype2'] = other_celltype
            
            if tmp_df.shape[0] <= n_marker_per_cmp:
                
                # no need to further rank, directly select all available genes
                scrna_marker_genes += tmp_df['names'].to_list()
                
                # combine DE result
                tmp_df['selected'] = 1
            
            else:
            
                # rank of pct.1/pct.2
                tmp_df['pct_divide'] = tmp_df['pct1'] / tmp_df['pct2']
                tmp_df.sort_values(by='pct_divide', ascending=False, inplace=True)
                tmp_df['pct_rank'] = range(tmp_df.shape[0])
                
                # rank of log fold change
                tmp_df.sort_values(by='logfoldchanges', ascending=False, inplace=True)
                tmp_df['logfc_rank'] = range(tmp_df.shape[0])
                
                tmp_df['comb_rank'] = tmp_df['pct_rank'] + tmp_df['logfc_rank']
                tmp_df.sort_values(by=['comb_rank', 'logfoldchanges'], ascending=[True, False], inplace=True)
                
                # select top X marker genes
                scrna_marker_genes += tmp_df['names'].to_list()[:n_marker_per_cmp]
                # combine DE result
                tmp_df['selected'] = 0
                tmp_df.loc[tmp_df.index.to_list()[:n_marker_per_cmp], 'selected'] = 1
            
            tmp_df.rename(columns={'names': 'gene'}, inplace=True)
            de_result_list.append(tmp_df.loc[:, ['gene', 'logfoldchanges', 'pvals', 'pvals_adj', 'pct1', 'pct2', 'celltype1', 'celltype2', 'selected']].copy())

    scrna_marker_genes = sorted(list(set(scrna_marker_genes)))
    print(f'finally selected {len(scrna_marker_genes)} cell-type marker genes\n')
    
    if save_result:
        pd.concat(de_result_list).to_csv(os.path.join(output_path, save_file_name), index=False)
    
    return scrna_marker_genes



def run_DE_only(ref_file, ref_anno_file, spatial_genes, n_marker_per_cmp, save_result=False):
    '''
    read scRNA-seq raw nUMI and cell-type annotation, then perform DE analysis.
    
    Note: the genes in scRNA-seq data need to be subsetted to overlapped genes with spatial data only.

    Parameters
    ----------
    ref_file : string
        full path of input csv file of raw nUMI counts in scRNA-seq data (cells * genes).
    ref_anno_file : string
        full path of input csv file of cell-type annotations for all cells in scRNA-seq data.
    spatial_genes : list
        genes included in spatial dataset.
    n_marker_per_cmp : int
        number of TOP marker genes for each comparison in DE
    save_result : bool
        if true, save dataframe of DE result to csv file
        
    Returns
    -------
    marker_gene_profile : DataFrame
        average gene expressions of identified cell-type specific marker genes from refer scRNA-seq data
    '''
    
    scrna_obj = read_scRNA_data(ref_file, ref_anno_file)
    
    # subset genes
    overlap_genes = list(set(spatial_genes).intersection(set(scrna_obj.var_names)))
    #if len(overlap_genes) < len(spatial_genes):
        #print(f'{len(spatial_genes)-len(overlap_genes)} genes in spatial data but not found in scRNA-seq data: {", ".join(set(spatial_genes).difference(set(overlap_genes)))}\n')
    
    scrna_obj = scrna_obj[:, overlap_genes].copy()
    
    # DE
    marker_genes = run_DE(scrna_obj, n_marker_per_cmp, save_result, 'DE celltype markers.csv')
    
    # generate average gene expressions (gene signature) for cell-types based on normalized values
    tmp_df = sc.get.obs_df(scrna_obj, keys=marker_genes)
    
    tmp_df['celltype'] = scrna_obj.obs['celltype']

    tmp_df = tmp_df.groupby(['celltype']).mean()
    
    return tmp_df



def rerun_DE(scRNA_df, scRNA_celltype, n_marker_per_cmp, save_result=False):
    '''
    rerun DE on CVAE transformed scRNA-seq data
    
    genes are only overlapped genes between spatial and scRNA-seq data
    
    gene expression values are already normalized by sequencing depth

    Parameters
    ----------
    scRNA_df : dataframe
        normalized gene expression after CVAE tranforming on scRNA-seq data (cells * genes).
    scRNA_celltype : dataframe
        cell-type annotations for cells in scRNA-seq data. Only 1 column named <celltype>
    n_marker_per_cmp : int
        number of TOP marker genes for each comparison in DE
    save_result : bool
        if true, save dataframe of DE result to csv file
        
    Returns
    -------
    marker_gene_list : list
        a list of cell-type specific marker genes based on DE on CVAE tranformed gene expressions.
    '''
    
    # first build a AnnData object and replace the normalized data with CVAE transformed gene expressions
    scrna_obj = sc.AnnData(scRNA_df)
    # make a DEEP COPY of raw nUMI count, though it's actually normalized values
    scrna_obj.layers['raw_nUMI'] = scrna_obj.X.copy()
    
    assert((scrna_obj.obs_names == scRNA_celltype.index).all())
    # add cell-type annotation to metadata
    scrna_obj.obs['celltype'] = pd.Categorical(scRNA_celltype.iloc[:,0])  # Categoricals are preferred for efficiency
    
    # do not normalize by sequencing depth as it's already normalized
    # so directly run DE on values in AnnData.X
    return run_DE(scrna_obj, n_marker_per_cmp, save_result, 'redo DE celltype markers.csv')



# check total size of a Python object such as a Dictionary
# ref https://code.activestate.com/recipes/577504/
def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    
    from sys import getsizeof, stderr
    from itertools import chain
    from collections import deque
    try:
        from reprlib import repr
    except ImportError:
        pass
    
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)