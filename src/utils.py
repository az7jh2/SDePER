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
from config import print, diagnosis_path
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



def read_spatial_data(spatial_file, filter_gene, n_hv_gene=0):
    '''
    read spatial data saved as a CSV file by Scanpy

    Parameters
    ----------
    spatial_file : string
        full path of input csv file of raw nUMI counts in spatial transcriptomic data (spots * genes).
    filter_gene : bool
        whether to filter genes before DE.
    n_hv_gene : int
        number of highly variable genes to be kept in spatial data. If equals 0, all genes are kept.
        
    Returns
    -------
    spatial_spot_obj : a AnnData object
        AnnData object of spatial spot data.
    tmp_df : dataframe
        dataframe of raw nUMI of spatial data (spots * genes).
    N : series
        sequencing depth per spot.
    '''
    
    # Read spatial spot-level data
    spatial_spot_obj = sc.read_csv(spatial_file)
    print(f'read spatial data from file {spatial_file}')
    print(f'total {spatial_spot_obj.n_obs} spots; {spatial_spot_obj.n_vars} genes\n')
    
    # check whether cell name and gene name are unique
    if len(set(spatial_spot_obj.obs_names.tolist())) < spatial_spot_obj.n_obs:
        raise Exception('spot barcodes in spatial data are not unique!')
        
    if len(set(spatial_spot_obj.var_names.tolist())) < spatial_spot_obj.n_vars:
        raise Exception('gene names in spatial data are not unique!')
    
    # note for spatial data, we do not filter out spots
    if filter_gene:
        # Remove genes present in <3 cells
        pre_n_gene = spatial_spot_obj.n_vars
        sc.pp.filter_genes(spatial_spot_obj, min_cells=3)
        if pre_n_gene > spatial_spot_obj.n_vars:
            print(f'filtering genes present in <3 spots: {pre_n_gene-spatial_spot_obj.n_vars} genes removed\n')
        else:
            print('filtering genes present in <3 spots: No genes removed\n')
            
    # make a DEEP COPY of raw nUMI count
    spatial_spot_obj.layers['raw_nUMI'] = spatial_spot_obj.X.copy()
    
    # calculate sequencing depth per cell, note currently X is nUMI, we make a deep copy to avoid dataframe change
    tmp_df = spatial_spot_obj.to_df().copy()
    N = tmp_df.sum(axis=1) # sum also works on sparse dataframe
    
    # Normalize each cell by total counts over ALL genes
    sc.pp.normalize_total(spatial_spot_obj, target_sum=1, inplace=True)
    
    # identify highly variable genes in spatial data, select TOP X HV genes
    # no need to consider highly variable genes in spatial data, as for cell-type deconvolution, we work on each spot independently
    if n_hv_gene >= spatial_spot_obj.n_vars:
        print(f'\nWARNING: use all {spatial_spot_obj.n_vars} genes for downstream analysis as available genes in spatial data <= specified highly varabile gene number {n_hv_gene}')
    
    elif n_hv_gene > 0:
        print(f'\nWARNING: identify {n_hv_gene} highly variable genes from spatial data and keep those genes only...')
        spatial_hv_genes = sc.pp.highly_variable_genes(spatial_spot_obj, layer='raw_nUMI', flavor='seurat_v3', n_top_genes=n_hv_gene, inplace=False)
        spatial_hv_genes = spatial_hv_genes.loc[spatial_hv_genes['highly_variable']==True].index.to_list()
        spatial_spot_obj = spatial_spot_obj[:, spatial_hv_genes].copy()
        
        tmp_df = tmp_df[spatial_hv_genes]
    
    # after sequencing depth calculation and normalization we remove mitochondrial genes
    non_mt_genes = [gene for gene in spatial_spot_obj.var_names if not gene.startswith('MT-') ]
    n_mt_genes = spatial_spot_obj.n_vars - len(non_mt_genes)
    print(f'filtering {n_mt_genes} mitochondrial genes\n')
    
    if n_mt_genes > 0:
        # the same filtering will be applied to all layers
        spatial_spot_obj = spatial_spot_obj[:, non_mt_genes].copy()
        tmp_df = tmp_df[non_mt_genes]
    
    print(f'finally remain {spatial_spot_obj.n_obs} spots; {spatial_spot_obj.n_vars} genes for downstream analysis\n')
    
    return spatial_spot_obj, tmp_df, N



def read_scRNA_data(ref_file, ref_anno_file, filter_cell, filter_gene):
    '''
    read scRNA-seq data saved as a CSV file by Scanpy, also read cell-type annotation, then subset cells with cell-type annotation

    Parameters
    ----------
    ref_file : string
        full path of input csv file of raw nUMI counts in scRNA-seq data (cells * genes).
    ref_anno_file : string
        full path of input csv file of cell-type annotations for all cells in scRNA-seq data.
    filter_cell : bool
        whether to filter cells before DE.
    filter_gene : bool
        whether to filter genes before DE.
        
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
    
    if filter_cell:
        # Remove cells with <200 genes
        pre_n_cell = scrna_obj.n_obs
        sc.pp.filter_cells(scrna_obj, min_genes=200)
        if pre_n_cell > scrna_obj.n_obs:
            print(f'filtering cells with <200 genes: {pre_n_cell-scrna_obj.n_obs} cells removed\n')
        else:
            print('filtering cells with <200 genes: No cells removed\n')
    
    if filter_gene:
        # Remove genes present in <10 cells
        pre_n_gene = scrna_obj.n_vars
        sc.pp.filter_genes(scrna_obj, min_cells=10)
        if pre_n_gene > scrna_obj.n_vars:
            print(f'filtering genes present in <10 cells: {pre_n_gene-scrna_obj.n_vars} genes removed\n')
        else:
            print('filtering genes present in <10 cells: No genes removed\n')
    
    # make a DEEP COPY of raw nUMI count
    scrna_obj.layers['raw_nUMI'] = scrna_obj.X.copy()
    # Normalize each cell by total counts over ALL genes
    sc.pp.normalize_total(scrna_obj, target_sum=1, inplace=True)
    
    # after sequencing depth calculation and normalization we remove mitochondrial genes
    non_mt_genes = [gene for gene in scrna_obj.var_names if not gene.startswith('MT-') ]
    n_mt_genes = scrna_obj.n_vars - len(non_mt_genes)
    print(f'filtering {n_mt_genes} mitochondrial genes\n')
    if n_mt_genes > 0:
        # the same filtering will be applied to all layers
        scrna_obj = scrna_obj[:, non_mt_genes].copy()
    
    print(f'finally remain {scrna_obj.n_obs} cells; {scrna_obj.n_vars} genes for downstream analysis\n')
    
    return scrna_obj



def run_DE(sc_obj, n_marker_per_cmp, use_fdr, p_val_cutoff, fc_cutoff, pct1_cutoff, pct2_cutoff, sortby_fc, save_result=False, save_file_name=None):
    '''
    differential on cell-types in scRNA-seq data.
    
    we compare each cell-type with another one cell-type at a time.
    
    only choose TOP X marker genes for one comparison with one cell-type vs another one cell-type, with filtering (the FDR adjusted p value <= 0.05 + fold change >= 1.2 + pct.1 >= 0.3 + pct.2 <= 0.1, and sorting by fold change (by default). Then combine all marker genes from all comparisons.
    
    Note: the genes in object are overlapped genes with spatial data only.
    
    cell-type annotation is saved in object metadata <celltype>
    
    gene expression in AnnData object has already been normalized by sequencing depth

    Parameters
    ----------
    sc_obj : AnnData object
        scRNA-seq data object.
    n_marker_per_cmp : int
        number of TOP marker genes for each comparison in DE
    use_fdr : bool
        whether to use FDR adjusted p value for filtering and sorting
    p_val_cutoff : float
        threshold of p value (or FDR if --use_fdr is true) in marker genes filtering
    fc_cutoff : float
        threshold of fold change (without log transform!) in marker genes filtering
    pct1_cutoff : float
        threshold of pct.1 in marker genes filtering
    pct2_cutoff : float
        threshold of pct.2 in marker genes filtering
    sortby_fc : bool
        whether to sort marker genes by fold change
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
        assert sub_df.shape[0] > 0, f'Error! there is no cell for cell-type `{celltype}`'
        # column sum divided by number of rows
        return ((sub_df>0).sum(axis=0)) / (sub_df.shape[0])
        
        
    
    print('Differential analysis across cell-types on scRNA-seq data...')
    celltypes = sorted(list(set(sc_obj.obs['celltype'])))
    
    scrna_marker_genes = list()
    de_result_list = []
    
    if use_fdr:
        pval_col = 'pvals_adj'
    else:
        pval_col = 'pvals'
    
    # first calculate pct for each cell-type
    pct_dict = {}
    for this_celltype in celltypes:
        pct_dict[this_celltype] = calc_pct(sc_obj, this_celltype)
    
    # perform test
    for t, this_celltype in enumerate(celltypes):
        
        print(f'{t/len(celltypes):.0%}...', end='')
        
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
            
            # filter genes
            tmp_df = tmp_df.loc[(tmp_df[pval_col]<=p_val_cutoff) & (tmp_df['logfoldchanges']>=np.log2(fc_cutoff)) & (tmp_df['pct1']>=pct1_cutoff) & (tmp_df['pct2']<=pct2_cutoff)]
            
            # add cell-types
            tmp_df['celltype1'] = this_celltype
            tmp_df['celltype2'] = other_celltype
            
            if tmp_df.shape[0] <= n_marker_per_cmp:
                
                if tmp_df.shape[0] < n_marker_per_cmp:
                    print(f'\nWARNING: only {tmp_df.shape[0]} genes passing filtering (<{n_marker_per_cmp}) for {this_celltype} vs {other_celltype}')
                
                # no need to further rank, directly select all available genes
                scrna_marker_genes += tmp_df['names'].to_list()
                
                # combine DE result
                tmp_df['selected'] = 1
            
            else:
                '''
                # rank of pct.1/pct.2
                tmp_df['pct_divide'] = tmp_df['pct1'] / tmp_df['pct2']
                tmp_df.sort_values(by='pct_divide', ascending=False, inplace=True)
                tmp_df['pct_rank'] = range(tmp_df.shape[0])
                
                # rank of log fold change
                tmp_df.sort_values(by='logfoldchanges', ascending=False, inplace=True)
                tmp_df['logfc_rank'] = range(tmp_df.shape[0])
                
                tmp_df['comb_rank'] = tmp_df['pct_rank'] + tmp_df['logfc_rank']
                tmp_df.sort_values(by=['comb_rank', 'logfoldchanges'], ascending=[True, False], inplace=True)
                '''
                
                # sort by fold change or p value
                if sortby_fc:
                    tmp_df.sort_values(by=[pval_col, 'logfoldchanges'], ascending=[True, False], inplace=True)
                else:
                    tmp_df.sort_values(by=['logfoldchanges', pval_col], ascending=[False, True], inplace=True)
                
                # select top X marker genes
                scrna_marker_genes += tmp_df['names'].to_list()[:n_marker_per_cmp]
                # combine DE result
                tmp_df['selected'] = 0
                tmp_df.loc[tmp_df.index.to_list()[:n_marker_per_cmp], 'selected'] = 1
            
            tmp_df.rename(columns={'names': 'gene'}, inplace=True)
            de_result_list.append(tmp_df.loc[:, ['gene', 'logfoldchanges', 'pvals', 'pvals_adj', 'pct1', 'pct2', 'celltype1', 'celltype2', 'selected']].copy())
    
    print('100%')
    
    scrna_marker_genes = sorted(list(set(scrna_marker_genes)))
    print(f'finally selected {len(scrna_marker_genes)} cell-type marker genes\n')
    
    if save_result:
        os.makedirs(os.path.join(diagnosis_path, 'celltype_markers'), exist_ok=True)
        pd.concat(de_result_list).to_csv(os.path.join(diagnosis_path, 'celltype_markers', save_file_name)+'.gz', index=False, compression='gzip')
    
    return scrna_marker_genes



def run_DE_only(ref_file, ref_anno_file, spatial_genes, n_marker_per_cmp, use_fdr, p_val_cutoff, fc_cutoff, pct1_cutoff, pct2_cutoff, sortby_fc, save_result=False, filter_cell=True, filter_gene=True):
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
    use_fdr : bool
        whether to use FDR adjusted p value for filtering and sorting
    p_val_cutoff : float
        threshold of p value (or FDR if --use_fdr is true) in marker genes filtering
    fc_cutoff : float
        threshold of fold change (without log transform!) in marker genes filtering
    pct1_cutoff : float
        threshold of pct.1 in marker genes filtering
    pct2_cutoff : float
        threshold of pct.2 in marker genes filtering
    sortby_fc : bool
        whether to sort marker genes by fold change
    save_result : bool
        if true, save dataframe of DE result to csv file
    filter_cell : bool
        whether to filter cells before DE
    filter_gene : bool
        whether to filter genes before DE
        
    Returns
    -------
    scrna_obj : AnnData object
        a AnnData object for scRNA-seq data
    marker_gene_profile : DataFrame
        average gene expressions of identified cell-type specific marker genes from refer scRNA-seq data
    '''
    
    scrna_obj = read_scRNA_data(ref_file, ref_anno_file, filter_cell, filter_gene)
    
    # subset genes
    overlap_genes = list(set(spatial_genes).intersection(set(scrna_obj.var_names)))
    print(f'get {len(overlap_genes)} overlapped genes between spatial data and reference scRNA-seq data\n')
    #if len(overlap_genes) < len(spatial_genes):
        #print(f'{len(spatial_genes)-len(overlap_genes)} genes in spatial data but not found in scRNA-seq data: {", ".join(set(spatial_genes).difference(set(overlap_genes)))}\n')
    
    scrna_obj = scrna_obj[:, overlap_genes].copy()
    
    # DE
    marker_genes = run_DE(scrna_obj, n_marker_per_cmp, use_fdr, p_val_cutoff, fc_cutoff, pct1_cutoff, pct2_cutoff, sortby_fc, save_result, 'DE_celltype_markers.csv')
    
    # generate average gene expressions (gene signature) for cell-types based on normalized values
    tmp_df = sc.get.obs_df(scrna_obj, keys=marker_genes)
    
    tmp_df['celltype'] = scrna_obj.obs['celltype']

    tmp_df = tmp_df.groupby(['celltype']).mean()
    
    return scrna_obj, tmp_df



def rerun_DE(scRNA_df, scRNA_celltype, n_marker_per_cmp, use_fdr, p_val_cutoff, fc_cutoff, pct1_cutoff, pct2_cutoff, sortby_fc, save_result=False, filter_gene=True):
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
    use_fdr : bool
        whether to use FDR adjusted p value for filtering and sorting
    p_val_cutoff : float
        threshold of p value (or FDR if --use_fdr is true) in marker genes filtering
    fc_cutoff : float
        threshold of fold change (without log transform!) in marker genes filtering
    pct1_cutoff : float
        threshold of pct.1 in marker genes filtering
    pct2_cutoff : float
        threshold of pct.2 in marker genes filtering
    sortby_fc : bool
        whether to sort marker genes by fold change
    save_result : bool
        if true, save dataframe of DE result to csv file
    filter_gene : bool
        whether to filter genes before DE.
        
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
    
    if filter_gene:
        # Remove genes present in <10 cells
        pre_n_gene = scrna_obj.n_vars
        sc.pp.filter_genes(scrna_obj, min_cells=10)
        if pre_n_gene > scrna_obj.n_vars:
            print(f'filtering genes present in <10 cells: {pre_n_gene-scrna_obj.n_vars} genes removed\n')
        else:
            print('filtering genes present in <10 cells: No genes removed\n')
    
    # do not normalize by sequencing depth as it's already normalized
    # so directly run DE on values in AnnData.X
    return run_DE(scrna_obj, n_marker_per_cmp, use_fdr, p_val_cutoff, fc_cutoff, pct1_cutoff, pct2_cutoff, sortby_fc, save_result, 'redo_DE_celltype_markers.csv')



# check total size of a Python object such as a Dictionary
# ref https://code.activestate.com/recipes/577504/
def total_size(o, handlers={}, verbose=False):
    """
    Returns the approximate memory footprint of an object and all of its contents.
    
    Automatically finds the contents of several built-in containers and their subclasses,
    including tuple, list, deque, dict, set, and frozenset. To search other containers, 
    add handlers that iterate over their contents.
    
    Parameters
    ----------
    o : object
        The object whose memory footprint is to be calculated.
    handlers : dict, optional
        A dictionary of handler functions keyed by container types. These handlers are used to iterate over container contents. Defaults to an empty dictionary.
    verbose : bool, optional
        If True, the function prints more information about what it's doing. Defaults to False.
    
    Returns
    -------
    obj_size : int
        An approximation of the total memory footprint of the object in bytes.
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



def check_decoder(cvae, decoder, data, labels):
    '''
    since we first create a decoder then update its weights based on the corresponding weights in CVAE, we need to double check the weights are updated correctly, and the decoded output matchs the CVAE output

    Parameters
    ----------
    cvae : Keras model
        already trained CVAE model
    decoder : Keras model
        a separated decoder whose weights are already updated, i.e. it should give the same decoded output with CVAE
    data : 2-D numpy array
        data used for checking decoder (columns are genes, rows are cells, spatial spots or pseudo-spots)
    labels : 1-D numpy array
        corresponding conditional variables for each row in data

    Returns
    -------
    None.
    '''
    
    from tensorflow.keras.models import Model
    
    # a tmp model to get the embedding after sampling and decoder output at the same time
    tmp_model = Model([cvae.get_layer('encoder_input').input, cvae.get_layer('cond_input').input],
                      [cvae.get_layer('z').output, cvae.get_layer('decoder_output_act').output],
                      name='tmp_model')
    # the preditions of embedding and decoder output
    [tmp_embedding, tmp_output] = tmp_model.predict([data, labels])
    # feed the embedding to the new decoder
    tmp_output2 = decoder.predict([tmp_embedding, labels])
    # are them the same?
    tmp = np.all((tmp_output-tmp_output2)<1e-12)
    assert(tmp==True)