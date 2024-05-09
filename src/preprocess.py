#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:13:23 2022

@author: hill103

this script stores functions related to pre-processing including:
    
    1. read data from CSV files
    
    2. determine whether to perform DE for cell-type specific marker genes:
        if user specified marker file, then DE step will be skipped
        
    3. determine whether to build CVAE to adjust the platform effect:
        if user specified use_cvae as True and also provide original scRNA-seq data and corresponding cell-type annotation, then CVAE will be built
        
    4. filter genes and spots before GLRM modeling
    
    5. check datasets to avoid some mistakes
"""



import pandas as pd
from cvae import build_CVAE_whole
from utils import read_spatial_data, run_DE_only
from config import print



def preprocess(spatial_file, ref_file, ref_anno_file, marker_file, A_file, use_cvae, n_hv_gene, n_marker_per_cmp, n_pseudo_spot, pseudo_spot_min_cell, pseudo_spot_max_cell, seq_depth_scaler, cvae_input_scaler, cvae_init_lr, num_hidden_layer, use_batch_norm, cvae_train_epoch, use_spatial_pseudo, redo_de, use_fdr, p_val_cutoff, fc_cutoff, pct1_cutoff, pct2_cutoff, sortby_fc, filter_cell, filter_gene, diagnosis):
    '''
    preprocess files

    Parameters
    ----------
    spatial_file : string
        full path of input csv file of raw nUMI counts in spatial transcriptomic data (spots * genes).
    ref_file : string
        full path of input csv file of raw nUMI counts in scRNA-seq data (cells * genes).
    ref_anno_file : string
        full path of input csv file of cell-type annotations for all cells in scRNA-seq data.
    marker_file : string
        full path of input csv file of cell-typee marker gene expression (cell-types * genes).
    A_file : string
        full path of input csv file of Adjacency Matrix of spots in spatial transcriptomic data (spots * spots).
    use_cvae : bool
        whether to build CVAE to adjust platform effect.
    n_hv_gene : int
        number of highly variable genes for CVAE.
    n_marker_per_cmp : int
        number of TOP marker genes for each comparison in DE.
    n_pseudo_spot : int
        number of pseudo-spots.
    pseudo_spot_min_cell : int
        minimum value of cells in pseudo-spot.
    pseudo_spot_max_cell : int
        maximum value of cells in pseudo-spot.
    seq_depth_scaler : int
        a scaler of scRNA-seq sequencing depth.
    cvae_input_scaler : int
        maximum value of the scaled input for CVAE.
    cvae_init_lr : float
        initial learning rate for training CVAE.
    num_hidden_layer : int
        number of hidden layers in encoder and decoder.
    use_batch_norm : bool
        whether to use Batch Normalization.
    cvae_train_epoch : int
        max number of training epochs for the CVAE.
    use_spatial_pseudo : int
        whether to generate "pseudo-spots" in spatial condition.
    redo_de : bool
        whether to redo DE after CVAE transformation.
    use_fdr : bool
        whether to use FDR adjusted p value for filtering and sorting.
    p_val_cutoff : float
        threshold of p value (or FDR if --use_fdr is true) in marker genes filtering.
    fc_cutoff : float
        threshold of fold change (without log transform!) in marker genes filtering.
    pct1_cutoff : float
        threshold of pct.1 in marker genes filtering.
    pct2_cutoff : float
        threshold of pct.2 in marker genes filtering.
    sortby_fc : bool
        whether to sort marker genes by fold change.
    diagnosis : bool
        if True save more information to files for diagnosis CVAE and hyper-parameter selection.
    filter_cell : bool
        whether to filter cells before DE.
    filter_gene : bool
        whether to filter genes before DE.

    Returns
    -------
    data : Dict
        a Dict contains all info need for modeling:
            X: a 2-D numpy matrix of celltype specific marker gene expression (celltypes * genes).\n
            Y: a 2-D numpy matrix of spatial gene expression (spots * genes).\n
            A: a 2-D numpy matrix of Adjacency matrix (spots * spots), or is None. Adjacency matrix of spatial sptots (1: connected / 0: disconnected). All 0 in diagonal.\n
            N: a 1-D numpy array of sequencing depth of all spots (length #spots). If it's None, use sum of observed marker gene expressions as sequencing depth.\n
            non_zero_mtx: If it's None, then do not filter zeros during regression. If it's a bool 2-D numpy matrix (spots * genes) as False means genes whose nUMI=0 while True means genes whose nUMI>0 in corresponding spots. The bool indicators can be calculated based on either observerd raw nUMI counts in spatial data, or CVAE transformed nUMI counts.\n
            spot_names: a list of string of spot barcodes. Only keep spots passed filtering.\n
            gene_names: a list of string of gene symbols. Only keep actually used marker gene symbols.\n
            celltype_names: a list of string of cell-type names.\n
            initial_guess: initial guess of cell-type proportions of spatial spots.
    '''
    
    # first determine whether to build CVAE
    if use_cvae:
        if ref_file is None or ref_anno_file is None:
            raise Exception('ERROR: building CVAE requires both reference scRNA-seq data and corresponding cell-type annotation specified! But at least one of them is not specified!')
            
        print('first build CVAE...\n')
        # build CVAE, and return the data dict including transformed spatial data and reference gene expression
        spatial_df, cvae_marker_df, new_markers, cvae_pred = build_CVAE_whole(spatial_file, ref_file, ref_anno_file, marker_file, n_hv_gene, n_marker_per_cmp, n_pseudo_spot, pseudo_spot_min_cell, pseudo_spot_max_cell, seq_depth_scaler, cvae_input_scaler, cvae_init_lr, num_hidden_layer, use_batch_norm, cvae_train_epoch, use_spatial_pseudo, redo_de, use_fdr, p_val_cutoff, fc_cutoff, pct1_cutoff, pct2_cutoff, sortby_fc, diagnosis, filter_cell, filter_gene)
        
        # calculate squencing depth, sum also works on sparse dataframe
        N = spatial_df.sum(axis=1)
    
    else:
        
        print('building CVAE skipped...\n')
        
        # read spatial data
        spatial_obj, spatial_df, N = read_spatial_data(spatial_file, filter_gene)
        
        new_markers = None
        cvae_pred = None
    
        
    # use marker genes from original scRNA-seq or CVAE transformed data
    if new_markers is None:
        
        # whether to perform DE
        if marker_file is not None:
            # directly use provide marker gene expression
            # the default mangle_dupe_cols=True will handle the duplicated columns
            marker_df = pd.read_csv(marker_file, index_col=0)
            print('user provided marker gene profile, DE will be skipped...\n')
            print(f'read {marker_df.shape[1]} marker genes from user specified marker gene file')
            
            # check whether cell name are unique
            if len(set(marker_df.index.to_list())) < marker_df.shape[0]:
                raise Exception('cell-type names in user provided marker gene profile are not unique!')
            
            # extract marker gene overlapped with spatial data
            marker_genes = sorted(list(set(spatial_df.columns) & set(marker_df.columns)))
            print(f'from user specified marker gene expression use {len(marker_genes)} marker genes overlapped with spatial + scRNA-seq data')
            # if len(marker_genes) < spatial_df.shape[1]:
            #     print(f'{spatial_df.shape[1]-len(marker_genes)} genes in overlapped gene list between spatial and scRNA-seq data but not found in user provided marker gene expression: {", ".join(set(spatial_df.columns).difference(set(marker_genes)))}\n')
            
        else:
            # perform DE, return the marker gene expression. The identified markers but not in spatial data has already been removed
            print('no marker gene profile provided. Perform DE to get cell-type marker genes on scRNA-seq data...\n')
            tmp_scrna_obj, marker_df = run_DE_only(ref_file, ref_anno_file, spatial_df.columns.tolist(), n_marker_per_cmp, use_fdr, p_val_cutoff, fc_cutoff, pct1_cutoff, pct2_cutoff, sortby_fc, diagnosis, filter_cell, filter_gene)
            # note dataframe index and columns returns a RangeIndex object rather than list
            marker_genes = marker_df.columns.to_list()
    
    else:
        
        print('use the marker genes derived from CVAE transformed scRNA-seq for downstream regression!')
        marker_genes = new_markers
    
    
    if use_cvae:
        # replace the marker gene profile with the one transformed by CVAE
        marker_df = cvae_marker_df
        
    
    nUMI_threshold = 5
    
    # exclude genes which are all zeros in spatial data
    print('\ngene filtering before modeling...')
    tmp = spatial_df[marker_genes].sum(axis=0)
    all_zero_genes = tmp[tmp < nUMI_threshold].index.to_list()
    if len(all_zero_genes) > 0:
        print(f'{len(all_zero_genes)} genes with nUMIs<{nUMI_threshold} in all spatial spots and need to be excluded')
    
        marker_genes = sorted(list(set(marker_genes) - set(all_zero_genes)))
        print(f'finally use {len(marker_genes)} genes for modeling')
    else:
        print('all genes passed filtering')
    
    # subset genes
    spatial_df = spatial_df[marker_genes]
    marker_df = marker_df[marker_genes]
    
    # reorder cell-type orders in marker gene profile for consistency
    celltype_order = sorted(marker_df.index.to_list())
    if cvae_pred is not None:
        assert celltype_order == cvae_pred.columns.to_list()
    marker_df = marker_df.loc[celltype_order, :]
    
    if A_file is None:
        A_df = None
    else:
        A_df = pd.read_csv(A_file, index_col=0)
        
    # filter spots based on sum of nUMI
    print('\nspot filtering before modeling...')
    exclude_spots = spatial_df.index[spatial_df.sum(axis=1) < nUMI_threshold].tolist()
    if len(exclude_spots) > 0:
        print(f'{len(exclude_spots):d} spots will be excluded as sum of nUMI < {nUMI_threshold:d}')
        spatial_df.drop(exclude_spots, inplace=True)
        if A_df is not None:
            A_df.drop(exclude_spots, inplace=True)
            A_df.drop(exclude_spots, axis=1, inplace=True)
    else:
        print('all spots passed filtering')
        
        
    if diagnosis and not use_cvae:
        # plot UMAP for raw input gene expressions here
        import scanpy as sc
        from diagnosis_plots import defineColor, rawInputUMAP
        # get raw input of scRNA-seq cells
        if 'tmp_scrna_obj' not in locals():
            if ref_file is not None and ref_anno_file is not None:
                from utils import read_scRNA_data
                tmp_scrna_obj = read_scRNA_data(ref_file, ref_anno_file, filter_cell, filter_gene)
            else:
                tmp_scrna_obj = None
        
        if tmp_scrna_obj is not None:
            tmp_scrna_celltype = sc.get.obs_df(tmp_scrna_obj, keys='celltype').to_frame()
            tmp_scrna_celltype['celltype'] = tmp_scrna_celltype['celltype'].astype(str)
            plot_colors = defineColor(spatial_df.shape[0], tmp_scrna_celltype)
            rawInputUMAP(spatial_df, sc.get.obs_df(tmp_scrna_obj, keys=marker_genes), tmp_scrna_celltype, plot_colors)

    
    # record the zeros in spatial data if filtering zeros is turned on
    filter_zero_gene = False
    use_original_nUMI = True
    
    if filter_zero_gene:
        print('\nCAUTION: filtering genes with nUMI=0 in spot before regression!')
        if use_original_nUMI:
            print('CAUTION: gene nUMI=0 determined by original spatial raw nUMI counts')
            # reload the spatial data, and find out genes with nUMI>0
            non_zero_df = pd.concat(chunk.astype('Sparse[int]') for chunk in pd.read_csv(spatial_file, index_col=0, chunksize=1e4))
            non_zero_df = non_zero_df.loc[spatial_df.index, spatial_df.columns]
            non_zero_df = non_zero_df > 0
        else:
            print('CAUTION: gene nUMI=0 determined by current CVAE transformed nUMI counts')
            # just use the current spatial data, and find out genes with nUMI>0
            non_zero_df = spatial_df > 0
        non_zero_mtx = non_zero_df.values
    else:
        non_zero_mtx = None
    
    
    # final check to avoid mistakes

    # check the spot barcodes is the same order
    if A_df is not None:
        assert((spatial_df.index==A_df.index).all())
        assert((A_df.index==A_df.columns).all())
        
    # check the gene names is the same order
    assert((spatial_df.columns==marker_df.columns).all())
    
    if A_df is not None:
        A = A_df.values
    else:
        A = None
    
    # return dense values
    return {'X': marker_df.values,
            'Y': spatial_df.values,
            'A': A,
            'N': N.values,
            'non_zero_mtx': non_zero_mtx,
            'spot_names': spatial_df.index.to_list(),
            'gene_names': spatial_df.columns.to_list(),
            'celltype_names': marker_df.index.to_list(),
            'initial_guess': cvae_pred}