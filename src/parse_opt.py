#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 03:23:09 2022

@author: hill103

this script stores functions related to receive and parse the command line parameters
checking on parameters will also be performed
"""



from getopt import getopt
import sys, os
from config import print, input_path, diagnosis_path, cur_version
import numpy as np
import pandas as pd
import copy



# default value for options
default_paramdict = {'spatial_file': None, 'ref_file': None, 'ref_celltype_file': None, 'marker_file': None, 'loc_file': None, 'A_file': None,
                     'n_cores': 1, 'threshold': 0, 'use_cvae': True, 'use_imputation': False, 'diagnosis': False, 'verbose': True,
                     'use_fdr': True, 'p_val_cutoff': 0.05, 'fc_cutoff': 1.2, 'pct1_cutoff': 0.3, 'pct2_cutoff': 0.1, 'sortby_fc': True, 'n_marker_per_cmp': 20, 'filter_cell': True, 'filter_gene': True,
                     'n_hv_gene': 200,  'n_pseudo_spot': 100000, 'pseudo_spot_min_cell': 2, 'pseudo_spot_max_cell':8, 'seq_depth_scaler': 10000, 'cvae_input_scaler': 10, 'cvae_init_lr':0.01, 'num_hidden_layer': 1, 'use_batch_norm': True, 'cvae_train_epoch': 500, 'use_spatial_pseudo': False, 'redo_de': True, 'seed': 383,
                     'lambda_r': None, 'lambda_r_range_min': 0.1, 'lambda_r_range_max': 50, 'lambda_r_range_k': 6,
                     'lambda_g': None, 'lambda_g_range_min': 0.1, 'lambda_g_range_max': 50, 'lambda_g_range_k': 6,
                     'diameter': 200, 'impute_diameter': [160, 114, 80], 'hole_min_spots': 1, 'preserve_shape': False
                    }



def usage():
    '''
    help information of the main function
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    None.
    '''
    
    print(f'''
runDeconvolution [option][value]...

    -h or --help            print this help messages.
    -v or --version         print version of SDePER
    
    
    --------------- Input options -------------------
    
    -q or --query           input csv file of raw nUMI counts of spatial transcriptomic data (spots * genes), with absolute or relative path. Rows as spots and columns as genes. Row header as spot barcodes and column header as gene symbols are both required.
    -r or --ref             input csv file of raw nUMI counts of scRNA-seq data (cells * genes), with absolute or relative path. Rows as cells and columns as genes. Row header as cell barcodes and column header as gene symbols are both required.
    -c or --ref_anno        input csv file of cell-type annotations for all cells in scRNA-seq data, with absolute or relative path. Rows as cells and only 1 column as cell-type annotation. Row header as cell barcodes and column header with arbitrary name are both required.
    -m or --marker          input csv file of already curated cell-type marker gene expression (cell-types * genes; already normalized by sequencing depth), with absolute or relative path. Rows as cell-types and columns as genes. Row header as cell-type names and column header as gene symbols are both required. If marker gene expression is provided, the built-in differential analysis will be skipped and genes from this csv file will be directly used for cell-type deconvolution, as well as CVAE building. If not provided, Wilcoxon rank sum test will be performed to select cell-type marker genes. Default value is {default_paramdict["marker_file"]}.
    -l or --loc             input csv file of row/column integer index (x,y) of spatial spots (spots * 2), with absolute or relative path. Rows as spots and columns are coordinates x (column index) and y (row index). Row header as spot barcodes and column header "x","y" are both required. NOTE 1) the column header must be either "x" or "y" (lower case), 2) x and y are integer index (1,2,3,...) not pixels. This spot location file is required for imputation. And the spot order should be consist with row order in spatial nUMI count data. Default value is {default_paramdict["loc_file"]}.
    -a or --adjacency       input csv file of Adjacency Matrix of spots in spatial transcriptomic data (spots * spots), with absolute or relative path. In Adjacency Matrix, entry value 1 represents corresponding two spots are adjacent spots according to the definition of neighborhood, while value 0 for non-adjacent spots. All diagonal entries are set as 0. Row header and column header as spot barcodes are both required. And the spot order should be consist with row order in spatial data. Default value is {default_paramdict["A_file"]}.
    
    
    --------------- Output options -------------------
    
    We do not provide options for renaming output files. All outputs are in the same folder as input files.
    The cell-type deconvolution result file is named as "celltype_proportions.csv".
    If imputation is enabled, for each specified spot diameter d µm, there will be three more output files: 1) imputed spot locations "impute_diameter_d_spot_loc.csv", 2) imputed spot cell-type proportions "impute_diameter_d_spot_celltype_prop.csv", 3) imputed spot gene expressions (already normalized by sequencing depth of spots) "impute_diameter_d_spot_gene_norm_exp.csv.gz".
    
    
    --------------- General options -------------------
    
    -n or --n_cores         number of CPU cores used for parallel computing. Default value is {default_paramdict["n_cores"]}, i.e. no parallel computing.
    --threshold             threshold for hard thresholding the estimated cell-type proportions, i.e. for one spot, estimated cell-type proportions smaller than this threshold value will be set to 0, then re-normalize all proportions of this spot to sum as 1. Default value is {default_paramdict["threshold"]}, which means no hard thresholding.
    --use_cvae              control whether to build Conditional Variational Autoencoder (CVAE) to remove the platform effect between spatial transcriptomic and scRNA-seq data (true/false). Default value is {default_paramdict["use_cvae"]}. Building CVAE requires raw nUMI counts and corresponding cell-type annotation of scRNA-seq data specified.
    --use_imputation        control whether to perform imputation (true/false). Default value is {default_paramdict["use_imputation"]}. Imputation requires the spot diameter (µm) at higher resolution to be specified.
    --diagnosis             if true, provide more output files related to CVAE building and hyper-parameter selection for diagnosis. Default value is {default_paramdict["diagnosis"]}.
    --verbose               control whether to print more info such as output of each ADMM iteration step during program running (true/false). Default value is {default_paramdict["verbose"]}.
    
    
    ----- Cell-type marker identification options ------
    
    Cell-type specific markers are identified by Differential analysis (DE) across cell-types in reference scRNA-seq data. We also perform cell and/or gene filtering before DE. Each time we ONLY compare the normalized gene expression (raw nUMI counts divided by sequencing depth) one cell-type (1st) vs another one cell-type (2nd) using Wilcoxon Rank Sum Test, then take the UNION of all identified markers for downstream analysis. We filter the marker genes with pre-set thresholds of p value (or FDR), fold change, pct.1 (percentage of cells expressed this marker in 1st cell-type) and pct.2 (percentage of cells expressed this marker in 2nd cell-type). Next we sort the marker genes by p value (or FDR) or fold change, and select the TOP ones. The options related to cell-type marker identification are listed as below:
    
    --use_fdr               whether to use FDR adjusted p value for filtering and sorting. Default value is {default_paramdict["use_fdr"]}, i.e. use FDR adjusted p value. If false orginal p value will be used instead.
    --p_val_cutoff          threshold of p value (or FDR if `--use_fdr` is true) in marker genes filtering. Default value is {default_paramdict["p_val_cutoff"]}, and only genes with p value (or FDR if `--use_fdr` is true) <= {default_paramdict["p_val_cutoff"]} will be kept.
    --fc_cutoff             threshold of fold change (without log transform!) in marker genes filtering. Default value is {default_paramdict["fc_cutoff"]}, and only genes with fold change >= {default_paramdict["fc_cutoff"]} will be kept.
    --pct1_cutoff           threshold of pct.1 (percentage of cells expressed this marker in 1st cell-type) in marker genes filtering. Default value is {default_paramdict["pct1_cutoff"]}, and only genes with pct.1 >= {default_paramdict["pct1_cutoff"]} will be kept.
    --pct2_cutoff           threshold of pct.2 (percentage of cells expressed this marker in 2nd cell-type) in marker genes filtering. Default value is {default_paramdict["pct2_cutoff"]}, and only genes with pct.2 <= {default_paramdict["pct2_cutoff"]} will be kept.
    --sortby_fc             whether to sort marker genes by fold change. Default value is {default_paramdict["sortby_fc"]}, i.e. sort marker genes by fold change then select TOP ones. If false, p value (or FDR if `--use_fdr` is true) will be used to sort marker genes instead.
    --n_marker_per_cmp      number of selected TOP marker genes for each comparison of ONE cell-type against another ONE cell-type using Wilcoxon Rank Sum Test. Default number is {default_paramdict["n_marker_per_cmp"]}. For each comparison, genes passing filtering will be selected first, then these marker genes will be sorted by fold change or p value (or FDR), and finally pick up specified number of genes with TOP ranks. If the number of available genes is less than the specified number, a WARNING will be shown in the program running log file.
    --filter_cell           whether to filter cells with <200 genes for reference scRNA-seq data before differential analysis. Default value is {default_paramdict["filter_cell"]}, i.e. filter cells first.
    --filter_gene           whether to filter genes presented in <10 cells for reference scRNA-seq data and <3 spots for spatial data before differential analysis. Default value is {default_paramdict["filter_gene"]}, i.e. filter genes first.
    
    
    --------------- CVAE related options ---------------
    
    We build Conditional Variational Autoencoder (CVAE) to adjust the platform effect between spatial transcriptomic and scRNA-seq data. The input of CVAE includes scRNA-seq cells, pseudo-spots generated by aggregating randomly selected cells and real spatial spots. To successfully train a neural network model is non-trivial, and model Topology together with most related hyper-parameters have been pre-fixed based on our experiences on analysis of various spatial transcriptomic datasets. The options can be tuned by users are listed as below:
    
    --n_hv_gene             number of highly variable genes identified in reference scRNA-seq data, and these HV genes will be used together with identified cell-type marker genes for building CVAE. Default number is {default_paramdict["n_hv_gene"]}. If the actual number of genes in scRNA-seq data is less than the specified value, all available genes in scRNA-seq data will be used for building CVAE.
    --n_pseudo_spot         maximum number of pseudo-spots generated by randomly combining scRNA-seq cells into one pseudo-spot in CVAE training. Default value is {default_paramdict["n_pseudo_spot"]}.
    --pseudo_spot_min_cell  minimum value of cells in one pseudo-spot when combining cells into pseudo-spots. Default value is {default_paramdict["pseudo_spot_min_cell"]}.
    --pseudo_spot_max_cell  maximum value of cells in one pseudo-spot when combining cells into pseudo-spots. Default value is {default_paramdict["pseudo_spot_max_cell"]}.
    --seq_depth_scaler      a scaler of scRNA-seq sequencing depth to transform CVAE decoded values (sequencing depth normalized gene expressions) back to raw nUMI counts. Default value is {default_paramdict["seq_depth_scaler"]} for all cells.
    --cvae_input_scaler     maximum value of the scaled input for CVAE input layer. Default value is {default_paramdict["cvae_input_scaler"]}, i.e. linearly scale all the sequencing depth normalized gene expressions to range [0, {default_paramdict["cvae_input_scaler"]}].
    --cvae_init_lr          initial learning rate for training CVAE. Default value is {default_paramdict["cvae_init_lr"]}. Although learning rate will decreasing automatically during training, large initial learning rate will cause training failure at the very beginning of training. If loss function value do NOT monotonically decrease, please try smaller initial learning rate.
    --num_hidden_layer      number of hidden layers in encoder and decoder of CVAE. Default value is {default_paramdict["num_hidden_layer"]}. The number of neurons in each hidden layer will be determined automatically.
    --use_batch_norm        whether to use Batch Normalization in CVAE training. Default value is {default_paramdict["use_batch_norm"]}. If true, enables Batch Normalization in CVAE training.
    --cvae_train_epoch      maximum number of training epochs for the CVAE. Default value is {default_paramdict["cvae_train_epoch"]}.
    --use_spatial_pseudo    whether to generate "pseudo-spots" in spatial condition by randomly combining existing spatial spots in CVAE training. Default value is {default_paramdict["use_spatial_pseudo"]}. If true, half of the total number specified by option `n_pseudo_spot` will be created.
    --redo_de               control whether to redo Differential analysis on CVAE transformed scRNA-seq gene expressions to get a new set of marker gene list of cell-types (true/false). Default value is {default_paramdict["redo_de"]}. It's recommended to redo Differential analysis since CVAE transformation may change the marker gene profile of cell-types.
    --seed                  seed value of TensorFlow to control the randomness in building CVAE. Default value is {default_paramdict["seed"]}.
    
    
    -------------- GLRM hyper-parameter related options ---------------
    
    We incorporate adaptive Lasso penalty and graph Laplacian penalty in GLRM, and use the hyper-parameters lambda_r and lambda_g to balance the strength of those two penalties respectively.
    
    --lambda_r              hyper-parameter for adaptive Lasso. Default value is {default_paramdict["lambda_r"]}, i.e. use cross-validation to find the optimal value. The list of lambda_r candidates will has total `lambda_r_range_k` values, and candidate values will be evenly selected on a log scale (geometric progression) from range [`lambda_r_range_min`, `lambda_r_range_max`]. If `lambda_r` is specified as a valid value, then `lambda_r_range_k`, `lambda_r_range_min` and `lambda_r_range_max` will be ignored.
    --lambda_r_range_min    minimum value of the range of lambda_r candidates used for hyper-parameter selection. Default value is {default_paramdict["lambda_r_range_min"]}.
    --lambda_r_range_max    maximum value of the range of lambda_r candidates used for hyper-parameter selection. Default value is {default_paramdict["lambda_r_range_max"]}.
    --lambda_r_range_k      number of lambda_r candidates used for hyper-parameter selection. Default value is {default_paramdict["lambda_r_range_k"]} (including the values of `lambda_r_range_min` and `lambda_r_range_max`).
    --lambda_g              hyper-parameter for graph Laplacian constrain, which depends on the edge weights used in the graph created from the Adjacency Matrix. Default value is {default_paramdict["lambda_g"]}, i.e. use cross-validation to find the optimal value. The list of lambda_g candidates will has total `lambda_g_range_k` values, and candidate values will be evenly selected on a log scale (geometric progression) from range [`lambda_g_range_min`, `lambda_g_range_max`]. If `lambda_g` is specified as a valid value, then `lambda_g_range_k`, `lambda_g_range_min` and `lambda_g_range_max` will be ignored.
    --lambda_g_range_min    minimum value of the range of lambda_g candidates used for hyper-parameter selection. Default value is {default_paramdict["lambda_g_range_min"]}.
    --lambda_g_range_max    maximum value of the range of lambda_g candidates used for hyper-parameter selection. Default value is {default_paramdict["lambda_g_range_max"]}.
    --lambda_g_range_k      number of lambda_g candidates used for hyper-parameter selection. Default value is {default_paramdict["lambda_g_range_k"]} (including the values of `lambda_g_range_min` and `lambda_g_range_max`).
    
    
    -------------- imputation related options ---------------
    
    --diameter              the physical distance (µm) between centers of two neighboring spatial spots. For Spatial Transcriptomics v1.0 technique it's 200 µm. For 10x Genomics Visium technique it's 100 µm. Default value is {default_paramdict["diameter"]}.
    --impute_diameter       the target distance (µm) between centers of two neighboring spatial spots after imputation. Either one number or an array of numbers separated by "," are supported. Default value is {",".join([str(x) for x in default_paramdict["impute_diameter"]])}, corresponding to the low, medium, high resolution for Spatial Transcriptomics v1.0 technique.
    --hole_min_spots        the minimum number of uncaptured spots required to recognize a hole in the tissue map. Holes with a number of spots less than or equal to this threshold in it are treated as if no hole exists and imputation will be performed within the hole. Default value is {default_paramdict["hole_min_spots"]}, meaning single-spot holes are imputed.
    --preserve_shape        whether to maintain the shape of the tissue map during imputation. If true, all border points are retained in imputation to preserve the tissue's original shape, although this may result in an irregular imputed grid. Default value is {default_paramdict["preserve_shape"]}.
''')


def parseOpt():
    '''
    parse the command line parameters and return them as a dict
    
    Parameters
    ----------
    None.

    Returns
    -------
    paramdict : Dict
        parsed command line parameters for model, including:
            spatial_file : full file path of spatial transcriptomic data\n
            ref_file : full file path of scRNA-seq data\n
            ref_celltype_file : full file path of the corresponding cell-type annotation of scRNA-seq data\n
            marker_file : full file path of user curated cell-type marker gene expression\n
            loc_file : full file path of spot locations in spatial transcriptomic data\n
            A_file : full file path Adjacency Matrix of spots in spatial transcriptomic data\n
            n_cores : number of CPU cores used for parallel computing\n
            lambda_r : hyper-parameter for Adaptive Lasso\n
            lambda_g : hyper-parameter for graph weight, affecting the Laplacian Matrix\n
            use_cvae : whether to build CVAE\n
            threshold : threshold for hard thresholding estimated cell-type proportion theta\n
            n_hv_gene : number of highly variable genes for CVAE\n
            n_marker_per_cmp : number of TOP marker genes for each comparison in DE\n
            n_pseudo_spot : number of pseudo-spots\n
            pseudo_spot_min_cell : minimum value of cells in pseudo-spot\n
            pseudo_spot_max_cell : maximum value of cells in pseudo-spot\n
            seq_depth_scaler : a scaler of scRNA-seq sequencing depth\n
            cvae_input_scaler : maximum value of the scaled input for CVAE\n
            cvae_init_lr : initial learning rate for training CVAE\n
            num_hidden_layer : number of hidden layers in encoder and decoder\n
            use_batch_norm : whether to use Batch Normalization\n
            cvae_train_epoch : max number of training epochs for the CVAE\n
            use_spatial_pseudo : whether to generate "pseudo-spots" in spatial condition\n
            redo_de : whether to redo DE after CVAE transformation\n
            seed : seed value for random in building CVAE\n
            diagnosis : True or False, if True save more information to files for diagnosis CVAE and hyper-parameter selection\n
            verbose : True or False, if True print more information during program running\n
            use_fdr : whether to use FDR adjusted p value for filtering and sorting\n
            p_val_cutoff : threshold of p value (or FDR if --use_fdr is true) in marker genes filtering\n
            fc_cutoff : threshold of fold change (without log transform!) in marker genes filtering\n
            pct1_cutoff : threshold of pct.1 in marker genes filtering\n
            pct2_cutoff : threshold of pct.2 in marker genes filtering\n
            sortby_fc : whether to sort marker genes by fold change\n
            filter_cell : whether to filter cells before DE\n
            filter_gene : whether to filter genes before DE\n
            use_imputation : whether to perform imputation\n
            diameter : the physical diameter of spatial spots\n
            impute_diameter : target spot diameter for imputation\n
            hole_min_spots : threshold of number of uncaptured spots to validate holes\n
            preserve_shape : whether to preserve the exact shape of tissue map
    '''
    
    # If there are no parameters, display prompt information and exit
    if len(sys.argv) == 1:
        print('No options exist!')
        print('Use -h or --help for detailed help!')
        sys.exit(1)
        
    # Define command line parameters.
    # The colon (:) after the short option name indicates that the option must have an additional argument
    # The equal sign (=) after the long option name indicates that the option must have an additional argument
    shortargs = 'hq:r:c:m:l:a:o:n:v'
    longargs = ['help', 'query=', 'ref=', 'ref_anno=', 'marker=', 'loc=', 'adjacency=', 'n_cores=', 'lambda_r=', 'lambda_r_range_min=', 'lambda_r_range_max=', 'lambda_r_range_k=', 'lambda_g=', 'lambda_g_range_min=', 'lambda_g_range_max=', 'lambda_g_range_k=', 'use_cvae=', 'threshold=', 'n_hv_gene=', 'n_marker_per_cmp=', 'n_pseudo_spot=', 'pseudo_spot_min_cell=', 'pseudo_spot_max_cell=', 'seq_depth_scaler=', 'cvae_input_scaler=', 'cvae_init_lr=', 'num_hidden_layer=', 'use_batch_norm=', 'cvae_train_epoch=', 'use_spatial_pseudo=', 'redo_de=', 'seed=', 'diagnosis=', 'verbose=', 'use_fdr=', 'p_val_cutoff=', 'fc_cutoff=', 'pct1_cutoff=', 'pct2_cutoff=', 'sortby_fc=', 'filter_cell=', 'filter_gene=', 'use_imputation=', 'diameter=', 'impute_diameter=', 'hole_min_spots=', 'preserve_shape=', 'version']
    
  
    # Parse the command line parameters
    # `sys.argv[0]` is the name of the python script, the rest are parameters
    # `opts` contains the parsed parameter information, `args` contains the remaining parameters that do not conform to the format
    opts, args = getopt(sys.argv[1:], shortargs, longargs)
    
    # If there are remaining parameters that do not conform to the format, display prompt information and exit
    if args:
        print('Invalid options exist!')
        print('Use -h or --help for detailed help')
        sys.exit(1)
        
    # Define a dict type parameter set to make the program more robust
    # a deep copy
    paramdict = copy.deepcopy(default_paramdict)
    
    
    for opt,val in opts:
        
        if opt in ('-h', '--help'):
            usage()
            sys.exit(0)  # exit with success
            
            
        if opt in ('-v', '--version'):
            print(f'SDePER v{cur_version}')
            sys.exit(0)  # exit with success
            
        
        if opt in ('-q', '--query'):
            tmp_file = os.path.join(input_path, val)
            if not os.path.isfile(tmp_file):
                # the input is not a valid existing filename
                raise Exception(f'Invalid input file `{tmp_file}` for spatial transcriptomic data!')
             # Use the `realpath` function to get the real absolute path
            paramdict['spatial_file'] = os.path.realpath(tmp_file)
            continue
        
        
        if opt in ('-r', '--ref'):
            tmp_file = os.path.join(input_path, val)
            if not os.path.isfile(tmp_file):
                # the input is not a valid existing filename
                raise Exception(f'Invalid input file `{tmp_file}` for reference scRNA-seq data!')
            # Use the `realpath` function to get the real absolute path
            paramdict['ref_file'] = os.path.realpath(tmp_file)
            continue
        
        
        if opt in ('-c', '--ref_anno'):
            tmp_file = os.path.join(input_path, val)
            if not os.path.isfile(tmp_file):
                # the input is not a valid existing filename
                raise Exception(f'Invalid input file `{tmp_file}` for cell-type annotation of reference scRNA-seq data!')
            # Use the `realpath` function to get the real absolute path
            paramdict['ref_celltype_file'] = os.path.realpath(tmp_file)
            continue
        
        
        if opt in ('-m', '--marker'):
            tmp_file = os.path.join(input_path, val)
            if not os.path.isfile(tmp_file):
                # the input is not a valid existing filename
                raise Exception(f'Invalid input file `{tmp_file}` for marker gene expression!')
            # Use the `realpath` function to get the real absolute path
            paramdict['marker_file'] = os.path.realpath(tmp_file)
            continue
        
        
        if opt in ('-l', '--loc'):
            tmp_file = os.path.join(input_path, val)
            if not os.path.isfile(tmp_file):
                # the input is not a valid existing filename
                raise Exception(f'Invalid input file `{tmp_file}` for spot location of spatial transcriptomic data!')
            # Use the `realpath` function to get the real absolute path
            paramdict['loc_file'] = os.path.realpath(tmp_file)
            continue
        
        
        if opt in ('-a', '--adjacency'):
            tmp_file = os.path.join(input_path, val)
            if not os.path.isfile(tmp_file):
                # the input is not a valid existing filename
                raise Exception('Invalid input file `{tmp_file}` for adjacency matrix of spatial transcriptomic data!')
            # Use the `realpath` function to get the real absolute path
            paramdict['A_file'] = os.path.realpath(tmp_file)
            continue
        
        
        if opt in ('-n', '--n_cores'):
            try:
                paramdict['n_cores'] = int(float(val))
                
                if paramdict['n_cores'] > os.cpu_count():
                    print(f'WARNING: user set using `{paramdict["n_cores"]}` CPU cores but system only has `{os.cpu_count()}` cores. Reset CPU cores to `{os.cpu_count()}`')
                    paramdict['n_cores'] = os.cpu_count()
                
                if paramdict['n_cores'] < 1:
                    print(f'WARNING: invalid option value `{paramdict["n_cores"]}` for CPU cores! Please use integer which >= 1. Currently CPU cores is set to be default value `{default_paramdict["n_cores"]}`')
                    paramdict['n_cores'] = default_paramdict["n_cores"]
            except:
                print(f'WARNING: unrecognized option value `{val}` for CPU cores! Please use numeric value. Currently CPU cores is set to be default value `{default_paramdict["n_cores"]}`!')
            continue
        
    
        if opt in ('--lambda_r'):
            if val.casefold() == 'none'.casefold():
                paramdict['lambda_r'] = None
                continue
            else:
                try:
                    paramdict['lambda_r'] = float(val)
                    
                    if paramdict['lambda_r'] < 0:
                        print(f'WARNING: negative option value `{paramdict["lambda_r"]}` for lambda_r! Please use non-negative value. Currently lambda_r is set to be default value `{default_paramdict["lambda_r"]}`!')
                        paramdict['lambda_r'] = default_paramdict["lambda_r"]
                except:
                    print(f'WARNING: unrecognized option value `{val}` for lambda_r! Please use numeric value or `none`. Currently lambda_r is set to be default value `{default_paramdict["lambda_r"]}`!')
                continue
            
            
        if opt in ('--lambda_r_range_min'):
            try:
                paramdict['lambda_r_range_min'] = float(val)
                
                if paramdict['lambda_r_range_min'] < 0:
                    print(f'WARNING: negative option value `{paramdict["lambda_r_range_min"]}` for lambda_r_range_min! Please use non-negative value. Currently lambda_r_range_min is set to be value `0`!')
                    paramdict['lambda_r_range_min'] = 0
            except:
                print(f'WARNING: unrecognized option value `{val}` for lambda_r_range_min! Please use numeric value. Currently lambda_r_range_min is set to be default value `{default_paramdict["lambda_r_range_min"]}`!')
            continue
        
        
        if opt in ('--lambda_r_range_max'):
            try:
                paramdict['lambda_r_range_max'] = float(val)
                # checking max value in the final check
            except:
                print(f'WARNING: unrecognized option value `{val}` for lambda_r_range_max! Please use numeric value. Currently lambda_r_range_max is set to be default value `{default_paramdict["lambda_r_range_max"]}`!')
            continue
        
        
        if opt in ('--lambda_r_range_k'):
            try:
                paramdict['lambda_r_range_k'] = int(float(val))
                
                if paramdict['lambda_r_range_k'] < 1:
                    print(f'WARNING: option value `{paramdict["lambda_r_range_k"]}` for lambda_r_range_k < 1! Please use integer which >= 1. Currently lambda_r_range_k is set to be value `1`!')
                    paramdict['lambda_r_range_k'] = 1
            except:
                print(f'WARNING: unrecognized option value `{val}` for lambda_r_range_k! Please use numeric value. Currently lambda_r_range_k is set to be default value `{default_paramdict["lambda_r_range_k"]}`!')
            continue
    
        
        if opt in ('--lambda_g'):
            if val.casefold() == 'none'.casefold():
                paramdict['lambda_g'] = None
                continue
            else:
                try:
                    paramdict['lambda_g'] = float(val)
                    
                    if paramdict['lambda_g'] < 0:
                        print(f'WARNING: negative option value `{paramdict["lambda_g"]}` for lambda_g! Please use non-negative value. Currently lambda_g is set to be default value `{default_paramdict["lambda_g"]}`!')
                        paramdict['lambda_g'] = default_paramdict["lambda_g"]
                except:
                    print(f'WARNING: unrecognized option value `{val}` for lambda_g! Please use numeric value or `none`. Currently lambda_g is set to be default value `None`!')
                continue
        
        
        if opt in ('--lambda_g_range_min'):
            try:
                paramdict['lambda_g_range_min'] = float(val)
                
                if paramdict['lambda_g_range_min'] < 0:
                    print(f'WARNING: negative option value `{paramdict["lambda_g_range_min"]}` for lambda_g_range_min! Please use non-negative value. Currently lambda_g_range_min is set to be value `0`!')
                    paramdict['lambda_g_range_min'] = 0
            except:
                print(f'WARNING: unrecognized option value `{val}` for lambda_g_range_min! Please use numeric value. Currently lambda_g_range_min is set to be default value `{default_paramdict["lambda_g_range_min"]}`!')
            continue
        
        
        if opt in ('--lambda_g_range_max'):
            try:
                paramdict['lambda_g_range_max'] = float(val)
                # checking max value in the final check
            except:
                print(f'WARNING: unrecognized option value `{val}` for lambda_g_range_max! Please use numeric value. Currently lambda_g_range_max is set to be default value `{default_paramdict["lambda_g_range_max"]}`!')
            continue
        
        
        if opt in ('--lambda_g_range_k'):
            try:
                paramdict['lambda_g_range_k'] = int(float(val))
                
                if paramdict['lambda_g_range_k'] < 1:
                    print(f'WARNING: option value `{paramdict["lambda_g_range_k"]}` for lambda_g_range_k < 1! Please use integer which >= 1. Currently lambda_g_range_k is set to be value `1`!')
                    paramdict['lambda_g_range_k'] = 1
            except:
                print(f'WARNING: unrecognized option value `{val}` for lambda_g_range_k! Please use numeric value. Currently lambda_g_range_k is set to be default value `{default_paramdict["lambda_g_range_k"]}`!')
            continue
        
        
        if opt in ('--use_cvae'):
            if val.casefold() == 'true'.casefold():
                paramdict['use_cvae'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['use_cvae'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for use_cvae! Please use string of true or false. Currently use_cvae is set to be default value `{default_paramdict["use_cvae"]}`!')
            continue
        
        
        if opt in ('--threshold'):
            try:
                paramdict['threshold'] = float(val)
                
                if paramdict['threshold'] < 0:
                    print(f'WARNING: negative option value `{paramdict["threshold"]}` for thoreshold! Please use non-negative value. Currently threshold is set to be default value `0`!')
                    paramdict['threshold'] = 0
            except:
                print(f'WARNING: unrecognized option value `{val}` for threshold! Please use numeric value. Currently threshold is set to be default value `{default_paramdict["threshold"]}`!')
            continue
        
        
        if opt in ('--n_hv_gene'):
            try:
                paramdict['n_hv_gene'] = int(float(val))
                
                if paramdict['n_hv_gene'] < 0:
                    print(f'WARNING: negative option value `{paramdict["n_hv_gene"]}` for n_hv_gene! Please use non-negative integer. Currently n_hv_gene is set to be default value `{default_paramdict["n_hv_gene"]}`!')
                    paramdict['n_hv_gene'] = default_paramdict["n_hv_gene"]
            except:
                print(f'WARNING: unrecognized option value `{val}` for n_hv_gene! Please use numeric value. Currently n_hv_gene is set to be default value `{default_paramdict["n_hv_gene"]}`!')
            continue
        
        
        if opt in ('--n_marker_per_cmp'):
            try:
                paramdict['n_marker_per_cmp'] = int(float(val))
                
                if paramdict['n_marker_per_cmp'] <= 0:
                    print(f'WARNING: non-positive option value `{paramdict["n_marker_per_cmp"]}` for n_marker_per_cmp! Please use positve integer. Currently n_marker_per_cmp is set to be default value `{default_paramdict["n_marker_per_cmp"]}`!')
                    paramdict['n_marker_per_cmp'] = default_paramdict["n_marker_per_cmp"]
            except:
                print(f'WARNING: unrecognized option value `{val}` for n_marker_per_cmp! Please use numeric value. Currently n_marker_per_cmp is set to be default value `{default_paramdict["n_marker_per_cmp"]}`!')
            continue
        
        
        if opt in ('--n_pseudo_spot'):
            try:
                paramdict['n_pseudo_spot'] = int(float(val))
                
                if paramdict['n_pseudo_spot'] < 0:
                    print(f'WARNING: negative option value `{paramdict["n_pseudo_spot"]}` for n_pseudo_spot! Please use non-negative integer. Currently n_pseudo_spot is set to be default value `{default_paramdict["n_pseudo_spot"]}`!')
                    paramdict['n_pseudo_spot'] = default_paramdict["n_pseudo_spot"]
            except:
                print(f'WARNING: unrecognized option value `{val}` for n_pseudo_spot! Please use numeric value. Currently n_pseudo_spot is set to be default value `{default_paramdict["n_pseudo_spot"]}`!')
            continue
        
        
        if opt in ('--pseudo_spot_min_cell'):
            try:
                paramdict['pseudo_spot_min_cell'] = int(float(val))
                
                if paramdict['pseudo_spot_min_cell'] < 2:
                    print(f'WARNING: invalid option value `{paramdict["pseudo_spot_min_cell"]}` for pseudo_spot_min_cell! Please use integer which >= 2. Currently pseudo_spot_min_cell is set to be value `2`!')
                    paramdict['pseudo_spot_min_cell'] = 2
            except:
                print(f'WARNING: unrecognized option value `{val}` for pseudo_spot_min_cell! Please use numeric value. Currently pseudo_spot_min_cell is set to be default value `{default_paramdict["pseudo_spot_min_cell"]}`!')
            continue
        
        
        if opt in ('--pseudo_spot_max_cell'):
            try:
                paramdict['pseudo_spot_max_cell'] = int(float(val))
                # checking pseudo_spot_max_cell leaves to the final check
            except:
                print(f'WARNING: unrecognized option value `{val}` for pseudo_spot_max_cell! Please use numeric value. Currently pseudo_spot_max_cell is set to be default value `{default_paramdict["pseudo_spot_max_cell"]}`!')
            continue
        
        
        if opt in ('--seq_depth_scaler'):
            try:
                paramdict['seq_depth_scaler'] = int(float(val))
                
                if paramdict['seq_depth_scaler'] <= 0:
                    print(f'WARNING: non-positive option value `{paramdict["seq_depth_scaler"]}` for seq_depth_scaler! Please use positve integer. Currently seq_depth_scaler is set to be default value `{default_paramdict["seq_depth_scaler"]}`!')
                    paramdict['seq_depth_scaler'] = default_paramdict["seq_depth_scaler"]
            except:
                print(f'WARNING: unrecognized option value `{val}` for seq_depth_scaler! Please use numeric value. Currently seq_depth_scaler is set to be default value `{default_paramdict["seq_depth_scaler"]}`!')
            continue
        
        
        if opt in ('--cvae_input_scaler'):
            try:
                paramdict['cvae_input_scaler'] = int(float(val))
                
                if paramdict['cvae_input_scaler'] <= 0:
                    print(f'WARNING: non-positive option value `{paramdict["cvae_input_scaler"]}` for cvae_input_scaler! Please use positve integer. Currently cvae_input_scaler is set to be default value `{default_paramdict["cvae_input_scaler"]}`!')
                    paramdict['cvae_input_scaler'] = default_paramdict["cvae_input_scaler"]
            except:
                print(f'WARNING: unrecognized option value `{val}` for cvae_input_scaler! Please use numeric value. Currently cvae_input_scaler is set to be default value `{default_paramdict["cvae_input_scaler"]}`!')
            continue
        
        
        if opt in ('--cvae_init_lr'):
            try:
                paramdict['cvae_init_lr'] = float(val)
                
                if paramdict['cvae_init_lr'] <= 0:
                    print(f'WARNING: non-positive option value `{paramdict["cvae_init_lr"]}` for cvae_init_lr! Please use positve value. Currently cvae_init_lr is set to be default value `{default_paramdict["cvae_init_lr"]}`!')
                    paramdict['cvae_init_lr'] = default_paramdict["cvae_init_lr"]
            except:
                print(f'WARNING: unrecognized option value `{val}` for cvae_init_lr! Please use numeric value. Currently cvae_init_lr is set to be default value `{default_paramdict["cvae_init_lr"]}`!')
            continue
        
        
        if opt in ('--num_hidden_layer'):
            try:
                paramdict['num_hidden_layer'] = int(float(val))
                
                if paramdict['num_hidden_layer'] < 1:
                    print(f'WARNING: invalid option value `{paramdict["num_hidden_layer"]}` for num_hidden_layer! Please use integer which >= 1. Currently num_hidden_layer is set to be value `1`!')
                    paramdict['num_hidden_layer'] = 1
            except:
                print(f'WARNING: unrecognized option value `{val}` for num_hidden_layer! Please use numeric value. Currently num_hidden_layer is set to be default value `{default_paramdict["num_hidden_layer"]}`!')
            continue
        
        
        if opt in ('--use_batch_norm'):
            if val.casefold() == 'true'.casefold():
                paramdict['use_batch_norm'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['use_batch_norm'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for use_batch_norm! Please use string of true or false. Currently use_batch_norm is set to be default value `{default_paramdict["use_batch_norm"]}`!')
            continue
        
        
        if opt in ('--cvae_train_epoch'):
            try:
                paramdict['cvae_train_epoch'] = int(float(val))
                
                if paramdict['cvae_train_epoch'] < 30:
                    print(f'WARNING: invalid option value `{paramdict["cvae_train_epoch"]}` for cvae_train_epoch! Please use integer which >= 30. Currently cvae_train_epoch is set to be value `30`!')
                    paramdict['cvae_train_epoch'] = 30
            except:
                print(f'WARNING: unrecognized option value `{val}` for cvae_train_epoch! Please use numeric value. Currently cvae_train_epoch is set to be default value `{default_paramdict["cvae_train_epoch"]}`!')
            continue
        
        
        if opt in ('--use_spatial_pseudo'):
            if val.casefold() == 'true'.casefold():
                paramdict['use_spatial_pseudo'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['use_spatial_pseudo'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for use_spatial_pseudo! Please use string of true or false. Currently use_spatial_pseudo is set to be default value `{default_paramdict["use_spatial_pseudo"]}`!')
            continue
        
        
        if opt in ('--redo_de'):
            if val.casefold() == 'true'.casefold():
                paramdict['redo_de'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['redo_de'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for redo_de! Please use string of true or false. Currently redo_de is set to be default value `{default_paramdict["redo_de"]}`!')
            continue
        
        
        if opt in ('--seed'):
            try:
                paramdict['seed'] = int(float(val))
            except:
                print(f'WARNING: unrecognized option value `{val}` for seed! Please use numeric value. Currently seed is set to be default value `{default_paramdict["seed"]}`!')
            continue
        
        
        if opt in ('--diagnosis'):
            if val.casefold() == 'true'.casefold():
                paramdict['diagnosis'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['diagnosis'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for diagnosis! Please use string of true or false. Currently verbose is set to be default value `{default_paramdict["diagnosis"]}`!')
            continue
        
        
        if opt in ('--verbose'):
            if val.casefold() == 'true'.casefold():
                paramdict['verbose'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['verbose'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for verbose! Please use string of true or false. Currently verbose is set to be default value `{default_paramdict["verbose"]}`!')
            continue
        
        
        if opt in ('--use_fdr'):
            if val.casefold() == 'true'.casefold():
                paramdict['use_fdr'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['use_fdr'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for use_fdr! Please use string of true or false. Currently use_fdr is set to be default value `{default_paramdict["use_fdr"]}`!')
            continue
        
        
        if opt in ('--p_val_cutoff'):
            try:
                paramdict['p_val_cutoff'] = float(val)
                
                if paramdict['p_val_cutoff'] < 0:
                    print(f'WARNING: negative option value `{paramdict["p_val_cutoff"]}` for p_val_cutoff! Please use non-negative value. Currently p_val_cutoff is set to be default value `{default_paramdict["p_val_cutoff"]}`!')
                    paramdict['p_val_cutoff'] = default_paramdict["p_val_cutoff"]
            except:
                print(f'WARNING: unrecognized option value `{val}` for p_val_cutoff! Please use numeric value. Currently p_val_cutoff is set to be default value `{default_paramdict["p_val_cutoff"]}`!')
            continue
        
        
        if opt in ('--fc_cutoff'):
            try:
                paramdict['fc_cutoff'] = float(val)
                
                if paramdict['fc_cutoff'] < 1:
                    print(f'WARNING: option value `{paramdict["fc_cutoff"]}` for fc_cutoff < 1! Please use a value >= 1. Currently fc_cutoff is set to be value `1`!')
                    paramdict['fc_cutoff'] = 1
            except:
                print(f'WARNING: unrecognized option value `{val}` for fc_cutoff! Please use numeric value. Currently fc_cutoff is set to be default value `{default_paramdict["fc_cutoff"]}`!')
            continue
        
        
        if opt in ('--pct1_cutoff'):
            try:
                paramdict['pct1_cutoff'] = float(val)
                
                if paramdict['pct1_cutoff'] > 1:
                    print(f'WARNING: option value `{paramdict["pct1_cutoff"]}` for pct1_cutoff > 1! Please use a value <= 1. Currently pct1_cutoff is set to be default value `{default_paramdict["pct1_cutoff"]}`!')
                    paramdict['pct1_cutoff'] = default_paramdict["pct1_cutoff"]
            except:
                print(f'WARNING: unrecognized option value `{val}` for pct1_cutoff! Please use numeric value. Currently pct1_cutoff is set to be default value `{default_paramdict["pct1_cutoff"]}`!')
            continue
        
        
        if opt in ('--pct2_cutoff'):
            try:
                paramdict['pct2_cutoff'] = float(val)
                
                if paramdict['pct2_cutoff'] < 0:
                    print(f'WARNING: option value `{paramdict["pct2_cutoff"]}` for pct2_cutoff < 0! Please use a value >= 0. Currently pct2_cutoff is set to be default value `{default_paramdict["pct2_cutoff"]}`!')
                    paramdict['pct2_cutoff'] = default_paramdict["pct2_cutoff"]
            except:
                print(f'WARNING: unrecognized option value `{val}` for pct2_cutoff! Please use numeric value. Currently pct2_cutoff is set to be default value `{default_paramdict["pct2_cutoff"]}`!')
            continue
         
        
        if opt in ('--sortby_fc'):
            if val.casefold() == 'true'.casefold():
                paramdict['sortby_fc'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['sortby_fc'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for sortby_fc! Please use string of true or false. Currently sortby_fc is set to be default value `{default_paramdict["sortby_fc"]}`!')
            continue
        
        
        if opt in ('--filter_cell'):
            if val.casefold() == 'true'.casefold():
                paramdict['filter_cell'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['filter_cell'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for filter_cell! Please use string of true or false. Currently filter_cell is set to be default value `{default_paramdict["filter_cell"]}`!')
            continue
        
        
        if opt in ('--filter_gene'):
            if val.casefold() == 'true'.casefold():
                paramdict['filter_gene'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['filter_gene'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for filter_gene! Please use string of true or false. Currently filter_gene is set to be default value `{default_paramdict["filter_gene"]}`!')
            continue
        
        
        if opt in ('--use_imputation'):
            if val.casefold() == 'true'.casefold():
                paramdict['use_imputation'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['use_imputation'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for use_imputation! Please use string of true or false. Currently use_imputation is set to be default value `{default_paramdict["use_imputation"]}`!')
            continue
        
        
        if opt in ('--diameter'):
            try:
                paramdict['diameter'] = int(float(val))
                
                if paramdict['diameter'] <= 0:
                    print(f'WARNING: non-positive option value `{paramdict["diameter"]}` for diameter! Please use positive integer. Currently diameter is set to be default value `{default_paramdict["diameter"]}`!')
                    paramdict['diameter'] = default_paramdict['diameter']
            except:
                print(f'WARNING: unrecognized option value `{val}` for diameter! Please use numeric value. Currently diameter is set to be default value `{default_paramdict["diameter"]}`!')
            continue

        
        if opt in ('--impute_diameter'):
            
            ori_list = val.split(',')
            tmp_list = []
            
            for x in ori_list:
                try:
                    tmp_val = int(float(x.strip()))
                    
                    if tmp_val <= 0:
                        print(f'WARNING: non-positive option value `{tmp_val}` for impute_diameter! Please use positive integer. Currently this value will be ignored!')
                    else:
                        tmp_list.append(tmp_val)
                except:
                    print(f'WARNING: unrecognized option value `{x.strip()}` for impute_diameter! Please use numeric value. Currently this value will be ignored!')
                    
            if len(tmp_list) == 0:
                print(f'WARNING: no valid value can be extracted from option value `{val}` for impute_diameter! Please use one numeric value or an array of numeric values separated by ",". Currently impute_diameter is set to be default value `{",".join([str(x) for x in default_paramdict["impute_diameter"]])}`!')
            else:
                paramdict['impute_stepsize'] = tmp_list
        
            continue
        
        
        if opt in ('--hole_min_spots'):
            try:
                paramdict['hole_min_spots'] = int(float(val))
                
                if paramdict['hole_min_spots'] < 0:
                    print(f'WARNING: negative option value `{paramdict["hole_min_spots"]}` for hole_min_spots! Please use non-negative integer. Currently hole_min_spots is set to be default value `{default_paramdict["hole_min_spots"]}`!')
                    paramdict['hole_threshold'] = default_paramdict['hole_min_spots']
            except:
                print(f'WARNING: unrecognized option value `{val}` for hole_min_spots! Please use numeric value. Currently hole_min_spots is set to be default value `{default_paramdict["hole_min_spots"]}`!')
            continue
        
        
        if opt in ('--preserve_shape'):
            if val.casefold() == 'true'.casefold():
                paramdict['preserve_shape'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['preserve_shape'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for preserve_shape! Please use string of true or false. Currently preserve_shape is set to be default value `{default_paramdict["preserve_shape"]}`!')
            continue
    
    
    # double check all options
    for k,v in paramdict.items():
        if v is None:
            if k == 'spatial_file':
                # spatial file can't to be None
                raise Exception('ERROR: file for spatial transcriptomic data not specified!')
            elif k == 'marker_file':
                # marker file and ref file either one can be None
                if paramdict['ref_file'] is None or paramdict['ref_celltype_file'] is None:
                    raise Exception('ERROR: either file for marker gene expression, or file for reference scRNA-seq data and corresponding cell-type annotation should be specified!')
            elif k == 'ref_file' or k == 'ref_celltype_file':
                if paramdict['marker_file'] is None:
                    raise Exception('ERROR: either file for marker gene expression, or file for reference scRNA-seq data and corresponding cell-type annotation should be specified!')


    # check CVAE
    if paramdict['use_cvae']:
        if paramdict['ref_file'] is None or paramdict['ref_celltype_file'] is None:
            print('WARNING: building CVAE requires reference scRNA-seq data and corresponding cell-type annotation specified! But at least one of them is not specified. use_cvae will be reset to `False`!')
            paramdict['use_cvae'] = False
            
            
    # check number of cells in pseudo-spot
    if paramdict['pseudo_spot_max_cell'] < paramdict['pseudo_spot_min_cell']:
        print(f'WARNING: option pseudo_spot_max_cell `{paramdict["pseudo_spot_max_cell"]}` < pseudo_spot_min_cell `{paramdict["pseudo_spot_min_cell"]}`. Reset pseudo_spot_max_cell to value `{paramdict["pseudo_spot_min_cell"]+1}`')
        paramdict['pseudo_spot_max_cell'] = paramdict['pseudo_spot_min_cell'] + 1
    
    
    # generate candidates for lambda_r
    if paramdict['lambda_r'] is None:
        if paramdict['lambda_r_range_max'] <= paramdict['lambda_r_range_min']:
            print(f'WARNING: option lambda_r_range_max `{paramdict["lambda_r_range_max"]}` <= lambda_r_range_min `{paramdict["lambda_r_range_min"]}`. Just use lambda_r_range_min `{paramdict["lambda_r_range_min"]}` for lambda_r!')
            paramdict['lambda_r'] = paramdict['lambda_r_range_min']
        else:
            paramdict['lambda_r'] = list(np.round(np.geomspace(paramdict['lambda_r_range_min'], paramdict['lambda_r_range_max'], num=paramdict['lambda_r_range_k']), 3))
            if len(paramdict['lambda_r']) == 1:
                paramdict['lambda_r'] = paramdict['lambda_r'][0]
    
    del paramdict['lambda_r_range_min'], paramdict['lambda_r_range_max'], paramdict['lambda_r_range_k']
    
    
    # generate candidates for lambda_g
    if paramdict['lambda_g'] is None:
        if paramdict['lambda_g_range_max'] <= paramdict['lambda_g_range_min']:
            print(f'WARNING: option lambda_g_range_max `{paramdict["lambda_g_range_max"]}` <= lambda_g_range_min `{paramdict["lambda_g_range_min"]}`. Just use lambda_g_range_min `{paramdict["lambda_g_range_min"]}` for lambda_g!')
            paramdict['lambda_g'] = paramdict['lambda_g_range_min']
        else:
            paramdict['lambda_g'] = list(np.round(np.geomspace(paramdict['lambda_g_range_min'], paramdict['lambda_g_range_max'], num=paramdict['lambda_g_range_k']), 3))
            if len(paramdict['lambda_g']) == 1:
                paramdict['lambda_g'] = paramdict['lambda_g'][0]
    
    del paramdict['lambda_g_range_min'], paramdict['lambda_g_range_max'], paramdict['lambda_g_range_k']
    
    
    # check imputation
    if paramdict['use_imputation']:
        # 1st: whether location provided
        if paramdict['loc_file'] is None:
            print('WARNING: perform imputation requires original spot locations, which is not provided. use_imputation will be reset to `False`!')
            paramdict['use_imputation'] = False
        else:
            # 2nd: whether original location of spatial spots provided by x and y coordinates
            tmp_df = pd.read_csv(paramdict['loc_file'], index_col=0)
            # whether column names are x and y
            if not ('x' in tmp_df.columns and 'y' in tmp_df.columns):
                print('WARNING: the column header of spot location csv file must be "x" and "y"! use_imputation will be reset to `False` since can not infer spot physical locations!')
                paramdict['use_imputation'] = False
            else:
                # 3rd: whether diameters in list smaller than the original physical diameter of spatial spots
                tmp_list = []
                for x in set(paramdict['impute_diameter']):
                    if x < paramdict['diameter']:
                        tmp_list.append(x)
                    else:
                        print(f'WARNING: specified spot diameter value `{x}` for imputation >= physical diameter of spatial spots `{paramdict["diameter"]}`. Skip it!')
                paramdict['impute_diameter'] = sorted(tmp_list, reverse=True)
                
                if len(paramdict['impute_diameter']) == 0:
                    print('WARNING: no valid values for imputate_diameter after checking! use_imputation will be reset to `False`!')
                    paramdict['use_imputation'] = False
    
    
    print('\nrunning options:')
    for k,v in paramdict.items():
        print(f'{k}: {v}')
        
    if paramdict['diagnosis']:
        # need to create subfolders first, otherwise got FileNotFoundError
        os.makedirs(diagnosis_path, exist_ok=True)
        
        # save options to file
        with open(os.path.join(diagnosis_path, 'SDePER_settings.txt'), "w") as file:
            for k, v in paramdict.items():
                file.write(f"{k}: {v}\n")

        
    return paramdict