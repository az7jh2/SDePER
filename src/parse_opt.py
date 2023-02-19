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
from config import print, input_path, cur_version
import numpy as np
import pandas as pd



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
    
    print('''
runCVAEGLRM [option][value]...
    -h or --help            print this help messages.
    -q or --query           input csv file of raw nUMI counts in spatial transcriptomic data (spots * genes), with absolute or relative path. Rows as spots and columns as genes. Row header as spot barcodes and column header as gene symbols are required.
    -r or --ref             input csv file of raw nUMI counts in scRNA-seq data (cells * genes), with absolute or relative path. Rows as cells and columns as genes. Row header as cell barcodes and column header as gene symbols are required.
    -c or --ref_anno        input csv file of cell-type annotations for all cells in scRNA-seq data, with absolute or relative path. Rows as cells and only 1 column as cell-type annotation. Row header as cell barcodes and column header as cell-types are required.
    -m or --marker          input csv file of already curated cell-type marker gene expression (cell-types * genes; already normalized to remove effect of sequencing depth), with absolute or relative path. Rows as cell-types and columns as genes. Row header as cell-type names and column header as gene symbols are required. If marker gene expression is provided, the built-in differential analysis will be skipped and genes from this csv file will be directly used for cell-type deconvolution, as well as CVAE building if needed. If not provided, Wilcoxon rank sum test will be performed to select cell-type marker genes.
    -l or --loc             input csv file of row/column integer index (x,y) of spatial spots (spots * 2), with absolute or relative path. Rows as spots and columns are coordinates x (column index) and y (row index). Row header as spot barcodes and column header "x","y" are both required. NOTE 1) the column header must be either "x" or "y" (lower case), 2) x and y are integer index (1,2,3,...) not pixels. This spot location file is required for imputation.
    -a or --adjacency       input csv file of Adjacency Matrix of spots in spatial transcriptomic data (spots * spots), with absolute or relative path. In the Adjacency Matrix, spots within neighborhood have a value 1, otherwises have a value 0, and diagonal entries are all 0. Row header and column header as spot barcodes are required. And the spot order should be consist with row order in spatial data. Default value is None.
    -n or --n_cores         number of CPU cores used for parallel computing. Default value is 1, i.e. no parallel computing.
    --lambda_r              hyper-parameter for Adaptive Lasso. Default value is None, i.e. use cross-validation to find the optimal value. The list of lambda_r candidates will has total lambda_r_range_k values, and candidate values will be evenly selected on a log scale (geometric progression) from range [lambda_r_range_min, lambda_r_range_max]. If lambda_r is specified as a valid value, then lambda_r_range_k, lambda_r_range_min and lambda_r_range_max will be ignored.
    --lambda_r_range_min    minimum value of the range of lambda_r candidates used for hyper-parameter selection. Default value is 0.1.
    --lambda_r_range_max    maximum value of the range of lambda_r candidates used for hyper-parameter selection. Default value is 100.
    --lambda_r_range_k      number of lambda_r candidates used for hyper-parameter selection. Default value is 8 (including the values of lambda_r_range_min and lambda_r_range_max).
    --lambda_g              hyper-parameter for Graph Laplacian Constrain, which equals the edge weights used in the Graph created from the Adjacency Matrix. Default value is None, i.e. use cross-validation to find the optimal value. The list of lambda_g candidates will has total lambda_g_range_k values, and candidate values will be evenly selected on a log scale (geometric progression) from range [lambda_g_range_min, lambda_g_range_max]. If lambda_g is specified as a valid value, then lambda_g_range_k, lambda_g_range_min and lambda_g_range_max will be ignored.
    --lambda_g_range_min    minimum value of the range of lambda_g candidates used for hyper-parameter selection. Default value is 0.1.
    --lambda_g_range_max    maximum value of the range of lambda_g candidates used for hyper-parameter selection. Default value is 100.
    --lambda_g_range_k      number of lambda_g candidates used for hyper-parameter selection. Default value is 8 (including the values of lambda_g_range_min and lambda_g_range_max).
    --use_cvae              control whether to build Conditional Variational Autoencoder (CVAE) to adjust the platform effect between spatial transcriptomic and scRNA-seq (true/false). Default value is true. Building CVAE requires raw nUMI counts and corresponding cell-type annotation of scRNA-seq data specified.
    --threshold             threshold for hard thresholding the estimated cell-type proportions, i.e. for one spot, estimated cell-type proportions smaller than this threshold value will be set to 0 , then re-normalize all proportions of this spot to sum as 1. Default value is 0, which means no hard thresholding.
    --n_hv_gene             number of highly variable genes identified in scRNA-seq data and then used for building CVAE. Default number is 1,000. If the actual number of genes in scRNA-seq data is less than the specified value, all genes in scRNA-seq data will be used for CVAE.
    --n_marker_per_cmp      number of selected TOP cell-type specified marker genes for each comparison of ONE cell-type against another ONE cell-type using Wilcoxon rank sum test. Default number is 30. For each comparison, genes with a FDR adjusted p value < 0.05 will be selected first, then these marker genes will be sorted by a combined rank of log fold change and pct.1/pct.2, and finally pick up specified number of gene with TOP ranks.
    --pseudo_spot_min_cell  minimum value of cells in one pseudo-spot for building CVAE. Default value is 2.
    --pseudo_spot_max_cell  maximum value of cells in one pseudo-spot for building CVAE. Default value is 8.
    --seq_depth_scaler      a scaler of scRNA-seq sequencing depth to transform sequencing depth normalized gene expressions back to raw nUMI after CVAE transformation. Default value is 10,000 for all cells.
    --cvae_input_scaler     maximum value of the scaled input for CVAE input layer. Default value is 10, i.e. scale all the sequencing depth normalized gene expressions to range [0, 10].
    --cvae_init_lr          initial learning rate for training CVAE. Default value is 0.003. Althoug learning rate will decreasing automatically during training, large initial learning rate will cause training failure at the very beginning of training. If loss function value do NOT monotonically decrease, please try smaller initial learning rate.
    --redo_de               control whether to redo Differential analysis on CVAE transformed scRNA-seq gene expression to get a new set of marker gene list for cell-types (true/false). Default value is true. It's recommended to redo Differential analysis since CVAE transformation may change the marker gene profile of cell-types.
    --seed                  seed value for random in building CVAE. Default value is 383.
    --diagnosis             if true, provide more outputs related to CVAE and hyper-parameter selection for diagnosis. Default value is false.
    --verbose               control whether to print more info of ADMM during program running (true/false). Default value is true.
    -v or --version         print version of CVAE-GLRM
    --use_imputation        control whether to perform imputation (true/false). Default value is false. Imputation requires the spot diameter (µm) at higher resolution to be specified.
    --diameter              the physical diameter (µm) of spatial spots. Default value is 200.
    --impute_diameter       the target spot diameter (µm) during imputation. Either one number or an array of numbers separated by "," are supported. Default value is 160,114,80, corresponding to the low, medium, high resolution.
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
            spatial_file : full file path of spatial transcriptomic data
            ref_file : full file path of scRNA-seq data
            ref_celltype_file : full file path of the corresponding cell-type annotation of scRNA-seq data
            marker_file : full file path of user curated cell-type marker gene expression
            loc_file : full file path of spot locations in spatial transcriptomic data
            A_file : full file path Adjacency Matrix of spots in spatial transcriptomic data
            n_cores : number of CPU cores used for parallel computing
            lambda_r : hyper-parameter for Adaptive Lasso
            lambda_g : hyper-parameter for graph weight, affecting the Laplacian Matrix
            use_cvae : whether to build CVAE
            threshold : threshold for hard thresholding estimated cell-type proportion theta
            n_hv_gene : number of highly variable genes for CVAE
            n_marker_per_cmp : number of TOP marker genes for each comparison in DE
            pseudo_spot_min_cell : minimum value of cells in pseudo-spot
            pseudo_spot_max_cell : maximum value of cells in pseudo-spot
            seq_depth_scaler : a scaler of scRNA-seq sequencing depth
            cvae_input_scaler : maximum value of the scaled input for CVAE
            cvae_init_lr : initial learning rate for training CVAE
            redo_de : whether to redo DE after CVAE transformation
            seed : seed value for random in building CVAE
            diagnosis : True or False, if True save more information to files for diagnosis CVAE and hyper-parameter selection
            verbose : True or False, if True print more information during program running
            use_imputation : whether to perform imputation
            diameter : the physical diameter of spatial spots
            impute_diameter : target spot diameter for imputation
    '''
    
    # 如果没有任何参数，显示提示信息，并退出
    if len(sys.argv) == 1:
        print('No options exist!')
        print('Use -h or --help for detailed help!')
        sys.exit(1)
        
    # 定义命令行参数
    # 短选项名后的冒号(:)表示该选项必须有附加的参数
    # 长选项名后的等号(=)表示该选项必须有附加的参数
    shortargs = 'hq:r:c:m:l:a:o:n:v'
    longargs = ['help', 'query=', 'ref=', 'ref_anno=', 'marker=', 'loc=', 'adjacency=', 'n_cores=', 'lambda_r=', 'lambda_r_range_min=', 'lambda_r_range_max=', 'lambda_r_range_k=', 'lambda_g=', 'lambda_g_range_min=', 'lambda_g_range_max=', 'lambda_g_range_k=', 'use_cvae=', 'threshold=', 'n_hv_gene=', 'n_marker_per_cmp=', 'pseudo_spot_min_cell=', 'pseudo_spot_max_cell=', 'seq_depth_scaler=', 'cvae_input_scaler=', 'cvae_init_lr=', 'redo_de=', 'seed=', 'diagnosis=', 'verbose=', 'use_imputation=', 'diameter=', 'impute_diameter=', 'version']
    
  
    # 解析命令行参数
    # sys.argv[0]为python脚本名，后续全为参数
    # opts为分析出的参数信息，args为不符合格式信息的剩余参数
    opts, args = getopt(sys.argv[1:], shortargs, longargs)
    
    # 如果存在不符合格式信息的剩余参数，显示提示信息，并退出
    if args:
        print('Invalid options exist!')
        print('Use -h or --help for detailed help')
        sys.exit(1)
        
    # 定义dict类型的参数集，使得程序更稳健
    paramdict = {'spatial_file': None, 'ref_file': None, 'ref_celltype_file': None, 'marker_file': None, 'loc_file': None, 'A_file': None, 'n_cores': 1, 'lambda_r': None, 'lambda_r_range_min': 0.1, 'lambda_r_range_max': 100, 'lambda_r_range_k': 8, 'lambda_g': None, 'lambda_g_range_min': 0.1, 'lambda_g_range_max': 100, 'lambda_g_range_k': 8, 'use_cvae': True, 'threshold': 0, 'n_hv_gene': 1000, 'n_marker_per_cmp': 30, 'pseudo_spot_min_cell': 2, 'pseudo_spot_max_cell':8, 'seq_depth_scaler': 10000, 'cvae_input_scaler': 10, 'cvae_init_lr':0.003, 'redo_de': True, 'seed': 383, 'diagnosis': False, 'verbose': True, 'use_imputation': False, 'diameter': 200, 'impute_diameter': [160, 114, 80]}
    
    for opt,val in opts:
        
        if opt in ('-h', '--help'):
            usage()
            sys.exit(0)  # exit with success
            
            
        if opt in ('-v', '--version'):
            print(f'CVAE-GLRM v{cur_version}')
            sys.exit(0)  # exit with success
            
        
        if opt in ('-q', '--query'):
            tmp_file = os.path.join(input_path, val)
            if not os.path.isfile(tmp_file):
                # 输入不是一个确实存在的文件名
                raise Exception(f'Invalid input file `{tmp_file}` for spatial transcriptomic data!')
            # 采用realpath函数，获得真实绝对路径
            paramdict['spatial_file'] = os.path.realpath(tmp_file)
            continue
        
        
        if opt in ('-r', '--ref'):
            tmp_file = os.path.join(input_path, val)
            if not os.path.isfile(tmp_file):
                # 输入不是一个确实存在的文件名
                raise Exception(f'Invalid input file `{tmp_file}` for reference scRNA-seq data!')
            # 采用realpath函数，获得真实绝对路径
            paramdict['ref_file'] = os.path.realpath(tmp_file)
            continue
        
        
        if opt in ('-c', '--ref_anno'):
            tmp_file = os.path.join(input_path, val)
            if not os.path.isfile(tmp_file):
                # 输入不是一个确实存在的文件名
                raise Exception(f'Invalid input file `{tmp_file}` for cell-type annotation of reference scRNA-seq data!')
            # 采用realpath函数，获得真实绝对路径
            paramdict['ref_celltype_file'] = os.path.realpath(tmp_file)
            continue
        
        
        if opt in ('-m', '--marker'):
            tmp_file = os.path.join(input_path, val)
            if not os.path.isfile(tmp_file):
                # 输入不是一个确实存在的文件名
                raise Exception(f'Invalid input file `{tmp_file}` for marker gene expression!')
            # 采用realpath函数，获得真实绝对路径
            paramdict['marker_file'] = os.path.realpath(tmp_file)
            continue
        
        
        if opt in ('-l', '--loc'):
            tmp_file = os.path.join(input_path, val)
            if not os.path.isfile(tmp_file):
                # 输入不是一个确实存在的文件名
                raise Exception(f'Invalid input file `{tmp_file}` for spot location of spatial transcriptomic data!')
            # 采用realpath函数，获得真实绝对路径
            paramdict['loc_file'] = os.path.realpath(tmp_file)
            continue
        
        
        if opt in ('-a', '--adjacency'):
            tmp_file = os.path.join(input_path, val)
            if not os.path.isfile(tmp_file):
                # 输入不是一个确实存在的文件名
                raise Exception('Invalid input file `{tmp_file}` for adjacency matrix of spatial transcriptomic data!')
            # 采用realpath函数，获得真实绝对路径
            paramdict['A_file'] = os.path.realpath(tmp_file)
            continue
        
        
        if opt in ('-n', '--n_cores'):
            try:
                paramdict['n_cores'] = int(float(val))
                
                if paramdict['n_cores'] > os.cpu_count():
                    print(f'WARNING: user set using `{paramdict["n_cores"]}` CPU cores but system only has `{os.cpu_count()}` cores. Reset CPU cores to `{os.cpu_count()}`')
                    paramdict['n_cores'] = os.cpu_count()
                
                if paramdict['n_cores'] < 1:
                    print(f'WARNING: invalid option value `{paramdict["n_cores"]}` for CPU cores! Please use integer which >= 1. Currently CPU cores is set to be default value `1`')
                    paramdict['n_cores'] = 1
            except:
                print(f'WARNING: unrecognized option value `{val}` for CPU cores! Please use numeric value. Currently CPU cores is set to be default value `1`!')
            continue
        
    
        if opt in ('--lambda_r'):
            if val.casefold() == 'none'.casefold():
                paramdict['lambda_r'] = None
                continue
            else:
                try:
                    paramdict['lambda_r'] = float(val)
                    
                    if paramdict['lambda_r'] < 0:
                        print(f'WARNING: negative option value `{paramdict["lambda_r"]}` for lambda_r! Please use non-negative value. Currently lambda_r is set to be default value `None`!')
                        paramdict['lambda_r'] = None
                except:
                    print(f'WARNING: unrecognized option value `{val}` for lambda_r! Please use numeric value or `none`. Currently lambda_r is set to be default value `None`!')
                continue
            
            
        if opt in ('--lambda_r_range_min'):
            try:
                paramdict['lambda_r_range_min'] = float(val)
                
                if paramdict['lambda_r_range_min'] <= 0:
                    print(f'WARNING: non-positive option value `{paramdict["lambda_r_range_min"]}` for lambda_r_range_min! Please use positve value. Currently lambda_r_range_min is set to be default value `0.1`!')
                    paramdict['lambda_r_range_min'] = 0.1
            except:
                print(f'WARNING: unrecognized option value `{val}` for lambda_r_range_min! Please use numeric value. Currently lambda_r_range_min is set to be default value `0.1`!')
            continue
        
        
        if opt in ('--lambda_r_range_max'):
            try:
                paramdict['lambda_r_range_max'] = float(val)
                # checking max value in the final check
            except:
                print(f'WARNING: unrecognized option value `{val}` for lambda_r_range_max! Please use numeric value. Currently lambda_r_range_max is set to be default value `100`!')
            continue
        
        
        if opt in ('--lambda_r_range_k'):
            try:
                paramdict['lambda_r_range_k'] = int(float(val))
                
                if paramdict['lambda_r_range_k'] < 1:
                    print(f'WARNING: option value `{paramdict["lambda_r_range_k"]}` for lambda_r_range_k < 1! Please use integer which >= 1. Currently lambda_r_range_k is set to be value `1`!')
                    paramdict['lambda_r_range_k'] = 1
            except:
                print(f'WARNING: unrecognized option value `{val}` for lambda_r_range_k! Please use numeric value. Currently lambda_r_range_k is set to be default value `8`!')
            continue
    
        
        if opt in ('--lambda_g'):
            if val.casefold() == 'none'.casefold():
                paramdict['lambda_g'] = None
                continue
            else:
                try:
                    paramdict['lambda_g'] = float(val)
                    
                    if paramdict['lambda_g'] < 0:
                        print(f'WARNING: negative option value `{paramdict["lambda_g"]}` for lambda_g! Please use non-negative value. Currently lambda_g is set to be default value `None`!')
                        paramdict['lambda_g'] = None
                except:
                    print(f'WARNING: unrecognized option value `{val}` for lambda_g! Please use numeric value or `none`. Currently lambda_g is set to be default value `None`!')
                continue
        
        
        if opt in ('--lambda_g_range_min'):
            try:
                paramdict['lambda_g_range_min'] = float(val)
                
                if paramdict['lambda_g_range_min'] <= 0:
                    print(f'WARNING: non-positive option value `{paramdict["lambda_g_range_min"]}` for lambda_g_range_min! Please use positve value. Currently lambda_g_range_min is set to be default value `0.1`!')
                    paramdict['lambda_g_range_min'] = 0.1
            except:
                print(f'WARNING: unrecognized option value `{val}` for lambda_g_range_min! Please use numeric value. Currently lambda_g_range_min is set to be default value `0.1`!')
            continue
        
        
        if opt in ('--lambda_g_range_max'):
            try:
                paramdict['lambda_g_range_max'] = float(val)
                # checking max value in the final check
            except:
                print(f'WARNING: unrecognized option value `{val}` for lambda_g_range_max! Please use numeric value. Currently lambda_g_range_max is set to be default value `100`!')
            continue
        
        
        if opt in ('--lambda_g_range_k'):
            try:
                paramdict['lambda_g_range_k'] = int(float(val))
                
                if paramdict['lambda_g_range_k'] < 1:
                    print(f'WARNING: option value `{paramdict["lambda_g_range_k"]}` for lambda_g_range_k < 1! Please use integer which >= 1. Currently lambda_g_range_k is set to be value `1`!')
                    paramdict['lambda_g_range_k'] = 1
            except:
                print(f'WARNING: unrecognized option value `{val}` for lambda_g_range_k! Please use numeric value. Currently lambda_g_range_k is set to be default value `8`!')
            continue
        
        
        if opt in ('--use_cvae'):
            if val.casefold() == 'true'.casefold():
                paramdict['use_cvae'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['use_cvae'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for use_cvae! Please use string of true or false. Currently use_cvae is set to be default value `True`!')
            continue
        
        
        if opt in ('--threshold'):
            try:
                paramdict['threshold'] = float(val)
                
                if paramdict['threshold'] < 0:
                    print(f'WARNING: negative option value `{paramdict["threshold"]}` for thoreshold! Please use non-negative value. Currently threshold is set to be default value `0`!')
                    paramdict['threshold'] = 0
            except:
                print(f'WARNING: unrecognized option value `{val}` for threshold! Please use numeric value. Currently threshold is set to be default value `0`!')
            continue
        
        
        if opt in ('--n_hv_gene'):
            try:
                paramdict['n_hv_gene'] = int(float(val))
                
                if paramdict['n_hv_gene'] < 0:
                    print(f'WARNING: negative option value `{paramdict["n_hv_gene"]}` for n_hv_gene! Please use non-negative integer. Currently n_hv_gene is set to be default value `1,000`!')
                    paramdict['n_hv_gene'] = 1000
            except:
                print(f'WARNING: unrecognized option value `{val}` for n_hv_gene! Please use numeric value. Currently n_hv_gene is set to be default value `1,000`!')
            continue
        
        
        if opt in ('--n_marker_per_cmp'):
            try:
                paramdict['n_marker_per_cmp'] = int(float(val))
                
                if paramdict['n_marker_per_cmp'] <= 0:
                    print(f'WARNING: non-positive option value `{paramdict["n_marker_per_cmp"]}` for n_marker_per_cmp! Please use positve integer. Currently n_marker_per_cmp is set to be default value `30`!')
                    paramdict['n_marker_per_cmp'] = 30
            except:
                print(f'WARNING: unrecognized option value `{val}` for n_marker_per_cmp! Please use numeric value. Currently n_marker_per_cmp is set to be default value `30`!')
            continue
        
        
        if opt in ('--pseudo_spot_min_cell'):
            try:
                paramdict['pseudo_spot_min_cell'] = int(float(val))
                
                if paramdict['pseudo_spot_min_cell'] < 2:
                    print(f'WARNING: invalid option value `{paramdict["pseudo_spot_min_cell"]}` for pseudo_spot_min_cell! Please use integer which >= 2. Currently pseudo_spot_min_cell is set to be default value `2`!')
                    paramdict['pseudo_spot_min_cell'] = 2
            except:
                print(f'WARNING: unrecognized option value `{val}` for pseudo_spot_min_cell! Please use numeric value. Currently pseudo_spot_min_cell is set to be default value `2`!')
            continue
        
        
        if opt in ('--pseudo_spot_max_cell'):
            try:
                paramdict['pseudo_spot_max_cell'] = int(float(val))
                # checking pseudo_spot_max_cell leaves to the final check
            except:
                print(f'WARNING: unrecognized option value `{val}` for pseudo_spot_max_cell! Please use numeric value. Currently pseudo_spot_max_cell is set to be default value `8`!')
            continue
        
        
        if opt in ('--seq_depth_scaler'):
            try:
                paramdict['seq_depth_scaler'] = int(float(val))
                
                if paramdict['seq_depth_scaler'] <= 0:
                    print(f'WARNING: non-positive option value `{paramdict["seq_depth_scaler"]}` for seq_depth_scaler! Please use positve integer. Currently seq_depth_scaler is set to be default value `10,000`!')
                    paramdict['seq_depth_scaler'] = 10000
            except:
                print(f'WARNING: unrecognized option value `{val}` for seq_depth_scaler! Please use numeric value. Currently seq_depth_scaler is set to be default value `10,000`!')
            continue
        
        
        if opt in ('--cvae_input_scaler'):
            try:
                paramdict['cvae_input_scaler'] = int(float(val))
                
                if paramdict['cvae_input_scaler'] <= 0:
                    print(f'WARNING: non-positive option value `{paramdict["cvae_input_scaler"]}` for cvae_input_scaler! Please use positve integer. Currently cvae_input_scaler is set to be default value `10`!')
                    paramdict['cvae_input_scaler'] = 10
            except:
                print(f'WARNING: unrecognized option value `{val}` for cvae_input_scaler! Please use numeric value. Currently cvae_input_scaler is set to be default value `10`!')
            continue
        
        
        if opt in ('--cvae_init_lr'):
            try:
                paramdict['cvae_init_lr'] = float(val)
                
                if paramdict['cvae_init_lr'] <= 0:
                    print(f'WARNING: non-positive option value `{paramdict["cvae_init_lr"]}` for cvae_init_lr! Please use positve value. Currently cvae_init_lr is set to be default value `0.003`!')
                    paramdict['cvae_init_lr'] = 0.003
            except:
                print(f'WARNING: unrecognized option value `{val}` for cvae_init_lr! Please use numeric value. Currently cvae_init_lr is set to be default value `0.003`!')
            continue
        
        
        if opt in ('--redo_de'):
            if val.casefold() == 'true'.casefold():
                paramdict['redo_de'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['redo_de'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for redo_de! Please use string of true or false. Currently redo_de is set to be default value `True`!')
            continue
        
        
        if opt in ('--seed'):
            try:
                paramdict['seed'] = int(float(val))
            except:
                print(f'WARNING: unrecognized option value `{val}` for seed! Please use numeric value. Currently seed is set to be default value `383`!')
            continue
        
        
        if opt in ('--diagnosis'):
            if val.casefold() == 'true'.casefold():
                paramdict['diagnosis'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['diagnosis'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for diagnosis! Please use string of true or false. Currently verbose is set to be default value `False`!')
            continue
        
        
        if opt in ('--verbose'):
            if val.casefold() == 'true'.casefold():
                paramdict['verbose'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['verbose'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for verbose! Please use string of true or false. Currently verbose is set to be default value `True`!')
            continue
        
        
        if opt in ('--use_imputation'):
            if val.casefold() == 'true'.casefold():
                paramdict['use_imputation'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['use_imputation'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for use_imputation! Please use string of true or false. Currently use_imputation is set to be default value `False`!')
            continue
        
        
        if opt in ('--diameter'):
            try:
                paramdict['diameter'] = int(float(val))
                
                if paramdict['diameter'] <= 0:
                    print(f'WARNING: option value `{paramdict["diameter"]}` for diameter <= 0! Please use integers > 0. Currently diameter is set to be default value `200`!')
                    paramdict['diameter'] = 200
            except:
                print(f'WARNING: unrecognized option value `{val}` for diameter! Please use numeric value. Currently diameter is set to be default value `200`!')
            continue
        
        
        if opt in ('--impute_diameter'):
            
            ori_list = val.split(',')
            tmp_list = []
            
            for x in ori_list:
                try:
                    tmp_val = int(float(x.strip()))
                    
                    if tmp_val <= 0:
                        print(f'WARNING: option value `{tmp_val}` for imputate_diameter <= 0! Please use integers > 0. Currently this value will be ignored!')
                    else:
                        tmp_list.append(tmp_val)
                except:
                    print(f'WARNING: unrecognized option value `{x.strip()}` for imputate_diameter! Please use numeric value. Currently this value will be ignored!')
                    
            if len(tmp_list) == 0:
                print(f'WARNING: no valid value can be extracted from option value `{val}` for imputate_diameter! Please use one numeric value or an array of numeric values separated by ",". Currently impute_diameter is set to be default value `160,114,80`!')
            else:
                paramdict['impute_diameter'] = tmp_list
        
            continue
    
    
    # 检查参数是否齐全
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
        
    return paramdict