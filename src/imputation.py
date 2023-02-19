#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 07:15:15 2023

@author: hill103

this script perform imputation on cell-types and gene expression of spatial spots
"""



from getopt import getopt
import sys, os
import pandas as pd
import numpy as np
from config import print, input_path, output_path, cur_version
from time import time



def create_grid(mid_loc, mid_nbr_dist, nbr_dist=2000):
    """
    Create a grid table with a smaller spot length based on the original grid table. 10 pixels = 1 µm
    :param mid_loc: the locations of spot centers, a pandas dataframe with two columns "x" and "y", and the index is the spot id
    :param mid_nbr_dist: the distance between two adjacent spots at original resolution
    :param nbr_dist: the distance between two adjacent spots, default is 2000, i.e. 200 µm
    :return: the new map with smaller spot length, a pandas dataframe with two columns "x" and "y"
    """
    xmin = np.min(mid_loc.loc[:, "x"])
    xmax = np.max(mid_loc.loc[:, "x"])
    ymin = np.min(mid_loc.loc[:, "y"])
    ymax = np.max(mid_loc.loc[:, "y"])
    grid_x = np.arange(xmin, xmax, nbr_dist)
    grid_y = np.arange(ymin, ymax, nbr_dist)
    grid_table = pd.DataFrame(columns=["x", "y"])
    if np.max(grid_x)<xmax:
        grid_x = np.append(grid_x, grid_x[-1]+nbr_dist)
    if np.max(grid_y)<ymax:
        grid_y = np.append(grid_y, grid_y[-1]+nbr_dist)
    mid_loc["border"] = False
    x_coord = np.unique(mid_loc.loc[:, "x"])
    y_coord = np.unique(mid_loc.loc[:, "y"])
    for x in x_coord:
        ymin = np.min(mid_loc.loc[mid_loc.loc[:, "x"]==x, "y"])
        ymax = np.max(mid_loc.loc[mid_loc.loc[:, "x"]==x, "y"])
        mid_loc.loc[(mid_loc.loc[:, "x"]==x) & (mid_loc.loc[:, "y"]==ymin), "border"] = True
        mid_loc.loc[(mid_loc.loc[:, "x"]==x) & (mid_loc.loc[:, "y"]==ymax), "border"] = True
    for y in y_coord:
        xmin = np.min(mid_loc.loc[mid_loc.loc[:, "y"]==y, "x"])
        xmax = np.max(mid_loc.loc[mid_loc.loc[:, "y"]==y, "x"])
        mid_loc.loc[(mid_loc.loc[:, "y"]==y) & (mid_loc.loc[:, "x"]==xmin), "border"] = True
        mid_loc.loc[(mid_loc.loc[:, "y"]==y) & (mid_loc.loc[:, "x"]==xmax), "border"] = True   

    inner_loc = mid_loc.loc[mid_loc["border"]==False, :]

    for i in range(len(grid_x)):
        for j in range(len(grid_y)):
            if (np.min((grid_x[i]- inner_loc.loc[:, "x"]) ** 2 + (grid_y[j]- inner_loc.loc[:, "y"]) ** 2) <  (mid_nbr_dist) ** 2) or (np.min((grid_x[i]- mid_loc.loc[:, "x"]) ** 2 + (grid_y[j]- mid_loc.loc[:, "y"]) ** 2) <  ((mid_nbr_dist - nbr_dist)/2) ** 2):
                grid_table = pd.concat([grid_table, pd.DataFrame([[grid_x[i],grid_y[j]]], columns=["x", "y"])], axis=0)         
    grid_table = grid_table.reset_index(drop=True)
    return grid_table


def find_nn(locL, thetaL, locS, thetaS):
    '''
    Find the nearest neighbor of the small spots in the large spots
    :param locL: the location of the large spots, a pandas dataframe with two columns "x" and "y"
    :param thetaL: the theta matrix of the large spots, a pandas dataframe with the index is the spot id and the columns are the cell types
    :param locS: the location of the small spots, a pandas dataframe with two columns "x" and "y"
    :param thetaS: the theta matrix of the small spots, a pandas dataframe with the index is the spot id and the columns are the cell types
    :return: the theta matrix of the small spots, a pandas dataframe with the columns are the cell types which are the same as the nearest large spots
    '''
    global nearest_loc
    nearest_loc = 1
    for locS_col in locS.index:
        nearest_dis = 100000000
        for locL_col in locL.index:
            dis = np.sqrt((locL.loc[locL_col, "x"] - locS.loc[locS_col, "x"])**2 + (locL.loc[locL_col, "y"] - locS.loc[locS_col, "y"])**2)
            if dis < nearest_dis:
                nearest_dis = dis
                nearest_loc = locL_col
        thetaS.iloc[:, locS_col] = thetaL.loc[:, nearest_loc]
    return thetaS


def construct_W(locS, sigma):
    '''
    Construct the Weight matrix
    :param locS: the location of the imputed small spots, a pandas dataframe with two columns "x" and "y"
    :param sigma: the parameter of the Gaussian kernel
    :return: the Weight matrix, a numpy array
    '''
    W = np.zeros((locS.shape[0], locS.shape[0]))
    for i in range(locS.shape[0]):
        for j in range(locS.shape[0]):
            W[i, j] = np.exp(-((locS.loc[i, "x"] - locS.loc[j, "x"])**2 + (locS.loc[i, "y"] - locS.loc[j, "y"])**2)/(2*sigma**2))
    return W


def do_random_walk(theta_hat, loc_hat, n_step=1, sigma=100, lam=0.5, option="general", q=1000):
    '''
    Do the random walk to impute the small spots
    :param theta_hat: the initial estimated theta matrix of the small spots, usually initialized with the nearest neighbor spots of the original map, a pandas dataframe with the columns are the cell types
    :param loc_hat: the location of the imputed small spots, a pandas dataframe with two columns "x" and "y"
    :param n_step: the number of steps of the random walk, an integer
    :param sigma: the parameter of the Gaussian kernel
    :param lam: the parameter of the random walk
    :param option: the option of the random walk, "lazy", "general" or "nn"
    :param q: the window radius parameter of the nearest neighbor random walk
    :return: the final estimated theta matrix of the small spots, a pandas dataframe with the columns are the cell types
    '''
    global theta_hat_final

    def construct_M_mat(theta, sigma, lam):
        '''
        Construct the M matrix
        :param theta: the estimated theta matrix of the small spots, a pandas dataframe with the columns are the cell types
        :param sigma: the parameter of the Gaussian kernel
        :param lam: the weight parameter between "walking" and "standing still"
        :return: the transition M matrix, a numpy array
        '''
        global M
        n_spot = theta.shape[1]
        W = construct_W(loc_hat, sigma)
        if option=="lazy":
            for i in range(W.shape[0]):
                W[i, i] = 0
            D = np.sum(W, axis=1)
            for i in range(W.shape[0]):
                W[i, :] = W[i, :] / D[i]
            M = lam * W + (1-lam) * np.eye(n_spot)
        elif option=="general":
            D = np.sum(W, axis=1)
            for i in range(W.shape[0]):
                W[i, :] = W[i, :] / D[i]
            M = W
        elif option=="nn":
            thres = np.exp((-q**2)/(2*sigma**2))
            W[W<thres] = 0
            D = np.sum(W, axis=1)
            for i in range(W.shape[0]):
                W[i, :] = W[i, :] / D[i]
            M = W
        return M

    M = construct_M_mat(theta_hat, sigma, lam)
    theta_hat_final = theta_hat
    for i in range(n_step):
        theta_hat_final = M @ theta_hat_final.T
    return theta_hat_final


def generate_map(mid_loc_path, celltype_path, mid_nbr_dist, nb_dist, sig=800, q=2000):
    '''
    Generate the imputed map
    :param mid_loc_path: the path of the location of the original large spots, a csv file with two columns "x" and "y"
    :param celltype_path: the path of the cell type proportion of the original large spots, a csv file with the columns are the cell types
    :param mid_nbr_dist: the distance between two adjacent spots at original resolution
    :param nb_dist: the distance between two adjacent imputed small spots
    :param sig: the sigma parameter of the Gaussian kernel
    :param q: the window radius parameter of the nearest neighbor random walk
    :return: theta_final, the final estimated theta matrix of the small spots, a pandas dataframe with the columns are the cell types
             loc_hat, the location of the imputed small spots, a pandas dataframe with two columns "x" and "y"
    '''
    mid_loc = pd.read_csv(mid_loc_path, index_col=0)
    celltype_prop = pd.read_csv(celltype_path, index_col=0)
    mid_loc = mid_loc * mid_nbr_dist
    small_loc = create_grid(mid_loc=mid_loc, mid_nbr_dist=mid_nbr_dist, nbr_dist=nb_dist)
    mid_loc = mid_loc.reset_index(drop=True)
    mid_theta = celltype_prop.reset_index(drop=True).T
    loc_hat = small_loc
    theta_hat = pd.DataFrame(index=loc_hat.index, columns=mid_theta.index).T
    theta_hat = find_nn(mid_loc, mid_theta, loc_hat, theta_hat)
    theta_final = do_random_walk(theta_hat=theta_hat, loc_hat=loc_hat, n_step=1, sigma=sig, lam=0.5, option="nn", q=q)
    #loc_hat.to_csv(os.path.join(output_path, f'impute_diameter_{int(nb_dist/10)}_spot_loc.csv'))
    #theta_final.to_csv(os.path.join(output_path, f'impute_diameter_{int(nb_dist/10)}_spot_celltype_prop.csv'))
    return theta_final, loc_hat


def generate_imputation_expression_map(mid_loc_path, celltype_path, spatial_count_path, mid_nbr_dist, nb_dist, sig=800, q=2000):
    '''
    Generate the imputed expression map. The default sig and q is optimized for original spot diameter 2000 pixel, and 10 pixels = 1 µm
    :param mid_loc_path: the path of the location of the original large spots, a csv file with two columns "x" and "y"
    :param celltype_path: the path of the cell type proportion of the original large spots, a csv file with the columns are the cell types
    :param spatial_count_path: the path of the spatial count of the original large spots, a csv file with the columns are the genes
    :param mid_nbr_dist: the distance between two adjacent spots at original resolution
    :param nb_dist: the distance between two adjacent imputed small spots
    :param sig: the sigma parameter of the Gaussian kernel
    :param q: the window radius parameter of the nearest neighbor random walk
    :return: X_new, the imputed expression map, with rows are genes and columns are the imputed small spots
             loc_hat, the location of the imputed small spots, a pandas dataframe with two columns "x" and "y"
             theta_hat, the final estimated theta matrix of the small spots, a pandas dataframe with the columns are the cell types
    '''
    theta_hat, loc_hat = generate_map(mid_loc_path, celltype_path, mid_nbr_dist, nb_dist, sig, q)
    spatial_count = pd.read_csv(spatial_count_path, index_col=0)
    X = spatial_count / spatial_count.sum(axis=0)
    V = pd.read_csv(celltype_path, index_col=0)
    V = np.matrix(V)
    X = np.matrix(X)
    X = X.T
    V = V.T
    B = X @ np.linalg.pinv(V)
    X_new = B @ theta_hat.T
    X_new.index = spatial_count.columns
    X_new.columns = loc_hat.index
    X_new[X_new<0] = 0
    #X_new.to_csv(os.path.join(output_path, f'impute_diameter_{int(nb_dist/10)}_spot_gene_norm_exp.csv'))
    return loc_hat, theta_hat, X_new


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
runImputation [option][value]...

    -h or --help            print this help messages.
    -v or --version         print version of CVAE-GLRM
    
    
    --------------- Input options -------------------
    
    -q or --query           input csv file of raw nUMI counts of spatial transcriptomic data (spots * genes), with absolute or relative path. Rows as spots and columns as genes. Row header as spot barcodes and column header as gene symbols are both required.
    -l or --loc             input csv file of row/column integer index (x,y) of spatial spots (spots * 2), with absolute or relative path. Rows as spots and columns are coordinates x (column index) and y (row index). Row header as spot barcodes and column header "x","y" are both required. NOTE 1) the column header must be either "x" or "y" (lower case), 2) x and y are integer index (1,2,3,...) not pixels. This spot location file is required for imputation. And the spot order should be consist with row order in spatial data.
    -p or --prop            input csv file of cell-type proportions of spots in spatial transcriptomic data (spots * cell-types), with absolute or relative path. It can be the result from cell-type deconvolution by CVAE-GLRM, or directly provided by user. Rows as spots and columns as cell-types. Row header as spot barcodes and column header as cell-type names are required. And the spot order should be consist with row order in spatial data.


    --------------- Output options -------------------
    
    We do not provide options for renaming output files. All outputs are in the same folder as input files.
    For each specified spot diameter d µm, there are three output files: 1) imputed spot locations "impute_diameter_d_spot_loc.csv", 2) imputed spot cell-type proportions "impute_diameter_d_spot_celltype_prop.csv", 3) imputed spot gene expressions (already normalized by sequencing depth of spots) "impute_diameter_d_spot_gene_norm_exp.csv".
    
    
    -------------- imputation related options ---------------
    
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
            loc_file : full file path of spot locations in spatial transcriptomic data
            prop_file : full file path of cell-type proportions of spots in spatial transcriptomic data
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
    shortargs = 'hq:l:p:v'
    longargs = ['help', 'query=', 'loc=', 'prop=', 'diameter=', 'impute_diameter=', 'version']
    
  
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
    paramdict = {'spatial_file': None, 'loc_file': None, 'prop_file': None,
                 'diameter': 200, 'impute_diameter': [160, 114, 80]}
    
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
        
        
        if opt in ('-l', '--loc'):
            tmp_file = os.path.join(input_path, val)
            if not os.path.isfile(tmp_file):
                # 输入不是一个确实存在的文件名
                raise Exception(f'Invalid input file `{tmp_file}` for spot location of spatial transcriptomic data!')
            # 采用realpath函数，获得真实绝对路径
            paramdict['loc_file'] = os.path.realpath(tmp_file)
            continue
        
        
        if opt in ('-p', '--prop'):
           tmp_file = os.path.join(input_path, val)
           if not os.path.isfile(tmp_file):
               # 输入不是一个确实存在的文件名
               raise Exception(f'Invalid input file `{tmp_file}` for spot cell-type proportion of spatial transcriptomic data!')
           # 采用realpath函数，获得真实绝对路径
           paramdict['prop_file'] = os.path.realpath(tmp_file)
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
            elif k == 'loc_file':
                # loc file can't be None
                raise Exception('ERROR: file for spot location of spatial transcriptomic data not specified!')
            elif k == 'prop_file':
                raise Exception('ERROR: file for spot cell-type proportion of spatial transcriptomic data not specified!')
    
    
    # check imputation
    # 2nd: whether original location of spatial spots provided by x and y coordinates
    tmp_df = pd.read_csv(paramdict['loc_file'], index_col=0)
    # whether column names are x and y
    if not ('x' in tmp_df.columns and 'y' in tmp_df.columns):
        raise Exception('ERROR: the column header of spot location csv file must be "x" and "y"!')
    # 3rd: whether diameters in list smaller than the original physical diameter of spatial spots
    tmp_list = []
    for x in set(paramdict['impute_diameter']):
        if x < paramdict['diameter']:
            tmp_list.append(x)
        else:
            print(f'WARNING: specified spot diameter value `{x}` for imputation >= physical diameter of spatial spots `{paramdict["diameter"]}`. Skip it!')
    paramdict['impute_diameter'] = sorted(tmp_list, reverse=True)
    
    if len(paramdict['impute_diameter']) == 0:
        raise Exception('ERROR: no valid values for imputate_diameter after checking! please specify diameter values smaller than physical diameter!')
    
    
    # check row index of spots whether consistent
    tmp_count = pd.read_csv(paramdict['spatial_file'], index_col=0)
    tmp_loc = pd.read_csv(paramdict['loc_file'], index_col=0)
    tmp_prop = pd.read_csv(paramdict['prop_file'], index_col=0)
    
    assert (tmp_count.index==tmp_loc.index).all(), 'ERROR: order of spot barcode in gene expression and spot location not consistent!'
    assert (tmp_count.index==tmp_prop.index).all(), 'ERROR: order of spot barcode in gene expression and spot proportion not consistent!'
    
    print('\nrunning options:')
    for k,v in paramdict.items():
        print(f'{k}: {v}')
        
    return paramdict
        

def main():
    # run as independent function
    print(f'\nCVAE-GLRM (Conditional Variational Autoencoder - Graph Laplacian Regularized stratified Model) v{cur_version}\n')
    
    start_time = time()

    paramdict = parseOpt()
    
    print('\n\nstart imputation!')
    
    for x in paramdict['impute_diameter']:
        impute_start = time()
        # 1 µm = 10 pixels
        result = generate_imputation_expression_map(paramdict['loc_file'], paramdict['prop_file'], paramdict['spatial_file'], paramdict['diameter']*10, x*10)
        # return imputed spot locations, cell-type proportions and gene expressions
        result[0].to_csv(os.path.join(output_path, f'impute_diameter_{x}_spot_loc.csv'))
        result[1].to_csv(os.path.join(output_path, f'impute_diameter_{x}_spot_celltype_prop.csv'))
        result[2].to_csv(os.path.join(output_path, f'impute_diameter_{x}_spot_gene_norm_exp.csv'))
        print(f'imputation for {x} µm finished. Elapsed time: {(time()-impute_start)/60.0:.2f} minutes')
    
    print(f'\n\nwhole pipeline finished. Total elapsed time: {(time()-start_time)/60.0:.2f} minutes.')

if __name__ == '__main__':
    main()