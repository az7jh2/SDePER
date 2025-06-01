#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 21:43:08 2022

@author: hill103

this script stores functions to build a CVAE for platform effect adjustment
"""



import os
from config import print, diagnosis_path
import numpy as np
import pandas as pd
import umap
from utils import read_spatial_data, read_scRNA_data, run_DE
from time import time
import random
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import scanpy as sc
sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Input, Dense, Activation, concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import set_random_seed

# dealing with the keras symbolic tensor error
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()



def celltype2props(celltype_anno, celltype_order):
    '''
    calculate cell-type proportions matrix given one cell-type annotation

    Parameters
    ----------
    celltype_anno : dataframe
        cell-type annotations. Only 1 column named <celltype>
    celltype_order : list
        already sorted unique cell-types. Its order matters, and will be the order in cell-type proportions (columns) and cell-type gene expression profile (rows)

    Returns
    -------
    celltype_prop : dataframe
        cell-type proportions, columns are already sorted cell-types
    '''
    
    celltype_stats = []
    for i in celltype_anno.index:
        celltype_stats.append({celltype_anno.loc[i, 'celltype']: 1})
    # transform to matrix
    celltype_prop = pd.DataFrame(celltype_stats, columns=celltype_order, index=celltype_anno.index)
    celltype_prop.fillna(0, inplace=True)
    # calculate cell-type proportions, divides each element in a row by the sum of that row
    celltype_prop = celltype_prop.div(celltype_prop.sum(axis=1), axis=0)
    
    return celltype_prop



def transferProps(query, ref, ref_props, n_neighbors=10, sigma=1, use_embedding='PCA', pca_dimension=None):
    '''
    transfer cell-type proportions by select K Nearest Neighbors in ref and take Gaussian weighted average of ref proportions

    Parameters
    ----------
    query : 2-D numpy matrix
        encoder embeddings of spatial spots (spots * latent layer neurons).
    ref : 2-D numpy matrix
        encoder embeddings of scRNA-seq cells and pseudo-spots in scRNA-seq condition (cells+pseudo-spots * latent layer neurons).
    ref_props : 2-D numpy matrix
        cell-type proportion matrix of scRNA-seq cells and pseudo-spots (cells+pseudo-spots * cell-types).
    n_neighbors : int, optional
        Number of neighbors to use. The default is 10.
    sigma : float, optional
        Standard deviation for the Gaussian weighting function. The default is 1.
    use_embedding : str, optional
        which embedding to use, either PCA, UMAP or none. The default is PCA.
    pca_dimension : int, optinal
        specify the number of dimensions for PCA reduction. If set to None, the reduced dimension will be one-third of the input dimension.

    Returns
    -------
    query_props : 2-D numpy matrix
        cell-type proportion matrix for spatial spots.
    '''
    
    assert query.shape[1] == ref.shape[1]
    n_celltype = ref_props.shape[1]
    
    if (query.shape[1]<=2) and (use_embedding!='none'):
        print(f'WARNING: original latent space dimension {query.shape[1]} <= 2, no need to use {use_embedding} embedding!')
        use_embedding = 'none'
    
    if use_embedding == 'PCA':
        # first take a PCA to avoid Curse of Dimensionality
        # we perform PCA without any normalization and scaling, and reduce the dimensionality to one-third of the original dimensions
        if pca_dimension is None:
            # orginal dimension: 3*#cell-types, reduced dimension: #cell-types
            reduced_pca_dimension = int(query.shape[1] / 3)
        else:
            reduced_pca_dimension = int(pca_dimension)
            
        principal_components = PCA(n_components=reduced_pca_dimension).fit_transform(np.vstack((query, ref)))
        # Split the principal components back into query and ref
        query_pc = principal_components[:query.shape[0], :]
        ref_pc = principal_components[query.shape[0]:, :]
    
    elif use_embedding == 'UMAP':
        all_umap = umap.UMAP(random_state=42).fit_transform(np.vstack((query, ref)))
        # Split the principal components back into query and ref
        query_pc = all_umap[:query.shape[0], :]
        ref_pc = all_umap[query.shape[0]:, :]
        
    elif use_embedding == 'none':
        query_pc = query
        ref_pc = ref
    
    else:
        raise Exception(f'unknow embedding {use_embedding}')
    
    print(f'embedding dimension: {query_pc.shape[1]}')
    
    # perform KNN on query data on reduced dimension
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(ref_pc)
    # find nearest neighbors
    distances, indices = nbrs.kneighbors(query_pc)
    # Calculate Gaussian weighted averages
    query_props = np.zeros((query.shape[0], ref_props.shape[1]))

    for i, (dists, inds) in enumerate(zip(distances, indices)):
        # Compute Gaussian weights
        weights = np.exp(-dists**2 / (2 * sigma**2))
        weights /= weights.sum()
        # Multiply weights with corresponding rows in ref_props (element-wise multiplication)
        # first inserts a new axis along the second dimension (column) of weights, changing the shape of weights from (K,) to (K, 1).
        # then perform multiplication with shape (K, m) * (K, 1), numpy broadcasting expands weights along the column dimension (1 dimension), matching the column size of ref_props, and perform element-wise multiplication
        weighted_props = ref_props[inds] * weights[:, np.newaxis]
        # calculate weighted average for the current query row; axis=0 sum across columns
        avg_props = np.sum(weighted_props, axis=0)
        # normalize the proportions to sum to 1
        query_props[i] = avg_props / np.sum(avg_props)
        
        # NOTE if all proportions are 0 due to very small weights, the initial guess will be all NaN
        if pd.isnull(query_props[i]).any():
            # replace it as a vector with all elements identical
            query_props[i, :] = np.full((n_celltype,), 1.0/n_celltype)
            continue
        
        # post-process theta to set theta<0.01 as 0 then re-normalize remaining theta to sum to 1
        tmp_ind = query_props[i, :] < 0.01
        
        if tmp_ind.all():
            # all elements < threashold, just leave it unchanged
            continue
        
        if tmp_ind.any():
            query_props[i, tmp_ind] = 0
            query_props[i, :] = query_props[i, :] / np.sum(query_props[i, :])
        
    return query_props



def generate_pseudo_spots(exp_df, celltype_anno, n_spot, celltype_order, pseudo_spot_min_cell, pseudo_spot_max_cell):
    '''
    generate pseudo-spots for CVAE training by randomly combining scRNA-seq cells
    
    UPDATE:
    
    now we separate the pseudo-spots and all scRNA-seq cells, i.e. we DO NOT add all cells to the end of the dataframe as special pseudo-spots with only one cell
    
    if n_spot=0, i.e. no pseudo-spots, then return an empty dataframe

    Parameters
    ----------
    exp_df : dataframe
        normalized gene expression (cell * genes)
    celltype_anno : dataframe or None
        cell-type annotations for cells in scRNA-seq data. Only 1 column named <celltype>
    n_spot : int
        number of pseudo spots need to be generated, including training and validation set
    celltype_order : list
        already sorted unique cell-types. Its order matters, and will be the order in cell-type proportions (columns) and cell-type gene expression profile (rows)
    pseudo_spot_min_cell : int
        minimum value of cells in pseudo-spot
    pseudo_spot_max_cell : int
        maximum value of cells in pseudo-spot

    Returns
    -------
    pseudo_spots_df : dataframe
        pseudo-spot gene expression (pseudo-spots * genes; NO original cells included)
    pseudo_spots_celltype_prop : dataframe
        pseudo-spot cell-type proportions (pseudo-spots * cell-types; NO original cells included)
    n_cell_in_spot : list
        number of cells/spots in pseudo-spots (NO original cells included)
    '''
    
    if n_spot == 0:
        return pd.DataFrame(columns=exp_df.columns), pd.DataFrame(columns=celltype_order), []
    
    pseudo_spots = []
    celltype_stats = []
    n_cell_in_spot = []
    
    n_cell_list = list(range(pseudo_spot_min_cell, pseudo_spot_max_cell+1))
    # cell barcode separated by cell-types
    type_cell_index = dict()
    for one_celltype in celltype_order:
        type_cell_index[one_celltype] = celltype_anno[celltype_anno['celltype']==one_celltype].index.to_list()
    
    
    # though it's possible to use multiprocessing to generate pseudo-spots parallelly, the big dataframe need to be shared across all subprocesses, and it may not be a good idea to share objects in multiprocessing as it may cause unknown problems. And the performance benefits by multiprocessing many not be such large
    # so considering the safety and performance benefits, just keep the simplest way to generate pseudo-spots one-by-one
    # to reduce randomness, pre-set the seed value for random
    random.seed(138)
    
    # Set to track the milestones to print
    milestones = set(round((n_spot / 10) * i) for i in range(1, 11))
    # Ensure that no milestone can exceed the total number of spots
    milestones = {m for m in milestones if m <= n_spot}

    for i in range(n_spot):
        
        if i+1 in milestones:
            # Calculate the current percentage, ensure it does not exceed 100%
            cur_progress = min(round((i+1) / n_spot, 2), 1)
            print(f'{cur_progress:.0%}...', end='')

        # first determine how many cells in this pseudo-spot
        this_num = random.sample(n_cell_list, 1)[0]
        n_cell_in_spot.append(this_num)
        
        this_cells = []
        for j in range(this_num):
            # select one cell-type
            selected_celltype = random.sample(celltype_order, 1)[0]
            # from this selected cell-type, randomly select one cell belong to that cell-type
            this_cells.append(random.sample(type_cell_index[selected_celltype], 1)[0])
        
        # take average of selected cells
        pseudo_spots.append(exp_df.loc[this_cells].mean(axis=0))
        
        # count the celltype of selected cells
        celltype_stats.append(celltype_anno.loc[this_cells, 'celltype'].value_counts().to_dict())
    
    # make below prints in a newline
    print('\n')

    # Build pseudo-spots dataframe
    # First n_valid_spot spots are used for validation, rest spots are used for training
    pseudo_spots_df = pd.concat(pseudo_spots, axis=1).transpose()
    pseudo_spots_df.reset_index(inplace=True, drop=True)
    pseudo_spots_df.index = ['scrna_pseudo' + str(idx) for idx in pseudo_spots_df.index]
    
    pseudo_spots_celltype_prop = pd.DataFrame(celltype_stats, columns=celltype_order, index=pseudo_spots_df.index)
    pseudo_spots_celltype_prop.fillna(0, inplace=True)
    # calculate cell-type proportions
    pseudo_spots_celltype_prop = pseudo_spots_celltype_prop.div(pseudo_spots_celltype_prop.sum(axis=1), axis=0)
    
    #import gc
    #import psutil
    #print(f'before gc and del variable RAM usage: {psutil.Process().memory_info().rss/1024**2:.2f} MB')
    #del pseudo_spots, celltype_stats
    #print(f'del variable without gc RAM usage: {psutil.Process().memory_info().rss/1024**2:.2f} MB')
    #gc.collect()
    #print(f'after gc RAM usage: {psutil.Process().memory_info().rss/1024**2:.2f} MB')
    
    return pseudo_spots_df, pseudo_spots_celltype_prop, n_cell_in_spot



def combine_spatial_spots(exp_df, n_spot, pseudo_spot_min_cell, pseudo_spot_max_cell):
    '''
    we also generate "pseudo-spots" by combining spatial spots
    
    here we do not know the true cell-type proportions in spatial spots, so we also do not know the proportions in generated "pseudo-spots". We just ignore it, and only use the expressions for CVAE training

    Parameters
    ----------
    exp_df : dataframe
        normalized gene expression (spots * genes)
    n_spot : int
        number of pseudo spots need to be generated, including training and validation set
    pseudo_spot_min_cell : int
        minimum value of cells in pseudo-spot
    pseudo_spot_max_cell : int
        maximum value of cells in pseudo-spot

    Returns
    -------
    pseudo_spots_df : dataframe
        pseudo-spot gene expression (pseudo-spots * genes; NO original cells included)
    '''
    if n_spot == 0:
        return pd.DataFrame(columns=exp_df.columns)
    
    pseudo_spots = []
    n_cell_list = list(range(pseudo_spot_min_cell, pseudo_spot_max_cell+1))
    spot_index = exp_df.index.to_list()
    
    # to reduce randomness, pre-set the seed value for random
    random.seed(154)
    
    # Set to track the milestones to print
    milestones = set(round((n_spot / 10) * i) for i in range(1, 11))
    # Ensure that no milestone can exceed the total number of spots
    milestones = {m for m in milestones if m <= n_spot}
    
    for i in range(n_spot):
        
        if i+1 in milestones:
            # Calculate the current percentage, ensure it does not exceed 100%
            cur_progress = min(round((i+1) / n_spot, 2), 1)
            print(f'{cur_progress:.0%}...', end='')
            
        # first determine how many spots in this pseudo-spot
        this_num = random.sample(n_cell_list, 1)[0]
        # then randomly select these number of spots and take average
        pseudo_spots.append(exp_df.loc[random.sample(spot_index, this_num)].mean(axis=0))
    
    # make below prints in a newline
    print('\n')
    
    # Build pseudo-spots dataframe
    pseudo_spots_df = pd.concat(pseudo_spots, axis=1).transpose()
    pseudo_spots_df.reset_index(inplace=True, drop=True)
    pseudo_spots_df.index = ['spatial_pseudo' + str(idx) for idx in pseudo_spots_df.index]
    
    return pseudo_spots_df



def augment_sc(exp_df, celltype_anno, target_count, pseudo_spot_min_cell, pseudo_spot_max_cell):
    '''
    augment single cells and balance #cells of cell types
    
    generate pseudo-spots by randomly combining scRNA-seq cells within the same cell type
    
    original single cells are put at the end

    Parameters
    ----------
    exp_df : dataframe
        normalized gene expression (cell * genes)
    celltype_anno : dataframe or None
        cell-type annotations for cells in scRNA-seq data. Only 1 column named <celltype>
    target_count : int
        target number of cells per cell type
    pseudo_spot_min_cell : int
        minimum value of cells in pseudo-spot
    pseudo_spot_max_cell : int
        maximum value of cells in pseudo-spot

    Returns
    -------
    pseudo_spots_df : dataframe
        pseudo-spot gene expression (pseudo-spots * genes; original cells included first)
    pseudo_spots_celltype_prop : dataframe
        pseudo-spot cell-type proportions (pseudo-spots * cell-types; original cells included first)
    n_cell_in_spot : list
        number of cells/spots in pseudo-spots (original cells included first)
    '''
    
    pseudo_spots = []
    celltype_stats = []
    n_cell_in_spot = []
    
    n_cell_list = list(range(pseudo_spot_min_cell, pseudo_spot_max_cell+1))
    n_cell_per_ct = celltype_anno.celltype.value_counts().to_dict()
    all_cts = n_cell_per_ct.keys()
    
    # cell barcode separated by cell-types
    type_cell_index = dict()
    for one_celltype in all_cts:
        type_cell_index[one_celltype] = celltype_anno[celltype_anno['celltype']==one_celltype].index.to_list()
    
    
    # though it's possible to use multiprocessing to generate pseudo-spots parallelly, the big dataframe need to be shared across all subprocesses, and it may not be a good idea to share objects in multiprocessing as it may cause unknown problems. And the performance benefits by multiprocessing many not be such large
    # so considering the safety and performance benefits, just keep the simplest way to generate pseudo-spots one-by-one
    # to reduce randomness, pre-set the seed value for random
    random.seed(169)
    
    for i, one_celltype in enumerate(all_cts):
        
        print(f'{i/len(all_cts):.0%}...', end='')
        
        for j in range(target_count-n_cell_per_ct[one_celltype]):
            # first determine how many cells in this pseudo-spot
            this_num = random.sample(n_cell_list, 1)[0]
            n_cell_in_spot.append(this_num)
            # from this selected cell-type, randomly select needed cells belong to that cell-type
            if n_cell_per_ct[one_celltype] < this_num:
                # sample with replace
                this_cells = random.choices(type_cell_index[one_celltype], k=this_num)
            else:
                # sample without replace
                this_cells = random.sample(type_cell_index[one_celltype], this_num)

            # take average of selected cells
            pseudo_spots.append(exp_df.loc[this_cells].mean(axis=0))
            celltype_stats.append(one_celltype)
    
    # make below prints in a newline
    print('100%')

    # Build pseudo-spots dataframe
    # last X spots are used for validation, rest spots are used for training
    pseudo_spots_df = pd.concat(pseudo_spots, axis=1).transpose()
    pseudo_spots_df.reset_index(inplace=True, drop=True)
    pseudo_spots_df.index = ['scrna_augment' + str(idx) for idx in pseudo_spots_df.index]
    
    pseudo_spots_celltypes = pd.DataFrame(celltype_stats, columns=['celltype'], index=pseudo_spots_df.index)
    n_cell_df = pd.DataFrame(n_cell_in_spot, columns=['ncell'], index=pseudo_spots_df.index)
    
    # shuffle all rows
    tmp_index = pseudo_spots_df.index.to_list()
    random.shuffle(tmp_index)
    pseudo_spots_df = pseudo_spots_df.loc[tmp_index].copy()
    pseudo_spots_celltypes = pseudo_spots_celltypes.loc[tmp_index].copy()
    n_cell_df = n_cell_df.loc[tmp_index].copy()
    
    # combine original single cells at the end
    combined_exp = pd.concat([pseudo_spots_df, exp_df], axis=0)
    combined_ct = pd.concat([pseudo_spots_celltypes, celltype_anno], axis=0)
    combined_n_cell = n_cell_df.ncell.to_list() + [1] * exp_df.shape[0]
    
    return combined_exp, combined_ct, combined_n_cell



def CVAE_keras_model(p, p_cond, latent_dim, p_encoder_lst, p_decoder_lst, hidden_act='elu', output_act='relu', use_batch_norm=True, cvae_init_lr=0.01):
    '''
    define a standard CVAE model based on Keras
    need to build a decoder separately as can not extract it from the whole model

    Parameters
    ----------
    p : int
        number of nodes in input layer
    p_cond : int
        number of conditional nodes in input layer
    latent_dim : int
        number of nodes in latent space
    p_encoder_lst : list of integers
        including number of nodes in each hidden layer of encoder, the length of list is the number of hidden layers
    p_decoder_lst : list of integers
        including number of nodes in each hidden layer of decoder, the length of list is the number of hidden layers
    hidden_act : string, optional
        activation function of hidden layers. Default is elu function
    output_act : string, optional
        activation function of output layer. Default is relu function
    use_batch_norm : bool, optional
        whether to use batch normalization. Default if True, i.e. use batch normalization
    cvae_init_lr : float
        initial learning rate for training CVAE
    
    Returns
    -------
    cvae : Keras model
        CVAE model. Encoder can be extracted from it
    decoder : Keras model
        corresponding decoder, reset its weights after CVAE training
    '''

    # Functions for CVAE
    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    def sampling(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    # the Keras framework support custom loss returning one value, but more correct way is returning an array of losses (one of sample in the input batch), and the reducing the done by Keras
    # when return an array of losses, we can also handle the specified sample_weight
    def KL_loss(obs, pred):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return kl_loss
    
    def reconstruction_loss(obs, pred):
        return K.sum(K.square(obs - pred), axis=-1)
    
    def vae_loss(obs, pred, C=1):
        # currently we set weight C=1
        return C*reconstruction_loss(obs, pred) + KL_loss(obs, pred)
    
    
    # Build encoder model
    X = Input(shape=(p,), name='encoder_input')
    cond = Input(shape=(p_cond,), name='cond_input')
    encoder_inputs = concatenate([X, cond])
    
    # add hidden layers
    for i, num in enumerate(p_encoder_lst):
        if i == 0:
            # 1st hidden layer
            if use_batch_norm:
                encoder_hidden = Dense(num, use_bias=False, name=f'encoder_layer{i}_w')(encoder_inputs)
                encoder_hidden = BatchNormalization(name=f'encoder_layer{i}_BN')(encoder_hidden)
                encoder_hidden = Activation(hidden_act, name=f'encoder_layer{i}_act')(encoder_hidden)
            else:
                encoder_hidden = Dense(num, use_bias=True, name=f'encoder_layer{i}_w')(encoder_inputs)
                encoder_hidden = Activation(hidden_act, name=f'encoder_layer{i}_act')(encoder_hidden)
        else:
            if use_batch_norm:
                encoder_hidden = Dense(num, use_bias=False, name=f'encoder_layer{i}_w')(encoder_hidden)
                encoder_hidden = BatchNormalization(name=f'encoder_layer{i}_BN')(encoder_hidden)
                encoder_hidden = Activation(hidden_act, name=f'encoder_layer{i}_act')(encoder_hidden)
            else:
                encoder_hidden = Dense(num, use_bias=True, name=f'encoder_layer{i}_w')(encoder_hidden)
                encoder_hidden = Activation(hidden_act, name=f'encoder_layer{i}_act')(encoder_hidden)
    
    
    # latent layer of z_mean and z_log_var
    if use_batch_norm:
        z_mean_pre = Dense(latent_dim, use_bias=False)(encoder_hidden)
        z_mean_pre = BatchNormalization()(z_mean_pre)
        z_mean = Activation('linear', name='z_mean')(z_mean_pre)
        
        z_log_var_pre = Dense(latent_dim, use_bias=False)(encoder_hidden)
        z_log_var_pre = BatchNormalization()(z_log_var_pre)
        z_log_var = Activation('linear', name='z_log_var')(z_log_var_pre)
    else:
        z_mean_pre = Dense(latent_dim, use_bias=True)(encoder_hidden)
        z_mean = Activation('linear', name='z_mean')(z_mean_pre)
        
        z_log_var_pre = Dense(latent_dim, use_bias=True)(encoder_hidden)
        z_log_var = Activation('linear', name='z_log_var')(z_log_var_pre)
    
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    
    # Build decoder model
    latent_plus_cond = concatenate([z, cond])
    
    # add hidden layers
    for i, num in enumerate(p_decoder_lst):
        if i == 0:
            # 1st hidden layer
            if use_batch_norm:
                decoder_hidden = Dense(num, use_bias=False, name=f'decoder_layer{i}_w')(latent_plus_cond)
                decoder_hidden = BatchNormalization(name=f'decoder_layer{i}_BN')(decoder_hidden)
                decoder_hidden = Activation(hidden_act, name=f'decoder_layer{i}_act')(decoder_hidden)
            else:
                decoder_hidden = Dense(num, use_bias=True, name=f'decoder_layer{i}_w')(latent_plus_cond)
                decoder_hidden = Activation(hidden_act, name=f'decoder_layer{i}_act')(decoder_hidden)
        else:
            if use_batch_norm:
                decoder_hidden = Dense(num, use_bias=False, name=f'decoder_layer{i}_w')(decoder_hidden)
                decoder_hidden = BatchNormalization(name=f'decoder_layer{i}_BN')(decoder_hidden)
                decoder_hidden = Activation(hidden_act, name=f'decoder_layer{i}_act')(decoder_hidden)
            else:
                decoder_hidden = Dense(num, use_bias=True, name=f'decoder_layer{i}_w')(decoder_hidden)
                decoder_hidden = Activation(hidden_act, name=f'decoder_layer{i}_act')(decoder_hidden)
    
    
    # output layer
    if use_batch_norm:
        decoder_hidden = Dense(p, use_bias=False, name='decoder_output_w')(decoder_hidden)
        decoder_hidden = BatchNormalization(name='decoder_output_BN')(decoder_hidden)
        decoder_output = Activation(output_act, name='decoder_output_act')(decoder_hidden)
    else:
        decoder_hidden = Dense(p, use_bias=True, name='decoder_output_w')(decoder_hidden)
        decoder_output = Activation(output_act, name='decoder_output_act')(decoder_hidden)
    
   
    # CVAE model = encoder + decoder
    # by using the Keras functional API, the variables will be created right away without needing to call .build(). When not using API, you can manually call `model.build()`
    cvae = Model([X, cond], decoder_output, name='cvae')
    
    # Optimizer, use old optimizers in legacy namespace
    adam = optimizers.legacy.Adam(learning_rate=cvae_init_lr, clipnorm=1.0, decay=0.0)
    cvae.compile(optimizer=adam, loss=vae_loss, metrics=[reconstruction_loss, KL_loss], experimental_run_tf_function=True)
    
    
    # Subset the decoder (build another new decoder and re-store weights)
    def build_new_decoder(p, p_cond, latent_dim, p_decoder_lst, hidden_act='elu', output_act='relu', use_batch_norm=False):
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        cond_input = Input(shape=(p_cond,), name='conditions')
        latent_plus_cond = concatenate([latent_inputs, cond_input])
        
        # add hidden layers
        for i, num in enumerate(p_decoder_lst):
            if i == 0:
                # 1st hidden layer
                if use_batch_norm:
                    decoder_hidden = Dense(num, use_bias=False, name=f'decoder_layer{i}_w')(latent_plus_cond)
                    decoder_hidden = BatchNormalization(name=f'decoder_layer{i}_BN')(decoder_hidden)
                    decoder_hidden = Activation(hidden_act, name=f'decoder_layer{i}_act')(decoder_hidden)
                else:
                    decoder_hidden = Dense(num, use_bias=True, name=f'decoder_layer{i}_w')(latent_plus_cond)
                    decoder_hidden = Activation(hidden_act, name=f'decoder_layer{i}_act')(decoder_hidden)
            else:
                if use_batch_norm:
                    decoder_hidden = Dense(num, use_bias=False, name=f'decoder_layer{i}_w')(decoder_hidden)
                    decoder_hidden = BatchNormalization(name=f'decoder_layer{i}_BN')(decoder_hidden)
                    decoder_hidden = Activation(hidden_act, name=f'decoder_layer{i}_act')(decoder_hidden)
                else:
                    decoder_hidden = Dense(num, use_bias=True, name=f'decoder_layer{i}_w')(decoder_hidden)
                    decoder_hidden = Activation(hidden_act, name=f'decoder_layer{i}_act')(decoder_hidden)
        
        # output layer
        if use_batch_norm:
            decoder_hidden = Dense(p, use_bias=False, name='decoder_output_w')(decoder_hidden)
            decoder_hidden = BatchNormalization(name='decoder_output_BN')(decoder_hidden)
            decoder_output = Activation(output_act, name='decoder_output_act')(decoder_hidden)
        else:
            decoder_hidden = Dense(p, use_bias=True, name='decoder_output_w')(decoder_hidden)
            decoder_output = Activation(output_act, name='decoder_output_act')(decoder_hidden)
    
        new_decoder = Model([latent_inputs, cond_input], decoder_output, name='new_decoder')
        return new_decoder
    
    new_decoder = build_new_decoder(p, p_cond, latent_dim, p_decoder_lst, hidden_act, output_act, use_batch_norm)
    
    return cvae, new_decoder



def build_CVAE(spatial_df, scRNA_df, scRNA_celltype, n_marker_per_cmp, n_pseudo_spot, pseudo_spot_min_cell, pseudo_spot_max_cell, seq_depth_scaler, cvae_input_scaler, cvae_init_lr, num_hidden_layer, use_batch_norm, cvae_train_epoch, use_spatial_pseudo, use_fdr, p_val_cutoff, fc_cutoff, pct1_cutoff, pct2_cutoff, sortby_fc, diagnosis, rerun_DE=True, filter_gene=True):
    '''
    build CVAE to adjust platform effect, return transformed spatial gene expression and scRNA-seq cell-type gene signature
    
    input gene expression in datasets only included genes needed for downstream analysis and already been normalized by sequencing depth

    Parameters
    ----------
    spatial_df : dataframe
        normalized gene expression in spatial transcriptomic data (spots * genes).
    scRNA_df : dataframe
        normalized gene expression in scRNA-seq data (cells * genes).
    scRNA_celltype : dataframe
        cell-type annotations for cells in scRNA-seq data. Only 1 column named <celltype>.
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
    rerun_DE : bool, optional
        whether to rerun DE on the CVAE transformed scRNA-seq data, since the DE genes might be different with before CVAE transforming.
    filter_gene : bool
        whether to filter genes before DE.
    

    Returns
    -------
    spatial_transformed_numi : dataframe
        CVAE transformed (platform effect adjusted) spatial spot gene raw nUMI counts (spots * genes).
    scRNA_decode_avg_df : dataframe
        CVAE decodered average gene expression (normalized) of cell-types in scRNA-seq data (cell-types * genes).
    new_markers : list or None
        marker genes from re-run DE on CVAE transformed scRNA-seq data. It will be None if not re-run DE (rerun_DE=False).
    cvae_pred : dataframe or None
        cell-type proportions of spatial spots predicted or transferred by CVAE. It will be None if no way to got initial guess of cell-type proportions (spots * cell-types).
    '''
    
    assert((scRNA_df.index == scRNA_celltype.index).all())
    assert((spatial_df.columns == scRNA_df.columns).all())
    
    
    if diagnosis:
        # first plot UMAP for raw input gene expressions
        from diagnosis_plots import defineColor, rawInputUMAP
        plot_colors = defineColor(spatial_df.shape[0], scRNA_celltype)
        rawInputUMAP(spatial_df, scRNA_df, scRNA_celltype, plot_colors)
        
    
    # some settings
    # max number of pseudo spots (training+validation, without training single cells)
    n_max_pseudo_spots = n_pseudo_spot
    # scaler to multiply the normalized gene values and transform back to raw nUMI counts
    depth_scaler = seq_depth_scaler
    # percentage of training pseudo spots
    training_pct = 0.8
    # max value when scaling the input gene expression of CVAE, while min is 0
    input_max = cvae_input_scaler
    # first log transform, then minmax scaling
    use_log_transform = True
    # whether to get initial guess of cell-type proportions
    do_initial_guess = True
    # whether to single cell augmentation and #cells of cell type balancing
    sc_augment = True
    
    # the order of celltypes matters, unify the order throughout whole pipeline, which will be determined here
    celltype_order = sorted(list(scRNA_celltype.celltype.unique()))
    n_celltype = len(celltype_order)
    celltype_count_dict = scRNA_celltype.celltype.value_counts().to_dict()
    
    
    # Randomly select cells into pseudo-spots, at most X pseudo-spots
    # total number of generated pseudo-spots (including training and validation pseudo-spots, NOT include scRNA-seq cells)
    n_pseudo_scrna = int(min(100 * spatial_df.shape[0] * n_celltype, n_max_pseudo_spots))
    n_train_pseudo_scrna = int(np.floor(n_pseudo_scrna * training_pct))
    n_valid_pseudo_scrna = int(n_pseudo_scrna - n_train_pseudo_scrna)
    print(f'generate {n_pseudo_scrna} pseudo-spots containing {pseudo_spot_min_cell} to {pseudo_spot_max_cell} cells from scRNA-seq cells...')
    # pseudo-spot gene expression (pseudo-spots * genes; NO scRNA-seq cells at the end)
    # pseudo-spot cell-type proportions (pseudo-spots * cell-types; NO scRNA-seq cells at the end)
    # number of cells in pseudo-spots (NO scRNA-seq cells at the end)
    pseudo_spots_df, pseudo_spots_celltype_prop, n_cell_in_spot = generate_pseudo_spots(scRNA_df, scRNA_celltype, n_pseudo_scrna, celltype_order, pseudo_spot_min_cell, pseudo_spot_max_cell)
    
    
    if sc_augment:
        print('\n#cells of cell types in reference scRNA-seq data:')
        for k, v in celltype_count_dict.items():
            print(f'{k}: {v}')
        max_count = max(celltype_count_dict.values())
        target_count = int(1.5 * max_count)
        print(f'HIGHLIGHT: augment single cells to {target_count} cells per cell type')
        scRNA_df, scRNA_celltype, scrna_n_cell = augment_sc(scRNA_df, scRNA_celltype, target_count, 2, 3)
        # update count
        celltype_count_dict = scRNA_celltype.celltype.value_counts().to_dict()
        # also update color palette
        if diagnosis:
            plot_colors = defineColor(spatial_df.shape[0], scRNA_celltype)
        
        # split training and validation
        n_train_scrna_cell = int(np.floor(scRNA_df.shape[0] * training_pct))
        n_valid_scrna_cell = int(scRNA_df.shape[0] - n_train_scrna_cell)
    
    else:
        n_train_scrna_cell = scRNA_df.shape[0]
        n_valid_scrna_cell = 0
        scrna_n_cell = [1] * n_train_pseudo_scrna
    
    # convert scRNA-seq cell-type annotation to proportions
    scrna_cell_celltype_prop = celltype2props(scRNA_celltype, celltype_order)
    
    
    # generate pseudo-spots by combining spatial spots
    if use_spatial_pseudo:
        n_pseudo_spatial = int(0.5 * n_pseudo_scrna)
    else:
        n_pseudo_spatial = 0
    n_train_pseudo_spatial = int(np.floor(n_pseudo_spatial * training_pct))
    n_valid_pseudo_spatial = int(n_pseudo_spatial - n_train_pseudo_spatial)
    print(f'generate {n_pseudo_spatial} pseudo-spots containing 2 to 6 spots from spatial spots...')
    pseudo_spatial_df = combine_spatial_spots(spatial_df, n_pseudo_spatial, 2, 6)
    
    
    if use_log_transform:
        # since the input dataframe is extracted from AnnData object, and will not be used in downstream analysis (we can extract from AnnData again), it's safe to modify them directly here
        print('\nHIGHLIGHT: first apply log transformation on sequencing depth normalized gene expressions, followed by Min-Max scaling')
        spatial_df = np.log1p(spatial_df)
        if pseudo_spots_df.shape[0] > 0:
            pseudo_spots_df = np.log1p(pseudo_spots_df)
        if pseudo_spatial_df.shape[0] > 0:
            pseudo_spatial_df = np.log1p(pseudo_spatial_df)
        # do not forget scRNA_df
        scRNA_df = np.log1p(scRNA_df)
    
    
    # Build training and validation data
    # first spots used for validation, rest spots used for training
    print(f'\n{"" : <24} | {"training": >9} | {"validation": >9}')
    print(f'{"spatial spots" : <24} | {spatial_df.shape[0]: >9} | {0: >9}')
    print(f'{"spatial pseudo-spots" : <24} | {n_train_pseudo_spatial: >9} | {n_valid_pseudo_spatial: >9}')
    print(f'{"scRNA-seq cells" : <24} | {n_train_scrna_cell: >9} | {n_valid_scrna_cell: >9}')
    print(f'{"scRNA-seq pseudo-spots" : <24} | {n_train_pseudo_scrna: >9} | {n_valid_pseudo_scrna: >9}\n')
    
    
    train_scrna_df = pd.concat([pseudo_spots_df.iloc[n_valid_pseudo_scrna:,:], scRNA_df.iloc[n_valid_scrna_cell:,:]], ignore_index=False)
    valid_scrna_df = pd.concat([pseudo_spots_df.iloc[:n_valid_pseudo_scrna,:], scRNA_df.iloc[:n_valid_scrna_cell,:]], ignore_index=False)
    
    train_spatial_df = pd.concat([pseudo_spatial_df.iloc[n_valid_pseudo_spatial:,:], spatial_df], ignore_index=False)
    valid_spatial_df = pseudo_spatial_df.iloc[:n_valid_pseudo_spatial,:]
    
    assert train_scrna_df.shape[0] == (n_train_pseudo_scrna + n_train_scrna_cell)
    assert valid_scrna_df.shape[0] == (n_valid_pseudo_scrna + n_valid_scrna_cell)
    assert train_spatial_df.shape[0] == (n_train_pseudo_spatial + spatial_df.shape[0])
    assert valid_spatial_df.shape[0] == n_valid_pseudo_spatial
    
    # scaling to [0,input_max] with each dataset separately
    # use only spatial spots + spatial pseudo-spots for spatial dataset scaling
    print(f'scaling inputs to range 0 to {input_max}')
    spatial_min_max_scaler = MinMaxScaler(feature_range=[0, input_max])
    train_spatial_data = spatial_min_max_scaler.fit_transform(train_spatial_df)
    if valid_spatial_df.shape[0] > 0:
        valid_spatial_data = spatial_min_max_scaler.transform(valid_spatial_df)
    else:
        valid_spatial_data = valid_spatial_df.values
    
    # use only training pseudo spots + single cells for scRNA-seq dataset scaling
    scRNA_min_max_scaler = MinMaxScaler(feature_range=[0, input_max])
    train_scrna_data = scRNA_min_max_scaler.fit_transform(train_scrna_df)
    if valid_scrna_df.shape[0] > 0:
        valid_scrna_data = scRNA_min_max_scaler.transform(valid_scrna_df)
    else:
        valid_scrna_data = valid_scrna_df.values
    
    
    # first spatial pseudo-spots then spatial spots then scRNA-seq pseudo-spots and scRNA-seq cells in data
    # we also consider whether to duplicate spatial data since they are few
    # update: use training sample weights
    data = np.vstack((train_spatial_data, train_scrna_data))
    labels = np.array([input_max]*train_spatial_data.shape[0] + [0.]*train_scrna_data.shape[0])
    labels = labels.reshape((len(labels), 1))
    
    # validation data
    valid_data = np.vstack((valid_spatial_data, valid_scrna_data))
    valid_labels = np.array([input_max]*valid_spatial_data.shape[0] + [0.]*valid_scrna_data.shape[0])
    valid_labels = valid_labels.reshape((len(valid_labels), 1))
    
    
    # training sample weights
    weight_pseudo_scrna = np.ones((n_train_pseudo_scrna,))
    weight_cell_scrna = np.ones((n_train_scrna_cell,))
    weight_pseudo_spatial = np.ones((n_train_pseudo_spatial,))
    weight_spot_spatial = np.ones((spatial_df.shape[0],))
    
    # weight sum of scRNA-seq cells : sum of scRNA pseudo spots = 1 : 1
    # always decrease the weights for cohort with more samples
    if n_train_pseudo_scrna > 0:
        if n_train_pseudo_scrna > n_train_scrna_cell:
            weight_pseudo_scrna *= n_train_scrna_cell / n_train_pseudo_scrna
        elif n_train_pseudo_scrna < n_train_scrna_cell:
            weight_cell_scrna *= n_train_pseudo_scrna / n_train_scrna_cell
            
    # weight sum of spatial spots : sum of spatial pseudo spots = 1 : 1
    # always decrease the weights for cohort with more samples
    if n_train_pseudo_spatial > 0:
        if n_train_pseudo_spatial > spatial_df.shape[0]:
            weight_pseudo_spatial *= spatial_df.shape[0] / n_train_pseudo_spatial
        elif n_train_pseudo_spatial < spatial_df.shape[0]:
            weight_spot_spatial *= n_train_pseudo_spatial / spatial_df.shape[0]
    
    # Final Balancing, re-weight spatial data to make sure the sum of spatial : sum of scRNA-seq = 1 : 1
    # since we have already adjusted the weights, here we can not rely on sample size any more, use the sum of weight instead
    # always decrease the weights for cohort with more samples
    if (np.sum(weight_pseudo_scrna)+np.sum(weight_cell_scrna)) < (np.sum(weight_pseudo_spatial)+np.sum(weight_spot_spatial)):
        # calculate factor beforehand to avoid update in weight causing factor change
        tmp_factor = (np.sum(weight_pseudo_scrna)+np.sum(weight_cell_scrna)) / (np.sum(weight_pseudo_spatial)+np.sum(weight_spot_spatial))
        weight_pseudo_spatial *= tmp_factor
        weight_spot_spatial *= tmp_factor
    elif (np.sum(weight_pseudo_scrna)+np.sum(weight_cell_scrna)) > (np.sum(weight_pseudo_spatial)+np.sum(weight_spot_spatial)):
        tmp_factor = (np.sum(weight_pseudo_spatial)+np.sum(weight_spot_spatial)) / (np.sum(weight_pseudo_scrna)+np.sum(weight_cell_scrna))
        weight_pseudo_scrna *= tmp_factor
        weight_cell_scrna *= tmp_factor
    
    sample_weight = np.concatenate([weight_pseudo_spatial, weight_spot_spatial,
                                    weight_pseudo_scrna, weight_cell_scrna])
    
    del train_spatial_data, valid_spatial_data, train_scrna_data, valid_scrna_data
    del train_spatial_df, valid_spatial_df, train_scrna_df, valid_scrna_df
    
    
    # Define CVAE
    # number of nodes in input layer (equals number of celltypes)
    p = data.shape[1]
    # number of nodes in conditional node
    p_cond = 1
    # Hyper-parameters
    latent_dim = n_celltype * 3
    # use geometric mean of latent and input dimension (a geometric progression)
    hidden_dim = list(np.floor(np.geomspace(latent_dim, p, num_hidden_layer+2)[1:num_hidden_layer+1]).astype('int'))
    
    print('\nCVAE structure:')
    print(f'Encoder: {" - ".join([str(x) for x in ([p+p_cond] + hidden_dim[::-1] + [latent_dim])])}')
    print(f'Decoder: {" - ".join([str(x) for x in ([latent_dim+p_cond] + hidden_dim + [p])])}\n')
    
    # note hidden layer in encoder is a reverse of the hidden_dim variable
    cvae, new_decoder = CVAE_keras_model(p, p_cond, latent_dim, hidden_dim[::-1], hidden_dim, use_batch_norm=use_batch_norm, cvae_init_lr=cvae_init_lr)
    
    # learning rate decay
    lrate = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=5e-4, cooldown=10, verbose=False)
    
    # early stopping based on validation loss
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=40, restore_best_weights=True, verbose=False)
    
    
    # change tensorflow seed value, set the same seed value for sampling samples from latent space to decoder before training
    # still has unknown randomness source even set seed here...
    set_random_seed(1154)
    
    # if not use Batch Normalization, we use all training data for one epoch
    if use_batch_norm:
        one_batch_size = 16384
        do_shuffle = True
    else:
        one_batch_size = data.shape[0]
        do_shuffle = False
    
    # Train CVAE
    # note when there is no pseudo-spots, then there is no validation data
    if valid_data.shape[0] == 0:
        print('\nStart training without validation data...\n')
        history_callback = cvae.fit([data, labels], data,
                                epochs=cvae_train_epoch,
                                batch_size=one_batch_size,
                                shuffle=do_shuffle,
                                callbacks=[lrate, early_stop],
                                sample_weight=sample_weight,
                                verbose=True)
    else:
         print('\nStart training...\n')
         history_callback = cvae.fit([data, labels], data,
                                validation_data=([valid_data, valid_labels], valid_data),
                                epochs=cvae_train_epoch,
                                batch_size=one_batch_size,
                                shuffle=do_shuffle,
                                callbacks=[lrate, early_stop],
                                sample_weight=sample_weight,
                                verbose=True)
    
    n_epoch = len(history_callback.history['loss'])
    
    if n_epoch < cvae_train_epoch:
        print(f'\ntraining finished in {n_epoch} epochs (early stop), transform data to adjust the platform effect...\n')
    else:
        print(f'\ntraining finished in {n_epoch} epochs (reach max pre-specified epoches), transform data to adjust the platform effect...\n')

    if diagnosis:
        # plot loss
        from diagnosis_plots import plotCVAELoss
        plotCVAELoss(history_callback)


    # postprocess the trained models
    # Subset the encoder
    encoder = Model([cvae.get_layer('encoder_input').input, cvae.get_layer('cond_input').input],
                    [cvae.get_layer('z_mean').output, cvae.get_layer('z_log_var').output],
                    name='encoder')
    
    # update layer weights for the decoder
    for layer in new_decoder.layers:
        if layer.name.endswith('w') or layer.name.endswith('BN'):
            new_decoder.get_layer(layer.name).set_weights(cvae.get_layer(layer.name).get_weights())
    
    '''
    # double check to ensure decoder is correct
    from utils import check_decoder
    check_decoder(cvae, new_decoder, data, labels)
    '''
    
    # Transform data to adjust platform effect
    # note if log transform is applied, the data in spatial_df, pseudo_spatial_df, scRNA_df and pseudo_spots_df are already log transformed
    
    # Decoder output of transformed latent embedding of spatial spot-level data
    spatial_embed = encoder.predict([spatial_min_max_scaler.transform(spatial_df), np.full((spatial_df.shape[0],1), input_max)])[0]
    
    tmp_output = new_decoder.predict([spatial_embed, np.full((spatial_embed.shape[0],1), 0)])
    
    spatial_transformed_df = pd.DataFrame(scRNA_min_max_scaler.inverse_transform(tmp_output),
                                          columns=spatial_df.columns,
                                          index=spatial_df.index)
    
    # log transformation back
    if use_log_transform:
        print('HIGHLIGHT: when transforming data, after reversed Min-Max Scaling, apply exp transformation then multiple the factor and round to integer')
        spatial_transformed_df = np.expm1(spatial_transformed_df)
        
    spatial_transformed_numi = np.rint(spatial_transformed_df * depth_scaler)

    
    # Decoder output of average marker gene expression of scRNA-seq cell-types
    # NOTE: here we also include augmented single cells into downstream analysis, which affects the cell-type marker gene profile
    scRNA_embed = encoder.predict([scRNA_min_max_scaler.transform(scRNA_df), np.full((scRNA_df.shape[0],1), 0)])[0]
    
    # decode it
    tmp_output = new_decoder.predict([scRNA_embed, np.full((scRNA_embed.shape[0],1), 0)])
    
    scRNA_decode_df = pd.DataFrame(scRNA_min_max_scaler.inverse_transform(tmp_output),
                                   columns=scRNA_df.columns,
                                   index=scRNA_df.index)
    
    # log transformation back
    if use_log_transform:
        scRNA_decode_df = np.expm1(scRNA_decode_df)
    
    # whether to rerun DE
    if rerun_DE:
        print('\nre-run DE on CVAE transformed scRNA-seq data!')
        from utils import rerun_DE
        new_markers = rerun_DE(scRNA_decode_df, scRNA_celltype, n_marker_per_cmp, use_fdr, p_val_cutoff, fc_cutoff, pct1_cutoff, pct2_cutoff, sortby_fc, diagnosis, filter_gene)
    else:
        new_markers = None
    
    # take average for all genes
    scRNA_decode_avg_df = scRNA_decode_df.copy()
    scRNA_decode_avg_df['celltype'] = scRNA_celltype.celltype
    scRNA_decode_avg_df = scRNA_decode_avg_df.groupby(['celltype']).mean()
    
    # re-order the rows to match the previous defined cell-type order
    scRNA_decode_avg_df = scRNA_decode_avg_df.loc[celltype_order, :]
    
    
    # transfer cell-type proportions as an initial guess
    if n_pseudo_scrna == 0:
        pseudo_spot_embed = np.empty((0, scRNA_embed.shape[1]))
    else:
        pseudo_spot_embed = encoder.predict([scRNA_min_max_scaler.transform(pseudo_spots_df), np.full((pseudo_spots_df.shape[0],1), 0)])[0]
    
    if do_initial_guess:
        use_embedding = 'none'
        if use_embedding == 'none':
            print('HIGHLIGHT: got initial guess of cell type proportions based on original CVAE latent embedding')
        else:
            print(f'\nHIGHLIGHT: got initial guess of cell type proportions based on {use_embedding} embedding of CVAE latent space')
        tmp_pred = transferProps(spatial_embed,
                                 np.vstack((scRNA_embed, pseudo_spot_embed)),
                                 pd.concat([scrna_cell_celltype_prop, pseudo_spots_celltype_prop]).values,
                                 n_neighbors=10, sigma=1, use_embedding=use_embedding)
        cvae_pred = pd.DataFrame(tmp_pred, index=spatial_df.index, columns=celltype_order)
        
        if diagnosis:
            os.makedirs(os.path.join(diagnosis_path, 'initial_guess'), exist_ok=True)
            cvae_pred.to_csv(os.path.join(diagnosis_path, 'initial_guess', 'celltype_props_by_transferring.csv'))
            
    else:
        cvae_pred = None


    # whether to save models and transformed data
    if diagnosis:
        # also embed pseudo-spots for diagnosis
        if n_pseudo_spatial == 0:
            pseudo_spatial_embed = np.empty((0, spatial_embed.shape[1]))
        else:
            pseudo_spatial_embed = encoder.predict([spatial_min_max_scaler.transform(pseudo_spatial_df), np.full((pseudo_spatial_df.shape[0],1), input_max)])[0]
        
        
        from diagnosis_plots import diagnosisCVAE
        diagnosisCVAE(cvae, encoder, new_decoder,
                      spatial_embed, spatial_transformed_df, spatial_transformed_numi, pseudo_spatial_embed,
                      scRNA_celltype, celltype_order, celltype_count_dict, scrna_cell_celltype_prop, scRNA_embed, scrna_n_cell,
                      pseudo_spots_celltype_prop, n_cell_in_spot, pseudo_spot_embed,
                      scRNA_decode_df, scRNA_decode_avg_df, new_markers, plot_colors)


    #print(f'before CVAE building function return RAM usage: {psutil.Process().memory_info().rss/1024**2:.2f} MB')
    
    return spatial_transformed_numi, scRNA_decode_avg_df, new_markers, cvae_pred



def build_CVAE_whole(spatial_file, ref_file, ref_anno_file, marker_file, n_hv_gene, n_marker_per_cmp, n_pseudo_spot, pseudo_spot_min_cell, pseudo_spot_max_cell, seq_depth_scaler, cvae_input_scaler, cvae_init_lr, num_hidden_layer, use_batch_norm, cvae_train_epoch, use_spatial_pseudo, redo_de, use_fdr, p_val_cutoff, fc_cutoff, pct1_cutoff, pct2_cutoff, sortby_fc, diagnosis, filter_cell, filter_gene):
    '''
    read related CSV files, build CVAE to adjust platform effect, return transformed spatial gene expression and scRNA-seq cell-type gene signature

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
    spatial_transformed_numi : dataframe
        CVAE transformed (platform effect adjusted) spatial spot gene raw nUMI counts (spots * genes).
    scRNA_decode_avg_df : dataframe
        CVAE decodered average gene expression (normalized) of cell-types in scRNA-seq data (cell-types * genes).
    new_markers : list or None
        marker genes from re-run DE on CVAE transformed scRNA-seq data. It will be None if not re-run DE.
    cvae_pred : dataframe or None
        cell-type proportions of spatial spots predicted or transferred by CVAE. It will be None if no way to got initial guess of cell-type proportions (spots * cell-types).
    '''
    
    start_time = time()
    
    # read spatial data
    spatial_spot_obj = read_spatial_data(spatial_file, filter_gene)[0]
    
    # read scRNA-seq data
    scrna_obj = read_scRNA_data(ref_file, ref_anno_file, filter_cell, filter_gene)
    
    # Overlap of genes between scRNA cell-level and spatial spot-level data
    overlap_genes = list(set(spatial_spot_obj.var_names).intersection(set(scrna_obj.var_names)))
    print(f'total {len(overlap_genes)} overlapped genes')
    # if len(overlap_genes) < spatial_spot_obj.n_vars:
    #     print(f'{spatial_spot_obj.n_vars-len(overlap_genes)} genes in spatial data but not found in scRNA-seq data: {", ".join(set(spatial_spot_obj.var_names).difference(set(overlap_genes)))}\n')
    
    # subset overlapped gene
    spatial_spot_obj = spatial_spot_obj[:, overlap_genes].copy()
    scrna_obj = scrna_obj[:, overlap_genes].copy()
    
    # how many genes used for CVAE
    if len(overlap_genes) <= n_hv_gene:
        
        # use all genes for CVAE
        final_gene_list = sorted(overlap_genes)
        print(f'\nuse all {len(final_gene_list)} genes for downstream analysis as there are less genes available than specified number {n_hv_gene}')
        
    else:
    
        # use selected highly variable genes + cell-type marker genes for CVAE
        
        # identify highly variable genes in scRNA-seq data, select TOP X HV genes
        # no need to consider highly variable genes in spatial data, as for cell-type deconvolution, we work on each spot independently
        print(f'\nidentify {n_hv_gene} highly variable genes from scRNA-seq data...')
        if n_hv_gene == 0:
            scrna_hv_genes = []
        else:
            scrna_hv_genes = sc.pp.highly_variable_genes(scrna_obj, layer='raw_nUMI', flavor='seurat_v3', n_top_genes=n_hv_gene, inplace=False)
            scrna_hv_genes = scrna_hv_genes.loc[scrna_hv_genes['highly_variable']==True].index.to_list()

        
        # identify cell-type marker genes
        print('\nidentify cell-type marker genes...')
        if marker_file is not None:
            # directly use provide marker gene expression
            marker_df = pd.read_csv(marker_file, index_col=0)
            print('user provided marker gene profile, DE will be skipped...\n')
            print(f'read {marker_df.shape[1]} marker genes from user specified marker gene file')
    
            # extract marker gene overlapped with spatial data
            marker_genes = list(set(overlap_genes) & set(marker_df.columns))
            print(f'from user specified marker gene expression use {len(marker_genes)} marker genes overlapped with spatial + scRNA-seq data')
            # if len(marker_genes) < len(overlap_genes):
            #     print(f'{len(overlap_genes)-len(marker_genes)} genes in overlapped gene list between spatial and scRNA-seq data but not found in user provided marker gene expression: {", ".join(set(overlap_genes).difference(set(marker_genes)))}\n')
            
        else:
            # perform DE, return the marker gene expression
            print('no marker gene profile provided. Perform DE to get cell-type marker genes on scRNA-seq data...\n')
            marker_genes = run_DE(scrna_obj, n_marker_per_cmp, use_fdr, p_val_cutoff, fc_cutoff, pct1_cutoff, pct2_cutoff, sortby_fc, diagnosis, 'DE_celltype_markers.csv')
        
        # final gene list for downstream analysis
        final_gene_list = sorted(list(set(scrna_hv_genes).union(set(marker_genes))))
        print(f'\nuse union of highly variable gene list and cell-type marker gene list derived from scRNA-seq data, finally get {len(final_gene_list)} genes for downstream analysis')
    
    
    print('\nstart CVAE building...\n')
    scrna_celltype = sc.get.obs_df(scrna_obj, keys='celltype').to_frame()
    scrna_celltype['celltype'] = scrna_celltype['celltype'].astype(str)
    
    # build CVAE
    (spatial_transformed_numi,
     scRNA_decode_avg_df,
     new_markers,
     cvae_pred) = build_CVAE(sc.get.obs_df(spatial_spot_obj, keys=final_gene_list),
                             sc.get.obs_df(scrna_obj, keys=final_gene_list),
                             scrna_celltype,
                             n_marker_per_cmp, n_pseudo_spot, pseudo_spot_min_cell, pseudo_spot_max_cell, seq_depth_scaler,
                             cvae_input_scaler, cvae_init_lr, num_hidden_layer, use_batch_norm, cvae_train_epoch, use_spatial_pseudo,
                             use_fdr, p_val_cutoff, fc_cutoff, pct1_cutoff, pct2_cutoff, sortby_fc,
                             diagnosis, rerun_DE=redo_de, filter_gene=filter_gene)
    
    print(f'\nplatform effect adjustment by CVAE finished. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.\n\n')
    
    return spatial_transformed_numi, scRNA_decode_avg_df, new_markers, cvae_pred