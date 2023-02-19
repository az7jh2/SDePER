#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 21:43:08 2022

@author: hill103

this script stores functions to build a CVAE for platform effect adjustment
"""



import os
from config import print, output_path
import numpy as np
import pandas as pd
from utils import read_spatial_data, read_scRNA_data, run_DE
from time import time
import random
from sklearn.preprocessing import MinMaxScaler
import scanpy as sc
sc.settings.verbosity = 0  # verbosity: errors (0), warnings (1), info (2), hints (3)

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Input, Dense, Activation, concatenate
#from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import set_random_seed

# dealing with the keras symbolic tensor error
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()



def build_CVAE(spatial_df, scRNA_df, scRNA_celltype, n_marker_per_cmp, pseudo_spot_min_cell, pseudo_spot_max_cell, seq_depth_scaler, cvae_input_scaler, cvae_init_lr, diagnosis, rerun_DE=True):
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
        cell-type annotations for cells in scRNA-seq data. Only 1 column named <celltype>
    n_marker_per_cmp : int
        number of TOP marker genes for each comparison in DE
    pseudo_spot_min_cell : int
        minimum value of cells in pseudo-spot
    pseudo_spot_max_cell : int
        maximum value of cells in pseudo-spot
    seq_depth_scaler : int
        a scaler of scRNA-seq sequencing depth
    cvae_input_scaler : int
        maximum value of the scaled input for CVAE
    cvae_init_lr : float
        initial learning rate for training CVAE
    diagnosis : bool
        if True save more information to files for diagnosis CVAE and hyper-parameter selection
    rerun_DE : bool, optional
        whether to rerun DE on the CVAE transformed scRNA-seq data, since the DE genes might be different with before CVAE transforming.

    Returns
    -------
    spatial_transformed_df : dataframe
        CVAE transformed (platform effect adjusted) spatial spot gene raw nUMI counts (spots * genes)
    scRNA_decode_df : dataframe
        CVAE decodered average gene expression (normalized) of cell-types in scRNA-seq data (cell-types * genes)
    new_markers : list or None
        marker genes from re-run DE on CVAE transformed scRNA-seq data. It will be None if not re-run DE (rerun_DE=False).
    '''
    
    assert((scRNA_df.index == scRNA_celltype.index).all())
    assert((spatial_df.columns == scRNA_df.columns).all())
    
    # some settings
    # max number of pseudo spots (training+validation, without training single cells)
    n_max_pseudo_spots = 5e5
    # scaler to multiply the normalized gene values and transform back to raw nUMI counts
    depth_scaler = seq_depth_scaler
    # percentage of training pseudo spots
    training_pct = 0.8
    # max value when scaling the input gene expression of CVAE, while min is 0
    input_max = cvae_input_scaler
    
    celltypes = sorted(list(scRNA_celltype.celltype.unique()))
    n_celltype = len(celltypes)
    celltype_count_dict = scRNA_celltype.celltype.value_counts().to_dict()
    
    # Randomly select cells into pseudo-spots, at most X pseudo-spots
    print(f'generate pseudo-spots containing {pseudo_spot_min_cell} to {pseudo_spot_max_cell} cells from scRNA-seq cells...')
    n_spot = int(min(100 * spatial_df.shape[0] * n_celltype, n_max_pseudo_spots))
    n_train_spot = int(np.floor(n_spot * training_pct))
    n_valid_spot = int(n_spot - n_train_spot)
    
    print(f'generate {n_train_spot} pseudo-spots for training and {n_valid_spot} pseudo-spots for validation')

    pseudo_spots = []
    celltype_stats = []
    n_cell_in_spot = []
    
    n_cell_list = list(range(pseudo_spot_min_cell, pseudo_spot_max_cell+1))
    all_cell_index = scRNA_df.index.to_list()
    # cell barcode separated by cell-types
    type_cell_index = dict()
    for one_celltype in celltypes:
        type_cell_index[one_celltype] = scRNA_celltype[scRNA_celltype['celltype']==one_celltype].index.to_list()
    
    # though it's possible to use multiprocessing to generate pseudo-spots parallelly, the big dataframe need to be shared across all subprocesses, and it may not be a good idea to share objects in multiprocessing as it may cause unknown problems. And the performance benefits by multiprocessing many not be such large
    # so considering the safety and performance benefits, just keep the simplest way to generate pseudo-spots one-by-one
    # to reduce randomness, pre-set the seed value for random
    random.seed(138)
    for i in range(n_spot):
        # first determine how many cells in this pseudo-spot
        this_num = random.sample(n_cell_list, 1)[0]
        n_cell_in_spot.append(this_num)
        
        this_cells = []
        for i in range(this_num):
            # select one cell-type
            selected_celltype = random.sample(celltypes, 1)[0]
            # from this selected cell-type, randomly select one cell belong to that cell-type
            this_cells.append(random.sample(type_cell_index[selected_celltype], 1)[0])
        
        # take average of selected cells
        pseudo_spots.append(scRNA_df.loc[this_cells].mean(axis=0))
        
        # count the celltype of selected cells
        celltype_stats.append(scRNA_celltype.loc[this_cells, 'celltype'].value_counts().to_dict())
    
    # Add the scRNA-seq cells as pseudo-spots containing only one cell to the end
    for this_cells in all_cell_index:
        pseudo_spots.append(scRNA_df.loc[this_cells])
        celltype_stats.append({scRNA_celltype.loc[this_cells, 'celltype']: 1})
        n_cell_in_spot.append(1)


    # Build pseudo-spots dataframe
    # First n_valid_spot spots are used for validation, rest spots are used for training
    pseudo_spots_df = pd.concat(pseudo_spots, axis=1).transpose()
    pseudo_spots_df.reset_index(inplace=True, drop=True)
    
    pseudo_spots_celltype = pd.DataFrame(celltype_stats, columns=celltypes)
    pseudo_spots_celltype.fillna(0, inplace=True)
    # calculate cell-type proportions
    pseudo_spots_celltype = pseudo_spots_celltype.div(pseudo_spots_celltype.sum(axis=1), axis=0)
    
    #import gc
    #import psutil
    #print(f'before gc and del variable RAM usage: {psutil.Process().memory_info().rss/1024**2:.2f} MB')
    del pseudo_spots, celltype_stats
    #print(f'del variable without gc RAM usage: {psutil.Process().memory_info().rss/1024**2:.2f} MB')
    #gc.collect()
    #print(f'after gc RAM usage: {psutil.Process().memory_info().rss/1024**2:.2f} MB')
    
    
    # Build training and validation data
    # nomalize to [0,input_max] with each dataset separately
    # use only training pseudo spots + single cells for scRNA-seq dataset normalization
    print(f'scaling inputs to range 0 to {input_max}')
    spatial_min_max_scaler = MinMaxScaler(feature_range=[0,input_max])
    tmp_spatial_data = spatial_min_max_scaler.fit_transform(spatial_df)
    
    scRNA_min_max_scaler = MinMaxScaler(feature_range=[0,input_max])
    scRNA_min_max_scaler.fit(pseudo_spots_df.iloc[n_valid_spot:,:])
    tmp_scRNA_data = scRNA_min_max_scaler.transform(pseudo_spots_df)
    
    # first n_valid_spot spots used for validation, rest spots used for training, as spots with only one cell are listed at the end
    oversample_scale = 1
    data = np.vstack((np.tile(tmp_spatial_data, (oversample_scale, 1)), tmp_scRNA_data[n_valid_spot:,:]))
    labels = np.array([input_max]*tmp_spatial_data.shape[0]*oversample_scale + [0.]*tmp_scRNA_data[n_valid_spot:,:].shape[0])
    labels = labels.reshape((len(labels), 1))
    
    # validation data
    valid_data = tmp_scRNA_data[:n_valid_spot,:]
    valid_labels = np.array([0.]*n_valid_spot)
    valid_labels = valid_labels.reshape((len(valid_labels), 1))
    
    n_scRNA_training_spot = tmp_scRNA_data[n_valid_spot:,:].shape[0]
    n_spatial_training_spot = tmp_spatial_data.shape[0]*oversample_scale
    print(f'in training -- spatial spots : pseudo-spots = {n_spatial_training_spot:d} : {n_scRNA_training_spot:d}')
    
    del tmp_spatial_data, tmp_scRNA_data
    
    # re-weight spatial data to make sure the sample size equals spatial:scRNA = 1:10
    sample_weight = np.ones((data.shape[0],))
    # also increase the weight of scRNA-seq cells
    #sample_weight = np.array([1.]*n_spatial_training_spot + [1.]*n_train_spot + [2.]*len(all_cell_index))

    if n_spatial_training_spot < 0.1 * n_scRNA_training_spot:
        sample_weight[:n_spatial_training_spot] *= 0.1 * n_scRNA_training_spot / n_spatial_training_spot
    
    
    # Define CVAE
    # number of nodes in input layer (equals number of celltypes)
    p = data.shape[1]
    # number of nodes in conditional node
    p_cond = 1
    
    
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
    def vae_loss(obs, pred):
        # VAE loss = mse_loss or xent_loss + kl_loss
        recon_loss = K.sum(K.square(obs - pred), axis=-1)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return recon_loss + kl_loss
    
    
    # Hyper-parameters
    latent_dim = n_celltype * 3
    # hidden_dim = (latent_dim * np.floor(np.sqrt(p/latent_dim))).astype('int')
    # use geometric mean of latent and input dimension
    hidden_dim = np.floor(np.sqrt(p * latent_dim))
    
    
    # Build encoder model
    X = Input(shape=(p,), name='encoder_input')
    cond = Input(shape=(p_cond,), name='cond_input')
    encoder_inputs = concatenate([X, cond])
    # hidden layer 1
    encoder_hidden = Dense(hidden_dim, use_bias=True)(encoder_inputs)
    #encoder_hidden = BatchNormalization()(encoder_hidden)
    encoder_hidden = Activation('elu')(encoder_hidden)
    # encoding layer z_mean
    z_mean_pre = Dense(latent_dim, use_bias=True)(encoder_hidden)
    #z_mean_pre = BatchNormalization()(z_mean_pre)
    z_mean = Activation('linear', name='z_mean')(z_mean_pre)
    # encoding layer z_log_var
    z_log_var_pre = Dense(latent_dim, use_bias=True)(encoder_hidden)
    #z_log_var_pre = BatchNormalization()(z_log_var_pre)
    z_log_var = Activation('linear', name='z_log_var')(z_log_var_pre)
    
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    
    # Build decoder model
    latent_plus_cond = concatenate([z, cond])
    # hidden layer 1
    decoder_hidden = Dense(hidden_dim, use_bias=True, name='decoder_layer1_w')(latent_plus_cond)
    #decoder_hidden = BatchNormalization(name='decoder_layer1_BN')(decoder_hidden)
    decoder_hidden = Activation('elu', name='decoder_layer1_act')(decoder_hidden)
    # output layer
    decoder_hidden = Dense(p, use_bias=True, name='decoder_output_w')(decoder_hidden)
    #decoder_hidden = BatchNormalization(name='decoder_output_BN')(decoder_hidden)
    decoder_output = Activation('relu', name='decoder_output_act')(decoder_hidden)
    
    # recorder the layer names which need to re-store weights
    layer_names = ['decoder_layer1_w', 'decoder_output_w']
    
    
    # CVAE model = encoder + decoder
    # by using the Keras functional API, the variables will be created right away without needing to call .build(). When not using API, you can manually call `model.build()`
    cvae = Model([X, cond], decoder_output, name='cvae')
    
    # Optimazer
    adam = optimizers.Adam(learning_rate=cvae_init_lr, clipnorm=1.0, decay=0.0)
    cvae.compile(optimizer=adam, loss=vae_loss, experimental_run_tf_function=True)
    
    
    # learning rate decay
    lrate = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=15, min_lr=1e-5, cooldown=5, verbose=False)
    
    
    # early stopping based on validation loss
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, restore_best_weights=True, verbose=False)
    
    
    # change tensorflow seed value, set the same seed value for sampling samples from latent space to decoder before training
    set_random_seed(1154)
    
    # Train CVAE
    print('\nStart training...\n')
    history_callback = cvae.fit([data, labels], data,
                                validation_data=([valid_data, valid_labels], valid_data),
                                epochs=1000,
                                batch_size=data.shape[0],
                                shuffle=False,
                                callbacks=[lrate, early_stop],
                                sample_weight=sample_weight,
                                verbose=True)
    
    n_epoch = len(history_callback.history['loss'])
    
    if n_epoch < 1000:
        print(f'\ntraining finished in {n_epoch} epochs (early stop), transform data to adjust the platform effect...\n')
    else:
        print(f'\ntraining finished in {n_epoch} epochs (reach max pre-specified epoches), transform data to adjust the platform effect...\n')

    
    # proprecess the trained models
    # Subset the encoder
    encoder = Model([cvae.get_layer('encoder_input').input, cvae.get_layer('cond_input').input],
                    [cvae.get_layer('z_mean').output, cvae.get_layer('z_log_var').output],
                    name='encoder')
    
    # Subset the decoder (build another new decoder and re-store weights)
    def build_new_decoder(p, p_cond, latent_dim):
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        cond_input = Input(shape=(p_cond,), name='conditions')
        latent_plus_cond = concatenate([latent_inputs, cond_input])
        # hidden layer 1
        decoder_hidden = Dense(hidden_dim, use_bias=True, name='decoder_layer1_w')(latent_plus_cond)
        #decoder_hidden = BatchNormalization(name='decoder_layer1_BN')(decoder_hidden)
        decoder_hidden = Activation('elu', name='decoder_layer1_act')(decoder_hidden)
        # output layer
        decoder_hidden = Dense(p, use_bias=True, name='decoder_output_w')(decoder_hidden)
        #decoder_hidden = BatchNormalization(name='decoder_output_BN')(decoder_hidden)
        decoder_output = Activation('relu', name='decoder_output_act')(decoder_hidden)
    
        new_decoder = Model([latent_inputs, cond_input], decoder_output, name='new_decoder')
        return new_decoder

    new_decoder = build_new_decoder(p, p_cond, latent_dim)
    for layer_name in layer_names:
        new_decoder.get_layer(layer_name).set_weights(cvae.get_layer(layer_name).get_weights())
    
    
    # Transform data to adjust platform effect
    # Decoder output of transformed latent embedding of spatial spot-level data
    spatial_embed = encoder.predict([spatial_min_max_scaler.transform(spatial_df), np.full((spatial_df.shape[0],1), input_max)])[0]
    
    tmp_output = new_decoder.predict([spatial_embed, np.full((spatial_embed.shape[0],1), 0)])
    
    spatial_transformed_df = pd.DataFrame(np.rint(scRNA_min_max_scaler.inverse_transform(tmp_output) * depth_scaler),
                                          columns=spatial_df.columns,
                                          index=spatial_df.index)

    
    # Decoder output of average marker gene expression of scRNA-seq cell-types
    scRNA_embed = encoder.predict([scRNA_min_max_scaler.transform(scRNA_df), np.full((scRNA_df.shape[0],1), 0)])[0]
    
    # decode it
    tmp_output = new_decoder.predict([scRNA_embed, np.full((scRNA_embed.shape[0],1), 0)])
    
    scRNA_decode_df = pd.DataFrame(scRNA_min_max_scaler.inverse_transform(tmp_output),
                                   columns=scRNA_df.columns,
                                   index=scRNA_df.index)
    
    # whether to rerun DE
    if rerun_DE:
        print('\nre-run DE on CVAE transformed scRNA-seq data!')
        from utils import rerun_DE
        new_markers = rerun_DE(scRNA_decode_df, scRNA_celltype, n_marker_per_cmp, diagnosis)
    else:
        new_markers = None
    
    # take average for all genes
    scRNA_decode_df['celltype'] = scRNA_celltype.celltype
    scRNA_decode_df = scRNA_decode_df.groupby(['celltype']).mean()
    
    # whether to save models and transformed data
    if diagnosis:
        print('\nsave variables related to CVAE to files!')
        spatial_transformed_df.to_csv(os.path.join(output_path, 'spatial_spots_transformToscRNA_decoded.csv'))
        scRNA_decode_df.to_csv(os.path.join(output_path, 'scRNA_decoded_avg_exp_bycelltypes.csv'))
        cvae.save(os.path.join(output_path, 'CVAE_whole.h5'))
        encoder.save(os.path.join(output_path, 'CVAE_encoder.h5'))
        new_decoder.save(os.path.join(output_path, 'CVAE_decoder.h5'))
        
        
        # plot variance of mu of spatial spots and scRNA-seq cells
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        
        latent_var = np.concatenate((spatial_embed, scRNA_embed), axis=0).var(axis=0)
        plt.figure()
        ax = sns.histplot(x=np.log10(latent_var))
        ax.set(xlabel='log10(var(mu))')
        plt.savefig(os.path.join(output_path, 'histogram of variance of latent neurons on spatial spots and scRNA-seq cells.png'))
        plt.close()
        
        
        # plot histogram of transformed spots nUMI sum
        plt.figure()
        ax = sns.histplot(x=np.sum(spatial_transformed_df.values, axis=1))
        ax.set(xlabel='Sum of nUMI per spatial spot after transformation')
        plt.savefig(os.path.join(output_path, 'histogram of sum of nUMI of spatial spots after transformation.png'))
        plt.close()
        
        
        # plot zero percentage of marker genes in spatial spots after transformation
        plt.figure()
        if new_markers is None:
            tmp_mtx = spatial_transformed_df.values
        else:
            tmp_mtx = spatial_transformed_df.loc[:, new_markers].values
        ax = sns.histplot(x=np.sum(tmp_mtx==0, axis=1)/tmp_mtx.shape[1])
        ax.set(xlabel=f'Zero percentage of {tmp_mtx.shape[1]} genes per spot after transformation')
        ax.set_xlim(xmin=0, xmax=1)
        plt.savefig(os.path.join(output_path, 'histogram of zero percentage of spatial spots after transformation.png'))
        plt.close()
        
        
        # plot umap of spatial spots and scRNA-seq cells plus pseudo spots
        import umap
        from distinctipy import distinctipy
        
        # embed of decoded average marker gene expression
        marker_embed = encoder.predict([scRNA_min_max_scaler.transform(scRNA_decode_df), np.full((scRNA_decode_df.shape[0],1), 0)])[0]
        
        pseudo_spot_embed = encoder.predict([scRNA_min_max_scaler.transform(pseudo_spots_df.iloc[:n_spot, :]), np.full((n_spot,1), 0)])[0]
        
        # the order will affect the point overlay, first row draw first
        # umap has embeded seed (default 42), by specify random_state, umap will use special mode to keep reproducibility
        all_umap = umap.UMAP(random_state=42).fit_transform(np.concatenate((pseudo_spot_embed, scRNA_embed, marker_embed, spatial_embed), axis=0))
        
        # add cell/spot count in the annotation
        plot_df = pd.DataFrame({'UMAP1': all_umap[:, 0],
                                'UMAP2': all_umap[:, 1],
                                'celltype': ['pseudo']*n_spot + [f'{x} ({celltype_count_dict[x]})' for x in scRNA_celltype.celltype.to_list()] + [f'{x} ({celltype_count_dict[x]})' for x in scRNA_decode_df.index.to_list()] + [f'spatial ({spatial_df.shape[0]})']*spatial_df.shape[0],
                                'dot_type': ['pseudo spot']*n_spot + ['cell']*scRNA_df.shape[0] + ['marker']*scRNA_decode_df.shape[0] + ['spatial spot']*spatial_df.shape[0],
                                'dataset': ['scRNA-seq']*(scRNA_df.shape[0]+scRNA_decode_df.shape[0]+n_spot) + ['spatial']*spatial_df.shape[0]
                                },
                               index = [f'pseudo{x}' for x in range(n_spot)] + scRNA_df.index.to_list() + [f'{x}-marker' for x in scRNA_decode_df.index.to_list()] + spatial_df.index.to_list())
        
        plot_sizes = {'cell': 20, 'spatial spot': 20, 'marker': 200, 'pseudo spot': 20}
        plot_markers = {'cell': 'o', 'spatial spot': 'o', 'marker': 'X', 'pseudo spot': 'o'}
        plot_colors = {}
        for one_celltype, one_color in zip([f'spatial ({spatial_df.shape[0]})']+[f'{x} ({celltype_count_dict[x]})' for x in celltypes], distinctipy.get_colors(n_celltype+1)):
            plot_colors[one_celltype] = one_color
        # assign pseudo spots as gray80
        plot_colors['pseudo'] = '#cccccc'
        
        #plt.figure(figsize=(6.4*2*2, 4.8*2))
        sns.set_style("darkgrid")
        
        # relplot return a FacetGrid object
        # specify figure size by Height (in inches) of each facet, and Aspect ratio of each facet
        fgrid = sns.relplot(data=plot_df, x='UMAP1', y='UMAP2', hue='celltype', size='dot_type', style='dot_type', sizes=plot_sizes, markers=plot_markers, palette=plot_colors, kind='scatter', col='dataset', col_order=['scRNA-seq', 'spatial'], height=4.8*2, aspect=6.4/4.8)
        fgrid.set(xlabel='Embedding Dimension 1', ylabel='Embedding Dimension 2')
        # Put the legend out of the figure
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # add cell-type annotations around marker coordinates
        # adjustText do not support seaborn relplot
        #from adjustText import adjust_text
        # fgrid.axes return an array of all axes in the figure
        ax = fgrid.axes[0, 0]
        texts = []
        sub_plot_df = plot_df.loc[plot_df['dot_type']=='marker']
        for one_row in sub_plot_df.index:
            texts.append(ax.text(sub_plot_df.at[one_row, 'UMAP1'], sub_plot_df.at[one_row, 'UMAP2'], sub_plot_df.at[one_row, 'celltype'].split(' (')[0], weight='bold'))
        #adjust_text(texts)
        plt.savefig(os.path.join(output_path, 'UMAP of spatial spots and scRNA-seq cells with markers.png'))
        plt.close()
        
        
        # plot distribution of number of cells in pseudo-spots
        # first n_spot + #scRNA-seq cells of the records are just the pseudo-spots + scRNA-seq data, with the row order matches
        tmp_df = plot_df.iloc[:(n_spot+scRNA_df.shape[0]), :]
        tmp_df = tmp_df.assign(n_cell_in_spot = n_cell_in_spot)
     
        # generate a colormap with a specified color for NA (spatial spots), but not work for relplot...
        #my_cmap = sns.color_palette("viridis", as_cmap=True)
        #my_cmap.set_bad(color=plot_colors[f'spatial ({spatial_df.shape[0]})'])
        
        # show the full legend of colorbar in relplot, otherwise it will only show a sample of evenly spaced values (The FacetGrid hue is categorical, not continuous)
        #fgrid = sns.relplot(data=tmp_df, x='UMAP1', y='UMAP2', hue='#cell_in_spot', palette='viridis', kind='scatter', col='dataset', col_order=['scRNA-seq', 'spatial'], height=4.8*2, aspect=6.4/4.8, legend='full')
        #fgrid.set(xlabel='Embedding Dimension 1', ylabel='Embedding Dimension 2')
        
        # instead use plt.subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6.4*4, 4.8*2))
        # left panel: scatter plot of pseudo-spots
        sc = ax1.scatter(tmp_df['UMAP1'], tmp_df['UMAP2'], c=tmp_df['n_cell_in_spot'], cmap='cubehelix', s=10, marker='o')
        ax1.set_title('dataset = scRNA-seq')
        ax1.set_xlabel('Embedding Dimension 1')
        ax1.set_ylabel('Embedding Dimension 2')
        
        # right panel: scatter plot of spatial spots
        tmp_df = plot_df.loc[plot_df['dot_type']=='spatial spot', :]
        ax2.scatter(tmp_df['UMAP1'], tmp_df['UMAP2'], color=plot_colors[f'spatial ({spatial_df.shape[0]})'], s=10, marker='o')
        ax2.set_title('dataset = spatial')
        ax2.set_xlabel('Embedding Dimension 1')
        ax2.set_ylabel('Embedding Dimension 2')
        
        # add colorbar with title to the most right (https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots, conflict with tight_layout)
        cbar = fig.colorbar(sc, ax=ax2)
        cbar.ax.set_title('#cell in spot')
        
        fig.tight_layout()
        fig.savefig(os.path.join(output_path, 'distribution of number of cells in pseudo-spots.png'))
        plt.close()
        
        
        # plot distribution of cell-type proportions of each cell-type
        # tried to use PDF format, but encountered error TeX capacity exceeded, since too many dots in figure
        for this_celltype in celltypes:
            # first n_spot + #scRNA-seq cells of the records are just the pseudo-spots + scRNA-seq data, with the row order matches
            tmp_df = plot_df.iloc[:(n_spot+scRNA_df.shape[0]), :]
            # don't forget the .values
            tmp_df = tmp_df.assign(proportion = pseudo_spots_celltype[this_celltype].values)
            
            # start to plot
            fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6.4*4, 4.8*2))
            (ax1, ax2) = axes.flat
            # left panel: scatter plot of pseudo-spots
            sc = ax1.scatter(tmp_df['UMAP1'], tmp_df['UMAP2'], c=tmp_df['proportion'], cmap='cubehelix', s=10, marker='o', norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
            # plot marker gene profiles with different marker also color (same color is hard to recognize the cross)
            ax1.scatter(plot_df.at[f'{this_celltype}-marker', 'UMAP1'], plot_df.at[f'{this_celltype}-marker', 'UMAP2'], color='red', s=120, marker='X')
            ax1.set_title('dataset = scRNA-seq')
            ax1.set_xlabel('Embedding Dimension 1')
            ax1.set_ylabel('Embedding Dimension 2')
            
            # right panel: scatter plot of spatial spots
            # interpolate the grid for contour plot
            from scipy.interpolate import griddata
            grid_x, grid_y = np.mgrid[tmp_df['UMAP1'].min():tmp_df['UMAP1'].max():0.025, tmp_df['UMAP2'].min():tmp_df['UMAP2'].max():0.025]
            grid_z = griddata(tmp_df.loc[:, ['UMAP1', 'UMAP2']].values, tmp_df['proportion'].values, (grid_x, grid_y), method='linear',  fill_value=np.nan)
            
            try:
                ax2.contourf(grid_x, grid_y, grid_z, cmap='cubehelix', norm=matplotlib.colors.Normalize(vmin=0, vmax=1), alpha=0.3)
            except:
                pass
            
            tmp_df2 = plot_df.loc[plot_df['dot_type']=='spatial spot', :]
            ax2.scatter(tmp_df2['UMAP1'], tmp_df2['UMAP2'], color=plot_colors[f'spatial ({spatial_df.shape[0]})'], s=10, marker='o')
            ax2.set_title('dataset = spatial')
            ax2.set_xlabel('Embedding Dimension 1')
            ax2.set_ylabel('Embedding Dimension 2')
            
            # add colorbar with title
            cbar = fig.colorbar(sc, ax=ax2)
            cbar.ax.set_title('proportion')
            
            fig.suptitle(this_celltype)
            
            fig.tight_layout()
            
            # make sure the file name is valid
            fig.savefig(os.path.join(output_path, f'distribution of {"".join(x for x in this_celltype if (x.isalnum() or x in "._- "))} proportions.png'))
            plt.close()
        
        
        # save spatial spot UMAP coordinates
        plot_df.loc[plot_df['dot_type']=='spatial spot', ['UMAP1', 'UMAP2']].to_csv(os.path.join(output_path, 'spatial spots UMAP coordinates.csv'))
        # save scRNA-seq cells UMAP coordinates
        plot_df.loc[plot_df['dot_type']=='cell', ['UMAP1', 'UMAP2']].to_csv(os.path.join(output_path, 'scRNA-seq cells UMAP coordinates.csv'))
    
    #print(f'before CVAE building function return RAM usage: {psutil.Process().memory_info().rss/1024**2:.2f} MB')
    
    return spatial_transformed_df, scRNA_decode_df, new_markers



def build_CVAE_whole(spatial_file, ref_file, ref_anno_file, marker_file, n_hv_gene, n_marker_per_cmp, pseudo_spot_min_cell, pseudo_spot_max_cell, seq_depth_scaler, cvae_input_scaler, cvae_init_lr, redo_de, diagnosis):
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
        number of TOP marker genes for each comparison in DE
    pseudo_spot_min_cell : int
        minimum value of cells in pseudo-spot
    pseudo_spot_max_cell : int
        maximum value of cells in pseudo-spot
    seq_depth_scaler : int
        a scaler of scRNA-seq sequencing depth
    cvae_input_scaler : int
        maximum value of the scaled input for CVAE
    cvae_init_lr : float
        initial learning rate for training CVAE
    redo_de : bool
        whether to redo DE after CVAE transformation
    diagnosis : bool
        if True save more information to files for diagnosis CVAE and hyper-parameter selection

    Returns
    -------
    spatial_transformed_df : dataframe
        CVAE transformed (platform effect adjusted) spatial spot gene raw nUMI counts (spots * genes)
    scRNA_decode_df : dataframe
        CVAE decodered average gene expression (normalized) of cell-types in scRNA-seq data (cell-types * genes)
    new_markers : list or None
        marker genes from re-run DE on CVAE transformed scRNA-seq data. It will be None if not re-run DE.
    '''
    
    start_time = time()
    
    # read spatial data
    spatial_spot_obj = read_spatial_data(spatial_file)
    
    # read scRNA-seq data
    scrna_obj = read_scRNA_data(ref_file, ref_anno_file)
    
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
            marker_genes = run_DE(scrna_obj, n_marker_per_cmp, diagnosis, 'DE celltype markers.csv')
        
        # final gene list for downstream analysis
        final_gene_list = sorted(list(set(scrna_hv_genes).union(set(marker_genes))))
        print(f'\nuse union of highly variable gene list and cell-type marker gene list derived from scRNA-seq data, finally get {len(final_gene_list)} genes for downstream analysis')
    
    
    print('\nstart CVAE building...\n')
    scrna_celltype = sc.get.obs_df(scrna_obj, keys='celltype').to_frame()
    scrna_celltype['celltype'] = scrna_celltype['celltype'].astype(str)
    
    # build CVAE
    spatial_df, scrna_avg_df, new_markers = build_CVAE(sc.get.obs_df(spatial_spot_obj, keys=final_gene_list),
                                          sc.get.obs_df(scrna_obj, keys=final_gene_list),
                                          scrna_celltype,
                                          n_marker_per_cmp, pseudo_spot_min_cell, pseudo_spot_max_cell, seq_depth_scaler,
                                          cvae_input_scaler, cvae_init_lr, diagnosis, rerun_DE=redo_de)
    
    print(f'\nplatform effect adjustment by CVAE finished. Elapsed time: {(time()-start_time)/60.0:.2f} minutes.\n\n')
    
    return spatial_df, scrna_avg_df, new_markers