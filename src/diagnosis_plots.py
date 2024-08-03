#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 06:02:34 2024

@author: hill103

We move all functions for generating diagnosis plots to here.
"""



import os
from config import print, diagnosis_path
#from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import seaborn as sns
sns.set()
import umap
from distinctipy import distinctipy
from scipy.interpolate import griddata
from sklearn.decomposition import PCA



def plotCVAELoss(history):
    '''
    plot training and validation loss in CVAE training
    
    NOTE validation loss may be unavailable if no validation data in training

    Parameters
    ----------
    history : History object
        History object returned by Keras Model.fit in CVAE training.

    Returns
    -------
    None.
    '''
    
    # need to create subfolders first, otherwise got FileNotFoundError
    os.makedirs(os.path.join(diagnosis_path, 'CVAE_training'), exist_ok=True)
    
    
    # plot total loss
    plt.figure(figsize=(6.4*2, 4.8*2))
    plt.plot(history.history['loss'], label='Training')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation')
    plt.ylabel('Total Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(diagnosis_path, 'CVAE_training', 'CVAE_total_loss_in_training.png'))
    plt.close()
    
    
    # plot components of loss
    plt.figure(figsize=(6.4*2, 4.8*2))
    
    # Training KL Loss
    plt.plot(history.history['KL_loss'], color='#1f77b4', marker='o', label='Training KL Loss')
    # Validation KL Loss
    if 'val_KL_loss' in history.history:
        plt.plot(history.history['val_KL_loss'], color='#ff7f0e', marker='o', label='Validation KL Loss')

    # Training Reconstruction Loss
    plt.plot(history.history['reconstruction_loss'], color='#1f77b4', marker='s', linestyle='dashed', label='Training Reconstruction Loss')
    # Validation Reconstruction Loss
    if 'val_reconstruction_loss' in history.history:
        plt.plot(history.history['val_reconstruction_loss'], color='#ff7f0e', marker='s', linestyle='dashed', label='Validation Reconstruction Loss')
    
    # Adding labels
    plt.xlabel('Epoch')
    plt.ylabel('Model Loss')
    # Add a legend
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(diagnosis_path, 'CVAE_training', 'CVAE_loss_components_in_training.png'))
    plt.close()



def defineColor(n_spatial_spot, scRNA_celltype):
    '''
    generate n visually distinct colours for cell-types. Use these colours across whole pipeline for consistency

    Parameters
    ----------
    n_spatial_spot : int
        number of spatial spots.
    scRNA_celltype : dataframe
        cell-type annotations for cells in scRNA-seq data. Only 1 column named <celltype>.

    Returns
    -------
    plot_colors : dict
        generated color palette, keys are cell-types together with the number of cells with this cell-type, and values are RGB colors.
    '''
    
    celltype_order = sorted(list(scRNA_celltype.celltype.unique()))
    celltype_count_dict = scRNA_celltype.celltype.value_counts().to_dict()
    
    plot_colors = {}
    for one_celltype, one_color in zip([f'spatial ({n_spatial_spot})']+[f'{x} ({celltype_count_dict[x]})' for x in celltype_order], distinctipy.get_colors(len(celltype_order)+1)):
        plot_colors[one_celltype] = one_color
    # assign pseudo spots as gray80
    plot_colors['pseudo'] = '#cccccc'
    
    return plot_colors



def rawInputUMAP(spatial_df, scRNA_df, scRNA_celltype, plot_colors):
    '''
    generate UMAP of spatial spots together with scRNA-seq cells

    Parameters
    ----------
    spatial_df : dataframe
        normalized gene expression in spatial transcriptomic data (spots * genes).
    scRNA_df : dataframe
        normalized gene expression in scRNA-seq data (cells * genes).
    scRNA_celltype : dataframe
        cell-type annotations for cells in scRNA-seq data. Only 1 column named <celltype>.
    plot_colors : dict
        color palette for plot.

    Returns
    -------
    None.
    '''
    
    assert((scRNA_df.index == scRNA_celltype.index).all())
    assert((spatial_df.columns == scRNA_df.columns).all())
    
    # need to create subfolders first, otherwise got FileNotFoundError
    os.makedirs(os.path.join(diagnosis_path, 'raw_input_data'), exist_ok=True)
    
    celltype_count_dict = scRNA_celltype.celltype.value_counts().to_dict()
    
    # take average within cell-types to get cell-type specific marker gene profile
    scRNA_avg_df = scRNA_df.copy()
    scRNA_avg_df['celltype'] = scRNA_celltype.celltype
    scRNA_avg_df = scRNA_avg_df.groupby(['celltype']).mean()
    
    # UMAP of raw input spatial and scRNA-seq cell gene expression
    # we also add marker gene expression profile here
    # the order will affect the point overlay, first row draw first
    # umap has embeded seed (default 42), by specify random_state, umap will use special mode to keep reproducibility
    # note here both gene expressions are sequencing normalized values, without log transform and scaling
    tmp_df = pd.concat([spatial_df, scRNA_df, scRNA_avg_df], ignore_index=True)
    
    all_umap = umap.UMAP(random_state=42).fit_transform(tmp_df.values)
    
    # add cell/spot count in the annotation
    plot_df = pd.DataFrame({'UMAP1': all_umap[:, 0],
                            'UMAP2': all_umap[:, 1],
                            'celltype': [f'spatial ({spatial_df.shape[0]})']*spatial_df.shape[0] +
                                        [f'{x} ({celltype_count_dict[x]})' for x in scRNA_celltype.celltype.to_list()] +
                                        [f'{x} ({celltype_count_dict[x]})' for x in scRNA_avg_df.index],
                            'dataset': ['spatial']*spatial_df.shape[0] +
                                       ['scRNA-seq']*scRNA_df.shape[0] +
                                       ['scRNA-seq']*scRNA_avg_df.shape[0],
                            'datatype': ['cell/spot']*spatial_df.shape[0] +
                                        ['cell/spot']*scRNA_df.shape[0] +
                                        ['marker']*scRNA_avg_df.shape[0]
                            },
                           index = spatial_df.index.to_list() +
                                   scRNA_df.index.to_list() +
                                   [f'{x}-marker' for x in scRNA_avg_df.index])
    
    plot_sizes = {'cell/spot': 20, 'marker': 200}
    plot_markers = {'cell/spot': 'o', 'marker': 'X'}
    # color for plot already defined

    sns.set_style("darkgrid")
    
    # relplot return a FacetGrid object
    # specify figure size by Height (in inches) of each facet, and Aspect ratio of each facet
    fgrid = sns.relplot(data=plot_df, x='UMAP1', y='UMAP2', hue='celltype', size='datatype', style='datatype', sizes=plot_sizes, markers=plot_markers, palette=plot_colors, kind='scatter', col='dataset', col_order=['scRNA-seq', 'spatial'], height=4.8*2, aspect=6.4/4.8)
    fgrid.set(xlabel='UMAP 1', ylabel='UMAP 2')
    # Put the legend out of the figure
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # add cell-type annotations around marker coordinates
    # adjustText do not support seaborn relplot
    #from adjustText import adjust_text
    # fgrid.axes return an array of all axes in the figure
    ax = fgrid.axes[0, 0]
    texts = []
    for one_row in scRNA_avg_df.index:
        texts.append(ax.text(plot_df.at[one_row+'-marker', 'UMAP1'], plot_df.at[one_row+'-marker', 'UMAP2'], one_row, weight='bold'))
    #adjust_text(texts)
    
    plt.savefig(os.path.join(diagnosis_path, 'raw_input_data', 'UMAP_raw_input_color_by_celltype.png'))
    plt.close()
    
    
    # also save UMAP coordinates
    plot_df[['UMAP1', 'UMAP2']].to_csv(os.path.join(diagnosis_path, 'raw_input_data', 'UMAP_coordinates_raw_input.csv.gz'), compression='gzip')



def diagnosisCVAE(cvae, encoder, decoder, spatial_embed, spatial_transformed_df, spatial_transformed_numi, pseudo_spatial_embed, scRNA_celltype, celltype_order, celltype_count_dict, scrna_cell_celltype_prop, scRNA_embed, scrna_n_cell, pseudo_spots_celltype_prop, n_cell_in_spot, pseudo_spot_embed, scRNA_decode_df, scRNA_decode_avg_df, new_markers, plot_colors):
    '''
    save CVAE related Keras models to h5 file, generate figures to diagnosis the training of CVAE

    Parameters
    ----------
    cvae : Keras model
        already trained CVAE model
    encoder: Keras model
        encoder part extract from CVAE model
    decoder : Keras model
        a separated decoder whose weights are already updated, i.e. it should give the same decoded output with CVAE
    spatial_embed : 2-D numpy array
        mu in latent space of spatial spots (spots * latent neurons)
    spatial_transformed_df : dataframe
        CVAE transformed (platform effect adjusted) spatial spot gene normalized expressions (spots * genes)
    spatial_transformed_numi : dataframe
        CVAE transformed (platform effect adjusted) spatial spot gene raw nUMI counts (spots * genes)
    pseudo_spatial_embed : dataframe
        mu in latent space of spatial pseudo-spots (spatial pseudo-spots * latent neurons). It will have 0 rows if no spatial pseudo-spots generated
    scRNA_celltype : dataframe
        cell-type annotations for cells in scRNA-seq data. Only 1 column named <celltype>
    celltype_order : list
        already sorted unique cell-types. Its order matters, and will be the order in pseudo_spots_celltype (columns) and cell-type gene expression profile (rows)
    celltype_count_dict : dict
        number of cells in reference scRNA-seq data for each cell-type
    scrna_cell_celltype_prop : dataframe
        scRNA-seq cells cell-type proportions (cells * cell-types)
    scRNA_embed : 2-D numpy array
        mu in latent space of scRNA-seq cells (cells * latent neurons)
    scrna_n_cell : list
        number of cells in scRNA-seq data. Note augmented single cells also included
    pseudo_spots_celltype_prop : dataframe
        pseudo-spot cell-type proportions (pseudo-spots * cell-types; NO scRNA-seq cells included)
    n_cell_in_spot : list
        number of cells in pseudo-spots (no scRNA-seq cells included)
    pseudo_spot_embed : dataframe
        mu in latent space of pseudo spots (pseudo-spots * latent neurons); NO scRNA-seq cells included. It will have 0 rows if no scRNA-seq pseudo-spots generated
    scRNA_decode_df : dataframe
        CVAE decodered gene expression (normalized) of scRNA-seq cells (cells * genes)
    scRNA_decode_avg_df : dataframe
        CVAE decodered average gene expression (normalized) of cell-types in scRNA-seq data (cell-types * genes)
    new_markers : list or None
        marker genes from re-run DE on CVAE transformed scRNA-seq data. It will be None if not re-run DE (rerun_DE=False)
    plot_colors : dict
        color palette for plot
        
    Returns
    -------
    None.
    '''
    
    print('\nsave variables related to CVAE to files!')
    
    # need to create subfolders first, otherwise got FileNotFoundError
    os.makedirs(os.path.join(diagnosis_path, 'CVAE_model'), exist_ok=True)
    os.makedirs(os.path.join(diagnosis_path, 'CVAE_transformed_data'), exist_ok=True)
    os.makedirs(os.path.join(diagnosis_path, 'CVAE_latent_space'), exist_ok=True)
    
    '''
    # plot model
    # need Graphviz binaries (https://graphviz.gitlab.io/, install in OS) and pydot python package
    # we do not plot it in package implementation
    plot_model(cvae, to_file=os.path.join(diagnosis_path, 'CVAE_model', 'CVAE_model.png'), show_shapes=True, show_layer_names=True)
    '''
    
    # save model
    cvae.save(os.path.join(diagnosis_path, 'CVAE_model', 'CVAE_whole.h5'))
    encoder.save(os.path.join(diagnosis_path, 'CVAE_model', 'CVAE_encoder.h5'))
    decoder.save(os.path.join(diagnosis_path, 'CVAE_model', 'CVAE_decoder.h5'))
    
    
    # save transformed data, which are used in GLRM modeling
    if new_markers is None:
        spatial_transformed_numi.to_csv(os.path.join(diagnosis_path, 'CVAE_transformed_data', 'spatial_spots_transformToscRNA_decoded.csv.gz'), compression='gzip')
    else:
        # save only marker genes used in downstream anlaysis
        spatial_transformed_numi[new_markers].to_csv(os.path.join(diagnosis_path, 'CVAE_transformed_data', 'spatial_spots_transformToscRNA_decoded.csv.gz'), compression='gzip')
    
    if new_markers is None:
        scRNA_decode_avg_df.to_csv(os.path.join(diagnosis_path, 'CVAE_transformed_data', 'scRNA_decoded_avg_exp_bycelltypes.csv.gz'), compression='gzip')
    else:
        # save only marker genes used in downstream anlaysis
        scRNA_decode_avg_df[new_markers].to_csv(os.path.join(diagnosis_path, 'CVAE_transformed_data', 'scRNA_decoded_avg_exp_bycelltypes.csv.gz'), compression='gzip')
        
    
    # plot variance of mu of spatial spots and scRNA-seq cells
    latent_var = np.concatenate((spatial_embed, scRNA_embed), axis=0).var(axis=0)
    plt.figure()
    ax = sns.histplot(x=np.log10(latent_var))
    ax.set(xlabel='log10(var(mu))')
    plt.savefig(os.path.join(diagnosis_path, 'CVAE_latent_space', 'histogram_latent_neurons_variance_spatial_spots_and_scRNA-seq_cells.png'))
    plt.close()
    
    
    # plot histogram of transformed spots nUMI sum
    plt.figure()
    ax = sns.histplot(x=np.sum(spatial_transformed_numi.values, axis=1))
    ax.set(xlabel='Sum of nUMI per spatial spot after transformation')
    plt.savefig(os.path.join(diagnosis_path, 'CVAE_transformed_data', 'histogram_nUMI_sum_per_spatial_spot_after_transformation.png'))
    plt.close()
    
    
    # plot histogram of transformed spots nUMI
    # exclude zeros since too many zeros
    plt.figure()
    tmp_values = spatial_transformed_numi.stack().values
    ax = sns.histplot(x=tmp_values[tmp_values>0])
    ax.set(xlabel='non-zero nUMI of all genes and all spatial spots after transformation')
    plt.savefig(os.path.join(diagnosis_path, 'CVAE_transformed_data', 'histogram_non-zero_nUMI_all_gene_all_spatial_spot_after_transformation.png'))
    plt.close()
    
    
    # plot zero percentage of marker genes in spatial spots after transformation
    plt.figure()
    if new_markers is None:
        tmp_mtx = spatial_transformed_numi.values
    else:
        tmp_mtx = spatial_transformed_numi.loc[:, new_markers].values
    ax = sns.histplot(x=np.sum(tmp_mtx==0, axis=1)/tmp_mtx.shape[1])
    ax.set(xlabel=f'Zero percentage of {tmp_mtx.shape[1]} genes per spot after transformation')
    ax.set_xlim(xmin=0, xmax=1)
    plt.savefig(os.path.join(diagnosis_path, 'CVAE_transformed_data', 'histogram_zero_percentage_per_spatial_spot_after_transformation.png'))
    plt.close()
    
    
    # plot umap of latent space of spatial spots and scRNA-seq cells plus pseudo spots
    # there is no way to add marker gene expression profile here, since the marker gene average expression is calculated on decoded values, it has no meaning even if we feed it to encoder
    # the order will affect the point overlay, first row draw first
    # umap has embeded seed (default 42), by specify random_state, umap will use special mode to keep reproducibility
    all_umap = umap.UMAP(random_state=42).fit_transform(np.concatenate((pseudo_spot_embed, pseudo_spatial_embed, scRNA_embed, spatial_embed), axis=0))
    
    # add cell/spot count in the annotation
    plot_df = pd.DataFrame({'UMAP1': all_umap[:, 0],
                            'UMAP2': all_umap[:, 1],
                            'celltype': ['pseudo']*pseudo_spot_embed.shape[0] +
                                        ['pseudo']*pseudo_spatial_embed.shape[0] + 
                                        [f'{x} ({celltype_count_dict[x]})' for x in scRNA_celltype.celltype.to_list()] +
                                        [f'spatial ({spatial_embed.shape[0]})']*spatial_embed.shape[0],
                            'dataset': ['scRNA-seq']*pseudo_spot_embed.shape[0] +
                                       ['spatial']*pseudo_spatial_embed.shape[0] +
                                       ['scRNA-seq']*scRNA_embed.shape[0] +
                                       ['spatial']*spatial_embed.shape[0],
                            'datatype': ['scrna-pseudo']*pseudo_spot_embed.shape[0] +
                                        ['spatial-pseudo']*pseudo_spatial_embed.shape[0] +
                                        ['scrna-cell']*scRNA_embed.shape[0] +
                                        ['spatial-spot']*spatial_embed.shape[0],
                            },
                           index = [f'scrna_pseudo{x}' for x in range(pseudo_spot_embed.shape[0])] +
                                   [f'spatial_pseudo{x}' for x in range(pseudo_spatial_embed.shape[0])] +
                                   scRNA_decode_df.index.to_list() +
                                   spatial_transformed_df.index.to_list())
    
    # plot colors already defined
    sns.set_style("darkgrid")
    
    # relplot return a FacetGrid object
    # specify figure size by Height (in inches) of each facet, and Aspect ratio of each facet
    fgrid = sns.relplot(data=plot_df, x='UMAP1', y='UMAP2', hue='celltype', s=20, marker='o', palette=plot_colors, kind='scatter', col='dataset', col_order=['scRNA-seq', 'spatial'], height=4.8*2, aspect=6.4/4.8)
    fgrid.set(xlabel='UMAP 1', ylabel='UMAP 2')
    # Put the legend out of the figure
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.savefig(os.path.join(diagnosis_path, 'CVAE_latent_space', 'UMAP_latent_mu_embedding_color_by_celltype.png'))
    plt.close()
    
    
    # save UMAP coordinates of latent space
    plot_df.loc[plot_df['datatype']=='spatial-spot', ['UMAP1', 'UMAP2']].to_csv(os.path.join(diagnosis_path, 'CVAE_latent_space', 'UMAP_coordinates_latent_mu_embedding_spatial_spots.csv.gz'), compression='gzip')
    # save scRNA-seq cells UMAP coordinates
    plot_df.loc[plot_df['datatype']=='scrna-cell', ['UMAP1', 'UMAP2']].to_csv(os.path.join(diagnosis_path, 'CVAE_latent_space', 'UMAP_coordinates_latent_mu_embedding_scRNA-seq_cells.csv.gz'), compression='gzip')
    
    
    # NOTE: based on UMAP
    # plot distribution of number of cells in pseudo-spots
    # add number of cells in pseudo-spot to dataframe
    plot_df['n_cell_in_spot'] = np.nan
    plot_df.loc[plot_df['datatype']=='scrna-cell', 'n_cell_in_spot'] = scrna_n_cell
    plot_df.loc[plot_df['datatype']=='scrna-pseudo', 'n_cell_in_spot'] = n_cell_in_spot
    
    sns.set_style("darkgrid")
 
    # generate a colormap with a specified color for NA (spatial spots), but not work for relplot...
    # show the full legend of colorbar in relplot, otherwise it will only show a sample of evenly spaced values (The FacetGrid hue is categorical, not continuous)
    # finally use plt.subplots instead
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6.4*4, 4.8*2))
    # left panel: scatter plot of pseudo-spots + scRNA-seq cells
    tmp_df = plot_df.loc[plot_df['dataset']=='scRNA-seq']
    sc = ax1.scatter(tmp_df['UMAP1'], tmp_df['UMAP2'], c=tmp_df['n_cell_in_spot'], cmap='cubehelix', s=10, marker='o')
    ax1.set_title('dataset = scRNA-seq')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    
    # right panel: scatter plot of spatial spots + spatial pseudo-spots
    # first draw spatial pseudo-spots, which lay at the bottom
    tmp_df = plot_df.loc[plot_df['datatype']=='spatial-pseudo']
    ax2.scatter(tmp_df['UMAP1'], tmp_df['UMAP2'], color=plot_colors['pseudo'], s=10, marker='o')
    tmp_df = plot_df.loc[plot_df['datatype']=='spatial-spot']
    ax2.scatter(tmp_df['UMAP1'], tmp_df['UMAP2'], color=plot_colors[f'spatial ({spatial_embed.shape[0]})'], s=10, marker='o')
    ax2.set_title('dataset = spatial')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    
    # add colorbar with title to the most right (https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots, conflict with tight_layout)
    cbar = fig.colorbar(sc, ax=ax2)
    cbar.ax.set_title('#cell in spot')
    
    fig.tight_layout()
    fig.savefig(os.path.join(diagnosis_path, 'CVAE_latent_space', 'UMAP_latent_mu_embedding_color_by_num_cell_in_spot.png'))
    plt.close()
    
    
    # NOTE: based on UMAP
    # plot distribution of cell-type proportions of each cell-type
    # create a proportion dataframe with the same row order
    prop_df = pd.concat([pseudo_spots_celltype_prop,
                         pd.DataFrame(0, index=['spatial_pseudo' + str(idx) for idx in range(pseudo_spatial_embed.shape[0])], columns=pseudo_spots_celltype_prop.columns),
                         scrna_cell_celltype_prop,
                         pd.DataFrame(0, index=spatial_transformed_df.index, columns=pseudo_spots_celltype_prop.columns)
                        ], ignore_index=False)
    assert prop_df.shape[0] == plot_df.shape[0]
    assert (prop_df.index == plot_df.index).all()
    
    # save multiple figures into one PDF
    with PdfPages(os.path.join(diagnosis_path, 'CVAE_latent_space', 'UMAP_latent_mu_embedding_color_by_celltype_proportion.pdf')) as pdf:
        for this_celltype in celltype_order:
        
            # start to plot
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6.4*4, 4.8*2))
            
            # left panel: scatter plot of pseudo-spots + scRNA-seq cells
            tmp_df = plot_df.loc[plot_df['dataset']=='scRNA-seq']
            # don't forget the .values
            tmp_df = tmp_df.assign(proportion = prop_df.loc[plot_df['dataset']=='scRNA-seq', this_celltype].values)
            
            sc = ax1.scatter(tmp_df['UMAP1'], tmp_df['UMAP2'], c=tmp_df['proportion'], cmap='cubehelix', s=10, marker='o', norm=Normalize(vmin=0, vmax=1))
       
            ax1.set_title('dataset = scRNA-seq')
            ax1.set_xlabel('UMAP 1')
            ax1.set_ylabel('UMAP 2')
        
            # right panel: scatter plot of spatial spots + spatial pseudo-spots
            # we also interpolate the grid and draw a contour plot of cell type proportions in right panel
            grid_x, grid_y = np.mgrid[tmp_df['UMAP1'].min():tmp_df['UMAP1'].max():0.025, tmp_df['UMAP2'].min():tmp_df['UMAP2'].max():0.025]
            grid_z = griddata(tmp_df.loc[:, ['UMAP1', 'UMAP2']].values, tmp_df['proportion'].values, (grid_x, grid_y), method='linear',  fill_value=np.nan)
        
            try:
                ax2.contourf(grid_x, grid_y, grid_z, cmap='cubehelix', norm=Normalize(vmin=0, vmax=1), alpha=0.3)
            except:
                pass
        
            # first draw spatial pseudo-spots, which lay at the bottom
            tmp_df = plot_df.loc[plot_df['datatype']=='spatial-pseudo']
            ax2.scatter(tmp_df['UMAP1'], tmp_df['UMAP2'], color=plot_colors['pseudo'], s=10, marker='o')
            tmp_df = plot_df.loc[plot_df['datatype']=='spatial-spot']
            ax2.scatter(tmp_df['UMAP1'], tmp_df['UMAP2'], color=plot_colors[f'spatial ({spatial_embed.shape[0]})'], s=10, marker='o')
        
            ax2.set_title('dataset = spatial')
            ax2.set_xlabel('UMAP 1')
            ax2.set_ylabel('UMAP 2')
        
            # add colorbar with title
            cbar = fig.colorbar(sc, ax=ax2)
            cbar.ax.set_title('proportion')
        
            fig.suptitle(this_celltype)
        
            fig.tight_layout()
            
            pdf.savefig(fig)  # Save the current figure to pdf
            plt.close(fig)  # Close the figure to free memory
    
    
    # plot PCA of latent space of spatial spots and scRNA-seq cells plus pseudo spots
    # the order will affect the point overlay, first row draw first
    all_pca = PCA(n_components=2).fit_transform(np.concatenate((pseudo_spot_embed, pseudo_spatial_embed, scRNA_embed, spatial_embed), axis=0))
    
    # add cell/spot count in the annotation
    plot_df = pd.DataFrame({'PC1': all_pca[:, 0],
                            'PC2': all_pca[:, 1],
                            'celltype': ['pseudo']*pseudo_spot_embed.shape[0] +
                                        ['pseudo']*pseudo_spatial_embed.shape[0] + 
                                        [f'{x} ({celltype_count_dict[x]})' for x in scRNA_celltype.celltype.to_list()] +
                                        [f'spatial ({spatial_embed.shape[0]})']*spatial_embed.shape[0],
                            'dataset': ['scRNA-seq']*pseudo_spot_embed.shape[0] +
                                       ['spatial']*pseudo_spatial_embed.shape[0] +
                                       ['scRNA-seq']*scRNA_embed.shape[0] +
                                       ['spatial']*spatial_embed.shape[0],
                            'datatype': ['scrna-pseudo']*pseudo_spot_embed.shape[0] +
                                        ['spatial-pseudo']*pseudo_spatial_embed.shape[0] +
                                        ['scrna-cell']*scRNA_embed.shape[0] +
                                        ['spatial-spot']*spatial_embed.shape[0],
                            },
                           index = [f'scrna_pseudo{x}' for x in range(pseudo_spot_embed.shape[0])] +
                                   [f'spatial_pseudo{x}' for x in range(pseudo_spatial_embed.shape[0])] +
                                   scRNA_decode_df.index.to_list() +
                                   spatial_transformed_df.index.to_list())
    
    # plot colors already defined
    sns.set_style("darkgrid")
    
    # relplot return a FacetGrid object
    # specify figure size by Height (in inches) of each facet, and Aspect ratio of each facet
    fgrid = sns.relplot(data=plot_df, x='PC1', y='PC2', hue='celltype', s=20, marker='o', palette=plot_colors, kind='scatter', col='dataset', col_order=['scRNA-seq', 'spatial'], height=4.8*2, aspect=6.4/4.8)
    fgrid.set(xlabel='PC 1', ylabel='PC 2')
    # Put the legend out of the figure
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.savefig(os.path.join(diagnosis_path, 'CVAE_latent_space', 'PCA_latent_mu_embedding_color_by_celltype.png'))
    plt.close()
    
    
    # save PCA coordinates of latent space
    plot_df.loc[plot_df['datatype']=='spatial-spot', ['PC1', 'PC2']].to_csv(os.path.join(diagnosis_path, 'CVAE_latent_space', 'PCA_coordinates_latent_mu_embedding_spatial_spots.csv.gz'), compression='gzip')
    # save scRNA-seq cells PCA coordinates
    plot_df.loc[plot_df['datatype']=='scrna-cell', ['PC1', 'PC2']].to_csv(os.path.join(diagnosis_path, 'CVAE_latent_space', 'PCA_coordinates_latent_mu_embedding_scRNA-seq_cells.csv.gz'), compression='gzip')
    
    
    # Plot PCA density
    # for safe to avoid program exit when density estimation failed
    try:
        plt.figure(figsize=(6.4*2, 4.8*2))
        sns.set_style('whitegrid')
        ax = sns.kdeplot(data=plot_df, x='PC1', y='PC2', fill=True)
        ax.set(xlabel='PC 1', ylabel='PC 2')
        plt.savefig(os.path.join(diagnosis_path, 'CVAE_latent_space', 'PCA_latent_mu_embedding_color_by_density.png'))
        plt.close()
    except:
        pass
    
    
    # UMAP of decoded spatial and scRNA-seq cell gene expression
    # now we can add marker gene expression profile here
    # the order will affect the point overlay, first row draw first
    # umap has embeded seed (default 42), by specify random_state, umap will use special mode to keep reproducibility
    # note we use normalized value rather than raw nUMI for spatial spots
    tmp_df = pd.concat([spatial_transformed_df, scRNA_decode_df, scRNA_decode_avg_df], ignore_index=True)
    # only use marker genes
    if new_markers is not None:
        tmp_df = tmp_df[new_markers]
    
    all_umap = umap.UMAP(random_state=42).fit_transform(tmp_df.values)
    
    # add cell/spot count in the annotation
    plot_df = pd.DataFrame({'UMAP1': all_umap[:, 0],
                            'UMAP2': all_umap[:, 1],
                            'celltype': [f'spatial ({spatial_transformed_df.shape[0]})']*spatial_transformed_df.shape[0] +
                                        [f'{x} ({celltype_count_dict[x]})' for x in scRNA_celltype.celltype.to_list()] +
                                        [f'{x} ({celltype_count_dict[x]})' for x in scRNA_decode_avg_df.index],
                            'dataset': ['spatial']*spatial_transformed_df.shape[0] +
                                       ['scRNA-seq']*scRNA_decode_df.shape[0] +
                                       ['scRNA-seq']*scRNA_decode_avg_df.shape[0],
                            'datatype': ['cell/spot']*spatial_transformed_df.shape[0] +
                                        ['cell/spot']*scRNA_decode_df.shape[0] +
                                        ['marker']*scRNA_decode_avg_df.shape[0]
                            },
                           index = spatial_transformed_df.index.to_list() +
                                   scRNA_decode_df.index.to_list() +
                                   [f'{x}-marker' for x in scRNA_decode_avg_df.index])
    
    plot_sizes = {'cell/spot': 20, 'marker': 200}
    plot_markers = {'cell/spot': 'o', 'marker': 'X'}
    # color for plot already defined

    sns.set_style("darkgrid")
    
    # relplot return a FacetGrid object
    # specify figure size by Height (in inches) of each facet, and Aspect ratio of each facet
    fgrid = sns.relplot(data=plot_df, x='UMAP1', y='UMAP2', hue='celltype', size='datatype', style='datatype', sizes=plot_sizes, markers=plot_markers, palette=plot_colors, kind='scatter', col='dataset', col_order=['scRNA-seq', 'spatial'], height=4.8*2, aspect=6.4/4.8)
    fgrid.set(xlabel='UMAP 1', ylabel='UMAP 2')
    # Put the legend out of the figure
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # add cell-type annotations around marker coordinates
    # adjustText do not support seaborn relplot
    #from adjustText import adjust_text
    # fgrid.axes return an array of all axes in the figure
    ax = fgrid.axes[0, 0]
    texts = []
    for one_row in scRNA_decode_avg_df.index:
        texts.append(ax.text(plot_df.at[one_row+'-marker', 'UMAP1'], plot_df.at[one_row+'-marker', 'UMAP2'], one_row, weight='bold'))
    #adjust_text(texts)
    
    plt.savefig(os.path.join(diagnosis_path, 'CVAE_transformed_data', 'UMAP_decoded_value_color_by_celltype.png'))
    plt.close()
    
    
    # also save UMAP coordinates
    plot_df[['UMAP1', 'UMAP2']].to_csv(os.path.join(diagnosis_path, 'CVAE_transformed_data', 'UMAP_coordinates_decoded_value.csv.gz'), compression='gzip')



def diagnosisParamsTuning(x, y, optimal_idx, x_label, y_label):
    '''
    generate scatter plot show performance in hyper parameter tuning by cross-validation

    Parameters
    ----------
    x : list
        a list of candidates of hyper parameter
    y : list
        corresponding performance for each hyper parameter candidate. NOTE not all candidates have corresponding performance due to early stop
    optimal_idx : int
        index of list to indicate the optimal hyper parameter value which achieve the best performance
    x_label : string
        label of x axis in the plot, representing the number of hyper parameter
    y_label : string
        label of y axis in the plot, representing the performance metric used

    Returns
    -------
    None.
    '''
    
    # need to create subfolders first, otherwise got FileNotFoundError
    os.makedirs(os.path.join(diagnosis_path, 'GLRM_params_tuning'), exist_ok=True)
    
    sns.set()
    
    fig, ax = plt.subplots()
    ax.plot(x[:len(y)], y, marker='o')
    
    # if the range of the x-axis exceeds 100 (where the ratio of the largest to smallest value is greater than 100), we utilize a logarithmic scale for the x-axis in the plot.
    # log scale do not support 0, so use symlog
    # "linear threshold" determines the range around zero within which the scale will be linear ranther than instead of logarithmic to avoid distortion that occurs with logarithmic scaling near zero and make the plot more readable
    if x[-1] / x[0] > 100:
        ax.set_xscale('symlog', linthresh=x[1])
    
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=45)
    ax.axvline(x[optimal_idx], color='red', linestyle='--')
        
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
        
    fig.tight_layout()
    fig.savefig(os.path.join(diagnosis_path, 'GLRM_params_tuning', f'tuning_{x_label}_by_cv.png'))
    plt.close()



def plot_imputation(df, grid_df, contours, hierarchy, figname, figsize=(6.4, 4.8)):
    '''
    draw scatter plot of spatial spots and imputed spots for diagnosis.

    Parameters
    ----------
    df : dataframe
        dataframe of spatial spots at original resolution, with columns 'x', 'y', 'border'.
    grid_df : dataframe
        dataframe of generated grid points at higher resolution, with columns 'x', 'y'.
    contours : tuple
        contours variable returned by cv2.findContours, used for creating grid.
    hierarchy : 3-D numpy array (1 * #contours * 4)
        hierarchy variable returned by cv2.findContours, used for creating grid.
    figname: string
        name of figure.
    figsize : tuple, optional
        figure size. The default is (6.4, 4.8).

    Returns
    -------
    None.
    '''
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()
    sns.set_style('white')
    
    plt.figure(figsize=figsize)
    
    # scatter plot of original spatial spots, highlighting edge spots
    sns.scatterplot(data=df, x='x', y='y', hue='border', s=100, alpha=0.5, legend=False)
    
    outer_edges = [] # each element is a single outer edge with several points
    inner_edges = [] # each element is a single inner edge with several points
    
    # Go through all contours and hierarchy
    for i, (cnt, hrc) in enumerate(zip(contours, hierarchy[0])):
        # Convert contour points back to original coordinates
        cnt = cnt.reshape(-1, 2)
        
        # Check if it's an outer or inner edge (hierarchy: [Next, Previous, First Child, Parent])
        if hrc[3] == -1:  # it's an outer edge if it has no parent
            outer_edges.append(cnt)
        else:  # it's an inner edge if it has a parent
            inner_edges.append(cnt)
    
    # add edges
    # connect the last point to the first one
    if len(outer_edges) > 0:
        for tmp_plot in outer_edges:
            tmp_plot = np.append(tmp_plot, [tmp_plot[0]], axis=0)
            plt.plot(tmp_plot[:, 0], tmp_plot[:, 1], 'g--')
    
    if len(inner_edges) > 0:
        for tmp_plot in inner_edges:
            tmp_plot = np.append(tmp_plot, [tmp_plot[0]], axis=0)
            plt.plot(tmp_plot[:, 0], tmp_plot[:, 1], 'r--')
            
    # add generated high resolution grid
    sns.scatterplot(data=grid_df, x='x', y='y', marker='X', color='k')
    
    # need to create subfolders first, otherwise got FileNotFoundError
    os.makedirs(os.path.join(diagnosis_path, 'imputation'), exist_ok=True)
    
    plt.savefig(os.path.join(diagnosis_path, 'imputation', figname))
    
    plt.close()