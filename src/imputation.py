#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 07:15:15 2023

@author: hill103

this script perform imputation on cell-types and gene expression of spatial spots

Steps:
    1. Identify edge spots (both outer and inner edges) at the original resolution
    2. Generate grid at a specified higher resolution (location of imputed smaller spots)
    3. Initialize cell type proportions for grid points
    4. Impute cell type proportions theta at higher resolution
    5. Further impute gene expression X at higher resolution
"""



from getopt import getopt
import sys, os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from config import print, input_path, output_path, cur_version
from time import time
import copy
import cv2



# --------------------------- imputation related functions -------------------------

def check_technique(df):
    '''
    check the input spatial tissue map is derived from ST or 10x Visium technique.
    The distance of two neighboring spots in the same row is 1 for ST and 2 for 10x.
    We only select 5 rows with most spots for checking.

    Parameters
    ----------
    df : dataframe
        dataframe of spatial spots, with columns 'x', 'y'.

    Returns
    -------
    tech_mode : str
        the technique for this dataset, either 'st' or 'visium'
    '''
    
    print('\nfirst determine the technique of this dataset')
    
    # 1. Count the points for each y (row).
    point_counts = df.groupby('y').size()
    
    # 2. Select the 5 rows with the most points.
    top_y_values = point_counts.nlargest(5).index
    
    # 3. Loop over each of the 5 rows
    tech_mode = 'visium'
    
    for y_val in top_y_values:
        subset = df[df['y'] == y_val]
    
        # Order the points from smallest x to largest x
        ordered_x = subset['x'].sort_values().values
    
        # Check if the difference between two neighboring x values is divisible by 2
        differences = np.diff(ordered_x)
        
        if np.any(differences % 2 != 0):
            # Found a difference that's not divisible by 2
            tech_mode = 'st'
            break

    print(f'\tafter checking, it is "{tech_mode}" technique')
    
    return tech_mode


def identify_edges(df, mode, keep_filled=False):
    '''
    identify edge spots. Column 'border' of Edge spots are True.
    NOTE edge spot identification is based on 'x' and 'y' coordinate indexes not pixels.
    when keep_filled=True, filled spots for 10x will be kept for creating grid.

    Parameters
    ----------
    df : dataframe
        dataframe of spatial spots, with columns 'x', 'y'.
    mode : string
        either 'st' for Spatial Transcriptomics technique, or 'visium' for 10x Genomics Visium technique.
    keep_filled : bool, optional
        if True, filled spots for 10x will be kept for creating grid.

    Returns
    -------
    df : dataframe
        dataframe with extra columns: 'border', 'inner_border', 'outer_border', 'filled'.
    contours : tuple
        contours variable returned by cv2.findContours, used for creating grid.
    hierarchy : 3-D numpy array (1 * #contours * 4)
        hierarchy variable returned by cv2.findContours, used for creating grid.
    contour_info : list of dicts
        parsed contour information used for creating grid.
    '''
    
    def fill_spots(df):
        '''
        fill spots for finding contour row by row given a dataframe containing points, for 10x Visium.
        Note: for one spot to be filled, also need to check whether its top or bottom spots exist!
        '''
        points = set(df[['x', 'y']].apply(tuple, axis=1))
        # Group by 'y'
        grouped = df.groupby('y')
        # Prepare a list to hold the new points
        new_points = []
        # For each group
        for name, group in grouped:
            # Get the 'x' values as a list
            xs = sorted(group['x'].tolist())
            # Check each pair of consecutive 'x' values
            for i in range(len(xs) - 1):
                # If the difference is equals 2, generate the missing points in between
                if xs[i+1] - xs[i] == 2:
                    # further check whether top and bottom spots exist
                    # Use `or` rather than `and` is to be consistent with how we define the neighbourhood of points in 10x Visium
                    # where the top and bottom spots are not counted as neighbours!!!
                    if ((xs[i]+1, name+1) in points) or ((xs[i]+1, name-1) in points):
                        new_points.append((xs[i]+1, name))
                        
        # return a dataframe
        new_df = pd.DataFrame(new_points, columns=['x', 'y'])
        new_df['filled'] = 1
        return new_df
    
    def count_missing_points_in_hole(df, distance):
        '''
        distance is 1 for ST, 2 for 10x Visium.
        '''
        df = df.copy(deep=True)
        df.sort_values(['y', 'x'], inplace=True)
        # Group by 'y'
        grouped = df.groupby('y')
        n = 0
        # For each group
        for _, group in grouped:
            # Get the 'x' values as a list
            xs = sorted(group['x'].tolist())
            # Check each pair of consecutive 'x' values
            for i in range(len(xs) - 1):
                # If the difference is > distance, count the missing points in between
                if xs[i+1] - xs[i] > distance:
                    # The formula `(stop - start - 1) // step` is used to calculate the number of steps between start and stop
                    # given a specific step size (both start and stop are exclusive; `start` is at the start of a step; not require `stop` is at the stop of a step)
                    # Formula `(stop - start - step) // step` requires both `start` and `stop` is at the start and stop of a step, respectively, which is also true here
                    n += (xs[i+1]-xs[i]-1) // distance
        return n
    
    def map_func(row, sets):
        # WARNING: This approach assumes that a point cannot belong to more than one inner edge.
        # If a point can belong to multiple edges, this script will only assign it to the first inner edge it belongs to
        # If the point is not in any set, it returns 0 
        for i, edge_set in enumerate(sets):
            # note it's (x,y) representation in opencv points
            if (row.x, row.y) in edge_set:
                return i+1
        return 0
    
    
    df = df.copy(deep=True)
    df['filled'] = 0
    
    # Calculate the min and max values of x and y
    max_x = df['x'].max()
    max_y = df['y'].max()
    
    # Append the new points to the original dataframe
    if mode == 'visium':
        filled_df = pd.concat([df, fill_spots(df)], ignore_index=False)
        if keep_filled:
            df = filled_df.copy(deep=True)
    elif mode == 'st':
        filled_df = df

    # Create a zeros matrix of the required size
    # note it's image with (row, column) representation
    image = np.zeros((max_y + 1, max_x + 1), np.uint8)

    # Fill in the points
    for _, row in filled_df.iterrows():
        image[int(row['y']), int(row['x'])] = 255
        
    # Find contours (https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
    # `RETR_TREE` find all contours (outer and inner)
    # `CHAIN_APPROX_NONE` store all contour points and reconstructs a full hierarchy of nested contours
    # The 'contours' variable contains the contours, a tuple of all found contours
    # each contour is a numpy array of (x, y) coordinates of boundary points, note the shape is (n,1,2), n is number of points in this contour
    # The 'hierarchy' variable contains info about the image topology (type numpy.ndarray)
    # note the shape is (1,m,4), m is number of contours found, equals the length of contours
    # access the info for i-th (0-based indices) contour via hierarchy[0]
    # hierarchy[0][i][0] ~ hierarchy[0][i][3] are [Next, Previous, First_Child, Parent]
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    
    # create a list of dicts including all contour info for creating grid
    contour_info = []
    
    # Extract outer and inner edge coordinates for mapping back to points in dataframe
    outer_edges = [] # each element is a single outer edge with several points
    inner_edges = [] # each element is a single inner edge with several points
    
    if mode == 'visium':
        this_distance = 2
    elif mode == 'st':
        this_distance = 1

    # Go through all contours and hierarchy ([Next, Previous, First Child, Parent])
    for i, contour in enumerate(contours):
        
        # If the contour has a parent, skip it (it's a hole)
        if hierarchy[0, i, 3] != -1:
            continue
        
        # Convert contour points back to original coordinates, i.e. a 2-D numpy array
        tmp_cnt = contour.reshape(-1, 2)
        outer_edges.append(tmp_cnt)
        
        tmp_dict = {'origin_idx': i, 'n_points': tmp_cnt.shape[0], 'holes': []}
        
        # Get all child contours (holes)
        # Get the first child of the contour (hierarchy: [Next, Previous, First Child, Parent])
        child_idx = hierarchy[0, i, 2]
        
        # Iterate through the list of child contours
        while child_idx != -1:
            
            tmp_cnt = contours[child_idx].reshape(-1, 2)
            inner_edges.append(tmp_cnt)
            
            tmp_dict['holes'].append({'child_idx': child_idx,
                                      'n_points': tmp_cnt.shape[0],
                                      'area': cv2.contourArea(tmp_cnt),
                                      'n_missing_spots': count_missing_points_in_hole(pd.DataFrame(tmp_cnt, columns=["x","y"]), distance=this_distance)})

            # Proceed to the next contour at the same level
            child_idx = hierarchy[0, child_idx, 0]
        
        # add dict to list
        contour_info.append(tmp_dict)
    
    print(f'total {len(contour_info)} set of outer borders\n')
    if len(contour_info) > 0:
        for i, this_info_dict in enumerate(contour_info):
            print(f'outer border {i} (0-based index in opencv contour {this_info_dict["origin_idx"]}):')
            print(f'total {this_info_dict["n_points"]} edge points in it')
            if len(this_info_dict['holes']) > 0:
                print(f'it contains total {len(this_info_dict["holes"])} holes (inner borders)')
                for j, this_child_dict in enumerate(this_info_dict['holes']):
                    print(f'\n\thole {j} (0-based index in opencv contour {this_child_dict["child_idx"]}):')
                    print(f'\ttotal {this_child_dict["n_points"]} edge points in it')
                    print(f'\tarea: {this_child_dict["area"]}')
                    print(f'\tmissing spots in hole: {this_child_dict["n_missing_spots"]}')
            else:
                print('no holes in it')
            print('\n')

   
    # Now `outer_edges` and `inner_edges` are lists of 2D NumPy arrays
    # where each array contains the **(x, y)** coordinates (this inconsistency makes me crazy)
    # label the dataframe
    if len(outer_edges) > 0:
        outer_edge_sets = [set(map(tuple, edge)) for edge in outer_edges]
        # Create a new column "outer_border" that indicates which inner edge a point belongs to
        df['outer_border'] = df.apply(map_func, sets=outer_edge_sets, axis=1)
        #print(df['outer_border'].value_counts())
    else:
        df['outer_border'] = 0
    
    if len(inner_edges) > 0:
        inner_edge_sets = [set(map(tuple, edge)) for edge in inner_edges]
        # Create a new column "inner_border" that indicates which inner edge a point belongs to
        df['inner_border'] = df.apply(map_func, sets=inner_edge_sets, axis=1)
        #print(df['inner_border'].value_counts())
    else:
        df['inner_border'] = 0
    
    df['border'] = (df['outer_border']>0) | (df['inner_border']>0)
    print(f'total identified {df["border"].sum()} border points (outer+inner)')
    
    return df, contours, hierarchy, contour_info


def create_grid(df, contours, hierarchy, contour_info, grid_step, miss_spot_threshold, add_edge=False):
    '''
    create a high resolution grid inside the contours.
    we first create a grid on the bounding box of the contour with any shape, then filter out any points outside the contour.
    we also consider the generated points falling inside the holes and determine whether to filter them.
    add_edge controls whether add (x,y) coordinates of edge points to the grid, which will cause the step of grid to be uneven.
    we filter the generated points row-by-row.
    creatied grid are independent across multiple outer contours if they exist,
    so in the imputed heatmap, there may be some empty lines inside one contour shape as some points at the same line are presented in other contours

    Parameters
    ----------
    df : dataframe
        dataframe of spatial spots, with columns 'x', 'y', and extra columns: 'border', 'inner_border', 'outer_border'.
    contours : tuple
        contours variable returned by cv2.findContours, used for creating grid.
    hierarchy : 3-D numpy array (1 * #contours * 4)
        hierarchy variable returned by cv2.findContours, used for creating grid.
    contour_info : list of dicts
        parsed contour information used for creating grid.
    grid_step : float
        a float number in (0,1), served as the steps of grid.
    miss_spot_threshold : int
        if the number of missing (uncaptured) spots in an identified hole is less or equal this value, this hole will be ignored as if there is no hole at all.
    add_edge : bool, optional
        if True, add (x,y) coordinates of contour edge points to the grid. Default is False.
    
    Returns
    -------
    grid_df : dataframe
        dataframe of generated poins (passed filtering), with columns 'x', 'y', 'contour_idx'.
    '''
    
    # Initialize an empty list to store all grids points
    grids_points = []

    # For each outer contour
    for contour_idx, this_info_dict in enumerate(contour_info):
        
        print(f'\nprocessing outer border {contour_idx} (0-based index in opencv contour {this_info_dict["origin_idx"]}):')
        print(f'\ttotal {this_info_dict["n_points"]} edge points in it')
        print(f'\twith {len(this_info_dict["holes"])} holes in it')
        
        this_contour = contours[this_info_dict["origin_idx"]]
        this_contour_points = this_contour.reshape(-1, 2)
        
        # if this contour only has 1 edge point, do not create grid, just return this point for imputation
        if this_info_dict["n_points"] == 1:
            # note it's a 1*2 array
            grids_points.append(this_contour_points.tolist()[0] + [contour_idx])
            continue
        
        # count holes larger than the threshold
        this_hole_contours = []
        this_hole_points = []
        
        for this_child_dict in this_info_dict['holes']:
            if this_child_dict["n_missing_spots"] > miss_spot_threshold:
                this_hole_contours.append(contours[this_child_dict['child_idx']])
                this_hole_points.append(contours[this_child_dict['child_idx']].reshape(-1, 2))

        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(this_contour.astype(int))
        x_line = np.round(np.arange(x, x+w, grid_step), decimals=3)
        y_line = np.round(np.arange(y, y+h, grid_step), decimals=3)
        # add end point if it's not included
        if x_line[-1] < x+w:
            x_line = np.append(x_line, x+w)
        if y_line[-1] < y+h:
            y_line = np.append(y_line, y+h)
            
        if add_edge:
            # add x and y coordinates of current outer contour points to grid
            x_line = np.concatenate((x_line, this_contour_points[:, 0]))
            y_line = np.concatenate((y_line, this_contour_points[:, 1]))
            
            # add x and y coordinates of edges of holes
            for one_hole_point_set in this_hole_points:
                x_line = np.concatenate((x_line, one_hole_point_set[:, 0]))
                y_line = np.concatenate((y_line, one_hole_point_set[:, 1]))
            
            # remove duplicates
            x_line = np.sort(np.unique(x_line))
            y_line = np.sort(np.unique(y_line))

        # Create a grid of points inside the bounding box
        xx, yy = np.meshgrid(x_line, y_line)
        
        n_this_grid_points = 0

        for j in range(xx.shape[1]):
            for i in range(xx.shape[0]):
                # when measureDist=False, point inside contour: +1; on contour: 0; outside: -1
                if cv2.pointPolygonTest(this_contour, (xx[i,j], yy[i,j]), False) >= 0:
                    # further check this grid point not in holes
                    this_point_in_hole = False
                    for one_hole_contour in this_hole_contours:
                        if cv2.pointPolygonTest(one_hole_contour, (xx[i,j], yy[i,j]), False) > 0:
                            this_point_in_hole = True
                            break
                    if not this_point_in_hole:
                        n_this_grid_points += 1
                        grids_points.append([xx[i,j], yy[i,j], contour_idx])
 
        print(f'\tget {n_this_grid_points} grid points')
    
    print(f'\ntotal {len(grids_points)} generated grid points')
    
    return pd.DataFrame(grids_points, columns=["x", "y", "contour_idx"])


def initialize_theta_grid_points(df, theta, grid_df):
    '''
    initialize cell type proportions grid_theta for generated grid points in higher resolution.
    for one grid point, grid_theta is set as the theta value of spatial spot in original resolution with the smallest Euclidean distance.
    if there are multiple spatial spots with the same smallest Euclidean distance, then take average of theta of those spots
    finally we will normalize all grid_theta to sum to 1.

    Parameters
    ----------
    df : dataframe
        dataframe of spatial spots at original resolution, with columns 'x', 'y'.
    theta : dataframe
        cell type proportions of spatial spots at original resolution, with columns are cell types.
    grid_df : dataframe
        dataframe of generated grid points at higher resolution, with columns 'x', 'y'.

    Returns
    -------
    grid_theta : dataframe
        cell type proportions of generated grid points at higher resolution, with columns are cell types.
    '''

    # Compute pairwise distances (row: grid point; column: spatial spot)
    distances = cdist(grid_df[['x','y']].to_numpy(), df[['x','y']].to_numpy(), 'euclidean')
    
    # For each row in grid, get the indices with the smallest distance in spatial spots
    # distances.min(axis=1)[:, None] generate a n*1 matrix with each row is the smallest distance
    # then create a boolean matrix of the same shape as distances, where each element is True if it is the minimum distance in its row
    # The np.where function returns the indices of True values in the boolean matrix. The output is a tuple of arrays, where the first array contains the row indices and the second array contains the corresponding column indices. If multiple True values in one row in the boolean matrix, then all of them will be recorded, and use the row indices to access them
    min_distance_indices = np.where(distances == distances.min(axis=1)[:, None])
    
    # Create an empty dataframe to store results
    grid_theta = pd.DataFrame(np.zeros((grid_df.shape[0], theta.shape[1])),
                              index=grid_df.index.tolist(), columns=theta.columns.tolist())
    
    # Iterate over each point in grid
    for i in range(grid_df.shape[0]):
        # Get indices in spatial spots with the smallest distance to the current point in grid
        # note min_distance_indices includes all pairs (grid point - spatial spot) with smallest distances
        relevant_indices = min_distance_indices[1][min_distance_indices[0] == i]
        
        # Compute the average of corresponding rows in theta and assign to grid_theta
        grid_theta.iloc[i] = theta.iloc[relevant_indices].mean()
    
    # Normalize each row in grid to sum to 1
    grid_theta = grid_theta.div(grid_theta.sum(axis=1), axis=0)
    
    return grid_theta


def construct_M_matrix(df, sigma, phi, lazy_rw=False, alpha=0.5):
    '''
    construct nearest neighbor random walk matrix M, also called transition probability matrix in compute science. The value in M represents the probability that, given a node, the random walker will stay on that node in the next step
        W(x,y) = exp(-||x-y||^2 / 2*sigma^2)
        
    matrix M is based on a Gaussian kernel W with SD/bandwidth `sigma` and distance threshold `phi`
        M = D^−1 * W
        
    if `lazy_rw` is True, i.e. use lazy random walk with a probability `alpha` of staying at the current node, an additional lazy transition matrix will be calculated
        P_lazy = alpha * I + (1−alpha) * P

    Parameters
    ----------
    df : dataframe
        dataframe of generated grid points at higher resolution, with columns 'x', 'y'.
    sigma : float
        standard deviation of Gaussian kernel W.
    phi : float
        threshold of Euclidean distance in Gaussian kernel W. If two points with distance > phi, the corresponding weight in W will be 0.
    lazy_rw : bool, optional
        if True, return lazy random walk matrix. The default is False.
    alpha : float, optional
        the probability `alpha` of staying at the current node. The default is 0.5.

    Returns
    -------
    M : 2-D numpy array
        the transition probalility matrix for random walk.
    '''
    
    # points with (x,y) coordinates
    points = df[['x','y']].to_numpy()
    
    # Compute pairwise distances using cdist
    distances = cdist(points, points)
    
    # Apply the threshold and compute the Gaussian kernel values
    W = np.exp(-distances**2 / (2 * sigma**2))  # Gaussian kernel with specified sigma
    W[distances > phi] = 0  # Apply the threshold
    
    D_inv = np.diag(1 / W.sum(axis=1))
    # Starting from NumPy version 1.10.0 and later, there's a dedicated matrix multiplication operator @
    M = D_inv @ W
    
    if lazy_rw:
        M = alpha * np.eye(df.shape[0]) + (1 - alpha) * M
    
    return M


def do_random_walk(df, theta, sigma, phi, n_step=1, lazy_rw=False, alpha=0.5):
    '''
    impute cell type proportion theta for generated grid points at higher resolution using one step random walk.
    
    note if the input cell type proportions are normalized (i.e., each row sums to 1), the imputed results using a valid transition probability matrix will also be normalized.

    Parameters
    ----------
    df : dataframe
        dataframe of generated grid points at higher resolution, with columns 'x', 'y'.
    theta : dataframe
        initial cell type proportions of generated grid points at higher resolution, with columns are cell types.
    sigma : float
        standard deviation of Gaussian kernel W.
    phi : float
        threshold of Euclidean distance in Gaussian kernel W. If two points with distance > phi, the corresponding weight in W will be 0.
    n_step : int, optional
        steps for random walk. The default is 1.
    lazy_rw : bool, optional
        if True, return lazy random walk matrix. The default is False.
    alpha : float, optional
        the probability `alpha` of staying at the current node. The default is 0.5.

    Returns
    -------
    impute_theta : dataframe
        imputed cell type proportions of generated grid points at higher resolution, with columns are cell types.
    '''
    
    # construct transition probability matrix M
    M = construct_M_matrix(df, sigma, phi, lazy_rw, alpha)
    
    # updating the rows of theta based on the transition probabilities in M
    impute_theta = theta.to_numpy()
    
    for i in range(n_step):
        # Starting from NumPy version 1.10.0 and later, there's a dedicated matrix multiplication operator @
        impute_theta = M @ impute_theta
        
    return pd.DataFrame(impute_theta, index=theta.index.tolist(), columns=theta.columns.tolist())


def impute_spots_with_theta(df, theta, grid_step=0.5, miss_spot_threshold=1, add_edge=False, sigma=0.35, phi=1, diagnosis=False):
    '''
    given the spatial spots location and cell type proportions, impute smaller spots at a higher resolution, return the imputed location and cell type proportions for these smaller spots.

    Parameters
    ----------
    df : dataframe
        dataframe of spatial spots at original resolution, with columns 'x', 'y'.
    theta : dataframe
        cell type proportions of spatial spots at original resolution, with columns are cell types.
    grid_step : float, optional
        step size of generated grid at higher resolution.It also equals the distance of two neighboring imputed spots, given the distance of two neighboring spatial spots is 1 for ST or 2 for 10x Visium techniques. The default is 0.5.
    miss_spot_threshold : int, optional
        if the number of missing (uncaptured) spots in an identified hole is less or equal this value, this hole will be ignored as if there is no hole at all. The default is 1.
    add_edge : bool, optional
        if True, add (x,y) coordinates of contour edge points to the grid. Default is False.
    sigma : float, optional
        standard deviation of Gaussian kernel W. The default is 0.35.
    phi : float, optional
        threshold of Euclidean distance in Gaussian kernel W. If two points with distance > phi, the corresponding weight in W will be 0. The default is 1.
    diagnosis : bool, optional
        if True save the scatter plot including spatial spots and imputed spots.

    Returns
    -------
    impute_df : dataframe
        dataframe of generated grid points at higher resolution, with columns 'x', 'y', 'contour_idx'.
    impute_theta: dataframe
        imputed cell type proportions of generated grid points at higher resolution, with columns are cell types.
    '''
    
    # check technique
    tech_mode = check_technique(df[['x','y']])
    
    # identify edge spots
    df_label_edge, contours, hierarchy, contour_info = identify_edges(df[['x','y']], tech_mode)
    assert (df_label_edge.index == df.index).all()
    
    # generate grid at higher resolution
    impute_df = create_grid(df_label_edge, contours, hierarchy, contour_info, grid_step, miss_spot_threshold, add_edge)
    
    if diagnosis:
        from diagnosis_plots import plot_imputation
        if tech_mode == 'st':
            plot_imputation(df_label_edge, impute_df, contours, hierarchy, f'imputation_stepsize_{grid_step}.png', figsize=(6.4*2, 4.8*2))
        elif tech_mode == 'visium':
            plot_imputation(df_label_edge, impute_df, contours, hierarchy, f'imputation_stepsize_{grid_step}.png', figsize=(6.4*4, 4.8*4))
            
    # initialize theta for grid points
    initial_theta = initialize_theta_grid_points(df_label_edge[['x','y']], theta, impute_df[['x','y']])
    assert (initial_theta.index == impute_df.index).all()
    assert (initial_theta.columns == theta.columns).all()
    
    # random walk to impute theta
    impute_theta = do_random_walk(impute_df[['x','y']], initial_theta, sigma, phi, n_step=1, lazy_rw=False)
    assert (impute_theta.index == impute_df.index).all()
    assert (impute_theta.columns == theta.columns).all()
    
    return impute_df, impute_theta


def impute_expression(X, theta, grid_theta):
    '''
    impute gene expression for generated grid points at higher resolution.
    The imputed expression are assured to sum to 1 for each spot/row.
    And all values in imputed expression are non-negative, as multiplying non-negative numbers will always yield a non-negative result.
    
    X_imputed = theta_imputed * theta^+ * X
    
    theta^+ = (theta^T * theta)^(−1) * theta^T

    Parameters
    ----------
    X : dataframe
        nUMI counts of spatial spots, rows are spots and columns are gene expressions
    theta : dataframe
        cell type proportions of spatial spots at original resolution, with columns are cell types.
    grid_theta : dataframe
        imputed cell type proportions of generated grid points at higher resolution, with columns are cell types.
        
    Returns
    -------
    X_imputed : dataframe
        imputed normalized gene expression of generated grid points at higher resolution, rows are grid points, columns are gene expressions.
    '''
    
    # normalize X by spot’s sequence depth, i.e. normalize each row by dividing by its sum
    X_norm = X.div(X.sum(axis=1), axis=0)
    
    X_imputed = grid_theta.to_numpy() @ np.linalg.pinv(theta.to_numpy()) @ X_norm.to_numpy()
    
    return pd.DataFrame(X_imputed, index=grid_theta.index.tolist(), columns=X.columns.tolist())


def do_imputation(loc_file, theta_file, spatial_file, grid_step=0.5, miss_spot_threshold=1, add_edge=False, sigma=0.35, phi=1, diagnosis=False):
    '''
    perform imputation given location, cell type proportions and nUMI counts of original spatial spots.
    
    optimal hyperparameters in the Gaussian kernel `sigma` and `phi` are selected based on analysis on coarse-grained data.
    
    this function can be called in Python to perform imputation.

    Parameters
    ----------
    loc_file : string
        full path of input csv file of spot locations in spatial transcriptomic data, with columns x and y (spots * 2).
    theta_file : string
        full path of input csv file of cell type proportions of spots in spatial transcriptomic data (spots * cell types).
    spatial_file : string
        full path of input csv file of raw nUMI counts in spatial transcriptomic data (spots * genes).
    grid_step : float, optional
        step size of generated grid at higher resolution.It also equals the distance of two neighboring imputed spots, given the distance of two neighboring spatial spots is 1 for ST or 2 for 10x Visium techniques. The default is 0.5.
    miss_spot_threshold : int, optional
        if the number of missing (uncaptured) spots in an identified hole is less or equal this value, this hole will be ignored as if there is no hole at all. The default is 1.
    add_edge : bool, optional
        if True, add (x,y) coordinates of contour edge points to the grid. Default is False.
    sigma : float, optional
        standard deviation of Gaussian kernel W. The default is 0.35.
    phi : float, optional
        threshold of Euclidean distance in Gaussian kernel W. If two points with distance > phi, the corresponding weight in W will be 0. The default is 1.
    diagnosis : bool, optional
        if True save the scatter plot including spatial spots and imputed spots.

    Returns
    -------
    impute_df : dataframe
        ataframe of generated grid poins at higher resolution (passed filtering), with columns 'x', 'y', 'contour_idx'.
    impute_theta : dataframe
        imputed cell type proportions of generated grid points at higher resolution, with columns are cell types.
    impute_X : dataframe
        imputed normalized gene expression of generated grid points at higher resolution, rows are grid points, columns are gene expressions.
    '''
    
    # read files
    df = pd.read_csv(loc_file, index_col=0)
    theta = pd.read_csv(theta_file, index_col=0)
    X = pd.read_csv(spatial_file, index_col=0)
    
    # we check spot order here, note some spots may not have cell-type proportions
    # Find the intersection of the indices
    common_index = df.index.intersection(theta.index).intersection(X.index)
    if len(common_index) < df.shape[0]:
        print(f'\nWARNING: {df.shape[0]-len(common_index)} spots removed as lacking of location or cell type proportion information')
    
    # Filter the dataframes to keep only the rows with common indices
    df = df.loc[common_index].copy()
    theta = theta.loc[common_index].copy()
    X = X.loc[common_index].copy()
    
    # make sure the x,y coordinates are integers
    df['x'] = df['x'].round().astype(int)
    df['y'] = df['y'].round().astype(int)
    
    # impute spots at higher resolution, also get imputed cell type proportion theta
    impute_df, impute_theta = impute_spots_with_theta(df, theta, grid_step, miss_spot_threshold, add_edge, sigma, phi, diagnosis)
    
    # impute gene expression at higher resolution
    impute_X = impute_expression(X, theta, impute_theta)
    
    return impute_df, impute_theta, impute_X



# --------------------------- usage related functions -------------------------

# default value for options
default_paramdict = {'spatial_file': None, 'loc_file': None, 'prop_file': None,
                     'diagnosis': False, 
                     'diameter': 200, 'impute_diameter': [160, 114, 80],
                     'hole_min_spots': 1, 'preserve_shape': False
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
runImputation [option][value]...

    -h or --help            print help messages.
    -v or --version         print version of SDePER.
    
    
    --------------- Input-related options -------------------
    
    -q or --query           input csv file of raw nUMI counts of spatial transcriptomic data (spots * genes), with absolute or relative path. Rows as spots and columns as genes. Row header as spot barcodes and column header as gene symbols are both required.
    -l or --loc             input csv file of row/column integer index (x,y) of spatial spots (spots * 2), with absolute or relative path. Rows as spots and columns are coordinates x (column index) and y (row index). Row header as spot barcodes and column header "x","y" are both required. NOTE 1) the column header must be either "x" or "y" (lower case), 2) x and y are integer index (1,2,3,...) not pixels. This spot location file is required for imputation. And the spot order should be consist with row order in spatial nUMI count data.
    -p or --prop            input csv file of cell-type proportions of spots in spatial transcriptomic data (spots * cell types), with absolute or relative path. It can be the result from cell type deconvolution by SDePER, or directly provided by user. Rows as spots and columns as cell types. Row header as spot barcodes and column header as cell type names are required. And the spot order should be consist with row order in spatial nUMI count data.


    --------------- Output-related options ------------------
    
    We do not provide options for renaming output files. All output files are in the same folder as input files.
    For each specified spot diameter d µm, there will be three more output files: 1) imputed spot locations "impute_diameter_d_spot_loc.csv", 2) imputed spot cell-type proportions "impute_diameter_d_spot_celltype_prop.csv", 3) imputed spot gene expressions (already normalized by sequencing depth of spots) "impute_diameter_d_spot_gene_norm_exp.csv.gz".
    
    
    --------------- imputation-related options --------------
    
    --diameter              the physical distance (µm) between centers of two neighboring spatial spots. For Spatial Transcriptomics v1.0 technique it's 200 µm. For 10x Genomics Visium technique it's 100 µm. Default value is {default_paramdict["diameter"]}.
    --impute_diameter       the target distance (µm) between centers of two neighboring spatial spots after imputation. Either one number or an array of numbers separated by "," are supported. Default value is {",".join([str(x) for x in default_paramdict["impute_diameter"]])}, corresponding to the low, medium, high resolution for Spatial Transcriptomics v1.0 technique.
    --hole_min_spots        the minimum number of uncaptured spots required to recognize a hole in the tissue map. Holes with a number of spots less than or equal to this threshold in it are treated as if no hole exists and imputation will be performed within the hole. Default value is {default_paramdict["hole_min_spots"]}, meaning single-spot holes are imputed.
    --preserve_shape        whether to maintain the shape of the tissue map during imputation. If true, all border points are retained in imputation to preserve the tissue's original shape, although this may result in an irregular imputed grid. Default value is {default_paramdict["preserve_shape"]}.
    --diagnosis             If true, a scatter plot displaying spatial spots and imputed spots is generated for diagnostic purposes. Default value is {default_paramdict["diagnosis"]}.
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
            loc_file : full file path of spot locations in spatial transcriptomic data\n
            prop_file : full file path of cell type proportions of spots in spatial transcriptomic data\n
            diameter : the physical diameter of spatial spots\n
            impute_diameter : target spot diameter for imputation\n
            hole_min_spots : threshold of number of uncaptured spots to validate holes\n
            preserve_shape : whether to preserve the exact shape of tissue map\n
            diagnosis : whether to draw plot for diagnosis
    '''
    
    # If there are no parameters, display prompt information and exit
    if len(sys.argv) == 1:
        print('No options exist!')
        print('Use -h or --help for detailed help!')
        sys.exit(1)
        
    # Define command line parameters.
    # The colon (:) after the short option name indicates that the option must have an additional argument
    # The equal sign (=) after the long option name indicates that the option must have an additional argument
    shortargs = 'hq:l:p:v'
    longargs = ['help', 'query=', 'loc=', 'prop=', 'diameter=', 'impute_diameter=', 'hole_min_spots=', 'preserve_shape=', 'diagnosis=', 'version']
    
  
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
        
        
        if opt in ('-l', '--loc'):
            tmp_file = os.path.join(input_path, val)
            if not os.path.isfile(tmp_file):
                # the input is not a valid existing filename
                raise Exception(f'Invalid input file `{tmp_file}` for spot location of spatial transcriptomic data!')
            # Use the `realpath` function to get the real absolute path
            paramdict['loc_file'] = os.path.realpath(tmp_file)
            continue
        
        
        if opt in ('-p', '--prop'):
           tmp_file = os.path.join(input_path, val)
           if not os.path.isfile(tmp_file):
               # the input is not a valid existing filename
               raise Exception(f'Invalid input file `{tmp_file}` for spot cell type proportion of spatial transcriptomic data!')
           # Use the `realpath` function to get the real absolute path
           paramdict['prop_file'] = os.path.realpath(tmp_file)
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
                paramdict['impute_diameter'] = tmp_list
        
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
        
        
        if opt in ('--diagnosis'):
            if val.casefold() == 'true'.casefold():
                paramdict['diagnosis'] = True
            elif val.casefold() == 'false'.casefold():
                paramdict['diagnosis'] = False
            else:
                print(f'WARNING: unrecognized option value `{val}` for diagnosis! Please use string of true or false. Currently diagnosis is set to be default value `{default_paramdict["diagnosis"]}`!')
            continue
        
        
    # double check all options
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
    
    
    # row order checking is skipped as some spot may not have corresponding cell-type proportions
    '''
    # check row index of spots whether consistent
    tmp_count = pd.read_csv(paramdict['spatial_file'], index_col=0)
    tmp_loc = pd.read_csv(paramdict['loc_file'], index_col=0)
    tmp_prop = pd.read_csv(paramdict['prop_file'], index_col=0)
    
    assert (tmp_count.index==tmp_loc.index).all(), 'ERROR: order of spot barcode in gene expression and spot location not consistent!'
    assert (tmp_count.index==tmp_prop.index).all(), 'ERROR: order of spot barcode in gene expression and spot proportion not consistent!'
    '''
    
    print('\nrunning options:')
    for k,v in paramdict.items():
        print(f'{k}: {v}')
        
    return paramdict


def main():
    # run as independent function, called from CLI
    print(f'\nSDePER (Spatial Deconvolution method with Platform Effect Removal) v{cur_version}\n')
    
    start_time = time()

    paramdict = parseOpt()
    
    print('\n\n######### Start imputation #########')
    
    for x in paramdict['impute_diameter']:
        print(f'\n\nimputation for {x} µm ...')
        impute_start = time()
        # we now totally discard the transforming from integer coordinates to pixels
        # we use stepsize = impute_diameter / diameter inside imputation function
        result = do_imputation(paramdict['loc_file'], paramdict['prop_file'], paramdict['spatial_file'], float(x)/paramdict['diameter'], paramdict['hole_min_spots'], paramdict['preserve_shape'], diagnosis=paramdict['diagnosis'])
        # return imputed spot locations, cell type proportions and gene expressions
        result[0].to_csv(os.path.join(output_path, f'impute_diameter_{x}_spot_loc.csv'))
        result[1].to_csv(os.path.join(output_path, f'impute_diameter_{x}_spot_celltype_prop.csv'))
        result[2].to_csv(os.path.join(output_path, f'impute_diameter_{x}_spot_gene_norm_exp.csv.gz'), compression='gzip')
        print(f'imputation for {x} µm finished. Elapsed time: {(time()-impute_start)/60.0:.2f} minutes')
    
    print(f'\n\nwhole pipeline finished. Total elapsed time: {(time()-start_time)/60.0:.2f} minutes.')



if __name__ == '__main__':
    main()