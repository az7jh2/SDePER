.. _deconvolution_opt:

runDeconvolution
================

.. option:: -h, --help

   Print help messages.

.. option:: -v, --version

   Print version of SDePER.


.. _deconvolution_opt_input:

Input-related options
---------------------

.. option:: -q, --query

   Input CSV file of raw nUMI counts of spatial transcriptomics data (spots * genes), with absolute or relative path.

   Rows as spots and columns as genes.

   Row header as spot barcodes and column header as gene symbols are both required.

   :Type: string

   :Default: ``None``


.. option:: -r, --ref

   Input CSV file of raw nUMI counts of scRNA-seq data (cells * genes), with absolute or relative path.

   Rows as cells and columns as genes.

   Row header as cell barcodes and column header as gene symbols are both required.

   :Type: string

   :Default: ``None``


.. option:: -c, --ref_anno

   Input CSV file of cell type annotations for all cells in scRNA-seq data, with absolute or relative path.

   Rows as cells and only 1 column as cell type annotation.

   Row header as cell barcodes and column header as arbitrary name are both required.

   :Type: string

   :Default: ``None``


.. option:: -m, --marker

   Input CSV file of already curated cell type marker gene expression (cell types * genes; already normalized by sequencing depth), with absolute or relative path.

   Rows as cell types and columns as genes.

   Row header as cell type names and column header as gene symbols are both required.

   If marker gene expression is provided, the built-in Differential analysis will be skipped. If not provided, Wilcoxon rank sum test will be performed to select cell type-specific marker genes.

   :Type: string

   :Default: ``None``

   .. tip::

      Check out the :ref:`Specify marker genes <use_marker>` page for how to provide manually selected marker genes and suppress the Differential analysis using this option.


.. option:: -l, --loc

   Input CSV file of row/column integer index (x,y) of spatial spots (spots * 2), with absolute or relative path.

   Rows as spots and columns are coordinates x (column index) and y (row index).

   Row header as spot barcodes and column header "x","y" are both required.

   :Type: string

   :Default: ``None``

   .. note::

      1. This spot location file is required for imputation.
      2. The column header must be "x" and "y" (lower case).
      3. x and y are integer index (1,2,3,...) not pixels.
      4. The spot order should be consist with row order in spatial nUMI count data.


.. option:: -a, --adjacency

   Input CSV file of Adjacency Matrix of spots in spatial transcriptomics data (spots * spots), with absolute or relative path.

   In Adjacency Matrix, entry value 1 represents corresponding two spots are adjacent spots according to the definition of neighborhood, while value 0 for non-adjacent spots. All diagonal entries are set as 0.

   Row header and column header as spot barcodes are both required.

   :Type: string

   :Default: ``None``

   .. note::

      1. The spot order should be consistent with row order in spatial nUMI count data.
      2. When Adjacency Matrix is not provided, graph Laplacian regularization will be disabled in fitting graph Laplacian regularized model (GLRM).


.. _deconvolution_opt_output:

Output-related options
----------------------

.. note::

   We do not provide options for renaming output files. For Docker/Singularity implementations, the output folder is the mounted folder where the input files are located. For pip/conda installations, the output folder is the current working directory where SDePER is executed.

   The cell type deconvolution result file is named as ``celltype_proportions.csv``.

   If imputation is enabled, for each specified spot diameter ``d`` µm, there will be three more output files:

      1. imputed spot locations ``impute_diameter_d_spot_loc.csv``,
      2. imputed spot cell type proportions ``impute_diameter_d_spot_celltype_prop.csv``,
      3. imputed spot gene expressions (already normalized by sequencing depth of spots) ``impute_diameter_d_spot_gene_norm_exp.csv.gz``.


.. _deconvolution_opt_general:

General options
---------------

.. option:: -n, --n_cores

   Number of CPU cores used for parallel computing.

   :Type: integer

   :Default: ``1``, i.e. no parallel computing


.. option:: --threshold

   Threshold for hard thresholding the estimated cell type proportions, i.e. for one spot, estimated cell type proportions smaller than this threshold value will be set to 0, then re-normalize all proportions of this spot to sum as 1.

   :Type: float

   :Default: ``0``, which means no hard thresholding


.. option:: --use_cvae

   Control whether to build Conditional Variational Autoencoder (CVAE) to remove the platform effect between spatial transcriptomics and reference scRNA-seq data (true/false).

   Building CVAE requires raw nUMI counts and corresponding cell type annotation of scRNA-seq data specified.

   :Type: boolean

   :Default: ``true``

   .. tip::

      It is recommended to enable CVAE when there is an anticipated presence of platform effect between the spatial transcriptomics and reference scRNA-seq data.


.. option:: --use_imputation

   Control whether to perform imputation (true/false).

   Imputation requires the spot diameter (µm) at higher resolution to be specified.

   :Type: boolean

   :Default: ``false``


.. option:: --diagnosis

   If true, provide more output files related to CVAE building and hyperparameter selection for diagnosis.

   :Type: boolean

   :Default: ``false``


.. option:: --verbose

   Control whether to print more info such as output of each ADMM iteration step during program running (true/false).

   :Type: boolean

   :Default: ``true``


.. _deconvolution_opt_de:

Cell type marker identification options
---------------------------------------

.. versionchanged:: 1.1.0

   Cell-type specific markers are identified by Differential analysis (DE) across cell-types in reference scRNA-seq data. We also perform cell and/or gene filtering before DE. Each time we ONLY compare the normalized gene expression (raw nUMI counts divided by sequencing depth) one cell-type (1st) vs another one cell-type (2nd) using **Wilcoxon Rank Sum Test**, then take the UNION of all identified markers for downstream analysis.

   Before version 1.1.0, for each comparison genes with a FDR adjusted p value < 0.05 will be selected first, then these marker genes will be sorted by a combined rank of log fold change and pct.1/pct.2, and finally pick up specified number of genes with TOP ranks.

   In version 1.1.0, the ranking strategy has been revised. Now we filter the marker genes with pre-set thresholds of p value (or FDR), fold change, pct.1 (percentage of cells expressed this marker in 1st cell-type) and pct.2 (percentage of cells expressed this marker in 2nd cell-type). Next we sort the marker genes by p value (or FDR) or fold change, and select the TOP ones.


.. option:: --n_marker_per_cmp

   Number of selected TOP marker genes for each comparison of ONE cell-type against another ONE cell-type using Wilcoxon Rank Sum Test. For each comparison, genes passing filtering will be selected first, then these marker genes will be sorted by fold change or p value (or FDR), and finally pick up specified number of genes with TOP ranks. If the number of available genes is less than the specified number, a WARNING will be shown in the program running log file.

   :Type: integer

   :Default: ``20``

   .. versionchanged:: 1.2.1

      Default value changed from 30 to 20.


.. option:: --use_fdr

   Whether to use FDR adjusted p value for filtering and sorting. If true use FDR adjusted p value; if false orginal p value will be used instead.

   :Type: boolean

   :Default: ``true``

   .. versionadded:: 1.1.0


.. option:: --p_val_cutoff

   Threshold of p value (or FDR if :option:`--use_fdr` is true) in marker genes filtering. Only genes with p value (or FDR if :option:`--use_fdr` is true) <= 0.05 will be kept.

   :Type: float

   :Default: ``0.05``

   .. versionadded:: 1.1.0


.. option:: --fc_cutoff

   Threshold of fold change (without log transform!) in marker genes filtering. By default only genes with fold change >= 1.2 will be kept.

   :Type: float

   :Default: ``1.2``

   .. versionadded:: 1.1.0


.. option:: --pct1_cutoff

   Threshold of pct.1 (percentage of cells expressed this marker in 1st cell-type) in marker genes filtering. By default only genes with pct.1 >= 0.3 will be kept.

   :Type: float

   :Default: ``0.3``

   .. versionadded:: 1.1.0


.. option:: --pct2_cutoff

   Threshold of pct.2 (percentage of cells expressed this marker in 2nd cell-type) in marker genes filtering. By default only genes with pct.2 <= 0.1 will be kept.

   :Type: float

   :Default: ``0.1``

   .. versionadded:: 1.1.0


.. option:: --sortby_fc

   Whether to sort marker genes by fold change. If true sort marker genes by fold change then select TOP ones. If false, p value (or FDR if :option:`--use_fdr` is true) will be used to sort marker genes instead.

   :Type: boolean

   :Default: ``true``

   .. versionadded:: 1.1.0


.. option:: --filter_cell

   Whether to filter cells with <200 genes for reference scRNA-seq data before differential analysis. NOTE we only apply cell filtering on reference data.

   :Type: boolean

   :Default: ``true``

   .. versionadded:: 1.1.0


.. option:: --filter_gene

   Whether to filter genes presented in <10 cells for reference scRNA-seq data and <3 spots for spatial data before differential analysis.

   :Type: boolean

   :Default: ``true``

   .. versionadded:: 1.1.0


.. _deconvolution_opt_cvae:

CVAE-related options
--------------------

.. note::

   We build Conditional Variational Autoencoder (CVAE) to adjust the platform effect between spatial transcriptomic and scRNA-seq data. To successfully train a neural network model is non-trivial. We already **pre-fix some hyperparameters** related to CVAE model Topology and optimizer based on our experiences on analysis of various spatial transcriptomics datasets. The options can be tuned by users are listed as below. We also provide guidance for setting each option right after the description of that option, and a summary of setting CVAE-related options in :ref:`Set CVAE-related options <use_cvae>` page.


.. option:: --n_hv_gene

   Number of highly variable genes identified in reference scRNA-seq data, and these HV genes will be used together with identified cell type marker genes for building CVAE.

   If the actual number of genes in scRNA-seq data is less than the specified value, all available genes in scRNA-seq data will be used for building CVAE.

   :Type: integer

   :Default: ``200``

   .. versionchanged:: 1.1.0

      Default value decreased from 1,000 to 200.

   .. note::

      Highly variable genes are used for building CVAE only, and cell type-specific marker genes will also be used for building CVAE. It's recommended to set the number of highly variable genes to be close to the number of identified marker genes.


.. option:: --n_pseudo_spot

   Maximum number of pseudo-spots generated by randomly combining scRNA-seq cells into one pseudo-spot in CVAE training.

   :Type: integer

   :Default: ``100,000``

   .. versionadded:: 1.2.0

   .. versionchanged:: 1.3.0

      Default value decreased from 500,000 to 100,000.


.. option:: --pseudo_spot_min_cell

   Minimum value of cells in one pseudo-spot when combining cells into pseudo-spots.

   :Type: integer

   :Default: ``2``

   .. tip::

      It's recommended to first make a rough estimate of how many cells in one spot in the spatial transcriptomics dataset, then set this option based on the estimation to make sure the number of cells in pseudo-spots are close to the spatial spots.


.. option:: --pseudo_spot_max_cell

   Maximum value of cells in one pseudo-spot when combining cells into pseudo-spots.

   :Type: integer

   :Default: ``8``

   .. tip::

      It's recommended to first make a rough estimate of how many cells in one spot in the spatial transcriptomics dataset, then set this option based on the estimation to make sure the number of cells in pseudo-spots are close to the spatial spots.


.. option:: --seq_depth_scaler

   A scaler of scRNA-seq sequencing depth to transform CVAE decoded values (sequencing depth normalized gene expressions) back to raw nUMI counts.

   :Type: integer

   :Default: ``10,000``


.. option:: --cvae_input_scaler

   Maximum value of the scaled input for CVAE input layer, i.e. linearly scale all the sequencing depth normalized gene expressions to range [0, `cvae_input_scaler`].

   :Type: integer

   :Default: ``10``

   .. danger::

      It is strongly recommended not to change the value of this option and use the default value 10.


.. option:: --cvae_init_lr

   Initial learning rate for training CVAE.

   :Type: float

   :Default: ``0.01``

   .. versionchanged:: 1.2.0

      Default learning rate increased from 0.003 to 0.01 as Batch Normalization incoporated.

   .. note::

      Although learning rate is set to decrease automatically based on the loss function value on validation data during training, large initial learning rate will cause training failure at the very beginning of training. **If loss function value NOT monotonically decrease, please try smaller initial learning rate**.


.. option:: --num_hidden_layer

   Number of hidden layers in encoder and decoder of CVAE. The number of neurons in each hidden layer will be determined automatically.

   :Type: integer

   :Default: ``1``

   .. versionadded:: 1.2.0

   .. versionchanged:: 1.5.0

      Default value changed from 2 to 1.


.. option:: --use_batch_norm

   Whether to use Batch Normalization in CVAE training.

   :Type: boolean

   :Default: ``true``

   .. versionadded:: 1.2.1


.. option:: --cvae_train_epoch

   Maximum number of training epochs for the CVAE.

   :Type: int

   :Default: ``500``

   .. versionadded:: 1.2.1


.. option:: --use_spatial_pseudo

   Whether to generate "pseudo-spots" in spatial condition by randomly combining existing spatial spots in CVAE training. When true, half of the total number specified by :option:`--n_pseudo_spot` will be created.

   :Type: boolean

   :Default: ``false``

   .. versionadded:: 1.2.1


.. option:: --redo_de

   Control whether to redo Differential analysis on CVAE transformed scRNA-seq gene expressions to get a new set of marker gene list of cell types (true/false).

   :Type: boolean

   :Default: ``true``

   .. tip::

      It is strongly recommended to redo Differential analysis since CVAE transformation may change the marker gene expression profile of cell types.


.. option:: --seed

   Seed value of TensorFlow to control the randomness in building CVAE.

   :Type: integer

   :Default: ``383``

   .. tip::

      Check out the :ref:`Reproducibility <reproducibility>` page for how to get reproducible results.


.. _deconvolution_opt_parameter:

GLRM hyperparameter-related options
-----------------------------------

.. note::

   We incorporate **adaptive Lasso penalty** and **graph Laplacian penalty** in GLRM, and use the hyperparameters `lambda_r` and `lambda_g` to balance the strength of those two penalties respectively.


.. option:: --lambda_r

   hyperparameter for adaptive Lasso.

   When the value of this option is not specified, cross-validation will be used to find the optimal value. The list of `lambda_r` candidates will has total `lambda_r_range_k` values, and candidate values will be evenly selected on a log scale (geometric progression) from range [`lambda_r_range_min`, `lambda_r_range_max`].

   If :option:`--lambda_r` is specified as a valid value, then :option:`--lambda_r_range_k`, :option:`--lambda_r_range_min` and :option:`--lambda_r_range_max` will be ignored.

   :Type: float

   :Default: ``None``


.. option:: --lambda_r_range_min

   Minimum value of the range of `lambda_r` candidates used for hyperparameter selection.

   :Type: float

   :Default: ``0.1``


.. option:: --lambda_r_range_max

   Maximum value of the range of `lambda_r` candidates used for hyperparameter selection.

   :Type: float

   :Default: ``100``


.. option:: --lambda_r_range_k

   Number of `lambda_r` candidates used for hyperparameter selection. When generating candidate list, both endpoints `lambda_r_range_min` and `lambda_r_range_max` are included.

   :Type: integer

   :Default: ``8``


.. option:: --lambda_g

   hyperparameter for graph Laplacian constrain, which depends on the edge weights used in the graph created from the Adjacency Matrix.

   When the value of this option is not specified, cross-validation will be used to find the optimal value. The list of `lambda_g` candidates will has total `lambda_g_range_k` values, and candidate values will be evenly selected on a log scale (geometric progression) from range [`lambda_g_range_min`, `lambda_g_range_max`].

   If :option:`--lambda_g` is specified as a valid value, then :option:`--lambda_g_range_k`, :option:`--lambda_g_range_min` and :option:`--lambda_g_range_max` will be ignored.

   :Type: float

   :Default: ``None``


.. option:: --lambda_g_range_min

   Minimum value of the range of `lambda_g` candidates used for hyperparameter selection.

   :Type: float

   :Default: ``0.1``


.. option:: --lambda_g_range_max

   Maximum value of the range of `lambda_g` candidates used for hyperparameter selection.

   :Type: float

   :Default: ``100``


.. option:: --lambda_g_range_k

   Number of `lambda_g` candidates used for hyperparameter selection. When generating candidate list, both endpoints `lambda_g_range_min` and `lambda_g_range_max` are included.

   :Type: integer

   :Default: ``8``


.. _deconvolution_opt_imputation:

Imputation-related options
--------------------------

.. option:: --diameter

   the physical distance (µm) between centers of two neighboring spatial spots. For Spatial Transcriptomics v1.0 technique it's 200 µm. For 10x Genomics Visium technique it's 100 µm.

   :Type: integer

   :Default: ``200``


.. option:: --impute_diameter

   the target distance (µm) between centers of two neighboring spatial spots after imputation.

   :Type: one integer or a string containing an array of numbers separated by ","

   :Default: ``160,114,80``, corresponding to the low, medium, high resolution for Spatial Transcriptomics v1.0 technique


.. option:: --hole_min_spots

   the minimum number of uncaptured spots required to recognize a hole in the tissue map. Holes with a number of spots less than or equal to this threshold in it are treated as if no hole exists and imputation will be performed within the hole. Default value is ``1``, meaning single-spot holes are imputed.

   :Type: integer

   :Default: ``1``

   .. versionadded:: 1.6.0


.. option:: --preserve_shape

   whether to maintain the shape of the tissue map during imputation. If true, all border points are retained in imputation to preserve the tissue's original shape, although this may result in an irregular imputed grid.

   :Type: boolean

   :Default: ``false``

   .. versionadded:: 1.6.0