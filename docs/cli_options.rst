CLI Options
===========

.. _deconvolution_opt:

runDeconvolution
----------------

.. option:: -h, --help

   Print help messages.

.. option:: -v, --version

   Print version of SDePER.


.. _deconvolution_opt_input:

Input-related options
~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~

.. note::

   We do not provide options for renaming output files. All output files are in the same folder as input files.

   The cell type deconvolution result file is named as ``celltype_proportions.csv``.

   If imputation is enabled, for each specified spot diameter ``d`` µm, there will be three more output files:

      1. imputed spot locations ``impute_diameter_d_spot_loc.csv``,
      2. imputed spot cell type proportions ``impute_diameter_d_spot_celltype_prop.csv``,
      3. imputed spot gene expressions (already normalized by sequencing depth of spots) ``impute_diameter_d_spot_gene_norm_exp.csv``.


.. _deconvolution_opt_general:

General options
~~~~~~~~~~~~~~~

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

   If true, provide more output files related to CVAE building and hyper-parameter selection for diagnosis.

   :Type: boolean

   :Default: ``false``


.. option:: --verbose

   Control whether to print more info such as output of each ADMM iteration step during program running (true/false).

   :Type: boolean

   :Default: ``true``


.. _deconvolution_opt_de:

Cell type marker identification options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. option:: --n_marker_per_cmp

   Number of selected TOP marker genes for each comparison of ONE cell type against another ONE cell type using Wilcoxon Rank Sum Test.

   For each comparison, genes with a FDR adjusted p value < 0.05 will be selected first, then these marker genes will be sorted by a combined rank of log fold change and pct.1/pct.2, and finally pick up specified number of gene with TOP ranks.

   :Type: integer

   :Default: ``30``


.. _deconvolution_opt_cvae:

CVAE-related options
~~~~~~~~~~~~~~~~~~~~

.. note::

   To successfully train a neural network model is non-trivial. We already **pre-fix some hyper-parameters** related to CVAE model Topology and optimizer based on our experiences on analysis of various spatial transcriptomics datasets. The options can be tuned by users are listed as below. We also provide guidance for setting each option right after the description of that option, and a summary of setting CVAE-related options in :ref:`Set CVAE-related options <use_cvae>` page.


.. option:: --n_hv_gene

   Number of highly variable genes identified in reference scRNA-seq data, and these HV genes will be used together with identified cell type marker genes for building CVAE.

   If the actual number of genes in scRNA-seq data is less than the specified value, all available genes in scRNA-seq data will be used for building CVAE.

   :Type: integer

   :Default: ``1000``

   .. note::

      Highly variable genes are used for building CVAE only, and cell type-specific marker genes will also be used for building CVAE. It's recoomended to set the number of highly variable genes to be close to the number of identified marker genes.


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

   :Default: ``10000``


.. option:: --cvae_input_scaler

   Maximum value of the scaled input for CVAE input layer, i.e. linearly scale all the sequencing depth normalized gene expressions to range [0, `cvae_input_scaler`].

   :Type: integer

   :Default: ``10``

   .. danger::

      It is strongly recommended not to change the value of this option and use the default value 10.


.. option:: --cvae_init_lr

   Initial learning rate for training CVAE.

   :Type: float

   :Default: ``0.003``

   .. note::

      Although learning rate is set to decrease automatically based on the loss function value on validation data during training, large initial learning rate will cause training failure at the very beginning of training. **If loss function value NOT monotonically decrease, please try smaller initial learning rate**.


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

GLRM hyper-parameter-related options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. option:: --lambda_r

   Hyper-parameter for adaptive Lasso.

   When the value of this option is not specified, cross-validation will be used to find the optimal value. The list of `lambda_r` candidates will has total `lambda_r_range_k` values, and candidate values will be evenly selected on a log scale (geometric progression) from range [`lambda_r_range_min`, `lambda_r_range_max`].

   If :option:`--lambda_r` is specified as a valid value, then :option:`--lambda_r_range_k`, :option:`--lambda_r_range_min` and :option:`--lambda_r_range_max` will be ignored.

   :Type: float

   :Default: ``None``


.. option:: --lambda_r_range_min

   Minimum value of the range of `lambda_r` candidates used for hyper-parameter selection.

   :Type: float

   :Default: ``0.1``


.. option:: --lambda_r_range_max

   Maximum value of the range of `lambda_r` candidates used for hyper-parameter selection.

   :Type: float

   :Default: ``100``


.. option:: --lambda_r_range_k

   Number of `lambda_r` candidates used for hyper-parameter selection. When generating candidate list, both endpoints `lambda_r_range_min` and `lambda_r_range_max` are included.

   :Type: integer

   :Default: ``8``


.. option:: --lambda_g

   Hyper-parameter for graph Laplacian constrain, which depends on the edge weights used in the graph created from the Adjacency Matrix.

   When the value of this option is not specified, cross-validation will be used to find the optimal value. The list of `lambda_g` candidates will has total `lambda_g_range_k` values, and candidate values will be evenly selected on a log scale (geometric progression) from range [`lambda_g_range_min`, `lambda_g_range_max`].

   If :option:`--lambda_g` is specified as a valid value, then :option:`--lambda_g_range_k`, :option:`--lambda_g_range_min` and :option:`--lambda_g_range_max` will be ignored.

   :Type: float

   :Default: ``None``


.. option:: --lambda_g_range_min

   Minimum value of the range of `lambda_g` candidates used for hyper-parameter selection.

   :Type: float

   :Default: ``0.1``


.. option:: --lambda_g_range_max

   Maximum value of the range of `lambda_g` candidates used for hyper-parameter selection.

   :Type: float

   :Default: ``100``


.. option:: --lambda_g_range_k

   Number of `lambda_g` candidates used for hyper-parameter selection. When generating candidate list, both endpoints `lambda_g_range_min` and `lambda_g_range_max` are included.

   :Type: integer

   :Default: ``8``


.. _deconvolution_opt_imputation:

Imputation-related options
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. option:: --diameter

   the physical diameter (µm) of spatial spots.

   :Type: integer

   :Default: ``200``


.. option:: --impute_diameter

   the target spot diameter (µm) during imputation.

   :Type: one integer or a string containing an array of numbers separated by ","

   :Default: ``160,114,80``, corresponding to the low, medium, high resolution




.. _imputation_opt:

runImputation
-------------

.. option:: -h, --help

   Print help messages.

.. option:: -v, --version

   Print version of SDePER.


.. _imputation_opt_input:

Input-related options
~~~~~~~~~~~~~~~~~~~~~

.. option:: -q, --query

   Input CSV file of raw nUMI counts of spatial transcriptomics data (spots * genes), with absolute or relative path.

   Rows as spots and columns as genes.

   Row header as spot barcodes and column header as gene symbols are both required.

   :Type: string

   :Default: ``None``


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


.. option:: -p, --prop

   Input csv file of cell type proportions of spots in spatial transcriptomics data (spots * cell types), with absolute or relative path.

   It can be the result from cell type deconvolution by SDePER, or directly provided by user.

   Rows as spots and columns as cell types.

   Row header as spot barcodes and column header as cell type names are required.

   :Type: string

   :Default: ``None``

   .. note::

      The spot order should be consist with row order in spatial nUMI count data.


.. _imputation_opt_output:

Output-related options
~~~~~~~~~~~~~~~~~~~~~~

.. note::

   We do not provide options for renaming output files. All output files are in the same folder as input files.

   For each specified spot diameter ``d`` µm, there will be three output files:

      1. imputed spot locations ``impute_diameter_d_spot_loc.csv``,
      2. imputed spot cell type proportions ``impute_diameter_d_spot_celltype_prop.csv``,
      3. imputed spot gene expressions (already normalized by sequencing depth of spots) ``impute_diameter_d_spot_gene_norm_exp.csv``.


.. _imputation_opt_imputation:

Imputation-related options
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. option:: --diameter

   the physical diameter (µm) of spatial spots.

   :Type: integer

   :Default: ``200``


.. option:: --impute_diameter

   the target spot diameter (µm) during imputation.

   :Type: one integer or a string containing an array of numbers separated by ","

   :Default: ``160,114,80``, corresponding to the low, medium, high resolution