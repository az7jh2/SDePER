.. _imputation_opt:

runImputation
=============

.. option:: -h, --help

   Print help messages.

.. option:: -v, --version

   Print version of SDePER.


.. _imputation_opt_input:

Input-related options
---------------------

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
----------------------

.. note::

   We do not provide options for renaming output files. All output files are in the same folder as input files.

   For each specified spot diameter ``d`` µm, there will be three output files:

      1. imputed spot locations ``impute_diameter_d_spot_loc.csv``,
      2. imputed spot cell type proportions ``impute_diameter_d_spot_celltype_prop.csv``,
      3. imputed spot gene expressions (already normalized by sequencing depth of spots) ``impute_diameter_d_spot_gene_norm_exp.csv.gz``.


.. _imputation_opt_general:

General options
---------------

.. option:: --diagnosis

   If true, a scatter plot displaying spatial spots and imputed spots is generated for diagnostic purposes.

   :Type: boolean

   :Default: ``false``


.. _imputation_opt_imputation:

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