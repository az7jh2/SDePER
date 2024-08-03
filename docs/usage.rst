Usage
=====

.. note::

   For tutorials on using SDePER, please refer to the `simulated <https://github.com/az7jh2/SDePER_Analysis/tree/main/Simulation>`_ and `real data analysis <https://github.com/az7jh2/SDePER_Analysis/tree/main/RealData>`_ examples in our GitHub repository `SDePER_Analysis <https://github.com/az7jh2/SDePER_Analysis>`_.


.. _deconvolution_usage:

Cell type deconvolution
-----------------------

SDePER requires **4 input files** for cell type deconvolution:

   1. raw nUMI counts of **spatial transcriptomics data** (spots × genes): ``spatial.csv``
   2. raw nUMI counts of **reference scRNA-seq data** (cells × genes): ``scrna_ref.csv``
   3. **cell type annotations** for all cells in scRNA-seq data (cells × 1): ``scrna_anno.csv``
   4. **adjacency matrix** of spots in spatial transcriptomics data (spots × spots): ``adjacency.csv``

To start cell type deconvolution with default settings, run:

.. tabs::

   .. tab:: conda / pip

      .. code-block:: bash

         runDeconvolution -q spatial.csv -r scrna_ref.csv -c scrna_anno.csv -a adjacency.csv

   .. tab:: Docker

      .. code-block:: bash
         :substitutions:

         docker run -it --rm -v <path>:/data az7jh2/sdeper:|cur_version| runDeconvolution -q spatial.csv -r scrna_ref.csv -c scrna_anno.csv -a adjacency.csv

   .. tab:: Singularity

      .. code-block:: bash
         :substitutions:

         singularity exec -B <path>:/data sdeper-|cur_version|.sif runDeconvolution -q spatial.csv -r scrna_ref.csv -c scrna_anno.csv -a adjacency.csv

``<path>`` is the valid and **absolute** path of the folder in the host machine where all input files locate, and this fold will be mounted in the Docker/Singularity image for data exchanging.

.. important::

   Please check out the :ref:`deconvolution input-related options <deconvolution_opt_input>` page for file format requirements of each input file, and :ref:`deconvolution output-related options <deconvolution_opt_output>` for descriptions of output files containing the result of cell type deconvotion.

   Examples of input and output files can also be found in our GitHub repository `SDePER_Analysis <https://github.com/az7jh2/SDePER_Analysis>`_.


.. _imputation_usage:

Imputation
----------

SDePER requires **3 input files** for imputation:

   1. raw nUMI counts of **spatial transcriptomics data** (spots × genes): ``spatial.csv``
   2. **row/column integer index** (*x*, *y*) of spots in spatial transcriptomics data (spots × 2): ``spatial_loc.csv``
   3. **cell type proportions** of spots in spatial transcriptomics data (spots × cell types): ``spatial_prop.csv``

To start imputation with default settings, run:

.. tabs::

   .. tab:: conda / pip

      .. code-block:: bash

         runImputation -q spatial.csv -l spatial_loc.csv -p spatial_prop.csv

   .. tab:: Docker

      .. code-block:: bash
         :substitutions:

         docker run -it --rm -v <path>:/data az7jh2/sdeper:|cur_version| runImputation -q spatial.csv -l spatial_loc.csv -p spatial_prop.csv

   .. tab:: Singularity

      .. code-block:: bash
         :substitutions:

         singularity exec -B <path>:/data sdeper-|cur_version|.sif runImputation -q spatial.csv -l spatial_loc.csv -p spatial_prop.csv

``<path>`` is the valid and **absolute** path of the folder in the host machine where all input files locate, and this fold will be mounted in the Docker/Singularity image for data exchanging.

.. important::

   Please check out the :ref:`imputation input-related options <imputation_opt_input>` page for file format requirements of each input file, and :ref:`imputation output-related options <imputation_opt_output>` for descriptions of output files containing the results of imputation.

   Examples of input and output files can also be found in our GitHub repository `SDePER_Analysis <https://github.com/az7jh2/SDePER_Analysis>`_.


.. tip::

   Imputation can also be run together with cell type deconvolution. To start cell type deconvolution followed by imputation with default settings, run:

   .. tabs::

      .. tab:: conda / pip

         .. code-block:: bash

            runDeconvolution -q spatial.csv -r scrna_ref.csv -c scrna_anno.csv -a adjacency.csv -l spatial_loc.csv --use_imputation true

      .. tab:: Docker

         .. code-block:: bash
            :substitutions:

            docker run -it --rm -v <path>:/data az7jh2/sdeper:|cur_version| runDeconvolution -q spatial.csv -r scrna_ref.csv -c scrna_anno.csv -a adjacency.csv -l spatial_loc.csv --use_imputation true

      .. tab:: Singularity

         .. code-block:: bash
            :substitutions:

            singularity exec -B <path>:/data sdeper-|cur_version|.sif runDeconvolution -q spatial.csv -r scrna_ref.csv -c scrna_anno.csv -a adjacency.csv -l spatial_loc.csv --use_imputation true

   ``<path>`` is the valid and **absolute** path of the folder in the host machine where all input files locate, and this fold will be mounted in the Docker/Singularity image for data exchanging.