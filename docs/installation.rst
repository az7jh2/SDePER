Installation
============

SDePER can be installed using conda, pip or Docker/Singularity.


.. attention::

   It is recommended to **install SDePER using conda or pip** because of the observed incompatibility between `Numba <https://numba.pydata.org/>`_ and Docker/Singularity. When running SDePER in a Docker or Singularity container, a significant increase in running time was noticed, as Numba cannot fully utilize the CPU resources for parallel computing.


Install SDePER
--------------

.. tabs::

   .. tab:: conda

      .. code-block:: bash

         conda create -n sdeper-env -c bioconda -c conda-forge python=3.9.12 sdeper

   .. tab:: pip

      .. code-block:: bash

         conda create -n sdeper-env python=3.9.12
         conda activate sdeper-env
         pip install sdeper

   .. tab:: Docker

      .. code-block:: bash
         :substitutions:

         docker pull az7jh2/sdeper:|cur_version|

   .. tab:: Singularity

      .. code-block:: bash
         :substitutions:

         singularity build sdeper-|cur_version|.sif docker://az7jh2/sdeper:|cur_version|


Test installation
-----------------

It should print the version.

.. tabs::

   .. tab:: conda

      .. code-block:: bash

         runDeconvolution -v

   .. tab:: pip

      .. code-block:: bash

         runDeconvolution -v

   .. tab:: Docker

      .. code-block:: bash
         :substitutions:

         docker run -it --rm az7jh2/sdeper:|cur_version| runDeconvolution -v

   .. tab:: Singularity

      .. code-block:: bash
         :substitutions:

         singularity exec sdeper-|cur_version|.sif runDeconvolution -v


After installing SDePER using any one of these methods, you can start using the package. Please check out :doc:`usage` page for commands for cell type deconvolution and imputation. The detailed descriptions of all options in commands are in :doc:`cli_options` page, and a guidance on setting the options is in :doc:`best_practice` page.