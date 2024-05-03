.. SDePER documentation master file, created by
   sphinx-quickstart on Fri Feb 17 03:14:22 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SDePER's documentation!
=====================================

.. image:: https://img.shields.io/badge/os-linux-blue
   :alt: OS
.. image:: https://img.shields.io/pypi/pyversions/sdeper
   :target: https://www.python.org/
   :alt: Python version
.. image:: https://img.shields.io/github/v/release/az7jh2/SDePER
   :target: https://github.com/az7jh2/SDePER
   :alt: GitHub release (latest by date)
.. image:: https://img.shields.io/pypi/v/sdeper
   :target: https://pypi.org/project/sdeper/
   :alt: PyPI version
.. image:: https://img.shields.io/conda/vn/bioconda/sdeper
   :target: https://anaconda.org/bioconda/sdeper
   :alt: Conda Version
.. image:: https://img.shields.io/docker/v/az7jh2/sdeper?label=docker
   :target: https://hub.docker.com/repository/docker/az7jh2/sdeper/general
   :alt: Docker Image Version (latest by date))
.. image:: https://img.shields.io/readthedocs/sdeper/latest
   :target: https://sdeper.readthedocs.io/en/latest/
   :alt: Read the Docs (version)


SDePER (\ **S**\ patial **De**\ convolution method with **P**\ latform **E**\ ffect **R**\ emoval) is a **hybrid** machine learning and regression method to deconvolve Spatial barcoding-based transcriptomic data using reference single-cell RNA sequencing data, considering **platform effects removal**, **sparsity** of cell types per capture spot and across-spots **spatial correlation** in cell type compositions. SDePER is also able to **impute** cell type compositions and gene expression at unmeasured locations in a tissue map with **enhanced resolution**..


Quick Start
-----------

SDePER currently supports only Linux operating systems such as Ubuntu, and is compatible with Python versions 3.9.12 up to but not including 3.11.

SDePER can be installed using conda

   .. code-block:: bash

      conda create -n sdeper-env -c bioconda -c conda-forge python=3.9.12 sdeper

or pip

   .. code-block:: bash

      conda create -n sdeper-env python=3.9.12
      conda activate sdeper-env
      pip install sdeper

SDePER requires **4 input files** for cell type deconvolution:

   1. raw nUMI counts of **spatial transcriptomics data** (spots × genes): ``spatial.csv``
   2. raw nUMI counts of **reference scRNA-seq data** (cells × genes): ``scrna_ref.csv``
   3. **cell type annotations** for all cells in scRNA-seq data (cells × 1): ``scrna_anno.csv``
   4. **adjacency matrix** of spots in spatial transcriptomics data (spots × spots): ``adjacency.csv``

To start cell type deconvolution, run:

   .. code-block:: bash

      runDeconvolution -q spatial.csv -r scrna_ref.csv -c scrna_anno.csv -a adjacency.csv


Check out :doc:`installation` page for detailed installation instructions, and :doc:`usage` page for commands for cell type deconvolution and imputation. The detailed descriptions of all options in commands are in :doc:`cli_options` page, and a guidance on setting the options is in :doc:`best_practice` page.


.. toctree::
   :maxdepth: 4
   :caption: Table of Contents
   :hidden:

   installation
   usage
   cli_options
   best_practice
   changelog


.. toctree::
   :maxdepth: 2
   :caption: Python API
   :hidden:

   modules