Changelog
=========

Version 1.6.2 (2024-08-05)
--------------------------

**Bug Fixes**:

* Upgrade certain dependencies to ensure successful installation via both PyPI and Conda.


Version 1.6.1 (2024-08-04)
--------------------------

**Bug Fixes**:

* Changed the dependency to the headless version of OpenCV to avoid installation issues and since no GUI functionality is required.


Version 1.6.0 (2024-08-03)
--------------------------

**Updates**:

* The :mod:`imputation` module has been completely rewritten, significantly improving contour finding and grid generation for enhanced performance and accuracy (`#1 <https://github.com/az7jh2/SDePER/issues/1>`_).


Version 1.5.0 (2024-07-12)
--------------------------

**Updates**:

* The optimization of cell type proportion :math:`\theta` is skipped if the initial value of :math:`\theta` indicates the presence of only one cell type in the spot.

* When predicting cell type proportions utilizing the CVAE latent space, the values of the CVAE latent space are now directly used instead of PCA embeddings for proportion transferring.

* The default number of hidden layers has been changed from 2 to 1.


Version 1.4.0 (2024-06-26)
--------------------------

**Updates**:

* Added single cell augmentation feature for scRNA-seq reference data.

* Implemented failure steps for handling irregular :math:`\theta` values after optimization.

* Added support for transferring cell type proportions based on PCA or UMAP embeddings of latent space, or directly on the original latent space in cell type proportion prediction using CVAE.

* Introduced caching in GLRM to accelerate computations by storing calculated likelihood values, reducing duplicate calculations. Note: This feature is disabled by default due to potential optimization failures caused by unknown reasons (`#5 <https://github.com/az7jh2/SDePER/issues/5>`_).

**Bug Fixes**:

* Resolved issue with spot name inconsistencies when spots are filtered out if cell type proportions predicted by CVAE were used for :math:`\theta` initialization in GLRM modeling.

* Fixed bug causing errors when plotting CVAE loss during training in the absence of validation data.


Version 1.3.1 (2024-06-06)
--------------------------

**Updates**:

* Added a step to remove mitochondrial genes during preprocessing.

* Introduced a PCA plot for visualizing the CVAE latent space and added density estimation based on PCA in diagnostic figures.


Version 1.3.0 (2024-05-09)
--------------------------

**Updates**:

* Introduced prediction of cell type proportions utilizing the CVAE latent space. Currently, the proportions are transferred from the scRNA-seq condition to the spatial condition in latent space. Then the predicted cell type proportions are used as initial value of :math:`\theta` for GLRM modeling (`#13 <https://github.com/az7jh2/SDePER/issues/13>`_).

* Reused :math:`\theta` and :math:`e^{\alpha}` estimations from stage 1 of GLRM modeling for initializing stage 2 (`#12 <https://github.com/az7jh2/SDePER/issues/12>`_).

* Increased the weight of spatial spots and scRNA-seq cells in CVAE training against generated pseudo-spots.

* Added support for retaining only highly variable genes in the spatial data. By default all genes are retained.

* SDePER options are written to a text file within the diagnosis folder, and only DE genes are retained in the CVAE-transformed data during saving if command-line option :option:`--redo_de` is ``true``.

* Decreased the default number of command-line option :option:`--n_pseudo_spot` to ``100,000``.


**Bug Fixes**:

* Resolved a bug where errors occurred during diagnostic UMAP drawing if only cell type markers were provided, and no scRNA-seq cells were available.


Version 1.2.1 (2024-05-03)
--------------------------

**Updates**:

* Implemented a GitHub Action triggered by new release publications to test the package installation (`#10 <https://github.com/az7jh2/SDePER/issues/10>`_).

* Added a diagnostic UMAP plot of raw data before platform effect removal using CVAE. Also included new diagnostic plots depicting CVAE training loss.

* Changed the default value of the :option:`--n_marker_per_cmp` command-line option to ``20``.

* Added three command-line options: :option:`--use_batch_norm`, :option:`--use_spatial_pseudo` and :option:`--cvae_train_epoch`.


Version 1.2.0 (2024-04-28)
--------------------------

**Updates**:

* Revised the :mod:`cvae` module, implementing several updates including (`#4 <https://github.com/az7jh2/SDePER/issues/4>`_):

   * Integration of Batch Normalization into the CVAE training process.
   * Inclusion of a logarithmic transformation in the preprocessing of gene expression data for CVAE input.
   * Generation of "pseudo-spots" under spatial conditions through the random combination of spatial spots.
   * Addition of two command-line options: :option:`--n_pseudo_spot` and :option:`--num_hidden_layer`. Also adjusted the default value of :option:`--cvae_init_lr`.

* Relocated all code related to generating diagnostic figures to a new module, :mod:`diagnosis_plots`. Additionally organized the output figures into a folder named `diagnosis` within the output path (`#6 <https://github.com/az7jh2/SDePER/issues/6>`_).


Version 1.1.0 (2024-04-20)
--------------------------

**Updates**:

* Improved differential analysis strategy for maker gene identification. Added 8 new related command-line options and modified the default value of 2 options (`#3 <https://github.com/az7jh2/SDePER/issues/3>`_).

* Updated help messages (`#7 <https://github.com/az7jh2/SDePER/issues/7>`_).

* Add support for installation via Conda (`#2 <https://github.com/az7jh2/SDePER/issues/2>`_, `#8 <https://github.com/az7jh2/SDePER/issues/8>`_).

* Add source code and relevant documentation into the package documentation (`#9 <https://github.com/az7jh2/SDePER/issues/9>`_).



Version 1.0.3 (2024-04-01)
--------------------------

**Bug Fixes**:

* Resolved the version determination bug in release v1.0.2 (`#8 <https://github.com/az7jh2/SDePER/issues/8>`_).

**Updates**:

* Automatically publishing new releases to PyPI using GitHub Actions.



Version 1.0.2 (2024-03-31)
--------------------------

**Updates**:

* Updated the version control to ensure compatibility with Bioconda installation (`#8 <https://github.com/az7jh2/SDePER/issues/8>`_).



Version 1.0.1 (2023-05-01)
--------------------------

**Bug Fixes**:

* Fixed a bug in imputation caused by a typo, which led to accessing an index outside the list size.



Version 1.0.0 (2023-03-20)
--------------------------

The first release of SDePER.