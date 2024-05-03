Changelog
=========

Version 1.2.1 (2024-05-03)
--------------------------

**Updates**:

* Implemented a GitHub Action triggered by new release publications to test the package installation (`#10 <https://github.com/az7jh2/SDePER/issues/10>`_).

* Added a diagnostic UMAP plot of raw data before platform effect removal using CVAE. Also included new diagnostic plots depicting CVAE training loss.

* Changed the default value of the :option:`--n_marker_per_cmp` command-line option to 20.

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

* Improved differential analysis strategy for maker gene identification. Added 8 new related options and modified the default value of 2 options (`#3 <https://github.com/az7jh2/SDePER/issues/3>`_).

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