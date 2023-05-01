Best Practice
=============

.. _use_cvae:

Set CVAE-related options
------------------------

The first step in SDePER is building a CVAE to remove the platform effect between spatial transcriptomics data and scRNA-seq data. To successfully train a neural network model is non-trivial. We provide a guidance for setting CVAE-related options as below:

* Set the number of highly variable genes (:option:`--n_hv_gene`) to be close to the number of identified cell type-specific marker genes.
* Set the number of minimum and maximum of the number of cells in one pseudo-spot (:option:`--pseudo_spot_min_cell` and :option:`--pseudo_spot_max_cell`) so that the number of cells in pseudo-spots is close to the number of cells in real spatial spots. The number of cells in a spot is affected by size of the capture spot and size of cells.
* Set :option:`--redo_de` as ``true`` (default value) since CVAE transformation may change the marker gene expression profile of cell types.
* Monitor the loss function values in each CVAE training epoch. If loss function value NOT decrease monotonically, try smaller initial learning rate (:option:`--cvae_init_lr`).
* Left the values for :option:`--seq_depth_scaler` and :option:`--cvae_input_scaler` as default.
* You can also try different seed values (:option:`--seed`) then select the most reasonable result for downstream analysis.
* When set :option:`--diagnosis` as ``true``, UMAPs of latent space of CVAE will be generated for diagnosing whether CVAE has been trainied successfully.


.. _use_marker:

Specify marker genes
--------------------

In SDePER, we provide the :option:`-m` or :option:`--marker` option for user to specify manually selected cell type-specific marker genes instead of identifying marker genes by Differential analysis on reference scRNA-seq data. The required CSV file for this option is a matrix with rows as cell types and columns as selected marker genes.

The usage of user-specified marker genes differs depending on whether Conditional Variational Autoencoder (CVAE) is enabled to remove the platform effect or not.

**In case of enabling CVAE in SDePER**:

When CVAE is enabled in SDePER, the union of the top highly variable genes identified from the reference scRNA-seq data and user-specified cell type marker genes are used in CVAE building. So **only the gene symbols** of this input is used in this situation, and user can provide a matrix with **all 0s** for this options.

**In case of disabling CVAE in SDePER**:

When CVAE is disabled in SDePER, the whole cell type-specific gene expression porfile provided by this input is directly used for fitting graph Laplacian regularized model (GLRM). In this situation the input matrix is required to be the **average expression profiles across cells from the given cell type** which can be calculated from the reference scRNA-seq data. It is recommend to take average on library size normalized gene expression **without log transformation**.



.. _reproducibility:

Reproducibility
---------------

The only randomness in SDePER comes from training CVAE, which is essentially a neural network. Using the same value for **number of CPU cores** (:option:`-n` or :option:`--n_cores`) and **random seed** (:option:`--seed`) across different runnings **in the same hardware** can assure you get the same trained model, then get **exactly the same** cell type deconvolution and imputation results.

.. warning::

   Technologically there is **no way to assure reproducibility across different hardware**. This is because the **hardware/floating point limitation**. The floating point standard specifies only how close from the real value the result should be. But the hardware can return any value that is that close. So the residual accumulates during CVAE training process, ultimately resulting in different trained models.


.. _st_platform:

Spot size in ST techonology
---------------------------

The **diameter of spot** and **distance between centers of two neighboring spots** are different across different spatial transcriptomic (ST) techonologies. It is recommended to set these 3 SDePER options (:option:`--pseudo_spot_min_cell`, :option:`--pseudo_spot_max_cell` and :option:`--diameter`) based on the ST techonology which is used to generate the spatial data. We provide a summary of spot size in different ST techonologies as below:

**Spatial Transcriptomics v1.0**

.. image:: https://raw.githubusercontent.com/az7jh2/SDePER_Analysis/main/RealData/ST_v1.0.png
    :alt: ST v1.0

The diameter of each spot is **100 µm** and the spot center to center distance is **200 µm**. Copied from Figure S2 in `Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. 2016, Science <https://www.science.org/doi/10.1126/science.aaf2403>`_.


**10x Genomics Visium**

.. image:: https://raw.githubusercontent.com/az7jh2/SDePER_Analysis/main/RealData/10x_Visium.png
    :alt: 10x Visium

The diameter of each spot is **55 µm** and the spot center to center distance is **100 µm**. Copied from Figure 1 in `Inside Visium spatial capture technology <https://pages.10xgenomics.com/rs/446-PBO-704/images/10x_BR060_Inside_Visium_Spatial_Technology.pdf>`_.