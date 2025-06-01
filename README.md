# SDePER
![OS](https://img.shields.io/badge/os-linux-blue) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sdeper)](https://www.python.org/) [![GitHub release (latest by date)](https://img.shields.io/github/v/release/az7jh2/SDePER)](https://github.com/az7jh2/SDePER) [![PyPI](https://img.shields.io/pypi/v/sdeper)](https://pypi.org/project/sdeper/)  [![Conda Version](https://img.shields.io/conda/vn/bioconda/sdeper)](https://anaconda.org/bioconda/sdeper) [![Docker Image Version (latest by date)](https://img.shields.io/docker/v/az7jh2/sdeper?label=docker)](https://hub.docker.com/r/az7jh2/sdeper) [![Read the Docs (version)](https://img.shields.io/readthedocs/sdeper/latest)](https://sdeper.readthedocs.io/en/latest/) [![DOI](https://zenodo.org/badge/585965825.svg)](https://zenodo.org/doi/10.5281/zenodo.8328020)

**SDePER** (**S**patial **De**convolution method with **P**latform **E**ffect **R**emoval) is a **hybrid** machine learning and regression method to deconvolve Spatial barcoding-based transcriptomic data using reference single-cell RNA sequencing data, considering **platform effects removal**, **sparsity** of cell types per capture spot and across-spots **spatial correlation** in cell type compositions. SDePER is also able to **impute** cell type compositions and gene expression at unmeasured locations in a tissue map with **enhanced resolution**.

## Quick Start

SDePER currently supports only Linux operating systems such as Ubuntu, and is compatible with Python 3.9.x and 3.10.x releases (3.11+ not yet supported).

SDePER can be installed via conda

```bash
conda create -n sdeper-env -c bioconda -c conda-forge python=3.9.12 sdeper
```

or pip

```bash
conda create -n sdeper-env python=3.9.12
conda activate sdeper-env
pip install sdeper
```

SDePER supports an **out-of-the-box** feature, meaning that users only need to provide the required **four input files** for cell type deconvolution. The package manages all aspects of file reading, preprocessing, cell type-specific marker gene identification, and more internally. The required files are:

1. raw nUMI counts of **spatial transcriptomics data** (spots × genes): `spatial.csv`
2. raw nUMI counts of **reference scRNA-seq data** (cells × genes): `scrna_ref.csv`
3. **cell type annotations** for all cells in scRNA-seq data (cells × 1): `scrna_anno.csv`
4. **adjacency matrix** of spots in spatial transcriptomics data (spots × spots; **optional**): `adjacency.csv`

To start cell type deconvolution using all default settings by running

```bash
runDeconvolution -q spatial.csv -r scrna_ref.csv -c scrna_anno.csv -a adjacency.csv
```

**Homepage**: [https://az7jh2.github.io/SDePER/](https://az7jh2.github.io/SDePER/).

**Full Documentation** for SDePER is available [here](https://sdeper.readthedocs.io/en/latest/).

**Example data and Analysis** using SDePER are summarized in [this page](https://sdeper.readthedocs.io/en/latest/vignettes1.html).

## Citation

If you use SDePER, please cite:

Yunqing Liu, Ningshan Li, Ji Qi *et al.* SDePER: a hybrid machine learning and regression method for cell-type deconvolution of spatial barcoding-based transcriptomic data. *Genome Biology* **25**, 271 (2024). https://doi.org/10.1186/s13059-024-03416-2
