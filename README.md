# SDePER
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sdeper)](https://www.python.org/) [![GitHub release (latest by date)](https://img.shields.io/github/v/release/az7jh2/SDePER)](https://github.com/az7jh2/SDePER) [![PyPI](https://img.shields.io/pypi/v/sdeper)](https://pypi.org/project/sdeper/) [![Docker Image Version (latest by date)](https://img.shields.io/docker/v/az7jh2/sdeper?label=docker)](https://hub.docker.com/r/az7jh2/sdeper) [![Read the Docs (version)](https://img.shields.io/readthedocs/sdeper/latest)](https://sdeper.readthedocs.io/en/latest/)

**SDePER** (**S**patial **De**convolution method with **P**latform **E**ffect **R**emoval) is a **hybrid** machine learning and regression method to deconvolve Spatial barcoding-based transcriptomic data using reference single-cell RNA sequencing data, considering **platform effects removal**, **sparsity** of cell types per capture spot and across-spots **spatial correlation** in cell type compositions. SDePER is also able to **impute** cell type compositions and gene expression at unmeasured locations in a tissue map with **enhanced resolution**.

## Quick Start

SDePER can be installed via `pip`

```bash
conda create -n sdeper-env python=3.9.12
conda activate sdeper-env
pip install sdeper
```

SDePER requires **4 input files** for cell type deconvolution:

1. raw nUMI counts of **spatial transcriptomics data** (spots × genes): `spatial.csv`
2. raw nUMI counts of **reference scRNA-seq data** (cells × genes): `scrna_ref.csv`
3. **cell type annotations** for all cells in scRNA-seq data (cells × 1): `scrna_anno.csv`
4. **adjacency matrix** of spots in spatial transcriptomics data (spots × spots): `adjacency.csv`

To start cell type deconvolution by running

```bash
runDeconvolution -q spatial.csv -r scrna_ref.csv -c scrna_anno.csv -a adjacency.csv
```

**Homepage**: [https://az7jh2.github.io/SDePER/](https://az7jh2.github.io/SDePER/).

**Full Documentation** for SDePER is available on [Read the Docs](https://sdeper.readthedocs.io/en/latest/).

**Example data and Analysis** using SDePER are available in our GitHub repository [SDePER_Analysis](https://github.com/az7jh2/SDePER_Analysis).
