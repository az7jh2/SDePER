# File Structure

This document provides a detailed overview of the file and directory organization within this repository, designed to facilitate easy navigation and comprehension for users and contributors.

## Source Code

- `src/`: Source code.
- `requirements.txt`: All dependencies required by the project.
- `LICENSE`: License file.

## Documentation for Hosting on Read the Docs

[Documentation page](https://sdeper.readthedocs.io/en/latest/).

- `docs/`: Documentation files.
  - `conf.py`: Sphinx configuration file to customize the documentation build process.
  - `requirements.txt`: Dependencies needed specifically for building the documentation. NOTE The **dependencies required by the package must also be installed** to display the source code in the documentation. To ensure successful documentation builds, we do not impose version restrictions on dependencies.
  - `_static/`: static asserts for that are used in the documentation, including images and CSS files.
- `.readthedocs.yaml`: Configuration file for Read the Docs, specifying build parameters and requirements to automate documentation builds.

## Docker Support

[Docker image page](https://hub.docker.com/r/az7jh2/sdeper).

- `Dockerfile`: Script including instructions to build a Docker image for the project.

## PyPI Distribution

[PyPI package page](https://pypi.org/project/sdeper/).

To publish a package on PyPI, ensure including the following files in addition to all relevant source code files:

- `MANIFEST.in`: Instructs setuptools on which additional non-code files should be included in the distribution.
- `pyproject.toml`: Configuration file for using setuptools, defining build and packaging settings.
- `setup.py`: Script for setting up the package for distribution.

## Bioconda Distribution

[Bioconda package page](https://anaconda.org/bioconda/sdeper).

- `meta.yaml`: Recipe file for publishing the software to Bioconda, specifying package name, version, dependencies, and other distribution details.

## GitHub Action workflow

All GitHub Action workflow files are in `.github/workflows/` folder.

- `publish-to-pypi.yml`: publishing package to PyPI trigged by a new release in GitHub.
- `test-installation.yml`: test SDePER when a new version is released.