#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 22:41:42 2023

@author: hill103

'setup.py' (this file) together with 'src' folder, 'MANIFEST.in', 'pyproject.toml', 'requirements.txt', 'README.md' and 'LICENSE' are required for publishing SDePER to PyPI

CAUTION: Avoid import custom modules from src folder, although it works in publishing PyPI package, but it won't work for bioconda package building because of different building environments
"""



import setuptools



# requirements.txt must be included in top-level manually, otherwise installing from source code tar.gz will fail
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f.readlines()]


# README.md will be automatically included
with open("README.md", "r") as f:
    long_description = f.read()


# Read version string
with open("src/version.py", "r") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("__version__"):
            # Extract the version string
            cur_version = line.split("=")[1].strip().strip('"').strip("'")
            break


setuptools.setup(
    name = "sdeper",    # short and all lower case
    version = cur_version,
    author = "Ningshan Li",
    author_email = "hill103.2@gmail.com",
    description = "Spatial Deconvolution method with Platform Effect Removal",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    python_requires = ">=3.9, <3.11",    # Specifies Python version support, PyPI (PEP 440) treats 3.9 as 3.9.0, 3.11 as 3.11.0, different with conda
    install_requires = requirements,    # Dependencies
    license_files = "LICENSE",    # license file will be include in top-level automatically (failed, specify it in MANIFEST.in)
    package_dir = {"": "src"},    # py files are in src folder
    # no need to specify 'packages=' since we only have one 'package' corresonding to the src folder
    # also no need to specify 'py_modules=', all py files under src folder will be recognized as modules
    # we need to include two non-python files
    # one way is include_package_data=True + MANIFEST.in, which can make sure the file is in top-level
    # the other way is using package_data WITHOUT include_package_data, which put file under src/eff-info
    include_package_data = True,
    #package_data = {
    #    '': ["requirements.txt"]
    #    },
    entry_points = {    # create wrappers for globally accessible function in Python scripts; only function are supported
        "console_scripts": [
            "runDeconvolution = cvaeglrm:main",
            "runImputation = imputation:main"
        ]
    },
    classifiers = [
        # Get strings from https://pypi.org/classifiers/
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux"],
    url = "https://az7jh2.github.io/SDePER/",    # homepage
    project_urls={
        # additional relevant URLs
        'Documentation': 'https://sdeper.readthedocs.io/en/latest/',
        'Source': 'https://github.com/az7jh2/SDePER',
        'Changelog': 'https://sdeper.readthedocs.io/en/latest/changelog.html',
        }
)