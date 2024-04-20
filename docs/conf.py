# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

from config import cur_version


# -- Project information -----------------------------------------------------

project = 'SDePER'
copyright = '2023, Ningshan Li'
author = 'Ningshan Li'

# The full version, including alpha/beta/rc tags
release = cur_version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# sphinx-prompt (https://pypi.org/project/sphinx-prompt/): useful to show commands outputs, Python REPL scripts and more
# Sphinx Substitution Extensions (https://github.com/adamtheturtle/sphinx-substitution-extensions): allow substitutions within code blocks
# sphinx-tabs (https://github.com/executablebooks/sphinx-tabs): create tabbed content
# sphinx-toolbox changeset (https://sphinx-toolbox.readthedocs.io/en/stable/extensions/changeset.html): customised version directives
# sphinxcontrib.jquery (https://pypi.org/project/sphinxcontrib-jquery/): ensures that jQuery is always installed for use in Sphinx themes or extensions
# sphinx.ext.autodoc (https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html): include documentation from docstrings
# sphinx.ext.viewcode (https://www.sphinx-doc.org/en/master/usage/extensions/viewcode.html): add links to highlighted source code
# sphinx.ext.napoleon (https://sphinxcontrib-napoleon.readthedocs.io/en/latest/): a pre-processor that parses NumPy and Google style docstrings and converts them to reStructuredText before Sphinx attempts to parse them
extensions = ['sphinx-prompt', 'sphinx_substitution_extensions', 'sphinx_tabs.tabs', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'sphinx_toolbox.changeset', 'sphinxcontrib.jquery'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# By default, Read the Docs uses the sphinx_rtd_theme; alabaster is the default theme of Sphinx
html_theme = 'sphinx_rtd_theme'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# -- Options for add substitutions -------------------------------------------

# Substitute the version string as latest version
# It's a good practice to use tex_escape when defining substitutions to ensure compatibility with LaTeX output
# But substitutions do not work inside code blocks

from sphinx.util.texescape import escape as tex_escape

rst_prolog = f"""
.. |cur_version| replace:: {tex_escape(cur_version)}
"""