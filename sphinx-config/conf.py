# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import numpy as np
import sys
sys.path.insert(0, os.path.abspath('../../prog_algs'))
sys.path.insert(0, os.path.abspath('../../prog_algs/src'))
sys.path.insert(0, os.path.abspath('../../prog_algs/examples'))

sys.path.insert(0, os.path.abspath('../../prog_models'))
sys.path.insert(0, os.path.abspath('../../prog_models/src'))
sys.path.insert(0, os.path.abspath('../../prog_models/examples'))

sys.path.insert(0, os.path.abspath('../../prog_server'))
sys.path.insert(0, os.path.abspath('../../prog_server/src'))
sys.path.insert(0, os.path.abspath('../../prog_server/examples'))

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'ProgPy Python Packages'
copyright = '2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.'
author = 'Chris Teubert, Matteo Corbetta, Chetan Kulkarni, Katelyn Jarvis, and Matthew Daigle'

# The full version, including alpha/beta/rc tags
release = "1.4.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', "sphinx.ext.githubpages", 'sphinx.ext.napoleon', 'sphinx.ext.intersphinx', 'sphinx_toolbox.collapse', 'sphinx_toolbox.shields', 'sphinx.ext.autosectionlabel', 'sphinx_tabs.tabs', 'sphinx_panels', 'rst2pdf.pdfbuilder']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster' # 'sphinx_rtd_theme'
html_theme = 'sphinx_book_theme'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# Set sphinx logo
html_logo = '_static/software_name.png'
html_favicon = '_static/nasa_logo.ico'

html_css_files = [
    'css/custom.css',
]

html_static_path = ['_static']
html_theme_options = {
    'description': "Python packages aiding research and development in prognostics and health management",
}

github_username = "nasa"
github_repository = "prog_models"
