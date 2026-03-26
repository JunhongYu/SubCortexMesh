# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from unittest.mock import MagicMock
import sys
import os

MOCK_MODULES = [
    'ants', 'antspyx', 'pyvista', 'vtk', 'sklearn',
    'sklearn.preprocessing', 'sklearn.neighbors',
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

sys.path.insert(0, os.path.abspath('../..')) #sphinx access

# -- Project information -----------------------------------------------------
project = 'SubCortexMesh'
copyright = '2026, Charly H. A. Billaud'
author = 'Charly H. A. Billaud'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # auto-generates docs from docstrings
    'sphinx.ext.napoleon',      # supports Google/NumPy-style docstrings
    'sphinx.ext.viewcode',      # adds [source] links
    'nbsphinx',                 # renders Jupyter notebooks as tutorials
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']