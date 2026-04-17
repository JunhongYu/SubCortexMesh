# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
import shutil

sys.path.insert(0, os.path.abspath('../..')) #sphinx access

# -- Project information -----------------------------------------------------
project = 'SubCortexMesh'
copyright = '2026, Charly H. A. Billaud'
author = 'Charly H. A. Billaud, Cognitive and Brain Health Laboratory - Nanyang Technological University'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

myst_enable_extensions = [
    "footnote",
]

autodoc_mock_imports = [
    'ants',
    'antspyx',
    'pyvista',
    'vtk',
    'vtkmodules',
    'numpy',
    'pandas',
    'matplotlib',
    'mpl_toolkits',
    'nibabel',
    'scipy',
    'sklearn',
    'brainstat',
    'brainspace',
    'matplotlib',
    'requests',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

# Copy figures into the build path so README links resolve
figures_src = os.path.abspath('../../figures')
figures_dst = os.path.abspath('figures')
if os.path.exists(figures_src) and not os.path.exists(figures_dst):
    shutil.copytree(figures_src, figures_dst)