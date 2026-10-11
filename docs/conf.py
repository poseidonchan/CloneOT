import os
import sys

sys.path.insert(0, os.path.abspath('..'))
import cloneot  # noqa: E402

project = 'CloneOT'
author = 'Yanshuo Chen, Wei Chen, Heng Huang'
copyright = '2026, Yanshuo Chen'
release = version = cloneot.__version__

extensions = [
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
]
autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_default_options = {'members': True, 'show-inheritance': False}
napoleon_numpy_docstring = True
typehints_use_signature = True
nb_execution_mode = 'off'                 # the tutorials are executed on a GPU and stored with their outputs
myst_enable_extensions = ['dollarmath', 'amsmath', 'colon_fence']
source_suffix = {'.md': 'myst-nb', '.ipynb': 'myst-nb'}
exclude_patterns = ['_build', 'jupyter_execute', '**.ipynb_checkpoints']

html_theme = 'furo'
html_title = 'CloneOT'
html_static_path = ['_static']
html_theme_options = {
    'source_repository': 'https://github.com/poseidonchan/CloneOT',
    'source_branch': 'main',
    'source_directory': 'docs/',
}
