# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Causal AI Scientist'
copyright = '2025, causalNLP'
author = 'causalNLP'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # Google/NumPy style docstrings
    "sphinx_autodoc_typehints",   # render type hints nicely
    "sphinx.ext.viewcode",        # links to source
    "sphinx_design",
    "nbsphinx",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_logo = "_static/cais.png"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}

"""
autodoc_mock_imports = [
    "openai",
    "anthropic",
    "langchain",
    "langchain_anthropic",
    "torch",
    "tensorflow",
    "sklearn",
    "pandas",
    "numpy",
]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/tests/**",
    "tests/**",
    "**/notebooks/**",
    "**/examples/**",
]
"""

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
typehints_fully_qualified = False
typehints_document_rtype = False
