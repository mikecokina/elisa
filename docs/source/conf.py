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
import warnings

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../src"))

# Suppress the DeprecationWarning emitted by elisa.conf.settings during import
# ("Variable `atlas` in configuration section `support` is not longer supported").
warnings.filterwarnings("ignore", category=DeprecationWarning, module="elisa")

from elisa import __version__ as release

# -- Project information -----------------------------------------------------

project = "elisa"
# noinspection PyShadowingBuiltins
copyright = "2026, Michal Cokina, Miroslav Fedurco"  # noqa: A001
author = "Michal Cokina, Miroslav Fedurco"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = {".rst": "restructuredtext"}

# The master toctree document.
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "nature"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "custom.css",  # Reference your CSS file
]

# Output file base name for HTML help builder.
htmlhelp_basename = "elisadocs"


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Suppress "more than one target found for cross-reference" warnings that arise
# because multiple modules define classes with identical short names (e.g. Plot,
# Orbit). Sphinx emits ref.python warnings.
suppress_warnings = ["ref.python"]

#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"
