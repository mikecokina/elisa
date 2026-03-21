# Configuration file for the Sphinx documentation builder.  # noqa: INP001
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# ---------------------------------------------------------------------------
# elisa's settings.py calls warnings.simplefilter("always", DeprecationWarning)
# *without* a catch_warnings() guard, so standard filter-based suppression
# cannot override it.  Patching warnings.warn itself is the only hook that
# survives that, because catch_warnings() only saves/restores
# warnings.filters and warnings.showwarning - not warnings.warn.
# ---------------------------------------------------------------------------
_original_warn = warnings.warn


def _suppress_atlas_deprecation(
    message: str | Warning,
    category: type[Warning] | None = None,
    stacklevel: int = 1,
    *,
    source: Any = None,
) -> None:
    """Suppress atlas-related DeprecationWarning but pass through others."""
    if "atlas" in str(message) and "not longer supported" in str(message):
        return
    _original_warn(message, category, stacklevel, source=source)


warnings.warn = _suppress_atlas_deprecation

from elisa import __version__ as release  # noqa: E402, F401

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

# Sphinx 9+ defaults to language="en" internally and searches for locale files
# even when no translations exist.  An empty list disables that lookup.
locale_dirs = []
