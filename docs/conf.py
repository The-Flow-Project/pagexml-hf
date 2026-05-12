"""Sphinx configuration for pagexml-hf documentation."""

import sys
from pathlib import Path

# Add pagexml_hf/ to path so autodoc can find the package
sys.path.insert(0, str(Path(__file__).parent.parent / "pagexml_hf"))

from pagexml_hf import __version__

project = "pagexml-hf"
copyright = "2026, Jonas Widmer, Dana Rebecca Meyer"
author = "Jonas Widmer, Dana Rebecca Meyer"
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autosummary_generate = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "sphinx_rtd_theme"
html_show_sphinx = False

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
