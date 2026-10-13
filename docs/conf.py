"""Sphinx configuration for the loopus² documentation."""

import sys
from importlib.metadata import version as _package_version
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _generate  # noqa: E402

project = "loopus²"
author = "the loopus² contributors"
copyright = "2026, the loopus² contributors"
release = _package_version("loopuss")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

# API reference: NumPy-style docstrings carry the types, so signatures stay
# short.
autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_google_docstring = False
napoleon_use_rtype = False

myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

html_theme = "furo"
html_title = f"loopus² {release}"
html_static_path = ["sfx"]
html_theme_options = {
    "light_logo": "logo_black_small.png",
    "dark_logo": "logo_white_small.png",
    "sidebar_hide_name": True,
}

# PDF: the LaTeX build starts from pdf.md.
latex_documents = [
    ("pdf", "loopuss.tex", f"loopus² {release}", author, "manual", True),
]

_generate.write_pages(HERE)
