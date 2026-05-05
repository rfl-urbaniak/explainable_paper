import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "PCI Paper"
copyright = "2026, Rafal Urbaniak"
author = "Rafal Urbaniak"

extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinxcontrib.jquery",
]

nbsphinx_execute = "never"

html_sourcelink_suffix = ""

master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
