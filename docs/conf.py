# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

project = "LLM Detector Benchmark 🔎"
copyright = "2024, Marluxiaboss"
author = "Marluxiaboss"
release = "0.1"
html_title = "LLM Detector Benchmark 🔎"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.todo",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"  # or 'sphinx_book_theme'
# html_theme = 'sphinx_book_theme'
html_static_path = ["_static"]

# Specify the favicon
html_favicon = "_static/glass.png"

# the logo at the top left of the page
# html_logo = "_static/glass.png"

master_doc = "index"


# font
"""
html_theme_options = {
    "light_css_variables": {
        "font-stack": "Arial, sans-serif",
        "font-stack--monospace": "Courier, monospace",
        "font-stack--headings": "Georgia, serif",
    },
}
"""

# autoapi
extensions.append("autoapi.extension")
autoapi_dirs = ["../detector_benchmark"]

# landing page
# templates_path = ["_templates"]
# html_additional_pages = {"index": "subpages/getting_started.md"}
