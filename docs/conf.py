# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Public Infrastructure Service Access"
author = "EiriniK"
release = "2.0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autoapi_dirs = ["../pisa"]

# AutoAPI configuration
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]

# Skip documenting module-level variables like loggers
autoapi_python_class_content = "class"


def skip_member(app, what, name, obj, skip, options):
    """Skip certain members from being documented."""
    import logging

    # Skip logger objects specifically
    if name == "logger" and isinstance(obj, logging.Logger):
        return True
    # Skip private members (starting with underscore) but not special methods like __init__
    if name.startswith("_") and not name.startswith("__"):
        return True
    return skip


def setup(sphinx):
    """Connect the skip function to the autoapi-skip-member event."""
    sphinx.connect("autoapi-skip-member", skip_member)


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme = "sphinx_rtd_theme"
