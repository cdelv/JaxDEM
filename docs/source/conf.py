# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import pathlib, sys, datetime
root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))      # import jaxdem from repo

project = 'JaxDEM'
author = 'Carlos Andres del Valle'
copyright = f"{datetime.datetime.now().year}, {author}"
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "myst_parser",
]

autosummary_generate = True
autodoc_typehints    = "description"

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "image_light": "logo-light.svg",
        "image_dark":  "logo-dark.svg",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links", "search-field"],
    "show_toc_level": 2,                  # numbered headings depth
    "collapse_navigation": True,          # ← collapsible sidebar

    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/cdelv/JaxDEM",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
    ],
}

# narrower content column (optional)
html_css_files = ["custom.css"]           # see step-3

html_static_path = ["_static"]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'