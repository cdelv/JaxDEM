# … existing imports …
import pathlib, sys, datetime
root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

project   = "JaxDEM"
author    = "Carlos Andres del Valle"
release   = ""
copyright = f"{datetime.datetime.now().year}, {author}"

# ----------------------------------------------------------------
# theme configuration
# ----------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"{project} {release}"        
html_short_title = html_title

html_theme_options = {
    "navbar_start":  ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end":    ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-button"],              
    "show_prev_next": True,

    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/cdelv/JaxDEM",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
    ],
}

html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"]
}
use_edit_page_button = True
navigation_with_keys = True
collapse_navigation = True
navbar_align = "left"

extensions = [
    "myst_parser",        
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google / NumPy style docstrings
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

html_static_path = ["_static"]
templates_path   = ["_templates"]
html_css_files   = ["custom.css"] 