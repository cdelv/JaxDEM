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
html_title = f"{project} {release}".strip()      
html_short_title = html_title

html_theme_options = {
    "navbar_start":  ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end":    ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-button"],              
    "show_prev_next": True,
    "show_nav_level": 2,
    "collapse_navigation": True,
    "navigation_with_keys" : True,
    "primary_sidebar_end": ["indices.html", "sidebar-ethical-ads.html"],
    "back_to_top_button": True,

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
    "index": ["custom-sidebar.html", "sidebar-nav-bs"],
    "*": ["sidebar-nav-bs", "sidebar-ethical-ads"]
}

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
templates_path = ["_templates"]
html_css_files   = ["custom.css"] 

# ------------------------------------------------------------------
# Automatic API reference  (docs/source/reference/api.rst)
# ------------------------------------------------------------------
import importlib, pkgutil
from pathlib import Path
from sphinx.util import logging

_log = logging.getLogger(__name__)


# ------------ helpers ---------------------------------------------
def _top_level_packages(repo_root: Path) -> list[str]:
    """All packages that sit directly in *repo_root*."""
    return sorted(m.name for m in pkgutil.iter_modules([str(repo_root)]) if m.ispkg)


def _immediate_submodules(mod_name: str) -> list[str]:
    """
    Depth-1 sub-modules of *mod_name*.
    If *mod_name* is not a package (no __path__), return [].
    """
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, "__path__"):
        return []                            # nothing below a plain module
    prefix = mod.__name__ + "."
    return sorted(
        m.name
        for m in pkgutil.walk_packages(mod.__path__, prefix)
        if "." not in m.name[len(prefix) :]
    )


def _tree_lines(mod_name: str, depth: int, maxdepth: int) -> list[str]:
    """Recursively build an indented list up to *maxdepth* levels."""
    tab_indent = "\t" * depth                        # real tab(s)
    lines = [f"   {tab_indent}{mod_name}"]           # 3 spaces for RST
    if depth < maxdepth:
        for child in _immediate_submodules(mod_name):
            lines.extend(_tree_lines(child, depth + 1, maxdepth))
    return lines


# ------------ writer ----------------------------------------------
def _write_api_index(app) -> None:
    src_dir = Path(__file__).parent            # docs/source/
    out     = src_dir / "reference" / "api.rst"
    out.parent.mkdir(parents=True, exist_ok=True)

    pkgs = _top_level_packages(root)           # ‹root› declared above

    lines: list[str] = ["API reference", "=============", ""]

    for pkg in pkgs:
        lines += [pkg, "-" * len(pkg), ""]

        lines += [
            ".. autosummary::",
            "   :toctree: generated",
            "   :nosignatures:",
            "   :recursive:",
            "",
        ]
        lines += _tree_lines(pkg, depth=0, maxdepth=2)  # package + 2 levels
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    _log.info("API index regenerated: %d top-level packages", len(pkgs))


# ------------ Sphinx hook -----------------------------------------
def setup(app):
    app.connect("builder-inited", _write_api_index)
    return {"parallel_read_safe": True}