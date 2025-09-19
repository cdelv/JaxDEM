import datetime
import importlib
import inspect
import os
import pathlib
import sys
import types

# ------------------------------------------------------------------
# Paths and project metadata
# ------------------------------------------------------------------

root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

project = "JaxDEM"
author = "Carlos Andres del Valle"
release = ""
copyright = f"{datetime.datetime.now().year}, {author}"

# ------------------------------------------------------------------
# Theme configuration
# ------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {release}".strip()
html_short_title = html_title

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-button"],
    "show_prev_next": True,
    "show_nav_level": 3,
    "collapse_navigation": True,
    "navigation_with_keys": True,
    "primary_sidebar_end": ["indices.html", "sidebar-ethical-ads.html"],
    "back_to_top_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/cdelv/JaxDEM",
            "icon": "fab fa-github",
            "type": "fontawesome",
        }
    ],
    "use_edit_page_button": False,
}

html_show_sourcelink = False
html_sidebars = {
    "index": ["custom-sidebar.html", "sidebar-nav-bs"],
    "*": ["sidebar-nav-bs", "sidebar-ethical-ads"],
}

# ------------------------------------------------------------------
# Sphinx extensions and options
# ------------------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r".*\.py",
    "ignore_pattern": r"__init__",
    "download_all_examples": False,
    "plot_gallery": True,
    "show_signature": False,
}

root_doc = "index"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]

# ------------------------------------------------------------------
# Linkcode configuration
# ------------------------------------------------------------------
from typing import Mapping, Optional, cast
import inspect
import importlib
import os
import types

github_user = "cdelv"
github_repo = "JaxDEM"
github_version = "main"


def linkcode_resolve(domain: str, info: Mapping[str, object]) -> Optional[str]:
    """
    Determine the URL to include in the "View source" link.
    Requires globals: root, github_user, github_repo, github_version.
    """
    if domain != "py":
        return None

    modname = cast(Optional[str], info.get("module"))
    fullname = cast(Optional[str], info.get("fullname"))
    if not modname or not fullname:
        return None

    # Import the module
    try:
        submod = importlib.import_module(modname)
    except ImportError:
        return None

    obj: object = submod

    # Traverse through the object parts (e.g., Class.method)
    for part in fullname.split("."):
        try:
            obj = cast(object, getattr(obj, part))
        except AttributeError:
            return None

        # Allow submodules to pass through
        if inspect.ismodule(obj):
            continue

        # Normalize to an inspectable object
        if not (
            inspect.isfunction(obj)
            or inspect.isclass(obj)
            or inspect.ismethod(obj)
            or inspect.iscode(obj)
        ):
            wrapped = getattr(obj, "__wrapped__", None)
            if wrapped is not None:
                obj = cast(object, wrapped)
            elif isinstance(obj, property) and obj.fget:
                obj = cast(object, obj.fget)
            elif inspect.isclass(obj):
                init = getattr(obj, "__init__", None)
                if inspect.isfunction(init):
                    obj = cast(object, init)
                else:
                    return None
            else:
                return None

    # Ignore built-ins or modules without Python source
    if isinstance(
        obj, (types.BuiltinFunctionType, types.BuiltinMethodType, types.ModuleType)
    ):
        return None

    # Choose the best target to inspect (stable for typing and runtime)
    target: object = obj
    if isinstance(target, property) and target.fget:
        target = target.fget  # type: ignore[assignment]
    elif inspect.isclass(target):
        init = getattr(target, "__init__", None)
        if inspect.isfunction(init):
            target = init  # type: ignore[assignment]
    else:
        wrapped = getattr(target, "__wrapped__", None)
        if wrapped is not None:
            target = cast(object, wrapped)

    # Resolve source file path
    try:
        fn = inspect.getsourcefile(target)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        fn = None

    if not fn or not os.path.isabs(fn):
        return None

    # Path relative to the repository root
    try:
        rel_fn = os.path.relpath(fn, start=root)
    except ValueError:
        return None

    # Line numbers
    try:
        src_lines, lineno = inspect.getsourcelines(target)  # type: ignore[arg-type]
    except (TypeError, OSError):
        lineno = None
        src_lines = []

    # Construct the GitHub URL
    url = (
        f"https://github.com/{github_user}/{github_repo}/blob/"
        f"{github_version}/{rel_fn}"
    )

    # Append line numbers if available
    if lineno is not None:
        end_lineno = lineno + len(src_lines) - 1
        if end_lineno >= lineno:
            url += f"#L{lineno}-L{end_lineno}"

    return url
