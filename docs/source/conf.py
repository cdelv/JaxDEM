# … existing imports …
import pathlib, sys, datetime

root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

project = "JaxDEM"
author = "Carlos Andres del Valle"
release = ""
copyright = f"{datetime.datetime.now().year}, {author}"

# ----------------------------------------------------------------
# theme configuration
# ----------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"{project} {release}".strip()
html_short_title = html_title

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-button"],
    "show_prev_next": True,
    "show_nav_level": 2,
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
        },
    ],
    "use_edit_page_button": False,
}

html_show_sourcelink = False
html_sidebars = {
    "index": ["custom-sidebar.html", "sidebar-nav-bs"],
    "*": ["sidebar-nav-bs", "sidebar-ethical-ads"],
}

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
# Link code
# ------------------------------------------------------------------

import os
import inspect
import sys
import types  # Import the types module
import importlib  # Ensure importlib is imported for consistency

# Configuration for the linkcode extension
# Adjust these for your repository
github_user = "cdelv"
github_repo = "JaxDEM"
github_version = "main"  # or "master" or a specific tag like "v1.0.0"


def linkcode_resolve(domain, info):
    """
    Determine the URL to include in the "View source" link.
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    # --- ADD THIS CHECK HERE ---
    if not modname or not fullname:
        return None
    # ---------------------------

    # Try to import the module
    try:
        submod = importlib.import_module(modname)
    except ImportError:
        return None

    obj = submod
    # Traverse through the object parts (e.g., Class.method)
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None  # Object part not found
        # Handle cases where getattr returns a non-inspectable proxy
        # like for slots or certain properties where inspection needs to go deeper
        if inspect.ismodule(
            obj
        ):  # If we hit a submodule, it's typically fine to continue
            continue
        if not (
            inspect.isfunction(obj)
            or inspect.isclass(obj)
            or inspect.ismethod(obj)
            or inspect.iscode(obj)
        ):
            # If it's not a standard inspectable type, try its __wrapped__ or fget (for properties)
            if hasattr(obj, "__wrapped__"):
                obj = obj.__wrapped__
            elif isinstance(obj, property) and obj.fget:
                obj = obj.fget
            elif hasattr(obj, "__init__") and inspect.isfunction(obj.__init__):
                # For classes, often __init__ has the source
                obj = obj.__init__
            else:
                # If still not a clear source-inspectable object, bail out.
                # This prevents errors for things like method-wrapper or built-ins.
                return None

    # Handle built-in objects or objects without a Python source file
    if isinstance(
        obj, (types.BuiltinFunctionType, types.BuiltinMethodType, types.ModuleType)
    ):
        return None  # Built-in, C-module, or just the module itself (no specific line to link to)

    # Get the source file path
    try:
        fn = inspect.getsourcefile(obj)
    except (TypeError, AttributeError):
        # This catches issues with method-wrapper, properties, etc.,
        # that inspect.getsourcefile might struggle with directly.
        # Fallback to fget for properties, or __init__ for classes if obj is a class
        if isinstance(obj, property) and obj.fget:
            fn = inspect.getsourcefile(obj.fget)
        elif inspect.isclass(obj) and hasattr(obj, "__init__"):
            fn = inspect.getsourcefile(obj.__init__)
        elif hasattr(obj, "__wrapped__"):  # For decorators
            fn = inspect.getsourcefile(obj.__wrapped__)
        else:
            fn = None  # Still no source file found

    if not fn:
        return None  # No source file to link to

    # Ensure fn is an absolute path
    if not os.path.isabs(fn):
        return None

    # Get the path relative to the repository root
    try:
        fn = os.path.relpath(fn, start=root)
    except ValueError:
        return None  # File is not under the specified root directory

    # Get line numbers
    try:
        lines, lineno = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        lineno = None

    # Construct the GitHub URL
    url_format = (
        f"https://github.com/{github_user}/{github_repo}/blob/{github_version}/{fn}"
    )

    # Append line numbers if available
    if lineno is not None:
        try:
            end_lineno = lineno + len(lines) - 1
            if end_lineno >= lineno:  # Prevent invalid ranges if lines is empty
                url_format += f"#L{lineno}-L{end_lineno}"
        except TypeError:
            pass  # Can happen if 'lines' isn't sequence-like

    return url_format
