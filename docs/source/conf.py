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
    "navigation_with_keys": False,
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
    "undoc-members": False,
    "inherited-members": True,
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
}


root_doc = "index"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]

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
        return []  # nothing below a plain module
    prefix = mod.__name__ + "."
    return sorted(
        m.name
        for m in pkgutil.walk_packages(mod.__path__, prefix)
        if "." not in m.name[len(prefix) :]
    )


# ------------ helpers ---------------------------------------------
# (Keep _top_level_packages and _immediate_submodules as they are)


def _format_for_autosummary(mod_name: str, maxdepth: int) -> list[str]:
    """
    Recursively generate lines for autosummary, starting from the
    immediate sub-modules of mod_name, respecting maxdepth.
    """
    lines: list[str] = []
    # Get immediate children of the current module
    children = _immediate_submodules(mod_name)

    # For depth 0 (top level for autosummary), just list the children directly
    # and then recursively list their children for deeper levels.
    for child_name in children:
        lines.append(f"   {child_name}")  # Indent by 3 spaces for autosummary

        # If maxdepth allows, recursively get children of this child
        # and increase depth for indentation
        if maxdepth > 0:  # maxdepth refers to levels *below* the current one
            lines.extend(_format_for_autosummary_recursive(child_name, 1, maxdepth))
    return lines


def _format_for_autosummary_recursive(
    mod_name: str, current_depth: int, maxdepth: int
) -> list[str]:
    """
    Helper for _format_for_autosummary to handle recursion and indentation.
    `current_depth` is the current indentation level relative to the first
    level of autosummary (i.e., `jaxdem.collider` is depth 0 in autosummary terms).
    """
    lines: list[str] = []
    if current_depth > maxdepth:
        return lines  # Stop recursion if maxdepth is exceeded

    children = _immediate_submodules(mod_name)
    indent = "   " * (current_depth + 1)  # Additional 3 spaces for each depth level

    for child_name in children:
        lines.append(f"{indent}{child_name}")
        lines.extend(
            _format_for_autosummary_recursive(child_name, current_depth + 1, maxdepth)
        )
    return lines


# ------------ writer ----------------------------------------------
def _write_api_index(app) -> None:
    src_dir = Path(__file__).parent  # docs/source/
    out = src_dir / "reference" / "api.rst"
    out.parent.mkdir(parents=True, exist_ok=True)

    top_level_pkgs_found = _top_level_packages(root)
    if not top_level_pkgs_found:
        _log.warning(
            "No top-level packages found under %s to build API reference.", root
        )
        return

    main_pkg_name = top_level_pkgs_found[0]  # Assuming 'jaxdem' is the first/only one

    lines: list[str] = [
        "API reference",
        "=============",
        "",  # Empty line before the first section
        main_pkg_name,  # The main package name as a section title
        "-" * len(main_pkg_name),  # Underline for the main package title
        "",  # Empty line after the title
        ".. autosummary::",
        "   :toctree: generated",
        "   :nosignatures:",
        "",  # Empty line after autosummary directive
    ]

    # Call the new helper to get the correctly formatted autosummary entries
    # We want to list the children of 'main_pkg_name' up to maxdepth.
    lines.extend(
        _format_for_autosummary(main_pkg_name, maxdepth=2)
    )  # maxdepth=2 to get 'jaxdem.module.submodule'
    lines.append("")  # Final empty line

    out.write_text("\n".join(lines), encoding="utf-8")
    _log.info("API index regenerated for package: %s", main_pkg_name)


# ... (rest of your setup function remains the same)


# ------------ Sphinx hook -----------------------------------------
def setup(app):
    app.connect("builder-inited", _write_api_index)
    return {"parallel_read_safe": True}


# ------------------------------------------------------------------
# Link code
# ------------------------------------------------------------------
# conf.py
# ... (rest of your conf.py, including existing imports and _write_api_index) ...

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
