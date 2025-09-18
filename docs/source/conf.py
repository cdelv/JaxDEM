import datetime
import importlib
import importlib.util
import inspect
import os
import pathlib
import pkgutil
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

github_user = "cdelv"
github_repo = "JaxDEM"
github_version = "main"  # e.g., "main", "master", or a tag like "v1.0.0"


def linkcode_resolve(domain, info):
    """
    Determine the URL to include in the "View source" link.
    """
    if domain != "py":
        return None

    modname = info.get("module")
    fullname = info.get("fullname")
    if not modname or not fullname:
        return None

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
            if hasattr(obj, "__wrapped__"):
                obj = obj.__wrapped__
            elif isinstance(obj, property) and obj.fget:
                obj = obj.fget
            elif hasattr(obj, "__init__") and inspect.isfunction(obj.__init__):
                obj = obj.__init__
            else:
                return None

    # Ignore built-ins or modules without Python source
    if isinstance(
        obj,
        (
            types.BuiltinFunctionType,
            types.BuiltinMethodType,
            types.ModuleType,
        ),
    ):
        return None

    # Resolve source file path
    try:
        fn = inspect.getsourcefile(obj)
    except (TypeError, AttributeError):
        if isinstance(obj, property) and obj.fget:
            fn = inspect.getsourcefile(obj.fget)
        elif inspect.isclass(obj) and hasattr(obj, "__init__"):
            fn = inspect.getsourcefile(obj.__init__)
        elif hasattr(obj, "__wrapped__"):
            fn = inspect.getsourcefile(obj.__wrapped__)
        else:
            fn = None

    if not fn or not os.path.isabs(fn):
        return None

    # Path relative to repo root
    try:
        fn = os.path.relpath(fn, start=root)
    except ValueError:
        return None

    # Line numbers
    try:
        lines, lineno = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        lineno = None
        lines = []

    url = (
        f"https://github.com/{github_user}/{github_repo}/blob/" f"{github_version}/{fn}"
    )

    if lineno is not None:
        try:
            end_lineno = lineno + len(lines) - 1
            if end_lineno >= lineno:
                url += f"#L{lineno}-L{end_lineno}"
        except TypeError:
            pass

    return url


# -----------------------------
# Auto-generate API pages
# -----------------------------
import pkgutil, importlib, importlib.util, pathlib


def _top_level_modules(package_name: str) -> list[str]:
    """Return sorted list of the package itself and its immediate children (no _prefixed)."""
    mods = {package_name}
    spec = importlib.util.find_spec(package_name)
    if spec and spec.submodule_search_locations:
        for mi in pkgutil.iter_modules(spec.submodule_search_locations):
            if mi.name.startswith("_"):
                continue
            mods.add(f"{package_name}.{mi.name}")
    return sorted(mods)


def _write_api_index(app) -> None:
    """source/reference/api.rst: top-level jaxdem modules (exclude bare 'jaxdem')."""
    modules = [m for m in _top_level_modules("jaxdem") if m != "jaxdem"]

    lines = [
        ":orphan:",
        ":html_theme.sidebar_secondary.remove:",
        "",
        "API reference",
        "=============",
        "",
        ".. autosummary::",
        "   :toctree: generated",
        "   :caption: Top-level modules",
        "   :nosignatures:",
        "",
        *[f"   {m}" for m in modules],
        "",
    ]
    path = pathlib.Path(__file__).parent / "reference" / "api.rst"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = "\n".join(lines)
    try:
        cur = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        cur = ""
    if cur != new:
        path.write_text(new, encoding="utf-8")


def _write_module_stubs(app) -> None:
    """Write reference/generated/<module>.rst for each top-level jaxdem.* module."""
    top = [m for m in _top_level_modules("jaxdem") if m != "jaxdem"]

    for mod in top:
        # list immediate children of this module (no deeper recursion)
        subs = [m for m in _top_level_modules(mod) if m != mod]

        lines = [
            f"{mod}",
            "=" * len(mod),
            "",
            f".. automodule:: {mod}",
            "",
            ".. autosummary::",
            "   :toctree: .",
            "   :nosignatures:",
            "",
            *[f"   {m}" for m in subs],
            "",
        ]

        stub = pathlib.Path(__file__).parent / "reference" / "generated" / f"{mod}.rst"
        stub.parent.mkdir(parents=True, exist_ok=True)
        new = "\n".join(lines)
        try:
            cur = stub.read_text(encoding="utf-8")
        except FileNotFoundError:
            cur = ""
        if cur != new:
            stub.write_text(new, encoding="utf-8")


def setup(app):
    app.connect("builder-inited", _write_api_index)
    app.connect("builder-inited", _write_module_stubs)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
