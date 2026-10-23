# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Configure the JaxDEM documentation, API reference, and example gallery."""

import datetime
import importlib
import inspect
import os
import pathlib
import shutil
import sys
import types

from sphinx.util import logging as sphinx_logging

# ------------------------------------------------------------------
# Paths and project metadata
# ------------------------------------------------------------------

root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

# ------------------------------------------------------------------
# Performance report generation
# ------------------------------------------------------------------
try:
    from tools.performance_report import generate_report

    generate_report(
        results_file=str(root / "benchmarks" / "results.json"),
        output_dir=str(pathlib.Path(__file__).parent / "benchmarks"),
    )
except Exception as e:
    print(f"Could not generate performance report: {e}")

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
    "show_toc_level": 3,
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
html_show_copyright = False
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
    "sphinx_gallery.gen_gallery",
]

autosummary_generate = True
autosummary_ignore_module_all = True
napoleon_use_ivar = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "ignore-module-all": True,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAXDEM_DOCS_VISUALS", "1")
render_example_scenes = os.environ["JAXDEM_DOCS_VISUALS"] == "1"
jaxdem_visuals_enabled = (
    render_example_scenes and os.environ.get("JAXDEM_DOCS_EXECUTE", "1") == "1"
)
if jaxdem_visuals_enabled:
    tags.add("jaxdem_visuals")
if (
    render_example_scenes
    and os.environ.get("JAXDEM_DOCS_EXECUTE", "1") == "1"
    and shutil.which(os.environ.get("JAXDEM_PVPYTHON", "pvpython")) is None
):
    raise RuntimeError(
        "Visual documentation requires ParaView's pvpython. Set JAXDEM_PVPYTHON "
        "to its executable, or use JAXDEM_DOCS_VISUALS=0 for a text/plot-only build. "
        "See the documentation contribution guide."
    )
from tools.example_visuals import start_visual_build

if render_example_scenes and os.environ.get("JAXDEM_DOCS_EXECUTE", "1") == "1":
    start_visual_build(
        root / "examples", pathlib.Path(__file__).parent / "_static" / "visuals"
    )

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    # Every published figure must come from the current example and library.
    "filename_pattern": r"\.py$",
    "run_stale_examples": True,
    "reset_modules": (
        "matplotlib",
        "seaborn",
        "tools.example_visuals.reset_jax_compilation_cache",
    ),
    "image_scrapers": (
        ("matplotlib", "tools.example_visuals.paraview_scraper")
        if render_example_scenes
        else ("matplotlib",)
    ),
    # Keep the hand-written gallery index while publishing input assets.
    "copyfile_regex": r".*\.(rst|stl|json)$",
    "ignore_pattern": r"__init__",
    "download_all_examples": False,
    "plot_gallery": os.environ.get("JAXDEM_DOCS_EXECUTE", "1") == "1",
    "show_signature": False,
}

root_doc = "index"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]


def _warn_missing_example_api_reference(app, env, node, contnode):
    """Check API links in tutorials and the packing-analysis example."""
    docname = node.get("refdoc", "")
    name = docname.rsplit("/", 1)[-1]
    is_checked_page = name.endswith("_guide") or name in {
        "introduction",
        "intro_to_rl",
        "packing_analysis",
    }
    if (
        docname.startswith("auto_examples/")
        and is_checked_page
        and node.get("refdomain") == "py"
    ):
        sphinx_logging.getLogger(__name__).warning(
            "Unresolved API reference in tutorial or example: %s",
            node.get("reftarget"),
            location=node,
            type="ref",
            subtype="guide_api",
        )


def setup(app):
    from tools.docs_media import SimulationImage

    app.add_config_value("jaxdem_visuals_enabled", False, "env")
    app.add_directive("simulation-image", SimulationImage)
    # Let other resolvers (including optional intersphinx) try first.
    app.connect("missing-reference", _warn_missing_example_api_reference, priority=999)


# ------------------------------------------------------------------
# Linkcode configuration
# ------------------------------------------------------------------
from typing import Mapping, Optional, cast

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
        f"https://github.com/{github_user}/{github_repo}/blob/{github_version}/{rel_fn}"
    )

    # Append line numbers if available
    if lineno is not None:
        end_lineno = lineno + len(src_lines) - 1
        if end_lineno >= lineno:
            url += f"#L{lineno}-L{end_lineno}"

    return url
