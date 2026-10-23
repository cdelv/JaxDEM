"""Generate documentation media from the just-executed Sphinx-Gallery example.

ParaView reads actual VTKWriter collections. There is deliberately no media
cache: library changes must be reflected in the pictures on the next build.
"""

from __future__ import annotations

import ast
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from textwrap import indent
import time
from urllib.parse import urlencode

from tools.example_scene_exports import _DIAGNOSTICS_ONLY, export_example

_build_library_hash: str | None = None
_build_example_hashes: dict[Path, str] = {}


def start_visual_build(examples: Path, output: Path) -> None:
    """Discard generated media and freeze provenance before example execution."""
    global _build_library_hash, _build_example_hashes
    root = Path(__file__).resolve().parents[1]
    _build_library_hash = _library_hash(root)
    _build_example_hashes = {
        source.resolve(): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in examples.glob("*.py")
    }
    if output.name != "visuals" or output.parent.name != "_static":
        raise ValueError(
            "Visual builds must use the generated _static/visuals directory"
        )
    if output.exists():
        shutil.rmtree(output)


def _last_statement_line(source: str) -> int:
    tree = ast.parse(source)
    return max((node.end_lineno or node.lineno for node in tree.body), default=0)


def paraview_scraper(block, block_vars, gallery_conf):
    """Render once, after the last Python statement has actually executed."""
    if not block_vars.get("execute_script", True) or block_vars.get(
        "formatted_exception"
    ):
        return ""
    source = Path(block_vars["src_file"])
    if source.parent.name != "examples":
        return ""
    if block_vars.get("_jaxdem_scene_rendered"):
        return ""
    last_line = _last_statement_line(source.read_text())
    if not block.lineno <= last_line <= block.lineno + len(block.content.splitlines()):
        return ""
    block_vars["_jaxdem_scene_rendered"] = True

    executable = os.environ.get("JAXDEM_PVPYTHON", "pvpython")
    pvpython = shutil.which(executable)
    if pvpython is None:
        raise RuntimeError(
            "Visual documentation requires ParaView's pvpython. Set "
            "JAXDEM_PVPYTHON to its path; see the contribution guide."
        )
    root = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if _build_example_hashes.get(source.resolve(), source_hash) != source_hash:
        raise RuntimeError(
            f"{source.name} changed during the documentation build; rerun it."
        )
    library_hash = _library_hash(root)
    if _build_library_hash is not None and library_hash != _build_library_hash:
        raise RuntimeError(
            "JaxDEM sources changed during the documentation build; rerun it."
        )
    source_dir = Path(gallery_conf["src_dir"])
    output = source_dir / "_static" / "visuals" / source.stem
    # Only this example's generated directory is replaced, never user output.
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=f"jaxdem-media-{source.stem}-"
    ) as temporary:
        manifest = export_example(
            source.stem, block_vars["example_globals"], Path(temporary), output
        )
        if manifest is None:
            output.rmdir()
            if source.stem in _DIAGNOSTICS_ONLY:
                return ""
            raise RuntimeError(
                f"{source.name} produced no expected simulation scene; check its State/System captures."
            )
        manifest_path = Path(temporary) / "scene.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        backend = os.environ.get("JAXDEM_PARAVIEW_BACKEND", "EGL")
        command = [
            pvpython,
            "--force-offscreen-rendering",
            "--opengl-window-backend",
            backend,
            str(root / "tools" / "render_paraview_scene.py"),
            str(manifest_path),
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300, check=False
        )
        if result.returncode:
            raise RuntimeError(
                f"ParaView rendering failed for {source.name}:\n"
                f"{result.stdout[-2000:]}\n{result.stderr[-6000:]}"
            )
        poster = output / "scene.png"
        if not poster.is_file() or poster.stat().st_size < 1000:
            raise RuntimeError(
                f"ParaView did not produce a valid poster for {source.name}"
            )
        from PIL import Image, ImageStat

        with Image.open(poster) as picture:
            if max(ImageStat.Stat(picture.convert("RGB")).stddev) < 2:
                raise RuntimeError(
                    f"ParaView produced a blank poster for {source.name}"
                )

    if (
        hashlib.sha256(source.read_bytes()).hexdigest() != source_hash
        or _library_hash(root) != library_hash
    ):
        raise RuntimeError(
            "Sources changed while rendering documentation; rerun the build."
        )
    provenance = json.loads((output / "render.json").read_text())
    provenance.update(
        example=source.name,
        example_sha256=source_hash,
        library_revision=_git_revision(root),
        library_sha256=library_hash,
        media_seconds=round(time.monotonic() - started, 3),
        simulation=manifest.get("export_metadata", {}),
    )
    (output / "render.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # Register the actual ParaView poster with Sphinx-Gallery as well, so its
    # thumbnail and notebook machinery know this is an example figure.
    gallery_image = Path(next(block_vars["image_path_iterator"]))
    shutil.copy2(poster, gallery_image)
    from sphinx_gallery.scrapers import figure_rst

    title = str(manifest.get("title", source.stem.replace("_", " ")))
    rst = figure_rst([str(gallery_image)], str(source_dir), fig_titles=title)
    if (output / "animation.mp4").is_file():
        # Retain the image for gallery thumbnails, notebooks and print, while
        # HTML uses the video's identical poster instead of showing it twice.
        rst = (
            ".. container:: animation-poster\n\n" + indent(rst.strip(), "    ") + "\n\n"
        )
    caption = manifest.get("caption", "ParaView rendering of this run's VTK output.")
    rst += f"\n{caption}\n\n"
    groups = manifest.get("export_metadata", {}).get("groups", [])
    if len(groups) > 1:
        rst += (
            "Left to right: "
            + "; ".join(str(group["label"]) for group in groups)
            + ".\n\n"
        )
    base = f"../_static/visuals/{source.stem}"
    if (output / "animation.mp4").is_file():
        rst += (
            ".. raw:: html\n\n"
            f'    <figure class="simulation-media"><video controls loop muted playsinline '
            f'preload="none" poster="{base}/scene.png" aria-label="{html.escape(title)}">'
            f'<source src="{base}/animation.mp4" type="video/mp4">'
            f'<a href="{base}/animation.mp4">Download the animation</a></video>'
            f'<figcaption>{html.escape(str(manifest.get("animation_caption", "Saved simulation frames; no interpolated states.")))}</figcaption></figure>\n\n'
        )
    if (output / "scene.vtkjs").is_file():
        query = urlencode(
            {
                "name": f"{source.stem}.vtkjs",
                "url": f"{source.stem}/scene.vtkjs",
                "noGirder": "true",
            }
        )
        viewer = html.escape(f"../_static/visuals/viewer.html?{query}", quote=True)
        rst += (
            ".. raw:: html\n\n"
            '    <details class="simulation-media interactive-scene">'
            "<summary>Explore this scene in 3D</summary>"
            f'<iframe src="{viewer}" loading="lazy" '
            f'title="Interactive ParaView scene: {html.escape(title)}"></iframe>'
            f'<p><a href="{viewer}" target="_blank" rel="noopener">'
            "Open the interactive scene in a separate tab</a></p></details>\n\n"
        )
    return rst


def reset_jax_compilation_cache(gallery_conf, filename):
    """Release compiled executables between independent gallery examples."""
    import gc
    import sys

    if "jax" in sys.modules:
        sys.modules["jax"].clear_caches()
    gc.collect()


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _library_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "jaxdem").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
