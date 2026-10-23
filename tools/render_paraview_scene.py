#!/usr/bin/env pvpython
# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Render VTKWriter collections using ParaView.

Run with the ``pvpython`` executable distributed with ParaView 6.2::

    pvpython --force-offscreen-rendering --opengl-window-backend EGL \
        tools/render_paraview_scene.py manifest.json

The JSON manifest selects PVD collections, their representations and the output
directory. All animation frames correspond to saved simulation timesteps. The
documentation scraper calls this renderer after executing each current example.
"""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
import zipfile


def _sample_times(times: list[float], maximum: int) -> list[float]:
    """Select native timesteps, including the first and last, without interpolation."""
    if len(times) <= maximum:
        return times
    indices = {round(i * (len(times) - 1) / (maximum - 1)) for i in range(maximum)}
    return [times[i] for i in sorted(indices)]


def _native_times(reader: Any) -> list[float]:
    values = getattr(reader, "TimestepValues", [])
    return sorted({float(value) for value in values})


def _same_times(left: list[float], right: list[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(left, right)
    )


def _valid_bounds(bounds: list[float]) -> bool:
    return all(math.isfinite(x) for x in bounds) and all(
        bounds[2 * axis] <= bounds[2 * axis + 1] for axis in range(3)
    )


def _shared_glance_viewer(paraview: Any, directory: Path) -> dict[str, Any]:
    """Share the bundled browser application; every scene still exports afresh."""
    resources = Path(
        paraview.servermanager.vtkPVFileInformation.GetParaViewSharedResourcesDirectory()
    )
    source = resources / "web" / "glance" / "ParaViewGlance.html"
    if not source.is_file():
        raise RuntimeError(f"The ParaView bundle has no Glance viewer at {source}")
    content = source.read_text(encoding="utf-8")
    # The official 6.2 inline bundle references two absent optional modules.
    # Neither handles vtkjs. Remove their dead tags so the inline application
    # does not request absent ITK image readers or service worker integration.
    content, removed = re.subn(
        r'<script\s+src="glance-external-(?:ITKReader|Workbox)\.[^"]+\.js"\s*></script>',
        "",
        content,
    )

    class Dependencies(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.external: list[str] = []

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            attributes = dict(attrs)
            asset = attributes.get("src") if tag == "script" else None
            if tag == "link" and attributes.get("rel") in {"stylesheet", "icon"}:
                asset = attributes.get("href")
            if asset and not asset.startswith(("data:", "#")):
                self.external.append(asset)

    dependencies = Dependencies()
    dependencies.feed(content)
    if dependencies.external:
        raise RuntimeError(
            "The bundled Glance viewer has unsupported external dependencies: "
            + ", ".join(dependencies.external)
        )
    if "processURLArgs" not in content or "autoLoadRemotes" not in content:
        raise RuntimeError("The bundled Glance viewer lacks name/url scene loading")
    target = directory / "viewer.html"
    encoded = content.encode("utf-8")
    if not target.is_file() or target.read_bytes() != encoded:
        with tempfile.NamedTemporaryFile(
            dir=directory, suffix=".html", delete=False
        ) as tmp:
            tmp.write(encoded)
            temporary_path = Path(tmp.name)
        temporary_path.replace(target)
    license_files = []
    for license_source in sorted((source.parent / "www").glob("*.LICENSE.txt")):
        license_target = directory / license_source.name
        shutil.copyfile(license_source, license_target)
        license_files.append(license_source.name)
    return {
        "interactive_viewer": "viewer.html",
        "interactive_scene": "scene.vtkjs",
        "viewer_source": str(source),
        "viewer_sha256": hashlib.sha256(encoded).hexdigest(),
        "viewer_bytes": len(encoded),
        "removed_optional_script_tags": removed,
        "viewer_license_files": license_files,
    }


def render(manifest_path: Path) -> dict[str, Any]:
    """Create a poster, video and scene for a shared Glance viewer."""
    try:
        import paraview
        from paraview import simple as pv
    except ImportError as exc:
        raise RuntimeError(
            "Run this script with ParaView's pvpython executable; installing vtk "
            "or running an ordinary Python interpreter does not provide ParaView."
        ) from exc

    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    output = Path(manifest["output"]).expanduser()
    if not output.is_absolute():
        output = manifest_path.parent / output
    output.mkdir(parents=True, exist_ok=True)
    size = [int(x) for x in manifest.get("size", [960, 600])]
    if len(size) != 2 or min(size) < 64 or max(size) > 4096:
        raise ValueError("size must contain two pixel dimensions between 64 and 4096")
    camera = manifest.get("camera", "xy")
    if camera not in {"xy", "isometric"}:
        raise ValueError("camera must be 'xy' or 'isometric'")
    maximum = int(manifest.get("max_frames", 48))
    fps = int(manifest.get("fps", 12))
    if maximum < 2 or maximum > 120 or fps < 1 or fps > 60:
        raise ValueError("max_frames must be 2..120 and fps must be 1..60")
    series = manifest.get("series", [])
    if not series:
        raise ValueError("The scene manifest must contain at least one series")
    animate = bool(manifest.get("animate", False))
    interactive = bool(manifest.get("interactive", False))
    time_label = str(manifest.get("time_label", "t"))
    ffmpeg = shutil.which(os.environ.get("FFMPEG", "ffmpeg"))
    if animate and ffmpeg is None:
        raise RuntimeError("Animation encoding requires ffmpeg on PATH (or FFMPEG)")

    pv._DisableFirstRenderCameraReset()
    view = pv.CreateView("RenderView")
    view.ViewSize = size
    view.Background = [0.965, 0.975, 0.985]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CenterAxesVisibility = 0
    view.UseFXAA = 1

    records: list[dict[str, Any]] = []
    pipelines: list[Any] = []
    primary_pipelines: list[Any] = []
    colored_pipelines: list[tuple[Any, str]] = []
    colored_displays: dict[str, list[Any]] = {}
    geometry_representations: list[Any] = []
    scalar_bars: dict[str, Any] = {}
    timeline: list[float] = []
    scalar_bar_count = 0
    for index, item in enumerate(series):
        path = Path(item["path"]).expanduser().resolve()
        if path.suffix.lower() != ".pvd" or not path.is_file():
            raise ValueError(f"Series {index} must identify an existing .pvd: {path}")
        kind = item["kind"]
        if kind not in {"particles", "surface", "edges", "domain"}:
            raise ValueError(f"Unsupported series kind: {kind!r}")
        reader = pv.PVDReader(registrationName=f"{kind}_{index}", FileName=str(path))
        reader.UpdatePipelineInformation()
        times = _native_times(reader)
        if len(times) > 1:
            if timeline and not _same_times(timeline, times):
                raise ValueError(
                    "Temporal PVD collections must share their native time grid; "
                    f"{path} has different saved timesteps"
                )
            timeline = times
        reader.UpdatePipeline(time=times[-1] if times else 0.0)
        data = reader.GetDataInformation()
        point_arrays = list(reader.PointData.keys())
        source = reader
        if "offset" in item:
            offset = [float(x) for x in item["offset"]]
            if len(offset) != 3 or not all(math.isfinite(x) for x in offset):
                raise ValueError("offset must be three finite coordinates")
            source = pv.Transform(Input=source)
            source.Transform.Translate = offset
        if kind == "particles":
            if "rad" not in point_arrays:
                raise ValueError(f"Particle series {path} has no PointData 'rad' array")
            source = pv.Glyph(Input=source, GlyphType="Sphere")
            source.GlyphType.Radius = 0.5
            source.GlyphType.ThetaResolution = (
                48 if data.GetNumberOfPoints() < 16 else 24
            )
            source.GlyphType.PhiResolution = 32 if data.GetNumberOfPoints() < 16 else 16
            source.OrientationArray = ["POINTS", "No orientation array"]
            source.ScaleArray = ["POINTS", "rad"]
            source.ScaleFactor = 2.0
            source.GlyphMode = "All Points"
        elif kind == "edges":
            source = pv.Tube(Input=source)
            source.Radius = float(item.get("tube_radius", 0.025))
            radius_by = item.get("radius_by")
            if radius_by:
                if radius_by not in point_arrays:
                    raise ValueError(
                        f"Edge series {path} has no radius PointData {radius_by!r}"
                    )
                source.Scalars = ["POINTS", radius_by]
                source.VaryRadius = "By Absolute Scalar"
                source.Radius = 1.0
            source.NumberofSides = 10
            source.Capping = 1
        display = pv.Show(source, view, "GeometryRepresentation")
        geometry_representations.append(display)
        display.Representation = "Wireframe" if kind == "domain" else "Surface"
        display.DiffuseColor = item.get(
            "color", [0.12, 0.48, 0.72] if kind == "particles" else [0.55, 0.62, 0.70]
        )
        display.AmbientColor = display.DiffuseColor
        display.Opacity = float(item.get("opacity", 0.22 if kind == "surface" else 1.0))
        display.LineWidth = 1.5 if kind == "domain" else 2.0
        display.Specular = 0.25 if kind == "particles" else 0.1
        color_by = item.get("color_by")
        if color_by:
            if color_by not in point_arrays:
                raise ValueError(
                    f"Series {path} has no PointData {color_by!r}; "
                    f"available arrays: {point_arrays}"
                )
            pv.ColorBy(display, ("POINTS", color_by))
            lookup = pv.GetColorTransferFunction(color_by)
            lookup.ApplyPreset("Viridis", True)
            display.RescaleTransferFunctionToDataRange(False, True)
            colored_pipelines.append((source, color_by))
            colored_displays.setdefault(color_by, []).append(display)
            # One legend keeps mixed particle/facet scenes readable. A repeated
            # field reuses its lookup table and therefore the same color scale.
            if scalar_bar_count == 0:
                display.SetScalarBarVisibility(view, True)
                bar = pv.GetScalarBar(lookup, view)
                bar.Title = {
                    "vel": "Speed",
                    "rad": "Radius",
                    "clump_id": "Clump",
                    "mass": "Mass",
                    "force": "Force",
                }.get(color_by, color_by.replace("_", " "))
                bar.ComponentTitle = ""
                bar.WindowLocation = "Lower Center"
                bar.Orientation = "Horizontal"
                bar.ScalarBarLength = 0.4
                bar.TitleFontSize = 15
                bar.LabelFontSize = 12
                bar.TitleColor = [0.12, 0.18, 0.25]
                bar.LabelColor = [0.12, 0.18, 0.25]
                scalar_bars[color_by] = bar
                scalar_bar_count += 1
        else:
            display.SetScalarColoring(None, 0)
        pipelines.append(source)
        primary_pipelines.append(source)
        if item.get("vectors"):
            vectors = item["vectors"]
            if vectors not in point_arrays:
                raise ValueError(f"Series {path} has no vector PointData {vectors!r}")
            arrows = pv.Glyph(Input=reader, GlyphType="Arrow")
            arrows.OrientationArray = ["POINTS", vectors]
            arrows.ScaleArray = ["POINTS", vectors]
            arrows.ScaleFactor = float(item.get("vector_scale", 1.0))
            arrows.GlyphMode = "All Points"
            # A requested 2D annotation overlay can lift arrows toward the
            # camera while preserving their projected origins and directions.
            if "offset" in item or "vector_offset" in item:
                base_offset = item.get("offset", [0.0, 0.0, 0.0])
                vector_offset = item.get("vector_offset", [0.0, 0.0, 0.0])
                if len(vector_offset) != 3:
                    raise ValueError("vector_offset must contain three coordinates")
                shifted = pv.Transform(Input=arrows)
                shifted.Transform.Translate = [
                    float(base_offset[axis]) + float(vector_offset[axis])
                    for axis in range(3)
                ]
                arrows = shifted
            arrow_display = pv.Show(arrows, view, "GeometryRepresentation")
            geometry_representations.append(arrow_display)
            arrow_display.DiffuseColor = [0.86, 0.25, 0.13]
            arrow_display.SetScalarColoring(None, 0)
            pipelines.append(arrows)
        records.append(
            {
                "path": str(path),
                "kind": kind,
                "native_times": times,
                "source_points": data.GetNumberOfPoints(),
                "source_cells": data.GetNumberOfCells(),
                "point_arrays": point_arrays,
                "representation": {
                    key: item[key]
                    for key in (
                        "color_by",
                        "color",
                        "opacity",
                        "offset",
                        "vectors",
                        "vector_scale",
                        "vector_offset",
                        "radius_by",
                        "tube_radius",
                    )
                    if key in item
                },
            }
        )

    if not timeline:
        single_times = [x["native_times"][-1] for x in records if x["native_times"]]
        timeline = [max(single_times, default=0.0)]
    # A failed or differently configured regeneration must never leave an old
    # movie, viewer or success record masquerading as current simulation output.
    for name in (
        "scene.png",
        "animation.mp4",
        "scene.vtkjs",
        "scene.html",
        "interactive.html",
        "render.json",
    ):
        (output / name).unlink(missing_ok=True)
    sampled = _sample_times(timeline, maximum)
    scene = pv.GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    bounds = [math.inf, -math.inf, math.inf, -math.inf, math.inf, -math.inf]
    clock = None
    if animate and len(timeline) > 1:
        clock = pv.Text(Text="")
        clock_display = pv.Show(clock, view)
        clock_display.WindowLocation = "Upper Right Corner"
        clock_display.FontSize = 14
        clock_display.Color = [0.12, 0.18, 0.25]

    def update(time: float) -> None:
        scene.AnimationTime = time
        view.ViewTime = time
        if clock is not None:
            clock.Text = f"{time_label} = {time:.4g}"
        for pipeline in pipelines:
            pipeline.UpdatePipeline(time=time)

    # Camera and color scale remain fixed through playback. Inspecting only
    # native saved frames avoids fabricated trajectories or changing zoom.
    ranges: dict[str, tuple[float, float]] = {}
    for time in sampled:
        update(time)
        for pipeline in pipelines:
            information = pipeline.GetDataInformation()
            current = information.GetBounds()
            if _valid_bounds(current):
                for axis in range(3):
                    bounds[2 * axis] = min(bounds[2 * axis], current[2 * axis])
                    bounds[2 * axis + 1] = max(
                        bounds[2 * axis + 1], current[2 * axis + 1]
                    )
        for pipeline, field in colored_pipelines:
            values = pipeline.PointData[field]
            current_range = values.GetRange(
                -1 if values.GetNumberOfComponents() > 1 else 0
            )
            if all(math.isfinite(value) for value in current_range):
                old_range = ranges.get(field, current_range)
                ranges[field] = (
                    min(old_range[0], current_range[0]),
                    max(old_range[1], current_range[1]),
                )
    if not _valid_bounds(bounds):
        raise RuntimeError(
            "ParaView loaded no finite visible geometry from the PVD files"
        )
    center = [(bounds[2 * axis] + bounds[2 * axis + 1]) / 2 for axis in range(3)]
    spans = [bounds[2 * axis + 1] - bounds[2 * axis] for axis in range(3)]
    extent = max(max(spans), 1e-6)
    view.CameraFocalPoint = center
    if camera == "xy":
        view.CameraPosition = [center[0], center[1], center[2] + 3 * extent]
        view.CameraViewUp = [0, 1, 0]
        view.CameraParallelScale = 0.6 * max(
            spans[1], spans[0] * size[1] / size[0], extent * 0.1
        )
    else:
        view.CameraPosition = [
            center[0] + 2 * extent,
            center[1] - 3 * extent,
            center[2] + 2 * extent,
        ]
        view.CameraViewUp = [0, 0, 1]
        view.CameraParallelScale = extent * 0.8
    # Reserve clear space for the clock and scalar-bar title in both camera
    # modes instead of letting fitted grains overlap the lower annotations.
    view.CameraParallelScale *= 1.15
    view.CenterOfRotation = center
    for field, (low, high) in ranges.items():
        lookup = pv.GetColorTransferFunction(field)
        lookup.RescaleTransferFunction(low, high)
        if field == "clump_id":
            first, last = round(low), round(high)
            count = min(last - first + 1, 8)
            labels = (
                [first]
                if count <= 1
                else sorted(
                    {
                        round(first + i * (last - first) / (count - 1))
                        for i in range(count)
                    }
                )
            )
            lookup.Discretize = 1
            lookup.NumberOfTableValues = max(2, min(last - first + 1, 256))
            if field in scalar_bars:
                bar = scalar_bars[field]
                bar.UseCustomLabels = 1
                bar.CustomLabels = labels
                bar.AutomaticLabelFormat = 0
                bar.LabelFormat = "{:.0f}"
                bar.RangeLabelFormat = "{:.0f}"
                bar.AddRangeLabels = 0
        if math.isclose(low, high, rel_tol=1e-5, abs_tol=1e-12):
            for display in colored_displays[field]:
                display.SetScalarColoring(None, 0)
            if field in scalar_bars:
                bar = scalar_bars[field]
                bar.Visibility = 0
                constant = pv.Text(Text=f"{bar.Title} = {low:.3g}")
                constant_display = pv.Show(constant, view)
                constant_display.WindowLocation = "Lower Center"
                constant_display.FontSize = 14
                constant_display.Color = [0.12, 0.18, 0.25]
    update(timeline[-1])
    pv.Render(view)
    view.GetClientSideObject().GetRenderer().ResetCameraClippingRange(bounds)
    for pipeline, record in zip(primary_pipelines, records):
        information = pipeline.GetDataInformation()
        record["geometry_points"] = information.GetNumberOfPoints()
        record["geometry_cells"] = information.GetNumberOfCells()
    pv.SaveScreenshot(str(output / "scene.png"), view, ImageResolution=size)

    animation_times: list[float] = []
    if animate and len(sampled) > 1:
        animation_times = sampled
        with tempfile.TemporaryDirectory(prefix="jaxdem-paraview-frames-") as frame_dir:
            for frame_index, time in enumerate(sampled):
                update(time)
                pv.Render(view)
                pv.SaveScreenshot(
                    str(Path(frame_dir) / f"frame_{frame_index:04d}.png"),
                    view,
                    ImageResolution=size,
                )
            command = [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(Path(frame_dir) / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                str(output / "animation.mp4"),
            ]
            subprocess.run(command, check=True)

    update(timeline[-1])
    interactive_metadata: dict[str, Any] = {}
    if interactive:
        # ParaView 6.2's named-actor exporter assumes every representation is
        # vtkCompositeRepresentation; a scalar bar or Text source crashes it.
        # Glance provides its own field/legend controls, so export only actual
        # geometry, retaining the clock and color bar in the poster and movie.
        view.Representations = geometry_representations
        pv.Render(view)
        exported_fields = sorted(
            {item["color_by"] for item in series if item.get("color_by")}
            | {item["vectors"] for item in series if item.get("vectors")}
            | {item["radius_by"] for item in series if item.get("radius_by")}
            | {
                field
                for field in ("rad", "clump_id")
                if any(field in record["point_arrays"] for record in records)
            }
        )
        # ParaView's exporter selection domain uses actor-qualified names
        # ("Glyph1:vel"), unlike ColorBy's plain array names. Plain names are
        # accepted silently but export no data arrays, disabling Glance fields.
        helper = paraview.servermanager.vtkSMViewExportHelper()
        proxy = helper.CreateExporter(str(output / "scene.vtkjs"), view.SMProxy)
        if proxy is None:
            raise RuntimeError("ParaView could not create the vtkjs scene exporter")
        proxy.UnRegister(None)
        exporter = paraview.servermanager._getPyProxy(proxy)
        exporter.ExportToGlance = 0
        exporter.PointArrays = [
            qualified
            for qualified in exporter.PointArrays.Available
            if qualified.rsplit(":", 1)[-1] in {*exported_fields, "Normals"}
        ]
        exporter.CellArrays = []
        exporter.Write()
        expected_arrays = {
            pipeline.SMProxy.GetLogName(): set(exported_fields).intersection(
                pipeline.PointData.keys()
            )
            for pipeline in pipelines
        }
        if not (output / "scene.vtkjs").is_file():
            raise RuntimeError("ParaView did not produce the requested vtkjs scene")
        try:
            with zipfile.ZipFile(output / "scene.vtkjs") as archive:
                indexes = [
                    name for name in archive.namelist() if name.endswith("index.json")
                ]
                if not indexes:
                    raise ValueError("missing scene index")
                exported_scene = json.loads(archive.read(min(indexes, key=len)))
                if not exported_scene.get("scene"):
                    raise ValueError("the scene contains no visible geometry")
                scene_arrays = {}
                for actor in exported_scene["scene"]:
                    data_path = actor[actor["type"]]["url"] + "/index.json"
                    dataset = json.loads(archive.read(data_path))
                    scene_arrays[actor["name"]] = {
                        entry["data"]["name"]
                        for entry in dataset.get("pointData", {}).get("arrays", [])
                    }
                for actor, required in expected_arrays.items():
                    if actor not in scene_arrays:
                        raise ValueError(f"the scene lost required actor: {actor}")
                    missing = required - scene_arrays[actor]
                    if missing:
                        raise ValueError(
                            f"{actor} lost required point arrays: {sorted(missing)}"
                        )
        except (zipfile.BadZipFile, KeyError, ValueError) as exc:
            raise RuntimeError(
                f"ParaView exported an invalid vtkjs scene: {exc}"
            ) from exc
        interactive_metadata = _shared_glance_viewer(paraview, output.parent)
        interactive_metadata["interactive_point_arrays"] = {
            name: sorted(arrays) for name, arrays in scene_arrays.items()
        }

    version = getattr(paraview, "__version__", None)
    if version is None:
        version = str(pv.GetParaViewVersion())
    metadata = {
        "title": manifest.get("title", "Simulation"),
        "renderer": "ParaView",
        "paraview_version": str(version),
        "manifest": str(manifest_path),
        "size": size,
        "camera": camera,
        "bounds": bounds,
        "poster_time": timeline[-1],
        "time_label": time_label,
        "native_times": timeline,
        "animation_times": animation_times,
        "fps": fps,
        "series": records,
        **interactive_metadata,
        "outputs": {
            name: (output / name).stat().st_size
            for name in (
                "scene.png",
                "animation.mp4",
                "scene.vtkjs",
                "interactive.html",
            )
            if (output / name).is_file()
        },
    }
    (output / "render.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON scene manifest")
    args = parser.parse_args()
    try:
        metadata = render(args.manifest)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"ParaView rendering failed: {exc}\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
