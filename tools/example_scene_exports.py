# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Export current example results for :mod:`tools.render_paraview_scene`.

This adapter never integrates a simulation, renders geometry, or reconstructs
states from position-only observations. Simulation geometry always passes
through ``VTKWriter``. Native VTK adds explicitly derived contact lines,
measured surface triangles, and diagnostic vector fields to that output.

``scene_states`` / ``scene_systems`` are the preferred capture interface.
They may be parallel lists, stacked State/System trajectories, or dictionaries
mapping comparison labels to either representation. A history requires its
actual matching System snapshots. Named ordinary rollout outputs and existing
VTKWriter collections are also recognized. Batch axes remain independent
experiments; they are never interpreted as time.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import shutil
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

_DIAGNOSTICS_ONLY = {
    "checkpoint_guide",
    "collider_guide",
    "custom_modules_guide",
    "force_manager_guide",
    "force_model_guide",
    "infer_contact_stiffness",
    "installation_guide",
    "integrator_guide",
    "materials_guide",
    "pseudo_log_rollout",
    "state_guide",
    "system_guide",
}
_INTERACTIVE = {
    "clump_construction_pipeline",
    "rigid_elastic_viscoplastic_loading",
    "rotating_drum",
    "soft_mode_rearrangement",
}
_EXISTING_WRITERS = {
    "environment_custom_policy",
    "facets_guide",
    "intro_to_rl",
    "introduction",
    "vtk_writer_guide",
}
_AQS = {"soft_mode_rearrangement", "shear_jammed_disks"}
_KINDS = {
    "spheres": "particles",
    "facets": "surface",
    "deformableelements": "surface",
    "deformableedges": "edges",
    "domain": "domain",
}
_COLORS = ([0.12, 0.48, 0.72], [0.88, 0.46, 0.12], [0.23, 0.63, 0.42])
_LOADING_COLORS = {"rigid": "#52657c", "elastic": "#258685", "viscoplastic": "#d07a28"}


def _indices(size: int, maximum: int = 48) -> list[int]:
    return sorted(
        {int(i) for i in np.rint(np.linspace(0, size - 1, min(size, maximum)))}
    )


def _slice(state: Any, system: Any, index: tuple[int, ...]) -> tuple[Any, Any]:
    """Slice both PyTrees together, retaining an explicitly shared scalar System."""
    import jax

    leading = state.pos_c.shape[:-2]
    if system.dt.shape not in ((), leading):
        raise ValueError(
            f"State/System axes differ: {leading} versus {system.dt.shape}"
        )
    selected = jax.tree.map(lambda x: x[index], state) if index else state
    selected_system = (
        jax.tree.map(lambda x: x[index], system) if system.dt.ndim and index else system
    )
    return selected, selected_system


def _frames(states: Any, systems: Any) -> list[tuple[Any, Any]]:
    """Expand one explicitly temporal capture without inventing System leaves."""
    from jaxdem import State, System

    if isinstance(states, (list, tuple)):
        if not isinstance(systems, (list, tuple)) or len(states) != len(systems):
            raise ValueError("Scene histories require equally sized State/System lists")
        frames = list(zip(states, systems, strict=True))
    elif isinstance(states, State) and isinstance(systems, System):
        if states.pos_c.ndim == 2:
            frames = [(states, systems)]
        elif states.pos_c.ndim == 3 and systems.dt.shape == states.pos_c.shape[:-2]:
            frames = [
                _slice(states, systems, (i,)) for i in range(states.pos_c.shape[0])
            ]
        else:
            raise ValueError(
                "A scene history must be (T,N,d) with a matching (T,) System"
            )
    else:
        raise TypeError("Scene captures must contain actual State and System objects")
    if not frames or any(
        not isinstance(state, State)
        or not isinstance(system, System)
        or state.pos_c.ndim != 2
        or system.dt.ndim != 0
        for state, system in frames
    ):
        raise ValueError("Each scene frame must be one complete State/System snapshot")
    return frames


def _batch_groups(state: Any, system: Any, maximum: int = 3) -> list[dict[str, Any]]:
    shape = state.pos_c.shape[:-2]
    if not shape:
        return [{"label": "Final configuration", "frames": [(state, system)]}]
    indices = list(np.ndindex(shape))[:maximum]
    return [
        {"label": f"Replica {index}", "frames": [_slice(state, system, index)]}
        for index in indices
    ]


def _groups(slug: str, namespace: dict[str, Any]) -> list[dict[str, Any]]:
    from jaxdem import State, System

    states, systems = namespace.get("scene_states"), namespace.get("scene_systems")
    if states is not None or systems is not None:
        if isinstance(states, dict):
            if not isinstance(systems, dict) or states.keys() != systems.keys():
                raise ValueError(
                    "Scene comparison dictionaries must have matching labels"
                )
            return [
                {"label": str(label), "frames": _frames(value, systems[label])}
                for label, value in states.items()
            ]
        return [{"label": "Saved trajectory", "frames": _frames(states, systems)}]

    if slug == "packing_analysis":
        data = namespace["analysis"].full
        return [
            {
                "label": "Full packing, including the isolated rattler",
                "frames": [(data.state, data.system)],
            }
        ]
    if slug == "batched_granular_experiments":
        result = namespace["result"]
        fractions = np.asarray(namespace["packing_fractions"])
        seed_count = len(namespace["seeds"])
        return [
            {
                "label": f"phi={phi:g}, seed=0",
                "frames": [_slice(result.state, result.system, (i * seed_count,))],
            }
            for i, phi in enumerate(fractions[:3])
        ]

    for state_name, system_name in (
        ("traj_state", "traj_system"),
        ("state_trajectory", "system_trajectory"),
    ):
        state, system = namespace.get(state_name), namespace.get(system_name)
        if isinstance(state, State) and isinstance(system, System):
            if state.pos_c.ndim == 3:
                return [{"label": "Saved trajectory", "frames": _frames(state, system)}]
            if state.pos_c.ndim == 4 and system.dt.shape == state.pos_c.shape[:-2]:
                return [
                    {
                        "label": f"Replica {b}",
                        "frames": [
                            _slice(state, system, (t, b))
                            for t in range(state.pos_c.shape[0])
                        ],
                    }
                    for b in range(min(state.pos_c.shape[1], 3))
                ]
            raise ValueError(
                "Named rollout State/System axes do not describe a supported trajectory"
            )

    if slug in {"particle_level_friction_calculation", "sasa_calculation"}:
        # These examples construct a geometric body, without a dynamics System.
        # A free export-only System supplies VTKWriter's configuration interface;
        # it neither evolves the body nor fabricates a historical state.
        state = namespace["central"]
        system = System.create(
            state=state, collider_type="naive", rotation_integrator_type=None
        )
        return [
            {
                "label": "Probed rigid clump",
                "frames": [(state, system)],
                "geometry_only_system": True,
            }
        ]

    if slug == "deformable_particle_guide":
        return _batch_groups(namespace["states"], namespace["systems"])

    pairs = (
        [("state", "sim_system")]
        if slug in {"clump_construction_pipeline", "dp_construction_pipeline"}
        else (
            [("state_sim", "system_sim")]
            if slug == "clump_guide"
            else [("state_f", "system_f"), ("state", "system")]
        )
    )
    for state_name, system_name in pairs:
        state, system = namespace.get(state_name), namespace.get(system_name)
        if isinstance(state, State) and isinstance(system, System):
            result = _batch_groups(state, system)
            if slug in {
                "sphere_jamming_rattlers_dynamical_matrix",
                "clump_jamming_dynamical_matrix_and_friction",
            }:
                result[0]["label"] = "Before rattler pruning"
                result.append(
                    {
                        "label": "After rattler pruning",
                        "frames": [(namespace["state_nr"], namespace["system_nr"])],
                    }
                )
            return result
    return []


def _pvd_entries(path: Path) -> list[ET.Element]:
    root = ET.parse(path).getroot()
    if root.attrib.get("type") != "Collection":
        raise ValueError(f"Not a VTK collection: {path}")
    entries = root.findall("./Collection/DataSet")
    if not entries:
        raise ValueError(f"Empty VTK collection: {path}")
    for entry in entries:
        if not np.isfinite(float(entry.attrib["timestep"])):
            raise ValueError(f"Nonfinite collection time: {path}")
        if not (path.parent / entry.attrib["file"]).is_file():
            raise FileNotFoundError(path.parent / entry.attrib["file"])
    return entries


def _kind(path: Path) -> str | None:
    writer = re.sub(r"^batch_\d+_", "", path.stem)
    return _KINDS.get(re.sub(r"[^a-z0-9]", "", writer.lower()))


def _series(
    directory: Path, label: str, color: list[float], color_by: str | None
) -> list[dict[str, Any]]:
    result = []
    for path in sorted(directory.glob("*.pvd")):
        kind = _kind(path)
        if kind is None:
            continue
        _pvd_entries(path)
        item: dict[str, Any] = {
            "path": str(path.resolve()),
            "kind": kind,
            "label": label,
        }
        if kind == "particles":
            item["color"] = color
            if color_by:
                item["color_by"] = color_by
        elif kind == "surface":
            item.update(color=color, opacity=0.38)
        elif kind == "edges":
            item.update(color=[0.22, 0.30, 0.36], tube_radius=0.012)
        result.append(item)
    return result


def _read_poly(path: Path) -> Any:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    if reader.GetErrorCode():
        raise RuntimeError(f"VTK could not read {path}")
    poly = reader.GetOutput()
    if poly is None:
        raise RuntimeError(f"VTK read no polydata from {path}")
    if poly.GetNumberOfPoints() and not np.all(
        np.isfinite(vtk_to_numpy(poly.GetPoints().GetData()))
    ):
        raise ValueError(f"Nonfinite physical VTK point coordinates in {path}")
    return poly


def _display_bounds(series: list[dict[str, Any]]) -> tuple[float, float]:
    """Include physical glyph radii, moving surfaces, and the actual domain."""
    low, high = np.inf, -np.inf
    for item in series:
        path = Path(item["path"])
        for entry in _pvd_entries(path):
            poly = _read_poly(path.parent / entry.attrib["file"])
            if not poly.GetNumberOfPoints():
                continue
            bounds = poly.GetBounds()
            radius = (
                poly.GetPointData().GetArray("rad")
                if item["kind"] == "particles"
                else None
            )
            padding = float(radius.GetRange()[1]) if radius is not None else 0.0
            low = min(low, bounds[0] - padding)
            high = max(high, bounds[1] + padding)
    if not np.isfinite(low + high):
        raise RuntimeError("VTKWriter exported no finite display geometry")
    return float(low), float(high)


def _write_poly(path: Path, poly: Any) -> None:
    import vtk

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"VTK could not write {path}")


def _add_fields(path: Path, values: dict[str, np.ndarray]) -> None:
    from vtk.util.numpy_support import numpy_to_vtk

    poly = _read_poly(path)
    for name, values_array in values.items():
        values_array = np.asarray(values_array)
        if values_array.shape[0] != poly.GetNumberOfPoints():
            raise ValueError(
                f"Derived {name!r} does not match the exported points in {path}"
            )
        if values_array.ndim == 2 and values_array.shape[1] == 2:
            values_array = np.pad(values_array, ((0, 0), (0, 1)))
        if not np.all(np.isfinite(values_array)):
            raise ValueError(f"Nonfinite derived {name!r} in {path}")
        array = numpy_to_vtk(np.ascontiguousarray(values_array), deep=True)
        array.SetName(name)
        poly.GetPointData().AddArray(array)
    _write_poly(path, poly)


def _derived_modes(
    frames: list[tuple[Any, Any]], namespace: dict[str, Any]
) -> list[dict[str, np.ndarray]]:
    """Reevaluate each equilibrium's actual mode; align only its arbitrary sign."""
    spectrum = namespace.get("spectrum")
    if not callable(spectrum):
        raise ValueError("Soft-mode export requires the example's spectrum function")
    fields = []
    previous_mode = None
    previous_state = previous_system = None
    for state, system in frames:
        _, _, flat_mode = spectrum(state, system)
        flat_mode = np.asarray(flat_mode)
        if previous_mode is not None and np.dot(flat_mode, previous_mode) < 0:
            flat_mode = -flat_mode
        previous_mode = flat_mode
        mode = flat_mode.reshape(state.pos_c.shape)
        np.testing.assert_allclose(np.linalg.norm(mode), 1.0, rtol=1e-6, atol=1e-8)
        displacement = np.zeros_like(mode)
        if previous_state is not None:
            affine = np.asarray(previous_state.pos).copy()
            domain = system.domain
            dgamma = float(domain.gamma - previous_system.domain.gamma)
            affine[:, domain.alpha] += dgamma * (
                np.asarray(previous_state.pos_c)[:, domain.beta]
                - float(previous_system.domain.anchor[domain.beta])
            )
            displacement = np.asarray(domain.displacement(state.pos, affine, system))
            displacement = displacement - displacement.mean(axis=0)
        fields.append({"soft_mode": mode, "nonaffine_displacement": displacement})
        previous_state, previous_system = state, system
    return fields


def _write_groups(
    slug: str, groups: list[dict[str, Any]], namespace: dict[str, Any], directory: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    import jax.numpy as jnp
    from jaxdem import VTKWriter
    from jaxdem.domains.free import FreeDomain

    series = []
    records = []
    dimension = 2
    cursor = 0.0
    time_grids = []
    for index, group in enumerate(groups):
        all_frames = group["frames"]
        selected = _indices(len(all_frames))
        frames = [all_frames[i] for i in selected]
        dimension = max(dimension, frames[0][0].pos_c.shape[-1])
        times = [
            float(system.domain.gamma if slug in _AQS else system.time)
            for _, system in frames
        ]
        if len(times) > 1 and not np.all(np.diff(times) > 0):
            raise ValueError(
                f"{slug}: saved history times must increase; batches are not trajectories"
            )
        if len(times) > 1:
            time_grids.append(times)
        export_dir = directory / f"group_{index:02d}"
        export_dir.mkdir(parents=True, exist_ok=True)
        writers = ["spheres", "facets", "deformable_elements", "deformable_edges"]
        if not isinstance(frames[0][1].domain, FreeDomain):
            writers.append("domain")
        with VTKWriter(directory=export_dir, writers=writers, max_workers=1) as writer:
            for frame_number, ((state, system), time) in enumerate(
                zip(frames, times, strict=True)
            ):
                if slug in _AQS:
                    # Unique file indices and strain coordinates are export labels,
                    # not fictitious physical integration steps or physical times.
                    system = replace(
                        system,
                        time=jnp.asarray(time),
                        step_count=jnp.asarray(frame_number),
                    )
                writer.save(state, system)

        for path in export_dir.glob("*.pvd"):
            entries = _pvd_entries(path)
            if len(entries) != len(frames) or not np.allclose(
                [float(entry.attrib["timestep"]) for entry in entries],
                times,
                rtol=1e-9,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"VTKWriter dropped or relabeled a captured frame: {path}"
                )

        color_by = namespace.get("scene_color_by", "rad")
        if slug == "packing_analysis":
            color_by = "clump_id"
        if slug == "rigid_elastic_viscoplastic_loading":
            color_by = None
        if slug == "rotating_drum":
            color_by = "vel"
        if slug == "soft_mode_rearrangement":
            fields = _derived_modes(frames, namespace)
            sphere_pvd = next(
                path for path in export_dir.glob("*.pvd") if _kind(path) == "particles"
            )
            entries = _pvd_entries(sphere_pvd)
            if len(entries) != len(fields):
                raise ValueError("VTKWriter dropped a soft-mode snapshot")
            for entry, values in zip(entries, fields, strict=True):
                _add_fields(sphere_pvd.parent / entry.attrib["file"], values)
            color_by = "nonaffine_displacement"

        color = list(_COLORS[index % len(_COLORS)])
        if (
            slug == "rigid_elastic_viscoplastic_loading"
            and group["label"] in _LOADING_COLORS
        ):
            hexadecimal = _LOADING_COLORS[group["label"]]
            color = [int(hexadecimal[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]
        group_series = _series(export_dir, group["label"], color, color_by)
        if slug == "rotating_drum":
            for item in group_series:
                if item["kind"] == "surface":
                    item["opacity"] = 0.08
        if namespace.get("scene_vectors"):
            for item in group_series:
                if item["kind"] == "particles":
                    item.update(
                        vectors=str(namespace["scene_vectors"]),
                        vector_scale=float(namespace.get("scene_vector_scale", 1.0)),
                    )
        if slug == "soft_mode_rearrangement":
            for item in group_series:
                if item["kind"] == "particles":
                    item.update(vectors="soft_mode", vector_scale=2.0)
                    if all(state.pos_c.shape[-1] == 2 for state, _ in frames):
                        # Move only the rendered XY projection layer, keeping
                        # native VTK particle positions and vectors unchanged.
                        max_radius = max(
                            float(np.asarray(state.rad).max(initial=0.0))
                            for state, _ in frames
                        )
                        item["vector_offset"] = [0.0, 0.0, 1.1 * max_radius]

        minimum, maximum = _display_bounds(group_series)
        span = max(maximum - minimum, 1e-3)
        offset = [cursor - minimum, 0.0, 0.0] if len(groups) > 1 else [0.0, 0.0, 0.0]
        if len(groups) > 1:
            for item in group_series:
                item["offset"] = offset
            cursor += span * 1.3
        series.extend(group_series)
        records.append(
            {
                "label": group["label"],
                "source_frames": len(all_frames),
                "exported_frames": len(frames),
                "selected_indices": selected,
                "particle_rows": [int(state.pos_c.shape[0]) for state, _ in frames],
                "source_step_counts": [int(system.step_count) for _, system in frames],
                "source_times": [float(system.time) for _, system in frames],
                "collection_times": times,
                "offset": offset,
                "geometry_only_system": group.get("geometry_only_system", False),
            }
        )
    if time_grids and any(
        len(grid) != len(time_grids[0])
        or not np.allclose(grid, time_grids[0], rtol=1e-9, atol=1e-12)
        for grid in time_grids[1:]
    ):
        raise ValueError("Comparison histories must share the same native time grid")
    return series, records, dimension


def _reuse_writer(
    namespace: dict[str, Any], directory: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    from jaxdem import VTKWriter
    from jaxdem.domains.free import FreeDomain

    writer = namespace.get("writer")
    if not isinstance(writer, VTKWriter):
        return None
    writer.block_until_ready()
    source_root = writer.directory.resolve()
    paths = sorted(source_root.glob("*.pvd"))
    if not paths:
        return None
    batch_names = sorted(
        {
            re.match(r"batch_\d+", path.name).group(0)
            for path in paths
            if re.match(r"batch_\d+", path.name)
        }
    )[:3]
    records = []
    series = []
    cursor = 0.0
    for index, batch in enumerate(batch_names):
        destination = directory / f"group_{index:02d}"
        destination.mkdir(parents=True, exist_ok=True)
        for path in paths:
            if not path.name.startswith(batch + "_") or _kind(path) is None:
                continue
            if _kind(path) == "domain":
                candidate = namespace.get("system")
                if candidate is None:
                    candidate = getattr(namespace.get("env"), "system", None)
                if candidate is not None and isinstance(candidate.domain, FreeDomain):
                    continue
            entries = _pvd_entries(path)
            chosen = [entries[i] for i in _indices(len(entries))]
            root = ET.Element("VTKFile", type="Collection", version="0.1")
            collection = ET.SubElement(root, "Collection")
            for entry in chosen:
                source = (source_root / entry.attrib["file"]).resolve()
                relative = source.relative_to(source_root)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                ET.SubElement(collection, "DataSet", dict(entry.attrib))
            ET.ElementTree(root).write(
                destination / path.name, encoding="utf-8", xml_declaration=True
            )
        group_series = _series(
            destination, f"Exported replica {index}", list(_COLORS[index]), "rad"
        )
        if len(batch_names) > 1:
            # Preserve source geometry and separate independent environments only
            # in the display pipeline, never by editing their exported states.
            minimum, maximum = _display_bounds(group_series)
            for item in group_series:
                item["offset"] = [cursor - minimum, 0.0, 0.0]
            cursor += 1.3 * max(maximum - minimum, 1e-3)
        series.extend(group_series)
        if group_series:
            records.append(
                {
                    "label": f"Exported replica {index}",
                    "reused_writer_output": True,
                    "source_directory": str(source_root),
                }
            )
    return (series, records) if series else None


def _auxiliary(
    directory: Path,
    name: str,
    points: np.ndarray,
    cells: np.ndarray,
    *,
    lines: bool,
    time: float,
    fields: dict[str, np.ndarray] | None = None,
) -> Path:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    directory.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=float)
    if points.shape[1] == 2:
        points = np.pad(points, ((0, 0), (0, 1)))
    if not np.all(np.isfinite(points)):
        raise ValueError("Auxiliary geometry contains nonfinite coordinates")
    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    poly.SetPoints(vtk_points)
    vtk_cells = vtk.vtkCellArray()
    for cell in cells:
        ids = vtk.vtkIdList()
        for point_id in cell:
            ids.InsertNextId(int(point_id))
        vtk_cells.InsertNextCell(ids)
    if lines:
        poly.SetLines(vtk_cells)
    else:
        poly.SetPolys(vtk_cells)
    path = directory / f"{name}.vtp"
    _write_poly(path, poly)
    if fields:
        _add_fields(path, fields)
    root = ET.Element("VTKFile", type="Collection", version="0.1")
    collection = ET.SubElement(root, "Collection")
    ET.SubElement(collection, "DataSet", timestep=format(time, ".12g"), file=path.name)
    pvd = directory / f"{name}.pvd"
    ET.ElementTree(root).write(pvd, encoding="utf-8", xml_declaration=True)
    return pvd


def _packing_edges(namespace: dict[str, Any], directory: Path) -> list[dict[str, Any]]:
    data = namespace["analysis"].full
    network = data.clump_contacts
    pairs = np.asarray(network.pair_ids)
    unique = pairs[:, 0] < pairs[:, 1]
    species = (
        np.asarray(network.pair_species)
        if network.pair_species is not None
        else np.zeros_like(pairs)
    )
    starts = np.asarray(data.clump_positions)[pairs[:, 0]]
    ends = starts - np.asarray(network.displacements)
    force = np.linalg.norm(np.asarray(network.forces), axis=-1)
    max_force = max(float(force.max(initial=0.0)), 1e-12)
    result = []
    for same, name, color in (
        (True, "within_species", [0.12, 0.48, 0.72]),
        (False, "between_species", [0.88, 0.46, 0.12]),
    ):
        selected = unique & ((species[:, 0] == species[:, 1]) == same)
        if not np.any(selected):
            continue
        points = np.stack((starts[selected], ends[selected]), axis=1).reshape(
            -1, starts.shape[-1]
        )
        count = int(selected.sum())
        pvd = _auxiliary(
            directory,
            name,
            points,
            np.arange(2 * count).reshape(-1, 2),
            lines=True,
            time=float(data.system.time),
            fields={
                "force_magnitude": np.repeat(force[selected], 2),
                "tube_radius": np.repeat(
                    0.025 * (1.0 + 3.0 * force[selected] / max_force), 2
                ),
            },
        )
        result.append(
            {
                "path": str(pvd.resolve()),
                "kind": "edges",
                "color": color,
                "radius_by": "tube_radius",
                "label": name.replace("_", " "),
            }
        )
        if data.state.pos_c.shape[-1] == 2:
            # Native segments stay in the particle plane; their display layer
            # sits in front of the largest sphere for a readable projection.
            max_radius = float(np.asarray(data.state.rad).max(initial=0.0))
            result[-1]["offset"] = [0.0, 0.0, 1.1 * max_radius]
    return result


def _metrics(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from vtk.util.numpy_support import vtk_to_numpy

    records = []
    for item in series:
        path = Path(item["path"])
        entries = _pvd_entries(path)
        frames = []
        for entry in entries:
            source = path.parent / entry.attrib["file"]
            poly = _read_poly(source)
            physical: dict[str, Any] = {}
            for field in ("rad", "vel", "force"):
                vtk_array = poly.GetPointData().GetArray(field)
                if vtk_array is None:
                    continue
                values = vtk_to_numpy(vtk_array)
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Nonfinite physical VTK field {field!r} in {source}"
                    )
                if not values.size:
                    continue
                if field == "rad":
                    # Facet thickness can legitimately be zero. Particle glyph
                    # radii represent physical spheres and must be positive.
                    invalid = values <= 0 if item["kind"] == "particles" else values < 0
                    if np.any(invalid):
                        raise ValueError(f"Invalid physical VTK radii in {source}")
                    physical["radius_min"] = float(values.min())
                    physical["radius_max"] = float(values.max())
                else:
                    vectors = values.reshape(poly.GetNumberOfPoints(), -1)
                    physical["speed_max" if field == "vel" else "force_norm_max"] = (
                        float(np.linalg.norm(vectors, axis=-1).max())
                    )
            frames.append(
                {
                    "time": float(entry.attrib["timestep"]),
                    "points": poly.GetNumberOfPoints(),
                    "cells": poly.GetNumberOfCells(),
                    "bounds": list(poly.GetBounds()),
                    "bytes": source.stat().st_size,
                    "physical_summary": physical,
                }
            )
        if item["kind"] == "particles" and any(
            frame["points"] == 0 for frame in frames
        ):
            raise RuntimeError(f"VTKWriter exported an empty particle frame: {path}")
        records.append(
            {
                "label": item.get("label"),
                "kind": item["kind"],
                "path": str(path),
                "point_arrays": [
                    poly.GetPointData().GetArrayName(i)
                    for i in range(poly.GetPointData().GetNumberOfArrays())
                ],
                "frames": frames,
            }
        )
    return records


def export_example(
    slug: str, namespace: dict[str, Any], work_dir: Path, output_dir: Path
) -> dict[str, Any] | None:
    """Return a renderer manifest for current results, or None for diagnostics only.

    Both directories are per-example. Files under ``work_dir / 'vtk'`` are
    newly exported on every call; cached images or VTK from earlier builds
    are never used. An existing VTKWriter owned by this namespace may supply
    its freshly written collections, copied before its temporary scope ends.
    """
    slug = Path(slug).stem
    if slug in _DIAGNOSTICS_ONLY:
        return None
    work_dir = Path(work_dir).resolve()
    output_dir = Path(output_dir).resolve()
    directory = work_dir / "vtk"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    reused = _reuse_writer(namespace, directory) if slug in _EXISTING_WRITERS else None
    if reused is not None:
        series, groups = reused
        source_state = namespace.get("state")
        if source_state is None:
            source_state = getattr(namespace.get("env"), "state", None)
        dimension = int(source_state.pos_c.shape[-1]) if source_state is not None else 2
    else:
        captures = _groups(slug, namespace)
        if not captures:
            return None
        series, groups, dimension = _write_groups(slug, captures, namespace, directory)
    if slug == "packing_analysis":
        series.extend(_packing_edges(namespace, directory / "contacts"))
    if slug == "sasa_calculation":
        central = namespace["central"]
        points = (
            np.asarray(central.pos_c)[0]
            + np.asarray(namespace["directions"])
            * np.asarray(namespace["separation"])[:, None]
        )
        path = _auxiliary(
            directory / "probe_surface",
            "accessible_surface",
            points,
            np.asarray(namespace["triangles"]),
            lines=False,
            time=0.0,
        )
        series.append(
            {
                "path": str(path.resolve()),
                "kind": "surface",
                "color": [0.76, 0.46, 0.16],
                "opacity": 0.2,
                "label": "Measured tracer-center surface",
            }
        )
    if not series:
        return None
    metrics = _metrics(series)
    animate = any(len(record["frames"]) > 1 for record in metrics)
    frame_count = max(len(record["frames"]) for record in metrics)
    manifest = {
        "title": str(namespace.get("scene_title", slug.replace("_", " ").capitalize())),
        "output": str(output_dir),
        "size": [960, 600],
        "max_frames": 48,
        "fps": max(1, min(8, frame_count // 2)) if animate else 8,
        "camera": namespace.get(
            "scene_camera",
            "xy" if dimension == 2 or slug == "rotating_drum" else "isometric",
        ),
        "animate": animate,
        "interactive": slug in _INTERACTIVE,
        "time_label": (
            "strain gamma"
            if slug in _AQS
            else str(namespace.get("scene_time_label", "t"))
        ),
        "series": series,
        "export_metadata": {
            "slug": slug,
            "source": "Current example State/System snapshots via VTKWriter; derived diagnostics via native VTK",
            "groups": groups,
            "geometry": metrics,
        },
    }
    manifest["caption"] = "ParaView rendering of this run's VTKWriter output."
    manifest["animation_caption"] = (
        "Saved native states at equally spaced video intervals; "
        "the clock shows their recorded physical time."
    )
    if slug in _AQS:
        manifest["animation_caption"] = (
            "Accepted equilibria in strain order. Playback speed represents "
            "neither a physical loading rate nor minimizer time."
        )
    if slug == "packing_analysis":
        manifest["caption"] = (
            "Particles are colored by clump ID. Blue network edges connect "
            "equal analysis species; orange edges connect different species. "
            "Contact tubes use radii 0.025*(1+3F/Fmax) and are projected in a "
            "layer in front of the spheres for visibility; native VTK "
            "coordinates remain in the particle plane."
        )
    elif slug == "soft_mode_rearrangement":
        manifest["caption"] = (
            "Colors show nonaffine displacement from the preceding saved "
            "equilibrium. Arrows show each equilibrium's unit-norm soft "
            "eigenmode with sign aligned and a fixed display scale. In this "
            "planar scene the arrow glyphs are projected in front of the "
            "spheres; native particle positions and vector data are unchanged."
        )
    elif slug == "rotating_drum":
        manifest["caption"] = (
            "Grain colors show instantaneous speed on a fixed scale. "
            "The imported facet walls are transparent to expose the bed; "
            "their simulation geometry and contacts are unchanged."
        )
    elif slug == "rigid_elastic_viscoplastic_loading":
        manifest["caption"] = (
            "Rigid grains use slate blue, elastic grains teal, and viscoplastic "
            "grains orange, matching the diagnostic plots. Display "
            "offsets separate the three experiments; each mesh and node field "
            "comes from that model's matching State/System snapshot."
        )
    (work_dir / "export.json").write_text(
        json.dumps(manifest["export_metadata"], indent=2) + "\n"
    )
    return manifest
