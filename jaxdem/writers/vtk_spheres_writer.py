# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""VTK writer that exports spheres data."""

from __future__ import annotations

import jax

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING
from functools import partial

import numpy as np

from . import VTKBaseWriter

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


def _write_sphere_points(
    state: State,
    filename: Path,
    binary: bool,
    mask: np.ndarray,
    label: str,
) -> None:
    """Shared implementation for the sphere-like point writers.

    Writes the masked particle centers as VTK points and attaches the
    per-particle :class:`State` fields (and quaternion components) as
    ``PointData`` arrays. Positions and 2-component vectors are padded
    to 3D as required by VTK.
    """
    import vtk  # type: ignore[import-untyped]
    import vtk.util.numpy_support as vtk_np  # type: ignore[import-untyped]

    pos = np.asarray(state.pos)[mask]
    if pos.shape[-1] == 2:
        pos = np.pad(pos, (*[(0, 0)] * (pos.ndim - 1), (0, 1)), "constant")

    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(vtk_np.numpy_to_vtk(pos, deep=False))
    poly.SetPoints(points)

    for fld in fields(state):
        name = fld.name
        if name == "pos" or name.startswith("_"):
            continue

        arr = getattr(state, name)
        if (
            isinstance(arr, np.ndarray)
            and arr.ndim >= 1
            and arr.shape[0] == mask.shape[0]
        ):
            arr = arr[mask]
            if arr.dtype == np.bool_:
                arr = arr.astype(np.int8)

            if arr.ndim == 2 and arr.shape[1] == 2:
                arr = np.pad(arr, ((0, 0), (0, 1)), "constant")

            vtk_arr = vtk_np.numpy_to_vtk(arr, deep=False)
            vtk_arr.SetName(name)
            poly.GetPointData().AddArray(vtk_arr)

    q_xyz = np.asarray(state.q.xyz)[mask]
    q_w = np.asarray(state.q.w)[mask]
    vtk_arr = vtk_np.numpy_to_vtk(q_xyz, deep=False)
    vtk_arr.SetName("q.xyz")
    poly.GetPointData().AddArray(vtk_arr)
    vtk_arr = vtk_np.numpy_to_vtk(q_w, deep=False)
    vtk_arr.SetName("q.w")
    poly.GetPointData().AddArray(vtk_arr)

    VTKBaseWriter._write_polydata(poly, filename, binary, label=label)


@VTKBaseWriter.register("spheres")
@dataclass(slots=True)
class VTKSpheresWriter(VTKBaseWriter):
    """A :class:`VTKBaseWriter` that writes particle centers as VTK points and
    attaches per-particle :class:`State` fields as ``PointData`` attributes.

    For each particle, its position is written as a point. Relevant per-particle
    fields (e.g., ``vel``, ``rad``, ``mass``) are exported as arrays.
    Positions and 2-component vectors are padded to 3D as required by VTK.
    """

    @classmethod
    def is_active(cls, state: State, system: System) -> bool:
        facet_id = np.asarray(state.facet_id)
        return bool(np.any(facet_id == -1))

    @classmethod
    @partial(jax.named_call, name="VTKSpheresWriter.write")
    def write(
        cls,
        state: State,
        system: System,
        filename: Path,
        binary: bool,
    ) -> None:
        mask = np.asarray(state.facet_id) == -1
        _write_sphere_points(state, filename, binary, mask, "VTK spheres writer")


@VTKBaseWriter.register("facet_spheres")
@dataclass(slots=True)
class VTKFacetSpheresWriter(VTKBaseWriter):
    """A :class:`VTKBaseWriter` that writes facet vertex spheres as VTK points."""

    @classmethod
    def is_active(cls, state: State, system: System) -> bool:
        facet_id = np.asarray(state.facet_id)
        return bool(np.any(facet_id != -1))

    @classmethod
    @partial(jax.named_call, name="VTKFacetSpheresWriter.write")
    def write(
        cls,
        state: State,
        system: System,
        filename: Path,
        binary: bool,
    ) -> None:
        mask = np.asarray(state.facet_id) != -1
        _write_sphere_points(state, filename, binary, mask, "VTK facet spheres writer")


__all__ = ["VTKSpheresWriter", "VTKFacetSpheresWriter"]
