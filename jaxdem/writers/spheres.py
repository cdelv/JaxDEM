# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""VTK writer that exports particle-centric data."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np

from . import VTKBaseWriter

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@VTKBaseWriter.register("spheres")
class SpheresWriter(VTKBaseWriter):
    """Write particle centers and per-particle data to ``.vtp`` files."""

    @classmethod
    def write(
        cls,
        state: "State",
        system: "System",
        filename: Path,
        binary: bool,
    ):
        pos = state.pos
        n = pos.shape[0]
        if pos.shape[-1] == 2:
            pos = np.pad(pos, (*[(0, 0)] * (pos.ndim - 1), (0, 1)), "constant")

        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(vtk_np.numpy_to_vtk(pos, deep=False))
        poly.SetPoints(points)

        for fld in fields(state):
            name = fld.name
            if name == "pos":
                continue
            arr = getattr(state, name)
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == n:
                if arr.dtype == np.bool_:
                    arr = arr.astype(np.int8)

                if arr.ndim == 2 and arr.shape[1] == 2:
                    arr = np.pad(arr, ((0, 0), (0, 1)), "constant")

                vtk_arr = vtk_np.numpy_to_vtk(arr, deep=False)
                vtk_arr.SetName(name)
                poly.GetPointData().AddArray(vtk_arr)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(filename))
        writer.SetInputData(poly)
        if binary:
            writer.SetDataModeToAppended()
            compressor = vtk.vtkZLibDataCompressor()
            writer.SetCompressor(compressor)
        else:
            writer.SetDataModeToAscii()
        ok = writer.Write()
        if ok != 1:
            raise RuntimeError("VTK spheres writer failed")


__all__ = ["SpheresWriter"]
