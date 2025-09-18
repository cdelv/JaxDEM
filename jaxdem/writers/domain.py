# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""VTK writer for domain geometry."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import vtk

from . import VTKBaseWriter

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@VTKBaseWriter.register("domain")
class DomainWriter(VTKBaseWriter):
    """Write the simulation domain as a ``vtkCubeSource``."""

    @classmethod
    def write(
        cls,
        state: "State",
        system: "System",
        filename: Path,
        binary: bool,
    ):
        box = system.domain.box_size
        anch = system.domain.anchor

        if box.shape[-1] == 2:
            box = np.pad(box, (*[(0, 0)] * (box.ndim - 1), (0, 1)), "constant")

        if anch.shape[-1] == 2:
            anch = np.pad(anch, (*[(0, 0)] * (anch.ndim - 1), (0, 1)), "constant")

        cube = vtk.vtkCubeSource()
        cube.SetXLength(float(box[0]))
        cube.SetYLength(float(box[1]))
        cube.SetZLength(float(box[2]))
        cube.SetCenter(*(anch + 0.5 * box))
        cube.Update()

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(filename))
        writer.SetInputData(cube.GetOutput())
        if binary:
            writer.SetDataModeToAppended()
            compressor = vtk.vtkZLibDataCompressor()
            writer.SetCompressor(compressor)
        else:
            writer.SetDataModeToAscii()
        ok = writer.Write()
        if ok != 1:
            raise RuntimeError("VTK domain writer failed")


__all__ = ["DomainWriter"]
