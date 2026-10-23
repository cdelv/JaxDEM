# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""VTK writer for domain geometry."""

from __future__ import annotations

import jax

from pathlib import Path
from typing import TYPE_CHECKING
from dataclasses import dataclass
from functools import partial

import numpy as np

from . import VTKBaseWriter

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@VTKBaseWriter.register("domain")
@dataclass(slots=True)
class VTKDomainWriter(VTKBaseWriter):
    """A :class:`VTKBaseWriter <jaxdem.writers.VTKBaseWriter>` that writes the simulation domain as a VTK geometric
    primitive.

    Orthogonal domains are axis-aligned cuboids (3D) or rectangles (2D).
    A :class:`LeesEdwardsDomain <jaxdem.domains.lees_edwards.LeesEdwardsDomain>`
    uses its current shear strain and flow/gradient axes to draw the sheared
    periodic cell, with its anchor fixed. Two-dimensional input is padded to
    3D as VTK requires.
    """

    @classmethod
    @partial(jax.named_call, name="VTKDomainWriter.write")
    def write(
        cls,
        state: State,
        system: System,
        filename: Path,
        binary: bool,
    ) -> None:
        import vtk  # type: ignore[import-untyped]

        from ..domains.lees_edwards import LeesEdwardsDomain

        box = np.asarray(system.domain.box_size)
        anch = np.asarray(system.domain.anchor)

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

        poly = cube.GetOutput()
        if isinstance(system.domain, LeesEdwardsDomain):
            domain = system.domain
            # Shear about the anchor, matching LeesEdwardsDomain.shear:
            # r_alpha -> r_alpha + gamma * (r_beta - anchor_beta).
            # Transform the normals as well as the points so oblique faces
            # have the correct lighting in ParaView.
            matrix = vtk.vtkMatrix4x4()
            matrix.Identity()
            matrix.SetElement(domain.alpha, domain.beta, float(domain.gamma))
            matrix.SetElement(
                domain.alpha, 3, -float(domain.gamma) * float(anch[domain.beta])
            )
            transform = vtk.vtkTransform()
            transform.SetMatrix(matrix)
            shear = vtk.vtkTransformPolyDataFilter()
            shear.SetInputData(poly)
            shear.SetTransform(transform)
            shear.Update()
            poly = shear.GetOutput()

        cls._write_polydata(poly, filename, binary, label="VTK domain writer")


__all__ = ["VTKDomainWriter"]
