# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""VTK writer that exports facets as lines or triangles."""

from __future__ import annotations

import jax

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any
from functools import partial

import numpy as np

from . import VTKBaseWriter

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@VTKBaseWriter.register("facets")
@dataclass(slots=True)
class VTKFacetsWriter(VTKBaseWriter):
    """A :class:`VTKBaseWriter` that writes facets as VTK lines (2D) or triangles (3D)."""

    @classmethod
    @partial(jax.named_call, name="VTKFacetsWriter.write")
    def write(
        cls,
        state: State,
        system: System,
        filename: Path,
        binary: bool,
    ) -> None:
        import vtk  # type: ignore[import-untyped]
        import vtk.util.numpy_support as vtk_np  # type: ignore[import-untyped]

        pos = np.asarray(state.pos)
        pos = np.asarray(state.pos)
        clump_id = np.asarray(state.clump_id)
        dim = state.dim
        facet_id = np.asarray(state.facet_id)

        facet_mask = facet_id != -1

        if not np.any(facet_mask):
            return  # No facets to write

        f_pos = pos[facet_mask]
        f_clump = clump_id[facet_mask]
        f_facet = facet_id[facet_mask]

        thick_arr = np.asarray(state.rad)[facet_mask]

        # Group vertices by facet
        unique_facets, inverse = np.unique(f_facet, return_inverse=True)

        # Fibonacci sphere for 3D Minkowski expansion
        samples = 42
        phi = np.pi * (3.0 - np.sqrt(5.0))
        y_sph = 1 - (np.arange(samples) / float(samples - 1)) * 2
        radius_sph = np.sqrt(1 - y_sph * y_sph)
        theta_sph = phi * np.arange(samples)
        x_sph = np.cos(theta_sph) * radius_sph
        z_sph = np.sin(theta_sph) * radius_sph
        sphere_pts = np.column_stack((x_sph, y_sph, z_sph))

        # Circle for 2D Minkowski expansion
        theta_circ = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        circle_pts = np.column_stack((np.cos(theta_circ), np.sin(theta_circ)))

        new_pos = []
        new_thick = []
        point_source_idx: Any = []
        cells = vtk.vtkCellArray()

        pt_idx = 0
        try:
            from scipy.spatial import ConvexHull  # type: ignore[import-untyped]
        except ImportError:
            ConvexHull = None

        for i in range(len(unique_facets)):
            idx = np.where(inverse == i)[0]

            # Apply minimum image convention to keep facets contiguous in periodic domains
            if type(system.domain).__name__ == "PeriodicDomain":
                box_size = np.asarray(system.domain.box_size)
                # Unwrap relative to the first vertex
                p0_ref = f_pos[idx[0]]
                for j in range(1, len(idx)):
                    dp = f_pos[idx[j]] - p0_ref
                    dp = dp - box_size * np.round(dp / box_size)
                    f_pos[idx[j]] = p0_ref + dp

            if len(idx) == 2 and pos.shape[-1] == 2:
                # 2D line
                p0, p1 = f_pos[idx[0]], f_pos[idx[1]]
                t = (thick_arr[idx[0]] + thick_arr[idx[1]]) / 2.0

                if ConvexHull is not None:
                    R = t / 2.0
                    pts = np.vstack([p0 + circle_pts * R, p1 + circle_pts * R])
                    hull = ConvexHull(pts)
                    ordered_pts = pts[hull.vertices]
                    new_pos.extend(ordered_pts)
                    new_thick.extend([t] * len(ordered_pts))
                    point_source_idx.extend([idx[0]] * len(ordered_pts))

                    poly_cell = vtk.vtkPolygon()
                    poly_cell.GetPointIds().SetNumberOfIds(len(ordered_pts))
                    for j in range(len(ordered_pts)):
                        poly_cell.GetPointIds().SetId(j, pt_idx + j)
                    cells.InsertNextCell(poly_cell)
                    pt_idx += len(ordered_pts)
                else:
                    # Fallback to simple quad
                    dx = p1[0] - p0[0]
                    dy = p1[1] - p0[1]
                    L = np.sqrt(dx * dx + dy * dy) + 1e-12
                    nx, ny = -dy / L, dx / L
                    n = np.array([nx, ny])
                    new_pos.extend(
                        [p0 + n * t / 2, p1 + n * t / 2, p1 - n * t / 2, p0 - n * t / 2]
                    )
                    new_thick.extend([t] * 4)
                    point_source_idx.extend([idx[0]] * 4)
                    quad = vtk.vtkQuad()
                    quad.GetPointIds().SetId(0, pt_idx)
                    quad.GetPointIds().SetId(1, pt_idx + 1)
                    quad.GetPointIds().SetId(2, pt_idx + 2)
                    quad.GetPointIds().SetId(3, pt_idx + 3)
                    cells.InsertNextCell(quad)
                    pt_idx += 4

            elif len(idx) == 3 and pos.shape[-1] == 3:
                # 3D triangle
                p0, p1, p2 = f_pos[idx[0]], f_pos[idx[1]], f_pos[idx[2]]
                t = (thick_arr[idx[0]] + thick_arr[idx[1]] + thick_arr[idx[2]]) / 3.0

                if ConvexHull is not None:
                    R = t / 2.0
                    pts = np.vstack(
                        [p0 + sphere_pts * R, p1 + sphere_pts * R, p2 + sphere_pts * R]
                    )
                    hull = ConvexHull(pts)
                    new_pos.extend(pts)
                    new_thick.extend([t] * len(pts))
                    point_source_idx.extend([idx[0]] * len(pts))

                    for simplex in hull.simplices:
                        tri = vtk.vtkTriangle()
                        tri.GetPointIds().SetId(0, pt_idx + simplex[0])
                        tri.GetPointIds().SetId(1, pt_idx + simplex[1])
                        tri.GetPointIds().SetId(2, pt_idx + simplex[2])
                        cells.InsertNextCell(tri)
                    pt_idx += len(pts)
                else:
                    # Fallback to flat prism
                    v1 = p1 - p0
                    v2 = p2 - p0
                    n = np.cross(v1, v2)
                    n = n / (np.linalg.norm(n) + 1e-12)
                    new_pos.extend(
                        [
                            p0 + n * t / 2,
                            p1 + n * t / 2,
                            p2 + n * t / 2,
                            p0 - n * t / 2,
                            p1 - n * t / 2,
                            p2 - n * t / 2,
                        ]
                    )
                    new_thick.extend([t] * 6)
                    point_source_idx.extend([idx[0]] * 6)

                    tri1 = vtk.vtkTriangle()
                    tri1.GetPointIds().SetId(0, pt_idx)
                    tri1.GetPointIds().SetId(1, pt_idx + 1)
                    tri1.GetPointIds().SetId(2, pt_idx + 2)
                    cells.InsertNextCell(tri1)
                    tri2 = vtk.vtkTriangle()
                    tri2.GetPointIds().SetId(0, pt_idx + 5)
                    tri2.GetPointIds().SetId(1, pt_idx + 4)
                    tri2.GetPointIds().SetId(2, pt_idx + 3)
                    cells.InsertNextCell(tri2)

                    for s_idx in [(0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)]:
                        q = vtk.vtkQuad()
                        q.GetPointIds().SetId(0, pt_idx + s_idx[0])
                        q.GetPointIds().SetId(1, pt_idx + s_idx[1])
                        q.GetPointIds().SetId(2, pt_idx + s_idx[2])
                        q.GetPointIds().SetId(3, pt_idx + s_idx[3])
                        cells.InsertNextCell(q)
                    pt_idx += 6

        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        if len(new_pos) > 0:
            new_pos_arr = np.array(new_pos, dtype=np.float32)
            if new_pos_arr.shape[-1] == 2:
                new_pos_arr = np.pad(new_pos_arr, ((0, 0), (0, 1)), "constant")
            points.SetData(vtk_np.numpy_to_vtk(new_pos_arr, deep=False))
        poly.SetPoints(points)
        poly.SetPolys(cells)

        if len(new_thick) > 0:
            vtk_thick = vtk_np.numpy_to_vtk(
                np.array(new_thick, dtype=np.float32), deep=False
            )
            vtk_thick.SetName("thickness")
            poly.GetPointData().AddArray(vtk_thick)

        # Add other state fields to the facets
        if len(point_source_idx) > 0:
            point_source_idx = np.array(point_source_idx, dtype=int)
            n_total = pos.shape[0]

            for fld in fields(state):
                name = fld.name
                if name in ("pos", "rad", "facet_id") or name.startswith("_"):
                    continue

                arr = getattr(state, name)
                if (
                    isinstance(arr, np.ndarray)
                    and arr.ndim >= 1
                    and arr.shape[0] == n_total
                ):
                    f_arr = arr[facet_mask]
                    if f_arr.dtype == np.bool_:
                        f_arr = f_arr.astype(np.int8)

                    if f_arr.ndim == 2 and f_arr.shape[1] == 2:
                        f_arr = np.pad(f_arr, ((0, 0), (0, 1)), "constant")

                    mapped_arr = f_arr[point_source_idx]
                    vtk_arr = vtk_np.numpy_to_vtk(mapped_arr, deep=False)
                    vtk_arr.SetName(name)
                    poly.GetPointData().AddArray(vtk_arr)

            # Handle quaternions
            if hasattr(state, "q"):
                for comp in ("xyz", "w"):
                    arr = getattr(state.q, comp)
                    if isinstance(arr, np.ndarray) and arr.shape[0] == n_total:
                        f_arr = arr[facet_mask]
                        mapped_arr = f_arr[point_source_idx]
                        vtk_arr = vtk_np.numpy_to_vtk(mapped_arr, deep=False)
                        vtk_arr.SetName(f"q.{comp}")
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
            raise RuntimeError("VTK facets writer failed")


__all__ = ["VTKFacetsWriter"]
