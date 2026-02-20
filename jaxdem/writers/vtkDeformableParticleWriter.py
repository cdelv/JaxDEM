# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""VTK writers for deformable particles."""

from __future__ import annotations

import jax

from pathlib import Path
from typing import TYPE_CHECKING, cast
from dataclasses import dataclass
from functools import partial

import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np

from . import VTKBaseWriter

if TYPE_CHECKING:  # pragma: no cover
    from ..bonded_forces import DeformableParticleModel
    from ..state import State
    from ..system import System


def _write_poly(poly: vtk.vtkPolyData, filename: Path, binary: bool) -> None:
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
        raise RuntimeError("VTK deformable particles writer failed")


def _as_points_3d(pos: np.ndarray) -> np.ndarray:
    if pos.shape[-1] == 2:
        return np.pad(pos, ((0, 0), (0, 1)), mode="constant")
    return pos


def _map_unique_ids_to_state_indices(state: State, connectivity: np.ndarray) -> np.ndarray:
    unique_id = np.asarray(state.unique_ID, dtype=np.int64)
    uid_to_idx = {int(uid): i for i, uid in enumerate(unique_id.tolist())}
    if connectivity.size == 0:
        return connectivity.astype(np.int64, copy=False)
    flat = connectivity.reshape(-1)
    mapped = np.asarray([uid_to_idx[int(uid)] for uid in flat], dtype=np.int64)
    return mapped.reshape(connectivity.shape)


def _add_cell_array(poly: vtk.vtkPolyData, name: str, values: np.ndarray) -> None:
    vtk_arr = vtk_np.numpy_to_vtk(values, deep=False)
    vtk_arr.SetName(name)
    poly.GetCellData().AddArray(vtk_arr)


def _compute_element_properties(
    vertices_per_element: np.ndarray, dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dim == 3:
        r1 = vertices_per_element[:, 0, :]
        r2 = vertices_per_element[:, 1, :] - vertices_per_element[:, 0, :]
        r3 = vertices_per_element[:, 2, :] - vertices_per_element[:, 0, :]
        face_normal = np.cross(r2, r3) / 2.0
        measure = np.linalg.norm(face_normal, axis=-1)
        safe = np.where(measure == 0.0, 1.0, measure)
        normals = face_normal / safe[:, None]
        partial_content = np.sum(face_normal * r1, axis=-1) / 3.0
        return normals, measure, partial_content

    r1 = vertices_per_element[:, 0, :2]
    r2 = vertices_per_element[:, 1, :2]
    edge = r2 - r1
    measure = np.linalg.norm(edge, axis=-1)
    safe = np.where(measure == 0.0, 1.0, measure)
    normals_2d = np.stack([edge[:, 1], -edge[:, 0]], axis=1) / safe[:, None]
    normals = np.pad(normals_2d, ((0, 0), (0, 1)), mode="constant")
    partial_content = 0.5 * (r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0])
    return normals, measure, partial_content


def _compute_bendings(
    *,
    dim: int,
    vertices_3d: np.ndarray,
    element_normals: np.ndarray,
    element_adjacency: np.ndarray,
    element_adjacency_edges_idx: np.ndarray,
) -> np.ndarray:
    n1 = element_normals[element_adjacency[:, 0]]
    n2 = element_normals[element_adjacency[:, 1]]
    cos = np.sum(n1 * n2, axis=-1)
    if dim == 3:
        hinge_vec = (
            vertices_3d[element_adjacency_edges_idx[:, 1]]
            - vertices_3d[element_adjacency_edges_idx[:, 0]]
        )
        hinge_norm = np.linalg.norm(hinge_vec, axis=-1)
        safe = np.where(hinge_norm == 0.0, 1.0, hinge_norm)
        tangent = hinge_vec / safe[:, None]
        sin = np.sum(np.cross(n1, n2) * tangent, axis=-1)
    else:
        n1_2d = n1[:, :2]
        n2_2d = n2[:, :2]
        sin = n1_2d[:, 0] * n2_2d[:, 1] - n1_2d[:, 1] * n2_2d[:, 0]
    return np.arctan2(sin, cos)


def _base_poly(state: State) -> tuple[vtk.vtkPolyData, np.ndarray, int]:
    pos = np.asarray(state.pos)
    dim = int(pos.shape[-1])
    if dim not in (2, 3):
        raise ValueError(f"Deformable particles require dim=2 or 3, got {dim}.")
    pos_3d = _as_points_3d(pos)
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(vtk_np.numpy_to_vtk(pos_3d, deep=False))
    poly.SetPoints(points)
    return poly, pos_3d, dim


@VTKBaseWriter.register("deformable_elements")
@dataclass(slots=True)
class VTKDeformableElementsWriter(VTKBaseWriter):
    @classmethod
    @partial(jax.named_call, name="VTKDeformableElementsWriter.write")
    def write(
        cls,
        state: State,
        system: System,
        filename: Path,
        binary: bool,
    ) -> None:
        model: DeformableParticleModel | None = cast("DeformableParticleModel | None", system.bonded_force_model)
        poly, pos_3d, dim = _base_poly(state)
        if model is None or model.elements is None:
            _write_poly(poly, filename, binary)
            return

        elements = np.asarray(model.elements, dtype=np.int64)
        element_idx = _map_unique_ids_to_state_indices(state, elements)
        n_elements = int(elements.shape[0])

        cells = vtk.vtkCellArray()
        verts_per_elem = int(elements.shape[1])
        if verts_per_elem == 3:
            for tri in element_idx:
                cell = vtk.vtkTriangle()
                cell.GetPointIds().SetId(0, int(tri[0]))
                cell.GetPointIds().SetId(1, int(tri[1]))
                cell.GetPointIds().SetId(2, int(tri[2]))
                cells.InsertNextCell(cell)
            poly.SetPolys(cells)
        elif verts_per_elem == 2:
            for seg in element_idx:
                cell = vtk.vtkLine()
                cell.GetPointIds().SetId(0, int(seg[0]))
                cell.GetPointIds().SetId(1, int(seg[1]))
                cells.InsertNextCell(cell)
            poly.SetLines(cells)
        else:
            raise ValueError(
                "Deformable particle elements must have shape (M, 2) or (M, 3). "
                f"Got shape={elements.shape}."
            )

        element_vertices = pos_3d[element_idx]
        normals, current_measures, partial_content = _compute_element_properties(
            element_vertices, dim
        )

        elements_id = (
            np.asarray(model.elements_ID, dtype=np.int32)
            if model.elements_ID is not None
            else np.full((n_elements,), -1, dtype=np.int32)
        )
        ec_by_element = np.full((n_elements,), np.nan, dtype=float)
        if model.ec is not None and np.any(elements_id >= 0):
            ec = np.asarray(model.ec, dtype=float)
            valid = elements_id >= 0
            ec_by_element[valid] = ec[elements_id[valid]]

        gamma = (
            np.asarray(model.gamma, dtype=float)
            if model.gamma is not None
            else np.full((n_elements,), np.nan, dtype=float)
        )
        initial_measures = (
            np.asarray(model.initial_element_measures, dtype=float)
            if model.initial_element_measures is not None
            else np.full((n_elements,), np.nan, dtype=float)
        )

        _add_cell_array(poly, "elements_ID", elements_id)
        _add_cell_array(poly, "ec", ec_by_element)
        _add_cell_array(poly, "gamma", gamma)
        _add_cell_array(poly, "initial_element_measures", initial_measures)
        _add_cell_array(poly, "current_element_measures", current_measures)
        _add_cell_array(poly, "partial_content", partial_content)
        _add_cell_array(poly, "element_normals", normals)

        _write_poly(poly, filename, binary)


@VTKBaseWriter.register("deformable_edge_adjacencies")
@dataclass(slots=True)
class VTKDeformableEdgeAdjacenciesWriter(VTKBaseWriter):
    @classmethod
    @partial(jax.named_call, name="VTKDeformableEdgeAdjacenciesWriter.write")
    def write(
        cls,
        state: State,
        system: System,
        filename: Path,
        binary: bool,
    ) -> None:
        model: DeformableParticleModel | None = cast("DeformableParticleModel | None", system.bonded_force_model)
        poly, pos_3d, dim = _base_poly(state)
        if (
            model is None
            or model.elements is None
            or model.element_adjacency is None
            or model.element_adjacency_edges is None
        ):
            _write_poly(poly, filename, binary)
            return

        element_adjacency = np.asarray(model.element_adjacency, dtype=np.int64)
        adjacency_edges = np.asarray(model.element_adjacency_edges, dtype=np.int64)
        n_adjacency = int(adjacency_edges.shape[0])

        adj_edge_idx = _map_unique_ids_to_state_indices(state, adjacency_edges)
        lines = vtk.vtkCellArray()
        for seg in adj_edge_idx:
            cell = vtk.vtkLine()
            cell.GetPointIds().SetId(0, int(seg[0]))
            cell.GetPointIds().SetId(1, int(seg[1]))
            lines.InsertNextCell(cell)
        poly.SetLines(lines)

        elements = np.asarray(model.elements, dtype=np.int64)
        element_idx = _map_unique_ids_to_state_indices(state, elements)
        element_vertices = pos_3d[element_idx]
        normals, _, _ = _compute_element_properties(element_vertices, dim)
        current_bendings = _compute_bendings(
            dim=dim,
            vertices_3d=pos_3d,
            element_normals=normals,
            element_adjacency=element_adjacency,
            element_adjacency_edges_idx=adj_edge_idx,
        )

        initial_bendings = (
            np.asarray(model.initial_bendings, dtype=float)
            if model.initial_bendings is not None
            else np.full((n_adjacency,), np.nan, dtype=float)
        )
        eb = (
            np.asarray(model.eb, dtype=float)
            if model.eb is not None
            else np.full((n_adjacency,), np.nan, dtype=float)
        )

        _add_cell_array(poly, "initial_bendings", initial_bendings)
        _add_cell_array(poly, "current_bendings", current_bendings)
        _add_cell_array(poly, "eb", eb)

        _write_poly(poly, filename, binary)


@VTKBaseWriter.register("deformable_edges")
@dataclass(slots=True)
class VTKDeformableEdgesWriter(VTKBaseWriter):
    @classmethod
    @partial(jax.named_call, name="VTKDeformableEdgesWriter.write")
    def write(
        cls,
        state: State,
        system: System,
        filename: Path,
        binary: bool,
    ) -> None:
        model: DeformableParticleModel | None = cast("DeformableParticleModel | None", system.bonded_force_model)
        poly, pos_3d, _ = _base_poly(state)
        if model is None or model.edges is None:
            _write_poly(poly, filename, binary)
            return

        edges = np.asarray(model.edges, dtype=np.int64)
        edge_idx = _map_unique_ids_to_state_indices(state, edges)
        n_edges = int(edge_idx.shape[0])

        lines = vtk.vtkCellArray()
        for seg in edge_idx:
            cell = vtk.vtkLine()
            cell.GetPointIds().SetId(0, int(seg[0]))
            cell.GetPointIds().SetId(1, int(seg[1]))
            lines.InsertNextCell(cell)
        poly.SetLines(lines)

        edge_vec = pos_3d[edge_idx[:, 1]] - pos_3d[edge_idx[:, 0]]
        current_lengths = np.linalg.norm(edge_vec, axis=-1)
        initial_lengths = (
            np.asarray(model.initial_edge_lengths, dtype=float)
            if model.initial_edge_lengths is not None
            else np.full((n_edges,), np.nan, dtype=float)
        )
        el = (
            np.asarray(model.el, dtype=float)
            if model.el is not None
            else np.full((n_edges,), np.nan, dtype=float)
        )

        _add_cell_array(poly, "initial_edge_lengths", initial_lengths)
        _add_cell_array(poly, "current_edge_lengths", current_lengths)
        _add_cell_array(poly, "el", el)

        _write_poly(poly, filename, binary)


__all__ = [
    "VTKDeformableElementsWriter",
    "VTKDeformableEdgeAdjacenciesWriter",
    "VTKDeformableEdgesWriter",
]
