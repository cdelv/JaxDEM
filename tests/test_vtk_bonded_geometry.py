# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Bonded meshes retain their actual geometry across periodic cell boundaries."""

import jax
import jax.numpy as jnp

from dataclasses import replace
from itertools import product
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import jaxdem as jd
from jaxdem.bonded_forces import (
    DeformableParticleModel,
    PlasticDeformableParticleModel,
)


def _read(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def _mesh(dim):
    if dim == 2:
        vertices = np.array([[0.9, 0.9], [1.1, 0.9], [1.1, 1.1], [0.9, 1.1]])
        elements = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        edges = elements.copy()
    else:
        vertices = np.array(
            [[0.9, 0.9, 0.9], [1.1, 0.9, 0.9], [0.9, 1.1, 0.9], [0.9, 0.9, 1.1]]
        )
        elements = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    return vertices, elements, edges


_CASES = [
    (*case, False)
    for case in product(
        (2, 3),
        ("periodic", "leesedwards"),
        ("snapshot", "batch", "trajectory", "shared_batch"),
    )
] + [(2, "leesedwards", "trajectory", True), (3, "periodic", "batch", True)]


@pytest.mark.parametrize("dim,domain,layout,plastic", _CASES)
def test_periodic_bonded_mesh_geometry_is_not_wrapped_per_node(
    tmp_path, dim, domain, layout, plastic
):
    vertices, elements, edges = _mesh(dim)
    model_type = PlasticDeformableParticleModel if plastic else DeformableParticleModel
    model = model_type.Create(
        vertices=vertices, elements=elements, edges=edges, em=1.0, el=1.0
    )
    state = jd.State.create(
        pos=jnp.asarray(vertices),
        rad=jnp.full(len(vertices), 0.025),
        vel=jnp.arange(vertices.size, dtype=float).reshape(vertices.shape),
    )
    domain_kw = {"box_size": jnp.ones(dim)}
    if domain == "leesedwards":
        domain_kw.update(gamma=0.25, gamma_dot=0.1)
    system = jd.System.create(
        state=state,
        collider_type="",
        domain_type=domain,
        domain_kw=domain_kw,
        bonded_force_model=model,
    )
    state_frames = [state]
    system_frames = [system]
    if layout != "snapshot":
        state_frames.append(replace(state, pos_c=state.pos_c + 0.04, vel=state.vel + 3))
        system_frames.append(
            replace(system, step_count=jnp.asarray(7), time=jnp.asarray(0.375))
        )
        state = jd.State.stack(state_frames)
        system = system if layout == "shared_batch" else jd.System.stack(system_frames)
    before = [np.array(leaf, copy=True) for leaf in jax.tree.leaves((state, system))]
    with jd.VTKWriter(
        directory=tmp_path,
        writers=["spheres", "deformable_edges", "deformable_elements"],
        max_workers=1,
    ) as writer:
        writer.save(state, system, trajectory=layout == "trajectory")
    for original, leaf in zip(before, jax.tree.leaves((state, system)), strict=True):
        np.testing.assert_array_equal(leaf, original)

    manifests = sorted(tmp_path.glob("*_deformable_edges.pvd"))
    assert len(manifests) == (2 if layout in {"batch", "shared_batch"} else 1)
    for batch_index, manifest in enumerate(manifests):
        entries = ET.parse(manifest).findall(".//DataSet")
        assert len(entries) == (2 if layout == "trajectory" else 1)
        for time_index, entry in enumerate(entries):
            index = time_index if layout == "trajectory" else batch_index
            expected_state = state_frames[index]
            expected_system = system_frames[0 if layout == "shared_batch" else index]
            assert float(entry.attrib["timestep"]) == float(expected_system.time)
            edge_file = tmp_path / entry.attrib["file"]
            expected_points = np.pad(
                np.asarray(expected_state.pos), ((0, 0), (0, 3 - dim))
            )
            # Every writer must use the same unchanged node coordinates, so
            # the sphere glyphs stay aligned with the edges and surface mesh.
            for kind in ("spheres", "deformable_edges", "deformable_elements"):
                poly = _read(
                    edge_file.with_name(
                        edge_file.name.replace("deformable_edges", kind)
                    )
                )
                np.testing.assert_allclose(
                    vtk_to_numpy(poly.GetPoints().GetData()), expected_points
                )
            poly = _read(edge_file)
            exported_edges = np.array(
                [
                    [poly.GetCell(i).GetPointId(0), poly.GetCell(i).GetPointId(1)]
                    for i in range(poly.GetNumberOfCells())
                ]
            )
            np.testing.assert_array_equal(exported_edges, edges)
            expected_lengths = np.linalg.norm(
                expected_points[edges[:, 1]] - expected_points[edges[:, 0]], axis=-1
            )
            np.testing.assert_allclose(
                vtk_to_numpy(poly.GetCellData().GetArray("current_edge_lengths")),
                expected_lengths,
            )
            # These short edges cross the cell boundary; independent wrapping
            # would change 0.2-unit bonds into 0.8-unit box-spanning chords.
            assert expected_lengths.max() < 0.3
            elements_file = edge_file.with_name(
                edge_file.name.replace("deformable_edges", "deformable_elements")
            )
            poly = _read(elements_file)
            exported_elements = np.array(
                [
                    [poly.GetCell(i).GetPointId(j) for j in range(dim)]
                    for i in range(poly.GetNumberOfCells())
                ]
            )
            np.testing.assert_array_equal(exported_elements, elements)


@pytest.mark.parametrize("domain", ["periodic", "leesedwards"])
def test_unbonded_vtk_particles_still_wrap_into_the_primary_cell(tmp_path, domain):
    state = jd.State.create(pos=jnp.array([[1.1, 0.9]]))
    domain_kw = {"box_size": jnp.ones(2)}
    if domain == "leesedwards":
        domain_kw["gamma"] = 0.25
    system = jd.System.create(
        state=state, collider_type="", domain_type=domain, domain_kw=domain_kw
    )
    with jd.VTKWriter(directory=tmp_path, writers=["spheres"]) as writer:
        writer.save(state, system)
    poly = _read(next(tmp_path.rglob("*.vtp")))
    np.testing.assert_allclose(
        vtk_to_numpy(poly.GetPoints().GetData()), [[0.1, 0.9, 0.0]], atol=1e-6
    )
