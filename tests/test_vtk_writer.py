# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""VTK writer tests: bonded meshes, physical field ownership, trajectories, and domains."""

import jax
import jax.numpy as jnp

from dataclasses import replace
from itertools import product
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import scipy.spatial
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import jaxdem as jd
from jaxdem.bonded_forces import (
    DeformableParticleModel,
    PlasticDeformableParticleModel,
)


def _read_poly(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def _bonded_mesh(dim):
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


@pytest.mark.parametrize(
    "dim,domain,layout,plastic",
    [
        (2, "periodic", "snapshot", False),
        (3, "periodic", "batch", True),
        (2, "leesedwards", "trajectory", True),
        (3, "leesedwards", "shared_batch", False),
    ],
)
def test_periodic_bonded_mesh_geometry_is_not_wrapped_per_node(
    tmp_path, dim, domain, layout, plastic
):
    vertices, elements, edges = _bonded_mesh(dim)
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
            for kind in ("spheres", "deformable_edges", "deformable_elements"):
                poly = _read_poly(
                    edge_file.with_name(
                        edge_file.name.replace("deformable_edges", kind)
                    )
                )
                np.testing.assert_allclose(
                    vtk_to_numpy(poly.GetPoints().GetData()), expected_points
                )
            poly = _read_poly(edge_file)
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
            assert expected_lengths.max() < 0.3
            elements_file = edge_file.with_name(
                edge_file.name.replace("deformable_edges", "deformable_elements")
            )
            poly = _read_poly(elements_file)
            exported_elements = np.array(
                [
                    [poly.GetCell(i).GetPointId(j) for j in range(dim)]
                    for i in range(poly.GetNumberOfCells())
                ]
            )
            np.testing.assert_array_equal(exported_elements, elements)


@pytest.mark.parametrize(
    "domain,expected",
    [
        ("periodic", [[0.1, 0.1, 0.0]]),
        # Crossing the shear-gradient boundary (y=1.1 -> image +1) shifts x by -gamma=-0.25.
        ("leesedwards", [[0.85, 0.1, 0.0]]),
    ],
)
def test_unbonded_vtk_particles_still_wrap_into_the_primary_cell(
    tmp_path, domain, expected
):
    state = jd.State.create(pos=jnp.array([[1.1, 1.1]]))
    domain_kw = {"box_size": jnp.ones(2)}
    if domain == "leesedwards":
        domain_kw["gamma"] = 0.25
    system = jd.System.create(
        state=state, collider_type="", domain_type=domain, domain_kw=domain_kw
    )
    with jd.VTKWriter(directory=tmp_path, writers=["spheres"]) as writer:
        writer.save(state, system)
    poly = _read_poly(next(tmp_path.rglob("*.vtp")))
    np.testing.assert_allclose(
        vtk_to_numpy(poly.GetPoints().GetData()), expected, atol=1e-6
    )


@pytest.mark.parametrize(
    "dim,hull,scale",
    [(2, True, 1.0), (2, False, 1e-12), (3, True, 1e-12), (3, False, 1.0)],
)
def test_facet_points_inherit_their_own_vertex_fields(
    tmp_path, monkeypatch, dim, hull, scale
):
    vertices = np.array([[9.8, 9.8], [0.8, 9.8]])
    if dim == 3:
        vertices = np.column_stack((np.vstack((vertices, [9.8, 0.8])), np.ones(3)))
    state = jd.State.add_facet(
        jd.State.create(dim=dim),
        jnp.asarray(vertices * scale),
        thickness=0.2 * scale,
        rigid=False,
    )
    velocity = np.arange(1, dim * dim + 1).reshape(dim, dim).astype(float)
    state = replace(state, vel=jnp.asarray(velocity), force=jnp.asarray(-2 * velocity))
    system = jd.System.create(
        state=state,
        collider_type="",
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(dim, 10.0 * scale)},
    )
    if not hull:
        monkeypatch.setattr(scipy.spatial, "ConvexHull", None)
    with jd.VTKWriter(directory=tmp_path, writers=["facets"]) as writer:
        writer.save(state, system)
    poly = _read_poly(next(tmp_path.rglob("*.vtp")))
    points = vtk_to_numpy(poly.GetPoints().GetData())[:, :dim] / scale
    unwrapped = vertices.copy()
    unwrapped[1, 0] += 10
    if dim == 3:
        unwrapped[2, 1] += 10
    distances = np.linalg.norm(points[:, None] - unwrapped[None], axis=-1)
    source = distances.argmin(axis=1)
    np.testing.assert_array_equal(np.unique(source), np.arange(dim))
    np.testing.assert_allclose(distances.min(axis=1), 0.2, rtol=2e-5)
    for name, values in (("vel", velocity), ("force", -2 * velocity)):
        expected = np.pad(values, ((0, 0), (0, 3 - dim)))[source]
        np.testing.assert_array_equal(
            vtk_to_numpy(poly.GetPointData().GetArray(name)), expected
        )


@pytest.mark.parametrize("time_axis", [0, 1])
def test_vtk_trajectory_preserves_batch_fields_and_file_times(tmp_path, time_axis):
    states, systems = [], []
    for t in range(2):
        frames, configurations = [], []
        for b in range(2):
            state = jd.State.create(
                pos=jnp.array([[100.0 * b + 10 * t, 0.0]]),
                vel=jnp.array([[b + 1.0, t + 2.0]]),
                mass=jnp.array([b + t + 1.0]),
            )
            system = jd.System.create(state=state, collider_type="")
            configurations.append(
                replace(
                    system,
                    step_count=jnp.array(10 + t),
                    time=jnp.array(0.125 + 0.25 * t),
                )
            )
            frames.append(state)
        states.append(jd.State.stack(frames))
        systems.append(jd.System.stack(configurations))
    state, system = jd.State.stack(states), jd.System.stack(systems)
    if time_axis == 1:
        state, system = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), (state, system))
    with jd.VTKWriter(directory=tmp_path, writers=["spheres"]) as writer:
        writer.save(state, system, trajectory=True, trajectory_axis=time_axis)
    directories = sorted(path for path in tmp_path.glob("batch_*") if path.is_dir())
    assert len(directories) == 2
    for b, directory in enumerate(directories):
        manifest = ET.parse(tmp_path / f"{directory.name}_spheres.pvd")
        entries = manifest.findall(".//DataSet")
        assert len(entries) == 2
        for t, entry in enumerate(entries):
            assert float(entry.attrib["timestep"]) == 0.125 + 0.25 * t
            assert entry.attrib["file"] == f"{directory.name}/spheres_{10 + t:08d}.vtp"
            poly = _read_poly(tmp_path / entry.attrib["file"])
            np.testing.assert_allclose(
                vtk_to_numpy(poly.GetPoints().GetData()), [[100 * b + 10 * t, 0, 0]]
            )
            np.testing.assert_array_equal(
                vtk_to_numpy(poly.GetPointData().GetArray("vel")), [[b + 1, t + 2, 0]]
            )
            np.testing.assert_array_equal(
                vtk_to_numpy(poly.GetPointData().GetArray("mass")), [b + t + 1]
            )


def _write_domain(tmp_path, dim, domain_type, *, binary=True, **domain_kw):
    box = np.array([2.0, 3.0, 4.0])[:dim]
    anchor = np.array([-1.25, 2.5, -0.75])[:dim]
    state = jd.State.create(
        pos=jnp.asarray((anchor + 0.25 * box)[None]), rad=jnp.array([0.1])
    )
    system = jd.System.create(
        state=state,
        collider_type="",
        domain_type=domain_type,
        domain_kw={"box_size": box, "anchor": anchor, **domain_kw},
    )
    with jd.VTKWriter(directory=tmp_path, writers=["domain"], binary=binary) as writer:
        writer.save(state, system)
    return _read_poly(next(tmp_path.rglob("domain_*.vtp"))), box, anchor


@pytest.mark.parametrize(
    "dim,alpha,beta,gamma",
    [
        (2, 0, 1, -0.4),
        (2, 1, 0, 0.65),
        (3, 0, 1, 0.65),
        (3, 1, 2, -0.4),
        (3, 2, 0, 0.65),
    ],
)
def test_lees_edwards_domain_exports_the_sheared_periodic_cell(
    tmp_path, dim, alpha, beta, gamma
):
    poly, box, anchor = _write_domain(
        tmp_path, dim, "leesedwards", gamma=gamma, alpha=alpha, beta=beta
    )
    points = np.unique(vtk_to_numpy(poly.GetPoints().GetData()), axis=0)

    lattice_edges = np.diag(box)
    lattice_edges[beta, alpha] = gamma * box[beta]
    corners = np.array(
        [
            anchor + np.asarray(image) @ lattice_edges
            for image in product((0, 1), repeat=dim)
        ]
    )
    corners = np.pad(corners, ((0, 0), (0, 3 - dim)))
    np.testing.assert_allclose(points, np.unique(corners, axis=0), atol=2e-6)

    normals = vtk_to_numpy(poly.GetPointData().GetNormals())
    for index in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(index)
        face = vtk_to_numpy(cell.GetPoints().GetData())
        edge_a, edge_b = face[1] - face[0], face[-1] - face[0]
        if np.linalg.norm(np.cross(edge_a, edge_b)) == 0:
            continue
        normal = normals[cell.GetPointId(0)]
        np.testing.assert_allclose([normal @ edge_a, normal @ edge_b], 0, atol=2e-6)


@pytest.mark.parametrize(
    "dim,domain_type,binary",
    [
        (2, "free", False),
        (3, "periodic", True),
        (2, "reflect", True),
        (3, "reflectsphere", False),
    ],
)
def test_orthogonal_domain_export_preserves_box_and_anchor(
    tmp_path, dim, domain_type, binary
):
    poly, box, anchor = _write_domain(tmp_path, dim, domain_type, binary=binary)
    points = np.unique(vtk_to_numpy(poly.GetPoints().GetData()), axis=0)
    corners = np.array(
        [anchor + np.asarray(image) * box for image in product((0, 1), repeat=dim)]
    )
    corners = np.pad(corners, ((0, 0), (0, 3 - dim)))
    np.testing.assert_allclose(points, np.unique(corners, axis=0), atol=2e-6)
