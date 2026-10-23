# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Physical field ownership and frame identity in real VTK files."""

import jax
import jax.numpy as jnp

from dataclasses import replace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import scipy.spatial
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import jaxdem as jd


def read_poly(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("hull", [False, True])
@pytest.mark.parametrize("scale", [1e-12, 1.0])
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
    poly = read_poly(next(tmp_path.rglob("*.vtp")))
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
            poly = read_poly(tmp_path / entry.attrib["file"])
            np.testing.assert_allclose(
                vtk_to_numpy(poly.GetPoints().GetData()), [[100 * b + 10 * t, 0, 0]]
            )
            np.testing.assert_array_equal(
                vtk_to_numpy(poly.GetPointData().GetArray("vel")), [[b + 1, t + 2, 0]]
            )
            np.testing.assert_array_equal(
                vtk_to_numpy(poly.GetPointData().GetArray("mass")), [b + t + 1]
            )
