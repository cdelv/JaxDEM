# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Dynamic and shared facet bounds must keep spatial searches conservative."""

import jax
import jax.numpy as jnp

from dataclasses import replace

import numpy as np
import pytest

import jaxdem as jd
from jaxdem.forces.facet_contact import facet_search_radii


def _system(state, collider, **kwargs):
    return jd.System.create(
        state=state,
        collider_type=collider,
        collider_kw=kwargs,
        force_model_type="sphere_facet_spring",
    )


@pytest.fixture(scope="module", params=[2, 3])
def stretch_reference(request):
    dim = request.param
    if dim == 2:
        vertices = jnp.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        faces = jnp.array([[0, 1], [0, 2]])
        point = jnp.array([[8.0, 0.15]])
    else:
        vertices = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = jnp.array([[0, 1, 2], [0, 3, 4]])
        point = jnp.array([[7.0, 1.0, 0.15]])
    state = jd.State.add_contact_facets(
        jd.State.create(pos=vertices, rad=jnp.full(vertices.shape[0], 0.1)), faces
    )
    state = jd.State.merge(state, jd.State.create(pos=point, rad=jnp.array([0.2])))
    deformed = replace(state, pos_c=state.pos_c.at[:-1].set(10.0 * vertices))
    reference, _ = jd.System.evaluate_forces(deformed, _system(deformed, "naive"))
    return dim, vertices, state, reference.force


@pytest.mark.parametrize("collider", ["CellList", "MultiCellList", "NeighborList"])
def test_shared_facet_stretch_reaches_contact_far_from_its_primary_vertex(
    stretch_reference, collider
):
    dim, vertices, state, reference_force = stretch_reference
    kwargs = (
        {
            "cutoff": 0.2,
            "skin": 0.1,
            "max_neighbors": state.N,
            "secondary_collider_type": "CellList",
            "secondary_collider_kw": {"cell_size": 0.25},
        }
        if collider == "NeighborList"
        else {"cell_size": 0.25}
    )
    system = _system(state, collider, **kwargs)
    before, system = jd.System.evaluate_forces(state, system)
    np.testing.assert_allclose(before.force, 0.0, atol=1e-9)

    deformed = replace(before, pos_c=before.pos_c.at[:-1].set(10.0 * vertices))
    bounds = facet_search_radii(deformed, system)
    np.testing.assert_allclose(bounds[0], 10.1, rtol=1e-6)
    after, system = jd.System.evaluate_forces(deformed, system)
    system.check_overflow()
    expected = np.zeros(dim)
    expected[-1] = 1500.0
    np.testing.assert_allclose(after.force[-1], expected, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(after.force.sum(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(after.force, reference_force, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("extra_only_face", [False, True])
def test_adding_same_size_topology_rebuilds_neighbor_cache_with_unchanged_global_reach(
    extra_only_face,
):
    if extra_only_face:
        pos = jnp.array(
            [
                [0.0, 0.0],
                [5.0, 0.0],
                [0.0, -1.0],
                [5.0, -1.0],
                [3.0, 0.15],
                [100.0, 100.0],
            ]
        )
        state = jd.State.create(pos=pos, rad=jnp.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.1]))
        state = jd.State.add_contact_facets(state, jnp.array([[0, 2], [1, 3]]))
    else:
        pos = jnp.array([[0.0, 0.0], [5.0, 0.0], [3.0, 0.15], [100.0, 100.0]])
        state = jd.State.create(pos=pos, rad=jnp.array([0.1, 0.1, 0.2, 0.1]))
    state = replace(state, _rad=state._rad.at[-1].set(1000.0))
    system = _system(
        state,
        "NeighborList",
        cutoff=0.2,
        skin=0.1,
        max_neighbors=state.N,
        secondary_collider_type="naive",
    )
    state, system = jd.System.evaluate_forces(state, system)
    np.testing.assert_allclose(state.force, 0.0, atol=1e-9)
    cutoff = system.collider.physical_cutoff_snapshot
    builds = int(system.collider.n_build_times)
    connected = jd.State.add_contact_facets(state, jnp.array([[0, 1]]))
    if extra_only_face:
        np.testing.assert_array_equal(connected.facet_id, state.facet_id)
    after, system = jd.System.evaluate_forces(connected, system)
    assert int(system.collider.n_build_times) == builds + 1
    assert system.collider.physical_cutoff_snapshot == cutoff
    np.testing.assert_allclose(after.force[-2, -1], 1500.0, rtol=1e-5)
    reference, _ = jd.System.evaluate_forces(connected, _system(connected, "naive"))
    np.testing.assert_allclose(after.force, reference.force, rtol=1e-5, atol=1e-4)

    moved = replace(after, pos_c=after.pos_c.at[-2, -1].add(0.01))
    _, reused = jd.System.evaluate_forces(moved, system)
    assert int(reused.collider.n_build_times) == builds + 1


def test_vtk_writer_exports_shared_faces_and_incidence_arrays(tmp_path):
    vtk = pytest.importorskip("vtk")
    from jaxdem.writers.vtk_facets_writer import VTKFacetsWriter
    from jaxdem.writers.vtk_spheres_writer import (
        VTKFacetSpheresWriter,
        VTKSpheresWriter,
    )

    plain = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]),
        rad=jnp.array([0.05, 0.1, 0.2]),
        mass=jnp.array([1.0, 2.0, 3.0]),
    )
    shared = jd.State.add_contact_facets(plain, jnp.array([[0, 1], [1, 2], [2, 0]]))

    def read_vtp(path):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        return reader.GetOutput()

    # Facets writer includes extra-only face IDs with per-vertex fields.
    facets_file = tmp_path / "shared.vtp"
    VTKFacetsWriter.write(
        jax.tree.map(np.asarray, shared), _system(shared, "naive"), facets_file, False
    )
    output = read_vtp(facets_file)
    assert output.GetNumberOfCells() == 3
    masses = output.GetPointData().GetArray("mass")
    assert masses.GetNumberOfTuples() == output.GetNumberOfPoints()
    assert {masses.GetValue(i) for i in range(masses.GetNumberOfTuples())} == {
        1.0,
        2.0,
        3.0,
    }
    thickness = output.GetPointData().GetArray("thickness")
    values = sorted(
        {thickness.GetValue(i) for i in range(thickness.GetNumberOfTuples())}
    )
    np.testing.assert_allclose(values, [0.05, 0.1, 0.2], rtol=1e-6)

    # Sphere writers handle both empty (plain) and multidimensional (shared) incidence arrays.
    for name, st, writer_cls, has_extra in (
        ("plain.vtp", plain, VTKSpheresWriter, False),
        ("shared_nodes.vtp", shared, VTKFacetSpheresWriter, True),
    ):
        node_file = tmp_path / name
        writer_cls.write(
            jax.tree.map(np.asarray, st), _system(st, "naive"), node_file, False
        )
        node_out = read_vtp(node_file)
        assert node_out.GetNumberOfPoints() == 3
        incidence = node_out.GetPointData().GetArray("facet_vertices_extra")
        if has_extra:
            assert incidence.GetNumberOfComponents() == 2 * st.facet_ids_extra.shape[-1]
        else:
            assert incidence is None


@pytest.fixture(scope="module", params=[2, 3])
def periodic_reference(request):
    dim = request.param
    if dim == 2:
        vertices = jnp.array([[9.8, 0.1], [10.2, 0.1], [10.4, 0.4]])
        faces = jnp.array([[0, 1], [1, 2]])
        sphere = jnp.array([[10.0, 0.25]])
    else:
        vertices = jnp.array(
            [[9.8, 0.1, 0.1], [10.2, 0.1, 0.1], [10.2, 0.5, 0.1], [9.8, 0.5, 0.1]]
        )
        faces = jnp.array([[0, 1, 2], [0, 2, 3]])
        sphere = jnp.array([[10.0, 0.3, 0.25]])
    original = jd.State.add_contact_facets(
        jd.State.create(pos=vertices, rad=jnp.full(len(vertices), 0.1)), faces
    )
    original = jd.State.merge(
        original, jd.State.create(pos=sphere, rad=jnp.array([0.2]))
    )
    wrapped = replace(
        original, pos_c=original.pos_c.at[:, 0].set(original.pos_c[:, 0] % 10.0)
    )
    reference_system = _system(original, "naive")
    expected_frames = []
    for displacement in (0.0, 0.01):
        free = replace(original, pos_c=original.pos_c.at[1, -1].add(displacement))
        expected, reference_system = jd.System.evaluate_forces(free, reference_system)
        _, reference_system, expected_energy = (
            reference_system.collider.compute_potential_energy(
                expected, reference_system
            )
        )
        expected_frames.append((displacement, expected.force, expected_energy))
    return dim, wrapped, expected_frames


@pytest.mark.parametrize(
    "collider", ["naive", "CellList", "MultiCellList", "NeighborList"]
)
def test_periodic_shared_surface_matches_unwrapped_contacts_after_deformation(
    periodic_reference, collider
):
    dim, wrapped, expected_frames = periodic_reference
    kwargs = {}
    if collider == "NeighborList":
        kwargs = {
            "cutoff": 0.2,
            "skin": 0.1,
            "max_neighbors": wrapped.N,
            "secondary_collider_type": "CellList",
            "secondary_collider_kw": {"cell_size": 0.2},
        }
    elif collider != "naive":
        kwargs = {"cell_size": 0.2}
    system = jd.System.create(
        state=wrapped,
        collider_type=collider,
        collider_kw=kwargs,
        force_model_type="sphere_facet_spring",
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(dim, 10.0), "anchor": jnp.zeros(dim)},
    )
    for displacement, expected_force, expected_energy in expected_frames:
        periodic = replace(wrapped, pos_c=wrapped.pos_c.at[1, -1].add(displacement))
        actual, system = jd.System.evaluate_forces(periodic, system)
        _, system, actual_energy = system.collider.compute_potential_energy(
            actual, system
        )
        system.check_overflow()
        assert np.linalg.norm(actual.force[-1]) > 0.0
        np.testing.assert_allclose(actual.force, expected_force, rtol=2e-5, atol=2e-3)
        np.testing.assert_allclose(actual_energy, expected_energy, rtol=2e-5, atol=2e-3)
        np.testing.assert_allclose(actual.force.sum(axis=0), 0.0, atol=2e-3)
