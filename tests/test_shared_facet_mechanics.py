# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Shared contact faces transfer each face's load to physical nodes once."""

import jax
import jax.numpy as jnp

from dataclasses import replace

import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils.linalg import cross


def _system(state, model="sphere_facet_spring"):
    material = jd.Material.create("elastic", young=20.0, poisson=0.25, density=1.0)
    return jd.System.create(
        state=state,
        collider_type="naive",
        force_model=jd.ForceModel.create(model) if isinstance(model, str) else model,
        mat_table=jd.MaterialTable.from_materials([material]),
    )


def _evaluate(state, system):
    loaded, _ = system.collider.compute_force(state, system)
    _, _, energy = system.collider.compute_potential_energy(state, system)
    return loaded, energy


def _surface(dim):
    if dim == 2:
        return jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.2]]), [[0, 1], [1, 2]]
    return jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.1]]
    ), [[0, 1, 2], [1, 3, 2]]


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("deform", [False, True])
def test_shared_sphere_contacts_match_independent_face_loads_and_energy_gradient(
    dim, deform
):
    vertices, faces = _surface(dim)
    if deform:
        vertices = vertices.at[1, 0].add(0.15).at[-1, -1].add(0.1)
    # A sphere contacts both faces, exercising the extra incidence at shared nodes.
    point = [1.0, 0.15] if dim == 2 else [0.5, 0.5, 0.18]
    nodes = jd.State.create(pos=vertices, rad=jnp.full(len(vertices), 0.1))
    state = jd.State.add_contact_facets(nodes, faces)
    sphere = jd.State.create(pos=jnp.array([point]), rad=jnp.array([0.2]))
    state = jd.State.merge(state, sphere)
    system = _system(state)
    loaded, energy = _evaluate(state, system)
    tolerance = 2e-6 if state.pos.dtype == jnp.float32 else 1e-10

    expected_force = np.zeros_like(state.pos)
    expected_energy = 0.0
    for face in faces:
        independent = jd.State.add_facet(
            jd.State.create(), vertices[jnp.array(face)], thickness=0.1, rigid=False
        )
        independent = jd.State.merge(independent, sphere)
        reference, face_energy = _evaluate(independent, _system(independent))
        assert float(face_energy) > 0.0
        np.add.at(expected_force, face, np.asarray(reference.force[:-1]))
        expected_force[-1] += np.asarray(reference.force[-1])
        expected_energy += float(face_energy)
    np.testing.assert_allclose(
        loaded.force, expected_force, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(energy, expected_energy, rtol=tolerance)
    np.testing.assert_allclose(jnp.sum(loaded.force, axis=0), 0.0, atol=tolerance)
    np.testing.assert_allclose(
        jnp.sum(cross(state.pos, loaded.force), axis=0), 0.0, atol=tolerance
    )
    np.testing.assert_array_equal(loaded.torque, 0.0)

    def potential(positions):
        return system.collider.compute_potential_energy(
            replace(state, pos_c=positions), system
        )[2]

    np.testing.assert_allclose(
        -jax.grad(potential)(state.pos_c), loaded.force, rtol=2e-5, atol=2e-6
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_adjacent_faces_have_no_artificial_self_contact(dim):
    vertices, faces = _surface(dim)
    state = jd.State.add_contact_facets(
        jd.State.create(pos=vertices, rad=jnp.full(len(vertices), 0.1)), faces
    )
    loaded, energy = _evaluate(state, _system(state, "facet_facet_spring"))
    np.testing.assert_array_equal(loaded.force, 0.0)
    np.testing.assert_array_equal(loaded.torque, 0.0)
    assert float(energy) == 0.0


@pytest.mark.parametrize("dim", [2, 3])
def test_two_shared_surfaces_match_separate_face_pair_contacts(dim):
    vertices, faces = _surface(dim)
    shift = jnp.zeros(dim).at[-1].set(0.15)
    first = jd.State.add_contact_facets(
        jd.State.create(
            pos=vertices,
            rad=jnp.full(len(vertices), 0.1),
            species_id=jnp.zeros(len(vertices), dtype=int),
        ),
        faces,
    )
    second = jd.State.add_contact_facets(
        jd.State.create(
            pos=vertices + shift,
            rad=jnp.full(len(vertices), 0.1),
            species_id=jnp.ones(len(vertices), dtype=int),
        ),
        faces,
    )
    state = jd.State.merge(first, second)
    router = jd.ForceRouter.from_dict(
        2, {(0, 1): jd.ForceModel.create("facet_facet_spring")}
    )
    loaded, energy = _evaluate(state, _system(state, router))
    tolerance = 2e-6 if state.pos.dtype == jnp.float32 else 1e-10
    expected_force = np.zeros_like(state.pos)
    expected_energy = 0.0
    for face_i in faces:
        for face_j in faces:
            a = jd.State.add_facet(
                jd.State.create(),
                vertices[jnp.array(face_i)],
                thickness=0.1,
                rigid=False,
                species_id=0,
            )
            b = jd.State.add_facet(
                jd.State.create(),
                vertices[jnp.array(face_j)] + shift,
                thickness=0.1,
                rigid=False,
                species_id=1,
            )
            pair = jd.State.merge(a, b)
            reference, pair_energy = _evaluate(pair, _system(pair, router))
            np.add.at(expected_force, face_i, np.asarray(reference.force[:dim]))
            np.add.at(
                expected_force,
                np.array(face_j) + len(vertices),
                np.asarray(reference.force[dim:]),
            )
            expected_energy += float(pair_energy)
    assert expected_energy > 0.0
    np.testing.assert_allclose(
        loaded.force, expected_force, rtol=10 * tolerance, atol=tolerance
    )
    np.testing.assert_allclose(energy, expected_energy, rtol=10 * tolerance)
    np.testing.assert_allclose(jnp.sum(loaded.force, axis=0), 0.0, atol=tolerance)
    np.testing.assert_allclose(
        jnp.sum(cross(state.pos, loaded.force), axis=0), 0.0, atol=tolerance
    )
