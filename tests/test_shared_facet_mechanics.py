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
def test_shared_sphere_contacts_match_independent_face_loads_and_energy_gradient(dim):
    vertices, faces = _surface(dim)
    point = [1.0, 0.15] if dim == 2 else [0.5, 0.5, 0.18]
    nodes = jd.State.create(pos=vertices, rad=jnp.full(len(vertices), 0.1))
    state = jd.State.add_contact_facets(nodes, faces)
    sphere = jd.State.create(pos=jnp.array([point]), rad=jnp.array([0.2]))
    state = jd.State.merge(state, sphere)
    # Deform live vertices after add_contact_facets so extra-incidence geometry tracks motion.
    deformed_vertices = vertices.at[1, 0].add(0.15).at[-1, -1].add(0.1)
    state = replace(state, pos_c=state.pos_c.at[:-1].set(deformed_vertices))
    system = _system(state)
    loaded, energy = _evaluate(state, system)
    tolerance = 2e-6 if state.pos.dtype == jnp.float32 else 1e-10

    expected_force = np.zeros_like(state.pos)
    expected_energy = 0.0
    for face in faces:
        independent = jd.State.add_facet(
            jd.State.create(),
            deformed_vertices[jnp.array(face)],
            thickness=0.1,
            rigid=False,
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


def _bonded_particle(dim):
    if dim == 2:
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.2]])
        faces = jnp.array([[0, 1], [1, 2]])
        edges = jnp.array([[0, 1], [1, 2], [2, 0]])
        sphere_position = jnp.array([[0.95, 0.15]])
    else:
        vertices = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.05]]
        )
        faces = jnp.array([[0, 1, 2], [1, 3, 2]])
        edges = jnp.array([[0, 1], [1, 3], [3, 2], [2, 0], [1, 2]])
        sphere_position = jnp.array([[0.53, 0.46, 0.18]])
    count = len(vertices)
    adjacency = [[j for j in range(count) if i != j] for i in range(count)]
    nodes = jd.State.create(
        pos=vertices,
        rad=jnp.full(count, 0.08),
        mass=jnp.arange(count, dtype=float) * 0.2 + 1.0,
        species_id=jnp.ones(count, dtype=int),
        bond_id=adjacency,
    )
    surface = jd.State.add_contact_facets(nodes, faces)
    state = jd.State.merge(
        surface,
        jd.State.create(
            pos=sphere_position, rad=jnp.array([0.2]), mass=jnp.array([0.7])
        ),
    )
    velocity = jnp.zeros_like(state.vel).at[1, -1].set(-0.3)
    velocity = velocity.at[0, 0].set(0.15).at[-1, -1].set(-0.1)
    state = replace(state, vel=velocity)
    bonded = jd.BondedForceModel.create(
        "deformableparticlemodel", vertices=vertices, edges=edges, el=50.0
    )
    return state, bonded, edges


def _bonded_system(state, bonded, collider):
    kwargs = {}
    if collider == "NeighborList":
        kwargs = {
            "cutoff": 0.2,
            "skin": 0.1,
            "max_neighbors": state.N,
            "secondary_collider_type": "CellList",
            "secondary_collider_kw": {"cell_size": 0.5},
        }
    elif collider != "naive":
        kwargs = {"cell_size": 0.5}
    material = jd.Material.create("elastic", young=20.0, poisson=0.25, density=1.0)
    router = jd.ForceRouter.from_dict(
        2, {(0, 1): jd.ForceModel.create("sphere_facet_spring")}
    )
    return jd.System.create(
        state=state,
        collider_type=collider,
        collider_kw=kwargs,
        force_model=router,
        bonded_force_model=bonded,
        mat_table=jd.MaterialTable.from_materials([material]),
        dt=0.002,
    )


def _rollout_bonded(state, system):
    initial, system = jd.System.initialize(state, system)
    final, system, history = jd.System.trajectory_rollout(
        initial, system, n=12, stride=2, save_fn=lambda st, _: (st.pos, st.vel, st.force)
    )
    system.check_overflow()
    return initial, final, system, history


@pytest.fixture(scope="module", params=[2, 3])
def bonded_reference(request):
    state, bonded, edges = _bonded_particle(request.param)
    initial, final, system, history = _rollout_bonded(
        state, _bonded_system(state, bonded, "naive")
    )
    return state, bonded, edges, initial, final, system, history


@pytest.mark.parametrize("collider", ["CellList", "MultiCellList", "NeighborList"])
def test_shared_node_bonded_trajectories_match_naive_and_conserve_momentum(
    bonded_reference, collider
):
    state, bonded, edges, initial, expected, _, expected_history = bonded_reference
    actual_initial, actual, system, history = _rollout_bonded(
        state, _bonded_system(state, bonded, collider)
    )
    assert actual.N == state.N
    np.testing.assert_array_equal(actual.mass, state.mass)
    for observed, predicted in zip(history, expected_history, strict=True):
        np.testing.assert_allclose(observed, predicted, rtol=2e-5, atol=3e-6)
    np.testing.assert_allclose(actual.force, expected.force, rtol=2e-5, atol=3e-6)
    np.testing.assert_allclose(
        actual_initial.force, initial.force, rtol=2e-5, atol=3e-6
    )
    assert np.linalg.norm(np.asarray(actual.force[-1])) > 0.1
    initial_lengths = jnp.linalg.norm(
        state.pos[edges[:, 1]] - state.pos[edges[:, 0]], axis=-1
    )
    final_lengths = jnp.linalg.norm(
        actual.pos[edges[:, 1]] - actual.pos[edges[:, 0]], axis=-1
    )
    assert float(jnp.max(jnp.abs(final_lengths - initial_lengths))) > 1e-3
    assert float(bonded.compute_potential_energy(actual.pos, actual, system)) > 1e-5
    initial_momentum = state.mass[:, None] * state.vel
    final_momentum = actual.mass[:, None] * actual.vel
    np.testing.assert_allclose(
        final_momentum.sum(axis=0), initial_momentum.sum(axis=0), atol=5e-6
    )
    np.testing.assert_allclose(
        jnp.sum(cross(actual.pos, final_momentum), axis=0),
        jnp.sum(cross(state.pos, initial_momentum), axis=0),
        atol=5e-6,
    )
    np.testing.assert_array_equal(actual.ang_vel, 0.0)


def test_shared_node_bonded_surface_supports_jit_and_batched_steps():
    state, bonded, _ = _bonded_particle(2)
    system = _bonded_system(state, bonded, "naive")
    state, system = jd.System.initialize(state, system)
    translated = replace(state, pos_c=state.pos_c + jnp.array([0.3, -0.2]))
    batched_state = jd.State.stack([state, translated])
    batched_system = jd.System.stack([system, system])
    advance = jax.jit(lambda st, sy: jd.System.step_dynamic(st, sy, n=jnp.array(6)))
    observed, observed_system = advance(batched_state, batched_system)
    expected, expected_system = jd.System.step(state, system, n=6)
    observed_system.check_overflow()
    expected_system.check_overflow()
    np.testing.assert_allclose(observed.pos[0], expected.pos, rtol=2e-5, atol=3e-6)
    np.testing.assert_allclose(
        observed.pos[1], expected.pos + jnp.array([0.3, -0.2]), rtol=2e-5, atol=3e-6
    )
    np.testing.assert_allclose(
        observed.vel,
        jnp.broadcast_to(expected.vel, observed.vel.shape),
        rtol=2e-5,
        atol=3e-6,
    )
    np.testing.assert_array_equal(observed.mass, batched_state.mass)
