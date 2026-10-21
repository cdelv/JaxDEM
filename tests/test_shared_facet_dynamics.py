# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Bonded deformable particles use shared contact nodes throughout dynamics."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils.linalg import cross


def _particle(dim):
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
    assert surface.N == nodes.N
    np.testing.assert_array_equal(surface.mass, nodes.mass)
    np.testing.assert_array_equal(surface.bond_id, nodes.bond_id)
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


def _system(state, bonded, collider):
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


def _save(state, system):
    del system
    return state.pos, state.vel, state.force


def _rollout(state, system):
    initial, system = jd.System.initialize(state, system)
    final, system, history = jd.System.trajectory_rollout(
        initial, system, n=12, stride=2, save_fn=_save
    )
    system.check_overflow()
    return initial, final, system, history


@pytest.fixture(scope="module", params=[2, 3])
def reference(request):
    state, bonded, edges = _particle(request.param)
    initial, final, system, history = _rollout(state, _system(state, bonded, "naive"))
    return state, bonded, edges, initial, final, system, history


@pytest.mark.parametrize("collider", ["CellList", "MultiCellList", "NeighborList"])
def test_shared_node_bonded_trajectories_match_naive_and_conserve_momentum(
    reference, collider
):
    state, bonded, edges, initial, expected, _, expected_history = reference
    actual_initial, actual, system, history = _rollout(
        state, _system(state, bonded, collider)
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
    state, bonded, _ = _particle(2)
    system = _system(state, bonded, "naive")
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
