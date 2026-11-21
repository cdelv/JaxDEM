# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Focused contracts for collider tracing, static/dynamic stepping, and opt-in validation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import dataclasses

import numpy as np
import pytest

import jaxdem as jdem
from jaxdem.colliders import DynamicCellList, NeighborList
from jaxdem.utils.quaternion import Quaternion


def _free_system(state: jdem.State) -> jdem.System:
    return jdem.System.create(
        state=state,
        dt=0.1,
        rotation_integrator_type=None,
    )


def test_traced_neighbor_constructor_uses_explicit_static_capacity() -> None:
    state = jdem.State.create(pos=jnp.array([[0.0, 0.0], [0.75, 0.0]]))

    @jax.jit
    def create(st: jdem.State) -> NeighborList:
        return NeighborList.Create(
            st,
            cutoff=1.0,
            max_neighbors=2,
            secondary_collider_kw={"search_range": 1},
        )

    collider = create(state)
    assert collider.neighbor_list.shape == (4,)
    host_cell_list = DynamicCellList.Create(
        state, cell_size=1.0, search_range=jnp.asarray(1)
    )
    assert host_cell_list.neighbor_mask.shape == (9, 2)


def test_static_step_supports_reverse_mode_and_dynamic_step_is_explicit() -> None:
    state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]))
    system = _free_system(state)

    def terminal_x(vx: jax.Array) -> jax.Array:
        initial = dataclasses.replace(state, vel=jnp.array([[vx, 0.0]]))
        final, _ = jdem.System.step(initial, system, n=3)
        return final.pos_c[0, 0]

    np.testing.assert_allclose(jax.grad(terminal_x)(2.0), 0.3, rtol=1e-6)

    def rollout_terminal_x(vx: jax.Array) -> jax.Array:
        initial = dataclasses.replace(state, vel=jnp.array([[vx, 0.0]]))
        final, _, _ = jdem.System.trajectory_rollout(
            initial, system, n=2, stride=2, save_fn=lambda s, _: s.pos_c
        )
        return final.pos_c[0, 0]

    np.testing.assert_allclose(jax.grad(rollout_terminal_x)(2.0), 0.4, rtol=1e-6)
    dynamic, _ = jdem.System.step_dynamic(state, system, n=jnp.asarray(2))
    assert int(dynamic.N) == 1
    with pytest.raises(ValueError, match="Python integer"):
        jdem.System.step(state, system, n=jnp.asarray(2))


def test_opt_in_validation_checks_physics_but_construction_stays_permissive() -> None:
    state = jdem.State.create(pos=jnp.zeros((1, 2)), mass=jnp.array([0.0]))
    assert state.mass[0] == 0.0
    with pytest.raises(ValueError, match="positive mass"):
        state.validate()

    state = jdem.State.create(pos=jnp.zeros((1, 2)), mat_id=jnp.array([1]))
    system = _free_system(state)
    with pytest.raises(ValueError, match="material table"):
        system.validate(state)

    with pytest.raises(ValueError, match="integer dtype"):
        dataclasses.replace(state, clump_id=state.clump_id.astype(float)).validate()
    with pytest.raises(ValueError, match="boolean dtype"):
        dataclasses.replace(state, fixed=state.fixed.astype(int)).validate()
    with pytest.raises(ValueError, match="mat_id entries must be nonnegative"):
        dataclasses.replace(state, mat_id=jnp.array([-1])).validate()
    with pytest.raises(ValueError, match="species_id entries must be nonnegative"):
        dataclasses.replace(state, species_id=jnp.array([-1])).validate()
    with pytest.raises(ValueError, match="nonnegative"):
        dataclasses.replace(state, rad=jnp.array([-1.0])).validate()

    malformed_q = Quaternion.create(jnp.ones((2, 1)), jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="array shapes"):
        dataclasses.replace(state, q=malformed_q).validate()
    with pytest.raises(ValueError, match="non-finite"):
        dataclasses.replace(
            state, facet_vertices=jnp.full(state.facet_vertices.shape, jnp.inf)
        ).validate()

    clump = jdem.State.create(
        pos=jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        clump_id=jnp.array([1, 1, 3, 3]),
        mass=jnp.array([2.0, 2.0, 4.0, 4.0]),
    )
    clump.validate()
    with pytest.raises(ValueError, match="mass.*not replicated"):
        dataclasses.replace(clump, mass=clump.mass.at[1].set(3.0)).validate()
