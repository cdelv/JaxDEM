# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import pytest

import jaxdem as jd


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _ScalarHistoryForce(jd.ForceModel):
    @property
    def supports_analytical_energy_gradient(self):
        return False

    def history_shape(self, dim):
        return ()

    def init_history(self, pair_shape, dim):
        return jnp.full(pair_shape, 3.0)

    def search_radii(self, state, system):
        return state.rad

    @staticmethod
    def force(i, j, pos, state, system, history, *, advance_history=True):
        force = (pos[j] - pos[i]) * history[..., None]
        torque = jnp.zeros(j.shape + state.torque[i].shape, dtype=force.dtype)
        return force, torque, history + advance_history

    @staticmethod
    def energy(i, j, pos, state, system):
        return jnp.zeros(jnp.shape(j), dtype=pos.dtype)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _MatrixHistoryForce(jd.ForceModel):
    @property
    def supports_analytical_energy_gradient(self):
        return False

    def history_shape(self, dim):
        return (2, 2)

    def init_history(self, pair_shape, dim):
        return jnp.full(pair_shape + (2, 2), 7.0)

    def search_radii(self, state, system):
        return state.rad

    @staticmethod
    def force(i, j, pos, state, system, history, *, advance_history=True):
        force = (pos[j] - pos[i]) * history[..., 0, 0, None]
        torque = jnp.zeros(j.shape + state.torque[i].shape, dtype=force.dtype)
        return force, torque, history + advance_history

    @staticmethod
    def energy(i, j, pos, state, system):
        return jnp.zeros(jnp.shape(j), dtype=pos.dtype)


def _case(force_model, capacity):
    state = jd.State.create(
        pos=jnp.asarray([[0.0, 0.0], [0.2, 0.1], [0.4, -0.1]]),
        rad=jnp.full((3,), 0.2),
    )
    system = jd.System.create(
        state=state,
        force_model=force_model,
        collider_type="NeighborList",
        collider_kw={
            "cutoff": 1.0,
            "skin": 0.2,
            "max_neighbors": capacity,
            "secondary_collider_type": "naive",
        },
    )
    state, system = system.collider.compute_force(state, system)

    neighbors = jnp.full((state.N * capacity,), -1, dtype=int)
    offsets = jnp.array([0, 1, 2, 2]) if capacity else jnp.zeros(4, dtype=int)
    if capacity:
        neighbors = neighbors.at[0].set(1).at[1].set(2)
    history = force_model.init_history(neighbors.shape, state.dim)
    values = jnp.arange(state.N * capacity, dtype=history.dtype).reshape(
        neighbors.shape
    )
    for _ in force_model.history_shape(state.dim):
        values = values[..., None]
    history = history + values
    collider = replace(
        system.collider, neighbor_list=neighbors, row_offsets=offsets, history=history
    )
    return state, replace(system, collider=collider), neighbors, history


@pytest.mark.parametrize("force_model", [_ScalarHistoryForce(), _MatrixHistoryForce()])
@pytest.mark.parametrize("capacity", [0, 4])
@pytest.mark.parametrize("advance_history", [False, True])
def test_neighbor_history_supports_arbitrary_payload_rank(
    force_model, capacity, advance_history
):
    state, system, neighbors, history = _case(force_model, capacity)

    result, updated_system = system.collider.compute_force(
        state, system, advance_history=advance_history
    )

    if advance_history:
        valid = neighbors != -1
        for _ in force_model.history_shape(state.dim):
            valid = valid[..., None]
        initialized = force_model.init_history(neighbors.shape, state.dim)
        expected = jnp.where(valid, history + 1, initialized)
    else:
        expected = history

    assert updated_system.collider.history.shape == history.shape
    assert jnp.array_equal(updated_system.collider.history, expected)
    assert jnp.all(jnp.isfinite(result.force))
    assert jnp.all(jnp.isfinite(result.torque))
