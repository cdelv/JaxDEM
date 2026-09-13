# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils.linalg import cross

_RTOL = 1e-12 if jax.config.jax_enable_x64 else 1e-6
_ATOL = 1e-12 if jax.config.jax_enable_x64 else 1e-7


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _HistoryForce(jd.ForceModel):
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
        return 0.5 * jnp.sum((pos[j] - pos[i]) ** 2, axis=-1)


def _system(capacity):
    pos = jnp.asarray([[0.0, 0.0], [0.2, 0.1], [0.4, -0.1]])
    pos_p = jnp.asarray([[0.1, 0.05], [-0.05, 0.1], [0.08, -0.06]])
    state = jd.State.create(pos=pos, pos_p=pos_p, rad=jnp.full((3,), 0.2))
    system = jd.System.create(
        state=state,
        force_model=_HistoryForce(),
        collider_type="NeighborList",
        collider_kw={
            "cutoff": 1.0,
            "skin": 0.2,
            "max_neighbors": capacity,
            "secondary_collider_type": "naive",
        },
    )
    state, system = system.collider.compute_force(state, system)
    return state, system


def _pair_reference(state, system, advance_history):
    col = system.collider
    offsets = np.asarray(col.row_offsets)
    force, torque = jnp.zeros_like(state.force), jnp.zeros_like(state.torque)
    history = (
        system.force_model.init_history(col.neighbor_list.shape, state.dim)
        if advance_history
        else col.history
    )
    for i in range(state.N):
        for slot in range(offsets[i], offsets[i + 1]):
            j = col.neighbor_list[slot]
            f, t, h = system.force_model.force(
                jnp.asarray(i),
                j,
                state.pos,
                state,
                system,
                col.history[slot],
                advance_history=advance_history,
            )
            force = force.at[i].add(f)
            torque = torque.at[i].add(t)
            if advance_history:
                history = history.at[slot].set(h)
    return force, torque + cross(state._pos_p_rot, force), history


@pytest.mark.parametrize("capacity", [0, 1, 7, 8, 9, 64])
@pytest.mark.parametrize("advance_history", [False, True])
def test_force_blocks_match_pair_reference(capacity, advance_history):
    state, system = _system(capacity)
    expected_force, expected_torque, expected_history = _pair_reference(
        state, system, advance_history
    )

    actual_state, actual_system = system.collider.compute_force(
        state, system, advance_history=advance_history
    )

    # Block accumulation groups floating-point additions differently.
    assert jnp.allclose(actual_state.force, expected_force, rtol=_RTOL, atol=_ATOL)
    assert jnp.allclose(actual_state.torque, expected_torque, rtol=_RTOL, atol=_ATOL)
    if capacity > 0:
        assert jnp.any(jnp.abs(expected_torque) > 0)
    assert jnp.array_equal(actual_system.collider.history, expected_history)


@pytest.mark.parametrize("advance_history", [False, True])
def test_force_blocks_cross_row_boundaries_and_reset_unused_history(advance_history):
    state, system = _system(9)
    # Three complete rows cross four-pair block boundaries; the unused tail
    # carries stale history to verify that only valid pairs retain memory.
    history = jnp.arange(108, dtype=float).reshape(27, 2, 2)
    system = replace(system, collider=replace(system.collider, history=history))
    expected = _pair_reference(state, system, advance_history)
    actual, updated = system.collider.compute_force(
        state, system, advance_history=advance_history
    )
    assert jnp.allclose(actual.force, expected[0], rtol=_RTOL, atol=_ATOL)
    assert jnp.allclose(actual.torque, expected[1], rtol=_RTOL, atol=_ATOL)
    assert jnp.array_equal(updated.collider.history, expected[2])


@pytest.mark.parametrize("advance_history", [False, True])
def test_force_rows_support_forward_and_reverse_mode(advance_history):
    state, system = _system(9)
    weights = jnp.arange(state.force.size, dtype=float).reshape(state.force.shape) + 1

    def blocked_observable(pos):
        moved = replace(state, pos_c=pos)
        result, updated = system.collider.compute_force(
            moved, system, advance_history=advance_history
        )
        return jnp.sum(weights * result.force) + updated.collider.history.sum()

    def pair_observable(pos):
        moved = replace(state, pos_c=pos)
        force, _, history = _pair_reference(moved, system, advance_history)
        return jnp.sum(weights * force) + history.sum()

    actual = jax.grad(blocked_observable)(state.pos_c)
    expected = jax.grad(pair_observable)(state.pos_c)
    assert jnp.all(jnp.isfinite(actual))
    assert jnp.any(actual != 0)
    assert jnp.allclose(actual, expected, rtol=_RTOL, atol=_ATOL)
    direction = jnp.arange(state.pos_c.size, dtype=state.pos_c.dtype).reshape(
        state.pos_c.shape
    )
    _, actual_jvp = jax.jvp(blocked_observable, (state.pos_c,), (direction,))
    _, expected_jvp = jax.jvp(pair_observable, (state.pos_c,), (direction,))
    assert jnp.allclose(actual_jvp, expected_jvp, rtol=_RTOL, atol=_ATOL)
