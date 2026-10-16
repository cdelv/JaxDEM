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
from jaxdem.colliders import _neighbor_cache

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
    def force(
        i, j, pos, state, system, history, *, advance_history=True, quasistatic=False
    ):
        force = (pos[j] - pos[i]) * history[..., 0, 0, None]
        torque = jnp.zeros(j.shape + state.torque[i].shape, dtype=force.dtype)
        return force, torque, history + advance_history

    @staticmethod
    def energy(i, j, pos, state, system, history):
        return 0.5 * jnp.sum((pos[j] - pos[i]) ** 2, axis=-1)


def _system(capacity, dim=2, n=3):
    pos = jnp.asarray([[0.0, 0.0], [0.2, 0.1], [0.4, -0.1]])
    pos_p = jnp.asarray([[0.1, 0.05], [-0.05, 0.1], [0.08, -0.06]])
    if n > 3:
        # Eight mutually touching particles, then an isolated empty row.
        pos = jnp.asarray(
            np.stack([0.02 * np.arange(n), 0.01 * np.sin(np.arange(n))], axis=-1)
        )
        pos = pos.at[-1].set(jnp.array([2.0, 2.0]))
        pos_p = jnp.asarray(
            np.stack(
                [0.01 * np.cos(np.arange(n)), 0.01 * np.sin(np.arange(n))], axis=-1
            )
        )
    pos = jnp.pad(pos, ((0, 0), (0, dim - 2)))
    pos_p = jnp.pad(pos_p, ((0, 0), (0, dim - 2)))
    state = jd.State.create(pos=pos, pos_p=pos_p, rad=jnp.full((n,), 0.2))
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
            f = (state.pos[j] - state.pos[i]) * col.history[slot, 0, 0]
            force = force.at[i].add(f)
            if advance_history:
                history = history.at[slot].set(col.history[slot] + 1)
    return force, torque + cross(state._pos_p_rot, force), history


@pytest.mark.parametrize("capacity", [0, 1, 9])
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
@pytest.mark.parametrize("dim", [2, 3])
def test_force_blocks_cross_row_boundaries_and_reset_unused_history(
    advance_history, dim, monkeypatch
):
    state, system = _system(9, dim, n=9)
    np.testing.assert_array_equal(np.diff(system.collider.row_offsets), [7] * 8 + [0])
    # Force a real scan plus a short final batch without a huge fixture.
    monkeypatch.setitem(_neighbor_cache._ROW_BATCH_SIZE, dim, 2)
    history = jnp.arange(system.collider.history.size, dtype=float).reshape(
        system.collider.history.shape
    )
    system = replace(system, collider=replace(system.collider, history=history))
    expected = _pair_reference(state, system, advance_history)
    evaluate = jax.jit(lambda s, y: _neighbor_cache.forces(s, y, advance_history))
    actual = evaluate(state, system)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("advance_history", [False, True])
@pytest.mark.parametrize("dim", [2, 3])
def test_force_rows_support_forward_and_reverse_mode(advance_history, dim, monkeypatch):
    state, system = _system(9, dim)
    monkeypatch.setitem(_neighbor_cache._ROW_BATCH_SIZE, dim, 2)
    weights = jnp.arange(state.force.size, dtype=float).reshape(state.force.shape) + 1

    def blocked_observable(pos):
        moved = replace(state, pos_c=pos)
        force, _, history = _neighbor_cache.forces(moved, system, advance_history)
        return jnp.sum(weights * force) + history.sum()

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
    h = 1e-4 if jax.config.jax_enable_x64 else 1e-2
    finite_difference = (
        blocked_observable(state.pos_c + h * direction)
        - blocked_observable(state.pos_c - h * direction)
    ) / (2 * h)
    np.testing.assert_allclose(actual_jvp, finite_difference, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(
        jnp.sum(actual * direction), actual_jvp, rtol=_RTOL, atol=_ATOL
    )
