# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import pytest

import jaxdem as jd
import jaxdem.colliders.neighbor_list as neighbor_list_module
from jaxdem.colliders import valid_interaction_mask
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


@pytest.fixture(autouse=True)
def _force_multiple_blocks(monkeypatch):
    monkeypatch.setattr(neighbor_list_module, "_FORCE_NEIGHBOR_PAIR_BUDGET", 24)
    jax.clear_caches()
    yield
    jax.clear_caches()


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


def _full_width_reference(state, system, advance_history):
    collider = system.collider
    nl = collider.neighbor_list
    history = collider.history
    iota = jax.lax.iota(dtype=int, size=state.N)

    def per_particle(i, pos_pi, neighbors, hist_i):
        valid = neighbors != -1
        safe_j = jnp.maximum(neighbors, 0)
        valid = valid * valid_interaction_mask(
            state.clump_id[i],
            state.clump_id[safe_j],
            state.bond_id[i],
            safe_j,
            system.interact_same_bond_id,
        )
        force, torque, new_history = system.force_model.force(
            i,
            safe_j,
            state.pos,
            state,
            system,
            hist_i,
            advance_history=advance_history,
        )
        force = jnp.where(valid[..., None], force, 0.0)
        torque = jnp.where(valid[..., None], torque, 0.0)
        if advance_history:
            initialized = system.force_model.init_history(neighbors.shape, state.dim)
            history_valid = valid
            for _ in range(new_history.ndim - history_valid.ndim):
                history_valid = history_valid[..., None]
            new_history = jnp.where(history_valid, new_history, initialized)
        force_sum = jnp.sum(force, axis=0)
        torque_sum = jnp.sum(torque, axis=0) + cross(pos_pi, force_sum)
        return force_sum, torque_sum, new_history

    return jax.vmap(per_particle)(iota, state._pos_p_rot, nl, history)


@pytest.mark.parametrize("capacity", [0, 1, 7, 8, 9, 64])
@pytest.mark.parametrize("advance_history", [False, True])
def test_force_blocks_match_full_width_reference(capacity, advance_history):
    state, system = _system(capacity)
    expected_force, expected_torque, expected_history = _full_width_reference(
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
def test_force_blocks_preserve_holes_and_late_valid_slots(advance_history):
    state, system = _system(9)
    neighbors = jnp.full((3, 9), -1, dtype=int)
    neighbors = neighbors.at[0, 0].set(1)
    neighbors = neighbors.at[0, 8].set(2)
    neighbors = neighbors.at[1, 8].set(0)
    history = jnp.arange(108, dtype=float).reshape(3, 9, 2, 2)
    collider = replace(system.collider, neighbor_list=neighbors, history=history)
    system = replace(system, collider=collider)
    expected_force, expected_torque, expected_history = _full_width_reference(
        state, system, advance_history
    )

    actual_state, actual_system = system.collider.compute_force(
        state, system, advance_history=advance_history
    )

    assert jnp.allclose(actual_state.force, expected_force, rtol=_RTOL, atol=_ATOL)
    assert jnp.allclose(actual_state.torque, expected_torque, rtol=_RTOL, atol=_ATOL)
    assert jnp.array_equal(actual_system.collider.history, expected_history)


def test_force_block_scan_supports_reverse_mode():
    state, system = _system(9)
    weights = jnp.arange(state.force.size, dtype=float).reshape(state.force.shape) + 1

    def blocked_observable(pos):
        moved = replace(state, pos_c=pos)
        result, _ = system.collider.evaluate_force(moved, system)
        return jnp.sum(weights * result.force)

    def full_width_observable(pos):
        moved = replace(state, pos_c=pos)
        force, _, _ = _full_width_reference(moved, system, False)
        return jnp.sum(weights * force)

    actual = jax.grad(blocked_observable)(state.pos_c)
    expected = jax.grad(full_width_observable)(state.pos_c)
    assert jnp.all(jnp.isfinite(actual))
    assert jnp.any(actual != 0)
    assert jnp.allclose(actual, expected, rtol=_RTOL, atol=_ATOL)


def test_default_budget_keeps_small_system_full_width(monkeypatch):
    monkeypatch.setattr(
        neighbor_list_module, "_FORCE_NEIGHBOR_PAIR_BUDGET", 8 * 1024 * 1024
    )
    jax.clear_caches()
    state, system = _system(64)
    expected_force, expected_torque, expected_history = _full_width_reference(
        state, system, True
    )

    actual_state, actual_system = system.collider.compute_force(state, system)

    assert jnp.allclose(actual_state.force, expected_force, rtol=_RTOL, atol=_ATOL)
    assert jnp.allclose(actual_state.torque, expected_torque, rtol=_RTOL, atol=_ATOL)
    assert jnp.array_equal(actual_system.collider.history, expected_history)
