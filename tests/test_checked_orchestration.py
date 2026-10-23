# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Failure counterexamples for the optional checked simulation drivers."""

import jax.numpy as jnp
import jax

from dataclasses import replace

import numpy as np
import pytest

import jaxdem as jdem
from jaxdem.minimizers import TerminationReason
from jaxdem.utils.packing_utils import (
    CompressionReason,
    compute_packing_fraction,
    quasistatic_compress_to_packing_fraction,
)


def _nonfinite_second_step(state, system):
    state = replace(state, vel=jnp.where(system.step_count == 2, jnp.nan, state.vel))
    return state, system


def _overflow_second_step(state, system):
    return state, replace(system, search_overflow=system.step_count == 2)


def _setup(callback=None):
    state = jdem.State.create(
        pos=jnp.array([[0.5, 0.5]]), rad=jnp.array([0.1]), vel=jnp.array([[0.2, -0.1]])
    )
    kwargs = {} if callback is None else {"user_post_step_actions": callback}
    system = jdem.System.create(
        state=state,
        domain_type="periodic",
        domain_kw={"box_size": jnp.ones(2)},
        dt=0.01,
        rotation_integrator_type=None,
        **kwargs,
    )
    return jdem.System.initialize(state, system)


@pytest.mark.parametrize(
    "callback,status",
    [
        (_nonfinite_second_step, jdem.SimulationStatus.NONFINITE),
        (_overflow_second_step, jdem.SimulationStatus.SEARCH_OVERFLOW),
    ],
)
def test_checked_step_rolls_back_and_does_not_lose_failure(callback, status):
    state, system = _setup(callback)
    expected_state, expected_system = jdem.System.step(state, system)
    result = jdem.System.step_checked(state, system, n=4)
    assert int(result.status) == status
    assert int(result.steps) == 1
    assert int(result.system.step_count) == 1
    np.testing.assert_array_equal(result.state.pos, expected_state.pos)
    assert np.any(np.asarray(result.state.pos) != np.asarray(state.pos))
    np.testing.assert_array_equal(result.system.key, expected_system.key)
    assert not bool(result.system.search_overflow)
    with pytest.raises(RuntimeError, match="Checked simulation stopped"):
        result.check()


def _mixed_second_step_failure(state, system):
    system = system.force_manager.add_force(
        state, system, jnp.full_like(state.force, 0.1 * system.step_count)
    )
    fail = system.step_count == 2
    state = replace(
        state, vel=jnp.where(fail & (state.mass[0] == 2), jnp.nan, state.vel)
    )
    return state, replace(system, search_overflow=fail & (state.mass[0] == 3))


def test_checked_batch_restores_history_randomness_and_queued_loads_per_replica():
    import jax
    from tests.test_sparse_neighbors import Remember
    from tests.test_checkpoints import _assert_leaves_equal

    pairs = []
    for mass in [1.0, 2.0, 3.0]:
        state = jdem.State.create(
            pos=jnp.array([[0.2, 0.2], [0.5, 0.25]]),
            vel=jnp.array([[0.1, -0.2], [-0.1, 0.2]]),
            mass=jnp.full(2, mass),
            rad=jnp.full(2, 0.2),
        )
        table = jdem.MaterialTable.from_materials(
            [jdem.Material.create("elastic", young=1.0, poisson=0.25, density=1.0)]
        )
        system = jdem.System.create(
            state=state,
            dt=0.01,
            seed=int(mass),
            mat_table=table,
            force_model=Remember(),
            collider_type="NeighborList",
            collider_kw={"cutoff": 0.4, "skin": 0.1, "max_neighbors": 1},
            linear_integrator_type="langevin",
            linear_integrator_kw={"gamma": 0.3, "temperature": 0.5, "k_B": 1.0},
            user_post_step_actions=_mixed_second_step_failure,
        )
        state, system = jdem.System.initialize(state, system)
        system = system.force_manager.add_force(
            state, system, jnp.full_like(state.force, 0.2)
        )
        pairs.append((state, system))
    state = jdem.State.stack([s for s, _ in pairs])
    system = jdem.System.stack([y for _, y in pairs])
    result = jdem.System.step_checked(state, system, n=3)
    np.testing.assert_array_equal(result.steps, [3, 1, 1])
    np.testing.assert_array_equal(
        result.status,
        [0, jdem.SimulationStatus.NONFINITE, jdem.SimulationStatus.SEARCH_OVERFLOW],
    )
    for index, (s, y) in enumerate(pairs):
        expected = jdem.System.step(s, y, n=3 if index == 0 else 1)
        actual = jax.tree.map(lambda x: x[index], (result.state, result.system))
        _assert_leaves_equal(
            expected, actual, "last valid replica", rtol=2e-6, atol=1e-7
        )
        assert not np.array_equal(np.asarray(actual[1].key), np.asarray(y.key))
        assert np.any(
            np.asarray(actual[1].collider.history) != np.asarray(y.collider.history)
        )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_checked_batch_and_valid_step_match_unchecked(batch_size):
    state, system = _setup()
    states = jdem.State.stack([state] * batch_size)
    systems = jdem.System.stack([system] * batch_size)
    expected, _ = jdem.System.step(states, systems, n=3)
    result = jdem.System.step_checked(states, systems, n=3)
    np.testing.assert_array_equal(result.state.pos, expected.pos)
    np.testing.assert_array_equal(result.steps, jnp.full((batch_size,), 3))
    result.check()


def test_invalid_initial_state_is_not_advanced():
    state, system = _setup()
    state = replace(state, force=jnp.full_like(state.force, jnp.inf))
    result = jdem.System.step_checked(state, system, n=3)
    assert int(result.status) == jdem.SimulationStatus.NONFINITE
    assert int(result.steps) == 0


def test_real_capacity_overflow_is_reported_before_a_step():
    state = jdem.State.create(pos=jnp.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]))
    system = jdem.System.create(
        state=state,
        collider_type="neighborlist",
        collider_kw={"cutoff": 2.0, "max_neighbors": 1},
    )
    state, system = jdem.System.initialize(state, system)
    result = jdem.System.step_checked(state, system, n=3)
    assert int(result.status) & jdem.SimulationStatus.SEARCH_OVERFLOW
    assert int(result.steps) == 0


def test_compression_reports_exhaustion_instead_of_claiming_target():
    state, system = _setup()
    phi = float(compute_packing_fraction(state, system))
    result = quasistatic_compress_to_packing_fraction(
        state,
        system,
        phi + 0.01,
        max_n_outer_steps=0,
    )
    assert result.reason == CompressionReason.MAX_STEPS
    assert result.steps == 0
    assert len(tuple(result)) == 4
    system = replace(system, target_fn=lambda st, _: jnp.sum(st.pos_c**2))
    failed = quasistatic_compress_to_packing_fraction(
        state,
        system,
        phi + 0.01,
        max_n_min_steps_per_outer=0,
        force_tol=0.0,
        torque_tol=0.0,
    )
    assert failed.reason == CompressionReason.MINIMIZATION_FAILED
    assert failed.minimizer_reason == TerminationReason.MAX_STEPS
    np.testing.assert_array_equal(failed.system.domain.box_size, system.domain.box_size)


def test_compression_reports_reached_target():
    state, system = _setup()
    phi = float(compute_packing_fraction(state, system))
    result = quasistatic_compress_to_packing_fraction(
        state,
        system,
        phi,
        max_n_outer_steps=0,
    )
    assert result.reason == CompressionReason.TARGET_REACHED


def test_compression_rejects_later_failure_without_losing_the_accepted_state():
    from tests.test_packing_protocol import _system, _RememberingSpring

    state, system = _system(force_model=_RememberingSpring())
    state = replace(state, fixed=jnp.ones_like(state.fixed))
    state, system = jdem.System.initialize(state, system)
    offsets = np.asarray(system.collider.row_offsets)
    history = (
        system.collider.history.at[offsets[0], 0].set(11).at[offsets[1], 0].set(29)
    )
    system = replace(
        system,
        collider=replace(system.collider, history=history),
        target_fn=lambda st, sy: jnp.where(sy.domain.box_size[0] < 9.3, jnp.nan, 0.0),
    )
    queued = jnp.array([[1.0, 2.0], [-3.0, 4.0]])
    system = system.force_manager.add_force(state, system, queued)
    phi = float(compute_packing_fraction(state, system))
    options = dict(step=phi / 10, max_n_min_steps_per_outer=0)
    accepted = quasistatic_compress_to_packing_fraction(
        state, system, phi * 1.3, max_n_outer_steps=1, **options
    )
    failed = quasistatic_compress_to_packing_fraction(
        state, system, phi * 1.3, max_n_outer_steps=3, **options
    )
    assert accepted.steps == failed.steps == 1
    assert failed.reason == CompressionReason.MINIMIZATION_FAILED
    assert failed.minimizer_reason == TerminationReason.NONFINITE
    np.testing.assert_allclose(
        failed.system.domain.box_size, np.full(2, 10 / np.sqrt(1.1))
    )
    valid = failed.system.collider.neighbor_list >= 0
    np.testing.assert_array_equal(failed.system.collider.neighbor_list[valid], [1, 0])
    np.testing.assert_array_equal(failed.system.collider.history[valid], [[12], [30]])
    np.testing.assert_array_equal(failed.system.force_manager.external_force, queued)
    assert jax.tree.structure((accepted.state, accepted.system)) == jax.tree.structure(
        (failed.state, failed.system)
    )
    for expected, actual in zip(
        jax.tree.leaves((accepted.state, accepted.system)),
        jax.tree.leaves((failed.state, failed.system)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_compression_scaling_respects_nonzero_domain_anchor():
    from jaxdem.utils.packing_utils import _scale_to_packing_fraction_grouped

    state, system = _setup()
    phi = compute_packing_fraction(state, system)
    anchor = jnp.array([3.0, -4.0])
    shifted_state = replace(state, pos_c=state.pos_c + anchor)
    shifted_system = replace(system, domain=replace(system.domain, anchor=anchor))
    group_id = jnp.array([0], dtype=jnp.int32)
    scaled, _ = _scale_to_packing_fraction_grouped(state, system, phi / 4, group_id)
    shifted_scaled, _ = _scale_to_packing_fraction_grouped(
        shifted_state, shifted_system, phi / 4, group_id
    )
    np.testing.assert_allclose(shifted_scaled.pos_c - anchor, scaled.pos_c, atol=1e-6)
