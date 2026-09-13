"""Failure counterexamples for the optional checked simulation drivers."""

from dataclasses import replace

import jax.numpy as jnp
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
    state = jdem.State.create(pos=jnp.array([[0.5, 0.5]]), rad=jnp.array([0.1]))
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
    np.testing.assert_array_equal(result.system.key, expected_system.key)
    assert not bool(result.system.search_overflow)
    with pytest.raises(RuntimeError, match="Checked simulation stopped"):
        result.check()


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
    result = jdem.System.step_checked(state, system, n=0)
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
