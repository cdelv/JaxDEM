from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import pytest

import jaxdem as jdem
from jaxdem.minimizers import TerminationReason
from jaxdem.utils.jamming import (
    JamReason,
    bisection_jam,
    pe_band_jam,
    pressure_bisection_jam,
)


def _nonfinite_system():
    state = jdem.State.create(
        pos=jnp.asarray([[jnp.nan, 0.0], [2.0, 0.0]]),
        rad=jnp.asarray([0.2, 0.2]),
    )
    return state, jdem.System.create(state=state, collider_type="naive")


@pytest.mark.parametrize("driver", [bisection_jam, pe_band_jam, pressure_bisection_jam])
def test_jamming_reports_nonfinite_minimization_without_retries(driver):
    state, system = _nonfinite_system()
    result = driver(
        state,
        system,
        n_minimization_steps=2,
        n_jamming_steps=3,
        verbose=False,
    )

    assert result.reason == JamReason.NONFINITE
    assert result.minimizer_reason == TerminationReason.NONFINITE
    assert int(result.steps) == 0
    assert len(result) == 6
    assert tuple(result)[2] is result.jammed_state


def test_jamming_outer_step_limit_has_explicit_reason():
    state = jdem.State.create(
        pos=jnp.asarray([[0.0, 0.0], [2.0, 0.0]]),
        rad=jnp.asarray([0.2, 0.2]),
    )
    system = jdem.System.create(state=state, collider_type="naive")
    result = pe_band_jam(
        state,
        system,
        n_minimization_steps=1,
        n_jamming_steps=0,
        verbose=False,
    )
    assert result.reason == JamReason.MAX_STEPS


def test_pressure_nan_is_not_accepted_as_target(monkeypatch):
    state = jdem.State.create(
        pos=jnp.asarray([[0.0, 0.0], [2.0, 0.0]]), rad=jnp.asarray([0.2, 0.2])
    )
    system = jdem.System.create(state=state, collider_type="naive")

    def nan_pressure(state, system, *_args):
        return state, system, jnp.asarray(jnp.nan)

    monkeypatch.setattr("jaxdem.utils.jamming.compute_contact_pressure", nan_pressure)
    result = pressure_bisection_jam(
        state, system, n_minimization_steps=1, n_jamming_steps=2, verbose=False
    )
    assert result.reason == JamReason.NONFINITE


def test_final_minimizer_failure_takes_priority_over_outer_status(monkeypatch):
    state = jdem.State.create(
        pos=jnp.asarray([[0.0, 0.0], [2.0, 0.0]]), rad=jnp.asarray([0.2, 0.2])
    )
    system = jdem.System.create(state=state, collider_type="naive")
    original = type(system).minimize
    calls = 0

    def staged_minimize(state, system, **kwargs):
        nonlocal calls
        calls += 1
        result = original(state, system, **kwargs)
        if calls == 3:
            return replace(
                result,
                energy=jnp.asarray(jnp.nan),
                reason=jnp.asarray(TerminationReason.NONFINITE),
            )
        return result

    def target_pressure(state, system, *_args):
        return state, system, jnp.asarray(1.005e-7)

    monkeypatch.setattr(type(system), "minimize", staticmethod(staged_minimize))
    monkeypatch.setattr(
        "jaxdem.utils.jamming.compute_contact_pressure", target_pressure
    )
    result = pressure_bisection_jam(
        state, system, n_minimization_steps=1, n_jamming_steps=2, verbose=False
    )
    assert result.reason == JamReason.NONFINITE
    assert result.minimizer_reason == TerminationReason.NONFINITE


def test_initial_minimizer_failure_rolls_back_to_input(monkeypatch):
    state = jdem.State.create(
        pos=jnp.asarray([[0.0, 0.0], [2.0, 0.0]]), rad=jnp.asarray([0.2, 0.2])
    )
    system = jdem.System.create(state=state, collider_type="naive")
    original = type(system).minimize

    def failed_minimize(state, system, **kwargs):
        result = original(state, system, **kwargs)
        bad_state = replace(
            result.state, pos_c=jnp.full_like(result.state.pos_c, jnp.nan)
        )
        return replace(
            result,
            state=bad_state,
            energy=jnp.asarray(jnp.nan),
            reason=jnp.asarray(TerminationReason.NONFINITE),
        )

    monkeypatch.setattr(type(system), "minimize", staticmethod(failed_minimize))
    result = pressure_bisection_jam(
        state, system, n_minimization_steps=1, n_jamming_steps=2, verbose=False
    )
    assert result.jammed_state is state
    assert result.jammed_system is system
    assert jnp.isnan(result.potential_energy)
