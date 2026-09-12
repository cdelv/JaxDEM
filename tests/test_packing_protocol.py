# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Packing-fraction protocol validation and force-coherence regressions."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import pytest

import jaxdem as jd
from jaxdem.utils.dynamics_routines import run_packing_fraction_protocol
from jaxdem.utils.packing_utils import scale_to_packing_fraction


def _system(*, force_model=None, integrator=None):
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.8, 0.0]]),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
    )
    system = jd.System.create(
        state=state,
        dt=0.01,
        domain_type="periodic",
        domain_kw={"box_size": jnp.array([10.0, 10.0])},
        force_model=force_model,
        linear_integrator=integrator,
        collider_type="NeighborList" if force_model is not None else "CellList",
        collider_kw=(
            {
                "cutoff": 1.1,
                "skin": 0.2,
                "max_neighbors": 2,
                "secondary_collider_type": "naive",
            }
            if force_model is not None
            else {"state": state}
        ),
    )
    return jd.System.initialize(state, system)


@pytest.mark.parametrize("strides", [[], [0], [0, 1]])
def test_protocol_supports_zero_one_and_two_frames(strides):
    state, system = _system()
    phi = jd.utils.compute_packing_fraction(state, system)
    targets = jnp.full((len(strides),), phi)

    final_state, final_system, (states, systems) = run_packing_fraction_protocol(
        state, system, strides=jnp.asarray(strides, dtype=int), phi_at_frames=targets
    )

    assert states.pos_c.shape[0] == len(strides)
    assert systems.time.shape == (len(strides),)
    assert int(final_system.step_count) == sum(strides)
    assert jnp.all(jnp.isfinite(final_state.force))


@pytest.mark.parametrize(
    ("strides", "targets", "message"),
    [
        (jnp.array(1), jnp.array([0.1]), "1D integer"),
        (jnp.array([0.0]), jnp.array([0.1]), "1D integer"),
        (jnp.array([0]), jnp.array([[0.1]]), "1D array"),
        (jnp.array([0, 1]), jnp.array([0.1]), "same length"),
        (jnp.array([-1]), jnp.array([0.1]), "nonnegative"),
    ],
)
def test_protocol_rejects_invalid_stride_shapes_and_values(strides, targets, message):
    state, system = _system()
    with pytest.raises(ValueError, match=message):
        run_packing_fraction_protocol(
            state, system, strides=strides, phi_at_frames=targets
        )


@pytest.mark.parametrize("unroll", [0, -1, True, 1.5])
def test_protocol_rejects_invalid_unroll(unroll):
    state, system = _system()
    with pytest.raises(ValueError, match="positive Python integer"):
        run_packing_fraction_protocol(
            state,
            system,
            strides=jnp.array([], dtype=int),
            phi_at_frames=jnp.array([]),
            unroll=unroll,
        )


def test_protocol_matches_explicit_step_scale_and_evaluate_sequence():
    state, system = _system()
    phi = jd.utils.compute_packing_fraction(state, system)
    strides = jnp.array([0, 2], dtype=int)
    targets = jnp.array([phi / 0.95**2, phi / 0.9**2])

    actual_state, actual_system, _ = run_packing_fraction_protocol(
        state, system, strides=strides, phi_at_frames=targets
    )

    expected_state, expected_system = state, system
    for stride, target in zip(strides, targets, strict=True):
        expected_state, expected_system = jd.System.step_dynamic(
            expected_state, expected_system, n=stride
        )
        expected_state, expected_system = scale_to_packing_fraction(
            expected_state, expected_system, target
        )
        expected_state, expected_system = jd.System.evaluate_forces(
            expected_state, expected_system
        )

    for actual, expected in (
        (actual_state.pos_c, expected_state.pos_c),
        (actual_state.vel, expected_state.vel),
        (actual_state.force, expected_state.force),
        (actual_system.domain.box_size, expected_system.domain.box_size),
        (actual_system.time, expected_system.time),
        (actual_system.step_count, expected_system.step_count),
    ):
        assert jnp.allclose(actual, expected)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _RememberingSpring(jd.forces.SpringForce):
    def history_shape(self, dim):
        return (1,)

    @staticmethod
    def force(i, j, pos, state, system, history, *, advance_history=True):
        force, torque, _ = jd.forces.SpringForce.force(
            i, j, pos, state, system, history, advance_history=advance_history
        )
        return force, torque, history + advance_history


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _InitializationMarker(jd.LinearIntegrator):
    @staticmethod
    def initialize(state, system):
        return replace(state, vel=state.vel + 7.0), system


def test_rescale_refresh_preserves_history_queued_load_and_integrator_state():
    state, system = _system(
        force_model=_RememberingSpring(), integrator=_InitializationMarker()
    )
    state, system = jd.System.step(state, system)
    queued = jnp.array([[3.0, 4.0], [-2.0, 1.0]])
    system = system.force_manager.add_force(state, system, queued)
    old_history = system.collider.history
    old_velocity = state.vel
    phi = jd.utils.compute_packing_fraction(state, system)

    result, updated, _ = run_packing_fraction_protocol(
        state,
        system,
        strides=jnp.array([0], dtype=int),
        phi_at_frames=jnp.array([phi / 0.9**2]),
    )
    reevaluated, _ = jd.System.evaluate_forces(result, updated)

    assert jnp.array_equal(updated.collider.history, old_history)
    assert jnp.array_equal(updated.force_manager.external_force, queued)
    assert jnp.array_equal(result.vel, old_velocity)
    assert jnp.allclose(result.force, reevaluated.force)
