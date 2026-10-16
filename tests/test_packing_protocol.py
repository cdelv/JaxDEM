# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Packing-fraction protocol validation and force-coherence regressions."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
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
        mat_table=jd.MaterialTable.from_materials(
            [jd.Material.create("elastic", young=1.0, poisson=0.25, density=1.0)]
        ),
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
    state = replace(state, vel=jnp.array([[0.1, 0.2], [-0.1, 0.3]]))
    state, system = jd.System.initialize(
        state, replace(system, collider=jd.Collider.create(""))
    )
    phi = jd.utils.compute_packing_fraction(state, system)
    targets = jnp.full((len(strides),), phi)

    final_state, final_system, (states, systems) = run_packing_fraction_protocol(
        state, system, strides=jnp.asarray(strides, dtype=int), phi_at_frames=targets
    )

    assert states.pos_c.shape[0] == len(strides)
    assert systems.time.shape == (len(strides),)
    assert int(final_system.step_count) == sum(strides)
    times = np.cumsum(strides) * float(system.dt)
    expected = np.asarray(state.pos_c)[None] + times[:, None, None] * np.asarray(
        state.vel
    )
    np.testing.assert_allclose(states.pos_c, expected, atol=1e-7)
    np.testing.assert_allclose(systems.time, times, atol=1e-7)
    np.testing.assert_array_equal(systems.step_count, np.cumsum(strides))
    np.testing.assert_allclose(
        systems.domain.box_size, np.full((len(strides), 2), 10.0)
    )
    np.testing.assert_array_equal(states.force, np.zeros((len(strides), 2, 2)))
    np.testing.assert_allclose(
        final_state.pos_c, expected[-1] if strides else state.pos_c
    )
    np.testing.assert_array_equal(final_state.vel, state.vel)
    if not strides:
        from tests.test_checkpoints import _assert_leaves_equal

        _assert_leaves_equal(
            (state, system), (final_state, final_system), "empty protocol"
        )


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

    actual_state, actual_system, (frames, frame_systems) = (
        run_packing_fraction_protocol(
            state, system, strides=strides, phi_at_frames=targets
        )
    )
    np.testing.assert_allclose(frames.pos_c[0], 0.95 * state.pos_c, atol=1e-7)
    np.testing.assert_allclose(frame_systems.domain.box_size[0], [9.5, 9.5])
    np.testing.assert_allclose(frames.force[0], [[-0.24, 0.0], [0.24, 0.0]], rtol=2e-6)

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
    def force(
        i, j, pos, state, system, history, *, advance_history=True, quasistatic=False
    ):
        force, torque, _ = jd.forces.SpringForce.force(
            i, j, pos, state, system, history
        )
        return force, torque, history + advance_history


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _InitializationMarker(jd.LinearIntegrator):
    @staticmethod
    def initialize(state, system):
        return replace(state, vel=state.vel + 7.0), system


def test_rescale_advances_history_once_and_preserves_load_and_integrator_state():
    state, system = _system(
        force_model=_RememberingSpring(), integrator=_InitializationMarker()
    )
    state, system = jd.System.step(state, system)
    queued = jnp.array([[3.0, 4.0], [-2.0, 1.0]])
    system = system.force_manager.add_force(state, system, queued)
    offsets = np.asarray(system.collider.row_offsets)
    np.testing.assert_array_equal(np.diff(offsets), [1, 1])
    history = (
        system.collider.history.at[offsets[0], 0].set(11).at[offsets[1], 0].set(29)
    )
    system = replace(system, collider=replace(system.collider, history=history))
    old_velocity = state.vel
    phi = jd.utils.compute_packing_fraction(state, system)

    result, updated, _ = run_packing_fraction_protocol(
        state,
        system,
        strides=jnp.array([0], dtype=int),
        phi_at_frames=jnp.array([phi / 0.9**2]),
    )
    reevaluated, _ = jd.System.evaluate_forces(result, updated)

    # The affine rescale advances pair history exactly once; the force
    # refresh afterwards does not.
    valid = updated.collider.neighbor_list >= 0
    np.testing.assert_array_equal(updated.collider.neighbor_list[valid], [1, 0])
    np.testing.assert_array_equal(updated.collider.history[valid], [[12], [30]])
    assert jnp.array_equal(updated.force_manager.external_force, queued)
    assert jnp.array_equal(result.vel, old_velocity)
    assert jnp.allclose(result.force, reevaluated.force)
    np.testing.assert_allclose(updated.domain.box_size, [9.0, 9.0])
    np.testing.assert_allclose(result.pos_c, 0.9 * state.pos_c, atol=1e-7)
    np.testing.assert_allclose(
        result.force - queued, [[-0.28, 0.0], [0.28, 0.0]], rtol=2e-6
    )
