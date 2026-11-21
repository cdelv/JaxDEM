# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Packing-fraction protocol, quasistatic compression, and force-coherence regressions."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.minimizers import TerminationReason
from jaxdem.minimizers.routines import minimize
from jaxdem.utils.dynamics_routines import run_packing_fraction_protocol
from jaxdem.utils.packing_utils import (
    CompressionReason,
    _scale_to_packing_fraction_grouped,
    compute_packing_fraction,
    quasistatic_compress_to_packing_fraction,
    scale_to_packing_fraction,
)
from jaxdem.utils.random_sphere_configuration import minimize_sphere_configuration


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


def _single_periodic():
    state = jd.State.create(
        pos=jnp.array([[0.5, 0.5]]), rad=jnp.array([0.1]), vel=jnp.array([[0.2, -0.1]])
    )
    system = jd.System.create(
        state=state,
        domain_type="periodic",
        domain_kw={"box_size": jnp.ones(2)},
        dt=0.01,
        rotation_integrator_type=None,
    )
    return jd.System.initialize(state, system)


@pytest.mark.parametrize("strides", [[], [0], [0, 1]])
def test_protocol_supports_zero_one_and_two_frames(strides):
    state, system = _system()
    state = replace(state, vel=jnp.array([[0.1, 0.2], [-0.1, 0.3]]))
    state, system = jd.System.initialize(
        state, replace(system, collider=jd.Collider.create(""))
    )
    phi = compute_packing_fraction(state, system)
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


def test_protocol_rejects_invalid_strides_and_unroll():
    state, system = _system()
    for strides, targets, message in (
        (jnp.array(1), jnp.array([0.1]), "1D integer"),
        (jnp.array([0.0]), jnp.array([0.1]), "1D integer"),
        (jnp.array([0]), jnp.array([[0.1]]), "1D array"),
        (jnp.array([0, 1]), jnp.array([0.1]), "same length"),
        (jnp.array([-1]), jnp.array([0.1]), "nonnegative"),
    ):
        with pytest.raises(ValueError, match=message):
            run_packing_fraction_protocol(
                state, system, strides=strides, phi_at_frames=targets
            )
    for unroll in (0, -1, True, 1.5):
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
    phi = compute_packing_fraction(state, system)
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
    phi = compute_packing_fraction(state, system)

    result, updated, _ = run_packing_fraction_protocol(
        state,
        system,
        strides=jnp.array([0], dtype=int),
        phi_at_frames=jnp.array([phi / 0.9**2]),
    )
    reevaluated, _ = jd.System.evaluate_forces(result, updated)

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


def test_compression_reports_reached_target_exhaustion_and_min_failure():
    state, system = _single_periodic()
    phi = float(compute_packing_fraction(state, system))

    reached = quasistatic_compress_to_packing_fraction(
        state, system, phi, max_n_outer_steps=0
    )
    assert reached.reason == CompressionReason.TARGET_REACHED

    exhausted = quasistatic_compress_to_packing_fraction(
        state, system, phi + 0.01, max_n_outer_steps=0
    )
    assert exhausted.reason == CompressionReason.MAX_STEPS
    assert exhausted.steps == 0
    assert len(tuple(exhausted)) == 4

    system_with_target = replace(system, target_fn=lambda st, _: jnp.sum(st.pos_c**2))
    failed = quasistatic_compress_to_packing_fraction(
        state,
        system_with_target,
        phi + 0.01,
        max_n_min_steps_per_outer=0,
        force_tol=0.0,
        torque_tol=0.0,
    )
    assert failed.reason == CompressionReason.MINIMIZATION_FAILED
    assert failed.minimizer_reason == TerminationReason.MAX_STEPS
    np.testing.assert_array_equal(
        failed.system.domain.box_size, system_with_target.domain.box_size
    )


def test_compression_rejects_later_failure_without_losing_the_accepted_state():
    state, system = _system(force_model=_RememberingSpring())
    state = replace(state, fixed=jnp.ones_like(state.fixed))
    state, system = jd.System.initialize(state, system)
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
    state, system = _single_periodic()
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


def test_sphere_relaxation_helper_uses_force_tolerance():
    pos = np.array([[0.0, 0.0], [0.9, 0.0]])
    relaxed, box = minimize_sphere_configuration(
        [0.5, 0.5], pos, phi=0.05, dim=2, force_tol=1e-10
    )
    delta = np.asarray(relaxed[1] - relaxed[0])
    delta = delta - np.asarray(box) * np.round(delta / np.asarray(box))
    assert np.linalg.norm(delta) >= 1.0 - 1e-8


def test_quasistatic_compression_forwards_mechanical_tolerances():
    xy = np.indices((3, 3)).reshape(2, -1).T * 1.1
    st = jd.State.create(pos=jnp.asarray(xy), rad=jnp.full(9, 0.5))
    mat = jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
    sy = jd.System.create(
        st.shape,
        collider_type="naive",
        dt=0.01,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(2, 3.3)},
        mat_table=jd.MaterialTable.from_materials([mat]),
        minimizer=jd.minimizers.fire,
    )
    target = float(compute_packing_fraction(st, sy)) + 0.01
    st, sy, phi, pe = quasistatic_compress_to_packing_fraction(
        st,
        sy,
        target,
        step=0.005,
        force_tol=1e-11,
        torque_tol=1e-11,
        max_n_min_steps_per_outer=2,
        max_n_outer_steps=5,
    )
    assert float(phi) == pytest.approx(target)
    checked = minimize(st, sy, max_steps=0, force_tol=1e-11, torque_tol=1e-11)
    assert checked.info.converged
    np.testing.assert_allclose(pe, checked[3])
