# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Failure and rollback checks for packing compression."""

import jax.numpy as jnp
import jax

from dataclasses import replace

import numpy as np

import jaxdem as jdem
from jaxdem.minimizers import TerminationReason
from jaxdem.utils.packing_utils import (
    CompressionReason,
    compute_packing_fraction,
    quasistatic_compress_to_packing_fraction,
)


def _setup():
    state = jdem.State.create(
        pos=jnp.array([[0.5, 0.5]]), rad=jnp.array([0.1]), vel=jnp.array([[0.2, -0.1]])
    )
    system = jdem.System.create(
        state=state,
        domain_type="periodic",
        domain_kw={"box_size": jnp.ones(2)},
        dt=0.01,
        rotation_integrator_type=None,
    )
    return jdem.System.initialize(state, system)


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
