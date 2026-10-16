# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Regression tests for zero-safe Euclidean norms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxdem.utils.linalg import norm, unit_and_norm


def test_tiny_nonzero_vectors_keep_their_norm_and_direction() -> None:
    vectors = jnp.asarray(
        [
            [3.0e-10, 4.0e-10, 0.0],
            [-2.0e-11, 4.0e-11, -8.0e-11],
        ]
    )
    expected_norms = np.array([5e-10, np.sqrt(84) * 1e-11])
    expected_units = np.array([[0.6, 0.8, 0], [-2, 4, -8] / np.sqrt(84)])

    units, lengths = unit_and_norm(vectors)

    np.testing.assert_allclose(norm(vectors), expected_norms, rtol=2e-6, atol=0.0)
    np.testing.assert_allclose(lengths, expected_norms, rtol=2e-6, atol=0.0)
    np.testing.assert_allclose(units, expected_units, rtol=2e-6, atol=0.0)
    np.testing.assert_allclose(
        jnp.linalg.norm(units, axis=-1), jnp.ones(2), rtol=2e-6, atol=0.0
    )


def test_surface_probe_orientation_has_a_finite_antipodal_pullback():
    from jaxdem.utils.surface_properties import _quat_from_x_to_3d

    for direction in (
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([-1.0, 0.0, 0.0]),
        jnp.array([0.0, 0.6, 0.8]),
    ):
        q = _quat_from_x_to_3d(direction)
        w, x, y, z = np.asarray(q)
        rotated_x = np.array(
            [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]
        )
        np.testing.assert_allclose(rotated_x, direction, atol=1e-6)
        assert np.isfinite(jax.jacrev(_quat_from_x_to_3d)(direction)).all()


def test_zero_norm_outputs_and_reverse_mode_derivatives_are_finite() -> None:
    zero = jnp.zeros(3)
    unit_vector, length = unit_and_norm(zero)

    np.testing.assert_array_equal(norm(zero), 0.0)
    np.testing.assert_array_equal(unit_vector, jnp.zeros_like(zero))
    np.testing.assert_array_equal(length, 0.0)

    norm_gradient = jax.grad(norm)(zero)
    length_gradient = jax.grad(lambda value: unit_and_norm(value)[1])(zero)
    unit_jacobian = jax.jacrev(lambda value: unit_and_norm(value)[0])(zero)

    assert jnp.all(jnp.isfinite(norm_gradient))
    assert jnp.all(jnp.isfinite(length_gradient))
    assert jnp.all(jnp.isfinite(unit_jacobian))
    np.testing.assert_array_equal(norm_gradient, jnp.zeros_like(zero))
    np.testing.assert_array_equal(length_gradient, jnp.zeros_like(zero))
    np.testing.assert_array_equal(unit_jacobian, jnp.zeros((3, 3)))
