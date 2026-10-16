"""Analytical union mass properties, without assumptions about quadrature order."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxdem.utils.clumps import _compute_uniform_union_properties


@pytest.mark.parametrize("dim,radius", [(2, 1.0), (3, 1.7)])
def test_duplicate_spheres_preserve_volume_mass_properties_and_world_geometry(
    dim, radius
):
    center = np.arange(1, dim + 1, dtype=float)
    pos = np.broadcast_to(center, (2, dim))
    mass = 2.3
    volume, com, moments, q, body = _compute_uniform_union_properties(
        jnp.asarray(pos),
        jnp.full(2, radius),
        jnp.asarray(mass),
        n_samples=32768,
    )
    expected_volume = np.pi * radius**2 if dim == 2 else 4 * np.pi * radius**3 / 3
    expected_moment = (0.5 if dim == 2 else 0.4) * mass * radius**2
    # Fixed error budgets for bounded geometric integration, not a fitted slope.
    np.testing.assert_allclose(volume, expected_volume, rtol=0.01)
    np.testing.assert_allclose(com, center, atol=0.01 * radius)
    np.testing.assert_allclose(moments, expected_moment, rtol=0.02)
    w, x, y, z = np.asarray(q)
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    rotation = np.eye(3) + 2 * w * skew + 2 * skew @ skew
    np.testing.assert_allclose(np.sum(np.asarray(q) ** 2), 1.0, atol=2e-6)
    reconstructed = np.asarray(body) @ rotation[:dim, :dim].T + np.asarray(com)
    np.testing.assert_allclose(reconstructed, pos, atol=2e-6)


@pytest.mark.parametrize("dim", [2, 3])
def test_overlapping_unit_spheres_have_analytical_union_volume_and_centroid(dim):
    # Centers separated by R=1: subtract the known circular lens / spherical lens.
    pos = np.zeros((2, dim))
    pos[:, 0] = [-0.5, 0.5]
    volume, center, _, _, _ = _compute_uniform_union_properties(
        jnp.asarray(pos),
        jnp.ones(2),
        jnp.asarray(1.0),
        n_samples=32768,
    )
    expected = 4 * np.pi / 3 + np.sqrt(3) / 2 if dim == 2 else 9 * np.pi / 4
    np.testing.assert_allclose(volume, expected, rtol=0.01)
    np.testing.assert_allclose(center, np.zeros(dim), atol=0.01)
