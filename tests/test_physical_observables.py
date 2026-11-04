# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Known displacement and rotation trajectories through the shipped kernels."""

import jax
import jax.numpy as jnp

import numpy as np
import pytest
from scipy.special import j0, j1

from jaxdem.analysis import LagBinsExact, evaluate_binned
from jaxdem.analysis import kernels
from jaxdem.analysis.bessel import j0 as jax_j0


@pytest.mark.parametrize("dim", [2, 3])
def test_ballistic_observables_distinguish_internal_motion_and_drift(dim):
    velocity = np.zeros((2, 2, dim))
    velocity[:, :, 0] = [[1.7, -0.3], [2.7, -1.3]]
    velocity[:, :, 1] = 0.3
    arrays = {"pos": jnp.asarray(np.arange(4)[:, None, None, None] * velocity)}
    result = evaluate_binned(kernels.msd_kernel, arrays, LagBinsExact(4, [1, 2]))
    np.testing.assert_allclose(result.mean, [[1, 4], [4, 16]], atol=1e-6)
    np.testing.assert_array_equal(result.counts, [3, 2])
    k = np.array([0.0, 4.9, 5.1, 10.0])
    radius = np.linalg.norm(velocity, axis=-1)
    argument = radius[:, :, None] * k
    expected = j0(argument) if dim == 2 else np.sinc(argument / np.pi)
    np.testing.assert_allclose(
        kernels.isf_self_isotropic_kernel(arrays, 0, 1, k=jnp.asarray(k)),
        expected.mean(axis=1),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        kernels.isf_self_isotropic_kernel(arrays, 0, 0, k=2.0), 1.0
    )
    kvecs = np.zeros((3, dim))
    kvecs[:, 0] = [0, np.pi / 2, np.pi]
    expected = np.cos(velocity @ kvecs.T).mean(axis=1)
    np.testing.assert_allclose(
        kernels.isf_self_kvecs_kernel(arrays, 0, 1, kvecs=jnp.asarray(kvecs)),
        expected,
        atol=1e-6,
    )
    x = jnp.array([0.0, 4.99, 5.01, 11.0])
    np.testing.assert_allclose(jax_j0(x), j0(np.asarray(x)), atol=1e-6)
    np.testing.assert_allclose(
        jax.vmap(jax.grad(jax_j0))(x), -j1(np.asarray(x)), atol=2e-6
    )


def test_rotational_observables_unwrap_branch_crossings_and_quaternion_signs():
    angle = np.deg2rad([170, 210, 260, 310, 370])[:, None] + np.array([0.0, 0.2])
    sign = np.array([1, -1, 1, -1, 1])[:, None]
    w = jnp.asarray((sign * np.cos(angle / 2))[..., None])
    xyz = np.zeros((5, 2, 3))
    xyz[..., 2] = sign * np.sin(angle / 2)
    unwrapped = kernels.unwrap_angles_2d(w, jnp.asarray(xyz))
    np.testing.assert_allclose(unwrapped, angle, atol=2e-6)
    arrays = {"theta": unwrapped}
    delta = np.deg2rad(200)
    np.testing.assert_allclose(
        kernels.msad_kernel_2d(arrays, 0, 4), delta**2, rtol=1e-6
    )
    np.testing.assert_allclose(
        kernels.isf_angular_kernel_2d(arrays, 0, 4, theta_0=jnp.array([0.0, 1.0, 2.0])),
        np.cos(np.arange(3) * delta),
        atol=1e-6,
    )
