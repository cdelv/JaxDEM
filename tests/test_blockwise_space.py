# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Block-diagonal Jacobians and independent marginal entropy integrals."""

import jax
import jax.numpy as jnp

import json

import distrax
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammainc

from jaxdem.rl.action_spaces import (
    ActionSpace,
    BlockwiseSpace,
    BoxSpace,
    FreeSpace,
    MaxNormSpace,
    Transformed,
)
from tests.test_action_spaces import isotropic_radial_entropy_reference


@pytest.mark.parametrize("nested", [False, True])
def test_blockwise_transform_inverse_and_block_diagonal_jacobian(nested):
    blocks = (
        (3, MaxNormSpace(max_norm=1.7)),
        (3, MaxNormSpace(max_norm=0.4)),
        (2, BoxSpace(-2.0, 3.0, width=0.7)),
        (1, FreeSpace()),
    )
    if nested:
        blocks = ((6, BlockwiseSpace(blocks[:2])), *blocks[2:])
    space = BlockwiseSpace(blocks)
    x = jnp.linspace(-0.6, 0.8, 9)
    y, log_det = jax.jit(lambda b, x: b.forward_and_log_det(x))(space, x)
    restored, inverse_log_det = jax.jit(space.inverse_and_log_det)(y)

    def radial(v, limit):
        r = np.linalg.norm(v)
        return (
            limit * (1 - 1e-6) * gammainc(v.size / 2, r * r / 2) ** (1 / v.size) * v / r
        )

    expected = np.concatenate(
        [
            radial(np.asarray(x[:3]), 1.7),
            radial(np.asarray(x[3:6]), 0.4),
            0.5 + 2.5 * (1 - 1e-6) * np.tanh(np.asarray(x[6:8]) / 0.7),
            np.asarray(x[8:]),
        ]
    )
    np.testing.assert_allclose(y, expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(restored, x, rtol=3e-6, atol=1e-6)
    np.testing.assert_allclose(inverse_log_det, -log_det, rtol=3e-6, atol=1e-6)
    jacobian = jax.jacfwd(space.forward)(x)
    np.testing.assert_allclose(
        log_det, np.linalg.slogdet(jacobian)[1], rtol=3e-6, atol=1e-6
    )
    for start, stop in [(0, 3), (3, 6), (6, 8), (8, 9)]:
        np.testing.assert_array_equal(jacobian[start:stop, :start], 0)
        np.testing.assert_array_equal(jacobian[start:stop, stop:], 0)


def test_batched_blockwise_entropy_and_gradients_against_marginal_integrals():
    std, width, eps = 0.3, 1.2, 1e-6
    space = BlockwiseSpace(
        (
            (3, MaxNormSpace(max_norm=1.7)),
            (9, MaxNormSpace(max_norm=0.4)),
            (2, BoxSpace(-2.0, 3.0, width=width)),
            (1, FreeSpace()),
        )
    )
    normal_entropy = 0.5 * np.log(2 * np.pi * np.e) + np.log(std)

    def expectation(function):
        return quad(
            lambda z: function(z) * np.exp(-z * z / 2) / np.sqrt(2 * np.pi), -12, 12
        )[0]

    box_entropy = normal_entropy + np.log(2.5 * (1 - eps) / width)
    box_entropy += expectation(
        lambda z: (
            2 * (np.log(2) - std * z / width - np.logaddexp(0, -2 * std * z / width))
        )
    )
    box_gradient = 1 + expectation(
        lambda z: -2 * std * z / width * np.tanh(std * z / width)
    )
    force_entropy, force_gradient = isotropic_radial_entropy_reference(
        3, std, 1.7 * (1 - eps)
    )
    torque_entropy, torque_gradient = isotropic_radial_entropy_reference(
        9, std, 0.4 * (1 - eps)
    )
    expected = force_entropy + torque_entropy + 2 * box_entropy + normal_entropy
    expected_gradient = np.repeat(
        [force_gradient, torque_gradient, box_gradient, 1], [3, 9, 2, 1]
    )

    def entropy(mean, log_std):
        normal = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        return Transformed(normal, space).entropy().sum()

    value, (mean_gradient, scale_gradient) = jax.jit(
        jax.value_and_grad(entropy, argnums=(0, 1))
    )(jnp.zeros((2, 3, 15)), jnp.full(15, np.log(std)))
    np.testing.assert_allclose(value, 6 * expected, rtol=3e-6, atol=1e-5)
    np.testing.assert_allclose(mean_gradient, 0, atol=1e-6)
    np.testing.assert_allclose(
        scale_gradient, 6 * expected_gradient, rtol=3e-6, atol=1e-5
    )


def test_nested_blockwise_factory_configuration_survives_json_roundtrip():
    original = BlockwiseSpace(
        (
            (3, BlockwiseSpace(((2, MaxNormSpace(max_norm=2.0)), (1, FreeSpace())))),
            (2, BoxSpace([-2.0, -1.0], [1.0, 4.0], width=0.7, eps=0.01)),
        )
    )
    metadata = json.loads(json.dumps(original.metadata))
    restored = ActionSpace.create("Blockwise", **metadata)
    assert restored.metadata == original.metadata
    x = jnp.array([0.2, -0.4, 0.7, 0.1, -0.3])
    apply = jax.jit(lambda b, x: b.forward_and_log_det(x))
    expected = apply(original, x)
    actual = apply(restored, x)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
    np.testing.assert_allclose(
        restored.log_det_expectation(x, jnp.full(5, 0.3)),
        original.log_det_expectation(x, jnp.full(5, 0.3)),
        rtol=2e-6,
    )


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_blockwise_rejects_invalid_block_sizes(size):
    with pytest.raises(ValueError, match="positive integers"):
        BlockwiseSpace(((size, FreeSpace()),))


def test_blockwise_rejects_missing_or_unsupported_children():
    with pytest.raises(ValueError, match="empty"):
        BlockwiseSpace(())
    with pytest.raises(TypeError, match="ActionSpace bijector"):
        BlockwiseSpace(((1, distrax.Tanh()),))


@pytest.mark.parametrize(
    "method", ["forward_and_log_det", "inverse_and_log_det", "log_det_expectation"]
)
def test_blockwise_rejects_unassigned_or_missing_coordinates(method):
    space = BlockwiseSpace(((2, MaxNormSpace()), (1, FreeSpace())))
    arguments = [jnp.zeros(4)]
    if method == "log_det_expectation":
        arguments.append(jnp.ones(4))
    with pytest.raises(ValueError, match="length 3"):
        jax.jit(getattr(space, method))(*arguments)
