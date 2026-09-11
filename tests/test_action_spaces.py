"""Numerical and contract tests for RL action-space bijectors."""

import math

import jax
import jax.numpy as jnp
import pytest

from jaxdem.rl.action_spaces.box_space import BoxSpace
from jaxdem.rl.action_spaces.max_norm_space import MaxNormSpace


@pytest.mark.parametrize("width", [0.0, -1.0, float("inf"), float("nan")])
def test_box_rejects_invalid_width(width: float) -> None:
    with pytest.raises(ValueError, match="width"):
        BoxSpace(-1.0, 1.0, width=width)


@pytest.mark.parametrize("eps", [0.0, -1.0, 1.0, float("nan")])
def test_action_spaces_reject_invalid_epsilon(eps: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        BoxSpace(-1.0, 1.0, eps=eps)
    with pytest.raises(ValueError, match="eps"):
        MaxNormSpace(eps=eps)


@pytest.mark.parametrize("radius", [0.0, -1.0, float("inf"), float("nan")])
def test_max_norm_rejects_invalid_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="max_norm"):
        MaxNormSpace(max_norm=radius)


def test_distinct_box_spaces_remain_distinct_as_jitted_arguments() -> None:
    narrow = BoxSpace(-1.0, 1.0)
    wide = BoxSpace(-2.0, 2.0)

    @jax.jit
    def apply(space: BoxSpace, value: jax.Array) -> jax.Array:
        return space.forward(value)

    assert apply(narrow, jnp.array(1.0)) != apply(wide, jnp.array(1.0))


def test_box_inverse_and_jacobian_near_boundary() -> None:
    space = BoxSpace(-3.0, 5.0, width=0.7, eps=1e-6)
    x = jnp.array([-4.5, 0.0, 4.5])
    y, log_det = space.forward_and_log_det(x)
    restored, inverse_log_det = space.inverse_and_log_det(y)
    assert jnp.allclose(restored, x, rtol=2e-3, atol=2e-3)
    assert jnp.allclose(inverse_log_det, -log_det, rtol=2e-3, atol=2e-3)
    # At these near-boundary points float32 tanh is rounded enough that its
    # autodiff derivative suffers cancellation. Check the stable expression
    # against an independent double-precision scalar formula instead.
    half = (1.0 - 1e-6) * (5.0 - (-3.0)) / 2.0
    expected = []
    for value in (-4.5, 0.0, 4.5):
        z = value / 0.7
        sec2_log = 2.0 * (math.log(2.0) - z - math.log1p(math.exp(-2.0 * z)))
        expected.append(math.log(half) - math.log(0.7) + sec2_log)
    assert jnp.allclose(log_det, jnp.asarray(expected), rtol=2e-5, atol=2e-5)

    resolvable_x = jnp.array([-2.0, 0.0, 2.0])
    _, resolvable_log_det = space.forward_and_log_det(resolvable_x)
    jacobian = jax.vmap(jax.grad(lambda value: space.forward(value)))(resolvable_x)
    assert jnp.allclose(
        jnp.log(jnp.abs(jacobian)), resolvable_log_det, rtol=2e-4, atol=2e-4
    )


@pytest.mark.parametrize(
    "x", [jnp.zeros(3), jnp.array([1e-7, -2e-7, 3e-7]), jnp.array([3.0, -2.0, 1.0])]
)
def test_max_norm_inverse_and_jacobian(x: jax.Array) -> None:
    space = MaxNormSpace(max_norm=2.0, eps=1e-6)
    y, log_det = space.forward_and_log_det(x)
    restored, inverse_log_det = space.inverse_and_log_det(y)
    assert jnp.allclose(restored, x, rtol=3e-4, atol=3e-6)
    assert jnp.allclose(inverse_log_det, -log_det, rtol=3e-4, atol=3e-5)
    jacobian = jax.jacfwd(space.forward)(x)
    _, numeric_log_det = jnp.linalg.slogdet(jacobian)
    assert jnp.allclose(numeric_log_det, log_det, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("eps", [0.1, 0.5])
def test_max_norm_large_margin_keeps_exact_inverse_and_jacobian(eps: float) -> None:
    space = MaxNormSpace(max_norm=2.0, eps=eps)
    x = jnp.array([0.4, -0.7, 1.1])
    y, log_det = space.forward_and_log_det(x)
    restored, inverse_log_det = space.inverse_and_log_det(y)
    jacobian = jax.jacfwd(space.forward)(x)
    _, numeric_log_det = jnp.linalg.slogdet(jacobian)

    assert jnp.allclose(restored, x, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(inverse_log_det, -log_det, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(numeric_log_det, log_det, rtol=2e-5, atol=2e-6)


def test_max_norm_origin_has_exact_linear_limit() -> None:
    space = MaxNormSpace(max_norm=2.0, eps=0.1)
    origin = jnp.zeros(3)
    scale = (1.0 - space.eps) * space.max_norm

    np_jacobian = jax.jacfwd(space.forward)(origin)
    reverse_jacobian = jax.jacrev(space.forward)(origin)
    np_hessian = jax.jacfwd(jax.jacfwd(space.forward))(origin)
    assert jnp.allclose(np_jacobian, scale * jnp.eye(3))
    assert jnp.allclose(reverse_jacobian, scale * jnp.eye(3))
    assert jnp.allclose(np_hessian, 0.0)
    assert jnp.allclose(space.forward_log_det_jacobian(origin), 3 * jnp.log(scale))
    assert jnp.allclose(jax.grad(space.forward_log_det_jacobian)(origin), jnp.zeros(3))
    assert jnp.allclose(
        jax.hessian(space.forward_log_det_jacobian)(origin),
        -2.0 * (3 + 2) / 3 * jnp.eye(3),
    )
    assert jnp.allclose(jax.jacrev(space.inverse)(origin), jnp.eye(3) / scale)


def test_max_norm_entropy_rejects_high_dimension_before_tensor_allocation() -> None:
    space = MaxNormSpace()
    with pytest.raises(ValueError, match="dimensions up to 6"):
        space.log_det_expectation(jnp.zeros(7), jnp.ones(7))
