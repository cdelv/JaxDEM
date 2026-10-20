"""Numerical and contract tests for RL action-space bijectors."""

import math

import jax
import jax.numpy as jnp
import pytest
import numpy as np
import distrax
from scipy.integrate import quad
from scipy.special import gammaln

from jaxdem.rl.action_spaces.box_space import BoxSpace
from jaxdem.rl.action_spaces.max_norm_space import MaxNormSpace
from jaxdem.rl.action_spaces import BlockwiseSpace, Transformed


@pytest.mark.parametrize("kind", ["box", "maxnorm"])
@pytest.mark.parametrize("mean,std", [(0.3, 0.4), (-0.7, 0.5), (1.1, 0.3)])
def test_transformed_entropy_and_derivatives_against_normal_integrals(kind, mean, std):
    eps = 1e-5
    if kind == "box":
        bijector = distrax.Block(BoxSpace(-2.0, 3.0, width=1.2, eps=eps), ndims=1)
        scale, width, center = 2.5 * (1 - eps), 1.2, 0.5
    else:
        bijector = MaxNormSpace(max_norm=1.7, eps=eps)
        scale, width, center = 1.7 * (1 - eps), 1.0, 0.0

    def expectation(function):
        return quad(
            lambda z: function(z) * np.exp(-z * z / 2) / np.sqrt(2 * np.pi),
            -10.0,
            10.0,
            epsabs=1e-11,
        )[0]

    def log_sech_squared(x):
        return 2 * (np.log(2) - abs(x) - np.log1p(np.exp(-2 * abs(x))))

    entropy = 0.5 * np.log(2 * np.pi * np.e) + np.log(std * scale / width)
    entropy += expectation(lambda z: log_sech_squared((mean + std * z) / width))
    expected_gradient = [
        expectation(lambda z: -2 * np.tanh((mean + std * z) / width) / width),
        1
        + expectation(
            lambda z: -2 * np.tanh((mean + std * z) / width) * std * z / width
        ),
    ]

    def transformed_entropy(parameters):
        normal = distrax.MultivariateNormalDiag(parameters[:1], jnp.exp(parameters[1:]))
        return Transformed(normal, bijector).entropy()

    value, gradient = jax.value_and_grad(transformed_entropy)(
        jnp.array([mean, np.log(std)])
    )
    # Finite quadrature has an approximation budget; the independent integral is tighter.
    np.testing.assert_allclose(value, entropy, rtol=0, atol=1e-3)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=0, atol=5e-3)
    latent = np.array([0.4])
    expected_action = center + scale * np.tanh(latent / width)
    np.testing.assert_allclose(bijector.forward(latent), expected_action, rtol=2e-6)
    normal = distrax.MultivariateNormalDiag(jnp.array([mean]), jnp.array([std]))
    actual_log_density = Transformed(normal, bijector).log_prob(
        jnp.asarray(expected_action)
    )
    expected_log_density = -0.5 * ((0.4 - mean) / std) ** 2 - np.log(
        std * np.sqrt(2 * np.pi)
    )
    expected_log_density -= np.log(scale / width) + log_sech_squared(0.4 / width)
    np.testing.assert_allclose(
        actual_log_density, expected_log_density, rtol=2e-6, atol=1e-6
    )


def test_radial_entropy_with_anisotropic_two_dimensional_normal():
    parameters = np.array([0.4, -0.2, np.log(0.25), np.log(0.4)])
    nodes, weights = np.polynomial.hermite_e.hermegauss(40)
    z = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1)
    probability = np.outer(weights, weights) / (2 * np.pi)
    scale = 1.7 * (1 - 1e-5)

    def reference(p):
        x = p[:2] + np.exp(p[2:]) * z
        r = np.linalg.norm(x, axis=-1)
        # Tangential eigenvalue s*tanh(r)/r; radial eigenvalue s*sech(r)^2.
        log_det = 2 * np.log(scale) + np.log(np.tanh(r) / r)
        log_det += 2 * (np.log(2) - r - np.log1p(np.exp(-2 * r)))
        return np.log(2 * np.pi * np.e) + p[2:].sum() + np.sum(probability * log_det)

    def actual(p):
        normal = distrax.MultivariateNormalDiag(p[:2], jnp.exp(p[2:]))
        return Transformed(normal, MaxNormSpace(max_norm=1.7, eps=1e-5)).entropy()

    h = 1e-5
    expected_gradient = np.array(
        [
            (reference(parameters + h * e) - reference(parameters - h * e)) / (2 * h)
            for e in np.eye(4)
        ]
    )
    value, gradient = jax.value_and_grad(actual)(jnp.asarray(parameters))
    np.testing.assert_allclose(value, reference(parameters), atol=1e-3, rtol=0)
    np.testing.assert_allclose(gradient, expected_gradient, atol=5e-3, rtol=0)


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


def isotropic_radial_entropy_reference(dimension, std, scale):
    """Integrate over the chi-distributed radius, independently of Laplace quadrature."""

    def expectation(function):
        def integrand(u):
            density = np.exp(
                (dimension - 1) * np.log(u)
                - u * u / 2
                - (dimension / 2 - 1) * np.log(2)
                - gammaln(dimension / 2)
            )
            return density * function(std * u)

        return quad(integrand, 0.0, np.sqrt(dimension) + 12.0, epsabs=1e-10)[0]

    def log_det(r):
        return (dimension - 1) * np.log(np.tanh(r) / r) + 2 * (
            np.log(2) - r - np.logaddexp(0, -2 * r)
        )

    def radial_derivative(r):
        log_tanh_derivative = 4 * np.exp(-2 * r) / -np.expm1(-4 * r)
        return (dimension - 1) * (log_tanh_derivative - 1 / r) - 2 * np.tanh(r)

    expected = dimension * (
        0.5 * np.log(2 * np.pi * np.e) + np.log(std * scale)
    ) + expectation(log_det)
    expected_scale_gradient = (
        1 + expectation(lambda r: r * radial_derivative(r)) / dimension
    )
    return expected, expected_scale_gradient


@pytest.mark.parametrize("dimension", [1, 6, 9, 16, 64])
@pytest.mark.parametrize("std", [0.1, 1.0, 10.0])
def test_radial_entropy_and_gradients_against_chi_integral(dimension, std):
    space = MaxNormSpace(max_norm=1.7)
    expected, expected_scale_gradient = isotropic_radial_entropy_reference(
        dimension, std, (1 - space.eps) * space.max_norm
    )

    def entropy(mean, log_std):
        normal = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        return Transformed(normal, space).entropy()

    value, (mean_gradient, scale_gradient) = jax.jit(
        jax.value_and_grad(entropy, argnums=(0, 1))
    )(jnp.zeros(dimension), jnp.full(dimension, np.log(std)))
    np.testing.assert_allclose(value, expected, rtol=3e-6, atol=1e-5)
    np.testing.assert_allclose(mean_gradient, 0, atol=1e-7)
    np.testing.assert_allclose(
        scale_gradient, expected_scale_gradient, rtol=3e-6, atol=1e-5
    )


@pytest.mark.parametrize("block_size", [8, 16])
def test_high_dimensional_policy_entropy_bonus_and_gradient(block_size):
    from flax import nnx

    from jaxdem.rl.models import Model
    from jaxdem.rl.trainers import TrajectoryData
    from jaxdem.rl.trainers.ppo_trainer import PPOTrainer

    dimension, coefficient = 16, 0.3
    space = MaxNormSpace(max_norm=1.7)
    if block_size < dimension:
        space = BlockwiseSpace(((block_size, space),) * (dimension // block_size))
    model = Model.create(
        "SharedActorCritic",
        observation_space_size=1,
        action_space_size=dimension,
        architecture=[],
        action_space=space,
        key=nnx.Rngs(42),
    )
    zero = jnp.zeros((2, 3))
    log_prob = jnp.full((2, 3), -dimension * np.log(2 * np.pi) / 2)
    latent = jnp.zeros((2, 3, dimension))
    td = TrajectoryData(
        obs=jnp.zeros((2, 3, 1)),
        action=latent,
        value=zero,
        log_prob=log_prob,
        ratio=jnp.ones((2, 3)),
        reward=zero,
        done=jnp.ones((2, 3), bool),
        terminated=jnp.ones((2, 3), bool),
        truncated=jnp.zeros((2, 3), bool),
        agent_mask=jnp.ones((2, 3), bool),
        bootstrap_value=zero,
        latent_action=latent,
        latent_log_prob=log_prob,
    )

    @nnx.jit
    def loss_and_grad(model, td):
        return nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
            model,
            td,
            ppo_clip_eps=jnp.array(0.2),
            ppo_value_coeff=jnp.array(0.0),
            ppo_entropy_coeff=jnp.array(coefficient),
            advantage_gamma=jnp.array(0.9),
            advantage_lambda=jnp.array(0.8),
            advantage_rho_clip=jnp.array(1.0),
            advantage_c_clip=jnp.array(1.0),
            last_value=jnp.zeros(3),
            vtrace=jnp.array(False),
        )

    (loss, aux), gradient = loss_and_grad(model, td)
    expected, scale_gradient = isotropic_radial_entropy_reference(
        block_size, 1.0, (1 - 1e-6) * 1.7
    )
    expected *= dimension // block_size
    np.testing.assert_allclose(aux["entropy"], expected, rtol=3e-6, atol=1e-5)
    np.testing.assert_allclose(loss, -coefficient * expected, rtol=3e-6, atol=1e-5)
    np.testing.assert_allclose(
        gradient.actor_sigma.log_std[...],
        -coefficient * scale_gradient,
        rtol=3e-6,
        atol=1e-5,
    )
    np.testing.assert_allclose(gradient.actor_mu.bias[...], 0.0, atol=1e-7)


@pytest.mark.parametrize("dimension", [9, 32])
def test_high_dimensional_anisotropic_radial_correction(dimension):
    # Two fluctuating coordinates and fixed offsets reduce this Gaussian limit
    # to an independent two-dimensional reference, including nonzero means.
    parameters = np.array([0.4, -0.2, np.log(0.25), np.log(0.4)])
    offsets = np.linspace(-0.4, 0.6, dimension - 2)
    nodes, weights = np.polynomial.hermite_e.hermegauss(40)
    z = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1)
    probability = np.outer(weights, weights) / (2 * np.pi)
    space = MaxNormSpace(max_norm=1.7)

    def reference(p):
        x = p[:2] + np.exp(p[2:]) * z
        r = np.sqrt(np.sum(x * x, axis=-1) + np.sum(offsets * offsets))
        log_det = dimension * np.log((1 - space.eps) * space.max_norm)
        log_det += (dimension - 1) * np.log(np.tanh(r) / r)
        log_det += 2 * (np.log(2) - r - np.logaddexp(0, -2 * r))
        return np.sum(probability * log_det)

    def actual(p):
        mean = jnp.concatenate([p[:2], jnp.asarray(offsets)])
        std = jnp.concatenate([jnp.exp(p[2:]), jnp.zeros(dimension - 2)])
        return space.log_det_expectation(mean, std)

    h = 1e-5
    expected_gradient = np.array(
        [
            (reference(parameters + h * e) - reference(parameters - h * e)) / (2 * h)
            for e in np.eye(4)
        ]
    )
    value, gradient = jax.jit(jax.value_and_grad(actual))(jnp.asarray(parameters))
    np.testing.assert_allclose(value, reference(parameters), rtol=3e-6, atol=1e-5)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-6, atol=1e-5)


def test_radial_correction_broadcasts_batches_and_has_deterministic_limit():
    space = MaxNormSpace(max_norm=1.7)
    mean = jnp.zeros((2, 3, 16))
    std = jnp.linspace(0.1, 0.5, 16)
    batched = jax.jit(space.log_det_expectation)(mean, std)
    scalar = space.log_det_expectation(mean[0, 0], std)
    np.testing.assert_allclose(batched, np.full((2, 3), scalar), rtol=2e-6)

    # A collapsed Gaussian has the Jacobian at its mean, even at the origin.
    for point in [jnp.zeros(16), jnp.linspace(-0.5, 1.0, 16)]:
        value, gradient = jax.value_and_grad(space.log_det_expectation)(
            point, jnp.zeros(16)
        )
        expected, expected_gradient = jax.value_and_grad(
            space.forward_log_det_jacobian
        )(point)
        np.testing.assert_allclose(value, expected, rtol=3e-6, atol=1e-5)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-6, atol=1e-5)
    np.testing.assert_allclose(
        jax.hessian(space.log_det_expectation)(jnp.zeros(16), jnp.zeros(16)),
        -2 * (16 + 2) / 3 * jnp.eye(16),
        rtol=3e-6,
    )


@pytest.mark.parametrize("eps", [1e-6, 0.1])
@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="tanh(8) inverse accuracy requires float64"
)
def test_box_margin_does_not_prematurely_clip_inverse(eps):
    space = BoxSpace(-3.0, 5.0, width=0.7, eps=eps)
    # tanh(8) lies beyond the old inverse cutoff even in float64.
    x = jnp.array([-5.6, 0.0, 5.6], dtype=jnp.float64)
    y, ld = space.forward_and_log_det(x)
    restored, ild = space.inverse_and_log_det(y)
    assert jnp.allclose(restored, x, rtol=1e-8, atol=1e-8)
    assert jnp.allclose(ild, -ld, rtol=1e-8, atol=1e-8)
