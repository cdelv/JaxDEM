"""MinGRU replay must preserve rollout outputs and learning gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxdem.rl.models import MinGRUActorCritic


def _outputs(model, observations, carry, done, *, sequence):
    if sequence:
        pi, value = model(
            observations, sequence=True, initial_carry=carry, done=done
        )
        return jnp.concatenate((pi.logits, value), axis=-1)
    model.h.set_value(carry)
    outputs = []
    for obs, boundary in zip(observations, done, strict=True):
        pi, value = model(obs)
        outputs.append(jnp.concatenate((pi.logits, value), axis=-1))
        model.reset(obs.shape, mask=boundary)
    return jnp.stack(outputs)


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("candidate_weight", [-0.5, 0.0, 0.5])
def test_replay_matches_rollout_gradients_at_zero_and_nonzero_candidates(
    num_layers, candidate_weight
):
    model = MinGRUActorCritic(
        2, 2, nnx.Rngs(7), hidden_features=2, gru_features=2,
        num_layers=num_layers, activation=lambda x: x, discrete=True,
    )
    model.encoder.layers[0].kernel[...] = jnp.eye(2)
    model.encoder.layers[0].bias[...] = jnp.zeros(2)
    # At zero, nonzero inputs still give nonzero candidate-kernel gradients.
    kernel = model.mingru_kernel[...]
    model.mingru_kernel[...] = kernel.at[:, :, :2].set(
        candidate_weight * jnp.eye(2)
    )
    model.fused_head.kernel[...] = jnp.asarray([[0.3, -0.7, 0.9], [0.6, 0.2, -0.4]])
    observations = jnp.asarray([
        [[1.0, -0.5], [0.4, 0.7]],
        [[0.2, 0.9], [-0.8, 0.3]],
        [[-0.6, 0.2], [0.5, -0.4]],
        [[0.7, 0.1], [-0.3, 0.8]],
    ])
    carry = jnp.full((2, num_layers, 2), 0.3)
    done = jnp.asarray([[False, True], [True, False], [False, False], [True, True]])
    graph, state = nnx.split(model)
    weights = jnp.arange(24, dtype=jnp.float32).reshape(4, 2, 3) / 24

    def loss(state, obs, *, sequence):
        current = nnx.merge(graph, state, copy=True)
        outputs = _outputs(current, obs, carry, done, sequence=sequence)
        return jnp.sum(outputs * weights), outputs

    with jax.default_matmul_precision("highest"):
        replay = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(
            state, observations, sequence=True
        )
        rollout = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(
            state, observations, sequence=False
        )
    for actual, expected in zip(jax.tree.leaves(replay), jax.tree.leaves(rollout), strict=True):
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("initial_value", [0.0, 1e-9, 1e-8])
def test_replay_preserves_small_positive_initial_carry(initial_value):
    model = MinGRUActorCritic(
        1, 1, nnx.Rngs(11), hidden_features=1, gru_features=1,
        num_layers=1, activation=lambda x: x, discrete=True,
    )
    model.encoder.layers[0].kernel[...] = jnp.ones((1, 1))
    model.encoder.layers[0].bias[...] = jnp.zeros(1)
    # A nearly closed update gate preserves the incoming carry. An open
    # highway gate exposes it to the critic, avoiding a large residual input.
    model.mingru_kernel[...] = jnp.asarray([[[-20.0, -20.0, 30.0]]])
    model.fused_head.kernel[...] = jnp.asarray([[0.0, 1.0]])
    model.fused_head.bias[...] = jnp.zeros(2)
    graph, state = nnx.split(model)
    obs = jnp.ones((3, 1, 1))
    done = jnp.zeros((3, 1), dtype=bool)
    carry = jnp.full((1, 1, 1), initial_value)

    def loss(carry, *, sequence):
        current = nnx.merge(graph, state, copy=True)
        return _outputs(current, obs, carry, done, sequence=sequence)[..., -1].sum()

    replay = jax.value_and_grad(loss)(carry, sequence=True)
    rollout = jax.value_and_grad(loss)(carry, sequence=False)
    np.testing.assert_allclose(replay[0], rollout[0], rtol=5e-6, atol=1e-17)
    assert np.isfinite(replay[1]).all()
    if initial_value > 0:
        np.testing.assert_allclose(replay[1], rollout[1], rtol=5e-6, atol=1e-6)
