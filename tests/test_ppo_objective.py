# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Production PPO loss and derivatives against scalar probability calculus."""

import jax
import jax.numpy as jnp

import distrax
from flax import nnx
import numpy as np
import pytest

from jaxdem.rl.models import Model
from jaxdem.rl.trainers import Trainer, TrajectoryData
from jaxdem.rl.trainers.ppo_trainer import PPOTrainer


class GaussianPolicy(Model):
    def __init__(self):
        self.mean = nnx.Param(jnp.asarray(0.25))
        self.log_scale = nnx.Param(jnp.asarray(0.0))
        self.value = nnx.Param(jnp.asarray(0.0))

    def __call__(self, x, sequence=False, **kwargs):
        shape = (*x.shape[:-1], 1)
        policy = distrax.MultivariateNormalDiag(
            jnp.broadcast_to(self.mean[...], shape),
            jnp.broadcast_to(jnp.exp(self.log_scale[...]), shape),
        )
        return policy, jnp.broadcast_to(self.value[...], shape)


def transition(reward, ratio):
    zero = jnp.zeros((1, 1))
    # Current Normal(.25,1) at action .75 has log density -log(2*pi)/2-1/8.
    log_prob = -0.5 * np.log(2 * np.pi) - 0.125 - np.log(ratio)
    return TrajectoryData(
        obs=jnp.zeros((1, 1, 1)),
        action=jnp.full((1, 1, 1), 0.75),
        value=zero,
        log_prob=jnp.full((1, 1), log_prob),
        ratio=jnp.full((1, 1), 99.0),
        reward=jnp.full((1, 1), reward),
        done=jnp.ones((1, 1), bool),
        terminated=jnp.ones((1, 1), bool),
        truncated=jnp.zeros((1, 1), bool),
        agent_mask=jnp.ones((1, 1), bool),
        bootstrap_value=zero,
    )


def loss_arguments(entropy=0.0):
    return dict(
        ppo_clip_eps=jnp.array(0.2),
        ppo_value_coeff=jnp.array(0.0),
        ppo_entropy_coeff=jnp.array(entropy),
        advantage_gamma=jnp.array(0.9),
        advantage_lambda=jnp.array(0.8),
        advantage_rho_clip=jnp.array(1.0),
        advantage_c_clip=jnp.array(1.0),
        last_value=jnp.zeros(1),
        vtrace=jnp.array(False),
    )


@pytest.mark.parametrize("advantage", [-2.0, 3.0])
@pytest.mark.parametrize("ratio", [0.5, 1.1, 1.5])
def test_clipped_policy_loss_and_gradients(advantage, ratio):
    td = transition(advantage, ratio)
    (loss, aux), gradient = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
        GaussianPolicy(), td, **loss_arguments()
    )
    clipped = (advantage > 0 and ratio > 1.2) or (advantage < 0 and ratio < 0.8)
    bounded = min(ratio, 1.2) if advantage > 0 else max(ratio, 0.8)
    expected_loss = -advantage * bounded
    # d log pi(a)/d mean=.5; d log pi(a)/d log(sigma)=.5^2-1=-.75.
    slope = 0.0 if clipped else -advantage * ratio
    np.testing.assert_allclose(aux["ratio"], ratio, rtol=2e-6)
    np.testing.assert_allclose(loss, expected_loss, rtol=2e-6)
    np.testing.assert_allclose(gradient.mean[...], 0.5 * slope, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(
        gradient.log_scale[...], -0.75 * slope, rtol=2e-6, atol=1e-7
    )
    np.testing.assert_array_equal(td.ratio, [[99.0]])


@pytest.mark.parametrize(
    "advantage,selected,expected",
    [(3.0, True, -3.6), (0.0, True, 0.0), (-2.0, False, 0.0)],
)
def test_clipped_policy_gradient_stays_zero_before_ratio_overflow(
    advantage, selected, expected
):
    with jax.enable_x64(False):
        td = transition(advantage, np.exp(100.0))
        (loss, _), gradient = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
            GaussianPolicy(),
            td,
            loss_mask=jnp.full((1, 1), selected),
            **loss_arguments(),
        )
        np.testing.assert_allclose(loss, expected, rtol=2e-6)
        np.testing.assert_array_equal(gradient.mean[...], 0.0)
        np.testing.assert_array_equal(gradient.log_scale[...], 0.0)


def test_entropy_bonus_value_sign_and_gradient():
    model = GaussianPolicy()
    model.log_scale[...] = jnp.asarray(np.log(1.7))
    (loss, aux), gradient = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
        model, transition(0.0, 1.0), **loss_arguments(entropy=0.3)
    )
    entropy = 0.5 * np.log(2 * np.pi * np.e) + np.log(1.7)
    np.testing.assert_allclose(aux["entropy"], entropy, rtol=2e-6)
    np.testing.assert_allclose(loss, -0.3 * entropy, rtol=2e-6)
    np.testing.assert_allclose(gradient.log_scale[...], -0.3, rtol=2e-6)
    np.testing.assert_allclose(gradient.mean[...], 0.0, atol=1e-7)


@pytest.mark.parametrize(
    "ratio,expected", [(1.1, [-0.44, -0.66, 1.1]), (0.5, [0.0, 0.0, 0.0])]
)
def test_categorical_policy_loss_uses_log_probability_gradient(ratio, expected):
    class CategoricalPolicy(Model):
        def __init__(self):
            self.logits = nnx.Param(jnp.log(jnp.array([0.2, 0.3, 0.5])))

        def __call__(self, x, sequence=False, **kwargs):
            return distrax.Categorical(
                logits=jnp.broadcast_to(self.logits[...], (*x.shape[:-1], 3))
            ), jnp.zeros((*x.shape[:-1], 1))

    td = transition(-2.0, ratio)
    td.action = jnp.full((1, 1), 2, dtype=int)
    td.log_prob = jnp.full((1, 1), np.log(0.5 / ratio))
    (loss, _), gradient = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
        CategoricalPolicy(), td, **loss_arguments()
    )
    np.testing.assert_allclose(loss, 2 * max(ratio, 0.8), rtol=2e-6)
    np.testing.assert_allclose(gradient.logits[...], expected, atol=2e-7)


@pytest.mark.parametrize(
    "value,expected_loss,expected_gradient",
    [
        (0.1, 0.5 * 2.9**2, -2.9),  # Inside the clip interval.
        (0.8, 0.5 * 2.6**2, 0.0),  # Clipped error wins; prediction gradient is zero.
        (-0.6, 0.5 * 3.6**2, -3.6),  # Unclipped error wins outside the interval.
    ],
)
def test_value_loss_selects_the_larger_error(value, expected_loss, expected_gradient):
    model = GaussianPolicy()
    model.value[...] = jnp.asarray(value)
    td = transition(3.0, 1.0)
    td.value = jnp.full((1, 1), 0.2)
    behavior_value = np.asarray(td.value).copy()
    arguments = loss_arguments()
    arguments["ppo_value_coeff"] = jnp.asarray(0.7)
    (loss, aux), gradient = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
        model, td, **arguments
    )
    np.testing.assert_allclose(aux["target_values"], [[3.0]], rtol=2e-6)
    np.testing.assert_allclose(aux["value_loss"], expected_loss, rtol=2e-6)
    np.testing.assert_allclose(
        gradient.value[...], 0.7 * expected_gradient, rtol=2e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        loss, aux["actor_loss"] + 0.7 * expected_loss, rtol=2e-6, atol=5e-7
    )
    np.testing.assert_array_equal(td.value, behavior_value)


@pytest.mark.parametrize("importance", [False, True])
def test_nontrivial_advantage_trace(importance):
    returns, advantages = Trainer.compute_advantages(
        value=jnp.array([[0.5], [0.75], [1.25]]),
        reward=jnp.array([[1.0], [2.0], [3.0]]),
        ratio=jnp.array([[0.4], [1.8], [0.7]]) if importance else jnp.ones((3, 1)),
        done=jnp.zeros((3, 1), bool),
        advantage_gamma=jnp.array(0.9),
        advantage_lambda=jnp.array(0.8),
        advantage_rho_clip=jnp.array(0.8 if importance else 1.0),
        advantage_c_clip=jnp.array(0.5 if importance else 1.0),
        last_value=jnp.array([2.0]),
        unroll=1,
    )
    expected = [1.2748448, 2.7946, 2.485] if importance else [4.72532, 4.931, 3.55]
    np.testing.assert_allclose(advantages[:, 0], expected, rtol=2e-6)
    np.testing.assert_allclose(
        returns[:, 0], np.array(expected) + [0.5, 0.75, 1.25], rtol=2e-6
    )
