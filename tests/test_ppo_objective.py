# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Production PPO loss, targets, latent ratios, and derivatives."""

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxdem.rl.action_spaces import BoxSpace, MaxNormSpace, Transformed
from jaxdem.rl.env_wrappers import vectorise_env
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


class ConstantPolicyCritic(Model):
    """Isolate critic updates from policy learning and action sampling."""

    def __init__(self, value=0.0):
        self.bias = nnx.Param(jnp.asarray([value]))

    def __call__(self, x, sequence=False, **kwargs):
        shape = (*x.shape[:-1], 1)
        policy = distrax.MultivariateNormalDiag(jnp.zeros(shape), jnp.ones(shape))
        return policy, jnp.broadcast_to(self.bias[...], shape)


def _trajectory():
    zeros = jnp.zeros((2, 1))
    done = jnp.array([[False], [True]])
    # pi_current(0) / pi_behavior(0) = 1/2. Stale ratio field is ignored.
    behavior_log_prob = jnp.full((2, 1), -0.5 * jnp.log(2 * jnp.pi) + jnp.log(2))
    return TrajectoryData(
        obs=jnp.zeros((2, 1, 1)),
        action=jnp.zeros((2, 1, 1)),
        value=zeros,
        log_prob=behavior_log_prob,
        ratio=jnp.full((2, 1), 99.0),
        reward=jnp.array([[1.0], [2.0]]),
        done=done,
        terminated=done,
        truncated=jnp.zeros_like(done),
        agent_mask=jnp.ones_like(done),
        bootstrap_value=zeros,
    )


def _loss_args(vtrace=False, clip=10.0):
    return dict(
        ppo_clip_eps=jnp.asarray(clip),
        ppo_value_coeff=jnp.asarray(1.0),
        ppo_entropy_coeff=jnp.asarray(0.0),
        advantage_gamma=jnp.asarray(1.0),
        advantage_lambda=jnp.asarray(1.0),
        advantage_rho_clip=jnp.asarray(1.0),
        advantage_c_clip=jnp.asarray(1.0),
        last_value=jnp.zeros(1),
        vtrace=jnp.asarray(vtrace),
    )


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


@pytest.mark.parametrize(
    "vtrace, expected_advantage, expected_returns, expected_gradient",
    [
        (False, [2.4, 1.4], [3.0, 2.0], -1.9),
        (True, [0.85, 0.7], [1.45, 1.3], -0.775),
    ],
)
def test_current_targets_raw_advantages_and_detachment(
    vtrace, expected_advantage, expected_returns, expected_gradient
):
    model = ConstantPolicyCritic(0.6)
    td = _trajectory()
    original = jax.tree.map(lambda x: x.copy(), td)
    (_, aux), gradient = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
        model, td, **_loss_args(vtrace)
    )

    assert jnp.allclose(aux["ratio"], 0.5)
    assert jnp.allclose(aux["advantages"][:, 0], jnp.array(expected_advantage))
    assert jnp.allclose(aux["target_values"][:, 0], jnp.array(expected_returns))
    assert jnp.allclose(
        aux["actor_loss"], -0.5 * jnp.mean(jnp.array(expected_advantage))
    )
    assert jnp.allclose(gradient.bias[...], expected_gradient)
    assert all(
        jnp.array_equal(before, after)
        for before, after in zip(jax.tree.leaves(original), jax.tree.leaves(td))
    )


def test_gae_ignores_vtrace_caps_when_vtrace_is_disabled():
    kwargs = _loss_args(False)
    kwargs.update(advantage_rho_clip=jnp.array(0.2), advantage_c_clip=jnp.array(0.1))
    _, aux = PPOTrainer.loss_fn(ConstantPolicyCritic(0.6), _trajectory(), **kwargs)
    assert jnp.allclose(aux["target_values"][:, 0], jnp.array([3.0, 2.0]))


def test_loss_selection_preserves_future_gae_context():
    td = _trajectory()
    model = ConstantPolicyCritic(0.6)
    mask = jnp.array([[True], [False]])
    (_, aux), gradient = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
        model, td, **_loss_args(), loss_mask=mask
    )
    # The unselected terminal reward is still needed for the first return.
    assert jnp.allclose(aux["target_values"][:, 0], jnp.array([3.0, 2.0]))
    assert jnp.allclose(aux["actor_loss"], -0.5 * 2.4)
    assert jnp.allclose(aux["value_loss"], 0.5 * 2.4**2)
    assert jnp.allclose(gradient.bias[...], -2.4)


@pytest.mark.parametrize("model_name", ["lstm", "mingru"])
def test_selected_recurrent_loss_keeps_prefix_gradients(model_name):
    if model_name == "lstm":
        from jaxdem.rl.models.lstm import LSTMActorCritic

        model = LSTMActorCritic(
            1,
            1,
            nnx.Rngs(17),
            hidden_features=4,
            lstm_features=4,
            carry_leading_shape=(1,),
        )
    else:
        from jaxdem.rl.models.mingru import MinGRUActorCritic

        model = MinGRUActorCritic(
            1,
            1,
            nnx.Rngs(17),
            hidden_features=4,
            gru_features=4,
            num_layers=2,
            carry_leading_shape=(1,),
        )
    td = _trajectory()
    td.obs = jnp.array([[[0.4]], [[0.8]]])
    initial_carry = model.carry
    kwargs = dict(_loss_args(), initial_carry=initial_carry)
    (actual, aux), grads = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
        model, td, **kwargs, loss_mask=jnp.array([[False], [True]])
    )

    def reference_loss(current_model):
        pi, values = current_model(
            td.obs, sequence=True, initial_carry=initial_carry, done=td.done
        )
        ratio = jnp.exp(pi.log_prob(td.action) - td.log_prob)[-1, 0]
        return (
            -aux["advantages"][-1, 0] * ratio
            + 0.5 * (values[-1, 0, 0] - aux["target_values"][-1, 0]) ** 2
        )

    expected, expected_grads = nnx.value_and_grad(reference_loss)(model)
    assert jnp.allclose(actual, expected, atol=1e-6)
    for actual_grad, expected_grad in zip(
        jax.tree.leaves(grads), jax.tree.leaves(expected_grads), strict=True
    ):
        assert jnp.allclose(actual_grad, expected_grad, atol=1e-6)


class SaturatedPolicy(Model):
    def __init__(self, space, dtype):
        self.mean = nnx.Param(jnp.asarray([30.0], dtype=dtype))
        self.bij = nnx.data(
            space if isinstance(space, MaxNormSpace) else distrax.Block(space, 1)
        )

    def __call__(self, obs, sequence=False, **kwargs):
        shape = (*obs.shape[:-1], 1)
        return Transformed(
            distrax.MultivariateNormalDiag(
                jnp.broadcast_to(self.mean[...], shape),
                jnp.full(shape, 0.1, dtype=self.mean[...].dtype),
            ),
            self.bij,
        ), jnp.zeros(shape)


@pytest.mark.parametrize(
    "space,use_x64",
    [(BoxSpace(-1.0, 1.0), False), (BoxSpace(-1.0, 1.0), True), (MaxNormSpace(), False)],
)
def test_saturated_actions_use_exact_latent_ratios_and_gradients(space, use_x64):
    from tests.test_ppo_math import CheckpointCounter

    with jax.enable_x64(use_x64):
        dtype = jnp.float64 if use_x64 else jnp.float32
        env = vectorise_env(CheckpointCounter.Create(), n=1)
        model = SaturatedPolicy(space, dtype)
        graphdef, graphstate = nnx.split((model,))
        _, _, _, td = Trainer.trajectory_rollout(
            env, graphdef, graphstate, jax.random.key(7), num_steps_epoch=2
        )
        td = jax.tree.map(lambda x: x[:, 0], td)
        assert td.latent_action.dtype == dtype
        args = _loss_args()
        (_, aux), grads = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
            model, td, **args
        )
        np.testing.assert_allclose(aux["ratio"], 1.0, rtol=2e-6, atol=2e-6)
        assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads))
        model.mean[...] += 0.02
        (_, aux), grads = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
            model, td, **args
        )
        expected_ratio = jnp.exp(
            distrax.MultivariateNormalDiag(
                model.mean[...], jnp.array([0.1], dtype=dtype)
            ).log_prob(td.latent_action)
            - td.latent_log_prob
        )
        np.testing.assert_allclose(aux["ratio"], expected_ratio, rtol=2e-5)
        expected_grad = -jnp.mean(
            aux["advantages"]
            * expected_ratio
            * (td.latent_action[..., 0] - model.mean[0])
            / 0.1**2
        )
        np.testing.assert_allclose(grads.mean[0], expected_grad, rtol=2e-4, atol=2e-4)
