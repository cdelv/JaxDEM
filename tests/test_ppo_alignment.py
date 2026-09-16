"""Contracts for sequential PPO updates with current, detached targets."""

from dataclasses import dataclass, replace

import distrax
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from jaxdem.rl.models import Model
from jaxdem.rl.trainers import TrajectoryData
from jaxdem.rl.trainers.ppo_trainer import PPOTrainer
from jaxdem.state import State
from jaxdem.system import System
from tests.test_ppo_math import TimeLimitCounterEnv


class ConstantPolicyCritic(Model):
    """Isolate critic updates from policy learning and action sampling."""

    def __init__(self, value=0.0):
        self.bias = nnx.Param(jnp.asarray([value]))

    def __call__(self, x, sequence=False, **kwargs):
        shape = (*x.shape[:-1], 1)
        policy = distrax.MultivariateNormalDiag(
            jnp.zeros(shape), jnp.ones(shape)
        )
        return policy, jnp.broadcast_to(self.bias[...], shape)


def _trajectory():
    zeros = jnp.zeros((2, 1))
    done = jnp.array([[False], [True]])
    # pi_current(0) / pi_behavior(0) = 1/2. The stale ratio field is
    # intentionally unrelated, to ensure the learner computes its own ratio.
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
    assert jnp.allclose(aux["actor_loss"], -0.5 * jnp.mean(jnp.array(expected_advantage)))
    # Actor targets and critic targets must both be detached. Only the
    # explicit prediction-minus-target term contributes to this parameter.
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


def test_value_clipping_uses_immutable_behavior_predictions_on_revisit():
    model = ConstantPolicyCritic(0.6)
    td = _trajectory()
    for prediction in (0.6, 0.8):
        model.bias[...] = jnp.array([prediction])
        (_, aux), gradient = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
            model, td, **_loss_args(clip=0.2)
        )
        # Targets remain [3, 2] for this terminal, gamma=lambda=1 rollout;
        # the behavior reference remains zero and its clipped value is 0.2.
        assert jnp.allclose(aux["value_loss"], 0.25 * (2.8**2 + 1.8**2))
        assert jnp.allclose(gradient.bias[...], 0.0)
        assert jnp.array_equal(td.value, jnp.zeros((2, 1)))
        assert jnp.allclose(aux["advantages"][:, 0], jnp.array([3.0, 2.0]) - prediction)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MarkedTerminalEnv(TimeLimitCounterEnv):
    """One-step episodes with a distinct reward for each environment slot."""

    @staticmethod
    def reset(env, key):
        del key
        return replace(env, env_params={
            "count": jnp.asarray(0),
            "marker": env.env_params.get("marker", jnp.asarray(1.0)),
        })

    @staticmethod
    def step(env, action):
        del action
        return replace(env, env_params={
            **env.env_params, "count": env.env_params["count"] + 1,
        })

    @staticmethod
    def observation(env):
        return env.env_params["marker"][None, None]

    @staticmethod
    def reward(env):
        return env.env_params["marker"][None]

    @staticmethod
    def terminated(env):
        return env.env_params["count"] > 0

    @staticmethod
    def truncated(env):
        return jnp.asarray(False)


def _sgd(learning_rate, eps):
    del eps
    return optax.sgd(learning_rate)


def test_epoch_visits_contiguous_batches_wraps_and_preserves_rollout():
    trainer = PPOTrainer.Create(
        MarkedTerminalEnv.Create(), ConstantPolicyCritic(), num_envs=4,
        num_steps_epoch=1, num_minibatches=4, minibatch_size=2,
        optimizer=_sgd, learning_rate=0.1, anneal_learning_rate=False,
        max_grad_norm=float("inf"), ppo_clip_eps=100.0,
        ppo_value_coeff=1.0, ppo_entropy_coeff=0.0,
    )
    trainer.env = replace(trainer.env, env_params={
        **trainer.env.env_params, "marker": jnp.arange(1.0, 5.0),
    })
    trainer, td, metrics = trainer.epoch(trainer, jnp.asarray(0))

    # SGD sees reward means 1.5, 3.5, 1.5, 3.5, giving these pre-update
    # critic predictions: 0, .15, .485, .5865. Every advantage must use
    # that visit's current value, including visits after wraparound.
    assert jnp.allclose(trainer.model.bias[...], 0.87785)
    assert jnp.allclose(metrics["actor_loss"], -(1.5 + 3.35 + 1.015 + 2.9135) / 4)
    assert jnp.array_equal(td.reward, jnp.array([[1.0, 2.0, 3.0, 4.0]]))
    assert jnp.array_equal(td.value, jnp.zeros((1, 4)))
    assert jnp.array_equal(td.ratio, jnp.ones((1, 4)))
    expected_lp = -0.5 * jnp.square(td.action[..., 0]) - 0.5 * jnp.log(2 * jnp.pi)
    assert jnp.allclose(td.log_prob, expected_lp)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_envs": 5, "num_steps_epoch": 64, "num_minibatches": 4}, "divisible"),
        ({"num_envs": 4, "num_steps_epoch": 4, "minibatch_size": 6}, "divisible"),
        ({"num_envs": 4, "num_steps_epoch": 4, "minibatch_size": 12}, "divide"),
    ],
)
def test_sequential_batching_rejects_silent_rounding(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PPOTrainer.Create(MarkedTerminalEnv.Create(), ConstantPolicyCritic(), **kwargs)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PaddedTerminalEnv(MarkedTerminalEnv):
    @classmethod
    def Create(cls, dim=2):
        state = State.create(pos=jnp.zeros((2, dim)))
        return cls(state, System.create(state.shape), {"count": jnp.asarray(0)})

    @staticmethod
    def observation(env):
        return jnp.zeros((2, 1))

    @staticmethod
    def reward(env):
        return jnp.array([1.0, 999.0])

    @staticmethod
    def agent_mask(env):
        return jnp.array([True, False])


def test_inactive_only_block_does_not_advance_optimizer_momentum():
    def momentum_sgd(learning_rate, eps):
        del eps
        return optax.sgd(learning_rate, momentum=0.9)

    trainer = PPOTrainer.Create(
        PaddedTerminalEnv.Create(), ConstantPolicyCritic(), num_envs=1,
        num_steps_epoch=1, num_minibatches=2, minibatch_size=1,
        optimizer=momentum_sgd, learning_rate=0.1, anneal_learning_rate=False,
        ppo_clip_eps=100.0, ppo_value_coeff=1.0, ppo_entropy_coeff=0.0,
    )
    trainer, _, metrics = trainer.epoch(trainer, jnp.asarray(0))
    assert jnp.allclose(trainer.model.bias[...], 0.1)
    assert jnp.allclose(metrics["score"], 1.0)
