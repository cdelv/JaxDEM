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
from jaxdem.rl.trainers.ppo_trainer import PPOTrainer, _minibatch_selection
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


@pytest.mark.parametrize(
    "horizon, batch_size, reward_means",
    [
        (1, 2, [1.5, 3.5, 1.5, 3.5]),
        (1, 3, [2.0, 7 / 3, 8 / 3, 3.0]),
        (3, 6, [1.5, 3.5, 1.5, 3.5]),
        (3, 9, [2.0, 7 / 3, 8 / 3, 3.0]),
        (3, 4, [1.25, 2.5, 3.75, 1.25]),
        (3, 5, [1.4, 3.0, 2.2, 2.4]),
    ],
)
def test_epoch_visits_contiguous_batches_wraps_and_preserves_rollout(
    horizon, batch_size, reward_means
):
    trainer = PPOTrainer.Create(
        MarkedTerminalEnv.Create(), ConstantPolicyCritic(), num_envs=4,
        num_steps_epoch=horizon, num_minibatches=4, minibatch_size=batch_size,
        optimizer=_sgd, learning_rate=0.1, anneal_learning_rate=False,
        max_grad_norm=float("inf"), ppo_clip_eps=100.0,
        ppo_value_coeff=1.0, ppo_entropy_coeff=0.0,
    )
    trainer.env = replace(trainer.env, env_params={
        **trainer.env.env_params, "marker": jnp.arange(1.0, 5.0),
    })
    trainer, td, metrics = trainer.epoch(trainer, jnp.asarray(0))

    # Terminal rewards make the analytical SGD updates independent of horizon.
    # The size-three case must wrap inside a batch, not clamp the slice start:
    # [1,2,3], [4,1,2], [3,4,1], [2,3,4].
    prediction = 0.0
    advantages = []
    for reward_mean in reward_means:
        advantages.append(reward_mean - prediction)
        prediction += 0.1 * advantages[-1]
    assert jnp.allclose(trainer.model.bias[...], prediction)
    assert jnp.allclose(metrics["actor_loss"], -sum(advantages) / 4)
    assert jnp.array_equal(
        td.reward, jnp.broadcast_to(jnp.arange(1.0, 5.0), (horizon, 4))
    )
    assert jnp.array_equal(td.value, jnp.zeros((horizon, 4)))
    assert jnp.array_equal(td.ratio, jnp.ones((horizon, 4)))
    expected_lp = -0.5 * jnp.square(td.action[..., 0]) - 0.5 * jnp.log(2 * jnp.pi)
    assert jnp.allclose(td.log_prob, expected_lp)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"num_envs": 5, "num_steps_epoch": 64, "num_minibatches": 3},
            r"num_envs \(5\).*num_steps_epoch \(64\).*"
            r"env.max_num_agents \(1\) = 320.*num_minibatches=3",
        ),
        (
            {"num_envs": 4, "num_steps_epoch": 4, "minibatch_size": 17},
            r"minibatch_size=17.*\[4, 16\].*num_envs \(4\).*"
            r"num_steps_epoch \(4\).*env.max_num_agents \(1\) = 16",
        ),
        (
            {"num_envs": 32, "num_steps_epoch": 100, "minibatch_size": 8},
            r"minibatch_size=8.*\[100, 3200\]",
        ),
        (
            {"num_envs": 4, "num_steps_epoch": 4, "num_minibatches": 8},
            r"minibatch_size=2.*\[4, 16\].*num_minibatches \(8\)",
        ),
    ],
)
def test_sequential_batching_rejects_silent_rounding(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PPOTrainer.Create(MarkedTerminalEnv.Create(), ConstantPolicyCritic(), **kwargs)


@pytest.mark.parametrize(
    "explicit_size, expected_size",
    [(None, 400), (100, 100), (101, 101), (800, 800), (3200, 3200)],
)
def test_minibatch_size_counts_transitions(explicit_size, expected_size):
    trainer = PPOTrainer.Create(
        MarkedTerminalEnv.Create(), ConstantPolicyCritic(), num_envs=32,
        num_steps_epoch=100, num_minibatches=8, minibatch_size=explicit_size,
        optimizer=_sgd,
    )
    assert trainer.minibatch_size == expected_size
    assert trainer.num_minibatches == 8


def test_default_batch_can_split_sequences_when_total_divides_exactly():
    trainer = PPOTrainer.Create(
        MarkedTerminalEnv.Create(), ConstantPolicyCritic(), num_envs=5,
        num_steps_epoch=64, num_minibatches=4, optimizer=_sgd,
    )
    assert trainer.minibatch_size == 80


@pytest.mark.parametrize("batch_size", range(3, 13))
def test_transition_selection_has_exact_size_and_wraps(batch_size):
    # Agent-major transition IDs: agent 0 owns 0..2, agent 1 owns 3..5, etc.
    for update in range(12):
        indices, mask = _minibatch_selection(3, 4, batch_size, jnp.asarray(update))
        positions = indices[None, :] * 3 + jnp.arange(3)[:, None]
        selected = sorted(positions[mask].tolist())
        expected = sorted((update * batch_size + i) % 12 for i in range(batch_size))
        assert selected == expected
        assert len(set(indices.tolist())) == len(indices)


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
            1, 1, nnx.Rngs(17), hidden_features=4, lstm_features=4,
            carry_leading_shape=(1,),
        )
    else:
        from jaxdem.rl.models.mingru import MinGRUActorCritic

        model = MinGRUActorCritic(
            1, 1, nnx.Rngs(17), hidden_features=4, gru_features=4, num_layers=2,
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
        # Replay both timesteps, but differentiate only the final-step loss.
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
