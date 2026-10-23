# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""PPO action collection and latent ratios using a synthetic training fixture."""

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace

import distrax
import numpy as np
import pytest
from flax import nnx

from jaxdem.rl.action_spaces import BoxSpace, MaxNormSpace, Transformed
from jaxdem.rl.environments import Environment, ThreeGears
from jaxdem.rl.env_wrappers import vectorise_env
from jaxdem.state import State
from jaxdem.system import System
from jaxdem.rl.models import SharedActorCritic, MinGRUActorCritic, Model
from jaxdem.rl.trainers import Trainer
from jaxdem.rl.trainers.ppo_trainer import PPOTrainer
from jaxdem.utils import advance_action
from tests.test_ppo_alignment import _loss_args


@pytest.mark.parametrize(
    "middle_done,expected", [(False, [2.75, 3.5, 3.0]), (True, [2.0, 2.0, 3.0])]
)
def test_drip_epoch_returns_the_known_backward_reward_sequence(
    monkeypatch, middle_done, expected
):
    from tests.test_caps import trajectory
    from tests.test_ppo_math import BanditModel, TimeLimitCounterEnv

    trainer = PPOTrainer.Create(
        TimeLimitCounterEnv.Create(),
        BanditModel(),
        num_envs=1,
        num_steps_epoch=3,
        num_minibatches=1,
        drip_decay=0.5,
        learning_rate=0.0,
    )
    original = trajectory(
        jnp.zeros((3, 1, 1, 1)),
        done=jnp.array([False, middle_done, False]).reshape(3, 1, 1),
    )
    original = replace(original, reward=jnp.array([1.0, 2.0, 3.0]).reshape(3, 1, 1))

    def rollout(env, graphdef, graphstate, key, *args, **kwargs):
        return env, graphstate, key, original

    monkeypatch.setattr(PPOTrainer, "trajectory_rollout", staticmethod(rollout))
    # A fresh trace prevents a cached epoch from retaining an unrelated rollout.
    _, collected, _ = jax.jit(PPOTrainer.epoch.__wrapped__)(trainer, jnp.array(0))
    np.testing.assert_allclose(collected.reward[:, 0], expected)
    np.testing.assert_array_equal(original.reward[:, 0, 0], [1.0, 2.0, 3.0])


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class CheckpointCounter(Environment):
    @classmethod
    def Create(cls):
        state = State.create(pos=jnp.zeros((1, 2)))
        return cls(
            state,
            System.create(state.shape),
            {
                "count": jnp.asarray(0),
                "limit": jnp.asarray(100),
                "terminal": jnp.asarray(False),
                "checkpoints": jnp.asarray(0),
                "previous": jnp.asarray(0.0),
            },
        )

    @staticmethod
    def reset(env, key):
        return replace(
            env,
            env_params={
                **env.env_params,
                "count": jnp.asarray(0),
                "checkpoints": jnp.asarray(0),
                "previous": jnp.asarray(0.0),
            },
        )

    @staticmethod
    def step(env, action):
        return replace(
            env, env_params={**env.env_params, "count": env.env_params["count"] + 1}
        )

    @staticmethod
    def checkpoint(env, action):
        p = env.env_params
        return replace(
            env,
            env_params={
                **p,
                "previous": p["count"].astype(float) ** 2,
                "checkpoints": p["checkpoints"] + 1,
            },
        )

    @staticmethod
    def observation(env):
        return env.env_params["count"].astype(float).reshape(1, 1)

    @staticmethod
    def reward(env):
        return (
            env.env_params["count"].astype(float) ** 2 - env.env_params["previous"]
        )[None]

    @staticmethod
    def terminated(env):
        p = env.env_params
        return (p["count"] >= p["limit"]) & p["terminal"]

    @staticmethod
    def truncated(env):
        p = env.env_params
        return (p["count"] >= p["limit"]) & ~p["terminal"]


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ActionLookupEnv(CheckpointCounter):
    """Two slots, one inactive; discrete actions must really be valid indices."""

    @classmethod
    def Create(cls):
        env = CheckpointCounter.Create()
        state = env.state.add(env.state, pos=jnp.ones((1, 2)), rad=jnp.ones(1))
        return cls(
            state,
            env.system,
            {
                **env.env_params,
                "limit": jnp.asarray(2),
                "total": jnp.asarray(0.0),
                "inactive_action": jnp.asarray(0.0),
            },
        )

    @staticmethod
    def agent_mask(env):
        return jnp.array([True, False])

    @staticmethod
    def observation(env):
        return jnp.broadcast_to(env.env_params["count"].astype(float), (2, 1))

    @staticmethod
    def reward(env):
        return jnp.array([env.env_params["total"], 0.0])

    @staticmethod
    def step(env, action):
        if jnp.issubdtype(action.dtype, jnp.integer):
            amount = jnp.array([0.0, 1.0, 4.0])[action[0]]
            inactive = action[1]
        else:
            amount, inactive = action[0, 0], action[1, 0]
        return replace(
            env,
            env_params={
                **env.env_params,
                "count": env.env_params["count"] + 1,
                "total": env.env_params["total"] + amount,
                "inactive_action": inactive.astype(float),
            },
        )


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("model_type", [SharedActorCritic, MinGRUActorCritic])
def test_tiny_ppo_epoch_accepts_both_action_types_with_padding(discrete, model_type):
    kwargs = {}
    if model_type is MinGRUActorCritic:
        kwargs.update(hidden_features=4, gru_features=4, num_layers=2)
    else:
        kwargs.update(architecture=[4])
    model = model_type(
        observation_space_size=1,
        action_space_size=3 if discrete else 1,
        key=nnx.Rngs(8),
        discrete=discrete,
        **kwargs,
    )
    tr = PPOTrainer.Create(
        ActionLookupEnv.Create(),
        model,
        num_envs=2,
        num_steps_epoch=2,
        num_minibatches=2,
        skip_frames=1,
    )
    before = [
        np.asarray(x).copy() for x in jax.tree.leaves(nnx.state(tr.model, nnx.Param))
    ]
    tr, td, metrics = tr.epoch(tr, jnp.asarray(0))
    assert jnp.issubdtype(td.action.dtype, jnp.integer) == discrete
    assert td.action.shape == ((2, 4) if discrete else (2, 4, 1))
    assert jnp.all(td.action[:, 1::2] == 0)
    assert jnp.all(tr.env.env_params["inactive_action"] == 0)
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(metrics))
    assert (td.latent_action is None) == discrete
    action = np.asarray(td.action[:, ::2])
    amount = np.array([0.0, 1.0, 4.0])[action] if discrete else action[..., 0]
    expected_reward = 2 * np.cumsum(amount, axis=0)
    np.testing.assert_allclose(td.reward[:, ::2], expected_reward, rtol=2e-6, atol=1e-7)
    np.testing.assert_array_equal(td.reward[:, 1::2], 0.0)
    np.testing.assert_allclose(
        tr.env.env_params["total"], expected_reward[-1], rtol=2e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        metrics["score"], expected_reward.mean(), rtol=2e-6, atol=1e-7
    )
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            before, jax.tree.leaves(nnx.state(tr.model, nnx.Param)), strict=True
        )
    )


def test_gear_observation_and_reward_use_whole_bodies_and_action_intervals():
    centers = jnp.array([[5.0, 5.0], [9.0, 5.0]])
    angles = jnp.repeat(jnp.array([0.0, jnp.pi / 2]), 2)
    quaternion = jnp.stack(
        (jnp.cos(angles / 2), jnp.zeros(4), jnp.zeros(4), jnp.sin(angles / 2)), axis=-1
    )
    state = State.create(
        pos=jnp.repeat(centers, 2, axis=0),
        pos_p=jnp.tile(jnp.array([[0.1, 0.0], [-0.1, 0.0]]), (2, 1)),
        clump_id=jnp.repeat(jnp.arange(2), 2),
        rad=jnp.full(4, 0.1),
        mass=jnp.ones(4),
        inertia=jnp.ones((4, 1)),
        vel=jnp.tile(jnp.array([[1.0, 0.0]]), (4, 1)),
        q=quaternion,
    )
    system = System.create(
        state=state, dt=0.1, collider_type=None, rotation_integrator_type=None
    )
    env = ThreeGears.Create(num_gears=2)
    env = replace(
        env,
        state=state,
        system=system,
        env_params={
            **env.env_params,
            "objective": centers + jnp.array([2.0, 0.0]),
            "friction": jnp.asarray(0.0),
            "attraction_mag": jnp.asarray(0.0),
        },
    )
    np.testing.assert_allclose(
        env.observation(env)[:, 11:13], [[1, 0], [-1, 0]], atol=1e-6
    )
    action = jnp.zeros((2, 1))
    endpoint, _, _ = advance_action(env, action, skip_frames=2)
    np.testing.assert_allclose(endpoint.reward(endpoint), 0.3, atol=2e-6)
    np.testing.assert_allclose(endpoint.env_params["prev_dist"], 2.0)
    np.testing.assert_allclose(
        endpoint.state.pos_c, state.pos_c + jnp.array([0.3, 0.0]), atol=2e-6
    )
    endpoint = replace(
        endpoint,
        state=replace(endpoint.state, vel=state.vel * jnp.array([3, 3, 2, 2])[:, None]),
    )
    np.testing.assert_allclose(endpoint.reward(endpoint), [-0.1, 0.15], atol=2e-6)
    np.testing.assert_allclose(endpoint.env_params["prev_ke"], 0.5)
    frozen, _, _ = advance_action(endpoint, action, terminated=jnp.asarray(True))
    np.testing.assert_allclose(frozen.reward(frozen), 0.0, atol=2e-6)


def test_partial_triangular_stack_targets_do_not_overlap():
    from jaxdem.rl.environments.two_gears import rr

    env = ThreeGears.Create(num_gears=4)
    env = env.reset(env, jax.random.key(9))
    targets = np.asarray(env.env_params["objective"])
    distances = np.linalg.norm(targets[:, None] - targets[None, :], axis=-1)
    distances[np.diag_indices(4)] = np.inf
    assert distances.min() >= 2 * float(rr) - 2e-6


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


@pytest.mark.parametrize("space", [BoxSpace(-1.0, 1.0), MaxNormSpace()])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_saturated_actions_use_exact_latent_ratios_and_gradients(space, dtype):
    env = vectorise_env(CheckpointCounter.Create(), n=1)
    model = SaturatedPolicy(space, dtype)
    graphdef, graphstate = nnx.split((model,))
    _, _, _, td = Trainer.trajectory_rollout(
        env, graphdef, graphstate, jax.random.key(7), num_steps_epoch=2
    )
    td = jax.tree.map(lambda x: x[:, 0], td)
    args = _loss_args()
    (_, aux), grads = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
        model, td, **args
    )
    np.testing.assert_allclose(aux["ratio"], 1.0, rtol=2e-6, atol=2e-6)
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads))
    # A changed policy must use the original latent, not just force ratios to one.
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
