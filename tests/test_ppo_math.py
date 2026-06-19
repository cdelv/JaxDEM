import math
import jax
import jax.numpy as jnp
import distrax
import pytest
import optax
from flax import linen as nn
from flax import nnx

from jaxdem.rl.trainers import Trainer
from jaxdem.rl.trainers.ppo_trainer import PPOTrainer
from jaxdem.rl.environments import Environment
from jaxdem.rl.models import Model
from jaxdem.system import System
from jaxdem.state import State
from jaxdem.rl.action_spaces import ActionSpace


def test_gae_analytical():
    rewards = jnp.array([[1.0], [2.0]])
    values = jnp.array([[0.5], [0.5]])
    dones = jnp.array([[0.0], [1.0]])
    ratios = jnp.array([[1.0], [1.0]])

    returns, advantages = Trainer.compute_advantages(
        value=values,
        reward=rewards,
        ratio=ratios,
        done=dones,
        advantage_rho_clip=jnp.array(1.0),
        advantage_c_clip=jnp.array(1.0),
        advantage_gamma=jnp.array(1.0),
        advantage_lambda=jnp.array(1.0),
        last_value=jnp.array([0.0]),
        unroll=1,
    )

    expected_advantages = jnp.array([[2.5], [1.5]])
    assert jnp.allclose(advantages, expected_advantages)
    assert jnp.allclose(returns, expected_advantages + values)


def test_clipped_surrogate_objective():
    def clipped_actor_loss(ratio, advantage, eps):
        ratio_bounded = jnp.where(
            advantage >= 0,
            jnp.minimum(ratio, 1.0 + eps),
            jnp.maximum(ratio, 1.0 - eps),
        )
        return -(advantage * ratio_bounded)

    eps = 0.2

    l1 = clipped_actor_loss(jnp.array(1.1), jnp.array(1.0), eps)
    assert jnp.allclose(l1, -1.1)

    l2 = clipped_actor_loss(jnp.array(1.5), jnp.array(1.0), eps)
    assert jnp.allclose(l2, -1.2)

    l3 = clipped_actor_loss(jnp.array(0.5), jnp.array(-1.0), eps)
    assert jnp.allclose(l3, 0.8)


def test_entropy_continuous():
    k = 3
    mu = jnp.zeros(k)
    sigma = jnp.ones(k)

    dist = distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    entropy = dist.entropy()

    expected_entropy = (k / 2.0) * math.log(2 * math.pi * math.e)
    assert jnp.allclose(entropy, expected_entropy)


def test_gradient_detachment():
    def advantage_fn(values, rewards):
        dones = jnp.zeros_like(rewards)
        ratios = jnp.ones_like(rewards)
        ret, adv = Trainer.compute_advantages(
            value=values,
            reward=rewards,
            ratio=ratios,
            done=dones,
            advantage_rho_clip=jnp.array(1.0),
            advantage_c_clip=jnp.array(1.0),
            advantage_gamma=jnp.array(1.0),
            advantage_lambda=jnp.array(1.0),
            last_value=jnp.array([0.0]),
            unroll=1,
        )
        return ret.sum() + adv.sum()

    values = jnp.array([[0.5], [0.5]])
    rewards = jnp.array([[1.0], [2.0]])

    grads = jax.grad(advantage_fn, argnums=(0, 1))(values, rewards)

    assert jnp.all(grads[0] == 0.0)
    assert jnp.all(grads[1] == 0.0)


@Model.register("BanditModel")
class BanditModel(Model):
    def __init__(
        self, observation_space_size=1, action_space_size=1, action_space=None, key=None
    ):
        self.action_mean = nnx.Param(jnp.zeros((1,)))
        self.value_bias = nnx.Param(jnp.zeros((1,)))

    def __call__(self, x, sequence=False, **kwargs):
        mean = jnp.broadcast_to(self.action_mean[...], (*x.shape[:-1], 1))
        sigma = jnp.broadcast_to(jnp.array([0.1]), (*x.shape[:-1], 1))
        dist = distrax.MultivariateNormalDiag(mean, sigma)
        val = jnp.broadcast_to(self.value_bias[...], (*x.shape[:-1], 1))
        return dist, val


from dataclasses import dataclass


@Environment.register("StatelessBandit")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class StatelessBanditEnv(Environment):
    @classmethod
    def Create(cls, dim: int = 2) -> "Environment":
        state = State.create(pos=jnp.zeros((1, 2)))
        system = System.create(state.shape)
        return cls(state=state, system=system, env_params={"reward": jnp.zeros((1,))})

    @staticmethod
    def reset(env, key):
        return env

    @staticmethod
    def step(env, action):
        reward = jnp.where(action[..., 0] > 0, 1.0, -1.0)
        reward = jnp.reshape(reward, (1,))
        from dataclasses import replace

        return replace(env, env_params={"reward": reward})

    @staticmethod
    def reward(env):
        return env.env_params.get("reward", jnp.zeros((env.max_num_agents,)))

    @staticmethod
    def done(env):
        return jnp.array(True, dtype=bool)

    @staticmethod
    def observation(env):
        return jnp.zeros((env.max_num_agents, 1))

    @property
    def action_space_size(self):
        return 1

    @property
    def action_space_shape(self):
        return (1,)

    @property
    def observation_space_size(self):
        return 1


def test_stateless_bandit():
    env = StatelessBanditEnv.Create()
    model = BanditModel(
        observation_space_size=1,
        action_space_size=1,
        action_space=ActionSpace.create("maxNorm", max_norm=1.0),
    )

    tr = PPOTrainer.Create(
        env=env,
        model=model,
        key=jax.random.PRNGKey(0),
        num_epochs=100,
        num_envs=128,
        num_minibatches=1,
        num_steps_epoch=2,
        optimizer=optax.adam,
        learning_rate=0.1,
    )

    tr = tr.train(tr, verbose=False, log=False)

    model, optimizer = nnx.merge(tr.graphdef, tr.graphstate)

    dist, val = model(jnp.zeros((1, 1)))

    # Verify the agent learned to output a positive mean action
    assert dist.mean() > 0.5
    # Verify the value function predicts the expected return (which should be ~1.0)
    assert val.mean() > 0.5


@Environment.register("DeterministicCorridor")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DeterministicCorridorEnv(Environment):
    @classmethod
    def Create(cls, dim: int = 2) -> "Environment":
        state = State.create(pos=jnp.zeros((1, 2)))
        system = System.create(state.shape)
        return cls(
            state=state,
            system=system,
            env_params={
                "reward": jnp.zeros((1,)),
                "done": jnp.array(False, dtype=bool),
            },
        )

    @staticmethod
    def reset(env, key):
        from dataclasses import replace

        new_pos = jnp.zeros_like(env.state.pos_c)
        new_state = replace(env.state, pos_c=new_pos, pos_p=new_pos)
        return replace(
            env,
            state=new_state,
            env_params={
                "reward": jnp.zeros((1,)),
                "done": jnp.array(False, dtype=bool),
            },
        )

    @staticmethod
    def step(env, action):
        pos_x = env.state.pos_c[..., 0] + action[..., 0]
        pos_y = env.state.pos_c[..., 1]
        new_pos = jnp.stack([pos_x, pos_y], axis=-1)
        from dataclasses import replace

        new_state = replace(env.state, pos_c=new_pos, pos_p=new_pos)

        reached_goal = pos_x >= 5.0
        reward = jnp.where(reached_goal, 10.0, -0.1)
        reward = jnp.reshape(reward, (env.max_num_agents,))

        done = jnp.reshape(reached_goal, ())

        from dataclasses import replace

        return replace(
            env, state=new_state, env_params={"reward": reward, "done": done}
        )

    @staticmethod
    def reward(env):
        return env.env_params.get("reward", jnp.zeros((env.max_num_agents,)))

    @staticmethod
    def done(env):
        d = env.env_params.get("done", jnp.array(False, dtype=bool))
        return jnp.reshape(d, ())

    @staticmethod
    def observation(env):
        return env.state.pos_c[..., 0:1]

    @property
    def action_space_size(self):
        return 1


def test_deterministic_corridor():
    from jaxdem.rl.models.mlp import SharedActorCritic

    env = DeterministicCorridorEnv.Create()
    model = SharedActorCritic(
        observation_space_size=1,
        action_space_size=1,
        action_space=ActionSpace.create("maxNorm", max_norm=2.0),
        key=nnx.Rngs(0),
    )

    tr = PPOTrainer.Create(
        env=env,
        model=model,
        key=jax.random.PRNGKey(0),
        num_epochs=300,
        num_envs=256,
        num_minibatches=4,
        num_steps_epoch=8,
        optimizer=optax.adam,
        learning_rate=3e-3,
        advantage_gamma=0.99,
    )

    tr = tr.train(tr, verbose=False, log=False)

    model, optimizer = nnx.merge(tr.graphdef, tr.graphstate)
    dist, val = model(jnp.zeros((1, 1)))
    mean = dist.distribution.mean()

    # Should strongly prefer moving right
    assert mean[0, 0] > 0.55
