import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial

from flax import nnx

from ..factory import Factory
from ..state import State
from ..system import System
from .environment import Environment


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class TrajectoryData:
    """
    Container for holding rl trajectory data
    """

    obs: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    done: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class RlAlgorithm(Factory["RlAlgorithm"], ABC):
    """
    Interface for ddefining RL algorithms
    """


@RlAlgorithm.register("ClippedPPO")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class ClippedPPO(RlAlgorithm):
    """
    Interface for ddefining RL algorithms
    """

    @staticmethod
    @jax.jit
    def step(
        env: "Environment",
        key: ArrayLike,
        graphdef: nnx.GraphDef,
        state: nnx.State,
    ) -> Tuple["Environment", ArrayLike, nnx.State, "TrajectoryData"]:
        """
        a
        """

        model, *_ = nnx.merge(graphdef, state)
        key, subkey = jax.random.split(key)

        obs = env.observation(env)
        pi, value = model(obs)
        action = pi.sample(seed=subkey)
        log_prob = pi.log_prob(action)

        action = jnp.clip(action, -1.0, 1.0)
        env = env.step(env, action)
        reward = env.reward(env)
        done = env.done(env)

        trajectory_data = TrajectoryData(obs, action, value, reward, log_prob, done)
        state = nnx.state((model, *_))
        return env, key, state, trajectory_data

    @staticmethod
    @partial(jax.jit, static_argnames=("num_steps"))
    def _steps(
        env: "Environment",
        key: ArrayLike,
        graphdef: nnx.GraphDef,
        state: nnx.State,
        num_steps: int = 1,
    ) -> Tuple["Environment", ArrayLike, nnx.State, "TrajectoryData"]:
        def body(carry, _):
            env, key, state = carry
            env, key, state, trajectory_data = step(env, key, graphdef, state)
            return (env, key, state), trajectory_data

        (env, key, state), trajectory_data = jax.lax.scan(
            body, (env, key, state), None, num_steps
        )
        return env, key, state, trajectory_data

    @partial(jax.jit, static_argnames=("num_steps", "stride"))
    def trajectory_rollout(env, key, graphdef, state, num_steps, stride: int = 1):
        def body(carry, _):
            env, key, state = carry
            env, key, state, trajectory_data = _steps(env, key, graphdef, state, stride)
            return (env, key, state), trajectory_data

        (env, key, state), trajectory_data = jax.lax.scan(
            body, (env, key, state), None, num_steps
        )
        return env, key, state, trajectory_data

    # @jax.jit
    # def trajectory_rollout(env, key, graphdef, state, num_steps_epoch):
    #     pass

    # @jax.jit
    # def calculate_advantages(gae_and_next_value, trajectory_data: TrajectoryData):
    #     pass

    # @nnx.jit
    # def loss_fn(model, traj_batch, gae, targets):
    #     passs

    # @jax.jit
    # def train_batch(carry, batch_info):
    #     pass

    # @jax.jit
    # def epoch(env: "rl.Environment", key, graphdef, state):
    #     pass

    # @jax.jit
    # def reset_if_done(env, key):
    #     pass
