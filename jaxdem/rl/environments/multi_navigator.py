# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Multi-agent navigation task with collision penalties."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from functools import partial

from . import Environment
from ...state import State
from ...system import System
from ...utils import lidar


@partial(jax.jit, static_argnames=("N",))
@partial(jax.named_call, name="multi_navigator._sample_objectives")
def _sample_objectives(key: ArrayLike, N: int, box: jax.Array) -> jax.Array:
    i = jax.lax.iota(jnp.int32, N)  # 0..N-1
    Lx, Ly = box.astype(jnp.float32)

    nx = jnp.ceil(jnp.sqrt(N * Lx / Ly)).astype(int)
    ny = jnp.ceil(N / nx).astype(int)

    ix = jnp.mod(i, nx)
    iy = i // nx

    dx = Lx / nx
    dy = Ly / ny

    xs = (ix + 0.5) * dx
    ys = (iy + 0.5) * dy
    base = jnp.stack([xs, ys], axis=1)

    noise = jax.random.uniform(key, (N, 2), minval=-1.0, maxval=1.0) * jnp.asarray(
        [dx / 4, dy / 4]
    )
    return base + noise


@Environment.register("multiNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MultiNavigator(Environment):
    """
    Multi-agent navigation environment with collision penalties.

    Agents seek fixed objectives in a 2D reflective box. Each step applies a
    force-like action, advances simple dynamics, updates LiDAR, and returns
    shaped rewards with an optional final bonus on goal.
    """

    n_lidar_rays: int = field(default=16, metadata={"static": True})
    """
    Number of lidar rays for the vision system.
    """

    @classmethod
    @partial(jax.named_call, name="MultiNavigator.Create")
    def Create(
        cls,
        N: int = 2,
        min_box_size: float = 1.0,
        max_box_size: float = 2.0,
        max_steps: int = 4000,
        final_reward: float = 0.05,  # 1.0
        shaping_factor: float = 1.0,
        prev_shaping_factor: float = 0.0,
        global_shaping_factor: float = 0.1,
        collision_penalty: float = -0.05,
        goal_threshold: float = 2 / 3,
        lidar_range: float = 0.45,
        n_lidar_rays: int = 16,
    ) -> "MultiNavigator":
        dim = 2
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = dict(
            objective=jnp.zeros_like(state.pos),
            min_box_size=jnp.asarray(min_box_size, dtype=float),
            max_box_size=jnp.asarray(max_box_size, dtype=float),
            max_steps=jnp.asarray(max_steps, dtype=int),
            final_reward=jnp.asarray(final_reward, dtype=float),
            collision_penalty=jnp.asarray(collision_penalty, dtype=float),
            shaping_factor=jnp.asarray(shaping_factor, dtype=float),
            prev_shaping_factor=jnp.asarray(prev_shaping_factor, dtype=float),
            global_shaping_factor=jnp.asarray(global_shaping_factor, dtype=float),
            goal_threshold=jnp.asarray(goal_threshold, dtype=float),
            prev_rew=jnp.zeros_like(state.rad),
            lidar_range=jnp.asarray(lidar_range, dtype=float),
            lidar=jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
        )
        action_space_size = dim
        action_space_shape = (dim,)
        observation_space_size = 2 * dim + n_lidar_rays

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            max_num_agents=N,
            action_space_size=action_space_size,
            action_space_shape=action_space_shape,
            observation_space_size=observation_space_size,
            n_lidar_rays=int(n_lidar_rays),
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.reset")
    def reset(env: "Environment", key: ArrayLike) -> "Environment":
        """
        Initialize the environment with randomly placed particles and velocities.

        Parameters
        ----------
        env: Environment
            Current environment instance.

        key : jax.random.PRNGKey
            JAX random number generator key.

        Returns
        -------
        Environment
            Freshly initialized environment.
        """
        root = key
        key_box = jax.random.fold_in(root, jnp.uint32(0))
        key_pos = jax.random.fold_in(root, jnp.uint32(1))
        key_objective = jax.random.fold_in(root, jnp.uint32(2))
        key_shuffle = jax.random.fold_in(root, jnp.uint32(3))
        key_vel = jax.random.fold_in(root, jnp.uint32(4))

        N = env.max_num_agents
        dim = env.state.dim
        box = jax.random.uniform(
            key_box,
            (dim,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
            dtype=float,
        )

        rad = 0.05
        pos = _sample_objectives(key_pos, int(N), box - 2 * rad) + rad
        objective = _sample_objectives(key_objective, int(N), box - 2 * rad) + rad
        env.env_params["objective"] = jax.random.permutation(key_shuffle, objective)

        vel = jax.random.uniform(
            key_vel, (N, dim), minval=-0.1, maxval=0.1, dtype=float
        )

        rad = rad * jnp.ones(N)
        env.state = State.create(pos=pos, vel=vel, rad=rad)
        env.system = System.create(
            env.state.shape,
            domain_type="reflect",
            domain_kw=dict(box_size=box, anchor=jnp.zeros_like(box)),
        )
        env.env_params["prev_rew"] = jnp.zeros_like(env.state.rad)

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.step")
    def step(env: "Environment", action: jax.Array) -> "Environment":
        """
        Advance one step. Actions are forces; simple drag is applied.

        Parameters
        ----------
        env : Environment
            The current environment.

        action : jax.Array
            The vector of actions each agent in the environment should take.

        Returns
        -------
        Environment
            The updated environment state.
        """
        force = (
            action.reshape(env.max_num_agents, *env.action_space_shape)
            - jnp.sign(env.state.vel) * 0.08
        )
        env.system = env.system.force_manager.add_force(env.state, env.system, force)

        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        d = jnp.vecdot(delta, delta)
        env.env_params["prev_rew"] = jnp.sqrt(d)

        env.state, env.system = env.system.step(env.state, env.system)

        env.env_params["lidar"] = lidar(env)

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.observation")
    def observation(env: "Environment") -> jax.Array:
        """
        Build per-agent observations.

        Contents per agent
        ------------------
        - Wrapped displacement to objective ``Δx`` (shape ``(2,)``).
        - Velocity ``v`` (shape ``(2,)``).
        - LiDAR proximities (shape ``(n_lidar_rays,)``).

        Returns
        -------
        jax.Array
            Array of shape ``(N, 2 * dim + n_lidar_rays)`` scaled by the
            maximum box size for normalization.
        """
        return jnp.concatenate(
            [
                env.system.domain.displacement(
                    env.env_params["objective"], env.state.pos, env.system
                ),
                env.state.vel,
                env.env_params["lidar"],
            ],
            axis=-1,
        ) / jnp.max(env.system.domain.box_size)

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.reward")
    def reward(env: "Environment") -> jax.Array:
        r"""
        Per-agent reward with distance shaping, goal bonus, and LiDAR penalties.

        Definitions
        -----------
        Let ``δ_i = displacement(x_i, objective)`` and ``d_i = ||δ_i||_2``.
        A “too-close” LiDAR hit occurs when proximity exceeds
        ``τ_i = max(0, R - κ r_i)`` with safety factor ``κ=2.0`` (approx.).

        Reward
        ------
        ``rew_i = (prev_shaping_factor * prev_rew_i - shaping_factor * d_i) + final_reward * 1[d_i < r_i * goal_threshold] + collision_penalty * (#hits_i) - global_shaping_factor * mean(d)``

        Returns
        -------
        jax.Array
            Shape ``(N,)`` reward vector, normalized by max box size.
        """
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        d = jnp.vecdot(delta, delta)
        d = jnp.sqrt(d)
        rew = (
            env.env_params["prev_shaping_factor"] * env.env_params["prev_rew"]
            - env.env_params["shaping_factor"] * d
        )

        closeness_thresh = jnp.maximum(
            0.0, env.env_params["lidar_range"] - 2.0 * env.state.rad[:, None]
        )
        n_hits = (
            (env.env_params["lidar"] > closeness_thresh).sum(axis=-1).astype(rew.dtype)
        )

        on_goal = d < env.env_params["goal_threshold"] * env.state.rad
        reward = (
            rew
            + env.env_params["final_reward"] * on_goal
            + env.env_params["collision_penalty"] * n_hits
        )
        reward -= jnp.mean(d) * env.env_params["global_shaping_factor"]
        return reward.reshape(env.max_num_agents) / jnp.max(env.system.domain.box_size)

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.done")
    def done(env: "Environment") -> jax.Array:
        """
        Returns a boolean indicating whether the environment has ended.
        The episode terminates when the maximum number of steps is reached.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            Boolean array indicating whether the episode has ended.
        """
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])


__all__ = ["MultiNavigator"]
