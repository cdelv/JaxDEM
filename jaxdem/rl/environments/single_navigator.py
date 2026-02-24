# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where a single agent navigates towards a target."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass
from functools import partial
from typing import Tuple

from . import Environment
from ...state import State
from ...system import System


@Environment.register("singleNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SingleNavigator(Environment):
    """Single-agent navigation environment toward a fixed target."""

    @classmethod
    @partial(jax.named_call, name="SingleNavigator.Create")
    def Create(
        cls,
        dim: int = 2,
        min_box_size: float = 2.0,
        max_box_size: float = 2.0,
        max_steps: int = 1000,
        final_reward: float = 10.0,
        shaping_factor: float = 5.0,
        friction: float = 0.5,
        observation_cap: float = 5.0,
        effort_cost: float = 0.05,
        velocity_cost: float = 0.05,
        goal_threshold: float = 2 / 3,
    ) -> SingleNavigator:
        """
        Custom factory method for this environment.
        """
        N = 1
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = dict(
            objective=jnp.zeros_like(state.pos),
            min_box_size=jnp.asarray(min_box_size, dtype=float),
            max_box_size=jnp.asarray(max_box_size, dtype=float),
            max_steps=jnp.asarray(max_steps, dtype=int),
            final_reward=jnp.asarray(final_reward, dtype=float),
            shaping_factor=jnp.asarray(shaping_factor, dtype=float),
            friction=jnp.asarray(friction, dtype=float),
            observation_cap=jnp.asarray(observation_cap, dtype=float),
            effort_cost=jnp.asarray(effort_cost, dtype=float),
            velocity_cost=jnp.asarray(velocity_cost, dtype=float),
            goal_threshold=jnp.asarray(goal_threshold, dtype=float),
            prev_rew=jnp.zeros_like(state.rad),
            last_action=jnp.zeros_like(state.pos),
        )

        return cls(
            state=state,
            system=system,
            env_params=env_params,
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="SingleNavigator.reset")
    def reset(env: Environment, key: ArrayLike) -> Environment:
        """
        Initialize the environment with a randomly placed particle and velocity.

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
        key_box, key_pos, key_objective, key_vel = jax.random.split(key, 4)
        N = env.max_num_agents
        dim = env.state.dim
        rad = jnp.array(0.05, dtype=float)

        box = jax.random.uniform(
            key_box,
            (dim,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
            dtype=float,
        )

        min_pos = rad * jnp.ones_like(box)
        pos = jax.random.uniform(
            key_pos,
            (N, dim),
            minval=min_pos,
            maxval=box - min_pos,
            dtype=float,
        )

        objective = jax.random.uniform(
            key_objective,
            (N, dim),
            minval=min_pos,
            maxval=box - min_pos,
            dtype=float,
        )
        env.env_params["objective"] = objective

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

        # Initialize previous distance for shaping
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        d = jnp.sqrt(jnp.vecdot(delta, delta))
        env.env_params["prev_rew"] = d
        env.env_params["last_action"] = jnp.zeros_like(vel)

        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="SingleNavigator.step")
    def step(env: Environment, action: jax.Array) -> Environment:
        """
        Advance one step. Actions are forces; viscous friction is applied.

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
        reshaped_action = action.reshape(env.max_num_agents, *env.action_space_shape)

        # Store action for effort penalty
        env.env_params["last_action"] = reshaped_action

        # Apply Viscous Friction: F = F_action - c * v
        force = reshaped_action - env.state.vel * env.env_params["friction"]
        env.system = env.system.force_manager.add_force(env.state, env.system, force)

        # Update Distance (for shaping)
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        d = jnp.vecdot(delta, delta)
        env.env_params["prev_rew"] = jnp.sqrt(d)

        env.state, env.system = env.system.step(env.state, env.system)

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SingleNavigator.observation")
    def observation(env: Environment) -> jax.Array:
        """
        Build per-agent observations.

        Contents per agent
        ------------------
        - Unit vector to objective (shape ``(2,)``).
        - Velocity ``v`` (shape ``(2,)``).
        - Absolute distance to objective, capped (shape ``(1,)``).

        Returns
        -------
        jax.Array
            Array of shape ``(N, 2 * dim + 1)``.
        """
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        dist = jnp.sqrt(jnp.vecdot(delta, delta))

        # Safe unit vector calculation
        unit_vec = delta / (dist[..., None] + 1e-6)

        # Capped distance
        capped_dist = jnp.minimum(dist, env.env_params["observation_cap"])

        return jnp.concatenate(
            [
                unit_vec,
                env.state.vel,
                capped_dist[..., None],
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SingleNavigator.reward")
    def reward(env: Environment) -> jax.Array:
        r"""
        Returns a vector of per-agent rewards.

        Goals:
        1. Move quickly (Distance Shaping)
        2. Least effort (Action Penalty)
        3. Stay still at goal (Velocity Penalty)
        4. Reach goal (Goal Bonus)

        Returns
        -------
        jax.Array
            Shape ``(N,)``. The normalized per-agent reward vector.
        """
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        dist = jnp.sqrt(jnp.vecdot(delta, delta))

        # 1. Shaping: (d_t-1 - d_t) * scale
        # Positive if moving closer
        shaping = (env.env_params["prev_rew"] - dist) * env.env_params["shaping_factor"]

        # 2. Effort Penalty: - w * ||action||^2
        action_mag_sq = jnp.vecdot(
            env.env_params["last_action"], env.env_params["last_action"]
        )
        effort_penalty = -env.env_params["effort_cost"] * action_mag_sq

        # 3. Velocity Penalty: - w * ||vel||^2
        # Encourages stopping, especially when shaping reward is 0 (at goal)
        vel_mag_sq = jnp.vecdot(env.state.vel, env.state.vel)
        velocity_penalty = -env.env_params["velocity_cost"] * vel_mag_sq

        # 4. Goal Bonus
        on_goal = dist < env.env_params["goal_threshold"] * env.state.rad
        goal_bonus = env.env_params["final_reward"] * on_goal

        return shaping + effort_penalty + velocity_penalty + goal_bonus

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="SingleNavigator.done")
    def done(env: Environment) -> jax.Array:
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

    @property
    def action_space_size(self) -> int:
        """
        Flattened action size per agent. Actions passed to :meth:`step` have shape ``(A, action_space_size)``.
        """
        return self.state.dim

    @property
    def action_space_shape(self) -> Tuple[int]:
        """
        Original per-agent action shape (useful for reshaping inside the environment).
        """
        return (self.state.dim,)

    @property
    def observation_space_size(self) -> int:
        """
        Flattened observation size per agent. :meth:`observation` returns shape ``(A, observation_space_size)``.
        """
        return 2 * self.state.dim + 1


__all__ = ["SingleNavigator"]
