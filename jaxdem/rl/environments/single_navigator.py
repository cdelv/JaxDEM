# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
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
        min_box_size: float = 1.0,
        max_box_size: float = 2.0,
        max_steps: int = 2000,
        final_reward: float = 0.05,
        shaping_factor: float = 1.0,
        prev_shaping_factor: float = 0.0,
        goal_threshold: float = 2 / 3,
    ) -> "SingleNavigator":
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
            prev_shaping_factor=jnp.asarray(prev_shaping_factor, dtype=float),
            goal_threshold=jnp.asarray(goal_threshold, dtype=float),
            prev_rew=jnp.zeros_like(state.rad),
        )

        return cls(
            state=state,
            system=system,
            env_params=env_params,
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="SingleNavigator.reset")
    def reset(env: "Environment", key: ArrayLike) -> "Environment":
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
        root = key
        key_box = jax.random.fold_in(root, jnp.uint32(0))
        key_pos = jax.random.fold_in(root, jnp.uint32(1))
        key_objective = jax.random.fold_in(root, jnp.uint32(2))
        key_vel = jax.random.fold_in(root, jnp.uint32(4))

        N = env.max_num_agents
        dim = env.state.dim
        rad = 0.05

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
        env.env_params["prev_rew"] = jnp.zeros_like(env.state.rad)
        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="SingleNavigator.step")
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

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SingleNavigator.observation")
    def observation(env: "Environment") -> jax.Array:
        """
        Build per-agent observations.

        Contents per agent
        ------------------
        - Wrapped displacement to objective ``Δx`` (shape ``(2,)``).
        - Velocity ``v`` (shape ``(2,)``).

        Returns
        -------
        jax.Array
            Array of shape ``(N, 2 * dim)`` scaled by the
            maximum box size for normalization.
        """
        return jnp.concatenate(
            [
                env.system.domain.displacement(
                    env.state.pos, env.env_params["objective"], env.system
                ),
                env.state.vel,
            ],
            axis=-1,
        ) / jnp.max(env.system.domain.box_size)

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SingleNavigator.reward")
    def reward(env: "Environment") -> jax.Array:
        r"""
        Returns a vector of per-agent rewards.

        **Equation**

        Let :math:`\delta_i=\operatorname{displacement}(\mathbf{x}_i,\mathbf{objective})`,
        :math:`d_i=\lVert\delta_i\rVert_2`, and :math:`\mathbf{1}[\cdot]` the indicator.
        With shaping factors :math:`\alpha_{\text{prev}},\alpha`, final reward :math:`R_f`,
        and radius :math:`r_i`:

        .. math::

           \mathrm{rew}^{\text{shape}}_i
           = \alpha_{\text{prev}}\,\mathrm{rew}^{\text{prev}}_i - \alpha\, d_i

        Final reward with global shaping penalty (mean distance across agents
        weighted by :math:`\beta`):

        .. math::

           \mathrm{rew}_i = \mathrm{rew}^{\text{shape}}_i + R_f\,\mathbf{1}[\,d_i < \text{goal\_threshold} r_i\,] - \beta\,\operatorname{mean}_k(d_k)

            Parameters
            ----------
            env : Environment
                Current environment.
        """
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        d = jnp.vecdot(delta, delta)
        d = jnp.sqrt(d)
        on_goal = d < env.env_params["goal_threshold"] * env.state.rad
        rew = (
            env.env_params["prev_shaping_factor"] * env.env_params["prev_rew"]
            - env.env_params["shaping_factor"] * d
        )
        env.env_params["prev_rew"] = rew
        reward = rew + env.env_params["final_reward"] * on_goal
        return reward.reshape(env.max_num_agents) / jnp.max(env.system.domain.box_size)

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="SingleNavigator.done")
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
        return 2 * self.state.dim


__all__ = ["SingleNavigator"]
