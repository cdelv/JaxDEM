# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Environment where a single agent navigates towards a target."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from . import Environment
from ...state import State
from ...system import System


@Environment.register("singleNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class SingleNavigator(Environment):
    """
    Defines an environment with a single sphere that has to travel to
    a pre-defined point in space.
    """

    @classmethod
    def Create(
        cls,
        dim: int = 2,
        min_box_size: float = 1.0,
        max_box_size: float = 2.0,
        max_steps: int = 2000,
        final_reward: float = 0.05,
        shaping_factor: float = 1.0,
    ) -> "SingleNavigator":
        """
        Custom factory method for this environment.
        """
        N = 1
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(dim)

        env_params = dict(
            objective=jnp.zeros_like(state.pos),
            min_box_size=jnp.asarray(min_box_size, dtype=float),
            max_box_size=jnp.asarray(max_box_size, dtype=float),
            max_steps=jnp.asarray(max_steps, dtype=int),
            final_reward=jnp.asarray(final_reward, dtype=float),
            shaping_factor=jnp.asarray(shaping_factor, dtype=float),
            prev_rew=jnp.zeros_like(state.rad),
        )
        action_space_size = dim
        action_space_shape = (dim,)
        observation_space_size = 2 * dim

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            max_num_agents=N,
            action_space_size=action_space_size,
            action_space_shape=action_space_shape,
            observation_space_size=observation_space_size,
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    def reset(env: "Environment", key: ArrayLike) -> "Environment":
        """
        Creates a particle inside the domain at a random initial position
        with a random initial velocity.

        Parameters
        ----------
        env: Environment
            Current environment

        key : jax.random.PRNGKey
            Jax random numbers key

        Returns
        -------
        Environment
            Freshly initialized environment.
        """
        key, key_pos, key_vel, key_box, key_objective = jax.random.split(key, 5)

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
        state = State.create(pos=pos, vel=vel, rad=rad)
        system = System.create(
            env.state.dim,
            domain_type="reflect",
            domain_kw=dict(box_size=box, anchor=jnp.zeros_like(box)),
        )
        env = replace(env, state=state, system=system)
        env.env_params["prev_rew"] = jnp.zeros_like(env.state.rad)
        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env", "action"))
    def step(env: "Environment", action: jax.Array) -> "Environment":
        """
        Advances the simulation state by a time steps. actions are interpreted as acceleration.

        Parameters
        ----------
        env : Environment
            The current environment.

        action : System
            The vector of actions each agent on the environment should take.

        Returns
        -------
        Environment
            The updated envitonment state.
        """
        a = action.reshape(env.max_num_agents, *env.action_space_shape)
        state = replace(env.state, accel=a - jnp.sign(env.state.vel) * 0.08)
        state, system = env.system.step(state, env.system)
        return replace(env, state=state, system=system)

    @staticmethod
    @jax.jit
    def observation(env: "Environment") -> jax.Array:
        """
        Return a vector corresponding to the environment observation. the obs is the displacement vector between the
        position of the particle and objective and the particle's velocity.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            Vector corresponding to the environment observation.
        """
        return jnp.concatenate(
            [
                env.system.domain.displacement(
                    env.state.pos, env.env_params["objective"], env.system
                ),
                env.state.vel,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    def reward(env: "Environment") -> jax.Array:
        r"""
        Return a vector of per-agent rewards.

        **Equation**

        Let :math:`\delta_i = \mathrm{displacement}(\mathbf{x}_i, \mathbf{objective})`,
        :math:`d_i = \lVert \delta_i \rVert_2`,
        and :math:`\mathbf{1}[\cdot]` the indicator. With
        shaping factor :math:`\alpha`, final reward :math:`R_f`,
        radius :math:`r_i`, and previous reward :math:`rew^{\text{prev}}_i`:

        .. math::

           rew^{\text{shape}}_i \;=\; rew^{\text{prev}}_i \;-\; \alpha\, d_i

        .. math::

           rew_i \;=\; rew^{\text{shape}}_i \;+\; R_f \,\mathbf{1}[\,d_i < r_i\,]

        The function updates :math:`rew^{\text{prev}}_i \leftarrow rew^{\text{shape}}_i`

        Parameters
        ----------
        env : Environment
            Current environment.
        """
        pos = env.state.pos
        objective = env.env_params["objective"]
        delta = env.system.domain.displacement(pos, objective, env.system)
        d = jnp.linalg.norm(delta, axis=-1)
        on_goal = d < env.state.rad
        rew = env.env_params["prev_rew"] - d * env.env_params["shaping_factor"]
        env.env_params["prev_rew"] = rew
        reward = rew + env.env_params["final_reward"] * on_goal
        return jnp.asarray(reward).reshape(env.max_num_agents)

    @staticmethod
    @jax.jit
    def done(env: "Environment") -> jax.Array:
        """
        Return a bool indicating when the environment ended. Its done when the max number of steps are reached.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            A bool indicating when the environment ended
        """
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])


__all__ = ["SingleNavigator"]
