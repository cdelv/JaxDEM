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
from ...utils import unit


@Environment.register("singleNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SingleNavigator(Environment):
    r"""
    Single-agent navigation environment toward a fixed target.

    The agent controls a force vector that is applied directly to a sphere
    inside a reflective box.  Viscous drag ``-friction * vel`` is added
    each step.  The reward uses exponential potential-based shaping:

    .. math::

       \mathrm{rew} = e^{-2\,d} - e^{-2\,d^{\mathrm{prev}}}

    Notes
    -----
    The observation vector per agent is:

    ============================  =========
    Feature                       Size
    ============================  =========
    Unit direction to objective   ``dim``
    Clamped displacement          ``dim``
    Velocity                      ``dim``
    ============================  =========
    """

    @classmethod
    @partial(jax.named_call, name="SingleNavigator.Create")
    def Create(
        cls,
        dim: int = 2,
        min_box_size: float = 2.0,
        max_box_size: float = 2.0,
        max_steps: int = 1000,
        friction: float = 0.5,
    ) -> SingleNavigator:
        """
        Create a single-agent navigator environment.

        Parameters
        ----------
        dim : int
            Spatial dimensionality (2 or 3).
        min_box_size, max_box_size : float
            Range for the random square domain side length.
        max_steps : int
            Episode length in physics steps.
        friction : float
            Viscous drag coefficient applied as ``-friction * vel``.

        Returns
        -------
        SingleNavigator
            A freshly constructed environment (call :meth:`reset` before use).
        """
        N = 1
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = dict(
            objective=jnp.zeros_like(state.pos),
            min_box_size=jnp.asarray(min_box_size, dtype=float),
            max_box_size=jnp.asarray(max_box_size, dtype=float),
            max_steps=jnp.asarray(max_steps, dtype=int),
            friction=jnp.asarray(friction, dtype=float),
            prev_dist=jnp.zeros_like(state.rad),
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
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        env.env_params["prev_dist"] = jnp.sqrt(jnp.vecdot(delta, delta))
        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="SingleNavigator.step")
    def step(env: Environment, action: jax.Array) -> Environment:
        """
        Advance one step. Actions are forces; simple drag is applied (-friction * vel).

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
        # Apply Viscous Friction: F = F_action - c * v
        force = reshaped_action - env.state.vel * env.env_params["friction"]
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        env.env_params["prev_dist"] = jnp.sqrt(jnp.vecdot(delta, delta))
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
        - Unit vector to objective (shape (dim,))  --> Direction
        - Clamped delta to objective (shape (dim,)) --> Local precision
        - Velocity (shape (dim,))

        Returns
        -------
        jax.Array
            Array of shape ``(N, 3 * dim)``
        """
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        return jnp.concatenate(
            [
                unit(delta),
                jnp.clip(delta, -1.0, 1.0),
                env.state.vel,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SingleNavigator.reward")
    def reward(env: Environment) -> jax.Array:
        r"""
        Returns a vector of per-agent rewards.

        **Reward:**

        .. math::

           \mathrm{rew}_i = e^{-2 \cdot d_i} - e^{-2 \cdot d_i^{\mathrm{prev}}}

        where :math:`d_i` is the distance from agent :math:`i` to the objective.

        Parameters
        -----------
        env : Environment
            Current environment.

        Returns
        -------
        jax.Array
            Shape ``(N,)``.
        """
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        dist = jnp.sqrt(jnp.vecdot(delta, delta))
        shaping_reward = jnp.exp(-2 * dist) - jnp.exp(-2 * env.env_params["prev_dist"])
        return shaping_reward

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
        return 3 * self.state.dim
