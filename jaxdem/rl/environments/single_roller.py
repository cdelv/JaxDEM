# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where a single agent rolls towards a target on the floor."""

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
from ...utils.linalg import norm, norm2, unit, cross


@partial(jax.named_call, name="single_roller.frictional_wall_force")
def frictional_wall_force(
    pos: jax.Array, state: State, system: System
) -> Tuple[jax.Array, jax.Array]:
    r"""
    Normal, frictional, and restitution forces for a sphere on a :math:`z = 0` plane.

    Combines a linear spring in the normal direction with Coulomb tangential
    friction and a velocity-proportional dashpot for restitution damping.

    Parameters
    ----------
    pos : jax.Array
        Particle positions, shape ``(N, 3)``.
    state : State
        Full simulation state (provides ``vel``, ``ang_vel``, ``rad``, ``mass``).
    system : System
        System configuration (provides ``dt``).

    Returns
    -------
    total_force : jax.Array
        Per-particle force, shape ``(N, 3)``.
    total_torque : jax.Array
        Per-particle torque, shape ``(N, 3)``.
    """
    k = 1e5
    mu = 0.4
    restitution = 0.1
    n = jnp.array([0.0, 0.0, 1.0])
    p = jnp.array([0.0, 0.0, 0.0])

    # Normal force
    dist = jnp.dot(pos - p, n) - state.rad
    penetration = jnp.minimum(0.0, dist)
    force_n = -k * penetration[..., None] * n

    # Normal velocity damping (restitution)
    v_n_scalar = jnp.sum(state.vel * n, axis=-1, keepdims=True)
    in_contact = (penetration < 0)[..., None]
    c_n = 2.0 * (1.0 - restitution) * jnp.sqrt(k * state.mass[..., None])
    c_n = jnp.minimum(c_n, 0.5 * state.mass[..., None] / system.dt)
    force_damping = -c_n * v_n_scalar * n * in_contact

    # Velocity at contact point
    radius_vec = -state.rad[..., None] * n
    v_at_contact = state.vel + cross(state.ang_vel, radius_vec)
    v_n = jnp.sum(v_at_contact * n, axis=-1, keepdims=True) * n
    v_t = v_at_contact - v_n

    # Coulomb friction
    f_t_mag = mu * jnp.sum(force_n * n, axis=-1, keepdims=True)
    t_dir = unit(v_t)
    force_t = -f_t_mag * t_dir

    total_force = force_n + force_damping + force_t
    total_torque = cross(radius_vec, force_t)
    return total_force, total_torque


@Environment.register("singleRoller3D")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SingleRoller3D(Environment):
    r"""
    Single-agent 3D navigation via torque-controlled rolling.

    The agent is a sphere resting on a :math:`z = 0` floor under gravity.
    Actions are 3-D torque vectors; translational motion arises from
    frictional contact with the floor (see :func:`frictional_wall_force`).
    A viscous drag ``-friction * vel`` and a fixed angular damping of
    ``0.05 * ang_vel`` are applied each step.

    The reward uses exponential potential-based shaping:

    .. math::

       \mathrm{rew} = e^{-2\,d} - e^{-2\,d^{\mathrm{prev}}}

    Notes
    -----
    The observation vector per agent is:

    ============================  =========
    Feature                       Size
    ============================  =========
    Unit direction to objective   2
    Clamped displacement (x, y)   2
    Velocity (x, y)               2
    Angular velocity              3
    ============================  =========
    """

    @classmethod
    @partial(jax.named_call, name="SingleRoller3D.Create")
    def Create(
        cls,
        min_box_size: float = 2.0,
        max_box_size: float = 2.0,
        max_steps: int = 1000,
        friction: float = 0.5,
        work_weight: float = 0.0,
    ) -> SingleRoller3D:
        """
        Create a single-agent roller environment.

        Parameters
        ----------
        min_box_size, max_box_size : float
            Range for the random square domain side length.
        max_steps : int
            Episode length in physics steps.
        friction : float
            Viscous drag coefficient applied as ``-friction * vel``.
        work_weight : float
            Penalty coefficient for large actions.

        Returns
        -------
        SingleRoller3D
            A freshly constructed environment (call :meth:`reset` before use).
        """
        dim = 3
        N = 1
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = dict(
            objective=jnp.zeros_like(state.pos),
            min_box_size=jnp.asarray(min_box_size, dtype=float),
            max_box_size=jnp.asarray(max_box_size, dtype=float),
            max_steps=jnp.asarray(max_steps, dtype=int),
            friction=jnp.asarray(friction, dtype=float),
            work_weight=jnp.asarray(work_weight, dtype=float),
            delta=jnp.zeros_like(state.pos),
            prev_dist=jnp.zeros_like(state.rad),
            curr_dist=jnp.zeros_like(state.rad),
            action=jnp.zeros_like(state.ang_vel),
        )

        return cls(
            state=state,
            system=system,
            env_params=env_params,
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="SingleRoller3D.reset")
    def reset(env: Environment, key: ArrayLike) -> Environment:
        """
        Randomly place the agent and objective on the floor.

        Parameters
        ----------
        env : Environment
            Current environment instance.
        key : ArrayLike
            JAX PRNG key.

        Returns
        -------
        Environment
            Freshly initialised environment.
        """
        key_box, key_pos, key_objective, key_vel = jax.random.split(key, 4)
        N = env.max_num_agents
        dim = env.state.dim
        rad_val = 0.05
        box = jax.random.uniform(
            key_box,
            (dim,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
            dtype=float,
        )
        min_pos = rad_val * jnp.ones_like(box)
        pos = jax.random.uniform(
            key_pos,
            (N, dim),
            minval=min_pos,
            maxval=box - min_pos,
            dtype=float,
        )
        pos = pos.at[:, 2].set(rad_val)
        objective = jax.random.uniform(
            key_objective,
            (N, dim),
            minval=min_pos,
            maxval=box - min_pos,
            dtype=float,
        )
        objective = objective.at[:, 2].set(rad_val)
        env.env_params["objective"] = objective
        vel = jax.random.uniform(
            key_vel, (N, dim), minval=-0.05, maxval=0.05, dtype=float
        )
        rad = rad_val * jnp.ones(N)
        env.state = State.create(pos=pos, vel=vel, rad=rad)
        env.system = System.create(
            env.state.shape,
            domain_type="reflect",
            domain_kw=dict(box_size=box, anchor=[0.0, 0.0, -1.0 * rad_val]),
            force_manager_kw=dict(
                gravity=[0.0, 0.0, -10.0],
                force_functions=(frictional_wall_force,),
            ),
        )
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        dist = norm(delta)
        env.env_params["delta"] = delta
        env.env_params["prev_dist"] = dist
        env.env_params["curr_dist"] = dist
        env.env_params["action"] = jnp.zeros_like(env.state.ang_vel)
        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="SingleRoller3D.step")
    def step(env: Environment, action: jax.Array) -> Environment:
        """
        Apply a torque action, advance physics by one step.

        Parameters
        ----------
        env : Environment
            Current environment.
        action : jax.Array
            3-D torque vector per agent.

        Returns
        -------
        Environment
            Updated environment after one physics step.
        """
        reshaped_action = action.reshape(env.max_num_agents, *env.action_space_shape)
        env.env_params["action"] = reshaped_action
        torque = reshaped_action - 0.07 * env.state.ang_vel
        force = -env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.system = env.system.force_manager.add_torque(env.state, env.system, torque)
        env.env_params["prev_dist"] = env.env_params["curr_dist"]
        env.state, env.system = env.system.step(env.state, env.system)
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        env.env_params["delta"] = delta
        env.env_params["curr_dist"] = norm(delta)
        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SingleRoller3D.observation")
    def observation(env: Environment) -> jax.Array:
        """
        Per-agent observation vector.

        Contents per agent:

        - Unit displacement to objective projected to x-y (shape ``(2,)``).
        - Clamped displacement to objective projected to x-y (shape ``(2,)``).
        - Velocity projected to x-y (shape ``(2,)``).
        - Angular velocity (shape ``(3,)``).

        Returns
        -------
        jax.Array
            Shape ``(N, 9)``.
        """
        delta = env.env_params["delta"]
        delta_2d = delta[..., :2]
        vel_2d = env.state.vel[..., :2]
        return jnp.concatenate(
            [
                unit(delta_2d),
                jnp.clip(delta_2d, -1.0, 1.0),
                vel_2d,
                env.state.ang_vel,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SingleRoller3D.reward")
    def reward(env: Environment) -> jax.Array:
        r"""
        Returns a vector of per-agent rewards.

        Exponential potential-based shaping:

        .. math::

           \mathrm{rew}_i = e^{-2 \cdot d_i} - e^{-2 \cdot d_i^{\mathrm{prev}}}

        Returns
        -------
        jax.Array
            Shape ``(N,)``.
        """
        shaping_reward = jnp.exp(-2 * env.env_params["curr_dist"]) - jnp.exp(
            -2 * env.env_params["prev_dist"]
        )
        work_penalty = env.env_params["work_weight"] * norm2(env.env_params["action"])
        return shaping_reward - work_penalty

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="SingleRoller3D.done")
    def done(env: Environment) -> jax.Array:
        """``True`` when ``step_count`` exceeds ``max_steps``."""
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """Per-agent flattened action dimensionality (3-D torque)."""
        return 3

    @property
    def action_space_shape(self) -> Tuple[int]:
        """Per-agent action tensor shape."""
        return (3,)

    @property
    def observation_space_size(self) -> int:
        """Per-agent flattened observation dimensionality (9)."""
        return 9


__all__ = ["SingleRoller3D"]
