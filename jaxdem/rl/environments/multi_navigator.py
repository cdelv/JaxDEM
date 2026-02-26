# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Multi-agent 2D navigation with collision avoidance and objective sensing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from functools import partial
from typing import Tuple, cast

from . import Environment
from ...state import State
from ...system import System
from ...utils import lidar_2d
from ...utils.linalg import unit
from ...materials import MaterialTable, Material
from ...material_matchmakers import MaterialMatchmaker


@partial(jax.jit, static_argnames=("N",))
@partial(jax.named_call, name="multi_navigator._sample_objectives")
def _sample_objectives(key: ArrayLike, N: int, box: jax.Array, rad: float) -> jax.Array:
    r"""
    Sample *N* positions on a jittered 2-D grid.
    """
    i = jax.lax.iota(int, N)
    Lx, Ly = box.astype(float)

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
        [jnp.maximum(0.0, dx / 2 - rad), jnp.maximum(0.0, dy / 2 - rad)]
    )
    return base + noise


@Environment.register("multiNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MultiNavigator(Environment):
    r"""
    Multi-agent 2-D navigation with collision avoidance and objective sensing.

    Each agent controls a force vector applied directly to a sphere inside a
    reflective box.  Viscous drag ``-friction * vel`` is added every step.
    The reward blends an individual exponential-shaping term with a
    cooperative team signal:

    .. math::

        r_i = \tfrac{1}{2}\,(\alpha\,s_i + \beta\,g_i) + \tfrac{1}{2}\,\overline{g}

    where :math:`s_i = e^{-2d_i} - e^{-2d_i^{\text{prev}}}` is the
    potential-based shaping, :math:`g_i` is a goal reward that activates
    when the agent is within two radii of its objective, and
    :math:`\overline{g}` is the team-average goal reward.

    Notes
    -----
    The observation vector per agent is:

    ====================================  =====================
    Feature                               Size
    ====================================  =====================
    Unit direction to objective           ``dim``
    Clamped displacement / lidar_range    ``dim``
    Velocity                              ``dim``
    LiDAR proximity (normalised)          ``n_lidar_rays``
    Radial relative velocity              ``n_lidar_rays``
    ====================================  =====================
    """

    n_lidar_rays: int = field(metadata={"static": True})
    """Number of angular bins for each LiDAR sensor."""

    @classmethod
    @partial(jax.named_call, name="MultiNavigator.Create")
    def Create(
        cls,
        N: int = 64,
        min_box_size: float = 1.0,
        max_box_size: float = 1.0,
        box_padding: float = 5.0,
        max_steps: int = 5760,
        friction: float = 0.5,
        shaping_weight: float = 1.0,
        goal_weight: float = 0.05,
        goal_sharpness: float = 10.0,
        goal_radius_factor: float = 2.0,
        lidar_range: float = 0.3,
        n_lidar_rays: int = 16,
    ) -> MultiNavigator:
        r"""
        Create a multi-agent navigator environment.

        Parameters
        ----------
        N : int
            Number of agents.
        min_box_size, max_box_size : float
            Range for the random square domain side length sampled at each
            :meth:`reset`.
        box_padding : float
            Extra padding around the domain in multiples of the particle
            radius.
        max_steps : int
            Episode length in physics steps.
        friction : float
            Viscous drag coefficient applied as ``-friction * vel``.
        shaping_weight : float
            Weight :math:`\alpha` of the exponential potential-based shaping
            term :math:`e^{-2d} - e^{-2d^{\text{prev}}}`.
        goal_weight : float
            Weight :math:`\beta` of the goal proximity reward that activates
            when the agent is within two radii of its objective.
        goal_sharpness : float
            Decay rate :math:`k` in the goal reward
            :math:`e^{-k\,d}`.  Larger values make the reward sharper
            and more localised around the objective.
        goal_radius_factor : float
            Multiplicative factor :math:`f` applied to the particle radius
            to define the goal activation threshold
            :math:`d < f \cdot r`.  Larger values widen the zone in
            which the goal reward is active.
        lidar_range : float
            Maximum detection range for the LiDAR sensor.
        n_lidar_rays : int
            Number of angular LiDAR bins spanning
            :math:`[-\pi, \pi)`.

        Returns
        -------
        MultiNavigator
            A freshly constructed environment (call :meth:`reset` before
            use).
        """
        dim = 2
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = dict(
            objective=jnp.zeros_like(state.pos),
            prev_dist=jnp.zeros_like(state.rad),
            min_box_size=jnp.asarray(min_box_size, dtype=float),
            max_box_size=jnp.asarray(max_box_size, dtype=float),
            box_padding=jnp.asarray(box_padding, dtype=float),
            max_steps=jnp.asarray(max_steps, dtype=int),
            friction=jnp.asarray(friction, dtype=float),
            shaping_weight=jnp.asarray(shaping_weight, dtype=float),
            goal_weight=jnp.asarray(goal_weight, dtype=float),
            goal_sharpness=jnp.asarray(goal_sharpness, dtype=float),
            goal_radius_factor=jnp.asarray(goal_radius_factor, dtype=float),
            lidar_range=jnp.asarray(lidar_range, dtype=float),
            lidar=jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
            lidar_idx=jnp.zeros((state.N, int(n_lidar_rays)), dtype=int),
        )

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            n_lidar_rays=int(n_lidar_rays),
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="MultiNavigator.reset")
    def reset(env: Environment, key: ArrayLike) -> Environment:
        """Reset the environment to a random initial configuration.

        Parameters
        ----------
        env : Environment
            The environment instance to reset (donated / consumed).
        key : ArrayLike
            PRNG key used to sample the domain, positions, objectives,
            and initial velocities.

        Returns
        -------
        Environment
            The environment with a fresh episode state.
        """
        key_box, key_pos, key_objective, key_shuffle, key_vel = jax.random.split(key, 5)
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
        padding = env.env_params["box_padding"] * rad
        pos = _sample_objectives(key_pos, int(N), box + padding, rad) - padding / 2
        objective = _sample_objectives(key_objective, int(N), box, rad)
        perm = jax.random.permutation(key_shuffle, jnp.arange(N, dtype=int))
        env.env_params["objective"] = objective[perm]

        vel = jax.random.uniform(
            key_vel, (N, dim), minval=-0.1, maxval=0.1, dtype=float
        )
        env.state = State.create(pos=pos, vel=vel, rad=rad * jnp.ones(N))

        matcher = MaterialMatchmaker.create("harmonic")
        mat_table = MaterialTable.from_materials(
            [Material.create("elastic", density=0.27, young=6e3, poisson=0.3)],
            matcher=matcher,
        )
        env.system = System.create(
            env.state.shape,
            dt=0.004,
            domain_type="reflect",
            domain_kw=dict(
                box_size=box + padding,
                anchor=jnp.zeros_like(box) - padding / 2,
            ),
            mat_table=mat_table,
        )

        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        env.env_params["prev_dist"] = jnp.sqrt(jnp.vecdot(delta, delta))

        _, _, env.env_params["lidar"], env.env_params["lidar_idx"], _ = lidar_2d(
            env.state,
            env.system,
            env.env_params["lidar_range"],
            cast(int, getattr(env, "n_lidar_rays")),
            env.max_num_agents,
        )

        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="MultiNavigator.step")
    def step(env: Environment, action: jax.Array) -> Environment:
        """Advance the environment by one physics step.

        Parameters
        ----------
        env : Environment
            Current environment (donated / consumed).
        action : jax.Array
            Force actions for every agent, shape ``(N * dim,)``.

        Returns
        -------
        Environment
            Updated environment after the physics integration.
        """
        force = (
            action.reshape(env.max_num_agents, *env.action_space_shape)
            - env.env_params["friction"] * env.state.vel
        )
        env.system = env.system.force_manager.add_force(env.state, env.system, force)

        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        env.env_params["prev_dist"] = jnp.sqrt(jnp.vecdot(delta, delta))

        env.state, env.system = env.system.step(env.state, env.system)

        _, _, env.env_params["lidar"], env.env_params["lidar_idx"], _ = lidar_2d(
            env.state,
            env.system,
            env.env_params["lidar_range"],
            cast(int, getattr(env, "n_lidar_rays")),
            env.max_num_agents,
            sense_edges=True,
        )

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.observation")
    def observation(env: Environment) -> jax.Array:
        """Build the per-agent observation vector.

        Returns
        -------
        jax.Array
            Observation matrix of shape ``(N, obs_dim)``.  See the class
            docstring for the feature layout.
        """
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )

        # Radial relative velocity: project onto ray direction (+ outward)
        lidar_idx = env.env_params["lidar_idx"]
        is_agent = lidar_idx >= 0
        safe_idx = jnp.where(is_agent, lidar_idx, 0)
        rel_vel = env.state.vel[safe_idx] - env.state.vel[:, None, :]

        n_rays = lidar_idx.shape[-1]
        angles = jnp.linspace(-jnp.pi, jnp.pi, n_rays, endpoint=False) + jnp.pi / n_rays
        ray_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        lidar_vr = jnp.sum(rel_vel * ray_dirs, axis=-1)
        lidar_vr = jnp.where(is_agent * (env.env_params["lidar"] > 0), lidar_vr, 0.0)

        return jnp.concatenate(
            [
                unit(delta),
                jnp.clip(
                    delta, -env.env_params["lidar_range"], env.env_params["lidar_range"]
                ),
                env.state.vel,
                env.env_params["lidar"],
                lidar_vr,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.reward")
    def reward(env: Environment) -> jax.Array:
        r"""Compute the per-agent reward.

        The reward combines exponential potential-based shaping with a
        proximity-gated goal reward and a cooperative team average:

        .. math::

            r_i = \tfrac{1}{2}\,(\alpha\,s_i + \beta\,g_i) + \tfrac{1}{2}\,\overline{g}

        Returns
        -------
        jax.Array
            Reward vector of shape ``(N,)``.
        """
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        dist = jnp.sqrt(jnp.vecdot(delta, delta))
        shaping = jnp.exp(-2 * dist) - jnp.exp(-2 * env.env_params["prev_dist"])

        at_goal = dist < env.env_params["goal_radius_factor"] * env.state.rad
        goal_rew = at_goal * jnp.exp(-env.env_params["goal_sharpness"] * dist)

        individual = (
            env.env_params["shaping_weight"] * shaping
            + env.env_params["goal_weight"] * goal_rew
        )

        # Difference reward: r_i = G - G_{-i}
        r_team = jnp.sum(individual)
        r_team_without = r_team - individual
        return r_team - r_team_without

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="MultiNavigator.done")
    def done(env: Environment) -> jax.Array:
        """Return ``True`` when the episode has exceeded ``max_steps``."""
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """Number of scalar actions per agent (equal to ``dim``)."""
        return self.state.dim

    @property
    def action_space_shape(self) -> Tuple[int]:
        """Shape of a single agent's action (``(dim,)``)."""
        return (self.state.dim,)

    @property
    def observation_space_size(self) -> int:
        """Dimensionality of a single agent's observation vector."""
        return 3 * self.state.dim + 2 * self.n_lidar_rays


__all__ = ["MultiNavigator"]
