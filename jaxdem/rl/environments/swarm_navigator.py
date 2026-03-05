# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Multi-agent 2-D swarm navigation toward shared objectives."""

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
from ...materials import MaterialTable, Material
from ...material_matchmakers import MaterialMatchmaker


@partial(jax.jit, static_argnames=("N",))
@partial(jax.named_call, name="swarm_navigator._sample_objectives")
def _sample_objectives(key: ArrayLike, N: int, box: jax.Array, rad: float) -> jax.Array:
    r"""Sample *N* positions on a jittered 2-D grid.

    Parameters
    ----------
    key : ArrayLike
        PRNG key for jitter noise.
    N : int
        Number of positions to sample.
    box : jax.Array
        Domain extents, shape ``(2,)``.
    rad : float
        Particle radius used to clamp the jitter noise.

    Returns
    -------
    jax.Array
        Positions of shape ``(N, 2)``.
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


@Environment.register("swarmNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SwarmNavigator(Environment):
    r"""Multi-agent 2-D swarm navigation with cooperative difference rewards.

    Each agent controls a force vector applied directly to a sphere inside a
    reflective box.  Viscous drag ``-friction * vel`` is added every step.
    Objectives are shared among all agents; each agent dynamically tracks
    its nearest objectives.

    The reward encourages unique target acquisition using *Difference
    Rewards*:

    .. math::

        R_i = R_i^{\text{goal}} + D_i^{\text{team}} - P_i^{\text{work}}
              - P_i^{\text{seek}}

    where :math:`D_i^{\text{team}} = G - G_{-i}` is the difference reward
    with :math:`G = w_g \bigl(\sum_j c_j\bigr)^2`, :math:`c_j` is 1 when
    agent *j* uniquely occupies a goal, and the seek penalty
    :math:`P_i^{\text{seek}}` discourages stalling when few empty
    objectives are visible.

    Notes
    -----
    The observation vector per agent is:

    ====================================  =====================
    Feature                               Size
    ====================================  =====================
    Velocity                              ``dim``
    LiDAR proximity (normalised)          ``n_lidar_rays``
    Unit direction to top *k* objectives  ``k_objectives * dim``
    Clamped displacement to top *k*       ``k_objectives * dim``
    Occupancy status of top *k*           ``k_objectives``
    ====================================  =====================
    """

    n_lidar_rays: int = field(metadata={"static": True})
    """Number of angular bins for the agent-to-agent LiDAR sensor."""
    k_objectives: int = field(metadata={"static": True})
    """Number of closest objectives tracked per agent."""

    @classmethod
    @partial(jax.named_call, name="SwarmNavigator.Create")
    def Create(
        cls,
        N: int = 64,
        min_box_size: float = 1.0,
        max_box_size: float = 1.0,
        box_padding: float = 20.0,
        max_steps: int = 5760,
        friction: float = 0.2,
        goal_weight: float = 0.002,
        global_weight: float = 0.0018,
        goal_radius_factor: float = 1.0,
        work_weight: float = 0.002,
        seek_weight: float = 0.55,
        lidar_range: float = 0.4,
        n_lidar_rays: int = 8,
        k_objectives: int = 4,
    ) -> SwarmNavigator:
        r"""
        Create a swarm navigator environment.

        Parameters
        ----------
        N : int
            Number of agents.
        min_box_size, max_box_size : float
            Range for the random square domain side length sampled at
            each :meth:`reset`.
        box_padding : float
            Extra padding around the domain in multiples of the particle
            radius.
        max_steps : int
            Episode length in physics steps.
        friction : float
            Viscous drag coefficient applied as ``-friction * vel``.
        goal_weight : float
            Weight of the individual goal reward for uniquely occupying
            a target.
        global_weight : float
            Weight :math:`w_g` of the quadratic team synergy reward
            :math:`G = w_g (\sum_j c_j)^2`.
        goal_radius_factor : float
            Multiplicative factor :math:`f` applied to the particle
            radius to define the goal activation threshold
            :math:`d < f \cdot r`.
        work_weight : float
            Weight of the quadratic action penalty :math:`\|a\|^2`.
        seek_weight : float
            Weight of the exploration penalty that discourages stalling
            when few empty objectives are visible.
        lidar_range : float
            Maximum detection range for the LiDAR sensor.
        n_lidar_rays : int
            Number of angular LiDAR bins spanning
            :math:`[-\pi, \pi)`.
        k_objectives : int
            Number of closest objectives tracked per agent.

        Returns
        -------
        SwarmNavigator
            A freshly constructed environment (call :meth:`reset` before
            use).
        """
        dim = 2
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = dict(
            objective=jnp.zeros_like(state.pos),
            action=jnp.zeros_like(state.pos),
            min_box_size=jnp.asarray(min_box_size, dtype=float),
            max_box_size=jnp.asarray(max_box_size, dtype=float),
            box_padding=jnp.asarray(box_padding, dtype=float),
            max_steps=jnp.asarray(max_steps, dtype=int),
            friction=jnp.asarray(friction, dtype=float),
            goal_weight=jnp.asarray(goal_weight, dtype=float),
            global_weight=jnp.asarray(global_weight, dtype=float),
            goal_radius_factor=jnp.asarray(goal_radius_factor, dtype=float),
            work_weight=jnp.asarray(work_weight, dtype=float),
            seek_weight=jnp.asarray(seek_weight, dtype=float),
            top_k_units=jnp.zeros((N, int(k_objectives) * dim), dtype=float),
            clipped_deltas=jnp.zeros((N, int(k_objectives) * dim), dtype=float),
            top_k_occupied=jnp.zeros((N, int(k_objectives)), dtype=float),
            lidar_range=jnp.asarray(lidar_range, dtype=float),
            lidar=jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
        )

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            n_lidar_rays=int(n_lidar_rays),
            k_objectives=int(k_objectives),
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="SwarmNavigator.reset")
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
        key_box, key_pos, key_objective, key_vel = jax.random.split(key, 4)
        N = env.max_num_agents
        dim = env.state.dim
        k = cast(int, getattr(env, "k_objectives"))
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

        env.env_params["objective"] = objective
        env.env_params["action"] = jnp.zeros_like(env.state.pos)

        # Agent-to-agent LiDAR
        _, _, env.env_params["lidar"], _, _ = lidar_2d(
            env.state,
            env.system,
            env.env_params["lidar_range"],
            cast(int, getattr(env, "n_lidar_rays")),
            env.max_num_agents,
        )

        # Top-k objective sensors
        deltas = env.system.domain.displacement(
            env.state.pos[:, None, :], objective[None, :, :], env.system
        )
        dist = jnp.linalg.norm(deltas, axis=-1)
        _, top_k_idx = jax.lax.top_k(-dist, k)
        top_k_deltas = jnp.take_along_axis(deltas, top_k_idx[..., None], axis=1)
        top_k_dist = jnp.take_along_axis(dist, top_k_idx, axis=1)
        top_k_units = top_k_deltas / jnp.maximum(top_k_dist[..., None], 1e-6)
        clip_range = env.env_params["lidar_range"]
        clipped = top_k_units * jnp.minimum(top_k_dist[..., None], clip_range)

        thresh = env.env_params["goal_radius_factor"] * env.state.rad[0]
        at_obj = (dist < thresh).astype(jnp.float32)
        count_at = jnp.sum(at_obj, axis=0)
        occ_others = (count_at[None, :] - at_obj) > 0
        top_k_occ = jnp.take_along_axis(
            occ_others.astype(jnp.float32), top_k_idx, axis=1
        )

        env.env_params["top_k_units"] = top_k_units.reshape(N, k * dim)
        env.env_params["clipped_deltas"] = clipped.reshape(N, k * dim)
        env.env_params["top_k_occupied"] = top_k_occ

        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="SwarmNavigator.step")
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
        N = env.max_num_agents
        dim = env.state.dim
        k = cast(int, getattr(env, "k_objectives"))

        force = (
            action.reshape(N, *env.action_space_shape)
            - env.env_params["friction"] * env.state.vel
        )
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.env_params["action"] = action.reshape(N, *env.action_space_shape)

        env.state, env.system = env.system.step(env.state, env.system)

        # Agent-to-agent LiDAR
        _, _, env.env_params["lidar"], _, _ = lidar_2d(
            env.state,
            env.system,
            env.env_params["lidar_range"],
            cast(int, getattr(env, "n_lidar_rays")),
            env.max_num_agents,
            sense_edges=True,
        )

        # Top-k objective sensors
        objective = env.env_params["objective"]
        deltas = env.system.domain.displacement(
            env.state.pos[:, None, :], objective[None, :, :], env.system
        )
        dist = jnp.linalg.norm(deltas, axis=-1)
        _, top_k_idx = jax.lax.top_k(-dist, k)
        top_k_deltas = jnp.take_along_axis(deltas, top_k_idx[..., None], axis=1)
        top_k_dist = jnp.take_along_axis(dist, top_k_idx, axis=1)
        top_k_units = top_k_deltas / jnp.maximum(top_k_dist[..., None], 1e-6)
        clip_range = env.env_params["lidar_range"]
        clipped = top_k_units * jnp.minimum(top_k_dist[..., None], clip_range)

        thresh = env.env_params["goal_radius_factor"] * env.state.rad[0]
        at_obj = (dist < thresh).astype(jnp.float32)
        count_at = jnp.sum(at_obj, axis=0)
        occ_others = (count_at[None, :] - at_obj) > 0
        top_k_occ = jnp.take_along_axis(
            occ_others.astype(jnp.float32), top_k_idx, axis=1
        )

        env.env_params["top_k_units"] = top_k_units.reshape(N, k * dim)
        env.env_params["clipped_deltas"] = clipped.reshape(N, k * dim)
        env.env_params["top_k_occupied"] = top_k_occ

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmNavigator.observation")
    def observation(env: Environment) -> jax.Array:
        """Build the per-agent observation vector.

        Returns
        -------
        jax.Array
            Observation matrix of shape ``(N, obs_dim)``.  See the class
            docstring for the feature layout.
        """
        return jnp.concatenate(
            [
                env.state.vel,
                env.env_params["lidar"],
                env.env_params["top_k_units"],
                env.env_params["clipped_deltas"],
                env.env_params["top_k_occupied"],
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmNavigator.reward")
    def reward(env: Environment) -> jax.Array:
        r"""Compute the per-agent cooperative reward using Difference Rewards.

        1. Individual goal reward for uniquely occupying a target:

           .. math::

               R_i^{\text{goal}} = w_{\text{goal}} \cdot c_i

        2. Team difference reward:

           .. math::

               D_i = G - G_{-i}, \quad
               G = w_g \Bigl(\sum_j c_j\Bigr)^2

        3. Penalties (work effort and seek):

           .. math::

               P_i^{\text{work}} = w_{\text{work}} \|a_i\|^2, \quad
               P_i^{\text{seek}} = w_{\text{seek}} \cdot s_i \cdot e^{-10\|v_i\|}

        Returns
        -------
        jax.Array
            Reward vector of shape ``(N,)``.
        """
        N = env.max_num_agents
        dim = env.state.dim
        k = cast(int, getattr(env, "k_objectives"))
        thresh = env.env_params["goal_radius_factor"] * env.state.rad[0]

        clipped_deltas = env.env_params["clipped_deltas"].reshape(N, k, dim)
        top_k_occupied = env.env_params["top_k_occupied"]

        dists = jnp.linalg.norm(clipped_deltas, axis=-1)
        occ_by_others = top_k_occupied > 0.5
        self_occ = dists < thresh

        contrib = self_occ & ~occ_by_others
        agent_contrib = jnp.any(contrib, axis=-1).astype(jnp.float32)

        # Individual goal reward
        ind_goal = env.env_params["goal_weight"] * agent_contrib

        # Team difference reward: D_i = G - G_{-i}
        total_occ = jnp.sum(agent_contrib)
        G_team = env.env_params["global_weight"] * (total_occ**2)
        G_without_i = env.env_params["global_weight"] * (
            (total_occ - agent_contrib) ** 2
        )
        D_team = G_team - G_without_i

        # Work penalty
        work = env.env_params["work_weight"] * jnp.sum(
            env.env_params["action"] ** 2, axis=-1
        )

        # Seek penalty: penalise low speed when few empty objectives visible
        is_occupied = self_occ | occ_by_others
        on_goal = jnp.any(self_occ, axis=-1)
        num_empty = jnp.sum(~is_occupied, axis=-1)
        should_seek = (num_empty <= 1) & ~on_goal
        speed = jnp.linalg.norm(env.state.vel, axis=-1)
        seek = env.env_params["seek_weight"] * should_seek * jnp.exp(-10.0 * speed)

        return ind_goal + D_team - work - seek

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="SwarmNavigator.done")
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
        return (
            self.state.dim
            + self.n_lidar_rays
            + self.k_objectives * self.state.dim
            + self.k_objectives * self.state.dim
            + self.k_objectives
        )


__all__ = ["SwarmNavigator"]
