# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Multi-agent 2-D swarm navigation with potential-based rewards."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from functools import partial

from . import Environment
from ...state import State
from ...system import System
from ...utils import lidar_2d, cross_lidar_2d
from ...materials import MaterialTable, Material
from ...material_matchmakers import MaterialMatchmaker
from ...utils.linalg import dot, norm, norm2, unit_and_norm


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
    r"""Multi-agent 2-D swarm navigation with potential-based rewards.

    Each agent controls a force vector applied directly to a sphere inside a
    reflective box.  Viscous drag ``-friction * vel`` is added every step.
    Objectives are shared among all agents; each agent dynamically tracks
    its *k* nearest objectives.  The potential-based shaping signal is
    computed independently for each of the *k* objectives and summed.
    Occupancy is determined via strict symmetry breaking: only the closest
    agent to each objective within the activation threshold may claim it.

    **Reward**

    .. math::

        R_i = w_s\,\sum_{j \in \text{top-}k}
                  (e^{-2d_{ij}} - e^{-2d_{ij}^{\mathrm{prev}}})
              + w_g\,\mathbf{1}[d_i < f \cdot r_i]
              - w_c\,\left\|\sum_j l_j\,\hat{r}_j\right\|
              - w_w\,\|a_i\|^2
              + w_v\,\mathbf{1}[\text{all }k\text{ occupied}]
              - \bar{r}_i

    where :math:`\bar{r}_i` is an EMA baseline updated with factor
    :math:`\alpha`.  All weights are constructor parameters stored in
    ``env_params``.

    Notes
    -----
    The observation vector per agent is:

    ====================================  =====================
    Feature                               Size
    ====================================  =====================
    Velocity                              ``dim``
    LiDAR proximity                       ``n_lidar_rays``
    LiDAR radial relative velocity        ``n_lidar_rays``
    LiDAR objective proximity             ``n_lidar_rays``
    Unit direction to top *k* objectives  ``k_objectives * dim``
    Clamped displacement to top *k*       ``k_objectives * dim``
    Occupancy status of top *k*           ``k_objectives``
    ====================================  =====================

    """

    n_lidar_rays: int = jax.tree.static()  # type: ignore[attr-defined]
    """Number of angular bins for the agent-to-agent LiDAR sensor."""
    k_objectives: int = jax.tree.static()  # type: ignore[attr-defined]
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
        shaping_weight: float = 2.0,
        goal_weight: float = 0.001,
        crowding_weight: float = 0.005,
        work_weight: float = 0.001 / 2,
        vacancy_weight: float = 0.005,
        goal_radius_factor: float = 1.0,
        alpha_r_bar: float = 0.07,
        lidar_range: float = 0.4,
        n_lidar_rays: int = 8,
        k_objectives: int = 5,
    ) -> SwarmNavigator:
        r"""Create a swarm navigator environment.

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
        shaping_weight : float
            Multiplier :math:`w_s` on the potential-based shaping signal
            summed over the *k* nearest objectives.
        goal_weight : float
            Bonus :math:`w_g` for uniquely claiming a target.
        crowding_weight : float
            Penalty :math:`w_c` per unit of LiDAR crowding vector norm.
        work_weight : float
            Weight :math:`w_w` of the quadratic action penalty
            :math:`\|a\|^2`.
        vacancy_weight : float
            Reward :math:`w_v` granted when all *k* nearest
            objectives are occupied.
        goal_radius_factor : float
            Multiplicative factor :math:`f` applied to the particle
            radius to define the goal activation threshold
            :math:`d < f \cdot r`.
        alpha_r_bar : float
            EMA smoothing factor :math:`\alpha` for the differential
            reward baseline :math:`\bar{r}`.
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

        env_params = {
            "objective": jnp.zeros_like(state.pos),
            "action": jnp.zeros_like(state.pos),
            "delta": jnp.zeros_like(state.pos),
            "prev_dist_all": jnp.zeros((state.N, state.N), dtype=float),
            "r_bar": jnp.zeros(state.N, dtype=float),
            "current_reward": jnp.zeros(state.N, dtype=float),
            "min_box_size": jnp.asarray(min_box_size, dtype=float),
            "max_box_size": jnp.asarray(max_box_size, dtype=float),
            "box_padding": jnp.asarray(box_padding, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "shaping_weight": jnp.asarray(shaping_weight, dtype=float),
            "goal_weight": jnp.asarray(goal_weight, dtype=float),
            "crowding_weight": jnp.asarray(crowding_weight, dtype=float),
            "work_weight": jnp.asarray(work_weight, dtype=float),
            "vacancy_weight": jnp.asarray(vacancy_weight, dtype=float),
            "goal_radius_factor": jnp.asarray(goal_radius_factor, dtype=float),
            "alpha_r_bar": jnp.asarray(alpha_r_bar, dtype=float),
            "lidar_range": jnp.asarray(lidar_range, dtype=float),
            "lidar": jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
            "lidar_idx": jnp.zeros((state.N, int(n_lidar_rays)), dtype=int),
            "lidar_vr": jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
            "lidar_obj": jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
            "top_k_units": jnp.zeros((state.N, int(k_objectives) * dim), dtype=float),
            "top_k_clipped": jnp.zeros((state.N, int(k_objectives) * dim), dtype=float),
            "top_k_occupied": jnp.zeros((state.N, int(k_objectives)), dtype=float),
        }

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            n_lidar_rays=int(n_lidar_rays),
            k_objectives=int(k_objectives),
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmNavigator.reset")
    def reset(env: "SwarmNavigator", key: ArrayLike) -> Environment:
        """Reset the environment to a random initial configuration.

        Parameters
        ----------
        env : Environment
            The environment instance to reset.
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
        k = env.k_objectives
        n_rays = env.n_lidar_rays
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
            domain_kw={
                "box_size": box + padding,
                "anchor": jnp.zeros_like(box) - padding / 2,
            },
            mat_table=mat_table,
        )

        env.env_params["objective"] = objective
        env.env_params["action"] = jnp.zeros_like(env.state.pos)
        env.env_params["r_bar"] = jnp.zeros(N, dtype=float)
        env.env_params["current_reward"] = jnp.zeros(N, dtype=float)

        # Agent-to-agent LiDAR
        _, _, env.env_params["lidar"], env.env_params["lidar_idx"], _ = lidar_2d(
            env.state,
            env.system,
            env.env_params["lidar_range"],
            n_rays,
            env.max_num_agents,
            sense_edges=True,
        )
        is_agent = env.env_params["lidar_idx"] >= 0
        safe_idx = jnp.where(is_agent, env.env_params["lidar_idx"], 0)
        rel_vel = (env.state.vel[safe_idx] - env.state.vel[:, None, :])[..., :2]
        angles = jnp.linspace(-jnp.pi, jnp.pi, n_rays, endpoint=False) + jnp.pi / n_rays
        ray_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        env.env_params["lidar_vr"] = jnp.where(
            is_agent & (env.env_params["lidar"] > 0),
            dot(rel_vel, ray_dirs),
            0.0,
        )

        # Objective LiDAR
        env.env_params["lidar_obj"], _, _ = cross_lidar_2d(
            env.state.pos,
            objective,
            env.system,
            env.env_params["lidar_range"],
            n_rays,
            N,
        )

        # Distances to all objectives
        deltas = env.system.domain.displacement(
            env.state.pos[:, None, :], objective[None, :, :], env.system
        )
        dist = norm(deltas)

        env.env_params["prev_dist_all"] = dist

        # Top-k observations
        thresh = env.env_params["goal_radius_factor"] * env.state.rad[0]
        at_obj = dist < thresh
        someone_at_obj = jnp.any(at_obj, axis=0)
        closest_agent = jnp.argmin(dist, axis=0)
        occ_by_others = someone_at_obj[None, :] & (
            closest_agent[None, :] != jnp.arange(N)[:, None]
        )

        _, top_k_idx = jax.lax.top_k(-dist, k)
        top_k_deltas = jnp.take_along_axis(deltas, top_k_idx[..., None], axis=1)
        top_k_units, top_k_dist = unit_and_norm(top_k_deltas)
        top_k_dist = top_k_dist[..., 0]
        clip_range = env.env_params["lidar_range"]
        clipped = top_k_units * jnp.minimum(top_k_dist[..., None], clip_range)
        top_k_occ = jnp.take_along_axis(
            occ_by_others.astype(jnp.float32), top_k_idx, axis=1
        )

        env.env_params["top_k_units"] = top_k_units.reshape(N, k * dim)
        env.env_params["top_k_clipped"] = clipped.reshape(N, k * dim)
        env.env_params["top_k_occupied"] = top_k_occ

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmNavigator.step")
    def step(env: "SwarmNavigator", action: jax.Array) -> Environment:
        """Advance the environment by one physics step.

        Applies force actions with viscous drag.  After integration the
        method updates all sensor caches and computes the reward with a
        differential baseline.  The shaping signal is summed over the
        *k* nearest objectives.

        Parameters
        ----------
        env : Environment
            Current environment.
        action : jax.Array
            Force actions for every agent, shape ``(N * dim,)``.

        Returns
        -------
        Environment
            Updated environment after physics integration, sensor
            updates, and reward computation.

        """
        N = env.max_num_agents
        dim = env.state.dim
        k = env.k_objectives
        n_rays = env.n_lidar_rays

        reshaped_action = action.reshape(N, *env.action_space_shape)
        env.env_params["action"] = reshaped_action
        force = reshaped_action - env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)

        prev_dist_all = env.env_params["prev_dist_all"]
        env.state, env.system = env.system.step(env.state, env.system)

        # Agent-to-agent LiDAR
        _, _, env.env_params["lidar"], env.env_params["lidar_idx"], _ = lidar_2d(
            env.state,
            env.system,
            env.env_params["lidar_range"],
            n_rays,
            env.max_num_agents,
            sense_edges=True,
        )
        is_agent = env.env_params["lidar_idx"] >= 0
        safe_idx = jnp.where(is_agent, env.env_params["lidar_idx"], 0)
        rel_vel = (env.state.vel[safe_idx] - env.state.vel[:, None, :])[..., :2]
        angles = jnp.linspace(-jnp.pi, jnp.pi, n_rays, endpoint=False) + jnp.pi / n_rays
        ray_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        env.env_params["lidar_vr"] = jnp.where(
            is_agent & (env.env_params["lidar"] > 0),
            dot(rel_vel, ray_dirs),
            0.0,
        )

        # Objective LiDAR
        objective = env.env_params["objective"]
        env.env_params["lidar_obj"], _, _ = cross_lidar_2d(
            env.state.pos,
            objective,
            env.system,
            env.env_params["lidar_range"],
            n_rays,
            N,
        )

        # Distances to all objectives
        deltas = env.system.domain.displacement(
            env.state.pos[:, None, :], objective[None, :, :], env.system
        )
        dist = norm(deltas)

        env.env_params["prev_dist_all"] = dist

        # Top-k observations and occupancy
        thresh = env.env_params["goal_radius_factor"] * env.state.rad[0]
        at_obj = dist < thresh
        someone_at_obj = jnp.any(at_obj, axis=0)
        closest_agent = jnp.argmin(dist, axis=0)
        occ_by_others = someone_at_obj[None, :] & (
            closest_agent[None, :] != jnp.arange(N)[:, None]
        )

        _, top_k_idx = jax.lax.top_k(-dist, k)
        top_k_deltas = jnp.take_along_axis(deltas, top_k_idx[..., None], axis=1)
        top_k_units, top_k_dist = unit_and_norm(top_k_deltas)
        top_k_dist = top_k_dist[..., 0]
        clip_range = env.env_params["lidar_range"]
        clipped = top_k_units * jnp.minimum(top_k_dist[..., None], clip_range)
        top_k_occ = jnp.take_along_axis(
            occ_by_others.astype(jnp.float32), top_k_idx, axis=1
        )

        env.env_params["top_k_units"] = top_k_units.reshape(N, k * dim)
        env.env_params["top_k_clipped"] = clipped.reshape(N, k * dim)
        env.env_params["top_k_occupied"] = top_k_occ

        # Shaping summed over top-k objectives
        top_k_prev = jnp.take_along_axis(prev_dist_all, top_k_idx, axis=1)
        shaping = jnp.sum(jnp.exp(-2 * top_k_dist) - jnp.exp(-2 * top_k_prev), axis=-1)

        valid_claim = at_obj & (closest_agent[None, :] == jnp.arange(N)[:, None])
        on_target = jnp.any(valid_claim, axis=-1).astype(jnp.float32)

        crowding = norm(
            jnp.sum(
                ray_dirs[None, ...] * env.env_params["lidar"][..., None],
                axis=1,
            )
        )

        # Occupancy reward: granted when all k nearest are occupied
        top_k_anyone = jnp.take_along_axis(
            jnp.broadcast_to(
                someone_at_obj[None, :], (N, someone_at_obj.shape[0])
            ).astype(jnp.float32),
            top_k_idx,
            axis=1,
        )
        all_k_occupied = jnp.all(top_k_anyone > 0, axis=-1)

        work = norm2(reshaped_action)

        R_raw = (
            env.env_params["shaping_weight"] * shaping
            + env.env_params["goal_weight"] * on_target
            - env.env_params["crowding_weight"] * crowding
            - env.env_params["work_weight"] * work
            + env.env_params["vacancy_weight"] * all_k_occupied
        )

        r_bar = env.env_params["r_bar"]
        alpha = env.env_params["alpha_r_bar"]
        env.env_params["r_bar"] = r_bar + alpha * (R_raw - r_bar)
        env.env_params["current_reward"] = R_raw - r_bar

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmNavigator.observation")
    def observation(env: "SwarmNavigator") -> jax.Array:
        """Build the per-agent observation vector from cached sensors.

        All state-dependent components are pre-computed in :meth:`step`
        and :meth:`reset`.  This method only concatenates cached arrays.

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
                env.env_params["lidar_vr"],
                env.env_params["lidar_obj"],
                env.env_params["top_k_units"],
                env.env_params["top_k_clipped"],
                env.env_params["top_k_occupied"],
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmNavigator.reward")
    def reward(env: "SwarmNavigator") -> jax.Array:
        """Return the reward cached by :meth:`step`.

        Returns
        -------
        jax.Array
            Reward vector of shape ``(N,)``.

        """
        return env.env_params["current_reward"]

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="SwarmNavigator.done")
    def done(env: "SwarmNavigator") -> jax.Array:
        """Return ``True`` when the episode has exceeded ``max_steps``."""
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """Number of scalar actions per agent (equal to ``dim``)."""
        return self.state.dim

    @property
    def action_space_shape(self) -> tuple[int]:
        """Shape of a single agent's action (``(dim,)``)."""
        return (self.state.dim,)

    @property
    def observation_space_size(self) -> int:
        """Dimensionality of a single agent's observation vector."""
        # vel(dim)
        # + lidar(n) + lidar_vr(n) + lidar_obj(n)
        # + top_k_units(k*dim) + top_k_clipped(k*dim) + top_k_occupied(k)
        dim = self.state.dim
        return dim + 3 * self.n_lidar_rays + (2 * dim + 1) * self.k_objectives


__all__ = ["SwarmNavigator"]
