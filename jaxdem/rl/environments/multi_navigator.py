# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Multi-agent 2-D navigation with collision avoidance and cooperative rewards."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from functools import partial

from . import Environment
from ...state import State
from ...system import System
from ...utils import lidar_2d, unit
from ...utils.linalg import dot, norm, norm2
from ...materials import MaterialTable, Material
from ...material_matchmakers import MaterialMatchmaker


@partial(jax.jit, static_argnames=("N",))
@partial(jax.named_call, name="multi_navigator._sample_objectives")
def _sample_objectives(key: ArrayLike, N: int, box: jax.Array, rad: float) -> jax.Array:
    r"""Sample *N* positions on a jittered 2-D grid."""
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
    r"""Multi-agent 2-D navigation with cooperative rewards.

    Each agent controls a force vector applied directly to a sphere inside a
    reflective box.  Viscous drag ``-friction * vel`` is added every step.
    Objectives are assigned one-to-one via a random permutation.  Each
    agent receives a random priority scalar at reset for symmetry breaking.

    **Reward**

    .. math::

        R_i = w_s\,(e^{-2d_i} - e^{-2d_i^{\mathrm{prev}}})
              + w_g\,\mathbf{1}[d_i < f \cdot r_i]
              - w_c\,\left\|\sum_j l_j\,\hat{r}_j\right\|
              - w_w\,\|a_i\|^2
              - \bar{r}_i

    where :math:`l_j` and :math:`\hat{r}_j` are the LiDAR readings and
    ray directions respectively, and :math:`\bar{r}_i` is an EMA
    baseline updated with factor :math:`\alpha`.  All weights
    (:math:`w_s, w_g, w_c, w_w, \alpha, f`) are constructor
    parameters stored in ``env_params``.

    Notes
    -----
    The observation vector per agent is:

    ====================================  =====================
    Feature                               Size
    ====================================  =====================
    Unit direction to objective           ``dim``
    Clamped displacement                  ``dim``
    Velocity                              ``dim``
    Own priority                          ``1``
    LiDAR proximity (normalised)          ``n_lidar_rays``
    Radial relative velocity              ``n_lidar_rays``
    LiDAR neighbour priority              ``n_lidar_rays``
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
        friction: float = 0.2,
        shaping_weight: float = 1.5,
        goal_weight: float = 0.001,
        crowding_weight: float = 0.005,
        work_weight: float = 0.001 / 2,
        goal_radius_factor: float = 1.0,
        alpha_r_bar: float = 0.07,
        lidar_range: float = 0.3,
        n_lidar_rays: int = 8,
    ) -> MultiNavigator:
        r"""Create a multi-agent navigator environment.

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
            Multiplier :math:`w_s` on the potential-based shaping signal.
        goal_weight : float
            Bonus :math:`w_g` for being on target.
        crowding_weight : float
            Penalty :math:`w_c` per unit of LiDAR proximity sum.
        work_weight : float
            Weight :math:`w_w` of the quadratic action penalty
            :math:`\|a\|^2`.
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

        Returns
        -------
        MultiNavigator
            A freshly constructed environment (call :meth:`reset` before
            use).

        """
        dim = 2
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = {
            "objective": jnp.zeros_like(state.pos),
            "permutation": jnp.arange(N, dtype=int),
            "action": jnp.zeros_like(state.pos),
            "delta": jnp.zeros_like(state.pos),
            "prev_dist": jnp.zeros_like(state.rad),
            "curr_dist": jnp.zeros_like(state.rad),
            "priority": jnp.zeros(state.N, dtype=float),
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
            "goal_radius_factor": jnp.asarray(goal_radius_factor, dtype=float),
            "alpha_r_bar": jnp.asarray(alpha_r_bar, dtype=float),
            "lidar_range": jnp.asarray(lidar_range, dtype=float),
            "lidar": jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
            "lidar_idx": jnp.zeros((state.N, int(n_lidar_rays)), dtype=int),
            "lidar_vr": jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
            "lidar_priority": jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
        }

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            n_lidar_rays=int(n_lidar_rays),
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.reset")
    def reset(env: "MultiNavigator", key: ArrayLike) -> Environment:
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
        key_box, key_pos, key_objective, key_shuffle, key_vel, key_prio = (
            jax.random.split(key, 6)
        )
        N = env.max_num_agents
        dim = env.state.dim
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
        perm = jax.random.permutation(key_shuffle, jnp.arange(N, dtype=int))
        env.env_params["objective"] = objective[perm]
        env.env_params["permutation"] = perm

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

        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        dist = norm(delta)
        env.env_params["delta"] = delta
        env.env_params["prev_dist"] = dist
        env.env_params["curr_dist"] = dist
        env.env_params["action"] = jnp.zeros_like(env.state.pos)
        env.env_params["r_bar"] = jnp.zeros(N, dtype=float)
        env.env_params["current_reward"] = jnp.zeros(N, dtype=float)

        priority = jax.random.uniform(key_prio, (N,), dtype=float)
        env.env_params["priority"] = priority

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
        rel_vel = env.state.vel[safe_idx] - env.state.vel[:, None, :]
        angles = jnp.linspace(-jnp.pi, jnp.pi, n_rays, endpoint=False) + jnp.pi / n_rays
        ray_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        env.env_params["lidar_vr"] = jnp.where(
            is_agent & (env.env_params["lidar"] > 0),
            dot(rel_vel, ray_dirs),
            0.0,
        )
        env.env_params["lidar_priority"] = jnp.where(is_agent, priority[safe_idx], 0.0)

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.step")
    def step(env: "MultiNavigator", action: jax.Array) -> Environment:
        """Advance the environment by one physics step.

        Applies force actions with viscous drag.  After integration the
        method updates LiDAR sensors, displacement caches, and computes
        the reward with a differential baseline.

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
        n_rays = env.n_lidar_rays

        reshaped_action = action.reshape(N, *env.action_space_shape)
        env.env_params["action"] = reshaped_action
        force = reshaped_action - env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)

        env.env_params["prev_dist"] = env.env_params["curr_dist"]
        env.state, env.system = env.system.step(env.state, env.system)

        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        curr_dist = norm(delta)
        env.env_params["delta"] = delta
        env.env_params["curr_dist"] = curr_dist

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
        rel_vel = env.state.vel[safe_idx] - env.state.vel[:, None, :]
        angles = jnp.linspace(-jnp.pi, jnp.pi, n_rays, endpoint=False) + jnp.pi / n_rays
        ray_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        env.env_params["lidar_vr"] = jnp.where(
            is_agent & (env.env_params["lidar"] > 0),
            dot(rel_vel, ray_dirs),
            0.0,
        )
        env.env_params["lidar_priority"] = jnp.where(
            is_agent, env.env_params["priority"][safe_idx], 0.0
        )

        shaping = jnp.exp(-2 * curr_dist) - jnp.exp(-2 * env.env_params["prev_dist"])
        thresh = env.env_params["goal_radius_factor"] * env.state.rad
        on_target = curr_dist < thresh
        crowding = norm(
            jnp.sum(ray_dirs[None, ...] * env.env_params["lidar"][..., None], axis=1)
        )
        work = norm2(reshaped_action)

        R_raw = (
            env.env_params["shaping_weight"] * shaping
            + env.env_params["goal_weight"] * on_target
            - env.env_params["crowding_weight"] * crowding
            - env.env_params["work_weight"] * work
        )

        r_bar = env.env_params["r_bar"]
        alpha = env.env_params["alpha_r_bar"]
        env.env_params["r_bar"] = r_bar + alpha * (R_raw - r_bar)
        env.env_params["current_reward"] = R_raw - r_bar

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.observation")
    def observation(env: "MultiNavigator") -> jax.Array:
        """Build the per-agent observation vector from cached sensors.

        All state-dependent components are pre-computed in :meth:`step`
        and :meth:`reset`.  This method only concatenates cached arrays.

        Returns
        -------
        jax.Array
            Observation matrix of shape ``(N, obs_dim)``.  See the class
            docstring for the feature layout.

        """
        delta = env.env_params["delta"]
        return jnp.concatenate(
            [
                unit(delta),
                jnp.clip(
                    delta,
                    -env.env_params["lidar_range"],
                    env.env_params["lidar_range"],
                ),
                env.state.vel,
                env.env_params["priority"][:, None],
                env.env_params["lidar"],
                env.env_params["lidar_vr"],
                env.env_params["lidar_priority"],
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.reward")
    def reward(env: "MultiNavigator") -> jax.Array:
        """Return the reward cached by :meth:`step`.

        Returns
        -------
        jax.Array
            Reward vector of shape ``(N,)``.

        """
        return env.env_params["current_reward"]

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="MultiNavigator.done")
    def done(env: "MultiNavigator") -> jax.Array:
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
        return 3 * self.state.dim + 1 + 3 * self.n_lidar_rays


__all__ = ["MultiNavigator"]
