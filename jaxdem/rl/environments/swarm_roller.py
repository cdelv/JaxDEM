# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Multi-agent 3-D swarm rolling environment with cooperative difference rewards."""

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
from ...utils import lidar_2d, cross_lidar_2d
from ...materials import MaterialTable, Material
from ...material_matchmakers import MaterialMatchmaker
from .multi_roller import frictional_wall_force, _sample_objectives_3d
from ...utils.linalg import dot, norm, norm2


@Environment.register("swarmRoller")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SwarmRoller(Environment):
    r"""Multi-agent 3-D rolling environment with cooperative difference rewards.

    Each agent is a sphere resting on a :math:`z = 0` floor under gravity.
    Actions are 3-D torque vectors; translational motion arises from
    frictional contact with the floor (see
    :func:`~.multi_roller.frictional_wall_force`).  Viscous drag
    ``-friction * vel`` and angular damping ``-0.05 * ang_vel`` are
    applied every step.

    Objectives are shared among all agents; each agent dynamically tracks
    its *k* nearest objectives.  Occupancy is determined via strict
    symmetry breaking: only the closest agent to each objective within the
    activation threshold may claim it.

    **Reward**

    .. math::

        R_i = w_g\,c_i + w_e\,c_i\,\mathbb{1}[\text{all others occ.}]
              + D_i - w_w\,\|a_i\|^2 - \bar{r}_i

    where :math:`c_i=1` when agent *i* uniquely claims a target,
    :math:`D_i = G - G_{-i}` with :math:`G = w_G(\sum_j c_j)^2`,
    and :math:`\bar{r}_i` is an EMA baseline updated with smoothing
    factor :math:`\alpha`.

    Notes
    -----
    The observation vector per agent is:

    ====================================  =====================
    Feature                               Size
    ====================================  =====================
    Velocity (x, y)                       2
    Angular velocity                      3
    LiDAR proximity                       ``n_lidar_rays``
    LiDAR radial relative velocity        ``n_lidar_rays``
    LiDAR objective proximity             ``n_lidar_rays``
    Unit direction to top *k* objectives  ``k_objectives * 2``
    Clamped displacement to top *k*       ``k_objectives * 2``
    Occupancy status of top *k*           ``k_objectives``
    ====================================  =====================
    """

    n_lidar_rays: int = field(metadata={"static": True})
    """Number of angular bins for the agent-to-agent LiDAR sensor."""
    k_objectives: int = field(metadata={"static": True})
    """Number of closest objectives tracked per agent."""

    @classmethod
    @partial(jax.named_call, name="SwarmRoller.Create")
    def Create(
        cls,
        N: int = 64,
        min_box_size: float = 1.0,
        max_box_size: float = 1.0,
        box_padding: float = 20.0,
        max_steps: int = 5760,
        friction: float = 0.08,
        goal_weight: float = 0.01,
        extra_weight: float = 0.01,
        global_weight: float = 0.001,
        work_weight: float = 0.005,
        goal_radius_factor: float = 1.0,
        alpha_r_bar: float = 0.02,
        lidar_range: float = 0.4,
        n_lidar_rays: int = 8,
        k_objectives: int = 4,
    ) -> SwarmRoller:
        r"""Create a swarm roller environment.

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
        extra_weight : float
            Weight of the extra bonus when on target and all other
            visible targets are also occupied.
        global_weight : float
            Weight :math:`w_G` of the quadratic team synergy reward
            :math:`G = w_G (\sum_j c_j)^2`.
        work_weight : float
            Weight of the negative work penalty computed from the
            magnitude square of the action.
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
        SwarmRoller
            A freshly constructed environment (call :meth:`reset` before
            use).
        """
        dim = 3
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = dict(
            objective=jnp.zeros_like(state.pos),
            action=jnp.zeros_like(state.torque),
            min_box_size=jnp.asarray(min_box_size, dtype=float),
            max_box_size=jnp.asarray(max_box_size, dtype=float),
            box_padding=jnp.asarray(box_padding, dtype=float),
            max_steps=jnp.asarray(max_steps, dtype=int),
            friction=jnp.asarray(friction, dtype=float),
            goal_weight=jnp.asarray(goal_weight, dtype=float),
            extra_weight=jnp.asarray(extra_weight, dtype=float),
            global_weight=jnp.asarray(global_weight, dtype=float),
            work_weight=jnp.asarray(work_weight, dtype=float),
            goal_radius_factor=jnp.asarray(goal_radius_factor, dtype=float),
            alpha_r_bar=jnp.asarray(alpha_r_bar, dtype=float),
            r_bar=jnp.zeros(N, dtype=float),
            current_reward=jnp.zeros(N, dtype=float),
            top_k_units=jnp.zeros((N, int(k_objectives) * 2), dtype=float),
            top_k_clipped=jnp.zeros((N, int(k_objectives) * 2), dtype=float),
            top_k_occupied=jnp.zeros((N, int(k_objectives)), dtype=float),
            lidar_range=jnp.asarray(lidar_range, dtype=float),
            lidar=jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
            lidar_idx=jnp.zeros((state.N, int(n_lidar_rays)), dtype=int),
            lidar_vr=jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
            lidar_obj=jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
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
    @partial(jax.named_call, name="SwarmRoller.reset")
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
        dim = 3
        k = cast(int, getattr(env, "k_objectives"))
        n_rays = cast(int, getattr(env, "n_lidar_rays"))
        rad_val = 0.05

        box = jax.random.uniform(
            key_box,
            (dim,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
            dtype=float,
        )
        padding = env.env_params["box_padding"] * rad_val

        pos = (
            _sample_objectives_3d(key_pos, int(N), box + padding, rad_val) - padding / 2
        )
        pos = pos.at[:, 2].set(rad_val)

        objective = _sample_objectives_3d(key_objective, int(N), box, rad_val)
        objective = objective.at[:, 2].set(rad_val)

        vel = jax.random.uniform(
            key_vel, (N, dim), minval=-0.05, maxval=0.05, dtype=float
        )
        vel = vel.at[:, 2].set(0.0)
        env.state = State.create(pos=pos, vel=vel, rad=rad_val * jnp.ones(N))

        matcher = MaterialMatchmaker.create("linear")
        mat_table = MaterialTable.from_materials(
            [
                Material.create(
                    "elasticfrict", density=0.27, young=6e4, poisson=0.3, mu=0.5, e=0.98
                )
            ],
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
            force_model_type="cundallstrack",
            force_manager_kw=dict(
                gravity=[0.0, 0.0, -10.0],
                force_functions=(frictional_wall_force,),
            ),
            mat_table=mat_table,
        )

        env.env_params["objective"] = objective
        env.env_params["action"] = jnp.zeros_like(env.state.torque)

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

        delta = env.system.domain.displacement(
            env.state.pos[:, None, :], objective[None, :, :], env.system
        )
        delta_2d = delta[..., :2]
        dist = norm(delta_2d)

        thresh = env.env_params["goal_radius_factor"] * env.state.rad[0]
        at_obj = dist < thresh

        someone_at_obj = jnp.any(at_obj, axis=0)
        closest_agent_to_obj = jnp.argmin(dist, axis=0)
        occ_by_others_global = someone_at_obj[None, :] & (
            closest_agent_to_obj[None, :] != jnp.arange(N)[:, None]
        )

        _, top_k_idx = jax.lax.top_k(-dist, k)
        top_k_deltas = jnp.take_along_axis(delta_2d, top_k_idx[..., None], axis=1)
        top_k_dist = jnp.take_along_axis(dist, top_k_idx, axis=1)
        top_k_units = top_k_deltas / jnp.maximum(top_k_dist[..., None], 1e-6)

        clip_range = env.env_params["lidar_range"]
        clipped = top_k_units * jnp.minimum(top_k_dist[..., None], clip_range)
        top_k_occ = jnp.take_along_axis(
            occ_by_others_global.astype(jnp.float32), top_k_idx, axis=1
        )

        env.env_params["top_k_units"] = top_k_units.reshape(N, k * 2)
        env.env_params["top_k_clipped"] = clipped.reshape(N, k * 2)
        env.env_params["top_k_occupied"] = top_k_occ

        # Objective LiDAR
        env.env_params["lidar_obj"], _, _ = cross_lidar_2d(
            env.state.pos,
            objective,
            env.system,
            env.env_params["lidar_range"],
            n_rays,
            N,
        )

        env.env_params["r_bar"] = jnp.zeros(N, dtype=float)
        env.env_params["current_reward"] = jnp.zeros(N, dtype=float)

        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="SwarmRoller.step")
    def step(env: Environment, action: jax.Array) -> Environment:
        """Advance the environment by one physics step.

        Applies torque actions with angular damping, viscous drag, and
        the Cundall–Strack contact model.  After integration the method
        updates all state-dependent sensor caches (LiDAR, top-*k*
        objectives, occupancy) and computes the cooperative reward with
        a differential baseline.

        Parameters
        ----------
        env : Environment
            Current environment (donated / consumed).
        action : jax.Array
            Torque actions for every agent, shape ``(N * 3,)``.

        Returns
        -------
        Environment
            Updated environment after physics integration, sensor
            updates, and reward computation.
        """
        N = env.max_num_agents
        k = cast(int, getattr(env, "k_objectives"))
        n_rays = cast(int, getattr(env, "n_lidar_rays"))

        # 1. Physics step
        torque = action.reshape(N, *env.action_space_shape) - 0.05 * env.state.ang_vel
        force = -env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.system = env.system.force_manager.add_torque(env.state, env.system, torque)
        env.env_params["action"] = action.reshape(N, *env.action_space_shape)

        env.state, env.system = env.system.step(env.state, env.system)

        # 2. Sensor update: LiDAR proximity + radial relative velocity
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

        # 3. Displacement to objectives (2-D projection)
        objective = env.env_params["objective"]
        delta = env.system.domain.displacement(
            env.state.pos[:, None, :], objective[None, :, :], env.system
        )
        delta_2d = delta[..., :2]
        dist = norm(delta_2d)

        thresh = env.env_params["goal_radius_factor"] * env.state.rad[0]
        at_obj = dist < thresh

        # 4. Strict symmetry breaking: only the closest agent claims each goal
        someone_at_obj = jnp.any(at_obj, axis=0)
        closest_agent_to_obj = jnp.argmin(dist, axis=0)
        occ_by_others_global = someone_at_obj[None, :] & (
            closest_agent_to_obj[None, :] != jnp.arange(N)[:, None]
        )

        # 5. Top-k objective observations
        _, top_k_idx = jax.lax.top_k(-dist, k)
        top_k_deltas = jnp.take_along_axis(delta_2d, top_k_idx[..., None], axis=1)
        top_k_dist = jnp.take_along_axis(dist, top_k_idx, axis=1)
        top_k_units = top_k_deltas / jnp.maximum(top_k_dist[..., None], 1e-6)

        clip_range = env.env_params["lidar_range"]
        clipped = top_k_units * jnp.minimum(top_k_dist[..., None], clip_range)
        top_k_occ = jnp.take_along_axis(
            occ_by_others_global.astype(jnp.float32), top_k_idx, axis=1
        )

        env.env_params["top_k_units"] = top_k_units.reshape(N, k * 2)
        env.env_params["top_k_clipped"] = clipped.reshape(N, k * 2)
        env.env_params["top_k_occupied"] = top_k_occ

        # 6. Objective LiDAR
        env.env_params["lidar_obj"], _, _ = cross_lidar_2d(
            env.state.pos,
            objective,
            env.system,
            env.env_params["lidar_range"],
            n_rays,
            N,
        )

        # 7. Cooperative reward with differential baseline
        valid_claim = at_obj & (closest_agent_to_obj[None, :] == jnp.arange(N)[:, None])
        on_target = jnp.any(valid_claim, axis=-1).astype(jnp.float32)

        num_occ_by_others = jnp.sum(top_k_occ, axis=-1)
        all_others_occupied = (num_occ_by_others >= (k - 1)).astype(jnp.float32)

        base_reward = env.env_params["goal_weight"] * on_target
        extra_reward = env.env_params["extra_weight"] * (
            on_target * all_others_occupied
        )

        total_occ = jnp.sum(on_target)
        G_team = env.env_params["global_weight"] * (total_occ**2)
        G_without_i = env.env_params["global_weight"] * ((total_occ - on_target) ** 2)
        D_team = G_team - G_without_i

        work_penalty = env.env_params["work_weight"] * norm2(env.env_params["action"])

        R_raw = base_reward + extra_reward + D_team - work_penalty

        r_bar = env.env_params["r_bar"]
        alpha = env.env_params["alpha_r_bar"]
        env.env_params["r_bar"] = r_bar + alpha * (R_raw - r_bar)
        env.env_params["current_reward"] = R_raw - r_bar

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmRoller.observation")
    def observation(env: Environment) -> jax.Array:
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
                env.state.vel[..., :2],
                env.state.ang_vel,
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
    @partial(jax.named_call, name="SwarmRoller.reward")
    def reward(env: Environment) -> jax.Array:
        """Return the differential reward cached by :meth:`step`.

        Returns
        -------
        jax.Array
            Reward vector of shape ``(N,)``.
        """
        return env.env_params["current_reward"]

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="SwarmRoller.done")
    def done(env: Environment) -> jax.Array:
        """Return ``True`` when the episode has exceeded ``max_steps``."""
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """Number of scalar actions per agent (3-D torque)."""
        return 3

    @property
    def action_space_shape(self) -> Tuple[int]:
        """Shape of a single agent's action (``(3,)``)."""
        return (3,)

    @property
    def observation_space_size(self) -> int:
        """Dimensionality of a single agent's observation vector."""
        # vel_2d(2) + ang_vel(3) + lidar(n) + lidar_vr(n) + lidar_obj(n) + units(k*2) + clipped(k*2) + occ(k)
        return 5 + 3 * self.n_lidar_rays + 5 * self.k_objectives


__all__ = ["SwarmRoller"]
