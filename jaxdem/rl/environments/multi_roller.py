# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Multi-agent 3-D rolling environment with LiDAR sensing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from functools import partial
from typing import cast

from . import Environment
from ...state import State
from ...system import System
from ...utils import lidar_2d, unit
from ...utils.linalg import cross, dot, norm, norm2
from ...materials import MaterialTable, Material
from ...material_matchmakers import MaterialMatchmaker


@partial(jax.jit, static_argnames=("N",))
@partial(jax.named_call, name="multi_roller._sample_objectives_3d")
def _sample_objectives_3d(
    key: ArrayLike, N: int, box: jax.Array, rad: float
) -> jax.Array:
    r"""Sample *N* positions on a jittered X-Y grid at floor level.

    Parameters
    ----------
    key : ArrayLike
        PRNG key for jitter noise.
    N : int
        Number of positions to sample.
    box : jax.Array
        Domain extents, shape ``(3,)``; only ``box[0]`` (X) and
        ``box[1]`` (Y) are used for the grid.
    rad : float
        Particle radius.  The Z coordinate is fixed at ``rad``.

    Returns
    -------
    jax.Array
        Positions of shape ``(N, 3)`` with ``Z = rad``.

    """
    i = jax.lax.iota(int, N)
    Lx, Ly = box[0], box[1]

    nx = jnp.ceil(jnp.sqrt(N * Lx / Ly)).astype(int)
    ny = jnp.ceil(N / nx).astype(int)

    ix = jnp.mod(i, nx)
    iy = i // nx

    dx = Lx / nx
    dy = Ly / ny

    xs = (ix + 0.5) * dx
    ys = (iy + 0.5) * dy
    zs = jnp.full_like(xs, rad)
    base = jnp.stack([xs, ys, zs], axis=1)

    noise = jax.random.uniform(key, (N, 3), minval=-1.0, maxval=1.0)
    noise_scale = jnp.asarray(
        [
            jnp.maximum(0.0, dx / 2 - rad),
            jnp.maximum(0.0, dy / 2 - rad),
            0.0,
        ]
    )
    return base + noise * noise_scale


@partial(jax.named_call, name="multi_roller.frictional_wall_force")
def frictional_wall_force(
    pos: jax.Array, state: State, system: System
) -> tuple[jax.Array, jax.Array]:
    r"""Normal, frictional, and restitution forces for spheres on a :math:`z = 0` plane.

    Combines a linear spring in the normal direction with Coulomb
    tangential friction and a velocity-proportional dashpot for
    restitution damping.  The dashpot coefficient is clamped to
    ``0.5 m / \Delta t`` for numerical stability with explicit
    integration.

    Parameters
    ----------
    pos : jax.Array
        Particle positions, shape ``(N, 3)``.
    state : State
        Full simulation state (provides ``vel``, ``ang_vel``, ``rad``,
        ``mass``).
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

    dist = pos[..., 2] - state.rad
    penetration = jnp.minimum(0.0, dist)
    force_n = -k * penetration[..., None] * n

    v_n_scalar = dot(state.vel, n)[..., None]
    in_contact = (penetration < 0)[..., None]
    c_n = 2.0 * (1.0 - restitution) * jnp.sqrt(k * state.mass[..., None])
    c_n = jnp.minimum(c_n, 0.5 * state.mass[..., None] / system.dt)
    force_damping = -c_n * v_n_scalar * n * in_contact

    radius_vec = -state.rad[..., None] * n
    v_at_contact = state.vel + cross(state.ang_vel, radius_vec)
    v_n = dot(v_at_contact, n)[..., None] * n
    v_t = v_at_contact - v_n

    f_t_mag = mu * dot(force_n, n)[..., None]
    t_dir = unit(v_t)
    force_t = -f_t_mag * t_dir

    total_force = force_n + force_damping + force_t
    total_torque = cross(radius_vec, force_t)
    return total_force, total_torque


@Environment.register("multiRoller")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MultiRoller(Environment):
    r"""Multi-agent 3-D rolling environment with cooperative rewards.

    Each agent is a sphere resting on a :math:`z = 0` floor under gravity.
    Actions are 3-D torque vectors; translational motion arises from
    frictional contact with the floor (see :func:`frictional_wall_force`).
    Viscous drag ``-friction * vel`` and angular damping
    ``-ang_damping * ang_vel`` are applied every step.  Objectives are
    assigned one-to-one via a random permutation.  Each agent receives a
    random priority scalar at reset for symmetry breaking.

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
    Unit direction to objective (x, y)    ``2``
    Clamped displacement (x, y)           ``2``
    Velocity (x, y)                       ``2``
    Angular velocity                      ``3``
    Own priority                          ``1``
    LiDAR proximity (normalised)          ``n_lidar_rays``
    Radial relative velocity              ``n_lidar_rays``
    LiDAR neighbour priority              ``n_lidar_rays``
    ====================================  =====================

    """

    n_lidar_rays: int = field(metadata={"static": True})
    """Number of angular bins for each LiDAR sensor."""

    @classmethod
    @partial(jax.named_call, name="MultiRoller.Create")
    def Create(
        cls,
        N: int = 64,
        min_box_size: float = 1.0,
        max_box_size: float = 1.0,
        box_padding: float = 5.0,
        max_steps: int = 5760,
        friction: float = 0.2,
        ang_damping: float = 0.07,
        shaping_weight: float = 1.5,
        goal_weight: float = 0.001,
        crowding_weight: float = 0.005,
        work_weight: float = 0.001 / 2,
        goal_radius_factor: float = 1.0,
        alpha_r_bar: float = 0.07,
        lidar_range: float = 0.3,
        n_lidar_rays: int = 8,
    ) -> MultiRoller:
        r"""Create a multi-agent roller environment.

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
        ang_damping : float
            Angular damping coefficient applied as
            ``-ang_damping * ang_vel``.
        shaping_weight : float
            Multiplier :math:`w_s` on the potential-based shaping signal.
        goal_weight : float
            Bonus :math:`w_g` for being on target.
        crowding_weight : float
            Penalty :math:`w_c` per unit of LiDAR crowding vector norm.
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
        MultiRoller
            A freshly constructed environment (call :meth:`reset` before
            use).

        """
        dim = 3
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = {
            "objective": jnp.zeros_like(state.pos),
            "permutation": jnp.arange(N, dtype=int),
            "action": jnp.zeros((state.N, 3), dtype=float),
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
            "ang_damping": jnp.asarray(ang_damping, dtype=float),
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
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="MultiRoller.reset")
    def reset(env: "MultiRoller", key: ArrayLike) -> Environment:
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
        key_box, key_pos, key_objective, key_shuffle, key_vel, key_prio = (
            jax.random.split(key, 6)
        )
        N = env.max_num_agents
        n_rays = env.n_lidar_rays
        rad = 0.05

        box = jax.random.uniform(
            key_box,
            (3,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
            dtype=float,
        )
        padding = env.env_params["box_padding"] * rad

        pos = _sample_objectives_3d(key_pos, int(N), box + padding, rad) - padding / 2
        pos = pos.at[:, 2].set(rad)

        objective = _sample_objectives_3d(key_objective, int(N), box, rad)
        objective = objective.at[:, 2].set(rad)
        perm = jax.random.permutation(key_shuffle, jnp.arange(N, dtype=int))
        env.env_params["objective"] = objective[perm]
        env.env_params["permutation"] = perm

        vel = jax.random.uniform(
            key_vel, (N, 3), minval=-0.05, maxval=0.05, dtype=float
        )
        vel = vel.at[:, 2].set(0.0)
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
            force_manager_kw={
                "gravity": [0.0, 0.0, -10.0],
                "force_functions": (frictional_wall_force,),
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
        env.env_params["action"] = jnp.zeros((N, 3), dtype=float)
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
        rel_vel_2d = rel_vel[..., :2]
        angles = jnp.linspace(-jnp.pi, jnp.pi, n_rays, endpoint=False) + jnp.pi / n_rays
        ray_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        env.env_params["lidar_vr"] = jnp.where(
            is_agent & (env.env_params["lidar"] > 0),
            dot(rel_vel_2d, ray_dirs),
            0.0,
        )
        env.env_params["lidar_priority"] = jnp.where(is_agent, priority[safe_idx], 0.0)

        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="MultiRoller.step")
    def step(env: "MultiRoller", action: jax.Array) -> Environment:
        """Advance the environment by one physics step.

        Applies torque actions with angular damping and viscous drag.
        After integration the method updates LiDAR sensors, displacement
        caches, and computes the reward with a differential baseline.

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
        n_rays = env.n_lidar_rays

        reshaped_action = action.reshape(N, *env.action_space_shape)
        env.env_params["action"] = reshaped_action
        torque = reshaped_action - env.env_params["ang_damping"] * env.state.ang_vel
        force = -env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.system = env.system.force_manager.add_torque(env.state, env.system, torque)

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
        rel_vel_2d = rel_vel[..., :2]
        angles = jnp.linspace(-jnp.pi, jnp.pi, n_rays, endpoint=False) + jnp.pi / n_rays
        ray_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        env.env_params["lidar_vr"] = jnp.where(
            is_agent & (env.env_params["lidar"] > 0),
            dot(rel_vel_2d, ray_dirs),
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
    @partial(jax.named_call, name="MultiRoller.observation")
    def observation(env: "MultiRoller") -> jax.Array:
        """Build the per-agent observation vector from cached sensors.

        All state-dependent components are pre-computed in :meth:`step`
        and :meth:`reset`.  This method only concatenates cached arrays.

        Returns
        -------
        jax.Array
            Observation matrix of shape ``(N, obs_dim)``.  See the class
            docstring for the feature layout.

        """
        delta_2d = env.env_params["delta"][..., :2]
        return jnp.concatenate(
            [
                unit(delta_2d),
                jnp.clip(
                    delta_2d,
                    -env.env_params["lidar_range"],
                    env.env_params["lidar_range"],
                ),
                env.state.vel[..., :2],
                env.state.ang_vel,
                env.env_params["priority"][:, None],
                env.env_params["lidar"],
                env.env_params["lidar_vr"],
                env.env_params["lidar_priority"],
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiRoller.reward")
    def reward(env: "MultiRoller") -> jax.Array:
        """Return the reward cached by :meth:`step`.

        Returns
        -------
        jax.Array
            Reward vector of shape ``(N,)``.

        """
        return env.env_params["current_reward"]

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="MultiRoller.done")
    def done(env: "MultiRoller") -> jax.Array:
        """Return ``True`` when the episode has exceeded ``max_steps``."""
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """Number of scalar actions per agent (3-D torque)."""
        return 3

    @property
    def action_space_shape(self) -> tuple[int]:
        """Shape of a single agent's action (``(3,)``)."""
        return (3,)

    @property
    def observation_space_size(self) -> int:
        """Dimensionality of a single agent's observation vector."""
        return 10 + 3 * self.n_lidar_rays


__all__ = ["MultiRoller"]
