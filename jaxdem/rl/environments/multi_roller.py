# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Multi-agent 3-D rolling environment with LiDAR sensing."""

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
from ...utils.linalg import cross, dot, norm, norm2, unit
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
) -> Tuple[jax.Array, jax.Array]:
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
    p = jnp.array([0.0, 0.0, 0.0])

    # Normal force
    dist = jnp.dot(pos - p, n) - state.rad
    penetration = jnp.minimum(0.0, dist)
    force_n = -k * penetration[..., None] * n

    # Normal velocity damping (restitution)
    v_n_scalar = dot(state.vel, n)[..., None]
    in_contact = (penetration < 0)[..., None]
    c_n = 2.0 * (1.0 - restitution) * jnp.sqrt(k * state.mass[..., None])
    c_n = jnp.minimum(c_n, 0.5 * state.mass[..., None] / system.dt)
    force_damping = -c_n * v_n_scalar * n * in_contact

    # Velocity at contact point
    radius_vec = -state.rad[..., None] * n
    v_at_contact = state.vel + cross(state.ang_vel, radius_vec)
    v_n = dot(v_at_contact, n)[..., None] * n
    v_t = v_at_contact - v_n

    # Coulomb friction
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
    r"""Multi-agent 3-D rolling environment with LiDAR and cooperative rewards.

    Each agent is a sphere resting on a :math:`z = 0` floor under gravity.
    Actions are 3-D torque vectors; translational motion arises from
    frictional contact with the floor (see :func:`frictional_wall_force`).
    Viscous drag ``-friction * vel`` and angular damping ``-0.05 * ang_vel``
    are applied every step.

    The reward uses *Difference Rewards incorporating Potential-Based
    Reward Shaping* (DRiP):

    .. math::

        R_i^{\text{DRiP}} = D_i + \alpha \, F_i

    where :math:`D_i = G - G_{-i}` is the difference reward with
    :math:`G = \sum_j R_j`, and
    :math:`F_i = \gamma\,e^{-2d_i} - e^{-2d_i^{\text{prev}}}` is the
    potential-based shaping signal.  The base reward
    :math:`R_i = \beta\,\mathbf{1}[d_i < f\,r_i] - 0.002\,\|a_i\|^2`
    combines a goal proximity indicator with an action-effort penalty.

    Notes
    -----
    The observation vector per agent is:

    ====================================  =====================
    Feature                               Size
    ====================================  =====================
    Unit direction to objective (x, y)    2
    Clamped displacement (x, y)           2
    Velocity (x, y)                       2
    Angular velocity                      3
    LiDAR proximity (normalised)          ``n_lidar_rays``
    Radial relative velocity              ``n_lidar_rays``
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
        friction: float = 0.08,
        shaping_weight: float = 2.0,
        goal_weight: float = 0.04,
        goal_radius_factor: float = 1.0,
        work_weight: float = 0.002,
        gamma: float = 0.99,
        lidar_range: float = 0.3,
        n_lidar_rays: int = 8,
    ) -> MultiRoller:
        r"""
        Create a multi-agent roller environment.

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
            Weight :math:`\alpha` of the PBRS shaping signal
            :math:`F_i = \gamma\,e^{-2d} - e^{-2d^{\text{prev}}}`.
        goal_weight : float
            Weight :math:`\beta` of the binary goal indicator that
            activates when the agent is within
            :math:`f \cdot r` of its objective.
        goal_radius_factor : float
            Multiplicative factor :math:`f` applied to the particle radius
            to define the goal activation threshold
            :math:`d < f \cdot r`.
        work_weight : float
            Weight of the quadratic action penalty :math:`\|a\|^2`.
        gamma : float
            Discount factor used in the PBRS signal :math:`F_i`.
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

        env_params = dict(
            objective=jnp.zeros_like(state.pos),
            prev_dist=jnp.zeros_like(state.rad),
            action=jnp.zeros_like(state.torque),
            min_box_size=jnp.asarray(min_box_size, dtype=float),
            max_box_size=jnp.asarray(max_box_size, dtype=float),
            box_padding=jnp.asarray(box_padding, dtype=float),
            max_steps=jnp.asarray(max_steps, dtype=int),
            friction=jnp.asarray(friction, dtype=float),
            shaping_weight=jnp.asarray(shaping_weight, dtype=float),
            goal_weight=jnp.asarray(goal_weight, dtype=float),
            goal_radius_factor=jnp.asarray(goal_radius_factor, dtype=float),
            work_weight=jnp.asarray(work_weight, dtype=float),
            gamma=jnp.asarray(gamma, dtype=float),
            lidar_range=jnp.asarray(lidar_range, dtype=float),
            lidar=jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
            lidar_idx=jnp.zeros((state.N, int(n_lidar_rays)), dtype=int),
            permutation=jnp.zeros(state.N, dtype=int),
        )

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            n_lidar_rays=int(n_lidar_rays),
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="MultiRoller.reset")
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
        dim = 3
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
        perm = jax.random.permutation(key_shuffle, jnp.arange(N, dtype=int))
        env.env_params["objective"] = objective[perm]
        env.env_params["permutation"] = perm

        vel = jax.random.uniform(
            key_vel, (N, dim), minval=-0.05, maxval=0.05, dtype=float
        )
        vel = vel.at[:, 2].set(0.0)
        rads = rad_val * jnp.ones(N)
        env.state = State.create(pos=pos, vel=vel, rad=rads)

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
            force_manager_kw=dict(
                gravity=[0.0, 0.0, -10.0],
                force_functions=(frictional_wall_force,),
            ),
            mat_table=mat_table,
        )

        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        env.env_params["prev_dist"] = norm(delta)
        env.env_params["action"] = jnp.zeros_like(env.state.torque)

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
    @partial(jax.named_call, name="MultiRoller.step")
    def step(env: Environment, action: jax.Array) -> Environment:
        """Advance the environment by one physics step.

        Parameters
        ----------
        env : Environment
            Current environment (donated / consumed).
        action : jax.Array
            Torque actions for every agent, shape ``(N * 3,)``.

        Returns
        -------
        Environment
            Updated environment after the physics integration.
        """
        torque = (
            action.reshape(env.max_num_agents, *env.action_space_shape)
            - 0.05 * env.state.ang_vel
        )
        force = -env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.system = env.system.force_manager.add_torque(env.state, env.system, torque)

        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        env.env_params["prev_dist"] = norm(delta)
        env.env_params["action"] = action.reshape(
            env.max_num_agents, *env.action_space_shape
        )

        env.state, env.system = env.system.step(env.state, env.system)

        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        dist = norm(delta)
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
    @partial(jax.named_call, name="MultiRoller.observation")
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
        # Project to floor plane (x, z)
        delta_2d = delta[..., [0, 1]]
        vel_2d = env.state.vel[..., [0, 1]]

        # Radial relative velocity: project onto ray direction (+ outward)
        lidar_idx = env.env_params["lidar_idx"]
        is_agent = lidar_idx >= 0
        safe_idx = jnp.where(is_agent, lidar_idx, 0)
        rel_vel = env.state.vel[safe_idx] - env.state.vel[:, None, :]
        # LiDAR operates in 2-D (x-y of state == x-z of world), take first 2 components
        rel_vel_2d = rel_vel[..., :2]

        n_rays = lidar_idx.shape[-1]
        angles = jnp.linspace(-jnp.pi, jnp.pi, n_rays, endpoint=False) + jnp.pi / n_rays
        ray_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        lidar_vr = dot(rel_vel_2d, ray_dirs)
        lidar_vr = jnp.where(is_agent & (env.env_params["lidar"] > 0), lidar_vr, 0.0)

        return jnp.concatenate(
            [
                unit(delta_2d),
                jnp.clip(
                    delta_2d,
                    -env.env_params["lidar_range"],
                    env.env_params["lidar_range"],
                ),
                vel_2d,
                env.state.ang_vel,
                env.env_params["lidar"],
                lidar_vr,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiRoller.reward")
    def reward(env: Environment) -> jax.Array:
        r"""Compute the per-agent DRiP reward.

        Combines *Difference Rewards* with *Potential-Based Reward
        Shaping* (DRiP).

        1. Base reward per agent:

           .. math::

               R_i = \beta\,\mathbf{1}[d_i < f\,r_i] - \text{work\_weight}\,\|a_i\|^2

        2. Potential-based shaping (PBRS):

           .. math::

               F_i = \gamma\,e^{-2d_i} - e^{-2d_i^{\text{prev}}}

        3. Difference reward:

           .. math::

               D_i = G - G_{-i}, \quad G = \textstyle\sum_j R_j

        4. Final reward:

           .. math::

               R_i^{\text{DRiP}} = D_i + \alpha\,F_i

        Returns
        -------
        jax.Array
            Reward vector of shape ``(N,)``.
        """
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        dist = norm(delta)

        # --- Base Action / Goal Reward ---
        work = norm2(env.env_params["action"])
        at_goal = dist < env.env_params["goal_radius_factor"] * env.state.rad

        # 1. Base Environmental Reward (R_i) now includes the SFM penalty
        R_i = (env.env_params["goal_weight"] * at_goal) - env.env_params[
            "work_weight"
        ] * work

        # 2. Absolute Potential Function (\Phi) (Attractive Component)
        current_potential = jnp.exp(-2 * dist)
        prev_potential = jnp.exp(-2 * env.env_params["prev_dist"])

        # 3. Standard PBRS Signal (F_i)
        gamma = env.env_params["gamma"]
        F_i = (gamma * current_potential) - prev_potential

        # ==========================================
        # Standard Difference Reward
        # D_i = G(s) - G(s_{-i})
        # ==========================================
        G_current = jnp.sum(R_i)
        G_without_i = G_current - R_i
        R_Difference = G_current - G_without_i

        # ==========================================
        # DRiP (Difference Rewards incorporating PBRS)
        # R_DRiP = D_i + \gamma \Phi_i(s') - \Phi_i(s)
        # ==========================================
        R_DRiP = R_Difference + env.env_params["shaping_weight"] * F_i
        return R_DRiP

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="MultiRoller.done")
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
        # unit_dir(2) + clamp_disp(2) + vel(2) + ang_vel(3) + lidar(n) + lidar_vr(n)
        return 9 + 2 * self.n_lidar_rays


__all__ = ["MultiRoller"]
