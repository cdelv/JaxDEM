# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

"""Multi-agent 3-D swarm rolling environment with magnetic interaction and pyramid objectives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from functools import partial
from typing import cast, Any

from . import Environment
from ...state import State
from ...system import System
from ...utils import lidar_3d, cross_lidar_3d
from ...materials import MaterialTable, Material
from ...material_matchmakers import MaterialMatchmaker
from .multi_roller import frictional_wall_force, _sample_objectives_3d
from ...utils.linalg import dot, norm, norm2, unit_and_norm


@Environment.register("swarmRoller3D")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SwarmRoller3D(Environment):
    r"""Multi-agent 3-D rolling environment with magnetic interaction and pyramid objectives.
    Extends the swarm roller with two additions:

    1. Each agent has an extra binary **magnet** action.  When two
       nearby agents both activate their magnets the mutual attraction
       is twice as strong:

       .. math::

        \mathbf{F}_{ij}^{\text{mag}} = -w_{\text{mag}} \, (m_i + m_j) \,
               \max\!\bigl(0,\; 1 - d/r_{\text{mag}}\bigr) \,
               \hat{n}_{ij}

       where :math:`m_i \in \{0, 1\}` is the magnet flag for agent *i*,
       :math:`d = \|r_{ij}\|`, and :math:`r_{\text{mag}}` is
       ``magnet_range``.

    2. **Pyramid objectives.** Objectives are arranged in a
       pyramid: base layer on the floor and elevated apex targets.
       Agents must stack on top of one another to reach elevated
       targets.  Occupancy uses full 3-D distance to prevent false
       apex claims.

    **Reward**

    .. math::

        R_i = w_s\,\sum_{j \in \text{top-}k}
                  (e^{-2d_{ij}} - e^{-2d_{ij}^{\mathrm{prev}}})
              + w_{th}\,\frac{1}{N}\sum_{m=1}^{N} z_m
              + w_g\,\mathbf{1}[\text{on target}]
              - w_w\,\|a_i\|^2
              - w_{\mathrm{vel}}\,\|v_i\|^2
              - \bar{r}_i

    where :math:`\bar{r}_i` is an EMA baseline updated with factor
    :math:`\alpha`, :math:`w_{th}` scales the reward for the average
    team height, :math:`w_g` is the bonus for being on a target,
    and :math:`w_{\mathrm{vel}}` penalises high agent velocity.
    All weights are constructor parameters stored in ``env_params``.

    Notes
    -----
    The observation vector per agent is:

    ====================================  ====================================
    Feature                               Size
    ====================================  ====================================
    Velocity (x, y, z)                    ``3``
    Angular velocity                      ``3``
    Magnet flag                           ``1``
    LiDAR proximity (normalised)          ``n_lidar_rays * n_lidar_elevation``
    Radial relative velocity              ``n_lidar_rays * n_lidar_elevation``
    Objective LiDAR proximity             ``n_lidar_rays * n_lidar_elevation``
    Unit direction to top *k* objectives  ``k_objectives * 3``
    Clamped displacement to top *k*       ``k_objectives * 3``
    Occupancy status of top *k*           ``k_objectives``
    ====================================  ====================================

    """

    n_lidar_rays: int = jax.tree.static()  # type: ignore[attr-defined]
    """Number of azimuthal bins for the 3-D LiDAR sensor."""
    n_lidar_elevation: int = jax.tree.static()  # type: ignore[attr-defined]
    """Number of elevation bins for the 3-D LiDAR sensor."""
    k_objectives: int = jax.tree.static()  # type: ignore[attr-defined]
    """Number of closest objectives tracked per agent."""
    n_objectives: int = jax.tree.static()  # type: ignore[attr-defined]
    """Number of shared objectives."""

    @classmethod
    @partial(jax.named_call, name="SwarmRoller3D.Create")
    def Create(
        cls,
        N: int = 5,
        n_objectives: int = 5,
        min_box_size: float = 1.0,
        max_box_size: float = 1.0,
        box_padding: float = 0.0,
        max_steps: int = 5760,
        friction: float = 0.2,
        ang_damping: float = 0.07,
        shaping_weight: float = 2.0,
        team_height_weight: float = 1.0,
        goal_weight: float = 0.0,
        work_weight: float = 0.00,
        velocity_weight: float = 0.018,
        goal_radius_factor: float = 1.0,
        alpha_r_bar: float = 0.07,
        lidar_range: float = 0.4,
        n_lidar_rays: int = 6,
        n_lidar_elevation: int = 6,
        k_objectives: int = 4,
        magnet_strength: float = 4e1,
        magnet_range: float = 0.12,
    ) -> SwarmRoller3D:
        r"""Create a swarm roller 3-D environment.

        Parameters
        ----------
        N : int
            Number of agents.
        n_objectives : int
            Number of shared objectives.
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
        ang_damping : float
            Angular damping coefficient applied as ``-ang_damping * ang_vel``.
        shaping_weight : float
            Multiplier :math:`w_s` on the potential-based shaping signal
            summed over the *k* nearest objectives.
        team_height_weight : float
            Weight :math:`w_{th}` scaling the average z-height of the swarm
            as a global reward.
        goal_weight : float
            Bonus :math:`w_g` for being positioned on a target.
        work_weight : float
            Weight :math:`w_w` of the quadratic action penalty
            :math:`\|a\|^2`.
        velocity_weight : float
            Penalty :math:`w_{\mathrm{vel}}` on the squared velocity
            magnitude :math:`\|v_i\|^2`.
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
            Number of azimuthal LiDAR bins spanning
            :math:`[-\pi, \pi)`.
        n_lidar_elevation : int
            Number of elevation LiDAR bins spanning :math:`[-\pi/2, \pi/2]`.
        k_objectives : int
            Number of closest objectives tracked per agent.
        magnet_strength : float
            Magnitude of the magnetic attraction force.
        magnet_range : float
            Maximum range for magnetic interaction (beyond this
            the force is zero).

        Returns
        -------
        SwarmRoller3D
            A freshly constructed environment (call :meth:`reset` before use).

        """
        dim = 3
        n_obj = int(n_objectives)
        k = int(k_objectives)
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)
        rad_val = 0.05
        pyr_rel = _pyramid_layout(n_obj, rad_val)
        pyr_half = jnp.max(jnp.abs(pyr_rel[:, :2])) + rad_val
        n_az = int(n_lidar_rays)
        n_el = int(n_lidar_elevation)
        n_lidar = n_az * n_el
        ray_dirs = _build_ray_dirs(n_az, n_el)
        env_params = {
            "objective": jnp.zeros((n_obj, dim)),
            "pyr_rel": pyr_rel,
            "pyr_half": pyr_half,
            "action": jnp.zeros((N, 4)),
            "magnet": jnp.zeros(N),
            "delta": jnp.zeros((N, dim), dtype=float),
            "prev_dist_all": jnp.zeros((N, n_obj), dtype=float),
            "r_bar": jnp.zeros(N, dtype=float),
            "current_reward": jnp.zeros(N, dtype=float),
            "min_box_size": jnp.asarray(min_box_size, dtype=float),
            "max_box_size": jnp.asarray(max_box_size, dtype=float),
            "box_padding": jnp.asarray(box_padding, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "ang_damping": jnp.asarray(ang_damping, dtype=float),
            "shaping_weight": jnp.asarray(shaping_weight, dtype=float),
            "team_height_weight": jnp.asarray(team_height_weight, dtype=float),
            "goal_weight": jnp.asarray(goal_weight, dtype=float),
            "work_weight": jnp.asarray(work_weight, dtype=float),
            "velocity_weight": jnp.asarray(velocity_weight, dtype=float),
            "goal_radius_factor": jnp.asarray(goal_radius_factor, dtype=float),
            "alpha_r_bar": jnp.asarray(alpha_r_bar, dtype=float),
            "magnet_strength": jnp.asarray(magnet_strength, dtype=float),
            "magnet_range": jnp.asarray(magnet_range, dtype=float),
            "lidar_range": jnp.asarray(lidar_range, dtype=float),
            "ray_dirs": ray_dirs,
            "lidar": jnp.zeros((N, n_lidar), dtype=float),
            "lidar_idx": jnp.zeros((N, n_lidar), dtype=int),
            "lidar_vr": jnp.zeros((N, n_lidar), dtype=float),
            "lidar_obj": jnp.zeros((N, n_lidar), dtype=float),
            "top_k_units": jnp.zeros((N, k * 3), dtype=float),
            "top_k_clipped": jnp.zeros((N, k * 3), dtype=float),
            "top_k_occupied": jnp.zeros((N, k), dtype=float),
        }

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            n_lidar_rays=n_az,
            n_lidar_elevation=n_el,
            k_objectives=k,
            n_objectives=n_obj,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmRoller3D.reset")
    def reset(env: "SwarmRoller3D", key: ArrayLike) -> Environment:
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
        key_box, key_pos, key_vel, key_pyr = jax.random.split(key, 4)
        N = env.max_num_agents
        k = env.k_objectives
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
        pyr_rel = env.env_params["pyr_rel"]
        pyr_half = env.env_params["pyr_half"]
        pyr_center = jax.random.uniform(
            key_pyr, (2,), minval=pyr_half, maxval=box[:2] - pyr_half
        )
        objective = pyr_rel.at[:, 0].add(pyr_center[0]).at[:, 1].add(pyr_center[1])
        vel = jax.random.uniform(
            key_vel, (N, dim), minval=-0.05, maxval=0.05, dtype=float
        )
        vel = vel.at[:, 2].set(0.0)
        env.state = State.create(pos=pos, vel=vel, rad=rad_val * jnp.ones(N))
        matcher = MaterialMatchmaker.create("linear")
        mat_table = MaterialTable.from_materials(
            [
                Material.create(
                    "elasticfrict",
                    density=0.27,
                    young=6e4,
                    poisson=0.3,
                    mu=0.5,
                    e=0.9,
                    mu_r=0.5,
                )
            ],
            matcher=matcher,
        )
        env.system = System.create(
            env.state.shape,
            dt=0.004,
            domain_type="reflect",
            domain_kw={
                "box_size": box + padding,
                "anchor": jnp.asarray(
                    [-padding / 2, -padding / 2, -2 * rad_val],
                    dtype=float,
                ),
            },
            force_model_type="cundallstrack",
            force_manager_kw={
                "gravity": [0.0, 0.0, -8.0],
                "force_functions": (frictional_wall_force,),
            },
            mat_table=mat_table,
        )
        env.env_params["objective"] = objective
        env.env_params["action"] = jnp.zeros((N, 4))
        env.env_params["magnet"] = jnp.zeros(N)
        env.env_params["r_bar"] = jnp.zeros(N, dtype=float)
        env.env_params["current_reward"] = jnp.zeros(N, dtype=float)
        n_az = env.n_lidar_rays
        n_el = env.n_lidar_elevation
        ray_dirs = env.env_params["ray_dirs"]
        env.env_params = _update_lidar_vr(
            env.state, env.system, env.env_params, ray_dirs, n_az, n_el, N
        )
        env.env_params = _update_topk_obs(
            env.state, env.system, env.env_params, objective, N, k
        )
        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmRoller3D.step")
    def step(env: "SwarmRoller3D", action: jax.Array) -> Environment:
        r"""Advance the environment by one physics step.

        Applies torque actions with angular damping, viscous drag, and
        pairwise magnetic attraction.  After integration the method
        updates all sensor caches and computes the reward with a
        differential baseline.  The shaping signal is summed over the
        *k* nearest objectives.

        Parameters
        ----------
        env : Environment
            Current environment.
        action : jax.Array
            Actions for every agent, shape ``(N * 4,)``
            (3-D torque + magnet flag).

        Returns
        -------
        Environment
            Updated environment after physics integration, sensor
            updates, and reward computation.

        """
        N = env.max_num_agents
        k = env.k_objectives
        act = action.reshape(N, 4)
        torque_act = act[:, :3]
        magnet = (act[:, 3] > 0.0).astype(jnp.float32)
        env.env_params["magnet"] = magnet
        env.env_params["action"] = act
        torque = torque_act - env.env_params["ang_damping"] * env.state.ang_vel
        force = -env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.system = env.system.force_manager.add_torque(env.state, env.system, torque)
        mag_force = _magnetic_force(
            env.state.pos,
            env.state.rad,
            magnet,
            env.env_params["magnet_strength"],
            env.env_params["magnet_range"],
            env.system,
        )
        env.system = env.system.force_manager.add_force(
            env.state, env.system, mag_force
        )
        prev_dist_all = env.env_params["prev_dist_all"]
        env.state, env.system = env.system.step(env.state, env.system)
        n_az = env.n_lidar_rays
        n_el = env.n_lidar_elevation
        ray_dirs = env.env_params["ray_dirs"]
        env.env_params = _update_lidar_vr(
            env.state, env.system, env.env_params, ray_dirs, n_az, n_el, N
        )
        objective = env.env_params["objective"]
        env.env_params = _update_topk_obs(
            env.state, env.system, env.env_params, objective, N, k
        )
        dist = env.env_params["prev_dist_all"]

        # Shaping summed over top-k objectives
        _, top_k_idx = jax.lax.top_k(-dist, k)
        top_k_dist = jnp.take_along_axis(dist, top_k_idx, axis=1)
        top_k_prev = jnp.take_along_axis(prev_dist_all, top_k_idx, axis=1)
        shaping = jnp.sum(jnp.exp(-2 * top_k_dist) - jnp.exp(-2 * top_k_prev), axis=-1)

        # On target reward
        thresh = env.env_params["goal_radius_factor"] * env.state.rad[0]
        at_obj = dist < thresh
        on_target = jnp.any(at_obj, axis=1).astype(jnp.float32)

        # Penalties
        work = norm2(torque_act)
        vel_penalty = norm2(env.state.vel)

        # Average team height
        team_height = jnp.mean(env.state.pos[:, 2])

        R_raw = (
            env.env_params["shaping_weight"] * shaping
            + env.env_params["team_height_weight"] * team_height
            + env.env_params["goal_weight"] * on_target
            - env.env_params["work_weight"] * work
            - env.env_params["velocity_weight"] * vel_penalty
        )

        r_bar = env.env_params["r_bar"]
        alpha = env.env_params["alpha_r_bar"]
        env.env_params["r_bar"] = r_bar + alpha * (R_raw - r_bar)
        env.env_params["current_reward"] = R_raw - r_bar
        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmRoller3D.observation")
    def observation(env: "SwarmRoller3D") -> jax.Array:
        r"""Build the per-agent observation vector from cached sensors.
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
                env.state.ang_vel,
                env.env_params["magnet"][:, None],
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
    @partial(jax.named_call, name="SwarmRoller3D.reward")
    def reward(env: "SwarmRoller3D") -> jax.Array:
        r"""Return the reward cached by :meth:`step`.

        Returns
        -------
        jax.Array
            Reward vector of shape ``(N,)``.

        """
        return env.env_params["current_reward"]

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="SwarmRoller3D.done")
    def done(env: "SwarmRoller3D") -> jax.Array:
        r"""Return ``True`` when the episode has exceeded ``max_steps``."""
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """Number of scalar actions per agent (3-D torque + magnet)."""
        return 4

    @property
    def action_space_shape(self) -> tuple[int]:
        """Shape of a single agent's action (``(4,)``)."""
        return (4,)

    @property
    def observation_space_size(self) -> int:
        """Dimensionality of a single agent's observation vector."""
        # vel(3) + ang_vel(3) + magnet(1)
        # + lidar(n_az*n_el) + lidar_vr(n_az*n_el) + lidar_obj(n_az*n_el)
        # + units(k*3) + clipped(k*3) + occ(k)
        n_lidar = self.n_lidar_rays * self.n_lidar_elevation
        return 7 + 3 * n_lidar + 7 * self.k_objectives


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_ray_dirs(n_az: int, n_el: int) -> jnp.ndarray:
    """Precompute unit ray direction vectors for the 3-D LiDAR grid."""
    az = jnp.linspace(-jnp.pi, jnp.pi, n_az, endpoint=False) + jnp.pi / n_az
    el = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_el, endpoint=True)
    flat_az = jnp.repeat(az, n_el)
    flat_el = jnp.tile(el, n_az)
    return jnp.stack(
        [
            jnp.cos(flat_el) * jnp.cos(flat_az),
            jnp.cos(flat_el) * jnp.sin(flat_az),
            jnp.sin(flat_el),
        ],
        axis=-1,
    )


def _update_lidar_vr(
    state: State,
    system: System,
    env_params: dict[str, Any],
    ray_dirs: jax.Array,
    n_az: int,
    n_el: int,
    N: int,
) -> dict[str, Any]:
    """Update agent-to-agent LiDAR, radial velocity, and objective LiDAR caches."""
    _, _, env_params["lidar"], env_params["lidar_idx"], _ = lidar_3d(
        state,
        system,
        env_params["lidar_range"],
        n_az,
        n_el,
        N,
        sense_edges=True,
    )
    is_agent = env_params["lidar_idx"] >= 0
    safe_idx = jnp.where(is_agent, env_params["lidar_idx"], 0)
    rel_vel = state.vel[safe_idx] - state.vel[:, None, :]
    env_params["lidar_vr"] = jnp.where(
        is_agent & (env_params["lidar"] > 0),
        dot(rel_vel, ray_dirs),
        0.0,
    )
    env_params["lidar_obj"], _, _ = cross_lidar_3d(
        state.pos,
        env_params["objective"],
        system,
        env_params["lidar_range"],
        n_az,
        n_el,
        N,
    )
    return env_params


def _update_topk_obs(
    state: State,
    system: System,
    env_params: dict[str, Any],
    objective: jax.Array,
    N: int,
    k: int,
) -> dict[str, Any]:
    """Compute distances to all objectives and update the top-k observation caches."""
    delta_all = system.domain.displacement(
        state.pos[:, None, :],
        objective[None, :, :],
        system,
    )
    dist = norm(delta_all)
    env_params["prev_dist_all"] = dist
    thresh = env_params["goal_radius_factor"] * state.rad[0]
    at_obj = dist < thresh
    someone_at_obj = jnp.any(at_obj, axis=0)
    closest_agent = jnp.argmin(dist, axis=0)
    occ_by_others = someone_at_obj[None, :] & (
        closest_agent[None, :] != jnp.arange(N)[:, None]
    )
    _, top_k_idx = jax.lax.top_k(-dist, k)
    top_k_deltas = jnp.take_along_axis(delta_all, top_k_idx[..., None], axis=1)
    top_k_units, top_k_dist = unit_and_norm(top_k_deltas)
    top_k_dist = top_k_dist[..., 0]
    clip_range = env_params["lidar_range"]
    clipped = top_k_units * jnp.minimum(top_k_dist[..., None], clip_range)
    top_k_occ = jnp.take_along_axis(
        occ_by_others.astype(jnp.float32),
        top_k_idx,
        axis=1,
    )
    env_params["top_k_units"] = top_k_units.reshape(N, k * 3)
    env_params["top_k_clipped"] = clipped.reshape(N, k * 3)
    env_params["top_k_occupied"] = top_k_occ
    return env_params


def _pyramid_layout(n_obj: int, rad: float) -> jnp.ndarray:
    r"""Build ``n_obj`` sphere positions in a square pyramid, centred at the origin.
    Each layer's grid is centred at ``(0, 0)`` so that all layers share
    the same vertical axis regardless of how many extra spheres are
    distributed to intermediate levels.
    """
    import math

    if n_obj <= 0:
        return jnp.zeros((0, 3))
    # Number of layers: largest h where the full square-pyramid
    # sum  h*(h+1)*(2*h+1)/6  fits within n_obj.
    h = 1
    while (h + 1) * (h + 2) * (2 * h + 3) // 6 <= n_obj:
        h += 1
    # Standard layer counts: layer k (bottom-up) has (h-k)² spheres.
    # Sides alternate odd/even so upper spheres fit in the crevices of
    # the layer below.  All extra objectives go to the base layer to
    # preserve the odd-even-odd grid pattern of upper layers.
    layer_counts = [(h - k) ** 2 for k in range(h)]
    layer_counts[0] += n_obj - sum(layer_counts)
    # Generate grid positions for each layer, centred at the origin.
    # For non-square counts, select the positions closest to the centre
    # so extra spheres end up in the middle, not on the edges.
    positions: list[list[float]] = []
    for k in range(h):
        count = layer_counts[k]
        side = math.ceil(math.sqrt(count))
        z = rad + k * rad * math.sqrt(2.0)
        candidates = []
        for i in range(side):
            for j in range(side):
                x = (i - (side - 1) / 2.0) * 2.0 * rad
                y = (j - (side - 1) / 2.0) * 2.0 * rad
                candidates.append((x * x + y * y, x, y))
        candidates.sort()
        for _, x, y in candidates[:count]:
            positions.append([x, y, z])
    return jnp.asarray(positions[:n_obj], dtype=float)


def _magnetic_force(
    pos: jax.Array,
    rad: jax.Array,
    magnet: jax.Array,
    strength: jax.Array,
    mag_range: jax.Array,
    system: System,
) -> jax.Array:
    r"""Compute pairwise magnetic attraction between particles.
    Follows the same displacement convention used by the Cundall–Strack
    force model and the naive collider:

    * ``rij = displacement(pos_i, pos_j) = pos_i − pos_j`` (from *j*
      toward *i*).
    * Newton's 3rd law is guaranteed because ``pair_mag`` is symmetric
      and ``n̂`` is antisymmetric: ``F_{ji} = −F_{ij}``.
    Contact friction is handled by the Cundall–Strack force model;
    the magnetic attraction increases particle overlap which in turn
    raises the CS friction cap.
    """
    N = pos.shape[0]
    # --- pairwise geometry (same convention as CundallStrackForce) ---
    # rij[i,j] = pos[i] - pos[j]  (vector from j toward i)
    rij = system.domain.displacement(pos[:, None, :], pos[None, :, :], system)
    n, r = unit_and_norm(rij)
    r = r[..., 0]
    # Symmetric activation: sum of booleans (0, 1, or 2)
    pair_mag = magnet[:, None] + magnet[None, :]
    decay = jnp.maximum(0.0, 1.0 - r / mag_range)
    mask = 1.0 - jnp.eye(N)
    # Per-pair magnetic pressing magnitude (symmetric ⇒ Newton's 3rd law)
    F_n_mag = strength * pair_mag * decay * mask
    # Attraction toward j: force_on_i = −F_n_mag · n̂  (opposite to rij)
    return jnp.sum(-F_n_mag[..., None] * n, axis=1)


__all__ = ["SwarmRoller3D"]
