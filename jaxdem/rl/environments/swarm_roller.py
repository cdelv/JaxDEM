# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where multiple rolling agents navigate towards nearby shared targets."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ...material_matchmakers import MaterialMatchmaker
from ...materials import Material, MaterialTable
from ...state import State
from ...system import System
from ...utils import cross_lidar_2d, lidar_2d, unit
from ...utils.linalg import cross, dot, norm
from . import Environment


@partial(jax.jit, static_argnames=("N",))
@partial(jax.named_call, name="swarm_roller._sample_objectives_3d")
def _sample_objectives_3d(
    key: ArrayLike, N: int, box: jax.Array, rad: float
) -> jax.Array:
    r"""Sample *N* positions on a jittered X-Y grid at floor level."""
    if N == 0:
        return jnp.zeros((0, 3), dtype=box.dtype)

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


@partial(jax.named_call, name="swarm_roller.frictional_wall_force")
def frictional_wall_force(
    pos: jax.Array, state: State, system: System
) -> tuple[jax.Array, jax.Array]:
    r"""Normal, frictional, and restitution forces for spheres on a :math:`z = 0` plane."""
    k = 2e5
    mu = 0.4
    restitution = 0.6
    n = jnp.array([0.0, 0.0, 1.0])

    dist = pos[..., 2] - state.rad
    penetration = jnp.minimum(0.0, dist)
    force_n = (-k * penetration)[..., None] * n

    v_n_scalar = dot(state.vel, n)[..., None]
    in_contact = (penetration < 0)[..., None]
    c_n = (2.0 * (1.0 - restitution) * jnp.sqrt(k * state.mass))[..., None]
    c_n = jnp.minimum(c_n, (0.5 * state.mass / system.dt)[..., None])
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


@Environment.register("swarmRoller")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SwarmRoller(Environment):
    r"""Multi-agent rolling environment toward nearby shared targets.

    Each agent controls a torque vector that is applied directly to a sphere
    on a :math:`z=0` floor. Translational drag ``-friction * vel`` and
    angular damping ``-friction * ang_vel`` are applied each step. Objectives
    are sampled globally, and each agent observes objective LiDAR and agent
    LiDAR.

    At reset, a small subset of agents is spawned in the central objective
    region while the rest are spawned in the outer padding ring.

    The reward uses exponential potential-based shaping:

    .. math::

        R_i = (S_i - S_i^{\mathrm{prev}})
              - w_{\mathrm{ke}}(K_i - K_i^{\mathrm{prev}})
              + w_{\mathrm{coop}} \cdot \frac{1}{N}\sum_m
                (S_m - S_m^{\mathrm{prev}})
              + w_{\mathrm{near}}\,\mathbf{1}[d_i \le r_i]

    where :math:`d_i` is the distance to the closest objective,
    :math:`K_i` is the translational kinetic energy of agent :math:`i`, and
    :math:`S_i = \sum_{r \in \text{obj-LiDAR}} e^{-4 d_{ir}}` sums exponential
    shaping over objectives detected by objective LiDAR rays.

    Notes
    -----
    The observation vector per agent is:

    ====================================  =================
    Feature                               Size
    ====================================  =================
    Velocity                              ``dim``
    Objective LiDAR proximity             ``n_lidar_rays``
    Agent LiDAR proximity                 ``n_lidar_rays``
    ====================================  =================
    """

    n_lidar_rays: int = jax.tree.static()
    """Number of angular bins for each LiDAR sensor."""

    @classmethod
    @partial(jax.named_call, name="SwarmRoller.Create")
    def Create(
        cls,
        N: int = 64,
        min_box_size: float = 20.0,
        max_box_size: float = 20.0,
        box_padding: float = 20.0,
        max_steps: int = 10000 * 10,
        friction: float = 0.2,
        ke_weight: float = 0.1,
        coop_weight: float = 0.2,
        near_goal_bonus: float = 0.1,
        lidar_range: float = 10.0,
        n_lidar_rays: int = 24,
    ) -> SwarmRoller:
        r"""Create a swarm roller environment.

        Parameters
        ----------
        N : int
            Number of agents and number of sampled objectives.
        min_box_size, max_box_size : float
            Range for the random square domain side length sampled at each
            :meth:`reset`.
        box_padding : float
            Extra padding around the domain in multiples of the particle
            radius. The padding region is used as the outer spawn ring.
        max_steps : int
            Episode length in physics steps.
        friction : float
            Translational and angular damping coefficient.
        ke_weight : float
            Weight for the differential kinetic energy penalty.
        coop_weight : float
            Weight for the shared team-progress bonus.
        near_goal_bonus : float
            Reward bonus applied when an agent is within one radius of
            its closest objective.
        lidar_range : float
            Maximum detection range for the LiDAR sensor.
        n_lidar_rays : int
            Number of angular LiDAR bins spanning
            :math:`[-\pi, \pi)`.
        Returns
        -------
        SwarmRoller
            A freshly constructed environment (call :meth:`reset` before
            use).

        """
        dim = 3
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = {
            "objective": jnp.zeros_like(state.pos),
            "curr_dist": jnp.zeros_like(state.rad),
            "prev_shaping_sum": jnp.zeros(state.N, dtype=float),
            "curr_shaping_sum": jnp.zeros(state.N, dtype=float),
            "curr_ke": jnp.zeros(state.N, dtype=float),
            "prev_ke": jnp.zeros(state.N, dtype=float),
            "min_box_size": jnp.asarray(min_box_size, dtype=float),
            "max_box_size": jnp.asarray(max_box_size, dtype=float),
            "box_padding": jnp.asarray(box_padding, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "ke_weight": jnp.asarray(ke_weight, dtype=float),
            "coop_weight": jnp.asarray(coop_weight, dtype=float),
            "near_goal_bonus": jnp.asarray(near_goal_bonus, dtype=float),
            "lidar_range": jnp.asarray(lidar_range, dtype=float),
            "lidar": jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
            "lidar_obj": jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
        }

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            n_lidar_rays=int(n_lidar_rays),
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmRoller.reset")
    def reset(env: "SwarmRoller", key: ArrayLike) -> Environment:
        """Initialize the environment with random positions and objectives.

        Parameters
        ----------
        env : Environment
            Current environment instance.
        key : ArrayLike
            JAX random number generator key.

        Returns
        -------
        Environment
            Freshly initialized environment.

        """
        key_box, key_pos_mid, key_pos_pad, key_objective, key_mix = jax.random.split(
            key, 5
        )
        N = env.max_num_agents
        n_rays = env.n_lidar_rays
        rad = 1.0

        box_xy = jax.random.uniform(
            key_box,
            (2,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
            dtype=float,
        )
        box = jnp.asarray([box_xy[0], box_xy[1], 2.0 * rad], dtype=float)
        padding = env.env_params["box_padding"] * rad

        n_middle = max(1, int(N) // 6)
        n_outer = int(N) - n_middle

        pos_mid = _sample_objectives_3d(key_pos_mid, n_middle, box, rad)

        key_left, key_right, key_bottom, key_top = jax.random.split(key_pos_pad, 4)
        n_left = n_outer // 4
        n_right = n_outer // 4
        n_bottom = n_outer // 4
        n_top = n_outer - n_left - n_right - n_bottom

        strip_w = padding / 2
        full_h = box[1] + padding

        left_box = jnp.asarray([strip_w, full_h, 2.0 * rad], dtype=float)
        right_box = jnp.asarray([strip_w, full_h, 2.0 * rad], dtype=float)
        bottom_box = jnp.asarray([box[0], strip_w, 2.0 * rad], dtype=float)
        top_box = jnp.asarray([box[0], strip_w, 2.0 * rad], dtype=float)

        left_anchor = jnp.asarray([-strip_w, -strip_w, 0.0], dtype=float)
        right_anchor = jnp.asarray([box[0], -strip_w, 0.0], dtype=float)
        bottom_anchor = jnp.asarray([0.0, -strip_w, 0.0], dtype=float)
        top_anchor = jnp.asarray([0.0, box[1], 0.0], dtype=float)

        pos_left = _sample_objectives_3d(key_left, n_left, left_box, rad) + left_anchor
        pos_right = (
            _sample_objectives_3d(key_right, n_right, right_box, rad) + right_anchor
        )
        pos_bottom = (
            _sample_objectives_3d(key_bottom, n_bottom, bottom_box, rad) + bottom_anchor
        )
        pos_top = _sample_objectives_3d(key_top, n_top, top_box, rad) + top_anchor

        pos_outer = jnp.concatenate([pos_left, pos_right, pos_bottom, pos_top], axis=0)
        pos = jnp.concatenate([pos_mid, pos_outer], axis=0)

        spawn_perm = jax.random.permutation(key_mix, jnp.arange(N, dtype=int))
        pos = pos[spawn_perm]
        env.env_params["objective"] = _sample_objectives_3d(
            key_objective, int(N), box, rad
        )
        env.state = State.create(pos=pos, rad=rad * jnp.ones(N), mass=jnp.ones(N))

        matcher = MaterialMatchmaker.create("harmonic")
        mat_table = MaterialTable.from_materials(
            [
                Material.create(
                    "elasticfrict",
                    density=1.0 / (4.0 / 3.0 * jnp.pi),
                    young=2e5,
                    poisson=0.3,
                    mu=0.1,
                    e=0.88,
                )
            ],
            matcher=matcher,
        )
        env.system = System.create(
            env.state.shape,
            dt=2e-3,
            domain_type="reflect",
            domain_kw={
                "box_size": box + padding,
                "anchor": jnp.zeros_like(box) - padding / 2,
            },
            force_manager_kw={
                "gravity": [0.0, 0.0, -1.0],
                "force_functions": (frictional_wall_force,),
            },
            mat_table=mat_table,
            force_model_type="cundallstrack",
        )

        objective = env.env_params["objective"]
        deltas_xy = env.system.domain.displacement(
            env.state.pos[:, None, :], objective[None, :, :], env.system
        )[..., :2]
        dist_all = norm(deltas_xy)
        env.env_params["curr_dist"] = jnp.min(dist_all, axis=1)

        import jaxdem.utils.thermal as thermal

        ke_t = thermal.compute_translational_kinetic_energy_per_particle(env.state)
        env.env_params["curr_ke"] = ke_t
        env.env_params["prev_ke"] = ke_t

        _, _, lidar, _, _ = lidar_2d(
            env.state,
            env.system,
            env.env_params["lidar_range"],
            n_rays,
            env.max_num_agents,
            sense_edges=True,
        )
        env.env_params["lidar"] = lidar
        env.env_params["lidar_obj"], _, _ = cross_lidar_2d(
            env.state.pos,
            objective,
            env.system,
            env.env_params["lidar_range"],
            n_rays,
            N,
        )
        obj_dist = env.env_params["lidar_range"] - env.env_params["lidar_obj"]
        obj_detected = env.env_params["lidar_obj"] > 0
        shaping_sum = jnp.sum(
            jnp.where(obj_detected, jnp.exp(-4 * obj_dist), 0.0), axis=1
        )
        env.env_params["prev_shaping_sum"] = shaping_sum
        env.env_params["curr_shaping_sum"] = shaping_sum

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmRoller.step")
    def step(env: "SwarmRoller", action: jax.Array) -> Environment:
        """Advance one step. Actions are torques; simple damping is applied.

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
        N = env.max_num_agents
        n_rays = env.n_lidar_rays

        reshaped_action = action.reshape(N, *env.action_space_shape)
        torque = reshaped_action - env.env_params["friction"] * env.state.ang_vel
        force = -env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.system = env.system.force_manager.add_torque(env.state, env.system, torque)

        env.env_params["prev_shaping_sum"] = env.env_params["curr_shaping_sum"]
        env.env_params["prev_ke"] = env.env_params["curr_ke"]
        env.state, env.system = env.system.step(env.state, env.system)

        objective = env.env_params["objective"]
        deltas_xy = env.system.domain.displacement(
            env.state.pos[:, None, :], objective[None, :, :], env.system
        )[..., :2]
        dist_all = norm(deltas_xy)
        env.env_params["curr_dist"] = jnp.min(dist_all, axis=1)

        _, _, lidar, _, _ = lidar_2d(
            env.state,
            env.system,
            env.env_params["lidar_range"],
            n_rays,
            env.max_num_agents,
            sense_edges=True,
        )
        env.env_params["lidar"] = lidar
        env.env_params["lidar_obj"], _, _ = cross_lidar_2d(
            env.state.pos,
            objective,
            env.system,
            env.env_params["lidar_range"],
            n_rays,
            N,
        )
        obj_dist = env.env_params["lidar_range"] - env.env_params["lidar_obj"]
        obj_detected = env.env_params["lidar_obj"] > 0
        env.env_params["curr_shaping_sum"] = jnp.sum(
            jnp.where(obj_detected, jnp.exp(-4 * obj_dist), 0.0), axis=1
        )

        import jaxdem.utils.thermal as thermal

        env.env_params["curr_ke"] = (
            thermal.compute_translational_kinetic_energy_per_particle(env.state)
        )

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmRoller.observation")
    def observation(env: "SwarmRoller") -> jax.Array:
        """Build per-agent observations.

        Contents per agent
        ------------------
        - Velocity (shape (dim,)).
        - Objective LiDAR proximity, normalized by ``lidar_range`` (shape (n_lidar_rays,)).
        - Agent LiDAR proximity, normalized by ``lidar_range`` (shape (n_lidar_rays,)).

        Returns
        -------
        jax.Array
            Array of shape ``(N, dim + 2 * n_lidar_rays)``

        """
        return jnp.concatenate(
            [
                env.state.vel,
                env.env_params["lidar_obj"] / env.env_params["lidar_range"],
                env.env_params["lidar"] / env.env_params["lidar_range"],
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmRoller.reward")
    def reward(env: "SwarmRoller") -> jax.Array:
        r"""Returns a vector of per-agent rewards.

        .. math::

           \mathrm{rew}_t = (S_t - S_t^{\mathrm{prev}})
           - w_{\text{ke}} (K_t - K_{t-1})
           + w_{\text{coop}} \cdot \mathrm{mean}\left(
           (S_t - S_t^{\mathrm{prev}})\right)
           + w_{\text{near}} \cdot \mathbf{1}[d_t \le r]

        where :math:`d_t` is the distance to the closest objective at step
        :math:`t`, :math:`K_t` is the kinetic energy at step :math:`t`, and
        :math:`S_t` is the per-agent sum of :math:`e^{-4d}` over objectives
        detected by objective LiDAR rays,
        :math:`w_{\text{ke}}` is the kinetic-energy penalty weight, and
        :math:`w_{\text{coop}}` weights a shared team-progress bonus, and
        :math:`w_{\text{near}}` weights a near-goal bonus.

        Parameters
        ----------
        env : Environment
            Current environment.

        Returns
        -------
        jax.Array
            Shape ``(N,)``.

        """
        shaping_reward = (
            env.env_params["curr_shaping_sum"] - env.env_params["prev_shaping_sum"]
        )
        ke_diff = env.env_params["curr_ke"] - env.env_params["prev_ke"]
        near_goal_bonus = env.env_params["near_goal_bonus"] * jnp.where(
            env.env_params["curr_dist"] <= env.state.rad, 1.0, 0.0
        )
        coop_bonus = env.env_params["coop_weight"] * jnp.mean(shaping_reward)
        return (
            shaping_reward
            - env.env_params["ke_weight"] * ke_diff
            + coop_bonus
            + near_goal_bonus
        )

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="SwarmRoller.done")
    def done(env: "SwarmRoller") -> jax.Array:
        """Returns a boolean indicating whether the environment has ended.
        The episode terminates when the maximum number of steps is reached.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            Boolean array indicating whether the environment has ended.

        """
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """Flattened action size per agent. Actions passed to :meth:`step` have shape ``(A, action_space_size)``."""
        return 3

    @property
    def action_space_shape(self) -> tuple[int]:
        """Original per-agent action shape (useful for reshaping inside the environment)."""
        return (3,)

    @property
    def observation_space_size(self) -> int:
        """Flattened observation size per agent. :meth:`observation` returns shape ``(A, observation_space_size)``."""
        return self.state.dim + 2 * self.n_lidar_rays


__all__ = ["SwarmRoller"]
