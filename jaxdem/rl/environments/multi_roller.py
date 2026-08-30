# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where multiple rolling agents navigate toward assigned targets."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import jaxdem.utils.thermal as thermal

from ...material_matchmakers import MaterialMatchmaker
from ...materials import Material, MaterialTable
from ...state import State
from ...system import System
from ...utils import lidar_2d, unit
from ...utils.linalg import cross, dot, norm
from . import Environment


@jax.jit(inline=True, static_argnames=("N",))
@partial(jax.named_call, name="multi_roller._sample_objectives_3d")
def _sample_objectives_3d(
    key: ArrayLike, N: int, box: jax.Array, rad: float
) -> jax.Array:
    r"""Sample *N* positions on a jittered X-Y grid at floor level."""
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


@Environment.register("multiRoller")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MultiRoller(Environment):
    r"""Multi-agent rolling environment toward assigned targets.

    Each agent controls a torque vector that acts directly on a sphere
    on a :math:`z=0` floor. Each step applies translational drag
    ``-friction * vel`` and angular damping ``-friction * ang_vel``.
    The environment samples objectives and assigns them one-to-one with a
    random permutation.

    The reward uses potential-based shaping with a proximity-gated
    kinetic-energy term:

    .. math::

        \varphi(d, K) = \exp\!\left(-2 d^{\mathrm{eff}} - \frac{K}{\text{ke\_tau}}\,e^{-\text{ke\_gate} \cdot d^{\mathrm{eff}}}\right)

    where :math:`d^{\mathrm{eff}} = \max(0, d - 0.5 r)`, :math:`d` is the
    distance to the assigned objective in the :math:`xy` plane, and
    :math:`K` is the translational kinetic energy. ``ke_tau`` sets the
    overall strength of the KE penalty. ``ke_gate`` controls how sharply KE
    sensitivity falls off with distance. A larger ``ke_gate`` means KE only
    matters very close to the objective. The per-agent shaping credit is
    :math:`F_i = \varphi(d^{\mathrm{eff}}_t, K_t) - \varphi(d^{\mathrm{eff}}_{t-1}, K_{t-1})`.

    Notes
    -----
    The observation vector per agent is:

    ============================  =================
    Feature                       Size
    ============================  =================
    Unit direction to objective   ``2``
    Clamped displacement          ``2``
    Velocity                      ``2``
    Angular velocity              ``3``
    LiDAR proximity (normalized)  ``n_lidar_rays``
    ============================  =================

    For realistic training parameters, ``skip_frames = 50`` gives a response
    rate of 200 Hz, so ``num_steps_epoch = 100`` gives a horizon of 0.5
    seconds.
    """

    n_lidar_rays: int = jax.tree.static()
    """Number of angular bins for each LiDAR sensor."""

    @classmethod
    @partial(jax.named_call, name="MultiRoller.Create")
    def Create(
        cls,
        N: int = 64,
        min_box_size: float = 20.0,
        max_box_size: float = 20.0,
        box_padding: float = 5.0,
        max_steps: int = 10000 * 10,
        friction: float = 0.2,
        ke_tau: float = 5.0,
        ke_gate: float = 4.0,
        near_goal_bonus: float = 0.1,
        lidar_range: float = 6.0,
        n_lidar_rays: int = 16,
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
            Translational and angular damping coefficient.
        ke_tau : float
            Overall strength of the KE term in the potential (larger =
            less important). See class docstring.
        ke_gate : float
            Distance decay rate of KE sensitivity (larger = KE only
            matters very close to the goal). See class docstring.
        near_goal_bonus : float
            Reward bonus applied when an agent is within one radius of
            its objective.
        lidar_range : float
            Maximum detection range for the LiDAR sensor.
        n_lidar_rays : int
            Number of angular LiDAR bins spanning
            :math:`[-\pi, \pi)`.

        Returns
        -------
        MultiRoller
            The constructed environment. Call :meth:`reset` before use.

        """
        dim = 3
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = {
            "objective": jnp.zeros_like(state.pos),
            "permutation": jnp.arange(N, dtype=int),
            "prev_dist": jnp.zeros_like(state.rad),
            "prev_ke": jnp.zeros(state.N, dtype=float),
            "min_box_size": jnp.asarray(min_box_size, dtype=float),
            "max_box_size": jnp.asarray(max_box_size, dtype=float),
            "box_padding": jnp.asarray(box_padding, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "ke_tau": jnp.asarray(ke_tau, dtype=float),
            "ke_gate": jnp.asarray(ke_gate, dtype=float),
            "near_goal_bonus": jnp.asarray(near_goal_bonus, dtype=float),
            "lidar_range": jnp.asarray(lidar_range, dtype=float),
            "lidar": jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
        }

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            n_lidar_rays=int(n_lidar_rays),
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiRoller.reset")
    def reset(env: MultiRoller, key: ArrayLike) -> Environment:
        """Initialize the environment with random positions and objectives.

        Parameters
        ----------
        env : Environment
            The current environment.
        key : ArrayLike
            JAX random number generator key.

        Returns
        -------
        Environment
            The initialized environment.

        """
        key_box, key_pos, key_objective, key_shuffle = jax.random.split(key, 4)
        N = env.max_num_agents
        n_rays = env.n_lidar_rays
        rad = 1.0

        box = jax.random.uniform(
            key_box,
            (3,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
            dtype=float,
        )
        padding = env.env_params["box_padding"] * rad

        pos = _sample_objectives_3d(key_pos, int(N), box + padding, rad) - jnp.array(
            [padding / 2, padding / 2, 0.0]
        )
        pos = pos.at[:, 2].set(rad)

        objective = _sample_objectives_3d(key_objective, int(N), box, rad)
        objective = objective.at[:, 2].set(rad)
        perm = jax.random.permutation(key_shuffle, jnp.arange(N, dtype=int))
        env.env_params["objective"] = objective[perm]
        env.env_params["permutation"] = perm

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

        delta_xy = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )[..., :2]
        dist = norm(delta_xy)
        env.env_params["delta_xy"] = delta_xy
        env.env_params["prev_dist"] = dist

        env.env_params["prev_ke"] = (
            thermal.compute_translational_kinetic_energy_per_particle(env.state)
        )

        _, _, lidar, _, _ = lidar_2d(
            env.state,
            env.system,
            env.env_params["lidar_range"],
            n_rays,
            sense_edges=True,
        )
        env.env_params["lidar"] = lidar

        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiRoller.step")
    def step(env: MultiRoller, action: jax.Array) -> Environment:
        """Advance one step. Actions are torques. The step also applies translational drag and angular damping.

        Parameters
        ----------
        env : Environment
            The current environment.
        action : jax.Array
            The per-agent torque vectors.

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

        env.env_params["prev_dist"] = norm(env.env_params["delta_xy"])
        env.env_params["prev_ke"] = (
            thermal.compute_translational_kinetic_energy_per_particle(env.state)
        )
        env.state, env.system = env.system.step(env.state, env.system)

        delta_xy = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )[..., :2]
        env.env_params["delta_xy"] = delta_xy

        _, _, lidar, _, _ = lidar_2d(
            env.state,
            env.system,
            env.env_params["lidar_range"],
            n_rays,
            sense_edges=True,
        )
        env.env_params["lidar"] = lidar

        return env

        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiRoller.observation")
    def observation(env: MultiRoller) -> jax.Array:
        """Build per-agent observations.

        Contents per agent
        ------------------
        - Unit vector to objective in the :math:`xy` plane (shape (2,)).
        - Clamped objective delta in the :math:`xy` plane (shape (2,)).
        - Velocity in the :math:`xy` plane (shape (2,)).
        - Angular velocity (shape (3,)).
        - LiDAR proximity, normalized by ``lidar_range`` (shape (n_lidar_rays,)).

        Returns
        -------
        jax.Array
            Array of shape ``(N, 9 + n_lidar_rays)``

        """
        delta_xy = env.env_params["delta_xy"]
        dist = norm(delta_xy)
        direction = delta_xy / jnp.where(dist > 0, dist, 1.0)[:, None]
        return jnp.concatenate(
            [
                direction,
                jnp.clip(delta_xy, -3.0, 3.0),
                env.state.vel[..., :2],
                env.state.ang_vel,
                env.env_params["lidar"] / env.env_params["lidar_range"],
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiRoller.reward")
    def reward(env: MultiRoller) -> jax.Array:
        r"""Return the per-agent rewards.

        Potential-based shaping with a proximity-gated KE term:

        .. math::

           \varphi(d, K) = \exp\!\left(-2 d^{\mathrm{eff}} - \frac{K}{\text{ke\_tau}}\,e^{-\text{ke\_gate} \cdot d^{\mathrm{eff}}}\right)

        The gate :math:`e^{-\text{ke\_gate} \cdot d^{\mathrm{eff}}}` suppresses
        the KE term away from the objective, so fast motion is free until the
        agent is close. ``ke_tau`` sets the overall strength of the penalty.

        Per-step reward:

        .. math::

           \mathrm{rew}_t = \frac{F_t + w_{\text{near}} \cdot \mathbf{1}[d_t \le r]}{w_{\text{near}}}

        where :math:`F_t = \varphi(d^{\mathrm{eff}}_t, K_t) - \varphi(d^{\mathrm{eff}}_{t-1}, K_{t-1})`,
        :math:`d^{\mathrm{eff}}_t = \max(0, d_t - 0.5 r)`, and
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
        curr_dist = norm(env.env_params["delta_xy"])
        prev_dist = env.env_params["prev_dist"]

        flat_rad = 0.5 * env.state.rad
        curr_eff_dist = jnp.maximum(0.0, curr_dist - flat_rad)
        prev_eff_dist = jnp.maximum(0.0, prev_dist - flat_rad)

        tau = env.env_params["ke_tau"]
        alpha = env.env_params["ke_gate"]
        ke_curr = thermal.compute_translational_kinetic_energy_per_particle(env.state)

        phi_curr = jnp.exp(
            -2 * curr_eff_dist - ke_curr * jnp.exp(-alpha * curr_eff_dist) / tau
        )
        phi_prev = jnp.exp(
            -2 * prev_eff_dist
            - env.env_params["prev_ke"] * jnp.exp(-alpha * prev_eff_dist) / tau
        )
        shaping_reward = phi_curr - phi_prev

        near_goal_bonus = env.env_params["near_goal_bonus"] * jnp.where(
            curr_dist <= 1.0 * env.state.rad, 1.0, 0.0
        )

        return (shaping_reward + near_goal_bonus) / env.env_params["near_goal_bonus"]

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiRoller.done")
    def done(env: MultiRoller) -> jax.Array:
        """Return whether the episode has ended.

        The episode ends when ``step_count`` exceeds ``max_steps``.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            A bool that is True when the episode has ended.

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
        return 9 + self.n_lidar_rays


__all__ = ["MultiRoller"]
