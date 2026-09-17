# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where multiple rolling agents navigate toward assigned targets."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from math import isfinite

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ...material_matchmakers import MaterialMatchmaker
from ...materials import Material, MaterialTable
from ...state import State
from ...system import System
from ...utils import lidar_2d, thermal
from ...utils.linalg import norm, unit
from . import Environment
from .single_roller import frictional_wall_force


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

    Each physics step smooths each requested torque:

    .. math::

       \mathbf{u}_{i,t} = \alpha\,\mathbf{a}_{i,t}
           + (1 - \alpha)\,\mathbf{u}_{i,t-1}.

    Here :math:`\mathbf{a}_{i,t}` is the requested torque and
    :math:`\mathbf{u}_{i,t}` is the applied torque before angular damping.
    ``action_alpha`` defaults to 0.22. The previous applied torque is from
    the immediately preceding physics step, independently of checkpoints.
    Reset initializes :math:`\mathbf{u}_{i,0}=\mathbf{0}`.

    As in MultiNavigator, reward uses a radius-scaled quartic potential:

    .. math::

       R_i = 1.5\,\mathrm{rad}_i,
       \qquad
       \Phi_i(d,K) = \exp\!\left[-\left(\frac{d}{R_i}\right)^4
           - \beta\frac{K}{K_{\mathrm{ref}}}\right],
       \qquad
       r_{i,t} = \Phi_i(d_{i,t},K_{i,t})
           - \Phi_i(d_{i,\mathrm{checkpoint}},K_{i,\mathrm{checkpoint}}).

    Distance is measured in 3-D from each particle's center to its own
    assigned objective, as in SingleRoller. The potential is flat near
    the objective and rapidly approaches zero far away. Total kinetic energy is

    .. math::

       K_i = \tfrac12 m_i\|\mathbf{v}_i\|^2
           + \tfrac12\boldsymbol{\omega}_{i,b}^{\top}
               I_{i,b}\boldsymbol{\omega}_{i,b}.

    Angular velocity and inertia are expressed in the body's principal frame.
    ``kinetic_energy_coeff`` sets :math:`\beta` (default zero), and
    ``kinetic_energy_scale`` sets :math:`K_{\mathrm{ref}}` (default one).
    At fixed distance, slowing translation or rotation increases the potential
    near the objective and has little effect far away. At rest, the original
    flat goal potential is recovered. Unchanged distance and energy give zero
    reward; there is no per-step motion cost or occupancy bonus. Reset gives
    zero reward.

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
    Normalized agent/wall LiDAR    ``n_lidar_rays``
    ============================  =================

    With ``dt = 0.002`` and ``skip_frames = 49``, each action spans 50
    physics steps (0.1 seconds). One checkpoint captures each agent's
    starting potential in ``prev_potential`` and distance in ``prev_dist``;
    physics steps preserve this history. Observations and rewards read the
    live state. Floor forces use the same implementation as SingleRoller.
    Particle contacts use normal spring forces and the naive all-pairs
    collider, as in MultiNavigator. Tangential particle-contact friction
    and contact history are not used; floor friction still drives rolling.
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
        max_steps: int = 100000,
        friction: float = 0.2,
        action_alpha: float = 0.22,
        lidar_range: float = 6.0,
        n_lidar_rays: int = 16,
        kinetic_energy_coeff: float = 0.0,
        kinetic_energy_scale: float = 1.0,
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
        action_alpha : float
            Fraction of the requested torque applied by the smoothing update
            each physics step, in ``[0, 1]``. One disables smoothing.
        lidar_range : float
            Maximum detection range for the LiDAR sensor.
        n_lidar_rays : int
            Number of angular LiDAR bins spanning
            :math:`[-\pi, \pi)`.

        kinetic_energy_coeff : float
            Nonnegative dimensionless strength of the kinetic-energy factor.
            Zero preserves the distance-only potential.
        kinetic_energy_scale : float
            Positive reference kinetic energy in simulation units. The energy
            factor is ``exp(-kinetic_energy_coeff * K / kinetic_energy_scale)``.
            Energy includes both translation and rotation.

        Returns
        -------
        MultiRoller
            The constructed environment. Call :meth:`reset` before use.

        """
        if not 0.0 <= action_alpha <= 1.0:
            raise ValueError("action_alpha must be in [0, 1]")
        if not isfinite(kinetic_energy_coeff) or kinetic_energy_coeff < 0:
            raise ValueError("kinetic_energy_coeff must be finite and nonnegative")
        if not isfinite(kinetic_energy_scale) or kinetic_energy_scale <= 0:
            raise ValueError("kinetic_energy_scale must be finite and positive")
        dim = 3
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = {
            "objective": jnp.zeros_like(state.pos),
            "permutation": jnp.arange(N, dtype=int),
            "prev_dist": jnp.zeros_like(state.rad),
            "prev_potential": jnp.zeros_like(state.rad),
            "kinetic_energy_coeff": jnp.asarray(kinetic_energy_coeff, dtype=float),
            "kinetic_energy_scale": jnp.asarray(kinetic_energy_scale, dtype=float),
            "min_box_size": jnp.asarray(min_box_size, dtype=float),
            "max_box_size": jnp.asarray(max_box_size, dtype=float),
            "box_padding": jnp.asarray(box_padding, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "action_alpha": jnp.asarray(action_alpha, dtype=float),
            "applied_action": jnp.zeros_like(state.torque),
            "lidar_range": jnp.asarray(lidar_range, dtype=float),
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
        objective = _sample_objectives_3d(key_objective, int(N), box, rad)
        perm = jax.random.permutation(key_shuffle, jnp.arange(N, dtype=int))
        env.env_params["objective"] = objective[perm]
        env.env_params["permutation"] = perm

        env.state = State.create(pos=pos, rad=rad * jnp.ones(N), mass=jnp.ones(N))

        matcher = MaterialMatchmaker.create("harmonic")
        mat_table = MaterialTable.from_materials(
            [
                Material.create(
                    "elastic",
                    density=1.0 / (4.0 / 3.0 * jnp.pi),
                    young=2e5,
                    poisson=0.3,
                )
            ],
            matcher=matcher,
        )
        env.system = System.create(
            env.state.shape,
            dt=2e-3,
            domain_type="reflectsphere",
            domain_kw={
                "box_size": box + padding,
                "anchor": jnp.array([-padding / 2, -padding / 2, -rad]),
            },
            force_manager_kw={
                "gravity": [0.0, 0.0, -1.0],
                "force_functions": (frictional_wall_force,),
            },
            mat_table=mat_table,
            force_model_type="spring",
            collider_type="naive",
        )
        env.state, env.system = System.initialize(env.state, env.system)

        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        env.env_params["prev_dist"] = norm(delta)
        scale = 1.5 * env.state.rad
        kinetic_energy = thermal.compute_translational_kinetic_energy_per_particle(
            env.state
        ) + thermal.compute_rotational_kinetic_energy_per_particle(env.state)
        energy_cost = (
            env.env_params["kinetic_energy_coeff"]
            * kinetic_energy
            / env.env_params["kinetic_energy_scale"]
        )
        env.env_params["prev_potential"] = jnp.exp(
            -((env.env_params["prev_dist"] / scale) ** 4) - energy_cost
        )
        env.env_params["applied_action"] = jnp.zeros_like(env.state.torque)
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiRoller.step")
    def step(env: MultiRoller, action: jax.Array) -> Environment:
        """Smooth the requested torque, then advance physics with damping.

        ``applied_action`` follows each accepted physics step. ``prev_dist``
        and ``prev_potential`` remain fixed until the next checkpoint or reset.

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
        reshaped_action = action.reshape(env.max_num_agents, *env.action_space_shape)
        alpha = env.env_params["action_alpha"]
        applied_action = (
            alpha * reshaped_action + (1 - alpha) * env.env_params["applied_action"]
        )
        env.env_params["applied_action"] = applied_action
        torque = applied_action - env.env_params["friction"] * env.state.ang_vel
        force = -env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.system = env.system.force_manager.add_torque(env.state, env.system, torque)

        env.state, env.system = env.system.step(env.state, env.system)
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiRoller.checkpoint")
    def checkpoint(env: MultiRoller, action: jax.Array) -> Environment:
        """Save the live distance and full potential before the next action.

        ``prev_potential`` includes both translational and rotational energy;
        ``prev_dist`` retains the 3-D center distance for inspection.
        The baseline does not depend on ``action``. Physics steps preserve it
        so reward covers the complete action, including skipped frames. Read
        reward before a subsequent checkpoint or reset replaces the baseline.
        The immediately preceding applied torque is preserved independently.
        """
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        env.env_params["prev_dist"] = norm(delta)
        scale = 1.5 * env.state.rad
        kinetic_energy = thermal.compute_translational_kinetic_energy_per_particle(
            env.state
        ) + thermal.compute_rotational_kinetic_energy_per_particle(env.state)
        energy_cost = (
            env.env_params["kinetic_energy_coeff"]
            * kinetic_energy
            / env.env_params["kinetic_energy_scale"]
        )
        env.env_params["prev_potential"] = jnp.exp(
            -((env.env_params["prev_dist"] / scale) ** 4) - energy_cost
        )

        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiRoller.observation")
    def observation(env: MultiRoller) -> jax.Array:
        """Build per-agent observations and LiDAR from the live physical state.

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
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        delta_xy = delta[..., :2]
        lr = env.env_params["lidar_range"]
        _, _, lidar, _, _ = lidar_2d(
            env.state, env.system, lr, env.n_lidar_rays, sense_edges=True
        )
        return jnp.concatenate(
            [
                unit(delta_xy),
                jnp.clip(delta_xy, -3.0, 3.0),
                env.state.vel[..., :2],
                env.state.ang_vel,
                lidar / lr,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiRoller.reward")
    def reward(env: MultiRoller) -> jax.Array:
        r"""Return the change in each agent's radius-scaled goal potential.

        .. math::

           \Phi_i(d,K) = \exp\!\left[-\left(
               \frac{d}{1.5\,\mathrm{rad}_i}\right)^4
               - \beta\frac{K}{K_{\mathrm{ref}}}\right],
           \qquad
           r_i = \Phi_i(d_i,K_i)
               - \Phi_i(d_{i,\mathrm{checkpoint}},K_{i,\mathrm{checkpoint}}).

        Distances use the live 3-D center positions and assigned objectives.
        ``kinetic_energy_coeff`` sets :math:`\beta`; ``kinetic_energy_scale``
        sets :math:`K_{\mathrm{ref}}`. Energy includes translation and rotation.
        The checkpoint baseline spans the complete action interval. With the
        default zero coefficient, reward depends only on distance.
        Returns an array of shape ``(N,)``.
        """
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        scale = 1.5 * env.state.rad
        kinetic_energy = thermal.compute_translational_kinetic_energy_per_particle(
            env.state
        ) + thermal.compute_rotational_kinetic_energy_per_particle(env.state)
        energy_cost = (
            env.env_params["kinetic_energy_coeff"]
            * kinetic_energy
            / env.env_params["kinetic_energy_scale"]
        )
        potential = jnp.exp(-((norm(delta) / scale) ** 4) - energy_cost)
        return potential - env.env_params["prev_potential"]

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiRoller.truncated")
    def truncated(env: MultiRoller) -> jax.Array:
        """Return whether the episode has ended.

        The episode ends when ``step_count`` reaches ``max_steps``.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            A bool that is True when the episode has ended.

        """
        return jnp.asarray(env.system.step_count >= env.env_params["max_steps"])

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
