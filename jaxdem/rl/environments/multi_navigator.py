# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where multiple agents navigate toward assigned targets."""

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


@jax.jit(inline=True, static_argnames=("N",))
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
    r"""Multi-agent navigation environment toward assigned targets.

    Each agent controls a force vector that acts directly on a sphere
    inside a reflective box. Each step adds viscous drag ``-friction * vel``.
    The environment samples objectives and assigns them one-to-one with a
    random permutation.

    Each physics step smooths each requested force:

    .. math::

       \mathbf{u}_t = \alpha\,\mathbf{a}_t
           + (1 - \alpha)\,\mathbf{u}_{t-1}

    Here :math:`\mathbf{a}_t` is the requested force at physics step
    :math:`t`, :math:`\mathbf{u}_t` is the applied force before drag, and
    :math:`\mathbf{u}_{t-1}` is the immediately preceding applied force.
    The parameter :math:`\alpha` is ``action_alpha``. Reset initializes
    :math:`\mathbf{u}_0 = \mathbf{0}`; checkpoints preserve this actuator
    state. With :math:`\alpha = 0.22`,
    a constant request completes 99% of its transition in 19 physics steps.

    The reward uses a quartic exponential potential measured from the
    action-start checkpoint to the live endpoint:

    .. math::

       R_i = 2\,\mathrm{rad}_i,
       \qquad
       K_i = \tfrac12 m_i \|\mathbf{v}_i\|^2,
       \qquad
       \Phi_i(d,K) = \exp\!\left[-\left(\frac{d}{R_i}\right)^4
           - \beta\frac{K}{K_{\mathrm{ref}}}\right].

    Here :math:`d` is the distance from agent :math:`i`'s center to its
    assigned objective, and :math:`\mathrm{rad}_i` is its particle radius.
    The potential is flat near the objective, making small departures
    inexpensive, and decays rapidly far away. Its scale follows each
    particle's own radius. ``kinetic_energy_coeff`` sets :math:`\beta`
    (default zero), and ``kinetic_energy_scale`` sets :math:`K_{\mathrm{ref}}`
    (default one). Rotation is disabled, so only translational energy enters.
    At fixed distance, slowing down earns credit near the objective and has
    little effect far away. This is a potential difference, not a per-step
    cost for maintaining constant speed.

    Per-action reward:

    .. math::

       r_{i,t} = \Phi_i(d_{i,t},K_{i,t})
           - \Phi_i(d_{i,\mathrm{checkpoint}},K_{i,\mathrm{checkpoint}}).

    At fixed energy, moving closer gives positive credit and moving farther
    gives negative credit. Unchanged distance and energy give zero. Reset
    initializes the historical potential from the initial state, giving zero reward.
    There are no additional neighbor, occupancy, or collision reward terms.

    Notes
    -----
    The observation vector per agent is:

    ============================  =================
    Feature                       Size
    ============================  =================
    Unit direction to objective   ``dim``
    Clamped displacement          ``dim``
    Velocity                      ``dim``
    Normalized agent/wall LiDAR    ``n_lidar_rays``
    ============================  =================

    With ``dt = 0.002`` and ``skip_frames = 49``, each action spans 50
    physics steps (0.1 seconds). One checkpoint captures each agent's
    ``prev_potential`` before the action; reward uses the live endpoint after
    all accepted physics steps. ``prev_dist`` retains the starting distance
    for inspection.
    """

    n_lidar_rays: int = jax.tree.static()
    """Number of angular bins for each LiDAR sensor."""

    @classmethod
    @partial(jax.named_call, name="MultiNavigator.Create")
    def Create(
        cls,
        N: int = 64,
        min_box_size: float = 20.0,
        max_box_size: float = 20.0,
        box_padding: float = 5.0,
        max_steps: int = 100000,
        friction: float = 0.2,
        action_alpha: float = 0.22,
        lidar_range: float = 10.0,
        n_lidar_rays: int = 16,
        kinetic_energy_coeff: float = 0.0,
        kinetic_energy_scale: float = 1.0,
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
        action_alpha : float
            Fraction of the requested force applied by the smoothing update
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

        Returns
        -------
        MultiNavigator
            The constructed environment. Call :meth:`reset` before use.

        """
        if not 0.0 <= action_alpha <= 1.0:
            raise ValueError("action_alpha must be in [0, 1]")
        if not isfinite(kinetic_energy_coeff) or kinetic_energy_coeff < 0:
            raise ValueError("kinetic_energy_coeff must be finite and nonnegative")
        if not isfinite(kinetic_energy_scale) or kinetic_energy_scale <= 0:
            raise ValueError("kinetic_energy_scale must be finite and positive")
        dim = 2
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape, rotation_integrator_type=None)

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
            "applied_action": jnp.zeros_like(state.force),
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
    @partial(jax.named_call, name="MultiNavigator.reset")
    def reset(env: MultiNavigator, key: ArrayLike) -> Environment:
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
        dim = env.state.dim
        rad = 1.0

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
        env.state = State.create(pos=pos, rad=rad * jnp.ones(N))

        matcher = MaterialMatchmaker.create("harmonic")
        mat_table = MaterialTable.from_materials(
            [
                Material.create(
                    "elastic",
                    density=1.0 / jnp.pi,
                    young=2e5,
                    poisson=0.3,
                )
            ],
            matcher=matcher,
        )
        env.system = System.create(
            env.state.shape,
            dt=2e-3,
            rotation_integrator_type=None,
            domain_type="reflectsphere",
            domain_kw={
                "box_size": box + padding,
                "anchor": jnp.zeros_like(box) - padding / 2,
            },
            mat_table=mat_table,
        )

        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        env.env_params["prev_dist"] = norm(delta)
        scale = 2 * env.state.rad
        kinetic_energy = thermal.compute_translational_kinetic_energy_per_particle(
            env.state
        )
        energy_cost = (
            env.env_params["kinetic_energy_coeff"]
            * kinetic_energy
            / env.env_params["kinetic_energy_scale"]
        )
        env.env_params["prev_potential"] = jnp.exp(
            -((env.env_params["prev_dist"] / scale) ** 4) - energy_cost
        )
        env.env_params["applied_action"] = jnp.zeros_like(env.state.force)
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiNavigator.step")
    def step(env: MultiNavigator, action: jax.Array) -> Environment:
        """Smooth the requested force, then advance physics with viscous drag.

        Smoothing uses the applied action from the immediately preceding
        physics step, including across policy actions and checkpoints.

        The historical potential remains fixed throughout the action. Use
        ``utils.advance_action`` to checkpoint once before repeating physics
        steps. Observations and rewards always read the live state.

        Parameters
        ----------
        env : Environment
            The current environment.

        action : jax.Array
            The per-agent action vectors.

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
        force = applied_action - env.state.vel * env.env_params["friction"]
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.state, env.system = env.system.step(env.state, env.system)
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiNavigator.checkpoint")
    def checkpoint(env: MultiNavigator, action: jax.Array) -> Environment:
        """Save the starting potential before the next action interval.

        Store the live distance in ``prev_dist`` and the full distance/energy
        potential in ``prev_potential``. Physics steps preserve both; reward
        subtracts the saved potential from the live endpoint's potential.

        Parameters
        ----------
        env : MultiNavigator
            The environment immediately before the next action.
        action : jax.Array
            The per-agent action about to be applied. This potential
            baseline does not depend on its value.

        Returns
        -------
        Environment
            The environment with the action-start potential saved. Read reward
            before the next checkpoint or reset replaces this baseline.
        """
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        env.env_params["prev_dist"] = norm(delta)
        scale = 2 * env.state.rad
        kinetic_energy = thermal.compute_translational_kinetic_energy_per_particle(
            env.state
        )
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
    @partial(jax.named_call, name="MultiNavigator.observation")
    def observation(env: MultiNavigator) -> jax.Array:
        r"""Build per-agent observations and LiDAR from the live physical state.

        Contents per agent
        ------------------
        - Unit vector to objective (shape (dim,))  --> Direction
        - Clamped delta to objective (shape (dim,)) --> Local precision
        - Velocity (shape (dim,))
        - Agent/wall LiDAR proximity (shape (n_lidar_rays,))

        LiDAR reports normalized proximity
        :math:`p_k = \max(0, 1 - d_{\min,k} / r_{\max})`, with zero for
        empty bins. Readings use live positions rather than checkpointed values.

        Returns
        -------
        jax.Array
            Array of shape ``(N, 3 * dim + n_lidar_rays)``

        """
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        lr = env.env_params["lidar_range"]
        _, _, lidar, _, _ = lidar_2d(
            env.state,
            env.system,
            lr,
            env.n_lidar_rays,
            sense_edges=True,
        )
        return jnp.concatenate(
            [
                unit(delta),
                jnp.clip(delta, -3.0, 3.0),
                env.state.vel,
                lidar / lr,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiNavigator.reward")
    def reward(env: MultiNavigator) -> jax.Array:
        r"""Return the potential change over the complete action interval.

        .. math::

           \Phi_i(d,K) = \exp\!\left[-\left(
               \frac{d}{2\,\mathrm{rad}_i}\right)^4
               - \beta\frac{K}{K_{\mathrm{ref}}}\right],
           \qquad
           r_{i,t} = \Phi_i(d_{i,t},K_{i,t})
               - \Phi_i(d_{i,\mathrm{checkpoint}},K_{i,\mathrm{checkpoint}}).

        ``kinetic_energy_coeff`` is :math:`\beta` and ``kinetic_energy_scale``
        is :math:`K_{\mathrm{ref}}`. Energy is translational because navigator
        rotation is disabled. With the default zero coefficient, this is the
        distance-only quartic potential. The complete starting potential is
        saved by reset/checkpoint; physics steps preserve that baseline.
        Returns per-agent rewards of shape ``(N,)``.
        """
        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        scale = 2 * env.state.rad
        kinetic_energy = thermal.compute_translational_kinetic_energy_per_particle(
            env.state
        )
        energy_cost = (
            env.env_params["kinetic_energy_coeff"]
            * kinetic_energy
            / env.env_params["kinetic_energy_scale"]
        )
        potential = jnp.exp(-((norm(delta) / scale) ** 4) - energy_cost)
        return potential - env.env_params["prev_potential"]

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiNavigator.truncated")
    def truncated(env: MultiNavigator) -> jax.Array:
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
        return self.state.dim

    @property
    def action_space_shape(self) -> tuple[int]:
        """Original per-agent action shape (useful for reshaping inside the environment)."""
        return (self.state.dim,)

    @property
    def observation_space_size(self) -> int:
        """Flattened observation size per agent. :meth:`observation` returns shape ``(A, observation_space_size)``."""
        return 3 * self.state.dim + self.n_lidar_rays


__all__ = ["MultiNavigator"]
