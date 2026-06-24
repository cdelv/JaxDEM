# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where multiple agents navigate towards assigned targets."""

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
from ...utils import lidar_2d
from ...utils.linalg import norm
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

    Each agent controls a force vector that is applied directly to a sphere
    inside a reflective box. Viscous drag ``-friction * vel`` is added each
    step. Objectives are sampled and assigned one-to-one via a random
    permutation.

    The reward uses exponential potential-based shaping:

    .. math::

        R_i = (e^{-2d_i} - e^{-2d_i^{\mathrm{prev}}})
              - w_{\mathrm{ke}}(K_i - K_i^{\mathrm{prev}})
              + w_{\mathrm{coop}} \cdot \frac{1}{N}\sum_j
                (e^{-2d_j} - e^{-2d_j^{\mathrm{prev}}})
              + w_{\mathrm{near}}\,\mathbf{1}[d_i \le r_i]

    where :math:`d_i` is the distance to the assigned objective and
    :math:`K_i` is the translational kinetic energy of agent :math:`i`.

    Notes
    -----
    The observation vector per agent is:

    ============================  =================
    Feature                       Size
    ============================  =================
    Unit direction to objective   ``dim``
    Clamped displacement          ``dim``
    Velocity                      ``dim``
    LiDAR proximity (normalised)  ``n_lidar_rays``
    ============================  =================

    If one wants some realistic parameters for training, ``skip_frames = 50``
    will give a response rate of 200 Hz, meaning that ``num_steps_epoch = 100``
    gives a horizon of 0.5 seconds.
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
        max_steps: int = 10000 * 10,
        friction: float = 0.2,
        ke_weight: float = 0.1,
        coop_weight: float = 0.2,
        near_goal_bonus: float = 0.1,
        lidar_range: float = 6.0,
        n_lidar_rays: int = 16,
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
        ke_weight : float
            Weight for the differential kinetic energy penalty.
        coop_weight : float
            Weight for the shared team-progress bonus.
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
        MultiNavigator
            A freshly constructed environment (call :meth:`reset` before
            use).

        """
        dim = 2
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape, rotation_integrator_type=None)

        env_params = {
            "objective": jnp.zeros_like(state.pos),
            "permutation": jnp.arange(N, dtype=int),
            "prev_dist": jnp.zeros_like(state.rad),
            "curr_dist": jnp.zeros_like(state.rad),
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
    def reset(env: "MultiNavigator", key: ArrayLike) -> Environment:
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
        key_box, key_pos, key_objective, key_shuffle = jax.random.split(key, 4)
        N = env.max_num_agents
        dim = env.state.dim
        n_rays = env.n_lidar_rays
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
        dist = norm(delta)
        env.env_params["delta"] = delta
        env.env_params["prev_dist"] = dist
        env.env_params["curr_dist"] = dist

        ke_t = thermal.compute_translational_kinetic_energy_per_particle(env.state)
        env.env_params["curr_ke"] = ke_t
        env.env_params["prev_ke"] = ke_t

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
    @partial(jax.named_call, name="MultiNavigator.step")
    def step(env: "MultiNavigator", action: jax.Array) -> Environment:
        """Advance one step. Actions are forces; simple drag is applied (-friction * vel).

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
        force = reshaped_action - env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)

        env.env_params["prev_dist"] = env.env_params["curr_dist"]
        env.env_params["prev_ke"] = env.env_params["curr_ke"]
        env.state, env.system = env.system.step(env.state, env.system)

        delta = env.system.domain.displacement(
            env.state.pos_c, env.env_params["objective"], env.system
        )
        env.env_params["delta"] = delta
        env.env_params["curr_dist"] = norm(delta)

        _, _, lidar, _, _ = lidar_2d(
            env.state,
            env.system,
            env.env_params["lidar_range"],
            n_rays,
            sense_edges=True,
        )
        env.env_params["lidar"] = lidar

        env.env_params["curr_ke"] = (
            thermal.compute_translational_kinetic_energy_per_particle(env.state)
        )

        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiNavigator.observation")
    def observation(env: "MultiNavigator") -> jax.Array:
        """Build per-agent observations.

        Contents per agent
        ------------------
        - Unit vector to objective (shape (dim,))  --> Direction
        - Clamped delta to objective (shape (dim,)) --> Local precision
        - Velocity (shape (dim,))
        - LiDAR proximity, normalized by ``lidar_range`` (shape (n_lidar_rays,))

        Returns
        -------
        jax.Array
            Array of shape ``(N, 3 * dim + n_lidar_rays)``

        """
        delta = env.env_params["delta"]
        direction = (
            delta
            / jnp.where(
                env.env_params["curr_dist"] > 0, env.env_params["curr_dist"], 1.0
            )[:, None]
        )
        return jnp.concatenate(
            [
                direction,
                jnp.clip(delta, -3.0, 3.0),
                env.state.vel,
                env.env_params["lidar"] / env.env_params["lidar_range"],
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiNavigator.reward")
    def reward(env: "MultiNavigator") -> jax.Array:
        r"""Returns a vector of per-agent rewards.

        .. math::

           \mathrm{rew}_t = (e^{-2 \cdot d_t} - e^{-2 \cdot d_t^{\mathrm{prev}}})
           - w_{\text{ke}} (K_t - K_{t-1})
           + w_{\text{coop}} \cdot \mathrm{mean}(e^{-2 \cdot d_t} - e^{-2 \cdot d_t^{\mathrm{prev}}})
           + w_{\text{near}} \cdot \mathbf{1}[d_t \le r]

        where :math:`d_t` is the distance to the objective at step :math:`t`,
        :math:`K_t` is the kinetic energy at step :math:`t`,
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
        shaping_reward = jnp.exp(-2 * env.env_params["curr_dist"]) - jnp.exp(
            -2 * env.env_params["prev_dist"]
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
    @jax.jit(inline=True)
    @partial(jax.named_call, name="MultiNavigator.done")
    def done(env: "MultiNavigator") -> jax.Array:
        """Returns a boolean indicating whether the environment has ended.
        The episode terminates when the maximum number of steps is reached.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            Boolean array indicating whether the episode has ended.

        """
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

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
