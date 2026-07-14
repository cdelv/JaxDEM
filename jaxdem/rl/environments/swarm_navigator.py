# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where multiple agents cooperatively cover a set of objectives."""

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
from ...utils import cross_lidar_2d, lidar_2d
from . import Environment


@jax.jit(static_argnames=("N",))
@partial(jax.named_call, name="swarm_navigator._sample_objectives")
def _sample_objectives(key: ArrayLike, N: int, box: jax.Array, gap: float) -> jax.Array:
    r"""Sample *N* positions on a jittered 2-D grid, centres kept >= ``gap`` apart."""
    i = jax.lax.iota(int, N)
    Lx, Ly = box.astype(float)
    nx = jnp.ceil(jnp.sqrt(N * Lx / Ly)).astype(int)
    ny = jnp.ceil(N / nx).astype(int)
    ix, iy = jnp.mod(i, nx), i // nx
    dx, dy = Lx / nx, Ly / ny
    base = jnp.stack([(ix + 0.5) * dx, (iy + 0.5) * dy], axis=1)
    # Jitter is capped so each centre stays >= gap/2 inside its cell, hence
    # adjacent centres are >= gap apart (no overlap when gap >= 2*rad).
    noise = jax.random.uniform(key, (N, 2), minval=-1.0, maxval=1.0) * jnp.asarray(
        [jnp.maximum(0.0, dx / 2 - gap / 2), jnp.maximum(0.0, dy / 2 - gap / 2)]
    )
    return base + noise


def _sample_padding_ring(
    key: ArrayLike, N: int, box: float, pad: float, gap: float
) -> jax.Array:
    r"""Sample *N* points on jittered grids filling the padding ring around ``box``.

    Four rectangular strips (bottom/top/left/right), each of thickness ``pad/2``,
    are filled by :func:`_sample_objectives` with keep-out ``gap`` so all centres
    are >= gap apart.
    """
    if N == 0:
        return jnp.zeros((0, 2))
    t = pad / 2.0
    L = box + pad
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n = N // 4
    n4 = N - 3 * n
    bottom = _sample_objectives(k1, n, jnp.asarray([L, t]), gap) + jnp.asarray([-t, -t])
    top = _sample_objectives(k2, n, jnp.asarray([L, t]), gap) + jnp.asarray([-t, box])
    left = _sample_objectives(k3, n, jnp.asarray([t, box]), gap) + jnp.asarray(
        [-t, 0.0]
    )
    right = _sample_objectives(k4, n4, jnp.asarray([t, box]), gap) + jnp.asarray(
        [box, 0.0]
    )
    return jnp.concatenate([bottom, top, left, right], axis=0)


@Environment.register("swarmNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SwarmNavigator(Environment):
    r"""Multi-agent cooperative objective coverage with local sensing.

    Each agent controls a force vector applied to a sphere in a reflective box,
    with viscous drag ``-friction * vel`` added each step. Objectives are
    sampled on a jittered grid inside the box; agents spawn in the padding ring
    around it. Three LiDAR sensors are refreshed each step — walls, objectives,
    and peers (other agents) — but only the objective and wall sensors appear
    in the observation; the peer sensor drives the contention penalty in the
    reward. ``lidar_obj_prev`` and ``lidar_agt_prev`` hold the previous step's
    objective and peer readings so the reward can difference them.

    Notes
    -----
    The observation vector per agent is:

    ============================  =================
    Feature                       Size
    ============================  =================
    Velocity                      ``dim``
    Objective LiDAR (normalised)  ``n_lidar_rays``
    Wall LiDAR (normalised)       ``n_lidar_rays``
    ============================  =================
    """

    n_lidar_rays: int = jax.tree.static()
    """Number of angular bins for each LiDAR sensor."""

    num_objectives: int = jax.tree.static()
    """Number of objectives sampled per environment."""

    @classmethod
    @partial(jax.named_call, name="SwarmNavigator.Create")
    def Create(
        cls,
        N: int = 64,
        num_objectives: int = 64,
        box_size: float = 20.0,
        box_padding: float = 10.0,
        max_steps: int = 10000,
        friction: float = 0.2,
        near_goal_bonus: float = 1e-2,
        lidar_range: float = 16.0,
        n_lidar_rays: int = 12,
        contention_strength: float = 15.0,
    ) -> SwarmNavigator:
        r"""Create a swarm navigator environment.

        Parameters
        ----------
        N : int
            Number of agents.
        num_objectives : int
            Number of objectives sampled per environment.
        box_size : float
            Side length of the square domain that holds the objectives.
        box_padding : float
            Thickness of the agent spawn ring around the box (in multiples of
            the particle radius).
        max_steps : int
            Episode length in physics steps.
        friction : float
            Viscous drag coefficient applied as ``-friction * vel``.
        near_goal_bonus : float
            Weight :math:`b` of the near-goal indicator :math:`\mathbf{1}[d \le r]`.
        lidar_range : float
            Maximum detection range :math:`L` for the LiDAR sensors.
        n_lidar_rays : int
            Number of angular LiDAR bins spanning :math:`[-\pi, \pi)`.
        contention_strength : float
            Maximum penalty :math:`P_{\max}` subtracted from an objective's
            LiDAR proximity when a peer sits on it; the bin-wise penalty ramps
            linearly from :math:`P_{\max}` (peer on the objective) to 0 (peer
            at :math:`L/4`) and is zero beyond.

        Returns
        -------
        SwarmNavigator
            A freshly constructed environment (call :meth:`reset` before use).
        """
        dim = 2
        n_obj = int(num_objectives)
        state = State.create(pos=jnp.zeros((int(N), dim)))
        env_params = {
            "objective": jnp.zeros((n_obj, dim)),
            "box_size": jnp.asarray(box_size, dtype=float),
            "box_padding": jnp.asarray(box_padding, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "near_goal_bonus": jnp.asarray(near_goal_bonus, dtype=float),
            "lidar_range": jnp.asarray(lidar_range, dtype=float),
            "contention_strength": jnp.asarray(contention_strength, dtype=float),
            "lidar": jnp.zeros((int(N), int(n_lidar_rays))),
            "lidar_obj": jnp.zeros((int(N), int(n_lidar_rays))),
            "lidar_obj_prev": jnp.zeros((int(N), int(n_lidar_rays))),
            "lidar_agt": jnp.zeros((int(N), int(n_lidar_rays))),
            "lidar_agt_prev": jnp.zeros((int(N), int(n_lidar_rays))),
        }
        return cls(
            state=state,
            system=System.create(state.shape, rotation_integrator_type=None),
            env_params=env_params,
            n_lidar_rays=int(n_lidar_rays),
            num_objectives=n_obj,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmNavigator.reset")
    def reset(env: SwarmNavigator, key: ArrayLike) -> Environment:
        """Initialise the environment with random agents (padding) and objectives (box)."""
        key_pos, key_obj = jax.random.split(key)
        N, rad = env.max_num_agents, 1.0
        gap = 2.05 * rad
        box_s = env.env_params["box_size"]
        box = box_s * jnp.ones(env.state.dim)
        padding = env.env_params["box_padding"] * rad

        env.env_params["objective"] = _sample_objectives(
            key_obj, env.num_objectives, box, gap
        )
        pos = _sample_padding_ring(key_pos, int(N), box_s, padding, gap)
        env.state = State.create(pos=pos, rad=rad * jnp.ones(N))

        matcher = MaterialMatchmaker.create("linear")
        mat_table = MaterialTable.from_materials(
            [Material.create("elastic", density=1.0 / jnp.pi, young=2e5, poisson=0.3)],
            matcher=matcher,
        )
        env.system = System.create(
            env.state.shape,
            dt=2e-3,
            rotation_integrator_type=None,
            domain_type="reflectsphere",
            domain_kw={
                "box_size": box + padding,
                "anchor": -padding / 2 * jnp.ones(env.state.dim),
            },
            mat_table=mat_table,
        )

        env = SwarmNavigator._sense(env)
        env.env_params["lidar_obj_prev"] = env.env_params["lidar_obj"]
        env.env_params["lidar_agt_prev"] = env.env_params["lidar_agt"]
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmNavigator._sense")
    def _sense(env: SwarmNavigator) -> SwarmNavigator:
        """Refresh wall, objective, and peer LiDAR readings."""
        objective = env.env_params["objective"]
        lr = env.env_params["lidar_range"]

        _, _, lidar, _, _ = lidar_2d(
            env.state, env.system, lr, env.n_lidar_rays, sense_edges=True
        )
        env.env_params["lidar"] = lidar
        lidar_obj, _, _ = cross_lidar_2d(
            env.state.pos, objective, env.system, lr, env.n_lidar_rays
        )
        env.env_params["lidar_obj"] = lidar_obj
        _, _, lidar_agt, _, _ = lidar_2d(
            env.state, env.system, lr, env.n_lidar_rays, sense_edges=False
        )
        env.env_params["lidar_agt"] = lidar_agt

        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmNavigator.step")
    def step(env: SwarmNavigator, action: jax.Array) -> Environment:
        """Advance one step. Actions are forces; drag ``-friction * vel`` is added."""
        N = env.max_num_agents
        force = (
            action.reshape(N, *env.action_space_shape)
            - env.env_params["friction"] * env.state.vel
        )
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.env_params["lidar_obj_prev"] = env.env_params["lidar_obj"]
        env.env_params["lidar_agt_prev"] = env.env_params["lidar_agt"]

        env.state, env.system = env.system.step(env.state, env.system)

        env = SwarmNavigator._sense(env)
        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmNavigator.observation")
    def observation(env: SwarmNavigator) -> jax.Array:
        """Velocity + objective LiDAR + wall LiDAR (all normalised), per agent."""
        lr = env.env_params["lidar_range"]
        return jnp.concatenate(
            [
                env.state.vel,
                env.env_params["lidar_obj"] / lr,
                env.env_params["lidar"] / lr,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmNavigator.reward")
    def reward(env: SwarmNavigator) -> jax.Array:
        r"""Potential-based shaping with a bin-wise contention penalty.

        For each objective LiDAR bin, the nearest agent (over all agent LiDAR
        bins, distance recovered with the law of cosines) subtracts from the
        objective's apparent proximity when it lies within ``lr/4`` of it
        (exponential decay, already negligible by ``lr/4``)::

            d_eff = d_obj + P_max * exp(-d_peer / tau),  tau = 1.0

        where ``d_peer`` is :math:`\min_a \sqrt{d_{obj}^2 + d_{agt,a}^2 - 2
        d_{obj} d_{agt,a} \cos(\Delta\theta)}``. Empty bins read at ``lr`` (max
        range); the resulting long-range inaccuracy is negligible since far
        objectives barely contribute.

        Per-step reward::

            R = near_goal_bonus * 1[d_min <= r] + 10 * (phi_t - phi_prev)

        where ``d_min`` is the closest objective distance and ``10`` is the
        shaping scale.
        """
        lr = env.env_params["lidar_range"]
        bonus = env.env_params["near_goal_bonus"]
        P_max = env.env_params["contention_strength"]
        gate = lr / 4.0
        tau = 1.0

        n = env.n_lidar_rays
        idx = jnp.arange(n)
        cos_delta = jnp.cos((idx[:, None] - idx[None, :]) * (2.0 * jnp.pi / n))

        def phi(lidar_obj: jax.Array, lidar_agt: jax.Array) -> jax.Array:
            d_obj = lr - lidar_obj
            d_agt = lr - lidar_agt
            D = jnp.sqrt(
                d_obj[:, :, None] ** 2
                + d_agt[:, None, :] ** 2
                - 2.0 * d_obj[:, :, None] * d_agt[:, None, :] * cos_delta
            )
            peer_dist = D.min(axis=-1)
            penalty = jnp.where(
                peer_dist < gate, P_max * jnp.exp(-peer_dist / tau), 0.0
            )
            d_eff = d_obj + penalty
            return jnp.exp(-2.0 * d_eff).sum(axis=-1)

        curr = phi(env.env_params["lidar_obj"], env.env_params["lidar_agt"])
        prev = phi(env.env_params["lidar_obj_prev"], env.env_params["lidar_agt_prev"])

        at_goal = (lr - env.env_params["lidar_obj"]).min(axis=-1) < env.state.rad
        return at_goal * bonus + 10 * (curr - prev)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmNavigator.done")
    def done(env: SwarmNavigator) -> jax.Array:
        """Episode terminates when ``max_steps`` is reached."""
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """Flattened action size per agent."""
        return self.state.dim

    @property
    def action_space_shape(self) -> tuple[int]:
        """Original per-agent action shape."""
        return (self.state.dim,)

    @property
    def observation_space_size(self) -> int:
        """Flattened observation size per agent."""
        return self.state.dim + 2 * self.n_lidar_rays


__all__ = ["SwarmNavigator"]
