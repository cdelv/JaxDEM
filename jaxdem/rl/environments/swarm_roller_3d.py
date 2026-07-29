# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""3-D swarm rolling agents covering pyramid objectives, with mutual attraction."""

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
from ...utils import cross_lidar_3d, lidar_3d
from ...utils.linalg import unit_and_norm
from . import Environment
from .multi_roller import frictional_wall_force


@jax.jit(static_argnames=("N",))
@partial(jax.named_call, name="swarm_roller_3d._sample_objectives_3d")
def _sample_objectives_3d(
    key: ArrayLike, N: int, box: jax.Array, gap: float, rad: float
) -> jax.Array:
    r"""Sample *N* positions on a jittered X-Y grid at floor level (``z = rad``).

    Centers stay >= ``gap`` apart in the X-Y plane.
    """
    i = jax.lax.iota(int, N)
    Lx, Ly = box[0], box[1]
    nx = jnp.ceil(jnp.sqrt(N * Lx / Ly)).astype(int)
    ny = jnp.ceil(N / nx).astype(int)
    ix, iy = jnp.mod(i, nx), i // nx
    dx, dy = Lx / nx, Ly / ny
    base = jnp.stack([(ix + 0.5) * dx, (iy + 0.5) * dy, jnp.full((N,), rad)], axis=1)
    noise = jax.random.uniform(key, (N, 3), minval=-1.0, maxval=1.0) * jnp.asarray(
        [jnp.maximum(0.0, dx / 2 - gap / 2), jnp.maximum(0.0, dy / 2 - gap / 2), 0.0]
    )
    return base + noise


def _sample_padding_ring_3d(
    key: ArrayLike, N: int, box: float, pad: float, gap: float, rad: float
) -> jax.Array:
    r"""Sample *N* points on jittered grids filling the padding ring around ``box``."""
    if N == 0:
        return jnp.zeros((0, 3))
    t = pad / 2.0
    L = box + pad
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n = N // 4
    n4 = N - 3 * n
    bottom = _sample_objectives_3d(k1, n, jnp.asarray([L, t]), gap, rad) + jnp.asarray(
        [-t, -t, 0.0]
    )
    top = _sample_objectives_3d(k2, n, jnp.asarray([L, t]), gap, rad) + jnp.asarray(
        [-t, box, 0.0]
    )
    left = _sample_objectives_3d(k3, n, jnp.asarray([t, box]), gap, rad) + jnp.asarray(
        [-t, 0.0, 0.0]
    )
    right = _sample_objectives_3d(
        k4, n4, jnp.asarray([t, box]), gap, rad
    ) + jnp.asarray([box, 0.0, 0.0])
    return jnp.concatenate([bottom, top, left, right], axis=0)


def _pyramid_layout(n_obj: int, rad: float) -> jnp.ndarray:
    r"""Build ``n_obj`` sphere positions in a square pyramid, centered at the origin."""
    import math

    if n_obj <= 0:
        return jnp.zeros((0, 3))
    h = 1
    while (h + 1) * (h + 2) * (2 * h + 3) // 6 <= n_obj:
        h += 1
    layer_counts = [(h - k) ** 2 for k in range(h)]
    layer_counts[0] += n_obj - sum(layer_counts)
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
    magnet: jax.Array,
    strength: jax.Array,
    mag_range: jax.Array,
    system: System,
) -> jax.Array:
    r"""Compute pairwise magnetic attraction between particles."""
    N = pos.shape[0]
    rij = system.domain.displacement(pos[:, None, :], pos[None, :, :], system)
    n, r = unit_and_norm(rij)
    pair_mag = magnet[:, None] + magnet[None, :]
    decay = jnp.maximum(0.0, 1.0 - r / mag_range)
    mask = 1.0 - jnp.eye(N)
    F_n_mag = strength * pair_mag * decay * mask
    return jnp.sum(-F_n_mag[..., None] * n, axis=1)


@Environment.register("swarmRoller3D")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SwarmRoller3D(Environment):
    r"""Multi-agent cooperative coverage of 3-D pyramid objectives with attraction.

    Same structure as :class:`SwarmRoller`: rolling-sphere agents with
    translational and angular drag, three LiDAR sensors (walls, objectives,
    peers), and a bin-wise contention-shaped reward. Two differences: the
    objectives form a square pyramid sensed with 3-D LiDAR, and the agents
    attract each other pairwise through a magnetic force.

    ============================  =========================
    Feature                       Size
    ============================  =========================
    Velocity                      ``dim``
    Angular velocity              ``dim``
    Objective LiDAR (normalized)  ``n_az * n_el``
    Wall LiDAR (normalized)       ``n_az * n_el``
    ============================  =========================
    """

    n_lidar_rays: int = jax.tree.static()
    """Number of azimuthal bins for each 3-D LiDAR sensor."""

    n_lidar_elevation: int = jax.tree.static()
    """Number of elevation bins for each 3-D LiDAR sensor."""

    num_objectives: int = jax.tree.static()
    """Number of objectives (pyramid spheres) sampled per environment."""

    @classmethod
    @partial(jax.named_call, name="SwarmRoller3D.Create")
    def Create(
        cls,
        N: int = 5,
        num_objectives: int = 5,
        box_size: float = 5.0,
        box_padding: float = 5.0,
        max_steps: int = 10000,
        friction: float = 0.2,
        near_goal_bonus: float = 1e-2,
        lidar_range: float = 16.0,
        n_lidar_rays: int = 8,
        n_lidar_elevation: int = 8,
        contention_strength: float = 15.0,
        magnet_strength: float = 4.0,
        magnet_range: float = 3.0,
    ) -> SwarmRoller3D:
        r"""Create a 3-D swarm roller environment with pyramid objectives.

        Parameters mirror :meth:`SwarmRoller.Create`, plus ``n_lidar_elevation``
        (3-D LiDAR elevation bins) and ``magnet_strength`` / ``magnet_range``
        for the inter-agent attraction.
        """
        dim = 3
        rad = 1.0
        n_obj = int(num_objectives)
        n_az = int(n_lidar_rays)
        n_el = int(n_lidar_elevation)
        n_lidar = n_az * n_el
        state = State.create(pos=jnp.zeros((int(N), dim)))

        pyr_rel = _pyramid_layout(n_obj, rad)
        pyr_half = jnp.max(jnp.abs(pyr_rel[:, :2])) + rad

        env_params = {
            "objective": jnp.zeros((n_obj, dim)),
            "pyr_rel": pyr_rel,
            "pyr_half": pyr_half,
            "box_size": jnp.asarray(box_size, dtype=float),
            "box_padding": jnp.asarray(box_padding, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "near_goal_bonus": jnp.asarray(near_goal_bonus, dtype=float),
            "lidar_range": jnp.asarray(lidar_range, dtype=float),
            "contention_strength": jnp.asarray(contention_strength, dtype=float),
            "magnet_strength": jnp.asarray(magnet_strength, dtype=float),
            "magnet_range": jnp.asarray(magnet_range, dtype=float),
            "lidar": jnp.zeros((int(N), n_lidar)),
            "lidar_obj": jnp.zeros((int(N), n_lidar)),
            "lidar_obj_prev": jnp.zeros((int(N), n_lidar)),
            "lidar_agt": jnp.zeros((int(N), n_lidar)),
            "lidar_agt_prev": jnp.zeros((int(N), n_lidar)),
        }
        return cls(
            state=state,
            system=System.create(state.shape),
            env_params=env_params,
            n_lidar_rays=n_az,
            n_lidar_elevation=n_el,
            num_objectives=n_obj,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmRoller3D.reset")
    def reset(env: SwarmRoller3D, key: ArrayLike) -> Environment:
        """Initialize with agents in the padding ring and a pyramid of objectives in the box."""
        key_pos, key_pyr = jax.random.split(key)
        N, rad = env.max_num_agents, 1.0
        gap = 2.05 * rad
        box_s = env.env_params["box_size"]
        padding = env.env_params["box_padding"] * rad

        pyr_rel = env.env_params["pyr_rel"]
        pyr_half = env.env_params["pyr_half"]
        pyr_center = jax.random.uniform(
            key_pyr, (2,), minval=pyr_half, maxval=box_s - pyr_half
        )
        env.env_params["objective"] = (
            pyr_rel.at[:, 0].add(pyr_center[0]).at[:, 1].add(pyr_center[1])
        )

        pos = _sample_padding_ring_3d(key_pos, int(N), box_s, padding, gap, rad)
        env.state = State.create(pos=pos, rad=rad * jnp.ones(N), mass=jnp.ones(N))

        matcher = MaterialMatchmaker.create("linear")
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
        box3 = box_s * jnp.ones(3)
        env.system = System.create(
            env.state.shape,
            dt=2e-3,
            domain_type="reflect",
            domain_kw={
                "box_size": box3 + padding,
                "anchor": jnp.asarray(
                    [-padding / 2, -padding / 2, -2 * rad], dtype=float
                ),
            },
            force_manager_kw={
                "gravity": [0.0, 0.0, -1.0],
                "force_functions": (frictional_wall_force,),
            },
            mat_table=mat_table,
            force_model_type="cundallstrack",
        )

        env = SwarmRoller3D._sense(env)
        env.env_params["lidar_obj_prev"] = env.env_params["lidar_obj"]
        env.env_params["lidar_agt_prev"] = env.env_params["lidar_agt"]
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmRoller3D._sense")
    def _sense(env: SwarmRoller3D) -> SwarmRoller3D:
        """Refresh wall, objective, and peer 3-D LiDAR readings."""
        objective = env.env_params["objective"]
        lr = env.env_params["lidar_range"]
        n_az = env.n_lidar_rays
        n_el = env.n_lidar_elevation
        N = env.max_num_agents

        _, _, lidar, _, _ = lidar_3d(
            env.state, env.system, lr, n_az, n_el, sense_edges=True
        )
        env.env_params["lidar"] = lidar
        lidar_obj, _, _ = cross_lidar_3d(
            env.state.pos, objective, env.system, lr, n_az, n_el
        )
        env.env_params["lidar_obj"] = lidar_obj
        _, _, lidar_agt, _, _ = lidar_3d(
            env.state, env.system, lr, n_az, n_el, sense_edges=False
        )
        env.env_params["lidar_agt"] = lidar_agt
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmRoller3D.step")
    def step(env: SwarmRoller3D, action: jax.Array) -> Environment:
        """Advance one step: drag, torque, mutual attraction, then physics and sensing."""
        N = env.max_num_agents
        torque = (
            action.reshape(N, *env.action_space_shape)
            - env.env_params["friction"] * env.state.ang_vel
        )
        force = -env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.system = env.system.force_manager.add_torque(env.state, env.system, torque)

        mag_force = _magnetic_force(
            env.state.pos,
            jnp.ones(N, dtype=float),
            env.env_params["magnet_strength"],
            env.env_params["magnet_range"],
            env.system,
        )
        env.system = env.system.force_manager.add_force(
            env.state, env.system, mag_force
        )

        env.env_params["lidar_obj_prev"] = env.env_params["lidar_obj"]
        env.env_params["lidar_agt_prev"] = env.env_params["lidar_agt"]

        env.state, env.system = env.system.step(env.state, env.system)

        env = SwarmRoller3D._sense(env)
        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmRoller3D.observation")
    def observation(env: SwarmRoller3D) -> jax.Array:
        """Velocity + angular velocity + objective LiDAR + wall LiDAR (normalized), per agent."""
        lr = env.env_params["lidar_range"]
        return jnp.concatenate(
            [
                env.state.vel,
                env.state.ang_vel,
                env.env_params["lidar_obj"] / lr,
                env.env_params["lidar"] / lr,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmRoller3D.reward")
    def reward(env: SwarmRoller3D) -> jax.Array:
        r"""Potential-based shaping with a bin-wise contention penalty.

        Same as :meth:`SwarmRoller.reward`, but the law-of-cosines bin geometry
        uses azimuth alignment (``az = bin // n_elevation``) because the bins
        are the flattened 3-D (azimuth, elevation) grid.
        """
        lr = env.env_params["lidar_range"]
        bonus = env.env_params["near_goal_bonus"]
        P_max = env.env_params["contention_strength"]
        gate = lr / 4.0
        tau = 1.0

        n_az = env.n_lidar_rays
        n_el = env.n_lidar_elevation
        M = n_az * n_el
        az = jnp.arange(M) // n_el
        cos_delta = jnp.cos((az[:, None] - az[None, :]) * (2.0 * jnp.pi / n_az))

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
    @partial(jax.named_call, name="SwarmRoller3D.done")
    def done(env: SwarmRoller3D) -> jax.Array:
        """The episode ends when ``step_count`` exceeds ``max_steps``."""
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """Flattened action size per agent (torque components)."""
        return 3

    @property
    def action_space_shape(self) -> tuple[int]:
        """Original per-agent action shape."""
        return (3,)

    @property
    def observation_space_size(self) -> int:
        """Flattened observation size per agent."""
        return 2 * self.state.dim + 2 * (self.n_lidar_rays * self.n_lidar_elevation)


__all__ = ["SwarmRoller3D"]
