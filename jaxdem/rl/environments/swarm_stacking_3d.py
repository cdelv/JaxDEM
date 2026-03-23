# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

"""Multi-agent 3-D swarm stacking environment with periodic boundaries."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from functools import partial
from typing import Any

from . import Environment
from ...state import State
from ...system import System
from ...utils import lidar_3d
from ...materials import MaterialTable, Material
from ...material_matchmakers import MaterialMatchmaker
from .multi_roller import frictional_wall_force, _sample_objectives_3d
from ...utils.linalg import dot, norm2, unit_and_norm


@Environment.register("swarmStacking3D")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SwarmStacking3D(Environment):
    r"""Multi-agent 3-D stacking environment with periodic boundaries.

    Agents must stack on top of one another to reach as high as possible.

    **Reward**

    .. math::

        R_i = w_{climb} (0.8 \cdot z_i + 0.2 \cdot \bar{z}_t)
              + w_{cohesion} \sum \text{lidar}
              - w_w\,\|\tau_i\|^2
              - w_{\mathrm{vel}}\,\|v_i\|^2
              - \bar{r}_i

    where :math:`\bar{z}_t` is the average height of the swarm.

    Boundary Conditions:
    - Periodic in X and Y.
    - Frictional floor at Z=0.
    - Effectively unbounded Z (large box size).
    """

    n_lidar_rays: int = field(metadata={"static": True})
    """Number of azimuthal bins for the 3-D LiDAR sensor."""
    n_lidar_elevation: int = field(metadata={"static": True})
    """Number of elevation bins for the 3-D LiDAR sensor."""

    @classmethod
    @partial(jax.named_call, name="SwarmStacking3D.Create")
    def Create(
        cls,
        N: int = 16,
        min_box_size: float = 0.5,
        max_box_size: float = 0.5,
        box_padding: float = 0.0,
        max_steps: int = 5760,
        friction: float = 0.2,
        ang_damping: float = 0.07,
        climb_weight: float = 20.0,
        cohesion_weight: float = 0.05,
        work_weight: float = 0.0,
        velocity_weight: float = 2.0,
        alpha_r_bar: float = 0.07,
        lidar_range: float = 0.5,
        n_lidar_rays: int = 8,
        n_lidar_elevation: int = 8,
        magnet_strength: float = 4e1,
        magnet_range: float = 0.12,
    ) -> SwarmStacking3D:
        """Create a swarm stacking 3-D environment."""
        dim = 3
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        n_az, n_el = int(n_lidar_rays), int(n_lidar_elevation)
        ray_dirs = _build_ray_dirs(n_az, n_el)

        env_params = {
            "min_box_size": jnp.asarray(min_box_size, dtype=float),
            "max_box_size": jnp.asarray(max_box_size, dtype=float),
            "box_padding": jnp.asarray(box_padding, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "ang_damping": jnp.asarray(ang_damping, dtype=float),
            "climb_weight": jnp.asarray(climb_weight, dtype=float),
            "cohesion_weight": jnp.asarray(cohesion_weight, dtype=float),
            "work_weight": jnp.asarray(work_weight, dtype=float),
            "velocity_weight": jnp.asarray(velocity_weight, dtype=float),
            "alpha_r_bar": jnp.asarray(alpha_r_bar, dtype=float),
            "lidar_range": jnp.asarray(lidar_range, dtype=float),
            "magnet_strength": jnp.asarray(magnet_strength, dtype=float),
            "magnet_range": jnp.asarray(magnet_range, dtype=float),
            "ray_dirs": ray_dirs,
            "magnet": jnp.zeros(N),
            "current_reward": jnp.zeros(N, dtype=float),
            "r_bar": jnp.zeros(N, dtype=float),
            "lidar": jnp.zeros((N, n_az * n_el), dtype=float),
            "lidar_vr": jnp.zeros((N, n_az * n_el), dtype=float),
            "lidar_idx": jnp.zeros((N, n_az * n_el), dtype=int),
        }

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            n_lidar_rays=n_az,
            n_lidar_elevation=n_el,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmStacking3D.reset")
    def reset(env: "SwarmStacking3D", key: ArrayLike) -> Environment:
        key_box, key_pos, key_vel = jax.random.split(key, 3)
        N = env.max_num_agents
        rad_val = 0.05
        padding = env.env_params["box_padding"] * rad_val

        box_xy = jax.random.uniform(
            key_box,
            (2,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
        )
        effective_box_xy = box_xy + padding

        # Periodic Z with large period to simulate unbounded Z
        box_size = jnp.array([effective_box_xy[0], effective_box_xy[1], 1.0e6])
        anchor = jnp.array([-padding / 2, -padding / 2, -10.0])

        pos = _sample_objectives_3d(key_pos, int(N), box_size, rad_val) - padding / 2
        pos = pos.at[:, 2].set(rad_val * 1.01)

        vel = jax.random.uniform(key_vel, (N, 3), minval=-0.05, maxval=0.05)
        vel = vel.at[:, 2].set(0.0)

        env.state = State.create(pos=pos, vel=vel, rad=rad_val * jnp.ones(N))

        matcher = MaterialMatchmaker.create("linear")
        mat_table = MaterialTable.from_materials(
            [
                Material.create(
                    "elasticfrict",
                    density=0.27,
                    young=6e3,
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
            domain_kw={"box_size": box_size, "anchor": anchor},
            force_model_type="cundallstrack",
            force_manager_kw={
                "gravity": [0.0, 0.0, -8.0],
                "force_functions": (frictional_wall_force,),
            },
            mat_table=mat_table,
        )

        env.env_params["magnet"] = jnp.zeros(N)
        env.env_params["r_bar"] = jnp.zeros(N, dtype=float)
        env.env_params["current_reward"] = jnp.zeros(N, dtype=float)

        env.env_params = _update_lidar_vr(
            env.state,
            env.system,
            env.env_params,
            env.env_params["ray_dirs"],
            env.n_lidar_rays,
            env.n_lidar_elevation,
            N,
        )

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmStacking3D.step")
    def step(env: "SwarmStacking3D", action: jax.Array) -> Environment:
        N = env.max_num_agents
        act = action.reshape(N, 4)
        torque_act = act[:, :3]
        magnet = (act[:, 3] > 0.0).astype(jnp.float32)

        env.env_params["magnet"] = magnet

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

        env.state, env.system = env.system.step(env.state, env.system)

        env.env_params = _update_lidar_vr(
            env.state,
            env.system,
            env.env_params,
            env.env_params["ray_dirs"],
            env.n_lidar_rays,
            env.n_lidar_elevation,
            N,
        )

        # Absolute height reward
        z = env.state.pos[:, 2]
        mean_z = jnp.mean(z)
        climb_reward = env.env_params["climb_weight"] * (0.3 * z + 0.7 * mean_z)

        # Cohesion reward (sum of lidar readings)
        lidar_sum = jnp.sum(env.env_params["lidar"], axis=-1)
        cohesion_reward = env.env_params["cohesion_weight"] * lidar_sum

        work = norm2(torque_act)
        vel_penalty = norm2(env.state.vel)

        R_raw = (
            climb_reward
            + cohesion_reward
            - env.env_params["work_weight"] * work
            - env.env_params["velocity_weight"] * vel_penalty
        )

        r_bar = env.env_params["r_bar"]
        env.env_params["r_bar"] = r_bar + env.env_params["alpha_r_bar"] * (
            R_raw - r_bar
        )
        env.env_params["current_reward"] = R_raw  # - r_bar

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmStacking3D.observation")
    def observation(env: "SwarmStacking3D") -> jax.Array:
        return jnp.concatenate(
            [
                env.state.vel,
                env.state.ang_vel,
                env.env_params["magnet"][:, None],
                env.env_params["lidar"],
                env.env_params["lidar_vr"],
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmStacking3D.reward")
    def reward(env: "SwarmStacking3D") -> jax.Array:
        return env.env_params["current_reward"]

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="SwarmStacking3D.done")
    def done(env: "SwarmStacking3D") -> jax.Array:
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        return 4

    @property
    def action_space_shape(self) -> tuple[int]:
        return (4,)

    @property
    def observation_space_size(self) -> int:
        # vel(3) + ang_vel(3) + magnet(1) + lidar + lidar_vr
        n_lidar = self.n_lidar_rays * self.n_lidar_elevation
        return 7 + 2 * n_lidar


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
    """Update agent-to-agent LiDAR and radial velocity."""
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
    return env_params


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
