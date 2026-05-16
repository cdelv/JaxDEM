# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

"""Multi-agent 3-D swarm rolling environment with magnetic interaction and pyramid objectives."""

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
from ...utils.linalg import norm, unit_and_norm
from . import Environment
from .multi_roller import _sample_objectives_3d, frictional_wall_force


@Environment.register("swarmRoller3D")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SwarmRoller3D(Environment):
    r"""Multi-agent 3-D rolling environment with magnetic interaction and pyramid objectives."""

    n_lidar_rays: int = jax.tree.static()  # type: ignore[attr-defined]
    """Number of azimuthal bins for the 3-D LiDAR sensor."""
    n_lidar_elevation: int = jax.tree.static()  # type: ignore[attr-defined]
    """Number of elevation bins for the 3-D LiDAR sensor."""

    @classmethod
    @partial(jax.named_call, name="SwarmRoller3D.Create")
    def Create(
        cls,
        N: int = 5,
        min_box_size: float = 10.0,
        max_box_size: float = 10.0,
        box_padding: float = 0.0,
        max_steps: int = 100000,
        friction: float = 0.2,
        lidar_range: float = 10.0,
        n_lidar_rays: int = 8,
        n_lidar_elevation: int = 8,
        magnet_strength: float = 4.0,
        magnet_range: float = 3.0,
        ke_weight: float = 0.1,
        coop_weight: float = 0.2,
        near_goal_bonus: float = 0.1,
    ) -> SwarmRoller3D:
        """Create a swarm roller 3-D environment."""
        dim = 3
        n_obj = int(N)
        rad_val = 1.0
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        pyr_rel = _pyramid_layout(n_obj, rad_val)
        pyr_half = jnp.max(jnp.abs(pyr_rel[:, :2])) + rad_val
        n_az = int(n_lidar_rays)
        n_el = int(n_lidar_elevation)
        n_lidar = n_az * n_el

        env_params = {
            "objective": jnp.zeros((n_obj, dim)),
            "pyr_rel": pyr_rel,
            "pyr_half": pyr_half,
            "curr_dist": jnp.zeros(N, dtype=float),
            "prev_shaping_sum": jnp.zeros(N, dtype=float),
            "curr_shaping_sum": jnp.zeros(N, dtype=float),
            "curr_ke": jnp.zeros(N, dtype=float),
            "prev_ke": jnp.zeros(N, dtype=float),
            "min_box_size": jnp.asarray(min_box_size, dtype=float),
            "max_box_size": jnp.asarray(max_box_size, dtype=float),
            "box_padding": jnp.asarray(box_padding, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "magnet_strength": jnp.asarray(magnet_strength, dtype=float),
            "magnet_range": jnp.asarray(magnet_range, dtype=float),
            "lidar_range": jnp.asarray(lidar_range, dtype=float),
            "ke_weight": jnp.asarray(ke_weight, dtype=float),
            "coop_weight": jnp.asarray(coop_weight, dtype=float),
            "near_goal_bonus": jnp.asarray(near_goal_bonus, dtype=float),
            "lidar": jnp.zeros((N, n_lidar), dtype=float),
            "lidar_obj": jnp.zeros((N, n_lidar), dtype=float),
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
    @partial(jax.named_call, name="SwarmRoller3D.reset")
    def reset(env: "SwarmRoller3D", key: ArrayLike) -> Environment:
        """Reset the environment to a random initial configuration."""
        key_box, key_pos, key_pyr = jax.random.split(key, 3)
        N = env.max_num_agents
        dim = 3
        rad_val = 1.0
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

        env.state = State.create(
            pos=pos,
            rad=rad_val * jnp.ones(N),
            mass=jnp.ones(N),
        )

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
                "anchor": jnp.asarray(
                    [-padding / 2, -padding / 2, -2 * rad_val],
                    dtype=float,
                ),
            },
            force_model_type="cundallstrack",
            force_manager_kw={
                "gravity": [0.0, 0.0, -1.0],
                "force_functions": (frictional_wall_force,),
            },
            mat_table=mat_table,
        )
        env.env_params["objective"] = objective
        env.env_params = _update_lidar(
            env.state,
            env.system,
            env.env_params,
            env.n_lidar_rays,
            env.n_lidar_elevation,
            N,
        )

        deltas = env.system.domain.displacement(
            env.state.pos[:, None, :], objective[None, :, :], env.system
        )
        dist_all = norm(deltas)
        env.env_params["curr_dist"] = jnp.min(dist_all, axis=1)

        import jaxdem.utils.thermal as thermal

        ke_t = thermal.compute_translational_kinetic_energy_per_particle(env.state)
        env.env_params["curr_ke"] = ke_t
        env.env_params["prev_ke"] = ke_t

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
    @partial(jax.named_call, name="SwarmRoller3D.step")
    def step(env: "SwarmRoller3D", action: jax.Array) -> Environment:
        """Advance the environment by one physics step."""
        N = env.max_num_agents
        torque_act = action.reshape(N, 3)
        torque = torque_act - env.env_params["friction"] * env.state.ang_vel
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
        env.env_params["prev_shaping_sum"] = env.env_params["curr_shaping_sum"]
        env.env_params["prev_ke"] = env.env_params["curr_ke"]

        env.state, env.system = env.system.step(env.state, env.system)
        objective = env.env_params["objective"]
        env.env_params = _update_lidar(
            env.state,
            env.system,
            env.env_params,
            env.n_lidar_rays,
            env.n_lidar_elevation,
            N,
        )

        deltas = env.system.domain.displacement(
            env.state.pos[:, None, :], objective[None, :, :], env.system
        )
        dist_all = norm(deltas)
        env.env_params["curr_dist"] = jnp.min(dist_all, axis=1)

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
    @partial(jax.named_call, name="SwarmRoller3D.observation")
    def observation(env: "SwarmRoller3D") -> jax.Array:
        """Build per-agent observations."""
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
    @partial(jax.named_call, name="SwarmRoller3D.reward")
    def reward(env: "SwarmRoller3D") -> jax.Array:
        """Return per-agent rewards."""
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
    @partial(jax.named_call, name="SwarmRoller3D.done")
    def done(env: "SwarmRoller3D") -> jax.Array:
        """Return ``True`` when the episode has exceeded ``max_steps``."""
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """Number of scalar actions per agent (3-D torque)."""
        return 3

    @property
    def action_space_shape(self) -> tuple[int]:
        """Shape of a single agent's action (``(3,)``)."""
        return (3,)

    @property
    def observation_space_size(self) -> int:
        """Dimensionality of a single agent's observation vector."""
        n_lidar = self.n_lidar_rays * self.n_lidar_elevation
        return 3 + 2 * n_lidar


def _update_lidar(
    state: State,
    system: System,
    env_params: dict[str, jax.Array],
    n_az: int,
    n_el: int,
    N: int,
) -> dict[str, jax.Array]:
    """Update agent and objective LiDAR caches."""
    _, _, env_params["lidar"], _, _ = lidar_3d(
        state,
        system,
        env_params["lidar_range"],
        n_az,
        n_el,
        N,
        sense_edges=True,
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


def _pyramid_layout(n_obj: int, rad: float) -> jnp.ndarray:
    r"""Build ``n_obj`` sphere positions in a square pyramid, centred at the origin."""
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
    r = r[..., 0]
    pair_mag = magnet[:, None] + magnet[None, :]
    decay = jnp.maximum(0.0, 1.0 - r / mag_range)
    mask = 1.0 - jnp.eye(N)
    F_n_mag = strength * pair_mag * decay * mask
    return jnp.sum(-F_n_mag[..., None] * n, axis=1)


__all__ = ["SwarmRoller3D"]
