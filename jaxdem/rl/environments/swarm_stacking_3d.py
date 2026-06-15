# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

"""Multi-agent 3-D swarm stacking environment with reflective boundaries."""

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
from ...utils import lidar_3d
from ...utils.linalg import unit_and_norm
from . import Environment
from .multi_roller import _sample_objectives_3d, frictional_wall_force


@Environment.register("swarmStacking3D")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SwarmStacking3D(Environment):
    r"""Multi-agent 3-D stacking environment with reflective boundaries.

    There are no separate objective markers: the agents themselves are the
    stacking targets (magnetic attraction pulls them together), so the reward
    shaping is computed from the agent LiDAR.
    """

    n_lidar_rays: int = jax.tree.static()
    """Number of azimuthal bins for the 3-D LiDAR sensor."""
    n_lidar_elevation: int = jax.tree.static()
    """Number of elevation bins for the 3-D LiDAR sensor."""

    @classmethod
    @partial(jax.named_call, name="SwarmStacking3D.Create")
    def Create(
        cls,
        N: int = 16,
        min_box_size: float = 20.0,
        max_box_size: float = 20.0,
        box_padding: float = 20.0,
        max_steps: int = 5760,
        friction: float = 0.2,
        lidar_range: float = 10.0,
        n_lidar_rays: int = 8,
        n_lidar_elevation: int = 8,
        magnet_strength: float = 4e1,
        magnet_range: float = 2.4,
        ke_weight: float = 0.1,
        coop_weight: float = 0.2,
        near_goal_bonus: float = 0.1,
    ) -> SwarmStacking3D:
        """Create a swarm stacking 3-D environment."""
        dim = 3
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)
        n_lidar = int(n_lidar_rays) * int(n_lidar_elevation)

        env_params = {
            "min_box_size": jnp.asarray(min_box_size, dtype=float),
            "max_box_size": jnp.asarray(max_box_size, dtype=float),
            "box_padding": jnp.asarray(box_padding, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "lidar_range": jnp.asarray(lidar_range, dtype=float),
            "magnet_strength": jnp.asarray(magnet_strength, dtype=float),
            "magnet_range": jnp.asarray(magnet_range, dtype=float),
            "ke_weight": jnp.asarray(ke_weight, dtype=float),
            "coop_weight": jnp.asarray(coop_weight, dtype=float),
            "near_goal_bonus": jnp.asarray(near_goal_bonus, dtype=float),
            "curr_dist": jnp.zeros(N, dtype=float),
            "prev_shaping_sum": jnp.zeros(N, dtype=float),
            "curr_shaping_sum": jnp.zeros(N, dtype=float),
            "curr_ke": jnp.zeros(N, dtype=float),
            "prev_ke": jnp.zeros(N, dtype=float),
            "lidar": jnp.zeros((N, n_lidar), dtype=float),
        }

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            n_lidar_rays=int(n_lidar_rays),
            n_lidar_elevation=int(n_lidar_elevation),
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmStacking3D.reset")
    def reset(env: "SwarmStacking3D", key: ArrayLike) -> Environment:
        key_box, key_pos, key_vel = jax.random.split(key, 3)
        N = env.max_num_agents
        rad_val = 1.0
        padding = env.env_params["box_padding"] * rad_val

        box_xy = jax.random.uniform(
            key_box,
            (2,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
        )
        effective_box_xy = box_xy + padding
        box_size = jnp.array([effective_box_xy[0], effective_box_xy[1], 1.0e6])
        anchor = jnp.array([-padding / 2, -padding / 2, -10.0])

        pos = _sample_objectives_3d(key_pos, int(N), box_size, rad_val) - padding / 2
        pos = pos.at[:, 2].set(rad_val)

        vel = jax.random.uniform(key_vel, (N, 3), minval=-1.0, maxval=1.0)
        vel = vel.at[:, 2].set(0.0)

        env.state = State.create(
            pos=pos,
            vel=vel,
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
            domain_kw={"box_size": box_size, "anchor": anchor},
            force_model_type="cundallstrack",
            force_manager_kw={
                "gravity": [0.0, 0.0, -1.0],
                "force_functions": (frictional_wall_force,),
            },
            mat_table=mat_table,
        )

        env.env_params = _update_lidar(
            env.state,
            env.system,
            env.env_params,
            env.n_lidar_rays,
            env.n_lidar_elevation,
            N,
        )

        import jaxdem.utils.thermal as thermal

        ke_t = thermal.compute_translational_kinetic_energy_per_particle(env.state)
        env.env_params["curr_ke"] = ke_t
        env.env_params["prev_ke"] = ke_t

        obj_dist = env.env_params["lidar_range"] - env.env_params["lidar"]
        obj_detected = env.env_params["lidar"] > 0
        shaping_sum = jnp.sum(
            jnp.where(obj_detected, jnp.exp(-4 * obj_dist), 0.0), axis=1
        )
        env.env_params["prev_shaping_sum"] = shaping_sum
        env.env_params["curr_shaping_sum"] = shaping_sum
        env.env_params["curr_dist"] = jnp.min(
            jnp.where(obj_detected, obj_dist, jnp.inf), axis=1
        )

        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmStacking3D.step")
    def step(env: "SwarmStacking3D", action: jax.Array) -> Environment:
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

        env.env_params = _update_lidar(
            env.state,
            env.system,
            env.env_params,
            env.n_lidar_rays,
            env.n_lidar_elevation,
            N,
        )

        obj_dist = env.env_params["lidar_range"] - env.env_params["lidar"]
        obj_detected = env.env_params["lidar"] > 0
        env.env_params["curr_shaping_sum"] = jnp.sum(
            jnp.where(obj_detected, jnp.exp(-4 * obj_dist), 0.0), axis=1
        )
        env.env_params["curr_dist"] = jnp.min(
            jnp.where(obj_detected, obj_dist, jnp.inf), axis=1
        )

        import jaxdem.utils.thermal as thermal

        env.env_params["curr_ke"] = (
            thermal.compute_translational_kinetic_energy_per_particle(env.state)
        )
        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmStacking3D.observation")
    def observation(env: "SwarmStacking3D") -> jax.Array:
        return jnp.concatenate(
            [
                env.state.vel,
                env.env_params["lidar"] / env.env_params["lidar_range"],
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmStacking3D.reward")
    def reward(env: "SwarmStacking3D") -> jax.Array:
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
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmStacking3D.done")
    def done(env: "SwarmStacking3D") -> jax.Array:
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        return 3

    @property
    def action_space_shape(self) -> tuple[int]:
        return (3,)

    @property
    def observation_space_size(self) -> int:
        n_lidar = self.n_lidar_rays * self.n_lidar_elevation
        return 3 + n_lidar


def _update_lidar(
    state: State,
    system: System,
    env_params: dict[str, jax.Array],
    n_az: int,
    n_el: int,
    N: int,
) -> dict[str, jax.Array]:
    """Update agent LiDAR caches."""
    _, _, env_params["lidar"], _, _ = lidar_3d(
        state,
        system,
        env_params["lidar_range"],
        n_az,
        n_el,
        N,
        sense_edges=True,
    )
    return env_params


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


__all__ = ["SwarmStacking3D"]
