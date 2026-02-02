# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Environment where multiple agents roll towards targets on a 3D floor."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from functools import partial
from typing import Tuple

import jaxdem as jdem
from . import Environment
from ...state import State
from ...system import System
from ...utils import lidar
from ...utils.linalg import cross, unit
from ...materials import MaterialTable, Material
from ...material_matchmakers import MaterialMatchmaker


@partial(jax.jit, static_argnames=("N",))
@partial(jax.named_call, name="multi_roller._sample_objectives_3d")
def _sample_objectives_3d(
    key: ArrayLike, N: int, box: jax.Array, rad: float
) -> jax.Array:
    """Samples points on the X-Z plane (floor) for 3D positioning."""
    i = jax.lax.iota(int, N)
    # Use X and Z dimensions for the floor grid
    Lx, Lz = box[0], box[2]

    nx = jnp.ceil(jnp.sqrt(N * Lx / Lz)).astype(int)
    nz = jnp.ceil(N / nx).astype(int)

    ix = jnp.mod(i, nx)
    iz = i // nx

    dx = Lx / nx
    dz = Lz / nz

    xs = (ix + 0.5) * dx
    zs = (iz + 0.5) * dz

    # Y is fixed at rad (on the floor)
    ys = jnp.full_like(xs, rad)

    base = jnp.stack([xs, ys, zs], axis=1)

    noise = jax.random.uniform(key, (N, 3), minval=-1.0, maxval=1.0)
    noise_scale = jnp.asarray(
        [
            jnp.maximum(0.0, dx / 2 - rad),
            0.0,  # No noise in Y
            jnp.maximum(0.0, dz / 2 - rad),
        ]
    )

    return base + noise * noise_scale


@partial(jax.named_call, name="multi_roller.frictional_wall_force")
def frictional_wall_force(
    pos: jax.Array, state: State, system: System
) -> Tuple[jax.Array, jax.Array]:
    """Calculates normal and frictional forces for spheres on a y=0 plane."""
    k = 1e5  # Normal stiffness
    mu = 0.4  # Friction coefficient
    n = jnp.array([0.0, 1.0, 0.0])
    p = jnp.array([0.0, 0.0, 0.0])

    # 1. Normal Force Calculation
    dist = jnp.dot(pos - p, n) - state.rad
    penetration = jnp.minimum(0.0, dist)
    force_n = -k * penetration[..., None] * n

    # 2. Velocity at the contact point
    # radius_vector points from center to contact point: -rad * n
    radius_vec = -state.rad[..., None] * n
    v_at_contact = state.vel + cross(state.angVel, radius_vec)

    # Tangential velocity component
    v_n = jnp.sum(v_at_contact * n, axis=-1, keepdims=True) * n
    v_t = v_at_contact - v_n

    # 3. Friction Force (Coulomb approximation)
    f_t_mag = mu * jnp.linalg.norm(force_n, axis=-1, keepdims=True)
    t_dir = unit(v_t)
    force_t = -f_t_mag * t_dir

    # 4. Total Force and Torque
    total_force = force_n + force_t
    total_torque = cross(radius_vec, force_t)

    return total_force, total_torque


@Environment.register("multiRoller")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MultiRoller(Environment):
    """
    Multi-agent 3D rolling environment.

    Agents are spheres that roll on a floor. They are controlled via 3D torque vectors.
    Includes collision handling, LiDAR sensing, and distance-based reward shaping.
    """

    n_lidar_rays: int = field(metadata={"static": True})

    @classmethod
    @partial(jax.named_call, name="MultiRoller.Create")
    def Create(
        cls,
        N: int = 64,
        min_box_size: float = 1.0,
        max_box_size: float = 1.0,
        box_padding: float = 5.0,
        max_steps: int = 5760,
        final_reward: float = 1.0,  # 1.0
        shaping_factor: float = 0.005,
        prev_shaping_factor: float = 0.0,
        global_shaping_factor: float = 0.0,
        collision_penalty: float = -0.005,
        goal_threshold: float = 2 / 3,
        lidar_range: float = 0.45,
        n_lidar_rays: int = 16,
    ) -> "MultiRoller":
        dim = 3
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = dict(
            objective=jnp.zeros_like(state.pos),
            min_box_size=jnp.asarray(min_box_size, dtype=float),
            max_box_size=jnp.asarray(max_box_size, dtype=float),
            box_padding=jnp.asarray(box_padding, dtype=float),
            max_steps=jnp.asarray(max_steps, dtype=int),
            final_reward=jnp.asarray(final_reward, dtype=float),
            collision_penalty=jnp.asarray(collision_penalty, dtype=float),
            shaping_factor=jnp.asarray(shaping_factor, dtype=float),
            prev_shaping_factor=jnp.asarray(prev_shaping_factor, dtype=float),
            goal_threshold=jnp.asarray(goal_threshold, dtype=float),
            prev_rew=jnp.zeros_like(state.rad),
            lidar_range=jnp.asarray(lidar_range, dtype=float),
            lidar=jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
            goal_scale=jnp.asarray(1.0, dtype=float),
        )

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            n_lidar_rays=int(n_lidar_rays),
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="MultiRoller.reset")
    def reset(env: "Environment", key: ArrayLike) -> "Environment":
        """
        Initialize the environment with randomly placed particles.

        Parameters
        ----------
        env: Environment
            Current environment instance.

        key : jax.random.PRNGKey
            JAX random number generator key.

        Returns
        -------
        Environment
            Freshly initialized environment.
        """
        key_box, key_pos, key_objective, key_shuffle, key_vel = jax.random.split(key, 5)
        N = env.max_num_agents
        dim = 3
        box = jax.random.uniform(
            key_box,
            (dim,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
            dtype=float,
        )

        rad_val = 0.05
        padding = env.env_params["box_padding"] * rad_val
        pos = (
            _sample_objectives_3d(key_pos, int(N), box + padding, rad_val) - padding / 2
        )
        pos = pos.at[:, 1].set(rad_val)
        objective = _sample_objectives_3d(key_objective, int(N), box, rad_val)
        objective = objective.at[:, 1].set(rad_val)
        env.env_params["goal_scale"] = jnp.max(box)
        perm = jax.random.permutation(key_shuffle, jnp.arange(N))
        env.env_params["objective"] = objective[perm]

        rads = rad_val * jnp.ones(N)
        env.state = State.create(pos=pos, rad=rads)

        matcher = MaterialMatchmaker.create("harmonic")
        mat_table = MaterialTable.from_materials(
            [Material.create("elastic", density=0.27, young=6e3, poisson=0.3)],
            matcher=matcher,
        )

        env.system = System.create(
            env.state.shape,
            dt=0.004,
            domain_type="reflect",
            domain_kw=dict(
                box_size=box + padding,
                anchor=jnp.zeros_like(box) - padding / 2,
            ),
            force_manager_kw=dict(
                gravity=[0.0, -10.0, 0.0],
                force_functions=(frictional_wall_force,),
            ),
            mat_table=mat_table,
        )

        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        env.env_params["prev_rew"] = jnp.linalg.norm(delta, axis=-1)
        env.env_params["lidar"] = lidar(env)

        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="MultiRoller.step")
    def step(env: "Environment", action: jax.Array) -> "Environment":
        torque = action.reshape(env.max_num_agents, 3)
        force_drag = -0.08 * env.state.vel
        torque_drag = -0.05 * env.state.angVel
        env.system = env.system.force_manager.add_force(
            env.state, env.system, force_drag
        )
        env.system = env.system.force_manager.add_torque(
            env.state, env.system, torque + torque_drag
        )
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        env.env_params["prev_rew"] = jnp.linalg.norm(delta, axis=-1)
        env.state, env.system = env.system.step(env.state, env.system)
        env.env_params["lidar"] = lidar(env)
        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiRoller.observation")
    def observation(env: "Environment") -> jax.Array:
        disp = (
            env.system.domain.displacement(
                env.env_params["objective"], env.state.pos, env.system
            )
            / env.env_params["goal_scale"]
        )
        return jnp.concatenate(
            [
                disp,
                env.state.vel,
                env.state.angVel,
                env.env_params["lidar"] / env.env_params["lidar_range"],
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiRoller.reward")
    def reward(env: "Environment") -> jax.Array:
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        d = jnp.linalg.norm(delta, axis=-1)
        rew_shape = (
            env.env_params["prev_shaping_factor"] * env.env_params["prev_rew"]
            - env.env_params["shaping_factor"] * d
        )
        closeness_thresh = jnp.maximum(
            0.0, env.env_params["lidar_range"] - 2.0 * env.state.rad[:, None]
        )
        n_hits = (env.env_params["lidar"] > closeness_thresh).sum(axis=-1).astype(float)
        on_goal = d < env.env_params["goal_threshold"] * env.state.rad
        reward = (
            rew_shape
            + env.env_params["final_reward"] * on_goal
            + env.env_params["collision_penalty"] * n_hits
        )
        return reward.reshape(env.max_num_agents)

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="MultiRoller.done")
    def done(env: "Environment") -> jax.Array:
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        return 3  # 3D Torque

    @property
    def action_space_shape(self) -> Tuple[int]:
        return (3,)

    @property
    def observation_space_size(self) -> int:
        # Disp(3) + Vel(3) + AngVel(3) + Lidar(rays)
        return 9 + self.n_lidar_rays


__all__ = ["MultiRoller"]
