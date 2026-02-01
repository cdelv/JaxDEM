# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Environment where a single agent rolls towards a target on the floor."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jaxdem as jdem
from . import Environment
from ...state import State
from ...system import System
from ...utils.linalg import cross, unit


def frictional_wall_force(
    pos: jax.Array, state: jdem.State, system: jdem.System
) -> Tuple[jax.Array, jax.Array]:
    """Calculates normal and frictional forces for a sphere on a y=0 plane."""
    k = 1e5  # Normal stiffness
    mu = 0.5  # Friction coefficient
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


@Environment.register("singleRoller3D")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SingleRoller3D(Environment):
    """Single-agent navigation where the agent rolls on a plane using torque control."""

    @classmethod
    @partial(jax.named_call, name="SingleRoller3D.Create")
    def Create(
        cls,
        min_box_size: float = 1.0,
        max_box_size: float = 1.0,
        max_steps: int = 6000,
        final_reward: float = 2.0,
        shaping_factor: float = 1.0,
        prev_shaping_factor: float = 1.0,
        goal_threshold: float = 2 / 3,
    ) -> "SingleRoller3D":
        dim = 3
        N = 1
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = dict(
            objective=jnp.zeros_like(state.pos),
            min_box_size=jnp.asarray(min_box_size, dtype=float),
            max_box_size=jnp.asarray(max_box_size, dtype=float),
            max_steps=jnp.asarray(max_steps, dtype=int),
            final_reward=jnp.asarray(final_reward, dtype=float),
            shaping_factor=jnp.asarray(shaping_factor, dtype=float),
            prev_shaping_factor=jnp.asarray(prev_shaping_factor, dtype=float),
            goal_threshold=jnp.asarray(goal_threshold, dtype=float),
            prev_rew=jnp.zeros_like(state.rad),
        )

        return cls(
            state=state,
            system=system,
            env_params=env_params,
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="SingleRoller3D.reset")
    def reset(env: "Environment", key: ArrayLike) -> "Environment":
        root = key
        key_box = jax.random.fold_in(root, jnp.uint32(0))
        key_pos = jax.random.fold_in(root, jnp.uint32(1))
        key_objective = jax.random.fold_in(root, jnp.uint32(2))
        key_vel = jax.random.fold_in(root, jnp.uint32(4))

        N = env.max_num_agents
        dim = env.state.dim
        rad_val = 0.05

        box = jax.random.uniform(
            key_box,
            (dim,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
            dtype=float,
        )

        # Ensure agent and objective are placed on the floor (y = rad)
        min_pos = rad_val * jnp.ones_like(box)
        pos = jax.random.uniform(
            key_pos,
            (N, dim),
            minval=min_pos,
            maxval=box - min_pos,
            dtype=float,
        )
        pos = pos.at[:, 1].set(rad_val)

        objective = jax.random.uniform(
            key_objective,
            (N, dim),
            minval=min_pos,
            maxval=box - min_pos,
            dtype=float,
        )
        objective = objective.at[:, 1].set(rad_val)
        env.env_params["objective"] = objective

        vel = jax.random.uniform(
            key_vel, (N, dim), minval=-0.05, maxval=0.05, dtype=float
        )
        rad = rad_val * jnp.ones(N)

        env.state = State.create(pos=pos, vel=vel, rad=rad)

        # Initialize system with gravity and the frictional wall force function
        env.system = System.create(
            env.state.shape,
            domain_type="reflect",
            domain_kw=dict(box_size=box, anchor=[0, -4 * rad_val, 0]),
            force_manager_kw=dict(
                gravity=[0.0, -10.0, 0.0],
                force_functions=(frictional_wall_force,),
            ),
        )

        env.env_params["prev_rew"] = jnp.zeros_like(env.state.rad)
        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="SingleRoller3D.step")
    def step(env: "Environment", action: jax.Array) -> "Environment":
        torque = action.reshape(env.max_num_agents, 3) - 0.05 * env.state.angVel
        force = -0.05 * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.system = env.system.force_manager.add_torque(env.state, env.system, torque)

        # Update reward tracking before physics step
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        env.env_params["prev_rew"] = jnp.linalg.norm(delta, axis=-1)

        # Physics integration
        env.state, env.system = env.system.step(env.state, env.system)

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SingleRoller3D.observation")
    def observation(env: "Environment") -> jax.Array:
        # Include angular velocity in observations for better control of rolling
        return jnp.concatenate(
            [
                env.system.domain.displacement(
                    env.state.pos, env.env_params["objective"], env.system
                ),
                env.state.vel,
                env.state.angVel,
            ],
            axis=-1,
        ) / jnp.max(env.system.domain.box_size)

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SingleRoller3D.reward")
    def reward(env: "Environment") -> jax.Array:
        delta = env.system.domain.displacement(
            env.state.pos, env.env_params["objective"], env.system
        )
        d = jnp.linalg.norm(delta, axis=-1)
        on_goal = d < env.env_params["goal_threshold"] * env.state.rad

        rew = (
            env.env_params["prev_shaping_factor"] * env.env_params["prev_rew"]
            - env.env_params["shaping_factor"] * d
        )
        reward = rew + env.env_params["final_reward"] * on_goal
        return reward.reshape(env.max_num_agents)

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="SingleRoller3D.done")
    def done(env: "Environment") -> jax.Array:
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        return 3  # 3D Torque vector

    @property
    def action_space_shape(self) -> Tuple[int]:
        return (3,)

    @property
    def observation_space_size(self) -> int:
        return 3 * self.state.dim  # Disp(3) + Vel(3) + AngVel(3)


__all__ = ["SingleRoller3D"]
