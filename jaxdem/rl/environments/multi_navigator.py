# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Multi-agent navigation task with collision penalties."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from . import Environment
from ...state import State
from ...system import System


@Environment.register("multiNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class MultiNavigator(Environment):
    """Multi-agent navigation with LIDAR-style proximity sensing."""

    @classmethod
    def Create(
        cls,
        N: int = 2,
        min_box_size: float = 1.0,
        max_box_size: float = 2.0,
        max_steps: int = 5000,
        final_reward: float = 0.05,
        shaping_factor: float = 1.0,
        collision_penalty: float = -2.0,
        lidar_range: float = 0.35,
        n_lidar_rays: int = 12,
    ) -> "MultiNavigator":
        dim = 2
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(dim)
        n_lidar_rays = int(n_lidar_rays)

        env_params = dict(
            objective=jnp.zeros_like(state.pos),
            min_box_size=jnp.asarray(min_box_size, dtype=float),
            max_box_size=jnp.asarray(max_box_size, dtype=float),
            max_steps=jnp.asarray(max_steps, dtype=int),
            final_reward=jnp.asarray(final_reward, dtype=float),
            collision_penalty=jnp.asarray(collision_penalty, dtype=float),
            shaping_factor=jnp.asarray(shaping_factor, dtype=float),
            prev_rew=jnp.zeros_like(state.rad),
            lidar_range=jnp.asarray(lidar_range, dtype=float),
            n_lidar_rays=n_lidar_rays,
            lidar=jnp.zeros((N, n_lidar_rays), dtype=float),
        )
        action_space_size = dim
        action_space_shape = (dim,)
        observation_space_size = 2 * dim + n_lidar_rays

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            max_num_agents=N,
            action_space_size=action_space_size,
            action_space_shape=action_space_shape,
            observation_space_size=observation_space_size,
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    def reset(env: "Environment", key: ArrayLike) -> "Environment":
        key, key_pos, key_vel, key_box, key_objective = jax.random.split(key, 5)

        N = env.max_num_agents
        dim = env.state.dim
        box = jax.random.uniform(
            key_box,
            (dim,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
            dtype=float,
        )

        rad = 0.05
        min_pos = rad * jnp.ones_like(box)
        pos = jax.random.uniform(
            key_pos,
            (N, dim),
            minval=min_pos,
            maxval=box - min_pos,
            dtype=float,
        )

        objective = jax.random.uniform(
            key_objective,
            (N, dim),
            minval=min_pos,
            maxval=box - min_pos,
            dtype=float,
        )
        env.env_params["objective"] = objective

        vel = jax.random.uniform(
            key_vel, (N, dim), minval=-0.1, maxval=0.1, dtype=float
        )

        rad = rad * jnp.ones(N)
        state = State.create(pos=pos, vel=vel, rad=rad)
        system = System.create(
            env.state.dim,
            domain_type="reflect",
            domain_kw=dict(box_size=box, anchor=jnp.zeros_like(box)),
        )
        env = replace(env, state=state, system=system)
        env.env_params["prev_rew"] = jnp.zeros_like(env.state.rad)

        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env", "action"))
    def step(env: "Environment", action: jax.Array) -> "Environment":
        a = action.reshape(env.max_num_agents, *env.action_space_shape)
        state = replace(env.state, accel=a - jnp.sign(env.state.vel) * 0.08)
        state, system = env.system.step(state, env.system)
        env = replace(env, state=state, system=system)
        return env

    @staticmethod
    @jax.jit
    def observation(env: "Environment") -> jax.Array:
        nbins = env.env_params["lidar"].shape[-1]
        R = env.env_params["lidar_range"]
        indices = jax.lax.iota(int, env.max_num_agents)

        pos = env.state.pos
        vel = env.state.vel
        obj = env.env_params["objective"]

        def lidar_for_i(i: jax.Array) -> jax.Array:
            rij = jax.vmap(
                lambda j: env.system.domain.displacement(pos[i], pos[j], env.system)
            )(indices)
            r = jnp.linalg.norm(rij, axis=-1)
            r = r.at[i].set(jnp.inf)

            theta = jnp.arctan2(rij[..., 1], rij[..., 0])
            bins = jnp.floor((theta + jnp.pi) * (nbins / (2.0 * jnp.pi))).astype(int)

            d_in = jnp.where(r < R, r, jnp.inf)
            d_bins = jnp.full((nbins,), jnp.inf, dtype=pos.dtype).at[bins].min(d_in)

            proximity = jnp.where(
                jnp.isfinite(d_bins), jnp.maximum(0.0, R - d_bins), 0.0
            )
            return proximity

        lidar = jax.vmap(lidar_for_i)(indices)
        env.env_params["lidar"] = lidar

        obs = jnp.concatenate([obj - pos, vel, lidar], axis=-1)
        return obs / R

    @staticmethod
    @jax.jit
    def reward(env: "Environment") -> jax.Array:
        pos = env.state.pos
        objective = env.env_params["objective"]
        delta = env.system.domain.displacement(pos, objective, env.system)
        d = jnp.linalg.norm(delta, axis=-1)
        on_goal = d < env.state.rad
        rew = env.env_params["prev_rew"] - d * env.env_params["shaping_factor"]
        env.env_params["prev_rew"] = rew

        prox = env.env_params["lidar"]
        R = env.env_params["lidar_range"]
        two_r = 2.05 * env.state.rad[:, None]

        closeness_thresh = jnp.maximum(0.0, R - two_r)
        hits = prox > closeness_thresh
        n_hits = hits.sum(axis=-1).astype(rew.dtype)

        reward = (
            rew
            + env.env_params["final_reward"] * on_goal
            + env.env_params["collision_penalty"] * n_hits
        )
        return reward.reshape(env.max_num_agents)

    @staticmethod
    @jax.jit
    def done(env: "Environment") -> jax.Array:
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])


__all__ = ["MultiNavigator"]
