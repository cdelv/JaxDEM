# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Multi-agent navigation task with collision penalties."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import ShapeDtypeStruct

from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.stats import qmc

from . import Environment
from ...state import State
from ...system import System


@partial(jax.jit, static_argnames=("N",))
@partial(jax.named_call, name="multi_navigator._sample_objectives")
def _sample_objectives(key: ArrayLike, N: int, box: jax.Array) -> jax.Array:
    i = jax.lax.iota(jnp.int32, N)  # 0..N-1
    Lx, Ly = box.astype(jnp.float32)

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
        [dx / 4, dy / 4]
    )
    return base + noise


@Environment.register("multiNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MultiNavigator(Environment):
    """Multi-agent navigation environment with collision penalties."""

    @classmethod
    @partial(jax.named_call, name="MultiNavigator.Create")
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
        system = System.create(state.shape)
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
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.reset")
    def reset(env: "Environment", key: ArrayLike) -> "Environment":
        """
        Initialize the environment with randomly placed particles and velocities.

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
        root = key
        key_box = jax.random.fold_in(root, jnp.uint32(0))
        key_pos = jax.random.fold_in(root, jnp.uint32(1))
        key_objective = jax.random.fold_in(root, jnp.uint32(2))
        key_shuffle = jax.random.fold_in(root, jnp.uint32(3))
        key_vel = jax.random.fold_in(root, jnp.uint32(4))

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
        pos = _sample_objectives(key_pos, int(N), box - 2 * rad) + rad
        objective = _sample_objectives(key_objective, int(N), box - 2 * rad) + rad
        env.env_params["objective"] = jax.random.permutation(key_shuffle, objective)

        vel = jax.random.uniform(
            key_vel, (N, dim), minval=-0.1, maxval=0.1, dtype=float
        )

        rad = rad * jnp.ones(N)
        env.state = State.create(pos=pos, vel=vel, rad=rad)
        env.system = System.create(
            env.state.shape,
            domain_type="reflect",
            domain_kw=dict(box_size=box, anchor=jnp.zeros_like(box)),
        )
        env.env_params["prev_rew"] = jnp.zeros_like(env.state.rad)

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.step")
    def step(env: "Environment", action: jax.Array) -> "Environment":
        """
        Advance the simulation by one step. Actions are interpreted as accelerations.

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
        force = (
            action.reshape(env.max_num_agents, *env.action_space_shape)
            - jnp.sign(env.state.vel) * 0.08
        )
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.state, env.system = env.system.step(env.state, env.system)
        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.observation")
    def observation(env: "Environment") -> jax.Array:
        """
        Returns the observation vector for each agent.

        LiDAR bins store proximity values as ``max(0, R - d_min)``; a value of 0 means
        no detection or that an object lies beyond the LiDAR range. The observation
        concatenates the displacement to the objective, the particle velocity, and the
        LiDAR readings normalized by ``R``.
        """
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
            r = jnp.vecdot(rij, rij)
            r = jnp.sqrt(r)
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

        obs = jnp.concatenate(
            [env.system.domain.displacement(obj, pos, env.system), vel, lidar], axis=-1
        )
        return obs / R

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiNavigator.reward")
    def reward(env: "Environment") -> jax.Array:
        r"""
        Returns a vector of per-agent rewards.

        **Equation**

        Let :math:`\delta_i=\operatorname{displacement}(\mathbf{x}_i,\mathbf{objective})`,
        :math:`d_i=\lVert\delta_i\rVert_2`, and :math:`\mathbf{1}[\cdot]` the indicator.
        With shaping factor :math:`\alpha`, final reward :math:`R_f`, radius :math:`r_i`,
        previous reward :math:`\mathrm{rew}^{\text{prev}}_i`, collision-penalty
        coefficient :math:`C_\mathrm{col}\le 0`, LiDAR range :math:`R`, measured proximities
        :math:`\mathrm{prox}_{i,j}`, and safety factor :math:`\kappa=2.05`:

        .. math::

           \mathrm{rew}^{\text{shape}}_i \;=\;
           \mathrm{rew}^{\text{prev}}_i \;-\; \alpha\, d_i

        Define per-beam “too close” hits using a distance threshold
        :math:`\tau_i = \max(0,\, R - \kappa\, r_i)`:

        .. math::

           \mathrm{hit}_{i,j} \;=\; \mathbf{1}\!\left[\,\mathrm{prox}_{i,j} > \tau_i\,\right],\qquad
           n^{\text{hits}}_i \;=\; \sum_j \mathrm{hit}_{i,j}

        Total reward:

        .. math::

           \mathrm{rew}_i \;=\;
           \mathrm{rew}^{\text{shape}}_i
           \;+\; R_f\,\mathbf{1}[\,d_i < r_i\,]
           \;+\; C_\mathrm{col}\, n^{\text{hits}}_i

        The function updates :math:`\mathrm{rew}^{\text{prev}}_i \leftarrow \mathrm{rew}^{\text{shape}}_i`
        and returns :math:`(\mathrm{rew}_i)_{i=1}^N` reshaped to ``(env.max_num_agents,)``.
        """
        pos = env.state.pos
        objective = env.env_params["objective"]
        delta = env.system.domain.displacement(pos, objective, env.system)
        d = jnp.vecdot(delta, delta)
        d = jnp.sqrt(d)
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
    @partial(jax.named_call, name="MultiNavigator.done")
    def done(env: "Environment") -> jax.Array:
        """
        Returns a boolean indicating whether the environment has ended.
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


__all__ = ["MultiNavigator"]
