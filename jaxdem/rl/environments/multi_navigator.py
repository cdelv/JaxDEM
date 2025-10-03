# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Multi-agent navigation task with collision penalties."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import ShapeDtypeStruct

from dataclasses import dataclass, replace
from functools import partial

import numpy as np
from scipy.stats import qmc

from . import Environment
from ...state import State
from ...system import System


def PoissonDisk(
    N: int,
    dim: int,
    rad: float,
    l_bounds: jax.Array,
    u_bounds: jax.Array,
    key: ArrayLike,
):
    numpy_seed = int(jax.random.randint(key, (), 0, jnp.iinfo(jnp.int32).max))

    sampler = qmc.PoissonDisk(
        d=int(dim),
        radius=2 * float(rad),
        l_bounds=np.asarray(l_bounds, dtype=float) + float(rad),
        u_bounds=np.asarray(u_bounds, dtype=float) - float(rad),
        seed=int(numpy_seed),
        ncandidates=200,
    )

    pts = jnp.asarray(sampler.random(N), dtype=float)
    m = int(pts.shape[0])
    if m != N:
        raise RuntimeError(
            "Could not place requested number of points without overlap: "
            f"requested N={N}, placed {m}. Try reducing the radius, increasing the box, or decreasing N."
        )

    return pts


@partial(jax.jit, static_argnums=(1, 2))
def _sample_objectives(key, N: int, dim: int, box, rad):
    """Greedy Poisson-disk-like sampler with: pairwise >= 2*rad and wall clearance >= 1*rad. N and dim are static under jit."""
    # fixed shapes only
    low = jnp.ones_like(box) * rad  # (dim,)
    high = box - low  # (dim,)

    K = int(32 * N)  # Python int, static shape
    key, kperm, kcand = jax.random.split(key, 3)
    cands = jax.random.uniform(kcand, (K, dim), minval=low, maxval=high)
    cands = cands[jax.random.permutation(kperm, K)]
    min_sep = 2.0 * rad
    idxs = jnp.arange(N)  # (N,)

    def body(carry, x):
        pts, cnt = carry  # pts:(N,dim), cnt:int32
        valid = idxs < cnt  # (N,)
        dists = jnp.linalg.norm(pts - x, axis=-1)
        dists = jnp.where(valid, dists, jnp.inf)
        dmin = dists.min()
        ok_pair = (cnt == 0) | (dmin >= min_sep)
        ok_slot = cnt < N
        ok = ok_pair & ok_slot
        pts = jax.lax.cond(ok, lambda P: P.at[cnt].set(x), lambda P: P, pts)
        cnt = cnt + jnp.int32(ok)
        return (pts, cnt), None

    init_pts = jnp.zeros((N, dim), dtype=box.dtype)
    (pts, cnt), _ = jax.lax.scan(body, (init_pts, jnp.int32(0)), cands)

    # fill remaining rows with the last accepted or center
    last_idx = jnp.maximum(cnt - 1, 0)
    fallback = jnp.where(cnt > 0, pts[last_idx], (low + high) * 0.5)
    filled = jnp.where((idxs[:, None] < cnt), pts, jnp.broadcast_to(fallback, (N, dim)))
    return filled


@Environment.register("multiNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class MultiNavigator(Environment):
    """Multi-agent navigation environment with collision penalties."""

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
    @partial(jax.jit, donate_argnames=("env",))
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
        # result_spec = ShapeDtypeStruct((N, dim), env.state.pos.dtype)
        # env.env_params["objective"] = jax.pure_callback(
        #     PoissonDisk,
        #     result_spec,
        #     N,
        #     dim,
        #     rad,
        #     jnp.zeros_like(box),
        #     box,
        #     key_objective,
        #     vmap_method="sequential",
        # )

        # pos = jax.pure_callback(
        #     PoissonDisk,
        #     result_spec,
        #     N,
        #     dim,
        #     rad,
        #     jnp.zeros_like(box),
        #     box,
        #     key_pos,
        #     vmap_method="sequential",
        # )
        pos = _sample_objectives(key_pos, int(N), int(dim), box, float(rad))
        objective = _sample_objectives(key_objective, int(N), int(dim), box, float(rad))
        env.env_params["objective"] = objective

        vel = jax.random.uniform(
            key_vel, (N, dim), minval=-0.1, maxval=0.1, dtype=float
        )

        rad = rad * jnp.ones(N)
        state = State.create(pos=pos, vel=vel, rad=rad)
        system = System.create(
            state.shape,
            domain_type="reflect",
            domain_kw=dict(box_size=box, anchor=jnp.zeros_like(box)),
        )
        env = replace(env, state=state, system=system)
        env.env_params["prev_rew"] = jnp.zeros_like(env.state.rad)

        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env", "action"))
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
        force = action.reshape(env.max_num_agents, *env.action_space_shape)
        force -= jnp.sign(env.state.vel) * 0.08
        system = env.system.force_manager.add_force(env.state, env.system, force)
        state, system = env.system.step(env.state, system)
        env = replace(env, state=state, system=system)
        return env

    @staticmethod
    @jax.jit
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
