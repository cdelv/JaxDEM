# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Multi-agent 2-D swarm navigation toward shared objectives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from functools import partial
from typing import Tuple, cast

from . import Environment
from ...state import State
from ...system import System
from ...utils import lidar_2d, cross_lidar_2d
from ...materials import MaterialTable, Material
from ...material_matchmakers import MaterialMatchmaker


@partial(jax.jit, static_argnames=("N",))
@partial(jax.named_call, name="swarm_navigator._sample_regular_perimeter")
def _sample_regular_perimeter(
    N: int, box: jax.Array, padding: float, rad: float
) -> jax.Array:
    r"""Sample *N* positions on concentric regular grids along the perimeter."""
    D = 3.0 * rad

    k = jnp.arange(N)

    x_min_k = -padding / 2.0 + D / 2.0 + k * D
    x_max_k = box[0] + padding / 2.0 - D / 2.0 - k * D
    y_min_k = -padding / 2.0 + D / 2.0 + k * D
    y_max_k = box[1] + padding / 2.0 - D / 2.0 - k * D

    W_k = jnp.maximum(0.0, x_max_k - x_min_k)
    H_k = jnp.maximum(0.0, y_max_k - y_min_k)
    P_k = 2.0 * W_k + 2.0 * H_k

    C_k = jnp.floor(jnp.where(P_k > 0, P_k / D, 0)).astype(jnp.int32)
    C_k = C_k.at[-1].set(N)

    cum_C = jnp.cumsum(C_k)
    start_idx = jnp.concatenate([jnp.array([0]), cum_C[:-1]])
    n_k = jnp.clip(N - start_idx, 0, C_k)

    i = jnp.arange(N)

    ring_idx = jnp.sum(i[:, None] >= cum_C[None, :], axis=1)
    ring_idx = jnp.clip(ring_idx, 0, N - 1)

    local_i = i - start_idx[ring_idx]
    local_n = jnp.maximum(1, n_k[ring_idx])
    local_P = P_k[ring_idx]

    s = local_i * (local_P / local_n)

    w = W_k[ring_idx]
    h = H_k[ring_idx]
    x_min = x_min_k[ring_idx]
    x_max = x_max_k[ring_idx]
    y_min = y_min_k[ring_idx]
    y_max = y_max_k[ring_idx]

    x = jnp.where(
        s < w,
        x_min + s,  # Bottom edge
        jnp.where(
            s < w + h,
            x_max,  # Right edge
            jnp.where(
                s < 2.0 * w + h, x_max - (s - (w + h)), x_min  # Top edge  # Left edge
            ),
        ),
    )

    y = jnp.where(
        s < w,
        y_min,  # Bottom edge
        jnp.where(
            s < w + h,
            y_min + (s - w),  # Right edge
            jnp.where(
                s < 2.0 * w + h,
                y_max,  # Top edge
                y_max - (s - (2.0 * w + h)),  # Left edge
            ),
        ),
    )

    return jnp.stack([x, y], axis=1)


@partial(jax.jit, static_argnames=("N",))
@partial(jax.named_call, name="multi_navigator._sample_objectives")
def _sample_objectives(key: ArrayLike, N: int, box: jax.Array, rad: float) -> jax.Array:
    r"""Sample *N* positions on a jittered 2-D grid."""
    i = jax.lax.iota(int, N)
    Lx, Ly = box.astype(float)

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
        [jnp.maximum(0.0, dx / 2 - rad), jnp.maximum(0.0, dy / 2 - rad)]
    )
    return base + noise


def _update_sensors(env: Environment) -> Environment:
    """Internal helper to compute and update all observations in the environment."""
    N = env.max_num_agents
    dim = env.state.dim
    k = cast(int, getattr(env, "k_objectives"))

    # Post-step local LiDAR update (ignoring idx output to save state overhead)
    (
        env.state,
        env.system,
        env.env_params["lidar"],
        _,
        _,
    ) = lidar_2d(
        env.state,
        env.system,
        env.env_params["lidar_range"],
        cast(int, getattr(env, "n_lidar_rays")),
        env.max_num_agents,
        sense_edges=True,
    )

    # Post-step objective LiDAR update
    env.env_params["obj_lidar"], _, _ = cross_lidar_2d(
        env.state.pos,
        env.env_params["objective"],
        env.system,
        env.env_params["lidar_range"],
        cast(int, getattr(env, "n_lidar_rays")),
        env.max_num_agents,
    )

    # 1. Compute pairwise displacements from all agents to all objectives
    pos_a = env.state.pos[:, None, :]  # (N, 1, dim)
    pos_b = env.env_params["objective"][None, :, :]  # (1, N_objs, dim)
    deltas = env.system.domain.displacement(
        pos_a, pos_b, env.system
    )  # (N, N_objs, dim)
    dist = jnp.linalg.norm(deltas, axis=-1)  # (N, N_objs)

    # 2. Extract top `k` closest objectives for each agent
    neg_dists, top_k_idx = jax.lax.top_k(-dist, k)
    top_k_dist = -neg_dists  # (N, k)

    # Gather the exact displacement vectors for the top k
    top_k_deltas = jnp.take_along_axis(
        deltas, top_k_idx[..., None], axis=1
    )  # (N, k, dim)

    # 3. Compute Unit Vectors
    top_k_units = top_k_deltas / jnp.maximum(top_k_dist[..., None], 1e-6)  # (N, k, dim)

    # 4. Compute Clipped Vectors (reusing unit vectors for speed)
    clip_range = env.env_params["lidar_range"]
    clipped_deltas = top_k_units * jnp.minimum(
        top_k_dist[..., None], clip_range
    )  # (N, k, dim)

    # 5. Compute Occupancy Status (by OTHER agents)
    thresh = env.env_params["goal_radius_factor"] * env.state.rad[0]

    dist_T = dist.T  # (N_objs, N_agents)
    n_top = min(2, env.max_num_agents)
    neg_dists_obs, _ = jax.lax.top_k(-dist_T, n_top)

    min_dist = -neg_dists_obs[:, 0]
    second_min_dist = (
        -neg_dists_obs[:, 1] if n_top > 1 else jnp.full_like(min_dist, jnp.inf)
    )

    dist_without_i = jnp.where(
        dist_T == min_dist[:, None], second_min_dist[:, None], min_dist[:, None]
    )  # (N_objs, N_agents)

    is_occupied_by_others = dist_without_i.T < thresh  # (N_agents, N_objs)

    # Look up the occupancy status for the chosen top k objectives
    top_k_occupied = jnp.take_along_axis(
        is_occupied_by_others, top_k_idx, axis=1
    ).astype(
        jnp.float32
    )  # (N, k)

    # Store computed observation parameters in env.env_params
    env.env_params["top_k_units"] = top_k_units.reshape(N, k * dim)
    env.env_params["clipped_deltas"] = clipped_deltas.reshape(N, k * dim)
    env.env_params["top_k_occupied"] = top_k_occupied

    return env


@Environment.register("swarmNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SwarmNavigator(Environment):
    r"""Multi-agent 2-D swarm navigation with cooperative difference rewards.

    Agents track their nearest objectives dynamically. The reward encourages
    unique target acquisition using difference rewards and applies an action penalty.
    """

    n_lidar_rays: int = field(metadata={"static": True})
    """Number of angular bins for agent-to-agent LiDAR sensors."""
    k_objectives: int = field(metadata={"static": True})
    """Number of closest objectives to observe."""

    @classmethod
    @partial(jax.named_call, name="SwarmNavigator.Create")
    def Create(
        cls,
        N: int = 64,
        min_box_size: float = 1.0,
        max_box_size: float = 1.0,
        box_padding: float = 20.0,
        max_steps: int = 5760,
        friction: float = 0.2,
        goal_weight: float = 0.002,
        global_weight: float = 0.0018,
        goal_radius_factor: float = 1.0,
        work_weight: float = 0.002,
        seek_weight: float = 0.55,
        lidar_range: float = 0.4,
        n_lidar_rays: int = 8,
        k_objectives: int = 4,
    ) -> SwarmNavigator:
        r"""Create a swarm navigator environment."""
        dim = 2
        state = State.create(pos=jnp.zeros((N, dim)))
        system = System.create(state.shape)

        env_params = dict(
            objective=jnp.zeros_like(state.pos),
            action=jnp.zeros_like(state.pos),
            min_box_size=jnp.asarray(min_box_size, dtype=float),
            max_box_size=jnp.asarray(max_box_size, dtype=float),
            box_padding=jnp.asarray(box_padding, dtype=float),
            max_steps=jnp.asarray(max_steps, dtype=int),
            friction=jnp.asarray(friction, dtype=float),
            goal_weight=jnp.asarray(goal_weight, dtype=float),
            global_weight=jnp.asarray(global_weight, dtype=float),
            goal_radius_factor=jnp.asarray(goal_radius_factor, dtype=float),
            work_weight=jnp.asarray(work_weight, dtype=float),
            seek_weight=jnp.asarray(seek_weight, dtype=float),
            # Pre-allocated arrays for constructed observations
            top_k_units=jnp.zeros((N, int(k_objectives) * dim), dtype=float),
            clipped_deltas=jnp.zeros((N, int(k_objectives) * dim), dtype=float),
            top_k_occupied=jnp.zeros((N, int(k_objectives)), dtype=float),
            lidar_range=jnp.asarray(lidar_range, dtype=float),
            lidar=jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
            obj_lidar=jnp.zeros((state.N, int(n_lidar_rays)), dtype=float),
        )

        return cls(
            state=state,
            system=system,
            env_params=env_params,
            n_lidar_rays=int(n_lidar_rays),
            k_objectives=int(k_objectives),
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="SwarmNavigator.reset")
    def reset(env: Environment, key: ArrayLike) -> Environment:
        """Reset the environment to a random initial configuration."""
        key_box, key_perimeter, key_inside, key_objective, key_vel = jax.random.split(
            key, 5
        )
        N = env.max_num_agents
        dim = env.state.dim
        rad = 0.05
        box = jax.random.uniform(
            key_box,
            (dim,),
            minval=env.env_params["min_box_size"],
            maxval=env.env_params["max_box_size"],
            dtype=float,
        )
        padding = env.env_params["box_padding"] * rad

        # Split spawns: ~75% perimeter, ~25% inside the box
        n_inside = N // 4
        n_perimeter = N - n_inside

        pos_perimeter = _sample_regular_perimeter(n_perimeter, box, padding, rad)
        pos_inside = _sample_objectives(key_inside, n_inside, box, rad)
        pos = jnp.concatenate([pos_perimeter, pos_inside], axis=0)

        objective = _sample_objectives(key_objective, int(N), box, rad)

        vel = jax.random.uniform(
            key_vel, (N, dim), minval=-0.1, maxval=0.1, dtype=float
        )
        env.state = State.create(pos=pos, vel=vel, rad=rad * jnp.ones(N))

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
            mat_table=mat_table,
        )

        env.env_params["objective"] = objective
        env.env_params["action"] = jnp.zeros_like(env.state.pos)

        # Update all constructed observation sensors
        env = _update_sensors(env)

        return env

    @staticmethod
    @partial(jax.jit, donate_argnames=("env",))
    @partial(jax.named_call, name="SwarmNavigator.step")
    def step(env: Environment, action: jax.Array) -> Environment:
        """Advance the environment by one physics step."""

        # 1. Physics integration
        force = (
            action.reshape(env.max_num_agents, *env.action_space_shape)
            - env.env_params["friction"] * env.state.vel
        )
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.env_params["action"] = action.reshape(
            env.max_num_agents, *env.action_space_shape
        )

        env.state, env.system = env.system.step(env.state, env.system)

        # 2. Process new observations
        env = _update_sensors(env)

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmNavigator.observation")
    def observation(env: Environment) -> jax.Array:
        """Build the per-agent observation vector."""
        obs_elements = [
            env.state.vel,  # (N, dim)
            env.env_params["lidar"],  # (N, n_lidar_rays)
            env.env_params["obj_lidar"],  # (N, n_lidar_rays)
            env.env_params["top_k_units"],  # (N, k * dim)
            env.env_params["clipped_deltas"],  # (N, k * dim)
            env.env_params["top_k_occupied"],  # (N, k)
        ]

        return jnp.concatenate(obs_elements, axis=-1)

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmNavigator.reward")
    def reward(env: Environment) -> jax.Array:
        r"""Compute the per-agent cooperative reward using Difference Rewards."""
        N = env.max_num_agents
        dim = env.state.dim
        k = cast(int, getattr(env, "k_objectives"))
        thresh = env.env_params["goal_radius_factor"] * env.state.rad[0]

        # Extract strictly from the available observation parameters
        clipped_deltas = env.env_params["clipped_deltas"].reshape(N, k, dim)
        top_k_occupied = env.env_params["top_k_occupied"]

        dists = jnp.linalg.norm(clipped_deltas, axis=-1)  # (N, k)
        occ_by_others = top_k_occupied > 0.5  # (N, k)
        self_occ = dists < thresh  # (N, k)

        # Contribution flag: true if agent i uniquely occupies at least one goal
        contrib = self_occ & ~occ_by_others
        # Cap contribution to 1 per agent (an agent can only truly claim 1 goal)
        agent_contrib = jnp.any(contrib, axis=-1).astype(jnp.float32)  # (N,)

        # 1. Individual Goal Reward
        ind_goal_reward = env.env_params["goal_weight"] * agent_contrib  # (N,)

        # ==========================================
        # DIFFERENTIAL REWARD FORMALISM (D_i = G - G_{-i})
        # ==========================================

        # A. Global System Reward (G)
        # The team objective is maximizing total unique occupancy synergy
        total_occ = jnp.sum(agent_contrib)
        G_team = env.env_params["global_weight"] * (total_occ**2)

        # B. Counterfactual System Reward (G_{-i})
        # What would the team synergy be if agent i's contribution was removed?
        total_occ_without_i = total_occ - agent_contrib
        G_team_without_i = env.env_params["global_weight"] * (total_occ_without_i**2)

        # C. Compute Differential Impact (D_i)
        # The agent receives exactly the marginal value they added to the global team objective
        D_team = G_team - G_team_without_i  # (N,)

        # ==========================================
        # INDIVIDUAL PENALTIES
        # ==========================================

        # 2. Local Work Penalty
        work_penalty = env.env_params["work_weight"] * jnp.sum(
            env.env_params["action"] ** 2, axis=-1
        )  # (N,)

        # 3. Seek Penalty
        # If 0 or 1 empty objectives are visible while not on a goal, penalize low velocity to encourage exploration
        is_occupied = self_occ | occ_by_others
        on_goal = jnp.any(self_occ, axis=-1)
        num_empty_seen = jnp.sum(~is_occupied, axis=-1)
        should_seek = (num_empty_seen <= 1) & ~on_goal

        speed = jnp.linalg.norm(env.state.vel, axis=-1)
        seek_penalty = (
            env.env_params["seek_weight"] * should_seek * jnp.exp(-10.0 * speed)
        )

        return ind_goal_reward + D_team - work_penalty - seek_penalty

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="SwarmNavigator.done")
    def done(env: Environment) -> jax.Array:
        """Return ``True`` when the episode has exceeded ``max_steps``."""
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """Number of scalar actions per agent (equal to ``dim``)."""
        return self.state.dim

    @property
    def action_space_shape(self) -> Tuple[int]:
        """Shape of a single agent's action (``(dim,)``)."""
        return (self.state.dim,)

    @property
    def observation_space_size(self) -> int:
        """Dimensionality of a single agent's observation vector."""
        return (
            self.state.dim  # Agent velocity
            + self.n_lidar_rays  # Local LiDAR rays
            + self.n_lidar_rays  # Objective LiDAR rays
            + (self.k_objectives * self.state.dim)  # Unit vectors to top k objectives
            + (
                self.k_objectives * self.state.dim
            )  # Clipped vectors to top k objectives
            + self.k_objectives  # Binary occupancy status of top k objectives
        )


__all__ = ["SwarmNavigator"]
