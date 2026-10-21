"""Isolated local-sensing ablations inspired by vacancy-aware swarm control.

These environments intentionally live outside the production environment.  They
test an adaptation of the paper, not a reproduction of its controller.  In
particular, ``exact_local`` grants richer local sensing than the baseline:
every goal and peer within range is visible, before angular-bin reduction.
All training variants remain decentralized at execution time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from jaxdem.rl.environments import SwarmNavigator
from jaxdem.rl.environments._swarm import _peer_geometry, bin_directions
from jaxdem.utils.environment import _bin_azimuth, _bin_min_dist_and_idx, _bin_spherical


VARIANTS: dict[str, dict[str, str]] = {
    "baseline": {},
    "simple": {"reward_mode": "rational"},
    "simple_reward": {"reward_mode": "rational"},
    "compact": {"observation_mode": "compact"},
    "combined": {"reward_mode": "rational", "observation_mode": "compact"},
    "exact_local": {
        "reward_mode": "rational",
        "observation_mode": "exact_local",
    },
    "exact_compact": {"observation_mode": "exact_local"},
    "exact_combined": {
        "reward_mode": "rational",
        "observation_mode": "exact_local",
    },
    "vacancy_lidar": {"observation_mode": "vacant_lidar"},
    "vacancy_combined": {
        "reward_mode": "rational",
        "observation_mode": "vacant_lidar",
    },
    "paper_compact": {"observation_mode": "paper"},
    "paper_combined": {"reward_mode": "rational", "observation_mode": "paper"},
}


def scan_goal_scores(
    objectives: jax.Array,
    peers: jax.Array,
    peer_ids: jax.Array,
    env: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return raw distances, one-scale shared kernel scores, and own scores.

    Kernel ``K(d)=1/(1+(d/(1.5 radius))**2)`` supplies both the goal value
    and the soft peer claim.  Walls, empty bins, and self returns make no claim.
    This uses exactly the baseline's angularly quantized scan information.
    """
    distance, separation_sq, valid_peer = _peer_geometry(
        objectives,
        peers,
        peer_ids,
        env,
        env.n_lidar_elevation if env.state.dim == 3 else None,
    )
    sigma_sq = (1.5 * env.state.rad[:, None]) ** 2
    own = jnp.where(objectives > 0, 1 / (1 + distance**2 / sigma_sq), 0)
    claims = jnp.where(
        valid_peer[:, None, :],
        1 / (1 + separation_sq / sigma_sq[:, :, None]),
        0,
    ).sum(axis=-1)
    return distance, own / (1 + claims), own


def compact_scan_features(
    objectives: jax.Array,
    peers: jax.Array,
    peer_ids: jax.Array,
    env: Any,
) -> jax.Array:
    """Weighted local displacement, vacancy mass, and own-goal proximity.

    The displacement is a kernel mean-shift step, divided by sensor range.
    A separate mass feature distinguishes balanced directions from empty scans.
    The own score is independent of peer claims so settling remains observable.
    These are approximate features of nearest-per-bin returns: a vacant goal
    behind an occupied goal in the same bin remains invisible.
    """
    distance, weights, own = scan_goal_scores(objectives, peers, peer_ids, env)
    directions = bin_directions(
        env.n_lidar_rays,
        env.n_lidar_elevation if env.state.dim == 3 else None,
    )
    displacement = distance[:, :, None] * directions[None, :, :]
    mass = weights.sum(axis=-1, keepdims=True)
    drift = (weights[:, :, None] * displacement).sum(axis=1) / jnp.maximum(
        mass, jnp.finfo(weights.dtype).tiny
    )
    return jnp.concatenate(
        [
            drift / env.env_params["lidar_range"],
            -jnp.expm1(-mass),
            own.max(axis=-1, keepdims=True),
        ],
        axis=-1,
    )


def exact_local_features(env: Any) -> jax.Array:
    """Compute the same compact features from all truly local detections.

    Visibility is measured from the observing agent, with strict distance <
    range.  Claims use only other agents visible to that observer.  Unseen
    goals and peers cannot influence an observer's features.  This bypasses
    angular quantization and nearest-return occlusion, so it must be reported
    as a richer sensor, rather than a benefit of compression alone.
    """
    if env.num_objectives == 0:
        return jnp.zeros((env.max_num_agents, env.state.dim + 2))
    displacement = env.env_params["objective"][None, :, :] - env.state.pos[:, None, :]
    distance_sq = jnp.sum(displacement**2, axis=-1)
    lr = env.env_params["lidar_range"]
    sigma_sq = (1.5 * env.state.rad[:, None]) ** 2
    all_goal_kernel = 1 / (1 + distance_sq / sigma_sq)
    own = jnp.where(distance_sq < lr**2, all_goal_kernel, 0)
    peer_displacement = env.state.pos[None, :, :] - env.state.pos[:, None, :]
    peer_distance_sq = jnp.sum(peer_displacement**2, axis=-1)
    visible_peers = (peer_distance_sq < lr**2) & ~jnp.eye(
        env.max_num_agents, dtype=bool
    )
    # For observer i, sum only visible peer j's goal-k kernel.  This matrix
    # product avoids materializing an (observer, goal, peer) tensor.
    claims = visible_peers.astype(own.dtype) @ all_goal_kernel
    weights = own / (1 + claims)
    mass = weights.sum(axis=-1, keepdims=True)
    drift = (weights[:, :, None] * displacement).sum(axis=1) / jnp.maximum(
        mass, jnp.finfo(weights.dtype).tiny
    )
    return jnp.concatenate(
        [drift / lr, -jnp.expm1(-mass), own.max(axis=-1, keepdims=True)],
        axis=-1,
    )


def vacant_objective_lidar(env: Any) -> jax.Array:
    """Remove locally occupied targets *before* nearest-per-bin reduction.

    A visible peer occupies a goal when its center is within its own radius.
    Goals already covered by the observer are preserved.  Unlike the baseline
    mask, this exposes farther vacant goals behind an occupied goal in a bin.
    It requires the same richer all-local sensing as ``exact_local``.
    """
    bins = env.n_lidar_rays * (
        env.n_lidar_elevation if env.state.dim == 3 else 1
    )
    if env.num_objectives == 0:
        return jnp.zeros((env.max_num_agents, bins))
    displacement = env.env_params["objective"][None, :, :] - env.state.pos[:, None, :]
    distance_sq = jnp.sum(displacement**2, axis=-1)
    lr = env.env_params["lidar_range"]
    peer_distance_sq = jnp.sum(
        (env.state.pos[None, :, :] - env.state.pos[:, None, :]) ** 2, axis=-1
    )
    visible_peers = (peer_distance_sq < lr**2) & ~jnp.eye(
        env.max_num_agents, dtype=bool
    )
    peer_occupies_goal = distance_sq <= env.state.rad[:, None] ** 2
    occupied = (
        visible_peers.astype(jnp.int32) @ peer_occupies_goal.astype(jnp.int32)
    ) > 0
    keep = (distance_sq < lr**2) & (peer_occupies_goal | ~occupied)
    if env.state.dim == 3:
        bin_indices = _bin_spherical(
            displacement, env.n_lidar_rays, env.n_lidar_elevation
        )
    else:
        bin_indices = _bin_azimuth(displacement, env.n_lidar_rays)
    nearest, _ = _bin_min_dist_and_idx(
        jnp.where(keep, jnp.sqrt(distance_sq), jnp.inf), bin_indices, bins
    )
    return jnp.maximum(0, lr - nearest)


def paper_vacancy_features(env: Any) -> jax.Array:
    """Paper-style raised-cosine mean shift over genuinely vacant local goals.

    Occupancy includes the observer itself: its covered goal cannot pull the
    exploration vector toward zero.  A separate own-goal score lets the policy
    choose whether to maintain coverage.  This point-target adaptation uses
    the paper's Eq. 6 kernel at the sensor range, with no additional scale.
    """
    if env.num_objectives == 0:
        return jnp.zeros((env.max_num_agents, env.state.dim + 2))
    displacement = env.env_params["objective"][None, :, :] - env.state.pos[:, None, :]
    distance_sq = jnp.sum(displacement**2, axis=-1)
    distance = jnp.sqrt(distance_sq)
    lr = env.env_params["lidar_range"]
    peer_distance_sq = jnp.sum(
        (env.state.pos[None, :, :] - env.state.pos[:, None, :]) ** 2, axis=-1
    )
    # Self is deliberately included in the occupancy test here.
    visible_agents = peer_distance_sq < lr**2
    covers_goal = distance_sq <= env.state.rad[:, None] ** 2
    occupied = (
        visible_agents.astype(jnp.int32) @ covers_goal.astype(jnp.int32)
    ) > 0
    visible_goals = distance_sq < lr**2
    weights = jnp.where(
        visible_goals & ~occupied, 0.5 * (1 + jnp.cos(jnp.pi * distance / lr)), 0
    )
    mass = weights.sum(axis=-1, keepdims=True)
    drift = (weights[:, :, None] * displacement).sum(axis=1) / jnp.maximum(
        mass, jnp.finfo(weights.dtype).tiny
    )
    own = jnp.where(
        visible_goals, 1 / (1 + distance_sq / (1.5 * env.state.rad[:, None]) ** 2), 0
    )
    return jnp.concatenate(
        [drift / lr, -jnp.expm1(-mass), own.max(axis=-1, keepdims=True)], axis=-1
    )


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ExperimentalSwarmNavigator(SwarmNavigator):
    """Ablate reward and observation independently without changing physics."""

    reward_mode: str = jax.tree.static(default="baseline")
    observation_mode: str = jax.tree.static(default="baseline")

    @classmethod
    def Create(
        cls,
        *,
        reward_mode: str = "baseline",
        observation_mode: str = "baseline",
        **kwargs: Any,
    ) -> ExperimentalSwarmNavigator:
        if reward_mode not in {"baseline", "rational"}:
            raise ValueError(f"Unknown reward_mode: {reward_mode}")
        if observation_mode not in {
            "baseline", "compact", "exact_local", "vacant_lidar", "paper"
        }:
            raise ValueError(f"Unknown observation_mode: {observation_mode}")
        # The base constructor instantiates cls and retains the subclass
        # defaults; these static choices are installed before any tracing.
        env = SwarmNavigator.Create.__func__(cls, **kwargs)
        env.reward_mode = reward_mode
        env.observation_mode = observation_mode
        return env

    @staticmethod
    @jax.jit(inline=True)
    def checkpoint(env: ExperimentalSwarmNavigator, action: jax.Array):
        if env.reward_mode == "baseline":
            return SwarmNavigator.checkpoint(env, action)
        return env

    @staticmethod
    @jax.jit
    def observation(env: ExperimentalSwarmNavigator) -> jax.Array:
        if env.observation_mode == "baseline":
            return SwarmNavigator.observation(env)
        lidar, objectives, ids = SwarmNavigator._sense(env)
        if env.observation_mode == "compact":
            features = compact_scan_features(objectives, lidar, ids, env)
        elif env.observation_mode == "vacant_lidar":
            features = vacant_objective_lidar(env) / env.env_params["lidar_range"]
        elif env.observation_mode == "paper":
            features = paper_vacancy_features(env)
        else:
            features = exact_local_features(env)
        return jnp.concatenate(
            [env.state.vel, features, lidar / env.env_params["lidar_range"]],
            axis=-1,
        )

    @staticmethod
    @jax.jit(inline=True)
    def reward(env: ExperimentalSwarmNavigator) -> jax.Array:
        if env.reward_mode == "baseline":
            return SwarmNavigator.reward(env)
        if env.num_objectives == 0:
            return jnp.zeros_like(env.state.rad)
        peers, objectives, ids = SwarmNavigator._sense(env)
        _, scores, _ = scan_goal_scores(objectives, peers, ids, env)
        return scores.max(axis=-1)

    @property
    def observation_space_size(self) -> int:
        if self.observation_mode in {"baseline", "vacant_lidar"}:
            return SwarmNavigator.observation_space_size.fget(self)
        bins = self.n_lidar_rays
        if self.state.dim == 3:
            bins *= self.n_lidar_elevation
        return 2 * self.state.dim + 2 + bins


def make_environment(
    variant: str,
    *,
    N: int = 64,
    num_objectives: int = 64,
    max_steps: int = 100000,
    seed: int | None = None,
    **kwargs: Any,
) -> SwarmNavigator:
    """Construct an unbatched baseline or isolated experimental environment."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {tuple(VARIANTS)}")
    options = dict(N=N, num_objectives=num_objectives, max_steps=max_steps, **kwargs)
    if variant == "baseline":
        env = SwarmNavigator.Create(**options)
    else:
        env = ExperimentalSwarmNavigator.Create(**options, **VARIANTS[variant])
    if seed is not None:
        env = env.reset(env, jax.random.key(seed))
    return env
