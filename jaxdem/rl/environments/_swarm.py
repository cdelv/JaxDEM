# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Shared local coverage rewards and observation filtering for swarm tasks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from typing import Any

from ...utils.environment import _bin_azimuth, _bin_min_dist_and_idx, _bin_spherical


def bin_directions(n_azimuth: int, n_elevation: int | None = None) -> jax.Array:
    """Return LiDAR bin-center directions, flattened azimuth then elevation."""
    azimuth = -jnp.pi + (jnp.arange(n_azimuth) + 0.5) * (2 * jnp.pi / n_azimuth)
    if n_elevation is None:
        return jnp.stack([jnp.cos(azimuth), jnp.sin(azimuth)], axis=-1)
    elevation = -jnp.pi / 2 + (jnp.arange(n_elevation) + 0.5) * (jnp.pi / n_elevation)
    azimuth, elevation = jnp.meshgrid(azimuth, elevation, indexing="ij")
    return jnp.stack(
        [
            jnp.cos(elevation) * jnp.cos(azimuth),
            jnp.cos(elevation) * jnp.sin(azimuth),
            jnp.sin(elevation),
        ],
        axis=-1,
    ).reshape(-1, 3)


def _peer_geometry(
    lidar_obj: jax.Array,
    lidar: jax.Array,
    peer_ids: jax.Array,
    env: Any,
    n_elevation: int | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if n_elevation is None:
        # A common bin-center offset cancels in 2-D. Keep the original
        # angular-difference calculation for existing swarm environments.
        angle = jnp.arange(env.n_lidar_rays) * (2 * jnp.pi / env.n_lidar_rays)
        cosine = jnp.cos(angle[:, None] - angle[None, :])
    else:
        directions = bin_directions(env.n_lidar_rays, n_elevation)
        cosine = jnp.clip(directions @ directions.T, -1.0, 1.0)
        cosine = jnp.where(jnp.eye(cosine.shape[0], dtype=bool), 1.0, cosine)
    distance = env.env_params["lidar_range"] - lidar_obj
    peer_distance = env.env_params["lidar_range"] - lidar
    separation_sq = jnp.maximum(
        distance[:, :, None] ** 2
        + peer_distance[:, None, :] ** 2
        - 2 * distance[:, :, None] * peer_distance[:, None, :] * cosine,
        0.0,
    )
    valid_peer = (
        (lidar > 0)
        & (peer_ids >= 0)
        & (peer_ids < env.max_num_agents)
        & (peer_ids != jnp.arange(env.max_num_agents)[:, None])
    )
    return distance, separation_sq, valid_peer


def shared_potential(
    lidar_obj: jax.Array,
    lidar: jax.Array,
    peer_ids: jax.Array,
    env: Any,
    *,
    n_elevation: int | None = None,
    approach_kernel: str = "legacy",
) -> jax.Array:
    """Evaluate shared settling and approach value from unfiltered live scans.

    Only valid peer returns can claim objectives; walls, self and empty bins
    contribute nothing. The environment is read without changing its history.
    ``cosine`` changes only the broad approach term to a raised cosine at the
    sensor range; ``legacy`` retains the original attraction arithmetic.
    """
    distance, separation_sq, valid_peer = _peer_geometry(
        lidar_obj, lidar, peer_ids, env, n_elevation
    )
    valid_objective = lidar_obj > 0
    radii = env.state.rad[:, None]
    scale = env.env_params["sharing_range"] * radii[:, :, None]
    claims = jnp.where(
        valid_peer[:, None, :],
        jnp.exp(-((separation_sq / scale**2) ** 2)),
        0.0,
    ).sum(axis=-1)
    weight = 1 / (1 + claims)
    settling = jnp.exp(-((distance / (1.5 * radii)) ** 4))
    local = jnp.where(valid_objective, weight * settling, 0.0).max(axis=-1)
    if approach_kernel == "cosine":
        attraction = 0.5 * (
            1 + jnp.cos(jnp.pi * distance / env.env_params["lidar_range"])
        )
    elif approach_kernel == "legacy":
        x = distance / radii
        decay = env.env_params["attraction_decay"]
        attraction = jnp.exp(-decay * x) * (
            env.env_params["attraction_constant"] / decay
            + env.env_params["attraction_quadratic"]
            * (x**2 / decay + 2 * x / decay**2 + 2 / decay**3)
        )
    else:
        raise ValueError("approach_kernel must be 'legacy' or 'cosine'")
    return local + env.env_params["attraction_coeff"] * jnp.where(
        valid_objective, weight * attraction, 0.0
    ).sum(axis=-1)


def filter_objectives(
    lidar_obj: jax.Array,
    lidar: jax.Array,
    peer_ids: jax.Array,
    env: Any,
    *,
    n_elevation: int | None = None,
) -> jax.Array:
    """Hide locally occupied goals below half range, preserving own goals.

    ``occupancy_radius_scale`` widens the peer-to-goal tolerance only; it
    defaults to one radius for environments without that parameter.
    """
    distance, separation_sq, valid_peer = _peer_geometry(
        lidar_obj, lidar, peer_ids, env, n_elevation
    )
    radii = env.state.rad[:, None]
    occupancy_radii = env.env_params.get("occupancy_radius_scale", 1.0) * radii
    occupied = jnp.any(
        valid_peer[:, None, :] & (separation_sq <= occupancy_radii[:, :, None] ** 2),
        axis=-1,
    )
    hide = (
        env.env_params["hide_occupied_objectives"]
        & (lidar_obj > 0)
        & (distance < env.env_params["lidar_range"] / 2)
        & (distance > radii)
        & occupied
    )
    return jnp.where(hide, 0.0, lidar_obj)


def vacant_objective_lidar(env: Any) -> jax.Array:
    """Read local vacant goals, removing occupied goals before bin reduction.

    For each observer, only goals and peers strictly inside its sensor range
    participate. A visible peer occupies a goal when its center is within its
    own radius; goals already covered by the observer are always preserved.
    This requires all local goal/peer detections before angular-bin reduction,
    so an occupied near goal can reveal a farther vacant goal in the same bin.
    The output is proximity in the original objective LiDAR bins, in 2-D/3-D.
    """
    bins = env.n_lidar_rays * (env.n_lidar_elevation if env.state.dim == 3 else 1)
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
