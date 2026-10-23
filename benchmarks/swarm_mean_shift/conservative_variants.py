# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Conservative swarm ablations preserving settling and broad attraction.

The cosine candidate replaces only the baseline's polynomial/exponential
approach potential with a raised cosine at the existing sensor range. Quartic
settling, quartic peer sharing, and action-start potential differences remain.
The vacancy candidate changes only the 12 objective LiDAR channels, removing
occupied objectives before nearest-return reduction. All variants retain the
same 26 observation entries, physical dynamics, and local execution sensors.

Suffixes ``s1`` and ``s2`` select the original seeded geometry and its exact
twofold coordinate scaling. Counts, particle radii, sensing and reward lengths
stay fixed. The caller supplies each condition's episode duration explicitly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from typing import Any

from jaxdem.rl.environments import SwarmNavigator
from jaxdem.rl.environments._swarm import _peer_geometry
from benchmarks.swarm_mean_shift.variants import vacant_objective_lidar

VARIANTS = {
    f"{name}_s{scale}": (float(scale), reward, observation)
    for scale in (1, 2)
    for name, reward, observation in (
        ("baseline", "baseline", "baseline"),
        ("cosine", "cosine", "baseline"),
        ("vacancy", "baseline", "vacancy"),
        ("cosine_vacancy", "cosine", "vacancy"),
    )
}
VARIANT_NAMES = tuple(VARIANTS)


def cosine_shared_potential(
    lidar_obj: jax.Array,
    lidar: jax.Array,
    peer_ids: jax.Array,
    env: Any,
) -> jax.Array:
    """Keep baseline settling/sharing and replace only the approach kernel.

    ``H(d)=(1+cos(pi*d/L))/2`` has no new length scale; its value and
    derivative approach zero at sensor range. Missing returns stay zero.
    The baseline's attraction_constant/quadratic/decay have no effect here.
    """
    distance, separation_sq, valid_peer = _peer_geometry(
        lidar_obj,
        lidar,
        peer_ids,
        env,
        env.n_lidar_elevation if env.state.dim == 3 else None,
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
    approach = 0.5 * (1 + jnp.cos(jnp.pi * distance / env.env_params["lidar_range"]))
    return local + env.env_params["attraction_coeff"] * jnp.where(
        valid_objective, weight * approach, 0.0
    ).sum(axis=-1)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ConservativeSwarmNavigator(SwarmNavigator):
    """Independent reward/observation ablations with matched reset geometry."""

    reward_mode: str = jax.tree.static(default="baseline")
    observation_mode: str = jax.tree.static(default="baseline")
    spacing_scale: float = jax.tree.static(default=1.0)
    base_box_size: float = jax.tree.static(default=20.0)
    base_box_padding: float = jax.tree.static(default=10.0)

    @classmethod
    def Create(
        cls,
        *,
        reward_mode: str = "baseline",
        observation_mode: str = "baseline",
        spacing_scale: float = 1.0,
        box_size: float = 20.0,
        box_padding: float = 10.0,
        **kwargs: Any,
    ) -> ConservativeSwarmNavigator:
        if reward_mode not in {"baseline", "cosine"}:
            raise ValueError(f"Unknown reward_mode: {reward_mode!r}")
        if observation_mode not in {"baseline", "vacancy"}:
            raise ValueError(f"Unknown observation_mode: {observation_mode!r}")
        if spacing_scale not in (1.0, 2.0):
            raise ValueError("spacing_scale must be 1 or 2")
        env = SwarmNavigator.Create.__func__(
            cls,
            box_size=box_size,
            box_padding=box_padding,
            **kwargs,
        )
        env.reward_mode = reward_mode
        env.observation_mode = observation_mode
        env.spacing_scale = float(spacing_scale)
        env.base_box_size = float(box_size)
        env.base_box_padding = float(box_padding)
        return env

    @staticmethod
    @jax.jit
    def reset(env: ConservativeSwarmNavigator, key: jax.Array):
        # Base reset constructs a fresh physical system and calls its own
        # baseline checkpoint, so the subclass must overwrite history last.
        env.env_params["box_size"] = jnp.asarray(env.base_box_size, dtype=float)
        env.env_params["box_padding"] = jnp.asarray(env.base_box_padding, dtype=float)
        env = SwarmNavigator.reset(env, key)
        scale = env.spacing_scale
        env.state.pos_c = env.state.pos_c * scale
        env.env_params["objective"] = env.env_params["objective"] * scale
        env.env_params["box_size"] = env.env_params["box_size"] * scale
        env.env_params["box_padding"] = env.env_params["box_padding"] * scale
        env.system.domain.box_size = env.system.domain.box_size * scale
        env.system.domain.inv_box_size = 1 / env.system.domain.box_size
        env.system.domain.anchor = env.system.domain.anchor * scale
        return ConservativeSwarmNavigator.checkpoint(
            env, jnp.zeros((env.max_num_agents, env.action_space_size))
        )

    @staticmethod
    @jax.jit(inline=True)
    def _potential(env: ConservativeSwarmNavigator) -> jax.Array:
        if env.reward_mode == "baseline":
            return SwarmNavigator._potential(env)
        if env.num_objectives == 0:
            return jnp.zeros_like(env.state.rad)
        peers, objectives, ids = SwarmNavigator._sense(env)
        return cosine_shared_potential(objectives, peers, ids, env)

    @staticmethod
    @jax.jit(inline=True)
    def checkpoint(env: ConservativeSwarmNavigator, action: jax.Array):
        env.env_params["prev_potential"] = ConservativeSwarmNavigator._potential(env)
        return env

    @staticmethod
    @jax.jit(inline=True)
    def reward(env: ConservativeSwarmNavigator) -> jax.Array:
        return (
            ConservativeSwarmNavigator._potential(env)
            - env.env_params["prev_potential"]
        )

    @staticmethod
    @jax.jit
    def observation(env: ConservativeSwarmNavigator) -> jax.Array:
        if env.observation_mode == "baseline":
            return SwarmNavigator.observation(env)
        peers, _, _ = SwarmNavigator._sense(env)
        lr = env.env_params["lidar_range"]
        return jnp.concatenate(
            [env.state.vel, vacant_objective_lidar(env) / lr, peers / lr],
            axis=-1,
        )


def make_environment(
    variant: str,
    *,
    N: int = 64,
    num_objectives: int = 64,
    max_steps: int = 100000,
    seed: int | None = None,
    **kwargs: Any,
) -> ConservativeSwarmNavigator:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected {VARIANT_NAMES}")
    spacing, reward, observation = VARIANTS[variant]
    env = ConservativeSwarmNavigator.Create(
        spacing_scale=spacing,
        reward_mode=reward,
        observation_mode=observation,
        N=N,
        num_objectives=num_objectives,
        max_steps=max_steps,
        **kwargs,
    )
    if seed is not None:
        env = env.reset(env, jax.random.key(seed))
    return env
