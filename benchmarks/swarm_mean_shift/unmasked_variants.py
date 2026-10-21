"""Matched local-sensing ablations that preserve occupied-goal returns.

``raw`` changes only the baseline's observation mask. ``raw_cosine_matched``
also uses the existing conservative cosine reward, with approach weight 0.111
to match the legacy default's unshared approach value at zero distance. This
matches that amplitude, not its entire shape or learning objective.
``raw_typed`` adds local wall/peer type information by signing the existing
peer/wall channels; the objective channels are untouched. ``raw_low_entropy``
shares the raw environment and lets the runner reduce PPO entropy.

All raw variants force ``hide_occupied_objectives=False`` and disable richer
vacancy sensing. They retain the nearest raw goal in every LiDAR bin, including
goals occupied by peers. Reward sharing remains soft and local. ``masked`` is
the unchanged control. Suffixes s1/s2 reuse exact onefold/twofold seeded geometry
from ``conservative_variants``; callers supply the appropriate time budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any

import jax
import jax.numpy as jnp

from jaxdem.rl.environments import SwarmNavigator

from benchmarks.swarm_mean_shift.conservative_variants import (
    ConservativeSwarmNavigator,
    make_environment as make_conservative_environment,
)


MATCHED_COSINE_COEFF = 0.111
VARIANTS = {
    f"{family}_s{scale}": (scale, masked, reward, coefficient)
    for scale in (1, 2)
    for family, masked, reward, coefficient in (
        ("masked", True, "baseline", 0.1),
        ("raw", False, "baseline", 0.1),
        ("raw_typed", False, "baseline", 0.1),
        ("raw_low_entropy", False, "baseline", 0.1),
        ("raw_cosine_matched", False, "cosine", MATCHED_COSINE_COEFF),
    )
}
VARIANT_NAMES = tuple(VARIANTS)


def signed_peer_wall_lidar(
    lidar: jax.Array, ids: jax.Array, env: Any,
) -> jax.Array:
    """Expose existing local return types: walls negative, peers positive."""
    observer_ids = jnp.arange(env.max_num_agents)[:, None]
    peers = (ids >= 0) & (ids < env.max_num_agents) & (ids != observer_ids)
    return jnp.where(lidar > 0, jnp.where(ids < 0, -lidar, jnp.where(peers, lidar, 0)), 0)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class TypedUnmaskedSwarmNavigator(ConservativeSwarmNavigator):
    """Raw goals plus signed local wall/peer returns in the same channels.

    ``observation_mode`` remains baseline for the shared factory's mode
    contract; the subclass supplies this extra type encoding. Reward,
    checkpointing, and reset are inherited unchanged.
    """

    @staticmethod
    @jax.jit
    def observation(env: TypedUnmaskedSwarmNavigator) -> jax.Array:
        peers, goals, ids = SwarmNavigator._sense(env)
        lr = env.env_params["lidar_range"]
        return jnp.concatenate(
            [env.state.vel, goals / lr, signed_peer_wall_lidar(peers, ids, env) / lr],
            axis=-1,
        )


def _require_fixed(kwargs: dict[str, Any], name: str, expected: Any) -> None:
    """Reject contradictory overrides instead of silently enabling masking."""
    if name not in kwargs:
        return
    requested = kwargs.pop(name)
    if isinstance(expected, bool):
        valid = bool(requested) == expected
    elif isinstance(expected, float):
        valid = isclose(float(requested), expected, rel_tol=1e-6, abs_tol=1e-9)
    else:
        valid = requested == expected
    if not valid:
        raise ValueError(f"{name} is fixed at {expected!r} for this unmasked ablation")


def make_environment(
    variant: str,
    *,
    N: int = 64,
    num_objectives: int = 64,
    max_steps: int = 100000,
    seed: int | None = None,
    **kwargs: Any,
) -> ConservativeSwarmNavigator:
    """Create a fixed masked control or a strictly unmasked local variant.

    The production kernel selector is kept at legacy because the existing
    conservative wrapper dispatches its own cosine score. This avoids applying
    two reward selectors. ``raw_typed`` uses an observation-only subclass;
    ``raw_low_entropy`` is exactly ``raw`` here, with PPO entropy controlled by
    the training runner. All modes retain the shared reset and reward history.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected {VARIANT_NAMES}")
    scale, masked, reward, coefficient = VARIANTS[variant]
    _require_fixed(kwargs, "hide_occupied_objectives", masked)
    _require_fixed(kwargs, "vacancy_aware_lidar", False)
    _require_fixed(kwargs, "approach_kernel", "legacy")
    _require_fixed(kwargs, "attraction_coeff", coefficient)
    for name in ("reward_mode", "observation_mode", "spacing_scale"):
        if name in kwargs:
            raise ValueError(f"{name} is selected by the ablation variant")
    if variant.startswith("raw_typed_"):
        env = TypedUnmaskedSwarmNavigator.Create(
            spacing_scale=float(scale), reward_mode=reward, observation_mode="baseline",
            N=N, num_objectives=num_objectives, max_steps=max_steps,
            hide_occupied_objectives=False, vacancy_aware_lidar=False,
            approach_kernel="legacy", attraction_coeff=coefficient, **kwargs,
        )
        if seed is not None:
            env = env.reset(env, jax.random.key(seed))
        return env
    return make_conservative_environment(
        f"{'baseline' if reward == 'baseline' else 'cosine'}_s{scale}",
        N=N,
        num_objectives=num_objectives,
        max_steps=max_steps,
        seed=seed,
        hide_occupied_objectives=masked,
        vacancy_aware_lidar=False,
        approach_kernel="legacy",
        attraction_coeff=coefficient,
        **kwargs,
    )
