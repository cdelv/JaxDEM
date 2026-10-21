"""Matched follow-ups separating a rational kernel from reward timing/scale.

Example (uses the unchanged matched PPO runner and all its defaults)::

    python -m benchmarks.swarm_mean_shift.reward_followups \
        --variants rational_delta rational_scaled --seeds 0 --epochs 120 \
        --output /tmp/swarm_reward_followups

Both variants retain the baseline's 26-feature observation and raw local LiDAR
geometry.  No global training information is introduced.  ``rational_delta``
changes the original experiment's potential to a one-scale rational maximum,
preserving its action-to-action difference objective.  ``rational_scaled``
keeps the rational state objective but multiplies it by 0.01 to probe PPO's
sensitivity to reward/value-loss scale.  These are distinct training objectives;
neither is claimed to be a policy-invariant transformation of the baseline.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Any, Iterator

import jax
import jax.numpy as jnp

from jaxdem.rl.environments import SwarmNavigator

from benchmarks.swarm_mean_shift.variants import ExperimentalSwarmNavigator


FOLLOWUP_NAMES = ("rational_delta", "rational_scaled")


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class RationalFollowupNavigator(ExperimentalSwarmNavigator):
    """Reuse the frozen rational score with either a delta or a small scale."""

    reward_transform: str = jax.tree.static(default="rational_delta")

    @classmethod
    def Create(
        cls, *, reward_transform: str = "rational_delta", **kwargs: Any
    ) -> RationalFollowupNavigator:
        if reward_transform not in FOLLOWUP_NAMES:
            raise ValueError(f"Unknown reward_transform: {reward_transform}")
        env = ExperimentalSwarmNavigator.Create.__func__(
            cls, reward_mode="rational", observation_mode="baseline", **kwargs
        )
        env.reward_transform = reward_transform
        return env

    @staticmethod
    @jax.jit
    def reset(env: RationalFollowupNavigator, key: jax.Array):
        # Base reset explicitly initializes the baseline's old potential.  A
        # delta follow-up must replace it with its own rational score now.
        env = SwarmNavigator.reset(env, key)
        return RationalFollowupNavigator.checkpoint(
            env, jnp.zeros((env.max_num_agents, env.action_space_size))
        )

    @staticmethod
    @jax.jit(inline=True)
    def checkpoint(env: RationalFollowupNavigator, action: jax.Array):
        if env.reward_transform == "rational_delta":
            env.env_params["prev_potential"] = ExperimentalSwarmNavigator.reward(env)
        # The scaled state reward does not read or update historical state.
        return env

    @staticmethod
    @jax.jit(inline=True)
    def reward(env: RationalFollowupNavigator) -> jax.Array:
        current = ExperimentalSwarmNavigator.reward(env)
        if env.reward_transform == "rational_delta":
            return current - env.env_params["prev_potential"]
        return 0.01 * current


def make_followup_environment(
    variant: str,
    *,
    N: int = 64,
    num_objectives: int = 64,
    max_steps: int = 100000,
    seed: int | None = None,
    **kwargs: Any,
) -> RationalFollowupNavigator:
    """Match the frozen runner's factory signature without editing its files."""
    if variant not in FOLLOWUP_NAMES:
        raise ValueError(f"Unknown follow-up {variant!r}; expected one of {FOLLOWUP_NAMES}")
    env = RationalFollowupNavigator.Create(
        reward_transform=variant,
        N=N,
        num_objectives=num_objectives,
        max_steps=max_steps,
        **kwargs,
    )
    if seed is not None:
        env = env.reset(env, jax.random.key(seed))
    return env


@contextmanager
def matched_followup_runner() -> Iterator[Any]:
    """Temporarily adapt runner choices/factory and record this module's hash.

    Source files are unchanged.  Patches are process-local and restored even if
    argument parsing or a run raises, so active experiments remain unaffected.
    """
    from benchmarks.swarm_mean_shift import run

    original_factory = run.make_environment
    original_names = run.VARIANT_NAMES
    original_hashes = run.source_hashes

    def hashes_with_followup(source_experiment: Path) -> dict[str, str | None]:
        hashes = original_hashes(source_experiment)
        source = Path(__file__).resolve()
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        return hashes

    run.make_environment = make_followup_environment
    run.VARIANT_NAMES = FOLLOWUP_NAMES
    run.source_hashes = hashes_with_followup
    try:
        yield run
    finally:
        run.make_environment = original_factory
        run.VARIANT_NAMES = original_names
        run.source_hashes = original_hashes


def main() -> None:
    # The frozen parser has baseline variant defaults.  Supply follow-up
    # defaults only when the caller did not explicitly choose any variants.
    original_argv = sys.argv
    if not any(arg == "--variants" or arg.startswith("--variants=") for arg in sys.argv[1:]):
        sys.argv = [sys.argv[0], "--variants", *FOLLOWUP_NAMES, *sys.argv[1:]]
    try:
        with matched_followup_runner() as runner:
            runner.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
