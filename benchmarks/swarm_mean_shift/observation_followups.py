"""Test vacancy guidance without discarding the baseline's observation.

Run with the unchanged matched PPO settings::

    python -m benchmarks.swarm_mean_shift.observation_followups \
        --variants paper_augmented --seeds 0 --epochs 120 \
        --output /tmp/swarm_observation_followups

``paper_augmented`` preserves every baseline feature in its original order and
appends the paper-style vacancy drift and mass: 26 + 3 = 29 features in 2-D.
Its reward, checkpoint, reset geometry, and dynamics are the production
baseline's.  The appended guidance requires richer local sensing: all targets
and peers within the observing agent's range, before angular-bin occlusion.
It never uses geometry outside that observer's range or a global coverage
score.  This is an adaptation to discrete target coverage, not a reproduction
of the paper's controller.
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

from benchmarks.swarm_mean_shift.variants import paper_vacancy_features


FOLLOWUP_NAMES = ("paper_augmented",)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PaperAugmentedNavigator(SwarmNavigator):
    """Append local vacancy guidance to the complete baseline observation."""

    @staticmethod
    @jax.jit
    def observation(env: PaperAugmentedNavigator) -> jax.Array:
        baseline = SwarmNavigator.observation(env)
        # Baseline LiDAR already contains own-goal information.  Append only
        # mean-shift displacement and mass, leaving out the duplicate own score.
        guidance = paper_vacancy_features(env)[:, : env.state.dim + 1]
        return jnp.concatenate([baseline, guidance], axis=-1)

    @property
    def observation_space_size(self) -> int:
        return SwarmNavigator.observation_space_size.fget(self) + self.state.dim + 1


def make_followup_environment(
    variant: str,
    *,
    N: int = 64,
    num_objectives: int = 64,
    max_steps: int = 100000,
    seed: int | None = None,
    **kwargs: Any,
) -> PaperAugmentedNavigator:
    """Match the frozen runner's factory signature without editing its files."""
    if variant not in FOLLOWUP_NAMES:
        raise ValueError(f"Unknown follow-up {variant!r}; expected one of {FOLLOWUP_NAMES}")
    env = PaperAugmentedNavigator.Create(
        N=N, num_objectives=num_objectives, max_steps=max_steps, **kwargs
    )
    if seed is not None:
        env = env.reset(env, jax.random.key(seed))
    return env


@contextmanager
def matched_followup_runner() -> Iterator[Any]:
    """Adapt the frozen runner process-locally and record this source's hash."""
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
