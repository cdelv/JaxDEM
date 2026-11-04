# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Combine rational potential differences with vacancy-aware goal LiDAR.

Run with the unchanged matched PPO settings::

    python -m benchmarks.swarm_mean_shift.combined_followup \
        --variants rational_delta_vacancy --seeds 0 1 2 --epochs 120 \
        --output /tmp/swarm_combined_followup

This combines two already-tested changes without adding another mechanism.
Reward is the action-start difference in the shared rational maximum, with
sigma fixed at 1.5 agent radii.  The observation keeps the baseline 26-feature
layout, removing locally occupied goals before nearest-per-bin reduction so a
farther vacant goal can become visible.  Covered goals under the observing
agent remain visible.  This observation requires richer local sensing than the
baseline: all objectives and peers within the observer's sensor range before
bin reduction.  Neither reward nor observation uses global coverage or geometry
outside the observing agent's range.  Dynamics and PPO settings are unchanged.
"""

from __future__ import annotations

import jax

from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
from typing import Any, Iterator

from benchmarks.swarm_mean_shift import reward_followups
from benchmarks.swarm_mean_shift.reward_followups import RationalFollowupNavigator

FOLLOWUP_NAMES = ("rational_delta_vacancy",)


def make_followup_environment(
    variant: str,
    *,
    N: int = 64,
    num_objectives: int = 64,
    max_steps: int = 100000,
    seed: int | None = None,
    **kwargs: Any,
) -> RationalFollowupNavigator:
    """Compose existing static modes before reset, vectorization, or tracing."""
    if variant not in FOLLOWUP_NAMES:
        raise ValueError(
            f"Unknown follow-up {variant!r}; expected one of {FOLLOWUP_NAMES}"
        )
    env = reward_followups.make_followup_environment(
        "rational_delta",
        N=N,
        num_objectives=num_objectives,
        max_steps=max_steps,
        **kwargs,
    )
    env.observation_mode = "vacant_lidar"
    if seed is not None:
        env = env.reset(env, jax.random.key(seed))
    return env


@contextmanager
def matched_followup_runner() -> Iterator[Any]:
    """Adapt only the current process and hash both follow-up source modules."""
    from benchmarks.swarm_mean_shift import run

    original_factory = run.make_environment
    original_names = run.VARIANT_NAMES
    original_hashes = run.source_hashes

    def hashes_with_followup(source_experiment: Path) -> dict[str, str | None]:
        hashes = original_hashes(source_experiment)
        for source in (
            Path(__file__).resolve(),
            Path(reward_followups.__file__).resolve(),
        ):
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
    if not any(
        arg == "--variants" or arg.startswith("--variants=") for arg in sys.argv[1:]
    ):
        sys.argv = [sys.argv[0], "--variants", *FOLLOWUP_NAMES, *sys.argv[1:]]
    try:
        with matched_followup_runner() as runner:
            runner.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
