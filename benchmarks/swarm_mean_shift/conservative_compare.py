# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Matched conservative swarm changes using the current action mapping.

Examples::

    python -m benchmarks.swarm_mean_shift.conservative_compare \
      --variants baseline_s1 cosine_s1 vacancy_s1 cosine_vacancy_s1 \
      --seeds 0 1 2 --output /tmp/conservative_spacing1

    python -m benchmarks.swarm_mean_shift.conservative_compare \
      --variants baseline_s2 cosine_s2 vacancy_s2 cosine_vacancy_s2 \
      --epochs 240 --schedule-epochs 4320 --max-physics-steps 200000 \
      --eval-at 40 120 240 --checkpoint-at 120 \
      --seeds 0 1 2 --output /tmp/conservative_spacing2

Production modules and the existing runner remain unchanged. Run inside the
new frozen runtime so every configuration uses the same current action map.
"""

from __future__ import annotations

import jax

import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import pickle
import sys
from typing import Any, Iterator

from benchmarks.swarm_mean_shift import run
from benchmarks.swarm_mean_shift.conservative_variants import make_environment

VARIANT_NAMES = tuple(
    f"{variant}_s{spacing}"
    for spacing in (1, 2)
    for variant in ("baseline", "cosine", "vacancy", "cosine_vacancy")
)
CLI_DEFAULTS = {
    "--variants": tuple(name for name in VARIANT_NAMES if name.endswith("_s1")),
    "--epochs": ("120",),
    "--schedule-epochs": ("2160",),
    "--max-physics-steps": ("100000",),
    "--eval-at": ("40", "120"),
}


@contextmanager
def matched_conservative_runner(
    *, checkpoint_at: tuple[int, ...] = ()
) -> Iterator[Any]:
    old_factory, old_names = run.make_environment, run.VARIANT_NAMES
    old_hashes, old_evaluate = run.source_hashes, run.evaluate

    def hashes(source_experiment: Path) -> dict[str, str | None]:
        result = old_hashes(source_experiment)
        for name in (
            "conservative_compare.py",
            "conservative_evaluate.py",
            "conservative_variants.py",
            "variants.py",
            "spacing_evaluate.py",
            "evaluate_checkpoints.py",
        ):
            path = Path(__file__).with_name(name).resolve()
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def evaluate_with_checkpoint(
        trainer: Any, base_env: Any, args: Any, directory: Path, epoch: int
    ) -> dict[str, Any]:
        from benchmarks.swarm_mean_shift.spacing_evaluate import add_spacing_metrics

        result = add_spacing_metrics(
            old_evaluate(trainer, base_env, args, directory, epoch),
            directory,
            epoch,
        )
        if epoch in checkpoint_at:
            for filename, value in (
                (f"graphstate_epoch_{epoch:04d}.pkl", trainer.graphstate),
                (f"trainer_leaves_epoch_{epoch:04d}.pkl", jax.tree.leaves(trainer)),
            ):
                with (directory / filename).open("wb") as stream:
                    pickle.dump(jax.device_get(value), stream)
        return result

    run.make_environment, run.VARIANT_NAMES = make_environment, VARIANT_NAMES
    run.source_hashes, run.evaluate = hashes, evaluate_with_checkpoint
    try:
        yield run
    finally:
        run.make_environment, run.VARIANT_NAMES = old_factory, old_names
        run.source_hashes, run.evaluate = old_hashes, old_evaluate


def main() -> None:
    # Strip only this wrapper's argument; the original runner parses all PPO
    # settings. Requested checkpoint epochs must be explicit evaluation epochs.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint-at", nargs="+", type=int, default=[])
    extra, arguments = parser.parse_known_args(sys.argv[1:])
    if any(epoch < 0 for epoch in extra.checkpoint_at):
        parser.error("Checkpoint epochs must be nonnegative")
    defaults: list[str] = []
    for flag, values in CLI_DEFAULTS.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in arguments):
            defaults.extend((flag, *values))
    original_argv = sys.argv
    sys.argv = [original_argv[0], *defaults, *arguments]
    try:
        with matched_conservative_runner(
            checkpoint_at=tuple(extra.checkpoint_at)
        ) as runner:
            runner.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
