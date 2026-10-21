"""Matched occupied-goal masking ablations using the current action mapping.

Examples::

    python -m benchmarks.swarm_mean_shift.unmasked_compare \
      --variants masked_s1 raw_s1 raw_cosine_matched_s1 \
      --seeds 0 1 2 --output /tmp/unmasked_spacing1

    python -m benchmarks.swarm_mean_shift.unmasked_compare \
      --variants masked_s2 raw_s2 raw_cosine_matched_s2 \
      --epochs 240 --schedule-epochs 4320 --max-physics-steps 200000 \
      --eval-at 40 120 240 --checkpoint-at 120 \
      --seeds 0 1 2 --output /tmp/unmasked_spacing2

Production modules and the existing runner remain unchanged. Run inside the
new frozen runtime so every configuration uses the same current action map.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from math import isclose
from pathlib import Path
import pickle
import sys
from typing import Any, Iterator

import jax

from benchmarks.swarm_mean_shift import run
from benchmarks.swarm_mean_shift.unmasked_variants import (
    VARIANT_NAMES, TypedUnmaskedSwarmNavigator, make_environment,
)


CLI_DEFAULTS = {
    "--variants": tuple(name for name in VARIANT_NAMES if name.endswith("_s1")),
    "--epochs": ("120",),
    "--schedule-epochs": ("2160",),
    "--max-physics-steps": ("100000",),
    "--eval-at": ("40", "120"),
}


def variant_metadata(variant: str) -> dict[str, Any]:
    if variant not in VARIANT_NAMES:
        raise ValueError(f"Unknown unmasked comparison variant: {variant!r}")
    cosine = variant.startswith("raw_cosine_matched_")
    return {
        "observation_occupancy_mask": variant.startswith("masked_"),
        "hide_occupied_objectives": variant.startswith("masked_"),
        "vacancy_aware_lidar": False,
        "approach_mode": "cosine" if cosine else "baseline",
        "attraction_coeff": 0.111 if cosine else 0.1,
        "peer_wall_encoding": "signed" if variant.startswith("raw_typed_") else "unsigned",
        "ppo_entropy_coeff": 1e-4 if variant.startswith("raw_low_entropy_") else 1e-3,
    }


def checked_environment(variant: str, **kwargs: Any) -> Any:
    """Verify the no-mask contract before constructing a trainer or rollout."""
    env = make_environment(variant, **kwargs)
    metadata = variant_metadata(variant)
    if bool(jax.device_get(env.env_params["hide_occupied_objectives"])) != metadata["observation_occupancy_mask"]:
        raise ValueError(f"Occupied-goal mask differs from declared {variant} condition")
    if bool(env.vacancy_aware_lidar):
        raise ValueError("Unmasked comparisons must not use the richer vacancy sensor")
    if env.reward_mode != metadata["approach_mode"]:
        raise ValueError("Approach mode differs from declared variant")
    if not isclose(float(jax.device_get(env.env_params["attraction_coeff"])), metadata["attraction_coeff"], rel_tol=1e-6):
        raise ValueError("Attraction amplitude differs from declared variant")
    if isinstance(env, TypedUnmaskedSwarmNavigator) != (metadata["peer_wall_encoding"] == "signed"):
        raise ValueError("Peer/wall return encoding differs from declared variant")
    return env


def validate_mask_config(config: dict[str, Any], variant: str) -> None:
    for key, expected in variant_metadata(variant).items():
        if config.get(key) != expected:
            raise ValueError(f"Checkpoint {variant} lacks matching {key}: {config.get(key)!r} != {expected!r}")


def checked_trainer_entropy(trainer: Any, expected: float) -> float:
    actual = float(jax.device_get(trainer.ppo_entropy_coeff))
    if not isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-10):
        raise ValueError(f"Trainer entropy coefficient differs from declared variant: {actual} != {expected}")
    return actual


@contextmanager
def matched_trainer_entropy(coefficient: float) -> Iterator[None]:
    """Set one PPO constructor argument without changing the shared runner.

    Preserve both the original descriptor and whether it was inherited. This
    makes nested use and exceptions restore the exact original factory binding.
    """
    trainer_type = run.rl.Trainer
    owned = "create" in trainer_type.__dict__
    original_descriptor = inspect.getattr_static(trainer_type, "create")
    original_bound = trainer_type.create

    def create_with_entropy(*args: Any, **kwargs: Any) -> Any:
        if "ppo_entropy_coeff" in kwargs and not isclose(
            float(kwargs["ppo_entropy_coeff"]), coefficient, rel_tol=1e-6, abs_tol=1e-10,
        ):
            raise ValueError("Explicit entropy coefficient conflicts with the declared variant")
        kwargs["ppo_entropy_coeff"] = coefficient
        trainer = original_bound(*args, **kwargs)
        checked_trainer_entropy(trainer, coefficient)
        return trainer

    trainer_type.create = staticmethod(create_with_entropy)
    try:
        yield
    finally:
        if owned:
            trainer_type.create = original_descriptor
        else:
            delattr(trainer_type, "create")


@contextmanager
def matched_unmasked_runner(*, checkpoint_at: tuple[int, ...] = ()) -> Iterator[Any]:
    old_factory, old_names = run.make_environment, run.VARIANT_NAMES
    old_hashes, old_evaluate = run.source_hashes, run.evaluate
    old_run_one = run.run_one

    def hashes(source_experiment: Path) -> dict[str, str | None]:
        result = old_hashes(source_experiment)
        for name in (
            "unmasked_compare.py", "unmasked_evaluate.py", "unmasked_variants.py",
            "unmasked_long_evaluate.py", "conservative_variants.py", "variants.py",
            "spacing_evaluate.py", "spacing_long_evaluate.py", "evaluate_checkpoints.py",
        ):
            path = Path(__file__).with_name(name).resolve()
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def evaluate_with_checkpoint(
        trainer: Any, base_env: Any, args: Any, directory: Path, epoch: int
    ) -> dict[str, Any]:
        from benchmarks.swarm_mean_shift.spacing_evaluate import add_spacing_metrics

        # run_one writes its final model/config fields after construction.
        # Record the verified leaf here, before any evaluation or completed-run
        # signal, so that write cannot overwrite the actual coefficient.
        actual_entropy = checked_trainer_entropy(trainer, args.ppo_entropy_coeff)
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        config.update({
            "actual_ppo_entropy_coeff": actual_entropy,
            "entropy_coefficient_matches_config": True,
        })
        temporary = config_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
        temporary.replace(config_path)
        result = add_spacing_metrics(
            old_evaluate(trainer, base_env, args, directory, epoch), directory, epoch,
        )
        if epoch in checkpoint_at:
            for filename, value in (
                (f"graphstate_epoch_{epoch:04d}.pkl", trainer.graphstate),
                (f"trainer_leaves_epoch_{epoch:04d}.pkl", jax.tree.leaves(trainer)),
            ):
                with (directory / filename).open("wb") as stream:
                    pickle.dump(jax.device_get(value), stream)
        return result

    def run_with_metadata(args: Any, variant: str, seed: int, graphdefs: dict[str, Any]) -> dict[str, Any]:
        # Install before run_one's first config write, so a completed-run signal
        # can never race a later metadata update from a queued evaluator.
        metadata = variant_metadata(variant)
        missing = object()
        previous = {key: getattr(args, key, missing) for key in metadata}
        for key, value in metadata.items():
            setattr(args, key, value)
        try:
            with matched_trainer_entropy(metadata["ppo_entropy_coeff"]):
                return old_run_one(args, variant, seed, graphdefs)
        finally:
            for key, value in previous.items():
                if value is missing:
                    delattr(args, key)
                else:
                    setattr(args, key, value)

    run.make_environment, run.VARIANT_NAMES = checked_environment, VARIANT_NAMES
    run.source_hashes, run.evaluate = hashes, evaluate_with_checkpoint
    run.run_one = run_with_metadata
    try:
        yield run
    finally:
        run.make_environment, run.VARIANT_NAMES = old_factory, old_names
        run.source_hashes, run.evaluate = old_hashes, old_evaluate
        run.run_one = old_run_one


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
        with matched_unmasked_runner(checkpoint_at=tuple(extra.checkpoint_at)) as runner:
            parsed = runner.parse_args()
            if len({name.rsplit("_s", 1)[1] for name in parsed.variants}) != 1:
                parser.error("Train each spacing separately so its episode and training budgets are explicit")
            available = set(parsed.eval_at) | {parsed.epochs}
            if any(epoch not in available or epoch > parsed.epochs for epoch in extra.checkpoint_at):
                parser.error("Checkpoint epochs must be evaluation epochs within the training budget")
            runner.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
