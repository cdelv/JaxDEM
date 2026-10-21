"""Fresh-layout confirmation for unmasked changes to swarm navigation.

Restores trusted local checkpoints using their exact training factory and
recorded source hashes. Defaults to 64 independent confirmation layouts, reset
seed 90000 and action seed 100000. The primary score is mean coverage over the
final 10 percent of action endpoints, evaluated at each checkpoint's original
episode horizon. Training and model state are never updated.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator

from benchmarks.swarm_mean_shift import evaluate_checkpoints as original
from benchmarks.swarm_mean_shift import run, spacing_evaluate, unmasked_variants
from benchmarks.swarm_mean_shift.unmasked_compare import (
    VARIANT_NAMES, checked_environment, checked_trainer_entropy,
    matched_trainer_entropy, validate_mask_config,
)


@contextmanager
def matched_unmasked_confirmation() -> Iterator[None]:
    old_factory, old_evaluate = original.make_any_environment, run.evaluate
    old_rebuild = original.rebuild_and_restore

    def restore_with_entropy(config: dict[str, Any], checkpoint: Path) -> tuple[Any, Any, dict[str, Any]]:
        validate_mask_config(config, config["variant"])
        expected = config["ppo_entropy_coeff"]
        with matched_trainer_entropy(expected):
            trainer, base_env, restoration = old_rebuild(config, checkpoint)
        actual = checked_trainer_entropy(trainer, expected)
        if config.get("entropy_coefficient_matches_config") is not True:
            raise ValueError("Checkpoint lacks verified training entropy metadata")
        if config.get("actual_ppo_entropy_coeff") != actual:
            raise ValueError("Restored entropy leaf differs from recorded training coefficient")
        return trainer, base_env, {
            **restoration, "ppo_entropy_coeff": actual,
            "entropy_coefficient_matches_config": True,
        }

    def evaluate_with_metrics(
        trainer: Any, base_env: Any, args: Any, directory: Path, epoch: int
    ) -> dict[str, Any]:
        return spacing_evaluate.add_spacing_metrics(
            old_evaluate(trainer, base_env, args, directory, epoch), directory, epoch,
        )

    original.make_any_environment = checked_environment
    original.rebuild_and_restore = restore_with_entropy
    run.evaluate = evaluate_with_metrics
    try:
        yield
    finally:
        original.make_any_environment, run.evaluate = old_factory, old_evaluate
        original.rebuild_and_restore = old_rebuild


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANT_NAMES, default=[
        name for name in VARIANT_NAMES if name.endswith("_s1")
    ])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--eval-actions", type=int)
    parser.add_argument("--eval-reset-seed", type=int, default=90000)
    parser.add_argument("--eval-action-seed", type=int, default=100000)
    parser.add_argument("--eval-curve-stride", type=int, default=100)
    parser.add_argument("--action-mode", choices=("stochastic", "deterministic"))
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    if min(args.eval_envs, args.eval_curve_stride) < 1 or (
        args.eval_actions is not None and args.eval_actions < 1
    ):
        parser.error("Evaluation environment count, action count, and curve stride must be positive")
    if any(seed < 0 for seed in (*args.seeds, args.eval_reset_seed, args.eval_action_seed)):
        parser.error("Seeds must be nonnegative")
    for attribute in ("input", "output", "runtime_root", "original_root"):
        value = getattr(args, attribute)
        if value is not None:
            setattr(args, attribute, value.resolve())
    if args.input == args.output:
        parser.error("Confirmation output must differ from training input")
    return args


def main() -> None:
    args = parse_args()
    original.validate_runtime_imports(args.runtime_root)
    sources = [Path(module.__file__).resolve() for module in (
        original, run, spacing_evaluate, unmasked_variants,
    )]
    sources.extend((Path(__file__).resolve(), Path(__file__).with_name("unmasked_compare.py").resolve()))
    for source in sources:
        if not source.is_relative_to(args.runtime_root):
            raise ValueError(f"Confirmation source must come from frozen runtime: {source}")
    source_hashes = {str(source): original.sha256(source) for source in sources}
    protocol = {
        "experiment": "unmasked-swarm-independent-confirmation",
        "variants": args.variants, "training_seeds": args.seeds,
        "eval_envs": args.eval_envs, "eval_actions": args.eval_actions,
        "eval_reset_seed": args.eval_reset_seed, "eval_action_seed": args.eval_action_seed,
        "primary_metric": spacing_evaluate.PRIMARY_METRIC,
        "primary_window": "final 10 percent of action endpoints, rounded up",
        "horizon": "original checkpoint episode horizon unless eval_actions is explicitly shorter",
        "action_mapping": "current frozen runtime; validated against each checkpoint's recorded hash",
        "source_hashes": source_hashes, "input": str(args.input),
        "runtime_root": str(args.runtime_root),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "unmasked_confirmation_protocol.json"
    encoded = json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("Existing unmasked confirmation protocol differs from this invocation")
    if not path.exists():
        path.write_text(encoded)
    with matched_unmasked_confirmation():
        for variant in args.variants:
            for seed in args.seeds:
                source_config = args.input / f"{variant}_seed_{seed}" / "config.json"
                if source_config.exists():
                    config = json.loads(source_config.read_text())
                    validate_mask_config(config, variant)
                    for filename in ("unmasked_variants.py", "unmasked_compare.py", "unmasked_evaluate.py"):
                        if not any(name.endswith("/" + filename) for name in config["source_hashes"]):
                            raise ValueError(f"Unmasked checkpoint lacks critical source hash: {filename}")
                original.evaluate_one(args, variant, seed)
                config_path = args.output / f"{variant}_seed_{seed}" / "config.json"
                if config_path.exists():
                    config = json.loads(config_path.read_text())
                    config["confirmation_source_hashes"].update(source_hashes)
                    config["primary_metric"] = spacing_evaluate.PRIMARY_METRIC
                    config["confirmation_protocol_sha256"] = original.sha256(path)
                    config_path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
