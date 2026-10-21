"""Independent confirmation of the doubled-spacing N=64 comparison.

Run from the isolated, frozen experiment runtime::

    python -m benchmarks.swarm_mean_shift.spacing_evaluate \
      --runtime-root /path/to/runtime --input /path/to/training \
      --output /path/to/confirmation --seeds 0 1 2 3 4

This restores matching doubled-spacing trainers; it does not retrain policies.
The default confirmation uses 64 fresh layouts (reset seed 50000, action seed
60000). The primary score is coverage over the final 10 percent of the episode,
which is 400 actions for the planned 4000-action doubled-duration episodes.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from benchmarks.swarm_mean_shift import evaluate_checkpoints as original
from benchmarks.swarm_mean_shift import run


VARIANTS = ("spacing_baseline", "spacing_rational_delta")
PRIMARY_METRIC = "coverage_final_10_percent_mean"


def add_spacing_metrics(
    evaluation: dict[str, Any], directory: Path, epoch: int
) -> dict[str, Any]:
    """Extend a completed evaluation and its saved JSONL using its exact archive.

    The original endpoint, episode-average, and final-200-action scores remain
    available. Coverage at actions 2000 and 4000 also makes the effect of giving
    the agents more time visible without requiring a second stochastic rollout.
    This helper is shared by training and independent confirmation.
    """
    archive = directory / f"evaluation_epoch_{epoch:04d}.npz"
    with np.load(archive) as arrays:
        coverage = np.asarray(arrays["coverage"])
    if coverage.ndim != 2 or coverage.shape != (
        evaluation["actions"], evaluation["num_envs"]
    ):
        raise ValueError(f"Unexpected coverage archive shape: {coverage.shape}")
    if not np.isfinite(coverage).all():
        raise ValueError("Nonfinite coverage in evaluation archive")
    actions = coverage.shape[0]
    window = max(1, (actions + 9) // 10)
    tail = coverage[-window:].mean(axis=0)
    evaluation.update({
        "primary_metric": PRIMARY_METRIC,
        "coverage_final_10_percent_actions": window,
        "coverage_final_10_percent_start_action": actions - window + 1,
        PRIMARY_METRIC: float(tail.mean()),
        "coverage_final_10_percent_std": float(tail.std()),
        "coverage_final_10_percent_per_episode": tail.tolist(),
    })
    for action in (2000, 4000, 8000):
        if action <= actions:
            values = coverage[action - 1]
            prefix = f"coverage_action_{action}"
            evaluation.update({
                f"{prefix}_mean": float(values.mean()),
                f"{prefix}_std": float(values.std()),
                f"{prefix}_per_episode": values.tolist(),
            })
    log_path = directory / "evaluation.jsonl"
    lines = log_path.read_text().splitlines()
    if not lines or json.loads(lines[-1]).get("epoch") != epoch:
        raise ValueError("Last saved evaluation does not match the augmented epoch")
    lines[-1] = json.dumps(evaluation, allow_nan=False)
    log_path.write_text("\n".join(lines) + "\n")
    return evaluation


@contextmanager
def matched_spacing_confirmation() -> Iterator[None]:
    """Adapt only this process's original evaluator factory and metric writer."""
    from benchmarks.swarm_mean_shift.spacing_followup import make_followup_environment

    old_factory = original.make_any_environment
    old_evaluate = run.evaluate

    def evaluate_with_spacing_metrics(
        trainer: Any, base_env: Any, args: Any, directory: Path, epoch: int
    ) -> dict[str, Any]:
        evaluation = old_evaluate(trainer, base_env, args, directory, epoch)
        return add_spacing_metrics(evaluation, directory, epoch)

    original.make_any_environment = make_followup_environment
    run.evaluate = evaluate_with_spacing_metrics
    try:
        yield
    finally:
        original.make_any_environment = old_factory
        run.evaluate = old_evaluate


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--eval-actions", type=int)
    parser.add_argument("--eval-reset-seed", type=int, default=50000)
    parser.add_argument("--eval-action-seed", type=int, default=60000)
    parser.add_argument("--eval-curve-stride", type=int, default=100)
    parser.add_argument("--action-mode", choices=("stochastic", "deterministic"))
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    if args.eval_envs < 1 or args.eval_curve_stride < 1 or (
        args.eval_actions is not None and args.eval_actions < 1
    ):
        parser.error("Evaluation environments, actions, and curve stride must be positive")
    if any(seed < 0 for seed in (*args.seeds, args.eval_reset_seed, args.eval_action_seed)):
        parser.error("Seeds must be nonnegative")
    for attribute in ("input", "output", "runtime_root", "original_root"):
        value = getattr(args, attribute)
        if value is not None:
            setattr(args, attribute, value.resolve())
    if args.input == args.output:
        parser.error("Confirmation output must differ from the training results directory")
    return args


def confirmation_protocol(args: Any) -> dict[str, Any]:
    """Describe the independently chosen keys, metric, and actual helper code."""
    from benchmarks.swarm_mean_shift import spacing_followup

    original.validate_runtime_imports(args.runtime_root)
    sources = [Path(module.__file__).resolve() for module in (
        original, run, spacing_followup,
    )]
    sources.append(Path(__file__).resolve())
    for source in sources:
        if not source.is_relative_to(args.runtime_root):
            raise ValueError(f"Confirmation helper must come from the frozen runtime: {source}")
    return {
        "experiment": "doubled-spacing-independent-confirmation",
        "training_variants": args.variants,
        "training_seeds": args.seeds,
        "eval_envs": args.eval_envs,
        "eval_actions": args.eval_actions,
        "episode_duration": "checkpoint max_physics_steps unless eval_actions is explicitly shorter",
        "eval_reset_seed": args.eval_reset_seed,
        "eval_action_seed": args.eval_action_seed,
        "primary_metric": PRIMARY_METRIC,
        "primary_window": "final 10 percent of action endpoints, rounded up",
        "spacing_scale": 2.0,
        "fixed_controls": "N, agent radii, sensors, force, dt, policy architecture, PPO gamma=0.99",
        "source_hashes": {str(source): original.sha256(source) for source in sources},
        "input": str(args.input),
        "runtime_root": str(args.runtime_root),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = confirmation_protocol(args)
    protocol_path = args.output / "spacing_confirmation_protocol.json"
    encoded = json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    if protocol_path.exists():
        if protocol_path.read_text() != encoded:
            raise ValueError("Existing spacing confirmation protocol differs from this invocation")
    else:
        protocol_path.write_text(encoded)
    protocol_hash = hashlib.sha256(encoded.encode()).hexdigest()
    with matched_spacing_confirmation():
        for variant in args.variants:
            for seed in args.seeds:
                source_config = args.input / f"{variant}_seed_{seed}" / "config.json"
                if source_config.exists():
                    config = json.loads(source_config.read_text())
                    if not any(path.endswith("/spacing_followup.py") for path in config["source_hashes"]):
                        raise ValueError("Spacing checkpoint lacks its factory source hash")
                original.evaluate_one(args, variant, seed)
                config_path = args.output / f"{variant}_seed_{seed}" / "config.json"
                if config_path.exists():
                    config = json.loads(config_path.read_text())
                    config["confirmation_source_hashes"].update(protocol["source_hashes"])
                    config["spacing_confirmation_protocol_sha256"] = protocol_hash
                    config["primary_metric"] = PRIMARY_METRIC
                    config_path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
