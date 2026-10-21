"""Test original policies on doubled-spacing scenes without further training.

Run from the new isolated spacing runtime, whose original files match the old
checkpoints' recorded source hashes::

    python -m benchmarks.swarm_mean_shift.spacing_zero_shot \
      --runtime-root /path/to/spacing/runtime --input /path/to/original/results \
      --output /path/to/spacing/results/zero_shot --seeds 0 1 2 3 4

The default episode has 4000 actions / 200000 physics steps. For a genuine
8000-action extra-time stress test, use --max-physics-steps 400000; increasing
only eval-actions would leave terminated scenes frozen and is disallowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import jax
import numpy as np

from jaxdem.rl.env_wrappers import vectorise_env
from benchmarks.swarm_mean_shift import evaluate_checkpoints as original
from benchmarks.swarm_mean_shift import run
from benchmarks.swarm_mean_shift import spacing_followup
from benchmarks.swarm_mean_shift.spacing_evaluate import add_spacing_metrics, PRIMARY_METRIC


VARIANTS = ("baseline", "rational_delta")


def audit_reset_geometry(old_base: Any, spaced_base: Any, args: Any) -> dict[str, Any]:
    """Confirm exact 2x coordinate scaling on the actual fresh evaluation keys."""
    keys = jax.random.split(jax.random.key(args.eval_reset_seed), args.eval_envs)
    old_env = vectorise_env(old_base, n=args.eval_envs)
    new_env = vectorise_env(spaced_base, n=args.eval_envs)
    old_env = old_env.reset(old_env, keys)
    new_env = new_env.reset(new_env, keys)
    previous = jax.device_get((old_env.state.pos, old_env.env_params["objective"]))
    current = jax.device_get((new_env.state.pos, new_env.env_params["objective"]))
    maximum_error = 0.0
    for label, old_array, new_array in zip(("agents", "objectives"), previous, current):
        error = float(np.max(np.abs(np.asarray(new_array) - 2.0 * np.asarray(old_array))))
        maximum_error = max(maximum_error, error)
        if not np.allclose(new_array, 2.0 * np.asarray(old_array), rtol=0, atol=2e-5):
            raise ValueError(f"{label} evaluation coordinates are not doubled: max error {error}")
    np.testing.assert_array_equal(jax.device_get(old_env.state.rad), jax.device_get(new_env.state.rad))
    np.testing.assert_array_equal(
        jax.device_get(old_env.env_params["lidar_range"]),
        jax.device_get(new_env.env_params["lidar_range"]),
    )
    return {
        "spacing_scale": 2.0,
        "maximum_coordinate_scaling_error": maximum_error,
        "agent_radii_unchanged": True,
        "lidar_range_unchanged": True,
        "original_layout_sha256": hashlib.sha256(
            b"".join(np.asarray(array).tobytes() for array in previous)
        ).hexdigest(),
        "spaced_layout_sha256": hashlib.sha256(
            b"".join(np.asarray(array).tobytes() for array in current)
        ).hexdigest(),
    }


def evaluate_one(args: Any, variant: str, seed: int) -> dict[str, Any]:
    source = args.input / f"{variant}_seed_{seed}"
    source_config = source / "config.json"
    checkpoint = source / "trainer_leaves.pkl"
    source_result = source / "result.json"
    if not all(path.is_file() for path in (source_config, checkpoint, source_result)):
        raise FileNotFoundError(f"Completed original run/config/checkpoint required: {source}")
    directory = args.output / source.name
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite zero-shot evaluation: {directory}")
    config = json.loads(source_config.read_text())
    completed = json.loads(source_result.read_text())
    if config["variant"] != variant or config["seed"] != seed:
        raise ValueError("Original source identity differs from its directory")
    if config["epochs"] != completed["epochs"]:
        raise ValueError("Original source run did not complete its declared training budget")
    if config["N"] != 64 or config["num_objectives"] != 64:
        raise ValueError("This comparison requires 64 agents and 64 objectives")
    if args.eval_reset_seed == config["eval_reset_seed"] or args.eval_action_seed == config["eval_action_seed"]:
        raise ValueError("Zero-shot evaluation keys must differ from original screening keys")
    original_root = args.original_root or original.recorded_repository_root(config)
    checked_sources = original.validate_source_hashes(
        config, runtime_root=args.runtime_root, original_root=original_root,
    )
    original_actions = (config["max_physics_steps"] + config["skip_frames"]) // (config["skip_frames"] + 1)
    maximum_actions = (args.max_physics_steps + config["skip_frames"]) // (config["skip_frames"] + 1)
    actions = args.eval_actions if args.eval_actions is not None else maximum_actions
    if actions > maximum_actions:
        raise ValueError(f"Evaluation would exceed its actual episode horizon: {actions} > {maximum_actions}")
    started = time.perf_counter()
    print(json.dumps({"event": "zero_shot_restore", "source_run": str(source), "actions": actions}), flush=True)
    # No spacing factory patch here: restoration must use the ORIGINAL trainer.
    trainer, original_base_env, restoration = original.rebuild_and_restore(config, checkpoint)
    checks = original.restored_checks(trainer)
    base_env = spacing_followup.make_followup_environment(
        f"spacing_{variant}", N=config["N"], num_objectives=config["num_objectives"],
        max_steps=args.max_physics_steps,
    )
    if base_env.observation_space_size != original_base_env.observation_space_size:
        raise ValueError("Zero-shot policy and spaced environment have different observation shapes")
    geometry = audit_reset_geometry(original_base_env, base_env, args)
    evaluation_config = {
        **config,
        "zero_shot": True,
        "evaluation_only": True,
        "source_run": str(source),
        "source_config_sha256": original.sha256(source_config),
        "checkpoint_sha256": original.sha256(checkpoint),
        "validated_source_hashes": checked_sources,
        "restoration": restoration,
        "geometry_audit": geometry,
        "original_training_episode_actions": original_actions,
        "original_training_max_physics_steps": config["max_physics_steps"],
        "max_physics_steps": args.max_physics_steps,
        "eval_actions": actions,
        "eval_envs": args.eval_envs,
        "eval_reset_seed": args.eval_reset_seed,
        "eval_action_seed": args.eval_action_seed,
        "eval_curve_stride": args.eval_curve_stride,
        "action_mode": args.action_mode or config["action_mode"],
        "output": str(args.output),
        "runtime_root": str(args.runtime_root),
        "original_root": str(original_root),
        "primary_metric": PRIMARY_METRIC,
        "confirmation_devices": [str(device) for device in jax.devices()],
        "confirmation_source_hashes": args.confirmation_source_hashes,
    }
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(evaluation_config, indent=2) + "\n")
    evaluation_args = argparse.Namespace(**evaluation_config)
    evaluation_args.output = args.output
    evaluation = run.evaluate(trainer, base_env, evaluation_args, directory, completed["epochs"])
    evaluation = add_spacing_metrics(evaluation, directory, completed["epochs"])
    if evaluation["layout_sha256"] != geometry["spaced_layout_sha256"]:
        raise ValueError("Evaluation layout differs from the independently audited scaled layout")
    result = {
        "variant": variant,
        "seed": seed,
        "epochs": completed["epochs"],
        "zero_shot": True,
        "evaluation_only": True,
        "source_run": str(source),
        "checkpoint_sha256": evaluation_config["checkpoint_sha256"],
        "wall_seconds": time.perf_counter() - started,
        **checks,
        "geometry_audit": geometry,
        "final_evaluation": evaluation,
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    run.append_json(args.output / "summary.jsonl", result)
    return result


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
    parser.add_argument("--max-physics-steps", type=int, default=200000)
    parser.add_argument("--eval-reset-seed", type=int, default=50000)
    parser.add_argument("--eval-action-seed", type=int, default=60000)
    parser.add_argument("--eval-curve-stride", type=int, default=100)
    parser.add_argument("--action-mode", choices=("stochastic", "deterministic"))
    args = parser.parse_args()
    if min(args.eval_envs, args.max_physics_steps, args.eval_curve_stride) < 1 or (
        args.eval_actions is not None and args.eval_actions < 1
    ):
        parser.error("Environment count, episode duration, actions, and curve stride must be positive")
    if any(seed < 0 for seed in (*args.seeds, args.eval_reset_seed, args.eval_action_seed)):
        parser.error("Seeds must be nonnegative")
    for attribute in ("input", "output", "runtime_root", "original_root"):
        value = getattr(args, attribute)
        if value is not None:
            setattr(args, attribute, value.resolve())
    if args.input == args.output:
        parser.error("Zero-shot output must differ from the original training directory")
    return args


def main() -> None:
    args = parse_args()
    original.validate_runtime_imports(args.runtime_root)
    sources = [Path(module.__file__).resolve() for module in (original, run, spacing_followup)]
    sources.extend((Path(__file__).resolve(), Path(__file__).with_name("spacing_evaluate.py").resolve()))
    for source in sources:
        if not source.is_relative_to(args.runtime_root):
            raise ValueError(f"Zero-shot helper must come from the frozen runtime: {source}")
    args.confirmation_source_hashes = {str(source): original.sha256(source) for source in sources}
    args.output.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        for seed in args.seeds:
            evaluate_one(args, variant, seed)


if __name__ == "__main__":
    main()
