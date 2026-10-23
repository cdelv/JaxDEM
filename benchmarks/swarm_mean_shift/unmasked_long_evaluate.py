# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Give unmasked swarm policies more evaluation time without retraining.

With the default --time-multiplier 2, original-spacing checkpoints run for
4000 actions / 200000 physics steps and doubled-spacing checkpoints run for
8000 actions / 400000 physics steps. The original checkpoint, spacing, reset
geometry, sensors, action mapping, and policy weights stay fixed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from benchmarks.swarm_mean_shift import (
    unmasked_compare,
    unmasked_evaluate,
    unmasked_variants,
    evaluate_checkpoints as original,
    run,
    spacing_evaluate,
    spacing_long_evaluate,
)


def evaluate_one(args: Any, variant: str, seed: int) -> None:
    source = args.input / f"{variant}_seed_{seed}"
    config_path, checkpoint, result_path = (
        source / "config.json",
        source / "trainer_leaves.pkl",
        source / "result.json",
    )
    if not all(path.is_file() for path in (config_path, checkpoint, result_path)):
        if args.skip_missing:
            print(
                json.dumps({"event": "skip_missing", "source_run": str(source)}),
                flush=True,
            )
            return
        raise FileNotFoundError(
            f"Completed checkpoint/config/result required: {source}"
        )
    directory = args.output / source.name
    if directory.exists():
        raise FileExistsError(
            f"Refusing to overwrite extended confirmation: {directory}"
        )
    config = json.loads(config_path.read_text())
    completed = json.loads(result_path.read_text())
    if (
        config["variant"] != variant
        or config["seed"] != seed
        or completed["epochs"] != config["epochs"]
    ):
        raise ValueError(
            "Checkpoint identity or completed training budget differs from config"
        )
    unmasked_compare.validate_mask_config(config, variant)
    if config["N"] != 64 or config["num_objectives"] != 64:
        raise ValueError("The unmasked comparison requires 64 agents and 64 objectives")
    if (
        args.eval_reset_seed == config["eval_reset_seed"]
        or args.eval_action_seed == config["eval_action_seed"]
    ):
        raise ValueError("Extended confirmation keys must differ from screening keys")
    for filename in (
        "unmasked_variants.py",
        "unmasked_compare.py",
        "unmasked_evaluate.py",
    ):
        if not any(path.endswith("/" + filename) for path in config["source_hashes"]):
            raise ValueError(f"Checkpoint lacks critical source hash: {filename}")
    original_root = args.original_root or original.recorded_repository_root(config)
    checked = original.validate_source_hashes(
        config,
        runtime_root=args.runtime_root,
        original_root=original_root,
    )
    training_steps = config["max_physics_steps"]
    evaluation_steps = training_steps * args.time_multiplier
    frames = config["skip_frames"] + 1
    maximum_actions = (evaluation_steps + frames - 1) // frames
    actions = args.eval_actions if args.eval_actions is not None else maximum_actions
    if actions > maximum_actions or actions * frames <= training_steps:
        raise ValueError(
            "Extended actions must exceed the training horizon and fit the real evaluation horizon"
        )
    started = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "unmasked_extended_restore",
                "run": source.name,
                "actions": actions,
            }
        ),
        flush=True,
    )
    # Reconstruct and restore with the ORIGINAL horizon/factory before creating
    # a separate evaluation-only environment with a longer real time limit.
    with unmasked_evaluate.matched_unmasked_confirmation():
        trainer, training_base, restoration = original.rebuild_and_restore(
            config, checkpoint
        )
    checks = original.restored_checks(trainer)
    evaluation_base = unmasked_compare.checked_environment(
        variant,
        N=config["N"],
        num_objectives=config["num_objectives"],
        max_steps=evaluation_steps,
    )
    expected_spacing = float(variant.rsplit("_s", 1)[1])
    if (
        float(training_base.spacing_scale) != expected_spacing
        or float(evaluation_base.spacing_scale) != expected_spacing
    ):
        raise ValueError("Extending time changed the checkpoint's spacing")
    if int(jax.device_get(evaluation_base.env_params["max_steps"])) != evaluation_steps:
        raise ValueError(
            "Evaluation environment did not receive the longer actual time limit"
        )
    count = evaluation_base.system.step_count
    evaluation_base.system.step_count = jnp.asarray(training_steps)
    if bool(jax.device_get(evaluation_base.truncated(evaluation_base))):
        raise ValueError(
            "Extended scene would freeze at the original training boundary"
        )
    evaluation_base.system.step_count = count
    geometry = spacing_long_evaluate.audit_equal_layouts(
        training_base, evaluation_base, args
    )
    evaluation_config = {
        **config,
        "evaluation_only": True,
        "extended_evaluation": True,
        "source_run": str(source),
        "source_config_sha256": original.sha256(config_path),
        "checkpoint_sha256": original.sha256(checkpoint),
        "validated_source_hashes": checked,
        "restoration": restoration,
        "geometry_audit": geometry,
        "training_max_physics_steps": training_steps,
        "training_episode_actions": (training_steps + frames - 1) // frames,
        "max_physics_steps": evaluation_steps,
        "time_multiplier": args.time_multiplier,
        "spacing_scale": expected_spacing,
        "eval_actions": actions,
        "eval_envs": args.eval_envs,
        "eval_reset_seed": args.eval_reset_seed,
        "eval_action_seed": args.eval_action_seed,
        "eval_curve_stride": args.eval_curve_stride,
        "action_mode": args.action_mode or config["action_mode"],
        "output": str(args.output),
        "runtime_root": str(args.runtime_root),
        "original_root": str(original_root),
        "primary_metric": spacing_evaluate.PRIMARY_METRIC,
        "primary_window_note": "Extended horizon's final 10 percent, separate from original-horizon confirmation",
        "confirmation_source_hashes": args.confirmation_source_hashes,
        "confirmation_devices": [str(device) for device in jax.devices()],
    }
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(evaluation_config, indent=2) + "\n"
    )
    evaluation_args = argparse.Namespace(**evaluation_config)
    evaluation_args.output = args.output
    old_rollout = run.evaluation_rollout
    observed: dict[str, Any] = {}

    def checked_rollout(*arguments: Any, **kwargs: Any) -> Any:
        final_env, graphstate, metrics = old_rollout(*arguments, **kwargs)
        counts = np.asarray(jax.device_get(final_env.system.step_count))
        expected = min(actions * frames, evaluation_steps)
        if not np.all(counts == expected) or not np.all(counts > training_steps):
            raise ValueError(
                f"Extended rollout stopped early: {counts.tolist()}, expected {expected}"
            )
        if bool(jax.device_get(jnp.any(final_env.system.collider.overflow))):
            raise ValueError("Collider overflow during extended evaluation")
        observed.update(
            {
                "final_physics_step_counts": counts.tolist(),
                "continued_past_training_horizon": True,
            }
        )
        return final_env, graphstate, metrics

    run.evaluation_rollout = checked_rollout
    try:
        evaluation = run.evaluate(
            trainer, evaluation_base, evaluation_args, directory, completed["epochs"]
        )
    finally:
        run.evaluation_rollout = old_rollout
    evaluation = spacing_evaluate.add_spacing_metrics(
        evaluation, directory, completed["epochs"]
    )
    if evaluation["layout_sha256"] != geometry["layout_sha256"]:
        raise ValueError(
            "Extended evaluation differs from the audited unchanged reset layout"
        )
    result = {
        "variant": variant,
        "seed": seed,
        "epochs": completed["epochs"],
        "evaluation_only": True,
        "extended_evaluation": True,
        "source_run": str(source),
        "checkpoint_sha256": evaluation_config["checkpoint_sha256"],
        "wall_seconds": time.perf_counter() - started,
        **checks,
        **observed,
        "training_max_physics_steps": training_steps,
        "evaluation_max_physics_steps": evaluation_steps,
        "time_multiplier": args.time_multiplier,
        "spacing_scale": expected_spacing,
        "geometry_audit": geometry,
        "final_evaluation": evaluation,
    }
    (directory / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    run.append_json(args.output / "summary.jsonl", result)


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=unmasked_compare.VARIANT_NAMES,
        default=[
            name for name in unmasked_compare.VARIANT_NAMES if name.endswith("_s1")
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--eval-actions", type=int)
    parser.add_argument("--time-multiplier", type=int, default=2)
    parser.add_argument("--eval-reset-seed", type=int, default=90000)
    parser.add_argument("--eval-action-seed", type=int, default=100000)
    parser.add_argument("--eval-curve-stride", type=int, default=100)
    parser.add_argument("--action-mode", choices=("stochastic", "deterministic"))
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    if args.time_multiplier < 2:
        parser.error("time-multiplier must be an integer of at least 2")
    if min(args.eval_envs, args.eval_curve_stride) < 1 or (
        args.eval_actions is not None and args.eval_actions < 1
    ):
        parser.error("Evaluation sizes and durations must be positive")
    if any(
        seed < 0 for seed in (*args.seeds, args.eval_reset_seed, args.eval_action_seed)
    ):
        parser.error("Seeds must be nonnegative")
    for attribute in ("input", "output", "runtime_root", "original_root"):
        value = getattr(args, attribute)
        if value is not None:
            setattr(args, attribute, value.resolve())
    if args.input == args.output:
        parser.error("Extended confirmation output must differ from training input")
    return args


def main() -> None:
    args = parse_args()
    original.validate_runtime_imports(args.runtime_root)
    sources = [
        Path(module.__file__).resolve()
        for module in (
            original,
            run,
            unmasked_variants,
            unmasked_compare,
            unmasked_evaluate,
            spacing_evaluate,
            spacing_long_evaluate,
        )
    ]
    sources.append(Path(__file__).resolve())
    for source in sources:
        if not source.is_relative_to(args.runtime_root):
            raise ValueError(
                f"Extended confirmation source must come from frozen runtime: {source}"
            )
    args.confirmation_source_hashes = {
        str(source): original.sha256(source) for source in sources
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        for seed in args.seeds:
            evaluate_one(args, variant, seed)


if __name__ == "__main__":
    main()
