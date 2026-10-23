# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Evaluate doubled-spacing checkpoints for 8000 actions without retraining.

The trainer is restored with its original 200000-step episode. Only a new
evaluation environment receives a 400000-step horizon. Reset coordinates,
particle radii, sensing, forces and action randomness remain unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from jaxdem.rl.env_wrappers import vectorise_env
from benchmarks.swarm_mean_shift import evaluate_checkpoints as original
from benchmarks.swarm_mean_shift import run, spacing_evaluate, spacing_followup

VARIANTS = ("spacing_baseline", "spacing_rational_delta")


def audit_equal_layouts(
    training_base: Any, evaluation_base: Any, args: Any
) -> dict[str, Any]:
    keys = jax.random.split(jax.random.key(args.eval_reset_seed), args.eval_envs)
    environments = []
    for base in (training_base, evaluation_base):
        env = vectorise_env(base, n=args.eval_envs)
        environments.append(env.reset(env, keys))
    old, new = environments
    for label, first, second in (
        ("agents", old.state.pos, new.state.pos),
        ("objectives", old.env_params["objective"], new.env_params["objective"]),
        ("radii", old.state.rad, new.state.rad),
        ("lidar_range", old.env_params["lidar_range"], new.env_params["lidar_range"]),
        ("domain_size", old.system.domain.box_size, new.system.domain.box_size),
        ("domain_anchor", old.system.domain.anchor, new.system.domain.anchor),
    ):
        first, second = jax.device_get((first, second))
        if not np.array_equal(first, second):
            raise ValueError(f"Extended evaluation changed reset {label}")
    layout = jax.device_get((new.state.pos, new.env_params["objective"]))
    return {
        "training_and_evaluation_layouts_equal": True,
        "agent_radii_unchanged": True,
        "lidar_range_unchanged": True,
        "domain_unchanged": True,
        "layout_sha256": hashlib.sha256(
            b"".join(np.asarray(a).tobytes() for a in layout)
        ).hexdigest(),
    }


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
            f"Completed source checkpoint/config/result required: {source}"
        )
    directory = args.output / source.name
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite extended evaluation: {directory}")
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
    if config["N"] != 64 or config["num_objectives"] != 64:
        raise ValueError(
            "The spacing confirmation requires 64 agents and 64 objectives"
        )
    if config["max_physics_steps"] != 200000:
        raise ValueError("Expected the unchanged 200000-step training episode")
    if (
        args.eval_reset_seed == config["eval_reset_seed"]
        or args.eval_action_seed == config["eval_action_seed"]
    ):
        raise ValueError(
            "Extended confirmation keys must differ from training screening keys"
        )
    original_root = args.original_root or original.recorded_repository_root(config)
    checked = original.validate_source_hashes(
        config,
        runtime_root=args.runtime_root,
        original_root=original_root,
    )
    maximum_actions = (args.max_physics_steps + config["skip_frames"]) // (
        config["skip_frames"] + 1
    )
    actions = args.eval_actions if args.eval_actions is not None else maximum_actions
    if actions > maximum_actions:
        raise ValueError(
            "Requested evaluation actions exceed the real evaluation horizon"
        )
    if actions * (config["skip_frames"] + 1) <= config["max_physics_steps"]:
        raise ValueError(
            "An extended evaluation must continue past the original training horizon"
        )
    started = time.perf_counter()
    print(
        json.dumps(
            {"event": "extended_restore", "run": source.name, "actions": actions}
        ),
        flush=True,
    )
    # This context changes the factory only while reconstructing the original
    # trainer. All restored optimizer/environment leaves keep their old horizon.
    with spacing_evaluate.matched_spacing_confirmation():
        trainer, training_base, restoration = original.rebuild_and_restore(
            config, checkpoint
        )
    checks = original.restored_checks(trainer)
    evaluation_base = spacing_followup.make_followup_environment(
        variant,
        N=config["N"],
        num_objectives=config["num_objectives"],
        max_steps=args.max_physics_steps,
    )
    if float(evaluation_base.spacing_scale) != 2.0:
        raise ValueError("The extended evaluation must retain the same 2x spacing")
    if (
        int(jax.device_get(evaluation_base.env_params["max_steps"]))
        != args.max_physics_steps
    ):
        raise ValueError(
            "Evaluation environment did not receive its longer real horizon"
        )
    # Verify truncation itself remains false at the OLD boundary. Reset below
    # starts a fresh scene; this probe never changes the restored training env.
    original_count = evaluation_base.system.step_count
    evaluation_base.system.step_count = jnp.asarray(config["max_physics_steps"])
    if bool(jax.device_get(evaluation_base.truncated(evaluation_base))):
        raise ValueError("Extended evaluation would freeze at the original horizon")
    evaluation_base.system.step_count = original_count
    geometry = audit_equal_layouts(training_base, evaluation_base, args)
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
        "training_max_physics_steps": config["max_physics_steps"],
        "training_episode_actions": (
            config["max_physics_steps"] + config["skip_frames"]
        )
        // (config["skip_frames"] + 1),
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
        "primary_metric": spacing_evaluate.PRIMARY_METRIC,
        "primary_window_note": "Extended horizon's final 10 percent; distinct from the primary 4000-action confirmation",
        "confirmation_source_hashes": args.confirmation_source_hashes,
        "confirmation_devices": [str(device) for device in jax.devices()],
    }
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(evaluation_config, indent=2) + "\n"
    )
    evaluation_args = argparse.Namespace(**evaluation_config)
    evaluation_args.output = args.output
    original_rollout = run.evaluation_rollout
    observed: dict[str, Any] = {}

    def checked_rollout(*arguments: Any, **kwargs: Any) -> Any:
        final_env, graphstate, metrics = original_rollout(*arguments, **kwargs)
        counts = np.asarray(jax.device_get(final_env.system.step_count))
        expected = min(actions * (config["skip_frames"] + 1), args.max_physics_steps)
        if not np.all(counts == expected) or not np.all(
            counts > config["max_physics_steps"]
        ):
            raise ValueError(
                f"Extended rollout stopped early: counts={counts.tolist()}, expected={expected}"
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
        run.evaluation_rollout = original_rollout
    evaluation = spacing_evaluate.add_spacing_metrics(
        evaluation, directory, completed["epochs"]
    )
    if evaluation["layout_sha256"] != geometry["layout_sha256"]:
        raise ValueError(
            "Extended rollout layout differs from the audited unchanged layout"
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
        "training_max_physics_steps": config["max_physics_steps"],
        "evaluation_max_physics_steps": args.max_physics_steps,
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
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--eval-actions", type=int)
    parser.add_argument("--max-physics-steps", type=int, default=400000)
    parser.add_argument("--eval-reset-seed", type=int, default=50000)
    parser.add_argument("--eval-action-seed", type=int, default=60000)
    parser.add_argument("--eval-curve-stride", type=int, default=100)
    parser.add_argument("--action-mode", choices=("stochastic", "deterministic"))
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    if min(args.eval_envs, args.max_physics_steps, args.eval_curve_stride) < 1 or (
        args.eval_actions is not None and args.eval_actions < 1
    ):
        parser.error("Evaluation sizes and durations must be positive")
    if args.max_physics_steps <= 200000:
        parser.error("Extended evaluation must have more than 200000 physics steps")
    if any(
        seed < 0 for seed in (*args.seeds, args.eval_reset_seed, args.eval_action_seed)
    ):
        parser.error("Seeds must be nonnegative")
    for attribute in ("input", "output", "runtime_root", "original_root"):
        value = getattr(args, attribute)
        if value is not None:
            setattr(args, attribute, value.resolve())
    if args.input == args.output:
        parser.error("Extended evaluation output must differ from training input")
    return args


def main() -> None:
    args = parse_args()
    original.validate_runtime_imports(args.runtime_root)
    sources = [
        Path(module.__file__).resolve()
        for module in (original, run, spacing_followup, spacing_evaluate)
    ]
    sources.append(Path(__file__).resolve())
    for source in sources:
        if not source.is_relative_to(args.runtime_root):
            raise ValueError(
                f"Extended evaluation source must come from frozen runtime: {source}"
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
