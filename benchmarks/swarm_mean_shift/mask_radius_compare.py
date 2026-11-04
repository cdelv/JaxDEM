# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Matched N=64 comparisons of the original and 20% wider objective mask.

Run each spacing separately inside a frozen runtime. Original spacing defaults
to 120 training epochs and 2000 evaluation actions; doubled spacing should use
240 epochs, a 4320-epoch schedule, and 200000 maximum physics steps. Every final
checkpoint is also evaluated with twice its ordinary episode time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from math import isclose
from pathlib import Path
import sys
from typing import Any, Iterator

from benchmarks.swarm_mean_shift import run
from benchmarks.swarm_mean_shift.conservative_variants import ConservativeSwarmNavigator
from benchmarks.swarm_mean_shift.spacing_evaluate import add_spacing_metrics

VARIANTS = {
    f"{family}_s{spacing}": (spacing, radius)
    for spacing in (1, 2)
    for family, radius in (("original", 1.0), ("wider", 1.2))
}
VARIANT_NAMES = tuple(VARIANTS)
PRIMARY_METRIC = "coverage_final_10_percent_mean"
CLI_DEFAULTS = {
    "--variants": ("original_s1", "wider_s1"),
    "--seeds": ("10", "11", "12", "13", "14"),
    "--epochs": ("120",),
    "--schedule-epochs": ("2160",),
    "--max-physics-steps": ("100000",),
    "--N": ("64",),
    "--num-objectives": ("64",),
    "--eval-envs": ("64",),
    "--eval-at": (),
    "--eval-reset-seed": ("110000",),
    "--eval-action-seed": ("120000",),
    "--action-mode": ("stochastic",),
}


def variant_metadata(variant: str) -> dict[str, Any]:
    spacing, radius = VARIANTS[variant]
    return {
        "occupancy_radius_scale": radius,
        "spacing_scale": float(spacing),
        "hide_occupied_objectives": True,
        "vacancy_aware_lidar": False,
        "approach_kernel": "legacy",
        "reward_mode": "baseline",
        "observation_mode": "baseline",
        "peer_wall_encoding": "unsigned",
        "primary_metric": PRIMARY_METRIC,
        "extended_episode_time_multiplier": 2,
        "ppo_entropy_coeff": 1e-3,
    }


def checked_environment(variant: str, **kwargs: Any) -> ConservativeSwarmNavigator:
    """Change only the peer-to-objective masking tolerance at each spacing."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown masking-radius variant: {variant!r}")
    metadata = variant_metadata(variant)
    controls = {
        name: metadata[name]
        for name in (
            "occupancy_radius_scale",
            "spacing_scale",
            "hide_occupied_objectives",
            "vacancy_aware_lidar",
            "approach_kernel",
            "reward_mode",
            "observation_mode",
        )
    }
    for name, expected in controls.items():
        if name in kwargs and kwargs.pop(name) != expected:
            raise ValueError(f"{name} is fixed at {expected!r} for {variant}")
    env = ConservativeSwarmNavigator.Create(**controls, **kwargs)
    verify_environment(env, variant)
    return env


def verify_environment(env: Any, variant: str) -> None:
    expected = variant_metadata(variant)
    for name in ("occupancy_radius_scale", "hide_occupied_objectives"):
        actual = float(jax.device_get(env.env_params[name]))
        if not isclose(actual, float(expected[name]), rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"Actual {variant} {name} differs: {actual}")
    for name in (
        "spacing_scale",
        "vacancy_aware_lidar",
        "approach_kernel",
        "reward_mode",
        "observation_mode",
    ):
        if getattr(env, name) != expected[name]:
            raise ValueError(f"Actual {variant} {name} differs")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def matched_radius_runner(*, skip_completed: bool = False) -> Iterator[Any]:
    old_factory, old_names = run.make_environment, run.VARIANT_NAMES
    old_hashes, old_evaluate, old_run_one = run.source_hashes, run.evaluate, run.run_one
    old_rollout = run.evaluation_rollout
    observed_endpoints: list[dict[str, int]] = []

    def observed_rollout(*args: Any, **kwargs: Any) -> Any:
        endpoint, model_state, metrics = old_rollout(*args, **kwargs)
        counts = jax.device_get(endpoint.system.step_count)
        expected = kwargs["actions"] * (kwargs["skip_frames"] + 1)
        minimum, maximum = int(counts.min()), int(counts.max())
        if minimum != expected or maximum != expected:
            raise ValueError(
                f"Evaluation stopped before its requested horizon: "
                f"steps={minimum}..{maximum}, expected={expected}"
            )
        observed_endpoints.append(
            {
                "endpoint_physics_step_count_min": minimum,
                "endpoint_physics_step_count_max": maximum,
                "requested_endpoint_physics_steps": expected,
            }
        )
        return endpoint, model_state, metrics

    def evaluate_once(
        trainer: Any,
        env: Any,
        args: Any,
        directory: Path,
        epoch: int,
    ) -> dict[str, Any]:
        observed_endpoints.clear()
        result = old_evaluate(trainer, env, args, directory, epoch)
        if len(observed_endpoints) != 1:
            raise ValueError("Expected exactly one observed evaluation rollout")
        result.update(observed_endpoints[0])
        return add_spacing_metrics(result, directory, epoch)

    def hashes(source_experiment: Path) -> dict[str, str | None]:
        result = old_hashes(source_experiment)
        for name in (
            "mask_radius_compare.py",
            "conservative_variants.py",
            "spacing_evaluate.py",
        ):
            path = Path(__file__).with_name(name).resolve()
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def evaluate_with_extended_time(
        trainer: Any,
        base_env: Any,
        args: Any,
        directory: Path,
        epoch: int,
    ) -> dict[str, Any]:
        verify_environment(base_env, args.variant)
        actual_entropy = float(jax.device_get(trainer.ppo_entropy_coeff))
        if not isclose(
            actual_entropy, args.ppo_entropy_coeff, rel_tol=1e-6, abs_tol=1e-10
        ):
            raise ValueError(
                f"Unexpected actual PPO entropy coefficient: {actual_entropy}"
            )
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        config.update(
            {
                "actual_occupancy_radius_scale": float(
                    jax.device_get(base_env.env_params["occupancy_radius_scale"])
                ),
                "actual_ppo_entropy_coeff": actual_entropy,
                "actual_environment_controls_verified": True,
            }
        )
        _write_json(config_path, config)
        result = evaluate_once(trainer, base_env, args, directory, epoch)
        if epoch == args.epochs:
            extended_args = copy.copy(args)
            extended_args.eval_actions = 2 * args.eval_actions
            extended_args.max_physics_steps = 2 * args.max_physics_steps
            extended_env = jax.tree.map(lambda leaf: leaf, base_env)
            extended_env.env_params["max_steps"] = jnp.asarray(
                extended_args.max_physics_steps,
                dtype=base_env.env_params["max_steps"].dtype,
            )
            extended_dir = directory / "extended"
            extended_dir.mkdir()
            extended_config = {
                **config,
                "evaluation_only": True,
                "eval_actions": extended_args.eval_actions,
                "max_physics_steps": extended_args.max_physics_steps,
                "checkpoint_training_epochs": epoch,
                "checkpoint_run": str(directory),
            }
            _write_json(extended_dir / "config.json", extended_config)
            extended = evaluate_once(
                trainer, extended_env, extended_args, extended_dir, epoch
            )
            if extended["layout_sha256"] != result["layout_sha256"]:
                raise ValueError("Extended evaluation changed the initial layouts")
            _write_json(extended_dir / "result.json", extended)
            # Preserve the ordinary rollout as the primary final evaluation;
            # the extended rollout is a separately labelled sensitivity check.
            result["extended_evaluation"] = extended
            log_path = directory / "evaluation.jsonl"
            lines = log_path.read_text().splitlines()
            lines[-1] = json.dumps(result, allow_nan=False)
            log_path.write_text("\n".join(lines) + "\n")
        return result

    def run_with_metadata(
        args: Any,
        variant: str,
        seed: int,
        graphdefs: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = {**variant_metadata(variant), "variant": variant}
        missing = object()
        previous = {key: getattr(args, key, missing) for key in metadata}
        for key, value in metadata.items():
            setattr(args, key, value)
        try:
            directory = args.output / f"{variant}_seed_{seed}"
            if skip_completed and (directory / "result.json").exists():
                config = json.loads((directory / "config.json").read_text())
                expected_config = vars(args).copy()
                expected_config.update(
                    {
                        "seed": seed,
                        "output": str(args.output),
                        "source_experiment": str(args.source_experiment),
                        "source_hashes": hashes(args.source_experiment),
                    }
                )
                for key, expected in expected_config.items():
                    if config.get(key) != expected:
                        raise ValueError(f"Cannot resume {directory}: mismatched {key}")
                result = json.loads((directory / "result.json").read_text())
                required = [
                    directory / f"evaluation_epoch_{args.epochs:04d}.npz",
                    directory / "extended" / "result.json",
                    directory / "extended" / f"evaluation_epoch_{args.epochs:04d}.npz",
                ]
                if args.checkpoint:
                    required.extend(
                        directory / name
                        for name in (
                            "graphstate.pkl",
                            "trainer_leaves.pkl",
                        )
                    )
                if not all(path.exists() for path in required):
                    raise ValueError(
                        f"Cannot resume {directory}: missing completed-run artifacts"
                    )
                if not (
                    result["epochs"] == args.epochs
                    and result["parameters_finite"]
                    and result["physical_state_finite"]
                    and not result["collider_overflow"]
                    and result["final_evaluation"]["extended_evaluation"]
                ):
                    raise ValueError(
                        f"Cannot resume {directory}: incomplete or invalid result"
                    )
                print(
                    json.dumps({"event": "skip_completed", "run": directory.name}),
                    flush=True,
                )
                return result
            return old_run_one(args, variant, seed, graphdefs)
        finally:
            for key, value in previous.items():
                if value is missing:
                    delattr(args, key)
                else:
                    setattr(args, key, value)

    run.make_environment, run.VARIANT_NAMES = checked_environment, VARIANT_NAMES
    run.source_hashes, run.evaluate, run.run_one = (
        hashes,
        evaluate_with_extended_time,
        run_with_metadata,
    )
    run.evaluation_rollout = observed_rollout
    try:
        yield run
    finally:
        run.make_environment, run.VARIANT_NAMES = old_factory, old_names
        run.source_hashes, run.evaluate, run.run_one = (
            old_hashes,
            old_evaluate,
            old_run_one,
        )
        run.evaluation_rollout = old_rollout


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--skip-completed", action="store_true")
    extra, arguments = parser.parse_known_args(sys.argv[1:])
    defaults: list[str] = []
    for flag, values in CLI_DEFAULTS.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in arguments):
            defaults.extend((flag, *values))
    original_argv = sys.argv
    sys.argv = [original_argv[0], *defaults, *arguments]
    try:
        with matched_radius_runner(skip_completed=extra.skip_completed) as runner:
            args = runner.parse_args()
            if len({VARIANTS[name][0] for name in args.variants}) != 1:
                parser.error(
                    "Train each spacing separately with its explicit duration and training budget"
                )
            args.output.mkdir(parents=True, exist_ok=True)
            graphdefs: dict[str, Any] = {}
            # Alternate conditions within each seed to reduce run-order drift.
            for seed in args.seeds:
                for variant in args.variants:
                    runner.run_one(args, variant, seed, graphdefs)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
