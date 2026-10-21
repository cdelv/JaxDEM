"""Evaluate trusted local swarm checkpoints on independent confirmation scenes.

No training is performed. Reconstruct the original trainer and restore its full
saved leaves, then evaluate a copied model with fresh environment/action keys::

    python -m benchmarks.swarm_mean_shift.evaluate_checkpoints \
      --input benchmarks/results/swarm-mean-shift/2026-10-20 \
      --output /tmp/swarm_confirmation --variants baseline vacancy_lidar \
      --seeds 0 1 2 --eval-envs 64

Pickle checkpoints must come from these trusted local experiments. Runtime and
variant source hashes, initial geometry, and restored leaf schemas are checked
before evaluation. Existing confirmation directories are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import jaxdem
import jaxdem.rl as rl
from benchmarks.swarm_mean_shift import run
from benchmarks.swarm_mean_shift.compare import make_any_environment


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_repository_root(config: dict[str, Any]) -> Path:
    suffix = "/jaxdem/rl/environments/swarm_navigator.py"
    candidates = [Path(filename[:-len(suffix)]) for filename in config.get("source_hashes", {}) if filename.endswith(suffix)]
    if len(candidates) != 1:
        raise ValueError("Cannot infer the original repository root from checkpoint source hashes")
    return candidates[0].resolve()


def mapped_source(path: Path, runtime_root: Path | None, original_root: Path | None) -> Path:
    if runtime_root is not None and original_root is not None and path.is_relative_to(original_root):
        return runtime_root / path.relative_to(original_root)
    return path


def validate_runtime_imports(runtime_root: Path | None) -> None:
    if runtime_root is None:
        return
    for label, module in (("jaxdem", jaxdem), ("run", run)):
        actual = Path(module.__file__).resolve()
        if not actual.is_relative_to(runtime_root):
            raise ValueError(f"Imported {label} must come from the frozen runtime {runtime_root}; got {actual}")


def validate_source_hashes(
    config: dict[str, Any], *, runtime_root: Path | None = None,
    original_root: Path | None = None,
) -> dict[str, str]:
    """Require recorded runtime, runner, and variant code before unpickling."""
    expected = config.get("source_hashes", {})
    required_suffixes = [
        "/jaxdem/rl/environments/swarm_navigator.py",
        "/jaxdem/rl/environments/_swarm.py",
        "/jaxdem/rl/trainers/ppo_trainer.py",
        "/jaxdem/rl/models/mingru.py",
        "/jaxdem/rl/action_spaces/max_norm_space.py",
        "/benchmarks/swarm_mean_shift/variants.py",
    ]
    followup = {
        "rational_delta": "reward_followups.py",
        "rational_scaled": "reward_followups.py",
        "paper_augmented": "observation_followups.py",
        "rational_delta_vacancy": "combined_followup.py",
    }.get(config["variant"])
    if followup:
        required_suffixes.append(f"/benchmarks/swarm_mean_shift/{followup}")
    for suffix in required_suffixes:
        if not any(path.endswith(suffix) for path in expected):
            raise ValueError(f"Checkpoint config lacks critical source hash: {suffix}")
    checked = {}
    for filename, digest in expected.items():
        path = mapped_source(Path(filename), runtime_root, original_root)
        if not path.is_file() or digest is None:
            raise ValueError(f"Critical checkpoint source is unavailable: {path}")
        actual = sha256(path)
        if actual != digest:
            raise ValueError(f"Checkpoint source hash differs from runtime file: {path}")
        checked[str(path)] = actual
    return checked


def rebuild_and_restore(config: dict[str, Any], checkpoint: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Create a matching graph definition and restore complete trainer leaves."""
    variant = config["variant"]
    factory = make_any_environment
    if variant == "rational_delta_vacancy":
        from benchmarks.swarm_mean_shift.combined_followup import make_followup_environment

        factory = make_followup_environment
    base_env = factory(
        variant, N=config["N"], num_objectives=config["num_objectives"],
        max_steps=config["max_physics_steps"],
    )
    model = run.make_model(base_env, config["model_seed"])
    trainer = rl.Trainer.create(
        "PPO", env=base_env, model=model, seed=config["trainer_seed"],
        num_steps_epoch=config["steps_per_epoch"], num_envs=config["num_envs"],
        num_epochs=config["schedule_epochs"], stop_at_epoch=config["epochs"],
        skip_frames=config["skip_frames"], num_minibatches=8,
        learning_rate=1e-3, caps_temporal_coeff=1e-4,
        caps_spatial_coeff=1e-3, caps_noise_std=0.01,
    )
    layout = jax.device_get((trainer.env.state.pos, trainer.env.env_params["objective"]))
    layout_hash = hashlib.sha256(b"".join(np.asarray(array).tobytes() for array in layout)).hexdigest()
    # CPU and CUDA can round the reset arithmetic differently. Exact hashes
    # validate same-backend reconstruction; cross-backend smoke checks record
    # the difference while source hashes and leaf schemas remain mandatory.
    current_backend = str(jax.devices()[0]).split(":")[0]
    original_backend = str(config.get("devices", [""])[0]).split(":")[0]
    cross_backend = current_backend != original_backend
    layout_matches = layout_hash == config.get("training_initial_layout_sha256")
    if not layout_matches and not cross_backend:
        raise ValueError("Reconstructed initial training layout differs from checkpoint config")
    if base_env.observation_space_size != config["observation_space_size"]:
        raise ValueError("Reconstructed observation shape differs from checkpoint config")
    parameter_count = sum(value.size for value in jax.tree.leaves(nnx.state(model, nnx.Param)))
    if parameter_count != config["model_parameter_count"]:
        raise ValueError("Reconstructed parameter count differs from checkpoint config")
    # Only trusted locally generated experiment checkpoints are accepted here.
    with checkpoint.open("rb") as stream:
        saved_leaves = pickle.load(stream)
    template_leaves, definition = jax.tree.flatten(trainer)
    if len(template_leaves) != len(saved_leaves):
        raise ValueError(f"Checkpoint leaf count {len(saved_leaves)} differs from template {len(template_leaves)}")
    for index, (template, saved) in enumerate(zip(template_leaves, saved_leaves)):
        # Python counters/scalars become typed arrays during the first JIT
        # epoch. Compare canonical JAX dtypes rather than requiring Python
        # scalars to expose an array dtype before training has occurred.
        template_dtype = jnp.asarray(template).dtype
        saved_dtype = jnp.asarray(saved).dtype
        if np.shape(template) != np.shape(saved) or template_dtype != saved_dtype:
            raise ValueError(
                f"Checkpoint leaf {index} shape/dtype differs from matching trainer template: "
                f"saved {np.shape(saved)} {saved_dtype}, "
                f"template {np.shape(template)} {template_dtype}"
            )
    trainer = jax.tree.unflatten(definition, jax.device_put(saved_leaves))
    return trainer, base_env, {
        "reconstructed_initial_layout_sha256": layout_hash,
        "initial_layout_hash_matches": layout_matches,
        "cross_backend_restore": cross_backend,
        "restored_leaf_count": len(saved_leaves),
    }


def restored_checks(trainer: Any) -> dict[str, bool]:
    model, *_ = nnx.merge(trainer.graphdef, trainer.graphstate, copy=True)
    leaves = jax.tree.leaves(nnx.state(model, nnx.Param))
    checks = {
        "physical_state_finite": bool(jax.device_get(
            jnp.all(jnp.isfinite(trainer.env.state.pos)) & jnp.all(jnp.isfinite(trainer.env.state.vel))
        )),
        "parameters_finite": bool(jax.device_get(jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(value)) for value in leaves
        ])))),
        "collider_overflow": bool(jax.device_get(jnp.any(trainer.env.system.collider.overflow))),
    }
    if not checks["physical_state_finite"] or not checks["parameters_finite"] or checks["collider_overflow"]:
        raise ValueError(f"Restored checkpoint failed state checks: {checks}")
    return checks


def evaluate_one(args: Any, variant: str, seed: int) -> None:
    source = args.input / f"{variant}_seed_{seed}"
    source_result = source / "result.json"
    checkpoint = source / "trainer_leaves.pkl"
    config_path = source / "config.json"
    if not all(path.is_file() for path in (source_result, checkpoint, config_path)):
        if args.skip_missing:
            print(json.dumps({"event": "skip_missing", "source_run": str(source)}), flush=True)
            return
        raise FileNotFoundError(f"Completed run/config/checkpoint required: {source}")
    directory = args.output / source.name
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite confirmation directory: {directory}")
    original = json.loads(config_path.read_text())
    completed = json.loads(source_result.read_text())
    if original["variant"] != variant or original["seed"] != seed:
        raise ValueError(f"Source run identity differs from directory: {source}")
    if completed["epochs"] != original["epochs"]:
        raise ValueError(f"Source completed epochs differ from config: {source}")
    original_root = args.original_root or recorded_repository_root(original)
    validate_runtime_imports(args.runtime_root)
    checked_sources = validate_source_hashes(
        original, runtime_root=args.runtime_root, original_root=original_root,
    )
    actions = args.eval_actions
    maximum_actions = (original["max_physics_steps"] + original["skip_frames"]) // (original["skip_frames"] + 1)
    if actions is None:
        actions = maximum_actions
    if actions > maximum_actions:
        raise ValueError(f"Confirmation evaluation exceeds original episode length: {actions} > {maximum_actions}")
    if args.eval_reset_seed == original["eval_reset_seed"] or args.eval_action_seed == original["eval_action_seed"]:
        raise ValueError("Confirmation reset and action seeds must differ from screening seeds")
    started = time.perf_counter()
    print(json.dumps({"event": "restore", "source_run": str(source), "devices": [str(device) for device in jax.devices()]}), flush=True)
    trainer, base_env, restoration = rebuild_and_restore(original, checkpoint)
    checks = restored_checks(trainer)
    directory.mkdir(parents=True)
    config = original.copy()
    config.update({
        "output": str(args.output), "source_run": str(source), "evaluation_only": True,
        "eval_envs": args.eval_envs, "eval_actions": actions,
        "eval_reset_seed": args.eval_reset_seed, "eval_action_seed": args.eval_action_seed,
        "action_mode": args.action_mode or original["action_mode"],
        "eval_curve_stride": args.eval_curve_stride,
        "checkpoint_sha256": sha256(checkpoint), "origin_config_sha256": sha256(config_path),
        "validated_source_hashes": checked_sources,
        "confirmation_source_hashes": {str(path): sha256(path) for path in (Path(__file__).resolve(), Path(run.__file__).resolve())},
        "confirmation_devices": [str(device) for device in jax.devices()],
        "restoration": restoration,
        "runtime_root": str(args.runtime_root) if args.runtime_root is not None else None,
        "original_root": str(original_root),
        "source_path_mapping": {
            filename: str(mapped_source(Path(filename), args.runtime_root, original_root))
            for filename in original["source_hashes"]
        },
    })
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    evaluation_args = argparse.Namespace(**config)
    evaluation_args.output = args.output
    evaluation_args.source_experiment = Path(config["source_experiment"])
    evaluation = run.evaluate(trainer, base_env, evaluation_args, directory, completed["epochs"])
    result = {
        "variant": variant, "seed": seed, "epochs": completed["epochs"],
        "evaluation_only": True, "source_run": str(source),
        "checkpoint_sha256": config["checkpoint_sha256"],
        "wall_seconds": time.perf_counter() - started,
        **checks, "final_evaluation": evaluation,
    }
    temporary = directory / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(directory / "result.json")
    run.append_json(args.output / "summary.jsonl", result)


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, help="Frozen runtime root; imported jaxdem and runner must originate here")
    parser.add_argument("--original-root", type=Path, help="Original repository root recorded in source hashes (inferred by default)")
    parser.add_argument("--variants", nargs="+", default=["baseline", "vacancy_lidar", "rational_delta", "paper_augmented"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--eval-actions", type=int)
    parser.add_argument("--eval-reset-seed", type=int, default=30000)
    parser.add_argument("--eval-action-seed", type=int, default=40000)
    parser.add_argument("--eval-curve-stride", type=int, default=100)
    parser.add_argument("--action-mode", choices=("stochastic", "deterministic"))
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    if args.eval_envs < 1 or args.eval_curve_stride < 1 or (args.eval_actions is not None and args.eval_actions < 1):
        parser.error("Evaluation environments, actions, and curve stride must be positive")
    if any(seed < 0 for seed in (*args.seeds, args.eval_reset_seed, args.eval_action_seed)):
        parser.error("Seeds must be nonnegative")
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if args.runtime_root is not None:
        args.runtime_root = args.runtime_root.resolve()
    if args.original_root is not None:
        args.original_root = args.original_root.resolve()
    if args.original_root is not None and args.runtime_root is None:
        parser.error("--original-root requires --runtime-root")
    if args.input == args.output:
        parser.error("Confirmation output must differ from the training results directory")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        for seed in args.seeds:
            evaluate_one(args, variant, seed)


if __name__ == "__main__":
    main()
