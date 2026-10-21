"""Matched PPO comparisons of the existing swarm navigator and local variants.

Run from the repository root, for example::

    python -m benchmarks.swarm_mean_shift.run --variants baseline combined \
        --seeds 0 1 2 --epochs 120 --eval-at 20 60 120 --output /tmp/swarm_compare

This leaves the source experiment and production environments unchanged. The
original experiment's planned 2160-epoch learning-rate schedule is retained even
when training stops at 120 epochs. Rewards are diagnostic only: distinct-goal
coverage from simulator positions is the common comparison metric.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import jaxdem.rl as rl
from jaxdem import utils
from jaxdem.rl.env_wrappers import vectorise_env

from benchmarks.swarm_mean_shift.variants import make_environment


VARIANT_NAMES = (
    "baseline", "simple", "compact", "combined", "exact_compact", "exact_combined",
    "vacancy_lidar", "vacancy_combined", "paper_compact", "paper_combined",
)


def append_json(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def source_hashes(source_experiment: Path) -> dict[str, str | None]:
    files = [
        source_experiment,
        Path(__file__),
        Path(__file__).with_name("variants.py"),
        REPOSITORY / "jaxdem/rl/environments/swarm_navigator.py",
        REPOSITORY / "jaxdem/rl/environments/_swarm.py",
        REPOSITORY / "jaxdem/rl/trainers/ppo_trainer.py",
        REPOSITORY / "jaxdem/rl/models/mingru.py",
        REPOSITORY / "jaxdem/rl/models/__init__.py",
        REPOSITORY / "jaxdem/rl/action_spaces/__init__.py",
        REPOSITORY / "jaxdem/rl/action_spaces/max_norm_space.py",
        REPOSITORY / "jaxdem/rl/environments/__init__.py",
        REPOSITORY / "jaxdem/rl/env_wrappers/__init__.py",
        REPOSITORY / "jaxdem/utils/environment.py",
    ]
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        for path in files
    }


def exact_metrics(env: Any) -> dict[str, jax.Array]:
    """Per-environment metrics; every objective is counted at most once.

    SwarmNavigator uses reflective domains, so direct center distances are exact.
    This deliberately bypasses LiDAR bins, masking, and angular occlusion.
    """
    positions = env.state.pos
    objectives = env.env_params["objective"]
    radii = env.state.rad
    separation_sq = jnp.sum(
        (positions[..., :, None, :] - objectives[..., None, :, :]) ** 2,
        axis=-1,
    )
    covered = jnp.any(separation_sq <= radii[..., :, None] ** 2, axis=-2)
    coverage = jnp.mean(covered, axis=-1)
    nearest_radii = jnp.sqrt(jnp.min(separation_sq / radii[..., :, None] ** 2, axis=-2))
    speed = jnp.mean(jnp.linalg.norm(env.state.vel, axis=-1), axis=-1)
    pair_distance_sq = jnp.sum(
        (positions[..., :, None, :] - positions[..., None, :, :]) ** 2, axis=-1
    )
    overlap = (
        pair_distance_sq < jnp.maximum(radii[..., :, None] + radii[..., None, :] - 1e-4, 0) ** 2
    ) & ~jnp.eye(env.max_num_agents, dtype=bool)
    return {
        "coverage": coverage,
        "nearest_agent_distance_radii": nearest_radii.mean(axis=-1),
        "speed": speed,
        "overlapping_agent_fraction": jnp.any(overlap, axis=-1).mean(axis=-1),
        "finite_state": jnp.all(jnp.isfinite(positions), axis=(-2, -1))
        & jnp.all(jnp.isfinite(env.state.vel), axis=(-2, -1)),
    }


@partial(jax.jit, static_argnames=("graphdef", "actions", "skip_frames", "action_mode"))
def evaluation_rollout(
    env: Any,
    graphstate: Any,
    key: jax.Array,
    *,
    graphdef: Any,
    actions: int,
    skip_frames: int,
    action_mode: str,
) -> tuple[Any, Any, dict[str, jax.Array]]:
    """Collect small metric arrays, without retaining full simulator snapshots."""
    def step(carry: Any, _: None) -> tuple[Any, dict[str, jax.Array]]:
        current, model_state, random_key, terminated, truncated = carry
        random_key, action_key = jax.random.split(random_key)
        model = nnx.merge(graphdef, model_state)
        distribution, _ = model(current.observation(current), sequence=False)
        if action_mode == "deterministic":
            action = rl.Model.policy_output(distribution)
        else:
            action = distribution.sample(seed=action_key)
        model_state = nnx.state(model)
        current, terminated, truncated = utils.advance_action(
            current, action, skip_frames=skip_frames,
            terminated=terminated, truncated=truncated,
        )
        return (current, model_state, random_key, terminated, truncated), exact_metrics(current)

    shape = jnp.shape(env.done(env))
    carry = (env, graphstate, key, jnp.zeros(shape, bool), jnp.zeros(shape, bool))
    (env, graphstate, _, _, _), metrics = jax.lax.scan(step, carry, None, length=actions)
    return env, graphstate, metrics


def make_model(env: Any, seed: int) -> Any:
    return rl.Model.create(
        "minGRUActorCritic",
        key=nnx.Rngs(jax.random.key(seed)),
        observation_space_size=env.observation_space_size,
        action_space_size=env.action_space_size,
        action_space=rl.ActionSpace.create("maxNorm", max_norm=2.0),
        actor_sigma_head=False,
        hidden_features=64,
        gru_features=64,
    )


def evaluate(tr: Any, base_env: Any, args: Any, directory: Path, epoch: int) -> dict[str, Any]:
    started = time.perf_counter()
    env = vectorise_env(base_env, n=args.eval_envs)
    reset_keys = jax.random.split(jax.random.key(args.eval_reset_seed), args.eval_envs)
    env = env.reset(env, reset_keys)
    # Merge a copy and split the model alone. Training env, optimizer, RNG and
    # recurrent carry are all untouched by evaluation.
    model, *_ = nnx.merge(tr.graphdef, tr.graphstate, copy=True)
    model.eval()
    model.reset(shape=(args.eval_envs, base_env.max_num_agents, 1), mask=None)
    graphdef, graphstate = nnx.split(model)
    layout = jax.device_get((env.state.pos, env.env_params["objective"]))
    layout_hash = hashlib.sha256(b"".join(np.asarray(a).tobytes() for a in layout)).hexdigest()
    initial = jax.device_get(exact_metrics(env))
    _, _, metrics = evaluation_rollout(
        env, graphstate, jax.random.key(args.eval_action_seed), graphdef=graphdef,
        actions=args.eval_actions, skip_frames=args.skip_frames, action_mode=args.action_mode,
    )
    arrays = {key: np.asarray(value) for key, value in jax.device_get(metrics).items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("Nonfinite evaluation metrics")
    if not arrays["finite_state"].all():
        raise FloatingPointError("Nonfinite evaluation physical state")
    np.savez_compressed(directory / f"evaluation_epoch_{epoch:04d}.npz", **arrays)
    result: dict[str, Any] = {
        "epoch": epoch,
        "action_mode": args.action_mode,
        "num_envs": args.eval_envs,
        "actions": args.eval_actions,
        "physics_steps_per_action": 1 + args.skip_frames,
        "layout_sha256": layout_hash,
        "wall_seconds": time.perf_counter() - started,
        "initial_coverage": np.asarray(initial["coverage"]).tolist(),
    }
    for key, value in arrays.items():
        endpoint = value[-1]
        time_average = value.mean(axis=0)
        result[f"{key}_endpoint_mean"] = float(endpoint.mean())
        result[f"{key}_endpoint_std"] = float(endpoint.std())
        result[f"{key}_endpoint_per_episode"] = endpoint.tolist()
        result[f"{key}_time_average_mean"] = float(time_average.mean())
        result[f"{key}_time_average_std"] = float(time_average.std())
        result[f"{key}_time_average_per_episode"] = time_average.tolist()
    tail = arrays["coverage"][-min(200, args.eval_actions):].mean(axis=0)
    result["coverage_last_200_actions_mean"] = float(tail.mean())
    result["coverage_last_200_actions_per_episode"] = tail.tolist()
    append_json(directory / "evaluation.jsonl", result)
    for step in range(0, args.eval_actions, args.eval_curve_stride):
        end = min(step + args.eval_curve_stride, args.eval_actions)
        record: dict[str, Any] = {"epoch": epoch, "action_start": step + 1, "action_end": end}
        record.update({key: float(value[step:end].mean()) for key, value in arrays.items()})
        append_json(directory / "evaluation_curve.jsonl", record)
    print(json.dumps({"event": "evaluation", "run": directory.name, "epoch": epoch,
                      "coverage_endpoint": result["coverage_endpoint_mean"],
                      "coverage_time_average": result["coverage_time_average_mean"],
                      "wall_seconds": result["wall_seconds"]}), flush=True)
    return result


def run_one(args: Any, variant: str, seed: int, graphdefs: dict[str, Any]) -> dict[str, Any]:
    directory = args.output / f"{variant}_seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "training.jsonl").exists() or (directory / "result.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {directory}")
    config = vars(args).copy()
    config.update({"variant": variant, "seed": seed, "model_seed": seed + 1,
                   "trainer_seed": seed + 6, "output": str(args.output),
                   "source_experiment": str(args.source_experiment),
                   "source_hashes": source_hashes(args.source_experiment),
                   "jax_version": jax.__version__, "python_version": platform.python_version(),
                   "package_versions": {
                       name: importlib.metadata.version(name)
                       for name in ("jax", "jaxlib", "flax", "optax", "distrax", "numpy")
                   },
                   "devices": [str(device) for device in jax.devices()],
                   "coverage_cutoff_radii": 1.0,
                   "training_decisions_per_epoch": args.num_envs * args.N * args.steps_per_epoch})
    try:
        config["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        config["git_revision"] = None
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"event": "start", "variant": variant, "seed": seed,
                      "devices": config["devices"]}), flush=True)
    base_env = make_environment(variant, N=args.N, num_objectives=args.num_objectives,
                                max_steps=args.max_physics_steps)
    model = make_model(base_env, seed + 1)
    tr = rl.Trainer.create(
        "PPO", env=base_env, model=model, seed=seed + 6,
        num_steps_epoch=args.steps_per_epoch, num_envs=args.num_envs,
        num_epochs=args.schedule_epochs, stop_at_epoch=args.epochs,
        skip_frames=args.skip_frames, num_minibatches=8,
        learning_rate=1e-3, caps_temporal_coeff=1e-4,
        caps_spatial_coeff=1e-3, caps_noise_std=0.01,
    )
    # The optimizer graph includes freshly created callable objects. Reusing its
    # equivalent static definition across seeds avoids redundant JIT compilation.
    if variant in graphdefs:
        tr.graphdef = graphdefs[variant]
    else:
        graphdefs[variant] = tr.graphdef
    layout = jax.device_get((tr.env.state.pos, tr.env.env_params["objective"]))
    config.update({
        "observation_space_size": base_env.observation_space_size,
        "action_space_size": base_env.action_space_size,
        "training_initial_layout_sha256": hashlib.sha256(
            b"".join(np.asarray(a).tobytes() for a in layout)
        ).hexdigest(),
        "model_parameter_count": sum(
            value.size for value in jax.tree.leaves(nnx.state(model, nnx.Param))
        ),
    })
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    evaluation_epochs = set(args.eval_at) | {args.epochs}
    final_evaluation = None
    if 0 in evaluation_epochs:
        final_evaluation = evaluate(tr, base_env, args, directory, 0)
    started = time.perf_counter()
    train_seconds = 0.0
    for index in range(args.epochs):
        epoch_started = time.perf_counter()
        tr, trajectory, data = tr.epoch(tr, jnp.asarray(index, dtype=int))
        data["collider_overflow"] = jnp.any(tr.env.system.collider.overflow)
        host_data = jax.device_get(data)
        record = {key: float(value) for key, value in host_data.items()}
        if not all(np.isfinite(value) for value in record.values()):
            raise FloatingPointError(f"Nonfinite training metric at epoch {index + 1}: {record}")
        if record["collider_overflow"]:
            raise RuntimeError(f"Collider overflow after epoch {index + 1}; comparison is invalid")
        record.update({"epoch": index + 1, "wall_seconds": time.perf_counter() - epoch_started,
                       "agent_policy_decisions": (index + 1) * args.num_envs * args.N * args.steps_per_epoch})
        train_seconds += record["wall_seconds"]
        append_json(directory / "training.jsonl", record)
        del trajectory
        if index == 0 or (index + 1) % args.log_every == 0:
            print(json.dumps({"event": "training", "run": directory.name, **record}), flush=True)
        if index + 1 in evaluation_epochs:
            final_evaluation = evaluate(tr, base_env, args, directory, index + 1)
    overflow = bool(jax.device_get(jnp.any(tr.env.system.collider.overflow)))
    physical_state_finite = bool(jax.device_get(
        jnp.all(jnp.isfinite(tr.env.state.pos)) & jnp.all(jnp.isfinite(tr.env.state.vel))
    ))
    trained_model, *_ = nnx.merge(tr.graphdef, tr.graphstate, copy=True)
    parameter_leaves = jax.tree.leaves(nnx.state(trained_model, nnx.Param))
    parameters_finite = bool(jax.device_get(jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(value)) for value in parameter_leaves
    ]))))
    if overflow:
        raise RuntimeError("Collider overflow occurred; comparison is invalid")
    if not physical_state_finite or not parameters_finite:
        raise FloatingPointError("Nonfinite final physical state or model parameters")
    if args.checkpoint:
        with (directory / "graphstate.pkl").open("wb") as stream:
            pickle.dump(jax.device_get(tr.graphstate), stream)
        with (directory / "trainer_leaves.pkl").open("wb") as stream:
            pickle.dump(jax.device_get(jax.tree.leaves(tr)), stream)
        (directory / "checkpoint_readme.txt").write_text(
            "Trusted local pickle files only. Recreate a trainer with config.json, the same\n"
            "variant, model dimensions, seeds and PPO settings, then restore:\n"
            "  leaves = pickle.load(open('trainer_leaves.pkl', 'rb'))\n"
            "  tr = jax.tree.unflatten(jax.tree.structure(tr), leaves)\n"
            "This restores environment, PRNG, recurrent carry, parameters and optimizer.\n"
            "graphstate.pkl is also available for restoring just model/optimizer state.\n"
            "Continue via tr.epoch(tr, epoch_index); the 2160-epoch schedule is stored\n"
            "in the optimizer graph created by the matching trainer configuration.\n"
        )
    result = {"variant": variant, "seed": seed, "epochs": args.epochs,
              "wall_seconds": time.perf_counter() - started,
              "training_seconds": train_seconds,
              "physical_state_finite": physical_state_finite,
              "parameters_finite": parameters_finite,
              "collider_overflow": overflow, "final_evaluation": final_evaluation}
    temporary_result = directory / "result.json.tmp"
    temporary_result.write_text(json.dumps(result, indent=2) + "\n")
    temporary_result.replace(directory / "result.json")
    append_json(args.output / "summary.jsonl", result)
    return result


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANT_NAMES, default=["baseline", "simple", "compact", "combined"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--schedule-epochs", type=int, default=2160)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--num-objectives", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--skip-frames", type=int, default=49)
    parser.add_argument("--max-physics-steps", type=int, default=100000)
    parser.add_argument("--eval-envs", type=int, default=32)
    parser.add_argument("--eval-actions", type=int, default=None)
    parser.add_argument("--eval-at", nargs="*", type=int, default=[20, 60, 120])
    parser.add_argument("--eval-reset-seed", type=int, default=10000)
    parser.add_argument("--eval-action-seed", type=int, default=20000)
    parser.add_argument("--eval-curve-stride", type=int, default=100)
    parser.add_argument("--action-mode", choices=("stochastic", "deterministic"), default="stochastic")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source-experiment", type=Path, default=Path("/home/wind/Documents/experiments/intro_to_rl2.py"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/swarm_mean_shift"))
    args = parser.parse_args()
    if args.eval_actions is None:
        args.eval_actions = (args.max_physics_steps + args.skip_frames) // (args.skip_frames + 1)
    for name in ("epochs", "schedule_epochs", "N", "num_objectives", "num_envs", "steps_per_epoch",
                 "max_physics_steps", "eval_envs", "eval_actions", "eval_curve_stride", "log_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.skip_frames < 0 or args.epochs > args.schedule_epochs:
        parser.error("skip-frames must be nonnegative and epochs <= schedule-epochs")
    if args.eval_actions > (args.max_physics_steps + args.skip_frames) // (args.skip_frames + 1):
        parser.error("eval-actions exceeds the episode length")
    args.output = args.output.resolve()
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    graphdefs: dict[str, Any] = {}
    for variant in args.variants:
        for seed in args.seeds:
            run_one(args, variant, seed, graphdefs)


if __name__ == "__main__":
    main()
