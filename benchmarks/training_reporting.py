# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Compare reporting overhead against a previous training loop.

Run from the repository root with
``python -m benchmarks.training_reporting --output /tmp/reporting.json``.
Use ``--profile bandit`` for a small workload that exposes host overhead.
The navigator profile uses the MultiNavigator and MinGRU configuration.
Results include raw timings, compilation time, and paired trial ratios.
Set ``--ref`` to the commit before the reporting feature. The default uses
``HEAD`` to compare uncommitted changes.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import tempfile
import time
import types
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm.auto import trange

import jaxdem.rl as rl
from jaxdem.rl.trainers import ppo_trainer


def previous_train(ref: str) -> Any:
    source = subprocess.check_output(
        ["git", "show", f"{ref}:jaxdem/rl/trainers/ppo_trainer.py"], text=True
    )
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "PPOTrainer"
    )
    function = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    function.decorator_list = []
    namespace = dict(vars(ppo_trainer))
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, f"{ref}:PPOTrainer.train", "exec"), namespace)
    return namespace["train"]


def make_trainer(profile: str, epochs: int) -> Any:
    if profile == "bandit":
        from tests.test_ppo_math import BanditModel, TimeLimitCounterEnv

        return rl.Trainer.create(
            "PPO",
            env=TimeLimitCounterEnv.Create(),
            model=BanditModel(),
            key=jax.random.key(6),
            num_envs=32,
            num_steps_epoch=8,
            num_minibatches=1,
            num_epochs=epochs,
        )
    box = float(np.sqrt(64 * np.pi / 0.62))
    env = rl.Environment.create(
        "MultiNavigator",
        N=64,
        min_box_size=box,
        max_box_size=box,
        friction=2.0,
    )
    env.env_params["max_steps"] = jnp.asarray(160 / float(env.system.dt), dtype=int)
    model_key, trainer_key = jax.random.split(jax.random.key(6))
    model = rl.Model.create(
        "MinGRUActorCritic",
        key=nnx.Rngs(model_key),
        observation_space_size=env.observation_space_size,
        action_space_size=env.action_space_size,
        action_space=rl.ActionSpace.create("maxNorm", max_norm=2.0),
        actor_sigma_head=False,
        hidden_features=64,
        gru_features=64,
        num_layers=4,
        activation=nnx.identity,
    )
    return rl.Trainer.create(
        "PPO",
        env=env,
        model=model,
        key=trainer_key,
        num_envs=32,
        num_steps_epoch=64,
        skip_frames=49,
        num_minibatches=8,
        learning_rate=1e-3,
        anneal_learning_rate=False,
        num_epochs=epochs,
        caps_temporal_coeff=1e-4,
        caps_spatial_coeff=1e-3,
        caps_noise_std=0.01,
    )


def accuracy_metrics(tr: Any, trajectory: Any) -> dict[str, jax.Array]:
    del trajectory
    if "objective" not in tr.env.env_params:
        return {"accuracy": (tr.env.state.rad > 0).mean()}
    distance = jnp.linalg.norm(
        tr.env.state.pos_c - tr.env.env_params["objective"], axis=-1
    )
    return {"accuracy": (distance <= tr.env.state.rad).mean()}


def no_op_report(tr: Any, epoch: int, metrics: Any) -> None:
    return None


@jax.jit
def identical_state(a: Any, b: Any) -> jax.Array:
    return jnp.stack(
        [jnp.all(x == y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))]
    ).all()


def run_trial(
    function: Any, trainer: Any, directory: Path, options: dict[str, Any]
) -> tuple[Any, dict[str, float]]:
    first = []

    def clock() -> float:
        now = time.perf_counter()
        if not first:
            first.append(now)
        return now

    output = io.StringIO()
    namespace = dict(function.__globals__)
    namespace["time"] = types.SimpleNamespace(perf_counter=clock)
    namespace["trange"] = lambda *args, **kwargs: trange(
        *args, file=output, mininterval=1e9, **kwargs
    )
    isolated = types.FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    isolated.__kwdefaults__ = function.__kwdefaults__
    started = time.perf_counter()
    with contextlib.redirect_stdout(output):
        result = isolated(trainer, directory=directory, **options)
        jax.block_until_ready(result)
    finished = time.perf_counter()
    return result, {
        "wall_seconds": finished - started,
        "initial_report_seconds": first[0] - started,
        "steady_seconds": finished - first[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("bandit", "navigator"), default="navigator"
    )
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=(
            "no_hooks",
            "display",
            "accuracy",
            "accuracy_display",
            "accuracy_callback",
        ),
    )
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 2 or args.repeats < 1 or args.save_every < 1:
        parser.error(
            "Use at least two epochs, one repeat, and a positive reporting interval."
        )

    baseline_ref = subprocess.check_output(
        ["git", "rev-parse", args.ref], text=True
    ).strip()
    baseline = previous_train(baseline_ref)
    trainer = make_trainer(args.profile, args.epochs)
    print(f"Warmup: {args.profile}, {jax.devices()}", flush=True)
    compile_start = time.perf_counter()
    current = trainer
    for epoch in range(4):
        current, _, _ = current.epoch(current, jnp.asarray(epoch))
        jax.block_until_ready(current)
    compilation_seconds = time.perf_counter() - compile_start
    print(f"Compilation and four warmup epochs: {compilation_seconds:.3f}s", flush=True)
    del current
    cases = {
        "baseline": (baseline, {}),
        "no_hooks": (ppo_trainer.PPOTrainer.train, {}),
        "display": (
            ppo_trainer.PPOTrainer.train,
            {"progress_metrics": {"score": ".3f", "steps_per_sec": ".2e"}},
        ),
        "accuracy": (ppo_trainer.PPOTrainer.train, {"metrics_fn": accuracy_metrics}),
        "accuracy_display": (
            ppo_trainer.PPOTrainer.train,
            {
                "metrics_fn": accuracy_metrics,
                "progress_metrics": {
                    "accuracy": ".1%",
                    "score": ".3f",
                    "steps_per_sec": ".2e",
                },
            },
        ),
        "accuracy_callback": (
            ppo_trainer.PPOTrainer.train,
            {
                "metrics_fn": accuracy_metrics,
                "progress_metrics": {
                    "accuracy": ".1%",
                    "score": ".3f",
                    "steps_per_sec": ".2e",
                },
                "on_report": no_op_report,
            },
        ),
    }
    if args.cases is not None:
        cases = {
            name: case
            for name, case in cases.items()
            if name == "baseline" or name in args.cases
        }
    data = {
        "profile": args.profile,
        "epochs": args.epochs,
        "repeats": args.repeats,
        "save_every": args.save_every,
        "jax_version": jax.__version__,
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "platform": platform.platform(),
        "baseline_ref": baseline_ref,
        "source_sha256": hashlib.sha256(
            Path(ppo_trainer.__file__).read_bytes()
        ).hexdigest(),
        "cases": list(cases),
        "paired_baseline": args.paired,
        "tensorboard_logging": True,
        "progress_output": "StringIO",
        "compilation_and_warmup_seconds": compilation_seconds,
        "trials": [],
    }
    rng = np.random.default_rng(6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference = None
    with tempfile.TemporaryDirectory(prefix="jaxdem-reporting-bench-") as temporary:
        for repeat in range(args.repeats):
            order = [name for name in cases if not args.paired or name != "baseline"]
            rng.shuffle(order)
            groups = [["baseline", name] for name in order] if args.paired else [order]
            for group_index, group in enumerate(groups):
                if args.paired:
                    rng.shuffle(group)
                rows = []
                for name in group:
                    function, options = cases[name]
                    result, times = run_trial(
                        function,
                        trainer,
                        Path(temporary) / f"{repeat}-{group_index}-{name}",
                        {"save_every": args.save_every, **options},
                    )
                    if reference is None:
                        reference = result
                    elif not bool(identical_state(reference, result)):
                        raise AssertionError(f"Training state changed in {name}")
                    rows.append(
                        {"repeat": repeat, "pair": group_index, "case": name, **times}
                    )
                    print(
                        f"Trial {repeat + 1}/{args.repeats}: {name} {times['steady_seconds']:.4f}s",
                        flush=True,
                    )
                baseline_seconds = next(
                    row["steady_seconds"] for row in rows if row["case"] == "baseline"
                )
                for row in rows:
                    row["baseline_seconds"] = baseline_seconds
                data["trials"].extend(rows)
                args.output.write_text(json.dumps(data, indent=2) + "\n")
    summaries = {}
    for name in cases:
        values = [
            row["steady_seconds"] for row in data["trials"] if row["case"] == name
        ]
        rows = [row for row in data["trials"] if row["case"] == name]
        ratios = [row["steady_seconds"] / row["baseline_seconds"] for row in rows]
        bootstrap = np.median(rng.choice(ratios, size=(10000, len(ratios))), axis=1)
        summaries[name] = {
            "median_seconds": float(np.median(values)),
            "median_paired_ratio": float(np.median(ratios)),
            "min_seconds": min(values),
            "max_seconds": max(values),
            "paired_ratio_95ci": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            "median_initial_report_seconds": float(
                np.median([row["initial_report_seconds"] for row in rows])
            ),
        }
    data["summary"] = summaries
    data["identical_training_state"] = True
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
