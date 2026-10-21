"""Summarize completed swarm comparisons without treating episodes as seeds.

Example::

    python -m benchmarks.swarm_mean_shift.summarize --input /tmp/swarm_compare

Uncertainty is the sample standard deviation and standard error across training
seeds. A single training seed has no estimated training-seed uncertainty. Paired
deltas use matching seeds, evaluation layouts, action modes, and training budgets.
Incomplete run directories are ignored and no numerical results are invented.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jaxdem-swarm-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


EVALUATION_METRICS = (
    "coverage_endpoint_mean",
    "coverage_time_average_mean",
    "coverage_last_200_actions_mean",
    "speed_endpoint_mean",
    "speed_time_average_mean",
    "nearest_agent_distance_radii_endpoint_mean",
    "nearest_agent_distance_radii_time_average_mean",
    "overlapping_agent_fraction_endpoint_mean",
    "overlapping_agent_fraction_time_average_mean",
)
RUN_METRICS = (
    "training_seconds",
    "wall_seconds",
    "first_epoch_seconds",
    "steady_epoch_seconds",
    "model_parameter_count",
    "observation_space_size",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def statistics(values: list[float]) -> dict[str, float | int | None]:
    """Sample statistics across independent training runs, never episodes."""
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0, "mean": None, "std": None, "se": None}
    if not np.isfinite(array).all():
        raise ValueError("Cannot summarize nonfinite measurements")
    deviation = float(array.std(ddof=1)) if len(array) > 1 else None
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "std": deviation,
        "se": deviation / math.sqrt(len(array)) if deviation is not None else None,
    }


def load_runs(source: Path) -> list[dict[str, Any]]:
    # The runner writes one immediate child directory per variant/seed. Avoid
    # inadvertently combining pilot runs or unrelated nested result collections.
    paths = sorted(source.glob("*/result.json"))
    if not paths:
        raise ValueError(f"No completed runs found at {source}/*/result.json")
    runs = []
    identities = set()
    for path in paths:
        result = json.loads(path.read_text())
        evaluation = result.get("final_evaluation")
        if not evaluation:
            print(f"Skipping run without final evaluation: {path}", file=sys.stderr)
            continue
        identity = (result["variant"], result["seed"])
        if identity in identities:
            raise ValueError(f"Duplicate variant/seed {identity}; select one results directory")
        identities.add(identity)
        directory = path.parent
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        training = read_jsonl(directory / "training.jsonl")
        metrics = {key: float(evaluation[key]) for key in EVALUATION_METRICS if key in evaluation}
        metrics.update({key: float(result[key]) for key in ("training_seconds", "wall_seconds") if key in result})
        metrics.update({key: float(config[key]) for key in ("model_parameter_count", "observation_space_size") if key in config})
        if training:
            metrics["first_epoch_seconds"] = float(training[0]["wall_seconds"])
            if len(training) > 1:
                metrics["steady_epoch_seconds"] = float(np.mean([row["wall_seconds"] for row in training[1:]]))
        for key, value in metrics.items():
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite {key} in {path}")
        runs.append({
            "variant": result["variant"], "seed": result["seed"],
            "epochs": result["epochs"], "directory": str(directory.resolve()),
            "evaluation_epoch": evaluation["epoch"],
            "evaluation_actions": evaluation["actions"],
            "evaluation_envs": evaluation["num_envs"],
            "action_mode": evaluation["action_mode"],
            "layout_sha256": evaluation["layout_sha256"],
            "physics_steps_per_action": evaluation["physics_steps_per_action"],
            "metrics": metrics,
            "checks": {key: result.get(key) for key in ("physical_state_finite", "parameters_finite", "collider_overflow")},
            "config": config,
            "evaluations": read_jsonl(directory / "evaluation.jsonl") or [evaluation],
        })
    if not runs:
        raise ValueError("No completed runs contain final evaluations")
    return runs


def comparison_signature(run: dict[str, Any]) -> tuple[Any, ...]:
    """Conditions that must agree before computing a paired difference."""
    configuration = run["config"]
    training_fields = (
        "N", "num_objectives", "num_envs", "steps_per_epoch", "skip_frames",
        "max_physics_steps", "schedule_epochs", "eval_action_seed",
    )
    return (
        run["epochs"], run["evaluation_epoch"], run["evaluation_actions"],
        run["evaluation_envs"], run["action_mode"], run["layout_sha256"],
        run["physics_steps_per_action"],
        *(configuration.get(key) for key in training_fields),
    )


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["variant"]].append(run)
    variants = {}
    warnings = []
    baseline = {run["seed"]: run for run in grouped.get("baseline", [])}
    for variant, members in sorted(grouped.items()):
        members.sort(key=lambda run: run["seed"])
        keys = sorted(set().union(*(run["metrics"] for run in members)))
        measurements = {
            key: statistics([run["metrics"][key] for run in members if key in run["metrics"]])
            for key in keys
        }
        if len({comparison_signature(run) for run in members}) > 1:
            warnings.append(f"{variant} has different run/evaluation conditions across seeds; inspect per_seed before interpreting its aggregate.")
        pairs = []
        if variant != "baseline":
            for run in members:
                reference = baseline.get(run["seed"])
                if reference is None:
                    continue
                if comparison_signature(run) != comparison_signature(reference):
                    warnings.append(f"Excluded {variant} seed {run['seed']} from paired comparison: conditions differ from baseline.")
                    continue
                differences = {key: value - reference["metrics"][key]
                               for key, value in run["metrics"].items() if key in reference["metrics"]}
                pairs.append({"seed": run["seed"], "candidate_minus_baseline": differences})
        paired_keys = sorted(set().union(*(pair["candidate_minus_baseline"] for pair in pairs))) if pairs else []
        variants[variant] = {
            "num_training_seeds": len(members), "seeds": [run["seed"] for run in members],
            "epochs": sorted({run["epochs"] for run in members}), "metrics": measurements,
            "paired_vs_baseline": {
                "num_pairs": len(pairs),
                "delta_definition": "candidate minus baseline; positive coverage delta is an improvement",
                "metrics": {key: statistics([pair["candidate_minus_baseline"][key] for pair in pairs if key in pair["candidate_minus_baseline"]]) for key in paired_keys},
                "per_seed": pairs,
            },
            "per_seed": [{key: value for key, value in run.items() if key not in {"config", "evaluations"}} for run in members],
        }
    return {
        "uncertainty": "Sample standard deviation (ddof=1) and standard error across training seeds; null for fewer than two seeds. Evaluation episodes are not independent training seeds.",
        "completed_runs": len(runs), "warnings": warnings, "variants": variants,
    }


def write_csv(aggregate: dict[str, Any], destination: Path) -> None:
    keys = sorted(set().union(*(item["metrics"] for item in aggregate["variants"].values())))
    columns = ["variant", "training_seeds", "seeds", "epochs"]
    columns += [f"{key}_{stat}" for key in keys for stat in ("mean", "std", "se")]
    paired_keys = sorted(set().union(*(item["paired_vs_baseline"]["metrics"] for item in aggregate["variants"].values())))
    columns += ["paired_baseline_seeds"]
    columns += [f"delta_{key}_{stat}" for key in paired_keys for stat in ("mean", "std", "se")]
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for variant, item in aggregate["variants"].items():
            row = {"variant": variant, "training_seeds": item["num_training_seeds"],
                   "seeds": json.dumps(item["seeds"]), "epochs": json.dumps(item["epochs"]),
                   "paired_baseline_seeds": item["paired_vs_baseline"]["num_pairs"]}
            for key, values in item["metrics"].items():
                row.update({f"{key}_{stat}": values[stat] for stat in ("mean", "std", "se")})
            for key, values in item["paired_vs_baseline"]["metrics"].items():
                row.update({f"delta_{key}_{stat}": values[stat] for stat in ("mean", "std", "se")})
            writer.writerow(row)


def plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
        "figure.dpi": 130, "savefig.dpi": 180, "legend.frameon": False,
    })


def color_map(variants: list[str]) -> dict[str, Any]:
    palette = plt.get_cmap("tab10" if len(variants) <= 10 else "tab20")
    result = {variant: palette(index % palette.N) for index, variant in enumerate(variants)}
    if "baseline" in result:
        result["baseline"] = "#222222"
    return result


def line_with_seed_uncertainty(ax: Any, x: np.ndarray, values: np.ndarray, *, color: Any, label: str) -> None:
    means = values.mean(axis=0)
    ax.plot(x, means, label=label, color=color, linewidth=2.1)
    if len(values) > 1:
        errors = values.std(axis=0, ddof=1) / math.sqrt(len(values))
        ax.fill_between(x, means - errors, means + errors, color=color, alpha=0.15, linewidth=0)


def plot_coverage(runs: list[dict[str, Any]], destination: Path, colors: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), constrained_layout=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["variant"]].append(run)
    plotted = False
    for variant, members in sorted(grouped.items()):
        by_length: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for run in members:
            by_length[(run["evaluation_epoch"], run["evaluation_actions"])].append(run)
        for (epoch, actions), matching in sorted(by_length.items()):
            coverage, speeds = [], []
            for run in matching:
                path = Path(run["directory"]) / f"evaluation_epoch_{epoch:04d}.npz"
                if not path.exists():
                    continue
                with np.load(path) as arrays:
                    coverage.append(np.asarray(arrays["coverage"]).mean(axis=1))
                    speeds.append(np.asarray(arrays["speed"]).mean(axis=1))
            if not coverage:
                continue
            plotted = True
            # Plot short time windows so 2,000 action trajectories remain clear.
            stride = max(1, math.ceil(actions / 100))
            windows = [(start, min(start + stride, actions)) for start in range(0, actions, stride)]
            x = np.asarray([(start + 1 + end) / 2 for start, end in windows])
            smooth = lambda data: np.asarray([[row[start:end].mean() for start, end in windows] for row in data])
            extra = f", epoch {epoch}" if len(by_length) > 1 else ""
            label = f"{variant.replace('_', ' ')} (n={len(coverage)}{extra})"
            line_with_seed_uncertainty(axes[0], x, smooth(coverage), color=colors[variant], label=label)
            line_with_seed_uncertainty(axes[1], x, smooth(speeds), color=colors[variant], label=label)
    axes[0].set(title="Distinct goals covered during held-out episodes", xlabel="Policy actions", ylabel="Coverage", ylim=(0, 1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(title="Agent speed during the same episodes", xlabel="Policy actions", ylabel="Mean speed (simulation units / second)")
    axes[1].set_ylim(bottom=0)
    if plotted:
        axes[0].legend(fontsize=8, loc="lower right")
    else:
        axes[0].text(0.5, 0.5, "Evaluation trajectory files unavailable", ha="center", va="center", transform=axes[0].transAxes)
    fig.suptitle("Final evaluation · means across training seeds; shading = ±1 SE", fontsize=12)
    fig.savefig(destination)
    plt.close(fig)


def plot_learning(runs: list[dict[str, Any]], destination: Path, colors: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), constrained_layout=True)
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        # Last record wins if a resumed run repeated an evaluation epoch.
        evaluations = {evaluation["epoch"]: evaluation for evaluation in run["evaluations"]}
        for epoch, evaluation in evaluations.items():
            grouped[run["variant"]][epoch].append(evaluation)
    for variant, epochs in sorted(grouped.items()):
        x = np.asarray(sorted(epochs))
        for ax, metric in zip(axes, ("coverage_endpoint_mean", "coverage_time_average_mean")):
            summaries = [statistics([evaluation[metric] for evaluation in epochs[epoch]]) for epoch in x]
            means = np.asarray([item["mean"] for item in summaries], dtype=float)
            errors = np.asarray([item["se"] if item["se"] is not None else 0 for item in summaries])
            counts = sorted({item["n"] for item in summaries})
            label = f"{variant.replace('_', ' ')} (n={','.join(map(str, counts))})"
            ax.plot(x, means, color=colors[variant], label=label, linewidth=2, marker="o", markersize=3.7)
            if any(item["se"] is not None for item in summaries):
                ax.fill_between(x, means - errors, means + errors, color=colors[variant], alpha=0.15, linewidth=0)
    for ax, title in zip(axes, ("Endpoint coverage", "Episode time-average coverage")):
        ax.set(title=title, xlabel="Training epochs", ylabel="Held-out coverage", ylim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Learning curves · means across training seeds; shading = ±1 SE", fontsize=12)
    fig.savefig(destination)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    destination = (args.output or source).resolve()
    try:
        runs = load_runs(source)
        aggregate = aggregate_runs(runs)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    destination.mkdir(parents=True, exist_ok=True)
    aggregate["source_directory"] = str(source)
    (destination / "aggregate.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")
    write_csv(aggregate, destination / "summary.csv")
    plot_style()
    colors = color_map(sorted(aggregate["variants"]))
    plot_coverage(runs, destination / "coverage_curves.png", colors)
    plot_learning(runs, destination / "learning_curves.png", colors)
    for warning in aggregate["warnings"]:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"Summarized {len(runs)} completed runs across {len(aggregate['variants'])} variants in {destination}")


if __name__ == "__main__":
    main()
