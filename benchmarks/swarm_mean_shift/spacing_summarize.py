"""Summarize completed doubled-spacing runs without counting episodes as seeds.

    python -m benchmarks.swarm_mean_shift.spacing_summarize --input RESULTS

Only immediate child ``result.json`` files are read. Both fresh-training names
(``spacing_baseline``, ``spacing_rational_delta``) and zero-shot names
(``baseline``, ``rational_delta``) are supported. Pairing requires the same
training seed, evaluation scenes and randomness, horizon, and training budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jaxdem-spacing-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy.stats import t as student_t


PRIMARY_METRIC = "coverage_final_10_percent_mean"
METRICS = (
    PRIMARY_METRIC,
    "coverage_endpoint_mean",
    "coverage_action_2000_mean",
    "coverage_action_4000_mean",
    "coverage_action_8000_mean",
    "coverage_time_average_mean",
    "coverage_last_200_actions_mean",
    "speed_endpoint_mean",
    "speed_time_average_mean",
    "overlapping_agent_fraction_endpoint_mean",
    "overlapping_agent_fraction_time_average_mean",
    "nearest_agent_distance_radii_endpoint_mean",
    "nearest_agent_distance_radii_time_average_mean",
)
ALIASES = {
    "spacing_baseline": "baseline",
    "baseline": "baseline",
    "spacing_rational_delta": "rational_delta",
    "rational_delta": "rational_delta",
}
LABELS = {"baseline": "Baseline", "rational_delta": "Simpler reward"}
COLORS = {"baseline": "#444444", "rational_delta": "#007f80"}
CONFIG_CONDITIONS = (
    "N", "num_objectives", "num_envs", "steps_per_epoch", "skip_frames",
    "max_physics_steps", "schedule_epochs", "eval_reset_seed", "eval_action_seed",
    "observation_space_size", "action_space_size", "model_parameter_count",
)


def statistics(values: list[float]) -> dict[str, Any]:
    """Sample statistics and Student-t mean interval across training seeds."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Measurements must be a finite one-dimensional sequence")
    n = len(array)
    if not n:
        return {"n": 0, "mean": None, "std": None, "se": None, "ci95": None}
    mean = float(array.mean())
    deviation = float(array.std(ddof=1)) if n > 1 else None
    error = deviation / math.sqrt(n) if deviation is not None else None
    interval = None
    if error is not None:
        margin = float(student_t.ppf(0.975, n - 1)) * error
        interval = [mean - margin, mean + margin]
    return {"n": n, "mean": mean, "std": deviation, "se": error, "ci95": interval}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_runs(source: Path) -> list[dict[str, Any]]:
    runs = []
    identities = set()
    for path in sorted(source.glob("*/result.json")):
        result = json.loads(path.read_text())
        evaluation = result.get("final_evaluation")
        if not evaluation:
            continue
        original_variant = result["variant"]
        if original_variant not in ALIASES:
            raise ValueError(f"Unsupported spacing variant {original_variant!r}: {path}")
        variant, seed = ALIASES[original_variant], result["seed"]
        if (variant, seed) in identities:
            raise ValueError(f"Duplicate {variant} seed {seed}; select one experiment collection")
        identities.add((variant, seed))
        config = json.loads((path.parent / "config.json").read_text())
        if config.get("variant") != original_variant or config.get("seed") != seed:
            raise ValueError(f"Result/config variant or seed identity differs: {path}")
        if result["epochs"] != config["epochs"] or evaluation["epoch"] != result["epochs"]:
            raise ValueError(f"Result/config completed training epochs differ: {path}")
        if PRIMARY_METRIC not in evaluation:
            raise ValueError(f"Missing declared primary metric {PRIMARY_METRIC}: {path}")
        metrics = {key: float(evaluation[key]) for key in METRICS if key in evaluation}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Nonfinite evaluation measurement: {path}")
        checks = {key: result.get(key) for key in (
            "physical_state_finite", "parameters_finite", "collider_overflow"
        )}
        if checks["physical_state_finite"] is False or checks["parameters_finite"] is False or checks["collider_overflow"] is True:
            raise ValueError(f"Run failed physical/parameter validation: {path}: {checks}")
        runs.append({
            "variant": variant, "original_variant": original_variant, "seed": seed,
            "epochs": result["epochs"], "zero_shot": bool(result.get("zero_shot", False)),
            "directory": str(path.parent.resolve()), "metrics": metrics, "checks": checks,
            "config": config, "evaluation": evaluation,
            "evaluations": read_jsonl(path.parent / "evaluation.jsonl") or [evaluation],
        })
    if not runs:
        raise ValueError(f"No completed spacing runs at {source}/*/result.json")
    if len({run["zero_shot"] for run in runs}) != 1:
        raise ValueError("Select separate result collections for zero-shot and retrained models")
    return runs


def comparison_conditions(run: dict[str, Any]) -> dict[str, Any]:
    evaluation, config = run["evaluation"], run["config"]
    conditions = {key: config.get(key) for key in CONFIG_CONDITIONS}
    conditions.update({key: evaluation.get(key) for key in (
        "epoch", "actions", "num_envs", "action_mode", "layout_sha256",
        "physics_steps_per_action", "coverage_final_10_percent_actions",
    )})
    conditions.update({"epochs": run["epochs"], "zero_shot": run["zero_shot"]})
    # Zero-shot configs retain the actual original training time limit in this
    # field while max_physics_steps describes the extended evaluation horizon.
    if run["zero_shot"]:
        conditions["original_training_max_physics_steps"] = config.get("original_training_max_physics_steps")
        conditions["original_training_episode_actions"] = config.get("original_training_episode_actions")
    return conditions


def pairing_problems(candidate: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    left, right = comparison_conditions(candidate), comparison_conditions(reference)
    problems = [key for key in left if left[key] is None or right[key] is None or left[key] != right[key]]
    for key in ("model_seed", "trainer_seed", "training_initial_layout_sha256"):
        if candidate["config"].get(key) is None or candidate["config"].get(key) != reference["config"].get(key):
            problems.append(key)
    return problems


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["variant"]].append(run)
    baseline = {run["seed"]: run for run in grouped.get("baseline", [])}
    warnings, variants = [], {}
    for variant, members in sorted(grouped.items()):
        members.sort(key=lambda run: run["seed"])
        condition_strings = {json.dumps(comparison_conditions(run), sort_keys=True) for run in members}
        if len(condition_strings) > 1:
            warnings.append(f"{LABELS[variant]} has differing training/evaluation conditions across seeds.")
        if len(members) < 5:
            warnings.append(f"{LABELS[variant]} has {len(members)} completed training seeds; the planned comparison uses five.")
        metric_names = [key for key in METRICS if any(key in run["metrics"] for run in members)]
        pairs = []
        if variant != "baseline":
            for run in members:
                reference = baseline.get(run["seed"])
                if reference is None:
                    warnings.append(f"No matched baseline for simpler reward seed {run['seed']}.")
                    continue
                problems = pairing_problems(run, reference)
                if problems:
                    warnings.append(f"Excluded simpler reward seed {run['seed']} from paired comparison: {', '.join(problems)} differ or are missing.")
                    continue
                pairs.append({"seed": run["seed"], "candidate_minus_baseline": {
                    key: run["metrics"][key] - reference["metrics"][key]
                    for key in METRICS if key in run["metrics"] and key in reference["metrics"]
                }})
        paired_names = [key for key in METRICS if any(key in pair["candidate_minus_baseline"] for pair in pairs)]
        variants[variant] = {
            "label": LABELS[variant], "num_training_seeds": len(members),
            "seeds": [run["seed"] for run in members],
            "epochs": sorted({run["epochs"] for run in members}),
            "metrics": {key: statistics([run["metrics"][key] for run in members if key in run["metrics"]]) for key in metric_names},
            "paired_vs_baseline": {
                "num_pairs": len(pairs), "seeds": [pair["seed"] for pair in pairs],
                "delta_definition": "Candidate minus matched baseline; positive coverage is better.",
                "metrics": {key: statistics([pair["candidate_minus_baseline"][key] for pair in pairs if key in pair["candidate_minus_baseline"]]) for key in paired_names},
                "per_seed": pairs,
            },
            "per_seed": [{
                "seed": run["seed"], "original_variant": run["original_variant"],
                "directory": run["directory"], "metrics": run["metrics"],
                "checks": run["checks"], "comparison_conditions": comparison_conditions(run),
                "model_seed": run["config"].get("model_seed"),
                "trainer_seed": run["config"].get("trainer_seed"),
                "training_initial_layout_sha256": run["config"].get("training_initial_layout_sha256"),
            } for run in members],
        }
    return {
        "primary_metric": PRIMARY_METRIC,
        "uncertainty": "Sample SD (ddof=1), SE, and two-sided 95% Student-t mean intervals across independent training seeds. Null uncertainty for one seed. Evaluation episodes are averaged within each trained model, never counted as independent training seeds.",
        "completed_runs": len(runs), "zero_shot": runs[0]["zero_shot"],
        "warnings": warnings, "variants": variants,
    }


def write_csv(aggregate: dict[str, Any], destination: Path) -> None:
    present = [key for key in METRICS if any(key in group["metrics"] for group in aggregate["variants"].values())]
    columns = ["variant", "label", "num_training_seeds", "seeds", "epochs", "num_matched_pairs"]
    for prefix in ("", "paired_delta_"):
        columns.extend(f"{prefix}{key}_{stat}" for key in present for stat in ("n", "mean", "std", "se", "ci95_lower", "ci95_upper"))
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for variant, group in aggregate["variants"].items():
            row = {"variant": variant, "label": group["label"], "num_training_seeds": group["num_training_seeds"], "seeds": json.dumps(group["seeds"]), "epochs": json.dumps(group["epochs"]), "num_matched_pairs": group["paired_vs_baseline"]["num_pairs"]}
            for prefix, measurements in (("", group["metrics"]), ("paired_delta_", group["paired_vs_baseline"]["metrics"])):
                for key, values in measurements.items():
                    row.update({f"{prefix}{key}_{stat}": values[stat] for stat in ("n", "mean", "std", "se")})
                    interval = values["ci95"]
                    row.update({f"{prefix}{key}_ci95_lower": interval[0] if interval else None, f"{prefix}{key}_ci95_upper": interval[1] if interval else None})
            writer.writerow(row)


def plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False,
        "figure.dpi": 130, "savefig.dpi": 180,
    })


def plot_line(ax: Any, x: np.ndarray, rows: np.ndarray, *, variant: str, label: str) -> None:
    mean = rows.mean(axis=0)
    ax.plot(x, mean, color=COLORS[variant], label=label, linewidth=2)
    if len(rows) > 1:
        error = rows.std(axis=0, ddof=1) / math.sqrt(len(rows))
        ax.fill_between(x, mean - error, mean + error, color=COLORS[variant], alpha=0.16, linewidth=0)


def plot_comparison(runs: list[dict[str, Any]], aggregate: dict[str, Any], destination: Path) -> None:
    fig, (curve, dots) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    grouped: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for run in runs:
        evaluation = run["evaluation"]
        path = Path(run["directory"]) / f"evaluation_epoch_{evaluation['epoch']:04d}.npz"
        if not path.is_file():
            continue
        with np.load(path) as archive:
            coverage = np.asarray(archive["coverage"])
        if coverage.shape != (evaluation["actions"], evaluation["num_envs"]) or not np.isfinite(coverage).all():
            raise ValueError(f"Invalid coverage archive: {path}")
        grouped[(run["variant"], evaluation["actions"])].append(coverage.mean(axis=1))
    for (variant, actions), rows in sorted(grouped.items()):
        stride = max(1, math.ceil(actions / 100))
        windows = [(start, min(start + stride, actions)) for start in range(0, actions, stride)]
        x = np.asarray([(start + 1 + end) / 2 for start, end in windows])
        values = np.asarray([[row[start:end].mean() for start, end in windows] for row in rows])
        plot_line(curve, x, values, variant=variant, label=f"{LABELS[variant]} (n={len(rows)})")
    if grouped:
        curve.legend(loc="lower right", fontsize=9)
    else:
        curve.text(0.5, 0.5, "Per-action archives unavailable", transform=curve.transAxes, ha="center")
    curve.set(title="Coverage over the longer episode", xlabel="Policy actions", ylabel="Distinct objectives covered", ylim=(0, 1))
    curve.yaxis.set_major_formatter(PercentFormatter(1))

    names = [name for name in ("baseline", "rational_delta") if name in aggregate["variants"]]
    valid_pairs = {pair["seed"] for pair in aggregate["variants"].get("rational_delta", {}).get("paired_vs_baseline", {}).get("per_seed", [])}
    per_seed = {name: {run["seed"]: run["metrics"][PRIMARY_METRIC] for run in runs if run["variant"] == name} for name in names}
    if len(names) == 2:
        for seed in sorted(valid_pairs):
            dots.plot([0, 1], [per_seed[names[0]][seed], per_seed[names[1]][seed]], color="#b3b8bd", linewidth=1, zorder=1)
    tick_labels = []
    for index, name in enumerate(names):
        group = aggregate["variants"][name]
        values = group["metrics"][PRIMARY_METRIC]
        seeds = sorted(per_seed[name])
        offsets = np.linspace(-0.08, 0.08, len(seeds)) if len(seeds) > 1 else [0]
        for offset, seed in zip(offsets, seeds):
            score = per_seed[name][seed]
            dots.scatter(index + offset, score, color=COLORS[name], s=35, zorder=3)
            dots.annotate(str(seed), (index + offset, score), xytext=(4, 3), textcoords="offset points", fontsize=8, color=COLORS[name])
        dots.errorbar(index + 0.16, values["mean"], yerr=values["se"], fmt="D", markersize=6, color=COLORS[name], capsize=4, zorder=4)
        tick_labels.append(f"{LABELS[name]}\nn={values['n']}, mean {values['mean']:.1%}")
    dots.set(xticks=range(len(names)), xticklabels=tick_labels, xlim=(-0.35, max(0.65, len(names) - 0.65)), ylim=(0, 1), ylabel="Final 10% mean coverage", title="Each dot is one trained model")
    dots.yaxis.set_major_formatter(PercentFormatter(1))
    phase = "Original policies, zero-shot" if aggregate["zero_shot"] else "Retrained policies"
    fig.suptitle(f"Doubled objective spacing · {phase.lower()} · bands and bars = ±1 SE across training seeds", fontsize=11)
    fig.savefig(destination)
    plt.close(fig)


def plot_learning(runs: list[dict[str, Any]], destination: Path) -> None:
    grouped: dict[str, dict[int, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for run in runs:
        for evaluation in run["evaluations"]:
            grouped[run["variant"]][evaluation["epoch"]][run["seed"]] = evaluation
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    for ax, metric, title in zip(axes, (PRIMARY_METRIC, "coverage_time_average_mean"), ("Final 10% mean coverage", "Whole-episode mean coverage")):
        plotted = False
        for variant, epochs in sorted(grouped.items()):
            x, summaries = [], []
            for epoch, by_seed in sorted(epochs.items()):
                values = [evaluation[metric] for evaluation in by_seed.values() if metric in evaluation]
                if values:
                    x.append(epoch)
                    summaries.append(statistics(values))
            if not summaries:
                continue
            plotted = True
            means = np.asarray([summary["mean"] for summary in summaries])
            counts = [summary["n"] for summary in summaries]
            ax.plot(x, means, color=COLORS[variant], marker="o", label=f"{LABELS[variant]} (n={','.join(map(str, sorted(set(counts))))})")
            errors = np.asarray([summary["se"] if summary["se"] is not None else np.nan for summary in summaries])
            ax.fill_between(x, means - errors, means + errors, color=COLORS[variant], alpha=0.16, linewidth=0)
        if plotted:
            ax.legend(loc="lower right", fontsize=9)
        else:
            ax.text(0.5, 0.5, "Training evaluation records unavailable", transform=ax.transAxes, ha="center")
        ax.set(title=title, xlabel="Training epochs", ylabel="Held-out coverage", ylim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Doubled objective spacing · recorded evaluations only · shading = ±1 SE across training seeds", fontsize=11)
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
        destination.mkdir(parents=True, exist_ok=True)
        aggregate["source_directory"] = str(source)
        plot_style()
        plot_comparison(runs, aggregate, destination / "comparison_spacing.png")
        plot_learning(runs, destination / "learning_spacing.png")
        (destination / "aggregate_spacing.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")
        write_csv(aggregate, destination / "summary_spacing.csv")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for warning in aggregate["warnings"]:
        print(f"Warning: {warning}")
    print(f"Summarized {len(runs)} completed runs in {destination}")


if __name__ == "__main__":
    main()
