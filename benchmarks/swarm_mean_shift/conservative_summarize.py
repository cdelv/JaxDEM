"""Summarize one collection of conservative swarm experiments on CPU.

    python -m benchmarks.swarm_mean_shift.conservative_summarize --input RESULTS

Only immediate child ``result.json`` files are loaded. Screening and independent
confirmation collections must be summarized separately. Each measurement is an
episode average for one trained model; uncertainty is across training seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jaxdem-conservative-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.stats import t as student_t


PRIMARY_METRIC = "coverage_final_10_percent_mean"
METRICS = (
    PRIMARY_METRIC, "coverage_endpoint_mean", "coverage_time_average_mean",
    "coverage_action_2000_mean", "coverage_action_4000_mean",
    "coverage_action_8000_mean", "speed_time_average_mean",
    "overlapping_agent_fraction_time_average_mean",
)
FAMILIES = ("baseline", "cosine", "vacancy", "cosine_vacancy")
LABELS = {
    "baseline": "Baseline", "cosine": "Cosine approach",
    "vacancy": "Vacancy LiDAR", "cosine_vacancy": "Cosine + vacancy",
}
COLORS = {
    "baseline": "#444444", "cosine": "#007f80", "vacancy": "#b26b17",
    "cosine_vacancy": "#8359a3",
}
CONFIG_CONDITIONS = (
    "N", "num_objectives", "num_envs", "steps_per_epoch", "skip_frames",
    "max_physics_steps", "schedule_epochs", "eval_reset_seed", "eval_action_seed",
    "observation_space_size", "action_space_size", "model_parameter_count",
)
EVALUATION_CONDITIONS = (
    "epoch", "actions", "num_envs", "action_mode", "layout_sha256",
    "physics_steps_per_action", "coverage_final_10_percent_actions",
)
SEED_CONDITIONS = ("model_seed", "trainer_seed", "training_initial_layout_sha256")


def statistics(values: list[float]) -> dict[str, Any]:
    """Sample SD, SE and two-sided Student-t interval across trained models."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Measurements must be a finite one-dimensional sequence")
    n = len(array)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "se": None, "ci95": None}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if n > 1 else None
    se = std / math.sqrt(n) if std is not None else None
    ci95 = None
    if se is not None:
        margin = float(student_t.ppf(0.975, n - 1)) * se
        ci95 = [mean - margin, mean + margin]
    return {"n": n, "mean": mean, "std": std, "se": se, "ci95": ci95}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_runs(source: Path) -> list[dict[str, Any]]:
    runs, identities = [], set()
    for path in sorted(source.glob("*/result.json")):
        result = json.loads(path.read_text())
        evaluation = result.get("final_evaluation")
        if not evaluation:
            continue
        variant, seed = result["variant"], result["seed"]
        match = re.fullmatch(r"(baseline|cosine|vacancy|cosine_vacancy)_s([12])", variant)
        if match is None:
            raise ValueError(f"Unsupported conservative variant {variant!r}: {path}")
        if (variant, seed) in identities:
            raise ValueError(f"Duplicate variant/seed identity {(variant, seed)}")
        identities.add((variant, seed))
        config = json.loads((path.parent / "config.json").read_text())
        if config.get("variant") != variant or config.get("seed") != seed:
            raise ValueError(f"Result/config variant or seed identity differs: {path}")
        if result["epochs"] != config["epochs"] or evaluation["epoch"] != result["epochs"]:
            raise ValueError(f"Result/config/evaluation training epochs differ: {path}")
        # In-progress directories have no final result and are ignored; a final
        # result missing a declared metric is malformed rather than incomplete.
        if PRIMARY_METRIC not in evaluation:
            raise ValueError(f"Missing primary metric {PRIMARY_METRIC}: {path}")
        actions = evaluation["actions"]
        if evaluation.get("coverage_final_10_percent_actions") != math.ceil(actions / 10):
            raise ValueError(f"Primary window is not the final 10 percent: {path}")
        metrics = {key: float(evaluation[key]) for key in METRICS if key in evaluation}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Nonfinite evaluation measurement: {path}")
        checks = {key: result.get(key) for key in (
            "physical_state_finite", "parameters_finite", "collider_overflow",
        )}
        if checks["physical_state_finite"] is False or checks["parameters_finite"] is False or checks["collider_overflow"] is True:
            raise ValueError(f"Run failed physical/parameter validation: {path}: {checks}")
        runs.append({
            "variant": variant, "family": match[1], "spacing_scale": int(match[2]),
            "seed": seed, "epochs": result["epochs"],
            "evaluation_only": bool(config.get("evaluation_only", False)),
            "directory": str(path.parent.resolve()), "metrics": metrics,
            "checks": checks, "config": config, "evaluation": evaluation,
            "evaluations": read_jsonl(path.parent / "evaluation.jsonl") or [evaluation],
        })
    if not runs:
        raise ValueError(f"No completed conservative runs at {source}/*/result.json")
    if len({run["evaluation_only"] for run in runs}) != 1:
        raise ValueError("Select separate result collections for screening and confirmation")
    return runs


def comparison_conditions(run: dict[str, Any]) -> dict[str, Any]:
    """Conditions shared across seeds and candidate/baseline within a scale."""
    conditions = {f"config.{key}": run["config"].get(key) for key in CONFIG_CONDITIONS}
    conditions.update({f"evaluation.{key}": run["evaluation"].get(key) for key in EVALUATION_CONDITIONS})
    conditions.update({
        "epochs": run["epochs"], "spacing_scale": run["spacing_scale"],
        "evaluation_only": run["evaluation_only"],
        "source_hashes": run["config"].get("source_hashes"),
    })
    return conditions


def pairing_problems(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    left, right = comparison_conditions(candidate), comparison_conditions(baseline)
    problems = [key for key in left if left[key] is None or right[key] is None or left[key] != right[key]]
    if not left["source_hashes"] or any(value is None for value in (left["source_hashes"] or {}).values()):
        problems.append("source_hashes missing provenance")
    for key in SEED_CONDITIONS:
        if candidate["config"].get(key) is None or candidate["config"].get(key) != baseline["config"].get(key):
            problems.append(key)
    return problems


def metric_statistics(records: list[dict[str, float]]) -> dict[str, Any]:
    return {
        key: statistics([record[key] for record in records if key in record])
        for key in METRICS if any(key in record for record in records)
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["variant"]].append(run)
    baselines = {
        (run["spacing_scale"], run["seed"]): run
        for run in runs if run["family"] == "baseline"
    }
    warnings, variants = [], {}
    for variant, members in sorted(grouped.items(), key=lambda item: (item[1][0]["spacing_scale"], FAMILIES.index(item[1][0]["family"]))):
        members.sort(key=lambda run: run["seed"])
        family, scale = members[0]["family"], members[0]["spacing_scale"]
        n_baseline = sum(key[0] == scale for key in baselines)
        if len(members) < 3:
            warnings.append(f"{variant} has only {len(members)} completed training seeds; interpret uncertainty cautiously.")
        if len({json.dumps(comparison_conditions(run), sort_keys=True) for run in members}) > 1:
            warnings.append(f"{variant} has differing training/evaluation conditions across seeds.")
        pairs = []
        if family != "baseline":
            for run in members:
                baseline = baselines.get((scale, run["seed"]))
                if baseline is None:
                    warnings.append(f"No matched baseline_s{scale} for {variant} seed {run['seed']}.")
                    continue
                problems = pairing_problems(run, baseline)
                if problems:
                    warnings.append(f"Excluded {variant} seed {run['seed']} from paired comparison: {', '.join(problems)} differ or are missing.")
                    continue
                pairs.append({
                    "seed": run["seed"], "baseline_metrics": baseline["metrics"],
                    "candidate_metrics": run["metrics"],
                    "candidate_minus_baseline": {
                        key: run["metrics"][key] - baseline["metrics"][key]
                        for key in METRICS if key in run["metrics"] and key in baseline["metrics"]
                    },
                })
        if family != "baseline" and len(pairs) < 3:
            warnings.append(f"{variant} has only {len(pairs)} valid matched training-seed pairs.")
        variants[variant] = {
            "family": family, "spacing_scale": scale, "label": LABELS[family],
            "num_training_seeds": len(members), "num_baseline_training_seeds": n_baseline,
            "seeds": [run["seed"] for run in members],
            "epochs": sorted({run["epochs"] for run in members}),
            "metrics": metric_statistics([run["metrics"] for run in members]),
            "paired_vs_baseline": {
                "baseline_variant": f"baseline_s{scale}", "num_pairs": len(pairs),
                "seeds": [pair["seed"] for pair in pairs],
                "delta_definition": "Candidate minus same-scale baseline with matched training seed and conditions; positive coverage is better.",
                "metrics": metric_statistics([pair["candidate_minus_baseline"] for pair in pairs]),
                "matched_baseline_metrics": metric_statistics([pair["baseline_metrics"] for pair in pairs]),
                "matched_candidate_metrics": metric_statistics([pair["candidate_metrics"] for pair in pairs]),
                "primary_wins": sum(pair["candidate_minus_baseline"][PRIMARY_METRIC] > 0 for pair in pairs),
                "primary_ties": sum(pair["candidate_minus_baseline"][PRIMARY_METRIC] == 0 for pair in pairs),
                "per_seed": pairs,
            },
            "per_seed": [{
                "seed": run["seed"], "directory": run["directory"],
                "metrics": run["metrics"], "checks": run["checks"],
                "comparison_conditions": comparison_conditions(run),
                **{key: run["config"].get(key) for key in SEED_CONDITIONS},
            } for run in members],
        }
    provenance = {json.dumps(run["config"].get("source_hashes"), sort_keys=True) for run in runs}
    if len(provenance) != 1:
        warnings.append("Recorded source hashes differ across the collection; inspect provenance before combining comparisons.")
    return {
        "primary_metric": PRIMARY_METRIC, "completed_runs": len(runs),
        "collection": "independent_confirmation" if runs[0]["evaluation_only"] else "training_screen",
        "uncertainty": "Mean, sample SD (ddof=1), SE and two-sided 95% Student-t intervals across independent training seeds. Episodes are averaged within each trained model. One seed has null uncertainty. Paired statistics include only validated matched training seeds; no multiplicity correction or preregistration is implied.",
        "source_provenance_uniform": len(provenance) == 1,
        "baseline_counts_by_spacing": {
            str(scale): sum(run["family"] == "baseline" and run["spacing_scale"] == scale for run in runs)
            for scale in sorted({run["spacing_scale"] for run in runs})
        },
        "warnings": warnings, "variants": variants,
    }


def write_csv(aggregate: dict[str, Any], destination: Path) -> None:
    metrics = [key for key in METRICS if any(key in group["metrics"] for group in aggregate["variants"].values())]
    columns = ["variant", "spacing_scale", "label", "num_training_seeds", "num_baseline_training_seeds", "seeds", "epochs", "num_matched_pairs", "primary_paired_wins"]
    for prefix in ("", "paired_delta_", "matched_baseline_", "matched_candidate_"):
        columns.extend(f"{prefix}{key}_{stat}" for key in metrics for stat in ("n", "mean", "std", "se", "ci95_lower", "ci95_upper"))
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for variant, group in aggregate["variants"].items():
            pairs = group["paired_vs_baseline"]
            row = {key: group[key] for key in ("spacing_scale", "label", "num_training_seeds", "num_baseline_training_seeds")}
            row.update({"variant": variant, "seeds": json.dumps(group["seeds"]), "epochs": json.dumps(group["epochs"]), "num_matched_pairs": pairs["num_pairs"], "primary_paired_wins": pairs["primary_wins"]})
            for prefix, measurements in (("", group["metrics"]), ("paired_delta_", pairs["metrics"]), ("matched_baseline_", pairs["matched_baseline_metrics"]), ("matched_candidate_", pairs["matched_candidate_metrics"])):
                for metric, values in measurements.items():
                    row.update({f"{prefix}{metric}_{stat}": values[stat] for stat in ("n", "mean", "std", "se")})
                    interval = values["ci95"]
                    row.update({f"{prefix}{metric}_ci95_lower": interval[0] if interval else None, f"{prefix}{metric}_ci95_upper": interval[1] if interval else None})
            writer.writerow(row)


def plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "legend.frameon": False,
        "figure.dpi": 130, "savefig.dpi": 180,
    })


def plot_comparison(runs: list[dict[str, Any]], aggregate: dict[str, Any], destination: Path) -> None:
    scales = sorted({run["spacing_scale"] for run in runs})
    fig, axes = plt.subplots(len(scales), 2, figsize=(13, 4.2 * len(scales)), squeeze=False, constrained_layout=True)
    for row, scale in enumerate(scales):
        curve, dots = axes[row]
        same_scale = [run for run in runs if run["spacing_scale"] == scale]
        curves: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
        for run in same_scale:
            evaluation = run["evaluation"]
            path = Path(run["directory"]) / f"evaluation_epoch_{evaluation['epoch']:04d}.npz"
            if not path.is_file():
                continue
            with np.load(path) as archive:
                coverage = np.asarray(archive["coverage"])
            if coverage.shape != (evaluation["actions"], evaluation["num_envs"]) or not np.isfinite(coverage).all():
                raise ValueError(f"Invalid coverage archive: {path}")
            curves[(run["family"], evaluation["actions"])].append(coverage.mean(axis=1))
        for (family, actions), rows in sorted(curves.items(), key=lambda item: FAMILIES.index(item[0][0])):
            stride = max(1, math.ceil(actions / 100))
            windows = [(start, min(start + stride, actions)) for start in range(0, actions, stride)]
            x = [(start + 1 + end) / 2 for start, end in windows]
            values = np.asarray([[series[start:end].mean() for start, end in windows] for series in rows])
            mean = values.mean(axis=0)
            curve.plot(x, mean, color=COLORS[family], label=f"{LABELS[family]} (n={len(rows)})", linewidth=2)
            if len(rows) > 1:
                error = values.std(axis=0, ddof=1) / math.sqrt(len(rows))
                curve.fill_between(x, mean - error, mean + error, color=COLORS[family], alpha=0.14, linewidth=0)
        if curves:
            curve.legend(loc="lower right", fontsize=8)
        else:
            curve.text(.5, .5, "Per-action archives unavailable", transform=curve.transAxes, ha="center")
        curve.set(title=f"{scale}× objective spacing · episode coverage", xlabel="Policy actions", ylabel="Distinct objectives covered", ylim=(0, 1))
        curve.yaxis.set_major_formatter(PercentFormatter(1))
        names = [f"{family}_s{scale}" for family in FAMILIES if f"{family}_s{scale}" in aggregate["variants"]]
        baseline = {run["seed"]: run["metrics"][PRIMARY_METRIC] for run in same_scale if run["family"] == "baseline"}
        baseline_offsets = dict(zip(sorted(baseline), np.linspace(-.08, .08, len(baseline)) if len(baseline) > 1 else [0]))
        tick_labels = []
        for index, name in enumerate(names):
            group = aggregate["variants"][name]
            family = group["family"]
            scores = {run["seed"]: run["metrics"][PRIMARY_METRIC] for run in same_scale if run["variant"] == name}
            offsets = dict(zip(sorted(scores), np.linspace(-.08, .08, len(scores)) if len(scores) > 1 else [0]))
            for pair in group["paired_vs_baseline"]["per_seed"]:
                seed = pair["seed"]
                dots.plot([baseline_offsets[seed], index + offsets[seed]], [baseline[seed], scores[seed]], color=COLORS[family], alpha=.13, linewidth=1, zorder=1)
            for seed, score in sorted(scores.items()):
                offset = offsets[seed]
                dots.scatter(index + offset, score, color=COLORS[family], s=29, zorder=3)
                dots.annotate(str(seed), (index + offset, score), xytext=(4, 3), textcoords="offset points", fontsize=7, color=COLORS[family])
            values = group["metrics"][PRIMARY_METRIC]
            dots.errorbar(index + .16, values["mean"], yerr=values["se"], fmt="D", markersize=5, color=COLORS[family], capsize=4, zorder=4)
            tick_labels.append(f"{LABELS[family]}\nn={values['n']} · {values['mean']:.1%}")
        dots.set(xticks=range(len(names)), xticklabels=tick_labels, xlim=(-.35, max(.65, len(names) - .65)), ylim=(0, 1), ylabel="Final 10% mean coverage", title=f"{scale}× spacing · one dot per training seed")
        dots.tick_params(axis="x", labelsize=8)
        dots.yaxis.set_major_formatter(PercentFormatter(1))
    phase = aggregate["collection"].replace("_", " ")
    fig.suptitle(f"Conservative baseline changes · {phase} · bands and bars = ±1 SE across training seeds", fontsize=12)
    fig.savefig(destination)
    plt.close(fig)


def plot_learning(runs: list[dict[str, Any]], destination: Path) -> None:
    scales = sorted({run["spacing_scale"] for run in runs})
    fig, axes = plt.subplots(1, len(scales), figsize=(6.5 * len(scales), 4.1), squeeze=False, constrained_layout=True)
    for ax, scale in zip(axes[0], scales):
        grouped: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
        for run in runs:
            if run["spacing_scale"] != scale:
                continue
            for evaluation in run["evaluations"]:
                if PRIMARY_METRIC in evaluation:
                    grouped[run["family"]][evaluation["epoch"]][run["seed"]] = evaluation[PRIMARY_METRIC]
        for family in FAMILIES:
            if family not in grouped:
                continue
            x, summaries = [], []
            for epoch, by_seed in sorted(grouped[family].items()):
                x.append(epoch)
                summaries.append(statistics(list(by_seed.values())))
            mean = np.asarray([item["mean"] for item in summaries])
            counts = sorted({item["n"] for item in summaries})
            errors = np.asarray([item["se"] if item["se"] is not None else np.nan for item in summaries])
            ax.plot(x, mean, color=COLORS[family], marker="o", label=f"{LABELS[family]} (n={','.join(map(str, counts))})")
            ax.fill_between(x, mean - errors, mean + errors, color=COLORS[family], alpha=.14, linewidth=0)
        ax.set(title=f"{scale}× objective spacing", xlabel="Training epochs", ylabel="Final 10% mean coverage", ylim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        if grouped:
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Recorded evaluations only · shading = ±1 SE across training seeds", fontsize=11)
    fig.savefig(destination)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source, destination = args.input.resolve(), (args.output or args.input).resolve()
    try:
        runs = load_runs(source)
        aggregate = aggregate_runs(runs)
        aggregate["source_directory"] = str(source)
        destination.mkdir(parents=True, exist_ok=True)
        plot_style()
        plot_comparison(runs, aggregate, destination / "comparison_conservative.png")
        plot_learning(runs, destination / "learning_conservative.png")
        (destination / "aggregate_conservative.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")
        write_csv(aggregate, destination / "summary_conservative.csv")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for warning in aggregate["warnings"]:
        print(f"Warning: {warning}")
    print(f"Summarized {len(runs)} completed runs in {destination}")


if __name__ == "__main__":
    main()
