# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Summarize one collection of unmasked reliability experiments on CPU.

    python -m benchmarks.swarm_mean_shift.unmasked_summarize --input RESULTS

Only immediate child ``result.json`` files are loaded. Screening and independent
confirmation collections must be summarized separately. Each measurement is an
episode average for one trained model; uncertainty is across training seeds.
The low-entropy candidate intentionally changes PPO entropy from 1e-3 to 1e-4.
The supplemental cosine candidate changes the approach kernel and its weight
to 0.111; this matches the unshared approach value at zero distance, not slopes.
Other recorded settings and exact source hashes must match for paired results.
Use --expected-variants for a selected-candidate confirmation collection; fresh
confirmation and extended-time inventories default to training seeds 5 through 9.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jaxdem-unmasked-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.stats import t as student_t

PRIMARY_METRIC = "coverage_final_10_percent_mean"
METRICS = (
    PRIMARY_METRIC,
    "coverage_endpoint_mean",
    "coverage_time_average_mean",
    "coverage_action_2000_mean",
    "coverage_action_4000_mean",
    "coverage_action_8000_mean",
    "speed_time_average_mean",
    "overlapping_agent_fraction_time_average_mean",
)
FAMILIES = ("masked", "raw", "raw_typed", "raw_low_entropy", "raw_cosine_matched")
VARIANT_PATTERN = rf"({'|'.join(FAMILIES)})_s([12])"
LABELS = {
    "masked": "Masked baseline",
    "raw": "Raw goals",
    "raw_typed": "Raw + typed returns",
    "raw_low_entropy": "Raw + low entropy",
    "raw_cosine_matched": "Raw + cosine",
}
TICK_LABELS = {
    family: label.replace("Masked baseline", "Masked\nbaseline")
    .replace("Raw goals", "Raw\ngoals")
    .replace("Raw + ", "Raw +\n")
    for family, label in LABELS.items()
}
COLORS = {
    "masked": "#444444",
    "raw": "#007f80",
    "raw_typed": "#b26b17",
    "raw_low_entropy": "#8359a3",
    "raw_cosine_matched": "#3c65b0",
}
CONFIG_CONDITIONS = (
    "N",
    "num_objectives",
    "num_envs",
    "steps_per_epoch",
    "skip_frames",
    "max_physics_steps",
    "schedule_epochs",
    "eval_reset_seed",
    "eval_action_seed",
    "observation_space_size",
    "action_space_size",
    "model_parameter_count",
    "action_mode",
    "coverage_cutoff_radii",
    "training_decisions_per_epoch",
    "package_versions",
    "jax_version",
    "python_version",
    "eval_at",
    "eval_envs",
    "eval_actions",
    "eval_curve_stride",
    "source_experiment",
)
EVALUATION_CONDITIONS = (
    "epoch",
    "actions",
    "num_envs",
    "action_mode",
    "layout_sha256",
    "physics_steps_per_action",
    "coverage_final_10_percent_actions",
)
SEED_CONDITIONS = ("model_seed", "trainer_seed", "training_initial_layout_sha256")
RELIABILITY_THRESHOLD = 0.8


def validate_family_metadata(config: dict[str, Any], family: str, path: Path) -> None:
    """Validate intended differences without importing the GPU training stack."""
    entropy = 1e-4 if family == "raw_low_entropy" else 1e-3
    expected = {
        "observation_occupancy_mask": family == "masked",
        "hide_occupied_objectives": family == "masked",
        "vacancy_aware_lidar": False,
        "approach_mode": "cosine" if family == "raw_cosine_matched" else "baseline",
        "attraction_coeff": 0.111 if family == "raw_cosine_matched" else 0.1,
        "peer_wall_encoding": "signed" if family == "raw_typed" else "unsigned",
        "ppo_entropy_coeff": entropy,
        "N": 64,
        "num_objectives": 64,
        "observation_space_size": 26,
        "action_space_size": 2,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"Invalid {family} metadata {key}: {config.get(key)!r} != {value!r}: {path}"
            )
    actual = config.get("actual_ppo_entropy_coeff")
    if not isinstance(actual, (int, float)) or not math.isclose(
        actual, entropy, rel_tol=1e-6, abs_tol=1e-10
    ):
        raise ValueError(f"Unverified actual training entropy coefficient: {path}")
    if config.get("entropy_coefficient_matches_config") is not True:
        raise ValueError(f"Missing verified training entropy metadata: {path}")


def learning_history(
    directory: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Use comparable screening layouts for learning changes, also after restore."""
    warnings = []
    origin = directory
    if config.get("evaluation_only"):
        source = config.get("source_run")
        if not source:
            return (
                [],
                "unavailable",
                ["Training source_run is absent; late-epoch changes unavailable."],
            )
        origin = Path(source)
        source_config = origin / "config.json"
        if not source_config.is_file():
            return (
                [],
                str(origin),
                [
                    "Training source config is unavailable; late-epoch changes unavailable."
                ],
            )
        encoded = source_config.read_bytes()
        recorded_hash = config.get("source_config_sha256") or config.get(
            "origin_config_sha256"
        )
        if hashlib.sha256(encoded).hexdigest() != recorded_hash:
            return (
                [],
                str(origin),
                ["Training source config checksum differs; learning history excluded."],
            )
        original = json.loads(encoded)
        for key in ("variant", "seed", "epochs", "source_hashes", *SEED_CONDITIONS):
            if original.get(key) is None or original.get(key) != config.get(key):
                return (
                    [],
                    str(origin),
                    [f"Training source {key} differs; learning history excluded."],
                )
    history = read_jsonl(origin / "evaluation.jsonl")
    by_epoch: dict[int, dict[str, Any]] = {}
    for item in history:
        if (
            PRIMARY_METRIC not in item
            or item.get("epoch", config["epochs"] + 1) > config["epochs"]
        ):
            continue
        value = item[PRIMARY_METRIC]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return (
                [],
                str(origin),
                ["Nonfinite training evaluation; learning history excluded."],
            )
        epoch = item["epoch"]
        if epoch in by_epoch and by_epoch[epoch] != item:
            return (
                [],
                str(origin),
                [
                    "Conflicting training evaluations at one epoch; learning history excluded."
                ],
            )
        by_epoch[epoch] = item
    history = [item for _, item in sorted(by_epoch.items())]
    if history:
        reference = history[-1]
        fields = [key for key in EVALUATION_CONDITIONS if key != "epoch"]
        if any(
            item.get(key) is None or item.get(key) != reference.get(key)
            for item in history
            for key in fields
        ):
            return (
                [],
                str(origin),
                [
                    "Training evaluation layouts/horizons differ; learning history excluded."
                ],
            )
        if any(
            item["coverage_final_10_percent_actions"] != math.ceil(item["actions"] / 10)
            for item in history
        ):
            return (
                [],
                str(origin),
                [
                    "Training evaluation primary windows differ; learning history excluded."
                ],
            )
    if not history or history[-1]["epoch"] != config["epochs"]:
        warnings.append(
            "No final training evaluation in history; late-epoch changes unavailable."
        )
    return history, str(origin), warnings


def late_change(run: dict[str, Any]) -> dict[str, Any] | None:
    """Final minus the declared late checkpoint on identical screening layouts."""
    reference_epoch = 40 if run["spacing_scale"] == 1 else 120
    by_epoch = {item["epoch"]: item for item in run["evaluations"]}
    final = by_epoch.get(run["epochs"])
    reference = by_epoch.get(reference_epoch)
    if not final or not reference or reference_epoch >= run["epochs"]:
        return None
    intermediate = [item for epoch, item in by_epoch.items() if epoch < run["epochs"]]
    best = max(intermediate, key=lambda item: item[PRIMARY_METRIC])
    return {
        "reference_epoch": reference_epoch,
        "final_epoch": run["epochs"],
        "reference_coverage": reference[PRIMARY_METRIC],
        "final_coverage": final[PRIMARY_METRIC],
        "final_minus_reference": final[PRIMARY_METRIC] - reference[PRIMARY_METRIC],
        "best_intermediate_epoch": best["epoch"],
        "final_minus_best_intermediate": final[PRIMARY_METRIC] - best[PRIMARY_METRIC],
        "history_origin": run["learning_history_origin"],
    }


def reliability(values: list[float]) -> dict[str, Any]:
    summary = statistics(values)
    array = np.asarray(values, dtype=float)
    count = int(np.sum(array >= RELIABILITY_THRESHOLD))
    return {
        "n": summary["n"],
        "mean": summary["mean"],
        "sample_sd": summary["std"],
        "minimum": float(array.min()) if len(array) else None,
        "q25": float(np.quantile(array, 0.25, method="linear")) if len(array) else None,
        "threshold": RELIABILITY_THRESHOLD,
        "num_at_or_above_threshold": count,
        "fraction_at_or_above_threshold": count / len(array) if len(array) else None,
    }


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
    if not source.is_dir():
        raise ValueError(f"Input collection directory does not exist: {source}")
    runs, identities = [], set()
    for path in sorted(source.glob("*/result.json")):
        result = json.loads(path.read_text())
        evaluation = result.get("final_evaluation")
        if not evaluation:
            continue
        variant, seed = result["variant"], result["seed"]
        match = re.fullmatch(VARIANT_PATTERN, variant)
        if match is None:
            raise ValueError(
                f"Unsupported unmasked reliability variant {variant!r}: {path}"
            )
        if (variant, seed) in identities:
            raise ValueError(f"Duplicate variant/seed identity {(variant, seed)}")
        identities.add((variant, seed))
        config = json.loads((path.parent / "config.json").read_text())
        if config.get("variant") != variant or config.get("seed") != seed:
            raise ValueError(f"Result/config variant or seed identity differs: {path}")
        validate_family_metadata(config, match[1], path)
        if (
            result["epochs"] != config["epochs"]
            or evaluation["epoch"] != result["epochs"]
        ):
            raise ValueError(f"Result/config/evaluation training epochs differ: {path}")
        # In-progress directories have no final result and are ignored; a final
        # result missing a declared metric is malformed rather than incomplete.
        if PRIMARY_METRIC not in evaluation:
            raise ValueError(f"Missing primary metric {PRIMARY_METRIC}: {path}")
        actions = evaluation["actions"]
        if evaluation.get("coverage_final_10_percent_actions") != math.ceil(
            actions / 10
        ):
            raise ValueError(f"Primary window is not the final 10 percent: {path}")
        metrics = {key: float(evaluation[key]) for key in METRICS if key in evaluation}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Nonfinite evaluation measurement: {path}")
        checks = {
            key: result.get(key)
            for key in (
                "physical_state_finite",
                "parameters_finite",
                "collider_overflow",
            )
        }
        if (
            checks["physical_state_finite"] is False
            or checks["parameters_finite"] is False
            or checks["collider_overflow"] is True
        ):
            raise ValueError(
                f"Run failed physical/parameter validation: {path}: {checks}"
            )
        history, history_origin, history_warnings = learning_history(
            path.parent, config
        )
        runs.append(
            {
                "variant": variant,
                "family": match[1],
                "spacing_scale": int(match[2]),
                "seed": seed,
                "epochs": result["epochs"],
                "evaluation_only": bool(config.get("evaluation_only", False)),
                "extended_evaluation": bool(config.get("extended_evaluation", False)),
                "directory": str(path.parent.resolve()),
                "metrics": metrics,
                "checks": checks,
                "config": config,
                "evaluation": evaluation,
                "evaluations": history,
                "learning_history_origin": history_origin,
                "learning_history_warnings": history_warnings,
            }
        )
    if len({(run["evaluation_only"], run["extended_evaluation"]) for run in runs}) > 1:
        raise ValueError(
            "Select separate result collections for screening, confirmation and extended time"
        )
    return runs


def comparison_conditions(run: dict[str, Any]) -> dict[str, Any]:
    """Conditions shared across seeds and candidate/baseline within a scale."""
    conditions = {f"config.{key}": run["config"].get(key) for key in CONFIG_CONDITIONS}
    conditions.update(
        {
            f"evaluation.{key}": run["evaluation"].get(key)
            for key in EVALUATION_CONDITIONS
        }
    )
    conditions.update(
        {
            "epochs": run["epochs"],
            "spacing_scale": run["spacing_scale"],
            "evaluation_only": run["evaluation_only"],
            "extended_evaluation": run["extended_evaluation"],
            "source_hashes": run["config"].get("source_hashes"),
        }
    )
    if run["evaluation_only"]:
        conditions["confirmation_source_hashes"] = run["config"].get(
            "confirmation_source_hashes"
        )
    if run["extended_evaluation"]:
        for key in (
            "time_multiplier",
            "training_max_physics_steps",
            "training_episode_actions",
        ):
            conditions[f"config.{key}"] = run["config"].get(key)
    for key in (
        "learning_rate",
        "num_minibatches",
        "caps_temporal_coeff",
        "caps_spatial_coeff",
        "caps_noise_std",
    ):
        if key in run["config"]:
            conditions[f"config.{key}"] = run["config"][key]
    return conditions


def pairing_problems(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    left, right = comparison_conditions(candidate), comparison_conditions(baseline)
    problems = [
        key
        for key in sorted(left.keys() | right.keys())
        if left.get(key) is None or right.get(key) is None or left[key] != right[key]
    ]
    for key in ("source_hashes", "confirmation_source_hashes"):
        hashes = left.get(key)
        if key in left and (
            not isinstance(hashes, dict)
            or not hashes
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in hashes.values()
            )
        ):
            problems.append(f"{key} missing provenance")
    for key in SEED_CONDITIONS:
        if candidate["config"].get(key) is None or candidate["config"].get(
            key
        ) != baseline["config"].get(key):
            problems.append(key)
    return problems


def metric_statistics(records: list[dict[str, float]]) -> dict[str, Any]:
    return {
        key: statistics([record[key] for record in records if key in record])
        for key in METRICS
        if any(key in record for record in records)
    }


def paired_summary(
    members: list[dict[str, Any]],
    controls: dict[tuple[int, int], dict[str, Any]],
    reference: str,
    warnings: list[str],
) -> dict[str, Any]:
    scale = members[0]["spacing_scale"]
    pairs = []
    for run in members:
        if run["family"] == reference:
            continue
        baseline = controls.get((scale, run["seed"]))
        if baseline is None:
            warnings.append(
                f"No matched {reference}_s{scale} for {run['variant']} seed {run['seed']}."
            )
            continue
        problems = pairing_problems(run, baseline)
        if problems:
            warnings.append(
                f"Excluded {run['variant']} seed {run['seed']} vs {reference}: {', '.join(problems)} differ or are missing."
            )
            continue
        pairs.append(
            {
                "seed": run["seed"],
                "baseline_metrics": baseline["metrics"],
                "candidate_metrics": run["metrics"],
                "candidate_minus_baseline": {
                    key: run["metrics"][key] - baseline["metrics"][key]
                    for key in METRICS
                    if key in run["metrics"] and key in baseline["metrics"]
                },
            }
        )
    if members[0]["family"] != reference and len(pairs) < 3:
        warnings.append(
            f"{members[0]['variant']} has only {len(pairs)} valid matched training-seed pairs vs {reference}."
        )
    return {
        "baseline_variant": f"{reference}_s{scale}",
        "num_pairs": len(pairs),
        "seeds": [pair["seed"] for pair in pairs],
        "delta_definition": f"Candidate minus same-spacing {reference} with matched training seed and validated conditions; positive coverage is better.",
        "metrics": metric_statistics(
            [pair["candidate_minus_baseline"] for pair in pairs]
        ),
        "matched_baseline_metrics": metric_statistics(
            [pair["baseline_metrics"] for pair in pairs]
        ),
        "matched_candidate_metrics": metric_statistics(
            [pair["candidate_metrics"] for pair in pairs]
        ),
        "primary_wins": sum(
            pair["candidate_minus_baseline"][PRIMARY_METRIC] > 0 for pair in pairs
        ),
        "primary_ties": sum(
            pair["candidate_minus_baseline"][PRIMARY_METRIC] == 0 for pair in pairs
        ),
        "per_seed": pairs,
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["variant"]].append(run)
    baselines = {
        (run["spacing_scale"], run["seed"]): run
        for run in runs
        if run["family"] == "masked"
    }
    raw_controls = {
        (run["spacing_scale"], run["seed"]): run
        for run in runs
        if run["family"] == "raw"
    }
    warnings, variants = [], {}
    for variant, members in sorted(
        grouped.items(),
        key=lambda item: (
            item[1][0]["spacing_scale"],
            FAMILIES.index(item[1][0]["family"]),
        ),
    ):
        members.sort(key=lambda run: run["seed"])
        family, scale = members[0]["family"], members[0]["spacing_scale"]
        n_baseline = sum(key[0] == scale for key in baselines)
        if len(members) < 3:
            warnings.append(
                f"{variant} has only {len(members)} completed training seeds; interpret uncertainty cautiously."
            )
        if (
            len(
                {
                    json.dumps(comparison_conditions(run), sort_keys=True)
                    for run in members
                }
            )
            > 1
        ):
            warnings.append(
                f"{variant} has differing training/evaluation conditions across seeds."
            )
        pairs = paired_summary(members, baselines, "masked", warnings)
        changes = []
        for run in members:
            warnings.extend(
                f"{variant} seed {run['seed']}: {warning}"
                for warning in run["learning_history_warnings"]
            )
            change = late_change(run)
            if change is not None:
                changes.append({"seed": run["seed"], **change})
        if len(changes) < len(members):
            warnings.append(
                f"{variant}: late-epoch changes available for {len(changes)}/{len(members)} trained models."
            )
        variants[variant] = {
            "family": family,
            "spacing_scale": scale,
            "label": LABELS[family],
            "num_training_seeds": len(members),
            "num_baseline_training_seeds": n_baseline,
            "seeds": [run["seed"] for run in members],
            "epochs": sorted({run["epochs"] for run in members}),
            "evaluation_actions": sorted(
                {run["evaluation"]["actions"] for run in members}
            ),
            "evaluation_envs": sorted(
                {run["evaluation"]["num_envs"] for run in members}
            ),
            "final_window_actions": sorted(
                {
                    run["evaluation"]["coverage_final_10_percent_actions"]
                    for run in members
                }
            ),
            "metrics": metric_statistics([run["metrics"] for run in members]),
            "reliability": reliability(
                [run["metrics"][PRIMARY_METRIC] for run in members]
            ),
            "late_epoch_change": {
                "definition": "Final minus epoch 40 at 1x spacing or epoch 120 at 2x, using original screening layouts and episode horizon. These are training-history diagnostics, not fresh-confirmation scores.",
                "final_minus_reference": statistics(
                    [item["final_minus_reference"] for item in changes]
                ),
                "final_minus_best_intermediate": statistics(
                    [item["final_minus_best_intermediate"] for item in changes]
                ),
                "num_regressions_vs_reference": sum(
                    item["final_minus_reference"] < 0 for item in changes
                ),
                "num_regressions_vs_best_intermediate": sum(
                    item["final_minus_best_intermediate"] < 0 for item in changes
                ),
                "per_seed": changes,
            },
            "paired_vs_baseline": pairs,
            "paired_vs_raw": (
                paired_summary(members, raw_controls, "raw", warnings)
                if family in ("raw_typed", "raw_low_entropy", "raw_cosine_matched")
                and any(key[0] == scale for key in raw_controls)
                else None
            ),
            "per_seed": [
                {
                    "seed": run["seed"],
                    "directory": run["directory"],
                    "metrics": run["metrics"],
                    "checks": run["checks"],
                    "comparison_conditions": comparison_conditions(run),
                    "learning_history_origin": run["learning_history_origin"],
                    "late_epoch_change": late_change(run),
                    **{key: run["config"].get(key) for key in SEED_CONDITIONS},
                }
                for run in members
            ],
        }
    provenance = {
        json.dumps(run["config"].get("source_hashes"), sort_keys=True) for run in runs
    }
    if len(provenance) > 1:
        warnings.append(
            "Recorded source hashes differ across the collection; inspect provenance before combining comparisons."
        )
    return {
        "primary_metric": PRIMARY_METRIC,
        "completed_runs": len(runs),
        "collection": (
            (
                "extended_confirmation"
                if runs[0]["extended_evaluation"]
                else (
                    "independent_confirmation"
                    if runs[0]["evaluation_only"]
                    else "training_screen"
                )
            )
            if runs
            else "unknown"
        ),
        "uncertainty": "Mean, sample SD (ddof=1), SE and two-sided 95% Student-t intervals across independent training seeds. Episodes are averaged within each trained model. One seed has null uncertainty. Paired statistics include only validated matched training seeds; no multiplicity correction or preregistration is implied.",
        "source_provenance_uniform": len(provenance) == 1,
        "intentional_differences": {
            "masked": "Full legacy reward, masked occupied-goal observations, entropy 1e-3.",
            "raw": "Full legacy reward, all raw goal returns, entropy 1e-3.",
            "raw_typed": "Raw goal returns; signed peer/wall channel encoding (+peer, -wall); entropy 1e-3.",
            "raw_low_entropy": "Raw environment; PPO entropy 1e-4 instead of 1e-3. This is a training change.",
            "raw_cosine_matched": "Supplemental raw-observation reward pilot: raised-cosine approach kernel with weight 0.111, matching the legacy unshared zero-distance approach amplitude (not its slopes or distance profile); entropy 1e-3.",
            "shared": "26 observation dimensions; vacancy-aware lidar disabled; same quartic settling, soft reciprocal peer sharing, and action-start potential-difference timing. All except raw_cosine_matched retain the full legacy approach reward. Other recorded training/evaluation settings and complete source-hash dictionaries must match within a paired spacing/seed comparison.",
        },
        "reliability_note": "Sample minimum and linearly interpolated 25th percentile describe these trained models. A three-seed screen is a noisy selection heuristic, not evidence of a population tail-risk guarantee. Threshold counts use model mean coverage >=80%, not individual episodes.",
        "baseline_counts_by_spacing": {
            str(scale): sum(
                run["family"] == "masked" and run["spacing_scale"] == scale
                for run in runs
            )
            for scale in sorted({run["spacing_scale"] for run in runs})
        },
        "warnings": warnings,
        "variants": variants,
    }


def write_csv(aggregate: dict[str, Any], destination: Path) -> None:
    metrics = [
        key
        for key in METRICS
        if any(key in group["metrics"] for group in aggregate["variants"].values())
    ]
    reliability_keys = (
        "sample_sd",
        "minimum",
        "q25",
        "threshold",
        "num_at_or_above_threshold",
        "fraction_at_or_above_threshold",
    )
    columns = [
        "variant",
        "spacing_scale",
        "label",
        "collection",
        "num_training_seeds",
        "num_baseline_training_seeds",
        "seeds",
        "epochs",
        "evaluation_actions",
        "evaluation_envs",
        "final_window_actions",
        "num_matched_pairs",
        "primary_paired_wins",
        "num_raw_pairs",
    ]
    columns.extend(f"reliability_{key}" for key in reliability_keys)
    columns.extend(
        (
            "late_epoch_num_models",
            "late_epoch_reference",
            "late_epoch_final_minus_reference_mean",
            "late_epoch_num_regressions",
            "late_epoch_final_minus_best_intermediate_mean",
            "late_epoch_num_regressions_vs_best_intermediate",
        )
    )
    for prefix in (
        "",
        "paired_delta_",
        "matched_baseline_",
        "matched_candidate_",
        "paired_delta_vs_raw_",
    ):
        columns.extend(
            f"{prefix}{key}_{stat}"
            for key in metrics
            for stat in ("n", "mean", "std", "se", "ci95_lower", "ci95_upper")
        )
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for variant, group in aggregate["variants"].items():
            pairs = group["paired_vs_baseline"]
            row = {
                key: group[key]
                for key in (
                    "spacing_scale",
                    "label",
                    "num_training_seeds",
                    "num_baseline_training_seeds",
                )
            }
            row.update(
                {
                    "variant": variant,
                    "seeds": json.dumps(group["seeds"]),
                    "epochs": json.dumps(group["epochs"]),
                    "num_matched_pairs": pairs["num_pairs"],
                    "primary_paired_wins": pairs["primary_wins"],
                }
            )
            row.update(
                {
                    "collection": aggregate["collection"],
                    **{
                        key: json.dumps(group[key])
                        for key in (
                            "evaluation_actions",
                            "evaluation_envs",
                            "final_window_actions",
                        )
                    },
                }
            )
            row.update(
                {
                    f"reliability_{key}": group["reliability"][key]
                    for key in reliability_keys
                }
            )
            changes = group["late_epoch_change"]
            row.update(
                {
                    "late_epoch_num_models": changes["final_minus_reference"]["n"],
                    "late_epoch_reference": 40 if group["spacing_scale"] == 1 else 120,
                    "late_epoch_final_minus_reference_mean": changes[
                        "final_minus_reference"
                    ]["mean"],
                    "late_epoch_num_regressions": changes[
                        "num_regressions_vs_reference"
                    ],
                    "late_epoch_final_minus_best_intermediate_mean": changes[
                        "final_minus_best_intermediate"
                    ]["mean"],
                    "late_epoch_num_regressions_vs_best_intermediate": changes[
                        "num_regressions_vs_best_intermediate"
                    ],
                    "num_raw_pairs": (
                        group["paired_vs_raw"]["num_pairs"]
                        if group["paired_vs_raw"]
                        else None
                    ),
                }
            )
            raw_metrics = (
                group["paired_vs_raw"]["metrics"] if group["paired_vs_raw"] else {}
            )
            for prefix, measurements in (
                ("", group["metrics"]),
                ("paired_delta_", pairs["metrics"]),
                ("matched_baseline_", pairs["matched_baseline_metrics"]),
                ("matched_candidate_", pairs["matched_candidate_metrics"]),
                ("paired_delta_vs_raw_", raw_metrics),
            ):
                for metric, values in measurements.items():
                    row.update(
                        {
                            f"{prefix}{metric}_{stat}": values[stat]
                            for stat in ("n", "mean", "std", "se")
                        }
                    )
                    interval = values["ci95"]
                    row.update(
                        {
                            f"{prefix}{metric}_ci95_lower": (
                                interval[0] if interval else None
                            ),
                            f"{prefix}{metric}_ci95_upper": (
                                interval[1] if interval else None
                            ),
                        }
                    )
            writer.writerow(row)


def plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "legend.frameon": False,
            "figure.dpi": 130,
            "savefig.dpi": 180,
        }
    )


def plot_comparison(
    runs: list[dict[str, Any]], aggregate: dict[str, Any], destination: Path
) -> None:
    scales = sorted({run["spacing_scale"] for run in runs})
    fig, axes = plt.subplots(
        len(scales),
        2,
        figsize=(13, 4.2 * len(scales)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, scale in enumerate(scales):
        curve, dots = axes[row]
        same_scale = [run for run in runs if run["spacing_scale"] == scale]
        curves: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
        for run in same_scale:
            evaluation = run["evaluation"]
            path = (
                Path(run["directory"])
                / f"evaluation_epoch_{evaluation['epoch']:04d}.npz"
            )
            if not path.is_file():
                continue
            with np.load(path) as archive:
                coverage = np.asarray(archive["coverage"])
            if (
                coverage.shape != (evaluation["actions"], evaluation["num_envs"])
                or not np.isfinite(coverage).all()
            ):
                raise ValueError(f"Invalid coverage archive: {path}")
            curves[(run["family"], evaluation["actions"])].append(coverage.mean(axis=1))
        for (family, actions), rows in sorted(
            curves.items(), key=lambda item: FAMILIES.index(item[0][0])
        ):
            stride = max(1, math.ceil(actions / 100))
            windows = [
                (start, min(start + stride, actions))
                for start in range(0, actions, stride)
            ]
            x = [(start + 1 + end) / 2 for start, end in windows]
            values = np.asarray(
                [
                    [series[start:end].mean() for start, end in windows]
                    for series in rows
                ]
            )
            mean = values.mean(axis=0)
            curve.plot(
                x,
                mean,
                color=COLORS[family],
                label=f"{LABELS[family]} (n={len(rows)})",
                linewidth=2,
            )
            if len(rows) > 1:
                error = values.std(axis=0, ddof=1) / math.sqrt(len(rows))
                curve.fill_between(
                    x,
                    mean - error,
                    mean + error,
                    color=COLORS[family],
                    alpha=0.14,
                    linewidth=0,
                )
        if curves:
            curve.legend(loc="lower right", fontsize=8)
        else:
            curve.text(
                0.5,
                0.5,
                "Per-action archives unavailable",
                transform=curve.transAxes,
                ha="center",
            )
        curve.set(
            title=f"{scale}× objective spacing · episode coverage",
            xlabel="Policy actions",
            ylabel="Distinct objectives covered",
            ylim=(0, 1),
        )
        curve.yaxis.set_major_formatter(PercentFormatter(1))
        names = [
            f"{family}_s{scale}"
            for family in FAMILIES
            if f"{family}_s{scale}" in aggregate["variants"]
        ]
        baseline = {
            run["seed"]: run["metrics"][PRIMARY_METRIC]
            for run in same_scale
            if run["family"] == "masked"
        }
        baseline_offsets = dict(
            zip(
                sorted(baseline),
                np.linspace(-0.08, 0.08, len(baseline)) if len(baseline) > 1 else [0],
            )
        )
        tick_labels = []
        for index, name in enumerate(names):
            group = aggregate["variants"][name]
            family = group["family"]
            scores = {
                run["seed"]: run["metrics"][PRIMARY_METRIC]
                for run in same_scale
                if run["variant"] == name
            }
            offsets = dict(
                zip(
                    sorted(scores),
                    np.linspace(-0.08, 0.08, len(scores)) if len(scores) > 1 else [0],
                )
            )
            for pair in group["paired_vs_baseline"]["per_seed"]:
                seed = pair["seed"]
                dots.plot(
                    [baseline_offsets[seed], index + offsets[seed]],
                    [baseline[seed], scores[seed]],
                    color=COLORS[family],
                    alpha=0.13,
                    linewidth=1,
                    zorder=1,
                )
            for seed, score in sorted(scores.items()):
                offset = offsets[seed]
                dots.scatter(
                    index + offset, score, color=COLORS[family], s=29, zorder=3
                )
                dots.annotate(
                    str(seed),
                    (index + offset, score),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color=COLORS[family],
                )
            values = group["metrics"][PRIMARY_METRIC]
            dots.errorbar(
                index + 0.16,
                values["mean"],
                yerr=values["se"],
                fmt="D",
                markersize=5,
                color=COLORS[family],
                capsize=4,
                zorder=4,
            )
            tick_labels.append(
                f"{TICK_LABELS[family]}\nn={values['n']} · {values['mean']:.1%}"
            )
        dots.set(
            xticks=range(len(names)),
            xticklabels=tick_labels,
            xlim=(-0.35, max(0.65, len(names) - 0.65)),
            ylim=(0, 1),
            ylabel="Final 10% mean coverage",
            title=f"{scale}× spacing · one dot per training seed",
        )
        dots.tick_params(axis="x", labelsize=8)
        dots.yaxis.set_major_formatter(PercentFormatter(1))
    phase = aggregate["collection"].replace("_", " ")
    fig.suptitle(
        f"Unmasked reliability · {phase} · bands and bars = ±1 SE across training seeds",
        fontsize=12,
    )
    fig.savefig(destination)
    plt.close(fig)


def plot_learning(runs: list[dict[str, Any]], destination: Path) -> None:
    scales = sorted({run["spacing_scale"] for run in runs})
    fig, axes = plt.subplots(
        1,
        len(scales),
        figsize=(6.5 * len(scales), 4.1),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, scale in zip(axes[0], scales):
        grouped: dict[str, dict[int, dict[int, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for run in runs:
            if run["spacing_scale"] != scale:
                continue
            for evaluation in run["evaluations"]:
                if PRIMARY_METRIC in evaluation:
                    grouped[run["family"]][evaluation["epoch"]][run["seed"]] = (
                        evaluation[PRIMARY_METRIC]
                    )
        for family in FAMILIES:
            if family not in grouped:
                continue
            x, summaries = [], []
            for epoch, by_seed in sorted(grouped[family].items()):
                x.append(epoch)
                summaries.append(statistics(list(by_seed.values())))
            mean = np.asarray([item["mean"] for item in summaries])
            counts = sorted({item["n"] for item in summaries})
            errors = np.asarray(
                [item["se"] if item["se"] is not None else np.nan for item in summaries]
            )
            ax.plot(
                x,
                mean,
                color=COLORS[family],
                marker="o",
                label=f"{LABELS[family]} (n={','.join(map(str, counts))})",
            )
            ax.fill_between(
                x,
                mean - errors,
                mean + errors,
                color=COLORS[family],
                alpha=0.14,
                linewidth=0,
            )
        ax.set(
            title=f"{scale}× objective spacing",
            xlabel="Training epochs",
            ylabel="Final 10% mean coverage",
            ylim=(0, 1),
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        if grouped:
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Original screening layouts and horizons · shading = ±1 SE across training seeds",
        fontsize=11,
    )
    fig.savefig(destination)
    plt.close(fig)


def plot_reliability(aggregate: dict[str, Any], destination: Path) -> None:
    scales = sorted(
        {group["spacing_scale"] for group in aggregate["variants"].values()}
    )
    fig, axes = plt.subplots(
        len(scales),
        3,
        figsize=(14, 3.7 * len(scales)),
        squeeze=False,
        constrained_layout=True,
    )
    measures = (
        ("minimum", "Minimum trained-model coverage"),
        ("q25", "Sample 25th percentile"),
        ("fraction_at_or_above_threshold", "Trained models reaching 80%"),
    )
    for row, scale in enumerate(scales):
        groups = [
            aggregate["variants"][f"{family}_s{scale}"]
            for family in FAMILIES
            if f"{family}_s{scale}" in aggregate["variants"]
        ]
        for ax, (key, title) in zip(axes[row], measures):
            for x, group in enumerate(groups):
                record = group["reliability"]
                value = record[key]
                ax.scatter(x, value, s=65, color=COLORS[group["family"]], zorder=3)
                text = (
                    f"{record['num_at_or_above_threshold']}/{record['n']}"
                    if key == "fraction_at_or_above_threshold"
                    else f"{value:.1%}"
                )
                ax.annotate(
                    text,
                    (x, value),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
            ax.set(
                xticks=range(len(groups)),
                xticklabels=[
                    f"{TICK_LABELS[group['family']]}\nn={group['num_training_seeds']}"
                    for group in groups
                ],
                title=f"{scale}× spacing · {title}",
                ylim=(-0.04, 1.1),
            )
            ax.tick_params(axis="x", labelsize=8)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            if key != "fraction_at_or_above_threshold":
                ax.axhline(
                    RELIABILITY_THRESHOLD, color="0.65", linestyle=":", linewidth=1
                )
    fig.suptitle(
        "Descriptive reliability across trained models · small-sample minima and quantiles are noisy",
        fontsize=12,
    )
    fig.savefig(destination)
    plt.close(fig)


def add_inventory(
    aggregate: dict[str, Any],
    source: Path,
    expected_variants: list[str] | None,
    expected_seeds: list[int] | None,
    protocol_path: Path | None,
) -> None:
    protocol = {}
    if protocol_path is not None:
        protocol = json.loads(protocol_path.read_text())
    else:
        for path in (source.parent / "protocol.json", source / "protocol.json"):
            if path.is_file():
                protocol.update(json.loads(path.read_text()))
    phase_protocol = source / "unmasked_confirmation_protocol.json"
    if phase_protocol.is_file():
        protocol.update(json.loads(phase_protocol.read_text()))
        if aggregate["collection"] == "unknown":
            aggregate["collection"] = "independent_confirmation"
    if "initial_families" in protocol:
        supplementary = [
            family
            for family in protocol.get("families", [])
            if family not in protocol["initial_families"]
        ]
        aggregate["screen_design"] = {
            "initial_families": protocol["initial_families"],
            "supplemental_families": supplementary,
            "supplemental_descriptions": {
                family: protocol.get(family) for family in supplementary
            },
            "selection": protocol.get("selection"),
        }
    if aggregate["collection"] == "unknown":
        aggregate["collection"] = "training_screen"
    screening = aggregate["collection"] == "training_screen"
    if expected_variants is None:
        if screening:
            expected_variants = [
                f"{family}_s{scale}"
                for scale in protocol.get("spacings", [1, 2])
                for family in protocol.get("families", FAMILIES)
            ]
        else:
            expected_variants = protocol.get("variants") or list(aggregate["variants"])
            if not protocol.get("variants"):
                aggregate["warnings"].append(
                    "Expected confirmation variants inferred from completed runs; use --expected-variants to include families not yet completed."
                )
    if expected_seeds is None:
        key = "initial_training_seeds" if screening else "confirmation_training_seeds"
        expected_seeds = protocol.get("training_seeds") or protocol.get(
            key, [0, 1, 2] if screening else [5, 6, 7, 8, 9]
        )
    if not all(re.fullmatch(VARIANT_PATTERN, variant) for variant in expected_variants):
        raise ValueError(
            "Expected variants must belong to supported screen families and spacings 1/2"
        )
    if (
        len(set(expected_variants)) != len(expected_variants)
        or len(set(expected_seeds)) != len(expected_seeds)
        or any(seed < 0 for seed in expected_seeds)
    ):
        raise ValueError(
            "Expected variants/seeds must be distinct; seeds must be nonnegative"
        )
    expected = {
        (variant, seed) for variant in expected_variants for seed in expected_seeds
    }
    completed = {
        (variant, seed)
        for variant, group in aggregate["variants"].items()
        for seed in group["seeds"]
    }
    identities = lambda values: [
        {"variant": variant, "seed": seed} for variant, seed in sorted(values)
    ]
    aggregate["inventory"] = {
        "expected_variants": expected_variants,
        "expected_training_seeds": expected_seeds,
        "expected_count": len(expected),
        "completed_expected_count": len(expected & completed),
        "complete": expected <= completed,
        "completed": identities(completed),
        "missing": identities(expected - completed),
        "unexpected_completed": identities(completed - expected),
    }
    if expected - completed:
        aggregate["warnings"].append(
            f"Incomplete collection: {len(expected & completed)}/{len(expected)} expected runs completed. Missing runs are not imputed."
        )
    if completed - expected:
        aggregate["warnings"].append(
            "Completed runs outside expected identities are retained and listed in inventory.unexpected_completed."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        help="Screen protocol; otherwise found in input or its parent",
    )
    parser.add_argument(
        "--collection",
        choices=(
            "training_screen",
            "independent_confirmation",
            "extended_confirmation",
        ),
        help="Declare collection phase, useful before any runs finish",
    )
    parser.add_argument(
        "--expected-variants",
        nargs="+",
        help="Inventory only; every completed selected-family run is retained",
    )
    parser.add_argument(
        "--expected-seeds",
        nargs="+",
        type=int,
        help="Inventory only; defaults to screen 0..2 or confirmation 5..9",
    )
    args = parser.parse_args()
    source, destination = args.input.resolve(), (args.output or args.input).resolve()
    try:
        runs = load_runs(source)
        aggregate = aggregate_runs(runs)
        if args.collection:
            if runs and aggregate["collection"] != args.collection:
                raise ValueError(
                    "Declared collection phase differs from completed run metadata"
                )
            aggregate["collection"] = args.collection
        aggregate["source_directory"] = str(source)
        add_inventory(
            aggregate,
            source,
            args.expected_variants,
            args.expected_seeds,
            args.protocol,
        )
        destination.mkdir(parents=True, exist_ok=True)
        plot_style()
        if runs:
            plot_comparison(runs, aggregate, destination / "comparison_unmasked.png")
            plot_learning(runs, destination / "learning_unmasked.png")
            plot_reliability(aggregate, destination / "reliability_unmasked.png")
        else:
            aggregate["warnings"].append(
                "No completed runs; descriptive statistics and figures are not yet available."
            )
        (destination / "aggregate_unmasked.json").write_text(
            json.dumps(aggregate, indent=2, allow_nan=False) + "\n"
        )
        write_csv(aggregate, destination / "summary_unmasked.csv")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for warning in aggregate["warnings"]:
        print(f"Warning: {warning}")
    print(f"Summarized {len(runs)} completed runs in {destination}")


if __name__ == "__main__":
    main()
