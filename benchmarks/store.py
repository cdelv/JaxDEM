# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Compact nested JSONL throughput storage and matrix comparison tables."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterator

from .cases import COLLIDERS, DIMS, DOMAINS, FORCES, MODES, REGIME_ORDER, REGIMES

DEFAULT_HISTORY_PATH = Path(__file__).resolve().parent / "history.jsonl"


def _series_key(rec: dict[str, Any]) -> tuple[str, str, str, bool]:
    return (
        rec["hw"],
        rec["profile"],
        rec["dtype"],
        bool(rec.get("only_spheres", False)),
    )


def iter_flat(rec: dict[str, Any]) -> Iterator[tuple[str, float]]:
    """Yield ``(case_name, mps)`` pairs from a nested v3 record."""
    data = rec.get("data", {})
    for dim, by_dom in data.items():
        for dom, by_frc in by_dom.items():
            for frc, by_mod in by_frc.items():
                for mod, by_col in by_mod.items():
                    for col, vals in by_col.items():
                        for reg, mps in zip(REGIME_ORDER, vals):
                            if mps is not None:
                                yield f"{dim}/{dom}/{frc}/{mod}/{col}/{reg}", float(mps)


def count_measurements(rec: dict[str, Any]) -> int:
    return sum(1 for _ in iter_flat(rec))


def _merge_nested(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dst)
    for dim, by_dom in src.items():
        d_dim = out.setdefault(dim, {})
        for dom, by_frc in by_dom.items():
            d_dom = d_dim.setdefault(dom, {})
            for frc, by_mod in by_frc.items():
                d_frc = d_dom.setdefault(frc, {})
                for mod, by_col in by_mod.items():
                    d_mod = d_frc.setdefault(mod, {})
                    for col, vals in by_col.items():
                        row = list(d_mod.get(col, [None] * len(REGIME_ORDER)))
                        for idx, v in enumerate(vals):
                            if v is not None:
                                row[idx] = v
                        d_mod[col] = row
    return out


def load_history(path: Path | str = DEFAULT_HISTORY_PATH) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def save_history(
    records: list[dict[str, Any]], path: Path | str = DEFAULT_HISTORY_PATH
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(rec, separators=(",", ":")) for rec in records]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def upsert_run(
    record: dict[str, Any],
    path: Path | str = DEFAULT_HISTORY_PATH,
    *,
    keep: int | None = None,
) -> list[dict[str, Any]]:
    """Insert or deep-merge a run for ``(commit, hw, profile, dtype, only_spheres)``."""
    records = load_history(path)
    key = (record["commit"],) + _series_key(record)

    replaced = False
    for idx, existing in enumerate(records):
        if (existing["commit"],) + _series_key(existing) == key:
            merged = dict(record)
            merged["data"] = _merge_nested(
                existing.get("data", {}), record.get("data", {})
            )
            records[idx] = merged
            replaced = True
            break
    if not replaced:
        records.append(record)

    if keep is not None and keep > 0:
        counts: dict[tuple[str, str, str, bool], int] = {}
        pruned: list[dict[str, Any]] = []
        for rec in reversed(records):
            sk = _series_key(rec)
            if counts.get(sk, 0) < keep:
                pruned.append(rec)
                counts[sk] = counts.get(sk, 0) + 1
        records = list(reversed(pruned))

    save_history(records, path)
    return records


def find_previous_run(
    record: dict[str, Any], history: list[dict[str, Any]]
) -> dict[str, Any] | None:
    target = _series_key(record)
    for rec in reversed(history):
        if _series_key(rec) == target and rec["commit"] != record["commit"]:
            return rec
    return None


def resolve_record(ref: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = Path(ref)
    if candidate.is_file():
        return json.loads(candidate.read_text(encoding="utf-8"))
    for rec in reversed(history):
        if rec["commit"].startswith(ref):
            return rec
    raise ValueError(f"Could not find benchmark record matching {ref!r}.")


def _fmt_delta(new_mps: float, old_mps: float | None) -> str:
    if old_mps is None or old_mps <= 0.0:
        return "—"
    pct = (new_mps - old_mps) / old_mps * 100.0
    sign = "+" if pct > 0 else ""
    flag = " faster" if pct > 1.0 else (" SLOWER" if pct < -5.0 else "")
    return f"{sign}{pct:5.1f}%{flag}"


def format_run_table(
    record: dict[str, Any], baseline: dict[str, Any] | None = None
) -> str:
    """Render Collider x 4-Regimes throughput matrices (M particle-steps/s)."""
    data = record.get("data", {})
    base_flat = dict(iter_flat(baseline)) if baseline else {}
    base_label = baseline["commit"] if baseline else "prev"

    out: list[str] = []
    all_ratios: list[float] = []

    for dim in DIMS:
        for dom in DOMAINS:
            for frc in FORCES:
                for mod in MODES:
                    by_col = data.get(dim, {}).get(dom, {}).get(frc, {}).get(mod, {})
                    if not by_col:
                        continue
                    out.append(
                        f"\n[{dim} | {dom} | {frc} | {mod}] Throughput (M particle-steps/s)"
                    )
                    header = (
                        f"{'Collider':<26}"
                        + "".join(f"{REGIMES[r]['short']:>13}" for r in REGIME_ORDER)
                        + f"  {'Δ vs ' + base_label:>14}"
                    )
                    out.append(header)
                    out.append("-" * len(header))

                    for col in COLLIDERS:
                        vals = by_col.get(col)
                        if not vals:
                            continue
                        cells: list[str] = []
                        col_ratios: list[float] = []
                        for reg, mps in zip(REGIME_ORDER, vals):
                            if mps is None:
                                cells.append(f"{'—':>13}")
                            else:
                                cells.append(f"{mps:>13.2f}")
                                old = base_flat.get(f"{dim}/{dom}/{frc}/{mod}/{col}/{reg}")
                                if old and old > 0 and mps > 0:
                                    col_ratios.append(mps / old)
                                    all_ratios.append(mps / old)

                        if col_ratios:
                            gm = math.exp(sum(math.log(x) for x in col_ratios) / len(col_ratios))
                            delta = _fmt_delta(gm, 1.0)
                        else:
                            delta = "—"
                        out.append(f"{col:<26}" + "".join(cells) + f"  {delta:>14}")

    if all_ratios:
        gm_all = math.exp(sum(math.log(x) for x in all_ratios) / len(all_ratios))
        out.append(
            f"\nOverall geometric mean throughput ratio vs {base_label}: "
            f"{gm_all:.2f}x ({len(all_ratios)} shared case(s))"
        )
    return "\n".join(out)


def format_compare_table(run_a: dict[str, Any], run_b: dict[str, Any]) -> str:
    """Render a side-by-side throughput comparison between two records."""
    flat_a = dict(iter_flat(run_a))
    flat_b = dict(iter_flat(run_b))
    shared = [k for k in flat_b if k in flat_a]
    if not shared:
        return "No shared benchmark cases between the two records."

    lines = [
        f"Throughput comparison (M particle-steps/s): "
        f"{run_a['commit']} ({run_a['date']}) -> {run_b['commit']} ({run_b['date']})",
        f"Hardware: {run_a['hw']} -> {run_b['hw']}",
        "",
        f"{'Case':<58} {'Base M/s':>10} {'New M/s':>10} {'Speedup':>9} {'Δ':>14}",
        "-" * 105,
    ]
    ratios: list[float] = []
    for name in shared:
        a, b = flat_a[name], flat_b[name]
        ratio = b / a if a > 0 else float("nan")
        if a > 0 and b > 0:
            ratios.append(ratio)
        lines.append(
            f"{name:<58} {a:>10.2f} {b:>10.2f} {ratio:>8.2f}x {_fmt_delta(b, a):>14}"
        )

    if ratios:
        gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        lines.append("-" * 105)
        lines.append(f"Geometric mean speedup: {gm:.2f}x across {len(ratios)} cases")
    return "\n".join(lines)
