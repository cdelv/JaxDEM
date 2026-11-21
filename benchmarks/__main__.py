# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Simple CLI for running, comparing, recording, and viewing JaxDEM throughput benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax

from .cases import (
    COLLIDERS,
    DIMS,
    DOMAINS,
    FORCES,
    MODES,
    NL_MAX_NEIGHBORS,
    NL_SKIN,
    PROFILES,
    REGIME_ORDER,
    select_cases,
)
from .dashboard import render_standalone_html
from .runner import run_custom, run_suite
from .store import (
    DEFAULT_HISTORY_PATH,
    count_measurements,
    find_previous_run,
    format_compare_table,
    format_run_table,
    load_history,
    resolve_record,
    upsert_run,
)


def _cmd_list(args: argparse.Namespace) -> None:
    cases = select_cases(
        pattern=args.filter,
        dim=args.dim,
        domain=args.domain,
        collider=args.collider,
        force=args.force,
        mode=args.mode,
        regime=args.regime,
    )
    print(f"{'Case':<60} {'φ':>5}  {'v':>5}")
    print("-" * 74)
    for c in cases:
        print(f"{c.name:<60} {c.phi:>5.2f}  {c.vel:>5.2f}")
    print(f"\nTotal: {len(cases)} case(s)")


def _cmd_history(args: argparse.Namespace) -> None:
    history = load_history(args.history)
    if not history:
        print(f"No runs recorded in {args.history}.")
        return
    print(f"{'Commit':<10} {'Date':<12} {'Profile':<8} {'Dtype':<6} {'Points':>6}  Hardware")
    print("-" * 84)
    for rec in history:
        print(
            f"{rec['commit']:<10} {rec['date']:<12} {rec['profile']:<8} "
            f"{rec['dtype']:<6} {count_measurements(rec):>6d}  {rec['hw']}"
        )


def _cmd_compare(args: argparse.Namespace) -> None:
    history = load_history(args.history)
    if args.ref_a is None and args.ref_b is None:
        if len(history) < 2:
            raise SystemExit("Need at least two recorded runs to compare.")
        run_a, run_b = history[-2], history[-1]
    elif args.ref_b is None:
        if not history:
            raise SystemExit("History is empty; pass two commits or JSON files.")
        run_a = history[-1]
        run_b = resolve_record(args.ref_a, history)
    else:
        run_a = resolve_record(args.ref_a, history)
        run_b = resolve_record(args.ref_b, history)
    print(format_compare_table(run_a, run_b))


def _cmd_report(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_standalone_html(history_path=args.history), encoding="utf-8")
    print(f"Wrote benchmark report to {out}")


def _cmd_run(args: argparse.Namespace) -> None:
    if args.x64:
        jax.config.update("jax_enable_x64", True)

    # Custom (phi, vel) single-point mode for external benchmarking outside the 4 fixed regimes.
    if args.phi is not None or args.vel is not None:
        if args.record:
            raise SystemExit(
                "--record only stores the 4 fixed report regimes; omit --phi/--vel to record."
            )
        dim = args.dim or "3d"
        res = run_custom(
            collider=args.collider or "cell_list",
            force=args.force or "spring",
            domain=args.domain or "periodic",
            dim=dim,
            mode=args.mode or "single",
            phi=args.phi if args.phi is not None else (0.60 if dim == "3d" else 0.72),
            vel=args.vel if args.vel is not None else 0.5,
            n=args.n,
            n_naive=args.n_naive,
            batch_size=args.batch_size,
            steps=args.steps,
            repeats=args.repeats,
            profile=args.profile,
            nl_skin=args.nl_skin,
            nl_max_neighbors=args.nl_max_neighbors,
            only_spheres=args.only_spheres,
        )
        print(
            f"{res['dim']}/{res['domain']}/{res['force']}/{res['mode']}/{res['collider']} "
            f"(phi={res['phi']}, vel={res['vel']}, B×N={res['total_particles']}, steps={res['steps']}): "
            f"{res['mps']:.2f} M pstep/s [{res['hw']}]"
        )
        return

    history = load_history(args.history)
    record = run_suite(
        pattern=args.filter,
        dim=args.dim,
        domain=args.domain,
        collider=args.collider,
        force=args.force,
        mode=args.mode,
        regime=args.regime,
        profile=args.profile,
        n=args.n,
        n_naive=args.n_naive,
        batch_size=args.batch_size,
        steps=args.steps,
        repeats=args.repeats,
        nl_skin=args.nl_skin,
        nl_max_neighbors=args.nl_max_neighbors,
        only_spheres=args.only_spheres,
    )
    baseline = find_previous_run(record, history)
    print(format_run_table(record, baseline))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved run JSON to {out}")

    if args.record:
        upsert_run(record, args.history, keep=args.keep)
        size_kb = Path(args.history).stat().st_size / 1024.0
        print(f"\nRecorded commit {record['commit']} in {args.history} ({size_kb:.1f} KB total).")


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-k", "--filter",
        help="Substring or comma-separated filter on case name.",
    )
    parser.add_argument("--dim", help=f"Comma-separated dims ({', '.join(DIMS)}).")
    parser.add_argument("--domain", help=f"Comma-separated domains ({', '.join(DOMAINS)}).")
    parser.add_argument("--collider", help=f"Comma-separated colliders ({', '.join(COLLIDERS)}).")
    parser.add_argument("--force", help=f"Comma-separated force laws ({', '.join(FORCES)}).")
    parser.add_argument("--mode", help=f"Comma-separated modes ({', '.join(MODES)}).")
    parser.add_argument("--regime", help=f"Comma-separated regimes ({', '.join(REGIME_ORDER)}).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Run, compare, and record JaxDEM spheres-in-a-box throughput benchmarks.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="Path to history JSONL file (default: benchmarks/history.jsonl).",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run benchmarks (default command).")
    _add_filter_args(p_run)
    p_run.add_argument("--phi", type=float, help="Custom packing fraction ([0, 0.7] 3D, [0, 0.8] 2D).")
    p_run.add_argument("--vel", type=float, help="Custom initial speed ([0, 1]).")
    p_run.add_argument("--profile", choices=tuple(PROFILES), help="Size preset (auto-detects gpu/cpu).")
    p_run.add_argument("-n", type=int, help="Override particle count N for spatial colliders.")
    p_run.add_argument("--n-naive", type=int, help="Override particle count N for naive / nl_naive.")
    p_run.add_argument("--batch-size", type=int, help="Override batch size B for batch mode (default: 8).")
    p_run.add_argument("--steps", type=int, help="Override steps per timed chunk (default GPU: 2000).")
    p_run.add_argument("--repeats", type=int, help="Override warm timing repeats.")
    p_run.add_argument("--nl-skin", type=float, default=NL_SKIN, help=f"NeighborList skin distance (default: {NL_SKIN}).")
    p_run.add_argument("--nl-max-neighbors", type=int, default=NL_MAX_NEIGHBORS, help=f"NeighborList average slot budget per particle (default: {NL_MAX_NEIGHBORS}).")
    p_run.add_argument("--only-spheres", action="store_true", help="Enable State.only_spheres=True fast path.")
    p_run.add_argument("--x64", action="store_true", help="Enable float64 precision.")
    p_run.add_argument("--record", action="store_true", help="Upsert this run into benchmarks/history.jsonl.")
    p_run.add_argument("--keep", type=int, help="Optional cap on commits kept per hardware/profile when recording.")
    p_run.add_argument("--out", type=Path, help="Optional path to save this run's JSON.")
    p_run.set_defaults(func=_cmd_run)

    p_list = sub.add_parser("list", help="List available benchmark cases.")
    _add_filter_args(p_list)
    p_list.set_defaults(func=_cmd_list)

    p_hist = sub.add_parser("history", help="Show recorded commits in history.jsonl.")
    p_hist.set_defaults(func=_cmd_history)

    p_cmp = sub.add_parser("compare", help="Compare two commits or saved run JSON files.")
    p_cmp.add_argument("ref_a", nargs="?", help="Baseline commit prefix or JSON file.")
    p_cmp.add_argument("ref_b", nargs="?", help="Candidate commit prefix or JSON file.")
    p_cmp.set_defaults(func=_cmd_compare)

    p_rep = sub.add_parser("report", help="Render a standalone HTML dashboard.")
    p_rep.add_argument("--out", type=Path, default=Path("/tmp/jaxdem_benchmarks.html"))
    p_rep.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    known = {"run", "list", "history", "compare", "report"}
    if not any(tok in known for tok in raw) and "-h" not in raw and "--help" not in raw:
        if raw[:1] == ["--history"] and len(raw) >= 2:
            raw = raw[:2] + ["run", *raw[2:]]
        elif raw[:1] and raw[0].startswith("--history="):
            raw = [raw[0], "run", *raw[1:]]
        else:
            raw = ["run", *raw]
    args = build_parser().parse_args(raw)
    args.func(args)


if __name__ == "__main__":
    main()
