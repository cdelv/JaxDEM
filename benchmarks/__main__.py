# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Simple CLI for running, comparing, recording, and viewing JaxDEM throughput benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
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
from .scaling import format_scaling_table, parse_bool_token, plot_scaling, run_scaling
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
    print(
        f"{'Commit':<10} {'Date':<12} {'Profile':<8} {'Dtype':<6} {'Points':>6}  Hardware"
    )
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


def _cmd_scaling(args: argparse.Namespace) -> None:
    if args.x64:
        jax.config.update("jax_enable_x64", True)

    if args.from_json:
        record = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
    else:

        def _split_str(raw: str) -> list[str]:
            return [tok.strip() for tok in raw.split(",") if tok.strip()]

        def _split_float(raw: str) -> list[float]:
            return [float(tok.strip()) for tok in raw.split(",") if tok.strip()]

        def _split_bool(raw: str) -> list[bool]:
            return [parse_bool_token(tok) for tok in raw.split(",") if tok.strip()]

        colliders = _split_str(args.collider) if args.collider else None
        ns = [int(tok) for tok in _split_str(args.ns)] if args.ns else None
        pallas_vals = _split_bool(args.pallas) if args.pallas is not None else None
        spheres_vals = _split_bool(args.only_spheres)

        record = run_scaling(
            colliders=colliders,
            ns=ns,
            min_exp=args.min_exp,
            max_exp=args.max_exp,
            step_exp=args.step_exp,
            min_n=args.min_n,
            max_n=args.max_n,
            dim=_split_str(args.dim),
            domain=_split_str(args.domain),
            force=_split_str(args.force),
            mode=_split_str(args.mode),
            phi=_split_float(args.phi),
            vel=_split_float(args.vel),
            alpha=_split_float(args.alpha),
            only_spheres=spheres_vals,
            pallas=pallas_vals,
            batch_size=args.batch_size,
            steps=args.steps,
            steps_naive=args.steps_naive,
            profile=args.profile,
            nl_skin=args.nl_skin,
            nl_max_neighbors=args.nl_max_neighbors,
            max_n_naive=args.max_n_naive,
            ignore_naive_limits=args.ignore_naive_limits,
            max_seconds=args.max_seconds,
        )

    print(format_scaling_table(record))
    if args.no_save:
        return

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved scaling JSON to {out}")

    if args.plot:
        plot_path = plot_scaling(record, args.plot)
        print(f"Saved log-log scaling plot to {plot_path}")


def _cmd_run(args: argparse.Namespace) -> None:
    if args.x64:
        jax.config.update("jax_enable_x64", True)

    pallas_opt = parse_bool_token(args.pallas) if args.pallas is not None else None

    # Custom (phi, vel) single-point mode for external benchmarking outside the fixed regimes.
    if args.phi is not None or args.vel is not None:
        if args.record and not args.no_save:
            raise SystemExit(
                "--record only stores the fixed report regimes; omit --phi/--vel to record."
            )
        dim = args.dim or "3d"
        res = run_custom(
            collider=args.collider or "cell_list",
            force=args.force or "spring",
            domain=args.domain or "periodic",
            dim=dim,
            mode=args.mode or "single",
            phi=args.phi if args.phi is not None else 0.55,
            vel=args.vel if args.vel is not None else 0.6,
            n=args.n,
            n_naive=args.n_naive,
            n_naive_batch=args.n_naive_batch,
            batch_size=args.batch_size,
            steps=args.steps,
            steps_naive=args.steps_naive,
            profile=args.profile,
            nl_skin=args.nl_skin,
            nl_max_neighbors=args.nl_max_neighbors,
            alpha=args.alpha,
            only_spheres=args.only_spheres,
            pallas=pallas_opt,
        )
        rb_str = (
            f", nl_rebuilds={res['nl_rebuilds']}"
            if res.get("nl_rebuilds") is not None
            else ""
        )
        pal_str = f", pallas={'on' if res.get('pallas') else 'off'}"
        print(
            f"{res['dim']}/{res['domain']}/{res['force']}/{res['mode']}/{res['collider']} "
            f"(phi={res['phi']}, vel={res['vel']}, alpha={res['alpha']:g}{pal_str}, "
            f"B×N={res['total_particles']}, steps={res['steps']}{rb_str}): "
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
        n_naive_batch=args.n_naive_batch,
        batch_size=args.batch_size,
        steps=args.steps,
        steps_naive=args.steps_naive,
        nl_skin=args.nl_skin,
        nl_max_neighbors=args.nl_max_neighbors,
        alpha=args.alpha,
        only_spheres=args.only_spheres,
        pallas=pallas_opt,
    )
    baseline = find_previous_run(record, history)
    print(format_run_table(record, baseline))

    if args.no_save:
        return

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved run JSON to {out}")

    if args.record:
        upsert_run(record, args.history, keep=args.keep)
        size_kb = Path(args.history).stat().st_size / 1024.0
        print(
            f"\nRecorded commit {record['commit']} in {args.history} ({size_kb:.1f} KB total)."
        )


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-k",
        "--filter",
        help="Substring or comma-separated filter on case name.",
    )
    parser.add_argument("--dim", help=f"Comma-separated dims ({', '.join(DIMS)}).")
    parser.add_argument(
        "--domain", help=f"Comma-separated domains ({', '.join(DOMAINS)})."
    )
    parser.add_argument(
        "--collider", help=f"Comma-separated colliders ({', '.join(COLLIDERS)})."
    )
    parser.add_argument(
        "--force", help=f"Comma-separated force laws ({', '.join(FORCES)})."
    )
    parser.add_argument("--mode", help=f"Comma-separated modes ({', '.join(MODES)}).")
    parser.add_argument(
        "--regime", help=f"Comma-separated regimes ({', '.join(REGIME_ORDER)})."
    )


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
    p_run.add_argument(
        "--phi", type=float, help="Custom packing fraction ([0, 0.7] 3D, [0, 0.8] 2D)."
    )
    p_run.add_argument("--vel", type=float, help="Custom initial speed ([0, 1]).")
    p_run.add_argument(
        "--profile", choices=tuple(PROFILES), help="Size preset (auto-detects gpu/cpu)."
    )
    p_run.add_argument(
        "-n", type=int, help="Override particle count N for spatial colliders."
    )
    p_run.add_argument(
        "--n-naive",
        type=int,
        help="Override single-mode particle count N for naive / nl_naive (default GPU: 3000).",
    )
    p_run.add_argument(
        "--n-naive-batch",
        type=int,
        help="Override per-sim batch particle count N for naive / nl_naive (default GPU: 2000).",
    )
    p_run.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size B for batch mode (default: 6).",
    )
    p_run.add_argument(
        "--steps", type=int, help="Override steps per timed chunk (default GPU: 1000)."
    )
    p_run.add_argument(
        "--steps-naive",
        type=int,
        help="Override steps for the naive collider (default GPU: 100).",
    )
    p_run.add_argument(
        "--nl-skin",
        type=float,
        default=NL_SKIN,
        help=f"NeighborList skin distance (default: {NL_SKIN}).",
    )
    p_run.add_argument(
        "--nl-max-neighbors",
        type=int,
        default=NL_MAX_NEIGHBORS,
        help=f"NeighborList average slot budget per particle (default: {NL_MAX_NEIGHBORS}).",
    )
    p_run.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Target mean polydispersity E[max_rad/r] >= 1.0 for r ~ U[r_min, 1.0] (default: 1.0).",
    )
    p_run.add_argument(
        "--only-spheres",
        action="store_true",
        help="Enable State.only_spheres=True fast path.",
    )
    p_run.add_argument(
        "--pallas",
        help="Override Pallas Triton GPU kernels: on or off (default: auto).",
    )
    p_run.add_argument("--x64", action="store_true", help="Enable float64 precision.")
    p_run.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "jaxdem_benchmark_cache",
        help="Persistent JAX compilation cache directory (default: <tmp>/jaxdem_benchmark_cache).",
    )
    p_run.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable persistent JAX compilation cache.",
    )
    p_run.add_argument(
        "--no-save",
        action="store_true",
        help="Print results to stdout only without writing to disk (default unless --record/--out is passed).",
    )
    p_run.add_argument(
        "--record",
        action="store_true",
        help="Upsert this run into benchmarks/history.jsonl (opt-in; nothing is saved by default).",
    )
    p_run.add_argument(
        "--keep",
        type=int,
        help="Optional cap on commits kept per hardware/profile when recording.",
    )
    p_run.add_argument(
        "--out", type=Path, help="Optional path to save this run's JSON."
    )
    p_run.set_defaults(func=_cmd_run)

    p_list = sub.add_parser("list", help="List available benchmark cases.")
    _add_filter_args(p_list)
    p_list.set_defaults(func=_cmd_list)

    p_hist = sub.add_parser("history", help="Show recorded commits in history.jsonl.")
    p_hist.set_defaults(func=_cmd_history)

    p_cmp = sub.add_parser(
        "compare", help="Compare two commits or saved run JSON files."
    )
    p_cmp.add_argument("ref_a", nargs="?", help="Baseline commit prefix or JSON file.")
    p_cmp.add_argument("ref_b", nargs="?", help="Candidate commit prefix or JSON file.")
    p_cmp.set_defaults(func=_cmd_compare)

    p_rep = sub.add_parser("report", help="Render a standalone HTML dashboard.")
    p_rep.add_argument("--out", type=Path, default=Path("/tmp/jaxdem_benchmarks.html"))
    p_rep.set_defaults(func=_cmd_report)

    p_scl = sub.add_parser(
        "scaling",
        help="Sweep throughput vs N in powers of 2 (log-log) across colliders and varying parameters.",
    )
    p_scl.add_argument(
        "--collider",
        help=f"Comma-separated colliders (default: all valid for --force; options: {', '.join(COLLIDERS)}).",
    )
    p_scl.add_argument(
        "--dim",
        default="3d",
        help=f"Dim or comma-separated dims ({', '.join(DIMS)}; default: 3d).",
    )
    p_scl.add_argument(
        "--domain",
        default="periodic",
        help=f"Domain or comma-separated domains ({', '.join(DOMAINS)}; default: periodic).",
    )
    p_scl.add_argument(
        "--force",
        default="spring",
        help=f"Force or comma-separated forces ({', '.join(FORCES)}; default: spring).",
    )
    p_scl.add_argument(
        "--mode",
        default="single",
        help=f"Mode or comma-separated modes ({', '.join(MODES)}; default: single).",
    )
    p_scl.add_argument(
        "--phi",
        default="0.55",
        help="Packing fraction or comma-separated list (default: 0.55).",
    )
    p_scl.add_argument(
        "--vel",
        default="0.15",
        help="Initial velocity or comma-separated list (default: 0.15).",
    )
    p_scl.add_argument(
        "--alpha",
        default="1.0",
        help="Polydispersity ratio or comma-separated list (default: 1.0).",
    )
    p_scl.add_argument(
        "--only-spheres",
        nargs="?",
        const="true",
        default="false",
        help="Enable only_spheres, or pass comma-separated false,true to compare both.",
    )
    p_scl.add_argument(
        "--pallas",
        help="Pallas Triton GPU kernels: on, off, or on,off to compare both curves (default: auto).",
    )
    p_scl.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        help="Profile for default steps/batch (auto-detects gpu/cpu).",
    )
    p_scl.add_argument(
        "--batch-size",
        type=int,
        help="Batch size B when --mode includes batch (default: 6).",
    )
    p_scl.add_argument(
        "--steps",
        type=int,
        help="Steps per warmup/timed chunk (default GPU: 1000, CPU: 200).",
    )
    p_scl.add_argument(
        "--steps-naive",
        type=int,
        help="Steps for naive collider (default GPU: 100, CPU: 50).",
    )
    p_scl.add_argument(
        "--min-exp",
        type=int,
        default=1,
        help="Min power of 2 for N (default: 1 -> N=2).",
    )
    p_scl.add_argument(
        "--max-exp",
        type=int,
        default=21,
        help="Max power of 2 for N (default: 21 -> N=2,097,152).",
    )
    p_scl.add_argument(
        "--step-exp",
        type=int,
        default=1,
        help="Exponent stride for powers of 2 (default: 1).",
    )
    p_scl.add_argument(
        "--min-n",
        type=int,
        help="Minimum N (overrides --min-exp, rounds up to power of 2).",
    )
    p_scl.add_argument(
        "--max-n",
        type=int,
        help="Maximum N (overrides --max-exp, rounds down to power of 2).",
    )
    p_scl.add_argument(
        "--ns", help="Explicit comma-separated N values (overrides --min-*/--max-*)."
    )
    p_scl.add_argument(
        "--max-n-naive",
        type=int,
        default=8_192,
        help="Cap N for naive/nl_naive (default: 8192).",
    )
    p_scl.add_argument(
        "--ignore-naive-limits",
        action="store_true",
        help="Disable both the N=8192 cap and reduced step count for naive/nl_naive.",
    )
    p_scl.add_argument(
        "--max-seconds",
        type=float,
        help="Stop sweeping larger N for a curve once one point exceeds this many seconds.",
    )
    p_scl.add_argument(
        "--nl-skin",
        type=float,
        default=NL_SKIN,
        help=f"NeighborList skin distance (default: {NL_SKIN}).",
    )
    p_scl.add_argument(
        "--nl-max-neighbors",
        type=int,
        default=NL_MAX_NEIGHBORS,
        help=f"NeighborList average slot budget per particle (default: {NL_MAX_NEIGHBORS}).",
    )
    p_scl.add_argument("--x64", action="store_true", help="Enable float64 precision.")
    p_scl.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "jaxdem_benchmark_cache",
        help="Persistent JAX compilation cache directory (default: <tmp>/jaxdem_benchmark_cache).",
    )
    p_scl.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable persistent JAX compilation cache.",
    )
    p_scl.add_argument(
        "--from-json",
        type=Path,
        help="Load an existing scaling JSON instead of running benchmarks.",
    )
    p_scl.add_argument(
        "--out", type=Path, help="Optional path to save scaling results JSON."
    )
    p_scl.add_argument(
        "--plot",
        type=Path,
        help="Optional path to save log-log plot (.png/.pdf/.svg via matplotlib, or interactive .html).",
    )
    p_scl.add_argument(
        "--no-save",
        action="store_true",
        help="Print table to stdout only without writing files.",
    )
    p_scl.set_defaults(func=_cmd_scaling)

    return parser


def _configure_compilation_cache(args: argparse.Namespace) -> None:
    """Enable JAX's persistent compilation cache across isolated benchmark scopes."""
    import os

    if (
        getattr(args, "no_cache", False)
        or os.environ.get("JAXDEM_BENCH_NO_CACHE") == "1"
    ):
        os.environ["JAXDEM_BENCH_NO_CACHE"] = "1"
        return
    cache_dir: Path | None = getattr(args, "cache_dir", None)
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_str = str(cache_dir)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_str)
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    jax.config.update(
        "jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"]
    )
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    known = {"run", "list", "history", "compare", "report", "scaling"}
    if not any(tok in known for tok in raw) and "-h" not in raw and "--help" not in raw:
        if raw[:1] == ["--history"] and len(raw) >= 2:
            raw = raw[:2] + ["run", *raw[2:]]
        elif raw[:1] and raw[0].startswith("--history="):
            raw = [raw[0], "run", *raw[1:]]
        else:
            raw = ["run", *raw]
    args = build_parser().parse_args(raw)
    _configure_compilation_cache(args)
    args.func(args)


if __name__ == "__main__":
    main()
