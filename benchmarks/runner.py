# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Device-synchronized throughput runner with high-performance throttle policy checks."""

from __future__ import annotations

import datetime
import gc
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

from .cases import (
    NL_MAX_NEIGHBORS,
    NL_SKIN,
    PROFILES,
    REGIME_ORDER,
    BenchmarkCase,
    build_case,
    build_spheres_in_box,
    select_cases,
)


def _block(tree: Any) -> None:
    jax.block_until_ready(tree)


def _check_collider_overflow(collider: Any) -> None:
    """Recursively verify zero overflow on a collider and any secondary collider."""
    if collider is None:
        return
    overflow = getattr(collider, "overflow", None)
    if overflow is not None and bool(np.any(np.asarray(overflow))):
        raise RuntimeError(
            f"benchmark collider ({type(collider).__name__}) reported overflow"
        )
    secondary = getattr(collider, "secondary_collider", None)
    if secondary is not None:
        _check_collider_overflow(secondary)


def _validate(output: Any) -> None:
    """Verify finiteness and zero collider/search overflow outside timed loops."""
    for leaf in jax.tree.leaves(output):
        arr = np.asarray(leaf)
        if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
            raise ValueError("benchmark produced non-finite output")

    stack = [output]
    while stack:
        item = stack.pop()
        if isinstance(item, (tuple, list)):
            stack.extend(item)
            continue
        if hasattr(item, "check_overflow"):
            item.check_overflow()
        search_overflow = getattr(item, "search_overflow", None)
        if search_overflow is not None and bool(np.any(np.asarray(search_overflow))):
            raise RuntimeError("benchmark system reported search_overflow")
        _check_collider_overflow(getattr(item, "collider", None))


def _get_nl_build_count(system: Any) -> np.ndarray | None:
    """Return NeighborList.n_build_times as an int64 host array, or None."""
    collider = getattr(system, "collider", None)
    cnt = getattr(collider, "n_build_times", None)
    if cnt is None:
        return None
    return np.asarray(cnt, dtype=np.int64)


def check_throttle_policy() -> list[str]:
    """Check Linux CPU/platform/GPU power profiles and return warnings if not high-performance."""
    warnings: list[str] = []

    platform_profile = Path("/sys/firmware/acpi/platform_profile")
    if platform_profile.is_file():
        try:
            val = platform_profile.read_text(encoding="utf-8").strip().lower()
            if val and val != "performance":
                warnings.append(
                    f"ACPI platform_profile is '{val}' (expected 'performance')."
                )
        except OSError:
            pass

    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov.is_file():
        try:
            val = gov.read_text(encoding="utf-8").strip().lower()
            if val and val != "performance":
                warnings.append(
                    f"CPU scaling_governor is '{val}' (expected 'performance')."
                )
        except OSError:
            pass

    epp = Path("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference")
    if epp.is_file():
        try:
            val = epp.read_text(encoding="utf-8").strip().lower()
            if val and val != "performance":
                warnings.append(
                    f"CPU energy_performance_preference is '{val}' (expected 'performance')."
                )
        except OSError:
            pass

    if shutil.which("powerprofilesctl"):
        try:
            val = (
                subprocess.check_output(
                    ["powerprofilesctl", "get"], text=True, stderr=subprocess.DEVNULL
                )
                .strip()
                .lower()
            )
            if val and val != "performance":
                warnings.append(
                    f"powerprofilesctl profile is '{val}' (run: powerprofilesctl set performance)."
                )
        except Exception:
            pass

    if shutil.which("tuned-adm"):
        try:
            out = subprocess.check_output(
                ["tuned-adm", "active"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            active = out.split(":")[-1].strip().lower()
            if active and "performance" not in active:
                warnings.append(
                    f"tuned-adm profile is '{active}' (run: sudo tuned-adm profile throughput-performance)."
                )
        except Exception:
            pass

    if jax.default_backend() == "gpu" and shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown",
                    "--format=csv,noheader",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            first = out.splitlines()[0] if out else ""
            if any(tok.strip().lower() == "active" for tok in first.split(",")):
                warnings.append(
                    f"GPU thermal/power throttling is active (nvidia-smi: {first})."
                )
        except Exception:
            pass

    return warnings


def _warn_if_throttled() -> list[str]:
    warnings = check_throttle_policy()
    if warnings:
        print(
            "\nWARNING: System power/throttle policy is NOT set to high performance!",
            file=sys.stderr,
        )
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
        print(
            "  Results may be noisy or artificially low.\n",
            file=sys.stderr,
            flush=True,
        )
    return warnings


def _git_info() -> tuple[str, str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"], text=True
        ).strip()
        date = subprocess.check_output(
            ["git", "show", "-s", "--format=%cs", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
        return commit, date, dirty
    except Exception:
        return "unknown", datetime.date.today().isoformat(), False


def get_hardware_label() -> str:
    devices = jax.devices()
    if not devices:
        return "unknown"
    dev = devices[0]
    kind = dev.device_kind.replace("NVIDIA GeForce ", "").strip()
    return f"{dev.platform}: {kind}"


def default_profile() -> str:
    return "gpu" if jax.default_backend() != "cpu" else "cpu"


def _round_sig(val: float, sig: int = 4) -> float:
    if val == 0.0 or not np.isfinite(val):
        return 0.0
    return float(f"{val:.{sig}g}")


def _run_and_time(
    built: tuple[Any, Any, Any, int, int],
) -> tuple[float, int, int, int | float | None]:
    """Warm up for ``actual_steps``, sync, then measure one run from the mixed state."""
    step_fn, state, system, total_particles, actual_steps = built
    del built

    # Ensure initial buffers are ready before launching warmup.
    _block((state, system))

    # Warmup runs the same step count so JIT compiles and particles mix off the lattice.
    state, system = step_fn(state, system)
    _block((state, system))
    _validate((state, system))
    builds_before = _get_nl_build_count(system)

    # Single timed measurement (no repeats) starting at step=actual_steps.
    t0 = time.perf_counter()
    state, system = step_fn(state, system)
    _block((state, system))
    elapsed_s = time.perf_counter() - t0

    _validate((state, system))
    builds_after = _get_nl_build_count(system)

    nl_rebuilds: int | float | None = None
    if builds_before is not None and builds_after is not None:
        delta = builds_after - builds_before
        if delta.ndim == 0:
            nl_rebuilds = int(delta)
        else:
            mean_val = float(np.mean(delta))
            nl_rebuilds = int(mean_val) if mean_val.is_integer() else round(mean_val, 1)

    del step_fn, state, system
    gc.collect()
    mps = _round_sig((total_particles * actual_steps) / max(elapsed_s, 1e-12) / 1e6)
    return mps, total_particles, actual_steps, nl_rebuilds


def run_custom(
    *,
    collider: str = "cell_list",
    force: str = "spring",
    domain: str = "periodic",
    dim: str = "3d",
    mode: str = "single",
    phi: float = 0.55,
    vel: float = 0.6,
    n: int | None = None,
    n_naive: int | None = None,
    n_naive_batch: int | None = None,
    batch_size: int | None = None,
    steps: int | None = None,
    steps_naive: int | None = None,
    profile: str | None = None,
    nl_skin: float = NL_SKIN,
    nl_max_neighbors: int = NL_MAX_NEIGHBORS,
    alpha: float = 1.0,
    only_spheres: bool = False,
) -> dict[str, Any]:
    """Run one custom spheres-in-a-box point at any valid ``phi``, ``vel``, and ``alpha``."""
    _warn_if_throttled()
    preset = PROFILES[profile or default_profile()]
    run_n = n if n is not None else preset["n"]
    run_n_naive = n_naive if n_naive is not None else preset["n_naive"]
    run_n_naive_batch = (
        n_naive_batch if n_naive_batch is not None else preset["n_naive_batch"]
    )
    run_batch = batch_size if batch_size is not None else preset["batch"]
    run_steps = steps if steps is not None else preset["steps"]
    run_steps_naive = steps_naive if steps_naive is not None else preset["steps_naive"]

    mps, total_particles, actual_steps, nl_rebuilds = _run_and_time(
        build_spheres_in_box(
            collider=collider,
            force=force,
            domain=domain,
            dim=dim,
            mode=mode,
            phi=phi,
            vel=vel,
            n=run_n,
            n_naive=run_n_naive,
            n_naive_batch=run_n_naive_batch,
            batch_size=run_batch,
            steps=run_steps,
            steps_naive=run_steps_naive,
            nl_skin=nl_skin,
            nl_max_neighbors=nl_max_neighbors,
            alpha=alpha,
            only_spheres=only_spheres,
        )
    )
    return {
        "dim": dim,
        "domain": domain,
        "force": force,
        "mode": mode,
        "collider": collider,
        "phi": phi,
        "vel": vel,
        "alpha": alpha,
        "total_particles": total_particles,
        "steps": actual_steps,
        "mps": mps,
        "nl_rebuilds": nl_rebuilds,
        "hw": get_hardware_label(),
    }


def _put_nested(
    data: dict[str, Any], case: BenchmarkCase, mps: float
) -> None:
    d_dim = data.setdefault(case.dim, {})
    d_dom = d_dim.setdefault(case.domain, {})
    d_frc = d_dom.setdefault(case.force, {})
    d_mod = d_frc.setdefault(case.mode, {})
    row = d_mod.setdefault(case.collider, [None] * len(REGIME_ORDER))
    row[REGIME_ORDER.index(case.regime)] = mps


def _put_rebuilds(
    rebuilds: dict[str, Any],
    dim: str,
    domain: str,
    force: str,
    mode: str,
    reg_idx: int,
    value: int | float,
) -> None:
    d_dim = rebuilds.setdefault(dim, {})
    d_dom = d_dim.setdefault(domain, {})
    d_frc = d_dom.setdefault(force, {})
    row = d_frc.setdefault(mode, [None] * len(REGIME_ORDER))
    row[reg_idx] = value


def run_suite(
    *,
    pattern: str | None = None,
    dim: str | None = None,
    domain: str | None = None,
    collider: str | None = None,
    force: str | None = None,
    mode: str | None = None,
    regime: str | None = None,
    profile: str | None = None,
    n: int | None = None,
    n_naive: int | None = None,
    n_naive_batch: int | None = None,
    batch_size: int | None = None,
    steps: int | None = None,
    steps_naive: int | None = None,
    nl_skin: float = NL_SKIN,
    nl_max_neighbors: int = NL_MAX_NEIGHBORS,
    alpha: float = 1.0,
    only_spheres: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run selected fixed-regime benchmarks and return a compact v3 throughput record."""
    prof_name = profile or default_profile()
    preset = PROFILES[prof_name]
    run_n = n if n is not None else preset["n"]
    run_n_naive = n_naive if n_naive is not None else preset["n_naive"]
    run_n_naive_batch = (
        n_naive_batch if n_naive_batch is not None else preset["n_naive_batch"]
    )
    run_batch = batch_size if batch_size is not None else preset["batch"]
    run_steps = steps if steps is not None else preset["steps"]
    run_steps_naive = steps_naive if steps_naive is not None else preset["steps_naive"]

    selected = select_cases(
        pattern=pattern,
        dim=dim,
        domain=domain,
        collider=collider,
        force=force,
        mode=mode,
        regime=regime,
    )
    if not selected:
        raise ValueError("No benchmark cases matched the requested filters.")

    throttle_warnings = _warn_if_throttled()
    commit, date, dirty = _git_info()
    hw = get_hardware_label()
    dtype = "f64" if jax.config.x64_enabled else "f32"

    if verbose:
        print(
            f"JaxDEM Spheres-in-Box | {hw} | profile={prof_name} "
            f"(N={run_n}, naive={run_n_naive}/{run_n_naive_batch}, B={run_batch}, "
            f"steps={run_steps}, steps_naive={run_steps_naive}, alpha={alpha:g}, "
            f"only_spheres={only_spheres}) | {dtype} | commit={commit}{'*' if dirty else ''}"
        )

    nested: dict[str, Any] = {}
    nl_counts_spatial: dict[tuple[str, str, str, str, int], dict[str, int | float]] = {}
    nl_counts_naive: dict[tuple[str, str, str, str, int], int | float] = {}
    t_suite = time.perf_counter()
    for i, case in enumerate(selected, 1):
        if verbose:
            print(f"  [{i:3d}/{len(selected):3d}] {case.name:<58s} ... ", end="", flush=True)
        mps, _, _, nl_rb = _run_and_time(
            build_case(
                case,
                n=run_n,
                n_naive=run_n_naive,
                n_naive_batch=run_n_naive_batch,
                batch_size=run_batch,
                steps=run_steps,
                steps_naive=run_steps_naive,
                nl_skin=nl_skin,
                nl_max_neighbors=nl_max_neighbors,
                alpha=alpha,
                only_spheres=only_spheres,
            )
        )
        _put_nested(nested, case, mps)
        if nl_rb is not None:
            key = (case.dim, case.domain, case.force, case.mode, REGIME_ORDER.index(case.regime))
            if case.collider == "nl_naive":
                nl_counts_naive[key] = nl_rb
            else:
                nl_counts_spatial.setdefault(key, {})[case.collider] = nl_rb
        if verbose:
            rb_str = f" ({nl_rb} rebuilds)" if nl_rb is not None else ""
            print(f"{mps:7.2f} M pstep/s{rb_str}")

    nl_rebuilds: dict[str, Any] = {}
    nl_naive_rebuilds: dict[str, Any] = {}
    for key in sorted(nl_counts_spatial):
        dim_k, dom_k, frc_k, mod_k, reg_idx = key
        by_col = nl_counts_spatial[key]
        vals = list(by_col.values())
        lo, hi = min(vals), max(vals)
        # Spatial NL colliders share identical N and initial conditions; flag any divergence
        # beyond tiny float32 chaotic drift (±2 rebuilds or 5%).
        tol = max(2.0, 0.05 * hi)
        if (hi - lo) > tol:
            details = ", ".join(f"{c}={v}" for c, v in by_col.items())
            print(
                f"WARNING: NL rebuild count mismatch for "
                f"{dim_k}/{dom_k}/{frc_k}/{mod_k}/{REGIME_ORDER[reg_idx]}: {details}",
                file=sys.stderr,
            )
        med = float(np.median(np.asarray(vals, dtype=np.float64)))
        canon: int | float = (
            int(round(med)) if all(isinstance(v, int) for v in vals) else round(med, 1)
        )
        _put_rebuilds(nl_rebuilds, dim_k, dom_k, frc_k, mod_k, reg_idx, canon)
    for key in sorted(nl_counts_naive):
        dim_k, dom_k, frc_k, mod_k, reg_idx = key
        _put_rebuilds(
            nl_naive_rebuilds, dim_k, dom_k, frc_k, mod_k, reg_idx, nl_counts_naive[key]
        )

    if verbose:
        print(f"Finished {len(selected)} case(s) in {time.perf_counter() - t_suite:.1f}s.")

    record: dict[str, Any] = {
        "v": 3,
        "commit": commit,
        "date": date,
        "dirty": dirty,
        "hw": hw,
        "profile": prof_name,
        "dtype": dtype,
        "n": run_n,
        "n_naive": run_n_naive,
        "n_naive_batch": run_n_naive_batch,
        "batch": run_batch,
        "steps": run_steps,
        "steps_naive": run_steps_naive,
        "alpha": alpha,
        "only_spheres": only_spheres,
        "jax": jax.__version__,
        "throttled": bool(throttle_warnings),
        "data": nested,
    }
    if nl_rebuilds:
        record["nl_rebuilds"] = nl_rebuilds
    if nl_naive_rebuilds:
        record["nl_naive_rebuilds"] = nl_naive_rebuilds
    return record
