# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Device-synchronized throughput runner with high-performance throttle policy checks."""

from __future__ import annotations

import datetime
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


def _validate(output: Any) -> None:
    """Verify finiteness and zero collider overflow outside timed loops."""
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
        collider = getattr(item, "collider", None)
        if collider is not None and bool(np.any(np.asarray(collider.overflow))):
            raise RuntimeError("benchmark collider reported overflow")


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


def _time_built(built: tuple[Any, int, int], repeats: int) -> float:
    fn, total_particles, actual_steps = built
    out = fn()
    _block(out)
    _validate(out)

    samples_s: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        _block(out)
        samples_s.append(time.perf_counter() - t0)

    _validate(out)
    med_s = float(np.median(np.asarray(samples_s, dtype=np.float64)))
    return _round_sig((total_particles * actual_steps) / max(med_s, 1e-12) / 1e6)


def run_custom(
    *,
    collider: str = "cell_list",
    force: str = "spring",
    domain: str = "periodic",
    dim: str = "3d",
    mode: str = "single",
    phi: float = 0.60,
    vel: float = 0.5,
    n: int | None = None,
    n_naive: int | None = None,
    batch_size: int | None = None,
    steps: int | None = None,
    repeats: int | None = None,
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
    run_batch = batch_size if batch_size is not None else preset["batch"]
    run_steps = steps if steps is not None else preset["steps"]
    run_repeats = repeats if repeats is not None else preset["repeats"]

    built = build_spheres_in_box(
        collider=collider,
        force=force,
        domain=domain,
        dim=dim,
        mode=mode,
        phi=phi,
        vel=vel,
        n=run_n,
        n_naive=run_n_naive,
        batch_size=run_batch,
        steps=run_steps,
        nl_skin=nl_skin,
        nl_max_neighbors=nl_max_neighbors,
        alpha=alpha,
        only_spheres=only_spheres,
    )
    mps = _time_built(built, run_repeats)
    return {
        "dim": dim,
        "domain": domain,
        "force": force,
        "mode": mode,
        "collider": collider,
        "phi": phi,
        "vel": vel,
        "alpha": alpha,
        "total_particles": built[1],
        "steps": built[2],
        "mps": mps,
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
    batch_size: int | None = None,
    steps: int | None = None,
    repeats: int | None = None,
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
    run_batch = batch_size if batch_size is not None else preset["batch"]
    run_steps = steps if steps is not None else preset["steps"]
    run_repeats = repeats if repeats is not None else preset["repeats"]

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
            f"(N={run_n}, naive={run_n_naive}, B={run_batch}, steps={run_steps}, "
            f"alpha={alpha:g}, only_spheres={only_spheres}) | {dtype} | commit={commit}{'*' if dirty else ''}"
        )

    nested: dict[str, Any] = {}
    t_suite = time.perf_counter()
    for i, case in enumerate(selected, 1):
        if verbose:
            print(f"  [{i:3d}/{len(selected):3d}] {case.name:<58s} ... ", end="", flush=True)
        built = build_case(
            case,
            n=run_n,
            n_naive=run_n_naive,
            batch_size=run_batch,
            steps=run_steps,
            nl_skin=nl_skin,
            nl_max_neighbors=nl_max_neighbors,
            alpha=alpha,
            only_spheres=only_spheres,
        )
        mps = _time_built(built, run_repeats)
        _put_nested(nested, case, mps)
        if verbose:
            print(f"{mps:7.2f} M pstep/s")

    if verbose:
        print(f"Finished {len(selected)} case(s) in {time.perf_counter() - t_suite:.1f}s.")

    return {
        "v": 3,
        "commit": commit,
        "date": date,
        "dirty": dirty,
        "hw": hw,
        "profile": prof_name,
        "dtype": dtype,
        "n": run_n,
        "n_naive": run_n_naive,
        "batch": run_batch,
        "steps": run_steps,
        "alpha": alpha,
        "only_spheres": only_spheres,
        "jax": jax.__version__,
        "throttled": bool(throttle_warnings),
        "data": nested,
    }
