#!/usr/bin/env python3
"""Reproducible pooled neighbor-cache GPU benchmark; run each case in a fresh process.

PYTHONPATH=. JAX_PLATFORMS=cuda JAX_ENABLE_X64=0 python benchmarks/pooled_neighbors.py \
    --n 1000000 --capacity 32 --scenario uniform
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jaxdem as jd


def lattice(n, spacing):
    side = int(np.ceil(n ** (1 / 3)))
    ids = np.arange(n)
    return (
        np.stack(
            (ids % side, (ids // side) % side, ids // (side * side)), axis=1
        ).astype(np.float32)
        * spacing
    )


def make_case(n, capacity, scenario):
    if scenario == "uniform":
        pos = lattice(n, 0.9)
    else:
        cluster = lattice(n // 2, 0.65)
        dilute = lattice(n - n // 2, 2.0)
        dilute[:, 0] += cluster[:, 0].max() + 5
        pos = np.concatenate((cluster, dilute))
    rng = np.random.default_rng(42)
    pos += rng.uniform(-0.005, 0.005, pos.shape).astype(np.float32)
    state = jd.State.create(
        pos=jnp.asarray(pos),
        rad=jnp.full(n, 0.5),
        vel=jnp.asarray(rng.uniform(-0.01, 0.01, pos.shape), dtype=jnp.float32),
        mass=jnp.ones(n),
    )
    material = jd.Material.create("elastic", young=100.0, poisson=0.0, density=1.0)
    system = jd.System.create(
        state=state,
        dt=1e-4,
        mat_table=jd.MaterialTable.from_materials([material]),
        collider_type="NeighborList",
        collider_kw={
            "cutoff": 1.0,
            "skin": 0.15,
            "max_neighbors": capacity,
        },
        domain_type="free",
    )
    return state, system


def measure(fn, args, repeats):
    start = time.perf_counter()
    executable = jax.jit(fn).lower(*args).compile()
    compile_s = time.perf_counter() - start
    warmup_start = time.perf_counter()
    warmup_runs = 0
    while warmup_runs < 4 or time.perf_counter() - warmup_start < 0.3:
        out = jax.block_until_ready(executable(*args))
        warmup_runs += 1
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = jax.block_until_ready(executable(*args))
        timings.append(time.perf_counter() - start)
    mem = executable.memory_analysis()
    memory = {
        k: int(getattr(mem, k))
        for k in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
            "alias_size_in_bytes",
        )
    }
    return out, {
        "seconds": timings,
        "median_s": statistics.median(timings),
        "compile_s": compile_s,
        "memory": memory,
        "warmup_runs": warmup_runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument(
        "--scenario", choices=("uniform", "clustered"), default="uniform"
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("This benchmark requires a GPU")
    state, system = make_case(args.n, args.capacity, args.scenario)
    state, system = jax.block_until_ready(jd.System.initialize(state, system))
    system.check_overflow()
    print("initialized", flush=True)
    if args.validate_only:
        ds, dy = make_case(args.n, args.capacity, args.scenario)
        dy = replace(
            dy, collider=jd.Collider.create("CellList", state=ds, cell_size=1.15)
        )
        ds, dy = jax.block_until_ready(jd.System.initialize(ds, dy))
        dy.check_overflow()
        validation = {
            "n": args.n,
            "scenario": args.scenario,
        }
        for field in ("force", "torque"):
            a, b = getattr(state, field), getattr(ds, field)
            validation[field + "_max_abs_error"] = float(jnp.max(jnp.abs(a - b)))
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(b), rtol=2e-5, atol=2e-4
            )
        energy = system.collider.compute_potential_energy(state, system)[2]
        reference_energy = dy.collider.compute_potential_energy(ds, dy)[2]
        validation["energy_abs_error"] = float(jnp.abs(energy - reference_energy))
        np.testing.assert_allclose(
            np.asarray(energy), np.asarray(reference_energy), rtol=2e-5, atol=1e-3
        )
        for name, fn in (
            ("reference", lambda: jd.System.step(ds, dy, n=10)),
            ("pooled", lambda: jd.System.step(state, system, n=10)),
        ):
            st, sy = jax.block_until_ready(fn())
            sy.check_overflow()
            if name == "reference":
                reference = st
            else:
                for field in ("pos", "vel"):
                    a, b = getattr(st, field), getattr(reference, field)
                    validation[field + "_max_abs_error"] = float(
                        jnp.max(jnp.abs(a - b))
                    )
                    np.testing.assert_allclose(
                        np.asarray(a), np.asarray(b), rtol=2e-5, atol=2e-5
                    )
        validation["passed"] = True
        encoded = json.dumps(validation, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n")
        print(encoded, flush=True)
        return
    nl = system.collider.neighbor_list
    counts = np.diff(np.asarray(system.collider.row_offsets))
    result = {
        "n": args.n,
        "capacity": args.capacity,
        "warmup_runs": 4,
        "minimum_warmup_seconds": 0.3,
        "scenario": args.scenario,
        "jax": jax.__version__,
        "module": jd.__file__,
        "device": jax.devices()[0].device_kind,
        "x64": jax.config.jax_enable_x64,
        "pair_count": int(counts.sum()),
        "max_neighbors": int(counts.max()),
        "cache_bytes": sum(
            x.nbytes
            for x in (
                nl,
                system.collider.row_offsets,
                system.collider.history,
            )
        ),
    }
    cached, result["cached_force"] = measure(
        lambda s, y: y.collider.evaluate_force(s, y), (state, system), args.repeats
    )
    print("cached force", result["cached_force"]["median_s"], flush=True)
    _, result["cached_energy"] = measure(
        lambda s, y: y.collider.compute_potential_energy(s, y),
        (state, system),
        args.repeats,
    )
    _, result["rebuild_and_force"] = measure(
        lambda s, y: y.collider.evaluate_force(
            s, replace(y, collider=y.collider.invalidate())
        ),
        (state, system),
        args.repeats,
    )
    print("rebuild", result["rebuild_and_force"]["median_s"], flush=True)
    stepped, result["ten_steps"] = measure(
        lambda s, y: jd.System.step(s, y, n=10), (state, system), args.repeats
    )
    stepped[1].check_overflow()
    result["step_median_s"] = result["ten_steps"]["median_s"] / 10
    result["step_rebuilds"] = int(
        stepped[1].collider.n_build_times - system.collider.n_build_times
    )
    result["force_norm"] = float(jnp.linalg.norm(cached[0].force))
    result["finite"] = bool(jnp.all(jnp.isfinite(stepped[0].pos)))
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
