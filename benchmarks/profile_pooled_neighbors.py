#!/usr/bin/env python3
"""Capture compiled sparse workloads with Nsight Systems or inspect their HLO."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
from pooled_neighbors import make_case

import jaxdem as jd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--scenario", choices=("uniform", "clustered"), default="clustered")
    p.add_argument("--capacity", type=int, default=24)
    p.add_argument(
        "--collider",
        choices=("NeighborList", "CellList", "MultiCellList"),
        default="NeighborList",
    )
    p.add_argument("--force-law", choices=("spring", "cundallstrack"), default="spring")
    p.add_argument(
        "--phase",
        choices=("steps", "force", "update", "rebuild", "kernel", "energy", "sources"),
        default="steps",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cuda-capture", action="store_true")
    p.add_argument("--hlo", action="store_true", help="Also save the compiled HLO")
    p.add_argument("--row-width", type=int)
    p.add_argument("--row-batch", type=int)
    p.add_argument("--cell-batch", type=int)
    p.add_argument("--search-batch", type=int)
    a = p.parse_args()
    if a.cell_batch is not None:
        from jaxdem.colliders import cell_list, multi_cell_list

        cell_list.PAIR_TRAVERSAL_BATCH_SIZE = a.cell_batch
        multi_cell_list.PAIR_TRAVERSAL_BATCH_SIZE = a.cell_batch
    if a.search_batch is not None:
        from jaxdem.colliders import _neighbor_cache

        _neighbor_cache._SEARCH_BATCH_SIZE = a.search_batch
    if a.row_width is not None:
        from jaxdem.colliders import _neighbor_cache

        _neighbor_cache._ROW_FORCE_UNROLL = a.row_width
    if a.row_batch is not None:
        from jaxdem.colliders import _neighbor_cache

        _neighbor_cache._ROW_BATCH_SIZE = a.row_batch
    s, y = make_case(a.n, a.capacity, a.scenario)
    if a.collider != "NeighborList":
        y = replace(y, collider=jd.Collider.create(a.collider, state=s, cell_size=1.15))
    if a.force_law == "cundallstrack":
        material = jd.Material.create(
            "elasticfrict",
            density=1.0,
            young=100.0,
            poisson=0.0,
            e=0.8,
            mu=0.5,
            mu_r=0.0,
        )
        y = jd.System.create(
            state=s,
            collider=y.collider,
            domain=y.domain,
            dt=y.dt,
            mat_table=jd.MaterialTable.from_materials([material]),
            force_model=jd.forces.CundallStrackForce(),
        )
    s, y = jax.block_until_ready(jd.System.initialize(s, y))
    y.check_overflow()
    if a.phase == "steps":
        fn = lambda s, y: jd.System.step(s, y, n=20)
    elif a.phase == "rebuild":
        fn = lambda s, y: y.collider.evaluate_force(
            s, replace(y, collider=y.collider.invalidate())
        )
    elif a.phase == "force":
        fn = lambda s, y: y.collider.evaluate_force(s, y)
    elif a.phase == "update":
        fn = lambda s, y: y.collider.compute_force(s, y)
    elif a.phase == "energy":
        fn = lambda s, y: y.collider.compute_potential_energy(s, y)
    elif a.phase == "sources":
        from jaxdem.colliders._neighbor_cache import pair_sources

        fn = lambda s, y: pair_sources(y.collider)
    else:
        from jaxdem.colliders._neighbor_cache import forces

        fn = lambda s, y: forces(s, y, advance_history=False)
    executable = jax.jit(fn).lower(s, y).compile()
    start = time.perf_counter()
    while time.perf_counter() - start < 2.0:
        jax.block_until_ready(executable(s, y))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.hlo:
        a.output.with_suffix(".hlo").write_text(executable.as_text())
    mem = executable.memory_analysis()
    result = {
        "n": a.n,
        "scenario": a.scenario,
        "capacity": a.capacity,
        "collider": a.collider,
        "force_law": a.force_law,
        "phase": a.phase,
        "module": jd.__file__,
        "row_width_override": a.row_width,
        "row_batch_override": a.row_batch,
        "cell_batch_override": a.cell_batch,
        "search_batch_override": a.search_batch,
        "minimum_warmup_seconds": 2.0,
        "devices": [str(device) for device in jax.devices()],
        "memory": {
            k: int(getattr(mem, k))
            for k in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
            )
        },
    }
    if a.cuda_capture:
        lib = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")
        if lib.cudaProfilerStart() != 0:
            raise RuntimeError("cudaProfilerStart failed")
    timings = []
    try:
        for _ in range(7):
            start = time.perf_counter()
            jax.block_until_ready(executable(s, y))
            timings.append(time.perf_counter() - start)
    finally:
        if a.cuda_capture:
            lib.cudaProfilerStop()
    result["seconds"] = timings
    a.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
