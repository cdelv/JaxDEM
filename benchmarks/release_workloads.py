# SPDX-License-Identifier: BSD-3-Clause
"""Bounded end-to-end release workloads with explicit timing phases."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jaxlib
import jax.numpy as jnp
import numpy as np

import jaxdem as jd
from jaxdem.writers.vtk_writer import VTKWriter
from benchmarks.base import (
    create_clumps_state,
    create_deformable_state,
    create_mixed_state,
    create_spheres_state,
)


def _block(tree: Any) -> None:
    jax.tree.map(
        lambda value: (
            value.block_until_ready() if hasattr(value, "block_until_ready") else value
        ),
        tree,
    )


def _validate(state: jd.State, system: jd.System) -> None:
    _block((state, system))
    system.check_overflow()
    leaves = [np.asarray(x) for x in jax.tree.leaves((state, system))]
    if any(
        np.issubdtype(x.dtype, np.number) and not np.all(np.isfinite(x)) for x in leaves
    ):
        raise ValueError("release workload produced non-finite output")


def _deformable_model(state: jd.State, start: int = 0) -> Any:
    """Build a physical 2-D ring model for fixture vertices from ``start``."""
    edges: list[tuple[int, int]] = []
    for row, neighbors in enumerate(np.asarray(state.bond_id)[start:], start=start):
        for neighbor in neighbors:
            if neighbor >= 0 and row < int(neighbor):
                edges.append((row, int(neighbor)))
    edge_array = jnp.asarray(edges, dtype=int)
    edge_lookup = {tuple(edge): i for i, edge in enumerate(edges)}
    adjacency: list[tuple[int, int]] = []
    for i, (_, end) in enumerate(edges):
        candidates = [
            idx for edge, idx in edge_lookup.items() if end in edge and idx != i
        ]
        if candidates:
            adjacency.append((i, candidates[0]))
    return jd.BondedForceModel.create(
        "deformable_particle_model",
        vertices=state.pos,
        elements=edge_array,
        edges=edge_array,
        element_adjacency=jnp.asarray(adjacency, dtype=int),
        em=1.0,
        eb=0.1,
        el=0.2,
        gamma=0.0,
    )


def build_workload(name: str, size: int, batched: bool) -> tuple[jd.State, jd.System]:
    domain_type = "periodic"
    domain_kw: dict[str, Any] = {"box_size": jnp.full((2,), 200.0)}
    bonded = None
    if name == "spheres":
        state = create_spheres_state(size, dim=2)
    elif name == "clumps":
        state = create_clumps_state(size, dim=2)
    elif name == "deformable":
        state = create_deformable_state(size, dim=2)
        bonded = _deformable_model(state)
    elif name == "mixed":
        state = create_mixed_state(size, dim=2)
        start = 2 * (size // 3)
        bonded = _deformable_model(state, start=start)
    elif name == "lees-edwards":
        state = create_spheres_state(size, dim=2)
        domain_type = "leesedwards"
        domain_kw = {
            "box_size": jnp.full((2,), 200.0),
            "gamma": -0.35,
            "alpha": 0,
            "beta": 1,
        }
    else:
        raise ValueError(f"unknown workload {name!r}")

    system = jd.System.create(
        state.shape,
        state=state,
        domain_type=domain_type,
        domain_kw=domain_kw,
        collider_type="celllist",
        bonded_force_model=bonded,
        interact_same_bond_id=False,
        dt=1e-4,
    )
    state, system = jd.System.initialize(state, system)
    if batched:
        state = jax.tree.map(
            lambda x: x[None] if isinstance(x, jax.Array) else x, state
        )
        system = jax.tree.map(
            lambda x: x[None] if isinstance(x, jax.Array) else x, system
        )
    _validate(state, system)
    return state, system


def measure(
    name: str, size: int, steps: int, repeat: int, batched: bool
) -> dict[str, Any]:
    state, system = build_workload(name, size, batched)

    def chunk(st: jd.State, sy: jd.System) -> tuple[jd.State, jd.System]:
        result = jd.System.step(st, sy, n=steps)
        _block(result)
        return result

    # Prevent an earlier structurally identical workload from supplying this
    # workload's executable during the compile/first-call phase.
    jax.clear_caches()  # type: ignore[no-untyped-call]
    start = time.perf_counter()
    compiled_state, compiled_system = chunk(state, system)
    compile_seconds = time.perf_counter() - start
    _validate(compiled_state, compiled_system)

    replay_times = []
    for _ in range(repeat):
        start = time.perf_counter()
        replay_state, replay_system = chunk(state, system)
        replay_times.append(time.perf_counter() - start)
    _validate(replay_state, replay_system)

    sustained_times = []
    current_state, current_system = state, system
    for _ in range(repeat):
        start = time.perf_counter()
        current_state, current_system = chunk(current_state, current_system)
        sustained_times.append(time.perf_counter() - start)
    _validate(current_state, current_system)

    return {
        "workload": name,
        "size": size,
        "batch": 1 if batched else None,
        "steps_per_chunk": steps,
        "repeat": repeat,
        "compile_first_call_seconds": compile_seconds,
        "warm_snapshot_replay_seconds": statistics.mean(replay_times),
        "warm_snapshot_replay_std_seconds": statistics.pstdev(replay_times),
        "sustained_stepped_state_seconds": statistics.mean(sustained_times),
        "sustained_stepped_state_std_seconds": statistics.pstdev(sustained_times),
        "validation": "outside timed regions after each phase",
    }


def measure_rollout_io(size: int, frames: int, directory: Path) -> dict[str, Any]:
    """Measure a complete sphere rollout followed by durable VTK completion."""
    state, system = build_workload("spheres", size, False)
    # Compile the rollout before the end-to-end measurement and validate it.
    final_state, final_system, trajectory = jd.System.trajectory_rollout(
        state, system, n=frames, stride=1
    )
    _validate(final_state, final_system)
    _block(trajectory)

    start = time.perf_counter()
    final_state, final_system, trajectory = jd.System.trajectory_rollout(
        state, system, n=frames, stride=1
    )
    trajectory_state, trajectory_system = trajectory
    with VTKWriter(
        directory=directory, clean=True, max_workers=1, writers=["spheres"]
    ) as writer:
        for frame in range(frames):
            frame_state = jax.tree.map(lambda x: x[frame], trajectory_state)
            frame_system = jax.tree.map(lambda x: x[frame], trajectory_system)
            writer.save(frame_state, frame_system)
    elapsed = time.perf_counter() - start
    _validate(final_state, final_system)
    files = list(directory.rglob("*.vtp"))
    if len(files) != frames:
        raise RuntimeError(f"VTK workload wrote {len(files)} frames, expected {frames}")
    return {
        "workload": "spheres-rollout-vtk",
        "size": size,
        "frames": frames,
        "rollout_and_io_seconds": elapsed,
        "files": len(files),
        "validation": "rollout precompiled; writer close and file count included in timing",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        action="append",
        choices=["spheres", "clumps", "deformable", "mixed", "lees-edwards"],
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--batch-one", action="store_true")
    parser.add_argument(
        "--io", action="store_true", help="also measure rollout plus VTK completion"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--note", default="", help="free-form run conditions recorded in metadata"
    )
    args = parser.parse_args()
    if args.size < 6 or args.steps < 1 or args.repeat < 1:
        parser.error("size >= 6, steps >= 1, and repeat >= 1 are required")
    workloads = args.workload or [
        "spheres",
        "clumps",
        "deformable",
        "mixed",
        "lees-edwards",
    ]
    output = (
        args.output
        or Path("benchmarks/results")
        / f"release-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    results = [
        measure(name, args.size, args.steps, args.repeat, args.batch_one)
        for name in workloads
    ]
    if args.io:
        results.append(
            measure_rollout_io(args.size, args.steps, output.with_suffix(".vtk"))
        )
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "worktree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "platform_version": jax.devices()[0].client.platform_version,
        "platform": jax.default_backend(),
        "x64_enabled": bool(jax.config.x64_enabled),
        "note": args.note,
        "results": results,
    }
    output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
