# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Reproducible clump mass-property timings and analytical accuracy checks.

Run from the repository root, for example::

    JAX_PLATFORMS=cpu python -m benchmarks.clump_properties --dtype float64
    JAX_PLATFORMS=cpu python -m benchmarks.clump_properties --dtype float32

Each JSON record separates the first call, which includes compilation, from
warmed calls. Compare the same command, hardware, and sample budget across
revisions. The public-state workload includes host grouping and dispatch.

The overlapping disk, two-sphere, and nine-member star cases exercise
analytical paths. The three-sphere workload has a common triple overlap,
so it exercises boundary quadrature; the 65-sphere chain exercises the dense
volume path. Both check accuracy using independent cross-section integrals. For sampled 3D cases, ``--samples`` sets the total
budget, with at least 32 points per member. Boundary integration also rounds
up to an even count per member. Analytical cases do not depend on this budget.
For complex sampled shapes, compare error as well as elapsed time: surface
quadrature is not necessarily faster than volume sampling at the same point
count, and the budgets needed for comparable accuracy can differ.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import argparse
import json
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np

import jaxdem as jd
from jaxdem.utils.clumps import _compute_uniform_union_properties


def _measure(
    name: str,
    function: Callable[[], Any],
    repeat: int,
    *,
    expected_volume: float | None = None,
    expected_inertia: np.ndarray | None = None,
) -> None:
    start = time.perf_counter()
    result = jax.block_until_ready(function())
    first_call = time.perf_counter() - start
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = jax.block_until_ready(function())
        durations.append(time.perf_counter() - start)

    if isinstance(result, jd.State):
        volumes, inertias = np.asarray(result.volume), np.asarray(result.inertia)
    else:
        volumes, inertias = np.asarray(result[0]), np.asarray(result[2])
    if not np.all(np.isfinite(volumes)) or not np.all(np.isfinite(inertias)):
        raise ValueError(f"{name} returned non-finite mass properties")
    if not np.all(volumes > 0.0) or not np.all(inertias > 0.0):
        raise ValueError(f"{name} returned nonpositive volume or inertia")

    record = {
        "workload": name,
        "first_call_ms": 1000.0 * first_call,
        "warm_median_ms": 1000.0 * statistics.median(durations),
        "repeat": repeat,
    }
    if expected_volume is not None:
        record["volume_relative_error_max"] = float(
            np.max(np.abs(volumes / expected_volume - 1.0))
        )
    if expected_inertia is not None:
        record["inertia_relative_error_max"] = float(
            np.max(np.abs(inertias / expected_inertia - 1.0))
        )
    print(json.dumps(record), flush=True)


def _uniform_cases(samples: int, repeat: int) -> None:
    def run(
        name: str,
        positions: np.ndarray,
        radii: np.ndarray,
        expected_volume: float | None = None,
        expected_inertia: np.ndarray | None = None,
    ) -> None:
        pos = jnp.asarray(positions, dtype=float)
        rad = jnp.asarray(radii, dtype=float)
        mass = jnp.asarray(1.0, dtype=float)
        _measure(
            name,
            lambda: _compute_uniform_union_properties(
                pos, rad, mass, n_samples=samples
            ),
            repeat,
            expected_volume=expected_volume,
            expected_inertia=expected_inertia,
        )

    disk_volume = 4.0 * np.pi / 3.0 + np.sqrt(3.0) / 2.0
    run(
        "overlapping_disks",
        np.array([[-0.5, 0.0], [0.5, 0.0]]),
        np.ones(2),
        disk_volume,
        np.array([(np.pi + 3.0 * np.sqrt(3.0) / 4.0) / disk_volume]),
    )
    run(
        "overlapping_spheres",
        np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        np.ones(2),
        9.0 * np.pi / 4.0,
        np.array([17.0 / 40.0, 59.0 / 80.0, 59.0 / 80.0]),
    )
    for member_count, workload_name in (
        (3, "three_spheres_with_triple_overlap"),
        (65, "sixty_five_spheres_with_dense_overlap"),
    ):
        centers = np.linspace(-0.7, 0.7, member_count)
        # The nearest center determines the circular cross-section radius.
        # Integrating its polynomial area/second moments is independent of
        # the library's boundary quadrature and detects triple-overlap errors.
        bounds = np.r_[
            centers[0] - 1.0, 0.5 * (centers[:-1] + centers[1:]), centers[-1] + 1.0
        ]
        volume, axial_second, radial_second = 0.0, 0.0, 0.0
        x_squared = np.polynomial.Polynomial([0.0, 0.0, 1.0])
        for center, lower, upper in zip(centers, bounds[:-1], bounds[1:]):
            radius_squared = np.polynomial.Polynomial(
                [1.0 - center**2, 2.0 * center, -1.0]
            )

            def integral(polynomial: np.polynomial.Polynomial) -> float:
                antiderivative = polynomial.integ()
                return float(antiderivative(upper) - antiderivative(lower))

            volume += np.pi * integral(radius_squared)
            axial_second += np.pi * integral(x_squared * radius_squared)
            radial_second += 0.25 * np.pi * integral(radius_squared**2)
        run(
            workload_name,
            np.column_stack((centers, np.zeros((member_count, 2)))),
            np.ones(member_count),
            volume,
            np.array(
                [
                    2.0 * radial_second,
                    axial_second + radial_second,
                    axial_second + radial_second,
                ]
            )
            / volume,
        )
    angles = np.arange(8) * (2.0 * np.pi / 8.0)
    asperities = 0.8 * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(8)))
    positions = np.concatenate((asperities, np.zeros((1, 3))))
    run(
        "sixteen_nine_member_clumps",
        np.broadcast_to(positions, (16, 9, 3)),
        np.r_[np.full(8, 0.25), 0.65],
    )
    radius = 0.001
    run(
        "tiny_separated_spheres",
        np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        np.full(2, radius),
        8.0 * np.pi * radius**3 / 3.0,
        np.array([0.4 * radius**2, 3.0 + 0.4 * radius**2, 3.0 + 0.4 * radius**2]),
    )


def _public_state_case(samples: int, repeat: int) -> None:
    # Twenty-four dumbbells and one larger ring expose global-padding costs.
    positions = []
    ids = []
    radii = []
    for i in range(24):
        positions.extend([[4.0 * i - 0.4, 0.0], [4.0 * i + 0.4, 0.0]])
        ids.extend([i, i])
        radii.extend([0.5, 0.5])
    angles = np.arange(32) * (2.0 * np.pi / 32.0)
    positions.extend(np.column_stack((np.cos(angles), np.sin(angles))) + [0.0, 5.0])
    ids.extend([24] * 32)
    radii.extend([0.2] * 32)
    state = jd.State.create(
        pos=jnp.asarray(positions, dtype=float),
        rad=jnp.asarray(radii, dtype=float),
        clump_id=jnp.asarray(ids, dtype=int),
    )
    material = jd.Material.create("elastic", density=1.0, young=10.0, poisson=0.25)
    table = jd.MaterialTable.from_materials([material])
    _measure(
        "public_mixed_member_counts",
        lambda: jd.utils.compute_clump_properties(state, table, n_samples=samples),
        repeat,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()
    if args.samples <= 0 or args.repeat <= 0:
        parser.error("--samples and --repeat must be positive")
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    print(
        json.dumps(
            {
                "jax_version": jax.__version__,
                "platform": jax.devices()[0].platform,
                "dtype": args.dtype,
                "samples": args.samples,
            }
        ),
        flush=True,
    )
    _uniform_cases(args.samples, args.repeat)
    _public_state_case(args.samples, args.repeat)


if __name__ == "__main__":
    main()
