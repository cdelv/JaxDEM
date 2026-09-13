# Spatial-search optimization

Force and energy traversal in CellList and MultiCellList now processes 16,384
particles per batch. This bounds temporary arrays and confines each dynamic loop
to the longest occupied cell in that batch. MultiCellList reuses particle positions
and search radii for its query boxes, and both colliders reuse the maximum search
radius. Direct force and energy evaluation now handles empty systems.

NeighborList rebuilds count neighboring cells in parallel within each particle
batch. Per-cell counts determine disjoint write intervals within each CSR row;
the fill pass writes directly into the shared pool. Allocation and overflow are
still global across all particle batches. The last partial batch is masked, and
relative offsets are clipped before adding row bases to avoid integer wrap.

## Million-particle results

RTX 5070 Ti Laptop GPU, JAX 0.11.0, float32, SpringForce, capacity budget 24.
The baseline is commit `1e6ceeb`, imported from
`/tmp/jaxdem-before-search-optimization`; final measurements use the working tree.
Each case runs in a fresh process with two seconds of warm-up and seven
synchronized samples. Compilation is excluded. Uniform and clustered inputs use
the same seeded particle configurations as the previous pooled-cache benchmarks.

Cell-list times include spatial search and force evaluation. NeighborList times
include a forced rebuild and force evaluation; they are not cached-step timings.

| Collider | Distribution | Before (ms) | After (ms) | Time reduction |
| --- | --- | ---: | ---: | ---: |
| CellList | uniform | 28.69 | 19.46 | 32.2% |
| CellList | clustered | 25.13 | 16.47 | 34.5% |
| MultiCellList | uniform | 26.64 | 20.89 | 21.6% |
| MultiCellList | clustered | 30.33 | 18.34 | 39.5% |
| NeighborList | uniform | 28.31 | 24.66 | 12.9% |
| NeighborList | clustered | 46.66 | 20.88 | 55.2% |

Compiled cell-list temporary workspace fell from 2.05–2.08 GB to 0.594 GB
(about 71% less). NeighborList rebuild workspace remains approximately 0.567 GB.
The persistent neighbor pool and its capacity semantics are unchanged.

Energy evaluation, including search:

| Collider | Distribution | Before (ms) | After (ms) | Time reduction |
| --- | --- | ---: | ---: | ---: |
| CellList | uniform | 18.87 | 12.48 | 33.8% |
| CellList | clustered | 19.33 | 11.04 | 42.9% |
| MultiCellList | uniform | 22.61 | 14.20 | 37.2% |
| MultiCellList | clustered | 23.35 | 12.86 | 44.9% |

The process monitor recorded 186 observations with no competing GPU
compute process detected. Raw samples and process observations are under `final/`.
Timings depend on particle ordering, distribution, force law and hardware.

## Validation and profiling

- 177 CPU/float64 regressions passed.
- 100 GPU/float32 regressions passed.
- Both million-particle cases pass force, torque, energy, and ten-step trajectory
  comparisons against CellList. Maximum force error is 2.05e-5 and maximum position
  error is 1.91e-6 across these comparisons.
- Tests cover partial traversal batches, batched simulations, periodic and
  Lees–Edwards boundaries, contact history, forward/reverse gradients, empty
  systems, exactly full pools, borrowing capacity across batches, and overflow.
- Focused mypy checks pass for all four changed collider modules. Changed helper,
  benchmark and test lint/format checks pass; the partition module retains its
  pre-existing quoted-annotation lint exceptions.

Nsight Systems identified repeated loop-condition synchronizations during rebuilds;
the original CUDA API summary is retained under `profiling/`. Experiments compared
whole-stencil traversal and several particle batch sizes. Whole-stencil traversal
increased memory use and regressed the uniform case; bounded traversal was retained.
Raw experiments and the Nsight capture are archived under
`/tmp/jaxdem-search-experiments` and `/tmp/jaxdem-search-profiles`.

Reproduce a force comparison from the repository root:

```bash
PYTHONPATH=. JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/profile_pooled_neighbors.py \
  --n 1000000 --collider CellList --scenario clustered --phase force \
  --output /tmp/cell-force
```

Use `--collider MultiCellList`, `--phase energy`, or
`--collider NeighborList --phase rebuild` for the other cases. Profiling-only
`--cell-batch` and `--search-batch` overrides support tuning experiments without
adding simulation configuration options.
