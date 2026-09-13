# Row-only neighbor evaluation

The pooled neighbor cache now exposes no force-reduction or force-block-size
setting. Force and energy evaluation accumulate by CSR row. The implementation
shares row traversal, pair masking and positions, and precomputes each batch's
maximum degree instead of repeatedly reducing a vector loop condition. Read-only
history evaluation uses the same batches as stateless forces. History-writing
laws retain the faster flattened four-neighbor row-block evaluation.

Redundant rebuild wrappers and the alternate force reducer have been removed.
The two cache modules contain 51 fewer lines than the pre-simplification snapshot,
including the added derivative rules.
Custom derivative rules preserve forward- and reverse-mode gradients through
dynamic row traversal by differentiating the same pair laws over fixed cache
slots. This fixed-slot work occurs only during differentiation.

## Measurements

Measurements use one million particles, a capacity budget of 24, float32,
JAX 0.11.0, and an RTX 5070 Ti Laptop GPU. The baseline is the pooled cache with
row reduction selected, captured before this simplification at
`/tmp/jaxdem-before-row-only`; it is not a different storage layout. Each case
runs in a fresh process with compilation excluded, warm-up, and seven
synchronized samples. Step timings divide a ten-step window with no rebuilds
by ten. Both versions run the same current benchmark script.

The uniform case has 5,940,000 directed pairs and maximum degree 6. The clustered
case has 12,661,876 directed pairs and maximum degree 26: particles borrow from
the shared pool despite the capacity budget of 24. Both index caches occupy
100,000,004 bytes, excluding contact history and search workspaces.

Median SpringForce times in milliseconds:

| Operation | Uniform before | Uniform after | Clustered before | Clustered after |
| --- | ---: | ---: | ---: | ---: |
| Cached force | 3.136 | 2.957 | 3.682 | 3.639 |
| Cached energy | 4.739 | 2.659 | 4.910 | 2.922 |
| Rebuild and force | 25.188 | 25.845 | 45.598 | 44.599 |
| Per step | 4.215 | 4.075 | 4.959 | 4.660 |

Step time decreased by 3.3% (uniform) and 6.0% (clustered); energy time decreased
by 43.9% and 40.5%. Rebuild times differ by only a few percent in either direction,
so this change does not establish a rebuild improvement. These results apply to
the tested particle ordering, force law and GPU; they are not universal speedups.

Clustered Cundall–Strack measurements in milliseconds:

| Operation | Before | After | Warm-up |
| --- | ---: | ---: | ---: |
| Read-only force | 47.470 | 24.725 | 0.5 s |
| Force with history update | 54.953 | 54.090 | 2 s |

Read-only force time decreased by 47.9%. History-writing timings varied within
the initial short-warm-up runs (55.714 versus 67.884 ms); the longer-warm-up
comparison is essentially unchanged, so no history-update speedup is claimed.
Both sets of raw samples are retained. The profiling script now warms for two
seconds by default to better cover these more expensive calls.

Final measurements and GPU-process observations are saved in `clean/`. The monitor
recorded 118 observations across the eight main runs, with no competing compute
process detected.
Earlier timings were affected by another GPU compute process and must not be
used to claim speedups. Experimental outputs are archived under
`/tmp/jaxdem-row-experiments`; large HLO dumps are under
`/tmp/jaxdem-sparse-profiles`.

## Validation

The final GPU/float32 regression run passed 69 tests, covering pooled overflow,
borrowed capacity, force and torque, history updates and remapping, both gradient
modes, and serialization. The CPU/float64 selection passed 133 distinct tests
before the final history-layout adjustment; the GPU run includes that adjustment.
An empty-particle gradient test initially failed in its naive reference evaluator;
it passes with the analytical zero-energy reference.

Both million-particle cases pass comparisons against the independent CellList
implementation for forces, torques, energy, and ten-step positions and velocities.
Raw validation results are in `final-repeat/validation-*.json`.
Focused mypy checks passed for the four changed core modules; lint and formatting
checks passed for the cache helper, updated tests and benchmark scripts.

Reproduce the main measurements with:

```bash
PYTHONPATH=. JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/pooled_neighbors.py \
  --n 1000000 --capacity 24 --scenario clustered --output /tmp/rows.json
```

Use `--validate-only` for the independent comparison. Frictional force and history
measurements use `profile_pooled_neighbors.py --force-law cundallstrack` with
`--phase force` (read-only) or `--phase update`. The profiling script also supports
Nsight Systems capture and compiled HLO output.
