# Final neighbor-cache review

The source-ID decoder now marks CSR row starts and propagates their maximum,
instead of expanding row lengths with `repeat`. Duplicate starts correctly skip
empty rows; slots past the occupied pool retain the invalid source sentinel.

At one million clustered particles with a capacity budget of 24, on the RTX 5070
Ti Laptop GPU (JAX 0.11.0, float32):

| Operation | Before | After |
| --- | ---: | ---: |
| Decode source IDs | 3.071 ms | 1.793 ms |
| Decoder temporary memory | 198.0 MB | 102.4 MB |
| Rebuild and evaluate frictional forces | 182.8 ms | 190.2 ms |

The decoder uses 42% less time and 95.6 MB less temporary memory. Full rebuild
samples varied from roughly 172–208 ms in both versions, so these measurements
do not establish an overall rebuild improvement. The full rebuild's peak
workspace is unchanged because other operations dominate it.

Each case runs in a fresh process with a two-second warm-up and seven synchronized
samples. No competing GPU compute process was present at case starts. The baseline
is `/tmp/jaxdem-before-final-review`. Reproduce with
`benchmarks/profile_pooled_neighbors.py --scenario clustered --force-law cundallstrack
--phase sources --output /tmp/sources`; use `--phase rebuild` for the full rebuild.
The separate synthetic decoder comparison in `source-decoding.json` also checks
all 24 million output slots for exact equality.

The review fixed empty-system construction and free-domain bounds, skipped an
unused minimum-radius calculation when cell size is supplied, and removed a
duplicate conversion. User-guide changes are limited to shared `max_neighbors`
capacity and overflow semantics.

Validation: 165 CPU/float64 tests passed; 88 distinct GPU/float32 cases passed.
The initial GPU run passed 87 cases and exposed float32 cancellation in one new
batched force comparison (a 0.00049 residual difference against opposing forces
of order 10,000). It passed with a force-specific absolute tolerance of 0.001;
position and history tolerances remain strict. Both logs are retained. Focused
mypy, lint, formatting and diff checks passed.

The 19 new CI cases cover batches of one and three simulations, selective
rebuilds, empty rows, history updates, nested `vmap` with forward/reverse gradients,
per-replica overflow and sticky failure flags, and empty simulations across all
three supported search backends with automatic, zero and positive capacities.
