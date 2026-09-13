# Pooled neighbor cache

`NeighborList` uses one fixed-capacity pool of `N * max_neighbors` directed pairs,
with `N + 1` CSR offsets. There is no layout option or per-particle neighbor limit.
An exactly full pool is valid; a pair count exceeding total capacity sets overflow.

Force and energy evaluation use row reduction exclusively. Read-only and stateless
force calls, and energy calls, share traversal in batches of 131,072 particles,
accumulating four neighbors at a time. Each batch computes its loop bound once;
pair evaluation reuses positions and source-particle calculations. Contact history
follows packed slots and is remapped by pair identity on rebuild. History updates
also use row reduction with a precomputed bound.

See [the row-only report](row-only/README.md) for the latest matched measurements
and validation. Forward- and reverse-mode differentiation remain supported through
custom derivative rules: ordinary evaluation traverses rows, while differentiation
evaluates the same pair laws over fixed cache slots.

The preserved RTX 5070 Ti Laptop GPU measurements at one million particles (SpringForce,
float32, JAX 0.11.0) were 4.35 ms/step for uniform particles and 5.09 ms/step for the
clustered case, with a capacity budget of 24 and a 100 MB index cache. Raw synchronized
samples are in `uniform.json` and `clustered.json`. These are historical timings from
before the row-only simplification. Timed
windows contain ten steps without rebuilding. Particle ordering, force law, and GPU
can change performance.

```bash
PYTHONPATH=. JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/pooled_neighbors.py \
  --n 1000000 --scenario clustered --capacity 24 \
  --output /tmp/pooled-neighbors.json
```

Add `--validate-only` to compare forces, torques, energy and ten-step trajectories against the
independent CellList implementation. `profile_pooled_neighbors.py` supports Nsight
Systems capture via `--cuda-capture`, HLO dumps via `--hlo`, and row batching experiments.
The retired comparison reports are archived at `/tmp/jaxdem-retired-neighbor-benchmarks`.

The pooled-only refactor passed 152 CPU/float64 regressions. The clustered million-particle
comparison against CellList passed; maximum force error was 2.05e-5 and ten-step
position error was 1.91e-6 (see `validation-million.json`).

All 81 GPU/float32 force, history, overflow, and serialization regressions passed.
Focused mypy and lint checks for the cache helper, tests, and benchmarks passed.
