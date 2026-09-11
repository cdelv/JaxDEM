# Execution validation

## 2026-09-11 dynamics and state-cache check

These are bounded development measurements, not production throughput claims.
They used Python 3.14.7, JAX 0.11.0, the CPU backend, and 32-bit JAX arrays. The
workload was a 512-particle cubic periodic grid (spacing 1.15, radius 0.5) with
Spring contact forces, a CellList with cell size 1.0, and `dt=1e-4`.

The static and dynamic APIs each ran one compiling, blocking 20-step call, then
five warmed, blocking calls. They produced equivalent positions, finite values,
and false sticky overflow flags.

| Path | First blocking call (ms) | Warm samples (ms) | Warm median (ms) |
|---|---:|---|---:|
| `System.step(..., n=20)` | 1463.940 | 65.590, 64.510, 61.620, 59.390, 61.285 | 61.620 |
| `System.step_dynamic(..., n=jnp.asarray(20))` | 976.030 | 57.786, 63.658, 61.323, 56.060, 59.889 | 59.889 |

The first-call values include compilation and are not comparable compiler-cost
estimates: the second compilation can reuse previously compiled primitives. The
warm samples overlap and were not collected under pinned cores or controlled
clocks, so they show that the split APIs have the same practical scale in this
case rather than establishing a stable speed difference. The static API exists
to retain reverse-mode support through static loop bounds; the dynamic API
supports traced counts and carries JAX's reverse-mode limitation for dynamic
loop bounds.

The State cache check constructed a 512-particle state with nonzero body-frame
offsets, warmed `dataclasses.replace(state, vel=state.vel)`, and timed 100
individual replacements while blocking the derived rotated-offset result after
each call. Median latency was 68.745 microseconds (minimum 62.659, maximum
401.695). This measures the current deterministic cache rebuild, Python dispatch,
and synchronization together. There is no historical control in this run, so it
does not establish whether the immutable-input redesign is faster or slower.

Exact scripts and machine-readable output are retained in
`/tmp/jaxdem-dynamics-bench.py`, `/tmp/jaxdem-dynamics-bench-results.json`, and
`/tmp/jaxdem-cache-bench.py` for this workspace session.

## CPU default-suite status

A default CPU collection excluding the separately assigned PPO, action-space,
analysis, writer, and checkpoint suites was started with output captured in
`/tmp/jaxdem-release-core-tests.log`. It reached 29% with no failures before it
was stopped during the slow facet/collider parameter matrix so known thermal
validation defects could be fixed first. This partial run is not reported as a
full-suite pass. After those fixes, the focused dynamics and construction suites
passed 15 tests.
