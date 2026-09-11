# Boundary-search microbenchmark notes

**Historical measurements:** the first two tables predate the feedback-driven
API and geometry simplification. The dated CPU recheck at the end measures the
current working tree.

These numbers are bounded development measurements, not production performance
claims. They exercise a 64-particle, three-dimensional CellList force evaluation
with the grid rebuilt on every call. The periodic and Lees–Edwards cases use the
same state, material, cell parameters, and spring contact model. Lees–Edwards uses
`gamma=0.173`, flow axis 2, and gradient axis 0. `B=2` is a `jax.vmap` over two
identical snapshots. Compilation is the first blocking call; steady time is the
mean of 20 calls after compilation. Every result was finite and every collider
overflow flag was false.

| Platform | Geometry | Batch | Compile (s) | Steady call (ms) |
|---|---:|---:|---:|---:|
| CPU | periodic | 1 | 0.264 | 0.402 |
| CPU | periodic | 2 | 0.229 | 0.528 |
| CPU | Lees–Edwards | 1 | 0.291 | 0.639 |
| CPU | Lees–Edwards | 2 | 0.276 | 1.060 |
| RTX 5070 Ti Laptop GPU | periodic | 1 | 0.307 | 0.441 |
| RTX 5070 Ti Laptop GPU | periodic | 2 | 0.305 | 0.441 |
| RTX 5070 Ti Laptop GPU | Lees–Edwards | 1 | 0.355 | 0.538 |
| RTX 5070 Ti Laptop GPU | Lees–Edwards | 2 | 0.355 | 0.463 |

The Lees–Edwards path emits two candidate alpha cells for every stencil entry,
then deduplicates them. The extra work is visible most clearly in the CPU B=1
measurement. These tiny workloads are dominated by dispatch, compilation, and
machine state, so the ratios should not be extrapolated. The GPU result merely
confirms that the accelerated path compiles, runs, returns finite forces, and does
not report overflow on the available device.

## Extracted HEAD comparison

For a direct ordinary-periodic regression check, tracked `HEAD` was extracted with
`git archive` into `/tmp/jaxdem-boundary-baseline`; the repository and its index
were not changed. A self-contained script was run once from each source tree with
the same installed dependencies and platform selection. It used a 3D periodic
Spring/CellList grid at spacing 1.15, radius 0.5, and `dt=1e-4`. The force number
is the mean of ten already-compiled, blocking evaluations. The chunk number is
one already-compiled, blocking 20-step call after an initial force evaluation.
Both versions returned finite forces/positions and false overflow flags.

| Platform | N | Source | Force eval (ms) | 20-step chunk (ms) |
|---|---:|---|---:|---:|
| CPU | 512 | extracted HEAD | 2.012 | 33.368 |
| CPU | 512 | working tree | 2.100 | 36.086 |
| CPU | 4096 | extracted HEAD | 6.807 | 150.160 |
| CPU | 4096 | working tree | 7.117 | 173.714 |
| RTX 5070 Ti Laptop GPU | 512 | extracted HEAD | 0.353 | 3.639 |
| RTX 5070 Ti Laptop GPU | 512 | working tree | 0.442 | 4.747 |
| RTX 5070 Ti Laptop GPU | 4096 | extracted HEAD | 0.767 | 7.434 |
| RTX 5070 Ti Laptop GPU | 4096 | working tree | 0.657 | 8.828 |

The CPU force evaluations were about 4--5% slower in the working tree in this
run. The GPU force result varied with size: slower at N=512 and faster at N=4096.
The working-tree 20-step chunks were slower in all four comparisons. These are
single short observations without process-level repetitions, clock locking, or
thermal controls. They establish that the ordinary path remains functional and
quantify a possible regression worth profiling; they do not establish stable
throughput ratios or attribute the chunk difference to one change.

## Geometry validity regime

The displacement and image-stencil tests use cutoffs smaller than half every box
length. In that regime, any interacting pair has a unique nearest image in the
gradient direction; images with a different gradient index are already farther
than the cutoff. Lees–Edwards then applies that image's shear offset and selects
the periodic flow image. Searches at or beyond half a box length can have multiple
equally valid images and are outside the supported unique minimum-image regime.
The collider may deduplicate those images into one particle index, so such a query
must not be interpreted as enumerating image multiplicity.

## Follow-up optimization check

A conditional signed-coordinate fast path was tested in three additional warm
CPU runs. It was slower or noisier (N=512 force calls 2.14–2.23 ms versus
1.84–2.04 ms for the unsigned implementation) and was discarded. No speculative
optimization is retained. N=4096 chunk times varied substantially across runs
(153–175 ms for the unsigned implementation), reinforcing the need for controlled
profiling before interpreting small differences. Conservative shear traversal and
metric-change rebuilds remain explicit performance tradeoffs.

## 2026-09-11 bounded CPU recheck

The current working tree was compared with tracked commit
`bafaa753db41de738bdd7938e32958a0ac2798be`, extracted using `git archive` to
`/tmp/jaxdem-boundary-20260911.fogDqh`; neither the checkout nor the index was
changed. Both sources ran under Python 3.14.7, JAX 0.11.0, CPU backend, and
32-bit JAX arrays on an Intel Core Ultra 9 285H. The workload was a cubic 3D
periodic grid with spacing 1.15, radius 0.5, Spring force, CellList cell size
1.0, and `dt=1e-4`. N=512 and N=4096 correspond to 8-cubed and 16-cubed grids.

Each source/case compiled and blocked one force evaluation and one 20-step
chunk before timing. Each reported force sample is the blocked elapsed time per
call across ten force evaluations; each chunk sample is one blocked 20-step
call. The table reports the median of three samples. All timed outputs had
finite force/position arrays, and all available collider and sticky overflow
flags were false.

| N | Source | Force samples (ms) | Force median (ms) | 20-step samples (ms) | 20-step median (ms) |
|---:|---|---|---:|---|---:|
| 512 | extracted HEAD | 1.727, 1.613, 1.574 | 1.613 | 36.140, 34.097, 33.174 | 34.097 |
| 512 | working tree | 1.863, 1.815, 1.725 | 1.815 | 36.281, 35.949, 35.504 | 35.949 |
| 4096 | extracted HEAD | 6.971, 6.657, 6.234 | 6.657 | 156.961, 155.889, 165.155 | 156.961 |
| 4096 | working tree | 7.098, 6.925, 6.647 | 6.925 | 160.315, 161.506, 162.457 | 161.506 |

These short sequential runs were not pinned to dedicated cores and did not
control clocks or thermal state. The two source trees ran in separate sequential
processes after other numerical validation jobs completed, but process order was
not randomized and the extracted-HEAD N=4096 chunk still varied by about 9 ms.
The measurements confirm that both implementations complete this bounded
workload with finite, non-overflowing results. They do not support a stable
performance ratio or attribution of the observed timing differences. GPU timing
was skipped because the NVIDIA driver was unavailable to `nvidia-smi` from the
unprivileged benchmark session; separate device availability does not make these
CPU measurements a GPU comparison.
