# Validated performance evidence

This page records bounded end-to-end measurements for the uncommitted tree based
on commit `48e523b09a4f72aafed692871d074ec0aa9a74ef`. It is reproducibility evidence,
not a claim that the current kernels are optimal and not a comparison with an
older release. The historical `benchmarks/results.json` entries predate the
corrected topology fixtures and should not be used as current baselines.

The harness calls `System.initialize` explicitly. Each workload clears JAX's
compilation caches, records the first compiled call, replays the same initialized
snapshot, and then advances the returned state and system through successive
chunks. Device completion is inside each timed region. Finite-value and collider
overflow validation runs outside timing after each phase. The I/O case precompiles
the rollout and times a fresh rollout, eight VTK submissions, writer shutdown,
and durable file-count verification together.

## Environment and commands

- Python 3.14.7, JAX/JAXLIB 0.11.0.
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12,227 MiB, compute capability
  12.0; float32 (`jax_enable_x64=false`). GPU use was coordinated exclusively.
- CPU: Intel Core Ultra 9 285H, 16 logical CPUs; float64. Other repository test
  processes may have overlapped this run, so CPU values are explicitly marked
  contended and are unsuitable for CPU/GPU ratios.

```bash
PYTHONPATH=/tmp/jaxdem-test-tools JAX_PLATFORMS=cuda python -m benchmarks.release_workloads \
  --size 256 --steps 8 --repeat 3 --io --output benchmarks/results/release-20260911-gpu-float32.json
PYTHONPATH=/tmp/jaxdem-test-tools JAX_PLATFORMS=cuda python -m benchmarks.release_workloads \
  --workload spheres --workload clumps --workload mixed --size 4096 --steps 16 --repeat 5 \
  --output benchmarks/results/release-20260911-gpu-float32-n4096.json
```

The CPU command used the first command's workload parameters with
`JAX_PLATFORMS=cpu` and recorded the contention note in its JSON metadata.

## Results

Times are seconds per chunk. Standard deviations and all raw metadata remain in
the linked JSON files.

| Device / workload | N | steps | compile + first call | warm snapshot replay | sustained stepped state |
|---|---:|---:|---:|---:|---:|
| GPU spheres | 256 | 8 | 1.3610 | 0.00347 | 0.00339 |
| GPU clumps | 256 | 8 | 1.3332 | 0.00307 | 0.00394 |
| GPU deformable | 256 | 8 | 1.8416 | 0.00250 | 0.00379 |
| GPU mixed | 256 | 8 | 1.7446 | 0.00283 | 0.00737 |
| GPU Lees–Edwards spheres | 256 | 8 | 1.6977 | 0.00290 | 0.00304 |
| GPU spheres | 4096 | 16 | 1.6262 | 0.00728 | 0.00286 |
| GPU clumps | 4096 | 16 | 1.5808 | 0.00339 | 0.00310 |
| GPU mixed | 4096 | 16 | 2.1719 | 0.00474 | 0.00515 |
| CPU spheres (contended) | 256 | 8 | 0.7962 | 0.00856 | 0.00847 |
| CPU clumps (contended) | 256 | 8 | 1.0466 | 0.00895 | 0.00765 |
| CPU deformable (contended) | 256 | 8 | 1.3096 | 0.00931 | 0.01007 |
| CPU mixed (contended) | 256 | 8 | 1.2428 | 0.00937 | 0.00975 |
| CPU Lees–Edwards spheres (contended) | 256 | 8 | 1.2271 | 0.01134 | 0.01136 |

The GPU sphere rollout plus eight VTK frames took 2.3211 s. The contended CPU
equivalent took 1.7981 s. These include serialization and filesystem behavior and
must not be interpreted as isolated kernel throughput.

The deformable and mixed workloads attach an actual 2-D
`DeformableParticleModel` using their ring connectivity; adjacency alone is not
presented as deformable physics. The fixture contracts also cover an explicit
batch-size-one execution, while the recorded release timings use unbatched
snapshots.

This campaign deliberately does not replace the former 100,000/500,000-particle
kernel matrix. It covers corrected full-rollout workloads through N=4096 within a
bounded run. Large production baselines, an uncontended CPU campaign, memory
peaks, and comparisons against a tagged release remain open evidence work.

Raw results:

- `benchmarks/results/release-20260911-gpu-float32.json`
- `benchmarks/results/release-20260911-gpu-float32-n4096.json`
- `benchmarks/results/release-20260911-cpu-float64-contended.json`
