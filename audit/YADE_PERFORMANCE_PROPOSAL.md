# YADE-derived performance proposals for JaxDEM

## Scope and evidence

This review inspected `/home/wind/Documents/YADE2/bare` at commit
`76b306a73` and recent optimization commits from 2026-09-10. It proposes
experiments only; it does not assume YADE timings transfer to JaxDEM. YADE uses
C++/Kokkos, mutable interaction rows, explicit host/device mirrors, and host
worker pools. JaxDEM uses immutable-shape JAX pytrees and compiler-managed
device execution, so several successful YADE techniques map to measurement
questions rather than direct ports.

## 1. Compact active interaction work before expensive force evaluation

**YADE evidence.** Commit `eea1b6b36` replaced a full interaction download and
host scan with an execution-space parallel scan. The current implementation
selects detached row IDs into reusable execution-space scratch in
`core/DetachedForceKernels.hpp:25-52`, then evaluates only those rows at
`:55-120`. Cache validity is tied to topology and neighbor generation at
`:154-173`. The commit message and source establish the data-movement change;
they do not establish a transferable speedup for JAX.

**JaxDEM applicability.** `NeighborList.compute_force` currently evaluates a
fixed `(N, max_neighbors)` buffer with validity masks
(`jaxdem/colliders/neighbor_list.py:659-704`). This is efficient at high
occupancy and potentially wasteful for sparse or heavily padded lists. A
compacted pair representation could reduce force-law work, especially for laws
with expensive history updates.

**Risk.** Dynamic compaction can introduce sorting/scatter cost, dynamic-count
handling, nondeterministic reduction order, and extra compilation variants.
JAX's fixed-shape execution may make the masked buffer faster. History must
remain associated with canonical directed neighbor slots across rebuilds.

**Experiment.** Add a benchmark-only prototype that obtains `(i,j,slot)` via a
fixed-capacity `nonzero`, evaluates the existing pair law over the compacted
array, and segment-sums forces. Compare compile time, warm time, peak memory,
force/torque, energy, and next-step history against the existing path at 5%,
25%, 50%, and 90% neighbor-buffer occupancy on CPU and GPU. Reject it unless a
representative sparse case wins without changing overflow or history behavior.

## 2. Separate parallel per-contact evaluation from ordered publication

**YADE evidence.** Commit `87c6d1fc6` parallelized LevelSet volume-law
evaluation while retaining ordered force publication. Workers own disjoint
contact histories and write per-row outputs
(`pkg/levelSet/LevelSetIg2.cpp:1266-1324`); `publishContactOutputs` applies body
force and torque updates in row order at `:1248-1264`. This avoids concurrent
writes to shared bodies and preserves a stable publication order.

**JaxDEM applicability.** JaxDEM already expresses pair evaluation as `vmap`
and reduces contributions by particle. The relevant lesson is to keep a pure
pair-law stage distinct from state/history evolution, matching the open
architecture goal of separating model evaluation from evolution. This could
also let energy, minimization, and force evaluation share one compiled pair
primitive rather than repeat law algebra.

**Risk.** Materializing pair outputs increases memory, while fused JAX code may
already avoid it. Enforcing a particular ordered reduction could reduce GPU
throughput and does not guarantee bitwise reproducibility across backends.

**Experiment.** Use compiler HLO/memory analysis and the release workload
harness to compare the present fused neighbor kernel with a benchmark-only pure
pair-output plus `segment_sum` variant. Test sphere, friction-history, clump,
and deformable workloads; require force/energy agreement and exactly-once
history evolution. Measure sustained stepped state separately from compilation.

## 3. Give every reusable cache an explicit structural/content revision

**YADE evidence.** Commit `2fabf4541` added a shared content modification
counter to interaction-extension links
(`core/InteractionExtension.hpp:30-47,261-268`). Wire mirror reuse then checks
constant-time structural and content revisions instead of scanning every cached
record (`pkg/dem/WirePMKernels.cu` in that commit). The counter is incremented
through `markModified`, including retirement.

**JaxDEM applicability.** JaxDEM's neighbor cache has explicit geometry and
topology snapshots plus an invalidation flag
(`jaxdem/colliders/neighbor_list.py:61-145,345-360`). The YADE result supports
centralizing cache validity rather than adding more ad hoc comparisons as new
history, topology, or domain parameters appear. A revision token could be a
host-side orchestration concept for infrequent structural edits; dynamic
positions should remain array comparisons or displacement criteria.

**Risk.** Mutable revision counters conflict with JAX's functional pytree
contract. A forgotten increment creates silent stale physics, and a dynamic
revision stored as an array can trigger unnecessary recompilation or rebuilds.

**Experiment.** First instrument current rebuild reasons and their device cost
without changing behavior. Then prototype a returned immutable
`geometry_revision`/`topology_revision` value updated only by public structural
operations. Property-test every supported update against forced rebuilds and
checkpoint continuation. Adopt only if it replaces measured expensive validity
work and cannot be bypassed by ordinary array replacement.

## 4. Reuse expensive algebraic intermediates inside force laws

**YADE evidence.** Commit `862617a1c` replaced repeated exponential and friction
limit calculations in lubrication with named reused values; current examples
include cached `gap` values in `pkg/dem/Lubrication.cpp:517,566` and a cached
`logarithmicGap` at `:596`. Commits `785c3bb76` and `0eedda4e1` form the same
campaign. Commit `3d2ba2219` batches PotentialBlock constraints and reuses input
scratch, while `14fc0d9c1` reuses solver allocation with a reset initial basis
(`pkg/potential/PotentialBlockContact.cpp:100,243-298`).

**JaxDEM applicability.** Common-subexpression elimination may already remove
repeated JAX algebra within one fused trace. It may not cross separately traced
force and energy functions, and repeated geometry can remain when several laws
consume the same pair data. The useful target is a shared pure pair-geometry
record consumed by force, energy, and history functions.

**Risk.** Manual caching can enlarge live ranges and register pressure, making
GPU kernels slower. Persisting derived arrays across steps risks stale values
and increases checkpoint/schema burden. Solver scratch reuse has little direct
meaning under XLA buffer planning.

**Experiment.** Inspect lowered HLO for repeated norm, reciprocal, exponential,
and material-mixing operations in the dominant JaxDEM laws. Change only one
confirmed duplicate at a time, retain the value within a single pure call, and
measure kernel time plus device memory. For force/energy sharing, compare a
joint `force_energy` primitive against separate calls in minimization; verify
gradients, energies, and extreme-gap numerics before considering persistence.

## 5. Avoid host synchronization in device-resident preparation paths

**YADE evidence.** Commit `aee8bfca0` changed Grid read-only preparation to
preserve device residency. Commit `eea1b6b36` documents a stronger correctness
reason at `core/DetachedForceKernels.hpp:162-169`: downloading all interaction
rows could overwrite pending writes, so selection now occurs where authoritative
headers reside. Commits `774f6125b`, `0284f2426`, and `8b2361a65` further split,
skip, and reuse Grid preparation work.

**JaxDEM applicability.** JAX normally keeps traced simulation work on device,
but host checks, Python scalar conversions, writer dispatch, and benchmark
validation create synchronization boundaries. The highest-value audit target is
the full `System.step`/neighbor-rebuild path, especially any helper invoked
inside repeated Python stepping rather than a compiled rollout.

**Risk.** Removing checks from public boundaries can hide overflow or invalid
physics. Moving all validation onto device can add work to every step. Async
writers intentionally initiate device-to-host copies and should remain an
explicit I/O boundary.

**Experiment.** Capture a device trace for warm snapshot replay and sustained
stepped-state workloads, counting transfers and synchronization points. Compare
one Python-step loop with `System.step(n=...)` and `trajectory_rollout`. Move
only accidental transfers; retain `check_overflow`, finite checks, and writer
completion as labeled host boundaries. Verify identical failure propagation.

## 6. Resolve force-law ownership before evaluating heterogeneous pairs

**YADE evidence.** Commit `1bdce5dcc` resolves interaction-law ownership once
per step in `core/InteractionLoop.hpp:230-241` and dispatches the selected work
thereafter. The compact kernels likewise accept one canonical contact workset;
see `pkg/dem/ElasticContactLawKernels.hpp:56-100` and
`pkg/dem/CohesiveFrictionalContactLawKernels.hpp:69-99`.

**JaxDEM applicability.** `ForceRouter.force` evaluates registered law branches
and selects by species-pair masks (`jaxdem/forces/router.py:89-130`). With
several expensive laws, partitioning canonical pair indices by law could avoid
arithmetic for inactive combinations.

**Risk.** Dynamic buckets add compaction, scatter, and launch overhead and make
history remapping harder. Compiler branch lowering may already avoid some work.

**Experiment.** Compare the current router with fixed-capacity per-law index
buffers for one, two, and four species at balanced and 95/5 populations, using
N=4,096 and 65,536. Record HLO size, compilation, warm execution, memory,
forces, energies, gradients, and next-step histories.

## 7. Specialize incompatible geometry before expensive evaluation

**YADE evidence.** Commit `96f5c86b5` stopped reading contact geometry through
an incompatible layout while applying torque. Current detached application
checks force flags before typed `L3GeomData` or `ScGeomData` access in
`core/DetachedForceKernels.hpp:87-110`.

**JaxDEM applicability.** Sphere/facet selection in
`jaxdem/forces/spring.py:191-267` and facet/facet selection at `:357-461` uses
many `jnp.where` expressions. JAX evaluates both operands, so expensive or
numerically invalid inactive geometry can remain in the compiled computation.

**Risk.** A vmapped `lax.cond` can lower back to selection, while geometry
partitioning repeats compaction costs. Changing safe inactive operands can alter
gradients at type boundaries.

**Experiment.** Inspect HLO and profile sphere-only, sphere/facet, and mixed
facet workloads. Compare current safe operands with a type-homogeneous benchmark
kernel. Include degenerate facets and NaN-prone inactive geometry; require
identical valid forces, energies, and gradients before timing.

## Recommended order

1. Profile transfers and lowered HLO before changing runtime code.
2. Try local algebra reuse where HLO proves duplication.
3. Prototype compact-pair and force-routing evaluation only for sparse or
   heterogeneous occupancy.
4. Use the pure pair-output experiment to inform the broader evaluation versus
   evolution design.
5. Consider revision tokens only after instrumentation shows cache-validity
   checks are material and every mutation path has a testable owner.

Each experiment should use the versioned release harness, report compilation,
warm snapshot replay, and sustained stepped-state separately, validate finite
outputs and overflow outside timing, and retain a forced-reference physics
comparison. YADE's recorded measurements are evidence for its own layouts and
hardware only.

For cross-code correctness anchoring, YADE trunk commit `011308d01` adds a
normal spring law intended for matched benchmark physics. Before evaluating any
proposal above, compare a two-sphere force/energy case and a dense periodic grid
between that law and JaxDEM's spring law. Treat agreement as a physics gate,
never as evidence that either implementation's timing transfers to the other.
