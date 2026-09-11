# JaxDEM 1.0 readiness audit

Audited on 2026-09-10 at commit `2598c85f09dbe2a91b8ff43238e85a900600e6eb`, using the working tree. The existing edits to `pyproject.toml` were included when checking typing configuration. Existing untracked simulations, images, and graph reports were excluded from the library review. The initial audit did not change the library; the follow-up below records the requested fixes.

**Recommendation: hold the stable release until the remaining unchecked items are addressed.** The original audit reproduced missed interactions, broken rigid-body invariants, and changed physics after checkpoint restoration. The requested plasticity and boundary/collider repairs now address many of those cases, as recorded below. The follow-up below implements the remaining numbered code fixes. Remote/full release validation and production benchmark regeneration remain open as items 34 and 37; architectural support-policy decisions are listed separately.

This is a broad source, numerical, packaging, documentation, and test audit of a 114-file, approximately 40,000-line library. It is not a proof of correctness or a claim that every line and parameter combination has been exercised. The reinforcement-learning environments, geometric construction algorithms, and analysis kernels received less numerical validation than the core solver. Selected long conservation cases and a minimum-dependency installation now pass. The full conservation matrix, multi-device execution, and TPU behavior remain release validation work.

**How to use this checklist**

- **P1:** fix before declaring the affected feature stable. A feature can instead be explicitly excluded from the 1.0 support contract and rejected when selected.
- **P2:** resolve before freezing its API, or document a deliberate limitation and add the corresponding validation.
- **P3:** documentation, maintainability, or evidence improvements.
- **Reproduced** means a small executable probe or existing test demonstrated the issue. **Source-confirmed** means the defect follows directly from the implementation. **Design/validation** items are concerns or missing evidence, not claimed experimental failures.

The `probes_*.py` files are historical reproductions for the audited revision; some use APIs superseded by these fixes. Run them against that revision, not as current regression tests. Captured outputs are in [evidence](evidence/). The assertion-based tests linked below validate the current implementation.

**Follow-up scope and deformable-model usage review — 2026-09-10**

The first implementation follow-up covered requested items **06, 08, 18, 25, 28, 29, and 30**, resolved in local commit `43f1efd`. The subsequent authorized fix also resolves **01**, as recorded below. Original severity labels are retained for traceability. Other findings remain open; reviewing an item's intended usage below does not imply that its implementation was changed.

The [deformable model guide](../examples/deformable_particle_guide.py), [DP construction pipeline](../examples/dp_construction_pipeline.py), and [general construction example](../examples/general_particle_construction.py) attach a bonded-force container to `System`, then use ordinary stepping. The general example documents selecting plastic variants through `dp_plasticity_type` and `dp_tau_s`; it does not prescribe a separate reference-update callback. Repeating this construction through `create_dp_container` on a deformed polygon confirms **01** for edge, perimeter, and bending plasticity: each direct functional update changes its reference, while all three simulation steps retain the original reference. See [the expanded probe](probes_plastic.py) and [follow-up output](evidence/plastic_followup.txt). This is a persistence defect in plastic variants, not evidence that the elastic model's reference state should evolve. That reproduction preceded the subsequent plasticity fix documented below; its captured output is historical evidence.

The examples also narrow the interpretation of **10–12**. DP mesh `elements`, `edges`, and adjacency belong to the bonded model; they are distinct from `State.add_connected_facet` contact geometry. Item **10** therefore does **not** establish broken shared-vertex connectivity in the intended DP workflow. The [clump construction pipeline](../examples/clump_construction_pipeline.py) supplies geometry-derived volume and inertia through `create_ga_state`, avoiding the low-level `add_clump` omissions in **11–12**. DP construction instead assigns each independent node a share of body mass and reference union volume; the volume sum is consistent with that reference-volume convention. Instantaneous deformed volume is a separate, explicitly unsupported quantity in `compute_particle_volume`. These findings should not be read as failures of every higher-level construction route.

**Follow-up validation**

These results record fix commit `43f1efd`. The two Langevin regression tests were subsequently removed at the user's request; batch-layout tests remain in `tests/test_step_layout.py`. Historical test outputs below are retained.

- Combined CPU regressions: **28 passed**, covering the new dynamics/writer tests, the complete PPO math suite (including bandit and corridor training), and public API tests. Seven existing Flax `.value` deprecation warnings remain outside this scope. See [output](evidence/followup-tests.txt).
- CUDA dynamics regressions: **5 passed**, including unbatched/B=1/B=2 stepping and Langevin coherence/covariance. GPU access required running outside the filesystem/device sandbox. See [output](evidence/followup-cuda-tests.txt).
- Mypy: **90 source files passed** under the existing configuration, which excludes/ignores RL. This is not a typing validation of the PPO changes. Diff whitespace checks passed.
- The plasticity follow-up reproduced item 01 through the example construction path. The full library suite, long conservation runs, and performance benchmarks were not rerun for these isolated fixes.

**Plasticity implementation and boundary/collider proposal — 2026-09-10**

Only issue **01** received additional library changes in this pass. Plastic reference evolution is now a timestep operation; direct force/energy evaluation and minimization hold that reference data fixed. Plastic relaxation during a quasistatic protocol must be an explicit physical update between minimizations.

Validation: **24 CPU tests passed** across `tests/test_plastic_reference_state.py`, `tests/test_step_layout.py`, and `tests/test_public_api.py` (17 focused tests plus 7 public API tests). This includes checkpoint continuation for edge, perimeter, and bending variants. The checkpoint run completed outside the sandbox after its sandboxed attempt stalled; checkpoint-library code was not changed. Mypy passed **90 source files** under the existing configuration, and diff whitespace checks passed. No new GPU or performance benchmark campaign was run for this fix. The seven warnings in the earlier PPO run and historical test counts above refer to that earlier follow-up.

The [boundary and collider proposal](BOUNDARIES_AND_COLLIDERS.md) gathers related findings and proposes shared domain geometry, conservative force-dependent search bounds, and atomic cache/history updates, plus targeted capacity, overflow, construction, and restart fixes. It includes the performance tradeoff for continuously changing shear and a staged validation plan. The boundary/collider implementation follow-up below supersedes the proposal status.

**Boundary/collider implementation — now committed in `48e523b`**

The accepted proposal is implemented with unsigned `uint64` hashes when JAX x64 is enabled and `uint32` otherwise. Particle indices remain signed. Changes cover free-domain bounds updates, force-dependent reach, fractional Lees–Edwards image searches, atomic neighbor-cache/history replacement, explicit query capacity, initial force preparation, contact-facet guards, checkpoint protocols, and packing callback dtypes. See the [implementation notes](BOUNDARIES_AND_COLLIDERS.md), [public contracts](../docs/source/user_guide/search_contracts.md), and [bounded performance measurements](BOUNDARY_PERFORMANCE.md).

This pass was subsequently committed with the approved review repairs in **`48e523b`**. The original audit captures historical failures; checked items below describe their repaired behavior. Broader release tasks such as the full supported-version/device matrix, strict documentation build, packaging, and production benchmark regeneration remain open. The new CPU regression workflow covers the boundary contract in both precision modes; it does not establish GPU release coverage.

**Boundary/collider validation record — before the feedback revision**

- CPU x64: **90 passed**, with three checkpoint cases run separately (**3 passed**). This covers the new contract suites, plasticity, batch layout, and public API. The final atomic-cache adjustment was additionally checked with **19 lifecycle tests passed**.
- CPU float32: **63 contract tests passed**, followed by **9 additional zero-radius/invalid-grid cases passed**. Hash boundary tests also spawn fresh processes and assert the actual x64 setting and dtype, avoiding inherited test configuration.
- Existing Hessian, exclusion, and clump-friction suites: **35 passed**. Checkpoint suite: **76 passed**. Construction, benchmark-fixture, and mesh suites: **16 passed** (overlaps the focused suites above; counts are not additive).
- CUDA: **38 passed** plus the packing callback test (**1 passed**) with `JAX_PLATFORMS=cuda,cpu`. The initial GPU-only configuration omitted the CPU callback device; enabling that required backend resolved the setup failure. Tests ran on an RTX 5070 Ti Laptop GPU.
- Mypy: **90 source files passed** under the existing configuration excluding RL. Black formatting and diff whitespace checks passed; Ruff passed with the existing late-import convention (`E402`) excluded.
- CPU/GPU measurements compare periodic Spring/CellList against extracted HEAD at N=512/4096, plus shear and batching microbenchmarks. All results were finite and non-overflowing. Some step-chunk measurements were slower; see [raw timings and limits](BOUNDARY_PERFORMANCE.md). No production throughput guarantee is claimed. A tested conditional fast path was slower and was discarded.

The full library test suite, strict documentation build, dependency-version matrix, long conservation runs, and production benchmark regeneration were not completed by this scoped follow-up. Existing expected rattler/H5 warnings and upstream deprecation warnings remain documented in their test outputs.

**Feedback revision — 2026-09-11, subsequently committed in `48e523b`**

- Collider cache ownership is declared by the `stateful` class property; pair-memory capability uses `supports_history`. The hardcoded collider-name set is gone.
- All force calls accept and return history arrays. Stateless history has trailing shape `(0,)`; `history_shape(dim)` declares each law's layout. Compositions concatenate histories and keep configurable law parameters as pytree data. Refresh validates shapes and preserves/slices history with neighbor indices.
- Removed newly introduced NumPy dependencies from domain, collider, force, and minimizer code. Hash bounds use native Python bit limits and JAX arrays; separate cast/product guards prevent distinct overflow failures.
- Removed the general `prepare_search` API and duplicate public wrapping operation. Only FreeDomain updates query bounds. Existing `shift` and its private point helper share image rules. Independent Lees–Edwards image tests exposed and fixed the stencil-offset sign; free-space padding is reduced before constructing domain fields.
- LJ's `cutoff_ratio` controls physical force, shifted energy, and search reach together (default 2.5). Larger cutoffs work through all four colliders, combined/routed laws, JIT batching, and checkpoints. [Cutoff regressions](../tests/test_force_cutoffs.py).
- Preserved COM-to-contact rotational velocity: `_pos_p_rot + r_ci` adds the COM-to-sphere-center offset to the sphere-center-to-contact arm.
- Removed `consume_external`/`include_external`. Initial preparation uses a temporary zero-buffer manager and restores the queued loads.
- Restored permissive `add_clump` defaults and removed mandatory/finite/identical mass-property guards. Items 11–12 now reflect the intended construct-then-compute workflow. Removed the example/test values added solely to satisfy those guards.
- Orbax rejects zero-length array payloads. Checkpoint I/O omits those payloads and reconstructs empty arrays from the target; runtime history remains uniformly an array. Full force-law configuration is restored directly.

Validation of this revision (groups overlap; do not add the counts):

- CPU: **40 passed** across force-search, minimizer, construction, clump-friction, and step-layout suites; **19 cutoff tests passed**; **23 lifecycle tests passed**; **21 geometry tests passed**.
- CPU float32: **62 force/cutoff/history tests passed** and **21 geometry tests passed**. The geometry fixture no longer imports a suite that forces x64 on; the float32 run explicitly verified x64 remained disabled after import. Hash boundary tests also assert both precision modes in fresh processes.
- GPU: **63 cutoff/history/geometry tests passed**, with default backend independently verified as `gpu` / `CudaDevice(0)`.
- Checkpoints: **79 passed** across the existing suite and three configurable-LJ restart cases, plus **1 nonempty composite-history restart test passed**. The latter verifies exact stored history and identical next-step continuation.
- Mypy with the restored current tooling reports **two existing argument-type errors** at `utils/geometric_asperity_creation.py:612` (NumPy arrays passed to `angle_between_normals`). That file is unchanged by this work. The new code's typing issues are resolved; this is not a clean whole-library type check.
- Ruff (existing late-import convention excluded), Black, and whitespace checks pass. Bounded measurements are recorded separately in [the performance report](BOUNDARY_PERFORMANCE.md); full release benchmarking remains open.

**Domain and force capability follow-up — subsequently committed in `48e523b`**

- Grid colliders now call `system.domain.update_bounds(...)`. The base operation is a no-op, and FreeDomain overrides it; collider code no longer identifies dynamic domains itself. The unused `del system` statement in `_shift` is removed.
- `metric_snapshot` remains a neighbor-cache validity check for periodic box lengths and shear. Geometry changes can alter neighbor membership without particle displacement; free-space bounds do not invalidate this snapshot.
- Explicit NeighborList capacities are preserved exactly and bypass all estimator computations. Only omitted capacities are estimated/clamped. Regressions cover widths 0, 1, and 7 independently of density/safety estimates.
- Analytical minimization uses `ForceModel.supports_analytical_energy_gradient`, with default true and explicit false overrides for incompatible laws. Compositions aggregate the properties; an opted-out law requires a target function. The hardcoded supported-law list is removed.
- Collider attribute suppressions are removed through concrete type narrowing and accurate constructor annotations. A narrow JAX `finfo` suppression remains because the external callable lacks typing; existing external Optax/Orbax import suppressions remain.
- The automatic force-preparation lifecycle from this pass was subsequently replaced by the explicit initializer described below.

Validation: geometry **21 passed** in each precision mode; neighbor lifecycle **26 passed** (checkpoint deselected); force-search and batch-layout **25 passed**; minimizer capability/gradient tests **9 passed**. The new initialization tests cover zero-step behavior and explicit force refresh after moving particles. Current mypy still reports only the two unchanged geometry-utility argument errors noted above.

**Explicit initialization lifecycle — subsequently committed in `48e523b`**

The accepted API is now `state, system = System.initialize(state, system)`, called
by the user before starting dynamics. It updates bounds, evaluates initial pair
and managed forces, then invokes both configured integrators' existing
`initialize` hooks. The hooks can implement velocity staggering; no special
half-step correction is needed by the current built-in integrators.

Removed the prior preparation method, automatic conditional in `step`, stored
initialization flag, and temporary zero-load manager used for startup. Normal
managed-load application consumes queued buffers. Initialization does not advance
time, timestep callbacks, or contact/plastic history. An already initialized
checkpoint continues directly; repeating initialization could repeat an
integrator's velocity adjustment. Minimization and post-reindex force evaluation
remain separate from integrator startup.

Examples and user documentation now demonstrate explicit initialization. Earlier
automatic-initialization descriptions and validation above are historical and
superseded by this API.

Validation for explicit initialization:

- CPU force/batch/plasticity suites: **36 passed**, with checkpoint cases run separately.
- Initializer and neighbor lifecycle suites: **30 passed**; an additional regression that `step` never secretly recomputes initial forces also passed.
- Float32 initializer/force/batch suites: **29 passed**. GPU initializer/batch suites: **7 passed**.
- Checkpoint suites, including plastic and pair-history continuation: **80 passed**.
- Minimizer suite: **10 passed**, including a hook that raises if initialization is invoked and preservation of nonzero velocities. Related Hessian/contact/rattler checks also passed.
- Mypy remains limited by the same two pre-existing geometry-utility argument errors; formatting, Ruff (existing late-import convention excluded), and whitespace checks pass.


**Completed-fix review — 2026-09-11**

The [completed-fix review](COMPLETED_FIXES_REVIEW.md) reassesses all 26 previously checked items. **22 retain completed status; 05, 16, 17, and 30 are reopened** with numerical counterexamples and minimal proposed repairs. Item 23 also needs its obsolete method docstring corrected. During this review, the explicitly requested analytical-gradient capability was changed to default true, with only false overrides required and composite aggregation retained. No other runtime repairs or commits were made in this review. Earlier validation records describe their dated snapshots, not certification of these newly exposed cases.

**Repairs from the completed-fix review — subsequently committed in `48e523b`**

Items **05, 16, 17, and 30** are repaired and checked again. Their counterexamples are retained as regression tests; the [review report](COMPLETED_FIXES_REVIEW.md) preserves the original evidence and records the repair validation. The stale NeighborList query and LawCombiner tuple docstrings are corrected. No unrelated open audit items were changed. Validation: **23 geometry, 36 lifecycle (including checkpoint), 12 PPO, and 76 existing checkpoint tests passed**. Additional float32 CPU and CUDA counterexample checks passed; [full validation details](COMPLETED_FIXES_REVIEW.md) include precision runs and existing typing/warning limitations.

**Remaining numbered items — 2026-09-11, uncommitted**

The user-approved prior repairs are committed as **`48e523b`**. The subsequent work addresses the 19 previously open numbered items. **43 of the 45 numbered entries are now checked; 34 and 37 remain open for release evidence.** There is no numbered item 00 in the original audit. No further commit or push has been made.

See [implementation and validation details](REMAINING_ITEMS_IMPLEMENTATION.md), [execution measurements](EXECUTION_VALIDATION.md), [validated rollout/I/O benchmarks](PERFORMANCE_RELEASE.md), and [public compatibility contracts](../docs/source/user_guide/release_contracts.md). Earlier validation records below are historical snapshots; this follow-up supersedes their outstanding implementation and typing notes. Local validation does not establish full release readiness.

**Installation and contact-history feedback — uncommitted**

JaxDEM now exposes feature extras only (`io`, `rl`, `docs`, `test`, and `all`). Hardware extras belong to JAX, with commands such as `pip install '.[all]' 'jax[cuda13]'`. Metrics use TensorBoardX without importing TensorFlow. Cundall–Strack now uses actual pair history and its built-in RL callers use history-capable neighbor lists. Quaternion freezing and custom bijector comparisons are removed. See [the feedback implementation report](INSTALLATION_AND_CONTACT_HISTORY.md) for behavior, validation, and limitations.

The separate [YADE performance proposal](YADE_PERFORMANCE_PROPOSAL.md) reviews `/home/wind/Documents/YADE2/bare`. Its optimization ideas are proposals only; none have been ported.

**Architecture and final review — uncommitted**

Architectural items A–G are implemented. The [final review](G_AND_FINAL_REVIEW.md)
records the RL support contract, removal of the duplicate wrapper package,
additional correctness fixes and simplifications, and local validation.
Items 34 and 37 remain deferred; the local checks do not certify a full release.

**Correctness and persistence blockers**

- [x] **01 · P1 · Plastic reference-state updates do not survive a simulation step. Resolved.** [BondedForceModel](../jaxdem/bonded_forces/__init__.py) now provides the pure `update_reference_state` hook, with an elastic no-op default. [System stepping](../jaxdem/system.py) calls it exactly once after drift and before force evaluation, and carries the returned model forward explicitly. [Deformable force evaluation](../jaxdem/bonded_forces/deformable_particle.py) no longer advances plasticity. The existing relaxation equations are unchanged. [Regression tests](../tests/test_plastic_reference_state.py) cover edge, perimeter, and bending persistence; fixed-edge references **1.1, 1.19, 1.271**; analytical force using frozen reference data; repeated force/energy evaluation; batching; split stepping; and Orbax checkpoint continuation for all three variants. This resolves plastic-state persistence, not the separate callback/static-configuration restoration defects in 13–14.

- [x] **02 · P1 · Accelerated Lees–Edwards search. Resolved.** The shared grid stencil follows shear images, includes both intersected flow cells for fractional offsets, and deduplicates repeated cells. CellList, MultiCellList, and NeighborList support the same image rules as pair forces. MultiCellList disables only its shear bounding-box pruning conservatively; the grid search remains accelerated. [Tests](../tests/test_boundary_search_geometry.py) include independent physical-image enumeration, arbitrary axes, unwrapped positions, and small boxes; [force/energy tests](../tests/test_force_search_contracts.py) cover 2D and 3D. Unique-image interaction reach must stay below half each periodic box length.

- [x] **03 · P1 · Shared interaction-reach contract. Resolved.** [ForceModel.search_radii](../jaxdem/forces/__init__.py) defines conservative per-primitive bounds. Builtins, compositions, current flexible-facet extents, CellList/MultiCellList, force NeighborList, and default Hessian cutoffs use it. Custom accelerated laws must declare a bound; naive all-pairs forces do not require one. Explicitly smaller Hessian cutoffs are documented approximations. [Regressions](../tests/test_force_search_contracts.py) cover LJ/WCA beyond contact range and facets deformed after construction.

- [x] **04 · P1 · Free-domain search preparation. Resolved.** Force, energy, and explicit queries derive free-space bounds from the actual points being searched through a pure domain hook. Minimizer trials use current geometry without invoking physical reflection. [Public contract](../docs/source/user_guide/search_contracts.md).

- [x] **05 · P1 · Neighbor-cache invalidation. Resolved after review.** Cache validity includes periodic box lengths, shear magnitude and canonical axes, physical interaction reach, and build-time skin. `invalidate_collider` handles unchanged-index topology edits; `refresh_collider` reconstructs allocations. [Counterexample regressions](../tests/test_boundary_search_geometry.py) verify that increasing skin or replacing shear axes rebuilds the list and matches Naive forces. Arbitrary nested edits still require explicit invalidation; continuous shear may rebuild every step.

- [x] **06 · P1 · Langevin noise tears rigid clumps apart. Resolved.** [Langevin](../jaxdem/integrators/langevin.py) now gathers translational noise by `clump_id`, sharing each body sample among its members while retaining the replicated body mass in the covariance. Independent spheres and prescribed fixed velocities are preserved. The original regressions verified common velocities/centers, independent body samples, fixed velocity, and the equilibrium OU variance `k_B*T/body_mass` on CPU and CUDA. Those two tests were subsequently removed at the user's request; the implementation fix remains.

- [x] **07 · P1 · Clump contact-point velocity. Resolved.** Cundall–Strack now includes the rotated member offset in the COM-to-contact arm. Constitutive forces and torque aggregation are otherwise unchanged. An [analytical regression](../tests/test_force_search_contracts.py) compares a rotating/translating member with the equivalent stationary contact.

- [x] **08 · P1 · Batch size one fails in `System.step`. Resolved.** [System.step](../jaxdem/system.py) now dispatches by layout: `(N,dim)` is unbatched and `(B,N,dim)` is mapped even for `B=1`. Unsupported state ranks and mismatched state/system ranks raise `ValueError` during tracing; no repeated host synchronization is added. [Regressions](../tests/test_step_layout.py) compare every dynamic state/system leaf after three moving-particle Verlet steps for unbatched, B=1, and B=2 runs, and exercise mismatched layouts. CPU and CUDA checks pass.

- [x] **09 · P1 · Explicit integrator initialization. Resolved.** The user calls `System.initialize(state, system)` before stepping a new trajectory. It computes forces with frozen physical history, applies managed loads normally, then invokes the linear and rotational integrator initialization hooks. `step` performs no automatic initialization and stores no initialization flag. [Initialization regressions](../tests/test_initialization.py) cover hook ordering, velocity staggering, batching, and the absence of automatic startup; [force tests](../tests/test_force_search_contracts.py) cover initial Verlet acceleration and load-buffer semantics.

- [x] **10 · P1 · Shared contact-facet incidence. Resolved by guard.** `State.add_connected_facet` rejects vertices already assigned to another contact facet, with a descriptive error. Shared vertices in deformable bonded meshes remain supported. [Construction regressions](../tests/test_construction_contracts.py).

- [x] **11 · P1 · Clump volume convention. Reviewed as intended usage.** Low-level `add_clump` accepts placeholder defaults so callers can subsequently use `compute_clump_properties`. No new validity guard is imposed. Geometry-aware facet/mesh builders now replicate total body volume consistently with `compute_particle_volume`. [Construction regressions](../tests/test_construction_contracts.py).

- [x] **12 · P1 · Clump inertia convention. Reviewed as intended usage.** Default member inertia is a construction placeholder, deliberately permitted. Callers remain responsible for replacing it with body inertia, for example through `compute_clump_properties`; new mandatory/finite/identical-property guards have been removed.

- [x] **13 · P1 · Checkpoint boundary protocols. Resolved.** Version-2 metadata persists importable pre/post-step callbacks. Saving validates callable identity through import-path round trips; strict restoration rejects missing required physics. [Checkpoint tests](../tests/test_checkpoints.py) verify continuation with non-default shear axes and both callbacks.

- [x] **14 · P1 · Checkpoint static component configuration. Resolved.** Domain and integrator construction metadata is restored alongside existing collider/force configuration and array leaves. Canonical Lees–Edwards axes survive restoration. Version-2 reconstruction cannot silently fall back to dropping this configuration. [Checkpoint tests](../tests/test_checkpoints.py).

- [x] **15 · P1 · Atomic neighbor-list/history replacement. Resolved.** `_check_and_rebuild` returns the updated collider, including remapped pair history. Energy and force evaluations retain this result together. Energy evaluation remaps the cache without advancing physical history. [Lifecycle regressions](../tests/test_neighbor_list_lifecycle.py).

- [x] **16 · P1 · Collider refresh and history allocation. Resolved after review.** `refresh_collider` preserves unchanged-index pair memory and initializes new capacity with the force model. Particle-count changes with nonempty history and shrinking history capacity require `reset_history=True`; reindexing remains an explicit caller responsibility. [Lifecycle regressions](../tests/test_neighbor_list_lifecycle.py) verify rejection of implicit resets, explicit reset values, and capacity growth.

- [x] **17 · P1 · Uniform history allocation. Resolved after review.** Force laws use uniform history arrays and `history_shape(dim)`. Combiners and routers concatenate child initializers as well as shapes. Rebuilds initialize new pairs through the law and copy surviving pair memory into that buffer. [Nonzero-initializer regressions](../tests/test_neighbor_list_lifecycle.py) cover direct/composite laws, new neighbors, capacity growth, and retained history.

- [x] **18 · P1 for RL · PPO depends on a model-private attribute outside its interface. Resolved.** Removed the mandatory `model._log_std` diagnostic from [PPOTrainer](../jaxdem/rl/trainers/ppo_trainer.py), including the generic `log_std` metric. Training no longer requires an undocumented model field or fabricates a substitute statistic. The existing custom-model bandit and corridor training tests in [test_ppo_math.py](../tests/test_ppo_math.py) both pass. Consumers of the removed diagnostic must obtain model-specific statistics separately.

**Numerical, API, and operational contracts**

- [x] **19 · P2 · Static and dynamic stepping. Resolved.** `System.step` keeps a Python integer count static; `System.step_dynamic` accepts traced scalar counts. Constant-stride rollouts retain reverse-mode support, while variable-stride loops use the explicit dynamic path. Invalid fractional/rank/negative host strides are rejected. [Derivative/layout regressions](../tests/test_dynamics_state_contracts.py) and [bounded compilation/warm timing evidence](EXECUTION_VALIDATION.md) define the tested scope; arbitrary collider differentiation is not promised.

- [x] **20 · P2 · State cache mutation contract. Resolved.** Quaternions remain mutable as requested. Replacing `q` or `pos_p` refreshes the rotated-offset cache, including `dataclasses.replace`; nested component edits require `State.refresh_rotation_cache()`, with missed refreshes detected by optional host validation. Python stack inspection is removed. Force search bounds take the maximum of current physical reach and stored search radii; current geometry drives search invalidation, and stepping refreshes inverse box lengths after domain application. Opt-in validation detects inconsistent stored geometry. [State tests](../tests/test_dynamics_state_contracts.py) and [replacement measurements](EXECUTION_VALIDATION.md) cover the new contract; direct private-cache edits remain unsupported.

- [x] **21 · P2 · Thermostat stage and body-weighted drift. Resolved.** Rescaling runs through `finalize_step` after both final kicks. Drift uses each free body mass once, independent of clump membership count. Temperature excludes prescribed fixed motion and rescaling preserves it. Zero kinetic energy remains zero under multiplicative scaling. [Counterexamples](../tests/test_dynamics_state_contracts.py) cover terminal torque kicks, unequal masses/member counts, fixed particles, and small total mass.

- [x] **22 · P2 · Conservative minimization and termination status. Resolved.** Objective evaluation freezes physical history, masks fixed DOFs, and uses conservative energy gradients. Incompatible driven laws require an explicit objective. `MinimizationResult.reason` distinguishes convergence, exhausted budget, nonfinite objective, and search overflow; legacy four-value unpacking/indexing remains supported. [Tests](../tests/test_minimizer_contracts.py) include nonfinite and sticky-overflow termination.

- [x] **23 · P2 · Explicit neighbor-query semantics. Resolved.** Arbitrary NeighborList queries delegate to the secondary collider, honor cutoff and requested width, and preserve the physical force cache/history. Explicit capacity is not packing-clamped; zero width truthfully reports overflow. `System.search_overflow` records force-search failure until the caller retries from a valid snapshot. [Query/lifecycle tests](../tests/test_neighbor_list_lifecycle.py).

- [x] **24 · P2 · Opt-in representation validation. Resolved.** `State.validate` and `System.validate` check shapes, finite physical data, radii, IDs/material/router bounds, normalized orientations, positive dynamic mass/inertia, replicated clump fields, timestep and domain geometry. Public State input guards use `ValueError`, including under Python `-O`. Validation stays outside compiled loops and remains optional so low-level clump placeholders are permitted. [Invalid-input regressions](../tests/test_dynamics_state_contracts.py).

- [x] **25 · P2 · Asynchronous writer failures are not observable through completion. Resolved.** [BaseAsyncWriter](../jaxdem/writers/async_base.py) retains worker failures and raises `AsyncWriterError`, including task names and original exceptions, after `block_until_ready` drains the queue or `close` joins the workers. Reported failures are consumed once, so cleanup after catching a completion error is safe. [VTKWriter](../jaxdem/writers/vtk_writer.py) propagates file errors with writer/path context and rejects an active writer that produces no expected file. [Five regressions](../tests/test_async_writer.py) cover aggregation, draining, shutdown, failed/missing VTK output, and successful public save/manifest creation. Lifecycle and cleaning policies in item 26 are unchanged.

- [x] **26 · P2 · Writer lifecycle and cleanup. Resolved.** Worker/queue counts are validated, close and submit are serialized, and closed writers reject submissions. Cleanup defaults to false. Async and checkpoint writers share a tested directory policy protecting root, cwd, and ancestors. [Lifecycle tests](../tests/test_async_writer.py).

- [x] **27 · P2 · Strict restoration and schema boundaries. Resolved.** Callable identity is checked at save time and strict restoration rejects missing energy functions. Schema versions 1 and 2 are recognized; malformed/future versions fail explicitly. Loading is documented for trusted inputs, and version 1 can restore only fields it actually saved. General migration of arbitrary historical Python layouts is not promised. [Schema tests](../tests/test_checkpoint_schema.py) and [continuation tests](../tests/test_checkpoints.py).

- [x] **28 · P2 for RL · Learning-rate annealing uses the wrong time unit. Resolved.** [PPOTrainer](../jaxdem/rl/trainers/ppo_trainer.py) maps completed optimizer updates to epochs using `num_minibatches // accumulate_n_gradients`, then evaluates the cosine schedule over `num_epochs`. The rate stays constant throughout each epoch, including accumulated updates. [Regressions](../tests/test_ppo_math.py) check within-epoch constancy, epoch boundaries, and continuation with retained optimizer state. Resuming requires the full optimizer state; `start_epoch` alone does not restore schedule progress or optimizer moments.

- [x] **29 · P2 for RL · Gradient accumulation occurs after the optimizer transformation. Resolved.** [PPOTrainer](../jaxdem/rl/trainers/ppo_trainer.py) now wraps clipping and the stateful optimizer in `optax.MultiSteps`, averaging raw gradients before either operation. Accumulation of one uses the direct optimizer chain. [Regressions](../tests/test_ppo_math.py) compare emitted updates, parameters, Adam moments, and counters with an equivalent larger batch across multiple groups of varying vector gradients. The optimizer state layout changes; this follow-up does not migrate optimizer states created by the previous transformation chain.

- [x] **30 · P2 for RL · Prioritized sampling and importance weights. Resolved after review.** [PPOTrainer](../jaxdem/rl/trainers/ppo_trainer.py) normalizes `priority**alpha + epsilon` in log space to avoid power and sum overflow. Probabilities below the dtype's normal range are floored and renormalized to preserve representable support. Categorical draws with replacement match importance weights; duplicate writes remain deterministic. [Regressions](../tests/test_ppo_math.py) include finite float32 overflow counterexamples, zero priorities, alpha zero, weighted expectations, and training. The full-correction claim remains limited to fixed categorical draws.

- [x] **31 · P2 for RL · Custom bijector comparisons removed as requested.** Box, max-norm, and free spaces inherit Distrax's conservative identity behavior. The custom parameter-comparison code and equality-only tests are removed. Invalid bounds, radius, and epsilon remain rejected; [numerical tests](../tests/test_action_spaces.py) retain inverse/Jacobian, boundary, origin, and JIT-argument coverage.

- [x] **32 · P1 for packaging · Supported Python/dependency baseline. Resolved.** Metadata and documentation now require Python 3.12+, JAX 0.8.1+, and Optax 0.2.6+. A fresh Python 3.12/JAX 0.8.1/Optax 0.2.6 environment builds and runs the core wheel outside the repository; Python 3.14/JAX 0.11.0 is exercised locally. CI defines the endpoint matrix. This is evidence for those configurations, not every future version combination.

- [x] **33 · P1 for packaging · Distribution contents and installed-wheel behavior. Resolved.** Setuptools discovers only `jaxdem` namespaces; the correctly named `MANIFEST.in` excludes unrelated generated/source directories and packages `py.typed`. [Distribution inspection](../tools/check_distribution.py) checks both artifacts. The wheel built from the sdist passes core-only tests outside the checkout at the minimum dependency baseline.

- [ ] **34 · P1 for release evidence · Release CI configuration implemented; remote evidence pending.** PR jobs now include CPU Python 3.12/3.14 with both precisions, typing, checkpoint/RL contracts, installed minimum-dependency wheels, and strict docs including packaging-only PRs. Scheduled/manual full CPU and opt-in self-hosted CUDA jobs are configured. **Still required:** enable the GPU runner/repository variable, require checks through branch protection, and obtain successful remote/full release runs. Four local periodic Verlet conservation cases passed; the complete platform/physics matrix was not completed. No push or repository-settings change was made.

- [x] **35 · P2 · Strict documentation and curated examples. Resolved.** The pinned Sphinx toolchain builds with warnings as errors after correcting duplicate re-export indexing and malformed source docs. CI performs one strict build. [Curated smoke runner](../tools/smoke_examples.py) executes introduction, materials, and custom-module guides in isolated working directories; all three passed. The complete gallery is rendered but is not automatically executed.

- [x] **36 · P3 · Canonical names and compatibility aliases. Resolved.** `CundallStrackForce` / `cundallstrack` now implements persistent tangential spring history, contact-frame transport, Coulomb limiting, and separation reset. The temporary uncommitted dashpot alias is removed. [History contracts](../docs/source/user_guide/contact_history.md) explicitly distinguish normal elastic energy diagnostics from complete frictional energy. RL uses only `env_wrappers`; the redundant pre-1.0 `envWrappers` compatibility directory was removed during G. Factory examples remove nonexistent generic syntax and document `Create` as the existing hook exception. Adjacency and asynchronous frame-writing descriptions now match runtime behavior. [Public API tests](../tests/test_release_contracts.py).

- [ ] **37 · P2 for performance evidence · Benchmark fixtures repaired; production baseline regeneration pending.** Fixtures have correct topology, balanced mixed populations, and coherent clump properties; end-to-end deformable workloads attach an actual bonded force model. [Fixture tests](../tests/test_benchmark_contracts.py) pass. [New validated measurements](PERFORMANCE_RELEASE.md) cover all populations, shear, and VTK at N=256, with sphere/clump/mixed CUDA rollouts at N=4096. Historical plots are labeled as predating corrected fixtures. **Still required:** regenerate production-scale baselines and an uncontended CPU campaign before using them in release performance claims.

- [x] **38 · P2 · Correctness-gated rollout and I/O measurements. Resolved within the documented workload scope.** [Release benchmark harness](../benchmarks/release_workloads.py) separates first compiled call, warmed snapshot replay, sustained state advancement, and completed rollout plus VTK output. It records versions/device/precision and rejects nonfinite/overflow outcomes. Independent workloads clear JAX caches before compilation timing. [Results and limitations](PERFORMANCE_RELEASE.md) preserve raw measurements and distinguish bounded evidence from production throughput claims; item 37 retains the outstanding production campaign.

- [x] **39 · P3 · Bounded analysis caches and explicit memory limits. Resolved.** Analysis keeps at most 16 compiled wrappers and exposes `clear_jit_cache`. Array values are no longer copied into static cache keys. Chunking transfers bounded pair chunks; retained host indices remain O(P), with optional `max_pairs` rejection before materialization and during custom enumeration. Empty output trees are stable with zero sums and NaN means. [Agreement/cache/budget tests](../tests/test_analysis_engine_contracts.py).

- [x] **40 · P2 · Core and optional dependency boundaries. Resolved.** Core keeps Optax because minimization is a standard System component. VTK, HDF5, and `orbax-checkpoint>=0.12.1` move to the `io` extra and lazy imports; missing features report the installation extra. Legacy utility module aliases remain available. [Blocked-optional-import tests](../tests/test_release_contracts.py), a fresh core-only wheel environment, current I/O/RL suites, and the strict docs build exercise the declared boundaries.

- [x] **41 · P2 · Public layout and reduction contracts. Resolved.** [Method-level shape table](../docs/source/user_guide/release_contracts.md) distinguishes snapshots, one-axis stepping batches, and arbitrary leading axes for thermal reductions. Thermal/potential helpers map snapshot kernels and reduce only the particle axis; potential energy requires matching state/system leading axes. Velocity scaling is explicitly single-snapshot with external `vmap` available. [Tests](../tests/test_dynamics_state_contracts.py) cover stacked snapshots, trajectories, and differing material tables; existing B=1 stepping tests remain.

- [x] **42 · P1 · Packing/jamming callback integer dtypes. Resolved.** Callback result shapes and host grouping arrays consistently use `int32`, independent of JAX floating-point precision. The packing regression runs in the CPU precision matrix. [Regression](../tests/test_force_search_contracts.py).

- [x] **43 · P1 · Hash dtype and pre-overflow checks. Resolved.** Hashes use `uint64` with x64 and `uint32` otherwise, with the unsigned maximum reserved as padding. Dimension conversion uses a strict representable bound and products are guarded before multiplication. Unsigned coordinate arithmetic handles valid axes above the signed range. Invalid hash geometry reports overflow. Fresh-process regressions explicitly verify both actual precision settings. [Tests](../tests/test_boundary_search_geometry.py).

- [x] **44 · P1 · Lees–Edwards clump wrapping. Resolved.** State wrapping applies the image rule to shared `pos_c` values, preserving the rotated member offsets and rigid-body centers. Search wrapping separately uses actual primitive positions; both use canonical integer shear axes. [Domain implementation](../jaxdem/domains/lees_edwards.py).

- [x] **45 · P1 for custom minimizers · Static optimizer computation identity. Resolved.** Optimizer equality/hash include constructor identity and recursively immutable metadata. Same-name closures with different captured rates cannot reuse an incompatible executable. Metadata export returns independent JSON-compatible containers. [Regression tests](../tests/test_minimizer_contracts.py) cover distinct compiled updates, attribute/nested mutation rejection, and serialization isolation.

**Architectural work to settle before freezing 1.0**

The A–F implementation keeps the existing particle-shaped execution arrays and unchecked stepping path. The contracts and regression evidence are recorded in [ARCHITECTURE_A_F.md](ARCHITECTURE_A_F.md). Shared facet incidence remains explicitly unsupported; construction rejects it instead of silently choosing a facet.

- [x] **A · Separate model evaluation from model evolution.** `System.evaluate_forces` and `Collider.evaluate_force` inspect forces without advancing contact/plastic history or consuming queued loads. Initialization remains explicit. Physical stepping owns bonded reference evolution, contact advancement, load consumption, both final kicks, thermostat finalization, and protocol callbacks. Read-only minimization uses the collider contract rather than concrete-type dispatch. [Lifecycle contract](../docs/source/user_guide/evaluation_contract.md).

- [x] **B · Make geometry and search capabilities explicit.** Domains declare `SearchGeometry` plus shear/cache hooks; hashed colliders validate compatibility, while displacement-only colliders support arbitrary domain metrics. NeighborList delegates geometry requirements to its underlying search implementation. Force reach, image-aware stencils, cutoff/capacity queries, and invalidation preserve their established contracts. Cache invalidation is a collider method; domain rescaling preserves surviving contact history.

- [x] **C · Separate body, member, and facet topology logically.** `BodyTopology` provides padded independent body slots, representative/member maps, and fixed/valid masks. State exposes optional validated body mass records. Minimization packs one independent position/rotation per body and uses representative body-total force/torque, preserving the tuned member-array execution format. Empty states and heterogeneous batches are covered. Shared-vertex facet incidence remains guarded until an explicit connectivity representation is supported.

- [x] **D · Use an explicit versioned serialization contract.** New Orbax/HDF5 files share schema 3 manifests and physical conventions. Current-schema loads require complete saved fields/static configuration; unsupported callables fail at save time. Legacy schemas 1/2 remain explicit migration paths. Tests continue both custom physics and nonempty contact history after restoration. [Serialization contract](../docs/source/user_guide/serialization_contract.md).

- [x] **E · Establish one mutation and PyTree contract.** Mutable host assembly and return/rebind semantics under JAX are documented, including whole-field versus nested quaternion edits, dynamic numerical leaves, static computation identity, and cache-aware updates. `species_capacity` belongs to force laws/compositions rather than a System-maintained type list. Existing permissive clump construction and mutable quaternions remain supported. [Mutation contract](../docs/source/user_guide/mutation_contract.md).

- [x] **F · Put checked orchestration around fast kernels.** Optional `System.step_checked` returns accepted steps and persistent overflow/nonfinite status, rolling back a rejected step without changing the unchecked loop. Compression and jamming expose termination reasons and propagate failed minimization instead of treating it as a successful packing. Explicit retry retains caller control of capacity, timestep, and history. Existing host validation and writer failure propagation remain available. Remote/full CI and production benchmark evidence remain deferred under 34/37.

- [x] **G · Scope RL separately from the DEM core's 1.0 promise.** RL remains experimental with an explicit [support contract](../docs/source/user_guide/rl_contract.md) and a separate CPU precision/Python CI matrix. Termination/truncation, pre-reset bootstrap, activity masks, recurrent resets, and epoch schedules are defined and tested. Existing model outputs/carry and PPO metrics serve diagnostics. Unused policy heads and the duplicate wrapper package are removed; scenario-only bed data lives under examples. The [final review](G_AND_FINAL_REVIEW.md) records validation and the remaining support limits.

**Suggested implementation order**

1. Repair persistence and search correctness: 01–05, 13–17; introduce the minimal contracts from A, B, and D that those fixes require.
2. Repair rigid-body behavior and initialization: 06–12, 20–21, 44; choose the topology and quantity conventions from C and E.
3. Stabilize public execution, optimization, errors, and supported combinations: 19, 22–27, 41, 45, F.
4. Decide RL scope, then resolve 18 and 28–31 if RL is included in the stable promise.
5. Establish CI/install/docs gates, fix benchmark fixtures and default-dtype failures, and measure the corrected implementation: 32–40, 42–43.
6. Publish a release candidate, compatibility/deprecation policy, changelog, support matrix, citation metadata, and versioned documentation; promote only after continuation, physics, packaging, and performance checks pass.

**Verification record**

The audit used Python 3.14.7 and JAX 0.11.0. Numerical probes ran on CPU with x64 enabled, plus a separate process with x64 disabled for findings 42–43. The available accelerator is an NVIDIA CUDA device; selected existing API/cache/PPO tests also ran there. Test and documentation tools were installed into a temporary environment and exposed to the existing simulation environment. An unrelated stale editable-install redirect in the system Python environment was identified and bypassed; it is not reported as a repository defect.

- Numerical reproductions: see the eight probe scripts and captured evidence. They demonstrate failures on the audited revision; they do not certify unaffected parameter combinations.
- GPU selection: **15 passed, 1 failed**. The failing test is `tests/test_ppo_math.py::test_stateless_bandit`, due to the private `_log_std` access in finding 18.
- Default pytest collection: **409 cases available; 320 passed, 68 skipped, 1 failed, and 20 slow facet/collider cases not completed** across disjoint CPU/GPU portions. The CPU portion supplied 214 passes/28 skips before interruption; the interrupted case subsequently passed on GPU, and a further GPU run supplied 90 passes/40 skips. See [validation record](evidence/validation.txt) for commands, backend splits, and uncompleted cases. This is not a complete-suite or per-platform pass. The commands omit `--full`; expensive conservation tests are consequently outside these runs.
- Mypy using the working-tree configuration: **no issues in 90 source files**. RL is excluded by the pre-existing local configuration change; this is not a whole-library typing certificate.
- Ruff default checks: **78 diagnostics** across library/tests, largely intentional delayed imports and unused imports. No undefined-name findings were returned. These are not counted as 78 independent release bugs; configure the intended import/formatting policy before adopting a lint gate.
- Wheel and sdist build: **succeeded**, but content inspection demonstrated finding 33. No complete clean-install/platform/dependency matrix was run.
- Strict documentation build: **failed with 1,401 warnings** after regenerating API/gallery sources, with gallery execution disabled. The build used Sphinx 9.1.0; see finding 35 for interpretation.
- No full throughput baseline, complete collider parameter matrix, multi-device, TPU, minimum-version, or long conservation campaign was completed. Those remain explicit release gates, not inferred passes.
