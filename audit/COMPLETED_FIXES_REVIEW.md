# Review of completed audit fixes

**Current status: the four issues reopened by this review are repaired, and
their counterexamples are regression tests. All 26 scoped items are closed.
Changes remain uncommitted.** See the repair follow-up below; the initial review
and counterexample outputs are retained as historical evidence.

Reviewed 2026-09-11 against the uncommitted working tree based on `bafaa75`.
Scope: the 26 checked numerical/API issues in [the audit](README.md), including
the accepted explicit-initialization and construction contracts. Three parallel
Sol reviewers examined dynamics, geometry/search, and persistence/RL; the main
reviewer checked history lifecycle and reconciled the findings.

**Initial verdict: 22 items retained completed status; 05, 16, 17, and 30 were reopened.** Passing
existing tests did not expose the counterexamples below. Item 15's atomic cache
replacement is correct; its new-pair initialization gap is tracked under 17.
Item 23 has correct runtime behavior but a stale public method docstring.

The only runtime change made during this review was the explicitly requested
analytical-gradient capability default: `ForceModel` returns `True`, ordinary
laws inherit it, and incompatible laws override it with `False`. Combiners and
routers still aggregate their components' capabilities. The defects below are
recorded for repair, not fixed by this review. Nothing was committed.

## Repair follow-up — 2026-09-11, uncommitted

The user subsequently authorized repairing all four reopened issues. **05, 16,
17, and 30 are now fixed**, with the counterexamples added to regression tests.
The original review below is retained as historical evidence; its reopening
verdicts describe the tree before this repair follow-up.

- **05:** cache validity now compares build-time skin and canonical shear axes.
  Both missed-force cases check rebuilding and agreement with Naive forces in
  [geometry tests](../tests/test_boundary_search_geometry.py).
- **16:** particle-count changes with nonempty history require explicit reset.
  Growing capacity preserves old slots and initializes new slots through the law.
- **17:** composites concatenate child initializers; remapping copies surviving
  memory into a law-initialized buffer. [Lifecycle tests](../tests/test_neighbor_list_lifecycle.py)
  cover nonzero initialization, nested/empty composites, force behavior, capacity
  growth, explicit reset, and new pairs.
- **30:** [PPO tests](../tests/test_ppo_math.py) retain the finite float32 overflow
  cases. Log-space normalization preserves the specified additive epsilon;
  probabilities smaller than the dtype's smallest normal value are floored and
  renormalized, and importance weights use that resulting distribution. Tests
  also check alpha zero, zero priorities, ordering, and finite correction weights.
- The stale NeighborList query and LawCombiner tuple docstrings are corrected.

The fixes add cache-key comparisons and initialization at allocation/rebuild
boundaries, with no new startup lifecycle or host synchronization in timesteps.
History remapping retains its existing comparison algorithm. Production
performance benchmarking remains separate release work.

Repair validation (overlapping precision/device runs are listed separately):

- CPU geometry: **23 passed**; both new cache counterexamples also passed with
  x64 disabled (**2 passed**).
- CPU history lifecycle: **36 passed**, including the persistent-history
  checkpoint; with x64 disabled, **35 passed, 1 checkpoint deselected**.
- CPU PPO: **12 passed**, with seven existing upstream deprecation warnings;
  the focused normalization tests also passed with x64 disabled (**2 passed**).
- Existing checkpoint suite: **76 passed**, with six existing H5/upstream warnings.
- CUDA with x64 disabled: cache and normalization regressions **4 passed**.
- Black, scoped Ruff, and diff whitespace checks pass. Whole-core mypy still
  reports only the two pre-existing argument errors in unchanged
  `utils/geometric_asperity_creation.py:612`; RL remains excluded by the existing
  mypy configuration. No new whole-library typing certificate is claimed.
- The CPU precision-matrix workflow includes the geometry/history suites and
  now explicitly runs the priority-normalization regressions.

## Initial review findings and proposed repairs (historical)

### P1 · 05: neighbor-cache validity omits skin and shear axes

[`_check_and_rebuild`](../jaxdem/colliders/neighbor_list.py#L59) compares current
displacement with the current skin, but does not record the skin used to build
the cached list. Its metric snapshot records box lengths and gamma, omitting
the Lees–Edwards axes.

Two numerical counterexamples:

- Free 2D domain, radius 0.5, cutoff 1, skin 0.1, centers `(0,0)` and `(1.6,0)`.
  Build the list, increase skin to 1, then move both centers 0.31 toward each
  other. Each displacement remains below the new half-skin threshold, although
  the old list has insufficient reach. Build count stays 1 and NeighborList
  gives zero force; Naive gives approximately `(-200,0)` and `(200,0)`.
- Lees–Edwards 3D box of side 10, gamma 0.4, radius 1, centers `(0,0,0.1)` and
  `(0,4,9.9)`. Build with axes `(alpha,beta)=(0,1)`, then replace the domain with
  axes `(1,2)` and the same box/gamma. Build count stays 1 and NeighborList gives
  zero force; Naive gives a nonzero contact force of magnitude about 18000.

**Repair:** snapshot and compare build-time skin, and include canonical shear
axes in the fixed-shape metric snapshot. These are small cache-key additions;
no new preparation API or unconditional rebuild is needed. Add regressions for
both transitions. An explicit invalidation works around these cases today.

### P1 · 16: resizing can silently discard pair memory

[`refresh_collider`](../jaxdem/colliders/__init__.py#L322) preserves history only
when the particle count is unchanged. With a force model supplied, a changed
count falls through to fresh allocation without requiring `reset_history=True`.
This contradicts the documented explicit-reset contract.

A custom one-component history initialized to 7, with existing contact history
set to 21, was refreshed from two particles to three. The call returned without
an error and all history was reset to 7. This also loses accumulated memory for
ordinary zero-initialized history laws.

**Repair:** reject particle-count changes with nonempty pair history unless
`reset_history=True`. Preserving memory across a resize would require an explicit
identity mapping; a guard is the simpler accepted contract. Equal particle counts
do not prove identity is unchanged: callers must still reset after reindexing.

### P2 · 17: custom history initialization is not compositional

[`LawCombiner`](../jaxdem/forces/law_combiner.py) and
[`ForceRouter`](../jaxdem/forces/router.py) combine history shapes but inherit
the base zero-filled `init_history`. They do not combine component initializers.
[`_remap_history_array`](../jaxdem/colliders/neighbor_list.py#L28) likewise fills
new neighbor slots with zeros regardless of the force law's initializer.

A custom Spring-derived law declared history shape `(1,)`, initialized history
to 7, and multiplied its force by that history. For the same overlapping pair:

| Construction | Initial pair history | Force magnitude |
| --- | ---: | ---: |
| Direct law | 7 | 49000 |
| One-law combiner | 0 | 0 |
| One-entry router | 0 | 0 |

After building a list without a contact and bringing the pair together, a
read-only rebuild also assigned history 0 to the new pair. Existing matching
pair histories are remapped correctly. Built-in stateless laws are unaffected;
this is a failure of the public custom-law extension contract.

**Repair:** concatenate child `init_history` results in composites. During a
rebuild, initialize the new history buffer through the law and overwrite matched
slots with their old values. Keep allocation shapes static. Zero must remain a
default initializer, not an assumption embedded in cache management.

### P2 · 30: finite priorities can produce reversed or invalid probabilities

[`_priority_probabilities`](../jaxdem/rl/trainers/ppo_trainer.py#L68) computes
the power before normalizing and replaces overflowed powers with zero. Its sum
can also overflow. Fresh-process probes explicitly verified float32:

| Finite nonnegative priorities | Alpha | Returned probabilities |
| --- | ---: | --- |
| `[1, 1e30]` | 2 | `[0.99999905, 9.9999806e-7]` |
| `[2e38, 2e38]` | 1 | `[0, 0]` |

The first reverses priority ordering; the second is not a probability
distribution. Categorical replacement, duplicate writeback, and importance
weights are otherwise consistent with the probabilities actually supplied.

**Repair:** normalize weights in log space, using
`logaddexp(alpha * log(priority), log(epsilon))` followed by stable softmax,
with explicit zero-priority/alpha-zero handling. This preserves the specified
`priority**alpha + epsilon` distribution. Scaling priorities by their maximum
and then adding the original epsilon would change that distribution. Test finite
power overflow, sum overflow, zero priorities, and alpha zero.

## Per-item assessment

“Retain” means the scoped repair is correct under its documented usage, not that
every implementation is proven globally optimal.

| Item | Verdict | Correctness and simplicity/performance assessment |
| --- | --- | --- |
| 01 | Retain | One returned plastic reference update per step, before force evaluation; direct evaluations remain frozen. Pure hook is appropriate. |
| 02 | Retain | Physical-image oracle and collider checks support fractional shear, arbitrary axes, and unwrapped positions in the documented unique-image regime. Conservative stencil/dedup is reasonable. |
| 03 | Retain | Force/search bounds agree for builtins, composites, and current facet geometry. Global maximum bounds can over-search polydisperse systems but are safe and simple. |
| 04 | Retain | Dynamic bounds use actual query geometry through the domain hook; minimization does not invoke reflective physics. |
| 05 | Reopen | Missing build-time skin and shear-axis cache inputs; see counterexamples above. |
| 06 | Retain | Langevin noise is gathered by compact clump ID, preserving body coherence and fixed velocities. Vectorized implementation is appropriate; no removed Langevin tests were recreated. |
| 07 | Retain | COM-to-member plus member-to-contact is the correct contact arm. Collider torque aggregation supplies the COM moment without double-counting the contact velocity. |
| 08 | Retain | Dispatch by array layout handles B=1 and larger batches consistently, with trace-time errors for incompatible layouts. No per-step host synchronization. |
| 09 | Retain | Explicit initialization computes frozen-history forces and normal managed loads before integrator hooks. No automatic flags or hidden startup paths. Caller owns initialization and restart usage. |
| 10 | Retain | Guard rejects shared contact-facet incidence before mutation; bonded deformable meshes remain supported. A simple guard fits current storage. |
| 11 | Retain | Permissive low-level volume defaults are intentional placeholders. Geometry builders replicate total body volume consistently. |
| 12 | Retain | Permissive inertia defaults are intentional; caller must compute body properties. Mandatory guards would contradict the requested workflow. |
| 13 | Retain | Importable callbacks survive metadata round trips; missing required callbacks cannot silently disappear during restoration. |
| 14 | Retain | Built-in domain/integrator configuration survives reconstruction. Broader plugin/schema evolution remains separate architecture work. |
| 15 | Retain | Neighbor indices and remapped history are returned atomically, including from energy evaluation. New-pair initializer loss is covered by 17. |
| 16 | Reopen | Same-index refresh works; changed particle count can bypass the promised explicit reset. |
| 17 | Reopen | Uniform array/shape API is sound, but custom initializers are lost in composition and on new neighbors. |
| 18 | Retain | Trainer no longer depends on model-private `_log_std`; no replacement private diagnostic was introduced. |
| 23 | Retain; docs follow-up | Exact queries honor cutoff/width and preserve the force cache, including truthful zero-width overflow. Method docstring still describes obsolete cached-query behavior. |
| 25 | Retain | Worker failures are recorded before task completion, surfaced after draining/joining, and consumed once. Separate lifecycle issue 26 remains open. |
| 28 | Retain | Learning-rate schedule is measured in epochs, correctly accounting for actual optimizer updates and restored optimizer state. |
| 29 | Retain | Raw gradients are averaged before clipping/Adam. Direct path for accumulation one avoids unnecessary wrapping. Pre-existing invalid-count validation is separate hardening. |
| 30 | Reopen | Sampling/importance-weight structure is repaired, but normalization fails on finite float32 inputs. |
| 42 | Retain | Host callback outputs and declared result shapes consistently use int32 across floating precision settings. |
| 43 | Retain | Precision-dependent unsigned hashes, reserved sentinel, cast limits, and pre-multiply guards are coherent. Explicit precision probes cover boundary arithmetic. |
| 44 | Retain | Wrapping shared body centers preserves rigid offsets. The shear-axis cache defect belongs to 05, not this wrapping operation. |

## Small follow-ups and limits of optimality claims

- Correct the NeighborList query docstring: it still says cutoff/width are
  ignored and a refreshed cached list is returned; the implementation delegates
  the explicit query to its secondary collider.
- Correct `LawCombiner.laws` documentation calling the tuple static. Its law
  parameters are now pytree data. The corresponding historical audit claim has
  been updated; the broader mutation/PyTree design issue remains open.
- History remapping compares each new slot against old slots, O(N K²) in
  comparisons for equal capacity K. This is straightforward for small lists;
  profile large stateful capacities before choosing a more involved mapping.
  No new measured slowdown or quadratic allocated-memory claim is made here.
- Continuously changing shear can force rebuilding, and MultiCellList disables
  shear AABB pruning conservatively. Those are explicit correctness/performance
  tradeoffs, not evidence that a more complicated implementation is warranted.
- The existing [performance measurements](BOUNDARY_PERFORMANCE.md) predate the
  final initialization revision. This review does not establish optimality or
  current-tree production throughput; benchmark representative rollouts before
  making release performance claims.

## Validation

Review runs, with potentially overlapping groups (do not sum):

- Geometry suite: **21 passed**.
- Neighbor-list lifecycle suite: **26 passed, 1 checkpoint deselected**.
- Dynamics/construction/friction/packing selection: **31 passed, 3 checkpoint
  cases deselected**.
- Focused callback/domain/integrator checkpoint selection: **13 passed**.
- PPO and asynchronous writers: **16 passed**, seven existing upstream warnings.
- Minimizer suite after changing the capability default: **10 passed**.
- Separate numerical probes reproduced all four reopened items, including a
  subprocess that asserted float32 for priority normalization.

Existing lifecycle tests use zero-initialized histories and therefore do not
establish general initializer preservation. Passing geometry tests do not cover
the two cache transitions above. No full conservation/device/dependency matrix
or new production benchmark campaign was run in this review. The known two
mypy argument errors in unchanged `utils/geometric_asperity_creation.py:612`
remain outside these fixes.
