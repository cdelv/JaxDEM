# Remaining numbered audit items

The prior approved work is committed as `48e523b` (2026-09-11). This report covers
subsequent, uncommitted changes for items **19–22, 24, 26–27, 31–41, and 45**.
Items already repaired were preserved. Existing unrelated files and the user's
mypy RL exclusion were retained. No push was made.

The [main checklist](README.md) now has **43 of 45 numbered items checked**.
Items **34 and 37 remain open** for release evidence. There was no item 00 in the
original numbered list. Architectural proposals A–G are separate design/support
policy questions, not seven additional claims of runtime defects.

The later [installation/contact-history feedback](INSTALLATION_AND_CONTACT_HISTORY.md) supersedes this snapshot's quaternion freezing, bijector comparisons, temporary dashpot alias, hardware extras, and metrics dependency decisions. Historical validation counts below refer to the implementation at that stage.

The [G and final review](G_AND_FINAL_REVIEW.md) also supersedes row 36's wrapper
alias decision: only `env_wrappers` remains; `envWrappers` was removed.

## Implementation

| Items | Result and rationale | Regression coverage |
| --- | --- | --- |
| 19 | Static Python counts retain static loop bounds in `System.step`; traced counts use `System.step_dynamic`. Constant-stride rollouts retain reverse-mode support. | Direct and rollout derivatives, dynamic counts, fractional/negative/rank guards. |
| 20 | Frozen quaternion components prevent hidden cache mutation. Whole-field replacement deterministically refreshes offsets without Python frame inspection. Current force reach and box geometry drive search/cache validity. | Quaternion mutation rejection, replacement positions, existing cache tests. |
| 21 | `finalize_step` thermostats run after both terminal kicks. COM drift uses body mass once, including total mass below one; fixed motion is excluded and preserved. | Terminal torque, unequal member counts/masses, fixed bodies, small masses, invalid intervals/thermal parameters. |
| 22 | Minimizers report convergence, budget exhaustion, nonfinite objective, or search overflow through `MinimizationResult.reason`, while retaining four-value tuple usage. | Conservative gradients, fixed coordinates, frozen history, nonfinite and overflow termination. |
| 24 | Optional host validators check numerical/shape/ID/body/domain contracts. Replicated-field checks use representative indices with linear work per field. Public State guards survive Python optimization. | Invalid IDs, negative radii, quaternion/facet/fixed types, nested router bounds, sparse clumps, inconsistent fields, Python `-O`. |
| 26–27 | Shared opt-in cleanup, safe writer lifecycle, strict callable restoration, and explicit schema 1/2 compatibility bounds. | Closed submission, invalid workers, protected directories, schema rejection, existing checkpoint continuation. |
| 31 | Bijector equivalence uses current parameters and is conservative with tracers. | Mutation, bounds, origin, inverse, stable log determinant, and resolvable autodiff checks in both precisions. |
| 32–33 | Python 3.12+, JAX 0.8.1+, Optax 0.2.6+; package only `jaxdem`, include `py.typed`, inspect wheel/sdist. | Fresh minimum-version core wheel installed and tested outside the repository. |
| 34–35 | PR CPU/precision/version tests, package/type checks, strict docs, curated example execution, and scheduled full runs are configured. | Local strict build and three executable guides pass; remote/full evidence remains open. |
| 36 | Canonical `LinearDashpotForce` and `env_wrappers` names retain old aliases. Factory/adjacency/writer descriptions match behavior. | Class/factory/import identity and executable custom-module guide. |
| 37–38 | Corrected populations/topology and real bonded deformable workloads feed a reproducible timing harness. Compile, snapshot replay, sustained stepping, and completed I/O are separate. | Fixture tests, finite/overflow/file gates, CPU and CUDA result JSONs. Production-scale regeneration remains open. |
| 39 | Analysis wrapper cache is bounded to 16 entries; device chunks are bounded and optional `max_pairs` limits host materialization. | Chunk agreement, cache eviction, large early budget rejection, empty output tree. |
| 40 | Core retains Optax. I/O libraries move to `io`, with lazy imports and actionable missing-extra errors. | Blocked optional imports, fresh core-only wheel, current I/O/RL tests and docs. |
| 41 | Public shape table distinguishes snapshot kernels, one-axis stepping batches, and arbitrary leading thermal reduction axes. | Stacked/trajectory reductions, matching systems with distinct material tables, existing B=1 stepping. |
| 45 | Static optimizer identity includes its constructor and immutable configuration; metadata export is detached and JSON-compatible. | Same-name closures produce distinct updates, nested/attribute mutation is rejected, exported metadata is independent. |

Explicit initialization remains the user's responsibility. No automatic startup
flag or hidden half-step protocol was introduced. Low-level clump placeholders
remain permitted; strict validation is an opt-in setup operation outside the hot
loop. Multiplicative thermostats do not create motion from zero kinetic energy.
No new Langevin test was added.

## Validation and limits

Runs used Python 3.14.7/JAX 0.11.0 unless otherwise stated. Test groups overlap;
the counts below must not be added as a whole-suite total.

- The integrated float32 CPU contract selection produced **87 passes and one
  failure** in an overly cancellation-sensitive near-boundary autodiff assertion.
  The action test now checks stable log determinants against an independent
  scalar double-precision reference, retaining boundary inverse coverage and
  comparing autodiff where the float32 slope is resolvable. The complete action
  suite then passed **21 tests in each precision**; the stable runtime formula
  was unchanged.
- Final state/thermal/stepping regressions include the independent review's fixed
  kinetic energy, small-body-mass drift, ID, and vectorized validation cases.
  The final focused suite passed **19 tests on float32 CPU**. The final CUDA
  float64 dynamics, layout, initialization, and minimizer selection passed
  **41 tests**.
- The I/O/RL/release group passed **50 tests**, including all **12 PPO math
  tests**. Seven existing Flax warnings were reported. Benchmark contracts later
  passed **16 tests**, including the added batch-one workload.
- Checkpoint selection passed **76 tests** and exposed one missing legacy
  `utils.h5` module alias. The alias was restored and that HDF5 regression passed
  separately. Earlier persistence suites and their evidence remain recorded in
  the main audit; these are not claims of all tests passing in one invocation.
- CUDA float64 long conservation selection passed **4 tests**: periodic Verlet
  spheres and clumps in 2D and 3D, with explicit initialization and unchanged
  conservation tolerances. **36 conservation cases were deselected.**
- Strict Sphinx warnings-as-errors build passed, including regenerated API and
  rendered gallery pages. The introduction, materials, and custom-module guides
  executed successfully in isolated working directories. The entire gallery was
  not executed.
- A fresh Python 3.12/JAX 0.8.1/Optax 0.2.6 core environment built and inspected
  wheel and sdist, then passed **2 installed-wheel smoke tests** outside the
  source checkout. The RL-only alias check was deliberately deselected in this
  core-only environment. The minimum-version source smoke/layout selection
  also passed **5 tests**.
- Mypy passed **91 source files** under the retained configuration excluding RL.
  The two geometry argument errors observed in earlier reports were repaired by
  passing JAX arrays at the typed helper boundary. This is not RL typing evidence.
- Black checks, scoped Ruff (the existing late-import convention excluded), and
  `git diff --check` pass for the changed source and tests.
- The broad default CPU run reached 29% without failures before being stopped in
  the slow facet/collider matrix. It is explicitly **not a complete suite pass**.

[Execution timings](EXECUTION_VALIDATION.md) cover static/dynamic stepping and
State replacement. [Performance evidence](PERFORMANCE_RELEASE.md) includes raw
CUDA float32 N=256/N=4096 rollout timings and an explicitly contended CPU float64
run. Timings are bounded development evidence; no optimality or production speed
improvement is asserted.

## Still required before release

1. **34:** enable the intended GPU runner and repository variable, require CI
   checks through branch protection, and obtain successful remote and complete
   scheduled/full release runs. The workflows are configured locally; no remote
   jobs or repository settings were changed.
2. **37:** regenerate the historical production-size benchmark matrix from the
   repaired fixtures, including an uncontended CPU campaign, before using those
   numbers in release claims. Current N≤4096 evidence does not certify the former
   100,000/500,000-particle baselines.
3. Set the support/deprecation decisions in architectural proposals A–G and the
   release checklist. Multi-device/TPU behavior, arbitrary historical component
   migration, and differentiation through every collider are not established by
   these bounded checks.

The numbered implementation fixes make those decisions and validation gates
reviewable; they do not on their own certify a 1.0 release.
