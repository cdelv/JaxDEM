# G and final implementation review

Changes remain uncommitted. This review covers the cumulative audit changes from
`2598c85f09dbe2a91b8ff43238e85a900600e6eb` through the working tree, including
the earlier approved commit `48e523b`. Independent reviews covered core physics
and state/search lifecycles, persistence and orchestration, and RL. The final
checks below target changed contracts; they are not a complete release matrix.

## G: experimental RL with explicit contracts

The DEM core's proposed 1.0 compatibility promise excludes `jaxdem.rl`.
The [RL contract](../docs/source/user_guide/rl_contract.md) defines the supported
interfaces and limitations. CI has a separate RL job for Python 3.12/3.14 and
CPU float32/float64, plus the existing optional CUDA job.

- Environments distinguish termination from truncation. Both reset the episode
  and stop the advantage trace; only true termination suppresses bootstrapping.
  Truncations use the final observation before reset, with a temporary recurrent
  carry. Built-in time limits report truncation at `step_count >= max_steps`.
- Frame skipping retains the first episode boundary for each environment.
  Reward is evaluated at the last accepted frame, not summed across frames.
  Reset work and per-environment reset keys are conditional on a boundary.
- Fixed agent slots carry an activity mask. Inactive samples do not contribute
  to losses, advantage normalization, diagnostics, or PER support. A disappearing
  agent terminates its own transition; reactivation starts with fresh memory.
  Collection and recurrent replay use the same reset boundaries. PER correction
  uses the active segment population rather than the padded population.
- Each environment must contain at least one active agent at every observation.
  Construction and the initial reset validate this requirement. Padding must be
  finite; entirely inactive environments are outside the supported contract.
- Schedules remain expressed in epochs; raw gradient accumulation precedes
  clipping and the optimizer. Model outputs, carry/reset methods, and existing
  PPO metrics supply diagnostics without another generic interface.
- Only `jaxdem.rl.env_wrappers` remains. The old `envWrappers` directory was a
  redundant compatibility shim and has been removed. Tests cover both wrapper
  orders, batch size one, validation, inherited boundaries, and JIT identity.
- The unused bed dataset moved to `examples/data/settled_bed.json`, preserving
  all 414 positions/radii. Scenario environments remain experimental examples;
  no runtime module imports this data.

## Corrections and simplifications found in the final review

| Finding | Final change and evidence |
| --- | --- |
| Contact-force analysis initialized fresh history, losing frictional state | A collider history query returns stored pair history without advancing it. An accumulating-law regression checks both measured force and unchanged history. |
| Composed force laws assumed one-dimensional history | Combiners/routers flatten internal storage and restore each child's declared trailing shape. Matrix, scalar, and empty-pair cases are tested. Explicit widths avoid ambiguous zero-sized reshapes. |
| State construction rotated offsets twice | Construction waits for `__post_init__`; later whole-field assignments still refresh the cache. State/cache/dynamics tests pass. A representative JAXPR shrank from 88 to 44 equations, but compiled StableHLO was identical, so this is not a demonstrated throughput improvement. |
| MaxNorm used inconsistent smoothed forward and exact inverse/Jacobian formulas | The radial map now has matching inverse and log determinant, with finite origin derivatives and the correct second-order term. Tests cover nontrivial epsilon values, forward/reverse Jacobians, inverse derivatives, and the origin Hessian. Entropy rejects dimensions above six before constructing exponential quadrature storage. |
| Policy models allocated unused parameters | Removed discrete sigma parameters, the unused sigma-head log-standard-deviation parameter, and redundant recurrent actor/critic heads. Recurrent policies retain their fused output head. |
| RL replay/reset boundaries were incomplete | Tests exercise truncation bootstrap before autoreset, individual-agent disappearance, reset-to-empty rejection, and LSTM/MinGRU replay through the actual PPO loss across inactivity/reactivation. |
| Documentation and package output could retain removed names | Removed obsolete generated API pages locally; clean artifact inspection confirms both deleted modules are absent and the canonical wrapper is present. Gallery data has an introduction and an indexed documentation page. |

Existing explicit initialization, mutable quaternion behavior, permissive clump
assembly, force-law capabilities, and fast unchecked stepping remain intact.
No manual force-law type list or new automatic initialization state was added.
Compatibility paths for supported checkpoint schemas and result unpacking have
actual callers and remain. The YADE optimization work remains a proposal.

## Validation

Executed locally on 2026-09-11, with Python 3.14.7/JAX 0.11.0 unless noted.
Groups overlap and should not be added into a unique test count.

| Configuration | Checks | Result |
| --- | --- | --- |
| CPU float32 | Evaluation/history, force cutoffs and checkpoint continuation, state cache, dynamics, minimizers, initialization, composed history | 80 passed |
| CPU float32 | Action spaces, wrappers, TensorBoard metrics from actual training, RL contact history | 32 passed |
| CPU float64 | Same group plus composed history | 36 passed |
| CPU float32 and float64 | PPO math, schedules, learning, and new boundary/mask regressions | 22 passed in each precision |
| CUDA float32 | PPO, action spaces, wrappers, and RL contact history; GPU backend verified | 53 passed |
| Core typing | Existing mypy configuration, retaining RL exclusions | 94 files passed |
| Documentation | Strict Sphinx build with warnings as errors | Passed |
| Style | Ruff on RL/forces and final regression files, Black on 30 files, diff whitespace | Passed |
| Packaging | Clean staged wheel/sdist, archive contents and extras, minimum-JAX installed-wheel core tests outside the checkout | Passed; 2 tests on Python 3.12/JAX 0.8.1 |

The wider Ruff scan still reports two pre-existing ambiguous `l` variable names
in `utils/random_sphere_configuration.py`; this review does not claim a clean
whole-repository lint run. Dependencies emit Flax `.value` and JAX compatibility
deprecation warnings. The wheel build reports the existing setuptools license
metadata deprecation.

The new recurrent replay equivalence tests initially failed on CUDA because
single-step and sequence matrix kernels accumulated differently at default TF32
precision. They now request highest matrix precision locally to isolate reset
semantics, without loosening tolerances or changing runtime precision. The full
CUDA group was rerun successfully after that correction.

Items **34** (successful remote/full release checks) and **37** (production-scale
benchmark regeneration) remain deferred. These focused checks establish the
reviewed contracts, not universal optimality, a new throughput claim, or complete
GPU/platform coverage.
