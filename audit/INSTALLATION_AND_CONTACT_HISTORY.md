# Installation and contact-history feedback

These changes implement the installation, Cundall–Strack, quaternion, and
bijector feedback after commit `48e523b`. They remain uncommitted. Existing
unrelated workspace files are preserved.

## Installation

JaxDEM extras select features: `io`, `rl`, `docs`, `test`, and `all`.
`all` includes the other four extras. CUDA/TPU extras are removed from JaxDEM.

From a checkout:

```bash
python -m pip install '.[all]' 'jax[cuda13]'
```

This command needs no repeated minimum version: pip combines `jax[cuda13]`
with JaxDEM's `jax>=0.8.1` dependency and any tighter requirements from its
extras. An existing compatible JAX installation may be retained. Users can
explicitly request a compatible version, e.g. `'jax[cuda13]==0.11.0'`, when
reproducibility requires it. The installation guide shows GitHub, local, and
future PyPI forms and links to JAX's platform/driver requirements.

The RL metrics path previously imported TensorFlow through Flax although the
extra only declared TensorBoard. It now uses TensorBoardX to write the same
scalar/text events, without TensorFlow. TensorBoard remains available for
reading/viewing logs. The training and optimizer computations are unchanged.

## Cundall–Strack

The canonical `CundallStrackForce` / `cundallstrack` now stores tangential
spring displacement and the previous normal in a `(2 * dim,)` history array.
It transports the spring between contact frames, follows common 3D spin,
accumulates tangential relative motion once per advancing force evaluation,
and applies a Coulomb return map to prevent spring windup. Separation clears
history. Read-only evaluations return history unchanged.

The temporary, uncommitted `LinearDashpotForce` alias is removed. This is a
change in contact physics from the previous viscous approximation. Existing
stateless trajectories/checkpoints cannot reconstruct missing tangential
memory. Current-history checkpoints retain exact continuation data.

The model requires a history-capable `NeighborList`. Advancing neighbor-list
evaluation also resets excluded and padding slots to the law's initializer,
preventing hidden history from accumulating before a pair becomes eligible.
The six built-in RL callers now use history-capable colliders and explicitly
initialize freshly constructed state/system pairs before sensing or saving
their reset snapshot. Compiled reset
construction uses explicit static capacities/stencils; host value validation
remains in ordinary constructors, while traced construction retains shape
checks and requires static allocation parameters.

**Energy limitation:** the existing `energy` signature has no pair-history
argument, so this law reports normal elastic energy only. It does not report
tangential stored energy or dissipated work and explicitly disables the
analytical energy-gradient capability. An explicit minimization objective is
required. See [the user guide](../docs/source/user_guide/contact_history.md).

## Removed restrictions

Quaternion freezing was introduced to prevent nested component mutation from
leaving `_pos_p_rot` stale. Quaternions are mutable again. Whole `state.q` or
`state.pos_p` assignment refreshes that cache automatically; after changing
`state.q.w` or `state.q.xyz`, call `state.refresh_rotation_cache()`. Optional
host validation reports a missed refresh. The cached layout is retained to
avoid repeatedly rotating member offsets in hot force paths.

BoxSpace, MaxNormSpace, and FreeSpace no longer override Distrax's bijector comparison.
Their equality-only tests are removed. Numerical inverse/Jacobian tests and
parameter validation remain.

## YADE review

The requested repository was found at `/home/wind/Documents/YADE2/bare`.
[The proposal](YADE_PERFORMANCE_PROPOSAL.md) cites its recent commits and
compares their changes with JaxDEM's current implementation. Candidate
experiments cover compact active contact work, heterogeneous force routing,
local algebra reuse, geometry specialization, and host synchronization.
No YADE optimization has been ported. Timings from its C++/Kokkos implementation
are not asserted to transfer to JAX.

## Validation

Groups overlap and must not be added as a full-suite total.

- The requested `pip install '.[all]' 'jax[cuda13]'` command resolved successfully
  in a **dry run** with the existing compatible JAX 0.11.0 CUDA installation.
  No installation or environment modification was performed by that dry run.
- Float32 CPU force/minimizer/import/state/action regression selection:
  **78 passed**.
- Cundall-specific numerical cases: **12 passed** in both float32 and float64,
  plus **1 checkpoint continuation test passed**. Cases include static hold,
  damping, Coulomb limits, contact loss/recontact, rotating normals, batched
  antiparallel normals, common spin, antisymmetry, and frozen history.
- Float64 CPU geometry/contact-history/lifecycle selection:
  **70 passed, 2 checkpoint cases deselected**.
- Float32 CUDA contact-history and neighbor lifecycle selection:
  **47 passed, 2 checkpoint cases deselected**.
- Small compiled ThreeGears and SwarmRoller3D resets verify finite initialized
  forces, allocated history, and clear overflow flags.
- Final float32 CPU reset/state/logging integration: **22 passed**.
- Python 3.12/JAX 0.8.1 contact-history tests: **12 passed**, and rebuilt
  core-only installed-wheel tests outside the source tree: **2 passed**.
- The real contact-history documentation example executes successfully.
- A two-epoch PPO run with logging enabled passes and produces readable scalar
  and text events without TensorFlow; the **12 existing PPO tests** also pass.
- Strict documentation build and core mypy (**91 source files**, RL excluded
  by the retained user configuration) pass. Scoped RL typing was also checked.

The earlier full-release gates in audit items 34 and 37 remain open. These
bounded checks do not certify the complete platform, conservation, or
production-performance matrix.
