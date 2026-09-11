# Execution, validation, and compatibility

Python 3.12 or newer is required. The core dependency baseline is JAX 0.8.1 and
Optax 0.2.6; minimization is part of the core API. CI checks Python 3.12 and 3.14
with float32 and float64, plus a core-only installed wheel at the minimum JAX and
Optax versions. These are supported test configurations, not a promise that every
future dependency release will remain compatible.

Install `JaxDEM[io]` for VTK, HDF5, and Orbax checkpoint support. Checkpoints use
the specific `orbax-checkpoint` distribution, version 0.12.1 or newer. Use
`JaxDEM[rl]` for reinforcement learning and `JaxDEM[docs]` for the pinned
documentation toolchain. Core imports and force computations do not import I/O
or RL dependencies. Requesting an unavailable I/O feature reports its install
extra. Add both `io` and `rl` for checkpointed training.

Metrics logging writes TensorBoard events through TensorBoardX, included in the
`rl` extra. It does not import TensorFlow or require TensorFlow GPU support.

Use `JaxDEM[all]` to install all feature extras (`io`, `rl`, `docs`, and `test`).
Hardware selection belongs to JAX: from a checkout, install
`python -m pip install '.[all]' 'jax[cuda13]'`, or use an exact compatible
JAX version such as `'jax[cuda13]==0.11.0'`. Pip resolves one shared JAX version
for all dependencies. JaxDEM does not define CUDA or TPU extras. See the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for
backend/platform requirements.

## Shapes and stepping

| Operation | Accepted state layout | Result semantics |
| --- | --- | --- |
| `System.initialize`, `System.step`, `System.step_dynamic` | `(N, dim)` or `(B, N, dim)` | A single snapshot or one batch axis, including B=1; state/system layouts must match. |
| `System.trajectory_rollout` | Single snapshot or one batch axis | Prepends a saved-frame axis to the selected output tree. |
| Spatial queries | `(N, dim)` | Single-snapshot kernels; use explicit `jax.vmap` for batches. |
| Thermal kinetic-energy helpers and `compute_temperature` | `(..., N, dim)` | Preserve leading batch/time axes and reduce the particle axis. |
| `compute_potential_energy` | Matching state/system leading axes | Maps snapshot evaluation over those axes. |
| VTK output | Arbitrary leading axes | Explicit `trajectory`/`trajectory_axis` select time; other axes are batches. |

`System.step(state, system, n=...)` takes a static Python integer. It supports
reverse-mode differentiation for differentiable force/collider combinations;
this is tested for simple Naive/Spring dynamics. Use `System.step_dynamic` for
a traced scalar count, such as a variable stride inside a scan. That dynamic
loop does not promise reverse-mode differentiation. Constant-stride rollouts
use the static path; variable strides use the dynamic path. Strides must be
nonnegative integers, not fractional floats that are silently truncated.

Initialization remains explicit: call `System.initialize` once before a new
trajectory; continue an initialized checkpoint directly.

## State edits and host validation

Quaternions are mutable. Replacing the whole quaternion (`state.q = new_q`) or
using `dataclasses.replace(state, q=new_q)` refreshes the rotated member-offset
cache automatically. After editing `state.q.w` or `state.q.xyz` directly, call
`state.refresh_rotation_cache()`. Host validation detects a missed refresh.
Similarly replace `pos_p` through the state. Private cached arrays are
implementation details. State construction does not inspect Python stack frames.

Call `system.validate(state)` at setup boundaries for physical values, shape and
identifier checks. It synchronizes device data and deliberately stays outside
compiled loops. Constructors remain permissive for the construct-then-compute
clump workflow; compute volume/inertia before requesting strict validation.
This is an optional checked workflow, not an automatic per-step cost.

## Minimization and analysis

Minimization returns `MinimizationResult`. Existing four-value unpacking and
indexing remain supported; inspect `result.reason` with
`jaxdem.minimizers.TerminationReason` to distinguish force/energy convergence
from exhausting the step budget, a nonfinite objective, or search overflow.
Objective evaluations freeze history and use conservative gradients. Arbitrary
driven forces require an explicit objective.

Analysis caches at most 16 compiled function wrappers; call
`jaxdem.analysis.clear_jit_cache()` to release those references. This does not
clear JAX's process-wide caches. `chunk_size` bounds device pair storage, while
the returned `pairs` object retains O(P) host indices. Set `max_pairs` to reject
an analysis exceeding your host allocation budget before materialization;
the default `None` imposes no host pair limit. Empty bins preserve the kernel's
output structure, with zero sums and undefined (NaN) means.

Thermal temperature statistics exclude fixed bodies. Drift removal weights each
free rigid body's total mass once, independent of its number of members.
Rescaling preserves prescribed fixed velocities and runs after both terminal
integration kicks. Multiplicative rescaling cannot create motion from zero
kinetic energy; use explicit velocity initialization when that is required.

## Names and persistence

`CundallStrackForce`, factory key `cundallstrack`, stores tangential spring
displacement per contact and requires a history-capable `NeighborList` collider.
See [contact-history usage and energy limits](contact_history.md). RL wrappers live in `jaxdem.rl.env_wrappers`. The former camel-case directory
has been removed; update imports to the canonical name. The established factory constructor hook
`Create` is the documented exception to snake_case methods.

Writer cleanup defaults to false. Request `clean=True` only for a directory
whose output may be removed; roots, the working directory, and its ancestors
are protected. Closed writers reject submissions. Checkpoint loading is for
trusted inputs because metadata resolves Python callables and classes. Schema
versions 1, 2, and 3 are recognized; malformed and future schema versions fail
explicitly. Version 1 can restore only the configuration it actually saved;
version 2 records the expanded restart contract; new schema 3 files add a common
Orbax/HDF5 manifest and strict field/convention validation. See the
[serialization contract](serialization_contract.md). No general migration from
arbitrary historical Python class layouts is promised.

## Release gates

The quick CPU, installed-wheel, type-check, and strict-docs jobs run on pull
requests, including packaging-only changes. Full CPU validation is scheduled
weekly and available through manual workflow dispatch. CUDA jobs require a
self-hosted Linux GPU runner and repository variable `JAXDEM_GPU_CI=true`;
without that setup they are skipped. Enable the runner and require the intended
checks in branch protection before publishing. Workflow configuration alone
does not establish that a remote run or a full conservation campaign passed.
