# Architectural work A–F

This pass implements the architectural contracts using the existing execution
layout. Changes remain uncommitted. Items 34 and 37 and the RL support-policy
decision G are outside this pass. The subsequent
[G and final review](G_AND_FINAL_REVIEW.md) records completion of G and additional
history/cache corrections across these changes.

## A. Evaluation and evolution

`System.evaluate_forces` observes instantaneous forces while preserving pending
loads, physical history, time, and integrator state. Collider implementations
expose `evaluate_force` for read-only pair evaluation. Initialization and
minimization use that interface without dispatching on a specific collider type.
Search-cache changes are returned explicitly.

The existing physical timestep order remains responsible for plastic reference
updates, contact-history advancement, load consumption, integration, thermostat
finalization, and callbacks. The public contract explains exactly which stages
run during evaluation, initialization, and stepping. Custom callbacks must
return their state changes; external side effects cannot be rolled back.

## B. Geometry and search

Domains declare their hashed search geometry and provide shear parameters and
cache snapshots through class-owned hooks. Hashed colliders reject undeclared or
unsupported custom geometry during System construction/validation. Naive and
other displacement-only implementations do not require a hashed geometry.
NeighborList delegates compatibility to its secondary collider.

The per-law interaction reach and explicit query cutoff/capacity remain the
search authority. Invalidation now belongs to each collider. Packing rescaling
invalidates the neighbor cache without resetting the history initialization
counter, so rebuilding preserves memory for surviving contacts. Scaling is about
the domain anchor, including translated boxes.

## C. Bodies and members

`BodyTopology` derives member-to-body maps, representative indices, counts, and
fixed/valid masks. `State.body_mass_properties(validate=True)` returns validated
representative body totals while preserving permissive low-level construction.
The representation uses N padded body slots for static shapes; it does not
promise a smaller allocation than the execution arrays.

Optimization acts on independent body coordinates and rotations. Analytical
gradients select the already aggregated body force/torque once; custom objective
autodiff accumulates member contributions through the shared body coordinate.
The simulation's execution layout and State checkpoint layout remain unchanged.
`State.velocity_at` centralizes COM-to-member/contact velocity and is used by
Cundall–Strack.

Shared facet incidence is explicitly rejected. Supporting shared mesh vertices
requires separate connectivity data and is not silently approximated by the
current one-facet-per-member representation.

## D. Persistence

Schema 3 manifests are shared between Orbax and HDF5. They identify payload,
storage, topology/coordinate/unit conventions, and trusted-input requirements.
Current payloads require complete fields and static configuration; arrays retain
contact/plastic state, keys, buffers, and topology. Importable callable identity
is validated before writing. Unsupported custom payloads fail explicitly.

Legacy Orbax schemas 1/2 and unversioned HDF5 have explicit compatibility paths.
Schema 3 records JAX precision mode and rejects a mismatch before decoding,
preventing an implicit float64-to-float32 restart conversion.
Exact continuation requires compatible code/dependencies and both State and
System, and does not call initialization again. The generic HDF5 object-tree
codec remains available with strict current-schema field validation; it is not
an arbitrary Python object serializer.

## E. Mutation and PyTrees

The documented contract separates mutable host assembly from JAX value
semantics. Numerical leaves are dynamic; structure and callable identity are
static. Nested quaternion edits require cache refresh, whole-field updates
refresh automatically, and collider invalidation/resizing is explicit.

Force-law species capacity is now a class capability composed recursively by
routers/combiners. System no longer maintains a list of known composite laws.
No new freezing restrictions or automatic clump mass-property requirements were
introduced.

## F. Checked orchestration

`System.step_checked` adds opt-in device checks and early stopping. It returns the
last accepted state/system, accepted step count, and persistent status bits.
Rejected steps roll back time, keys, history, and pending loads along with the
particle state. The ordinary stepping path does not pay for these reductions.
Retries are explicit; the library does not silently resize, change timesteps, or
discard history.

Compression and jamming distinguish search termination from minimizer failure.
Tuple-compatible result objects preserve existing unpacking while exposing
status. Failed relaxation, NaN pressure/energy, and exhausted budgets are not
reported as a successful packing. Existing writer exceptions and host validation
complete the checked workflow.

## Validation

Focused regression coverage includes body-coordinate gradients and heterogeneous
topology, custom-domain compatibility, initialization and read-only history,
cache rebuilds, actual neighbor-capacity overflow, rollback and batches of one,
serialization migrations, custom callbacks, nonempty-history continuation, and
compression/jamming failure counterexamples. See the final validation record
below for executed checks. Production timings and the remote full release matrix
remain deferred; this pass makes no new throughput claim.

Executed locally on 2026-09-11 (groups overlap):

| Configuration | Checks | Result |
| --- | --- | --- |
| CPU float64, JAX 0.11.0 | Checked stepping, body/minimizer, geometry, initialization | 62 passed |
| CPU float32, JAX 0.11.0 | Checked stepping, evaluation/history, HDF5/schema, Cundall–Strack, dynamics, neighbor lifecycle | 87 passed; 3 checkpoint cases handled separately |
| CPU float64, JAX 0.11.0 | Final compression/jamming failure and rollback counterexamples | 16 passed |
| CUDA float32, JAX 0.11.0 | Checked stepping, read-only evaluation, Cundall–Strack history | 27 passed; checkpoint case handled separately |
| CPU float32, Python 3.12/JAX 0.8.1 | Checked stepping, body/minimizer contracts, read-only evolution | 34 passed |
| CPU float32, current schema | Final schema/HDF5/Orbax migrations and history continuation, including precision guard | 17 passed |
| Static checks | Mypy core; strict Sphinx build; scoped Ruff and Black; diff whitespace | Passed; mypy checked 94 files |

A broader Ruff scan also reported three existing diagnostics in untouched
`rl/models/__init__.py` and `utils/random_sphere_configuration.py`. Those files
were not changed for A–F. The focused checks on the architectural changes pass.
