# Boundary conditions and colliders: implementation record

Status: implemented in the uncommitted working tree, including the cache/history
repairs from the [completed-fix review](COMPLETED_FIXES_REVIEW.md). This record incorporates the
feedback on the first implementation. See the [audit checklist](README.md) and
[public search contracts](../docs/source/user_guide/search_contracts.md) for usage.
The [performance report](BOUNDARY_PERFORMANCE.md) separates historical timings
from a dated recheck that predates the final explicit-initialization revision.

## Geometry

All grid searches call `Domain.update_bounds(points, system, padding)`. Its default
is a no-op; FreeDomain overrides it to update its bounding box from actual
query/database points and padding. FreeDomain.apply uses that same method. Search
never calls reflective boundary physics. The former `prepare_search` API and the
collider-side free-domain type check have been removed.

The existing `shift` operation delegates its image calculation to `_shift`.
Grid hashing uses that same calculation for member/query positions; output
wrapping applies it to body centers, preserving clump offsets. There is no second
public wrapping method. Periodic and Lees–Edwards displacement use current box
lengths directly, so stale inverse-box fields do not alter interaction geometry.

Lees–Edwards retains the orthogonal grid and adjusts boundary-crossing stencil
cells for fractional shear offsets. The stencil must locate the database's
canonical cell: its flow-axis offset has the opposite sign to the physical shear
image offset. Tests caught and corrected a sign error here. Repeated cell hashes
are deduplicated in small boxes. MultiCellList conservatively disables its AABB
pruning under shear while retaining grid traversal.

The supported interaction regime has reach smaller than half each periodic box
length. Queries return particle indices, not all periodic-image copies. The
independent validation enumerates physical images in 2D/3D, with swapped axes,
positive/negative shear, and unwrapped coordinates; collider agreement is also
checked for forces, energy, and self/cross queries.

## Force reach and history

Each finite-range force law declares `search_radii(state, system)`: a nonzero pair
interaction must lie within the sum of the two radii. Contact laws use particle
radii, facet laws include current geometry, and compositions take conservative
maxima. This is an O(N) bounds calculation. Physical `rad` and geometric `_rad`
retain their distinct meanings.

Lennard–Jones exposes `cutoff_ratio` (default 2.5), measured in pair sigma
`rad[i] + rad[j]`. The same parameter controls the force mask, shifted energy, and
collider reach. Configure it with `force_model_kw={"cutoff_ratio": 4.0}`. Increasing
it works through CellList, MultiCellList, NeighborList, combiners, routers, batching,
and restart. Cell size or an undersized cache cutoff cannot silently truncate
physical interactions. WCA retains its model-defined cutoff. Explicit arbitrary
neighbor queries continue to use their requested cutoff.

Every pair-force call accepts history and returns `(force, torque, history)`.
`history_shape(dim)` declares a flat trailing vector; stateless laws use `(0,)`.
`init_history(pair_shape, dim)` always returns an array. Combiners and routers
concatenate component histories and initializers. New pairs receive the law's
initial history; surviving pairs retain their stored values. `advance_history=False` provides read-only force
evaluation without a separate force API. Composite law configuration is pytree
data, allowing configurable law parameters to participate in JIT and batching.

Collider capabilities are class properties: `stateful` describes persistent cache
state; `supports_history` identifies pair-memory support. CellList/MultiCellList
cache search data, while NeighborList also persists contact memory. Nonempty
history requires that capability at construction. No name list determines these
properties.

## Neighbor cache, capacity, and hashes

Cache rebuilds install indices and remapped history together. Energy evaluation
can remap a cache without evolving physical memory. Displacement, skin changes,
periodic box/shear/axis changes, and force-reach changes trigger rebuilding; topology/index changes use explicit
invalidation or refresh. Growing neighbor capacity preserves surviving pairs and initializes new slots;
shrinking nonempty history, changing particle count with nonempty history, or
reindexing requires an explicit history reset.

NeighborList delegates arbitrary queries to its secondary collider without
replacing its physical cache. Explicit capacity bypasses estimation and is preserved exactly, including
zero-width results with truthful overflow. Omitted capacities are estimates.

Hashes are `uint64` with JAX x64 enabled and `uint32` otherwise; particle indices
remain signed for their `-1` sentinel. Bounds are derived once from the native bit
width. Float-to-integer validation and pre-multiplication validation are distinct:
the former prevents an invalid cast, the latter prevents a product of individually
valid dimensions from wrapping. The guards use Python constants and JAX arrays,
with no NumPy dependency in the search/force/domain implementation.

`System.search_overflow` remains sticky after incomplete force evaluations.
`check_overflow()` synchronizes at a host chunk boundary. Resizing after a failure
requires retrying from a valid pre-failure snapshot; later successful searches
cannot repair the trajectory.

## Dynamics and construction

`System.initialize(state, system)` is the explicit startup operation. It updates
dynamic bounds and search caches, computes pair and managed forces, then calls
both configured integrators' `initialize` hooks. A staggered integrator can perform
its velocity half-step adjustment there; the current built-ins use the no-op
hook. No physical timestep, reflective impulse, protocol callback, or history
advance occurs during initialization.

The user calls this method before starting a trajectory. `System.step` does no
automatic preparation, stores no initialization flag, and simply advances its
input state. Initialized checkpoints resume without repeating the hooks. Managed
loads use ordinary `ForceManager.apply` semantics, including consuming one-shot
buffers; the previous temporary-manager initialization path has been removed.

Minimizer evaluation remains separate: it computes conservative forces with
frozen history and queued loads, and never invokes integrator initialization.

For a clump contact, `_pos_p_rot` points from the body COM to the member sphere
center, and `r_ci` points from that center to the contact point. Their sum is the
COM-to-contact arm. Rotational contact velocity therefore uses
`omega × (_pos_p_rot + r_ci)`; omitting `r_ci` would omit surface rotation, even
for a single sphere. The kinematic regression preserves this correction.

Low-level `add_clump` deliberately accepts placeholder volume and inertia defaults.
Users can subsequently call `compute_clump_properties`; the new mandatory,
finite-value, and identical-value guards have been removed. This reclassifies the
low-level omissions in audit items 11–12 as intended construction usage. Final
body properties are replicated across members. Geometry-aware facet/mesh builders
retain the corrected total-volume replication. The unsupported shared-contact-
vertex guard is separate from valid shared-node deformable bonded meshes.

Checkpoint restoration passes the complete reconstructed force law into System
construction, preserving configurable physical parameters and nested laws. Boundary
protocol callbacks and static axes remain covered by the prior restart changes.

The analytical minimizer checks a force-law capability property rather than a
list of implementation names. Custom laws can declare the same contract;
combiners and routers require support from every component. An explicit target
function remains available for other objectives.

## Remaining release work

This work covers the related findings in 02–05, 07, 09–17, 23, 43–44, and the
boundary/search portions of 20, 22, 24, 27, 34–38, 41–42. It does not close the
independent quaternion mutation policy, thermostat stage, minimizer termination
semantics, full documentation cleanup, or release device/version matrix.

Changing shear conservatively rebuilds NeighborList caches every step. This is a
known performance cost. Reusing them under changing geometry requires a proven
skin-displacement bound; no speculative reuse or quadratic fallback is introduced.
Production benchmark regeneration remains release work.
