# Boundary and spatial-search contracts

Spatial-search and force kernels operate on one snapshot, with positions shaped
`(N, dim)`. Use `jax.vmap` with matching stacked states and systems for independent
snapshots. `System.step` provides this wrapper for `(B, N, dim)`, including `B=1`.
Trajectory axes are storage axes; explicitly map or reduce them when evaluating
snapshot helpers.

## Geometry and interaction reach

`Domain.search_geometry` declares `SearchGeometry.ORTHOGONAL` or
`SearchGeometry.SHEAR_PERIODIC` for hashed searches. The default `None` means a
custom domain has not opted into those search assumptions. CellList and
MultiCellList validate the declaration; NeighborList delegates compatibility to
its secondary collider. Naive only uses the displacement contract and accepts
arbitrary domain metrics. `System.create` and host `System.validate` perform the
compatibility check.

Custom domains opting in must implement consistent displacement and image rules.
`shear_search_parameters()` supplies the strain and shear axes for shear-periodic
hashing. `search_geometry_snapshot()` records values that can change membership
without particle motion; override it when a custom metric has such parameters.
A class declaration alone cannot make an incompatible stencil correct.

Grid searches call `system.domain.update_bounds(pos, system, padding)` for the
actual query points. The base method returns the system unchanged; dynamic domains
override it. FreeDomain uses the same method in `apply`. Search never invokes
reflective boundary actions.
The existing `shift` method and internal point helper `_shift` share image rules:
search wraps member positions; output wrapping acts on shared body centers, so
wrapping a clump preserves its member offsets. Periodic displacement reads current
box lengths directly.

Each finite-range force law defines `search_radii(state, system)`. Nonzero pair
interactions must lie within the sum of these two bounds. Contact laws use physical
radii, Lennard–Jones uses `cutoff_ratio*rad` (default `2.5`), and WCA uses
`2**(1/6)*rad`. Contact-facet laws
also calculate current geometric extents. Compositions take conservative maxima.
Custom laws must declare this hook for spatial colliders; naive all-pairs force
evaluation supports laws without finite bounds.

Configure LJ's physical cutoff through the force law:

```python
system = jd.System.create(
    state=state,
    mat_table=lj_material_table,
    force_model_type="lennardjones",
    force_model_kw={"cutoff_ratio": 4.0},
    collider_type="NeighborList",
)
```

This sets `r_cut[i,j] = 4 * (rad[i] + rad[j])` for force, shifted energy,
and search. It works inside combiners and routers and is saved in checkpoints.
A collider's requested cell size or cache cutoff cannot truncate the law's reach.
WCA retains its physical minimum-potential cutoff. Custom finite-range laws expose
their own parameters through `search_radii`; use that hook when adding a longer
interaction range.

CellList, MultiCellList, NeighborList, and Hessian analysis use these bounds.
Physical `rad` and geometric `_rad` are not interchangeable.

Lees–Edwards grid searches include fractional shear-image offsets. MultiCellList
currently disables its bounding-box pruning under shear, while retaining grid
traversal. The supported unique-image interaction regime has reach smaller than
half each periodic box length. Queries return particle indices, not multiple
copies of periodic images.

## Contact analysis

Contact diagnostics use the collider and force model configured on the state and
system. Pressure, stress, and per-sphere contact counts support JIT compilation
and reduce pair contributions directly.

For several analyses of the same configuration, collect one contact snapshot:

```python
from jaxdem.utils import get_contacts, get_group_contacts, get_clump_rattler_ids

state, system, contacts = get_contacts(state, system)
state, system, groups = get_group_contacts(state, system, contacts=contacts)
state, system, rattlers, non_rattlers = get_clump_rattler_ids(
    state, system, contacts=contacts
)
```

`ContactData` contains directed sphere-pair IDs, forces, torques about the source
clump COM, and domain-aware displacements. `GroupContactData` contains directed
group-pair IDs, total forces and moments, centroid displacements, friction
ratios, and constituent contact counts.
Both collections contain active interactions without padding; collecting them is
a host operation with variable-length output.

For a system using `CundallStrackForce` directly, the force law can also report
its resolved contact components:

```python
from jaxdem.forces import get_cundall_strack_contacts
from jaxdem.utils import get_sphere_rattler_ids

state, system, cs_contacts = get_cundall_strack_contacts(state, system)
contacts = cs_contacts.as_contact_data(state, system)
state, system, rattlers, non_rattlers = get_sphere_rattler_ids(
    state, system, zc=2, contacts=contacts
)
```

`CundallStrackContactData` contains directed sphere-pair IDs, normal and
tangential force vectors, force-law torques about the source sphere center,
mixed pair friction coefficients, and a `mobilized` mask. A contact is
mobilized when its tangential trial force is on or beyond its Coulomb limit.
Its `as_contact_data` method combines the force components, shifts torques to
the source clump COM, and computes domain-aware displacements without another
force evaluation. The resulting generic `ContactData` can be reused for stress,
pressure, coordination, grouping, and rattler analysis. Here `zc=2` is an
explicit 2D frictional-sphere convention rather than the frictionless default.

Diagnostics preserve contact history. Use the returned system to retain any
refreshed collider cache. A snapshot can be reused only while the state and
force-model inputs, including contact history, are unchanged. Collect new contacts
after removing particles.

## Energy Hessians

Hessian analysis uses the configured collider and force model without separate
search arguments:

```python
from jaxdem.utils import non_bonded_hessian, clump_non_bonded_hessian

state, system, H = non_bonded_hessian(state, system)
state, system, H_clump = clump_non_bonded_hessian(state, system)
```

Sphere coordinates are flattened positions. Clump coordinates contain each
clump's translation followed by its infinitesimal rotation. The optional
`rotation_scale` supplies one positive length per clump to express rotations as
`R*omega`.

Candidate collection is a host operation. It shares contact analysis's search
and exclusion rules, but includes zero-force pairs because their energy can have
nonzero curvature. No contact forces are evaluated and history is not advanced.
Use the returned system to reuse any refreshed cache. Search overflow raises
`ValueError` before differentiation.

`pair_non_bonded_hessian(state, system)` returns `(state, system, pair_ids, blocks)`.
IDs are unique sorted pairs with `i < j`, without padding. Each block includes
the collider's directed interaction masks and half-sum energy convention; for
reciprocal interactions it is one pair-potential Hessian.

Differentiation and assembly run in bounded compiled batches. The assembled
matrix is dense, so its storage grows quadratically with the number of degrees
of freedom. These functions differentiate the force law's potential energy;
dissipative and history-dependent force contributions are not a potential-energy
Hessian. Add `bonded_hessian(state, system)`'s matrix to the sphere Hessian when
bonded energy is present.

## Queries, cache changes, and capacity

`create_neighbor_list` honors its explicit cutoff and `max_neighbors`, returning
exactly that row width and signed `-1` padding. Zero width returns an empty result
and reports overflow when qualifying neighbors exist. Always inspect its returned
overflow flag. NeighborList delegates these arbitrary queries to its secondary
collider; they do not replace its physical force cache or contact history.

An explicit NeighborList capacity bypasses estimation and is preserved exactly. Omitted
capacities are estimates and can overflow in dense or overlapping configurations.
The force cache shares `N * max_neighbors` slots: individual particles may exceed
`max_neighbors`. An exactly full pool is valid; capacity overflow occurs only when
the total directed-pair count exceeds the pool. Each batched simulation has its own pool.
Hashes use `uint64` with JAX x64 enabled and `uint32` otherwise; particle indices
remain signed. Invalid hash geometry is reported as overflow.

`Collider.stateful` is a class property describing cached search state.
`history_pair_shape(state_shape)` is the leading shape of a collider's pair
memory, or `None` for colliders without any. The naive
collider stores it densely as `history[i, j]` with shape `(N, N, H)`, so every
pair keeps its slot for the life of the collider and small frictional systems
need no neighbor list. `NeighborList` stores it in its pooled cache and remaps
it by pair on every rebuild. Cell lists have no pair memory and only run
stateless laws. `System.create` allocates the buffer from the collider's
`history_pair_shape` and the law's `history_shape`.

Every force law has the same signature:
`force(i, j, pos, state, system, history, advance_history=True, quasistatic=False)`
returns `(force, torque, history)` and `energy(i, j, pos, state, system, history)`
returns a scalar. `history_shape(dim)` is the trailing shape of the pair memory;
stateless laws return `(0,)`, return their input history unchanged, and ignore
`advance_history`. `init_history(pair_shape, dim)` allocates the history array,
including zero-width arrays for stateless laws. Combiners and routers
concatenate the component history vectors and their initializers. Read-only
evaluation passes `advance_history=False`; a law with memory must then preserve
it. `quasistatic=True` means no time passes: the law switches off every
velocity-dependent force, and its memory update may depend on velocity only
through `velocity * dt`.

Positions can change without time passing: `Domain.shear`, `Domain.resize`,
and each accepted relaxation step. `jaxdem.colliders.advance_history(state_old,
system_old, state_new, system_new)` is the one owner of that update. It
expresses the change as velocities over `dt`, including the domain's image
rates (`gamma_dot`, `box_rate`) through `Domain.rates_between`, and evaluates
the force model once with `advance_history=True, quasistatic=True`, so `dt`
cancels exactly and no dashpot acts. Dynamics does not use it: the integrator's
velocities advance history inside `compute_force`. Editing `box_size` or
`gamma` by hand bypasses this update; use `resize` and `shear`.

Force NeighborList caches rebuild when the skin budget is exhausted. Particle
motion since the last build counts twice, a strain change counts
`|dgamma| * L_beta`, and a box change counts `max |dL|`.
Changes to the search axes, the required force reach, or the skin itself always
rebuild. Small affine steps therefore do not rebuild, and continuous shear
rebuilds only as often as particle motion would. Call `invalidate_collider`
after changing exclusions or other topology with unchanged indices. Use
`refresh_collider(state, collider, force_model)` after changing allocation
sizes. Growing the neighbor capacity preserves pair memory. Shrinking a
nonempty history buffer, changing the particle count with nonempty history, or
reindexing requires an explicit history reset (`reset_history=True`); history
cannot be matched by old array indices after particle identities move.

`remove_rattlers` maps surviving particle pairs back to their original indices
and transfers their complete contact history into the resized cache before
evaluating forces. This includes tangential displacement and previous contact
normals for Cundall–Strack contacts. Newly discovered pairs use the force law's
history initializer; removal and force evaluation do not advance pair history.

## Preparation, evaluation, and failure handling

Call `state, system = System.initialize(state, system)` after construction and
before the first dynamics step. It updates dynamic bounds and collider caches,
evaluates pair and managed forces, then calls both integrators' `initialize`
hooks. These can apply a velocity staggering correction; the built-in hooks are
no-ops. Initialization does not advance time, timestep callbacks, boundary
impulses, or physical history. Managed one-shot loads queued before initialization
are applied and consumed by that force evaluation.

`System.step` performs no automatic initialization and stores no initialization
flag. An initialized checkpoint continues without calling `initialize` again.
This is an integrator startup operation, not a general force-refresh query:
repeating it can repeat an integrator's half-kick. When starting a new trajectory
after edits, supply initial velocities in the convention expected by that
integrator before initializing.

`System.evaluate_forces` refreshes instantaneous forces and search caches without
advancing history, consuming queued loads, or initializing an integrator.

Minimization evaluates rate-independent forces (`quasistatic=True`) and
holds load buffers fixed. The default energy minimizer supports laws declaring
`supports_analytical_energy_gradient=True`. The base capability is true: laws
whose rate-independent force/torque is not the negative energy gradient must
override it to false. Combined and routed laws propagate any component opt-out.
For unsupported laws, provide an explicit `target_fn`. Custom managed forces
must supply their matching energy function. Minimization uses independent
logical body coordinates and reports its termination reason explicitly.

Force models with pair history are relaxed by FIRE and damped Newtonian only.
The initial evaluation holds history fixed. Each accepted update then advances
history from the anchor to the trial configuration through `advance_history`
and uses that traversal's forces as the residuals. `CundallStrackForce` reports
normal plus tangential spring energy; dissipated work is not part of the energy.

Successful minimization requires both the maximum free-body force norm to be at
most `force_tol` (default `1e-12`) and the maximum torque norm to be at most
`torque_tol` (defaulting to the numerical value of `force_tol`). These are
Euclidean norms per logical body; fixed bodies are excluded. Energy magnitude
and energy changes do not terminate relaxation. Minimization accepts
`force_tol` and `torque_tol` as convergence arguments. For custom targets, the
residuals use the target's translational and rotational gradients.

FIRE and damped Newtonian evaluate physical energy once at exit. Line-search
optimizers evaluate the objective during optimization. `MinimizationResult`
unpacks into `(state, system, steps, energy)` and exposes `.info` with
force/torque residuals, finiteness, and convergence. `.reason` distinguishes
convergence from step exhaustion, nonfinite values and search overflow.

Jamming classifies energy or pressure only after mechanical convergence. Its
`pe_tol` specifies an energy classification threshold. `JamResult` supports
six-value unpacking and exposes `.reason`, `.minimizer_reason`, and `.info`.
`.steps` counts attempted trial minimizations, including the initial and failed
trials.
`.max_minimization_steps` records the maximum work of any trial. Passing
`return_info=True` also returns `(result, result.info)`.

Bisection returns its stored lower/upper states. The pressure and energy-band
drivers return the last evaluated trial on failure, along with the last
below-band state in the unjammed fields. Failed searches report NaN packing
fraction and energy; inspect their reason before accepting a packing. Accepted
states are returned directly without another minimization. Pressure bracket
exhaustion outside the target band is a failure, not target acceptance.

`system.search_overflow` records whether a force evaluation used incomplete
search results. Check `system.check_overflow()` outside compiled code after each
completed simulation chunk. This synchronizes only at the host boundary. A later
successful search does not repair earlier steps: increase capacity and restart
from a valid snapshot saved before failure.

## Estimating neighbor capacity

`estimate_neighbor_capacity` sizes a periodic sphere or rigid-clump system at a
specified **bounding-sphere packing fraction**. It rescales a separate sphere
analogue, relaxes it with FIRE, restores the clumps, and measures candidates using
the force model's search radii, configured cutoff, skin, and interaction
exclusions. The input configuration and contact history are unchanged.

```python
from jaxdem.utils import estimate_neighbor_capacity

sizing = estimate_neighbor_capacity(state, system, packing_fraction=0.84)
collider = jd.Collider.create(
    "NeighborList",
    state=state,
    cutoff=sizing.cutoff,
    skin=sizing.skin,
    max_neighbors=sizing.max_neighbors,
)
```

The default relaxation budget is 1,000,000 steps, with early convergence at a
maximum force norm of `1e-8`. The analogue uses unit maximum bounding radius,
mass, and spring stiffness. Relaxation runs once; three orientation samples reuse
the relaxed centers. Sphere-only states need one sample. Failure to converge
raises an error instead of producing a capacity recommendation.

`sizing.max_neighbors` is the average budget for a pool of
`state.N * max_neighbors` directed pairs. `sizing.query_max_neighbors` is the
per-row capacity for dense queries at the measured list cutoff. Both include a
20% multiplicative or two-neighbor additive margin, whichever is larger, and
round upward to multiples of eight. These settings are configurable. The sample
statistics, bounding radii, probe box, and relaxation diagnostics are returned.

For an existing configuration, `measure_neighbor_candidates(state, system)`
performs only the count. Neither helper allocates a trial pair buffer or evaluates
clump forces. Counts are exact for the sampled configurations; later motion can
require more capacity, so keep the production overflow checks. Direct CellList
and MultiCellList force traversal does not need this pooled capacity setting.

## Construction and restart

Low-level `State.add_clump` accepts placeholder volume and inertia defaults.
Callers may then use `compute_clump_properties` to assign geometry-derived body
properties; choosing when to do so is their responsibility. Final rigid-body
volume and principal inertia are replicated across members. Geometry-aware
facet/mesh builders calculate these quantities directly.
Contact facets cannot reuse a vertex already assigned to another contact facet;
shared vertices in deformable bonded meshes remain supported.

Checkpoints record component configuration, force-law parameters, and boundary
callbacks. Empty history arrays are reconstructed on load; nonempty pair memory
is preserved.
Callbacks and custom force/energy functions must be importable module-level
functions. Saving rejects paths that cannot resolve back to the same callable.
Strict restoration rejects missing physics. Load checkpoints only from trusted
sources: importing saved callables executes their Python modules.
