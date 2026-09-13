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


CellList, MultiCellList, and force NeighborList searches use these bounds. Hessian
helpers use them for their default cutoff; an explicitly smaller Hessian cutoff
is an approximation. Physical `rad` and geometric `_rad` are not interchangeable.

Lees–Edwards grid searches include fractional shear-image offsets. MultiCellList
currently disables its bounding-box pruning under shear, while retaining grid
traversal. The supported unique-image interaction regime has reach smaller than
half each periodic box length. Queries return particle indices, not multiple
copies of periodic images.

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

`Collider.stateful` is a class property describing cached collider state.
`supports_history` identifies colliders that persist pair memory (NeighborList).

All pair-force calls accept and return a history array:
`force(..., history, advance_history=True) -> (force, torque, history)`.
`history_shape(dim)` specifies its trailing shape; stateless laws return `(0,)`.
`init_history(pair_shape, dim)` allocates the array, including empty arrays.
Combiners and routers concatenate the component history vectors and their
initializers. New neighbor pairs receive the law's initial history; surviving
pairs retain their stored history when a list is rebuilt. Stateless laws
return their input history unchanged. Read-only evaluation passes
`advance_history=False`; a custom law must then preserve its physical memory.
Laws with nonempty history require a collider with `supports_history=True`.

Force NeighborList caches rebuild when displacement exhausts the skin, the skin
configuration changes, the periodic metric changes, or the required force reach
changes. `metric_snapshot` stores the box lengths, shear, and shear axes at the
last build: changing any of these can change
neighbor membership even when particle coordinates have not moved. Free domains
use a constant snapshot because their bounding box is only a search envelope. Changing shear can
therefore rebuild every step. Call `invalidate_collider` after changing exclusions
or other topology with unchanged indices. Use `refresh_collider(state, collider,
force_model)` after changing allocation sizes. Growing the neighbor capacity
preserves pair memory. Shrinking a nonempty history buffer, changing the particle
count with nonempty history, or reindexing requires an explicit history reset
(`reset_history=True`); history cannot be matched by old array indices after
particle identities move.

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

Minimization evaluates conservative potential energy and
holds physical history and load buffers fixed. The default energy minimizer supports
laws declaring `supports_analytical_energy_gradient=True`, using zero-velocity
forces to exclude damping. The base capability is true: laws whose zero-velocity
force/torque is not the negative energy gradient must override it to false.
Combined and routed laws propagate any component opt-out. For unsupported laws,
provide an explicit `target_fn`. Custom managed forces must supply their matching energy function. Minimization uses independent logical body
coordinates and reports its termination reason explicitly.

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
