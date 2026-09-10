# Boundary conditions and colliders: proposed repair plan

Status: proposal only. The separately authorized plasticity fix (01) is implemented and tested; the boundary and collider changes below have not been made.

The simplest coherent solution is to keep the existing colliders and their tuned array layouts, and give them three shared contracts: **domain geometry, conservative interaction bounds, and atomic cache updates**. Capacity checks, checkpoint metadata, and a few contact-representation fixes complete the work. A new universal collider or a general simulation framework is unnecessary.

The domain family covered here is free, periodic, Lees–Edwards, general reflective, and sphere-only reflective. The collider family is naive, CellList, MultiCellList, and NeighborList, including its secondary builder and cross-query paths. Sphere-only reflection remains restricted to spheres; this proposal preserves that intended specialization.

**Findings gathered from the audit**

| Area | Audit items | What must change |
| --- | --- | --- |
| Sheared periodic geometry | 02, 44 | Search the same periodic images as the force law; wrap one center per rigid body. |
| Interaction reach and current search geometry | 03, 04; geometry-cache portion of 20 | Supply force-dependent bounds; prepare free-space bounds for the actual configuration being queried; update derived inverse-box data consistently. |
| Neighbor cache and contact memory | 05, 15, 16, 17 | Invalidate on relevant changes; update indices and history together; allocate history with static shapes. |
| Query semantics, capacity, integer range | 23, 43; architectural item B | Honor requested cutoff and output width, report every truncation, and check hash limits before unsafe integer arithmetic. |
| Contact geometry and construction | 07, 10, 11, 12 | Use COM-to-contact velocity; reject unsupported shared contact-facet vertices; require coherent clump mass properties. These are distinct from DP bonded-mesh connectivity. |
| Evolution versus evaluation | 09, 22; related to 01 | Search/force inspection must not apply boundary impulses, advance plasticity/contact history, or consume one-shot loads. Initial acceleration needs its own preparation contract. |
| Boundary protocols and restart configuration | 13, 14; strict-restore portion of 27 | Restore callbacks and static component settings, including Lees–Edwards axes, as well as array leaves. |
| Validation and boundary-changing workflows | relevant portions of 24, 41, 42 | Define snapshot/batch shapes, validate geometry, and use explicit callback integer dtypes in packing/jamming. |
| Evidence and documentation | relevant portions of 34–38 | Repair fixtures and query documentation; require correctness before comparing collider performance. |

The table gathers related work without claiming that a search refactor fixes every part of each item. For example, 22 also includes minimizer convergence semantics, and 20 includes quaternion-cache mutation. Item39's host-side temporal/bin-pair enumeration is a separate analysis scaling issue, not the spatial neighbor-search cache.

**1. Domain owns image rules; search preparation does not apply physics**

The current distinction between `periodic=True` and `False` is insufficient: [CellList hashing](../jaxdem/colliders/cell_list.py) treats both ordinary periodic and Lees–Edwards domains as an orthogonal grid. Also, [FreeDomain.apply](../jaxdem/domains/free.py) prepares a search box, whereas reflective `apply` methods change positions and velocities.

Introduce one pure search-preparation hook, with a no-op default for fixed domains. It prepares the geometry used by the current query, including a free-space bounding box and consistent inverse-box values. Force, energy, minimizer trial, and explicit neighbor-query entry points use it; internal prepared kernels avoid repeating that work within one evaluation. Cross queries prepare bounds for their actual query/database point sets. In dynamics this preparation occurs after drift, so it describes the positions actually searched.

Keep physical reflection in the timestep's boundary stage. Do not fix free-domain minimization by calling arbitrary `domain.apply` inside its objective: that would apply reflective impulses during trial evaluations. This separation need not change the current reflection integrator ordering.

Add a domain-owned point-wrapping helper. Search uses wrapped **member positions**; state/output wrapping applies the same image rule to **body centers** (`pos_c`), preserving shared clump centers and offsets. DP nodes have independent centers and still wrap independently. Displacement and wrapping must use the same shear axes and offset; redundant one-hot axis arrays should be derived from canonical axis settings rather than become competing sources of truth.

For Lees–Edwards search, retain the orthogonal cell database but make boundary-crossing cell queries image-aware. A query crossing the gradient boundary shifts along the flow axis by the matching shear image offset. Fractional-cell shifts require visiting every intersected cell, not rounding to one column. Deduplicate repeated cell hashes in small boxes. The shared helper must cover self queries and cross queries; MultiCellList bounding boxes and pruning must use the same canonical images. Do not assume fixing hashes alone proves its box pruning conservative.

This preserves the ordinary periodic/free implementations as specialized paths. I would first guard unsupported Lees–Edwards/collider combinations, then remove each guard only when that entire path passes validation. A guard must depend on the domain type/capability, not just whether its current `gamma` happens to be zero. No silent switch to a quadratic collider.

I also considered hashing all particles in unsheared coordinates. That requires transformed conservative bounds and can widen searches throughout the box. Image-aware boundary queries fit the current orthogonal grids more directly; a general lattice representation is not required to fix the present supported domains.

**2. Force models supply conservative search bounds**

Add a force-model hook returning a conservative per-search-primitive radius `q_i`, with the contract that nonzero nonbonded interactions imply a center separation no greater than `q_i + q_j`. For simple spheres, contact laws use `rad`, LJ uses `2.5*rad`, and WCA uses `2**(1/6)*rad`. Facet bounds additionally account for the geometric extent from the vertex used as the search key to the possible contact location. Flexible facets need refreshed bounds or a validated maximum extent; construction-time radii cannot silently become stale after deformation.

Combiners and routers take conservative bounds over their possible laws. Custom laws declare a bound or use an explicitly supported unbounded/all-pairs path. Computing these bounds is an O(N) preparation operation, not an N-by-N pair calculation. Keep physical contact radii separate from search radii.

Use those bounds everywhere:

- CellList stencil coverage and effective cell size must cover the maximum required reach, including user-specified cell sizes.
- MultiCellList keeps per-primitive bounds for its boxes, preserving its advantage for unequal sizes.
- A force NeighborList covers at least the required reach plus its skin.
- Hessian helpers use the same physical reach by default; an explicit smaller cutoff is a documented approximation or an error, rather than an accidentally incomplete default.

Sensor/alignment queries still use their explicit geometric cutoff. They do not inherit the force law's cutoff. Long bonded interactions remain in the bonded model and do not enlarge the nonbonded contact search unnecessarily.

**3. Make neighbor-cache replacement one operation**

Refactor [_check_and_rebuild](../jaxdem/colliders/neighbor_list.py) to return an updated collider, rather than a tuple whose history field each caller must remember to store. Every force, energy, and cached-list entry point uses that same result. A rebuilt list and its remapped history are installed atomically. Energy evaluation can rearrange a cache while preserving history associated with each pair; it must not advance physical history.

The validity check needs:

- The existing displacement/skin check.
- A small snapshot of the **physical periodic metric**, including box lengths and shear offset. A changed free-space search bounding box is not a changed physical metric and should not force a cached geometric pair list to rebuild.
- Explicit invalidation for changes to exclusions/topology, indexing, force reach, or geometry bounds. Route supported edits through one invalidate/refresh operation; direct edits that bypass it need a documented re-prepare requirement. Do not claim arbitrary nested mutation can be detected for free.

Initially rebuild on any physical metric change. This is the simplest safe rule, but continuous shear or box deformation can then rebuild every step. That is a real performance cost, not a constant-time optimization. Measure it. Reuse under changing shear should be enabled only after a conservative bound on the additional image displacement has been derived and tested against the remaining skin; an arbitrary strain tolerance is insufficient.

`refresh_collider` must receive the force model (directly or through `System`) and initialize correctly shaped history before entering a compiled loop. Preserve history for surviving pairs during a same-indexing resize/rebuild. Make a destructive history reset explicit. Reindexing requires an identity mapping or a clear reset/error contract. Remove the ineffective history placeholder. For issue17, make the history allocation shape static or leave this construction-time helper unjitted; no new history framework is needed.

The plasticity fix establishes an exactly-once timestep stage for bonded evolution. Contact-memory integration needs the analogous distinction between a dynamics update and a read-only force/objective query; cache remapping is not physical integration. A first implementation can make this a static evaluation mode, with no timestep-based guessing or hidden `dt=0` workaround.

**4. Separate arbitrary neighbor queries from the force cache**

`NeighborList.create_neighbor_list` currently documents that it ignores its requested cutoff and capacity, although callers such as Hessian and alignment helpers pass those arguments expecting them to matter. Preserve the public query signature but make it fulfill the requested radius, interaction-mask semantics, and `(N, max_neighbors)` shape. The simplest initial implementation delegates an arbitrary query to the secondary collider without replacing the physical force cache or its history. Internal force evaluation uses a separately named cached-list helper. Optimize safe cache reuse later if needed; do not return a skin-expanded list as an exact-radius query result without filtering it.

Explicit capacity is an allocation request: preserve its output width, pad as needed, and do not clamp it using a non-overlap packing assumption. Run density/packing estimates only when the user omits capacity; label them estimates. Handle zero particle radii without dividing by them in a constructor heuristic. Reject invalid capacities/cutoffs clearly. For capacity zero, either reject it consistently or return the empty shape with overflow determined from whether any qualifying neighbors exist. I favor preserving the existing empty-shape API with truthful overflow.

For issue43, select the actual JAX integer dtype explicitly, derive its limit from that dtype, and check dimensions/products **before** casting or multiplying integers. Use guarded integer multiplication (dimension-by-dimension division against the remaining limit) after validating safe dimensions; guard the float-to-integer boundary conservatively in float32. Returning an overflow flag after hashes have already wrapped is too late. Invalid geometry must not enter normal hash traversal.

Keep immediate query overflow separate from a sticky simulation failure status. Later successful searches must not erase evidence that a prior step used truncated interactions. Check accumulated status at host completion/chunk boundaries, preserving asynchronous execution within a chunk. Resizing/retrying must restart from a valid pre-failure state; enlarging a buffer after advancing with missing forces does not repair the trajectory.

**5. Small associated fixes, without a larger rewrite**

- **07:** share COM-to-contact velocity calculation with the full member offset and contact arm. Preserve the existing force law; do not change its constitutive model as part of this correction.
- **10:** for 1.0, reject shared vertices in the contact-facet representation that stores only one incidence row. Continue supporting shared vertices in DP bonded meshes. General contact-facet incidence can be a separate feature.
- **11–12:** make low-level clump construction require validated body mass properties when geometry-derived values are unavailable. Retain the higher-level builders that already supply union volume and inertia. Do not silently approximate an extended body's inertia as one sphere's inertia.
- **13–14, 27:** add complete boundary/component configuration and importable protocol callbacks to checkpoint metadata, validate round-trip resolution, and fail on required physics that cannot be restored. Restore canonical axis/box settings and derive their caches. This is a targeted extension of existing component serialization, not a replacement storage format.
- **09, 22:** use the read-only evaluation path to prepare initial forces and conservative minimizer evaluations. Define handling of one-shot loads and fixed degrees of freedom explicitly. The remaining minimizer convergence/termination work remains its own task.
- **24, 41, 42:** validate finite positive box lengths and supported geometry/force/collider combinations outside hot loops; specify single-snapshot kernels and batching wrappers; use matching explicit `int32` callback shapes/results in packing and jamming.

**Implementation order and acceptance**

I would split this into six reviewable changes: (1) truthful query/capacity/hash behavior and unsupported-combination guards; (2) physical search bounds and pure geometry preparation; (3) atomic cache/history refresh and invalidation; (4) Lees–Edwards image queries and body wrapping; (5) contact construction and restart fixes; (6) workflow/benchmark/documentation validation. The evaluation/history lifecycle should be settled while doing (2–3), alongside the separate 01 fix.

Use one reusable verification matrix rather than unrelated tests for every method. Compare force, energy, self-neighbor, and cross-neighbor results against all-pairs calculations, with identical masks and cutoffs. Independently enumerate periodic images for small analytical boundary cases so a displacement bug shared by all colliders cannot validate itself. Cover free, ordinary periodic, Lees–Edwards, and reflective domains; 2D/3D; both x64 settings; B=1 and B>1; arbitrary axes; unwrapped coordinates; small boxes; fractional-cell shear offsets; unequal sizes; flexible facets; and explicit overflow. Check and document the supported minimum-image regime rather than assuming an arbitrary large cutoff is physically unambiguous.

Add sequences that change shear, box size, topology, radii/bounds, and neighbor ordering while holding other inputs fixed. Check persistence of pair history through energy queries, refresh, resize, and restart. Test each free-domain minimizer trial against its own bounding geometry. Invalid configurations should fail before a result can be mistaken for valid physics.

Repair benchmark topology/population fixtures (37), propagate benchmark failures (38), and run small correctness checks before timing. Record compilation, steady-state stepping, rebuild-heavy deformation, batched workloads, and memory on CPU/GPU. Ordinary periodic performance should remain the baseline; measure the additional shear work separately. Continuous-shear neighbor-cache reuse is the main performance tradeoff requiring evidence. Strict query documentation and automated regression gates cover the relevant parts of 34–36.

This plan closes the shared failure paths while preserving the existing kernels. It deliberately does not claim that serialization, topology, or minimizer semantics become correct merely by adding a geometry helper.
