r"""Clumps (Rigid Bodies)
----------------------------------------

A **clump** is a rigid body made of several spheres that move together.
Every sphere in the same clump shares its center-of-mass position
(``pos_c``), orientation (``q``), velocity (``vel``), angular velocity
(``ang_vel``), mass, and inertia. The remaining fields are stored
per-sphere: the body-frame offset (``pos_p``), the radius (``rad``),
the volume (``volume``), and the ID fields (``mat_id``, ``species_id``,
``bond_id``, ``clump_id``) — see the table below.

This guide covers:

- The data model: which fields are shared and which are per-sphere.
- Creating clumps manually and with :py:meth:`~jaxdem.state.State.add_clump`.
- Computing clump mass, inertia, and center of mass with
  :py:func:`~jaxdem.utils.compute_clump_properties`.
- How colliders, force aggregation, and integrators handle clumps.
- Practical tips and common pitfalls.
"""

# %%
# The Clump Data Model
# ~~~~~~~~~~~~~~~~~~~~~~
# Each particle slot in the state stores its *own* copy of the shared
# fields. ``clump_id`` encodes clump membership: every slot with the same
# ``clump_id`` belongs to the same rigid body.
#
# .. list-table::
#    :header-rows: 1
#
#    * - Field
#      - Shared by clump
#      - Per-sphere
#    * - ``pos_c`` (center of mass position)
#      - ✓
#      -
#    * - ``pos_p`` (body-frame offset relative to COM)
#      -
#      - ✓
#    * - ``q`` (quaternion orientation)
#      - ✓
#      -
#    * - ``vel`` (linear velocity of COM)
#      - ✓
#      -
#    * - ``ang_vel`` (angular velocity)
#      - ✓
#      -
#    * - ``force`` (total aggregated linear force)
#      - ✓
#      -
#    * - ``torque`` (total aggregated torque)
#      - ✓
#      -
#    * - ``mass`` (total mass of the clump)
#      - ✓
#      -
#    * - ``inertia`` (total inertia tensor of the clump)
#      - ✓
#      -
#    * - ``rad`` (radius of the sphere)
#      -
#      - ✓
#    * - ``volume`` (volume of the sphere)
#      -
#      - ✓
#    * - ``clump_id`` (clump identifier)
#      -
#      - ✓
#    * - ``bond_id`` (connectivity/connection IDs)
#      -
#      - ✓
#    * - ``mat_id`` (material ID of the sphere)
#      -
#      - ✓
#    * - ``species_id`` (species ID of the sphere)
#      -
#      - ✓
#    * - ``fixed`` (immobility flag)
#      - ✓
#      -
#
# .. note::
#
#    The state stores ``volume`` per sphere: ``State.create`` defaults it to each
#    sphere's own hypersphere volume, while
#    :py:meth:`~jaxdem.state.State.add_clump` broadcasts a provided clump
#    volume to every member sphere. Packing-fraction utilities read one
#    value per clump (via a segment max).
#
# The actual position of each sphere in the lab frame is a **computed
# property**:
#
# .. math::
#
#    \texttt{pos} = \texttt{pos\_c} + R(q)\;\texttt{pos\_p}
#
# For a lone sphere ``pos_p = 0``, so ``pos == pos_c``.

import jax.numpy as jnp
import numpy as np

import jaxdem as jdem

# %%
# Creating a Clump Manually
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The simplest way is to create individual spheres and assign the same
# ``clump_id`` to those that form a rigid body. We also set their shared
# ``pos_c`` (center of mass) and per-sphere ``pos_p`` (offset from the
# center of mass in the body frame).

# Two spheres forming a dumbbell clump, plus one free sphere.
pos_c = jnp.array(
    [
        [2.0, 0.0],  # sphere 0 — dumbbell COM
        [2.0, 0.0],  # sphere 1 — same COM
        [6.0, 0.0],  # sphere 2 — free sphere
    ]
)
pos_p = jnp.array(
    [
        [-0.5, 0.0],  # sphere 0 is 0.5 to the left of COM
        [0.5, 0.0],  # sphere 1 is 0.5 to the right of COM
        [0.0, 0.0],  # sphere 2 — lone sphere, no offset
    ]
)
clump_id = jnp.array([0, 0, 1])  # 0,0 → same clump, 1 → separate clump

state = jdem.State.create(
    pos=pos_c,
    pos_p=pos_p,
    rad=jnp.array([0.6, 0.6, 1.0]),
    clump_id=clump_id,
)
print("clump_id:", state.clump_id)
print("pos_c   :", state.pos_c)
print("pos_p   :", state.pos_p)
print("pos     :", state.pos)

# %%
# ``state.pos`` gives the true lab-frame position of each
# sphere. In the dumbbell (clump 0) the two spheres sit at different
# locations even though they share the same ``pos_c``.


# %%
# Using :py:meth:`~jaxdem.state.State.add_clump`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A more convenient way to append a clump to an existing state is
# :py:meth:`~jaxdem.state.State.add_clump`. It broadcasts shared fields
# (velocity, mass, material, …) to all spheres automatically and assigns
# a single ``clump_id`` to the whole group.

state_base = jdem.State.create(
    pos=jnp.array([[0.0, 0.0]]),
    rad=jnp.array([1.0]),
)
print("Before add_clump: N =", state_base.N, " clump_ids =", state_base.clump_id)

state_with_clump = jdem.State.add_clump(
    state_base,
    pos=jnp.array([[3.0, 0.0], [3.0, 0.0]]),  # COM for each sphere
    pos_p=jnp.array([[-0.4, 0.0], [0.4, 0.0]]),
    rad=jnp.array([0.5, 0.5]),
    vel=jnp.array([1.0, 0.0]),  # broadcast to both spheres
    mass=jnp.array(2.0),  # broadcast to both spheres
)
print(
    "After add_clump:  N =",
    state_with_clump.N,
    " clump_ids =",
    state_with_clump.clump_id,
)
print("Velocities:\n", state_with_clump.vel)


# %%
# Computing Clump Properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# When you define a clump by placing overlapping spheres at arbitrary
# positions, you must compute the correct center of mass, total mass, and
# inertia tensor. This is **not trivial** because spheres may overlap.
# Summing individual volumes would over-count shared regions.
#
# :py:func:`~jaxdem.utils.compute_clump_properties` computes the mass properties
# of the union. In 2D it splits intersecting circles into exposed arcs and
# integrates those arcs analytically. In 3D it combines analytical sphere
# moments with numerical integration where needed: boundary quadrature for
# sparse unions and direct volume quadrature for dense, heavily overlapping
# unions. The dense path avoids undersampling tiny exposed boundary patches.
# Separated and nested spheres have analytical properties in either
# dimension, as do overlapping pairs of spheres in 3D. Analytical and sparse
# boundary paths avoid spending samples in empty bounding-box space.
# For clumps with at most 32 members, pairwise lens corrections are also
# analytical when the overlap graph has no triangles. The shortcut is
# conservative: more complicated overlap patterns use numerical integration.
# The resulting density field determines:
#
# - total mass (accounting for overlap)
# - center of mass (``pos_c``)
# - principal moments of inertia (``inertia``)
# - principal-axes orientation (``q``)
# - body-frame offsets (``pos_p``) relative to the new center of mass
#
# These quantities are moments of the material union, not sums of
# independent overlapping spheres. With union :math:`\Omega` and material
# density :math:`\rho(\mathbf{x})`,
#
# .. math::
#
#    V=\int_\Omega dV,\qquad
#    M=\int_\Omega\rho(\mathbf{x})\,dV,\qquad
#    \mathbf{c}=\frac{1}{M}\int_\Omega\rho(\mathbf{x})\mathbf{x}\,dV,
#
# and the three-dimensional inertia about the COM is
#
# .. math::
#
#    \mathbf{I}=\int_\Omega\rho(\mathbf{x})
#    \left[\lVert\mathbf{x}-\mathbf{c}\rVert^2\mathbf{1}
#    -(\mathbf{x}-\mathbf{c})(\mathbf{x}-\mathbf{c})^T\right]dV.
#
# In 2D, area replaces volume and only
# :math:`I_{zz}=\int_\Omega\rho\lVert\mathbf{x}-\mathbf{c}\rVert^2dA`
# is needed. Diagonalizing the 3D tensor defines principal moments and a
# body-frame orientation. If :math:`R` maps that frame into the world,
# the member offsets become :math:`\mathbf{p}_i=R^T(\mathbf{x}_i-\mathbf{c})`;
# consequently :math:`\mathbf{c}+R\mathbf{p}_i=\mathbf{x}_i`, so changing
# the stored COM and principal frame does not move the physical spheres.
#
# The shared mass and inertia belong to the whole rigid body. Although
# those values are broadcast to its member slots, summing all slots would
# count the same body repeatedly; select one representative per clump when
# computing body-level kinetic energy or total mass.
#
# Where materials overlap, the utility uses the largest material density
# at each point, so neither volume nor overlapping mass is counted twice.
# It preserves the world-space sphere positions while updating the body
# frame. Principal frames have consistent signs and handedness; exactly
# repeated moments use a deterministic frame because their physical
# principal axes are not unique.
#
# .. important::
#
#    ``compute_clump_properties`` requires a
#    :py:class:`~jaxdem.materials.MaterialTable` because the mass
#    computation depends on material density.

mat = jdem.Material.create("elastic", density=2.0, young=1e4, poisson=0.3)
mat_table = jdem.MaterialTable.from_materials([mat])

# Place two overlapping spheres at known positions
clump_state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [0.8, 0.0]]),
    rad=jnp.array([0.5, 0.5]),
    clump_id=jnp.array([0, 0]),
    mat_table=mat_table,
)
print("Before compute_clump_properties:")
print("  pos_c:", clump_state.pos_c)
print("  pos_p:", clump_state.pos_p)
print("  mass :", clump_state.mass)

clump_state = jdem.utils.compute_clump_properties(clump_state, mat_table)
print("\nAfter compute_clump_properties:")
print("  pos_c:", clump_state.pos_c)
print("  pos_p:", clump_state.pos_p)
print("  mass :", clump_state.mass)
print("  inertia:", clump_state.inertia)


# %%
# Solid Member Spheres
# ~~~~~~~~~~~~~~~~~~~~
# Every member disk or sphere is solid. The mass model uses their union:
# material occupies points inside at least one member, and overlaps count
# once. Space outside all supplied members stays empty, including enclosed
# cavities. To leave the center empty, supply a clump without a center sphere.
#
# The geometric-asperity builder ``create_ga_state`` distinguishes:
#
# - ``core_type="hollow"`` (default) generates only the solid asperities,
#   without adding a center sphere. The name refers to the clump composition;
#   each member sphere is still solid.
# - ``core_type="solid"``: a central core contributes both material and
#   contact geometry, in addition to the asperities.
# - ``core_type="phantom"``: the core contributes mass, volume, and inertia,
#   then is removed from the contact geometry. Its interior is still material.
#
# ``compute_clump_properties`` derives mass from the material density table.
# The uniform-union helper used by ``create_ga_state`` instead preserves the
# requested total body mass and spreads it uniformly over the chosen union.
# Both models preserve geometric voids and avoid double-counting overlaps.


# %%
# Accuracy and Construction Cost
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The same call works in both dimensions. ``n_samples`` controls the total
# numerical integration budget per 3D clump; it has no effect on analytical
# 2D arc integration or analytical 3D cases. The numerical budget has a floor
# of 32 points per member; boundary quadrature also rounds up to an even
# number of points per member. Increase it when a more accurate inertia
# tensor is needed; there is no universal sample count that makes every
# shape exact.
#
# Any number of members may overlap. The 2D arc integrals remain analytical;
# general 3D integration evaluates coverage by all other members, including
# triple and higher-order overlaps. The pairwise analytical shortcut is used
# only when its overlap-graph check rules out triple intersections.
#
# Adaptive selection keeps volume integration for dense cases where one
# member alone guarantees useful bounding-box occupancy, and retains
# boundary integration for sparse unions. A positive finite result does
# not certify accuracy. For a 3D body requiring quadrature, compare
# increasing budgets, for example ``50_000``, ``200_000``, and ``800_000``,
# until volume, center of mass, and inertia meet the needed tolerances.
# Compare inertia in world coordinates, ``R(q) @ diag(inertia) @ R(q).T``:
# quaternion signs and axes within an almost repeated eigenspace can change
# even when the physical tensor agrees. Unresolved nonpositive properties
# raise ``ValueError``; increasing samples or precision can resolve them.
#
# ``sample_batch_size`` bounds the temporary sample arrays, while
# ``clump_batch_size`` bounds the number of bodies processed together.
# These settings control memory use without changing the integration rule.
# Samples are generated as needed rather than stored as one large point
# array, and their angular phases retain precision in float32.
#
# The host wrapper groups clumps by their actual member counts. Small
# clumps therefore do not acquire padding to the largest clump in the
# state, and sparse IDs do not require a table indexed by their largest
# value. Single-sphere bodies keep their existing state properties.
# Compute these properties during construction, before running dynamics.
# The grouping step runs on the host, so call the wrapper outside
# ``jax.jit``; its numerical kernels are compiled internally.


# %%
# Collision Detection and Clumps
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# All colliders automatically **skip** interactions between spheres
# that belong to the same clump.
# :py:func:`~jaxdem.colliders.valid_interaction_mask` handles this:
#
# .. code-block:: python
#
#    is_bonded = jnp.any(bond_id_i == idx_j[..., None], axis=-1)
#    mask = (clump_i != clump_j) * (~is_bonded | interact_same_bond_id)
#
# So spheres inside a clump never exert contact forces on each
# other — they are a rigid assembly by construction. Colliders mask
# bonded pairs (connected via ``bond_id``) by default. Bonded pairs
# interact when
# :py:attr:`~jaxdem.system.System.interact_same_bond_id` is ``True``.
# The colliders **always** mask same-clump pairs, regardless of that flag.
#
# There are no special collider requirements: clumps work with all colliders.


# %%
# ``clump_id`` vs ``bond_id``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These two identifiers serve different purposes:
#
# - ``clump_id`` — **rigid body grouping**. Spheres with the same
#   ``clump_id`` are physically fused: they share velocity, position,
#   and orientation and **never** collide with each other.
# - ``bond_id`` — **connectivity masking**. The colliders **disable**
#   (mask out) pairwise interactions between spheres connected by a
#   bond. The ``bond_id`` array holds the unique IDs of the spheres
#   each sphere is connected to. Setting
#   :py:attr:`~jaxdem.system.System.interact_same_bond_id` to ``True``
#   re-enables contact forces between bonded pairs (same-clump pairs stay
#   masked either way).
#
# In short: ``clump_id`` controls rigid-body aggregation *and*
# collision masking, while ``bond_id`` controls localized connection masking.
# See :doc:`../auto_examples/deformable_particle_guide` for details on
# deformable particles.

# %%
# Force Aggregation
# ~~~~~~~~~~~~~~~~~~~
# The :py:class:`~jaxdem.forces.force_manager.ForceManager` handles clump-level force
# aggregation in its ``apply`` step. The pipeline is:
#
# 1. The collider writes per-sphere contact forces/torques into
#    ``state.force`` / ``state.torque``.
# 2. The force manager adds external forces (gravity, custom functions).
# 3. Particle-frame forces induce extra torque via the lever arm
#    :math:`\tau_i = r_i \times F_i`, where :math:`r_i = R(q) \cdot pos\_p_i`.
# 4. The force manager **sums** forces and torques over each clump with
#    ``jax.ops.segment_sum``, using ``clump_id`` as the segment key.
# 5. It **broadcasts** the aggregated values back, so every sphere
#    in the clump sees the same total force and torque.
#
# This makes the clump accelerate and rotate as a single rigid body.


# %%
# Running a Simulation with Clumps
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A short free-fall interval illustrates rigid-body aggregation and the
# fixed-particle contract. The dumbbell stays above the fixed support during
# this interval, so its center of mass follows ``y(t) = y(0) - g*t**2/2``.
# Contact with the support is outside this short tutorial calculation.

# Fixed floor sphere
pos_floor = jnp.array([[0.0, 0.0]])
rad_floor = jnp.array([1.0])

state_sim = jdem.State.create(pos=pos_floor, rad=rad_floor, mat_table=mat_table)
state_sim.fixed = jnp.array([True])

# Add a dumbbell clump above the floor
state_sim = jdem.State.add_clump(
    state_sim,
    pos=jnp.array([[0.0, 4.0], [0.0, 4.0]]),
    pos_p=jnp.array([[-0.4, 0.0], [0.4, 0.0]]),
    rad=jnp.array([0.4, 0.4]),
)
print("N:", state_sim.N, " clump_ids:", state_sim.clump_id)

# Compute clump mass and inertia
state_sim = jdem.utils.compute_clump_properties(state_sim, mat_table)

system_sim = jdem.System.create(
    state_sim.shape,
    dt=1e-3,
    force_model_type="spring",
    mat_table=mat_table,
    force_manager_kw={"gravity": jnp.array([0.0, -9.81])},
)

# Run a short, resolved interval and compare its center of mass with free fall.
state_sim, system_sim = jdem.System.initialize(state_sim, system_sim)
initial_com = np.asarray(state_sim.pos_c[1]).copy()
state_sim, system_sim = system_sim.step(state_sim, system_sim, n=50)
elapsed = float(system_sim.time)
expected_com = initial_com + np.array([0.0, -0.5 * 9.81 * elapsed**2])
np.testing.assert_allclose(state_sim.pos_c[1], expected_com, atol=2e-5)
np.testing.assert_allclose(state_sim.pos[0], pos_floor[0], atol=1e-7)
np.testing.assert_allclose(state_sim.pos_c[1], state_sim.pos_c[2], atol=1e-7)
print("Dumbbell COM:", state_sim.pos_c[1])
print("Floor position (unchanged):  ", state_sim.pos[0])


# %%
# Integration and Clumps
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# The linear and rotational integrators act on ``pos_c``, ``vel``, ``q``,
# and ``ang_vel``. Because all spheres in a clump share these fields
# (with the same values broadcast to every slot), the integrator moves
# the entire clump as one rigid body without any special branching.
#
# ``state.pos`` derives the actual sphere positions from ``pos_c`` and
# ``pos_p`` on each access.


# %%
# Reflective Domains and Clumps
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# When a sphere inside a clump hits a reflective boundary, JaxDEM
# **aggregates** the velocity correction over the whole clump before
# applying it. This stops individual spheres from escaping while the
# correction pulls the rest of the body back, and preserves rigid-body
# integrity.


# %%
# Common Pitfalls
# ~~~~~~~~~~~~~~~~~
# 1. **Forgetting to call ``compute_clump_properties``.**
#    If you build a clump manually and skip this step, the mass, inertia,
#    and center of mass will be those of a single sphere. The simulation
#    will not be physically correct.
#
# 2. **Overlapping clumps with the same** ``clump_id``.
#    If two separate bodies accidentally share a ``clump_id``, JaxDEM
#    treats them as one rigid body. They do not interact through contact
#    forces and they move together.
#
# 3. **Editing** ``clump_id`` **after construction.**
#    :py:meth:`~jaxdem.state.State.create` relabels input groups to contiguous
#    IDs, so inputs such as ``[10, 10, 100, 100]`` become ``[0, 0, 1, 1]``.
#    Preserve that invariant when editing arrays directly: aggregation indexes
#    an array with ``N`` segments, so IDs outside ``[0, N)`` are invalid.
#
# 4. **Shared fields must be identical** across a clump.
#    If you manually set ``vel``, ``pos_c``, ``q``, etc., make sure all
#    slots belonging to the same clump receive the *same* value.
#    :py:meth:`~jaxdem.state.State.add_clump` handles this automatically by broadcasting.
