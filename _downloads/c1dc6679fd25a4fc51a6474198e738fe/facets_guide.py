# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Facets (Triangular & Line Boundaries)
-----------------------------------------

A **facet** is a 2D line segment (2 vertices) or 3D triangle (3 vertices) used
for flat walls, polygonal obstacles, imported meshes, or continuous contact
surfaces on deformable particles. Facets reuse JaxDEM's sphere broad-phase and
rigid-body integration, while narrow-phase contact resolves exact
point--triangle or segment--segment distances.

For an application importing an STL boundary mesh, see
:doc:`../auto_examples/rotating_drum`.
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp

import jaxdem as jdem

# %%
# Creating Single Facets and Triangular Meshes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use :py:meth:`~jaxdem.state.State.add_facet` for one segment/triangle and
# :py:meth:`~jaxdem.state.State.add_mesh` for a full mesh. Assign facets their
# own ``species_id`` (e.g. ``1`` for facets, ``0`` for spheres) so a
# :py:class:`~jaxdem.forces.router.ForceRouter` dispatches sphere--facet and
# facet--facet laws separately.
#
# * ``rigid=True`` groups vertices into one rigid clump (``filled=True`` for a
#   solid closed polyhedron; ``filled=False`` for a hollow shell or open mesh).
# * ``rigid=False`` leaves vertices independent (combine with a bonded model
#   for elasticity).
# * ``thickness`` sets the Minkowski expansion radius stored in ``state.rad``:
#   faces shift outward by ``thickness``, while edges and vertices become
#   smoothly rounded pipes and spheres of radius ``thickness``.

empty_state = jdem.State.create(pos=jnp.zeros((0, 3)))

L = 1.0
tri = jnp.array([[L, 0.0, -L / 2], [-L, 0.0, -L / 2], [0.0, 0.0, L]])
state_single = jdem.State.add_facet(
    empty_state,
    tri,
    thickness=0.1,
    mass=5.0,
    vel=jnp.array([0.0, -2.0, 0.0]),
    species_id=1,
    rigid=True,
)
print(f"Single 3D facet: N={state_single.N}, clump_id={state_single.clump_id}")

mesh_vertices = jnp.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
mesh_faces = jnp.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
state_rigid_mesh = jdem.State.add_mesh(
    empty_state,
    mesh_vertices,
    mesh_faces,
    thickness=0.1,
    rigid=True,
    filled=True,
    mass=10.0,
    species_id=1,
)
print(f"Tetrahedron mesh: N={state_rigid_mesh.N} (4 faces x 3 vertices)")

# %%
# Connecting Facets and Shared Contact Surfaces
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# * :py:meth:`~jaxdem.state.State.add_connected_facet` builds facets
#   incrementally by mixing existing vertex indices (scalars) with new
#   coordinates (arrays). Connected vertices must share ``species_id`` and
#   rigid/flexible status.
# * :py:meth:`~jaxdem.state.State.add_contact_facets` attaches facet geometry
#   directly onto the existing nodes of a deformable particle without adding
#   mass or degrees of freedom:

seed = jdem.State.create(
    pos=jnp.array([[0.0, 0.0, 0.0]]), rad=jnp.array([0.1]), species_id=jnp.array([1])
)
fixed_facet = jdem.State.add_connected_facet(
    seed,
    [0, jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])],
    thickness=0.1,
    rigid=False,
    mass=2.5,
    species_id=1,
)
fixed_facet.fixed = jnp.ones(fixed_facet.N, dtype=bool)
print("Connected triangle vertices:", fixed_facet.N)

# Attach contact segments to a 2D deformable square:
nodes = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
elements = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]])
dp_state = jdem.State.create(
    pos=nodes,
    rad=jnp.full(4, 0.05),
    mass=jnp.full(4, 0.25),
    bond_id=[[1, 3], [0, 2], [1, 3], [0, 2]],
    species_id=jnp.ones(4, dtype=int),
)
dp_state = jdem.State.add_contact_facets(dp_state, elements)
print(f"Deformable contact surface: {dp_state.N} physical nodes, {len(elements)} facets")

# %%
# Force Routing and Sphere--Facet Impact
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Route sphere--sphere ``(0, 0)``, sphere--facet ``(0, 1)``, and facet--facet
# ``(1, 1)`` interactions via :py:class:`~jaxdem.forces.router.ForceRouter`.
# :py:class:`~jaxdem.forces.spring.SphereFacetSpringForce` and
# :py:class:`~jaxdem.forces.spring.FacetFacetSpringForce` are undamped elastic
# normal springs driven by ``young_eff`` (contact thickness is read from the
# facet vertices' ``rad``).
#
# Launch a unit-mass sphere (radius 0.1) toward the fixed triangle (thickness
# 0.1); contact begins at height ``0.1 + 0.1 = 0.2``:

mat_table = jdem.MaterialTable.from_materials(
    [jdem.Material.create("elastic", young=1e3, poisson=0.3, density=1.0)]
)
router = jdem.ForceRouter.from_dict(
    S=2,
    mapping={
        (0, 0): jdem.ForceModel.create("spring"),
        (0, 1): jdem.ForceModel.create("sphere_facet_spring"),
        (1, 1): jdem.ForceModel.create("facet_facet_spring"),
    },
)

impact_state = jdem.State.merge(
    fixed_facet,
    jdem.State.create(
        pos=jnp.array([[0.2, 0.2, 0.8]]),
        vel=jnp.array([[0.0, 0.0, -1.5]]),
        rad=jnp.array([0.1]),
        mass=jnp.array([1.0]),
        species_id=jnp.array([0]),
    ),
)
impact_system = jdem.System.create(
    state=impact_state,
    collider_type="naive",
    force_model=router,
    mat_table=mat_table,
    dt=0.0005,
    rotation_integrator_type=None,
)
impact_state, impact_system = jdem.System.initialize(impact_state, impact_system)
state_stepped, system, (scene_states, scene_systems) = jdem.System.trajectory_rollout(
    impact_state, impact_system, strides=jnp.asarray([0] + [50] * 40)
)
scene_title = "An elastic sphere bouncing from a fixed facet"
system.check_overflow()

overlap = jnp.maximum(0.2 - scene_states.pos[:, -1, 2], 0.0)
print(f"Max overlap: {float(overlap.max()):.4f}")
print(
    f"Normal velocity: in={float(scene_states.vel[0, -1, 2]):.3f}, "
    f"out={float(scene_states.vel[-1, -1, 2]):.3f}"
)

# %%
# Selecting Contact Feature Queries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Both facet laws allow disabling unused geometric queries statically (compiled
# out by JAX):
#
# * ``sphere_facet_spring``: ``face``, ``edge``, ``vertex`` (in 2D, only
#   ``edge`` and ``vertex`` apply).
# * ``facet_facet_spring``: ``vertex_triangle``, ``edge_edge``,
#   ``edge_triangle`` (in 2D, only ``edge_edge`` applies).

selective_router = jdem.ForceRouter.from_dict(
    S=2,
    mapping={
        (0, 0): jdem.ForceModel.create("spring"),
        (0, 1): jdem.ForceModel.create("sphere_facet_spring", edge=False, vertex=False),
        (1, 1): jdem.ForceModel.create(
            "facet_facet_spring", vertex_triangle=False, edge_triangle=False
        ),
    },
)
print("Selective router created:", type(selective_router).__name__)

# %%
# VTK Visualization
# ~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.writers.vtk_writer.VTKWriter` separates ordinary spheres
# (``spheres``), facet vertices (``facet_spheres``), and facet faces
# (``facets``). Note that the ``facets`` writer draws sheets at half-thickness
# (``rad / 2``) while contacts use full Minkowski ``rad``, so ParaView may show
# a small visual gap of ``rad / 2`` at contact:

tmp_dir = Path(tempfile.mkdtemp(prefix="jaxdem-facets-guide-"))
with jdem.VTKWriter(
    directory=tmp_dir, writers=["facets", "spheres", "facet_spheres"]
) as writer:
    writer.save(scene_states, scene_systems, trajectory=True, trajectory_axis=0)
print("Saved facet VTK output to:", tmp_dir)

# %%
# Notes & Limitations
# ~~~~~~~~~~~~~~~~~~~
# * **Shared edges/vertices:** Adjacent faces sharing a node do not repel each
#   other, but a sphere contacting a shared edge can accumulate additive
#   penalties from multiple incident faces.
# * **Minimization & AD:** At zero separation the contact normal is not always
#   a smooth energy gradient; prefer force-only minimizers like
#   :py:func:`~jaxdem.minimizers.optimizers.fire` for relaxing facet contacts.
# * **Manual connectivity edits:** If you mutate facet connectivity arrays
#   directly after creating a system, invalidate the collider via
#   :py:func:`jaxdem.colliders.invalidate_collider` before the next step.
