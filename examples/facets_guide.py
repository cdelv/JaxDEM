r"""Facets (Triangular & Line Boundaries)
-----------------------------------------

A **facet** in JaxDEM is a 2D line segment or 3D triangle used to model boundaries, walls, or polygonal obstacles. Clump-based walls are bumpy because they are made of spheres. Facets give flat faces.

Internally, facets reuse the existing sphere-based broad-phase collision detection and rigid-body integration. The geometric resolution (point-triangle or segment-segment shortest distance) happens during the narrow-phase force calculation.

This guide covers:

- The internal representation of a facet (spheres mapped to vertices).
- Creating single facets dynamically using :py:meth:`~jaxdem.state.State.add_facet`.
- Creating whole triangular meshes using :py:meth:`~jaxdem.state.State.add_mesh`.
- Dynamic facet connectivity using :py:meth:`~jaxdem.state.State.add_connected_facet`.
- Shared-node contact surfaces on bonded deformable particles.
- The "thickness" parameter (Minkowski expansion) for rounded edge-edge collisions.
- Force Routing (how to trigger facet force models).
- Visualizing facets in VTK.
"""

# %%
# Internal Representation
# ~~~~~~~~~~~~~~~~~~~~~~~~
# JaxDEM represents contact facets through particle keys at their vertices:
#
# - **3D Facets** use 3 spheres (one at each vertex).
# - **2D Facets** use 2 spheres.
#
# Depending on the application, facets can be:
#
# - **Rigid**: Vertices share the same ``clump_id`` and belong to a rigid clump with a computed center of mass (COM) and moment of inertia.
# - **Independent vertices**: Vertices have unique ``clump_id`` values and
#   may move independently. This provides deforming contact geometry, not a
#   cohesive elastic shell: add a bonded model to enforce material elasticity.
#
# To detect collisions with facets during broad-phase:
#
# - The standard ``CellList`` recomputes each facet's current extent from its
#   search-key vertex to all facet vertices, adds thickness, and takes the
#   maximum with the stored ``_rad`` lower bound.
# - ``MultiCellList`` also uses these current bounds, including every face
#   incident on a shared node.
#
# Each vertex stores its first face in ``facet_id`` / ``facet_vertices``.
# Shared nodes store additional faces in ``facet_ids_extra`` /
# ``facet_vertices_extra``. Forces from all incident faces accumulate on the
# same physical node; no extra mass or integration degree of freedom is added.

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import jaxdem as jdem

# Initialize an empty state
empty_state = jdem.State.create(
    pos=jnp.zeros((0, 3)),
)

# %%
# Creating a Single Facet
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Create a single facet with the :py:meth:`~jaxdem.state.State.add_facet`
# method. Pass a ``(3, 3)`` array of vertices (or ``(2, 2)`` in 2D).
# The method computes the center of mass, true moment of inertia, and relative
# vertex offsets.
#
# ``safety_factor`` enlarges the stored ``_rad`` lower bound established at
# construction. The colliders still recompute current flexible-facet geometry;
# use a factor above one when the stored bound should include extra margin.

L = 1.0
vertices = jnp.array([[L, 0.0, -L / 2], [-L, 0.0, -L / 2], [0.0, 0.0, L]])

state = jdem.State.add_facet(
    empty_state,
    vertices,
    thickness=0.1,
    mass=5.0,  # Total mass of the facet
    vel=jnp.array([0.0, -2.0, 0.0]),  # Initial velocity (1 value per facet)
    species_id=1,  # Species ID used for Force Routing
    rigid=True,  # Rigid facet clump
    safety_factor=1.2,  # Safety factor to enlarge detection box
)

print(f"Number of particles in state: {state.N} (3 vertices for 1 facet)")
print(f"Shared Clump ID: {state.clump_id}")

# %%
# Creating a Triangular Mesh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:meth:`~jaxdem.state.State.add_mesh` builds entire boundary walls or
# polyhedra from a collection of mesh vertices and face connectivity indices (faces).
#
# - **Rigid vs. independent vertices**: ``rigid=True`` groups the mesh into
#   one body; ``rigid=False`` creates independent face vertices. Shared input
#   coordinates are duplicated between faces; no shared-node constraint or
#   bonded elasticity is added automatically.
# - **Solid vs. Shell**: If ``rigid=True``, set ``filled=True`` to calculate the center of mass and moment of inertia of a solid filled polyhedron, or ``filled=False`` for a hollow shell/boundary. For open meshes (uneven surfaces, sheets, or boundaries that are not closed), set ``filled=False`` (shell mode). Only then do the center of mass and inertia calculations give physically correct values.

mesh_vertices = jnp.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
mesh_faces = jnp.array(
    [
        [0, 2, 1],  # bottom face
        [0, 1, 3],  # front face
        [0, 3, 2],  # side face
        [1, 2, 3],  # diagonal face
    ]
)

# Create a rigid, solid (filled) mesh
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

# Create a mesh with independently moving face vertices
state_flex_mesh = jdem.State.add_mesh(
    empty_state,
    mesh_vertices,
    mesh_faces,
    thickness=0.1,
    rigid=False,
    mass=10.0,
    species_id=1,
)

print(f"Rigid mesh particles: {state_rigid_mesh.N} (4 faces * 3 vertices = 12)")
print(f"Flexible mesh particles: {state_flex_mesh.N}")

# %%
# Dynamic Facet Connection
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# To construct custom boundaries incrementally from ordinary particles and new
# vertices, use :py:meth:`~jaxdem.state.State.add_connected_facet`.
#
# You specify the vertices of the new facet using a list of ``vertex_specs``.
#
# - A **scalar integer** index refers to an existing particle in the State,
#   including one already used by another facet.
# - An **ArrayLike** position array of shape ``(dim,)`` defines a new vertex.
#
# **Rules & Constraints**:
#
# 1. **Same Species ID**: All connected existing and new vertices must share the same ``species_id``.
# 2. **No Hybrid Facets**: You cannot mix rigid and flexible vertices in the same facet (i.e., you cannot connect a rigid clumped vertex to a flexible vertex).
# 3. **Dynamic Clump Update**: If you connect new vertices to an existing rigid clump, ``add_connected_facet`` updates the center of mass, moment of inertia, and relative vertex offsets of that clump.
# 4. **Single Clump for Rigid Connections**: If the new facet shares multiple existing rigid vertices, they must already belong to the same rigid clump. You cannot connect vertices from two different rigid clumps into one facet.
# 5. **Shared nodes**: Existing nodes may belong to several facets. Duplicate
#    faces or repeated vertices within one face are rejected.

# Start with one ordinary particle, then reuse it as one vertex of a triangle.
# Its radius is retained, so choose it to match the intended facet thickness.
connection_seed = jdem.State.create(
    pos=jnp.array([[0.0, 0.0, 0.0]]), rad=jnp.array([0.1]), species_id=jnp.array([1])
)
state_connected = jdem.State.add_connected_facet(
    connection_seed,
    [0, jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])],
    thickness=0.1,
    rigid=False,
    mass=2.5,
    species_id=1,
)

print(f"After connection, N = {state_connected.N} (2 new vertices added)")

# %%
# Shared Contact Nodes on a Deformable Particle
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Attach contact geometry to an existing bonded mesh with
# :py:meth:`~jaxdem.state.State.add_contact_facets`. This leaves nodal masses,
# positions, radii, and bonds unchanged. The bonded model supplies elasticity;
# the contact facets follow the deforming nodes and transfer contact forces
# through closest-point barycentric weights.

nodes = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
elements = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]])
dp_state = jdem.State.create(
    pos=nodes,
    rad=jnp.full(4, 0.05),  # Uniform contact thickness
    mass=jnp.full(4, 0.25),
    bond_id=[[1, 3], [0, 2], [1, 3], [0, 2]],
    species_id=jnp.ones(4, dtype=int),
)
dp_state = jdem.State.add_contact_facets(dp_state, elements)
dp_model = jdem.BondedForceModel.create(
    "deformable_particle_model",
    vertices=nodes,
    elements=elements,
    edges=elements,
    el=100.0,
    em=100.0,
    ec=100.0,
)
print(f"Shared surface: {dp_state.N} physical nodes, {len(elements)} facets")
# Pass ``bonded_force_model=dp_model`` when creating the System, and route
# sphere/DP contacts to ``sphere_facet_spring`` and DP/DP contacts to
# ``facet_facet_spring`` as shown below. In 3D, use triangle elements (M, 3).
#
# Adjacent faces sharing a node do not generate facet-facet repulsion.
# Contact penalties are additive per face: a sphere near a shared edge or
# vertex can contact several incident faces, increasing the effective stiffness.
# These laws do not merge contacts into a single surface-union contact.
#
# Neighbor lists rebuild automatically when these constructors add facets.
# Directly replacing connectivity while retaining IDs requires explicit
# replacing ``system.collider`` with ``system.collider.invalidate()`` before
# the next contact evaluation.

# %%
# Force Routing & Thickness (Minkowski Sum)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Even though the broad-phase collider triggers on the inflated
# vertex spheres, the chosen Force Model (e.g. ``FacetFacetSpringForce``)
# sets the **actual** collision shape.
#
# Facets have a physical property called ``thickness``, passed during facet/mesh creation.
# The collision engine expands the thin mathematical triangle or segment by a
# **Minkowski sum** with a sphere. The sphere radius is the vertex particle's
# radius, which stores the ``thickness`` value (i.e., :math:`R = \text{thickness}`).
#
# Because of this expansion:
# - **Faces** are flat planes shifted outwards by :math:`R`.
# - **Edges** are rounded cylinders (pipes) of radius :math:`R`.
# - **Vertices** are rounded spheres of radius :math:`R`.
#
# Edge-to-edge and vertex-to-vertex collisions then resolve without
# sharp-corner singularities.
#
# JaxDEM supports the following narrow-phase force models for facets:
# - ``sphere_facet_spring``: For sphere-to-facet contacts.
# - ``facet_facet_spring``: For facet-to-facet contacts.
#
# At zero separation these models choose a contact normal that may not be
# an energy gradient. Use FIRE or another force-only optimizer for physical
# relaxation. Value-based optimizers differentiate energy directly. An
# intersecting contact can have a flat energy, so an energy minimum need not
# be force balanced. Energy AD requires Naive or NeighborList; direct CellList
# and MultiCellList traversals do not support reverse-mode differentiation.
#
# The ``ForceRouter`` routes the contacts. In a typical simulation with normal spheres (species 0)
# and boundaries/facets (species 1), you route collisions as:
# - ``(0, 0)`` -> ``spring`` or ``hertz`` (sphere-sphere contacts)
# - ``(1, 0)`` -> ``sphere_facet_spring`` (sphere-facet contacts)
# - ``(1, 1)`` -> ``facet_facet_spring`` (facet-facet contacts)

# Create a material
mat = jdem.Material.create(
    "elasticfrict", young=1e3, poisson=0.3, density=1.0, mu=0.5, e=0.5, mu_r=0.1
)
mat_table = jdem.MaterialTable.from_materials([mat])

# Route sphere/facet and facet/facet contacts separately.
# Note that the force models take no thickness parameter: the contact
# thickness comes from the facet vertices' ``state.rad``, set via the
# ``thickness`` argument of ``add_facet`` / ``add_mesh``.
router = jdem.ForceRouter.from_dict(
    S=2,
    mapping={
        (0, 1): jdem.ForceModel.create("sphere_facet_spring"),
        (1, 1): jdem.ForceModel.create("facet_facet_spring"),
    },
)

# Spatial colliders such as ``"cell_list"`` and ``"multi_cell_list"`` work with
# facets and derive their search bounds from the current facet geometry.
# This four-particle walkthrough uses naive search to keep compilation short.
# Passing ``state=`` to ``System.create`` forwards it to colliders that need it,
# so there is no need for ``collider_kw={"state": ...}``.
# Hold the connected triangle fixed and add one sphere just above its face.
# Its center is 0.15 above the plane, while the radius plus facet expansion
# is 0.2: overlap=0.05, giving a normal spring force 1000*0.05=50.
state_connected.fixed = jnp.ones(state_connected.N, dtype=bool)
state_connected = jdem.State.merge(
    state_connected,
    jdem.State.create(pos=[[0.2, 0.2, 0.15]], rad=[0.1], species_id=[0]),
)
system = jdem.System.create(
    state=state_connected,
    collider_type="naive",
    force_model=router,
    mat_table=mat_table,
    dt=1e-4,
)

# Check an actual sphere/facet interaction before advancing.
state_connected, system = jdem.System.initialize(state_connected, system)
np.testing.assert_allclose(
    state_connected.force[-1], [0.0, 0.0, 50.0], rtol=2e-5, atol=1e-5
)
print("Sphere normal contact force:", state_connected.force[-1])
fixed_positions = np.asarray(state_connected.pos[:3]).copy()
state_stepped, system = system.step(state_connected, system, n=2)
np.testing.assert_allclose(state_stepped.pos[:3], fixed_positions, atol=1e-7)
assert float(state_stepped.pos[-1, 2]) > 0.15
print("Sphere moved away from the fixed facet.")

# %%
# Selecting Contact Calculations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Both facet laws enable all contact calculations by default. Their switches
# are static JAX configuration: changing a switch compiles a new kernel with
# the disabled calculations omitted. The same choices apply to force and
# potential energy.
#
# For ``sphere_facet_spring``, ``face``, ``edge``, and ``vertex`` select the
# feature containing the closest point. For example, a face-only wall ignores
# contacts whose closest point lies on an edge or vertex; it does not substitute
# a farther enabled feature. In 2D, ``edge`` controls the open line segment and
# ``vertex`` controls its endpoints; ``face`` has no effect.
sphere_face_only = jdem.ForceModel.create(
    "sphere_facet_spring", edge=False, vertex=False
)

# For ``facet_facet_spring``, the switches select three geometry query groups:
#
# - ``vertex_triangle``: each vertex against the opposite triangle, including
#   its edges and vertices.
# - ``edge_edge``: each pair of edges, including their endpoints.
# - ``edge_triangle``: edge piercings through the opposite triangle.
#
# The closest remaining candidate defines the contact. These query groups
# overlap physical contact types: parallel face-face contacts, for example,
# can be detected by edge-edge queries. In 2D only ``edge_edge`` is used.
# The following law omits vertex-triangle and piercing queries, retaining
# edge-edge distances. Use such a reduction only when those omitted queries
# are unnecessary for the geometry: it can miss interior crossings or contacts
# with a face away from its boundary.
facet_edges_only = jdem.ForceModel.create(
    "facet_facet_spring", vertex_triangle=False, edge_triangle=False
)

# Configure each species pair separately through the usual router.
selective_router = jdem.ForceRouter.from_dict(
    S=2,
    mapping={
        (0, 0): jdem.ForceModel.create("spring"),
        (0, 1): sphere_face_only,
        (1, 1): facet_edges_only,
    },
)
# Pass ``force_model=selective_router`` to ``System.create`` to use these
# choices. Disabling every applicable switch makes that law return zero
# force, torque, and potential energy.

# %%
# VTK Visualization
# ~~~~~~~~~~~~~~~~~~
# When writing the simulation state, ``VTKWriter`` splits the output:
# - It writes normal spheres (non-facet particles, where ``facet_id == -1``) with the standard ``spheres`` writer.
# - It writes facet vertex spheres (where ``facet_id != -1``) to a separate PVD collection named ``facet_spheres``.
# - It exports the facet surfaces/triangles with the ``facets`` writer.
#
# The VTK rendering has one nuance. The collision engine uses the full vertex radius (``state.rad``)
# as the Minkowski expansion radius for contacts. The VTK writer draws the facets with a Minkowski radius of
# ``state.rad / 2`` (half-thickness) to show the physical sheet thickness. In ParaView you may therefore
# see a small gap of size ``state.rad / 2`` between colliding spheres
# and the rendered facet surface.
tmp_dir = Path(tempfile.mkdtemp(prefix="jaxdem-facets-guide-"))
with jdem.VTKWriter(
    directory=tmp_dir, writers=["facets", "spheres", "facet_spheres"]
) as writer:
    writer.save(state_stepped, system)
print(f"Saved VTK output (check the generated {tmp_dir} directory).")
# Output is retained for ParaView inspection; remove the directory when done.
