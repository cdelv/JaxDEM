r"""Facets (Triangular & Line Boundaries)
----------------------------------------

A **facet** in JaxDEM is a physically accurate 2D line segment or 3D triangle used to model complex boundaries, walls, or arbitrarily shaped polygonal obstacles. Unlike standard clump-based walls which are bumpy (made of spheres), facets provide perfectly flat faces.

Under the hood, facets are seamlessly integrated into the existing sphere-based broad-phase collision detection and rigid body integration. The actual geometric resolution (point-triangle or segment-segment shortest distance) happens exactly during the narrow-phase force calculation.

This guide covers:

- The internal representation of a facet (spheres mapped to vertices).
- Creating single facets dynamically using :py:meth:`~jaxdem.state.State.add_facet`.
- Creating whole triangular meshes using :py:meth:`~jaxdem.state.State.add_mesh`.
- Dynamic facet connectivity using :py:meth:`~jaxdem.state.State.add_connected_facet`.
- The "thickness" parameter (Minkowski expansion) for robust edge-edge collisions.
- Force Routing (how to trigger facet force models).
- Visualizing facets accurately in VTK.
"""

# %%
# Internal Representation
# ~~~~~~~~~~~~~~~~~~~~~~~~
# To maintain high GPU performance, JaxDEM does not track facets in a separate
# array. Instead, a facet is internally represented as a clump of spheres:
#
# - **3D Facets** use 3 spheres (one at each vertex).
# - **2D Facets** use 2 spheres.
#
# Depending on the application, facets can be:
# - **Rigid**: Vertices share the same ``clump_id`` and belong to a rigid clump with a computed center of mass (COM) and moment of inertia.
# - **Flexible/Deformable**: Vertices behave like individual dynamic spheres with unique ``clump_id``s, allowing the facet to deform under forces.
#
# In order for the broad-phase spatial hashers (like ``CellList`` and ``MultiCellList``)
# to detect collisions with the facet efficiently without sacrificing tightness, every vertex is
# assigned a search radius (``_rad``) equal to the maximum distance from the center of mass to
# any vertex. This guarantees the union of their spherical AABBs tightly covers
# the entire facet without gaps.
#
# Both the standard ``CellList`` (DynamicCellList) and ``MultiCellList`` (DynamicMultiCellList)
# utilize this ``_rad`` property to automatically determine cell sizes and search ranges.
#
# Furthermore, each vertex particle stores the unique IDs of the vertices that form its facet in ``state.facet_vertices``, enabling $O(1)$ lookup of the facet shape during narrow-phase contact resolution.

import jax.numpy as jnp
import jaxdem as jdem
import numpy as np

# Initialize an empty state
empty_state = jdem.State.create(
    pos=jnp.zeros((0, 3)),
    rad=jnp.zeros((0,)),
    mass=jnp.zeros((0,)),
    species_id=jnp.zeros((0,), dtype=int),
)

# %%
# Creating a Single Facet
# ~~~~~~~~~~~~~~~~~~~~~~~~
# The simplest way to create a facet is using the :py:meth:`~jaxdem.state.State.add_facet`
# method. You pass it a ``(3, 3)`` array of vertices (or ``(2, 2)`` in 2D), and
# it handles computing the center of mass, true moment of inertia, and relative
# vertex offsets.
#
# Use the ``safety_factor`` parameter to multiply the search radius ``_rad``, which is
# particularly useful for expanding the broad-phase detection box of fast-moving or flexible facets.

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
# You can define entire boundary walls or polyhedra using :py:meth:`~jaxdem.state.State.add_mesh`.
# You pass a collection of mesh vertices and face connectivity indices (faces).
#
# - **Rigid vs. Flexible**: Set ``rigid=True`` to group all vertex particles into a single rigid clump (e.g. for a moving obstacle) or ``rigid=False`` for a deformable boundary mesh.
# - **Solid vs. Shell**: If ``rigid=True``, set ``filled=True`` to calculate the center of mass and moment of inertia of a solid filled polyhedron, or ``filled=False`` for a hollow shell/boundary.

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
    species_id=2,
)

# Create a flexible/deformable mesh
state_flex_mesh = jdem.State.add_mesh(
    empty_state,
    mesh_vertices,
    mesh_faces,
    thickness=0.1,
    rigid=False,
    mass=10.0,
    species_id=3,
)

print(f"Rigid mesh particles: {state_rigid_mesh.N} (4 faces * 3 vertices = 12)")
print(f"Flexible mesh particles: {state_flex_mesh.N}")

# %%
# Dynamic Facet Connection
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# To construct custom boundaries incrementally or share vertices between adjacent facets,
# use :py:meth:`~jaxdem.state.State.add_connected_facet`.
#
# You specify the vertices of the new facet using a list of ``vertex_specs``.
# - A **scalar integer** unique ID refers to an existing vertex in the State.
# - An **ArrayLike** position array of shape ``(dim,)`` defines a new vertex.
#
# **Rules & Constraints**:
# 1. **Same Species ID**: All connected vertices (both existing and new) must share the same ``species_id``.
# 2. **No Hybrid Facets**: You cannot mix rigid and flexible vertices in the same facet (i.e., you cannot connect a rigid clumped vertex to a flexible vertex).
# 3. **Dynamic Clump Update**: If you connect new vertices to a rigid clump, the center of mass, moment of inertia, and relative vertex offsets of that clump are dynamically updated.

# Add a connected facet to the rigid mesh, sharing vertices 0 and 1, and introducing a new vertex
new_vertex_pos = jnp.array([0.5, 0.5, -1.0])
state_connected = jdem.State.add_connected_facet(
    state_rigid_mesh,
    [0, 1, new_vertex_pos],
    thickness=0.1,
    rigid=True,
    mass=2.5,
    species_id=2,  # must match the existing vertices' species ID
)

print(f"After connection, N = {state_connected.N} (1 new vertex added)")

# %%
# Force Routing & Thickness (Minkowski Sum)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Even though the broad-phase collider triggers based on the inflated
# vertex spheres, the **actual** collision shape is controlled entirely by
# the chosen Force Model (e.g. ``FacetFacetSpringForce``).
#
# Facets have a physical property called ``thickness``. The collision engine
# expands the infinitely thin mathematical triangle using a **Minkowski sum**
# with a sphere of radius $R = \text{thickness} / 2$.
#
# Because of this expansion:
# - **Faces** are perfectly flat planes shifted outwards.
# - **Edges** are perfectly rounded cylinders (pipes).
# - **Vertices** are perfectly rounded spheres.
#
# This guarantees that edge-to-edge collisions resolve smoothly without
# sharp-corner singularities!

# Create a material
mat = jdem.Material.create(
    "elasticfrict", young=1e7, poisson=0.3, density=2000.0, mu=0.5, e=0.5, mu_r=0.1
)
mat_table = jdem.MaterialTable.from_materials([mat])

# Route collisions between species 2 (Facet mesh vertices) and species 2
router = jdem.ForceRouter.from_dict(
    S=4, mapping={(2, 2): jdem.ForceModel.create("facet_facet_spring", thickness=0.1)}
)

# Spatial colliders such as "CellList" and "MultiCellList" are fully compatible with
# facets. Standard "CellList" uses the facet's search radius `_rad` automatically.
system = jdem.System.create(
    state_connected.shape,
    collider_type="CellList",
    collider_kw={"state": state_connected},
    force_model_type="forcerouter",
    force_model_kw={"table": router.table},
    mat_table=mat_table,
    dt=1e-4,
)

# Run a simulation step to verify
state_stepped, system = system.step(state_connected, system)
print("Simulation step successful with CellList collider.")

# %%
# VTK Visualization
# ~~~~~~~~~~~~~~~~~~
# When writing the simulation state using ``VTKWriter``:
# - Normal spheres (non-facet particles, where ``facet_id == -1``) are written using the standard `spheres` writer.
# - Facet vertex spheres (where ``facet_id != -1``) are written to a separate PVD collection named `facet_spheres`.
# - The actual facet surfaces/triangles are exported using the `facets` writer, which computes the true 3D Minkowski solid using the force model's ``thickness``.

writer = jdem.VTKWriter(writers=["facets", "spheres", "facet_spheres"])
writer.save(state_connected, system)
print("Saved VTK output (check the generated frames/ directory).")
