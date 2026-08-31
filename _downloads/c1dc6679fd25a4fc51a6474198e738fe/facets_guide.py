r"""Facets (Triangular & Line Boundaries)
----------------------------------------

A **facet** in JaxDEM is a 2D line segment or 3D triangle used to model boundaries, walls, or polygonal obstacles. Clump-based walls are bumpy because they are made of spheres. Facets give flat faces.

Internally, facets reuse the existing sphere-based broad-phase collision detection and rigid-body integration. The geometric resolution (point-triangle or segment-segment shortest distance) happens during the narrow-phase force calculation.

This guide covers:

- The internal representation of a facet (spheres mapped to vertices).
- Creating single facets dynamically using :py:meth:`~jaxdem.state.State.add_facet`.
- Creating whole triangular meshes using :py:meth:`~jaxdem.state.State.add_mesh`.
- Dynamic facet connectivity using :py:meth:`~jaxdem.state.State.add_connected_facet`.
- The "thickness" parameter (Minkowski expansion) for rounded edge-edge collisions.
- Force Routing (how to trigger facet force models).
- Visualizing facets in VTK.
"""

# %%
# Internal Representation
# ~~~~~~~~~~~~~~~~~~~~~~~~
# JaxDEM does not track facets in a separate
# array. It stores a facet as a clump of spheres:
#
# - **3D Facets** use 3 spheres (one at each vertex).
# - **2D Facets** use 2 spheres.
#
# Depending on the application, facets can be:
# - **Rigid**: Vertices share the same ``clump_id`` and belong to a rigid clump with a computed center of mass (COM) and moment of inertia.
# - **Flexible/Deformable**: Vertices behave like individual dynamic spheres with unique ``clump_id`` values, allowing the facet to deform under forces.
#
# To detect collisions with facets during broad-phase:
#
# - The standard ``CellList`` (DynamicCellList) assigns each vertex an inflated search radius (``_rad``), equal to the maximum distance from the COM to any vertex. Standard spherical queries then cover the entire facet without gaps.
# - The ``MultiCellList`` (DynamicMultiCellList) is tighter: it computes the joint axis-aligned bounding box (AABB) of the facet vertices instead of using ``_rad`` directly. This reduces cell registration overhead.
#
# Each vertex particle also stores the unique IDs of its facet vertices in ``state.facet_vertices``. This gives :math:`O(1)` lookup of the facet shape during narrow-phase contact resolution.

import jax.numpy as jnp

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
# Use the ``safety_factor`` parameter to multiply the search radius ``_rad``.
# This helps for fast-moving or flexible facets, whose broad-phase detection
# box needs more margin.

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
# - **Rigid vs. Flexible**: Set ``rigid=True`` to group all vertex particles into a single rigid clump (e.g. for a moving obstacle) or ``rigid=False`` for a deformable boundary mesh.
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

# Create a flexible/deformable mesh
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
# To construct custom boundaries incrementally or share vertices between adjacent facets,
# use :py:meth:`~jaxdem.state.State.add_connected_facet`.
#
# You specify the vertices of the new facet using a list of ``vertex_specs``.
# - A **scalar integer** unique ID refers to an existing vertex in the State.
# - An **ArrayLike** position array of shape ``(dim,)`` defines a new vertex.
#
# **Rules & Constraints**:
# 1. **Same Species ID**: All connected existing and new vertices must share the same ``species_id``.
# 2. **No Hybrid Facets**: You cannot mix rigid and flexible vertices in the same facet (i.e., you cannot connect a rigid clumped vertex to a flexible vertex).
# 3. **Dynamic Clump Update**: If you connect new vertices to an existing rigid clump, ``add_connected_facet`` updates the center of mass, moment of inertia, and relative vertex offsets of that clump.
# 4. **Single Clump for Rigid Connections**: If the new facet shares multiple existing rigid vertices, they must already belong to the same rigid clump. You cannot connect vertices from two different rigid clumps into one facet.

# Add a connected facet to the rigid mesh, sharing vertices 0 and 1, and introducing a new vertex
new_vertex_pos = jnp.array([0.5, 0.5, -1.0])
state_connected = jdem.State.add_connected_facet(
    state_rigid_mesh,
    [0, 1, new_vertex_pos],
    thickness=0.1,
    rigid=True,
    mass=2.5,
    species_id=1,  # must match the existing vertices' species ID
)

print(f"After connection, N = {state_connected.N} (1 new vertex added)")

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

# Route collisions between species 1 (Facet mesh vertices) and species 1.
# Note that the force models take no thickness parameter: the contact
# thickness comes from the facet vertices' ``state.rad``, set via the
# ``thickness`` argument of ``add_facet`` / ``add_mesh``.
router = jdem.ForceRouter.from_dict(
    S=2, mapping={(1, 1): jdem.ForceModel.create("facet_facet_spring")}
)

# Spatial colliders such as "cell_list" and "multi_cell_list" work with
# facets. Standard "cell_list" uses the facet's search radius `_rad` automatically.
# Passing ``state=`` to ``System.create`` forwards it to colliders that need it,
# so there is no need for ``collider_kw={"state": ...}``.
system = jdem.System.create(
    state=state_connected,
    collider_type="multi_cell_list",
    force_model=router,
    mat_table=mat_table,
    dt=1e-4,
)

# Run a simulation step to verify
state_stepped, system = system.step(state_connected, system)
print("Simulation step successful.")

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
import tempfile
from pathlib import Path

tmp_dir = Path(tempfile.gettempdir()) / "jaxdem_facets_guide"
with jdem.VTKWriter(
    directory=tmp_dir, writers=["facets", "spheres", "facet_spheres"]
) as writer:
    writer.save(state_connected, system)
print(f"Saved VTK output (check the generated {tmp_dir} directory).")
