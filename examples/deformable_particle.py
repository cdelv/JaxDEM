"""
Deformable particles
---------------------

In this example, we'll set up 2D and 3D deformable particle simulations using JaxDEM.
The API works consistently across dimensions, with the primary difference being the
definition of the mesh elements (faces in 3D vs. edges in 2D).
"""

# %%
# Imports
# ~~~~~~~
import jax
import jax.numpy as jnp
import jaxdem as jdem
import math
from typing import Tuple, List


# %%
# Mesh Generation Helpers
# ~~~~~~~~~~~~~~~~~~~~~~~
# We define helper functions to generate the initial geometry.
# For 3D, we use an icosphere (approximating a sphere with triangles).
# For 2D, we use a discretized circle (approximating a circle with line segments).


def icosphere(
    r: float = 1.0, n: int = 3
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Generates a 3D icosphere mesh (vertices and triangular faces)."""
    t = (1.0 + 5.0**0.5) / 2.0
    v = [
        (-1.0, t, 0.0),
        (1.0, t, 0.0),
        (-1.0, -t, 0.0),
        (1.0, -t, 0.0),
        (0.0, -1.0, t),
        (0.0, 1.0, t),
        (0.0, -1.0, -t),
        (0.0, 1.0, -t),
        (t, 0.0, -1.0),
        (t, 0.0, 1.0),
        (-t, 0.0, -1.0),
        (-t, 0.0, 1.0),
    ]
    f = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    cache = {}

    def mid(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key not in cache:
            cache[key] = len(v)
            p1, p2 = v[i], v[j]
            new_vert = (p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2])
            v.append(new_vert)
        return cache[key]

    for _ in range(n):
        f = [
            sub
            for tri in f
            for sub in (
                (tri[0], mid(tri[0], tri[1]), mid(tri[2], tri[0])),
                (tri[1], mid(tri[1], tri[2]), mid(tri[0], tri[1])),
                (tri[2], mid(tri[2], tri[0]), mid(tri[1], tri[2])),
                (mid(tri[0], tri[1]), mid(tri[1], tri[2]), mid(tri[2], tri[0])),
            )
        ]
    final_verts = [tuple(c * r / sum(k**2 for k in p) ** 0.5 for c in p) for p in v]
    return final_verts, f


def circle(
    r: float = 1.0, n: int = 40
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
    """Generates a 2D circular mesh (vertices and line segments)."""
    vertices = []
    edges = []
    cx, cy = (0.0, 0.0)
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        vertices.append((x, y))

    for i in range(n):
        edges.append((i, (i + 1) % n))

    return vertices, edges


# %%
# 3D Deformable Particle
# ~~~~~~~~~~~~~~~~~~~~~~
# To create deformable particles, we use the :py:class:`~jaxdem.containers.DeformableParticleContainer`.
# This container initializes the necessary topology (mesh connectivity) and reference configuration
# (initial areas/volumes) required to compute elastic forces.
#
# **Force Parameters:**
#
# * ``em`` (Measure Elasticity): Controls the stiffness of the surface (Area in 3D, Length in 2D). Acts like a rubber membrane.
# * ``ec`` (Content Elasticity): Controls the incompressibility of the body (Volume in 3D, Area in 2D). Acts like internal fluid pressure.
# * ``gamma`` (Surface Tension): A force that actively minimizes the surface area.
# * ``eb`` (Bending Elasticity): Controls the stiffness against bending between adjacent faces/edges.
# * ``el`` (Edge Elasticity): Controls the stiffness of the wireframe edges (springs between vertices).
#
# **Managing Multiple Particles:**
# The container allows simulating multiple deformable bodies simultaneously using **ID arrays**.
# For example, ``elements_ID`` maps each face to a specific body index. If ``elements_ID[i] == k``,
# then element ``i`` is part of body ``k`` and will use the material properties defined at index ``k`` (e.g., ``em[k]``).
#
# **Particle IDs vs. Indices:**
# A crucial detail in JaxDEM is that the connectivity arrays (`elements`, `edges`) store the **unique Particle IDs**
# (corresponding to `state.ID`), **not** the current array index in `state.pos`.
# This is necessary because performance-critical components like the Cell List collider frequently reorder
# particles in memory to optimize memory access. The deformable particle force function automatically handles
# the mapping from persistent IDs to current memory locations at every time step.

vertices, faces = icosphere(2.0, 2)

DP_container = jdem.DeformableParticleContainer.create(
    vertices=vertices,
    elements=faces,
    ec=[10000.0],  # Controls Volume Conservation
    em=[10.0],  # Controls Surface Area Conservation
    gamma=[1.0],  # Surface Tension (minimizes surface area)
)

state = jdem.State.create(
    pos=vertices,
    rad=0.05 * jnp.ones(len(vertices)),
)

# %%
# Force Manager Integration
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# We register the force function generated by the container into the ``ForceManager``.
# The :py:class:`~jaxdem.containers.DeformableParticleContainer` will remove from the force
# computation all calculations required for the unused forces.
#
# **Hybrid Simulations:**
# Because the deformable model is just another force, you can seamlessly mix deformable particles with
# standard granular spheres, clumps, or rigid bodies in the same simulation.
# The collider handles the contact physics (treating nodes as spheres with radius ``state.rad``), while
# the `ForceFunction` applies the elastic deformation forces.
#
# The nodes of a deformable mesh could even be rigid bodies themselves, allowing for complex meta-structures.

system = jdem.System.create(
    state.shape,
    force_manager_kw=dict(
        gravity=jnp.array([0.0, -0.1, 0.0]),
        force_functions=(DP_container.create_force_function(DP_container),),
    ),
)

# %%
# Driving the Simulation
# ~~~~~~~~~~~~~~~~~~~~~~~
# Finally, we drive the simulation by stepping the system forward.

writer = jdem.VTKWriter(directory="/tmp/frames")
state, system = system.step(state, system, n=10)
writer.save(state, system)

# %%
# 2D Deformable Particle
# ~~~~~~~~~~~~~~~~~~~~~~
# The API is identical for 2D particles. However, the physical meaning of the parameters changes slightly:
# * ``elements`` are now line segments (edges).
# * ``ec`` conserves the enclosed 2D Area.
# * ``em`` conserves the Perimeter length.

vertices_2D, edges = circle(r=2.0, n=20)

DP_container_2D = jdem.DeformableParticleContainer.create(
    vertices=vertices_2D,
    elements=edges,
    ec=[1000.0],  # Area stiffness
    em=[10.0],  # Perimeter stiffness
    gamma=[1.0],  # Line tension
)

state2D = jdem.State.create(
    pos=vertices_2D,
    rad=0.05 * jnp.ones(len(vertices_2D)),
)

system2D = jdem.System.create(
    state2D.shape,
    force_manager_kw=dict(
        gravity=jnp.array([0.0, -0.1]),
        force_functions=(DP_container_2D.create_force_function(DP_container_2D),),
    ),
)

state2D, system2D = system2D.step(state2D, system2D, n=10)
writer.save(state2D, system2D)
