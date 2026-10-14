"""SASA of a rigid clump via a spherical tracer
==================================================

SASA ("solvent-accessible surface area") is the surface that the
center of a spherical probe traces as it rolls over a body.
Equivalently, it is the boundary of the Minkowski sum of the body with
a ball of the probe's radius. This example builds a single rigid GA
clump, probes it with a smooth sphere of radius ``r_tracer``, and
estimates the SASA from the per-direction center-to-center
separations that
:func:`~jaxdem.utils.surface_properties.compute_surface_properties`
reports.

For each approach direction ``d_i`` on ``S^2``, sphere-ray intersections
locate the first contact from outside at ``target_overlap``. This also
works when the ray crosses several disconnected contact intervals.
The center-to-center separation ``separation[i]`` is the distance from
the central clump's COM to the SASA surface along ``d_i``. A missed ray
or a contact without a resolved normal force gives ``NaN``.

Numerics
--------

* ``target_overlap`` is how much interpenetration the "just-touching"
  configuration tolerates. Keep it much smaller than any physical
  length scale in the problem, so the tracer is effectively at zero
  contact (we define SASA at contact, not inside the body).
* ``separation_tolerance`` is retained for API compatibility and
  validation. It must be positive and smaller than ``target_overlap``;
  the direct contact roots have no iteration tolerance.
* Enable x64 before any JAX operation to resolve the small overlap
  used here (``1e-10``) at the body's length scale. Float32 roundoff
  can erase that overlap and leave the measured force unresolved.
"""

# %%
# Enable x64 before any other JAX work.
import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import numpy as np

from jaxdem.utils.particle_creation import create_ga_state, create_sphere_state
from jaxdem.utils.surface_properties import compute_surface_properties

# %%
# Central rigid clump + tracer sphere
# -----------------------------------
# The central body is a 20-asperity rigid clump with a solid core. The
# tracer is a single smooth sphere built with :func:`create_sphere_state`,
# which skips the Thomson mesh and union-volume MC that
# :func:`create_ga_state` uses.

central = create_ga_state(
    N=1,
    nv=20,
    dim=3,
    particle_radius=0.5,
    asperity_radius=0.1,
    particle_type="clump",
    core_type="solid",  # fill the central void so each approach ray meets the body
    n_samples=10_000_000,  # the default. ~10M samples give decent accuracy for the clump COM
    seed=0,
    mesh_kwargs={"steps": 1_000},
)

tracer_radius = 0.05
tracer = create_sphere_state(radii=tracer_radius, dim=3)


# %%
# Probe the surface
# -----------------
# Keep ``target_overlap`` small and the compatibility tolerance below it.
# ``n_orientations = n_rolls = 1`` because a smooth sphere has no
# orientation degree of freedom.

target_overlap = 1e-10
separation_tolerance = 1e-12
n_points = 1024

result = compute_surface_properties(
    central,
    tracer,
    target_overlap=target_overlap,
    separation_tolerance=separation_tolerance,
    n_points=n_points,
    n_orientations=1,
    n_rolls=1,
)

separation = np.asarray(result["separation"]).reshape(n_points)
print(
    f"separation: min={separation.min():.4f}  max={separation.max():.4f}  mean={separation.mean():.4f}"
)


# %%
# SASA from the per-direction separations
# ---------------------------------------
# Triangulate the approach directions on ``S^2``. Their 3D convex hull
# is a Delaunay triangulation of points on the sphere. Lift each
# triangle vertex to the SASA surface with ``separation[i] *
# direction[i]`` and sum the Euclidean triangle areas. The result is
# the surface area of the polyhedron with the sampled tracer-center
# positions as vertices. It converges to the true SASA as
# ``n_points`` grows.

from scipy.spatial import ConvexHull

directions = np.asarray(result["approach_directions"])  # (N, 3) on S^2
surface_points = separation[:, None] * directions  # (N, 3) on SASA
triangles = ConvexHull(directions).simplices  # (F, 3) vertex idx
tri = surface_points[triangles]  # (F, 3, 3)
e1 = tri[:, 1] - tri[:, 0]
e2 = tri[:, 2] - tri[:, 0]
sasa = 0.5 * float(np.sum(np.linalg.norm(np.cross(e1, e2), axis=-1)))

print(f"estimated SASA = {sasa:.4f}")
