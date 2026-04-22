# %%
"""SASA of a rigid clump via a spherical tracer
==================================================

SASA ("solvent-accessible surface area") is the surface traced by the
center of a spherical probe as it is rolled over a body's surface —
equivalently, the boundary of the Minkowski sum of the body with a
ball of the probe's radius. This example builds a single rigid GA
clump, probes it with a smooth sphere of radius ``r_tracer``, and
estimates the SASA from the per-direction center-to-center
separations reported by
:func:`~jaxdem.utils.surfaceProperties.compute_surface_properties`.

For each approach direction ``d_i`` on ``S^2``, the probe bisects the
tracer's position along the center-to-center axis until the maximum
pairwise sphere overlap equals ``target_overlap``. The corresponding
center-to-center separation ``separation[i]`` is the distance from
the central clump's COM to the SASA surface along ``d_i``.

Numerics
--------

* ``target_overlap`` is how much interpenetration is tolerated at the
  "just-touching" configuration. It should be much smaller than any
  physical length scale in the problem so the tracer is effectively at
  zero contact (SASA is defined at contact, not inside the body).
* ``separation_tolerance`` is the bisection's convergence tolerance on
  the center-to-center separation. **It must be strictly smaller than**
  ``target_overlap`` — otherwise the final bracket is wider than the
  overlap band and the bisection can converge to a no-contact separation,
  silently reporting the wrong surface.
* Both values are orders of magnitude below the float32 epsilon
  (``~1.2e-7``), so **x64 must be enabled before any JAX operation runs**
  or the bisection cannot resolve the bracket.
"""

# %%
# Enable x64 before any other JAX work.
import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import numpy as np

from jaxdem.utils.particleCreation import create_ga_state, create_sphere_state
from jaxdem.utils.surfaceProperties import compute_surface_properties


# %%
# Central rigid clump + tracer sphere
# -----------------------------------
# The central body is a 20-asperity rigid clump with a solid core; the
# tracer is a single smooth sphere built with :func:`create_sphere_state`
# (which skips the Thomson mesh and union-volume MC used by
# :func:`create_ga_state`).

central = create_ga_state(
    N=1,
    nv=20,
    dim=3,
    particle_radius=0.5,
    asperity_radius=0.1,
    n_steps=1_000,
    particle_type="clump",
    core_type="solid",  # clumps need to be solid for the sasa protocol to be robust
    n_samples=10_000_000,  # you need at least 10m sampling to get decent accuracy for the clump COM
    seed=0,
)

tracer_radius = 0.05
tracer = create_sphere_state(radii=tracer_radius, dim=3)


# %%
# Probe the surface
# -----------------
# ``target_overlap`` > ``separation_tolerance`` so the bisection can
# resolve the contact band. ``n_orientations = n_rolls = 1`` because a
# smooth sphere has no orientation degree of freedom.

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
print(f"separation: min={separation.min():.4f}  max={separation.max():.4f}  mean={separation.mean():.4f}")


# %%
# SASA from the per-direction separations
# ---------------------------------------
# Triangulate the approach directions on ``S^2`` (their 3D convex hull
# is a Delaunay triangulation of points on the sphere), lift each
# triangle vertex to the SASA surface with ``separation[i] *
# direction[i]``, and sum the Euclidean triangle areas. The result is
# the surface area of the polyhedron whose vertices are the sampled
# tracer-center positions -- convergent to the true SASA as
# ``n_points`` grows.

from scipy.spatial import ConvexHull

directions = np.asarray(result["approach_directions"])        # (N, 3) on S^2
surface_points = separation[:, None] * directions             # (N, 3) on SASA
triangles = ConvexHull(directions).simplices                  # (F, 3) vertex idx
tri = surface_points[triangles]                               # (F, 3, 3)
e1 = tri[:, 1] - tri[:, 0]
e2 = tri[:, 2] - tri[:, 0]
sasa = 0.5 * float(np.sum(np.linalg.norm(np.cross(e1, e2), axis=-1)))

print(f"estimated SASA = {sasa:.4f}")
