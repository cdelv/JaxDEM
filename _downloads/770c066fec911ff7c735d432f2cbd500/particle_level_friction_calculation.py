# %%
"""Particle-level friction coefficient of a rigid clump
=========================================================

Same setup as the SASA example, but the quantity of interest is the
probe-reported friction coefficient ``mu = |F_t| / |F_n|`` instead of
the surface area. We take the mean of ``mu`` over the sampled approach
directions; since the directions are a near-uniform Fibonacci lattice
on ``S^2``, that's the surface-averaged friction to leading order.

Numerics match :file:`sasa_calculation.py`: ``target_overlap = 1e-10``
and ``separation_tolerance = 1e-12``, which requires x64.
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
central = create_ga_state(
    N=1,
    nv=20,
    dim=3,
    particle_radius=0.5,
    asperity_radius=0.1,
    particle_type="clump",
    core_type="solid",  # clumps need to be solid for the sasa protocol to be robust
    n_samples=10_000_000,  # you need at least 10m sampling to get decent accuracy for the clump COM
    seed=0,
    mesh_kwargs={"steps": 1_000},
)

tracer_radius = 0.05
tracer = create_sphere_state(radii=tracer_radius, dim=3)


# %%
# Probe the surface
n_points = 1024

result = compute_surface_properties(
    central,
    tracer,
    target_overlap=1e-10,
    separation_tolerance=1e-12,
    n_points=n_points,
    n_orientations=1,
    n_rolls=1,
)

mu = np.asarray(result["mu"]).reshape(n_points)
print(f"mu: min={mu.min():.4f}  max={mu.max():.4f}")


# %%
# Mean friction across the sampled approach directions.
mean_friction = float(mu.mean())
print(f"mean friction = {mean_friction:.4f}")
