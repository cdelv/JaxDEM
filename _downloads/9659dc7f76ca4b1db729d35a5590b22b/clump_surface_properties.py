# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Clump surface properties: SASA and geometric friction
=========================================================

Probe a 3D rigid geometric-asperity clump with a spherical tracer using
:func:`~jaxdem.utils.surface_properties.compute_surface_properties` to measure
two shape-driven quantities from a single call:

1. **Solvent-accessible surface area (SASA):** triangulate the probed contact
   radii :math:`\rho_i \hat{\mathbf{d}}_i` on :math:`S^2` (valid for
   star-shaped / solid-core bodies without overhangs).
2. **Particle-level geometric friction:** decompose each probe contact force
   along the COM axis :math:`\hat{\mathbf{d}}_i` into
   :math:`\mu_i = \|F_{t,i}\| / |F_{n,i}|`, comparing both solid-angle and
   surface-area-weighted averages against a smooth sphere (:math:`\mu = 0`).

Enable float64 (``x64``) before running JAX so ``target_overlap = 1e-10`` is
resolved cleanly. For non-spherical tracers, increase ``n_orientations`` and
``n_rolls`` to average over tracer orientations.
"""

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from jaxdem.utils.particle_creation import create_ga_state, create_sphere_state
from jaxdem.utils.surface_properties import compute_surface_properties

# %%
# 1) Build the Central Clump, Smooth Reference, and Spherical Tracer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

central = create_ga_state(
    N=1,
    nv=8,
    dim=3,
    particle_radius=0.5,
    asperity_radius=0.1,
    particle_type="clump",
    core_type="solid",
    n_samples=2_048,
    seed=0,
    mesh_type="fibonacci",
)
tracer_radius = 0.05
tracer = create_sphere_state(radii=tracer_radius, dim=3)
smooth_sphere = create_sphere_state(radii=0.5, dim=3)

# %%
# 2) Probe Contact Separations and Geometric Friction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``n_orientations = n_rolls = 1`` suffices because the spherical tracer is
# rotationally symmetric:

n_points = 192
probe_kw = dict(
    target_overlap=1e-10,
    separation_tolerance=1e-12,
    n_orientations=1,
    n_rolls=1,
)
result = compute_surface_properties(central, tracer, n_points=n_points, **probe_kw)
separation = np.asarray(result["separation"]).reshape(n_points)
mu = np.asarray(result["mu"]).reshape(n_points)
directions = np.asarray(result["approach_directions"])  # (n_points, 3) on S^2

# Triangulate unit directions on S^2 (never the concave surface itself).
triangles = ConvexHull(directions).simplices


def triangle_areas(radii: np.ndarray, dirs: np.ndarray, tris: np.ndarray) -> np.ndarray:
    pts = radii[:, None] * dirs
    v = pts[tris]
    return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=-1)


clump_tri_areas = triangle_areas(separation, directions, triangles)
sasa = float(clump_tri_areas.sum())

# Compare solid-angle mean (equal weight per Fibonacci ray) vs SASA area-weighted mean:
vertex_areas = np.zeros(n_points)
np.add.at(vertex_areas, triangles, clump_tri_areas[:, None] / 3.0)
mean_mu_angle = float(mu.mean())
mean_mu_area = float(np.sum(vertex_areas * mu) / np.sum(vertex_areas))

print(
    f"Separation: min={separation.min():.4f}, max={separation.max():.4f}, "
    f"estimated SASA={sasa:.4f}"
)
print(
    f"Geometric friction mu: max={mu.max():.4f} | "
    f"solid-angle mean={mean_mu_angle:.4f}, area-weighted mean={mean_mu_area:.4f}"
)

# %%
# 3) Validate Against a Smooth Sphere and Check Angular Resolution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A smooth sphere of radius :math:`R = 0.5` has exact SASA
# :math:`4\pi (R + r_{\text{tracer}})^2` and :math:`\mu = 0`:

exact_area = 4.0 * np.pi * (0.5 + tracer_radius) ** 2
angular_counts = np.array([32, 64, 128, n_points])
area_errors = []
for count in angular_counts:
    ref = compute_surface_properties(smooth_sphere, tracer, n_points=int(count), **probe_kw)
    ref_sep = np.asarray(ref["separation"]).reshape(int(count))
    ref_dirs = np.asarray(ref["approach_directions"])
    ref_area = float(triangle_areas(ref_sep, ref_dirs, ConvexHull(ref_dirs).simplices).sum())
    area_errors.append(abs(ref_area / exact_area - 1.0))
    if count == n_points:
        print(
            f"Smooth sphere (N={count}): SASA={ref_area:.4f} vs exact {exact_area:.4f} "
            f"(err={area_errors[-1]:.2%}), max |mu|={float(np.max(np.abs(ref['mu']))):.2e}"
        )

# %%
# Summary Plots: Angular Convergence & Geometric Friction Distribution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")

axes[0].loglog(angular_counts, area_errors, "o-", color="tab:orange")
axes[0].set(
    xticks=angular_counts,
    xticklabels=[str(n) for n in angular_counts],
    xlabel="Fibonacci probe directions",
    ylabel="Relative smooth-sphere SASA error",
    title="Angular discretization convergence",
)
axes[0].minorticks_off()
axes[0].grid(alpha=0.2)

sorted_mu = np.sort(mu)
cdf = np.arange(1, n_points + 1) / n_points
axes[1].step(np.r_[0.0, sorted_mu], np.r_[0.0, cdf], where="post", color="tab:purple", label="Rough clump")
axes[1].axvline(mean_mu_angle, color="tab:orange", ls=":", label=f"Solid-angle mean ({mean_mu_angle:.3f})")
axes[1].axvline(mean_mu_area, color="tab:green", ls="--", label=f"Area-weighted mean ({mean_mu_area:.3f})")
axes[1].set(
    xlabel=r"Geometric force ratio $\mu = |F_t| / |F_n|$",
    ylabel="Cumulative fraction of directions",
    ylim=(0.0, 1.05),
    title="Shape-induced geometric friction",
)
axes[1].legend(fontsize="small")
axes[1].grid(alpha=0.2)
plt.show()
