"""Smooth-sphere baseline tests for :func:`compute_surface_properties`.

When both the central and tracer particles are perfect single spheres (no
surface asperities, smooth convex bodies) the surface probe reduces to an
analytically known sphere-on-sphere contact:

* At the user-specified target overlap ``delta``, the center-to-center
  separation must be ``r_central + r_tracer - delta`` **independent of the
  approach direction** — a sphere is isotropic.
* The interaction force between two smooth spheres is purely radial (along
  the center-to-center axis), so the tangential force is zero and the
  probe-reported friction ``mu = |F_t| / |F_n|`` must be zero.
* The polyhedral surface area reconstructed from the sampled tracer-center
  positions must converge to the analytical sphere area
  ``4*pi*R**2`` in 3D (perimeter ``2*pi*R`` in 2D), where
  ``R = r_central + r_tracer - delta``.

These conditions are exercised in 2D (circles) and 3D (spheres). The tracer
orientation has no degree of freedom in this symmetric setting, so we
sample only ``n_orientations = 1`` (and ``n_rolls = 1`` in 3D).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils.surfaceProperties import compute_surface_properties

jax.config.update("jax_enable_x64", True)


def _single_sphere_state(radius: float, dim: int) -> jd.State:
    """Build a single-sphere :class:`State` for use as central or tracer.

    The sphere is positioned at the origin; ``compute_surface_properties``
    will translate it internally for each probe. ``clump_id=[0]`` ensures
    the state is treated as a single clump (required by the function).
    """
    pos = jnp.zeros((1, dim), dtype=float)
    return jd.State.create(
        pos=pos,
        rad=jnp.asarray([radius], dtype=float),
        mass=jnp.ones((1,), dtype=float),
        clump_id=jnp.asarray([0], dtype=int),
    )


@pytest.mark.parametrize(
    "dim, r_central, r_tracer, target_overlap, n_points",
    [
        (2, 0.5, 0.3, 1e-10, 64),
        (2, 0.4, 0.2, 1e-10, 128),
        (3, 0.5, 0.3, 1e-10, 96),
        (3, 0.4, 0.2, 1e-10, 144),
    ],
)
def test_smooth_sphere_separation_is_uniform(
    dim: int,
    r_central: float,
    r_tracer: float,
    target_overlap: float,
    n_points: int,
) -> None:
    """Separation must equal ``r_central + r_tracer - target_overlap`` at every probe."""
    central = _single_sphere_state(r_central, dim)
    tracer = _single_sphere_state(r_tracer, dim)

    result = compute_surface_properties(
        central,
        tracer,
        target_overlap=target_overlap,
        n_points=n_points,
        n_orientations=1,
        n_rolls=1,
        separation_tolerance=1e-12,
    )

    separation = np.asarray(result["separation"])
    expected = r_central + r_tracer - target_overlap

    # The bisection converges to max-overlap == target_overlap; the
    # separation is monotone in max-overlap for smooth spheres, so the
    # error is bounded by ``separation_tolerance`` (not target_overlap).
    assert np.all(np.isfinite(separation)), "non-finite separation returned"
    np.testing.assert_allclose(separation, expected, atol=1e-8, rtol=0)


@pytest.mark.parametrize(
    "dim, r_central, r_tracer, target_overlap, n_points",
    [
        (2, 0.5, 0.3, 1e-10, 64),
        (2, 0.4, 0.2, 1e-10, 128),
        (3, 0.5, 0.3, 1e-10, 96),
        (3, 0.4, 0.2, 1e-10, 144),
    ],
)
def test_smooth_sphere_friction_is_zero(
    dim: int,
    r_central: float,
    r_tracer: float,
    target_overlap: float,
    n_points: int,
) -> None:
    """Smooth-sphere contact is purely radial -> mu = |F_t| / |F_n| = 0."""
    central = _single_sphere_state(r_central, dim)
    tracer = _single_sphere_state(r_tracer, dim)

    result = compute_surface_properties(
        central,
        tracer,
        target_overlap=target_overlap,
        n_points=n_points,
        n_orientations=1,
        n_rolls=1,
        separation_tolerance=1e-12,
    )

    mu = np.asarray(result["mu"])

    assert np.all(np.isfinite(mu)), "non-finite mu returned"
    # Tolerance reflects the numerical floor of the tangential-force
    # decomposition at the converged separation; smooth-sphere mu should
    # be zero up to normal-direction round-off.
    np.testing.assert_allclose(mu, 0.0, atol=1e-8)


@pytest.mark.parametrize(
    "dim, r_central, r_tracer, target_overlap, n_points, rtol",
    [
        # Perimeter converges as 1/n_points^2 for a regular polygon
        # inscribed in a circle; 512 samples give ~ppm relative error.
        (2, 0.5, 0.3, 1e-10, 512, 1e-4),
        (2, 0.4, 0.2, 1e-10, 512, 1e-4),
        # Surface area converges as 1/n_points for a convex polyhedron
        # inscribed in a sphere; 2048 Fibonacci samples give ~0.2% at
        # these radii, so 0.5% leaves a comfortable margin.
        (3, 0.5, 0.3, 1e-10, 2048, 5e-3),
        (3, 0.4, 0.2, 1e-10, 2048, 5e-3),
    ],
)
def test_smooth_sphere_sasa_from_separation(
    dim: int,
    r_central: float,
    r_tracer: float,
    target_overlap: float,
    n_points: int,
    rtol: float,
) -> None:
    """Polyhedral SASA reconstructed from the probe's separations converges
    to ``4*pi*R**2`` in 3D / ``2*pi*R`` in 2D, with ``R = r_central +
    r_tracer - target_overlap``.

    Method (matches the SASA example):
      * 3D: triangulate the approach directions on ``S^2`` (their 3D
        convex hull), lift triangle vertices to the SASA surface using the
        bisection's per-direction separations, and sum Euclidean triangle
        areas.
      * 2D: approach directions are equispaced on ``S^1``, so connecting
        consecutive lifted samples in order gives the inscribed polygon;
        its perimeter is the 2D analogue of SASA.
    """
    from scipy.spatial import ConvexHull

    central = _single_sphere_state(r_central, dim)
    tracer = _single_sphere_state(r_tracer, dim)

    result = compute_surface_properties(
        central,
        tracer,
        target_overlap=target_overlap,
        n_points=n_points,
        n_orientations=1,
        n_rolls=1,
        separation_tolerance=1e-12,
    )
    directions = np.asarray(result["approach_directions"])
    separation = np.asarray(result["separation"]).reshape(n_points)
    surface_points = separation[:, None] * directions

    R_sasa = r_central + r_tracer - target_overlap

    if dim == 3:
        triangles = ConvexHull(directions).simplices
        tri = surface_points[triangles]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        sasa = 0.5 * float(np.sum(np.linalg.norm(np.cross(e1, e2), axis=-1)))
        expected = 4.0 * np.pi * R_sasa**2
    else:
        # 2D: approach directions from ``_sample_directions`` are in CCW
        # order (equispaced polar angles starting at 0), so connecting
        # consecutive samples gives an inscribed regular polygon.
        closed = np.concatenate([surface_points, surface_points[:1]], axis=0)
        edges = np.linalg.norm(np.diff(closed, axis=0), axis=-1)
        sasa = float(edges.sum())
        expected = 2.0 * np.pi * R_sasa

    # Polygon/polyhedron inscribed in a sphere under-estimates the true
    # surface; assert convergence from below within the expected
    # discretization tolerance.
    assert sasa <= expected + 1e-12
    np.testing.assert_allclose(sasa, expected, rtol=rtol, atol=0)
