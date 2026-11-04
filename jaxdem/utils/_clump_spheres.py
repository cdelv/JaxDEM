# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Analytical overlap moments for two spheres."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _cap_moments(
    center: jax.Array,
    direction: jax.Array,
    radius: jax.Array,
    height: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Integrate one spherical cap's raw volume, first, and second moments."""
    # Integrate the smaller cap and use complements for caps larger than a
    # hemisphere. This avoids cancellation in fifth-degree polynomials near
    # a complete sphere.
    large = height > radius
    h = jnp.where(large, 2.0 * radius - height, height)
    r = radius
    volume_small = jnp.pi * h * h * (r - h / 3.0)
    ratio = h / r
    axial_small = (
        jnp.pi
        * r**3
        * h**2
        * (1.0 + ratio * (-5.0 / 3.0 + ratio * (1.0 - ratio / 5.0)))
    )
    transverse_small = (
        jnp.pi * r**2 * h**3 * (1.0 / 3.0 + ratio * (-1.0 / 4.0 + ratio / 20.0))
    )
    volume = jnp.where(large, 4.0 * jnp.pi * r**3 / 3.0 - volume_small, volume_small)
    full_second = 4.0 * jnp.pi * r**5 / 15.0
    axial = jnp.where(large, full_second - axial_small, axial_small)
    transverse = jnp.where(large, full_second - transverse_small, transverse_small)
    first_axis = jnp.pi * (height * (2.0 * r - height)) ** 2 / 4.0

    axis_outer = direction[..., :, None] * direction[..., None, :]
    center_outer = center[..., :, None] * center[..., None, :]
    mixed = (
        center[..., :, None] * direction[..., None, :]
        + direction[..., :, None] * center[..., None, :]
    )
    first = volume[..., None] * center + first_axis[..., None] * direction
    second = (
        volume[..., None, None] * center_outer
        + first_axis[..., None, None] * mixed
        + axial[..., None, None] * axis_outer
        + transverse[..., None, None] * (jnp.eye(3, dtype=center.dtype) - axis_outer)
    )
    return volume, first, second


@jax.jit
def _two_sphere_overlap_moments(
    positions: jax.Array,
    radii: jax.Array,
    densities: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return exact partial-overlap corrections for two sphere unions.

    Inputs have shapes ``(N, 2, 3)``, ``(N, 2)``, and ``(N, 2)``. Return
    intersection volume and its minimum-density-weighted mass, first moment,
    and second moment, about the supplied origin. Subtracting these from the
    individual sphere moments produces the maximum-density union moments.

    Containment, coincidence, disjointness, and tangency return zeros because
    those configurations are handled by the caller's analytical baseline.
    Proper intersections split into two caps along the common circle plane.
    """
    first_center, second_center = positions[:, 0, :], positions[:, 1, :]
    first_radius, second_radius = radii[:, 0], radii[:, 1]
    delta = second_center - first_center
    distance_squared = jnp.sum(delta * delta, axis=-1)
    safe_distance = jnp.sqrt(jnp.where(distance_squared > 0.0, distance_squared, 1.0))
    distance = jnp.where(distance_squared > 0.0, safe_distance, 0.0)
    proper = (distance > jnp.abs(first_radius - second_radius)) & (
        distance < first_radius + second_radius
    )
    direction = delta / safe_distance[:, None]
    gap = first_radius + second_radius - distance
    first_height = (
        gap * (distance + second_radius - first_radius) / (2.0 * safe_distance)
    )
    second_height = (
        gap * (distance + first_radius - second_radius) / (2.0 * safe_distance)
    )
    first_height = jnp.where(
        proper, jnp.clip(first_height, 0.0, 2.0 * first_radius), 0.0
    )
    second_height = jnp.where(
        proper, jnp.clip(second_height, 0.0, 2.0 * second_radius), 0.0
    )
    volume_a, first_a, second_a = _cap_moments(
        first_center, direction, first_radius, first_height
    )
    volume_b, first_b, second_b = _cap_moments(
        second_center, -direction, second_radius, second_height
    )
    volume = volume_a + volume_b
    density = jnp.minimum(densities[:, 0], densities[:, 1])
    return (
        volume,
        density * volume,
        density[:, None] * (first_a + first_b),
        density[:, None, None] * (second_a + second_b),
    )
