# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Exact boundary-arc moments for unions of disks."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def _arc_moments(
    center: jax.Array,
    radius: jax.Array,
    start: jax.Array,
    stop: jax.Array,
) -> jax.Array:
    """Return area, first moments, and symmetric second moments per arc.

    Columns are ``(area, x, y, xx, xy, yy)``. The divergence theorem gives
    the moment of degree ``k`` as ``integral(x**k * (x dot n)) / (2 + k)``.
    On a circle, ``x dot n = radius + center dot n``, so only trigonometric
    monomials up to degree three need to be integrated.
    """
    delta = stop - start
    midpoint = 0.5 * (start + stop)
    sin_half = jnp.sin(0.5 * delta)
    int_c = 2.0 * jnp.cos(midpoint) * sin_half
    int_s = 2.0 * jnp.sin(midpoint) * sin_half
    cos_two = jnp.cos(2.0 * midpoint) * jnp.sin(delta)
    sin_two = jnp.sin(2.0 * midpoint) * jnp.sin(delta)
    int_cc = 0.5 * (delta + cos_two)
    int_cs = 0.5 * sin_two
    int_ss = 0.5 * (delta - cos_two)
    cos_three = jnp.cos(3.0 * midpoint) * jnp.sin(1.5 * delta) / 6.0
    sin_three = jnp.sin(3.0 * midpoint) * jnp.sin(1.5 * delta) / 6.0
    int_ccc = 0.75 * int_c + cos_three
    int_ccs = 0.25 * int_s + sin_three
    int_css = 0.25 * int_c - cos_three
    int_sss = 0.75 * int_s - sin_three

    x, y = center[0], center[1]
    r = radius
    int_h = r * delta + x * int_c + y * int_s
    int_ch = r * int_c + x * int_cc + y * int_cs
    int_sh = r * int_s + x * int_cs + y * int_ss
    int_cch = r * int_cc + x * int_ccc + y * int_ccs
    int_csh = r * int_cs + x * int_ccs + y * int_css
    int_ssh = r * int_ss + x * int_css + y * int_sss

    moments = jnp.stack(
        (
            r * int_h / 2.0,
            r * (x * int_h + r * int_ch) / 3.0,
            r * (y * int_h + r * int_sh) / 3.0,
            r * (x * x * int_h + 2.0 * x * r * int_ch + r * r * int_cch) / 4.0,
            r
            * (x * y * int_h + x * r * int_sh + y * r * int_ch + r * r * int_csh)
            / 4.0,
            r * (y * y * int_h + 2.0 * y * r * int_sh + r * r * int_ssh) / 4.0,
        ),
        axis=-1,
    )

    # Entire circles have analytic moments. Besides avoiding roundoff in
    # sin(2*pi), this keeps separated, nested, and coincident disks exact.
    area = jnp.pi * r * r
    full = area * jnp.stack(
        (jnp.ones_like(x), x, y, x * x + r * r / 4.0, x * y, y * y + r * r / 4.0)
    )
    return jnp.where((delta == 2.0 * jnp.pi)[..., None], full, moments)


@partial(jax.jit, static_argnames=("uniform_density",))
def _disk_union_moments(
    positions: jax.Array,
    radii: jax.Array,
    densities: jax.Array,
    *,
    uniform_density: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Integrate a disk union and its maximum-density field exactly.

    Inputs have shapes ``(nv, 2)``, ``(nv,)``, and ``(nv,)``. Return union
    area, mass, density-weighted first moment, and density-weighted second
    moment about the supplied coordinate origin. The caller should center
    the positions near the body to avoid cancellation of large raw moments.

    Pairwise circle intersections partition each boundary into arcs with
    constant coverage. The outward density jump on one arc is the positive
    difference between that disk's density and the largest covering density.
    Equal coincident boundaries are owned in index order: later disks add
    only any increase in density, so their contributions telescope to the
    maximum density. Scanning circles bounds intermediate storage to
    ``O(nv**2)``.

    When ``uniform_density`` is true, all input densities must be identical.
    An angular event sweep determines exposed arcs without an arc-by-disk
    coverage table, reducing work to ``O(nv**2 * log(nv))`` and intermediate
    storage to ``O(nv)``.
    """
    nv = positions.shape[0]
    indices = jnp.arange(nv)
    dtype = positions.dtype
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=dtype)
    radii_squared = radii * radii

    def accumulate(
        i: jax.Array, carry: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        area_total, mass_moments = carry
        center = positions[i]
        radius = radii[i]
        difference = positions - center
        distance_squared = jnp.sum(difference * difference, axis=-1)
        distance = jnp.sqrt(jnp.where(distance_squared > 0.0, distance_squared, 1.0))
        intersects = (
            (distance_squared > 0.0)
            & (distance > jnp.abs(radius - radii))
            & (distance < radius + radii)
        )
        cosine = (radius * radius + distance_squared - radii_squared) / (
            2.0 * radius * distance
        )
        half_width = jnp.arccos(jnp.clip(cosine, -1.0, 1.0))
        direction = jnp.arctan2(
            difference[:, 1],
            jnp.where(distance_squared > 0.0, difference[:, 0], 1.0),
        )
        lower = jnp.where(intersects, jnp.mod(direction - half_width, two_pi), 0.0)
        upper = jnp.where(intersects, jnp.mod(direction + half_width, two_pi), 0.0)
        angles = jnp.concatenate(
            (jnp.zeros((1,), dtype=dtype), lower, upper, two_pi[None])
        )
        order = jnp.argsort(angles)
        endpoints = angles[order]
        start, stop = endpoints[:-1], endpoints[1:]
        center_distance = jnp.where(distance_squared > 0.0, distance, 0.0)
        coincident = (distance_squared == 0.0) & (radii == radius)
        contained = (center_distance + radius <= radii) & ~intersects
        contained = jnp.where(coincident, indices < i, contained) & (indices != i)
        if uniform_density:
            # A crossing disk covers lower -> upper. Wrapped arcs are
            # already active at angle zero. Numerically clipped full- and
            # zero-width caps have constant coverage and need no events.
            event_active = intersects & (half_width > 0.0) & (half_width < jnp.pi)
            enter = event_active.astype(jnp.int32)
            events = jnp.concatenate(
                (jnp.zeros((1,), jnp.int32), enter, -enter, jnp.zeros((1,), jnp.int32))
            )[order]
            initial = (
                jnp.sum(contained)
                + jnp.sum(event_active & (lower > upper))
                + jnp.sum(intersects & (half_width >= jnp.pi))
            )
            coverage = initial + jnp.cumsum(events)[:-1]
            exposed = coverage == 0
            density_jump = jnp.where(exposed, densities[i], 0.0)
        else:
            midpoint = 0.5 * (start + stop)
            normal = jnp.stack((jnp.cos(midpoint), jnp.sin(midpoint)), axis=-1)
            # Test every disk without a (n_arcs, nv, 2) offset tensor.
            projection = normal @ difference.T
            covered = (
                2.0 * radius * projection
                > (radius * radius + distance_squared - radii_squared)[None, :]
            )
            # Without crossings, coverage is constant up to isolated
            # tangent points, including a midpoint exactly at tangency.
            covered = jnp.where(intersects[None, :], covered, contained[None, :])
            # Earlier coincident circles cover later ones; later boundaries
            # add only density increases, telescoping to the maximum.
            covered = jnp.where(coincident[None, :], indices[None, :] < i, covered)
            covered = covered & (indices[None, :] != i)
            exposed = ~jnp.any(covered, axis=-1)
            outside_density = jnp.max(
                jnp.where(covered, densities[None, :], 0.0), axis=-1
            )
            density_jump = jnp.maximum(densities[i] - outside_density, 0.0)

        moments = _arc_moments(center, radius, start, stop)
        return (
            area_total + jnp.sum(jnp.where(exposed, moments[:, 0], 0.0)),
            mass_moments + jnp.sum(moments * density_jump[:, None], axis=0),
        )

    area, moments = jax.lax.fori_loop(
        0,
        nv,
        accumulate,
        (jnp.zeros((), dtype=dtype), jnp.zeros((6,), dtype=dtype)),
    )
    second = jnp.stack(
        (jnp.stack((moments[3], moments[4])), jnp.stack((moments[4], moments[5])))
    )
    return area, moments[0], moments[1:3], second
