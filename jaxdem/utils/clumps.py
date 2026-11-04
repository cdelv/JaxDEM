# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Mass properties of unions of disks and spheres with maximum overlap density."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import replace
from functools import partial
import operator
from typing import TYPE_CHECKING, Any

import numpy as np

from ._clump_arcs import _disk_union_moments
from ._clump_axes import _principal_frame
from ._clump_spheres import _two_sphere_overlap_moments
from .quaternion import Quaternion

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..materials import MaterialTable


def _phase_fraction(indices: jax.Array, steps: jax.Array, dtype: Any) -> jax.Array:
    """Multiply phases modulo 2**32 before converting to floating point."""
    phase = indices.astype(jnp.uint32) * steps + jnp.uint32(0x80000000)
    if jnp.dtype(dtype) == jnp.float32:
        return ((phase >> 8).astype(dtype) + 0.5) * (1.0 / 2**24)
    return phase.astype(dtype) * (1.0 / 2**32)


@partial(jax.jit, static_argnames=("n", "dim"))
def _generate_golden_lattice(n: int, dim: int = 2) -> jax.Array:
    """Generate a fixed-point lattice without large-index float32 cancellation."""
    return _golden_lattice_points(
        jnp.arange(n, dtype=jnp.uint32), dim, jnp.asarray(0.0, dtype=float).dtype
    )


def _golden_lattice_points(indices: jax.Array, dim: int, dtype: Any) -> jax.Array:
    """Generate the same lattice points regardless of sample chunk boundaries."""
    phi = {2: 1.32471795724475, 3: 1.22074408460576}[dim]
    steps = jnp.asarray([round(2**32 / phi ** (i + 1)) for i in range(dim)], jnp.uint32)
    return _phase_fraction(indices.astype(jnp.uint32)[:, None] + 1, steps, dtype)


@partial(jax.jit, static_argnames=("n_directions", "dtype"))
def _fibonacci_normals(indices: jax.Array, n_directions: int, dtype: Any) -> jax.Array:
    """Return indexed antithetic Fibonacci sphere normals, generated in chunks."""
    pair = indices // 2
    z = (pair.astype(dtype) + 0.5) / (n_directions // 2)
    angle = (2.0 * jnp.pi) * _phase_fraction(pair, jnp.uint32(0x9E3779B9), dtype)
    transverse = jnp.sqrt(jnp.maximum(0.0, 1.0 - z * z))
    normals = jnp.stack(
        (transverse * jnp.cos(angle), transverse * jnp.sin(angle), z), -1
    )
    return jnp.where((indices % 2 == 0)[..., None], normals, -normals)


def _validate_budgets(
    n_samples: int, sample_batch_size: int, clump_batch_size: int
) -> None:
    for name, value in (
        ("n_samples", n_samples),
        ("sample_batch_size", sample_batch_size),
        ("clump_batch_size", clump_batch_size),
    ):
        try:
            integer = operator.index(value)
        except TypeError as error:
            raise TypeError(f"{name} must be a positive integer.") from error
        if isinstance(value, (bool, np.bool_)) or integer <= 0:
            raise ValueError(f"{name} must be a positive integer.")


@jax.jit
def _geometry_validity(pos: jax.Array, rad: jax.Array, density: jax.Array) -> jax.Array:
    return jnp.stack(
        (
            jnp.all(jnp.isfinite(pos)),
            jnp.all(jnp.isfinite(rad) & (rad > 0.0)),
            jnp.all(jnp.isfinite(density) & (density > 0.0)),
        )
    )


def _validate_geometry(pos: jax.Array, rad: jax.Array, density: jax.Array) -> None:
    valid = np.asarray(_geometry_validity(pos, rad, density))
    for good, message in zip(
        valid,
        (
            "All positions must be finite.",
            "All radii must be finite and strictly positive.",
            "All densities or clump masses must be finite and strictly positive.",
        ),
    ):
        if not good:
            raise ValueError(message)


@jax.jit
def _properties_are_physical(properties: tuple[jax.Array, ...]) -> jax.Array:
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in properties]))
    return (
        finite
        & jnp.all(properties[0] > 0.0)
        & jnp.all(properties[1] > 0.0)
        & jnp.all(properties[3] > 0.0)
    )


def _validate_properties(properties: tuple[jax.Array, ...]) -> None:
    if not bool(_properties_are_physical(properties)):
        raise ValueError(
            "Union integration did not resolve positive finite mass properties; "
            "increase n_samples or use higher precision."
        )


def _compute_uniform_union_properties(
    pos: jax.Array,
    rad: jax.Array,
    clump_mass: jax.Array,
    n_samples: int = 50_000,
    sample_batch_size: int = 4096,
    clump_batch_size: int = 32,
) -> tuple[jax.Array, ...]:
    """Return volume, COM, principal moments, quaternion, and body-frame centers.

    Each member is a filled disk (2D) or ball (3D). The prescribed total mass
    is distributed uniformly over their union, counting overlaps once. Empty
    gaps and enclosed cavities outside that union remain empty: the utility
    does not fill a convex hull or infer an interior core. The supplied
    centers and radii alone define which regions contain material.
    Inputs have shapes
    ``pos=(N,nv,dim)`` or ``(nv,dim)``, ``rad=(N,nv)``, ``(nv,)`` or scalar,
    and ``clump_mass=(N,)`` or scalar. Outputs retain the leading body axis
    unless the positions describe a single body.

    Disk unions use exact circular-arc integrals. Disjoint, contained, nested,
    and two-sphere unions are analytical. Small clumps with at most 32 members
    and no overlap-graph triangles also use exact pair corrections. Other 3D
    configurations choose numerical boundary or volume integration according
    to overlap density. The dense-volume path avoids relying on tiny exposed
    boundary patches when many members overlap. Arbitrarily many simultaneous
    overlaps are supported; the numerical paths do not truncate the
    calculation at pairwise overlap terms.

    ``n_samples`` sets the 3D numerical integration budget, with a minimum
    resolution of 32 points per member. Boundary integration rounds up to an
    even number per member. The budget has no effect in 2D or analytical 3D
    cases. Samples are generated in bounded chunks; chunk sizes do not change
    the rule. For numerical 3D cases, compare volume, COM, and the reconstructed
    world-frame inertia tensor at increasing budgets. Positive finite output
    is not an accuracy certificate. Unresolved nonpositive properties raise a
    ValueError.

    The returned unit quaternion uses ``[w,x,y,z]`` and rotates body coordinates
    into world coordinates. Its signs and degenerate principal axes have a
    deterministic convention. Exact rotational symmetry has no unique physical
    orientation, and nearly repeated moments make individual axes sensitive
    to numerical error. This host-validated utility is not itself a jittable
    function.
    """
    _validate_budgets(n_samples, sample_batch_size, clump_batch_size)
    pos = jnp.asarray(pos, dtype=float)
    single_clump = pos.ndim == 2
    if single_clump:
        pos = pos[None, ...]
    if pos.ndim != 3 or pos.shape[-1] not in (2, 3) or pos.shape[-2] == 0:
        raise ValueError(
            "pos must have shape (N, nv, dim) or (nv, dim), nv > 0, dim=2 or 3."
        )
    n_clumps, nv, _dim = pos.shape
    rad = jnp.asarray(rad, dtype=pos.dtype)
    if single_clump and rad.shape == (nv, 1):
        rad = rad.reshape(1, nv)
    if rad.ndim not in (0, 1, 2) or (rad.ndim == 1 and rad.shape != (nv,)):
        raise ValueError("rad must be scalar, (nv,), or (N, nv).")
    try:
        rad = jnp.broadcast_to(rad, (n_clumps, nv))
        clump_mass = jnp.broadcast_to(jnp.asarray(clump_mass, pos.dtype), (n_clumps,))
    except ValueError as error:
        raise ValueError("rad or clump_mass has incompatible shape.") from error
    _validate_geometry(pos, rad, clump_mass)
    result = _union_properties_kernel(
        pos,
        rad,
        jnp.ones_like(rad),
        clump_mass,
        n_samples,
        sample_batch_size,
        clump_batch_size,
    )
    _validate_properties(result)
    volume, _mass, com, inertia, q, pos_p = result
    properties = volume, com, inertia, q, pos_p
    return tuple(value[0] for value in properties) if single_clump else properties


def _sphere_boundary_corrections(
    positions: jax.Array,
    radii: jax.Array,
    densities: jax.Array,
    partial: jax.Array,
    container_density: jax.Array,
    volume_coefficient: jax.Array,
    density_coefficient: jax.Array,
    n_samples: int,
    sample_batch_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Integrate the boundary portions removed from analytical sphere moments."""
    nv = positions.shape[0]
    n_directions = 2 * max(16, (n_samples + 2 * nv - 1) // (2 * nv))
    total = nv * n_directions
    batch = min(sample_batch_size, total)
    batches = (total + batch - 1) // batch
    offsets = jnp.arange(batch)
    area_per_point = 4.0 * jnp.pi * radii * radii / n_directions
    difference = positions[None, :, :] - positions[:, None, :]
    cap_offset = (
        radii[:, None] ** 2
        + jnp.sum(difference * difference, axis=-1)
        - radii[None, :] ** 2
    ) / (2.0 * radii[:, None])

    def accumulate(
        carry: tuple[jax.Array, ...], batch_index: jax.Array
    ) -> tuple[tuple[jax.Array, ...], None]:
        index = batch_index * batch + offsets
        source = jnp.minimum(index // n_directions, nv - 1)
        normals = _fibonacci_normals(
            index % n_directions, n_directions, positions.dtype
        )
        points = positions[source] + radii[source, None] * normals
        # A spherical overlap cuts a cap bounded by a plane in normal space.
        # Its precomputed offset avoids per-sample point-distance tensors.
        projection = jnp.einsum("si,sji->sj", normals, difference[source])
        covered = (projection > cap_offset[source]) & partial[source]
        outside_density = jnp.maximum(
            container_density[source],
            jnp.max(jnp.where(covered, densities[None, :], 0.0), axis=-1),
        )
        removed_density = density_coefficient[source] - jnp.maximum(
            densities[source] - outside_density, 0.0
        )
        removed_volume = volume_coefficient[source] & jnp.any(covered, axis=-1)
        flux = jnp.sum(points * normals, axis=-1) * area_per_point[source]
        flux = jnp.where(index < total, flux, 0.0)
        weighted = flux * removed_density
        increments = (
            jnp.sum(jnp.where(removed_volume, flux / 3.0, 0.0)),
            jnp.sum(weighted / 3.0),
            jnp.sum((weighted / 4.0)[:, None] * points, axis=0),
            jnp.einsum("n,ni,nj->ij", weighted / 5.0, points, points),
        )
        return tuple(old + new for old, new in zip(carry, increments)), None

    initial: tuple[jax.Array, ...] = (
        jnp.zeros((), positions.dtype),
        jnp.zeros((), positions.dtype),
        jnp.zeros((3,), positions.dtype),
        jnp.zeros((3, 3), positions.dtype),
    )
    corrections = jax.lax.scan(accumulate, initial, jnp.arange(batches))[0]
    return corrections[0], corrections[1], corrections[2], corrections[3]


def _dense_sphere_union_moments(
    positions: jax.Array,
    radii: jax.Array,
    densities: jax.Array,
    n_samples: int,
    sample_batch_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Integrate dense unions directly, avoiding tiny exposed-patch aliasing."""
    lower = jnp.min(positions - radii[:, None], axis=0)
    extent = jnp.max(positions + radii[:, None], axis=0) - lower
    count = max(n_samples, 32 * positions.shape[0])
    batch = min(sample_batch_size, count)
    offsets = jnp.arange(batch)

    def accumulate(
        carry: tuple[jax.Array, ...], step: jax.Array
    ) -> tuple[tuple[jax.Array, ...], None]:
        indices = step * batch + offsets
        points = lower + _golden_lattice_points(indices, 3, positions.dtype) * extent
        delta = points[:, None, :] - positions[None, :, :]
        covered = jnp.sum(delta * delta, axis=-1) < radii[None, :] ** 2
        valid = indices < count
        inside = jnp.any(covered, axis=-1) & valid
        rho = jnp.where(
            valid, jnp.max(jnp.where(covered, densities[None, :], 0.0), axis=-1), 0.0
        )
        increments = (
            jnp.sum(inside.astype(positions.dtype)),
            jnp.sum(rho),
            jnp.sum(rho[:, None] * points, axis=0),
            jnp.einsum("n,ni,nj->ij", rho, points, points),
        )
        return tuple(old + new for old, new in zip(carry, increments)), None

    initial: tuple[jax.Array, ...] = (
        jnp.zeros((), positions.dtype),
        jnp.zeros((), positions.dtype),
        jnp.zeros((3,), positions.dtype),
        jnp.zeros((3, 3), positions.dtype),
    )
    moments = jax.lax.scan(
        accumulate, initial, jnp.arange((count + batch - 1) // batch)
    )[0]
    measure = jnp.prod(extent) / count
    return (
        moments[0] * measure,
        moments[1] * measure,
        moments[2] * measure,
        moments[3] * measure,
    )


def _sphere_union_moments(
    positions: jax.Array,
    radii: jax.Array,
    densities: jax.Array,
    n_samples: int,
    sample_batch_size: int,
) -> tuple[jax.Array, ...]:
    """Compute batched moments, retaining intrinsic isotropic inertia separately."""
    n, nv, _ = positions.shape
    difference = positions[:, :, None, :] - positions[:, None, :, :]
    distance = jnp.sqrt(jnp.sum(difference * difference, axis=-1))
    ri, rj = radii[:, :, None], radii[:, None, :]
    indices = jnp.arange(nv)
    distinct = indices[:, None] != indices[None, :]
    coincident = (distance == 0.0) & (ri == rj)
    contains = (
        (distance + ri <= rj)
        & distinct
        & (~coincident | (indices[None, :] < indices[:, None]))
    )
    partial = (distance < ri + rj) & (distance > jnp.abs(ri - rj)) & distinct
    container_density = jnp.max(
        jnp.where(contains, densities[:, None, :], 0.0), axis=-1
    )
    volume_coefficient = ~jnp.any(contains, axis=-1)
    density_coefficient = jnp.maximum(densities - container_density, 0.0)
    sphere_volume = (4.0 * jnp.pi / 3.0) * radii**3
    weights = sphere_volume * density_coefficient
    mass = jnp.sum(weights, axis=-1)
    # Center the analytic baseline before subtracting the small boundary
    # corrections. This avoids world-scale parallel-axis cancellation.
    shift = jnp.sum(weights[..., None] * positions, axis=1) / mass[:, None]
    positions = positions - shift[:, None, :]
    volume = jnp.sum(sphere_volume * volume_coefficient, axis=-1)
    first = jnp.sum(weights[..., None] * positions, axis=1)
    second = jnp.einsum("ni,nij,nik->njk", weights, positions, positions)
    isotropic_second = jnp.sum(weights * radii**2 / 5.0, axis=-1)
    relevant = (volume_coefficient | (density_coefficient > 0.0)) & jnp.any(
        partial, axis=-1
    )
    has_partial = jnp.any(relevant, axis=-1)
    # A heavily overlapped union can expose only a thin belt on each member.
    # Reusing one angular rule on those belts causes coherent quadrature
    # error. Prefer direct volume integration when overlap is high and the
    # largest member alone guarantees a useful bounding-box occupancy.
    if nv > 2:
        lower = jnp.min(positions - radii[..., None], axis=1)
        upper = jnp.max(positions + radii[..., None], axis=1)
        box_volume = jnp.prod(upper - lower, axis=-1)
        dense = (jnp.sum(sphere_volume, axis=-1) > 2.0 * box_volume) & (
            jnp.max(sphere_volume, axis=-1) >= 0.05 * box_volume
        )
    else:
        dense = jnp.zeros((n,), dtype=bool)

    def integrate() -> tuple[jax.Array, ...]:
        return jax.vmap(
            _sphere_boundary_corrections, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None)
        )(
            positions,
            radii,
            densities,
            partial,
            container_density,
            volume_coefficient,
            density_coefficient,
            n_samples,
            sample_batch_size,
        )

    def zero_corrections() -> tuple[jax.Array, ...]:
        return (
            jnp.zeros((n,), positions.dtype),
            jnp.zeros((n,), positions.dtype),
            jnp.zeros((n, 3), positions.dtype),
            jnp.zeros((n, 3, 3), positions.dtype),
        )

    if nv == 2:
        corrections = _two_sphere_overlap_moments(positions, radii, densities)
    elif 2 < nv <= 32:
        # No triple intersection is possible without an overlap-graph
        # triangle. Pairwise inclusion-exclusion is exact for small sparse
        # clumps. Bound this shortcut to avoid cubic work for large clumps.
        adjacent = (distance < ri + rj) & distinct
        paths = adjacent.astype(jnp.int32) @ adjacent.astype(jnp.int32)
        triangle_free = ~jnp.any((paths > 0) & adjacent, axis=(1, 2))
        pairs = np.stack(np.triu_indices(nv, k=1), axis=-1)
        pair_count = len(pairs)

        def pair_corrections() -> tuple[jax.Array, ...]:
            result = _two_sphere_overlap_moments(
                positions[:, pairs].reshape((-1, 2, 3)),
                radii[:, pairs].reshape((-1, 2)),
                densities[:, pairs].reshape((-1, 2)),
            )
            return tuple(
                value.reshape((n, pair_count, *value.shape[1:])).sum(axis=1)
                for value in result
            )

        analytic = jax.lax.cond(
            jnp.any(triangle_free & has_partial), pair_corrections, zero_corrections
        )
        sampled = jax.lax.cond(
            jnp.any(~triangle_free & has_partial & ~dense), integrate, zero_corrections
        )
        use_dense = dense & ~triangle_free & has_partial
        corrections = tuple(
            jnp.where(triangle_free.reshape((n,) + (1,) * (a.ndim - 1)), a, b)
            for a, b in zip(analytic, sampled)
        )
    else:
        corrections = jax.lax.cond(
            jnp.any(has_partial & ~dense), integrate, zero_corrections
        )
        use_dense = dense & has_partial
    dv, dm, df, ds = corrections
    volume, mass, first, second = volume - dv, mass - dm, first - df, second - ds
    point_psd = ~has_partial
    if nv > 2:
        direct = jax.lax.cond(
            jnp.any(use_dense),
            lambda: jax.vmap(
                _dense_sphere_union_moments, in_axes=(0, 0, 0, None, None)
            )(positions, radii, densities, n_samples, sample_batch_size),
            zero_corrections,
        )
        volume, mass, first, second = tuple(
            jnp.where(use_dense.reshape((n,) + (1,) * (a.ndim - 1)), a, b)
            for a, b in zip(direct, (volume, mass, first, second))
        )
        isotropic_second = jnp.where(use_dense, 0.0, isotropic_second)
        point_psd = point_psd | use_dense
    return (
        volume,
        mass,
        first,
        second,
        isotropic_second,
        shift,
        point_psd,
    )


def _property_batch(
    positions: jax.Array,
    radii: jax.Array,
    densities: jax.Array,
    target_mass: jax.Array | None,
    n_samples: int,
    sample_batch_size: int,
) -> tuple[jax.Array, ...]:
    """Integrate and diagonalize one bounded body batch."""
    n, _nv, dim = positions.shape
    origin = positions[:, 0, :]
    local = positions - origin[:, None, :]
    if dim == 2:
        weights = jnp.pi * radii**2 * densities
        shift = (
            jnp.sum(weights[..., None] * local, axis=1)
            / jnp.sum(weights, axis=1)[:, None]
        )
        local = local - shift[:, None, :]
        origin = origin + shift
        # Put this condition outside vmap so a uniform-density body batch
        # executes only the cheaper coverage-event sweep.
        volume, mass, first, second = jax.lax.cond(
            jnp.all(densities == densities[:, :1]),
            lambda: jax.vmap(partial(_disk_union_moments, uniform_density=True))(
                local, radii, densities
            ),
            lambda: jax.vmap(_disk_union_moments)(local, radii, densities),
        )
        com = first / mass[:, None]
        covariance = second - mass[:, None, None] * com[:, :, None] * com[:, None, :]
        polar = covariance[:, 0, 0] + covariance[:, 1, 1]
        anisotropy_x = covariance[:, 0, 0] - covariance[:, 1, 1]
        anisotropy_y = 2.0 * covariance[:, 1, 0]
        degenerate = jnp.hypot(anisotropy_x, anisotropy_y) <= (
            64.0 * np.finfo(np.dtype(local.dtype)).eps * jnp.abs(polar)
        )
        angle = jnp.where(
            degenerate, 0.0, 0.5 * jnp.arctan2(anisotropy_y, anisotropy_x)
        )
        q = jnp.stack(
            (
                jnp.cos(angle / 2.0),
                jnp.zeros(n, local.dtype),
                jnp.zeros(n, local.dtype),
                jnp.sin(angle / 2.0),
            ),
            axis=-1,
        )
        moments = polar[:, None]
    else:
        volume, mass, first, second, intrinsic, shift, exact = _sphere_union_moments(
            local, radii, densities, n_samples, sample_batch_size
        )
        local = local - shift[:, None, :]
        origin = origin + shift
        com = first / mass[:, None]
        covariance = second - mass[:, None, None] * com[:, :, None] * com[:, None, :]
        # Sum the other two diagonal terms explicitly. trace(C)-Cii can
        # erase intrinsic moments of very thin bodies in float32.
        diagonal = jnp.stack(
            (
                covariance[:, 1, 1] + covariance[:, 2, 2],
                covariance[:, 0, 0] + covariance[:, 2, 2],
                covariance[:, 0, 0] + covariance[:, 1, 1],
            ),
            axis=-1,
        )
        tensor = -covariance
        tensor = tensor.at[:, jnp.arange(3), jnp.arange(3)].set(diagonal)
        moments, q = _principal_frame(tensor, 2.0 * intrinsic, exact)
        if radii.shape[1] == 2:
            # Two weighted sphere centers lie on one axis. Their point-mass
            # inertia about that axis is exactly zero, even if eigh leaves a
            # positive roundoff residual larger than the intrinsic moment.
            moments = moments.at[:, 0].set(
                jnp.where(exact, 2.0 * intrinsic, moments[:, 0])
            )
    actual_mass = mass if target_mass is None else target_mass
    moments = moments * (actual_mass / mass)[:, None]
    quaternion = Quaternion(q[:, None, 0:1], q[:, None, 1:])
    body = Quaternion.rotate_back(quaternion, local - com[:, None, :])
    return volume, actual_mass, com + origin, moments, q, body


@partial(
    jax.jit, static_argnames=("n_samples", "sample_batch_size", "clump_batch_size")
)
def _union_properties_kernel(
    pos: jax.Array,
    rad: jax.Array,
    vertex_density: jax.Array,
    clump_mass: jax.Array | None,
    n_samples: int,
    sample_batch_size: int,
    clump_batch_size: int,
) -> tuple[jax.Array, ...]:
    """Bounded union integration; overlapping density is the maximum density."""
    n, nv, dim = pos.shape
    if n == 0:
        return (
            jnp.zeros((0,), pos.dtype),
            jnp.zeros((0,), pos.dtype),
            jnp.zeros((0, dim), pos.dtype),
            jnp.zeros((0, 1 if dim == 2 else 3), pos.dtype),
            jnp.zeros((0, 4), pos.dtype),
            jnp.zeros_like(pos),
        )
    if n <= clump_batch_size:
        return _property_batch(
            pos, rad, vertex_density, clump_mass, n_samples, sample_batch_size
        )
    pad = (-n) % clump_batch_size

    def chunks(value: jax.Array) -> jax.Array:
        if pad:
            value = jnp.concatenate(
                (value, jnp.broadcast_to(value[:1], (pad, *value.shape[1:])))
            )
        return value.reshape((-1, clump_batch_size, *value.shape[1:]))

    inputs = (chunks(pos), chunks(rad), chunks(vertex_density))
    if clump_mass is None:
        result = jax.lax.map(
            lambda arrays: _property_batch(
                arrays[0], arrays[1], arrays[2], None, n_samples, sample_batch_size
            ),
            inputs,
        )
    else:
        result = jax.lax.map(
            lambda arrays: _property_batch(
                arrays[0], arrays[1], arrays[2], arrays[3], n_samples, sample_batch_size
            ),
            (*inputs, chunks(clump_mass)),
        )
    return tuple(value.reshape((n + pad, *value.shape[2:]))[:n] for value in result)


def compute_clump_properties(
    state: State,
    mat_table: MaterialTable,
    n_samples: int = 50_000,
    sample_batch_size: int = 4096,
    clump_batch_size: int = 32,
) -> State:
    """Compute rigid union properties for multi-member clumps in an unbatched state.

    Each member is a filled disk or ball with volumetric material density
    ``mat_table.density[state.mat_id]``. The density at any covered point is
    the maximum density of its containing members, without double counting.
    Gaps and enclosed cavities outside the union stay empty. The supplied
    centers and radii alone define the material geometry; omit a central
    sphere when the clump should have no material there. Zero/negative-density
    members cannot be used to subtract a cavity.

    Disk unions are exact. Sphere unions use analytical cases where possible
    and otherwise choose boundary or volume quadrature according to overlap
    density. The dense-volume path avoids dependence on tiny exposed boundary
    patches. Simultaneous triple and higher-order overlaps are supported.
    ``n_samples`` controls the 3D numerical budget; the two batch sizes bound
    intermediate memory without changing the rule. Check convergence using
    volume, COM, and the world-frame inertia tensor at increasing budgets.
    Positive finite output alone does not certify accuracy.

    Principal frames use deterministic signs and handle degenerate moments.
    Nearly equal principal moments still make individual axes sensitive to
    integration error. Single-member clumps retain all existing state values.

    Host grouping uses compact IDs and separate batches for each member count,
    so gaps in IDs or one unusually large clump do not create padded work for
    all other clumps. This grouping/validation step cannot be wrapped in jit;
    the integration itself is compiled by JAX.
    """
    _validate_budgets(n_samples, sample_batch_size, clump_batch_size)
    if state.clump_id.ndim != 1:
        raise ValueError("compute_clump_properties requires an unbatched state.")
    if state.N == 0:
        return state
    ids = np.asarray(state.clump_id)
    _groups, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    if not np.any(counts > 1):
        return state
    pos = state.pos
    density = mat_table.density[state.mat_id]
    _validate_geometry(pos, state.rad, density)
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(counts) - counts
    selected = []
    properties: list[tuple[jax.Array, ...]] = []
    for nv in np.unique(counts[counts > 1]):
        groups = np.flatnonzero(counts == nv)
        selection = order[starts[groups, None] + np.arange(nv)[None, :]]
        result = _union_properties_kernel(
            pos[selection],
            state.rad[selection],
            density[selection],
            None,
            n_samples,
            sample_batch_size,
            clump_batch_size,
        )
        volume, mass, com, inertia, q, body = result

        def members(value: jax.Array) -> jax.Array:
            return jnp.repeat(value, int(nv), axis=0)

        selected.append(selection.reshape(-1))
        properties.append(
            (
                members(volume),
                members(mass),
                members(com),
                members(inertia),
                members(q),
                body.reshape((-1, state.dim)),
            )
        )
    index = np.concatenate(selected)
    volume, mass, com, inertia, q, body = (
        jnp.concatenate(items, axis=0) for items in zip(*properties)
    )
    _validate_properties((volume, mass, com, inertia, q, body))
    return replace(
        state,
        volume=state.volume.at[index].set(volume),
        mass=state.mass.at[index].set(mass),
        pos_c=state.pos_c.at[index].set(com),
        inertia=state.inertia.at[index].set(inertia),
        q=Quaternion(
            state.q.w.at[index].set(q[:, 0:1]), state.q.xyz.at[index].set(q[:, 1:])
        ),
        pos_p=state.pos_p.at[index].set(body),
    )
