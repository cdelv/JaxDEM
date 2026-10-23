# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Independent mass-property predictions for rigid unions of disks and spheres."""

import jax
import jax.numpy as jnp

from dataclasses import replace

import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils.clumps import (
    _compute_uniform_union_properties,
    compute_clump_properties,
)


def _rotation(quaternion):
    """Construct a rotation independently of the library quaternion helpers."""
    w, x, y, z = np.asarray(quaternion)
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + 2 * w * skew + 2 * skew @ skew


def _world_inertia(moments, quaternion):
    rotation = _rotation(quaternion)
    return rotation @ np.diag(np.asarray(moments)) @ rotation.T


def _precision_tolerance(array):
    return 4e-6 if np.asarray(array).dtype == np.float32 else 2e-12


def _uniform(positions, radii, mass=1.0, **kwargs):
    return _compute_uniform_union_properties(
        jnp.asarray(positions), jnp.asarray(radii), jnp.asarray(mass), **kwargs
    )


def _material_table(densities):
    return jd.MaterialTable.from_materials(
        [
            jd.Material.create("elastic", density=rho, young=10.0, poisson=0.25)
            for rho in densities
        ]
    )


def _assert_geometry(properties, original):
    volume, center, moments, quaternion, body = map(np.asarray, properties)
    assert np.all(np.isfinite(volume))
    assert np.all(np.isfinite(center))
    assert np.all(np.isfinite(moments))
    assert np.all(moments > 0.0)
    np.testing.assert_allclose(np.sum(quaternion**2), 1.0, atol=2e-6)
    rotation = _rotation(quaternion)[: original.shape[-1], : original.shape[-1]]
    np.testing.assert_allclose(
        body @ rotation.T + center,
        original,
        rtol=_precision_tolerance(body),
        atol=_precision_tolerance(body),
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kind", ["duplicate", "contained"])
def test_single_ball_unions_have_exact_properties_and_stable_orientation(dim, kind):
    center = np.arange(dim, dtype=float) + 0.25
    positions = np.broadcast_to(center, (2, dim)).copy()
    radii = np.array([1.2, 1.2 if kind == "duplicate" else 0.3])
    if kind == "contained":
        positions[1, 0] += 0.2
    mass = 2.7
    first = _uniform(positions, radii, mass, n_samples=17)
    second = _uniform(positions, radii, mass, n_samples=32768)
    coefficient = 0.5 if dim == 2 else 0.4
    volume = np.pi * 1.2**2 if dim == 2 else 4 * np.pi * 1.2**3 / 3
    tolerance = _precision_tolerance(first[0])
    for result in (first, second):
        np.testing.assert_allclose(result[0], volume, rtol=tolerance)
        np.testing.assert_allclose(result[1], center, atol=tolerance)
        np.testing.assert_allclose(
            result[2], coefficient * mass * 1.2**2, rtol=tolerance
        )
        _assert_geometry(result, positions)
    # Isotropic bodies should not acquire sample-dependent principal frames.
    np.testing.assert_allclose(first[3], second[3], atol=tolerance)


@pytest.mark.parametrize("dim", [2, 3])
def test_disjoint_union_matches_sphere_moments_and_parallel_axis_theorem(dim):
    positions = np.array([[-1.0, 0.2, 0.4], [1.4, -0.5, 0.1], [0.0, 1.6, -0.9]])[
        :, :dim
    ]
    radii = np.array([0.2, 0.3, 0.4])
    mass = 3.5
    weights = radii**dim / np.sum(radii**dim)
    center = np.sum(weights[:, None] * positions, axis=0)
    delta = positions - center
    intrinsic = (0.5 if dim == 2 else 0.4) * radii**2
    result = _uniform(positions, radii, mass, n_samples=17)
    tolerance = _precision_tolerance(result[0])
    unit_volume = np.pi if dim == 2 else 4 * np.pi / 3
    np.testing.assert_allclose(
        result[0], unit_volume * np.sum(radii**dim), rtol=tolerance
    )
    np.testing.assert_allclose(result[1], center, rtol=tolerance, atol=tolerance)
    if dim == 2:
        expected = mass * np.sum(weights * (intrinsic + np.sum(delta**2, axis=1)))
        np.testing.assert_allclose(result[2], expected, rtol=tolerance)
    else:
        expected = mass * np.sum(
            weights[:, None, None]
            * (
                (intrinsic + np.sum(delta**2, axis=1))[:, None, None] * np.eye(3)
                - delta[:, :, None] * delta[:, None, :]
            ),
            axis=0,
        )
        np.testing.assert_allclose(
            _world_inertia(result[2], result[3]),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )
    _assert_geometry(result, positions)


def test_sparse_tiny_spheres_never_lose_their_volume_or_inertia():
    positions = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    radius = 1e-3
    result = _uniform(positions, np.full(2, radius), n_samples=17)
    tolerance = _precision_tolerance(result[0])
    np.testing.assert_allclose(result[0], 8 * np.pi * radius**3 / 3, rtol=tolerance)
    np.testing.assert_allclose(result[1], 0.0, atol=tolerance)
    np.testing.assert_allclose(
        result[2],
        [0.4 * radius**2, 3.0 + 0.4 * radius**2, 3.0 + 0.4 * radius**2],
        rtol=2e-5,
        atol=2e-7 if np.asarray(result[0]).dtype == np.float32 else 2e-14,
    )
    _assert_geometry(result, positions)


def test_overlapping_disks_have_exact_lens_volume_and_polar_inertia():
    positions = np.array([[-0.5, 0.0], [0.5, 0.0]])
    mass = 2.3
    result = _uniform(positions, np.ones(2), mass, n_samples=17)
    volume = 4 * np.pi / 3 + np.sqrt(3) / 2
    # Direct cross-section integration about the union's center of mass.
    polar_area_moment = np.pi + 3 * np.sqrt(3) / 4
    tolerance = _precision_tolerance(result[0])
    np.testing.assert_allclose(result[0], volume, rtol=tolerance)
    np.testing.assert_allclose(result[1], 0.0, atol=tolerance)
    np.testing.assert_allclose(
        result[2], mass * polar_area_moment / volume, rtol=tolerance
    )
    _assert_geometry(result, positions)


def test_closed_ring_of_solid_disks_preserves_its_central_cavity():
    # Three pairwise-overlapping unit disks enclose a hole: their common
    # circumcenter lies outside every disk, and there is no triple overlap.
    separation = 1.8
    circumradius = separation / np.sqrt(3.0)
    angles = 2.0 * np.pi * np.arange(3) / 3.0
    positions = circumradius * np.column_stack((np.cos(angles), np.sin(angles)))
    assert np.all(np.linalg.norm(positions, axis=1) > 1.0)
    lens_area = 2.0 * np.arccos(separation / 2.0) - (
        separation * np.sqrt(4.0 - separation**2) / 2.0
    )
    expected_area = 3.0 * np.pi - 3.0 * lens_area

    # Independently integrate the lens's polar area moment by vertical
    # sections. x=h*(1-t**2) removes the square-root endpoint singularity.
    nodes, weights = np.polynomial.legendre.leggauss(32)
    t, weights = (nodes + 1.0) / 2.0, weights / 2.0
    h = 1.0 - separation / 2.0
    x = h * (1.0 - t**2)
    section_radius = t * np.sqrt(2.0 * h - h**2 * t**2)
    lens_polar = (
        8.0
        * h
        * np.sum(weights * t * (x**2 * section_radius + section_radius**3 / 3.0))
    )
    expected_polar = 3.0 * np.pi * (0.5 + circumradius**2) - 3.0 * (
        lens_polar + lens_area * circumradius**2 / 4.0
    )
    result = _uniform(positions, np.ones(3), n_samples=17)
    tolerance = _precision_tolerance(result[0])
    np.testing.assert_allclose(result[0], expected_area, rtol=tolerance)
    np.testing.assert_allclose(result[1], 0.0, atol=tolerance)
    np.testing.assert_allclose(
        result[2], expected_polar / expected_area, rtol=tolerance
    )
    _assert_geometry(result, positions)


def test_overlapping_spheres_match_independent_lens_inertia_prediction():
    positions = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    mass = 2.3
    result = _uniform(positions, np.ones(2), mass, n_samples=32768)
    coarse = _uniform(positions, np.ones(2), mass, n_samples=17)
    tolerance = _precision_tolerance(result[0])
    # Cross-section integration gives I_parallel=17M/40, I_transverse=59M/80.
    np.testing.assert_allclose(result[0], 9 * np.pi / 4, rtol=tolerance)
    np.testing.assert_allclose(result[1], 0.0, atol=tolerance)
    np.testing.assert_allclose(
        result[2], mass * np.array([17 / 40, 59 / 80, 59 / 80]), rtol=tolerance
    )
    for index in (0, 1, 2):
        np.testing.assert_allclose(
            coarse[index], result[index], rtol=tolerance, atol=tolerance
        )
    _assert_geometry(result, positions)


def test_rotated_unequal_sphere_lens_matches_independent_cross_section_integrals():
    radii = np.array([1.2, 0.8])
    separation = 1.1
    direction = np.array([0.3, -0.7, 0.2])
    direction /= np.linalg.norm(direction)
    origin = np.array([-0.4, 0.7, -0.2])
    positions = np.stack([origin, origin + separation * direction])
    switch = (radii[0] ** 2 - radii[1] ** 2 + separation**2) / (2 * separation)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    volume, first, axial, transverse_origin = 0.0, 0.0, 0.0, 0.0
    # Each section is a disk; the larger cross-section switches at the
    # intersection plane. Gaussian integration is exact for these polynomials.
    for lower, upper, center, radius in (
        (-radii[0], switch, 0.0, radii[0]),
        (switch, separation + radii[1], separation, radii[1]),
    ):
        x = (upper + lower) / 2 + nodes * (upper - lower) / 2
        weight = weights * (upper - lower) / 2
        section_radius_sq = radius**2 - (x - center) ** 2
        volume += np.pi * np.sum(weight * section_radius_sq)
        first += np.pi * np.sum(weight * x * section_radius_sq)
        axial += np.pi / 2 * np.sum(weight * section_radius_sq**2)
        transverse_origin += np.pi * np.sum(
            weight * (x**2 * section_radius_sq + section_radius_sq**2 / 4)
        )
    center = first / volume
    mass = 1.7
    axial = mass * axial / volume
    transverse = mass * (transverse_origin / volume - center**2)
    expected_tensor = axial * np.outer(direction, direction) + transverse * (
        np.eye(3) - np.outer(direction, direction)
    )
    result = _uniform(positions, radii, mass, n_samples=17)
    tolerance = 1e-5 if np.asarray(result[0]).dtype == np.float32 else 3e-12
    np.testing.assert_allclose(result[0], volume, rtol=tolerance)
    np.testing.assert_allclose(
        result[1], origin + center * direction, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        _world_inertia(result[2], result[3]),
        expected_tensor,
        rtol=tolerance,
        atol=tolerance,
    )
    _assert_geometry(result, positions)


def test_partial_sphere_overlap_uses_maximum_density_without_double_counting():
    state = jd.State.create(
        pos=jnp.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        rad=jnp.ones(2),
        mat_id=jnp.array([0, 1]),
        clump_id=jnp.zeros(2, dtype=int),
    )
    actual = compute_clump_properties(state, _material_table([1.0, 3.0]), n_samples=17)
    # Add the two full spheres and remove the lens at the lower density.
    # Sphere/lens cross-section integrals give these COM-frame moments.
    mass = 59 * np.pi / 12
    center = 16 / 59
    axial = 971 * np.pi / 480
    transverse = 3257 * np.pi / 960 - mass * center**2
    tolerance = _precision_tolerance(actual.mass)
    np.testing.assert_allclose(actual.volume, 9 * np.pi / 4, rtol=tolerance)
    np.testing.assert_allclose(actual.mass, mass, rtol=tolerance)
    np.testing.assert_allclose(actual.pos_c, [[center, 0.0, 0.0]] * 2, atol=tolerance)
    quaternion = np.r_[np.asarray(actual.q.w[0]), np.asarray(actual.q.xyz[0])]
    np.testing.assert_allclose(
        _world_inertia(actual.inertia[0], quaternion),
        np.diag([axial, transverse, transverse]),
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(actual.pos, state.pos, atol=2e-6)


@pytest.mark.parametrize("radius", [1.0, 0.4])
def test_sphere_chain_matches_independent_cross_section_moments(radius):
    centers = np.array([-0.7, 0.0, 0.7])
    direction = np.array([0.4, -0.2, 0.7])
    direction /= np.linalg.norm(direction)
    positions = centers[:, None] * direction
    nodes, weights = np.polynomial.legendre.leggauss(8)
    volume, axial, transverse = 0.0, 0.0, 0.0
    for lower, upper, center in zip(
        [centers[0] - radius, -0.35, 0.35],
        [-0.35, 0.35, centers[-1] + radius],
        centers,
    ):
        x = (upper + lower) / 2 + nodes * (upper - lower) / 2
        weight = weights * (upper - lower) / 2
        section_radius_sq = radius**2 - (x - center) ** 2
        volume += np.pi * np.sum(weight * section_radius_sq)
        axial += np.pi / 2 * np.sum(weight * section_radius_sq**2)
        transverse += np.pi * np.sum(
            weight * (x**2 * section_radius_sq + section_radius_sq**2 / 4)
        )
    mass = 1.7
    expected_tensor = (
        mass
        / volume
        * (
            axial * np.outer(direction, direction)
            + transverse * (np.eye(3) - np.outer(direction, direction))
        )
    )
    result = _uniform(positions, np.full(3, radius), mass, n_samples=32768)
    # R=.4 overlaps only adjacent spheres and is exactly pairwise integrable;
    # R=1 has a genuine triple overlap and exercises surface quadrature.
    tolerance = _precision_tolerance(result[0]) if radius == 0.4 else 0.015
    volume_tolerance = tolerance if radius == 0.4 else 0.008
    position_tolerance = tolerance if radius == 0.4 else 0.006
    np.testing.assert_allclose(result[0], volume, rtol=volume_tolerance)
    np.testing.assert_allclose(result[1], 0.0, atol=position_tolerance)
    np.testing.assert_allclose(
        _world_inertia(result[2], result[3]),
        expected_tensor,
        rtol=tolerance,
        atol=position_tolerance,
    )
    if radius == 0.4:
        coarse = _uniform(positions, np.full(3, radius), mass, n_samples=17)
        for index in (0, 1, 2):
            np.testing.assert_allclose(
                coarse[index], result[index], rtol=tolerance, atol=tolerance
            )
    _assert_geometry(result, positions)


def test_many_strongly_overlapping_spheres_resolve_union_mass_properties():
    # All 65 spheres contain the origin. This exceeds the small-clump
    # analytical shortcut and exercises high-multiplicity numerical overlap.
    centers = np.linspace(-0.7, 0.7, 65)
    direction = np.array([0.4, -0.2, 0.7])
    direction /= np.linalg.norm(direction)
    positions = centers[:, None] * direction
    radius = 1.0
    breaks = np.r_[
        centers[0] - radius,
        (centers[:-1] + centers[1:]) / 2.0,
        centers[-1] + radius,
    ]
    nodes, weights = np.polynomial.legendre.leggauss(8)
    volume, axial, transverse = 0.0, 0.0, 0.0
    # The nearest center supplies the largest disk cross-section on each
    # interval. These section moments are polynomials, so Gauss integration
    # provides an independent analytical oracle, including all overlap orders.
    for center, lower, upper in zip(centers, breaks[:-1], breaks[1:]):
        x = (lower + upper) / 2.0 + nodes * (upper - lower) / 2.0
        weight = weights * (upper - lower) / 2.0
        section_radius_sq = radius**2 - (x - center) ** 2
        volume += np.pi * np.sum(weight * section_radius_sq)
        axial += np.pi / 2.0 * np.sum(weight * section_radius_sq**2)
        transverse += np.pi * np.sum(
            weight * (x**2 * section_radius_sq + section_radius_sq**2 / 4.0)
        )
    expected_tensor = (
        axial * np.outer(direction, direction)
        + transverse * (np.eye(3) - np.outer(direction, direction))
    ) / volume
    result = _uniform(positions, np.ones(65), n_samples=50_000)
    np.testing.assert_allclose(result[0], volume, rtol=0.005)
    np.testing.assert_allclose(result[1], 0.0, atol=0.004)
    np.testing.assert_allclose(
        _world_inertia(result[2], result[3]),
        expected_tensor,
        rtol=0.01,
        atol=0.003,
    )
    rechunked = _uniform(
        positions, np.ones(65), n_samples=50_000, sample_batch_size=1000
    )
    tolerance = 3e-5 if np.asarray(result[0]).dtype == np.float32 else 3e-11
    for index in (0, 1, 2):
        np.testing.assert_allclose(
            rechunked[index], result[index], rtol=tolerance, atol=tolerance
        )
    np.testing.assert_allclose(
        _world_inertia(rechunked[2], rechunked[3]),
        _world_inertia(result[2], result[3]),
        rtol=tolerance,
        atol=tolerance,
    )
    _assert_geometry(result, positions)


def test_mixed_dense_and_sparse_clumps_match_individual_integration():
    direction = np.array([0.4, -0.2, 0.7])
    direction /= np.linalg.norm(direction)
    positions = np.stack(
        [np.linspace(-span, span, 65)[:, None] * direction for span in (0.7, 5.0)]
    )
    # The compact union uses bulk integration; the spread-out union keeps
    # boundary integration. Batching must preserve both bodies' results.
    batched = _uniform(positions, np.ones(65), n_samples=8192)
    tolerance = 3e-5 if np.asarray(batched[0]).dtype == np.float32 else 3e-11
    for index, centers in enumerate(positions):
        individual = _uniform(centers, np.ones(65), n_samples=8192)
        for field in (0, 1, 2):
            np.testing.assert_allclose(
                batched[field][index],
                individual[field],
                rtol=tolerance,
                atol=tolerance,
            )
        np.testing.assert_allclose(
            _world_inertia(batched[2][index], batched[3][index]),
            _world_inertia(individual[2], individual[3]),
            rtol=tolerance,
            atol=tolerance,
        )
        _assert_geometry(individual, centers)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("inner_radius", [0.5, 1.0])
def test_maximum_material_density_preserves_dense_concentric_core(dim, inner_radius):
    positions = jnp.zeros((2, dim))
    state = jd.State.create(
        pos=positions,
        rad=jnp.array([1.0, inner_radius]),
        mat_id=jnp.array([0, 1]),
        clump_id=jnp.zeros(2, dtype=int),
    )
    actual = compute_clump_properties(
        state, _material_table([1.0, 3.0]), n_samples=32768
    )
    unit_volume = np.pi if dim == 2 else 4 * np.pi / 3
    expected_mass = unit_volume * (1 + 2 * inner_radius**dim)
    expected_moment = (
        (0.5 if dim == 2 else 0.4) * unit_volume * (1 + 2 * inner_radius ** (dim + 2))
    )
    tolerance = _precision_tolerance(actual.mass) if dim == 2 else 0.015
    np.testing.assert_allclose(actual.volume, unit_volume, rtol=tolerance)
    np.testing.assert_allclose(actual.mass, expected_mass, rtol=tolerance)
    np.testing.assert_allclose(actual.pos_c, 0.0, atol=0.006 if dim == 3 else tolerance)
    np.testing.assert_allclose(actual.inertia, expected_moment, rtol=tolerance)
    np.testing.assert_allclose(actual.pos, positions, atol=2e-6)


def test_three_dimensional_inertia_transforms_as_a_world_tensor():
    positions = np.array([[-1.0, 0.2, 0.4], [1.4, -0.5, 0.1], [0.0, 1.6, -0.9]])
    radii = np.array([0.2, 0.3, 0.4])
    quaternion = np.array([0.8, 0.2, -0.4, 0.4])
    quaternion /= np.linalg.norm(quaternion)
    rotation = _rotation(quaternion)
    translation = np.array([100.0, -200.0, 300.0])
    first = _uniform(positions, radii, n_samples=17)
    moved_positions = positions @ rotation.T + translation
    moved = _uniform(moved_positions, radii, n_samples=17)
    tolerance = 6e-5 if np.asarray(first[0]).dtype == np.float32 else 2e-11
    np.testing.assert_allclose(moved[0], first[0], rtol=tolerance)
    np.testing.assert_allclose(
        moved[1], np.asarray(first[1]) @ rotation.T + translation, atol=tolerance
    )
    np.testing.assert_allclose(
        _world_inertia(moved[2], moved[3]),
        rotation @ _world_inertia(first[2], first[3]) @ rotation.T,
        rtol=tolerance,
        atol=tolerance,
    )
    _assert_geometry(moved, moved_positions)


def test_nonconsecutive_ids_preserve_singletons_and_world_geometry():
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [2.0, 0.0], [7.0, 2.0]]),
        rad=jnp.array([0.5, 0.5, 0.2]),
        clump_id=jnp.array([0, 0, 1]),
        mass=jnp.array([9.0, 9.0, 6.0]),
        volume=jnp.array([8.0, 8.0, 3.0]),
        inertia=jnp.array([[7.0], [7.0], [2.0]]),
        q=jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [np.cos(0.4), 0.0, 0.0, np.sin(0.4)],
            ]
        ),
    )
    state = replace(state, clump_id=jnp.array([7, 7, 100000]))
    actual = compute_clump_properties(state, _material_table([2.0]), n_samples=17)
    tolerance = _precision_tolerance(actual.mass)
    np.testing.assert_array_equal(actual.clump_id, state.clump_id)
    np.testing.assert_allclose(actual.mass[:2], np.pi, rtol=tolerance)
    np.testing.assert_allclose(actual.inertia[:2, 0], 9 * np.pi / 8, rtol=tolerance)
    np.testing.assert_allclose(actual.pos_c[:2], [[1.0, 0.0]] * 2, atol=tolerance)
    for name in ("mass", "volume", "inertia", "pos_c", "pos_p"):
        np.testing.assert_array_equal(
            getattr(actual, name)[-1], getattr(state, name)[-1]
        )
    np.testing.assert_array_equal(actual.q.w[-1], state.q.w[-1])
    np.testing.assert_array_equal(actual.q.xyz[-1], state.q.xyz[-1])
    np.testing.assert_allclose(actual.pos, state.pos, atol=2e-6)


@pytest.mark.parametrize("dim", [2, 3])
def test_empty_state_properties_are_a_no_op(dim):
    state = jd.State.create(dim=dim)
    actual = compute_clump_properties(state, _material_table([1.0]))
    assert actual.shape == state.shape
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("samples", [17, 4099])
def test_sample_chunking_preserves_the_requested_integration_rule(samples):
    positions = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.1, 0.4, 0.2]])
    radii = np.array([1.0, 0.8, 0.6])
    first = _uniform(positions, radii, n_samples=samples, sample_batch_size=128)
    second = _uniform(positions, radii, n_samples=samples, sample_batch_size=1000)
    tolerance = 2e-5 if np.asarray(first[0]).dtype == np.float32 else 2e-11
    for idx in (0, 1, 2):
        np.testing.assert_allclose(
            second[idx], first[idx], rtol=tolerance, atol=tolerance
        )
    np.testing.assert_allclose(
        _world_inertia(second[2], second[3]),
        _world_inertia(first[2], first[3]),
        rtol=tolerance,
        atol=tolerance,
    )
    _assert_geometry(first, positions)
    _assert_geometry(second, positions)


def test_clump_chunking_preserves_individual_mass_properties():
    template = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.1, 0.4, 0.2]])
    positions = np.stack([template + np.array([i, -2 * i, 0.2 * i]) for i in range(5)])
    radii = np.array([1.0, 0.8, 0.6])
    first = _uniform(positions, radii, n_samples=2049, clump_batch_size=1)
    second = _uniform(positions, radii, n_samples=2049, clump_batch_size=3)
    tolerance = 3e-5 if np.asarray(first[0]).dtype == np.float32 else 3e-11
    for idx in (0, 1, 2):
        np.testing.assert_allclose(
            second[idx], first[idx], rtol=tolerance, atol=tolerance
        )
    for i in range(5):
        np.testing.assert_allclose(
            _world_inertia(second[2][i], second[3][i]),
            _world_inertia(first[2][i], first[3][i]),
            rtol=tolerance,
            atol=tolerance,
        )


def test_surface_normals_are_balanced_and_resolve_large_float32_indices():
    from jaxdem.utils.clumps import _fibonacci_normals

    normals = np.asarray(_fibonacci_normals(jnp.arange(4096), 4096, jnp.float32))
    np.testing.assert_allclose(np.sum(normals**2, axis=-1), 1.0, atol=3e-7)
    np.testing.assert_allclose(np.mean(normals, axis=0), 0.0, atol=2e-6)

    indices = jnp.arange(900000, 910000)
    high = np.asarray(_fibonacci_normals(indices, 1000000, jnp.float32))
    halves = np.concatenate(
        [
            np.asarray(_fibonacci_normals(indices[:5000], 1000000, jnp.float32)),
            np.asarray(_fibonacci_normals(indices[5000:], 1000000, jnp.float32)),
        ]
    )
    np.testing.assert_array_equal(high, halves)
    np.testing.assert_allclose(np.sum(high**2, axis=-1), 1.0, atol=3e-7)
    longitude = (np.arctan2(high[:, 1], high[:, 0]) / (2 * np.pi)) % 1.0
    histogram = np.histogram(longitude, bins=np.linspace(0.0, 1.0, 65))[0]
    # Multiplying a large float32 index before modulo collapses longitudes
    # onto a few phases and leaves half these bins empty.
    assert np.max(np.abs(histogram / (len(indices) / 64) - 1.0)) < 0.1


@pytest.mark.parametrize(
    "keyword", ["n_samples", "sample_batch_size", "clump_batch_size"]
)
@pytest.mark.parametrize("invalid", [0, -1, 1.5])
def test_integration_budgets_require_positive_integers(keyword, invalid):
    with pytest.raises((TypeError, ValueError), match=keyword):
        _uniform([[0.0, 0.0], [1.0, 0.0]], [1.0, 1.0], **{keyword: invalid})


@pytest.mark.parametrize(
    "positions,radii,mass",
    [
        ([[0.0, np.nan], [1.0, 0.0]], [1.0, 1.0], 1.0),
        ([[0.0, np.inf], [1.0, 0.0]], [1.0, 1.0], 1.0),
        ([[0.0, 0.0], [1.0, 0.0]], [1.0, np.nan], 1.0),
        ([[0.0, 0.0], [1.0, 0.0]], [1.0, np.inf], 1.0),
        ([[0.0, 0.0], [1.0, 0.0]], [1.0, 0.0], 1.0),
        ([[0.0, 0.0], [1.0, 0.0]], [1.0, 1.0], np.nan),
        ([[0.0, 0.0], [1.0, 0.0]], [1.0, 1.0], np.inf),
    ],
)
def test_union_properties_reject_nonfinite_or_nonphysical_input(positions, radii, mass):
    with pytest.raises(ValueError):
        _uniform(positions, radii, mass)


@pytest.mark.parametrize(
    "positions,radii,mass",
    [
        (np.zeros((2, 4)), np.ones(2), 1.0),
        (np.zeros((2, 3)), np.ones(3), 1.0),
        (np.zeros((2, 2, 3)), np.ones((2, 2)), np.ones(3)),
    ],
)
def test_union_properties_reject_incompatible_shapes(positions, radii, mass):
    with pytest.raises(ValueError):
        _uniform(positions, radii, mass)
