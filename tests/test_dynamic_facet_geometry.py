# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Deforming facets retain closest features and conservative nodal mechanics."""

import jax
import jax.numpy as jnp

from dataclasses import replace

import numpy as np
import pytest

import jaxdem as jd
from jaxdem.forces import FacetFacetSpringForce, SphereFacetSpringForce
from jaxdem.forces.facet_contact import (
    facet_normal,
    point_triangle_distance,
    segment_segment_distance,
    triangle_triangle_distance,
)


@pytest.fixture(scope="module")
def query_system():
    state = jd.State.create(pos=jnp.zeros((1, 3)))
    return jd.System.create(state=state, collider_type="")


@pytest.mark.parametrize("height", [1e-3, 1e-8])
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_triangle_flattening_preserves_face_distance_and_nodal_weights(
    query_system, height, scale
):
    triangle = scale * jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, height, 0.0]])
    point = scale * jnp.array([0.75, height / 4, 0.1])
    projection = scale * np.array([0.75, height / 4, 0.0])
    expected_weights = [0.25, 0.5, 0.25]
    for flags in (
        {},
        {"edge": False},
        {"vertex": False},
        {"edge": False, "vertex": False},
    ):
        distance, closest, weights = point_triangle_distance(
            point, *triangle, query_system, **flags
        )
        np.testing.assert_allclose(distance / scale, 0.1, rtol=3e-6)
        np.testing.assert_allclose(
            closest / scale, projection / scale, rtol=3e-6, atol=1e-10
        )
        np.testing.assert_allclose(weights, expected_weights, rtol=3e-6, atol=1e-7)
    # Disabling the actual closest face must not substitute a farther edge.
    distance, closest, weights = point_triangle_distance(
        point, *triangle, query_system, face=False
    )
    assert np.isinf(distance)
    assert np.all(np.isfinite(closest))
    assert np.all(np.isfinite(weights))


@pytest.mark.parametrize("height", [1e-3, 1e-8])
def test_thin_triangle_short_edge_is_not_mistaken_for_an_endpoint(query_system, height):
    triangle = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, height, 0.0]])
    point = jnp.array([1.2, 0.4 * height, 0.1])
    for flags in ({}, {"face": False, "vertex": False}):
        distance, closest, weights = point_triangle_distance(
            point, *triangle, query_system, **flags
        )
        np.testing.assert_allclose(distance, np.sqrt(0.2**2 + 0.1**2), rtol=3e-6)
        np.testing.assert_allclose(
            closest, [1.0, 0.4 * height, 0.0], rtol=3e-6, atol=1e-12
        )
        np.testing.assert_allclose(weights, [0.0, 0.6, 0.4], rtol=3e-6, atol=1e-7)
    distance, _, _ = point_triangle_distance(point, *triangle, query_system, edge=False)
    assert np.isinf(distance)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("height", [1e-4, 1e-8])
def test_nearly_parallel_edges_retain_their_interior_closest_points(dim, height):
    state = jd.State.create(pos=jnp.zeros((1, dim)))
    system = jd.System.create(state=state, collider_type="")
    first = jnp.zeros((2, dim)).at[1, 0].set(1.0)
    second = first.at[:, 1].set(jnp.array([-height / 2, height / 2]))
    gap = height * 0.01 if dim == 3 else 0.0
    if dim == 3:
        second = second.at[:, 2].set(gap)
    distance, point1, point2, weights1, weights2 = segment_segment_distance(
        *first, *second, system
    )
    np.testing.assert_allclose(distance, gap, rtol=3e-6, atol=gap * 1e-4)
    np.testing.assert_allclose(point1, first.mean(axis=0), rtol=3e-6, atol=1e-12)
    np.testing.assert_allclose(point2, second.mean(axis=0), rtol=3e-6, atol=1e-12)
    np.testing.assert_allclose(weights1, [0.5, 0.5], rtol=3e-6)
    np.testing.assert_allclose(weights2, [0.5, 0.5], rtol=3e-6)


@pytest.mark.parametrize("scale", [1e-6, 1.0])
def test_small_nearly_parallel_edges_retain_finite_distance_gradients(scale):
    state = jd.State.create(pos=jnp.zeros((1, 3)))
    system = jd.System.create(state=state, collider_type="")
    endpoints = scale * jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 5e-9, 0.1], [1.0, -5e-9, 0.1]]
    )
    distance, gradient = jax.value_and_grad(
        lambda points: segment_segment_distance(*points, system)[0]
    )(endpoints)
    np.testing.assert_allclose(distance / scale, 0.1, rtol=3e-6)
    expected = np.zeros((4, 3))
    expected[:, 2] = [-0.5, -0.5, 0.5, 0.5]
    np.testing.assert_allclose(gradient, expected, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("scale", [1e-6, 1.0])
def test_small_thin_triangle_retains_finite_distance_gradients(dim, scale):
    state = jd.State.create(pos=jnp.zeros((1, dim)))
    system = jd.System.create(state=state, collider_type="")
    triangle = scale * jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1e-8]])
    point = scale * jnp.array([0.75, 2.5e-9])
    if dim == 3:
        triangle = jnp.pad(triangle, ((0, 0), (0, 1)))
        point = jnp.concatenate((point, jnp.array([0.1 * scale])))
    query = jnp.concatenate((point[None, :], triangle))
    distance, gradient = jax.value_and_grad(
        lambda points: point_triangle_distance(*points, system)[0]
    )(query)
    expected = np.zeros((4, dim))
    if dim == 3:
        expected[:, 2] = [1.0, -0.25, -0.5, -0.25]
    np.testing.assert_allclose(distance / scale, 0.1 if dim == 3 else 0.0, atol=1e-7)
    np.testing.assert_allclose(gradient, expected, rtol=3e-6, atol=1e-7)


def test_rotated_thin_triangle_uses_the_plane_of_its_represented_vertices():
    state = jd.State.create(pos=jnp.zeros((1, 3)))
    system = jd.System.create(state=state, collider_type="")
    # Quantized B and C remain distinct. Forming AB x AC loses their small
    # transverse separation in float32; AB x BC preserves the plane.
    query = jnp.array(
        [
            [0.00627528597, 0.0416911133, -0.0184270795],
            [0.0963810608, 0.0968032256, -0.0142982295],
            [-0.0245889686, 0.0226264633, -0.0201781522],
            [-0.0245881658, 0.0226250924, -0.0201773886],
        ]
    )
    point, a, b, c = np.asarray(query, dtype=np.float64)
    normal = np.cross(b - a, c - b)
    normal /= np.linalg.norm(normal)
    signed_distance = np.dot(point - a, normal)
    projection = point - signed_distance * normal
    distance, closest, weights = point_triangle_distance(*query, system)
    np.testing.assert_allclose(distance, abs(signed_distance), rtol=2e-5, atol=2e-8)
    np.testing.assert_allclose(closest, projection, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(weights @ query[1:], closest, rtol=2e-6, atol=2e-8)
    gradient = jax.grad(lambda points: point_triangle_distance(*points, system)[0])(
        query
    )
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=3e-6)
    np.testing.assert_allclose(jnp.cross(query, gradient).sum(axis=0), 0.0, atol=3e-7)


@pytest.mark.parametrize("scale", [1e-6, 1.0])
def test_small_thin_facet_pairs_keep_rejected_intersection_gradients_finite(scale):
    state = jd.State.create(pos=jnp.zeros((1, 3)))
    system = jd.System.create(state=state, collider_type="")
    vertices = scale * jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1e-8, 0.0],
            [0.75, 0.25e-8, 0.1],
            [0.65, 0.5e-8, 0.5],
            [0.85, 0.2e-8, 0.8],
        ]
    )
    distance, gradient = jax.value_and_grad(
        lambda points: triangle_triangle_distance(*points, system)[0]
    )(vertices)
    expected = np.zeros((6, 3))
    expected[:, 2] = [-0.25, -0.5, -0.25, 1.0, 0.0, 0.0]
    np.testing.assert_allclose(distance / scale, 0.1, rtol=3e-6)
    np.testing.assert_allclose(gradient, expected, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("scale", [1e-6, 1.0])
def test_zero_separation_uses_the_normal_of_small_thin_facets(dim, scale):
    vertices = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    point = jnp.array([0.5, 0.0])
    if dim == 3:
        vertices = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1e-8, 0.0]])
        point = jnp.array([0.75, 2.5e-9, 0.0])
    vertices, point = scale * vertices, scale * point
    nodes = jd.State.create(pos=vertices, rad=jnp.full(dim, 0.1 * scale))
    state = jd.State.add_contact_facets(nodes, jnp.arange(dim)[None, :])
    state = jd.State.merge(
        state, jd.State.create(pos=point[None, :], rad=jnp.array([0.2 * scale]))
    )
    system = jd.System.create(
        state=state, collider_type="naive", force_model=SphereFacetSpringForce()
    )
    normal = facet_normal(
        vertices[0], vertices[1], vertices[2] if dim == 3 else None, system
    )
    expected = np.zeros(dim)
    expected[-1] = 1.0
    np.testing.assert_allclose(normal, expected, atol=1e-7)
    loaded, _ = jd.System.evaluate_forces(state, system)
    assert np.all(np.isfinite(loaded.force))
    assert float(jnp.abs(loaded.force[-1, -1])) > 0.0
    np.testing.assert_allclose(loaded.force[:, :-1], 0.0, atol=1e-7)
    np.testing.assert_allclose(loaded.force.sum(axis=0), 0.0, atol=1e-4)


@pytest.mark.parametrize("collapsed", [False, True])
def test_collapsed_triangle_returns_finite_boundary_points(query_system, collapsed):
    triangle = (
        jnp.zeros((3, 3))
        if collapsed
        else jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    )
    point = jnp.array([0.75, 0.2, 0.1])
    distance, closest, weights = point_triangle_distance(point, *triangle, query_system)
    expected = [0.0, 0.0, 0.0] if collapsed else [0.75, 0.0, 0.0]
    np.testing.assert_allclose(closest, expected, atol=1e-7)
    np.testing.assert_allclose(
        distance, np.linalg.norm(np.asarray(point) - expected), rtol=3e-6
    )
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-7)
    assert np.all(np.asarray(weights) >= 0.0)
    np.testing.assert_allclose(weights @ triangle, closest, atol=1e-7)
    distance, closest, weights = point_triangle_distance(
        point, *triangle, query_system, edge=False, vertex=False
    )
    assert np.isinf(distance)
    assert np.all(np.isfinite(closest))
    assert np.all(np.isfinite(weights))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("collapsed", [False, True])
def test_live_facet_collapse_keeps_contact_loads_finite_and_conservative(
    dim, collapsed
):
    vertices = jnp.eye(dim).at[0].set(0.0)
    state = jd.State.add_facet(jd.State.create(), vertices, thickness=0.1, rigid=False)
    deformed = jnp.zeros((dim, dim))
    if not collapsed:
        deformed = deformed.at[:, 0].set(jnp.arange(dim))
    point = jnp.zeros(dim).at[0].set(0.05 if collapsed else 0.75)
    point = point.at[-1].set(0.15)
    state = jd.State.merge(
        replace(state, pos_c=deformed),
        jd.State.create(pos=point[None, :], rad=jnp.array([0.2])),
    )
    system = jd.System.create(
        state=state, collider_type="naive", force_model=SphereFacetSpringForce()
    )
    loaded, _ = jd.System.evaluate_forces(state, system)
    energy = system.collider.compute_potential_energy(state, system)[2]
    assert np.all(np.isfinite(loaded.force))
    assert np.isfinite(energy) and energy > 0.0
    assert float(jnp.linalg.norm(loaded.force[-1])) > 0.0
    np.testing.assert_allclose(loaded.force.sum(axis=0), 0.0, atol=2e-4)
    force = np.asarray(loaded.force)
    moment = (
        state.pos[:, 0] * force[:, 1] - state.pos[:, 1] * force[:, 0]
        if dim == 2
        else np.cross(state.pos, force)
    )
    np.testing.assert_allclose(moment.sum(axis=0), 0.0, atol=2e-4)


def _normal(vertices):
    edge = vertices[1] - vertices[0]
    normal = (
        np.array([-edge[1], edge[0]])
        if vertices.shape[-1] == 2
        else np.cross(edge, vertices[2] - vertices[0])
    )
    return normal / np.linalg.norm(normal)


def _geometry(dim, frame, family):
    if dim == 2:
        vertices = (
            np.array([[0.0, 0.0], [2.0, 0.0]])
            if frame == 0
            else np.array([[0.3, -0.2], [2.4, 0.8]])
        )
        weights = np.array([0.35, 0.65])
    else:
        vertices = (
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
            if frame == 0
            else np.array([[0.2, -0.1, -0.2], [2.5, 0.4, 0.1], [-0.3, 2.6, 0.2]])
        )
        weights = np.array([0.4, 0.35, 0.25])
    normal = _normal(vertices)
    nearest = weights @ vertices + 0.08 * normal
    if family == "sphere":
        return vertices, nearest[None, :], weights, normal
    # The other facet's nearest vertex is unique and projects inside the first.
    others = np.stack(
        [
            nearest + (0.4 + 0.1 * i) * normal + 0.1 * (vertices[i + 1] - vertices[0])
            for i in range(dim - 1)
        ]
    )
    return vertices, np.concatenate((nearest[None, :], others)), weights, normal


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("family", ["sphere", "facet"])
def test_live_vertex_positions_preserve_energy_gradient_and_momentum(dim, family):
    vertices, other, _, _ = _geometry(dim, 0, family)
    state = jd.State.add_facet(
        jd.State.create(),
        jnp.asarray(vertices),
        thickness=0.1,
        rigid=False,
        species_id=jnp.array([1]),
    )
    if family == "sphere":
        state = jd.State.merge(
            state, jd.State.create(pos=jnp.asarray(other), rad=jnp.array([0.2]))
        )
        law = jd.ForceRouter.from_dict(2, {(0, 1): SphereFacetSpringForce()})
    else:
        state = jd.State.add_facet(
            state,
            jnp.asarray(other),
            thickness=0.2,
            rigid=False,
            species_id=jnp.array([1]),
        )
        law = FacetFacetSpringForce()
    system = jd.System.create(state=state, collider_type="naive", force_model=law)
    stiffness = float(system.mat_table.young_eff[0, 0])
    force_magnitude = stiffness * (0.1 + 0.2 - 0.08)
    for frame in range(2):
        vertices, other, weights, normal = _geometry(dim, frame, family)
        positions = jnp.asarray(np.concatenate((vertices, other)))
        moved = replace(state, pos_c=positions)
        actual, _ = jd.System.evaluate_forces(moved, system)

        def energy(position):
            trial = replace(state, pos_c=position)
            return system.collider.compute_potential_energy(trial, system)[2]

        value, gradient = jax.value_and_grad(energy)(positions)
        np.testing.assert_allclose(value, 0.5 * stiffness * 0.22**2, rtol=5e-6)
        np.testing.assert_allclose(
            actual.force, -gradient, rtol=2e-5, atol=3e-6 * stiffness
        )
        np.testing.assert_allclose(
            actual.force[:dim],
            -force_magnitude * weights[:, None] * normal,
            rtol=2e-5,
            atol=3e-6 * stiffness,
        )
        np.testing.assert_allclose(actual.force.sum(axis=0), 0.0, atol=3e-6 * stiffness)
        np.testing.assert_allclose(actual.torque, 0.0, atol=3e-6 * stiffness)
        force = np.asarray(actual.force)
        moment = (
            positions[:, 0] * force[:, 1] - positions[:, 1] * force[:, 0]
            if dim == 2
            else np.cross(positions, force)
        )
        np.testing.assert_allclose(moment.sum(axis=0), 0.0, atol=3e-6 * stiffness)
