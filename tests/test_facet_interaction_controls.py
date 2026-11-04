# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Static facet query controls omit geometry and apply to force and energy."""

import jax
import jax.numpy as jnp

import numpy as np
import pytest

import jaxdem as jd
from jaxdem.forces import FacetFacetSpringForce, SphereFacetSpringForce
from jaxdem.forces.facet_contact import (
    point_segment_distance,
    point_triangle_distance,
    triangle_triangle_distance,
)

SPHERE_CASES = [
    (3, "face", [0.5, 0.5, 0.1]),
    (3, "edge", [0.5, -0.1, 0.1]),
    (3, "vertex", [-0.1, -0.1, 0.1]),
    (2, "edge", [1.0, 0.1]),
    (2, "vertex", [-0.1, 0.1]),
]
FACET_GROUPS = ("vertex_triangle", "edge_edge", "edge_triangle")


def _sphere_state(dim, point, *, rigid=True):
    vertices = (
        [[0.0, 0.0], [2.0, 0.0]]
        if dim == 2
        else [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    state = jd.State.add_facet(
        jd.State.create(),
        jnp.array(vertices),
        thickness=0.1,
        rigid=rigid,
        species_id=jnp.array([1]),
    )
    sphere = jd.State.create(pos=jnp.array([point]), rad=jnp.array([0.2]))
    return jd.State.merge(state, sphere)


def _facet_vertices(group):
    if group == "vertex_triangle":
        return jnp.array(
            [
                [[0.5, 0.5, 0.1], [0.7, 0.5, 0.1], [0.5, 0.7, 0.1]],
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            ]
        )
    if group == "edge_edge":
        root3 = np.sqrt(3.0)
        # Projected faces overlap, but no projected vertex lies inside the other.
        return jnp.array(
            [
                [[0.0, 2.0, 0.0], [-root3, -1.0, 0.0], [root3, -1.0, 0.0]],
                [[0.0, -2.0, 0.1], [root3, 1.0, 0.1], [-root3, 1.0, 0.1]],
            ]
        )
    return jnp.array(
        [
            [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, -0.5, -1.0], [0.0, -0.5, 1.0], [0.0, 0.5, 1.0]],
        ]
    )


def _facet_state(vertices, *, rigid=True):
    state = jd.State.create()
    for triangle in vertices:
        state = jd.State.add_facet(
            state,
            triangle,
            thickness=0.1,
            rigid=rigid,
        )
    return state


def _pair(state, law, i, j, history=None):
    system = jd.System.create(state=state, force_model=law)
    if history is None:
        history = law.init_history(jnp.shape(j), state.pos.shape[-1])
    force, torque, actual_history = law.force(i, j, state.pos, state, system, history)
    energy = law.energy(i, j, state.pos, state, system, history)
    return force, torque, energy, actual_history


@pytest.mark.parametrize("dim,feature,point", SPHERE_CASES)
def test_sphere_feature_selection_suppresses_force_and_energy(dim, feature, point):
    state = _sphere_state(dim, point)
    all_enabled = _pair(state, SphereFacetSpringForce(), dim, 0)
    assert np.linalg.norm(all_enabled[0]) > 0.0
    assert float(all_enabled[2]) > 0.0

    only = {name: name == feature for name in ("face", "edge", "vertex")}
    selected = _pair(state, SphereFacetSpringForce(**only), dim, 0)
    tolerance = 1e-6 if state.pos.dtype == jnp.float32 else 1e-12
    # Equivalent projection formulas can introduce roundoff perpendicular to
    # the force in float32, so scale the absolute tolerance by its magnitude.
    force_scale = max(1.0, float(jnp.linalg.norm(all_enabled[0])))
    for actual, expected in zip(selected, all_enabled):
        np.testing.assert_allclose(
            actual, expected, rtol=tolerance, atol=tolerance * force_scale
        )

    omitted = _pair(state, SphereFacetSpringForce(**{feature: False}), dim, 0)
    for actual in omitted[:3]:
        np.testing.assert_array_equal(actual, 0.0)


@pytest.mark.parametrize("group", FACET_GROUPS)
def test_facet_query_groups_have_independent_computational_effects(group):
    vertices = _facet_vertices(group)
    state = _facet_state(vertices)
    default = _pair(state, FacetFacetSpringForce(), 0, 3)
    assert np.linalg.norm(default[0]) > 0.0
    assert float(default[2]) > 0.0

    only = {name: name == group for name in FACET_GROUPS}
    selected = _pair(state, FacetFacetSpringForce(**only), 0, 3)
    np.testing.assert_allclose(selected[0], default[0], atol=1e-12)
    np.testing.assert_allclose(selected[2], default[2], atol=1e-12)
    omitted = _pair(state, FacetFacetSpringForce(**{group: False}), 0, 3)
    for actual in omitted[:3]:
        np.testing.assert_array_equal(actual, 0.0)

    # Directed calls must still choose reciprocal forces when candidates tie.
    reverse = _pair(state, FacetFacetSpringForce(**only), 3, 0)
    np.testing.assert_allclose(reverse[0], -selected[0], atol=1e-12)
    np.testing.assert_allclose(reverse[2], selected[2], atol=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("family", ["sphere", "facet"])
def test_all_disabled_has_finite_zero_force_energy_and_unchanged_history(dim, family):
    history = jnp.array([0.3, -0.8])
    if family == "sphere":
        state = _sphere_state(dim, [0.5, 0.1] if dim == 2 else [0.5, 0.5, 0.1])
        law = SphereFacetSpringForce(face=False, edge=False, vertex=False)
        i, j = dim, 0
    else:
        vertices = (
            jnp.array([[[0.0, 0.0], [2.0, 0.0]], [[0.5, 0.1], [1.5, 0.1]]])
            if dim == 2
            else _facet_vertices("vertex_triangle")
        )
        state = _facet_state(vertices)
        law = FacetFacetSpringForce(**dict.fromkeys(FACET_GROUPS, False))
        i, j = 0, dim
    force, torque, energy, actual_history = _pair(state, law, i, j, history)
    for actual in (force, torque, energy):
        assert np.all(np.isfinite(actual))
        np.testing.assert_array_equal(actual, 0.0)
    np.testing.assert_array_equal(actual_history, history)


def test_two_dimensional_facet_contacts_use_edge_edge_group():
    state = _facet_state(
        jnp.array(
            [
                [[0.0, 0.0], [2.0, 0.0]],
                [[0.5, 0.1], [1.5, 0.1]],
            ]
        )
    )
    default = _pair(state, FacetFacetSpringForce(), 0, 2)
    only = _pair(
        state, FacetFacetSpringForce(vertex_triangle=False, edge_triangle=False), 0, 2
    )
    assert np.linalg.norm(default[0]) > 0.0
    for actual, expected in zip(only, default):
        np.testing.assert_array_equal(actual, expected)
    without = _pair(state, FacetFacetSpringForce(edge_edge=False), 0, 2)
    for actual in without[:3]:
        np.testing.assert_array_equal(actual, 0.0)


def test_sphere_switches_apply_in_batched_calls_and_nested_composition():
    state = _sphere_state(3, [0.5, 0.5, 0.1])
    spheres = jd.State.create(
        pos=jnp.array([[0.5, -0.1, 0.1], [-0.1, -0.1, 0.1]]),
        rad=jnp.full(2, 0.2),
    )
    state = jd.State.merge(state, spheres)
    law = SphereFacetSpringForce(edge=False, vertex=False)
    router = jd.ForceRouter.from_dict(2, {(0, 1): jd.LawCombiner(laws=(law,))})
    system = jd.System.create(state=state, force_model=router)
    i, j = 0, jnp.array([3, 4, 5])
    history = router.init_history((3,), 3)
    force, torque, _ = router.force(i, j, state.pos, state, system, history)
    energy = router.energy(i, j, state.pos, state, system, history)
    expected = _pair(state, law, i, j)
    np.testing.assert_allclose(force, expected[0], atol=1e-12)
    np.testing.assert_allclose(torque, expected[1], atol=1e-12)
    np.testing.assert_allclose(energy, expected[2], atol=1e-12)
    assert np.linalg.norm(force[0]) > 0.0
    np.testing.assert_array_equal(force[1:], 0.0)
    np.testing.assert_array_equal(energy[1:], 0.0)


@pytest.mark.parametrize("rigid", [False, True])
def test_selective_sphere_contact_preserves_reciprocity_and_partition(rigid):
    state = _sphere_state(3, [0.5, 0.5, 0.1], rigid=rigid)
    law = SphereFacetSpringForce(edge=False, vertex=False)
    router = jd.ForceRouter.from_dict(2, {(0, 1): law})
    system = jd.System.create(state=state, force_model=router)
    actual, system = system.collider.compute_force(state, system)
    actual, _ = system.force_manager.apply(actual, system)
    sphere_force = actual.force[-1]
    assert np.linalg.norm(sphere_force) > 0.0
    if rigid:
        np.testing.assert_allclose(actual.force[0], -sphere_force, atol=1e-12)
    else:
        np.testing.assert_allclose(
            jnp.sum(actual.force[:3], axis=0), -sphere_force, atol=1e-12
        )
        weights = jnp.array([0.5, 0.25, 0.25])
        np.testing.assert_allclose(
            actual.force[:3], -weights[:, None] * sphere_force, atol=1e-12
        )
        np.testing.assert_array_equal(actual.torque[:3], 0.0)


def test_geometry_queries_support_broadcasting_and_collapsed_segments():
    state = jd.State.create(pos=jnp.zeros((1, 3)))
    system = jd.System.create(state=state)
    triangle = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    points = jnp.array([case[2] for case in SPHERE_CASES if case[0] == 3])
    distance, closest, coords = point_triangle_distance(
        points,
        *triangle,
        system,
        edge=False,
        vertex=False,
    )
    np.testing.assert_allclose(distance[0], 0.1)
    assert np.all(np.isinf(distance[1:]))
    assert np.all(np.isfinite(closest))
    assert np.all(np.isfinite(coords))

    origin = jnp.zeros(3)
    point = jnp.array([0.1, 0.0, 0.0])
    distance, closest, coords = point_segment_distance(
        point,
        origin,
        origin,
        system,
        edge=False,
    )
    np.testing.assert_allclose(distance, 0.1)
    np.testing.assert_allclose(closest, origin)
    np.testing.assert_allclose(jnp.sum(coords), 1.0)
    distance, closest, coords = point_segment_distance(
        point,
        origin,
        origin,
        system,
        vertex=False,
    )
    assert np.isinf(distance)
    assert np.all(np.isfinite(closest))
    assert np.all(np.isfinite(coords))


def test_disabled_triangle_query_families_are_absent_from_trace():
    state = jd.State.create(pos=jnp.zeros((1, 3)))
    system = jd.System.create(state=state)
    vertices = _facet_vertices("edge_triangle").reshape((6, 3))

    def query(points, **flags):
        return triangle_triangle_distance(*points, system, **flags)

    full = str(jax.make_jaxpr(query)(vertices))
    reduced = str(
        jax.make_jaxpr(
            lambda points: query(points, edge_edge=False, edge_triangle=False)
        )(vertices)
    )
    assert "segment_segment_distance" in full
    assert "_segment_triangle_intersection" in full
    assert "point_triangle_distance" in reduced
    assert "segment_segment_distance" not in reduced
    assert "_segment_triangle_intersection" not in reduced


@pytest.mark.parametrize(
    "family,flags",
    [
        ("sphere_facet_spring", {"edge": False, "vertex": False}),
        ("facet_facet_spring", {"edge_edge": False, "edge_triangle": False}),
    ],
)
def test_controls_survive_metadata_and_hdf5_round_trip(tmp_path, family, flags):
    h5 = pytest.importorskip("jaxdem.utils.h5")
    law = jd.ForceModel.create(family, **flags)
    from_metadata = jd.ForceModel.create(family, **law.metadata)
    path = str(tmp_path / "facet-law.h5")
    h5.save(law, path)
    restored = h5.load(path)
    for actual in (from_metadata, restored):
        for flag, value in flags.items():
            assert getattr(actual, flag) is value
        assert not any(
            isinstance(leaf, bool) for leaf in jax.tree_util.tree_leaves(actual)
        )
        # Static flags are not array leaves and survive a tree reconstruction.
        reconstructed = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(actual),
            jax.tree_util.tree_leaves(actual),
        )
        for flag, value in flags.items():
            assert getattr(reconstructed, flag) is value


@pytest.mark.parametrize(
    "family,flag",
    [
        ("sphere_facet_spring", "face"),
        ("sphere_facet_spring", "edge"),
        ("sphere_facet_spring", "vertex"),
        ("facet_facet_spring", "vertex_triangle"),
        ("facet_facet_spring", "edge_edge"),
        ("facet_facet_spring", "edge_triangle"),
    ],
)
def test_controls_reject_non_boolean_configuration(family, flag):
    with pytest.raises(TypeError, match=flag + " must be a Python bool"):
        jd.ForceModel.create(family, **{flag: "false"})


@pytest.mark.parametrize("family", ["sphere", "facet"])
def test_batched_disabled_pairs_keep_output_shapes(family):
    if family == "sphere":
        state = _sphere_state(3, [0.5, 0.5, 0.1])
        law = SphereFacetSpringForce(face=False, edge=False, vertex=False)
    else:
        state = _facet_state(_facet_vertices("vertex_triangle"))
        law = FacetFacetSpringForce(**dict.fromkeys(FACET_GROUPS, False))
    force, torque, energy, history = _pair(
        state, law, jnp.array([0, 3]), jnp.array([3, 0])
    )
    assert force.shape == (2, 3)
    assert torque.shape == (2, 3)
    assert energy.shape == (2,)
    assert history.shape == (2, 0)
    for actual in (force, torque, energy):
        np.testing.assert_array_equal(actual, 0.0)


@pytest.mark.parametrize("family", ["sphere", "facet"])
def test_rejected_selective_contact_has_finite_zero_energy_gradient(family):
    if family == "sphere":
        state = _sphere_state(3, [0.5, -0.1, 0.1])
        law = SphereFacetSpringForce(edge=False, vertex=False)
    else:
        state = _facet_state(_facet_vertices("edge_edge"))
        law = FacetFacetSpringForce(vertex_triangle=False, edge_edge=False)
    system = jd.System.create(state=state, force_model=law)
    history = law.init_history((), 3)

    def energy(positions):
        return law.energy(0, 3, positions, state, system, history)

    value, gradient = jax.value_and_grad(energy)(state.pos)
    np.testing.assert_array_equal(value, 0.0)
    assert np.all(np.isfinite(gradient))
    np.testing.assert_array_equal(gradient, 0.0)


def test_face_only_projection_retains_thin_nonzero_area_triangle():
    state = jd.State.create(pos=jnp.zeros((1, 3)))
    system = jd.System.create(state=state)
    triangle = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1e-8, 0.0],
        ]
    )
    point = jnp.array([0.75, 2.5e-9, 0.1])
    distance, closest, coords = point_triangle_distance(
        point,
        *triangle,
        system,
        edge=False,
        vertex=False,
    )
    assert np.isfinite(distance)
    np.testing.assert_allclose(distance, 0.1, rtol=1e-6)
    np.testing.assert_allclose(coords, [0.25, 0.5, 0.25], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(closest, [0.75, 2.5e-9, 0.0], rtol=1e-6, atol=1e-8)
