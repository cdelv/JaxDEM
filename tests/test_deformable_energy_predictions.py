"""Independent polygon/polyhedron energies and signed hinge moments."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("term", ["content", "measure_and_tension"])
def test_deformable_energy_and_forces_match_known_geometry(dim, term):
    if dim == 2:
        reference = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
        elements = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        content0, measures0 = 2.0, np.array([2.0, 1.0, 2.0, 1.0])
        theta = np.deg2rad(27)
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
    else:
        reference = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        elements = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        content0, measures0 = 1 / 3, np.array([1.0, 1.0, 0.5, 1.5])
        from scipy.spatial.transform import Rotation

        rotation = Rotation.from_euler("xyz", np.deg2rad([20, -35, 15])).as_matrix()
    em = np.arange(1.0, 5.0) if term != "content" else np.zeros(4)
    gamma = np.arange(2.0, 6.0) / 10 if term != "content" else np.zeros(4)
    ec = 3.0 if term == "content" else 0.0
    model = jd.BondedForceModel.create(
        "deformableparticlemodel",
        vertices=reference,
        elements=elements,
        elements_id=np.zeros(4, dtype=int),
        ec=ec,
        em=em,
        gamma=gamma,
    )

    def energy(points):
        if dim == 2:
            a, b = points, np.roll(points, -1, axis=0)
            content = np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
            measures = np.linalg.norm(b - a, axis=-1)
        else:
            content = np.linalg.det(points[1:] - points[0]) / 6
            triangles = points[elements]
            measures = (
                np.linalg.norm(
                    np.cross(
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 0],
                    ),
                    axis=-1,
                )
                / 2
            )
        return (
            0.5 * ec * (content - content0) ** 2 / content0
            + 0.5 * np.sum(em * (measures - measures0) ** 2 / measures0)
            + np.sum(gamma * measures)
        )

    scale = 1.25
    points = scale * reference @ rotation.T + np.arange(dim) * 0.7
    state = jd.State.create(pos=jnp.asarray(points))
    system = jd.System.create(state=state, collider_type="", bonded_force_model=model)
    expected = (
        0.5 * ec * content0 * (scale**dim - 1) ** 2
        + 0.5 * np.sum(em * measures0) * (scale ** (dim - 1) - 1) ** 2
        + scale ** (dim - 1) * np.sum(gamma * measures0)
    )
    np.testing.assert_allclose(
        model.compute_potential_energy(state.pos, state, system), expected, rtol=2e-6
    )
    force, _ = model.compute_forces(state.pos, state, system)
    predicted = np.empty_like(points)
    for index in np.ndindex(points.shape):
        delta = np.zeros_like(points)
        delta[index] = 1e-5
        predicted[index] = -(energy(points + delta) - energy(points - delta)) / (2e-5)
    np.testing.assert_allclose(force, predicted, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.sum(force, axis=0), 0.0, atol=2e-6)
    np.testing.assert_allclose(
        np.sum(
            np.cross(
                np.pad(points, ((0, 0), (0, 3 - dim))),
                np.pad(np.asarray(force), ((0, 0), (0, 3 - dim))),
            ),
            axis=0,
        ),
        0.0,
        atol=1e-5,
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_bending_wraps_signed_angles_and_has_the_correct_restoring_moment(dim):
    def points(theta):
        if dim == 2:
            return jnp.array(
                [[0.0, 0.0], [1.0, 0.0], [1 + jnp.cos(theta), jnp.sin(theta)]]
            )
        return jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -jnp.cos(theta), -jnp.sin(theta)],
            ]
        )

    rest, current = jnp.deg2rad(170.0), jnp.deg2rad(-170.0)
    elements = [[0, 1], [1, 2]] if dim == 2 else [[0, 1, 2], [1, 0, 3]]
    model = jd.BondedForceModel.create(
        "deformableparticlemodel",
        vertices=points(rest),
        elements=elements,
        element_adjacency=[[0, 1]],
        element_adjacency_edges=[[0, 1]] if dim == 3 else None,
        initial_bendings=[rest],
        eb=2.3,
        w_b=[1.7],
    )
    state = jd.State.create(pos=points(current))
    system = jd.System.create(state=state, collider_type="", bonded_force_model=model)
    delta = np.deg2rad(20.0)
    np.testing.assert_allclose(
        model.compute_potential_energy(state.pos, state, system),
        0.5 * 2.3 * 1.7 * delta**2,
        rtol=2e-6,
    )
    derivative = jax.grad(
        lambda theta: model.compute_potential_energy(points(theta), state, system)
    )(current)
    np.testing.assert_allclose(derivative, 2.3 * 1.7 * delta, rtol=2e-6)
