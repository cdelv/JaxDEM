# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Known geometry, sampling rules and reconstruction of asperity meshes."""

from __future__ import annotations

import jax.numpy as jnp

import numpy as np
import pytest

from jaxdem.utils import (
    generate_arclength_mesh,
    generate_faceted_mesh,
    generate_fibonacci_sphere_mesh,
    generate_helix_mesh,
    generate_icosphere_mesh,
    generate_thomson_mesh,
    generate_torus_mesh,
)
from jaxdem.utils.meshes import _icosahedron_faces, dimensionless_overlap_energy
from jaxdem.utils.particle_creation import create_ga_state

# ---------------------------------------------------------------- icosphere


def test_icosphere_3d_valid_counts_match_subdivision_formula():
    for level in range(3):
        nv = 10 * 4**level + 2
        pos = np.asarray(generate_icosphere_mesh(nv=nv, N=1, dim=3))
        assert pos.shape == (nv, 3)
        np.testing.assert_allclose(np.linalg.norm(pos, axis=-1), 1.0, atol=1e-10)
        assert np.unique(np.round(pos, decimals=9), axis=0).shape[0] == nv


def test_icosphere_2d_is_regular_ngon():
    nv = 7
    pos = np.asarray(generate_icosphere_mesh(nv=nv, N=1, dim=2))
    assert pos.shape == (nv, 2)
    np.testing.assert_allclose(np.linalg.norm(pos, axis=-1), 1.0, atol=1e-12)
    diffs = np.diff(pos, axis=0, append=pos[:1])
    edge_lens = np.linalg.norm(diffs, axis=-1)
    np.testing.assert_allclose(edge_lens, edge_lens[0], atol=1e-12)


def test_icosphere_batches_replicate_and_apply_aspect_ratio():
    N, nv = 3, 42
    pos = np.asarray(
        generate_icosphere_mesh(nv=nv, N=N, dim=3, aspect_ratio=[1.0, 1.0, 2.0])
    )
    assert pos.shape == (N, nv, 3)
    for idx in range(1, N):
        np.testing.assert_array_equal(pos[idx], pos[0])
    max_abs = np.max(np.abs(pos), axis=(0, 1))
    np.testing.assert_allclose(max_abs, [0.5, 0.5, 1.0], atol=1e-12)


# ---------------------------------------------------------------- fibonacci


@pytest.mark.parametrize("nv", [1, 32])
def test_fibonacci_3d_arbitrary_nv_on_unit_sphere(nv: int):
    pos = np.asarray(generate_fibonacci_sphere_mesh(nv=nv, N=1, dim=3))
    assert pos.shape == (nv, 3)
    np.testing.assert_allclose(np.linalg.norm(pos, axis=-1), 1.0, atol=1e-10)
    assert np.unique(pos, axis=0).shape[0] == nv
    np.testing.assert_array_equal(
        pos, np.asarray(generate_fibonacci_sphere_mesh(nv=nv, N=1, dim=3))
    )
    if nv > 1:
        np.testing.assert_allclose(pos.mean(axis=0), 0.0, atol=0.1)


def test_fibonacci_2d_evenly_spaces_points():
    nv = 11
    pos = np.asarray(generate_fibonacci_sphere_mesh(nv=nv, N=1, dim=2))
    assert pos.shape == (nv, 2)
    np.testing.assert_allclose(np.linalg.norm(pos, axis=-1), 1.0, atol=1e-12)
    angles = np.sort(np.mod(np.arctan2(pos[:, 1], pos[:, 0]), 2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    np.testing.assert_allclose(gaps, 2 * np.pi / nv, atol=1e-12)


# ---------------------------------------------------------------- torus


def test_torus_surface_geometry():
    nv, r = 300, 0.25
    R = 1.0 - r
    pos = np.asarray(generate_torus_mesh(nv=nv, N=1, dim=3, tube_ratio=r))
    assert pos.shape == (nv, 3)
    max_abs = np.max(np.abs(pos), axis=0)
    assert np.all(max_abs <= 1.0 + 1e-12)
    assert max_abs.max() > 0.95
    assert max_abs[0] > (1.0 - r) - 1e-3 and max_abs[1] > (1.0 - r) - 1e-3
    assert max_abs[2] <= r + 1e-12
    np.testing.assert_allclose(max_abs[2], r, atol=5e-2)

    # Implicit torus equation: (sqrt(x^2 + y^2) - R)^2 + z^2 == r^2.
    rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    residual = (rho - R) ** 2 + pos[:, 2] ** 2 - r**2
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)

    # Genus-1 inner hole: closest surface point sits at radius R - r = 1 - 2r.
    np.testing.assert_allclose(
        np.min(np.linalg.norm(pos, axis=-1)), 1.0 - 2.0 * r, atol=2e-2
    )


# ---------------------------------------------------------------- helix


def test_helix_3d_is_chiral_and_fits_unit_z():
    nv = 100
    pos = np.asarray(
        generate_helix_mesh(nv=nv, N=1, dim=3, n_turns=3.0, helix_radius=0.3)
    )
    assert pos.shape == (nv, 3)
    np.testing.assert_allclose(pos[0, 2], -1.0, atol=1e-12)
    np.testing.assert_allclose(pos[-1, 2], 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(pos[:, :2], axis=-1), 0.3, atol=1e-10)
    theta = np.unwrap(np.arctan2(pos[:, 1], pos[:, 0]))
    assert np.all(np.diff(theta) > 0.0)
    np.testing.assert_allclose(theta[-1] - theta[0], 6.0 * np.pi, atol=1e-10)


def test_helix_2d_is_archimedean_spiral():
    nv = 50
    pos = np.asarray(generate_helix_mesh(nv=nv, N=1, dim=2, n_turns=2.0))
    assert pos.shape == (nv, 2)
    r = np.linalg.norm(pos, axis=-1)
    assert np.all(np.diff(r) > 0.0)
    np.testing.assert_allclose(r[-1], 1.0, atol=1e-12)
    theta = np.unwrap(np.arctan2(pos[1:, 1], pos[1:, 0]))
    np.testing.assert_allclose(theta, 4 * np.pi * r[1:], atol=1e-12)


# ---------------------------------------------------------------- arclength & thomson


def test_arclength_circle_is_regular_ngon():
    nv = 12
    pos = np.asarray(generate_arclength_mesh(nv=nv, N=1, dim=2))
    assert pos.shape == (nv, 2)
    np.testing.assert_allclose(np.linalg.norm(pos, axis=-1), 1.0, atol=1e-12)
    neighbor_dists = np.linalg.norm(np.roll(pos, -1, axis=0) - pos, axis=-1)
    np.testing.assert_allclose(neighbor_dists, neighbor_dists[0], atol=1e-12)


def test_arclength_ellipse_gives_uniform_neighbor_distances():
    nv = 60
    pos = np.asarray(
        generate_arclength_mesh(nv=nv, N=1, dim=2, aspect_ratio=[1.0, 2.0])
    )
    assert pos.shape == (nv, 2)
    max_abs = np.max(np.abs(pos), axis=0)
    assert max_abs[0] <= 0.5 + 1e-12 and max_abs[1] <= 1.0 + 1e-12
    np.testing.assert_allclose(max_abs[1], 1.0, atol=1e-2)
    np.testing.assert_allclose(
        (pos[:, 0] / 0.5) ** 2 + (pos[:, 1] / 1.0) ** 2, 1.0, atol=1e-10
    )
    d = np.linalg.norm(np.roll(pos, -1, axis=0) - pos, axis=-1)
    assert float(np.std(d) / np.mean(d)) < 5e-3


def test_thomson_mesh_takes_the_known_projected_repulsion_step(monkeypatch):
    from jaxdem.utils import meshes

    initial = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(
        meshes,
        "random_points_on_hyper_ellipsoid",
        lambda *args, **kw: (initial, jnp.ones(2)),
    )
    pos, energy = generate_thomson_mesh(nv=2, N=1, dim=2, alpha=2, lr=8, steps=1)
    expected = np.array([[4, -1], [-1, 4]]) / np.sqrt(17)
    np.testing.assert_allclose(pos, expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(energy, 17 / 50, rtol=2e-6)


def test_dimensionless_overlap_energy_penalizes_fractional_overlap():
    big_pair = jnp.array([[0.0, 0.0], [0.9, 0.0]])
    small_pair = jnp.array([[0.0, 0.0], [0.1, 0.0]])
    big_energy = dimensionless_overlap_energy(big_pair, jnp.array([0.5, 0.5]))
    small_energy = dimensionless_overlap_energy(small_pair, jnp.array([0.1, 0.1]))
    assert small_energy > big_energy
    np.testing.assert_allclose(float(small_energy / big_energy), 25.0, rtol=1e-6)


def test_thomson_mesh_accepts_batched_asperity_radii():
    radii = jnp.array([[1.0, 1.2, 1.4, 1.6], [2.1, 2.1, 2.1, 2.1]])
    pos, energy = generate_thomson_mesh(
        nv=4,
        N=2,
        dim=2,
        steps=2,
        seed=0,
        asperity_radii=radii,
        overlap_strength=10.0,
    )
    assert pos.shape == (2, 4, 2)
    assert energy.shape == (2,)
    expected = np.zeros(2)
    for b in range(2):
        for i in range(4):
            for j in range(i + 1, 4):
                distance = np.linalg.norm(np.asarray(pos[b, i] - pos[b, j]))
                overlap = max(0.0, 1 - distance / float(radii[b, i] + radii[b, j]))
                expected[b] += 1 / distance + 5 * overlap**2
    np.testing.assert_allclose(energy, expected, rtol=3e-6, atol=3e-6)


# ---------------------------------------------------------------- faceted


def test_faceted_2d_vertices_and_non_divisible_edge_fillers():
    # nv == n_facets: vertices only on the unit circle.
    verts_only = np.asarray(generate_faceted_mesh(nv=7, N=1, dim=2, n_facets=7))
    assert verts_only.shape == (7, 2)
    np.testing.assert_allclose(np.linalg.norm(verts_only, axis=-1), 1.0, atol=1e-12)

    # nv - n_facets = 7 extras across 3 edges: first edge gets 3, others 2.
    n_facets, nv = 3, 10
    pos = np.asarray(generate_faceted_mesh(nv=nv, N=1, dim=2, n_facets=n_facets))
    assert pos.shape == (nv, 2)
    angles = 2 * np.pi * np.arange(n_facets) / n_facets
    vertices = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    np.testing.assert_allclose(pos[:n_facets], vertices, atol=1e-12)
    start = n_facets
    for edge, count in enumerate([3, 2, 2]):
        fraction = np.arange(1, count + 1)[:, None] / (count + 1)
        expected = vertices[edge] + fraction * (
            vertices[(edge + 1) % n_facets] - vertices[edge]
        )
        np.testing.assert_allclose(pos[start : start + count], expected, atol=1e-12)
        start += count


def test_faceted_3d_vertices_and_face_interiors_lie_on_icosahedron():
    nv = 12 + 40
    pos = np.asarray(generate_faceted_mesh(nv=nv, N=1, dim=3))
    assert pos.shape == (nv, 3)
    verts = pos[:12]
    np.testing.assert_allclose(np.linalg.norm(verts, axis=-1), 1.0, atol=1e-12)
    phi = (1 + np.sqrt(5)) / 2
    expected = np.array(
        [(0, a, b * phi) for a in [-1, 1] for b in [-1, 1]]
        + [(a, b * phi, 0) for a in [-1, 1] for b in [-1, 1]]
        + [(b * phi, 0, a) for a in [-1, 1] for b in [-1, 1]]
    ) / np.sqrt(1 + phi**2)
    distances = np.linalg.norm(verts[:, None] - expected[None], axis=-1)
    np.testing.assert_allclose(distances.min(axis=1), 0, atol=1e-12)
    assert np.unique(distances.argmin(axis=1)).size == 12

    # Every face filler must lie in the plane of one of the 20 icosahedron faces
    # (outward normal . p == inradius) and strictly inside the unit sphere.
    faces = np.asarray(_icosahedron_faces())
    face_centers = verts[faces].mean(axis=1)
    face_normals = face_centers / np.linalg.norm(face_centers, axis=-1, keepdims=True)
    inradius = np.einsum("ij,ij->i", face_normals, verts[faces[:, 0]])
    face_pos = pos[12:]
    assert np.all(np.linalg.norm(face_pos, axis=-1) < 1.0)
    plane_residuals = np.abs(face_pos @ face_normals.T - inradius[None, :])
    np.testing.assert_allclose(plane_residuals.min(axis=1), 0.0, atol=1e-12)


def test_faceted_applies_aspect_ratio():
    pos = np.asarray(
        generate_faceted_mesh(nv=12, N=1, dim=3, aspect_ratio=[1.0, 1.0, 2.0])
    )
    max_abs = np.max(np.abs(pos), axis=0)
    np.testing.assert_allclose(max_abs[0] / max_abs[2], 0.5, atol=1e-12)
    np.testing.assert_allclose(max_abs[1] / max_abs[2], 0.5, atol=1e-12)
    assert np.all(np.linalg.norm(pos, axis=-1) <= 1.0 + 1e-12)


# ---------------------------------------------------- create_ga_state dispatch


@pytest.mark.parametrize(
    "mesh_type,dim,nv,aspect_ratio,kwargs",
    [
        ("fibonacci", 3, 20, None, {}),
        ("icosphere", 3, 42, None, {}),
        ("torus", 3, 40, None, {"tube_ratio": 0.25}),
        ("helix", 3, 40, None, {"n_turns": 3.0, "helix_radius": 0.3}),
        ("faceted", 3, 32, None, {}),
        ("arclength", 2, 16, [1.0, 2.0], {}),
    ],
)
def test_create_ga_state_dispatches_to_each_mesh_type(
    mesh_type: str, dim: int, nv: int, aspect_ratio, kwargs: dict
):
    state = create_ga_state(
        N=2,
        nv=nv,
        dim=dim,
        particle_radius=1.0,
        asperity_radius=0.1,
        particle_type="clump",
        core_type="hollow",
        aspect_ratio=aspect_ratio,
        n_samples=128,
        seed=0,
        mesh_type=mesh_type,
        mesh_kwargs=kwargs,
    )
    assert int(state.clump_id.max()) + 1 == 2
    assert state.N == 2 * nv
    generators = {
        "fibonacci": generate_fibonacci_sphere_mesh,
        "icosphere": generate_icosphere_mesh,
        "torus": generate_torus_mesh,
        "helix": generate_helix_mesh,
        "faceted": generate_faceted_mesh,
        "arclength": generate_arclength_mesh,
    }
    expected = 0.9 * generators[mesh_type](
        nv=nv, N=2, dim=dim, aspect_ratio=aspect_ratio, **kwargs
    )
    np.testing.assert_allclose(state.pos.reshape(2, nv, dim), expected, atol=2e-6)


def test_create_ga_state_thomson_steps_via_mesh_kwargs():
    common = dict(N=1, nv=4, dim=2, seed=7)
    mesh_kwargs = dict(steps=2, alpha=2.0, lr=0.4)
    state = create_ga_state(
        **common,
        particle_radius=1.0,
        asperity_radius=0.2,
        particle_type="dp",
        n_samples=256,
        mesh_type="thomson",
        core_type="hollow",
        mesh_kwargs=mesh_kwargs,
    )
    mesh, _ = generate_thomson_mesh(**common, **mesh_kwargs)
    np.testing.assert_allclose(state.pos, 0.8 * mesh, rtol=2e-6, atol=2e-7)


def test_create_ga_state_uses_polydisperse_radii_with_thomson_mesh():
    state = create_ga_state(
        N=1,
        nv=8,
        dim=2,
        particle_radius=1.0,
        asperity_radius=0.2,
        particle_type="clump",
        core_type="hollow",
        n_samples=256,
        seed=0,
        mesh_type="thomson",
        mesh_kwargs={"steps": 5, "overlap_strength": 10.0},
        asperity_dispersity_type="bidisperse",
        asperity_dispersity_kwargs={"size_ratio": 2.0, "fraction_small": 0.5},
    )
    assert state.N == 8
    unique_radii = np.unique(np.round(np.asarray(state.rad), decimals=12))
    assert unique_radii.size == 2
    np.testing.assert_allclose(unique_radii[1] / unique_radii[0], 2.0, rtol=1e-6)
    expected, _ = generate_thomson_mesh(
        nv=8,
        N=1,
        dim=2,
        seed=0,
        steps=5,
        overlap_strength=10.0,
        asperity_radii=state.rad / 0.8,
    )
    unaware, _ = generate_thomson_mesh(
        nv=8, N=1, dim=2, seed=0, steps=5, overlap_strength=0.0
    )
    np.testing.assert_allclose(state.pos, 0.8 * expected, rtol=2e-6, atol=2e-7)
    assert not np.allclose(state.pos, 0.8 * unaware, atol=1e-5)
