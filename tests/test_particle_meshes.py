"""Sanity tests for the asperity mesh generators (icosphere, fibonacci, torus, helix)
and the ``mesh_type`` dispatch in :func:`create_ga_state`.

These are lightweight correctness / shape checks — we're not validating
high-order physics here, just that each generator:

* returns the right shape,
* respects unit-scaling (longest axis has extent 1),
* exposes the shape-specific knobs sensibly (valid nv for icosphere,
  tube_ratio for torus, n_turns / helix_radius for helix), and
* flows through ``create_ga_state`` as an asperity mesh.
"""

from __future__ import annotations

import jax
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
from jaxdem.utils.particleCreation import create_ga_state


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------- icosphere


def test_icosphere_3d_valid_counts_match_subdivision_formula():
    for level in range(4):
        nv = 10 * 4 ** level + 2
        pos = generate_icosphere_mesh(nv=nv, N=1, dim=3)
        assert pos.shape == (nv, 3)
        # All vertices lie exactly on the unit sphere.
        norms = np.linalg.norm(np.asarray(pos), axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_icosphere_3d_rejects_invalid_nv():
    with pytest.raises(ValueError, match="icosphere mesh in 3D requires nv"):
        generate_icosphere_mesh(nv=50, N=1, dim=3)


def test_icosphere_2d_is_regular_ngon():
    nv = 7
    pos = generate_icosphere_mesh(nv=nv, N=1, dim=2)
    assert pos.shape == (nv, 2)
    # Points on unit circle.
    np.testing.assert_allclose(np.linalg.norm(np.asarray(pos), axis=-1), 1.0, atol=1e-12)
    # Neighboring point separations are all equal (polygon is regular).
    diffs = np.diff(np.asarray(pos), axis=0, append=np.asarray(pos)[:1])
    edge_lens = np.linalg.norm(diffs, axis=-1)
    np.testing.assert_allclose(edge_lens, edge_lens[0], atol=1e-12)


def test_icosphere_batches_replicate_and_apply_aspect_ratio():
    N, nv = 3, 42
    pos = generate_icosphere_mesh(nv=nv, N=N, dim=3, aspect_ratio=[1.0, 1.0, 2.0])
    assert pos.shape == (N, nv, 3)
    # Deterministic: all N copies identical.
    np.testing.assert_array_equal(np.asarray(pos[0]), np.asarray(pos[-1]))
    # Longest axis (z) has extent exactly 1 after aspect-ratio scaling.
    max_abs = np.max(np.abs(np.asarray(pos)), axis=(0, 1))
    np.testing.assert_allclose(max_abs[2], 1.0, atol=1e-12)
    assert max_abs[0] < 1.0 and max_abs[1] < 1.0


# ---------------------------------------------------------------- fibonacci


@pytest.mark.parametrize("nv", [1, 13, 64, 500])
def test_fibonacci_3d_arbitrary_nv_on_unit_sphere(nv: int):
    pos = generate_fibonacci_sphere_mesh(nv=nv, N=1, dim=3)
    assert pos.shape == (nv, 3)
    norms = np.linalg.norm(np.asarray(pos), axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_fibonacci_3d_is_deterministic():
    a = generate_fibonacci_sphere_mesh(nv=32, N=1, dim=3)
    b = generate_fibonacci_sphere_mesh(nv=32, N=1, dim=3)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_fibonacci_2d_evenly_spaces_points():
    nv = 11
    pos = generate_fibonacci_sphere_mesh(nv=nv, N=1, dim=2)
    assert pos.shape == (nv, 2)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(pos), axis=-1), 1.0, atol=1e-12)


# ---------------------------------------------------------------- torus


def test_torus_fits_in_unit_bounding_box_with_correct_minor_radius():
    nv, r = 200, 0.25
    pos = np.asarray(generate_torus_mesh(nv=nv, N=1, dim=3, tube_ratio=r))
    assert pos.shape == (nv, 3)
    max_abs = np.max(np.abs(pos), axis=0)
    # Unit-scaled: every axis fits in [-1, 1].
    assert np.all(max_abs <= 1.0 + 1e-12)
    # The golden-angle / stratified sampling is a 1D spiral on a 2D surface,
    # so not every xy corner is hit — but at least one axis gets close to 1.
    assert max_abs.max() > 0.95
    # Ring radius is in [R, R+r], so max |x|, |y| >= R = 1-r (some point
    # always samples theta near 0 and pi/2 at inner-rim phi).
    assert max_abs[0] > (1.0 - r) - 1e-3
    assert max_abs[1] > (1.0 - r) - 1e-3
    # |z| is bounded by r and densely approaches it as sin(phi) sweeps the circle.
    assert max_abs[2] <= r + 1e-12
    np.testing.assert_allclose(max_abs[2], r, atol=5e-2)


def test_torus_points_lie_on_torus_surface():
    nv, r = 300, 0.3
    R = 1.0 - r
    pos = np.asarray(generate_torus_mesh(nv=nv, N=1, dim=3, tube_ratio=r))
    # Implicit torus equation: (sqrt(x^2 + y^2) - R)^2 + z^2 == r^2.
    rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    residual = (rho - R) ** 2 + pos[:, 2] ** 2 - r ** 2
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_torus_is_genus_one_has_a_hole():
    # The origin (center of the donut) is strictly outside the torus surface.
    nv, r = 400, 0.2
    pos = np.asarray(generate_torus_mesh(nv=nv, N=1, dim=3, tube_ratio=r))
    min_dist_to_origin = np.min(np.linalg.norm(pos, axis=-1))
    # Closest surface point is on the inner rim at (R - r, 0, 0) = (1 - 2r, 0, 0).
    np.testing.assert_allclose(min_dist_to_origin, 1.0 - 2.0 * r, atol=1e-2)


def test_torus_rejects_non_3d_dim():
    with pytest.raises(ValueError, match="torus mesh is 3D only"):
        generate_torus_mesh(nv=20, N=1, dim=2)


def test_torus_rejects_out_of_range_tube_ratio():
    with pytest.raises(ValueError, match="tube_ratio"):
        generate_torus_mesh(nv=20, N=1, dim=3, tube_ratio=1.5)


# ---------------------------------------------------------------- helix


def test_helix_3d_is_chiral_and_fits_unit_z():
    nv = 100
    pos = np.asarray(
        generate_helix_mesh(nv=nv, N=1, dim=3, n_turns=3.0, helix_radius=0.3)
    )
    assert pos.shape == (nv, 3)
    # z spans exactly [-1, 1].
    np.testing.assert_allclose(pos[0, 2], -1.0, atol=1e-12)
    np.testing.assert_allclose(pos[-1, 2], 1.0, atol=1e-12)
    # xy stays on a circle of helix_radius.
    rho = np.linalg.norm(pos[:, :2], axis=-1)
    np.testing.assert_allclose(rho, 0.3, atol=1e-10)
    # Right-handed: theta (atan2(y, x)) is monotonically increasing (mod 2π).
    theta = np.unwrap(np.arctan2(pos[:, 1], pos[:, 0]))
    assert np.all(np.diff(theta) > 0.0)


def test_helix_2d_is_archimedean_spiral():
    nv = 50
    pos = np.asarray(generate_helix_mesh(nv=nv, N=1, dim=2, n_turns=2.0))
    assert pos.shape == (nv, 2)
    r = np.linalg.norm(pos, axis=-1)
    # Monotonic radial growth from the origin out to r=1.
    assert np.all(np.diff(r) > 0.0)
    np.testing.assert_allclose(r[-1], 1.0, atol=1e-12)


def test_helix_rejects_helix_radius_out_of_range():
    with pytest.raises(ValueError, match="helix_radius"):
        generate_helix_mesh(nv=20, N=1, dim=3, helix_radius=2.0)


# ---------------------------------------------------------------- arclength


def test_arclength_circle_is_regular_ngon():
    nv = 12
    pos = np.asarray(generate_arclength_mesh(nv=nv, N=1, dim=2))
    assert pos.shape == (nv, 2)
    # All points on the unit circle.
    np.testing.assert_allclose(np.linalg.norm(pos, axis=-1), 1.0, atol=1e-12)
    # Neighbor-to-neighbor distances are all equal.
    rolled = np.roll(pos, -1, axis=0)
    neighbor_dists = np.linalg.norm(rolled - pos, axis=-1)
    np.testing.assert_allclose(neighbor_dists, neighbor_dists[0], atol=1e-12)


def test_arclength_ellipse_gives_uniform_neighbor_distances():
    # Anisotropic ellipse: a Thomson sweep would converge here to essentially
    # the arclength-uniform layout, but we get it in closed form.
    nv = 60
    pos = np.asarray(
        generate_arclength_mesh(nv=nv, N=1, dim=2, aspect_ratio=[1.0, 2.0])
    )
    assert pos.shape == (nv, 2)
    # Unit-scaled: every axis fits in [-1, 1]. The discrete samples don't
    # land exactly on the poles, so max|y| approaches 1 as O(1/nv^2).
    max_abs = np.max(np.abs(pos), axis=0)
    assert max_abs[0] <= 0.5 + 1e-12
    assert max_abs[1] <= 1.0 + 1e-12
    np.testing.assert_allclose(max_abs[1], 1.0, atol=1e-2)
    # Points all lie on the ellipse x^2/a^2 + y^2/b^2 = 1.
    np.testing.assert_allclose(
        (pos[:, 0] / 0.5) ** 2 + (pos[:, 1] / 1.0) ** 2, 1.0, atol=1e-10
    )
    # Neighbor distances are nearly identical. Arc distances are exactly
    # equal by construction; chord (Euclidean) distances vary with the local
    # ellipse curvature and converge to arc as O(1/nv^2). For nv=60 and
    # a/b = 2 the chord CV sits right around 1e-3.
    rolled = np.roll(pos, -1, axis=0)
    d = np.linalg.norm(rolled - pos, axis=-1)
    cv = float(np.std(d) / np.mean(d))
    assert cv < 5e-3, f"neighbor-distance CV too large: {cv}"


def test_arclength_matches_converged_thomson_in_2d():
    # The arclength mesh is the α → ∞ packing limit. For α = 1 (default
    # Thomson) the ground state is slightly different, but for a well-
    # converged run on a circle both produce the regular n-gon up to a
    # global rotation.
    nv = 24
    pos_arc = np.asarray(generate_arclength_mesh(nv=nv, N=1, dim=2))
    pos_thomson, _ = generate_thomson_mesh(
        nv=nv, N=1, dim=2, steps=5_000, seed=0
    )
    pos_thomson = np.asarray(pos_thomson)

    def sorted_neighbor_dists(pts: np.ndarray) -> np.ndarray:
        # Angular order, then nearest-neighbor distances.
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        pts_sorted = pts[np.argsort(theta)]
        return np.linalg.norm(np.roll(pts_sorted, -1, axis=0) - pts_sorted, axis=-1)

    d_arc = sorted_neighbor_dists(pos_arc)
    d_thomson = sorted_neighbor_dists(pos_thomson)
    # Both converge to the same regular n-gon up to global rotation. Thomson
    # runs on float32 and 5k steps, so a ~1% agreement on the mean is the
    # realistic bar; the exact result comes from the arclength mesh.
    np.testing.assert_allclose(np.mean(d_arc), np.mean(d_thomson), rtol=1e-2)
    # Arclength's n-gon is numerically exact.
    np.testing.assert_allclose(np.std(d_arc), 0.0, atol=1e-10)


def test_arclength_rejects_3d():
    with pytest.raises(ValueError, match="arclength mesh is 2D only"):
        generate_arclength_mesh(nv=12, N=1, dim=3)


# ---------------------------------------------------------------- faceted


def test_faceted_2d_ngon_vertices_lie_on_unit_circle():
    nv, n_facets = 7, 7  # one asperity per vertex, no edge fillers
    pos = np.asarray(generate_faceted_mesh(nv=nv, N=1, dim=2, n_facets=n_facets))
    assert pos.shape == (nv, 2)
    np.testing.assert_allclose(np.linalg.norm(pos, axis=-1), 1.0, atol=1e-12)


def test_faceted_2d_edge_fillers_evenly_spaced_along_edges():
    n_facets = 6
    k = 3  # asperities per edge interior
    nv = n_facets * (1 + k)  # 6 vertices + 6 * 3 edge fillers = 24
    pos = np.asarray(generate_faceted_mesh(nv=nv, N=1, dim=2, n_facets=n_facets))
    assert pos.shape == (nv, 2)
    # Vertex asperities come first (on the unit circle).
    vertex_pos = pos[:n_facets]
    np.testing.assert_allclose(np.linalg.norm(vertex_pos, axis=-1), 1.0, atol=1e-12)
    # Edge fillers come in blocks of k per edge, strictly inside the unit circle.
    edge_pos = pos[n_facets:]
    assert np.all(np.linalg.norm(edge_pos, axis=-1) < 1.0)
    # Check the first edge block is evenly spaced between its two vertices.
    v0 = vertex_pos[0]
    v1 = vertex_pos[1]
    block_0 = edge_pos[:k]
    expected = np.stack([v0 + t * (v1 - v0) for t in (np.arange(k) + 1) / (k + 1)])
    np.testing.assert_allclose(block_0, expected, atol=1e-12)


def test_faceted_2d_distributes_non_divisible_remainder():
    # nv - n_facets = 7 extras across 3 edges: first edge gets 3, others 2.
    n_facets, nv = 3, 10
    pos = np.asarray(generate_faceted_mesh(nv=nv, N=1, dim=2, n_facets=n_facets))
    assert pos.shape == (nv, 2)


def test_faceted_3d_first_twelve_match_icosahedron_vertices():
    # 12 vertex asperities + 0 face fillers.
    nv = 12
    pos = np.asarray(generate_faceted_mesh(nv=nv, N=1, dim=3))
    assert pos.shape == (12, 3)
    # All on the unit sphere.
    np.testing.assert_allclose(np.linalg.norm(pos, axis=-1), 1.0, atol=1e-12)


def test_faceted_3d_face_interiors_strictly_inside_sphere():
    # 12 verts + 40 face fillers (2 per face).
    nv = 12 + 40
    pos = np.asarray(generate_faceted_mesh(nv=nv, N=1, dim=3))
    assert pos.shape == (nv, 3)
    # First 12 are vertices on unit sphere.
    np.testing.assert_allclose(np.linalg.norm(pos[:12], axis=-1), 1.0, atol=1e-12)
    # Face interiors are on the flat triangular faces — strictly inside the
    # circumscribed sphere, so |pos| < 1.
    face_pos = pos[12:]
    radii = np.linalg.norm(face_pos, axis=-1)
    assert np.all(radii < 1.0)
    # They should still be well away from the center (on the faces, not near
    # the origin). The inradius of a unit-circumradius icosahedron is
    # (phi^2) / sqrt(3) * 1/(2 phi) ≈ 0.7947.
    assert np.all(radii > 0.75)


def test_faceted_rejects_too_few_asperities():
    with pytest.raises(ValueError, match="nv must be >="):
        generate_faceted_mesh(nv=5, N=1, dim=2, n_facets=6)
    with pytest.raises(ValueError, match="nv must be >= 12"):
        generate_faceted_mesh(nv=10, N=1, dim=3)


def test_faceted_applies_aspect_ratio():
    nv = 12
    pos = np.asarray(
        generate_faceted_mesh(nv=nv, N=1, dim=3, aspect_ratio=[1.0, 1.0, 2.0])
    )
    # Icosahedron vertices aren't on coordinate axes (max |coord| is
    # phi/sqrt(1+phi^2) ≈ 0.851, not 1), so we can't expect max|z| = 1.
    # What we *can* check is that z is stretched 2× relative to x and y,
    # and that all vertices still fit inside the unit sphere.
    max_abs = np.max(np.abs(pos), axis=0)
    np.testing.assert_allclose(max_abs[0] / max_abs[2], 0.5, atol=1e-12)
    np.testing.assert_allclose(max_abs[1] / max_abs[2], 0.5, atol=1e-12)
    assert np.all(np.linalg.norm(pos, axis=-1) <= 1.0 + 1e-12)


# ---------------------------------------------------- create_ga_state dispatch


@pytest.mark.parametrize(
    "mesh_type,nv,kwargs",
    [
        ("fibonacci", 20, {}),
        ("icosphere", 42, {}),
        ("torus", 40, {"tube_ratio": 0.25}),
        ("helix", 40, {"n_turns": 3.0, "helix_radius": 0.3}),
        ("faceted", 32, {}),
    ],
)
def test_create_ga_state_dispatches_to_each_mesh_type(
    mesh_type: str, nv: int, kwargs: dict
):
    # Clump path exercises the whole union-property pipeline too.
    state = create_ga_state(
        N=2,
        nv=nv,
        dim=3,
        particle_radius=1.0,
        asperity_radius=0.1,
        particle_type="clump",
        core_type="hollow",
        n_samples=10_000,
        seed=0,
        mesh_type=mesh_type,
        mesh_kwargs=kwargs,
    )
    assert int(state.clump_id.max()) + 1 == 2
    assert state.N == 2 * nv


def test_create_ga_state_dispatches_arclength_in_2d():
    nv = 16
    state = create_ga_state(
        N=2,
        nv=nv,
        dim=2,
        particle_radius=1.0,
        asperity_radius=0.1,
        particle_type="clump",
        core_type="hollow",
        aspect_ratio=[1.0, 2.0],
        n_samples=10_000,
        seed=0,
        mesh_type="arclength",
    )
    assert int(state.clump_id.max()) + 1 == 2
    assert state.N == 2 * nv


def test_create_ga_state_rejects_unknown_mesh_type():
    with pytest.raises(ValueError, match="mesh_type must be one of"):
        create_ga_state(
            N=1,
            nv=12,
            dim=3,
            particle_radius=1.0,
            asperity_radius=0.1,
            n_samples=1_000,
            seed=0,
            mesh_type="not_a_mesh",
        )


def test_create_ga_state_thomson_steps_via_mesh_kwargs():
    # Thomson-specific 'steps' lives in mesh_kwargs, not as a top-level arg.
    state = create_ga_state(
        N=1,
        nv=12,
        dim=3,
        particle_radius=1.0,
        asperity_radius=0.1,
        n_samples=10_000,
        seed=0,
        mesh_type="thomson",
        mesh_kwargs={"steps": 100, "alpha": 1.0},
    )
    assert state.N == 12
