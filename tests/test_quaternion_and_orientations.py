# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Tests for Quaternion algebra, AD behavior, and randomize_orientations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import jaxdem as jd
from jaxdem.utils.quaternion import Quaternion
from jaxdem.utils.randomize_orientations import randomize_orientations


def test_quaternion_rotvec_roundtrip_double_cover_and_finite_grad_at_zero():
    rotvecs = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, -0.7, 0.2],
            [np.pi, 0.0, 0.0],
            [0.0, np.pi / np.sqrt(2.0), np.pi / np.sqrt(2.0)],
        ],
        dtype=float,
    )
    q = Quaternion.from_rotvec(rotvecs)
    norms = jnp.sqrt(q.w[..., 0] ** 2 + jnp.sum(q.xyz**2, axis=-1))
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    recovered = Quaternion.to_rotvec(q)
    np.testing.assert_allclose(recovered[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(recovered[1], rotvecs[1], atol=1e-6)
    # At theta = pi, +u*pi and -u*pi represent the same rotation; verify via Rotation.
    for idx in (2, 3):
        R_orig = Rotation.from_rotvec(np.asarray(rotvecs[idx])).as_matrix()
        R_rec = Rotation.from_rotvec(np.asarray(recovered[idx])).as_matrix()
        np.testing.assert_allclose(R_rec, R_orig, atol=1e-5)

    # Double cover: -q must map to the exact same rotvec with angle in [0, pi].
    q_neg = Quaternion(-q.w, -q.xyz)
    np.testing.assert_allclose(Quaternion.to_rotvec(q_neg)[1], rotvecs[1], atol=1e-6)

    # Gradients at the identity (theta = 0) must be finite and match exact first-order limits.
    jac_from = jax.jacfwd(lambda rv: Quaternion.from_rotvec(rv).xyz)(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(jac_from))
    np.testing.assert_allclose(jac_from, 0.5 * np.eye(3), atol=1e-6)

    jac_to = jax.jacfwd(
        lambda xyz: Quaternion.to_rotvec(Quaternion(jnp.array([1.0]), xyz))
    )(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(jac_to))
    np.testing.assert_allclose(jac_to, 2.0 * np.eye(3), atol=1e-6)


def test_quaternion_small_rotvec_unit_conj_inv_and_matmul():
    small_rv = jnp.array([3e-4, -5e-4, 2e-4], dtype=float)
    q_exact = Quaternion.from_rotvec(small_rv)
    q_approx = Quaternion.from_small_rotvec(small_rv)
    np.testing.assert_allclose(q_approx.w, q_exact.w, atol=1e-7)
    np.testing.assert_allclose(q_approx.xyz, q_exact.xyz, atol=1e-7)

    # Zero quaternion stays zero without NaN under unit() and inv().
    q_zero = Quaternion(jnp.zeros((1, 1)), jnp.zeros((1, 3)))
    q_zero_u = Quaternion.unit(q_zero)
    assert jnp.all(q_zero_u.w == 0.0) and jnp.all(q_zero_u.xyz == 0.0)
    q_zero_inv = Quaternion.inv(q_zero)
    assert jnp.all(q_zero_inv.w == 0.0) and jnp.all(q_zero_inv.xyz == 0.0)

    # Non-unit quaternion: q @ inv(q) == identity; unit quaternion: inv == conj.
    q_raw = Quaternion(jnp.array([[2.0]]), jnp.array([[1.0, -3.0, 0.5]]))
    prod = q_raw @ Quaternion.inv(q_raw)
    np.testing.assert_allclose(prod.w, [[1.0]], atol=1e-6)
    np.testing.assert_allclose(prod.xyz, [[0.0, 0.0, 0.0]], atol=1e-6)

    q_u = Quaternion.unit(q_raw)
    q_u_inv = Quaternion.inv(q_u)
    q_u_conj = Quaternion.conj(q_u)
    np.testing.assert_allclose(q_u_inv.w, q_u_conj.w, atol=1e-6)
    np.testing.assert_allclose(q_u_inv.xyz, q_u_conj.xyz, atol=1e-6)


def test_quaternion_rotate_and_matmul_match_scipy_in_3d_and_2d():
    rv1 = np.array([0.6, -0.3, 0.8])
    rv2 = np.array([-0.4, 0.9, -0.2])
    v3 = np.array([1.2, -0.5, 0.7])

    q1 = Quaternion.from_rotvec(jnp.asarray(rv1))
    q2 = Quaternion.from_rotvec(jnp.asarray(rv2))
    r1 = Rotation.from_rotvec(rv1)
    r2 = Rotation.from_rotvec(rv2)

    rotated = Quaternion.rotate(q1, jnp.asarray(v3))
    np.testing.assert_allclose(rotated, r1.apply(v3), atol=1e-6)
    np.testing.assert_allclose(Quaternion.rotate_back(q1, rotated), v3, atol=1e-6)

    composed = q1 @ q2
    np.testing.assert_allclose(
        Quaternion.rotate(composed, jnp.asarray(v3)),
        (r1 * r2).apply(v3),
        atol=1e-6,
    )

    # 2D rotation about z-axis matches 2x2 rotation matrix.
    theta = 1.15
    q2d = Quaternion.from_rotvec(jnp.array([0.0, 0.0, theta]))
    v2 = jnp.array([1.3, -0.8])
    expected_2d = np.array(
        [
            v2[0] * np.cos(theta) - v2[1] * np.sin(theta),
            v2[0] * np.sin(theta) + v2[1] * np.cos(theta),
        ]
    )
    rot_2d = Quaternion.rotate(q2d, v2)
    np.testing.assert_allclose(rot_2d, expected_2d, atol=1e-6)
    np.testing.assert_allclose(Quaternion.rotate_back(q2d, rot_2d), v2, atol=1e-6)

    # Wildcard 0D empty vector passes through; unsupported dimension raises ValueError.
    empty = jnp.zeros((2, 0))
    assert Quaternion.rotate(q1, empty).shape == (2, 0)
    with pytest.raises(ValueError, match="2D or 3D"):
        Quaternion.rotate(q1, jnp.ones((4,)))


def test_quaternion_from_rotation_matrix_covers_all_shepperd_branches():
    # Cover trace > 0 (identity + general) and the 3 negative-trace diagonal branches
    # (180-degree rotations about x, y, and z axes).
    rotations = [
        Rotation.identity(),
        Rotation.from_rotvec([0.5, -0.7, 0.4]),
        Rotation.from_rotvec([np.pi, 0.0, 0.0]),
        Rotation.from_rotvec([0.0, np.pi, 0.0]),
        Rotation.from_rotvec([0.0, 0.0, np.pi]),
        Rotation.from_rotvec([np.pi * 0.99, 0.05, -0.02]),
    ]
    mats = jnp.asarray([r.as_matrix() for r in rotations], dtype=float)
    qs = Quaternion.from_rotation_matrix(mats)

    norms = jnp.sqrt(qs.w[..., 0] ** 2 + jnp.sum(qs.xyz**2, axis=-1))
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    probe = jnp.array([0.7, -1.1, 0.4], dtype=float)
    probes = jnp.broadcast_to(probe, (len(rotations), 3))
    actual = Quaternion.rotate(qs, probes)
    expected = np.stack([r.apply(np.asarray(probe)) for r in rotations], axis=0)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("dim", [2, 3])
def test_randomize_orientations_preserves_spheres_and_syncs_clumps(dim):
    # 5 particles: clump 0 (idx 0, 1), clump 1 (idx 2, 3), single sphere (idx 4).
    pos = jnp.zeros((5, dim), dtype=float)
    clump_id = jnp.array([0, 0, 1, 1, 2], dtype=int)
    state = jd.State.create(pos=pos, clump_id=clump_id)
    sphere_q_before = Quaternion(
        jnp.array([[np.cos(0.3)]]),
        jnp.array([[0.0, 0.0, np.sin(0.3)]]),
    )
    state.q = Quaternion(
        w=state.q.w.at[4:5].set(sphere_q_before.w),
        xyz=state.q.xyz.at[4:5].set(sphere_q_before.xyz),
    )

    out = randomize_orientations(state, jax.random.key(42))

    norms = jnp.sqrt(out.q.w[..., 0] ** 2 + jnp.sum(out.q.xyz**2, axis=-1))
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    # Sphere (idx 4) must remain unchanged.
    np.testing.assert_allclose(out.q.w[4], sphere_q_before.w[0], atol=1e-6)
    np.testing.assert_allclose(out.q.xyz[4], sphere_q_before.xyz[0], atol=1e-6)

    # Members of the same clump must share identical orientation.
    np.testing.assert_allclose(out.q.w[0], out.q.w[1], atol=1e-6)
    np.testing.assert_allclose(out.q.xyz[0], out.q.xyz[1], atol=1e-6)
    np.testing.assert_allclose(out.q.w[2], out.q.w[3], atol=1e-6)
    np.testing.assert_allclose(out.q.xyz[2], out.q.xyz[3], atol=1e-6)

    # Distinct clumps must get distinct orientations.
    assert not jnp.allclose(out.q.xyz[0], out.q.xyz[2])

    if dim == 2:
        np.testing.assert_allclose(out.q.xyz[:, :2], 0.0, atol=1e-7)
    else:
        assert jnp.any(jnp.abs(out.q.xyz[:4, :2]) > 1e-3)


def test_randomize_orientations_supports_multi_leading_batch_axes():
    base = jd.State.create(
        pos=jnp.zeros((3, 3), dtype=float),
        clump_id=jnp.array([0, 0, 1], dtype=int),
    )
    batched = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (2, 3, *x.shape)).copy(), base
    )

    out = randomize_orientations(batched, jax.random.key(7))
    assert out.q.w.shape == (2, 3, 3, 1)
    assert out.q.xyz.shape == (2, 3, 3, 3)

    norms = jnp.sqrt(out.q.w[..., 0] ** 2 + jnp.sum(out.q.xyz**2, axis=-1))
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    # Sphere (idx 2) unchanged across all batch slots; clump (0, 1) synced per slot.
    np.testing.assert_allclose(out.q.w[:, :, 2, :], 1.0, atol=1e-6)
    np.testing.assert_allclose(out.q.xyz[:, :, 2, :], 0.0, atol=1e-6)
    np.testing.assert_allclose(out.q.xyz[:, :, 0, :], out.q.xyz[:, :, 1, :], atol=1e-6)

    # Distinct batch slots receive independent orientations.
    assert not jnp.allclose(out.q.xyz[0, 0, 0], out.q.xyz[0, 1, 0])
    assert not jnp.allclose(out.q.xyz[0, 0, 0], out.q.xyz[1, 0, 0])
