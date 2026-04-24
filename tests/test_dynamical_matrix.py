"""Analytical tests for :mod:`jaxdem.utils.dynamicalMatrix`.

For a linear-spring contact potential between two spheres at positions
:math:`r_i, r_j` with radii summing to :math:`R` and overlap
:math:`s = R - r` (where :math:`r = |r_i - r_j|`), the pair energy is

.. math::
    U = \\tfrac{1}{2} k s^2 \\,\\Theta(s)

and the 4-block hessian of :math:`U` with respect to :math:`(r_i, r_j)`
is determined by a single :math:`(dim, dim)` matrix

.. math::
    h = k \\left[ \\hat{n}\\hat{n}^T - (s / r)\\,(I - \\hat{n}\\hat{n}^T) \\right]

where :math:`\\hat{n} = (r_i - r_j) / r`. The full block is

.. math::
    H = \\begin{pmatrix} h & -h \\\\ -h & h \\end{pmatrix}

which is symmetric, translationally invariant (``H.sum(axis=-1) = 0`` on
every row of each row-block), and vanishes outside contact (``s <= 0``).
These are the four facts we assert.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils.dynamicalMatrix import (
    bonded_hessian,
    clump_non_bonded_hessian,
    non_bonded_hessian,
    pair_non_bonded_hessian,
)

jax.config.update("jax_enable_x64", True)


def _build_two_sphere_system(
    pos_i,
    pos_j,
    rad_i=0.5,
    rad_j=0.5,
    young=1.0,
    dim=None,
    collider_type="naive",
):
    """Build a 2-sphere state + spring system with the given geometry."""
    pos = jnp.stack([jnp.asarray(pos_i, dtype=float), jnp.asarray(pos_j, dtype=float)])
    rad = jnp.asarray([rad_i, rad_j], dtype=float)
    state = jd.State.create(
        pos=pos,
        rad=rad,
        mass=jnp.ones(2),
        clump_id=jnp.array([0, 1], dtype=int),
    )
    if dim is None:
        dim = state.dim
    mats = [jd.Material.create("elastic", young=young, poisson=0.5, density=1.0)]
    mat_table = jd.MaterialTable.from_materials(
        mats, matcher=jd.MaterialMatchmaker.create("harmonic")
    )
    collider_kw = {}
    if collider_type in ("StaticCellList", "CellList"):
        collider_kw = {"state": state}
    system = jd.System.create(
        state_shape=state.shape,
        dt=1e-2,
        linear_integrator_type="",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type=collider_type,
        collider_kw=collider_kw,
        mat_table=mat_table,
        domain_kw={"box_size": jnp.ones(dim) * 10.0},
    )
    return state, system


def _analytical_pair_block(r_i, r_j, R, k, dim):
    """Reference (2*dim, 2*dim) spring hessian block."""
    r_ij = np.asarray(r_i, dtype=float) - np.asarray(r_j, dtype=float)
    r = float(np.linalg.norm(r_ij))
    if r >= R:
        return np.zeros((2 * dim, 2 * dim))
    s = R - r
    n = r_ij / r
    n_outer = np.outer(n, n)
    I = np.eye(dim)
    h = k * (n_outer - (s / r) * (I - n_outer))
    H = np.block([[h, -h], [-h, h]])
    return H


def _analytical_clump_hessian_2d(world_pos, pos_c, rad, clump_id, k):
    """Reference 2D clump hessian built from the explicit rigid-body derivation
    (original `compute_hessian_clumps_2d` convention). Returns an
    ``(n_clumps*3, n_clumps*3)`` matrix in ``(δR_x, δR_y, ω)`` ordering.

    For each ordered pair ``(μ, ν)`` of spheres in different clumps that are
    in contact, with ``n̂ = (r_μ - r_ν)/r``, ``s = R - r``, ``t = -k·s``,
    ``c = k``, ``tr = t/r``, and Jacobians

        E_μ = [[1, 0], [0, 1], [-p_μ_y, p_μ_x]]    (p_μ = r_μ - r_cI)

    the contributions are

        p_μ = E_μ · n̂
        off(ci, cj)  =  -tr · E_μ E_ν^T - (c - tr) · p_μ p_ν^T
        diag(ci, ci) =   tr · E_μ E_μ^T + (c - tr) · p_μ p_μ^T
        diag[2, 2]   += -t · (n̂ · p_μ)           # rigid-body 2nd-order term

    The sum over ordered pairs fills both off-diagonal entries and both
    diagonal blocks by symmetry.
    """
    world_pos = np.asarray(world_pos, dtype=float)
    pos_c = np.asarray(pos_c, dtype=float)
    rad = np.asarray(rad, dtype=float)
    clump_id = np.asarray(clump_id, dtype=int)
    N = world_pos.shape[0]
    N_c = int(clump_id.max()) + 1
    df = 3
    H = np.zeros((N_c * df, N_c * df))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if clump_id[i] == clump_id[j]:
                continue
            r_vec = world_pos[i] - world_pos[j]
            r = float(np.linalg.norm(r_vec))
            R = rad[i] + rad[j]
            if r >= R:
                continue
            nhat = r_vec / r
            s = R - r
            t = -k * s
            c = k
            tr = t / r

            p_mu = world_pos[i] - pos_c[i]
            p_nu = world_pos[j] - pos_c[j]

            E_mu = np.array([[1.0, 0.0], [0.0, 1.0], [-p_mu[1], p_mu[0]]])
            E_nu = np.array([[1.0, 0.0], [0.0, 1.0], [-p_nu[1], p_nu[0]]])

            p_mu_proj = E_mu @ nhat
            p_nu_proj = E_nu @ nhat

            EE_cross = E_mu @ E_nu.T
            pp_cross = np.outer(p_mu_proj, p_nu_proj)
            off_block = -tr * EE_cross - (c - tr) * pp_cross

            EE_same = E_mu @ E_mu.T
            pp_same = np.outer(p_mu_proj, p_mu_proj)
            diag_block = tr * EE_same + (c - tr) * pp_same
            diag_block[2, 2] -= t * float(nhat @ p_mu)

            ci, cj = int(clump_id[i]), int(clump_id[j])
            H[ci * df : (ci + 1) * df, cj * df : (cj + 1) * df] += off_block
            H[ci * df : (ci + 1) * df, ci * df : (ci + 1) * df] += diag_block

    return H


def _analytical_clump_hessian_3d(world_pos, pos_c, rad, clump_id, k):
    """Reference 3D clump hessian, generalizing the 2D derivation.

    The 3D Jacobian ``E_μ`` is a ``(6, 3)`` matrix built as block-vertical
    ``[[I_3], [skew(p_μ)]]``, where the lower block follows from
    ``∂(ω × p)_α / ∂ω_a = skew(p)[α, a]`` (row ``a+3`` of E_μ is the row of
    ``skew(p_μ)`` that equals ``ẑ_a × p_μ``).

    All the pair-block algebra is unchanged from 2D, except the rigid-body
    2nd-order correction is now a ``3×3`` sub-matrix on the rotation block:

    From ``R(ω) p ≈ p + ω×p + ½ ω×(ω×p)`` we get
    ``∂² r_μ_i / ∂ω_a ∂ω_b = ½(δ_ia p_b + δ_ib p_a) - p_i δ_ab``
    which, chain-ruled with ``∂r/∂r_μ = n̂`` and multiplied by ``t = dφ/dr``,
    gives the correction

        C = t · [½(n̂ ⊗ p_μ + p_μ ⊗ n̂) - (n̂·p_μ) I_3]

    applied to ``diag_block[3:, 3:]``. This reduces to the 2D
    ``-t·(n̂·p)`` on the single ω-entry when ``p`` and ``n̂`` lie in the xy
    plane (trace-like term survives, symmetric part vanishes).
    """
    world_pos = np.asarray(world_pos, dtype=float)
    pos_c = np.asarray(pos_c, dtype=float)
    rad = np.asarray(rad, dtype=float)
    clump_id = np.asarray(clump_id, dtype=int)
    N = world_pos.shape[0]
    N_c = int(clump_id.max()) + 1
    df = 6
    H = np.zeros((N_c * df, N_c * df))

    def skew(p):
        return np.array(
            [
                [0.0, -p[2], p[1]],
                [p[2], 0.0, -p[0]],
                [-p[1], p[0], 0.0],
            ]
        )

    for i in range(N):
        for j in range(N):
            if i == j or clump_id[i] == clump_id[j]:
                continue
            r_vec = world_pos[i] - world_pos[j]
            r = float(np.linalg.norm(r_vec))
            R = rad[i] + rad[j]
            if r >= R:
                continue
            nhat = r_vec / r
            s = R - r
            t = -k * s
            c = k
            tr = t / r

            p_mu = world_pos[i] - pos_c[i]
            p_nu = world_pos[j] - pos_c[j]

            E_mu = np.vstack([np.eye(3), skew(p_mu)])  # (6, 3)
            E_nu = np.vstack([np.eye(3), skew(p_nu)])  # (6, 3)

            p_mu_proj = E_mu @ nhat  # (6,)
            p_nu_proj = E_nu @ nhat  # (6,)

            EE_cross = E_mu @ E_nu.T  # (6, 6)
            pp_cross = np.outer(p_mu_proj, p_nu_proj)
            off_block = -tr * EE_cross - (c - tr) * pp_cross

            EE_same = E_mu @ E_mu.T
            pp_same = np.outer(p_mu_proj, p_mu_proj)
            diag_block = tr * EE_same + (c - tr) * pp_same

            # Rigid-body 2nd-order correction on the rotation sub-block.
            n_dot_p = float(nhat @ p_mu)
            C = t * (
                0.5 * (np.outer(nhat, p_mu) + np.outer(p_mu, nhat))
                - n_dot_p * np.eye(3)
            )
            diag_block[3:, 3:] += C

            ci, cj = int(clump_id[i]), int(clump_id[j])
            H[ci * df : (ci + 1) * df, cj * df : (cj + 1) * df] += off_block
            H[ci * df : (ci + 1) * df, ci * df : (ci + 1) * df] += diag_block

    return H


# Pair-block correctness — compare to analytical spring hessian
@pytest.mark.parametrize(
    "dim, pos_i, pos_j",
    [
        (2, [0.0, 0.0], [0.8, 0.0]),  # aligned
        (2, [0.0, 0.0], [0.5, 0.5]),  # 45-degree, r ≈ 0.707, s ≈ 0.293
        (3, [0.0, 0.0, 0.0], [0.8, 0.0, 0.0]),  # aligned
        (3, [0.0, 0.0, 0.0], [0.3, 0.4, 0.6]),  # r ≈ 0.781, s ≈ 0.219
    ],
)
def test_pair_block_matches_analytical_in_contact(dim, pos_i, pos_j):
    """In-contact pair hessian block matches the analytical spring form."""
    state, system = _build_two_sphere_system(pos_i, pos_j)
    _, _, pair_ids, blocks = pair_non_bonded_hessian(
        state, system, cutoff=10.0, max_neighbors=4
    )
    # Find the (0, 1) pair (not (1, 0), and not a padding/self pair).
    pair_ids_np = np.asarray(pair_ids)
    mask = (pair_ids_np[:, 0] == 0) & (pair_ids_np[:, 1] == 1)
    assert mask.any()
    k_index = int(np.argmax(mask))
    H_pair = np.asarray(blocks[k_index])

    expected = _analytical_pair_block(pos_i, pos_j, R=1.0, k=1.0, dim=dim)
    np.testing.assert_allclose(H_pair, expected, atol=1e-12)


@pytest.mark.parametrize(
    "dim, pos_i, pos_j",
    [
        (2, [0.0, 0.0], [1.5, 0.0]),
        (3, [0.0, 0.0, 0.0], [1.2, 0.8, 0.5]),
    ],
)
def test_pair_block_is_zero_out_of_contact(dim, pos_i, pos_j):
    """For separation beyond R = r_i + r_j the pair hessian is zero."""
    state, system = _build_two_sphere_system(pos_i, pos_j)
    _, _, pair_ids, blocks = pair_non_bonded_hessian(
        state, system, cutoff=10.0, max_neighbors=4
    )
    pair_ids_np = np.asarray(pair_ids)
    mask = (pair_ids_np[:, 0] == 0) & (pair_ids_np[:, 1] == 1)
    k_index = int(np.argmax(mask))
    H_pair = np.asarray(blocks[k_index])
    np.testing.assert_allclose(H_pair, 0.0, atol=1e-14)


# Pair-block structural properties


@pytest.mark.parametrize(
    "dim, pos_i, pos_j",
    [
        (2, [0.0, 0.0], [0.5, 0.5]),
        (3, [0.0, 0.0, 0.0], [0.3, 0.4, 0.6]),
    ],
)
def test_pair_block_is_symmetric_and_translation_invariant(dim, pos_i, pos_j):
    state, system = _build_two_sphere_system(pos_i, pos_j)
    _, _, pair_ids, blocks = pair_non_bonded_hessian(
        state, system, cutoff=10.0, max_neighbors=4
    )
    pair_ids_np = np.asarray(pair_ids)
    mask = (pair_ids_np[:, 0] == 0) & (pair_ids_np[:, 1] == 1)
    k_index = int(np.argmax(mask))
    H = np.asarray(blocks[k_index])

    # H is a symmetric matrix.
    np.testing.assert_allclose(H, H.T, atol=1e-12)

    # Translational invariance: sum over (r_i, r_j) vanishes.
    # For the 4-block [[h, -h], [-h, h]], column sums are zero.
    np.testing.assert_allclose(H.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(H.sum(axis=1), 0.0, atol=1e-12)


# Full system hessian — agrees with pair block for a 2-sphere system


@pytest.mark.parametrize(
    "dim, pos_i, pos_j",
    [
        (2, [0.0, 0.0], [0.5, 0.5]),
        (3, [0.0, 0.0, 0.0], [0.3, 0.4, 0.6]),
    ],
)
def test_full_hessian_agrees_with_pair_block(dim, pos_i, pos_j):
    """For a two-sphere system, the full (N*dim, N*dim) hessian equals the
    analytical one-pair hessian. Each pair appears twice in the neighbor
    list, but `non_bonded_hessian` undoes that with a 0.5 factor (matching
    the `0.5 * sum` convention used in the potential-energy path)."""
    state, system = _build_two_sphere_system(pos_i, pos_j)
    _, _, H_full = non_bonded_hessian(state, system, cutoff=10.0, max_neighbors=4)
    H_full_np = np.asarray(H_full)

    expected_one_pair = _analytical_pair_block(pos_i, pos_j, R=1.0, k=1.0, dim=dim)
    np.testing.assert_allclose(H_full_np, expected_one_pair, atol=1e-12)


# Clump hessian


@pytest.mark.parametrize("dim", [2, 3])
def test_clump_hessian_matches_sphere_lift_for_single_sphere_clumps(dim):
    """For single-sphere clumps (p_i_lab = 0), the rotation chain-rule
    Jacobian is ``J = [I, 0]`` (rotating about the COM moves nothing
    since the sphere IS the COM). So the clump hessian's translational
    block equals the sphere hessian and the rotational block is zero."""
    if dim == 2:
        state, system = _build_two_sphere_system([0.0, 0.0], [0.5, 0.5])
    else:
        state, system = _build_two_sphere_system([0.0, 0.0, 0.0], [0.3, 0.4, 0.6])
    _, _, H_clump = clump_non_bonded_hessian(
        state, system, cutoff=10.0, max_neighbors=4
    )
    rot_dim = 1 if dim == 2 else 3
    group_dim = dim + rot_dim
    H_clump_np = np.asarray(H_clump)
    # Expected assembled translational block: 2 * sphere pair block
    # (for the two neighbor-list directions).
    _, _, H_sphere = non_bonded_hessian(state, system, cutoff=10.0, max_neighbors=4)
    H_sphere_np = np.asarray(H_sphere)

    # Extract translational rows/cols of H_clump (indices 0..dim-1 and
    # group_dim..group_dim+dim-1 for the two clumps).
    trans_idx = np.concatenate([np.arange(dim), np.arange(dim) + group_dim])
    H_trans = H_clump_np[np.ix_(trans_idx, trans_idx)]
    np.testing.assert_allclose(H_trans, H_sphere_np, atol=1e-12)

    # Rotational rows/cols should be identically zero for single-sphere clumps.
    rot_idx = np.concatenate(
        [np.arange(dim, group_dim), np.arange(dim, group_dim) + group_dim]
    )
    H_rot = H_clump_np[np.ix_(rot_idx, rot_idx)]
    np.testing.assert_allclose(H_rot, 0.0, atol=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_clump_hessian_is_symmetric(dim):
    """Clump hessian must be symmetric (second derivative of a scalar)."""
    if dim == 2:
        state, system = _build_two_sphere_system([0.0, 0.0], [0.5, 0.5])
    else:
        state, system = _build_two_sphere_system([0.0, 0.0, 0.0], [0.3, 0.4, 0.6])
    _, _, H = clump_non_bonded_hessian(state, system, cutoff=10.0, max_neighbors=4)
    H_np = np.asarray(H)
    np.testing.assert_allclose(H_np, H_np.T, atol=1e-12)


def test_clump_hessian_translational_null_mode():
    """Sum of columns across translational DOFs must vanish (rigid
    translation of the whole system leaves energy unchanged)."""
    state, system = _build_two_sphere_system([0.0, 0.0, 0.0], [0.3, 0.4, 0.6])
    _, _, H = clump_non_bonded_hessian(state, system, cutoff=10.0, max_neighbors=4)
    H_np = np.asarray(H)
    dim = 3
    rot_dim = 3
    group_dim = dim + rot_dim
    n_clumps = 2
    # Build a uniform translation eigenvector: translate every clump by
    # the same (δR, 0, 0, 0) direction. Expect H @ v = 0.
    v = np.zeros(n_clumps * group_dim)
    for I in range(n_clumps):
        v[I * group_dim + 0] = 1.0  # translate in +x
    np.testing.assert_allclose(H_np @ v, 0.0, atol=1e-12)


def test_clump_hessian_offset_geometry_matches_closed_form():
    """Two-sphere clump contacting a single-sphere clump off-axis in 2D.

    Clump 0: spheres at (0, 0), (2, 0); COM (1, 0); pos_p for contact sphere (1, 0).
    Clump 1: sphere at (2, 0.8); COM (2, 0.8).

    Only the B–C pair is in contact with overlap s = 0.2 and force
    purely in -y. The jacobian J = ∂r_BC/∂q at q=0 (coords ordered
    (δR0_x, δR0_y, ω_0, δR1_x, δR1_y, ω_1)) is

        J_x = [ 1,  0,  0, -1,  0, 0]
        J_y = [ 0,  1,  1,  0, -1, 0]

    With ∂²φ/∂v² = [[-0.25, 0], [0, 1]] at the current overlap, the
    ∂φ/∂v · ∂²r/∂q² term contributes zero here (force is purely in y
    while ∂²r/∂ω_0² is in x), so H = J^T · M · J exactly. The neighbor
    list visits the pair twice but `clump_non_bonded_hessian` undoes
    that double-count with a 0.5 factor, so the single-pair formula
    matches directly.
    """
    pos_c = jnp.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.8]])
    pos_p = jnp.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    rad = jnp.full((3,), 0.5)
    state = jd.State.create(
        pos=pos_c,
        pos_p=pos_p,
        rad=rad,
        mass=jnp.ones(3),
        clump_id=jnp.array([0, 0, 1], dtype=int),
    )
    mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
    mat_table = jd.MaterialTable.from_materials(
        mats, matcher=jd.MaterialMatchmaker.create("harmonic")
    )
    system = jd.System.create(
        state_shape=state.shape,
        dt=1e-2,
        linear_integrator_type="",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw={"box_size": jnp.ones(2) * 10.0},
    )

    _, _, H = clump_non_bonded_hessian(state, system, cutoff=10.0, max_neighbors=4)
    H_np = np.asarray(H)

    # Analytical per-pair block from H_single = J^T M J.
    M = np.array([[-0.25, 0.0], [0.0, 1.0]])
    J = np.array([[1.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, -1.0, 0.0]])
    expected = J.T @ M @ J

    np.testing.assert_allclose(H_np, expected, atol=1e-12)


def test_clump_hessian_rotation_scale_rescales_rotation_block():
    """Passing rotation_scale rescales rotation rows/cols by 1/R."""
    state, system = _build_two_sphere_system([0.0, 0.0, 0.0], [0.3, 0.4, 0.6])
    # For single-sphere clumps the rotation block is zero anyway, so
    # use the bigger regression check: full hessian stays symmetric and
    # the translational block is unchanged.
    _, _, H_unscaled = clump_non_bonded_hessian(
        state, system, cutoff=10.0, max_neighbors=4
    )
    R = jnp.array([0.5, 0.5])
    _, _, H_scaled = clump_non_bonded_hessian(
        state, system, cutoff=10.0, max_neighbors=4, rotation_scale=R
    )
    H_unscaled_np = np.asarray(H_unscaled)
    H_scaled_np = np.asarray(H_scaled)

    # Translational block unchanged.
    dim, rot_dim = 3, 3
    group_dim = dim + rot_dim
    trans_idx = np.concatenate([np.arange(dim), np.arange(dim) + group_dim])
    np.testing.assert_allclose(
        H_scaled_np[np.ix_(trans_idx, trans_idx)],
        H_unscaled_np[np.ix_(trans_idx, trans_idx)],
        atol=1e-12,
    )
    # Scaled still symmetric.
    np.testing.assert_allclose(H_scaled_np, H_scaled_np.T, atol=1e-12)


def test_clump_hessian_2d_matches_explicit_rigid_body_formula():
    """Full 2D clump hessian compared to the explicit rigid-body formula
    (original `compute_hessian_clumps_2d` derivation). Geometry is chosen
    so that the rigid-body 2nd-order correction ``-t·(n̂·p_μ)`` is non-zero,
    which exercises the rotation-rotation diagonal term that is absent in
    the "force purely perpendicular to lever arm" tests.

    Geometry (all radii 0.5, so R = 1.0):
        Clump 0 (two spheres):
            A at (0, 0), pos_p = (-1, -0.5)
            B at (2, 1), pos_p = ( 1,  0.5)
            COM = (1, 0.5)
        Clump 1 (single sphere):
            C at (2.5, 1.5), pos_p = (0, 0)

    Only the B-C pair is in contact:
        r_BC = (-0.5, -0.5), r = 1/sqrt(2), s = 1 - 1/sqrt(2) ≈ 0.2929
        n̂ = (-1, -1)/sqrt(2)
        p_B = (1, 0.5) so n̂ · p_B = -1.5/sqrt(2) ≠ 0    (non-trivial correction)
    """
    pos_c = jnp.array([[1.0, 0.5], [1.0, 0.5], [2.5, 1.5]])
    pos_p = jnp.array([[-1.0, -0.5], [1.0, 0.5], [0.0, 0.0]])
    rad = jnp.full((3,), 0.5)
    state = jd.State.create(
        pos=pos_c,
        pos_p=pos_p,
        rad=rad,
        mass=jnp.ones(3),
        clump_id=jnp.array([0, 0, 1], dtype=int),
    )
    # Sanity: world positions and pair geometry.
    world_pos = np.asarray(state.pos)
    np.testing.assert_allclose(world_pos[0], [0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(world_pos[1], [2.0, 1.0], atol=1e-14)
    np.testing.assert_allclose(world_pos[2], [2.5, 1.5], atol=1e-14)

    mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
    mat_table = jd.MaterialTable.from_materials(
        mats, matcher=jd.MaterialMatchmaker.create("harmonic")
    )
    system = jd.System.create(
        state_shape=state.shape,
        dt=1e-2,
        linear_integrator_type="",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw={"box_size": jnp.ones(2) * 20.0},
    )

    _, _, H = clump_non_bonded_hessian(state, system, cutoff=5.0, max_neighbors=4)
    H_np = np.asarray(H)

    expected = _analytical_clump_hessian_2d(
        world_pos=world_pos,
        pos_c=np.asarray(state.pos_c),
        rad=np.asarray(state.rad),
        clump_id=np.asarray(state.clump_id),
        k=1.0,
    )

    np.testing.assert_allclose(H_np, expected, atol=1e-12)


def test_clump_hessian_3d_matches_explicit_rigid_body_formula():
    """Full 3D clump hessian compared to the explicit rigid-body formula
    (3D generalization of the 2D derivation — see
    :func:`_analytical_clump_hessian_3d`). Geometry is chosen so that
    ``n̂·p_μ != 0`` so the rotation-block 2nd-order correction is non-trivial,
    and the pair direction has components along all three spatial axes so
    every 6×6 sub-block of the hessian has a chance to be exercised.

    Clump 0 (two spheres):
        A at (0, 0, 0),     pos_p = (-1, -0.5, -0.25)
        B at (2, 1, 0.5),   pos_p = ( 1,  0.5,  0.25)
        COM = (1, 0.5, 0.25)
    Clump 1 (single sphere):
        C at (2.5, 1.5, 0.7), pos_p = (0, 0, 0)

    Only B-C in contact: r_BC = (-0.5, -0.5, -0.2), r = sqrt(0.54),
    s = 1 - sqrt(0.54), n̂·p_B = -0.8/sqrt(0.54) ≠ 0.
    """
    pos_c = jnp.array([[1.0, 0.5, 0.25], [1.0, 0.5, 0.25], [2.5, 1.5, 0.7]])
    pos_p = jnp.array([[-1.0, -0.5, -0.25], [1.0, 0.5, 0.25], [0.0, 0.0, 0.0]])
    rad = jnp.full((3,), 0.5)
    state = jd.State.create(
        pos=pos_c,
        pos_p=pos_p,
        rad=rad,
        mass=jnp.ones(3),
        clump_id=jnp.array([0, 0, 1], dtype=int),
    )
    world_pos = np.asarray(state.pos)
    np.testing.assert_allclose(world_pos[0], [0.0, 0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(world_pos[1], [2.0, 1.0, 0.5], atol=1e-14)
    np.testing.assert_allclose(world_pos[2], [2.5, 1.5, 0.7], atol=1e-14)

    mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
    mat_table = jd.MaterialTable.from_materials(
        mats, matcher=jd.MaterialMatchmaker.create("harmonic")
    )
    system = jd.System.create(
        state_shape=state.shape,
        dt=1e-2,
        linear_integrator_type="",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw={"box_size": jnp.ones(3) * 20.0},
    )

    _, _, H = clump_non_bonded_hessian(state, system, cutoff=5.0, max_neighbors=4)
    H_np = np.asarray(H)

    expected = _analytical_clump_hessian_3d(
        world_pos=world_pos,
        pos_c=np.asarray(state.pos_c),
        rad=np.asarray(state.rad),
        clump_id=np.asarray(state.clump_id),
        k=1.0,
    )

    np.testing.assert_allclose(H_np, expected, atol=1e-12)


# Cross-collider consistency — the 0.5 double-count correction must be
# consistent across collider implementations.
@pytest.mark.parametrize("collider_type", ["naive", "StaticCellList", "CellList"])
@pytest.mark.parametrize(
    "pos_i, pos_j",
    [([0.0, 0.0], [0.5, 0.5]), ([0.0, 0.0], [0.8, 0.0])],
)
def test_non_bonded_hessian_is_collider_invariant(collider_type, pos_i, pos_j):
    """`non_bonded_hessian` should match the analytical spring hessian
    regardless of collider type (naive, NeighborList, CellList)."""
    state, system = _build_two_sphere_system(pos_i, pos_j, collider_type=collider_type)
    _, _, H_full = non_bonded_hessian(state, system, cutoff=10.0, max_neighbors=4)
    H_full_np = np.asarray(H_full)

    expected = _analytical_pair_block(pos_i, pos_j, R=1.0, k=1.0, dim=2)
    np.testing.assert_allclose(H_full_np, expected, atol=1e-12)


# Bonded hessian — analytical test against a single harmonic edge.


def test_bonded_hessian_matches_analytical_harmonic_edge():
    """For a DP with two vertices and a single edge (harmonic-bond-like),
    the potential is

        U = (1/2) · el · (L - L_0)^2 / L_0    (one edge, single sum)

    which is a harmonic bond with effective stiffness ``k = el/L_0``.
    Its hessian w.r.t. ``(r_0, r_1)`` is

        h = k · n̂ n̂^T + (k · δ / L) · (I - n̂ n̂^T)
        H = [[h, -h], [-h, h]]

    where ``n̂ = (r_0 - r_1)/L``, ``δ = L - L_0``. The stretched geometry
    ``L_0 = 1``, ``L = 1.5`` exercises both the radial (``n̂ n̂^T``) and
    transverse (``(I - n̂ n̂^T)``) stiffening terms.
    """
    # Rest configuration defines L_0 = 1.0.
    vertices = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    edges = jnp.array([[0, 1]], dtype=int)
    el_value = 2.0

    dp = jd.BondedForceModel.create(
        "deformableparticlemodel",
        vertices=vertices,
        edges=edges,
        el=el_value,
    )

    # Current (stretched) positions.
    pos = jnp.array([[0.0, 0.0], [1.5, 0.0]], dtype=float)
    state = jd.State.create(
        pos=pos,
        rad=jnp.full((2,), 0.1),
        mass=jnp.ones(2),
    )
    system = jd.System.create(
        state_shape=state.shape,
        dt=1e-2,
        linear_integrator_type="",
        rotation_integrator_type="",
        domain_type="free",
        force_model_type="spring",
        collider_type="naive",
        bonded_force_model=dp,
    )

    _, _, H = bonded_hessian(state, system)
    H_np = np.asarray(H)

    # Analytical harmonic-bond hessian.
    L = 1.5
    L_0 = 1.0
    delta = L - L_0
    k_eff = el_value / L_0
    n = np.array([-1.0, 0.0])  # (r_0 - r_1)/L; sign irrelevant (n⊗n even)
    I = np.eye(2)
    nn = np.outer(n, n)
    h = k_eff * nn + (k_eff * delta / L) * (I - nn)
    expected = np.block([[h, -h], [-h, h]])

    np.testing.assert_allclose(H_np, expected, atol=1e-10)


def test_bonded_hessian_returns_zeros_when_no_bonded_model():
    """When ``system.bonded_force_model`` is ``None``, `bonded_hessian`
    returns the zero matrix of the correct shape so that
    ``H_total = non_bonded + bonded`` works without branching."""
    state, system = _build_two_sphere_system([0.0, 0.0], [0.5, 0.0])
    assert system.bonded_force_model is None
    _, _, H = bonded_hessian(state, system)
    assert H.shape == (state.N * state.dim, state.N * state.dim)
    np.testing.assert_allclose(np.asarray(H), 0.0, atol=1e-14)
