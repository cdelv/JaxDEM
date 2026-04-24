# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

"""Functions for calculating the dynamical matrix (hessian of the
potential energy w.r.t. the system's generalized coordinates).

- :func:`non_bonded_hessian` Works for spheres and for
  deformable-particle (DP) particles with using the positions of
  the spheres in both cases as the coordinates.
- :func:`bonded_hessian` Calculates the contribution from
  ``system.bonded_force_model`` (DP internal elastic / plastic
  energies: em, ec, eb, el, gamma). Adds to the non-bonded hessian by
  linearity: ``H_total = H_non_bonded + H_bonded``.
- :func:`clump_non_bonded_hessian` Works for rigid clumps
  using the center of mass coordinates and infinitesimal rotations
  :math:`(\\delta r, \\omega)`. Can also define it in terms of a
  scaling of the rotations :math:`(\\delta r, R \\omega)`.

- Works unchanged for any existing (or future) force model that
  correctly implements ``energy``.
- The bonded and non-bonded contributions stay algorithmically
  independent; the total dynamical matrix is the simple sum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System

def _pair_non_bonded_hessian_block(
    i: jax.Array,
    j: jax.Array,
    state: "State",
    system: "System",
) -> jax.Array:
    r"""Second derivative of the pair energy w.r.t. ``(r_i, r_j)``.

    Computed as

    .. math::
        H_{\mathrm{pair}} \;=\;
        \begin{pmatrix}
          \partial^2 \phi / \partial r_i \, \partial r_i &
          \partial^2 \phi / \partial r_i \, \partial r_j \\
          \partial^2 \phi / \partial r_j \, \partial r_i &
          \partial^2 \phi / \partial r_j \, \partial r_j
        \end{pmatrix}

    For a pair potential depending only on :math:`r_i - r_j`, all four
    sub-blocks share the same magnitude and sign pattern
    :math:`\mathrm{diag}(h, h)` off the main diagonal with :math:`-h`.
    We don't exploit that symmetry here — the autograd call recovers
    it for free.

    Returns a ``(2*dim, 2*dim)`` matrix. Padding entries (``j == -1``)
    and self-pairs (``i == j``) are zeroed via ``jnp.where``. We use
    the branch-select form rather than multiplicative masking because
    the hessian can contain NaN at singular geometries (e.g. ``r=0``
    from an ``i==safe_j`` slot collapse), and ``NaN * 0 = NaN`` while
    ``jnp.where(False, NaN, 0) = 0``.
    """
    dim = state.dim
    pos = state.pos
    valid = (j != -1) & (i != j)
    safe_j = jnp.maximum(j, 0)

    def phi(r_pair: jax.Array) -> jax.Array:
        r_i = r_pair[:dim]
        r_j = r_pair[dim:]
        pos_new = pos.at[i].set(r_i).at[safe_j].set(r_j)
        return system.force_model.energy(i, safe_j, pos_new, state, system)

    r_pair = jnp.concatenate([pos[i], pos[safe_j]])
    H = jax.hessian(phi)(r_pair)  # (2*dim, 2*dim)
    return jnp.where(valid, H, jnp.zeros_like(H))


def pair_non_bonded_hessian(
    state: "State",
    system: "System",
    cutoff: float | None = None,
    max_neighbors: int | None = None,
) -> tuple["State", "System", jax.Array, jax.Array]:
    r"""Per-pair non-bonded hessian blocks for every neighbor-list pair.

    Returns
    -------
    state : State
        Potentially updated state (after neighbor-list rebuild).
    system : System
        Potentially updated system.
    pair_ids : jax.Array
        ``(M, 2)`` int array of ``(i, j)`` sphere index pairs, where
        ``M = state.N * max_neighbors``. Padding pairs have ``j == -1``
        and corresponding ``blocks`` entries are zero.
    blocks : jax.Array
        ``(M, 2*dim, 2*dim)`` hessian blocks. ``blocks[k, :dim, :dim]``
        is :math:`\partial^2 \phi / \partial r_i^2`, the upper-right
        sub-block is :math:`\partial^2 \phi / (\partial r_i \partial r_j)`,
        and so on.
    """
    if cutoff is None:
        cutoff = float(jnp.max(state.rad)) * 3.0
    if max_neighbors is None:
        max_neighbors = 100

    state, system, nl, overflow = system.collider.create_neighbor_list(
        state, system, cutoff, max_neighbors
    )
    if overflow:
        raise ValueError("Neighbor list overflowed. Increase max_neighbors.")

    sphere_ids = jax.lax.iota(dtype=int, size=state.N)
    n_neighbors = nl.shape[1]
    i_ids = jnp.repeat(sphere_ids[:, None], n_neighbors, axis=1).ravel()
    j_ids = nl.ravel()

    blocks = jax.vmap(
        lambda i, j: _pair_non_bonded_hessian_block(i, j, state, system)
    )(i_ids, j_ids)

    pair_ids = jnp.column_stack((i_ids, j_ids))
    return state, system, pair_ids, blocks


def non_bonded_hessian(
    state: "State",
    system: "System",
    cutoff: float | None = None,
    max_neighbors: int | None = None,
) -> tuple["State", "System", jax.Array]:
    r"""Assemble the dense ``(N*dim, N*dim)`` non-bonded hessian of
    :math:`\partial^2 U / \partial r \, \partial r` where ``r``
    concatenates all sphere positions.

    Scatters the per-pair blocks from :func:`pair_non_bonded_hessian`
    into a dense matrix. Symmetric by construction; translational
    null modes (``sum of rows in each block-row == 0``) follow from
    each pair's 4-block structure.

    Each pair appears twice in the neighbor list (``(i, j)`` and
    ``(j, i)``) and each lane scatters the full 4-quadrant hessian
    into ``(i, i), (i, j), (j, i), (j, j)``, so every slot accumulates
    two identical contributions. We divide by 2 at the end to undo
    this, matching the ``0.5 * sum`` convention used in
    :meth:`NeighborList.compute_potential_energy`.

    See :func:`pair_non_bonded_hessian` for how padding entries are
    handled — padding blocks are zero and contribute nothing.
    """
    state, system, pair_ids, blocks = pair_non_bonded_hessian(
        state, system, cutoff, max_neighbors
    )
    N = int(state.N)
    dim = int(state.dim)

    # Flat-index view: map each pair's 4 sub-blocks to their positions
    # in the (N*dim, N*dim) matrix. For pair (i, j) the 2*dim row/col
    # indices are [i*dim, ..., i*dim+dim-1, j*dim, ..., j*dim+dim-1].
    di = jnp.arange(dim)  # (dim,)
    i_flat = pair_ids[:, 0:1] * dim + di[None, :]  # (M, dim)
    j_flat = pair_ids[:, 1:2] * dim + di[None, :]  # (M, dim)
    # Safe-index: padding j = -1 -> j_flat negative; mask at add time.
    valid = (pair_ids[:, 1:2] != -1) & (pair_ids[:, 0:1] != pair_ids[:, 1:2])
    j_flat = jnp.where(valid, j_flat, i_flat)  # redirect padding to self slot;
    # blocks[k] is already zero for padding entries, so the scatter adds
    # zero there and we avoid out-of-bounds indices.
    pair_rows = jnp.concatenate([i_flat, j_flat], axis=-1)  # (M, 2*dim)

    row_idx = pair_rows[:, :, None]
    col_idx = pair_rows[:, None, :]
    H_full = jnp.zeros((N * dim, N * dim), dtype=blocks.dtype)
    H_full = H_full.at[row_idx, col_idx].add(blocks)

    return state, system, 0.5 * H_full


def bonded_hessian(
    state: "State",
    system: "System",
) -> tuple["State", "System", jax.Array]:
    r"""Dense ``(N*dim, N*dim)`` hessian of the bonded potential energy
    :math:`\partial^2 U_{bonded} / \partial r \, \partial r`.

    Returns zeros when ``system.bonded_force_model`` is ``None``. The
    total dynamical matrix is the sum
    ``bonded_hessian + non_bonded_hessian`` (by linearity).

    Note: this function does **not** apply a ``0.5`` double-count
    correction. The bonded potential energy is a plain
    ``sum(over bonds/elements)``—each bond appears exactly once, so
    the hessian of the total energy is the direct sum of per-bond
    hessians with no correction. The ``0.5`` factor in
    :func:`non_bonded_hessian` undoes neighbor-list double-counting,
    which does not apply here.
    """
    n_total = int(state.N) * int(state.dim)
    if system.bonded_force_model is None:
        return state, system, jnp.zeros((n_total, n_total), dtype=state.pos.dtype)

    def u_bonded(pos: jax.Array) -> jax.Array:
        return system.bonded_force_model.compute_potential_energy(
            pos, state, system
        )

    h = jax.hessian(u_bonded)(state.pos)  # (N, dim, N, dim)
    return state, system, h.reshape(n_total, n_total)


def _rotation_perturbation(omega: jax.Array, p_lab: jax.Array) -> jax.Array:
    r"""Second-order-exact rigid-body displacement from a rotation tangent.

    Returns :math:`\omega \times p + \tfrac{1}{2}\, \omega \times (\omega \times p)`,
    the Taylor expansion of :math:`R(\omega) p - p` through quadratic order.
    Higher-order terms vanish under ``∂²/∂ω²`` evaluated at ``ω = 0``, so
    this is exact for the hessian and avoids any manifold / quaternion
    calculus.
    """
    dim = p_lab.shape[-1]
    if dim == 2:
        w = omega[0]
        cross1 = w * jnp.stack([-p_lab[1], p_lab[0]])
        cross2 = -(w ** 2) * p_lab  # ω × (ω × p) in 2D
        return cross1 + 0.5 * cross2
    cross1 = jnp.cross(omega, p_lab)
    cross2 = jnp.cross(omega, cross1)
    return cross1 + 0.5 * cross2


def _pair_clump_non_bonded_hessian_block(
    i: jax.Array,
    j: jax.Array,
    state: "State",
    system: "System",
) -> jax.Array:
    r"""Contribution of a single sphere pair ``(i, j)`` to the clump-pair
    hessian block, parameterized by ``q = (δr_cI, ω_I, δr_cJ, ω_J)``.

    Returns a ``(2*group_dim, 2*group_dim)`` matrix where
    ``group_dim = dim + rot_dim`` (3 in 2D, 6 in 3D). Same padding /
    self-pair safety trick as the sphere version.
    """
    dim = state.dim
    rot_dim = 1 if dim == 2 else 3
    group_dim = dim + rot_dim
    pos = state.pos

    # Valid when: j is not padding, and the two spheres belong to
    # different clumps (intra-clump pairs contribute nothing to the
    # inter-clump hessian).
    safe_j = jnp.maximum(j, 0)
    valid_pad = j != -1
    valid_clump = state.clump_id[i] != state.clump_id[safe_j]
    valid = valid_pad & valid_clump

    r_i_cur = pos[i]
    r_j_cur = pos[safe_j]
    p_i_lab = r_i_cur - state.pos_c[i]
    p_j_lab = r_j_cur - state.pos_c[safe_j]

    def phi(q_pair: jax.Array) -> jax.Array:
        q_i = q_pair[:group_dim]
        q_j = q_pair[group_dim:]
        delta_r_i = q_i[:dim] + _rotation_perturbation(q_i[dim:], p_i_lab)
        delta_r_j = q_j[:dim] + _rotation_perturbation(q_j[dim:], p_j_lab)
        r_i_new = r_i_cur + delta_r_i
        r_j_new = r_j_cur + delta_r_j
        pos_new = pos.at[i].set(r_i_new).at[safe_j].set(r_j_new)
        return system.force_model.energy(i, safe_j, pos_new, state, system)

    q_zero = jnp.zeros(2 * group_dim, dtype=pos.dtype)
    h = jax.hessian(phi)(q_zero)  # (2*group_dim, 2*group_dim)
    return jnp.where(valid, h, jnp.zeros_like(h))


def clump_non_bonded_hessian(
    state: "State",
    system: "System",
    cutoff: float | None = None,
    max_neighbors: int | None = None,
    rotation_scale: jax.Array | None = None,
) -> tuple["State", "System", jax.Array]:
    r"""Dense ``(n_clumps*group_dim, n_clumps*group_dim)`` clump hessian.

    Generalized coordinates per clump are ``(δr_c, ω)`` -- translation +
    small-rotation tangent (scalar in 2D, 3-vector in 3D). The hessian
    is accumulated by summing per-sphere-pair contributions via the
    rigid-body chain rule ``r_i = r_cI + p_i^lab``; see
    :func:`_pair_clump_non_bonded_hessian_block`.

    Parameters
    ----------
    rotation_scale : jax.Array, optional
        ``(n_clumps,)`` array of length scales (e.g., bounding-sphere
        radii) for the ``R·θ`` convention. When provided, the rotation
        rows/columns of clump ``I`` are divided by ``rotation_scale[I]``
        on output. Default (``None``) returns the angle-based hessian
        (ω in radians).
    """
    if cutoff is None:
        cutoff = float(jnp.max(state.rad)) * 3.0
    if max_neighbors is None:
        max_neighbors = 100

    state, system, nl, overflow = system.collider.create_neighbor_list(
        state, system, cutoff, max_neighbors
    )
    if overflow:
        raise ValueError("Neighbor list overflowed. Increase max_neighbors.")

    sphere_ids = jax.lax.iota(dtype=int, size=state.N)
    n_neighbors = nl.shape[1]
    i_ids = jnp.repeat(sphere_ids[:, None], n_neighbors, axis=1).ravel()
    j_ids = nl.ravel()

    blocks = jax.vmap(
        lambda i, j: _pair_clump_non_bonded_hessian_block(i, j, state, system)
    )(i_ids, j_ids)  # (M, 2*group_dim, 2*group_dim)

    # Clump index per sphere pair.
    clump_i = state.clump_id[i_ids]
    clump_j = state.clump_id[jnp.maximum(j_ids, 0)]
    # Invalid sphere pairs redirect to (0, 0) (or any same-clump target)
    # so the scatter adds zero; blocks are already zeroed for invalid.
    valid = (j_ids != -1) & (clump_i != clump_j)
    clump_j = jnp.where(valid, clump_j, clump_i)

    dim = int(state.dim)
    rot_dim = 1 if dim == 2 else 3
    group_dim = dim + rot_dim
    n_clumps = int(jnp.max(state.clump_id)) + 1

    dg = jnp.arange(group_dim)
    i_flat = clump_i[:, None] * group_dim + dg[None, :]  # (M, group_dim)
    j_flat = clump_j[:, None] * group_dim + dg[None, :]
    pair_rows = jnp.concatenate([i_flat, j_flat], axis=-1)  # (M, 2*group_dim)

    row_idx = pair_rows[:, :, None]
    col_idx = pair_rows[:, None, :]
    h_full = jnp.zeros((n_clumps * group_dim, n_clumps * group_dim), dtype=blocks.dtype)
    h_full = h_full.at[row_idx, col_idx].add(blocks)
    # Undo neighbor-list double-counting (see note in `non_bonded_hessian`).
    h_full = 0.5 * h_full

    if rotation_scale is not None:
        # Build a per-row scale vector: 1 for translational rows/cols,
        # 1/R_I for rotation rows/cols of clump I. Apply via outer product.
        rs = jnp.asarray(rotation_scale, dtype=h_full.dtype)
        is_rot = (dg >= dim).astype(h_full.dtype)  # (group_dim,)
        # Per clump, scale = 1 for trans indices, 1/R_I for rot indices.
        # Broadcast rs[:, None] * is_rot[None, :] + (1 - is_rot[None, :])  →  (n_clumps, group_dim)
        per_clump_scale = (1.0 - is_rot[None, :]) + is_rot[None, :] / rs[:, None]
        row_scale = per_clump_scale.reshape(-1)  # (n_clumps*group_dim,)
        h_full = h_full * row_scale[:, None] * row_scale[None, :]

    return state, system, h_full


def zero_mode_mask(
    eigenvalues: jax.Array,
    rel_gap: float = 1e4,
) -> jax.Array:
    r"""Boolean mask identifying numerically-zero eigenvalues via gap detection.

    Given a 1-D array of eigenvalues (in any order), we sort by
    :math:`|\lambda|` ascending and look for the largest relative gap
    :math:`|\lambda_{k+1}| / |\lambda_k|`. Entries below the first gap that
    exceeds ``rel_gap`` are flagged as numerically zero. The mask is
    returned aligned with the original eigenvalue ordering, so it can be
    used directly to slice eigenvectors too (e.g. ``evecs[:, ~mask]`` for
    finite modes).

    This is robust to problem scale: a hessian with ``|λ_max| ~ 1`` and
    zero modes at ``~ 1e-16`` gives the same mask as one with
    ``|λ_max| ~ 1e6`` and zero modes at ``~ 1e-10``, because the
    criterion is the *ratio* between successive magnitudes, not an
    absolute threshold.

    Parameters
    ----------
    eigenvalues : jax.Array
        1-D array of eigenvalues (e.g. from :func:`jax.numpy.linalg.eigvalsh`
        or :func:`jax.numpy.linalg.eigh`). Ordering is not required.
    rel_gap : float, optional
        Minimum ratio ``|λ_{k+1}| / |λ_k|`` that counts as the zero /
        finite boundary. Default ``1e4`` (four orders of magnitude). True
        zero modes at machine precision vs. real modes of order
        :math:`k \cdot \mathrm{overlap}` in a jammed spring packing
        typically sit 10-14 orders of magnitude apart, so the threshold
        is not sensitive.

    Returns
    -------
    jax.Array
        Boolean array of the same shape as ``eigenvalues``; ``True``
        where the eigenvalue is below the first large relative gap.
        If no gap larger than ``rel_gap`` is found (all eigenvalues are
        comparable in magnitude), returns all-``False``.
    """
    e = jnp.asarray(eigenvalues)
    abs_e = jnp.abs(e)
    sorted_idx = jnp.argsort(abs_e)
    sorted_abs = abs_e[sorted_idx]
    # Floor below which eigenvalues are indistinguishable from zero for
    # this problem's scale. Using machine eps times the largest magnitude
    # collapses exact zeros and machine-precision non-zeros into a single
    # "numerical zero" plateau so the gap-detection heuristic picks up
    # the real gap between numerical zeros and finite modes, instead of
    # latching onto the ratio between an exact 0 and some ~1e-16 value.
    eps = jnp.finfo(sorted_abs.dtype).eps
    tiny = jnp.finfo(sorted_abs.dtype).tiny
    floor = jnp.maximum(eps * jnp.max(abs_e), tiny)
    s_safe = jnp.maximum(sorted_abs, floor)
    ratios = s_safe[1:] / s_safe[:-1]
    # Boundary position: index of the first entry whose ratio to the next
    # exceeds rel_gap. If none, there is no gap.
    gap_exceeds = ratios > rel_gap
    any_gap = jnp.any(gap_exceeds)
    first_gap = jnp.argmax(gap_exceeds.astype(jnp.int32))
    n_zero = jnp.where(any_gap, first_gap + 1, 0)
    # Build a mask on the sorted order, then scatter back to original order.
    positions = jax.lax.iota(dtype=jnp.int32, size=e.shape[0])
    sorted_mask = positions < n_zero
    mask = jnp.zeros(e.shape[0], dtype=bool).at[sorted_idx].set(sorted_mask)
    return mask


__all__ = [
    "pair_non_bonded_hessian",
    "non_bonded_hessian",
    "bonded_hessian",
    "clump_non_bonded_hessian",
    "zero_mode_mask",
]
