# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Energy Hessians in sphere and rigid-clump coordinates.

Non-bonded contributions use the configured collider's candidates and the
force model's pair energy. Bonded contributions differentiate the bonded
model's total energy. Their sum gives the total potential-energy Hessian.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from ._pair_candidates import _collect_pair_candidates

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


_HESSIAN_BATCH_SIZE = 128


def _hessian_pairs(state: State, system: System) -> tuple[System, jax.Array, jax.Array]:
    """Collect unique sphere pairs and their two directed interaction masks."""
    system, pairs, valid, _ = _collect_pair_candidates(state, system)
    directed = np.asarray(pairs)[np.asarray(valid)]
    pairs_np, inverse = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True
    )
    directions = np.zeros((len(pairs_np), 2), dtype=bool)
    directions[inverse, (directed[:, 0] > directed[:, 1]).astype(int)] = True
    pairs = jnp.asarray(pairs_np)
    if len(pairs_np):
        radii = system.force_model.search_radii(state, system)
        i, j = pairs.T
        dr = system.domain.displacement(state.pos[i], state.pos[j], system)
        within_reach = np.asarray(
            jnp.sum(dr * dr, axis=-1) <= (radii[i] + radii[j]) ** 2
        )
        pairs = pairs[within_reach]
        directions = directions[within_reach]
    return system, pairs, jnp.asarray(directions)


def _rotation_perturbation(omega: jax.Array, p_lab: jax.Array) -> jax.Array:
    r"""Rigid displacement ``omega cross p + omega cross (omega cross p) / 2``.

    The second-order expansion is exact for the Hessian at ``omega = 0``.
    """
    if p_lab.shape[-1] == 2:
        w = omega[0]
        return w * jnp.stack([-p_lab[1], p_lab[0]]) - 0.5 * w**2 * p_lab
    first = jnp.cross(omega, p_lab)
    return first + 0.5 * jnp.cross(omega, first)


def _pair_hessian_block(
    state: State,
    system: System,
    pair: jax.Array,
    directions: jax.Array,
    clumps: bool,
) -> jax.Array:
    """Differentiate one unordered pair's contribution to total energy."""
    i, j = pair
    dim = state.dim
    dof = dim + state.ang_vel.shape[-1] if clumps else dim
    pos = state.pos

    def energy(q: jax.Array) -> jax.Array:
        qi, qj = q[:dof], q[dof:]
        di, dj = qi[:dim], qj[:dim]
        if clumps:
            di = di + _rotation_perturbation(qi[dim:], state._pos_p_rot[i])
            dj = dj + _rotation_perturbation(qj[dim:], state._pos_p_rot[j])
        displaced = pos.at[i].add(di).at[j].add(dj)
        forward = system.force_model.energy(i, j, displaced, state, system)
        reverse = system.force_model.energy(j, i, displaced, state, system)
        return 0.5 * (
            jnp.where(directions[0], forward, 0.0)
            + jnp.where(directions[1], reverse, 0.0)
        )

    return jax.hessian(energy)(jnp.zeros(2 * dof, dtype=pos.dtype))


@partial(jax.jit, static_argnames=("clumps",))
def _pair_blocks(
    state: State,
    system: System,
    pairs: jax.Array,
    directions: jax.Array,
    *,
    clumps: bool,
) -> jax.Array:
    dof = state.dim + state.ang_vel.shape[-1] if clumps else state.dim
    if not pairs.shape[0]:
        return jnp.empty((0, 2 * dof, 2 * dof), dtype=state.pos_c.dtype)
    return jax.lax.map(
        lambda args: _pair_hessian_block(state, system, args[0], args[1], clumps),
        (pairs, directions),
        batch_size=min(pairs.shape[0], _HESSIAN_BATCH_SIZE),
    )


@partial(jax.jit, static_argnames=("clumps", "n_bodies"))
def _assemble_hessian(
    state: State,
    system: System,
    pairs: jax.Array,
    directions: jax.Array,
    *,
    clumps: bool,
    n_bodies: int,
) -> jax.Array:
    """Differentiate and scatter batches without retaining all pair blocks."""
    dof = state.dim + state.ang_vel.shape[-1] if clumps else state.dim
    matrix = jnp.zeros((n_bodies * dof, n_bodies * dof), dtype=state.pos_c.dtype)
    n_pairs = pairs.shape[0]
    if not n_pairs:
        return matrix

    def accumulate(
        matrix: jax.Array, args: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, None]:
        pair_batch, direction_batch = args
        blocks = jax.vmap(
            lambda pair, mask: _pair_hessian_block(state, system, pair, mask, clumps)
        )(pair_batch, direction_batch)
        bodies = state.clump_id[pair_batch] if clumps else pair_batch
        rows = (bodies[..., None] * dof + jnp.arange(dof)).reshape(-1, 2 * dof)
        return matrix.at[rows[:, :, None], rows[:, None, :]].add(blocks), None

    batch_size = min(n_pairs, _HESSIAN_BATCH_SIZE)
    full = n_pairs // batch_size * batch_size
    matrix, _ = jax.lax.scan(
        accumulate,
        matrix,
        (
            pairs[:full].reshape(-1, batch_size, 2),
            directions[:full].reshape(-1, batch_size, 2),
        ),
    )
    if full < n_pairs:
        matrix, _ = accumulate(matrix, (pairs[full:], directions[full:]))
    return matrix


def pair_non_bonded_hessian(
    state: State, system: System
) -> tuple[State, System, jax.Array, jax.Array]:
    r"""Return compact sphere-pair IDs and their potential-energy Hessian blocks.

    Parameters
    ----------
    state, system : State, System
        One 2D or 3D configuration and its configured collider and force model.

    Returns
    -------
    state : State
        Input state.
    system : System
        System carrying any refreshed collider cache. History is not advanced.
    pair_ids : jax.Array
        Sorted unique sphere pairs ``(i, j)`` with ``i < j``, shape ``(M, 2)``.
        Pairs obey the force model's search bounds and collider exclusions.
        There are no padding rows. Zero-force candidates are included.
    blocks : jax.Array
        Shape ``(M, 2*dim, 2*dim)``, in coordinate order ``(r_i, r_j)``.
        Each block differentiates ``(m_ij E_ij + m_ji E_ji) / 2``, with
        directed interaction masks ``m`` from the collider. For reciprocal
        interactions this is the Hessian of one pair potential. Blocks can
        be zero for candidates outside a law's actual interaction range.

    Notes
    -----
    Collection is a host operation; differentiation runs in compiled batches.
    An incomplete collider search raises ValueError before differentiation.
    """
    system, pairs, directions = _hessian_pairs(state, system)
    blocks = _pair_blocks(state, system, pairs, directions, clumps=False)
    return state, system, pairs, blocks


def non_bonded_hessian(state: State, system: System) -> tuple[State, System, jax.Array]:
    r"""Return state, refreshed system, and the dense sphere-coordinate Hessian.

    The matrix has shape ``(N*dim, N*dim)`` in flattened sphere-position
    order. Each unique pair contributes one block with the energy weighting
    described in :func:`pair_non_bonded_hessian`. Pair blocks are evaluated
    and scattered in bounded batches; output storage is quadratic in ``N*dim``.

    Search settings come from the configured collider and force model.
    Candidate collection runs on the host. Particle dynamics and contact
    history are unchanged. Incomplete searches raise ValueError.
    """
    system, pairs, directions = _hessian_pairs(state, system)
    matrix = _assemble_hessian(
        state, system, pairs, directions, clumps=False, n_bodies=state.N
    )
    return state, system, matrix


def bonded_hessian(state: State, system: System) -> tuple[State, System, jax.Array]:
    r"""Return state, system, and the Hessian of the bonded potential energy.

    The matrix has shape ``(N*dim, N*dim)`` in flattened sphere-position
    order. It is zero when ``system.bonded_force_model`` is None. Add it to
    :func:`non_bonded_hessian` to include both energy contributions.
    """
    n_total = state.N * state.dim
    bonded_model = system.bonded_force_model
    if bonded_model is None:
        return state, system, jnp.zeros((n_total, n_total), dtype=state.pos_c.dtype)

    def energy(pos: jax.Array) -> jax.Array:
        return bonded_model.compute_potential_energy(pos, state, system)

    hessian = jax.hessian(energy)(state.pos)
    return state, system, hessian.reshape(n_total, n_total)


def clump_non_bonded_hessian(
    state: State,
    system: System,
    *,
    rotation_scale: jax.Array | None = None,
) -> tuple[State, System, jax.Array]:
    r"""Return state, refreshed system, and the dense rigid-clump energy Hessian.

    Coordinates for each clump are ``(delta r_c, omega)``: translation and
    an infinitesimal rotation vector (one component in 2D, three in 3D).
    The matrix includes the second-order rotational displacement and has
    shape ``(G*dof, G*dof)``, with ``dof = dim + angular_dof`` and
    ``G = max(clump_id) + 1``. Rows and columns are indexed by clump ID.

    Parameters
    ----------
    state, system : State, System
        One configuration and its configured collider and force model.
    rotation_scale : jax.Array, optional
        Positive finite length scales, shape ``(G,)``. Rotation rows and
        columns for clump ``I`` are divided by ``rotation_scale[I]`` for
        coordinates ``(delta r_c, R*omega)``. Omit for angle coordinates.

    Notes
    -----
    Candidate collection runs on the host; pair differentiation and assembly
    run in bounded compiled batches. Output storage is quadratic in ``G*dof``.
    Contact history and particle dynamics are unchanged. Incomplete collider
    searches raise ValueError. See :func:`pair_non_bonded_hessian` for the
    directed energy weighting and inclusion of zero-force candidates.
    """
    system, pairs, directions = _hessian_pairs(state, system)
    n_clumps = int(jnp.max(state.clump_id, initial=-1)) + 1
    scales = None
    if rotation_scale is not None:
        scales = jnp.asarray(rotation_scale, dtype=state.pos_c.dtype)
        if scales.shape != (n_clumps,) or not bool(
            jnp.all(jnp.isfinite(scales) & (scales > 0))
        ):
            raise ValueError(
                "rotation_scale must contain one positive finite length per clump ID."
            )
    matrix = _assemble_hessian(
        state, system, pairs, directions, clumps=True, n_bodies=n_clumps
    )
    if scales is not None:
        dof = state.dim + state.ang_vel.shape[-1]
        scale = jnp.where(
            jnp.arange(dof)[None, :] < state.dim, 1.0, 1.0 / scales[:, None]
        )
        scale = scale.reshape(-1)
        matrix = matrix * scale[:, None] * scale[None, :]
    return state, system, matrix


def zero_mode_mask(
    eigenvalues: jax.Array,
    rel_gap: float = 1e4,
) -> jax.Array:
    r"""Boolean mask identifying numerically-zero eigenvalues via gap detection.

    Given a 1-D array of eigenvalues (in any order), we sort by
    :math:`|\lambda|` ascending and look for the largest relative gap
    :math:`|\lambda_{k+1}| / |\lambda_k|`. The function flags entries below
    the first gap that exceeds ``rel_gap`` as numerically zero. The function
    returns the mask aligned with the original eigenvalue ordering, so it
    can slice eigenvectors directly (e.g. ``evecs[:, ~mask]`` for
    finite modes).

    The result does not depend on the problem scale: a hessian with
    ``|λ_max| ~ 1`` and zero modes at ``~ 1e-16`` gives the same mask as
    one with ``|λ_max| ~ 1e6`` and zero modes at ``~ 1e-10``, because the
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
        zero modes at machine precision and real modes of order
        :math:`k \cdot \mathrm{overlap}` in a jammed spring packing
        typically sit 10-14 orders of magnitude apart. The threshold is
        therefore not sensitive.

    Returns
    -------
    jax.Array
        Boolean array of the same shape as ``eigenvalues``. ``True``
        where the eigenvalue is below the first large relative gap.
        If the function finds no gap larger than ``rel_gap`` (all
        eigenvalues are comparable in magnitude), it returns
        all-``False``.
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
    eps = np.finfo(sorted_abs.dtype).eps
    tiny = np.finfo(sorted_abs.dtype).tiny
    floor = jnp.maximum(eps * jnp.max(abs_e), tiny)
    s_safe = jnp.maximum(sorted_abs, floor)
    ratios = s_safe[1:] / s_safe[:-1]
    # Boundary position: index of the first entry whose ratio to the next
    # exceeds rel_gap. If none, there is no gap.
    gap_exceeds = ratios > rel_gap
    any_gap = jnp.any(gap_exceeds)
    first_gap = jnp.argmax(gap_exceeds.astype(int))
    n_zero = jnp.where(any_gap, first_gap + 1, 0)
    # Build a mask on the sorted order, then scatter back to original order.
    positions = jax.lax.iota(dtype=int, size=e.shape[0])
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
