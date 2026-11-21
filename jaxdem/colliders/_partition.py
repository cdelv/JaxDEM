# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Shared spatial-partitioning helpers for the grid-based colliders.

These helpers centralize grid-parameter computation and cell-hash dtype for
``cell_list.py``, ``multi_cell_list.py``, and ``neighbor_list.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

from jax.custom_batching import custom_vmap

from ..utils import linalg


if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@custom_vmap
def _shared_all(condition: Any) -> Any:
    return condition


@_shared_all.def_vmap
def _shared_all_mapped(axis_size: int, in_batched: Any, condition: Any) -> Any:
    return _shared_all(jnp.all(condition)), False


@custom_vmap
def _shared_max(value: Any) -> Any:
    return value


@_shared_max.def_vmap
def _shared_max_mapped(axis_size: int, in_batched: Any, value: Any) -> Any:
    return _shared_max(jnp.max(value, initial=0)), False


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _vec_sum(x: jax.Array, axis: int) -> jax.Array:
    """Sum a 2D/3D vector array along ``axis`` via a single variadic reduction.

    Using a variadic ``lax.reduce`` over the vector components allows XLA GPU
    to fuse per-pair scalar force magnitudes and vector projections into one
    reduction kernel without spilling ``(N, K)`` intermediates to VRAM.
    """
    axis = axis % x.ndim
    d = x.shape[-1]
    if d not in (2, 3) or axis == x.ndim - 1:
        return jnp.sum(x, axis=axis)
    zero = jnp.zeros((), dtype=x.dtype)
    if d == 3:
        r0, r1, r2 = jax.lax.reduce(
            (x[..., 0], x[..., 1], x[..., 2]),
            (zero, zero, zero),
            lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]),
            dimensions=(axis,),
        )
        return jnp.stack((r0, r1, r2), axis=-1)
    r0, r1 = jax.lax.reduce(
        (x[..., 0], x[..., 1]),
        (zero, zero),
        lambda a, b: (a[0] + b[0], a[1] + b[1]),
        dimensions=(axis,),
    )
    return jnp.stack((r0, r1), axis=-1)


@_vec_sum.defjvp
def _vec_sum_jvp(axis: int, primals: Any, tangents: Any) -> Any:
    (x,), (dx,) = primals, tangents
    return _vec_sum(x, axis), jnp.sum(dx, axis=axis)


def _compacted_stencil_decode(
    starts: jax.Array, pref: jax.Array, slots: jax.Array
) -> jax.Array:
    """Map compact candidate indices ``slots`` into sorted particle ranks via binary search.

    ``starts`` and ``pref`` have shape ``(M,)`` where ``pref = cumsum(lens)``
    over the ``M`` neighbor cells of one particle. Returns ``(K,)`` sorted
    member indices for each ``slot`` in ``slots`` (valid when ``slot < pref[-1]``).
    """
    m = int(starts.shape[0])
    lo = jnp.zeros_like(slots)
    step = 1 << ((m - 1).bit_length() - 1) if m > 1 else 0
    while step > 0:
        mid = jnp.minimum(lo + step, m - 1)
        lo = jnp.where(pref[mid - 1] <= slots, mid, lo)
        step >>= 1
    prev_p = jnp.where(lo > 0, pref[jnp.maximum(lo - 1, 0)], 0)
    return starts[lo] + (slots - prev_p)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class _SortedState:
    """Lightweight particle-order view for index-equivariant pair force laws.

    Permuting particle fields into spatial cell order makes inner-loop gathers
    coalesced without triggering ``State.__post_init__`` quaternion rotations.
    XLA dead-code-eliminates any field that the active force model does not read.
    """

    pos_c: jax.Array
    _pos_p_rot: jax.Array
    vel: jax.Array
    ang_vel: jax.Array
    force: jax.Array
    torque: jax.Array
    rad: jax.Array
    _rad: jax.Array
    mass: jax.Array
    inertia: jax.Array
    clump_id: jax.Array
    bond_id: jax.Array
    mat_id: jax.Array
    species_id: jax.Array
    fixed: jax.Array
    orig_id: jax.Array
    only_spheres: bool = jax.tree.static(default=False)

    @property
    def N(self) -> int:
        return self.pos_c.shape[-2]

    @property
    def dim(self) -> int:
        return self.pos_c.shape[-1]

    @property
    def pos(self) -> jax.Array:
        if self.only_spheres:
            return self.pos_c
        return self.pos_c + self._pos_p_rot

    @partial(jax.jit, inline=True)
    def velocity_at(self, index: jax.Array | int, offset: jax.Array) -> jax.Array:
        arm = offset if self.only_spheres else self._pos_p_rot[..., index, :] + offset
        return self.vel[..., index, :] + linalg.cross_3X3D_1X2D(
            self.ang_vel[..., index, :], arm
        )


def _permute_state(state: State, perm: jax.Array) -> _SortedState:
    """Return a sorted-order particle view of ``state`` indexed by ``perm``."""
    return _SortedState(
        pos_c=state.pos_c[perm],
        _pos_p_rot=state._pos_p_rot[perm],
        vel=state.vel[perm],
        ang_vel=state.ang_vel[perm],
        force=state.force,
        torque=state.torque,
        rad=state.rad[perm],
        _rad=state._rad[perm],
        mass=state.mass[perm],
        inertia=state.inertia[perm],
        clump_id=state.clump_id[perm],
        bond_id=state.bond_id[perm],
        mat_id=state.mat_id[perm],
        species_id=state.species_id[perm],
        fixed=state.fixed[perm],
        orig_id=perm,
        only_spheres=state.only_spheres,
    )


def _can_use_sorted_order(state: State, system: System) -> bool:
    """True when the force law is invariant under particle sorting."""
    return system.force_model.supports_sorted_order


def _index_limit() -> int:
    """Largest signed particle-index value supported by the active JAX precision."""
    return (1 << (63 if jax.config.jax_enable_x64 else 31)) - 1


def _effective_cell_size(collider: Any, reach: jax.Array) -> jax.Array:
    """Inflate cell size when ``reach`` exceeds the fixed stencil span."""
    stencil_reach = jnp.maximum(jnp.max(jnp.abs(collider.neighbor_mask)), 1)
    return jnp.maximum(collider.cell_size, reach / stencil_reach)


def shared_bounded_loop(body: Callable[..., Any], *, first_step: bool = False) -> Any:
    """Return a loop over ``body(index, carry, operands)``.

    Each explicit ``vmap`` axis shares the maximum bound. The body must mask
    exhausted work. A prototype call supplies carry variance, while its values
    remain unused. With the default option, zero bounds skip the body. Manual
    mesh axes retain independent bounds.

    ``first_step=True`` executes the first step before the loop. The body must
    leave its carry unchanged when the bound is zero. This option preserves
    the first step's values and manual-axis variance.
    """

    def run(steps: Any, initial: Any, operands: Any) -> Any:
        init_leaves, init_tree = jax.tree_util.tree_flatten(initial)
        op_leaves, op_tree = jax.tree_util.tree_flatten(operands)

        def scalar_loop(steps: Any, init_l: list[Any], op_l: list[Any]) -> Any:
            init_val = jax.tree_util.tree_unflatten(init_tree, init_l)
            op_val = jax.tree_util.tree_unflatten(op_tree, op_l)
            if first_step:
                init_val = body(jnp.asarray(0, steps.dtype), init_val, op_val)
                start = 1
            else:
                prototype = body(jnp.asarray(0, steps.dtype), init_val, op_val)
                init_val = jax.tree.map(
                    lambda start, result: start + jnp.zeros_like(result),
                    init_val,
                    prototype,
                )
                start = 0
            return jax.lax.fori_loop(
                jnp.asarray(start, steps.dtype),
                steps,
                lambda index, carry: body(index, carry, op_val),
                init_val,
            )

        @custom_vmap
        def dispatch(steps: Any, init_l: list[Any], op_l: list[Any]) -> Any:
            return scalar_loop(steps, init_l, op_l)

        @dispatch.def_vmap
        def mapped(
            axis_size: int,
            in_batched: Any,
            steps: Any,
            init_l: list[Any],
            op_l: list[Any],
        ) -> Any:
            steps_batched, carry_batched, op_batched = in_batched
            if not (steps_batched or any(carry_batched) or any(op_batched)):
                result = scalar_loop(steps, init_l, op_l)
                return result, jax.tree.map(lambda _: False, result)
            init_l = [
                v if b else jnp.broadcast_to(v, (axis_size,) + v.shape)
                for v, b in zip(init_l, carry_batched, strict=True)
            ]
            carry_axes = [0] * len(init_l)
            op_axes = [0 if b else None for b in op_batched]

            def batch_body(index: Any, carry_val: Any, arg_val: Any) -> Any:
                return jax.vmap(
                    lambda lc, la: body(
                        index,
                        jax.tree_util.tree_unflatten(init_tree, lc),
                        jax.tree_util.tree_unflatten(op_tree, la),
                    ),
                    in_axes=(carry_axes, op_axes),
                    axis_size=axis_size,
                )(
                    jax.tree_util.tree_leaves(carry_val),
                    jax.tree_util.tree_leaves(arg_val),
                )

            result = shared_bounded_loop(batch_body, first_step=first_step)(
                jnp.max(steps, initial=0) if steps_batched else steps,
                jax.tree_util.tree_unflatten(init_tree, init_l),
                jax.tree_util.tree_unflatten(op_tree, op_l),
            )
            return result, jax.tree.map(lambda _: True, result)

        return dispatch(steps, init_leaves, op_leaves)

    return run

# Bound query-axis vectorization during neighbor-list construction.
# Zero-capacity queries retain vmap: batched lax.map cannot reshape zero-width
# outputs, and those queries have no neighbor buffers to bound.
NEIGHBOR_QUERY_BATCH_SIZE = 65_536

# Bound force/energy traversal workspaces and the longest-lane loop per batch.
PAIR_TRAVERSAL_BATCH_SIZE = 16_384


#: Number of candidate particles each while-loop iteration of the pair
#: traversals visits. A vmapped ``lax.while_loop`` runs until the longest
#: lane finishes. Visiting several candidates reduces loop iterations at
#: the price of at most ``PAIR_UNROLL - 1`` extra masked pair evaluations
#: per lane.
PAIR_UNROLL = 4


@jax.jit
@partial(jax.named_call, name="cell_list._get_spatial_partition_grid")
def _get_spatial_partition_grid(
    pos: jax.Array,
    system: System,
    cell_size: jax.Array,
    iota: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute sorted cell hashes, permutation, and grid metadata without building stencils."""
    grid_dims, grid_strides, cell_size, hash_overflow = _grid_params(
        system.domain.box_size, cell_size, system.domain.periodic
    )

    hash_dtype = grid_strides.dtype
    pos = system.domain._shift(pos, system)
    if system.domain.periodic:
        inv_box = 1.0 / system.domain.box_size
        rel = (pos - system.domain.anchor) * inv_box
        frac = rel - jnp.floor(rel)
        frac = jnp.where(frac >= 1.0, 0.0, frac)
        p_cell_coords = jnp.floor(frac * grid_dims).astype(hash_dtype)
    else:
        inv_cell = 1.0 / cell_size
        p_cell_coords = jnp.floor((pos - system.domain.anchor) * inv_cell).astype(
            hash_dtype
        )

    p_cell_hash = jnp.dot(p_cell_coords, grid_strides)
    p_cell_hash, perm = jax.lax.sort([p_cell_hash, iota], num_keys=1)
    return (
        perm,
        p_cell_hash,
        p_cell_coords,
        grid_dims,
        grid_strides,
        cell_size,
        hash_overflow,
    )


@jax.jit(inline=True)
def _neighbor_hashes_from_coords(
    p_cell_coords: jax.Array,
    grid_dims: jax.Array,
    grid_strides: jax.Array,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    domain: Any,
) -> jax.Array:
    """Map per-particle cell coordinates to stencil neighbor cell hashes."""
    hash_dtype = grid_strides.dtype
    hash_sentinel = jnp.bitwise_not(jnp.asarray(0, hash_dtype))
    base_coords = p_cell_coords[:, None, :]
    offsets = neighbor_mask[None, :, :]
    nonnegative = offsets >= 0
    offset_magnitude = jnp.abs(offsets).astype(hash_dtype)

    if domain.periodic:
        dims = grid_dims[None, None, :]
        quotient = offset_magnitude // dims
        remainder = offset_magnitude % dims
        positive_carry = base_coords >= (dims - remainder)
        positive_coords = jnp.where(
            positive_carry,
            base_coords - (dims - remainder),
            base_coords + remainder,
        )
        negative_borrow = remainder > base_coords
        negative_coords = jnp.where(
            negative_borrow,
            dims - (remainder - base_coords),
            base_coords - remainder,
        )
        coords = jnp.where(nonnegative, positive_coords, negative_coords)
        image = jnp.where(
            nonnegative,
            quotient.astype(offsets.dtype) + positive_carry.astype(offsets.dtype),
            -quotient.astype(offsets.dtype) - negative_borrow.astype(offsets.dtype),
        )
        coords = domain.neighbor_cells(coords, image, grid_dims, cell_size)
        return jnp.dot(coords.astype(hash_dtype), grid_strides)

    positive_oob = offset_magnitude >= grid_dims - base_coords
    negative_oob = offset_magnitude > base_coords
    out_of_bounds = jnp.any(jnp.where(nonnegative, positive_oob, negative_oob), axis=-1)
    neighbor_cell_coords = jnp.where(
        nonnegative, base_coords + offset_magnitude, base_coords - offset_magnitude
    )
    neighbor_cell_hashes = jnp.dot(neighbor_cell_coords, grid_strides)
    return jnp.where(out_of_bounds, hash_sentinel, neighbor_cell_hashes)


@jax.jit
@partial(jax.named_call, name="cell_list._get_spatial_partition")
def _get_spatial_partition(
    pos: jax.Array,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    iota: jax.Array,
) -> tuple[jax.Array, ...]:
    """Compute the spatial hashing and partitioning of the cell list.

    Returns
    -------
    tuple[jax.Array, ...]
        ``(perm, p_cell_hash, neighbor_cell_hashes, hash_overflow)``.
    """
    (
        perm,
        p_cell_hash,
        p_cell_coords,
        grid_dims,
        grid_strides,
        eff_cell_size,
        hash_overflow,
    ) = _get_spatial_partition_grid(pos, system, cell_size, iota)
    neighbor_cell_hashes = _neighbor_hashes_from_coords(
        p_cell_coords,
        grid_dims,
        grid_strides,
        eff_cell_size,
        neighbor_mask,
        system.domain,
    )
    return (
        perm,
        p_cell_hash,
        neighbor_cell_hashes,
        hash_overflow,
    )


def _lees_edwards_compact_stencil(
    b_coords: jax.Array,
    dims_i: jax.Array,
    strides_i: jax.Array,
    eff_cell_size: jax.Array,
    offsets_mask: jax.Array,
    cell_offsets: jax.Array,
    domain: Any,
    r: int,
) -> tuple[jax.Array, jax.Array]:
    """Build exact duplicate-free Lees-Edwards stencil cell intervals ``(starts, ends)``.

    Requires ``dims_i[alpha] >= 2*r + 2`` and ``dims_i[d] >= 2*r + 1`` for ``d != alpha``.
    Produces ``(2r+1)^D + r*(2r+1)^(D-2)`` cells (30 in 3D, 10 in 2D for ``r=1``)
    instead of ``2*(2r+1)^D`` with ``O(M^2)`` deduplication.
    """
    dim = int(b_coords.shape[-1])
    a, b = int(domain.alpha), int(domain.beta)
    dims_a = dims_i[a]
    dims_b = dims_i[b]

    # Scalar shear shifts for the two possible beta-boundary crossings (img_b = +1, -1).
    shift_mag = domain.gamma * domain.box_size[b] / eff_cell_size[a]
    s_pos = -shift_mag
    s_neg = shift_mag
    lo_pos = jnp.floor(s_pos).astype(int)
    hi_pos = jnp.ceil(s_pos).astype(int)
    lo_neg = jnp.floor(s_neg).astype(int)
    hi_neg = jnp.ceil(s_neg).astype(int)
    need_hi_pos = hi_pos != lo_pos
    need_hi_neg = hi_neg != lo_neg
    lo_pos_m = jnp.mod(lo_pos, dims_a)
    lo_neg_m = jnp.mod(lo_neg, dims_a)

    def _wrap_a(v: jax.Array) -> jax.Array:
        t = jnp.where(v >= dims_a, v - dims_a, v)
        return jnp.where(t < 0, t + dims_a, jnp.where(t >= dims_a, t - dims_a, t))

    # 1. Base (2r+1)^D cells with floor(alpha_shift) applied on beta-crossing entries.
    raw_b = b_coords[:, None, b] + offsets_mask[None, :, b]
    img_b = jnp.where(raw_b < 0, -1, jnp.where(raw_b >= dims_b, 1, 0))
    wb = raw_b - img_b * dims_b
    lo_m = jnp.where(img_b == 0, 0, jnp.where(img_b > 0, lo_pos_m, lo_neg_m))
    wa = _wrap_a(b_coords[:, None, a] + offsets_mask[None, :, a] + lo_m)
    h_base = wa * strides_i[a] + wb * strides_i[b]
    if dim == 3:
        o = 3 - a - b
        dims_o = dims_i[o]
        raw_o = b_coords[:, None, o] + offsets_mask[None, :, o]
        wo = jnp.where(
            raw_o < 0, raw_o + dims_o, jnp.where(raw_o >= dims_o, raw_o - dims_o, raw_o)
        )
        h_base = h_base + wo * strides_i[o]
    starts_base = cell_offsets[h_base]
    ends_base = cell_offsets[h_base + 1]

    # 2. Extra hi = lo + 1 cells at off_a = +(r + 1) along the crossing beta boundary.
    off_k = jnp.arange(1, r + 1, dtype=int)
    cb = b_coords[:, b : b + 1]
    valid_neg = (cb - off_k[None, :] < 0) & need_hi_neg
    valid_pos = (cb + off_k[None, :] >= dims_b) & need_hi_pos
    valid_bk = valid_neg | valid_pos
    wb_bk = jnp.where(valid_neg, cb - off_k[None, :] + dims_b, cb + off_k[None, :] - dims_b)
    wb_bk = jnp.where(valid_bk, wb_bk, 0)
    lo_bk = jnp.where(valid_neg, lo_neg_m, lo_pos_m)
    wa_bk = _wrap_a(b_coords[:, a : a + 1] + (r + 1) + lo_bk)
    h_ab = wa_bk * strides_i[a] + wb_bk * strides_i[b]

    if dim == 2:
        h_ext = h_ab
        valid_ext = valid_bk
    else:
        o = 3 - a - b
        dims_o = dims_i[o]
        off_o = jnp.arange(-r, r + 1, dtype=int)
        raw_o = b_coords[:, o : o + 1] + off_o[None, :]
        wo = jnp.where(
            raw_o < 0, raw_o + dims_o, jnp.where(raw_o >= dims_o, raw_o - dims_o, raw_o)
        )
        h_ext = (h_ab[:, :, None] + wo[:, None, :] * strides_i[o]).reshape(
            b_coords.shape[0], -1
        )
        valid_ext = jnp.broadcast_to(
            valid_bk[:, :, None], (b_coords.shape[0], r, 2 * r + 1)
        ).reshape(b_coords.shape[0], -1)

    starts_ext = jnp.where(valid_ext, cell_offsets[h_ext], 0)
    ends_ext = jnp.where(valid_ext, cell_offsets[h_ext + 1], 0)
    return (
        jnp.concatenate((starts_base, starts_ext), axis=1),
        jnp.concatenate((ends_base, ends_ext), axis=1),
    )


def _dedup_periodic_stencil(
    stencil: jax.Array,
    grid_dims: jax.Array,
    neighbor_mask: jax.Array,
    domain: Any,
) -> jax.Array:
    """Deduplicate periodic stencil hashes only when cells can wrap onto themselves."""
    if not domain.periodic:
        return stencil
    if hasattr(domain, "gamma"):
        return jax.vmap(_dedup_stencil_hashes)(stencil)
    min_unique = (2 * jnp.max(jnp.abs(neighbor_mask)) + 1).astype(grid_dims.dtype)
    all_distinct = _shared_all(jnp.all(grid_dims >= min_unique))
    return jax.lax.cond(
        all_distinct,
        lambda s: s,
        lambda s: jax.vmap(_dedup_stencil_hashes)(s),
        stencil,
    )


@jax.jit(inline=True)
@partial(jax.named_call, name="cell_list._dedup_stencil_hashes")
def _dedup_stencil_hashes(stencil_hashes: jax.Array) -> jax.Array:
    """Deduplicate one particle's stencil hashes using the unsigned sentinel."""
    indices = jnp.arange(stencil_hashes.size)
    is_duplicate = jnp.zeros(stencil_hashes.shape, dtype=bool)
    # Small static slices fuse with their reductions instead of materializing
    # an (N, stencil, stencil) temporary for a large shear-periodic stencil.
    for start in range(0, stencil_hashes.size, 16):
        end = min(start + 16, stencil_hashes.size)
        mask = (indices[start:end, None] < indices[None, :]) & (
            stencil_hashes[start:end, None] == stencil_hashes[None, :]
        )
        is_duplicate |= jnp.any(mask, axis=0)
    sentinel = jnp.bitwise_not(jnp.asarray(0, stencil_hashes.dtype))
    return jnp.where(is_duplicate, sentinel, stencil_hashes)


def _traverse_neighbor_row(
    stencil: jax.Array,
    cell_starts: jax.Array,
    sorted_hashes: jax.Array,
    n_db: int,
    local_capacity: int,
    candidate_valid: Callable[[jax.Array], jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Build one neighbor row directly while walking its stencil in order."""
    j_arr = jax.lax.iota(dtype=int, size=PAIR_UNROLL)
    last_db = max(1, n_db) - 1
    init_carry = (
        jnp.array(0, dtype=int),
        cell_starts[0],
        jnp.array(0, dtype=int),
        jnp.full((local_capacity,), -1, dtype=int),
        jnp.array(False),
    )

    def cond_fun(
        val: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> bool:
        stencil_idx, _, c, _, _ = val
        return cast(bool, (stencil_idx < stencil.shape[0]) & (c < local_capacity + 1))

    def body_fun(
        val: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        stencil_idx, k, c, nl, overflow = val
        target_cell_hash = stencil[stencil_idx]
        actual_k_arr = k + j_arr
        safe_k_arr = jnp.minimum(actual_k_arr, last_db)
        in_cell_arr = (actual_k_arr < n_db) & (
            sorted_hashes[safe_k_arr] == target_cell_hash
        )
        valid_arr = jax.vmap(candidate_valid)(safe_k_arr).astype(bool) & in_cell_arr

        valid_counts = valid_arr.astype(c.dtype)
        write_idx_arr = jnp.where(
            valid_arr, c + jnp.cumsum(valid_counts) - valid_counts, local_capacity
        )
        nl = nl.at[write_idx_arr].set(safe_k_arr, mode="drop")
        c = c + jnp.sum(valid_counts)
        overflow = overflow | (c > local_capacity)

        next_k = k + PAIR_UNROLL
        remains_in_cell = (next_k < n_db) & (
            sorted_hashes[jnp.minimum(next_k, last_db)] == target_cell_hash
        )
        next_stencil_idx = jnp.where(remains_in_cell, stencil_idx, stencil_idx + 1)
        safe_stencil_idx = jnp.minimum(next_stencil_idx, stencil.shape[0] - 1)
        next_k = jnp.where(remains_in_cell, next_k, cell_starts[safe_stencil_idx])
        return next_stencil_idx, next_k, c, nl, overflow

    _, _, _, neighbor_row, row_overflow = jax.lax.while_loop(
        cond_fun, body_fun, init_carry
    )
    return neighbor_row, row_overflow


def _make_direct_row_body(
    sorted_hashes: jax.Array,
    n_db: int,
    local_capacity: int,
    candidate_valid: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Build a reusable row-traversal closure over ``_traverse_neighbor_row``."""

    @jax.jit(inline=True)
    def row_body(
        stencil: jax.Array, cell_starts: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return _traverse_neighbor_row(
            stencil, cell_starts, sorted_hashes, n_db, local_capacity, candidate_valid
        )

    return row_body


@jax.jit(inline=True)
@partial(jax.named_call, name="multi_cell_list._loose_cell_aabbs")
def _loose_cell_aabbs(
    member_min: jax.Array,
    member_max: jax.Array,
    sorted_hash: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Per-loose-cell axis-aligned bounding box, broadcast to each member.

    The hash-sorted particles of a loose cell form a contiguous run. The
    function therefore computes the cell's expandable AABB (the union of
    every member's box ``[member_min, member_max]``) as a segmented min/max
    reduction over those runs. It returns the result as
    ``(cell_center, cell_half_extent)`` indexed by sorted particle.
    """
    N = member_min.shape[0]
    # Dense, contiguous segment id per sorted particle (0-based cell rank).
    seg_start = jnp.concatenate(
        [jnp.array([True]), sorted_hash[1:] != sorted_hash[:-1]]
    )
    seg_id = jnp.cumsum(seg_start) - 1

    cell_min = jax.ops.segment_min(member_min, seg_id, num_segments=N)[seg_id]
    cell_max = jax.ops.segment_max(member_max, seg_id, num_segments=N)[seg_id]

    cell_center = 0.5 * (cell_min + cell_max)
    cell_half = 0.5 * (cell_max - cell_min)
    return cell_center, cell_half


@jax.jit(inline=True)
def _sorted_cell_run_ends(hashes: jax.Array) -> jax.Array:
    """Return each sorted member's exclusive cell boundary."""
    n = hashes.shape[0]
    if n == 0:
        return jnp.zeros_like(hashes, dtype=int)
    last = jnp.concatenate((hashes[1:] != hashes[:-1], jnp.ones((1,), bool)))
    ends = jnp.where(last, jnp.arange(1, n + 1), n)
    return jax.lax.cummin(ends, axis=0, reverse=True)


@jax.jit(inline=True, static_argnames=("quasistatic",))
def _force_only_pair_fn(
    acc: jax.Array,
    i: jax.Array,
    j: jax.Array,
    pos: jax.Array,
    state: "State",
    valid: jax.Array,
    system: "System",
    quasistatic: bool = False,
) -> jax.Array:
    history = system.force_model.init_history(jnp.shape(j), pos.shape[-1])
    f, _, _ = system.force_model.force(
        i,
        j,
        pos,
        state,
        system,
        history,
        advance_history=False,
        quasistatic=quasistatic,
    )
    f = jnp.where((valid > 0)[..., None], f, 0.0)
    return acc + f


@jax.jit(inline=True, static_argnames=("quasistatic",))
def _force_pair_fn(
    acc: tuple[jax.Array, jax.Array],
    i: jax.Array,
    j: jax.Array,
    pos: jax.Array,
    state: "State",
    valid: jax.Array,
    system: "System",
    quasistatic: bool = False,
) -> tuple[jax.Array, jax.Array]:
    history = system.force_model.init_history(jnp.shape(j), pos.shape[-1])
    f, t, _ = system.force_model.force(
        i,
        j,
        pos,
        state,
        system,
        history,
        advance_history=False,
        quasistatic=quasistatic,
    )
    f = jnp.where((valid > 0)[..., None], f, 0.0)
    t = jnp.where((valid > 0)[..., None], t, 0.0)
    return acc[0] + f, acc[1] + t


@jax.jit(inline=True)
def _energy_pair_fn(
    acc: jax.Array,
    i: jax.Array,
    j: jax.Array,
    pos: jax.Array,
    state: "State",
    valid: jax.Array,
    system: "System",
) -> jax.Array:
    history = system.force_model.init_history(jnp.shape(j), pos.shape[-1])
    e = system.force_model.energy(i, j, pos, state, system, history)
    e = jnp.where(valid > 0, e, 0.0)
    return acc + 0.5 * e


@jax.jit(inline=True, static_argnames=("periodic",))
def _grid_params(
    box_size: jax.Array, cell_size: jax.Array, periodic: bool
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute grid dimensions, hashing strides, and the effective cell size.

    Parameters
    ----------
    box_size : jax.Array
        Physical extents of the domain per axis.
    cell_size : jax.Array
        Requested (scalar) grid cell size.
    periodic : bool
        Whether the domain is periodic. Periodic grids are floored so an
        integer number of cells tiles the box, and the cell size is inflated
        to stay commensurate. Non-periodic grids are ceiled.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        ``(grid_dims, grid_strides, cell_size, hash_overflow)`` where
        ``cell_size`` is the effective per-axis-uniform cell size of the grid
        and ``hash_overflow`` is a scalar boolean. ``hash_overflow`` is True
        when the total number of grid cells exceeds the range of the
        cell-hash dtype.
    """
    # Cell hashes are unsigned so the full native JAX integer range is
    # available. Particle indices remain signed because -1 is their sentinel.
    hash_dtype = jnp.uint64 if jax.config.jax_enable_x64 else jnp.uint32
    hash_bits = 64 if jax.config.jax_enable_x64 else 32
    hash_limit = 1 << hash_bits
    max_hash = jnp.asarray(hash_limit - 1, dtype=hash_dtype)
    dims_float = (
        jnp.floor(box_size / cell_size) if periodic else jnp.ceil(box_size / cell_size)
    )
    dims_float = jnp.maximum(dims_float, 1)
    dims_valid = jnp.all(jnp.isfinite(dims_float)) & jnp.all(
        dims_float < jnp.asarray(float(hash_limit), dtype=box_size.dtype)
    )
    safe_dims_float = jnp.where(dims_valid, dims_float, 1)
    grid_dims = safe_dims_float.astype(hash_dtype)
    if periodic:
        cell_size = box_size / grid_dims

    # Derive the scalar from grid_dims so it carries the same manual-axis
    # variance under shard_map.
    product = jnp.ones_like(grid_dims[0])
    hash_overflow = ~dims_valid
    strides = []
    for axis in range(grid_dims.shape[0]):
        dim = grid_dims[axis]
        strides.append(product)
        can_multiply = dim <= max_hash // product
        # Select a safe factor before multiplication so overflowing products are never evaluated.
        safe_dim = jnp.where(can_multiply, dim, jnp.asarray(1, dim.dtype))
        product = product * safe_dim
        hash_overflow = hash_overflow | ~can_multiply
    grid_strides = jnp.stack(strides, axis=0)

    return grid_dims, grid_strides, cell_size, hash_overflow


def _saturated_prefix_sum(counts: jax.Array, limit: int) -> jax.Array:
    """Return nonnegative prefix sums clipped to a signed index limit."""
    if counts.size == 0:
        return counts
    unsigned_dtype = jnp.uint64 if counts.dtype == jnp.int64 else jnp.uint32
    clipped = jnp.minimum(counts, limit).astype(unsigned_dtype)
    totals = jnp.cumsum(clipped, dtype=unsigned_dtype)
    # The first crossing cannot wrap: both operands are at most signed max.
    # The maximum retains that crossing after later unsigned sums wrap.
    totals = jax.lax.cummax(totals, axis=0)
    return jnp.minimum(totals, jnp.asarray(limit, unsigned_dtype)).astype(counts.dtype)


def _capacity_offsets(counts: jax.Array, capacity: int) -> tuple[jax.Array, jax.Array]:
    """Saturate at capacity + 1 so even severe overflows cannot wrap integers."""
    totals = _saturated_prefix_sum(counts, capacity + 1)
    overflow = totals[-1] > capacity
    return (
        jnp.concatenate(
            (jnp.zeros_like(counts, shape=(1,)), jnp.minimum(totals, capacity))
        ),
        overflow,
    )
