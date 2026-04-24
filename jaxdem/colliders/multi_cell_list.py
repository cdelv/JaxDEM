# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Multi-Cell List collider implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Union, cast

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from ..utils.linalg import norm2
from . import Collider, valid_interaction_mask
from .cell_list import _energy_kernel, _force_kernel, _force_reduce

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@partial(jax.jit, static_argnames=("periodic", "max_hashes"))
@partial(jax.named_call, name="multi_cell_list._get_aabb_hashes")
def _get_aabb_hashes(
    center: jax.Array,
    aabb: jax.Array,
    cell_size: jax.Array,
    grid_dims: jax.Array,
    grid_strides: jax.Array,
    domain_anchor: jax.Array,
    periodic: bool,
    max_hashes: int,
) -> jax.Array:
    """Computes the hashes of all grid cells overlapped by an AABB.

    Parameters
    ----------
    center : jax.Array
        AABB center position.
    aabb : jax.Array
        AABB extent (full width).
    cell_size : jax.Array
        Grid cell size.
    grid_dims : jax.Array
        Number of cells per axis.
    grid_strides : jax.Array
        Strides for hashing ND coordinates to 1D.
    domain_anchor : jax.Array
        Domain minimum corner.
    periodic : bool
        Whether the domain is periodic.
    max_hashes : int
        Maximum number of hashes to return (padding).

    Returns
    -------
    jax.Array
        padded_hashes
    """
    # Compute absolute spatial bounds
    xmin = center - aabb / 2.0
    xmax = center + aabb / 2.0

    # Get RAW (unwrapped) cell coordinates for the bounds relative to anchor
    min_coords_raw = jnp.floor((xmin - domain_anchor) / cell_size).astype(int)
    max_coords_raw = jnp.floor((xmax - domain_anchor) / cell_size).astype(int)

    # Calculate how many cells the AABB spans in each dimension
    num_cells_dim = max_coords_raw - min_coords_raw + 1
    total_cells = jnp.prod(num_cells_dim)

    # Create static 1D index array
    idx = jax.lax.iota(size=max_hashes, dtype=int)

    # Compute dynamic strides for the LOCAL AABB grid
    local_strides = jnp.concatenate(
        [jnp.array([1], dtype=int), jnp.cumprod(num_cells_dim[:-1])]
    )

    # Unravel 1D indices to ND local offsets via broadcasting
    cell_coords = (
        min_coords_raw[None, :]
        + (idx[:, None] // local_strides[None, :]) % num_cells_dim[None, :]
    )

    # Apply boundary conditions
    if periodic:
        cell_coords = cell_coords % grid_dims[None, :]
        valid_cells = True
    else:
        valid_cells = (cell_coords >= 0).all(axis=-1) * (
            cell_coords < grid_dims[None, :]
        ).all(axis=-1)
        # Prevent negative coordinate aliasing by clamping to grid boundaries
        cell_coords = jnp.clip(cell_coords, 0, grid_dims[None, :] - 1)

    # Spatial Hashingcell_coords
    hashes = jnp.dot(cell_coords, grid_strides)
    hashes = jnp.where((idx < total_cells) * valid_cells, hashes, -1)

    return hashes


_get_aabb_hashes_vmap = jax.vmap(
    _get_aabb_hashes, in_axes=(0, 0, None, None, None, None, None, None)
)


@partial(jax.jit, static_argnames=("max_hashes",))
@partial(jax.named_call, name="multi_cell_list._get_multi_cell_partition")
def _get_multi_cell_partition(
    pos: jax.Array,
    rad_or_search_rad: jax.Array,
    system: System,
    cell_size: jax.Array,
    max_hashes: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Hashes particles into multiple grid cells and sorts them by hash.

    Parameters
    ----------
    pos : jax.Array
        Particle positions.
    rad_or_search_rad : jax.Array
        Radius or search radius for AABB computation.
    system : System
        The system configuration.
    cell_size : jax.Array
        Grid cell size.
    max_hashes : int
        Maximum number of cells a particle can occupy.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        (sorted_hashes, perm, original_hashes)
    """
    N, dim = pos.shape
    if system.domain.periodic:
        grid_dims = jnp.floor(system.domain.box_size / cell_size).astype(int)
    else:
        grid_dims = jnp.ceil(system.domain.box_size / cell_size).astype(int)

    grid_strides = jnp.concatenate(
        [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
    )

    aabb = 2.0 * rad_or_search_rad
    if aabb.ndim == 1:
        aabb = jnp.repeat(aabb[:, None], dim, axis=1)

    hashes = _get_aabb_hashes_vmap(
        pos,
        aabb,
        cell_size,
        grid_dims,
        grid_strides,
        system.domain.anchor,
        system.domain.periodic,
        max_hashes,
    )

    p_ids = jnp.broadcast_to(jax.lax.iota(size=N, dtype=int)[:, None], (N, max_hashes))
    flat_hashes = hashes.ravel()
    flat_ids = p_ids.ravel()

    sorted_hashes, sorted_ids = jax.lax.sort([flat_hashes, flat_ids], num_keys=1)
    perm = sorted_ids

    return sorted_hashes, perm, hashes


@partial(jax.jit, static_argnames=("weighted",))
@partial(jax.named_call, name="multi_cell_list._compute_canonical_hash")
def _compute_canonical_hash(
    pos_i: jax.Array,
    pos_j: jax.Array,
    rad_i: jax.Array,
    rad_j: jax.Array,
    cell_size: jax.Array,
    system: System,
    weighted: bool = True,
) -> jax.Array:
    """Computes a unique (canonical) cell hash for an interaction pair."""
    dr = system.domain.displacement(pos_i, pos_j, system)
    pos_j_unwrapped = pos_i - dr

    xmin_int = jnp.maximum(pos_i - rad_i, pos_j_unwrapped - rad_j)
    xmax_int = jnp.minimum(pos_i + rad_i, pos_j_unwrapped + rad_j)

    M_shifted = (xmin_int + xmax_int) / 2.0 - system.domain.anchor

    if system.domain.periodic:
        grid_dims = jnp.floor(system.domain.box_size / cell_size).astype(int)
        M_cell = jnp.floor(M_shifted / cell_size).astype(int) % grid_dims
    else:
        grid_dims = jnp.ceil(system.domain.box_size / cell_size).astype(int)
        M_cell = jnp.floor(M_shifted / cell_size).astype(int)
        M_cell = jnp.clip(M_cell, 0, grid_dims - 1)

    grid_strides = jnp.concatenate(
        [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
    )
    return jnp.dot(M_cell, grid_strides)


@partial(jax.named_call, name="multi_cell_list._compute_interaction_multi")
def _compute_interaction_multi(
    state: State,
    system: System,
    traverse_fn: Callable[..., Any],
    interaction_fn: Callable[..., Any],
    reduce_fn: Callable[..., Any] | None = None,
) -> tuple[State, Any]:
    collider = cast(Union["MultiCellList", "DynamicMultiCellList"], system.collider)

    pos_p = state.q.rotate(state.q, state.pos_p)
    pos = state.pos_c + pos_p

    sorted_hashes, perm, hashes = _get_multi_cell_partition(
        pos,
        state.rad,
        system,
        collider.cell_size,
        collider.max_hashes,
    )

    iota = jax.lax.iota(dtype=int, size=state.N)

    def per_hash(target_hash: jax.Array, idx: jax.Array) -> Any:
        return traverse_fn(
            target_hash,
            state,
            system,
            pos,
            sorted_hashes,
            perm,
            idx,
            collider.cell_size,
            interaction_fn,
        )

    flat_results = jax.vmap(per_hash)(sorted_hashes, perm)

    # Use scatter_add instead of segment_sum or sort
    summed = jax.tree.map(lambda x: jnp.zeros((state.N, *x.shape[1:])), flat_results)
    summed = jax.tree.map(lambda s, f: s.at[perm].add(f), summed, flat_results)

    if reduce_fn is not None:
        summed = reduce_fn(summed, pos_p)

    return state, summed


def _static_traverse_multi_cell(
    target_hash: jax.Array,
    state: State,
    system: System,
    pos: jax.Array,
    sorted_hashes: jax.Array,
    perm: jax.Array,
    idx: jax.Array,
    cell_size: jax.Array,
    interaction_fn: Callable[..., Any],
    max_occupancy: int,
    weighted: bool = True,
) -> Any:
    """Fixed-occupancy (unrolled) traversal of a single cell."""
    start_idx = jnp.searchsorted(sorted_hashes, target_hash, side="left")
    flat_indices = start_idx + jax.lax.iota(dtype=int, size=max_occupancy)

    total_elements = sorted_hashes.shape[0]
    safe_flat = jnp.minimum(flat_indices, total_elements - 1)

    j_indices = perm[safe_flat]

    valid_hash = (
        (target_hash != -1)
        & (flat_indices < total_elements)
        & (sorted_hashes[safe_flat] == target_hash)
    )

    canonical_hash = jax.vmap(
        lambda j: _compute_canonical_hash(
            pos[idx],
            pos[j],
            state.rad[idx],
            state.rad[j],
            cell_size,
            system,
            weighted=weighted,
        )
    )(j_indices)

    valid_canonical = target_hash == canonical_hash

    valid = (
        valid_hash
        & valid_canonical
        & valid_interaction_mask(
            state.clump_id[j_indices],
            state.clump_id[idx],
            state.bond_id[j_indices],
            state.bond_id[idx],
            system.interact_same_bond_id,
        )
        & (idx != j_indices)
    )

    res = interaction_fn(idx, j_indices, valid, pos, state, system)
    return jax.tree.map(lambda x: jnp.sum(x, axis=0), res)


def _dynamic_traverse_multi_cell(
    target_hash: jax.Array,
    state: State,
    system: System,
    pos: jax.Array,
    sorted_hashes: jax.Array,
    perm: jax.Array,
    idx: jax.Array,
    cell_size: jax.Array,
    interaction_fn: Callable[..., Any],
    init_val: Any,
    weighted: bool = True,
) -> Any:
    """Dynamic (while-loop) traversal of a single cell."""
    start_idx = jnp.searchsorted(sorted_hashes, target_hash, side="left")
    total_elements = sorted_hashes.shape[0]

    def cond_fun(val: tuple[jax.Array, Any]) -> bool:
        flat_idx, _ = val
        in_bounds = flat_idx < total_elements
        safe_idx = jnp.minimum(flat_idx, total_elements - 1)
        matches_hash = sorted_hashes[safe_idx] == target_hash
        return cast(bool, in_bounds & matches_hash & (target_hash != -1))

    def body_fun(val: tuple[jax.Array, Any]) -> tuple[jax.Array, Any]:
        flat_idx, acc = val
        j = perm[flat_idx]

        canonical_hash = _compute_canonical_hash(
            pos[idx],
            pos[j],
            state.rad[idx],
            state.rad[j],
            cell_size,
            system,
            weighted=weighted,
        )

        valid = (
            (target_hash == canonical_hash)
            & valid_interaction_mask(
                state.clump_id[j],
                state.clump_id[idx],
                state.bond_id[j],
                state.bond_id[idx],
                system.interact_same_bond_id,
            )
            & (idx != j)
        )

        res = interaction_fn(idx, j, valid, pos, state, system)
        new_acc = jax.tree.map(lambda a, b: a + b, acc, res)
        return flat_idx + 1, new_acc

    _, final_acc = jax.lax.while_loop(cond_fun, body_fun, (start_idx, init_val))
    return final_acc


@partial(jax.named_call, name="multi_cell_list._compute_neighbor_list_common_multi")
def _compute_neighbor_list_common_multi(
    state: State,
    system: System,
    traverse_fn: Callable[..., Any],
    max_neighbors: int,
    search_rad: jax.Array,
) -> tuple[State, System, jax.Array, jax.Array]:
    """Common logic for neighbor list computation in Multi-Cell List."""
    if max_neighbors == 0:
        empty = jnp.empty((state.N, 0), dtype=int)
        return state, system, empty, jnp.asarray(False)

    collider = cast(Union["MultiCellList", "DynamicMultiCellList"], system.collider)

    pos = state.pos

    sorted_hashes, perm, hashes = _get_multi_cell_partition(
        pos,
        search_rad,
        system,
        collider.cell_size,
        collider.max_hashes,
    )

    flat_state = jax.tree.map(lambda x: x[perm], state)
    flat_pos = pos[perm]
    total_flat = sorted_hashes.shape[0]
    iota_flat = jax.lax.iota(int, total_flat)

    def per_hash(
        target_hash: jax.Array, flat_idx_i: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return traverse_fn(
            flat_idx_i,
            flat_pos[flat_idx_i],
            target_hash,
            flat_state,
            system,
            flat_pos,
            sorted_hashes,
            iota_flat,
            collider.cell_size,
        )

    # We cannot use jax.ops.segment_sum for neighbor lists because they are arrays of indices.
    # Instead, we should just vmap over N, but that defeats the linear access optimization.
    # Wait, neighbor list is only built every few steps, so we can afford non-linear memory access for it!
    # Let's revert neighbor list to the original per_particle nested vmap!
    iota = jax.lax.iota(int, state.N)

    def per_particle(
        idx: jax.Array, pos_i: jax.Array, p_hashes: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return traverse_fn(
            idx,
            pos_i,
            p_hashes,
            state,
            system,
            pos,
            sorted_hashes,
            perm,
            collider.cell_size,
        )

    neighbor_list, overflows = jax.vmap(per_particle)(iota, pos, hashes)
    return state, system, neighbor_list, jnp.any(overflows)


@Collider.register("MultiCellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MultiCellList(Collider):
    """Implicit multi-cell list (spatial hashing) collider using static fixed-occupancy traversal.

    Allows large particles to span multiple cells, drastically reducing max_occupancy padding.
    Each particle can occupy up to ``max_hashes`` grid cells. Pair interactions are
    de-duplicated using a canonical cell hash based on the interaction center.
    """

    cell_size: jax.Array
    max_hashes: int = jax.tree.static()
    max_cells_per_axis: int = jax.tree.static()
    max_occupancy: int = jax.tree.static()

    @classmethod
    def Create(
        cls,
        state: State,
        cell_size: ArrayLike | None = None,
        box_size: ArrayLike | None = None,
        max_occupancy: int | None = None,
        max_hashes: int | None = None,
    ) -> Self:
        min_rad = jnp.min(state.rad)
        max_rad = jnp.max(state.rad)
        alpha = max_rad / min_rad

        if cell_size is None:
            # Default to match StaticCellList for consistency and robustness
            cell_size = 2.0 * max_rad if alpha < 2.5 else 0.5 * max_rad
        cell_size_val = jnp.asarray(cell_size, dtype=float)

        if box_size is not None:
            box_size = jnp.asarray(box_size, dtype=float)
            grid_dims = jnp.floor(box_size / cell_size_val).astype(int)
            grid_dims = jnp.maximum(grid_dims, 1)
            cell_size_val = jnp.min(box_size / grid_dims)

        max_rad = jnp.max(state.rad)
        S = jnp.ceil(2 * max_rad / cell_size_val).astype(int)
        max_cells_per_axis = int(S + 1)

        if max_hashes is None:
            max_hashes = max_cells_per_axis**state.dim

        if max_occupancy is None:
            # Estimate occupancy based on volume fraction phi=1.0
            cell_vol = cell_size_val**state.dim
            avg_particle_vol = jnp.mean(state.volume)
            expected_occ = cell_vol / jnp.maximum(avg_particle_vol, 1e-12)
            max_occupancy = int(
                jnp.maximum(32, jnp.ceil(expected_occ + 5.0 * jnp.sqrt(expected_occ)))
            )

        return cls(
            cell_size=cell_size_val,
            max_hashes=int(max_hashes),
            max_cells_per_axis=max_cells_per_axis,
            max_occupancy=max_occupancy,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiCellList.compute_force")
    def compute_force(state: State, system: System) -> tuple[State, System]:
        collider = cast(MultiCellList, system.collider)
        traverse = partial(
            _static_traverse_multi_cell,
            max_occupancy=collider.max_occupancy,
            weighted=True,
        )
        state, (state.force, state.torque) = _compute_interaction_multi(
            state, system, traverse, _force_kernel, _force_reduce
        )
        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="MultiCellList.compute_potential_energy")
    def compute_potential_energy(state: State, system: System) -> jax.Array:
        collider = cast(MultiCellList, system.collider)
        traverse = partial(
            _static_traverse_multi_cell,
            max_occupancy=collider.max_occupancy,
            weighted=True,
        )
        _, energy = _compute_interaction_multi(state, system, traverse, _energy_kernel)
        return jnp.sum(energy)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    @partial(jax.named_call, name="MultiCellList.create_neighbor_list")
    def create_neighbor_list(
        state: State, system: System, cutoff: float, max_neighbors: int
    ) -> tuple[State, System, jax.Array, jax.Array]:
        collider = cast(MultiCellList, system.collider)
        MAX_OCCUPANCY = collider.max_occupancy
        cutoff_sq = cutoff**2

        def traverse(
            idx: jax.Array,
            pos_i: jax.Array,
            p_hashes: jax.Array,
            state: State,
            system: System,
            pos: jax.Array,
            sorted_hashes: jax.Array,
            perm: jax.Array,
            cell_size: jax.Array,
            weighted: bool = False,
        ) -> tuple[jax.Array, jax.Array]:
            total_elements = sorted_hashes.shape[0]

            def per_cell(target_hash: jax.Array) -> tuple[jax.Array, jax.Array]:
                start_idx = jnp.searchsorted(sorted_hashes, target_hash, side="left")
                flat_indices = start_idx + jax.lax.iota(dtype=int, size=MAX_OCCUPANCY)
                safe_flat = jnp.minimum(flat_indices, total_elements - 1)
                j_indices = perm[safe_flat]

                valid_hash = (
                    (target_hash != -1)
                    & (flat_indices < total_elements)
                    & (sorted_hashes[safe_flat] == target_hash)
                )
                canonical_hash = jax.vmap(
                    lambda j: _compute_canonical_hash(
                        pos_i,
                        pos[j],
                        state.rad[idx],
                        state.rad[j],
                        cell_size,
                        system,
                        weighted=weighted,
                    )
                )(j_indices)
                valid_canonical = target_hash == canonical_hash

                dr = system.domain.displacement(pos_i, pos[j_indices], system)
                dist_sq = norm2(dr)

                valid = (
                    valid_hash
                    & valid_canonical
                    & valid_interaction_mask(
                        state.clump_id[j_indices],
                        state.clump_id[idx],
                        state.bond_id[j_indices],
                        state.bond_id[idx],
                        system.interact_same_bond_id,
                    )
                    & (dist_sq <= cutoff_sq)
                    & (idx != j_indices)
                )

                return jnp.where(valid, j_indices, -1), valid

            candidates, valid_mask = jax.vmap(per_cell)(p_hashes)
            candidates = candidates.ravel()
            valid_mask = valid_mask.ravel()

            num_neighbors = jnp.sum(valid_mask)
            overflow = num_neighbors > max_neighbors

            topk = jax.lax.top_k(
                jnp.where(valid_mask, candidates, -1),
                min(max_neighbors, candidates.size),
            )[0]
            if max_neighbors > candidates.size:
                topk = jnp.concatenate(
                    [
                        topk,
                        jnp.full(
                            (max_neighbors - candidates.size,), -1, dtype=topk.dtype
                        ),
                    ]
                )
            return topk, overflow

        search_rad = jnp.maximum(state.rad, cutoff / 2.0)
        return _compute_neighbor_list_common_multi(
            state, system, traverse, max_neighbors, search_rad
        )

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    def create_cross_neighbor_list(
        pos_a: jax.Array,
        pos_b: jax.Array,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError(
            "MultiCellList does not yet support cross_neighbor_list"
        )


@Collider.register("DynamicMultiCellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DynamicMultiCellList(Collider):
    """Implicit multi-cell list (spatial hashing) collider using dynamic while-loop traversal.

    Allows large particles to span multiple cells, completely bypassing max_occupancy limits.
    Each particle can occupy up to ``max_hashes`` grid cells. Pair interactions are
    de-duplicated using a canonical cell hash based on the interaction center.
    """

    cell_size: jax.Array
    max_hashes: int = jax.tree.static()
    max_cells_per_axis: int = jax.tree.static()

    @classmethod
    def Create(
        cls,
        state: State,
        cell_size: ArrayLike | None = None,
        box_size: ArrayLike | None = None,
        max_hashes: int | None = None,
    ) -> Self:
        min_rad = jnp.min(state.rad)
        max_rad = jnp.max(state.rad)
        alpha = max_rad / min_rad

        if cell_size is None:
            cell_size = 2.0 * max_rad if alpha < 2.5 else 0.5 * max_rad
        cell_size_val = jnp.asarray(cell_size, dtype=float)

        if box_size is not None:
            box_size = jnp.asarray(box_size, dtype=float)
            grid_dims = jnp.floor(box_size / cell_size_val).astype(int)
            grid_dims = jnp.maximum(grid_dims, 1)
            cell_size_val = jnp.min(box_size / grid_dims)

        max_rad = jnp.max(state.rad)
        S = jnp.ceil(2 * max_rad / cell_size_val).astype(int)
        max_cells_per_axis = int(S + 1)

        if max_hashes is None:
            max_hashes = max_cells_per_axis**state.dim

        return cls(
            cell_size=cell_size_val,
            max_hashes=int(max_hashes),
            max_cells_per_axis=max_cells_per_axis,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="DynamicMultiCellList.compute_force")
    def compute_force(state: State, system: System) -> tuple[State, System]:
        zero_f = (
            jnp.zeros(state.dim, dtype=float),
            jnp.zeros(1 if state.dim == 2 else 3, dtype=float),
        )
        traverse = partial(
            _dynamic_traverse_multi_cell, init_val=zero_f, weighted=False
        )
        state, (state.force, state.torque) = _compute_interaction_multi(
            state, system, traverse, _force_kernel, _force_reduce
        )
        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="DynamicMultiCellList.compute_potential_energy")
    def compute_potential_energy(state: State, system: System) -> jax.Array:
        traverse = partial(
            _dynamic_traverse_multi_cell,
            init_val=jnp.array(0.0, dtype=float),
            weighted=True,
        )
        _, energy = _compute_interaction_multi(state, system, traverse, _energy_kernel)
        return jnp.sum(energy)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    @partial(jax.named_call, name="DynamicMultiCellList.create_neighbor_list")
    def create_neighbor_list(
        state: State, system: System, cutoff: float, max_neighbors: int
    ) -> tuple[State, System, jax.Array, jax.Array]:
        collider = cast(DynamicMultiCellList, system.collider)
        cutoff_sq = cutoff**2

        def traverse(
            idx: jax.Array,
            pos_i: jax.Array,
            p_hashes: jax.Array,
            state: State,
            system: System,
            pos: jax.Array,
            sorted_hashes: jax.Array,
            perm: jax.Array,
            cell_size: jax.Array,
            weighted: bool = False,
        ) -> tuple[jax.Array, jax.Array]:
            total_elements = sorted_hashes.shape[0]

            def stencil_body(
                target_cell_hash: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                start_idx = jnp.searchsorted(
                    sorted_hashes, target_cell_hash, side="left"
                )
                local_capacity = max_neighbors // 2 + 1
                init_carry = (
                    start_idx,
                    jnp.array(0, dtype=int),
                    jnp.full((local_capacity,), -1, dtype=int),
                    jnp.array(False),
                )

                def cond_fun(
                    val: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ) -> bool:
                    flat_idx, c, _, _ = val
                    in_bounds = flat_idx < total_elements
                    safe_idx = jnp.minimum(flat_idx, total_elements - 1)
                    matches_hash = sorted_hashes[safe_idx] == target_cell_hash
                    has_space = c < local_capacity + 1
                    return cast(
                        bool,
                        in_bounds & matches_hash & has_space & (target_cell_hash != -1),
                    )

                def body_fun(
                    val: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    flat_idx, c, nl, overflow = val
                    j_idx = perm[flat_idx]

                    canonical_hash = _compute_canonical_hash(
                        pos_i,
                        pos[j_idx],
                        state.rad[idx],
                        state.rad[j_idx],
                        cell_size,
                        system,
                        weighted=weighted,
                    )

                    dr = system.domain.displacement(pos_i, pos[j_idx], system)
                    dist_sq = norm2(dr)

                    valid = (
                        (target_cell_hash == canonical_hash)
                        & valid_interaction_mask(
                            state.clump_id[j_idx],
                            state.clump_id[idx],
                            state.bond_id[j_idx],
                            state.bond_id[idx],
                            system.interact_same_bond_id,
                        )
                        & (dist_sq <= cutoff_sq)
                        & (idx != j_idx)
                    )

                    nl = jax.lax.cond(
                        valid,
                        lambda nl_: nl_.at[c].set(j_idx, mode="drop"),
                        lambda nl_: nl_,
                        nl,
                    )
                    c_new = c + valid.astype(c.dtype)
                    return flat_idx + 1, c_new, nl, overflow + (c_new > local_capacity)

                _, local_c, local_nl, local_overflow = jax.lax.while_loop(
                    cond_fun, body_fun, init_carry
                )
                return local_nl, local_c, local_overflow

            final_n_list, stencil_counts, stencil_overflows = jax.vmap(stencil_body)(
                p_hashes
            )
            row_offsets = jnp.cumsum(stencil_counts) - stencil_counts
            local_iota = jnp.arange(final_n_list.shape[1])
            target_indices = row_offsets[:, None] + local_iota[None, :]
            valid_mask = local_iota[None, :] < stencil_counts[:, None]
            safe_indices = jnp.where(
                valid_mask.flatten(), target_indices.flatten(), max_neighbors
            )
            result = jnp.full((max_neighbors,), -1, dtype=final_n_list.dtype)
            final_n_list = result.at[safe_indices].set(
                final_n_list.flatten(), mode="drop"
            )
            overflow_flag = (
                jnp.sum(stencil_overflows)
                > 0 + (jnp.sum(stencil_counts) > max_neighbors)
            ).astype(bool)
            return final_n_list, overflow_flag

        search_rad = jnp.maximum(state.rad, cutoff / 2.0)
        return _compute_neighbor_list_common_multi(
            state, system, traverse, max_neighbors, search_rad
        )

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    def create_cross_neighbor_list(
        pos_a: jax.Array,
        pos_b: jax.Array,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError(
            "DynamicMultiCellList does not yet support cross_neighbor_list"
        )


__all__ = ["MultiCellList", "DynamicMultiCellList"]
