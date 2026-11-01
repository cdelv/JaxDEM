# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Multi-cell (loose-grid / UGrid) collider — a JAX port of dragon-space's loose/tight grid."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, cast

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from ..utils.linalg import cross, norm2
from ..domains import SearchGeometry
from . import Collider, valid_interaction_mask
from ._cell_options import validate_cell_options
from ._cell_lookup import _cell_starts
from ._partition import (
    NEIGHBOR_QUERY_BATCH_SIZE,
    PAIR_TRAVERSAL_BATCH_SIZE,
    PAIR_UNROLL,
    _dedup_stencil_hashes,
    _energy_pair_fn,
    _force_pair_fn,
    _get_spatial_partition,
    _loose_cell_aabbs,
    _make_direct_row_body,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.jit(inline=True, static_argnames=("pair_fn",))
def _traverse_pairs_legacy(
    state: State,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    pair_fn: Callable[..., Any],
    init_acc: Any,
) -> tuple[Any, jax.Array]:
    """Fold a per-pair kernel over candidate pairs of the loose-grid partition.

    Each loose cell carries an expandable AABB, the union of its members'
    boxes. A loop-invariant cell-overlap flag gates the inner ``while_loop``.
    The traversal skips a stencil cell when the cell's box does not overlap
    the query box of particle ``i``. It never walks that cell's member run.
    This is the vectorized, periodic-correct replacement
    for the original UGrid's tight grid. The while loop visits ``PAIR_UNROLL``
    candidates per iteration.

    Returns
    -------
    tuple[Any, jax.Array]
        The per-particle accumulator pytree, and
        the partition's ``hash_overflow`` flag.
    """
    N = state.N
    if N == 0:
        empty = jax.tree.map(lambda x: jnp.empty((0,) + x.shape, x.dtype), init_acc)
        return empty, jnp.asarray(False)
    pos = state.pos
    search_rad = system.force_model.search_radii(state, system)
    j_arr = jax.lax.iota(dtype=int, size=PAIR_UNROLL)
    iota = jax.lax.iota(dtype=int, size=N)
    (
        perm,
        p_cell_hash,
        p_neighbor_cell_hashes,
        hash_overflow,
    ) = _get_spatial_partition(pos, system, cell_size, neighbor_mask, iota)
    p_neighbor_cell_starts = _cell_starts(p_cell_hash, p_neighbor_cell_hashes)

    # Conservative per-particle AABBs. For pure spheres this is just
    # pos +/- rad.
    xmin = pos - search_rad[:, None]
    xmax = pos + search_rad[:, None]

    # We need aabb_center and aabb_half in sorted order for cell_center computation
    xmin_sorted = xmin[perm]
    xmax_sorted = xmax[perm]
    cell_center, cell_half = _loose_cell_aabbs(xmin_sorted, xmax_sorted, p_cell_hash)

    def per_particle(
        orig_idx: jax.Array,
        neighbor_hashes: jax.Array,
        neighbor_starts: jax.Array,
    ) -> Any:
        if system.domain.periodic:
            neighbor_hashes = _dedup_stencil_hashes(neighbor_hashes)

        center_i = pos[orig_idx]
        half_i = search_rad[orig_idx]

        def per_cell(target_hash: jax.Array, start_idx: jax.Array) -> Any:
            safe_start = jnp.minimum(start_idx, N - 1)

            # Loose-cell AABB prune: does this cell's expanded box reach the
            # query box of particle i? Loop-invariant over the member run.
            dr_cell = system.domain.displacement(
                center_i, cell_center[safe_start], system
            )
            aabb_overlap = jnp.all(jnp.abs(dr_cell) <= half_i + cell_half[safe_start])
            if hasattr(system.domain, "gamma"):
                # These AABBs live in the primary orthogonal image.  The LE
                # stencil is conservative, but an orthogonal AABB rejection
                # across a shear image is not yet proven conservative.
                aabb_overlap = jnp.asarray(True)
            cell_overlap = (
                (start_idx < N)
                * (p_cell_hash[safe_start] == target_hash)
                * aabb_overlap
            )

            def cond_fun(val: tuple[jax.Array, Any]) -> bool:
                k, _ = val
                return cast(
                    bool,
                    (k < N)
                    * (p_cell_hash[jnp.minimum(k, N - 1)] == target_hash)
                    * cell_overlap,
                )

            def body_fun(val: tuple[jax.Array, Any]) -> tuple[jax.Array, Any]:
                k, acc = val
                kj = jnp.minimum(k + j_arr, N - 1)
                in_cell = ((k + j_arr) < N) * (p_cell_hash[kj] == target_hash)

                orig_kj = perm[kj]

                valid = in_cell * valid_interaction_mask(
                    state.clump_id[orig_kj],
                    state.clump_id[orig_idx],
                    state.bond_id[orig_kj],
                    orig_idx,
                    system.interact_same_bond_id,
                )

                zero_acc = jax.tree.map(jnp.zeros_like, acc)
                vec_acc = pair_fn(zero_acc, orig_idx, orig_kj, pos, state, valid)
                acc = jax.tree.map(lambda a, v: a + jnp.sum(v, axis=0), acc, vec_acc)

                return k + PAIR_UNROLL, acc

            _, final_acc = jax.lax.while_loop(cond_fun, body_fun, (start_idx, init_acc))
            return final_acc

        cell_results = jax.vmap(per_cell)(neighbor_hashes, neighbor_starts)
        return jax.tree.map(lambda x: x.sum(axis=0), cell_results)

    acc = jax.lax.map(
        lambda row: per_particle(*row),
        (iota, p_neighbor_cell_hashes, p_neighbor_cell_starts),
        batch_size=min(N, PAIR_TRAVERSAL_BATCH_SIZE),
    )
    return acc, hash_overflow


@jax.jit(inline=True, static_argnames=("pair_fn",))
def _traverse_pairs(
    state: State,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    pair_fn: Callable[..., Any],
    init_acc: Any,
) -> tuple[Any, jax.Array]:
    """Keep scalar traversal and share loop bounds under explicit batching."""
    from ._dynamic_pairs import _can_share_pair_loop, _dispatch_pair_traversal

    if _can_share_pair_loop(pair_fn):
        return _dispatch_pair_traversal(
            state,
            system,
            cell_size,
            neighbor_mask,
            pair_fn,
            init_acc,
            pair_system=cast(partial[Any], pair_fn).keywords["system"],
            legacy=_traverse_pairs_legacy,
            prune_cells=True,
            pair_unroll=PAIR_UNROLL,
            batch_size=PAIR_TRAVERSAL_BATCH_SIZE,
        )
    return _traverse_pairs_legacy(
        state, system, cell_size, neighbor_mask, pair_fn, init_acc
    )


@Collider.register("MultiCellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DynamicMultiCellList(Collider):
    r"""Loose-cell collider with dynamic or fixed-capacity static traversal.

    A regular grid assigns each particle center to one cell. Sorted cell
    hashes place each cell's members in one contiguous run. Each loose cell
    also stores the union of its members' axis-aligned search boxes.

    A fixed stencil selects candidate cells. A conservative box test skips
    cells whose members cannot contact the query particle. Shear-periodic
    domains skip this box rejection and retain the conservative stencil.

    With ``tile_capacity=None``, occupancy sets the dynamic traversal bound.
    Explicit environment batches share the longest active bound within each
    bounded query batch. The dynamic path has no per-cell occupancy limit.

    Set ``tile_capacity`` to a nonnegative Python integer for static traversal.
    This global budget covers four-candidate tiles across all source particles
    and stencil cells in one environment. Rejected cells allocate no tiles.
    Dense cells can use spare capacity from other cells. The static path has
    no per-cell occupancy limit. Each environment uses the same static budget.

    Set ``clump_aware=True`` to exclude same-clump members before static tile
    allocation. All paths exclude same-clump interactions during evaluation.
    External candidates can still exhaust the global tile budget.

    An overflow invalidates the result. After force or energy evaluation,
    call ``system.check_overflow()``. For neighbor queries, check the returned
    overflow flag. The static path reports tile and hash overflow.

    For a fixed stencil, dynamic storage scales as :math:`O(N)`.
    Static storage scales as :math:`O(N + C)`, where :math:`C` is
    ``tile_capacity``. Sorting costs :math:`O(N \log N)` in both paths.

    The dynamic path supports forward differentiation. Set ``tile_capacity``
    or use a ``NeighborList`` cache for reverse differentiation.
    """

    supported_search_geometries = frozenset(
        (SearchGeometry.ORTHOGONAL, SearchGeometry.SHEAR_PERIODIC)
    )

    neighbor_mask: jax.Array
    """Integer offsets defining the neighbor stencil (M, dim)."""

    cell_size: jax.Array
    """Linear size of a loose grid cell (scalar)."""

    tile_capacity: int | None = jax.tree.static(default=None)
    """Global tile budget; None selects the occupancy-dependent traversal."""

    clump_aware: bool = jax.tree.static(default=False)
    """Exclude same-clump members before static tile allocation."""

    _prune_cells: ClassVar[bool] = True

    @property
    def stateful(self) -> bool:
        return True

    @classmethod
    def Create(
        cls,
        state: State,
        cell_size: ArrayLike | None = None,
        search_range: ArrayLike | None = None,
        box_size: ArrayLike | None = None,
        max_hashes: int | None = None,
        tile_capacity: int | None = None,
        clump_aware: bool = False,
    ) -> Self:
        """Create a DynamicMultiCellList instance from the reference state.

        Parameters
        ----------
        state : State
            Reference state containing positions and radii.
        cell_size : float, optional
            Loose grid cell size. Defaults to ``2 * r_max``.
        search_range : int, optional
            Number of neighboring cells to search per axis.
        box_size : ArrayLike, optional
            Bounding dimensions of the physical box. Needed only when the box
            size is small compared with the cell size.
        max_hashes : int, optional
            Deprecated and ignored. Accepted for backward compatibility with
            the previous AABB-registration multi-cell list. The loose-grid
            implementation stores every particle in a single cell.

        tile_capacity : int, optional
            Global budget of four-candidate traversal tiles per environment.
            None keeps dynamic traversal. A supplied integer selects static
            traversal, including zero for an empty budget.
        clump_aware : bool, optional
            Exclude same-clump members before static tile allocation. This
            option affects allocation only in the static mode.

        Returns
        -------
        DynamicMultiCellList
            A configured DynamicMultiCellList instance.
        """
        del max_hashes  # deprecated no-op, kept for API compatibility

        validate_cell_options(tile_capacity, clump_aware)
        max_rad = jnp.max(state._rad, initial=0.0)

        if cell_size is None:
            cell_size = jnp.where(max_rad > 0, 2.0 * max_rad, 1.0)
        cell_size = jnp.asarray(cell_size, dtype=float)
        if cell_size.ndim != 0 or not bool(jnp.isfinite(cell_size) & (cell_size > 0)):
            raise ValueError("cell_size must be a finite positive scalar")
        if search_range is not None:
            sr_value = float(jnp.asarray(search_range))
            if (
                not bool(jnp.isfinite(sr_value))
                or sr_value < 1
                or sr_value != int(sr_value)
            ):
                raise ValueError("search_range must be a positive integer")

        if box_size is not None:
            box_size = jnp.asarray(box_size, dtype=float)
            for _ in range(2):
                if search_range is None:
                    sr = jnp.maximum(1, jnp.ceil(2 * max_rad / cell_size).astype(int))
                else:
                    sr = jnp.asarray(search_range, dtype=int)
                min_grids_per_axis = 2 * sr + 1
                grid_dims = jnp.floor(box_size / cell_size).astype(int)
                grid_dims = jnp.maximum(grid_dims, min_grids_per_axis)
                cell_size = jnp.min(box_size / grid_dims)

        if search_range is None:
            search_range = jnp.maximum(1, jnp.ceil(2 * max_rad / cell_size).astype(int))
        search_range = jnp.asarray(search_range, dtype=int)

        r = jnp.arange(-search_range, search_range + 1, dtype=int)
        mesh = jnp.meshgrid(*([r] * state.dim), indexing="ij")
        neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)

        return cls(
            neighbor_mask=neighbor_mask.astype(int),
            cell_size=jnp.asarray(cell_size, dtype=float),
            tile_capacity=tile_capacity,
            clump_aware=clump_aware,
        )

    @staticmethod
    @jax.jit(inline=True, static_argnames=("advance_history", "quasistatic"))
    @partial(jax.named_call, name="DynamicMultiCellList.compute_force")
    def compute_force(
        state: State,
        system: System,
        *,
        advance_history: bool = True,
        quasistatic: bool = False,
    ) -> tuple[State, System]:
        """Compute pairwise contact forces and torques with DynamicMultiCellList.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            The updated state and system, including search bounds and overflow status.
        """
        if cast(DynamicMultiCellList, system.collider).tile_capacity is not None:
            from ._static_cell import _static_compute_force

            return _static_compute_force(
                state, system, advance_history=advance_history, quasistatic=quasistatic
            )

        collider = cast(DynamicMultiCellList, system.collider)
        search_radius = jnp.max(
            system.force_model.search_radii(state, system), initial=0.0
        )
        system = system.domain.update_bounds(state.pos, system, padding=search_radius)
        search_range = jnp.maximum(jnp.max(jnp.abs(collider.neighbor_mask)), 1)
        cell_size = jnp.maximum(collider.cell_size, 2.0 * search_radius / search_range)
        (sum_f, sum_t), hash_overflow = _traverse_pairs(
            state,
            system,
            cell_size,
            collider.neighbor_mask,
            partial(_force_pair_fn, system=system, quasistatic=quasistatic),
            (
                jnp.zeros_like(state.force, shape=state.force.shape[1:]),
                jnp.zeros_like(state.torque, shape=state.torque.shape[1:]),
            ),
        )
        state.force = sum_f
        state.torque = sum_t + cross(state._pos_p_rot, sum_f)
        system.collider.overflow = hash_overflow
        return state, system

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="DynamicMultiCellList.compute_potential_energy")
    def compute_potential_energy(
        state: State, system: System
    ) -> tuple[State, System, jax.Array]:
        """Compute the total non-bonded potential energy of the system.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System, jax.Array]
            Tuple of (state, system, energy).
        """
        if cast(DynamicMultiCellList, system.collider).tile_capacity is not None:
            from ._static_cell import _static_compute_potential_energy

            return _static_compute_potential_energy(state, system)

        collider = cast(DynamicMultiCellList, system.collider)
        search_radius = jnp.max(
            system.force_model.search_radii(state, system), initial=0.0
        )
        system = system.domain.update_bounds(state.pos, system, padding=search_radius)
        search_range = jnp.maximum(jnp.max(jnp.abs(collider.neighbor_mask)), 1)
        cell_size = jnp.maximum(collider.cell_size, 2.0 * search_radius / search_range)
        energy, hash_overflow = _traverse_pairs(
            state,
            system,
            cell_size,
            collider.neighbor_mask,
            partial(_energy_pair_fn, system=system),
            jnp.zeros_like(state.pos_c, shape=(), dtype=float),
        )
        system.collider.overflow = hash_overflow
        return state, system, jnp.sum(energy)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",), inline=True)
    @partial(jax.named_call, name="DynamicMultiCellList.create_neighbor_list")
    def create_neighbor_list(
        state: State, system: System, cutoff: float, max_neighbors: int
    ) -> tuple[State, System, jax.Array, jax.Array]:
        """Create a neighbor list of shape (N, max_neighbors) with DynamicMultiCellList.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.
        cutoff : float
            Verlet search cutoff radius.
        max_neighbors : int
            Static size of neighbor buffer per particle.

        Returns
        -------
        Tuple[State, System, jax.Array, jax.Array]
            State, system, neighbor list, and overflow flag.
        """
        if cast(DynamicMultiCellList, system.collider).tile_capacity is not None:
            from ._static_cell import _static_create_neighbor_list

            return _static_create_neighbor_list(state, system, cutoff, max_neighbors)

        if max_neighbors < 0:
            raise ValueError("max_neighbors must be non-negative")
        cutoff_sq = cutoff**2
        N = state.N
        if N == 0:
            return (
                state,
                system,
                jnp.empty((0, max_neighbors), dtype=int),
                jnp.asarray(False),
            )
        system = system.domain.update_bounds(state.pos, system, padding=cutoff)

        collider = cast(DynamicMultiCellList, system.collider)
        iota = jax.lax.iota(int, N)
        pos = state.pos

        # Inflate the cell size so the fixed stencil reach covers the requested
        # cutoff (mirrors DynamicCellList.create_neighbor_list).
        search_range = jnp.maximum(jnp.max(collider.neighbor_mask), 1)
        cell_size = jnp.maximum(collider.cell_size, cutoff / search_range)

        # 1. Spatial Partitioning
        (
            perm,
            p_cell_hash,
            p_neighbor_hashes,
            hash_overflow,
        ) = _get_spatial_partition(pos, system, cell_size, collider.neighbor_mask, iota)
        p_neighbor_starts = _cell_starts(p_cell_hash, p_neighbor_hashes)

        permuted_pos = pos[perm]
        cell_center, cell_half = _loose_cell_aabbs(
            permuted_pos, permuted_pos, p_cell_hash
        )

        local_capacity = max_neighbors

        def traverse(
            idx: jax.Array,
            pos_i: jax.Array,
            stencil: jax.Array,
            cell_starts: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            if system.domain.periodic:
                stencil = _dedup_stencil_hashes(stencil)

            def candidate_valid(k: jax.Array) -> jax.Array:
                orig_k = perm[k]
                dr = system.domain.displacement(pos_i, permuted_pos[k], system)
                d_sq = norm2(dr)
                return valid_interaction_mask(
                    state.clump_id[orig_k],
                    state.clump_id[idx],
                    state.bond_id[orig_k],
                    idx,
                    system.interact_same_bond_id,
                ) * (d_sq <= cutoff_sq)

            row_body = _make_direct_row_body(
                p_cell_hash, N, local_capacity, candidate_valid
            )

            def mask_cell(target_hash: jax.Array, start_idx: jax.Array) -> jax.Array:
                # Loose-cell prune: skip the cell unless its members' box can
                # contain a point within ``cutoff`` of the query.
                safe_start = jnp.minimum(start_idx, N - 1)
                dr_cell = system.domain.displacement(
                    pos_i, cell_center[safe_start], system
                )
                overlap = jnp.all(jnp.abs(dr_cell) <= cutoff + cell_half[safe_start])
                if hasattr(system.domain, "gamma"):
                    overlap = jnp.asarray(True)
                sentinel = jnp.bitwise_not(jnp.asarray(0, target_hash.dtype))
                return jnp.where(overlap, target_hash, sentinel)

            masked_stencil = jax.vmap(mask_cell)(stencil, cell_starts)
            return row_body(masked_stencil, cell_starts)

        if max_neighbors == 0:
            topk, row_overflows = jax.vmap(traverse)(
                iota, pos, p_neighbor_hashes, p_neighbor_starts
            )
        else:
            topk, row_overflows = jax.lax.map(
                lambda args: traverse(*args),
                (iota, pos, p_neighbor_hashes, p_neighbor_starts),
                batch_size=min(N, NEIGHBOR_QUERY_BATCH_SIZE),
            )

        mask = topk != -1
        topk = jnp.where(mask, perm[topk], -1)

        overflow_flag = jnp.any(row_overflows) | hash_overflow

        return state, system, topk, overflow_flag

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",), inline=True)
    @partial(jax.named_call, name="DynamicMultiCellList.create_cross_neighbor_list")
    def create_cross_neighbor_list(
        pos_a: jax.Array,
        pos_b: jax.Array,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Create a cross-neighbor list between pos_a (query) and pos_b (database).

        Parameters
        ----------
        pos_a : jax.Array
            Query positions, shape (N_A, dim).
        pos_b : jax.Array
            Database positions, shape (N_B, dim).
        system : System
            The configuration of the simulation.
        cutoff : float
            Verlet search cutoff radius.
        max_neighbors : int
            Static size of neighbor buffer per particle.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            Cross-neighbor list of shape (N_A, max_neighbors) and overflow flag.
        """
        if cast(DynamicMultiCellList, system.collider).tile_capacity is not None:
            from ._static_cell import _static_create_cross_neighbor_list

            return _static_create_cross_neighbor_list(
                pos_a, pos_b, system, cutoff, max_neighbors
            )

        if max_neighbors < 0:
            raise ValueError("max_neighbors must be non-negative")
        n_a = pos_a.shape[0]
        n_b = pos_b.shape[0]
        if n_a == 0:
            return jnp.empty((0, max_neighbors), dtype=int), jnp.asarray(False)
        if n_b == 0:
            return jnp.full((n_a, max_neighbors), -1, dtype=int), jnp.asarray(False)

        system = system.domain.update_bounds(
            jnp.concatenate((pos_a, pos_b), axis=0), system, padding=cutoff
        )

        collider = cast(DynamicMultiCellList, system.collider)

        search_range = jnp.maximum(jnp.max(collider.neighbor_mask), 1)
        cell_size = jnp.maximum(collider.cell_size, cutoff / search_range)

        # 1. Permute pos_b into cells
        iota_b = jax.lax.iota(int, n_b)
        (
            perm_b,
            p_cell_hash_b,
            _,
            hash_overflow_b,
        ) = _get_spatial_partition(
            pos_b, system, cell_size, collider.neighbor_mask, iota_b
        )
        pos_b_permuted = pos_b[perm_b]
        cell_center_b, cell_half_b = _loose_cell_aabbs(
            pos_b_permuted, pos_b_permuted, p_cell_hash_b
        )

        # 2. Get query neighbor stencils
        iota_a = jax.lax.iota(int, n_a)
        (
            _,
            _,
            p_neighbor_hashes_a,
            hash_overflow_a,
        ) = _get_spatial_partition(
            pos_a, system, cell_size, collider.neighbor_mask, iota_a
        )
        p_neighbor_starts_a = _cell_starts(p_cell_hash_b, p_neighbor_hashes_a)

        cutoff_sq = cutoff**2
        local_capacity = max_neighbors

        # 3. For each original-A point, find neighbors in permuted B
        def traverse(
            pos_ai: jax.Array,
            stencil: jax.Array,
            cell_starts: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            if system.domain.periodic:
                stencil = _dedup_stencil_hashes(stencil)

            def candidate_valid(k: jax.Array) -> jax.Array:
                dr = system.domain.displacement(pos_ai, pos_b_permuted[k], system)
                return norm2(dr) <= cutoff_sq

            row_body = _make_direct_row_body(
                p_cell_hash_b, n_b, local_capacity, candidate_valid
            )

            def mask_cell(target_hash: jax.Array, start_idx: jax.Array) -> jax.Array:
                safe_start = jnp.minimum(start_idx, n_b - 1)
                dr_cell = system.domain.displacement(
                    pos_ai, cell_center_b[safe_start], system
                )
                overlap = jnp.all(jnp.abs(dr_cell) <= cutoff + cell_half_b[safe_start])
                if hasattr(system.domain, "gamma"):
                    overlap = jnp.asarray(True)
                sentinel = jnp.bitwise_not(jnp.asarray(0, target_hash.dtype))
                return jnp.where(overlap, target_hash, sentinel)

            masked_stencil = jax.vmap(mask_cell)(stencil, cell_starts)
            return row_body(masked_stencil, cell_starts)

        if max_neighbors == 0:
            topk, row_overflows = jax.vmap(traverse)(
                pos_a, p_neighbor_hashes_a, p_neighbor_starts_a
            )
        else:
            topk, row_overflows = jax.lax.map(
                lambda args: traverse(*args),
                (pos_a, p_neighbor_hashes_a, p_neighbor_starts_a),
                batch_size=min(n_a, NEIGHBOR_QUERY_BATCH_SIZE),
            )

        # 4. Map permuted-B indices back to original B indices
        valid_mask_nl = topk != -1
        safe_indices_nl = jnp.where(valid_mask_nl, topk, 0)
        topk = jnp.where(valid_mask_nl, perm_b[safe_indices_nl], -1)

        overflow_flag = jnp.any(row_overflows) | hash_overflow_a | hash_overflow_b

        return topk, overflow_flag


__all__ = ["DynamicMultiCellList"]
