# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Multi-cell (loose-grid / UGrid) collider — a JAX port of dragon-space's loose/tight grid."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from ..utils.linalg import cross, norm2
from . import Collider, valid_interaction_mask
from ._partition import (
    _energy_pair_fn,
    _force_pair_fn,
    _pack_stencil_lists,
)
from .cell_list import _dedup_stencil_hashes, _get_spatial_partition

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


#: Number of candidate particles each ``while_loop`` iteration of the pair and
#: neighbor-list traversals visits. A vmapped ``lax.while_loop`` runs until the
#: *longest* lane finishes, and every iteration costs a device->host round-trip
#: of the loop predicate on GPU (stalling async dispatch), so visiting several
#: candidates per iteration divides the number of synchronizations by
#: ``PAIR_UNROLL`` at the price of at most ``PAIR_UNROLL - 1`` extra masked
#: pair evaluations per lane. Empirically ``4`` is the sweet spot on both CPU
#: and GPU.
PAIR_UNROLL = 4


@jax.jit(inline=True)
@partial(jax.named_call, name="multi_cell_list._get_base_search_rad")
def _get_base_search_rad(state: State, system: System) -> jax.Array:
    """Returns the base search radius for the cell list.

    The base search radius determines the spatial extent each particle is considered
    to occupy when assigning it to cells or computing loose AABBs.
    """
    return state._rad


@jax.jit(inline=True)
@partial(jax.named_call, name="multi_cell_list._loose_cell_aabbs")
def _loose_cell_aabbs(
    member_min: jax.Array,
    member_max: jax.Array,
    sorted_hash: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Per-loose-cell axis-aligned bounding box, broadcast to each member.

    The hash-sorted particles of a loose cell form a contiguous run, so the
    cell's expandable AABB (the union of every member's box
    ``[member_min, member_max]``) is a segmented min/max reduction over those
    runs. The result is returned as ``(cell_center, cell_half_extent)``
    indexed by sorted particle.
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


def _make_stencil_body(
    sorted_hashes: jax.Array,
    n_db: int | jax.Array,
    local_capacity: int,
    candidate_valid: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    """Build the per-stencil-cell scan kernel for the neighbor-list builders.

    Mirrors :func:`cell_list._make_stencil_body` but visits ``PAIR_UNROLL``
    consecutive sorted entries per ``while_loop`` iteration. Entries past the
    end of the cell are masked out by the hash check (same-hash entries are
    contiguous after the sort), and writes past ``local_capacity`` are dropped
    while flagging overflow.
    """

    @jax.jit(inline=True)
    def stencil_body(
        target_cell_hash: jax.Array, start_idx: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        init_carry = (
            start_idx,
            jnp.array(0, dtype=int),
            jnp.full((local_capacity,), -1, dtype=int),
            jnp.array(False),  # overflow flag
        )

        def cond_fun(
            val: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> bool:
            k, c, _, _ = val
            safe_k = jnp.minimum(k, jnp.maximum(1, n_db) - 1)
            in_cell = (k < n_db) * (sorted_hashes[safe_k] == target_cell_hash)
            has_space = c < local_capacity + 1
            return cast(bool, in_cell * has_space)

        def body_fun(
            val: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            k, c, nl, overflow = val
            j_arr = jnp.arange(PAIR_UNROLL)
            safe_k_arr = jnp.minimum(k + j_arr, jnp.maximum(1, n_db) - 1)
            in_cell_arr = ((k + j_arr) < n_db) * (
                sorted_hashes[safe_k_arr] == target_cell_hash
            )
            valid_arr = in_cell_arr * candidate_valid(safe_k_arr)

            valid_counts = valid_arr.astype(c.dtype)
            num_valid = jnp.sum(valid_counts)

            cumsum = jnp.cumsum(valid_counts) - valid_counts
            write_idx_arr = c + cumsum
            write_idx_arr = jnp.where(valid_arr, write_idx_arr, local_capacity)

            nl = nl.at[write_idx_arr].set(safe_k_arr, mode="drop")
            c = c + num_valid
            overflow = overflow | (c > local_capacity)
            return k + PAIR_UNROLL, c, nl, overflow

        _, local_c, local_nl, local_overflow = jax.lax.while_loop(
            cond_fun, body_fun, init_carry
        )
        return local_nl, local_c, local_overflow

    return stencil_body


@jax.jit(inline=True, donate_argnames=("state",))
def _reorder_state(state: State, perm: jax.Array) -> State:
    return jax.tree.map(lambda x: x[perm], state)


@jax.jit(inline=True, static_argnames=("pair_fn",))
def _traverse_pairs(
    state: State,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    pair_fn: Callable[..., Any],
    init_acc: Any,
) -> tuple[State, Any, jax.Array]:
    """Fold a per-pair kernel over candidate pairs of the loose-grid partition.

    Each loose cell carries an expandable AABB (the union of its members'
    boxes); a stencil cell whose box does not overlap particle ``i``'s query
    box is skipped wholesale — the inner ``while_loop`` is gated by a
    loop-invariant cell-overlap flag, so its member run is never walked. This
    is the vectorised, periodic-correct replacement for the original UGrid's
    tight grid. The while loop visits ``PAIR_UNROLL`` candidates per iteration.

    Returns
    -------
    tuple[State, Any, jax.Array]
        The cell-hash-sorted state, the per-particle accumulator pytree, and
        the partition's ``hash_overflow`` flag.
    """
    iota = jax.lax.iota(dtype=int, size=state.N)
    (
        perm,
        p_cell_hash,
        p_neighbor_cell_hashes,
        hash_overflow,
    ) = _get_spatial_partition(state.pos, system, cell_size, neighbor_mask, iota)

    # Conservative per-particle AABBs. For pure spheres this is just
    # pos +/- rad.
    search_rad = _get_base_search_rad(state, system)
    xmin = state.pos - search_rad[:, None]
    xmax = state.pos + search_rad[:, None]

    state = jax.tree.map(lambda x: x[perm], state)
    pos = state.pos
    N = state.N
    xmin = xmin[perm]
    xmax = xmax[perm]
    aabb_center = 0.5 * (xmin + xmax)
    aabb_half = 0.5 * (xmax - xmin)

    cell_center, cell_half = _loose_cell_aabbs(xmin, xmax, p_cell_hash)

    def per_particle(idx: jax.Array, neighbor_hashes: jax.Array) -> Any:
        if system.domain.periodic:
            neighbor_hashes = _dedup_stencil_hashes(neighbor_hashes)

        center_i = aabb_center[idx]
        half_i = aabb_half[idx]

        def per_cell(target_hash: jax.Array) -> Any:
            start_idx = jnp.searchsorted(
                p_cell_hash, target_hash, side="left", method="scan_unrolled"
            )
            safe_start = jnp.minimum(start_idx, N - 1)

            # Loose-cell AABB prune: does this cell's expanded box reach the
            # query box of particle i? Loop-invariant over the member run.
            dr_cell = system.domain.displacement(
                center_i, cell_center[safe_start], system
            )
            aabb_overlap = jnp.all(jnp.abs(dr_cell) <= half_i + cell_half[safe_start])
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
                # Visit PAIR_UNROLL consecutive sorted entries per iteration.
                j_arr = jnp.arange(PAIR_UNROLL)
                kj = jnp.minimum(k + j_arr, N - 1)
                in_cell = ((k + j_arr) < N) * (p_cell_hash[kj] == target_hash)

                valid = in_cell * valid_interaction_mask(
                    state.clump_id[kj],
                    state.clump_id[idx],
                    state.bond_id[kj],
                    state.unique_id[idx],
                    system.interact_same_bond_id,
                )

                zero_acc = jax.tree.map(jnp.zeros_like, acc)
                vec_acc = pair_fn(zero_acc, idx, kj, pos, state, valid)
                acc = jax.tree.map(lambda a, v: a + jnp.sum(v, axis=0), acc, vec_acc)

                return k + PAIR_UNROLL, acc

            _, final_acc = jax.lax.while_loop(cond_fun, body_fun, (start_idx, init_acc))
            return final_acc

        cell_results = jax.vmap(per_cell)(neighbor_hashes)
        return jax.tree.map(lambda x: x.sum(axis=0), cell_results)

    acc = jax.vmap(per_particle)(iota, p_neighbor_cell_hashes)
    return state, acc, hash_overflow


@Collider.register("MultiCellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DynamicMultiCellList(Collider):
    r"""Multi-cell (loose-grid / UGrid) collider — a JAX port of dragon-space's loose/tight grid.

    This is the spatial-partitioning strategy of the ``UGrid`` / loose-grid
    structure (the loose/tight "double grid" popularised by dragon-space and
    the fastest CPU collider in the ``DynamicSpatialPartitioning`` benchmarks),
    adapted to JAX's static-shape, rebuilt-every-frame, fully-vectorised model.

    **Loose grid.** Like a cell list, the domain is a regular grid and every
    particle is binned into exactly one cell by its *center*. Particles are
    sorted by cell hash so each cell's members are a contiguous run. Unlike a
    plain cell list, each loose cell additionally carries an **expandable
    AABB** — the union of its members' boxes ``center +/- rad`` — computed by a
    segmented min/max reduction over the sorted runs.

    **Query.** For every particle ``i``, the fixed ``neighbor_mask`` stencil
    enumerates candidate loose cells. Before walking a cell's member run, the
    cell's expandable AABB is tested against ``i``'s query box; non-overlapping
    cells are skipped entirely. This loose-cell pruning is the vectorised,
    periodic-correct stand-in for the original algorithm's *tight grid*, whose
    only job on a scalar CPU was to enumerate the few loose cells actually near
    a query rather than a full fixed stencil.

    The prune only ever skips cells whose members are all non-contacting, so
    forces are bit-identical to :class:`~jaxdem.colliders.cell_list.DynamicCellList`.
    The two coincide when every loose cell is full and tight; this collider
    pulls ahead when stencil cells are sparsely or asymmetrically occupied (so
    their boxes do not reach the query), which is exactly the regime —
    polydispersity, loose packings, cells larger than the contact range — that
    motivates the loose/tight design.

    What does **not** carry over from the CPU original is its incremental
    ``insert``/``move``/``remove`` of a persistent mutable structure: JAX
    rebuilds the partition functionally each step (a near-sorted ``sort`` plus
    a segmented reduction), which is the price of running on GPU/TPU,
    ``vmap``-ing over environments, and differentiating through the simulation.

    Constructor Parameters
    ----------------------
    - **cell_size**: Loose-cell side length. Larger cells mean fewer, fuller
      cells (longer member runs but a smaller stencil and more effective AABB
      pruning); smaller cells mean a larger stencil. If ``None``, defaults to
      :math:`2 r_{max}`.
    - **search_range**: Stencil reach in cells per axis. If ``None``, chosen so
      every contact within :math:`2 r_{max}` is covered by the stencil.
    - **box_size**: Physical box extents; only needed when the box is small
      relative to the cell size under periodic boundaries.

    Complexity
    ----------
    - Time: :math:`O(N \log N)` from the sort, plus
      :math:`O(N \cdot M \cdot \langle K \rangle)` for traversal (``M`` =
      stencil size, :math:`\langle K \rangle` = average occupancy), reduced by
      AABB cell-skipping.
    - Memory: :math:`O(N)`.
    """

    neighbor_mask: jax.Array
    """Integer offsets defining the neighbor stencil (M, dim)."""

    cell_size: jax.Array
    """Linear size of a loose grid cell (scalar)."""

    @classmethod
    def Create(
        cls,
        state: State,
        cell_size: ArrayLike | None = None,
        search_range: ArrayLike | None = None,
        box_size: ArrayLike | None = None,
        max_hashes: int | None = None,
    ) -> Self:
        """Creates a DynamicMultiCellList instance based on the reference state.

        Parameters
        ----------
        state : State
            Reference state containing positions and radii.
        cell_size : float, optional
            Loose grid cell size. Defaults to ``2 * r_max``.
        search_range : int, optional
            Number of neighboring cells to search per axis.
        box_size : ArrayLike, optional
            Bounding dimensions of the physical box. Only needed when the box
            size is small compared with the cell size.
        max_hashes : int, optional
            Deprecated and ignored. Accepted for backward compatibility with
            the previous AABB-registration multi-cell list; the loose-grid
            implementation stores every particle in a single cell.

        Returns
        -------
        DynamicMultiCellList
            A configured DynamicMultiCellList instance.
        """
        del max_hashes  # deprecated no-op, kept for API compatibility

        max_rad = jnp.max(state._rad)

        if cell_size is None:
            cell_size = 2.0 * max_rad
        cell_size = jnp.asarray(cell_size, dtype=float)

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
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="DynamicMultiCellList.compute_force")
    def compute_force(state: State, system: System) -> tuple[State, System]:
        """Computes pairwise contact forces and torques using DynamicMultiCellList.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated state and unmodified system.
        """
        collider = cast(DynamicMultiCellList, system.collider)
        state, (sum_f, sum_t), hash_overflow = _traverse_pairs(
            state,
            system,
            collider.cell_size,
            collider.neighbor_mask,
            partial(_force_pair_fn, system=system),
            (jnp.zeros_like(state.force[0]), jnp.zeros_like(state.torque[0])),
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
        """Computes the total non-bonded potential energy of the system.

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
        collider = cast(DynamicMultiCellList, system.collider)
        state, energy, hash_overflow = _traverse_pairs(
            state,
            system,
            collider.cell_size,
            collider.neighbor_mask,
            partial(_energy_pair_fn, system=system),
            jnp.asarray(0.0, dtype=float),
        )
        system.collider.overflow = hash_overflow
        return state, system, jnp.sum(energy)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",), inline=True)
    @partial(jax.named_call, name="DynamicMultiCellList.create_neighbor_list")
    def create_neighbor_list(
        state: State, system: System, cutoff: float, max_neighbors: int
    ) -> tuple[State, System, jax.Array, jax.Array]:
        """Creates a neighbor list of shape (N, max_neighbors) using DynamicMultiCellList.

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
            Sorted state, system, neighbor list, and overflow flag.
        """
        cutoff_sq = cutoff**2
        N = state.N

        if max_neighbors == 0:
            empty = jnp.empty((N, 0), dtype=int)
            return state, system, empty, jnp.asarray(False)

        collider = cast(DynamicMultiCellList, system.collider)
        iota = jax.lax.iota(int, N)
        pos = state.pos

        # Inflate the cell size so the fixed stencil reach covers the requested
        # cutoff (mirrors DynamicCellList.create_neighbor_list).
        search_range = jnp.maximum(jnp.max(collider.neighbor_mask), 1)
        cell_size = jnp.maximum(collider.cell_size, cutoff / search_range)

        (
            perm,
            p_cell_hash,
            p_neighbor_hashes,
            hash_overflow,
        ) = _get_spatial_partition(
            state.pos, system, cell_size, collider.neighbor_mask, iota
        )

        # state = _reorder_state(state, perm)
        sorted_pos = pos[perm]

        sorted_clump_id = state.clump_id[perm]
        sorted_bond_id = state.bond_id[perm]
        sorted_unique_id = state.unique_id[perm]

        cell_center, cell_half = _loose_cell_aabbs(sorted_pos, sorted_pos, p_cell_hash)

        local_capacity = max_neighbors

        def traverse(
            idx: jax.Array,
            pos_i: jax.Array,
            stencil: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            if system.domain.periodic:
                stencil = _dedup_stencil_hashes(stencil)

            cell_starts = jnp.searchsorted(
                p_cell_hash, stencil, side="left", method="scan_unrolled"
            )

            def candidate_valid(k: jax.Array) -> jax.Array:
                dr = system.domain.displacement(pos_i, sorted_pos[k], system)
                d_sq = norm2(dr)
                return valid_interaction_mask(
                    sorted_clump_id[k],
                    sorted_clump_id[idx],
                    sorted_bond_id[k],
                    sorted_unique_id[idx],
                    system.interact_same_bond_id,
                ) * (d_sq <= cutoff_sq)

            stencil_body = _make_stencil_body(
                p_cell_hash, N, local_capacity, candidate_valid
            )

            def one_cell(
                target_hash: jax.Array, start_idx: jax.Array
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                # Loose-cell prune: skip the cell unless its members' box can
                # contain a point within ``cutoff`` of the query.
                safe_start = jnp.minimum(start_idx, N - 1)
                dr_cell = system.domain.displacement(
                    pos_i, cell_center[safe_start], system
                )
                reach = cutoff + cell_half[safe_start]
                overlap = jnp.all(jnp.abs(dr_cell) <= reach)
                masked_hash = jnp.where(overlap, target_hash, -1)
                return stencil_body(masked_hash, start_idx)

            final_n_list, stencil_counts, stencil_overflows = jax.vmap(one_cell)(
                stencil, cell_starts
            )
            return final_n_list, stencil_counts, stencil_overflows

        all_final_n_list, all_stencil_counts, all_stencil_overflows = jax.vmap(
            traverse
        )(iota, sorted_pos, p_neighbor_hashes)

        sorted_topk, count_overflow = _pack_stencil_lists(
            all_final_n_list, all_stencil_counts, max_neighbors
        )

        mask = sorted_topk != -1
        unsorted_elements = jnp.where(mask, perm[sorted_topk], -1)
        topk = jnp.empty_like(unsorted_elements).at[perm].set(unsorted_elements)

        overflow_flag = jnp.any(all_stencil_overflows) | count_overflow | hash_overflow

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
        """Creates a cross-neighbor list between pos_a (query) and pos_b (database).

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
        n_a = pos_a.shape[0]
        n_b = pos_b.shape[0]
        if n_a == 0:
            return jnp.empty((0, max_neighbors), dtype=int), jnp.asarray(False)
        if n_b == 0:
            return jnp.full((n_a, max_neighbors), -1, dtype=int), jnp.asarray(False)

        if max_neighbors == 0:
            empty = jnp.empty((n_a, 0), dtype=int)
            return empty, jnp.asarray(False)

        collider = cast(DynamicMultiCellList, system.collider)

        search_range = jnp.maximum(jnp.max(collider.neighbor_mask), 1)
        cell_size = jnp.maximum(collider.cell_size, cutoff / search_range)

        # 1. Sort pos_b into cells
        iota_b = jax.lax.iota(int, n_b)
        (
            perm_b,
            p_cell_hash_b,
            _,
            hash_overflow_b,
        ) = _get_spatial_partition(
            pos_b, system, cell_size, collider.neighbor_mask, iota_b
        )
        pos_b_sorted = pos_b[perm_b]

        # 2. Get query neighbor stencils
        iota_a = jax.lax.iota(int, n_a)
        (
            perm_a,
            _,
            p_neighbor_hashes_a,
            hash_overflow_a,
        ) = _get_spatial_partition(
            pos_a, system, cell_size, collider.neighbor_mask, iota_a
        )
        pos_a_sorted = pos_a[perm_a]

        cutoff_sq = cutoff**2
        local_capacity = max_neighbors

        # 3. For each sorted-A point, find neighbors in sorted B
        def traverse(
            pos_ai: jax.Array,
            stencil: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            if system.domain.periodic:
                stencil = _dedup_stencil_hashes(stencil)

            cell_starts = jnp.searchsorted(
                p_cell_hash_b, stencil, side="left", method="scan_unrolled"
            )

            def candidate_valid(k: jax.Array) -> jax.Array:
                dr = system.domain.displacement(pos_ai, pos_b_sorted[k], system)
                return norm2(dr) <= cutoff_sq

            stencil_body = _make_stencil_body(
                p_cell_hash_b, n_b, local_capacity, candidate_valid
            )
            final_n_list, stencil_counts, stencil_overflows = jax.vmap(stencil_body)(
                stencil, cell_starts
            )
            return final_n_list, stencil_counts, stencil_overflows

        all_final_n_list, all_stencil_counts, all_stencil_overflows = jax.vmap(
            traverse
        )(pos_a_sorted, p_neighbor_hashes_a)

        topk, count_overflow = _pack_stencil_lists(
            all_final_n_list, all_stencil_counts, max_neighbors
        )

        # 4. Map sorted-B indices back to original B indices
        valid_mask_nl = topk != -1
        safe_indices_nl = jnp.where(valid_mask_nl, topk, 0)
        topk = jnp.where(valid_mask_nl, perm_b[safe_indices_nl], -1)

        # 5. Unsort from sorted-A order back to original A order
        inv_perm_a = jnp.empty_like(perm_a)
        inv_perm_a = inv_perm_a.at[perm_a].set(iota_a)
        topk = topk[inv_perm_a]

        overflow_flag = (
            jnp.any(all_stencil_overflows)
            | count_overflow
            | hash_overflow_a
            | hash_overflow_b
        )

        return topk, overflow_flag


__all__ = ["DynamicMultiCellList"]
