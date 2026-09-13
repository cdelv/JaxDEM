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
from ..domains import SearchGeometry
from . import Collider, valid_interaction_mask
from ._partition import (
    NEIGHBOR_QUERY_BATCH_SIZE,
    PAIR_TRAVERSAL_BATCH_SIZE,
    _cell_starts,
    _energy_pair_fn,
    _force_pair_fn,
)
from .cell_list import (
    _dedup_stencil_hashes,
    _get_spatial_partition,
    _make_direct_row_body,
)

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


@jax.jit(inline=True, static_argnames=("pair_fn",))
def _traverse_pairs(
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


@Collider.register("MultiCellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DynamicMultiCellList(Collider):
    r"""Multi-cell (loose-grid / UGrid) collider — a JAX port of dragon-space's loose/tight grid.

    This collider adapts the spatial-partitioning strategy of the ``UGrid`` /
    loose-grid structure to JAX's static-shape, rebuilt-every-frame, fully
    vectorized model. dragon-space popularized the loose/tight "double grid".
    It was the fastest CPU collider in the
    ``DynamicSpatialPartitioning`` benchmarks.

    **Loose grid.** As in a cell list, the domain is a regular grid and the
    collider bins every particle into exactly one cell by its *center*. To
    build the cell index, an internal permutation sorts the hashes so each
    cell's members form a contiguous run. Unlike a plain cell list, each loose
    cell also carries an **expandable AABB** — the union of its members' boxes
    ``center +/- rad``. A segmented min/max reduction over the sorted runs
    computes this AABB.

    **Query.** For every particle ``i``, the fixed ``neighbor_mask`` stencil
    enumerates candidate loose cells. Before the walk of a cell's member run,
    the collider tests the cell's expandable AABB against the query box of
    ``i``. It skips non-overlapping cells entirely. This loose-cell pruning
    replaces the original algorithm's *tight grid* in a vectorized,
    periodic-correct way. The tight grid's only job on a scalar CPU was to
    enumerate the few loose cells near a query instead of a full fixed
    stencil.

    The prune only skips cells whose members are all non-contacting, so
    forces are bit-identical to :class:`~jaxdem.colliders.cell_list.DynamicCellList`.
    The two coincide when every loose cell is full and tight. This collider
    is faster when stencil cells are sparsely or asymmetrically occupied, so
    their boxes do not reach the query. That regime — polydispersity, loose
    packings, cells larger than the contact range — motivates the loose/tight
    design.

    The incremental ``insert``/``move``/``remove`` operations of the CPU
    original do **not** carry over. JAX rebuilds the partition functionally
    each step as a permutation plus a segmented reduction. This is the price
    of running on GPU/TPU, batching with ``vmap``, and differentiating
    through the simulation.

    Constructor Parameters
    ----------------------
    - **cell_size**: Loose-cell side length. Larger cells give fewer, fuller
      cells: longer member runs, a smaller stencil, and more effective AABB
      pruning. Smaller cells give a larger stencil. If ``None``, defaults to
      :math:`2 r_{max}`.
    - **search_range**: Stencil reach in cells per axis. If ``None``, the
      constructor chooses it so the stencil covers every contact within
      :math:`2 r_{max}`.
    - **box_size**: Physical box extents. Needed only when the box is small
      relative to the cell size under periodic boundaries.

    Complexity
    ----------
    - Time: :math:`O(N \log N)` from the sort, plus
      :math:`O(N \cdot M \cdot \langle K \rangle)` for traversal (``M`` =
      stencil size, :math:`\langle K \rangle` = average occupancy), reduced by
      AABB cell-skipping.
    - Memory: :math:`O(N)`.
    """

    supported_search_geometries = frozenset(
        (SearchGeometry.ORTHOGONAL, SearchGeometry.SHEAR_PERIODIC)
    )

    neighbor_mask: jax.Array
    """Integer offsets defining the neighbor stencil (M, dim)."""

    cell_size: jax.Array
    """Linear size of a loose grid cell (scalar)."""

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

        Returns
        -------
        DynamicMultiCellList
            A configured DynamicMultiCellList instance.
        """
        del max_hashes  # deprecated no-op, kept for API compatibility

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
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="DynamicMultiCellList.compute_force")
    def compute_force(state: State, system: System) -> tuple[State, System]:
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
            A tuple containing the updated state and unmodified system.
        """
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
            partial(_force_pair_fn, system=system),
            (
                jnp.zeros(state.force.shape[1:], dtype=state.force.dtype),
                jnp.zeros(state.torque.shape[1:], dtype=state.torque.dtype),
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
