# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Cell List :math:`O(N \log N)` collider implementation."""

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
    _grid_params,
    _pack_stencil_lists,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.jit
@partial(jax.named_call, name="cell_list._get_spatial_partition")
def _get_spatial_partition(
    pos: jax.Array,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    iota: jax.Array,
) -> tuple[jax.Array, ...]:
    """Computes spatial hashing and partitioning for the cell list.

    Returns
    -------
    tuple[jax.Array, ...]
        ``(perm, p_cell_hash, neighbor_cell_hashes, hash_overflow)``.
    """
    grid_dims, grid_strides, cell_size, hash_overflow = _grid_params(
        system.domain.box_size, cell_size, system.domain.periodic
    )

    if system.domain.periodic:
        p_cell_coords = jnp.floor(
            (((pos - system.domain.anchor) / system.domain.box_size) % 1) * grid_dims
        ).astype(int)
    else:
        p_cell_coords = jnp.floor((pos - system.domain.anchor) / cell_size).astype(int)

    p_cell_hash = jnp.dot(p_cell_coords, grid_strides)

    p_cell_hash, perm = jax.lax.sort([p_cell_hash, iota], num_keys=1)
    p_cell_coords = p_cell_coords[perm]

    neighbor_cell_coords = p_cell_coords[:, None, :] + neighbor_mask

    if system.domain.periodic:
        neighbor_cell_coords -= grid_dims * jnp.floor(
            neighbor_cell_coords / grid_dims
        ).astype(int)
        neighbor_cell_hashes = jnp.dot(neighbor_cell_coords, grid_strides)
    else:
        out_of_bounds = jnp.any(
            (neighbor_cell_coords < 0) | (neighbor_cell_coords >= grid_dims),
            axis=-1,
        )
        neighbor_cell_hashes = jnp.dot(neighbor_cell_coords, grid_strides)
        neighbor_cell_hashes = jnp.where(out_of_bounds, -1, neighbor_cell_hashes)

    return (
        perm,
        p_cell_hash,
        neighbor_cell_hashes,
        hash_overflow,
    )


@jax.jit(inline=True)
@partial(jax.named_call, name="cell_list._dedup_stencil_hashes")
def _dedup_stencil_hashes(stencil_hashes: jax.Array) -> jax.Array:
    """Deduplicate one particle's stencil hashes, padding duplicates with -1."""
    mask = jnp.triu(stencil_hashes[:, None] == stencil_hashes[None, :], k=1)
    is_duplicate = jnp.any(mask, axis=0)
    return stencil_hashes * (~is_duplicate).astype(int) - is_duplicate.astype(int)


def _make_stencil_body(
    sorted_hashes: jax.Array,
    n_db: int | jax.Array,
    local_capacity: int,
    candidate_valid: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    """Build the per-stencil-cell scan kernel shared by the neighbor-list builders.

    The returned ``stencil_body(target_cell_hash, start_idx)`` walks the
    hash-sorted database from ``start_idx`` while the cell hash matches
    ``target_cell_hash``, appending every candidate index ``k`` for which
    ``candidate_valid(k)`` holds into a ``local_capacity``-sized buffer
    (padded with ``-1``), and returns ``(neighbor_buffer, count, overflow)``.

    Parameters
    ----------
    sorted_hashes : jax.Array
        Cell hashes of the database points, sorted ascending.
    n_db : int | jax.Array
        Number of database points.
    local_capacity : int
        Static size of the per-cell neighbor buffer.
    candidate_valid : Callable[[jax.Array], jax.Array]
        Boolean predicate evaluated on each candidate database index.
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
            safe_k = jnp.minimum(k, jnp.maximum(1, n_db) - 1)
            valid = candidate_valid(safe_k)
            nl = jax.lax.cond(
                valid,
                lambda nl_: nl_.at[c].set(safe_k, mode="drop"),
                lambda nl_: nl_,
                nl,
            )
            c_new = c + valid.astype(c.dtype)
            return k + 1, c_new, nl, overflow + (c_new > local_capacity)

        _, local_c, local_nl, local_overflow = jax.lax.while_loop(
            cond_fun, body_fun, init_carry
        )
        return local_nl, local_c, local_overflow

    return stencil_body


#: Number of candidate particles each while-loop iteration of the pair
#: traversals visits. A vmapped ``lax.while_loop`` runs until the *longest*
#: lane finishes and every iteration costs a device->host round-trip of the
#: loop predicate on GPU (stalling async dispatch), so visiting several
#: candidates per iteration divides the number of synchronizations by
#: ``PAIR_UNROLL`` at the price of at most ``PAIR_UNROLL - 1`` extra masked
#: pair evaluations per lane.
PAIR_UNROLL = 4


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
    """Fold a per-pair kernel over all candidate pairs of the cell partition.

    Sorts the state by cell hash, then for every particle walks its occupied
    stencil cells and accumulates ``pair_fn`` over the particles they contain
    (``valid`` already carries the clump/bond interaction mask). This is the
    single traversal backing both :meth:`DynamicCellList.compute_force` and
    :meth:`DynamicCellList.compute_potential_energy`.

    Returns
    -------
    tuple[State, Any, jax.Array]
        The cell-hash-sorted state, the per-particle accumulator pytree
        (each leaf has a leading ``N`` axis), and the ``hash_overflow`` flag
        of the partition.
    """
    iota = jax.lax.iota(dtype=int, size=state.N)
    (
        perm,
        p_cell_hash,
        p_neighbor_cell_hashes,
        hash_overflow,
    ) = _get_spatial_partition(state.pos, system, cell_size, neighbor_mask, iota)

    state = _reorder_state(state, perm)
    pos = state.pos
    N = state.N

    def per_particle(idx: jax.Array, neighbor_hashes: jax.Array) -> Any:
        if system.domain.periodic:
            neighbor_hashes = _dedup_stencil_hashes(neighbor_hashes)

        def per_cell(target_hash: jax.Array) -> Any:
            start_idx = jnp.searchsorted(
                p_cell_hash, target_hash, side="left", method="scan_unrolled"
            )

            def cond_fun(val: tuple[jax.Array, Any]) -> bool:
                k, _ = val
                return cast(bool, (k < N) * (p_cell_hash[k] == target_hash))

            def body_fun(val: tuple[jax.Array, Any]) -> tuple[jax.Array, Any]:
                k, acc = val
                # Visit PAIR_UNROLL consecutive sorted entries per iteration.
                # Entries past the end of the cell are masked out by the hash
                # check (same-hash entries are contiguous after the sort).
                for j in range(PAIR_UNROLL):
                    kj = jnp.minimum(k + j, N - 1)
                    in_cell = ((k + j) < N) * (p_cell_hash[kj] == target_hash)
                    valid = in_cell * valid_interaction_mask(
                        state.clump_id[kj],
                        state.clump_id[idx],
                        state.bond_id[kj],
                        state.unique_id[idx],
                        system.interact_same_bond_id,
                    )
                    acc = pair_fn(acc, idx, kj, pos, state, valid)
                return k + PAIR_UNROLL, acc

            _, final_acc = jax.lax.while_loop(cond_fun, body_fun, (start_idx, init_acc))
            return final_acc

        cell_results = jax.vmap(per_cell)(neighbor_hashes)
        return jax.tree.map(lambda x: x.sum(axis=0), cell_results)

    acc = jax.vmap(per_particle)(iota, p_neighbor_cell_hashes)
    return state, acc, hash_overflow


@Collider.register("CellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DynamicCellList(Collider):
    r"""Implicit cell-list (spatial hashing) collider using dynamic while-loops.

    This collider accelerates short-range pair interactions by partitioning the
    domain into a regular grid of cubic/square cells of side length ``cell_size``.
    Each particle is assigned to a cell, particles are sorted by cell hash, and
    interactions are evaluated only against particles in the same or neighboring
    cells given by ``neighbor_mask``.

    This implementation does not use a fixed ``max_occupancy`` array padding.
    Instead, it uses a dynamic ``jax.lax.while_loop`` to iterate over the exact number of particles present in each neighboring cell.

    The operation of this collider can be understood as the following nested loop:

    .. code-block:: python

        for particle in particles: # parallel
            for hash in stencil(particle): # parallel
                while next_neighbor in cell(hash): # sequential
                    ...

    Because the innermost loop is evaluated sequentially, the computational cost
    is driven by the *average* cell occupancy rather than the maximum possible
    occupancy. This makes the total theoretical cost:

    .. math::
        O(N \cdot \text{neighbor\_mask\_size} \cdot \langle K \rangle)

    where :math:`\langle K \rangle` is the average cell occupancy. To understand
    how this scales, let's analyze the cost components:

    * Stencil size:
        The stencil size depends on the ratio between the cell size (:math:`L`) and
        the radius of the largest particle (:math:`r_{max}`).

        .. math::
            \text{neighbor\_mask\_size} = \left( 2\left\lceil \frac{2r_{max}}{L} \right\rceil + 1 \right)^{dim}

    * Average occupancy:
        The average number of particles that occupy a cell depends on the cell
        volume and the macroscopic number density (:math:`\rho`):

        .. math::
            \langle K \rangle = \rho L^{dim}

    To express this in terms of the local volume fraction :math:`\phi` (the ratio of
    volume actually occupied by particles to the total cell volume) and our
    normalized cell size :math:`L^\prime = L/r_{max}`, we use the average particle
    volume :math:`\langle V \rangle`:

    .. math::
        \langle K \rangle = \phi \frac{L^{dim}}{\langle V \rangle} = \phi \frac{(L^\prime r_{max})^{dim}}{\langle V \rangle}

    Knowing that the volume of the largest particle is :math:`V_{max} = k_v r_{max}^{dim}`
    (where :math:`k_v` is the geometric volume factor, such as :math:`4\pi/3` in 3D or
    :math:`\pi` in 2D), we find the final theoretical cost:

    .. math::
        \text{cost} \approx N \left( 2\left\lceil \frac{2}{L^\prime} \right\rceil + 1 \right)^{dim} \left( \frac{\phi}{k_v} \frac{V_{max}}{\langle V \rangle} (L^\prime)^{dim} \right)

    * The Polydispersity Advantage:
        In the static cell list, cost scales with the ratio of the largest to smallest
        particle volume (:math:`V_{max}/V_{min} \propto \alpha^{dim}`, where
        :math:`\alpha = r_{max}/r_{min}`). In this dynamic list, the cost scales with
        the ratio of the largest to the *average* particle volume
        (:math:`V_{max}/\langle V \rangle`). Thus, the severe :math:`O(\alpha^{dim})` padding
        penalty is significantly reduced or offset.

    Constructor Parameters
    ----------------------
    - **cell_size**: Linear size of the grid cells. A larger cell size reduces neighbor
      stencil size but increases cell occupancy (longer sequential loops). A smaller cell
      size reduces occupancy but expands the stencil exponentially, which increases compilation
      overhead. If None, defaults to :math:`2 r_{max}` (for systems with low polydispersity
      :math:`\alpha < 2.5`), or :math:`0.5 r_{max}` (for highly polydisperse systems).
    - **search_range**: Neighborhood range in cell units. Dictates how many cells are searched
      along each dimension. If None, it is dynamically computed to guarantee that all potential
      contacts within :math:`2 r_{max}` are visited. Setting this higher expands the search stencil.
    - **box_size**: Bounding dimensions of the physical domain. This is only needed when the physical box size is small
      compared with the cell size (to ensure the minimum grid size requirement of `2 * search_range + 1` cells per axis
      is met under periodic boundary conditions).

    This collider is suitable for large systems with low to moderate polydispersity (:math:`\alpha < 2.5`) and medium to high packing fractions. Highly polydisperse systems (:math:`\alpha \ge 3.0`) or systems containing rigid clumps with large internal overlaps will reduce performance significantly. This is because overlaps artificially inflate the local cell occupancy :math:`\langle K \rangle` far beyond the macroscopic physical volume fraction :math:`\phi`, leading to longer sequential loops and reduced GPU thread efficiency.

    Complexity
    ----------
    - Time: :math:`O(N)` - :math:`O(N \log N)` from sorting, plus :math:`O(N \cdot M \cdot \langle K \rangle)`
      for neighbor probing (M = ``neighbor_mask_size``, :math:`\langle K \rangle` = average occupancy).
      The state is close to sorted every frame.
    - Memory: :math:`O(N)`.

    Notes
    -----
    - **Batching with ``vmap``**: If you use ``jax.vmap`` to evaluate multiple
      simulation environments simultaneously, be aware of JAX's SIMD execution model.
      Because the innermost ``while`` loop executes sequentially, the loop must continue
      running for *all* environments in the batch until the environment with the highest
      local cell occupancy finishes its iterations. Consequently, the computational cost
      of a batched execution is bottlenecked by the single worst-case occupancy across
      the entire batch.
    """

    neighbor_mask: jax.Array
    """Integer offsets defining the neighbor stencil (M, dim)."""

    cell_size: jax.Array
    """Linear size of a grid cell (scalar)."""

    @classmethod
    def Create(
        cls,
        state: State,
        cell_size: ArrayLike | None = None,
        search_range: ArrayLike | None = None,
        box_size: ArrayLike | None = None,
    ) -> Self:
        """Creates a DynamicCellList instance based on the reference state.

        Parameters
        ----------
        state : State
            Reference state containing positions and radii.
        cell_size : float, optional
            Grid cell size.
        search_range : int, optional
            Number of neighboring cells to search.
        box_size : ArrayLike, optional
            Bounding dimensions of physical box. Only needed when the box size is small compared with the cell size.

        Returns
        -------
        DynamicCellList
            A configured DynamicCellList instance.
        """
        min_rad = jnp.min(state._rad)
        max_rad = jnp.max(state._rad)
        alpha = max_rad / min_rad

        if cell_size is None:
            cell_size = jnp.where(alpha < 2.5, 2.0 * max_rad, 0.5 * max_rad)

        if box_size is not None:
            box_size = jnp.asarray(box_size, dtype=float)
            for _ in range(2):
                if search_range is None:
                    sr = jnp.ceil(2 * max_rad / cell_size).astype(int)
                    sr = jnp.maximum(1, sr)
                else:
                    sr = jnp.asarray(search_range, dtype=int)
                min_grids_per_axis = 2 * sr + 1
                grid_dims = jnp.floor(box_size / cell_size).astype(int)
                grid_dims = jnp.maximum(grid_dims, min_grids_per_axis)
                cell_size = jnp.min(box_size / grid_dims)

        if search_range is None:
            search_range = jnp.ceil(2 * max_rad / cell_size).astype(int)
            search_range = jnp.maximum(1, search_range)
        search_range = jnp.array(search_range, dtype=int)

        r = jnp.arange(-search_range, search_range + 1, dtype=int)
        mesh = jnp.meshgrid(*([r] * state.dim), indexing="ij")
        neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)

        return cls(
            neighbor_mask=neighbor_mask.astype(int),
            cell_size=jnp.asarray(cell_size, dtype=float),
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="DynamicCellList.compute_force")
    def compute_force(state: State, system: System) -> tuple[State, System]:
        """Computes pairwise contact forces and torques using DynamicCellList.

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
        collider = cast(DynamicCellList, system.collider)
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
    @partial(jax.named_call, name="DynamicCellList.compute_potential_energy")
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
        collider = cast(DynamicCellList, system.collider)
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
    def create_neighbor_list(
        state: State, system: System, cutoff: float, max_neighbors: int
    ) -> tuple[State, System, jax.Array, jax.Array]:
        """Creates a neighbor list of shape (N, max_neighbors) using DynamicCellList.

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

        collider = cast(DynamicCellList, system.collider)
        iota = jax.lax.iota(int, N)
        pos = state.pos

        # Inflate the cell size so the fixed stencil reach covers the
        # requested cutoff. The stencil spans ``search_range`` cells per axis,
        # so all pairs within ``search_range * cell_size`` are guaranteed to
        # be visited; for larger cutoffs we grow the cells accordingly
        # (mirroring DynamicMultiCellList's search-radius inflation).
        search_range = jnp.maximum(jnp.max(collider.neighbor_mask), 1)
        cell_size = jnp.maximum(collider.cell_size, cutoff / search_range)

        # 1. Spatial Partitioning
        (
            perm,
            p_cell_hash,
            p_neighbor_hashes,
            hash_overflow,
        ) = _get_spatial_partition(pos, system, cell_size, collider.neighbor_mask, iota)

        # Permute state to sorted order
        sorted_state = jax.tree.map(lambda x: x[perm], state)
        sorted_pos = sorted_state.pos

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
                    sorted_state.clump_id[k],
                    sorted_state.clump_id[idx],
                    sorted_state.bond_id[k],
                    sorted_state.unique_id[idx],
                    system.interact_same_bond_id,
                ) * (d_sq <= cutoff_sq)

            stencil_body = _make_stencil_body(
                p_cell_hash, N, local_capacity, candidate_valid
            )
            final_n_list, stencil_counts, stencil_overflows = jax.vmap(stencil_body)(
                stencil, cell_starts
            )
            return final_n_list, stencil_counts, stencil_overflows

        all_final_n_list, all_stencil_counts, all_stencil_overflows = jax.vmap(
            traverse
        )(iota, sorted_pos, p_neighbor_hashes)

        topk, count_overflow = _pack_stencil_lists(
            all_final_n_list, all_stencil_counts, max_neighbors
        )

        overflow_flag = jnp.any(all_stencil_overflows) | count_overflow | hash_overflow

        return sorted_state, system, topk, overflow_flag

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",), inline=True)
    @partial(jax.named_call, name="DynamicCellList.create_cross_neighbor_list")
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

        collider = cast(DynamicCellList, system.collider)

        # Inflate the cell size so the fixed stencil reach covers the
        # requested cutoff (see ``create_neighbor_list``).
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
        n_a = pos_a.shape[0]
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
