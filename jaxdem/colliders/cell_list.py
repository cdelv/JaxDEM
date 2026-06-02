# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Cell List :math:`O(N log N)` collider implementation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from ..utils.linalg import cross, norm2
from . import Collider, valid_interaction_mask

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
    """Computes spatial hashing and partitioning for the cell list."""
    if system.domain.periodic:
        grid_dims = jnp.floor(system.domain.box_size / cell_size).astype(int)
    else:
        grid_dims = jnp.ceil(system.domain.box_size / cell_size).astype(int)

    grid_strides = jnp.concatenate(
        [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
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

    return (
        perm,
        p_cell_coords,
        p_cell_hash,
        neighbor_cell_coords,
        neighbor_cell_hashes,
    )


@jax.jit
@partial(jax.named_call, name="cell_list._dedup_stencil_hashes")
def _dedup_stencil_hashes(
    stencil_hashes: jax.Array, has_duplicates: bool = True
) -> jax.Array:
    """Deduplicate one particle's stencil hashes, padding duplicates with -1."""

    def dedup_hashes() -> jax.Array:
        mask = jnp.triu(stencil_hashes[:, None] == stencil_hashes[None, :], k=1)
        is_duplicate = jnp.any(mask, axis=0)
        return jnp.where(is_duplicate, -1, stencil_hashes)

    return jax.lax.cond(
        has_duplicates,
        dedup_hashes,
        lambda: stencil_hashes,
    )


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
        min_rad = jnp.min(state.rad)
        max_rad = jnp.max(state.rad)
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
    @jax.jit
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
        iota = jax.lax.iota(dtype=int, size=state.N)
        pos_p = state._pos_p_rot
        pos = state.pos

        (
            perm,
            _,
            p_cell_hash,
            _,
            p_neighbor_cell_hashes,
        ) = _get_spatial_partition(
            pos, system, collider.cell_size, collider.neighbor_mask, iota
        )

        state = jax.tree.map(lambda x: x[perm], state)
        pos = state.pos
        pos_p = state._pos_p_rot

        if system.domain.periodic:
            grid_dims = jnp.floor(system.domain.box_size / collider.cell_size).astype(
                int
            )
            stencil_range = (
                collider.neighbor_mask.max(axis=0)
                - collider.neighbor_mask.min(axis=0)
                + 1
            )
            has_duplicates = jnp.any(grid_dims < stencil_range)
        else:
            has_duplicates = jnp.array(False)

        def per_particle(
            idx: jax.Array, pos_pi: jax.Array, neighbor_hashes: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            if system.domain.periodic:
                neighbor_hashes = _dedup_stencil_hashes(neighbor_hashes, has_duplicates)

            def per_cell(target_hash: jax.Array) -> tuple[jax.Array, jax.Array]:
                start_idx = jnp.searchsorted(
                    p_cell_hash, target_hash, side="left", method="scan_unrolled"
                )

                def cond_fun(
                    val: tuple[jax.Array, tuple[jax.Array, jax.Array]],
                ) -> bool:
                    k, _ = val
                    return cast(bool, (k < state.N) * (p_cell_hash[k] == target_hash))

                def body_fun(
                    val: tuple[jax.Array, tuple[jax.Array, jax.Array]],
                ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
                    k, acc = val
                    valid = valid_interaction_mask(
                        state.clump_id[k],
                        state.clump_id[idx],
                        state.bond_id[k],
                        state.unique_id[idx],
                    )
                    f, t = system.force_model.force(idx, k, pos, state, system)
                    mask = valid if valid.ndim == 0 else valid[:, None]
                    new_acc = (acc[0] + f * mask, acc[1] + t * mask)
                    return k + 1, new_acc

                init_val = (
                    jnp.zeros(state.dim, dtype=float),
                    jnp.zeros(1 if state.dim == 2 else 3, dtype=float),
                )
                _, final_acc = jax.lax.while_loop(
                    cond_fun, body_fun, (start_idx, init_val)
                )
                return final_acc

            cell_results = jax.vmap(per_cell)(neighbor_hashes)
            sum_f = cell_results[0].sum(axis=0)
            sum_t = cell_results[1].sum(axis=0) + cross(pos_pi, sum_f)
            return sum_f, sum_t

        state.force, state.torque = jax.vmap(per_particle)(
            iota, pos_p, p_neighbor_cell_hashes
        )
        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="DynamicCellList.compute_potential_energy")
    def compute_potential_energy(state: State, system: System) -> jax.Array:
        """Computes the total non-bonded potential energy of the system.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        jax.Array
            Scalar potential energy.
        """
        collider = cast(DynamicCellList, system.collider)
        iota = jax.lax.iota(dtype=int, size=state.N)
        pos = state.pos

        (
            perm,
            _,
            p_cell_hash,
            _,
            p_neighbor_cell_hashes,
        ) = _get_spatial_partition(
            pos, system, collider.cell_size, collider.neighbor_mask, iota
        )

        state = jax.tree.map(lambda x: x[perm], state)
        pos = state.pos

        if system.domain.periodic:
            grid_dims = jnp.floor(system.domain.box_size / collider.cell_size).astype(
                int
            )
            stencil_range = (
                collider.neighbor_mask.max(axis=0)
                - collider.neighbor_mask.min(axis=0)
                + 1
            )
            has_duplicates = jnp.any(grid_dims < stencil_range)
        else:
            has_duplicates = jnp.array(False)

        def per_particle(idx: jax.Array, neighbor_hashes: jax.Array) -> jax.Array:
            if system.domain.periodic:
                neighbor_hashes = _dedup_stencil_hashes(neighbor_hashes, has_duplicates)

            def per_cell(target_hash: jax.Array) -> jax.Array:
                start_idx = jnp.searchsorted(
                    p_cell_hash, target_hash, side="left", method="scan_unrolled"
                )

                def cond_fun(val: tuple[jax.Array, jax.Array]) -> bool:
                    k, _ = val
                    return cast(bool, (k < state.N) * (p_cell_hash[k] == target_hash))

                def body_fun(
                    val: tuple[jax.Array, jax.Array],
                ) -> tuple[jax.Array, jax.Array]:
                    k, acc = val
                    valid = valid_interaction_mask(
                        state.clump_id[k],
                        state.clump_id[idx],
                        state.bond_id[k],
                        state.unique_id[idx],
                    )
                    e = system.force_model.energy(idx, k, pos, state, system)
                    return k + 1, acc + 0.5 * e * valid

                _, final_acc = jax.lax.while_loop(
                    cond_fun, body_fun, (start_idx, jnp.array(0.0, dtype=float))
                )
                return final_acc

            cell_results = jax.vmap(per_cell)(neighbor_hashes)
            return cell_results.sum(axis=0)

        energy = jax.vmap(per_particle)(iota, p_neighbor_cell_hashes)
        return jnp.sum(energy)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
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

        # 1. Spatial Partitioning
        (
            perm,
            _,
            p_cell_hash,
            _,
            p_neighbor_hashes,
        ) = _get_spatial_partition(
            pos, system, collider.cell_size, collider.neighbor_mask, iota
        )

        # Permute state to sorted order
        sorted_state = jax.tree.map(lambda x: x[perm], state)
        sorted_pos = sorted_state.pos

        if system.domain.periodic:
            grid_dims = jnp.floor(system.domain.box_size / collider.cell_size).astype(
                int
            )
            stencil_range = (
                collider.neighbor_mask.max(axis=0)
                - collider.neighbor_mask.min(axis=0)
                + 1
            )
            has_duplicates = jnp.any(grid_dims < stencil_range)
        else:
            has_duplicates = jnp.array(False)

        local_capacity = max_neighbors

        def traverse(
            idx: jax.Array,
            pos_i: jax.Array,
            stencil: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            if system.domain.periodic:
                stencil = _dedup_stencil_hashes(stencil, has_duplicates)

            cell_starts = jnp.searchsorted(
                p_cell_hash, stencil, side="left", method="scan_unrolled"
            )

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
                    in_cell = (k < N) * (p_cell_hash[k] == target_cell_hash)
                    has_space = c < local_capacity + 1
                    return cast(bool, in_cell * has_space)

                def body_fun(
                    val: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    k, c, nl, overflow = val
                    dr = system.domain.displacement(pos_i, sorted_pos[k], system)
                    d_sq = norm2(dr)
                    valid = valid_interaction_mask(
                        sorted_state.clump_id[k],
                        sorted_state.clump_id[idx],
                        sorted_state.bond_id[k],
                        sorted_state.unique_id[idx],
                    ) * (d_sq <= cutoff_sq)
                    nl = jax.lax.cond(
                        valid,
                        lambda nl_: nl_.at[c].set(k, mode="drop"),
                        lambda nl_: nl_,
                        nl,
                    )
                    c_new = c + valid.astype(c.dtype)
                    return k + 1, c_new, nl, overflow + (c_new > local_capacity)

                _, local_c, local_nl, local_overflow = jax.lax.while_loop(
                    cond_fun, body_fun, init_carry
                )
                return local_nl, local_c, local_overflow

            final_n_list, stencil_counts, stencil_overflows = jax.vmap(stencil_body)(
                stencil, cell_starts
            )
            return final_n_list, stencil_counts, stencil_overflows

        all_final_n_list, all_stencil_counts, all_stencil_overflows = jax.vmap(
            traverse
        )(iota, sorted_pos, p_neighbor_hashes)

        # Vectorized prefix-sum packing
        row_offsets = jnp.cumsum(all_stencil_counts, axis=-1) - all_stencil_counts
        local_iota = jnp.arange(local_capacity)
        target_indices = row_offsets[:, :, None] + local_iota[None, None, :]
        valid_mask = local_iota[None, None, :] < all_stencil_counts[:, :, None]

        safe_indices = jnp.where(
            valid_mask.reshape(N, -1),
            target_indices.reshape(N, -1),
            max_neighbors,
        )

        topk = jnp.full((N, max_neighbors), -1, dtype=all_final_n_list.dtype)
        row_idx = jnp.arange(N)[:, None]
        topk = topk.at[row_idx, safe_indices].set(
            all_final_n_list.reshape(N, -1), mode="drop"
        )

        overflow_flag = jnp.any(all_stencil_overflows) | jnp.any(
            jnp.sum(all_stencil_counts, axis=-1) > max_neighbors
        )

        jax.lax.cond(
            overflow_flag,
            lambda: jax.debug.print(
                "WARNING: DynamicCellList neighbor list overflow detected (max_neighbors={max_neighbors} is too small). Some neighbors have been missed.",
                max_neighbors=max_neighbors,
            ),
            lambda: None,
        )

        return sorted_state, system, topk, overflow_flag

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
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

        # 1. Sort pos_b into cells
        iota_b = jax.lax.iota(int, n_b)
        (
            perm_b,
            _,
            p_cell_hash_b,
            _,
            p_neighbor_hashes_a,
        ) = _get_spatial_partition(
            pos_b, system, collider.cell_size, collider.neighbor_mask, iota_b
        )
        pos_b_sorted = pos_b[perm_b]

        # 2. Get query neighbor stencils
        n_a = pos_a.shape[0]
        iota_a = jax.lax.iota(int, n_a)
        (
            perm_a,
            _,
            _,
            _,
            p_neighbor_hashes_a,
        ) = _get_spatial_partition(
            pos_a, system, collider.cell_size, collider.neighbor_mask, iota_a
        )
        pos_a_sorted = pos_a[perm_a]

        if system.domain.periodic:
            grid_dims = jnp.floor(system.domain.box_size / collider.cell_size).astype(
                int
            )
            stencil_range = (
                collider.neighbor_mask.max(axis=0)
                - collider.neighbor_mask.min(axis=0)
                + 1
            )
            has_duplicates = jnp.any(grid_dims < stencil_range)
        else:
            has_duplicates = jnp.array(False)

        cutoff_sq = cutoff**2
        local_capacity = max_neighbors

        # 3. For each sorted-A point, find neighbors in sorted B
        def traverse(
            pos_ai: jax.Array,
            stencil: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            if system.domain.periodic:
                stencil = _dedup_stencil_hashes(stencil, has_duplicates)

            cell_starts = jnp.searchsorted(
                p_cell_hash_b, stencil, side="left", method="scan_unrolled"
            )

            def stencil_body(
                target_cell_hash: jax.Array, start_idx: jax.Array
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                init_carry = (
                    start_idx,
                    jnp.array(0, dtype=int),
                    jnp.full((local_capacity,), -1, dtype=int),
                    jnp.array(False),
                )

                def cond_fun(
                    val: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ) -> bool:
                    k, c, _, _ = val
                    safe_k = jnp.minimum(k, jnp.maximum(1, n_b) - 1)
                    in_cell = (k < n_b) * (p_cell_hash_b[safe_k] == target_cell_hash)
                    has_space = c < local_capacity + 1
                    return cast(bool, in_cell * has_space)

                def body_fun(
                    val: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    k, c, nl, overflow = val
                    safe_k = jnp.minimum(k, jnp.maximum(1, n_b) - 1)
                    dr = system.domain.displacement(
                        pos_ai, pos_b_sorted[safe_k], system
                    )
                    d_sq = norm2(dr)
                    valid = d_sq <= cutoff_sq
                    nl = jax.lax.cond(
                        valid,
                        lambda nl_: nl_.at[c].set(k, mode="drop"),
                        lambda nl_: nl_,
                        nl,
                    )
                    c_new = c + valid.astype(c.dtype)
                    return k + 1, c_new, nl, overflow + (c_new > local_capacity)

                _, local_c, local_nl, local_overflow = jax.lax.while_loop(
                    cond_fun, body_fun, init_carry
                )
                return local_nl, local_c, local_overflow

            final_n_list, stencil_counts, stencil_overflows = jax.vmap(stencil_body)(
                stencil, cell_starts
            )
            return final_n_list, stencil_counts, stencil_overflows

        all_final_n_list, all_stencil_counts, all_stencil_overflows = jax.vmap(
            traverse
        )(pos_a_sorted, p_neighbor_hashes_a)

        # Vectorized prefix-sum packing
        row_offsets = jnp.cumsum(all_stencil_counts, axis=-1) - all_stencil_counts
        local_iota = jnp.arange(local_capacity)
        target_indices = row_offsets[:, :, None] + local_iota[None, None, :]
        valid_mask = local_iota[None, None, :] < all_stencil_counts[:, :, None]

        safe_indices = jnp.where(
            valid_mask.reshape(n_a, -1),
            target_indices.reshape(n_a, -1),
            max_neighbors,
        )

        topk = jnp.full((n_a, max_neighbors), -1, dtype=all_final_n_list.dtype)
        row_idx = jnp.arange(n_a)[:, None]
        topk = topk.at[row_idx, safe_indices].set(
            all_final_n_list.reshape(n_a, -1), mode="drop"
        )

        # 4. Map sorted-B indices back to original B indices
        valid_mask_nl = topk != -1
        safe_indices_nl = jnp.where(valid_mask_nl, topk, 0)
        topk = jnp.where(valid_mask_nl, perm_b[safe_indices_nl], -1)

        # 5. Unsort from sorted-A order back to original A order
        inv_perm_a = jnp.empty_like(perm_a)
        inv_perm_a = inv_perm_a.at[perm_a].set(iota_a)
        topk = topk[inv_perm_a]

        overflow_flag = jnp.any(all_stencil_overflows) | jnp.any(
            jnp.sum(all_stencil_counts, axis=-1) > max_neighbors
        )

        jax.lax.cond(
            overflow_flag,
            lambda: jax.debug.print(
                "WARNING: DynamicCellList cross neighbor list overflow detected (max_neighbors={max_neighbors} is too small). Some neighbors have been missed.",
                max_neighbors=max_neighbors,
            ),
            lambda: None,
        )

        return topk, overflow_flag
