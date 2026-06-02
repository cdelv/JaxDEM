# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Multi-Cell List collider implementations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast, Any

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
        valid_cells: bool | jax.Array = True
    else:
        valid_cells = (cell_coords >= 0).all(axis=-1) * (
            cell_coords < grid_dims[None, :]
        ).all(axis=-1)
        # Prevent negative coordinate aliasing by clamping to grid boundaries
        cell_coords = jnp.clip(cell_coords, 0, grid_dims[None, :] - 1)

    # Spatial Hashing
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
        grid_dims = jnp.maximum(grid_dims, 1)
        cell_size = system.domain.box_size / grid_dims
    else:
        grid_dims = jnp.ceil(system.domain.box_size / cell_size).astype(int)

    grid_strides = jnp.concatenate(
        [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
    )

    aabb = 2.0 * rad_or_search_rad
    if aabb.ndim == 1:
        aabb = jnp.repeat(aabb[:, None], dim, axis=1)

    # Check for overflow/truncation of AABB hashes
    xmin = pos - aabb / 2.0
    xmax = pos + aabb / 2.0
    min_coords_raw = jnp.floor((xmin - system.domain.anchor) / cell_size).astype(int)
    max_coords_raw = jnp.floor((xmax - system.domain.anchor) / cell_size).astype(int)
    num_cells_dim = max_coords_raw - min_coords_raw + 1
    total_cells = jnp.prod(num_cells_dim, axis=-1)

    overflow = jnp.any(total_cells > max_hashes)
    jax.lax.cond(
        overflow,
        lambda: jax.debug.print(
            "WARNING: DynamicMultiCellList particle hash capacity overflow detected. Some particles overlapped with more than max_hashes={max_hashes} cells. Some collisions may have been missed. Please increase max_hashes.",
            max_hashes=max_hashes,
        ),
        lambda: None,
    )

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


@partial(jax.jit)
@partial(jax.named_call, name="multi_cell_list._compute_canonical_hash")
def _compute_canonical_hash(
    pos_i: jax.Array,
    pos_j: jax.Array,
    rad_i: jax.Array,
    rad_j: jax.Array,
    cell_size: jax.Array,
    system: System,
) -> jax.Array:
    """Computes a unique (canonical) cell hash for an interaction pair."""
    dr = system.domain.displacement(pos_i, pos_j, system)
    pos_j_unwrapped = pos_i - dr

    if system.domain.periodic:
        grid_dims = jnp.floor(system.domain.box_size / cell_size).astype(int)
        grid_dims = jnp.maximum(grid_dims, 1)
        cell_size = system.domain.box_size / grid_dims
        min_c_i = jnp.floor((pos_i - rad_i - system.domain.anchor) / cell_size).astype(
            int
        )
        min_c_j = jnp.floor(
            (pos_j_unwrapped - rad_j - system.domain.anchor) / cell_size
        ).astype(int)
        min_coords_int = jnp.maximum(min_c_i, min_c_j)
        M_cell = min_coords_int % grid_dims
    else:
        grid_dims = jnp.ceil(system.domain.box_size / cell_size).astype(int)
        min_c_i = jnp.floor((pos_i - rad_i - system.domain.anchor) / cell_size).astype(
            int
        )
        min_c_j = jnp.floor(
            (pos_j_unwrapped - rad_j - system.domain.anchor) / cell_size
        ).astype(int)
        min_coords_int = jnp.maximum(min_c_i, min_c_j)
        M_cell = jnp.clip(min_coords_int, 0, grid_dims - 1)

    grid_strides = jnp.concatenate(
        [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
    )
    return jnp.dot(M_cell, grid_strides)


@Collider.register("MultiCellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DynamicMultiCellList(Collider):
    r"""Implicit multi-cell list (spatial hashing) collider using dynamic while-loop traversal.

    This collider partitions the domain into a regular grid of cubic/square cells
    of side length ``cell_size``. Unlike standard cell lists where cell size is bounded
    by the largest particle diameter to ensure immediate neighbor stencil searching,
    the multi-cell list allows particles to span multiple cells. Each particle is
    registered in every grid cell that overlaps its axis-aligned bounding box (AABB),
    up to a maximum of ``max_hashes`` cells.

    This formulation decouples the grid cell size from the largest particle size,
    making it exceptionally well-suited for systems with extreme polydispersity or
    large aspect ratios.

    When particles span multiple cells, a pair of overlapping particles :math:`(i, j)`
    will be present in the same cell in multiple grid locations. To avoid evaluating
    their interaction multiple times, a canonical cell check is performed.
    Let :math:`\mathbf{x}_i` and :math:`\mathbf{x}_j` be the positions of two particles.
    An interaction is evaluated in cell :math:`c` if and only if:

    .. math::
        c = \text{canonical\_hash}(\mathbf{x}_i, \mathbf{x}_j)

    where the canonical hash is uniquely determined by the spatial coordinates of the
    interaction midpoint under the domain's boundary conditions.

    Runtime and Cost Analysis
    -------------------------
    Let :math:`L` be the cell size, :math:`r` represent particle radii, and :math:`dim`
    be the spatial dimension. The number of cells overlapped by a particle of radius
    :math:`r` is given by:

    .. math::
        M(r) \approx \left( \frac{2r}{L} + 1 \right)^{dim}

    The maximum number of cell hashes a particle can occupy is bounded by the largest
    particle:

    .. math::
        M_{max} \approx \left( \frac{2r_{max}}{L} + 1 \right)^{dim}

    We set the static padding parameter ``max_hashes`` :math:`\ge M_{max}`. The spatial
    partitioning step hashes and sorts :math:`N \cdot \text{max\_hashes}` cell-particle
    references, introducing a sorting cost of :math:`O(N \cdot \text{max\_hashes} \log(N \cdot \text{max\_hashes}))`.

    The average occupancy of a grid cell (the average number of particle references
    occupying a cell), denoted as :math:`\langle K \rangle`, is:

    .. math::
        \langle K \rangle = \frac{N \langle M \rangle}{N_{cells}} = \rho L^{dim} \langle M \rangle

    where :math:`\rho = N / V_{domain}` is the macroscopic number density and
    :math:`\langle M \rangle` is the average number of cells overlapped by a particle.
    Expressing :math:`\rho` in terms of the packing fraction :math:`\phi` and the average
    particle volume :math:`\langle V \rangle` (:math:`\rho = \phi / \langle V \rangle`):

    .. math::
        \langle K \rangle = \phi \frac{L^{dim}}{\langle V \rangle} \langle M \rangle

    Since each particle :math:`i` queries :math:`M(r_i)` cells during traversal, the expected
    number of pairwise checks per particle of radius :math:`r_i` is :math:`M(r_i) \langle K \rangle`.
    Averaged over all particles, the total contact detection cost scales as:

    .. math::
        \text{cost} \approx N \langle M \rangle \langle K \rangle = N \phi \frac{L^{dim}}{\langle V \rangle} \langle M \rangle^2

    * **Optimal Cell Size**:
      There is a clear trade-off in selecting the cell size :math:`L`:
      - As :math:`L \to 0`, the number of overlapped cells :math:`\langle M \rangle \propto L^{-dim}`
        explodes, increasing sorting complexity and the number of cells each particle queries.
      - As :math:`L \to \infty`, the cell occupancy :math:`\langle K \rangle` increases, leading
        to larger sequential dynamic loops.
      The optimal cell size is typically chosen to be comparable to the median particle
      diameter (e.g. :math:`L \approx 2 r_{median}`), which balances the two costs.

    * **The Polydispersity Advantage**:
      In a standard cell list, a single giant particle forces the cell size :math:`L \ge 2r_{max}`.
      In highly polydisperse systems where :math:`\alpha = r_{max}/r_{min} \gg 1`, this results in
      massive cells relative to the tiny particles, leading to extremely high cell occupancies and
      redundant distance checks.
      By contrast, ``DynamicMultiCellList`` allows :math:`L` to remain small (scaled to :math:`r_{median}`).
      Small particles occupy only :math:`1` or :math:`2^{dim}` cells, while large particles occupy
      many cells. This avoids the stencil explosion for small particles, keeping the average
      occupancy low and significantly outperforming standard cell lists at high polydispersity.
      However, due to the descrete nature of the grid, we dont pay the penalty of increasing polidispersity
      until r//L exceeds the cell size. Meaning that two systems with different polydispersity
      (but the same cell size) could have the same performance.

    Constructor Parameters
    ----------------------
    - **cell_size**: Linear size of the grid cells. Smaller cell sizes allow finer-grained partitioning
      (lower cell occupancies) but cause large particles to overlap more cells (inflating the sorting array).
      A larger cell size reduces the cells overlapped by large particles but increases cell occupancies.
      If None, it defaults to the median particle diameter :math:`2 r_{median}`.
    - **max_hashes**: Static padding parameter representing the maximum number of grid cells a single particle
      is allowed to overlap. Must be set high enough to cover the largest particle's overlap capacity.
      If None, it defaults to the geometric cell overlap limit: :math:`\text{max\_hashes} = (\lceil 2r_{max}/L \rceil + 1)^{dim}`.
      Setting this higher increases compilation time and memory allocation, while setting it too small results in
      clipping of the AABB, causing missed interactions.

    This collider is suitable for systems with high polydispersity (:math:`\alpha \ge 3.0`).
    It allows the cell size to remain small without suffering from stencil explosions. While clumps with large internal overlaps inflate insertion counts,
    the canonical midpoint deduplication ensures that each pair interaction is evaluated exactly once.
    However, the high density of overlapping constituent spheres still increases local cell occupancies :math:`\langle K \rangle`.
    It is less suitable for monodisperse systems with few overlaps where standard cell lists are faster.

    Complexity
    ----------
    - Time: :math:`O(N \cdot M_{max} \log(N \cdot M_{max}))` sorting overhead, plus
      :math:`O(N \cdot \langle M \rangle \cdot \langle K \rangle)` traversal.
    - Memory: :math:`O(N \cdot M_{max})` to store the padded spatial hashing table.
    """

    cell_size: jax.Array
    """
    Linear size of a grid cell (scalar).
    """

    max_hashes: int = jax.tree.static()
    """
    Static padding parameter representing the maximum number of grid cells a particle
    is allowed to overlap.
    """

    @classmethod
    def Create(
        cls,
        state: State,
        cell_size: ArrayLike | None = None,
        max_hashes: int | None = None,
    ) -> Self:
        """Creates a DynamicMultiCellList instance based on the reference state.

        Parameters
        ----------
        state : State
            Reference state containing positions and radii.
        cell_size : float, optional
            Grid cell size.
        max_hashes : int, optional
            Maximum cells a single particle can overlap.

        Returns
        -------
        DynamicMultiCellList
            A configured DynamicMultiCellList instance.
        """
        if cell_size is None:
            rad = jnp.median(state.rad)
            cell_size = 2.0 * rad
        cell_size = jnp.asarray(cell_size, dtype=float)

        if max_hashes is None:
            max_rad = jnp.max(state.rad)
            S = jnp.ceil(2 * max_rad / cell_size).astype(int) + 1
            max_hashes = int(S**state.dim)

        return cls(
            cell_size=jnp.asarray(cell_size, dtype=float),
            max_hashes=int(max_hashes),
        )

    @staticmethod
    @jax.jit
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

        pos_p = state._pos_p_rot
        pos = state.pos

        sorted_hashes, perm, _ = _get_multi_cell_partition(
            pos,
            state.rad,
            system,
            collider.cell_size,
            collider.max_hashes,
        )

        total_elements = sorted_hashes.shape[0]

        def per_hash(
            target_hash: jax.Array, idx: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            start_idx = jnp.searchsorted(sorted_hashes, target_hash, side="left")

            def cond_fun(val: tuple[jax.Array, tuple[jax.Array, jax.Array]]) -> bool:
                flat_idx, _ = val
                in_bounds = flat_idx < total_elements
                safe_idx = jnp.minimum(flat_idx, total_elements - 1)
                matches_hash = sorted_hashes[safe_idx] == target_hash
                return cast(bool, in_bounds * matches_hash * (target_hash != -1))

            def body_fun(
                val: tuple[jax.Array, tuple[jax.Array, jax.Array]],
            ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
                flat_idx, acc = val
                j = perm[flat_idx]

                canonical_hash = _compute_canonical_hash(
                    pos[idx],
                    pos[j],
                    state.rad[idx],
                    state.rad[j],
                    collider.cell_size,
                    system,
                )

                valid = (
                    (target_hash == canonical_hash)
                    * valid_interaction_mask(
                        state.clump_id[j],
                        state.clump_id[idx],
                        state.bond_id[j],
                        state.unique_id[idx],
                    )
                    * (idx != j)
                )

                f, t = system.force_model.force(idx, j, pos, state, system)
                mask = valid if valid.ndim == 0 else valid[:, None]
                new_acc = (acc[0] + f * mask, acc[1] + t * mask)
                return flat_idx + 1, new_acc

            zero_f = (
                jnp.zeros(state.dim, dtype=float),
                jnp.zeros(1 if state.dim == 2 else 3, dtype=float),
            )
            _, final_acc = jax.lax.while_loop(cond_fun, body_fun, (start_idx, zero_f))
            return final_acc

        flat_results = jax.vmap(per_hash)(sorted_hashes, perm)

        # Use scatter_add to accumulate
        sum_f = jnp.zeros((state.N, state.dim), dtype=float)
        sum_t = jnp.zeros((state.N, 1 if state.dim == 2 else 3), dtype=float)
        sum_f = sum_f.at[perm].add(flat_results[0])
        sum_t = sum_t.at[perm].add(flat_results[1])

        sum_t += cross(pos_p, sum_f)

        state.force = sum_f
        state.torque = sum_t

        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="DynamicMultiCellList.compute_potential_energy")
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
        collider = cast(DynamicMultiCellList, system.collider)

        pos = state.pos

        sorted_hashes, perm, _ = _get_multi_cell_partition(
            pos,
            state.rad,
            system,
            collider.cell_size,
            collider.max_hashes,
        )

        total_elements = sorted_hashes.shape[0]

        def per_hash(target_hash: jax.Array, idx: jax.Array) -> jax.Array:
            start_idx = jnp.searchsorted(sorted_hashes, target_hash, side="left")

            def cond_fun(val: tuple[jax.Array, jax.Array]) -> bool:
                flat_idx, _ = val
                in_bounds = flat_idx < total_elements
                safe_idx = jnp.minimum(flat_idx, total_elements - 1)
                matches_hash = sorted_hashes[safe_idx] == target_hash
                return cast(bool, in_bounds * matches_hash * (target_hash != -1))

            def body_fun(
                val: tuple[jax.Array, jax.Array],
            ) -> tuple[jax.Array, jax.Array]:
                flat_idx, acc = val
                j = perm[flat_idx]

                canonical_hash = _compute_canonical_hash(
                    pos[idx],
                    pos[j],
                    state.rad[idx],
                    state.rad[j],
                    collider.cell_size,
                    system,
                )

                valid = (
                    (target_hash == canonical_hash)
                    * valid_interaction_mask(
                        state.clump_id[j],
                        state.clump_id[idx],
                        state.bond_id[j],
                        state.unique_id[idx],
                    )
                    * (idx != j)
                )

                e = system.force_model.energy(idx, j, pos, state, system)
                return flat_idx + 1, acc + 0.5 * e * valid

            _, final_acc = jax.lax.while_loop(
                cond_fun, body_fun, (start_idx, jnp.array(0.0, dtype=float))
            )
            return final_acc

        flat_results = jax.vmap(per_hash)(sorted_hashes, perm)

        # Use scatter_add to accumulate potential energy per particle
        energy_per_particle = jnp.zeros(state.N, dtype=float)
        energy_per_particle = energy_per_particle.at[perm].add(flat_results)

        return jnp.sum(energy_per_particle)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
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
            Unmodified state, system, neighbor list, and overflow flag.
        """
        if max_neighbors == 0:
            empty = jnp.empty((state.N, 0), dtype=int)
            return state, system, empty, jnp.asarray(False)

        collider = cast(DynamicMultiCellList, system.collider)
        cutoff_sq = cutoff**2
        pos = state.pos
        search_rad = jnp.maximum(state.rad, cutoff / 2.0)

        sorted_hashes, perm, hashes = _get_multi_cell_partition(
            pos,
            search_rad,
            system,
            collider.cell_size,
            collider.max_hashes,
        )

        total_elements = sorted_hashes.shape[0]
        n = state.N
        max_hashes = collider.max_hashes

        # Compute cell starts globally to avoid searchsorted inside vmap
        all_cell_starts = jnp.searchsorted(sorted_hashes, hashes, side="left")

        # Flattened execution to reduce compilation graphs drastically
        flat_hashes = hashes.ravel()
        flat_cell_starts = all_cell_starts.ravel()
        flat_iota = jnp.repeat(jnp.arange(n), max_hashes)
        flat_pos = jnp.repeat(pos, max_hashes, axis=0)

        def flat_stencil_body(
            idx: jax.Array,
            pos_i: jax.Array,
            target_cell_hash: jax.Array,
            start_idx: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            local_capacity = max_neighbors
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
                safe_idx = jnp.minimum(flat_idx, jnp.maximum(1, total_elements) - 1)
                matches_hash = sorted_hashes[safe_idx] == target_cell_hash
                has_space = c < local_capacity + 1
                return cast(
                    bool,
                    in_bounds * matches_hash * has_space * (target_cell_hash != -1),
                )

            def body_fun(
                val: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                flat_idx, c, nl, overflow = val
                safe_idx = jnp.minimum(flat_idx, jnp.maximum(1, total_elements) - 1)
                j_idx = perm[safe_idx]

                dr = system.domain.displacement(pos_i, pos[j_idx], system)
                dist_sq = norm2(dr)

                canonical_hash = _compute_canonical_hash(
                    pos_i,
                    pos[j_idx],
                    search_rad[idx],
                    search_rad[j_idx],
                    collider.cell_size,
                    system,
                )

                valid = (
                    (target_cell_hash == canonical_hash)
                    * valid_interaction_mask(
                        state.clump_id[j_idx],
                        state.clump_id[idx],
                        state.bond_id[j_idx],
                        state.unique_id[idx],
                    )
                    * (dist_sq <= cutoff_sq)
                    * (idx != j_idx)
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

        final_n_list, stencil_counts, stencil_overflows = jax.vmap(flat_stencil_body)(
            flat_iota, flat_pos, flat_hashes, flat_cell_starts
        )

        all_final_n_list = final_n_list.reshape(n, max_hashes, -1)
        all_stencil_counts = stencil_counts.reshape(n, max_hashes)
        all_stencil_overflows = stencil_overflows.reshape(n, max_hashes)

        # Global prefix-sum packing
        local_capacity = max_neighbors
        row_offsets = jnp.cumsum(all_stencil_counts, axis=-1) - all_stencil_counts
        local_iota = jnp.arange(local_capacity)
        target_indices = row_offsets[:, :, None] + local_iota[None, None, :]
        valid_mask = local_iota[None, None, :] < all_stencil_counts[:, :, None]

        safe_indices = jnp.where(
            valid_mask.reshape(state.N, -1),
            target_indices.reshape(state.N, -1),
            max_neighbors,
        )

        neighbor_list = jnp.full(
            (state.N, max_neighbors), -1, dtype=all_final_n_list.dtype
        )
        row_idx = jnp.arange(state.N)[:, None]
        neighbor_list = neighbor_list.at[row_idx, safe_indices].set(
            all_final_n_list.reshape(state.N, -1), mode="drop"
        )

        overflow_flag = jnp.any(all_stencil_overflows) | jnp.any(
            jnp.sum(all_stencil_counts, axis=-1) > max_neighbors
        )
        jax.lax.cond(
            overflow_flag,
            lambda: jax.debug.print(
                "WARNING: DynamicMultiCellList neighbor list overflow detected (max_neighbors={max_neighbors} is too small). Some neighbors have been missed.",
                max_neighbors=max_neighbors,
            ),
            lambda: None,
        )

        return state, system, neighbor_list, overflow_flag

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
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
        cutoff_sq = cutoff**2
        max_hashes = collider.max_hashes

        # 1. Sort pos_b into cells
        rad_b = jnp.full(n_b, cutoff / 2.0)
        sorted_hashes_b, perm_b, _ = _get_multi_cell_partition(
            pos_b,
            rad_b,
            system,
            collider.cell_size,
            collider.max_hashes,
        )
        pos_b_sorted = pos_b[perm_b]

        # 2. Get cell coordinates for query points in pos_a
        rad_a = jnp.full(n_a, cutoff / 2.0)
        _, _, hashes_a = _get_multi_cell_partition(
            pos_a,
            rad_a,
            system,
            collider.cell_size,
            collider.max_hashes,
        )

        total_elements = sorted_hashes_b.shape[0]

        # Compute cell starts globally to avoid searchsorted inside vmap
        all_cell_starts = jnp.searchsorted(sorted_hashes_b, hashes_a, side="left")

        # Flattened execution
        flat_hashes_a = hashes_a.ravel()
        flat_cell_starts = all_cell_starts.ravel()
        flat_pos_a = jnp.repeat(pos_a, max_hashes, axis=0)

        def flat_stencil_body(
            pos_ai: jax.Array,
            target_cell_hash: jax.Array,
            start_idx: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            local_capacity = max_neighbors
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
                safe_idx = jnp.minimum(flat_idx, jnp.maximum(1, total_elements) - 1)
                matches_hash = sorted_hashes_b[safe_idx] == target_cell_hash
                has_space = c < local_capacity + 1
                return cast(
                    bool,
                    in_bounds * matches_hash * has_space * (target_cell_hash != -1),
                )

            def body_fun(
                val: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                flat_idx, c, nl, overflow = val
                safe_idx = jnp.minimum(flat_idx, jnp.maximum(1, total_elements) - 1)
                j_idx = perm_b[safe_idx]

                dr = system.domain.displacement(pos_ai, pos_b_sorted[safe_idx], system)
                dist_sq = norm2(dr)

                canonical_hash = _compute_canonical_hash(
                    pos_ai,
                    pos_b_sorted[safe_idx],
                    cutoff / 2.0,
                    cutoff / 2.0,
                    collider.cell_size,
                    system,
                )

                valid = (target_cell_hash == canonical_hash) * (dist_sq <= cutoff_sq)

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

        final_n_list, stencil_counts, stencil_overflows = jax.vmap(flat_stencil_body)(
            flat_pos_a, flat_hashes_a, flat_cell_starts
        )

        all_final_n_list = final_n_list.reshape(n_a, max_hashes, -1)
        all_stencil_counts = stencil_counts.reshape(n_a, max_hashes)
        all_stencil_overflows = stencil_overflows.reshape(n_a, max_hashes)

        # Global prefix-sum packing
        local_capacity = max_neighbors
        row_offsets = jnp.cumsum(all_stencil_counts, axis=-1) - all_stencil_counts
        local_iota = jnp.arange(local_capacity)
        target_indices = row_offsets[:, :, None] + local_iota[None, None, :]
        valid_mask = local_iota[None, None, :] < all_stencil_counts[:, :, None]

        safe_indices = jnp.where(
            valid_mask.reshape(n_a, -1),
            target_indices.reshape(n_a, -1),
            max_neighbors,
        )

        neighbor_list = jnp.full((n_a, max_neighbors), -1, dtype=all_final_n_list.dtype)
        row_idx = jnp.arange(n_a)[:, None]
        neighbor_list = neighbor_list.at[row_idx, safe_indices].set(
            all_final_n_list.reshape(n_a, -1), mode="drop"
        )

        overflow_flag = jnp.any(all_stencil_overflows) | jnp.any(
            jnp.sum(all_stencil_counts, axis=-1) > max_neighbors
        )
        jax.lax.cond(
            overflow_flag,
            lambda: jax.debug.print(
                "WARNING: DynamicMultiCellList cross neighbor list overflow detected (max_neighbors={max_neighbors} is too small). Some neighbors have been missed.",
                max_neighbors=max_neighbors,
            ),
            lambda: None,
        )

        return neighbor_list, overflow_flag

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "cell_size": float(self.cell_size),
            "max_hashes": int(self.max_hashes),
        }

