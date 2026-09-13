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
from ..domains import SearchGeometry
from . import Collider, valid_interaction_mask
from ._partition import (
    NEIGHBOR_QUERY_BATCH_SIZE,
    PAIR_TRAVERSAL_BATCH_SIZE,
    _cell_starts,
    _energy_pair_fn,
    _force_pair_fn,
    _grid_params,
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
    """Compute the spatial hashing and partitioning of the cell list.

    Returns
    -------
    tuple[jax.Array, ...]
        ``(perm, p_cell_hash, neighbor_cell_hashes, hash_overflow)``.
    """
    grid_dims, grid_strides, cell_size, hash_overflow = _grid_params(
        system.domain.box_size, cell_size, system.domain.periodic
    )

    hash_dtype = grid_strides.dtype
    hash_sentinel = jnp.bitwise_not(jnp.asarray(0, hash_dtype))
    pos = system.domain._shift(pos, system)
    if system.domain.periodic:
        p_cell_coords = jnp.floor(
            (((pos - system.domain.anchor) / system.domain.box_size) % 1) * grid_dims
        ).astype(hash_dtype)
    else:
        p_cell_coords = jnp.floor((pos - system.domain.anchor) / cell_size).astype(
            hash_dtype
        )

    p_cell_hash = jnp.dot(p_cell_coords, grid_strides)

    p_cell_hash, perm = jax.lax.sort([p_cell_hash, iota], num_keys=1)

    # Note: we compute neighbor coords using original (unsorted) p_cell_coords
    base_coords = p_cell_coords[:, None, :]
    offsets = neighbor_mask[None, :, :]
    nonnegative = offsets >= 0
    offset_magnitude = jnp.where(nonnegative, offsets, -offsets).astype(hash_dtype)

    def wrapped_offset_coords() -> tuple[jax.Array, jax.Array]:
        """Unsigned modular coordinates plus signed periodic image counts."""
        dims = grid_dims[None, None, :]
        quotient = offset_magnitude // dims
        remainder = offset_magnitude % dims
        positive_carry = base_coords >= (dims - remainder)
        positive_coords = jnp.where(
            (remainder == 0) | ~positive_carry,
            base_coords + remainder,
            base_coords - (dims - remainder),
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
        return coords, image

    if system.domain.periodic:
        neighbor_cell_coords, periodic_image = wrapped_offset_coords()

        # A gradient-boundary image in Lees-Edwards shifts the queried image
        # along the flow direction. Two adjacent alpha cells cover fractional
        # shifts conservatively; duplicate cells are removed during traversal.
        shear = system.domain.shear_search_parameters()
        if shear is not None:
            gamma, alpha, beta = shear
            beta_image = periodic_image[..., beta]
            alpha_shift = (
                -beta_image.astype(cell_size.dtype)
                * gamma
                * system.domain.box_size[beta]
                / cell_size[alpha]
            )
            lo = jnp.floor(alpha_shift).astype(offsets.dtype)
            hi = jnp.ceil(alpha_shift).astype(offsets.dtype)
            alpha_offsets = jnp.stack((lo, hi), axis=-1).reshape(
                lo.shape[0], 2 * lo.shape[1]
            )
            expanded = jnp.repeat(neighbor_cell_coords, 2, axis=1)
            alpha_base = expanded[..., alpha]
            alpha_nonnegative = alpha_offsets >= 0
            alpha_mag = jnp.where(
                alpha_nonnegative, alpha_offsets, -alpha_offsets
            ).astype(hash_dtype)
            alpha_dim = grid_dims[alpha]
            alpha_rem = alpha_mag % alpha_dim
            alpha_pos = jnp.where(
                (alpha_rem == 0) | (alpha_base < alpha_dim - alpha_rem),
                alpha_base + alpha_rem,
                alpha_base - (alpha_dim - alpha_rem),
            )
            alpha_neg = jnp.where(
                alpha_rem > alpha_base,
                alpha_dim - (alpha_rem - alpha_base),
                alpha_base - alpha_rem,
            )
            neighbor_cell_coords = expanded.at[..., alpha].set(
                jnp.where(alpha_nonnegative, alpha_pos, alpha_neg)
            )
        neighbor_cell_hashes = jnp.dot(
            neighbor_cell_coords.astype(hash_dtype), grid_strides
        )
    else:
        positive_oob = offset_magnitude >= grid_dims - base_coords
        negative_oob = offset_magnitude > base_coords
        out_of_bounds = jnp.any(
            jnp.where(nonnegative, positive_oob, negative_oob), axis=-1
        )
        neighbor_cell_coords = jnp.where(
            nonnegative, base_coords + offset_magnitude, base_coords - offset_magnitude
        )
        neighbor_cell_hashes = jnp.dot(neighbor_cell_coords, grid_strides)
        neighbor_cell_hashes = jnp.where(
            out_of_bounds, hash_sentinel, neighbor_cell_hashes
        )

    return (
        perm,
        p_cell_hash,
        neighbor_cell_hashes,
        hash_overflow,
    )


@jax.jit(inline=True)
@partial(jax.named_call, name="cell_list._dedup_stencil_hashes")
def _dedup_stencil_hashes(stencil_hashes: jax.Array) -> jax.Array:
    """Deduplicate one particle's stencil hashes using the unsigned sentinel."""
    mask = jnp.triu(stencil_hashes[:, None] == stencil_hashes[None, :], k=1)
    is_duplicate = jnp.any(mask, axis=0)
    sentinel = jnp.bitwise_not(jnp.asarray(0, stencil_hashes.dtype))
    return jnp.where(is_duplicate, sentinel, stencil_hashes)


def _make_direct_row_body(
    sorted_hashes: jax.Array,
    n_db: int | jax.Array,
    local_capacity: int,
    candidate_valid: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Build one neighbor row directly while walking its stencil in order.

    The returned function carries one ``local_capacity``-wide row across all
    stencil cells. This avoids materializing per-cell rows of shape
    ``(stencil_size, local_capacity)`` before packing.

    Parameters
    ----------
    sorted_hashes : jax.Array
        Cell hashes of the database points, sorted ascending.
    n_db : int | jax.Array
        Number of database points.
    local_capacity : int
        Static size of the output neighbor row.
    candidate_valid : Callable[[jax.Array], jax.Array]
        Boolean predicate evaluated on each candidate database index.
    """

    @jax.jit(inline=True)
    def row_body(
        stencil: jax.Array, cell_starts: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        j_arr = jax.lax.iota(dtype=int, size=PAIR_UNROLL)
        init_carry = (
            jnp.array(0, dtype=int),
            cell_starts[0],
            jnp.array(0, dtype=int),
            jnp.full((local_capacity,), -1, dtype=int),
            jnp.array(False),  # overflow flag
        )

        def cond_fun(
            val: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> bool:
            stencil_idx, _, c, _, _ = val
            in_stencil = stencil_idx < stencil.shape[0]
            has_space = c < local_capacity + 1
            return cast(bool, in_stencil * has_space)

        def body_fun(
            val: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            stencil_idx, k, c, nl, overflow = val
            target_cell_hash = stencil[stencil_idx]
            actual_k_arr = k + j_arr
            safe_k_arr = jnp.minimum(actual_k_arr, jnp.maximum(1, n_db) - 1)
            in_cell_arr = (actual_k_arr < n_db) * (
                sorted_hashes[safe_k_arr] == target_cell_hash
            )
            valid_arr = jax.vmap(candidate_valid)(safe_k_arr) * in_cell_arr

            valid_counts = valid_arr.astype(c.dtype)
            num_valid = jnp.sum(valid_counts)

            cumsum = jnp.cumsum(valid_counts) - valid_counts
            write_idx_arr = c + cumsum
            write_idx_arr = jnp.where(valid_arr, write_idx_arr, local_capacity)

            nl = nl.at[write_idx_arr].set(safe_k_arr, mode="drop")
            c = c + num_valid
            overflow = overflow | (c > local_capacity)

            next_k = k + PAIR_UNROLL
            safe_next_k = jnp.minimum(next_k, jnp.maximum(1, n_db) - 1)
            remains_in_cell = (next_k < n_db) & (
                sorted_hashes[safe_next_k] == target_cell_hash
            )
            next_stencil_idx = jnp.where(remains_in_cell, stencil_idx, stencil_idx + 1)
            safe_stencil_idx = jnp.minimum(next_stencil_idx, stencil.shape[0] - 1)
            next_k = jnp.where(remains_in_cell, next_k, cell_starts[safe_stencil_idx])
            return next_stencil_idx, next_k, c, nl, overflow

        _, _, _, neighbor_row, row_overflow = jax.lax.while_loop(
            cond_fun, body_fun, init_carry
        )
        return neighbor_row, row_overflow

    return row_body


#: Number of candidate particles each while-loop iteration of the pair
#: traversals visits. A vmapped ``lax.while_loop`` runs until the longest
#: lane finishes. Visiting several candidates reduces loop iterations at
#: the price of at most ``PAIR_UNROLL - 1`` extra masked pair evaluations
#: per lane.
PAIR_UNROLL = 4


@jax.jit(inline=True, static_argnames=("pair_fn",))
def _traverse_pairs(
    state: State,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    pair_fn: Callable[..., Any],
    init_acc: Any,
) -> tuple[Any, jax.Array]:
    """Fold a per-pair kernel over all candidate pairs of the cell partition.

    The traversal sorts the cell hashes. For every particle, it then walks
    the occupied stencil cells and accumulates ``pair_fn`` over the particles
    they contain (``valid`` already carries the clump/bond interaction mask).
    This single traversal backs both :meth:`DynamicCellList.compute_force`
    and :meth:`DynamicCellList.compute_potential_energy`.

    Returns
    -------
    tuple[Any, jax.Array]
        The per-particle accumulator pytree
        (each leaf has a leading ``N`` axis), and the ``hash_overflow`` flag
        of the partition.
    """
    N = state.N
    if N == 0:
        empty = jax.tree.map(lambda x: jnp.empty((0,) + x.shape, x.dtype), init_acc)
        return empty, jnp.asarray(False)
    pos = state.pos
    j_arr = jax.lax.iota(dtype=int, size=PAIR_UNROLL)
    iota = jax.lax.iota(dtype=int, size=state.N)
    (
        perm,
        p_cell_hash,
        p_neighbor_cell_hashes,
        hash_overflow,
    ) = _get_spatial_partition(pos, system, cell_size, neighbor_mask, iota)
    p_neighbor_cell_starts = _cell_starts(p_cell_hash, p_neighbor_cell_hashes)

    def per_particle(
        orig_idx: jax.Array,
        neighbor_hashes: jax.Array,
        neighbor_starts: jax.Array,
    ) -> Any:
        if system.domain.periodic:
            neighbor_hashes = _dedup_stencil_hashes(neighbor_hashes)

        def per_cell(target_hash: jax.Array, start_idx: jax.Array) -> Any:
            def cond_fun(val: tuple[jax.Array, Any]) -> bool:
                k, _ = val
                return cast(bool, (k < N) * (p_cell_hash[k] == target_hash))

            def body_fun(val: tuple[jax.Array, Any]) -> tuple[jax.Array, Any]:
                k, acc = val
                kj = jnp.minimum(k + j_arr, N - 1)
                in_cell = ((k + j_arr) < N) * (p_cell_hash[kj] == target_hash)

                valid = in_cell * valid_interaction_mask(
                    state.clump_id[perm[kj]],
                    state.clump_id[orig_idx],
                    state.bond_id[perm[kj]],
                    orig_idx,
                    system.interact_same_bond_id,
                )

                zero_acc = jax.tree.map(jnp.zeros_like, acc)
                vec_acc = pair_fn(zero_acc, orig_idx, perm[kj], pos, state, valid)
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


@Collider.register("CellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DynamicCellList(Collider):
    r"""Implicit cell-list (spatial hashing) collider using dynamic while-loops.

    This collider accelerates short-range pair interactions by partitioning the
    domain into a regular grid of cubic/square cells of side length ``cell_size``.
    It assigns each particle to a cell and permutes the particles internally by
    cell hash. It evaluates interactions only against particles in the same cell
    or in the neighboring cells given by ``neighbor_mask``.

    This implementation does not use a fixed ``max_occupancy`` array padding.
    Instead, it uses a dynamic ``jax.lax.while_loop`` to iterate over the exact number of particles present in each neighboring cell.

    The collider runs the following nested loop:

    .. code-block:: python

        for particle in particles: # parallel
            for hash in stencil(particle): # parallel
                while next_neighbor in cell(hash): # sequential
                    ...

    Because the collider evaluates the innermost loop sequentially, the
    *average* cell occupancy drives the computational cost, not the maximum
    possible occupancy. This gives the total theoretical cost:

    .. math::
        O(N \cdot \text{neighbor\_mask\_size} \cdot \langle K \rangle)

    where :math:`\langle K \rangle` is the average cell occupancy. The cost
    has two components:

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

    The volume of the largest particle is :math:`V_{max} = k_v r_{max}^{dim}`,
    where :math:`k_v` is the geometric volume factor (such as :math:`4\pi/3`
    in 3D or :math:`\pi` in 2D). This gives the final theoretical cost:

    .. math::
        \text{cost} \approx N \left( 2\left\lceil \frac{2}{L^\prime} \right\rceil + 1 \right)^{dim} \left( \frac{\phi}{k_v} \frac{V_{max}}{\langle V \rangle} (L^\prime)^{dim} \right)

    * The Polydispersity Advantage:
        In the static cell list, cost scales with the ratio of the largest to smallest
        particle volume (:math:`V_{max}/V_{min} \propto \alpha^{dim}`, where
        :math:`\alpha = r_{max}/r_{min}`). In this dynamic list, the cost scales with
        the ratio of the largest to the *average* particle volume
        (:math:`V_{max}/\langle V \rangle`). This dynamic list therefore reduces
        or offsets the severe :math:`O(\alpha^{dim})` padding penalty.

    Constructor Parameters
    ----------------------
    - **cell_size**: Linear size of the grid cells. A larger cell size reduces neighbor
      stencil size but increases cell occupancy (longer sequential loops). A smaller cell
      size reduces occupancy but expands the stencil exponentially, which increases compilation
      overhead. If None, defaults to :math:`2 r_{max}` (for systems with low polydispersity
      :math:`\alpha < 2.5`), or :math:`0.5 r_{max}` (for highly polydisperse systems).
    - **search_range**: Neighborhood range in cell units. Sets how many cells the stencil
      searches along each dimension. If None, the constructor computes it so the stencil
      visits all potential contacts within :math:`2 r_{max}`. A higher value expands the
      search stencil.
    - **box_size**: Bounding dimensions of the physical domain. Needed only when the box is
      small compared with the cell size, to meet the minimum grid size of
      ``2 * search_range + 1`` cells per axis under periodic boundary conditions.

    This collider suits large systems with low to moderate polydispersity (:math:`\alpha < 2.5`) and medium to high packing fractions. Highly polydisperse systems (:math:`\alpha \ge 3.0`) or systems containing rigid clumps with large internal overlaps reduce performance significantly. Overlaps artificially inflate the local cell occupancy :math:`\langle K \rangle` far beyond the macroscopic physical volume fraction :math:`\phi`. This lengthens the sequential loops and reduces GPU thread efficiency.

    Complexity
    ----------
    - Time: :math:`O(N)` - :math:`O(N \log N)` from sorting internally, plus :math:`O(N \cdot M \cdot \langle K \rangle)`
      for neighbor probing (M = ``neighbor_mask_size``, :math:`\langle K \rangle` = average occupancy).
    - Memory: :math:`O(N)`.

    Notes
    -----
    - **Batching with ``vmap``**: If you use ``jax.vmap`` to evaluate multiple
      simulation environments simultaneously, be aware of JAX's SIMD execution model.
      The innermost ``while`` loop executes sequentially. It must keep
      running for *all* environments in the batch until the environment with the highest
      local cell occupancy finishes its iterations. The single worst-case occupancy
      across the entire batch therefore sets the cost of a batched execution.
    """

    supported_search_geometries = frozenset(
        (SearchGeometry.ORTHOGONAL, SearchGeometry.SHEAR_PERIODIC)
    )

    neighbor_mask: jax.Array
    """Integer offsets defining the neighbor stencil (M, dim)."""

    cell_size: jax.Array
    """Linear size of a grid cell (scalar)."""

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
    ) -> Self:
        """Create a DynamicCellList instance from the reference state.

        Parameters
        ----------
        state : State
            Reference state containing positions and radii.
        cell_size : float, optional
            Grid cell size.
        search_range : int, optional
            Number of neighboring cells to search. A Python integer is
            required when ``cell_size`` is traced so the stencil shape is
            static.
        box_size : ArrayLike, optional
            Bounding dimensions of the physical box. Needed only when the box
            size is small compared with the cell size.

        Returns
        -------
        DynamicCellList
            A configured DynamicCellList instance.
        """
        max_rad = jnp.max(state._rad, initial=0.0)
        if cell_size is None:
            min_rad = jnp.min(state._rad, initial=jnp.inf)
            alpha = max_rad / jnp.where(min_rad > 0, min_rad, 1.0)
            cell_size = jnp.where(
                max_rad > 0, jnp.where(alpha < 2.5, 2.0 * max_rad, 0.5 * max_rad), 1.0
            )

        cell_size = jnp.asarray(cell_size, dtype=float)
        if cell_size.ndim != 0:
            raise ValueError("cell_size must be a finite positive scalar")
        if not isinstance(cell_size, jax.core.Tracer) and not bool(
            jnp.isfinite(cell_size) & (cell_size > 0)
        ):
            raise ValueError("cell_size must be a finite positive scalar")
        search_range_int: int | None = None
        if search_range is not None:
            if isinstance(search_range, bool):
                raise ValueError("search_range must be a positive integer")
            if isinstance(search_range, int):
                search_range_int = search_range
            else:
                search_range_array = jnp.asarray(search_range)
                if search_range_array.ndim != 0:
                    raise ValueError("search_range must be a positive integer")
                if isinstance(search_range_array, jax.core.Tracer):
                    raise ValueError(
                        "CellList.Create requires a Python integer `search_range` "
                        "when called while tracing."
                    )
                search_range_value = float(search_range_array)
                if not bool(
                    jnp.isfinite(search_range_array)
                ) or search_range_value != int(search_range_value):
                    raise ValueError("search_range must be a positive integer")
                search_range_int = int(search_range_value)
            if search_range_int < 1:
                raise ValueError("search_range must be a positive integer")

        if box_size is not None:
            box_size = jnp.asarray(box_size, dtype=float)
            for _ in range(2):
                if search_range is None:
                    sr = jnp.ceil(2 * max_rad / cell_size).astype(int)
                    sr = jnp.maximum(1, sr)
                else:
                    sr = jnp.asarray(search_range_int, dtype=int)
                min_grids_per_axis = 2 * sr + 1
                grid_dims = jnp.floor(box_size / cell_size).astype(int)
                grid_dims = jnp.maximum(grid_dims, min_grids_per_axis)
                cell_size = jnp.min(box_size / grid_dims)

        if search_range is None:
            if isinstance(cell_size, jax.core.Tracer) or isinstance(
                max_rad, jax.core.Tracer
            ):
                raise ValueError(
                    "CellList.Create requires explicit static `search_range` "
                    "when `cell_size` is traced."
                )
            search_range = jnp.ceil(2 * max_rad / cell_size).astype(int)
            search_range = jnp.maximum(1, search_range)
            search_range_int = int(search_range.item())
        resolved_search_range = cast(int, search_range_int)
        r = jnp.arange(-resolved_search_range, resolved_search_range + 1, dtype=int)
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
        """Compute pairwise contact forces and torques with DynamicCellList.

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
    @partial(jax.named_call, name="DynamicCellList.compute_potential_energy")
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
        collider = cast(DynamicCellList, system.collider)
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
    def create_neighbor_list(
        state: State, system: System, cutoff: float, max_neighbors: int
    ) -> tuple[State, System, jax.Array, jax.Array]:
        """Create a neighbor list of shape (N, max_neighbors) with DynamicCellList.

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
        p_neighbor_starts = _cell_starts(p_cell_hash, p_neighbor_hashes)

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
                dr = system.domain.displacement(pos_i, pos[orig_k], system)
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
            return row_body(stencil, cell_starts)

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
    @partial(jax.named_call, name="DynamicCellList.create_cross_neighbor_list")
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

        collider = cast(DynamicCellList, system.collider)

        # Inflate the cell size so the fixed stencil reach covers the
        # requested cutoff (see ``create_neighbor_list``).
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
                dr = system.domain.displacement(pos_ai, pos_b[perm_b[k]], system)
                return norm2(dr) <= cutoff_sq

            row_body = _make_direct_row_body(
                p_cell_hash_b, n_b, local_capacity, candidate_valid
            )
            return row_body(stencil, cell_starts)

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
