# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Sweep and prune :math:`O(N log N)` collider implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from ..utils.linalg import cross
from . import Collider, valid_interaction_mask

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@jax.tree_util.register_dataclass
@dataclass
class DummyState:
    pos: jax.Array
    rad: jax.Array
    clump_id: jax.Array
    bond_id: jax.Array
    unique_id: jax.Array
    N: int
    dim: int


@Collider.register("SweepAndPrune")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SweepAndPrune(Collider):
    r"""PCA-aligned 1D slab partitioning Sweep and Prune with shifted multi-pass approach.

    This collider is an implementation of a variation of the Sweep and Prune algorithm.
    This variation is based on the paper:
        Real-time Collision Culling of a Million Bodies on Graphics Processing Units,  Liu et al. (2011)

    Mathematical Formalism & Slab Offsets
    --------------------------------------
    In :math:`dim` dimensions, the space is partitioned into slabs of width :math:`bin\_size = 4 r_{max}`
    along the perpendicular axes to the sweeping direction. To handle boundaries and prevent boundary-crossing particles from
    being missed, the algorithm performs :math:`2^{dim-1}` parallel passes, each with a different
    perpendicular coordinate shift of :math:`2 r_{max}` or :math:`0.0`.

    In each pass :math:`p`, a particle's perpendicular coordinates are mapped to a 1D cell index
    :math:`\text{HASH}_p`. To sweep all slabs in parallel without dynamic coordinate queries,
    the coordinates along the principal (sweeping) axis, :math:`x_{proj}`, are offset by:

    .. math::
        x_{proj, shifted} = x_{proj} + \text{HASH}_p \cdot L_{proj}

    where :math:`L_{proj} = L_{box} + 2 \cdot \text{cutoff} + 1.0` is a spacing buffer ensuring
    that slabs are arranged end-to-end along a single concatenated line with no overlap.
    Particles are then sorted along this concatenated line, grouping all particles in the same slab
    together.

    For each particle :math:`i`, we query a static window of :math:`2K` neighboring indices in the
    sorted array (:math:`K` to the left and :math:`K` to the right). Interactions are evaluated
    only if both particles reside in the same cell slab and pass the canonical deduplication check to
    prevent duplicate calculations across different passes.

    Runtime and Cost Analysis
    -------------------------
    Since the search window :math:`2K` is a static compilation parameter, the number of candidate
    checks performed per particle is fixed. The total pair evaluation cost scales as:

    .. math::
        \text{cost} \approx N \cdot 2^{dim-1} \cdot 2K

    * **Advantages**:
      Unlike cell lists, the number of distance evaluations per particle is constant and completely
      independent of particle size, polydispersity, packing fraction, or cell size.

      Sorting complexity is :math:`O(2^{dim-1} \cdot N \log N)` per time step, which is highly efficient
      as the state remains mostly sorted.

    * **Polydispersity Penalty and Window Overflow**:
      The safety of the collider (preventing missed contacts) relies on the window size :math:`K` being
      large enough to cover all overlapping particles in the projected slab. Let :math:`W = 4r_{max}`
      be the slab width. The expected number of particles in a search volume of length :math:`2r_{max}` is:

      .. math::
        \lambda = \rho (2 \cdot W^{dim-1} \cdot r_{max}) = \frac{2 \cdot 4^{dim-1}}{k_v} \cdot \phi \cdot \frac{V_{max}}{\langle V \rangle}

      where :math:`\phi` is the volume fraction, :math:`k_v` is the geometric shape factor, and
      :math:`V_{max} / \langle V \rangle` is the ratio of largest to average particle volume.
      For highly polydisperse systems where :math:`\alpha = r_{max}/r_{min} \gg 1`, the volume ratio
      scales as :math:`O(\alpha^{dim})`. This requires a large window size :math:`K` to guarantee
      correctness, which increases the constant overhead of the search loops.

      However, due to the partition of space, the search window :math:`K` does not depend on the system size like in the standard Sweep and Prune algorithm.

    Constructor Parameters
    ----------------------
    - **K**: The static search window size (number of sorted neighbors to check in each direction). A larger `K` avoids
      candidate window overflow warnings in clustered regions but increases execution overhead. Default is `8`.

    This collider is suitable mid to high polydispersity systems. However, it is also suitable for systems with overlapping rigid clumps at the cost of increasing :math:`K`.

    Complexity
    ----------
    - Time: :math:`O(2^{dim-1} \cdot N \log N)` sorting, plus :math:`O(2^{dim} \cdot N \cdot K)` traversal.
    - Memory: :math:`O(2^{dim} \cdot N \cdot K)` to store candidate indices and masks.
    """

    K: int = jax.tree.static()
    """
    The static search window radius (number of sorted neighbors to check in each direction).
    """

    @classmethod
    def Create(
        cls,
        state: State,
        K: int | None = None,
    ) -> Self:
        """Creates a SweepAndPruneShifted instance based on the reference state.

        Parameters
        ----------
        state : State
            Reference state containing positions and radii.
        max_neighbors : int, optional
            Ignored parameter (retained for signature compatibility).
        K : int, optional
            Static search window radius size.

        Returns
        -------
        SweepAndPruneShifted
            A configured SweepAndPruneShifted instance.
        """
        if K is None:
            max_rad = jnp.max(state.rad)
            min_rad = jnp.median(state.rad)
            particle_volume = min_rad**state.dim
            box_volume = 4 * max_rad**state.dim
            K = int(jnp.ceil(box_volume / particle_volume).astype(int)) + 2
            K = max(K, 4)

        return cls(
            K=int(K),
        )

    @staticmethod
    def _find_candidates(
        state: State | DummyState,
        system: System,
        cutoff: float,
        is_neighbor_list: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Finds collision candidates along binned 1D cell slabs.

        Parameters
        ----------
        state : State | DummyState
            The state containing positions, radii, clump IDs, and bond IDs.
        system : System
            The configuration of the simulation.
        cutoff : float
            Interaction cutoff distance.
        is_neighbor_list : bool, default False
            Whether neighbor list rules or touching radii are used for contact limit check.

        Returns
        -------
        Tuple[jax.Array, jax.Array, jax.Array]
            Array of candidate indices, validity boolean masks, and window overflow flag.
        """
        collider = cast(SweepAndPrune, system.collider)
        box_size = system.domain.box_size
        anchor = system.domain.anchor
        if system.domain.periodic:
            pos = state.pos - box_size * jnp.floor((state.pos - anchor) / box_size)
        else:
            pos = state.pos

        n, dim = pos.shape

        # Sweep and Prune is correct sweeping along any axis. Using a static axis
        # avoids dynamic slicing, which is extremely expensive on GPUs.
        axis = dim - 1
        perp_axes = slice(0, dim - 1)

        proj = pos[:, axis]
        max_rad = jnp.max(state.rad)

        # Perpendicular binning parameters
        search_limit = cutoff if is_neighbor_list else (2.0 * max_rad)
        bin_size = 2.0 * search_limit
        box_perp = box_size[perp_axes]
        anchor_perp = anchor[perp_axes]

        grid_dims = jnp.floor(box_perp / bin_size).astype(int)
        grid_dims = jnp.maximum(1, grid_dims)

        W = box_perp / grid_dims
        grid_strides = jnp.concatenate(
            [jnp.array([1], dtype=int), jnp.cumprod(grid_dims[:-1])]
        )

        K = collider.K
        search_K = K + 1
        iota_k = jnp.concatenate(
            [jnp.arange(-search_K, 0), jnp.arange(1, search_K + 1)]
        )

        # Precompute reciprocals to replace expensive divisions with multiplications
        inv_box_size = 1.0 / box_size
        inv_W = 1.0 / W

        # Compute unshifted cell coordinates for all particles
        c_unshifted = ((pos[:, perp_axes] - anchor_perp) * inv_W).astype(int)
        c_unshifted = jnp.minimum(c_unshifted, grid_dims - 1)

        # We only shift the perpendicular dimensions, so we have 2^(dim-1) combinations.
        import itertools

        perp_combos = list(itertools.product([0, 1], repeat=dim - 1))
        perp_combos_arr = jnp.array(perp_combos, dtype=bool)

        shifts_perp_list = []
        for combo in perp_combos:
            shift_perp_v = jnp.zeros(dim - 1)
            for i in range(dim - 1):
                val = jnp.where(combo[i] == 1, search_limit, 0.0)
                shift_perp_v = shift_perp_v.at[i].set(val)
            shifts_perp_list.append(shift_perp_v)
        shifts_perp = jnp.stack(shifts_perp_list)
        num_passes = len(perp_combos)

        L_proj = box_size[axis] + 2.0 * cutoff + 1.0

        def single_pass(
            shift_perp_v: jax.Array, combo_p: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            dx_perp = pos[:, perp_axes] - shift_perp_v - anchor_perp
            wrapped_dx_perp = jnp.where(dx_perp < 0.0, dx_perp + box_perp, dx_perp)

            c = (wrapped_dx_perp * inv_W).astype(int)
            c = jnp.minimum(c, grid_dims - 1)
            HASH_p = jnp.dot(c, grid_strides)

            # Offset sweeping axis coordinates by HASH_p * L_proj to align bins in series
            proj_shifted = proj + HASH_p * L_proj
            order = jnp.argsort(proj_shifted)

            pos_sorted = pos[order]
            rad_sorted = state.rad[order]
            clump_sorted = state.clump_id[order]
            bond_sorted = state.bond_id[order]
            uid_sorted = state.unique_id[order]

            HASH_p_sorted = HASH_p[order]

            # Start indices via prefix maximum
            changes = jnp.concatenate(
                [jnp.array([True]), HASH_p_sorted[1:] != HASH_p_sorted[:-1]]
            )
            change_idx = jnp.where(changes, jnp.arange(n), 0)
            start_indices = jax.lax.associative_scan(jnp.maximum, change_idx)

            # End indices via prefix minimum on reversed array
            changes_right = jnp.concatenate(
                [HASH_p_sorted[:-1] != HASH_p_sorted[1:], jnp.array([True])]
            )
            change_idx_right = jnp.where(changes_right, jnp.arange(1, n + 1), n)
            end_indices = jax.lax.associative_scan(jnp.minimum, change_idx_right[::-1])[
                ::-1
            ]

            n_slabs = end_indices - start_indices

            r_local = jnp.arange(n, dtype=int) - start_indices
            r_temp = r_local[:, None] + iota_k[None, :]
            r_wrapped = jnp.where(r_temp < 0, r_temp + n_slabs[:, None], r_temp)
            r_wrapped = jnp.where(
                r_wrapped >= n_slabs[:, None], r_wrapped - n_slabs[:, None], r_wrapped
            )
            r_wrapped = jnp.clip(r_wrapped, 0, n_slabs[:, None] - 1)
            idx_candidates = start_indices[:, None] + r_wrapped

            pos_cand = pos_sorted[idx_candidates]
            rad_cand = rad_sorted[idx_candidates]

            if dim == 2:
                dr_x = pos_sorted[:, None, 0] - pos_cand[..., 0]
                dr_y = pos_sorted[:, None, 1] - pos_cand[..., 1]
                if system.domain.periodic:
                    dr_x = dr_x - box_size[0] * jnp.round(dr_x * inv_box_size[0])
                    dr_y = dr_y - box_size[1] * jnp.round(dr_y * inv_box_size[1])
                dist_sq = dr_x**2 + dr_y**2
                dx = jnp.abs(dr_y)
            else:
                dr_x = pos_sorted[:, None, 0] - pos_cand[..., 0]
                dr_y = pos_sorted[:, None, 1] - pos_cand[..., 1]
                dr_z = pos_sorted[:, None, 2] - pos_cand[..., 2]
                if system.domain.periodic:
                    dr_x = dr_x - box_size[0] * jnp.round(dr_x * inv_box_size[0])
                    dr_y = dr_y - box_size[1] * jnp.round(dr_y * inv_box_size[1])
                    dr_z = dr_z - box_size[2] * jnp.round(dr_z * inv_box_size[2])
                dist_sq = dr_x**2 + dr_y**2 + dr_z**2
                dx = jnp.abs(dr_z)

            if is_neighbor_list:
                overlap = (dist_sq <= cutoff**2) * (dx <= cutoff + 1e-3)
            else:
                overlap_limit = rad_sorted[:, None] + rad_cand + cutoff
                overlap = (dist_sq <= overlap_limit**2) * (dx <= overlap_limit + 1e-3)

            valid = valid_interaction_mask(
                clump_sorted[:, None],
                clump_sorted[idx_candidates],
                bond_sorted[:, None],
                uid_sorted[idx_candidates],
            )

            in_same_cell = HASH_p_sorted[:, None] == HASH_p_sorted[idx_candidates]
            candidates_orig = order[idx_candidates]

            # Canonical cell check for duplicate checking (replaces expensive check_pair gathers)
            c_i_unshifted = c_unshifted[order]
            c_j_unshifted = c_unshifted[candidates_orig]
            combo_canonical = c_i_unshifted[:, None, :] != c_j_unshifted
            if dim == 2:
                not_in_prev = combo_canonical[..., 0] == combo_p[0]
            else:
                not_in_prev = (combo_canonical[..., 0] == combo_p[0]) & (
                    combo_canonical[..., 1] == combo_p[1]
                )

            not_duplicate_offset = jnp.abs(iota_k)[None, :] < n_slabs[:, None]
            is_neighbor_pass = (
                overlap * valid * in_same_cell * not_in_prev * not_duplicate_offset
            )

            rank_p = (
                jnp.empty_like(order).at[order].set(jnp.arange(n, dtype=order.dtype))
            )

            return rank_p, candidates_orig, is_neighbor_pass

        rank_ps, candidates_origs, is_neighbor = jax.vmap(single_pass)(
            shifts_perp, perp_combos_arr
        )

        # Check if the boundary candidate indices are duplicates of any inner candidate indices
        candidates_origs_inner = candidates_origs[:, :, 1:-1]
        is_duplicate_left = jnp.any(
            candidates_origs[:, :, 0, None] == candidates_origs_inner, axis=-1
        )
        is_duplicate_right = jnp.any(
            candidates_origs[:, :, -1, None] == candidates_origs_inner, axis=-1
        )

        # A boundary neighbor is a real overflow if it's not a duplicate of an inner candidate
        real_overflow_left = is_neighbor[:, :, 0] * ~is_duplicate_left
        real_overflow_right = is_neighbor[:, :, -1] * ~is_duplicate_right
        overflow_flag = jnp.any(real_overflow_left | real_overflow_right)

        # Slice out boundary elements to keep K elements for force / neighbor list
        is_neighbor_actual = is_neighbor[:, :, 1:-1]
        candidates_origs_actual = candidates_origs[:, :, 1:-1]

        # Map back to original order and concatenate the passes
        all_candidates_list = [
            candidates_origs_actual[p][rank_ps[p]] for p in range(num_passes)
        ]
        all_candidates = jnp.concatenate(all_candidates_list, axis=1)

        all_is_neighbor_list = [
            is_neighbor_actual[p][rank_ps[p]] for p in range(num_passes)
        ]
        all_is_neighbor = jnp.concatenate(all_is_neighbor_list, axis=1)

        return all_candidates, all_is_neighbor, overflow_flag

    @staticmethod
    @jax.jit
    def compute_force(state: State, system: System) -> tuple[State, System]:
        """Computes pairwise contact forces and torques using SweepAndPruneShifted.

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
        collider = cast(SweepAndPrune, system.collider)

        all_candidates, all_is_neighbor, overflow_flag = collider._find_candidates(
            state, system, 0.0, is_neighbor_list=False
        )

        # Warn if sweep search window overflow occurred during force calculation
        jax.lax.cond(
            overflow_flag,
            lambda: jax.debug.print(
                "WARNING: SweepAndPrune candidate window overflow detected during force computation (K={K} is too small). Some collisions may have been missed. Please increase K.",
                K=collider.K,
            ),
            lambda: None,
        )

        pos_global = state.pos
        iota = jax.lax.iota(dtype=int, size=state.N)

        def per_particle_force(
            i: jax.Array, pos_pi: jax.Array, candidates: jax.Array, mask: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            def per_candidate_force(
                j_id: jax.Array, valid: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                safe_j = jnp.maximum(j_id, 0)
                f, t = system.force_model.force(i, safe_j, pos_global, state, system)
                return f * valid, t * valid

            forces, torques = jax.vmap(per_candidate_force)(candidates, mask)
            f_sum = jnp.sum(forces, axis=0)
            t_sum = jnp.sum(torques, axis=0) + cross(pos_pi, f_sum)
            return f_sum, t_sum

        state.force, state.torque = jax.vmap(per_particle_force)(
            iota, state._pos_p_rot, all_candidates, all_is_neighbor
        )

        return state, system

    @staticmethod
    @jax.jit
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
        collider = cast(SweepAndPrune, system.collider)

        all_candidates, all_is_neighbor, overflow_flag = collider._find_candidates(
            state, system, 0.0, is_neighbor_list=False
        )

        # Warn if sweep search window overflow occurred during force calculation
        jax.lax.cond(
            overflow_flag,
            lambda: jax.debug.print(
                "WARNING: SweepAndPrune candidate window overflow detected during force computation (K={K} is too small). Some collisions may have been missed. Please increase K.",
                K=collider.K,
            ),
            lambda: None,
        )

        pos_global = state.pos
        iota = jax.lax.iota(dtype=int, size=state.N)

        def per_particle_energy(
            i: jax.Array, candidates: jax.Array, mask: jax.Array
        ) -> jax.Array:
            def per_neighbor_energy(j_id: jax.Array, valid: jax.Array) -> jax.Array:
                safe_j = jnp.maximum(j_id, 0)
                e = system.force_model.energy(i, safe_j, pos_global, state, system)
                return e * valid

            return 0.5 * jnp.sum(jax.vmap(per_neighbor_energy)(candidates, mask))

        return jnp.sum(
            jax.vmap(per_particle_energy)(iota, all_candidates, all_is_neighbor)
        )

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    def create_neighbor_list(
        state: State,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> tuple[State, System, jax.Array, jax.Array]:
        """Creates a neighbor list of shape (N, max_neighbors) using SweepAndPruneShifted.

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
        collider = cast(SweepAndPrune, system.collider)

        all_candidates, all_is_neighbor, search_overflow = collider._find_candidates(
            state, system, cutoff, is_neighbor_list=True
        )

        n = state.N
        neighbors = jnp.where(all_is_neighbor, all_candidates, -1)

        # Vectorized prefix-sum packing (replaces expensive row-wise top_k sorting)
        indices = jnp.cumsum(all_is_neighbor, axis=-1) - 1
        col_idx = jnp.where(all_is_neighbor, indices, max_neighbors)
        row_idx = jnp.arange(n)[:, None]

        nl_final = jnp.full((n, max_neighbors), -1, dtype=neighbors.dtype)
        nl_final = nl_final.at[row_idx, col_idx].set(neighbors, mode="drop")

        total_neighbors_count = jnp.sum(all_is_neighbor, axis=-1)
        storage_overflow = jnp.any(total_neighbors_count > max_neighbors)

        # Warn if sweep search window overflow occurred during neighbor list build
        jax.lax.cond(
            search_overflow,
            lambda: jax.debug.print(
                "WARNING: SweepAndPrune candidate window overflow detected during neighbor list build (K={K} is too small).",
                K=collider.K,
            ),
            lambda: None,
        )

        # Warn if storage overflow occurred during neighbor list build
        jax.lax.cond(
            storage_overflow,
            lambda: jax.debug.print(
                "WARNING: SweepAndPrune neighbor list storage overflow detected (max_neighbors={max_neighbors} is too small).",
                max_neighbors=max_neighbors,
            ),
            lambda: None,
        )

        return state, system, nl_final, storage_overflow | search_overflow

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
        if max_neighbors == 0:
            n_a = pos_a.shape[0]
            empty = jnp.empty((n_a, 0), dtype=int)
            return empty, jnp.asarray(False)

        n_a = pos_a.shape[0]
        n_b = pos_b.shape[0]

        # Merge pos_a and pos_b into one array
        pos = jnp.concatenate([pos_a, pos_b], axis=0)
        rad = jnp.full(n_a + n_b, cutoff / 2.0)

        clump_id = jnp.concatenate(
            [jnp.zeros(n_a, dtype=int), jnp.ones(n_b, dtype=int)]
        )
        bond_id = jnp.full((n_a + n_b, 1), -1, dtype=int)
        unique_id = jnp.arange(n_a + n_b, dtype=int)

        dummy_state = DummyState(
            pos=pos,
            rad=rad,
            clump_id=clump_id,
            bond_id=bond_id,
            unique_id=unique_id,
            N=n_a + n_b,
            dim=pos.shape[1],
        )

        collider = cast(SweepAndPrune, system.collider)

        all_candidates, all_is_neighbor, search_overflow = collider._find_candidates(
            dummy_state, system, cutoff, is_neighbor_list=True
        )

        # Slice to only query points from pos_a
        candidates_a = all_candidates[:n_a]
        is_neighbor_a = all_is_neighbor[:n_a]

        # Mask interactions to only include target pos_b points (index >= n_a)
        is_cross_neighbor = is_neighbor_a * (candidates_a >= n_a)
        candidates_b = candidates_a - n_a
        neighbors = jnp.where(is_cross_neighbor, candidates_b, -1)

        # Vectorized prefix-sum packing
        indices = jnp.cumsum(is_cross_neighbor, axis=-1) - 1
        col_idx = jnp.where(is_cross_neighbor, indices, max_neighbors)
        row_idx = jnp.arange(n_a)[:, None]

        nl_final = jnp.full((n_a, max_neighbors), -1, dtype=neighbors.dtype)
        nl_final = nl_final.at[row_idx, col_idx].set(neighbors, mode="drop")

        total_neighbors_count = jnp.sum(is_cross_neighbor, axis=-1)
        storage_overflow = jnp.any(total_neighbors_count > max_neighbors)

        # Warn if sweep search window overflow occurred during cross neighbor list build
        jax.lax.cond(
            search_overflow,
            lambda: jax.debug.print(
                "WARNING: SweepAndPrune cross neighbor list search overflow detected (K={K} is too small).",
                K=collider.K,
            ),
            lambda: None,
        )

        # Warn if storage overflow occurred
        jax.lax.cond(
            storage_overflow,
            lambda: jax.debug.print(
                "WARNING: SweepAndPrune cross neighbor list storage overflow detected (max_neighbors={max_neighbors} is too small).",
                max_neighbors=max_neighbors,
            ),
            lambda: None,
        )

        return nl_final, storage_overflow | search_overflow
