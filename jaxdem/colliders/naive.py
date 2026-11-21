# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Naive :math:`O(N^2)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Self

from jax.custom_batching import custom_vmap

from ..utils.linalg import cross, norm2
from . import Collider, valid_interaction_mask
from ._partition import _vec_sum

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Collider.register("naive")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class NaiveSimulator(Collider):
    r"""Implementation that computes forces and potential energies using a naive :math:`O(N^2)` all-pairs loop.

    This collider evaluates interactions between all particle pairs directly, without
    any spatial partitioning or binning.

    The total force acting on particle :math:`i` is the direct sum of its interactions
    with all other particles :math:`j` in the system:

    .. math::
        \mathbf{F}_i = \sum_{j=0}^{N-1} \mathbf{F}_{ij}(\mathbf{x}_i, \mathbf{x}_j, r_i, r_j) \cdot M_{ij}

    where :math:`\mathbf{F}_{ij}` is the force vector computed by the physical force model,
    and :math:`M_{ij}` is the interaction eligibility mask. The mask accounts for:

    - Clump member exclusions (internal clump particles do not exert forces on each other)
    - Bond connectivity exclusions
    - Contact overlap/cutoff checks

    Runtime and Cost Analysis
    -------------------------
    This collider always evaluates a fixed number of pair checks:

    .. math::
        \text{cost} \approx N^2 \cdot C_{interaction}

    where :math:`C_{interaction}` is the cost of a single pairwise force/energy query.

    Because the algorithm does not partition space into cells or project coordinates onto axes,
    its execution time is completely independent of:

    - The spatial distribution or packing fraction :math:`\phi` of the system
    - The particle polydispersity :math:`\alpha`

    * **Performance Trade-off**:

      - **For small systems (:math:`N \le 10^3 - 2 \cdot 10^3` depending on the GPU)**: NaiveSimulator is often the
        fastest collider because it does no sorting, hashing, or bookkeeping. This gives full GPU thread
        utilization and short JIT compilation times.
      - **For large systems (:math:`N \ge 10^4`)**: The quadratic complexity :math:`O(N^2)` becomes a severe
        performance bottleneck, and spatial partitioning colliders are significantly faster.

    Pair History
    ------------
    The collider stores force-model history densely: ``history[i, j]`` is the
    stored history of the directed pair ``(i, j)``, the interaction on ``i``
    from ``j``. Every pair keeps the same slot for the life of the collider,
    so no remapping is ever needed. Stateless laws have a zero-width trailing
    axis, which costs nothing. :meth:`System.create <jaxdem.system.System.create>` allocates the buffer.

    Complexity
    ----------
    - Time: :math:`O(N^2)`.
    - Memory: :math:`O(N^2 H)` for a law with history width :math:`H`, otherwise :math:`O(N)`.
    """

    history: jax.Array = field(
        default_factory=lambda: jnp.zeros((0, 0, 0)), kw_only=True
    )
    """Dense directed pair history with force-model-defined trailing dimensions."""

    def history_pair_shape(self, state_shape: tuple[int, ...]) -> tuple[int, ...]:
        n = state_shape[-2]
        return tuple(state_shape[:-2]) + (n, n)

    def refresh(
        self,
        state: State,
        force_model: Any | None = None,
        *,
        reset_history: bool = False,
    ) -> Self:
        """Reallocate the dense history when the particle count changes."""
        history = self.history
        tail = history.shape[2:]
        if history.shape[:2] == (state.N, state.N) and not reset_history:
            return self
        if force_model is not None:
            history = force_model.init_history((state.N, state.N), state.dim)
        elif tail == (0,) or reset_history:
            history = jnp.zeros((state.N, state.N) + tail, dtype=history.dtype)
        else:
            raise ValueError("force_model is required to initialize refreshed history")
        return replace(self, history=history)

    def reindex_history(
        self, state: State, system: System, new_to_old: jax.Array
    ) -> System:
        """Gather the dense history rows and columns of the surviving particles."""
        if self.history.size == 0:
            return system
        history = self.history[new_to_old[:, None], new_to_old[None, :]]
        collider = _naive_collider(system.collider)
        return replace(system, collider=replace(collider, history=history))

    @property
    def metadata(self) -> dict[str, Any]:
        """The naive collider has no construction parameters to persist."""
        return {}

    @staticmethod
    def get_history(
        state: State, system: System, neighbor_list: jax.Array
    ) -> jax.Array:
        """Gather stored history for an explicit ``(N, k)`` pair query."""
        collider = _naive_collider(system.collider)
        history = _dense_history(state, system, collider.history)
        rows = jnp.arange(state.N)[:, None]
        gathered = history[rows, jnp.maximum(neighbor_list, 0)]
        initialized = system.force_model.init_history(neighbor_list.shape, state.dim)
        mask = (neighbor_list >= 0).reshape(
            neighbor_list.shape + (1,) * (gathered.ndim - 2)
        )
        return jnp.where(mask, gathered, initialized)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="NaiveSimulator.compute_potential_energy")
    def compute_potential_energy(
        state: State, system: System
    ) -> tuple[State, System, jax.Array]:
        r"""Compute the total potential energy of the system with a naive :math:`O(N^2)` all-pairs loop.

        This method iterates over all particle pairs (i, j) and sums the
        potential energy contributions of the ``system.force_model``, using
        the stored pair history.

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
        iota = jax.lax.iota(dtype=int, size=state.N)
        pos = state.pos
        collider = _naive_collider(system.collider)
        history = _dense_history(state, system, collider.history)

        def row_energy(i: jax.Array, row_history: jax.Array) -> jax.Array:
            e_ij = jax.vmap(
                lambda j, h: system.force_model.energy(i, j, pos, state, system, h)
            )(iota, row_history)
            mask = valid_interaction_mask(
                state.clump_id[i],
                state.clump_id,
                state.bond_id[i],
                iota,
                system.interact_same_bond_id,
            )
            e_ij = jnp.where(mask > 0, e_ij, 0.0)
            return 0.5 * e_ij.sum(axis=0)

        energy = jnp.sum(jax.vmap(row_energy)(iota, history))
        return state, system, energy

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",), inline=True)
    @partial(jax.named_call, name="NaiveSimulator.create_neighbor_list")
    def create_neighbor_list(
        state: State,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> tuple[State, System, jax.Array, jax.Array]:
        r"""Compute a neighbor list with a naive :math:`O(N^2)` all-pairs search.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.
        cutoff : float
            The interaction radius (force cutoff).
        max_neighbors : int
            Maximum number of neighbors to store per particle.

        Returns
        -------
        Tuple[State, System, jax.Array, jax.Array]
            A tuple containing:

            - state: The simulation state.
            - system: The simulation system.
            - neighbor_list: Array of shape (N, max_neighbors) containing neighbor indices.
            - overflow: Boolean flag. True when any particle has more than
              ``max_neighbors`` neighbors.

        """
        if max_neighbors < 0:
            raise ValueError("max_neighbors must be non-negative")

        # Preserve documented semantics: always return shape (N, max_neighbors),
        # padded with -1. Clamp `lax.top_k` when `max_neighbors` exceeds N.
        iota = jax.lax.iota(dtype=int, size=state.N)
        pos = state.pos
        cutoff_sq = jnp.asarray(cutoff, dtype=pos.dtype) ** 2

        def per_particle(i: jax.Array) -> tuple[jax.Array, jax.Array]:
            dr = system.domain.displacement(pos[i], pos, system)  # (N, dim)
            valid = valid_interaction_mask(
                state.clump_id[i],
                state.clump_id,
                state.bond_id[i],
                iota,
                system.interact_same_bond_id,
            ) * (norm2(dr) <= cutoff_sq)
            overflow_flag = jnp.sum(valid) > max_neighbors
            k_eff = min(max_neighbors, iota.shape[0])
            topk = (
                jax.lax.top_k(jnp.where(valid, iota, -1), k_eff)[0]
                if k_eff > 0
                else jnp.empty((0,), dtype=int)
            )
            if k_eff < max_neighbors:
                topk = jnp.concatenate(
                    [topk, jnp.full((max_neighbors - k_eff,), -1, dtype=topk.dtype)]
                )
            return topk, overflow_flag

        nl, overflows = jax.vmap(per_particle)(iota)
        return state, system, nl, jnp.any(overflows)

    @staticmethod
    @jax.jit(inline=True, static_argnames=("advance_history", "quasistatic"))
    @partial(jax.named_call, name="NaiveSimulator.compute_force")
    def compute_force(
        state: State,
        system: System,
        *,
        advance_history: bool = True,
        quasistatic: bool = False,
    ) -> tuple[State, System]:
        r"""Compute the total force acting on each particle with a naive :math:`O(N^2)` all-pairs loop.

        This method sums the force contributions of the ``system.force_model``
        over all particle pairs (i, j) and updates the particle forces. With
        ``advance_history`` the dense pair history is advanced in place.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.
        advance_history : bool, optional
            Advance the stored pair history over this step.
        quasistatic : bool, optional
            Forwarded to the force law. True omits velocity-dependent terms.

        Returns
        -------
        Tuple[State, System]
            The updated ``State`` with computed forces and the ``System`` with
            the (possibly advanced) pair history.

        """
        iota = jax.lax.iota(dtype=int, size=state.N)
        pos_p = state._pos_p_rot
        pos = state.pos
        collider = _naive_collider(system.collider)
        tail = system.force_model.history_shape(state.dim)
        allocated = collider.history.shape == (state.N, state.N) + tail
        if advance_history and not allocated and tail != (0,):
            raise ValueError(
                "NaiveSimulator history is not allocated for this state. Build the "
                "system with System.create or call refresh_collider."
            )
        history = (
            collider.history
            if allocated
            else system.force_model.init_history((state.N, state.N), state.dim)
        )
        initialized = system.force_model.init_history(iota.shape, state.dim)

        has_torque = system.force_model.has_pair_torque
        # For stateless all-pairs systems, use variadic _vec_sum and tile large
        # pair grids (accounting for explicit vmap batches) to stay in L2 cache.
        if (not advance_history or tail == (0,)) and tail == (0,) and state.N > 0:
            force, torque = _naive_stateless_force_entry(state, system, quasistatic)
            state.force = force
            state.torque = torque
            return state, system

        def per_particle_i(
            i: jax.Array, pos_pi: jax.Array, row_history: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            res_f, res_t, new_history = system.force_model.force(
                i,
                iota,
                pos,
                state,
                system,
                row_history,
                advance_history=advance_history,
                quasistatic=quasistatic,
            )
            mask = valid_interaction_mask(
                state.clump_id[i],
                state.clump_id,
                state.bond_id[i],
                iota,
                system.interact_same_bond_id,
            )
            res_f = jnp.where(mask[..., None] > 0, res_f, 0.0)
            f_i = jnp.sum(res_f, axis=0)
            if has_torque:
                res_t = jnp.where(mask[..., None] > 0, res_t, 0.0)
                t_i = jnp.sum(res_t, axis=0) + (
                    0.0 if state.only_spheres else cross(pos_pi, f_i)
                )
            else:
                t_i = (
                    jnp.zeros_like(state.torque[i])
                    if state.only_spheres
                    else cross(pos_pi, f_i)
                )
            if advance_history:
                keep = mask.reshape(mask.shape + (1,) * (new_history.ndim - 1))
                new_history = jnp.where(keep > 0, new_history, initialized)
            else:
                new_history = row_history
            return f_i, t_i, new_history

        state.force, state.torque, history = jax.vmap(per_particle_i)(
            iota, pos_p, history
        )
        if not (advance_history and allocated):
            return state, system
        return state, replace(system, collider=replace(collider, history=history))


def _naive_collider(collider: Collider) -> NaiveSimulator:
    """Require the dense-history collider used by the naive evaluation methods."""
    if not isinstance(collider, NaiveSimulator):
        raise TypeError(
            "NaiveSimulator evaluation requires a NaiveSimulator collider, "
            f"got {type(collider).__name__}."
        )
    return collider


def _naive_stateless_force(
    state: State,
    system: System,
    quasistatic: bool,
    row_tile: int,
    col_tile: int,
) -> tuple[jax.Array, jax.Array]:
    """Stateless O(N^2) force reduction with variadic _vec_sum and L2 tiling."""
    n = state.N
    pos = state.pos
    pos_p = state._pos_p_rot
    has_torque = system.force_model.has_pair_torque

    if row_tile >= n and col_tile >= n:
        iota = jax.lax.iota(dtype=int, size=n)
        row_hist = system.force_model.init_history((n,), state.dim)

        def row_small(i: jax.Array) -> Any:
            res_f, res_t, _ = system.force_model.force(
                i,
                iota,
                pos,
                state,
                system,
                row_hist,
                advance_history=False,
                quasistatic=quasistatic,
            )
            mask = valid_interaction_mask(
                state.clump_id[i],
                state.clump_id,
                state.bond_id[i],
                iota,
                system.interact_same_bond_id,
            ).astype(bool)
            f_m = jnp.where(mask[..., None], res_f, 0.0)
            if not has_torque:
                return f_m
            return f_m, jnp.where(mask[..., None], res_t, 0.0)

        raw = jax.vmap(row_small)(iota)
        if not has_torque:
            force = _vec_sum(raw, 1)
            torque = (
                jnp.zeros_like(state.torque)
                if state.only_spheres
                else cross(pos_p, force)
            )
            return force, torque
        force = _vec_sum(raw[0], 1)
        torque = _vec_sum(raw[1], 1)
        return force, (torque if state.only_spheres else torque + cross(pos_p, force))

    row_tile = min(n, row_tile)
    col_tile = min(n, col_tile)
    exact_rows = (n % row_tile) == 0
    exact_cols = (n % col_tile) == 0
    padded_r = (n + row_tile - 1) // row_tile * row_tile
    padded_c = (n + col_tile - 1) // col_tile * col_tile
    row_blocks = jnp.arange(padded_r).reshape(-1, row_tile)
    col_blocks = jnp.arange(padded_c).reshape(-1, col_tile)
    tile_hist = system.force_model.init_history((col_tile,), state.dim)

    def eval_row_block(rows: jax.Array) -> Any:
        safe_rows = rows if exact_rows else jnp.minimum(rows, n - 1)

        def eval_col_block(carry: Any, cols: jax.Array) -> tuple[Any, None]:
            safe_cols = cols if exact_cols else jnp.minimum(cols, n - 1)
            in_range = None if exact_cols else (cols < n)

            def row_fn(i: jax.Array) -> Any:
                res_f, res_t, _ = system.force_model.force(
                    i,
                    safe_cols,
                    pos,
                    state,
                    system,
                    tile_hist,
                    advance_history=False,
                    quasistatic=quasistatic,
                )
                mask = valid_interaction_mask(
                    state.clump_id[i],
                    state.clump_id[safe_cols],
                    state.bond_id[i],
                    safe_cols,
                    system.interact_same_bond_id,
                ).astype(bool)
                if in_range is not None:
                    mask = in_range & mask
                f_masked = jnp.where(mask[..., None], res_f, 0.0)
                if not has_torque:
                    return f_masked
                t_masked = jnp.where(mask[..., None], res_t, 0.0)
                return f_masked, t_masked

            raw = jax.vmap(row_fn)(safe_rows)
            if not has_torque:
                return carry + _vec_sum(raw, 1), None
            return (
                carry[0] + _vec_sum(raw[0], 1),
                carry[1] + _vec_sum(raw[1], 1),
            ), None

        init_acc: Any = (
            jnp.zeros((row_tile, state.force.shape[-1]), dtype=state.force.dtype)
            if not has_torque
            else (
                jnp.zeros((row_tile, state.force.shape[-1]), dtype=state.force.dtype),
                jnp.zeros((row_tile, state.torque.shape[-1]), dtype=state.torque.dtype),
            )
        )
        acc, _ = jax.lax.scan(eval_col_block, init_acc, col_blocks)
        return acc

    if row_blocks.shape[0] == 1:
        acc = eval_row_block(row_blocks[0])
    else:
        acc = jax.lax.map(eval_row_block, row_blocks)
        acc = jax.tree.map(lambda a: a.reshape((-1,) + a.shape[2:])[:n], acc)
    if not has_torque:
        force = acc[:n]
        torque = (
            jnp.zeros_like(state.torque)
            if state.only_spheres
            else cross(pos_p, force)
        )
        return force, torque
    force, torque = acc[0][:n], acc[1][:n]
    return force, (torque if state.only_spheres else torque + cross(pos_p, force))


def _naive_tile_sizes(n: int, batch_size: int) -> tuple[int, int]:
    """Choose L2-friendly (row_tile, col_tile) sizes for ``batch_size * n * n`` pairs."""
    if batch_size * n * n <= 4_194_304:
        return n, n
    if batch_size > 1:
        return min(n, 256), min(n, 256)
    return min(n, 4096), min(n, 1024)


def _naive_stateless_force_dispatch(quasistatic: bool) -> Any:
    """Select L2-friendly tile sizes accounting for an enclosing ``jax.vmap`` batch."""

    @custom_vmap
    def dispatch(state: State, system: System) -> tuple[jax.Array, jax.Array]:
        row_tile, col_tile = _naive_tile_sizes(state.N, 1)
        return _naive_stateless_force(state, system, quasistatic, row_tile, col_tile)

    @dispatch.def_vmap
    def mapped(
        axis_size: int, in_batched: Any, state: State, system: System
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[bool, bool]]:
        axes = jax.tree.map(lambda batched: 0 if batched else None, in_batched)
        n = state.pos_c.shape[-2]
        row_tile, col_tile = _naive_tile_sizes(n, axis_size)
        out = jax.vmap(
            lambda st, sy: _naive_stateless_force(
                st, sy, quasistatic, row_tile, col_tile
            ),
            in_axes=tuple(axes),
            axis_size=axis_size,
        )(state, system)
        return out, (True, True)

    return dispatch


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _naive_stateless_force_entry(
    state: State, system: System, quasistatic: bool
) -> tuple[jax.Array, jax.Array]:
    """Differentiable entry point for stateless all-pairs force evaluation."""
    return _naive_stateless_force_dispatch(quasistatic)(state, system)


@_naive_stateless_force_entry.defjvp
def _naive_stateless_force_jvp(
    quasistatic: bool, primals: Any, tangents: Any
) -> Any:
    def fixed_tiled(st: State, sy: System) -> tuple[jax.Array, jax.Array]:
        n = st.N
        row_tile = n if n <= 512 else min(n, 512)
        col_tile = n if n <= 512 else min(n, 512)
        return _naive_stateless_force(st, sy, quasistatic, row_tile, col_tile)

    return _naive_stateless_force_entry(*primals, quasistatic), jax.jvp(
        fixed_tiled, primals, tangents
    )[1]


def _dense_history(state: State, system: System, history: jax.Array) -> jax.Array:
    """Return the ``(N, N, H)`` history, or the law's initial history if unallocated."""
    if history.shape == (state.N, state.N) + system.force_model.history_shape(
        state.dim
    ):
        return history
    return system.force_model.init_history((state.N, state.N), state.dim)


__all__ = ["NaiveSimulator"]
