# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Naive :math:`O(N^2)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any

from ..utils.linalg import cross, norm2
from . import Collider, valid_interaction_mask

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
    ) -> Collider:
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
        history = self.history[new_to_old][:, new_to_old]
        return replace(system, collider=replace(system.collider, history=history))

    @property
    def metadata(self) -> dict[str, Any]:
        """The naive collider has no construction parameters to persist."""
        return {}

    @staticmethod
    def get_history(
        state: State, system: System, neighbor_list: jax.Array
    ) -> jax.Array:
        """Gather stored history for an explicit ``(N, k)`` pair query."""
        collider = system.collider
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
        history = _dense_history(state, system, system.collider.history)

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
        # padded with -1. But `lax.top_k` requires k <= len(candidates), so we
        # clamp internally when `max_neighbors` exceeds N.
        if max_neighbors == 0:
            empty = jnp.empty((state.N, 0), dtype=int)
            iota = jax.lax.iota(dtype=int, size=state.N)
            pos = state.pos
            cutoff_sq = jnp.asarray(cutoff, dtype=pos.dtype) ** 2

            def any_neighbor(i: jax.Array) -> jax.Array:
                dr = system.domain._displacement(pos[i], pos, system)
                valid = valid_interaction_mask(
                    state.clump_id[i],
                    state.clump_id,
                    state.bond_id[i],
                    iota,
                    system.interact_same_bond_id,
                ) * (norm2(dr) <= cutoff_sq)
                return jnp.any(valid)

            return state, system, empty, jnp.any(jax.vmap(any_neighbor)(iota))

        iota = jax.lax.iota(dtype=int, size=state.N)
        pos = state.pos
        cutoff_sq = jnp.asarray(cutoff, dtype=pos.dtype) ** 2

        def per_particle(i: jax.Array) -> tuple[jax.Array, jax.Array]:
            dr = system.domain._displacement(pos[i], pos, system)  # (N, dim)
            dist_sq = norm2(dr)
            valid = valid_interaction_mask(
                state.clump_id[i],
                state.clump_id,
                state.bond_id[i],
                iota,
                system.interact_same_bond_id,
            ) * (dist_sq <= cutoff_sq)
            num_neighbors = jnp.sum(valid)
            overflow_flag = num_neighbors > max_neighbors
            candidates = jnp.where(valid, iota, -1)
            k_eff = min(max_neighbors, candidates.shape[0])
            topk = jax.lax.top_k(candidates, k_eff)[0]
            # If max_neighbors > N, pad back to the requested buffer size.
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
        collider = system.collider
        allocated = _history_allocated(state, system, collider.history)
        if (
            advance_history
            and not allocated
            and system.force_model.history_shape(state.dim) != (0,)
        ):
            raise ValueError(
                "NaiveSimulator history is not allocated for this state. Build the "
                "system with System.create or call refresh_collider."
            )
        history = _dense_history(state, system, collider.history)
        initialized = system.force_model.init_history(iota.shape, state.dim)

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
            res_t = jnp.where(mask[..., None] > 0, res_t, 0.0)
            if advance_history:
                keep = mask.reshape(mask.shape + (1,) * (new_history.ndim - 1))
                new_history = jnp.where(keep > 0, new_history, initialized)
            else:
                new_history = row_history
            f_i = jnp.sum(res_f, axis=0)
            t_i = jnp.sum(res_t, axis=0) + cross(pos_pi, f_i)
            return f_i, t_i, new_history

        state.force, state.torque, history = jax.vmap(per_particle_i)(
            iota, pos_p, history
        )
        if not (advance_history and allocated):
            return state, system
        return state, replace(system, collider=replace(collider, history=history))


def _history_allocated(state: State, system: System, history: jax.Array) -> bool:
    """Whether ``history`` is the ``(N, N, H)`` buffer for this state and law."""
    n = state.N
    tail = system.force_model.history_shape(state.dim)
    return history.shape == (n, n) + tail


def _dense_history(state: State, system: System, history: jax.Array) -> jax.Array:
    """Return the ``(N, N, H)`` history, or the law's initial history if unallocated."""
    if _history_allocated(state, system, history):
        return history
    return system.force_model.init_history((state.N, state.N), state.dim)


__all__ = ["NaiveSimulator"]
