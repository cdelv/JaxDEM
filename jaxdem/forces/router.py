# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Force model router selecting laws based on species pairs."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System

from . import ForceModel
from .law_combiner import LawCombiner


@ForceModel.register("forcerouter")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ForceRouter(ForceModel):
    r"""A `ForceModel` that selects the force law from the species of the interacting particles.

    The router holds a symmetric :math:`S \times S` lookup table of force laws,
    where :math:`S` is the number of species. For a particle pair
    :math:`(i, j)`, the router evaluates the law at
    ``table[species_id[i]][species_id[j]]``.

    Notes
    -----
    - Use :meth:`from_dict` to build the table from a mapping of species pairs.
      Pairs not present in the mapping default to an empty :class:`LawCombiner`,
      which produces zero force, torque, and energy.
    - Dispatch evaluates every law in the table and selects the result with
      :func:`jax.lax.select_n`. The cost grows quadratically with the number
      of species, for scalar and batched calls alike.
    - :attr:`required_material_properties` is the union of the requirements of
      all laws in the table.
    """

    # Keep the table as pytree data so configurable law arrays participate in
    # transformations just like laws held by LawCombiner.
    table: tuple[tuple[ForceModel, ...], ...] = field(default=())
    """A symmetric :math:`S \\times S` table where entry ``table[a][b]`` is the :class:`ForceModel` that governs interactions between species ``a`` and ``b``."""

    @property
    def supports_analytical_energy_gradient(self) -> bool:
        """Whether every law reachable through the table supports minimization."""
        return all(
            law.supports_analytical_energy_gradient for row in self.table for law in row
        )

    @property
    def species_capacity(self) -> int | None:
        """Smallest capacity imposed by this table or a nested router."""
        capacities = [len(self.table)]
        capacities.extend(
            capacity
            for row in self.table
            for law in row
            if (capacity := law.species_capacity) is not None
        )
        return min(capacities)

    def history_shape(self, dim: int) -> tuple[int, ...]:
        return (
            sum(
                math.prod(self.table[a][b].history_shape(dim))
                for a in range(len(self.table))
                for b in range(a, len(self.table))
            ),
        )

    def init_history(self, pair_shape: tuple[int, ...], dim: int) -> jax.Array:
        histories = [
            self.table[a][b]
            .init_history(pair_shape, dim)
            .reshape((*pair_shape, math.prod(self.table[a][b].history_shape(dim))))
            for a in range(len(self.table))
            for b in range(a, len(self.table))
        ]
        if not histories:
            return ForceModel.init_history(self, pair_shape, dim)
        return jnp.concatenate(histories, axis=-1)

    def search_radii(self, state: State, system: System) -> jax.Array:
        """Conservative per-primitive reach across every routable law."""
        radii = jnp.zeros_like(state.rad)
        for row in self.table:
            for law in row:
                sub_system = dataclasses.replace(system, force_model=law)
                radii = jnp.maximum(radii, law.search_radii(state, sub_system))
        return radii

    @staticmethod
    @jax.jit(static_argnames=("advance_history",))
    @partial(jax.named_call, name="ForceRouter.force")
    def force(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
        *,
        advance_history: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        router = cast(ForceRouter, system.force_model)
        S = len(router.table)

        si = state.species_id[i]
        sj = state.species_id[j]

        f_map = {}
        t_map = {}
        history_parts = []
        offset = 0

        for a in range(S):
            for b in range(a, S):
                law = router.table[a][b]
                child_shape = law.history_shape(pos.shape[-1])
                width = math.prod(child_shape)
                h_ab = history[..., offset : offset + width].reshape(
                    (*history.shape[:-1], *child_shape)
                )
                sys_law = dataclasses.replace(system, force_model=law)
                f, t, nh = law.force(
                    i,
                    j,
                    pos,
                    state,
                    sys_law,
                    h_ab,
                    advance_history=advance_history,
                )

                f_map[(a, b)] = jnp.asarray(f, dtype=float)
                t_map[(a, b)] = jnp.asarray(t, dtype=float)

                mask = ((si == a) * (sj == b)) | ((si == b) * (sj == a))
                selected = jnp.where(
                    mask.reshape((*mask.shape, *((1,) * len(child_shape)))), nh, h_ab
                )
                history_parts.append(selected.reshape((*history.shape[:-1], width)))
                offset += width

                if a != b:
                    f_map[(b, a)] = f_map[(a, b)]
                    t_map[(b, a)] = t_map[(a, b)]

        pair_to_law = [[0] * S for _ in range(S)]
        law_idx = 0
        for a in range(S):
            for b in range(a, S):
                pair_to_law[a][b] = law_idx
                pair_to_law[b][a] = law_idx
                law_idx += 1
        idx = jnp.asarray(pair_to_law, dtype=si.dtype)[si, sj]
        f_results = [f_map[(a, b)] for a in range(S) for b in range(a, S)]
        t_results = [t_map[(a, b)] for a in range(S) for b in range(a, S)]

        idx_f = idx
        if jnp.ndim(idx) > 0:
            while idx_f.ndim < f_results[0].ndim:
                idx_f = idx_f[..., None]
            idx_f = jnp.broadcast_to(idx_f, f_results[0].shape)

        idx_t = idx
        if jnp.ndim(idx) > 0:
            while idx_t.ndim < t_results[0].ndim:
                idx_t = idx_t[..., None]
            idx_t = jnp.broadcast_to(idx_t, t_results[0].shape)

        f_final = jax.lax.select_n(idx_f, *f_results)
        t_final = jax.lax.select_n(idx_t, *t_results)

        new_history = (
            jnp.concatenate(history_parts, axis=-1) if history_parts else history
        )
        return f_final, t_final, new_history

    @property
    def required_material_properties(self) -> tuple[str, ...]:
        """Names of the material properties this force model needs.

        The sorted union of the material properties required by all laws in
        the table. Each name must be present in :attr:`System.mat_table`.
        Used for validation.
        """
        return tuple(
            sorted(
                {
                    p
                    for row in self.table
                    for law in row
                    for p in law.required_material_properties
                }
            )
        )

    @staticmethod
    @partial(jax.named_call, name="ForceRouter.from_dict")
    def from_dict(S: int, mapping: dict[tuple[int, int], ForceModel]) -> ForceRouter:
        """Build a :class:`ForceRouter` from a mapping of species pairs to force laws.

        The router symmetrizes the mapping: entry ``(a, b)`` also fills
        ``(b, a)``. Pairs not present in the mapping default to an empty
        :class:`LawCombiner` (zero force, torque, and energy).

        Parameters
        ----------
        S : int
            Number of species. The resulting table has shape ``S x S``.
        mapping : dict[tuple[int, int], ForceModel]
            Mapping from species-index pairs to the force law that governs
            interactions between those species.

        Returns
        -------
        ForceRouter
            A router with the fully populated, symmetric lookup table.

        """
        empty = LawCombiner()  # zero-force default
        m: list[list[ForceModel]] = [[empty for _ in range(S)] for _ in range(S)]
        for (a, b), law in mapping.items():
            m[a][b] = m[b][a] = law
        return ForceRouter(table=tuple(tuple(r) for r in m))

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="ForceRouter.energy")
    def energy(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
    ) -> jax.Array:
        """Compute the potential energy of the interaction between particle :math:`i` and particle :math:`j` with the law their species select.

        Parameters
        ----------
        i : int
            Index of the first particle.
        j : int
            Index of the second particle.
        pos : jax.Array
            Particle positions used to evaluate the interaction.
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        jax.Array
            Scalar potential energy computed by the law at
            ``table[species_id[i]][species_id[j]]``.

        """
        router = cast(ForceRouter, system.force_model)
        S = len(router.table)

        si = state.species_id[i]
        sj = state.species_id[j]
        idx = si * S + sj

        e_map = {}

        for a in range(S):
            for b in range(a, S):
                law = router.table[a][b]
                sys_law = dataclasses.replace(system, force_model=law)

                e = law.energy(i, j, pos, state, sys_law)

                e_map[(a, b)] = jnp.asarray(e, dtype=float)
                if a != b:
                    e_map[(b, a)] = e_map[(a, b)]

        e_results = [e_map[(a, b)] for a in range(S) for b in range(S)]
        idx_e = idx
        if jnp.ndim(idx) > 0:
            while idx_e.ndim < e_results[0].ndim:
                idx_e = idx_e[..., None]
            idx_e = jnp.broadcast_to(idx_e, e_results[0].shape)

        e_final = jax.lax.select_n(idx_e, *e_results)
        return e_final


__all__ = ["ForceRouter"]
