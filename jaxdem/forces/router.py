# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Force model router selecting laws based on species pairs."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, cast

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
    r"""A `ForceModel` implementation that dispatches to different force laws based on the species of the interacting particles.

    The router holds a symmetric :math:`S \times S` lookup table of force laws,
    where :math:`S` is the number of species. For a particle pair :math:`(i, j)`,
    the law at ``table[species_id[i]][species_id[j]]`` is evaluated.

    Notes
    -----
    - Use :meth:`from_dict` to build the table from a mapping of species pairs.
      Pairs not present in the mapping default to an empty :class:`LawCombiner`,
      which produces zero force, torque, and energy.
    - For per-pair scalar calls, dispatch uses :func:`jax.lax.switch`, so only
      the selected law is evaluated at runtime.
    - For batched calls, all :math:`S^2` laws are evaluated for every pair and
      the result is selected afterwards, so the cost grows quadratically with
      the number of species.
    - :attr:`required_material_properties` is the union of the requirements of
      all laws in the table.
    """

    # `table` is a plain (pytree data) field, unlike `LawCombiner.laws` which
    # is static: the contained law dataclasses carry no array leaves, but
    # keeping the table as data lets the entries participate in tree mapping.
    table: tuple[tuple[ForceModel, ...], ...] = field(default=())
    """A symmetric :math:`S \\times S` table where entry ``table[a][b]`` is the :class:`ForceModel` governing interactions between species ``a`` and ``b``."""

    @property
    def requires_history(self) -> bool:
        return any(law.requires_history for row in self.table for law in row)

    @jax.jit(inline=True)
    def init_history(self, shape: tuple[int, ...]) -> Any:
        return tuple(
            tuple(
                law.init_history(shape) if law.requires_history else None for law in row
            )
            for row in self.table
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="ForceRouter.force_and_history")
    def force_and_history(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: Any,
    ) -> tuple[jax.Array, jax.Array, Any]:
        router = cast(ForceRouter, system.force_model)
        S = len(router.table)

        si = state.species_id[i]
        sj = state.species_id[j]
        idx = si * S + sj

        f_map = {}
        t_map = {}
        h_map = {}

        for a in range(S):
            for b in range(a, S):
                law = router.table[a][b]
                sys_law = dataclasses.replace(system, force_model=law)
                h_ab = history[a][b]

                if law.requires_history:
                    f, t, nh = law.force_and_history(i, j, pos, state, sys_law, h_ab)
                else:
                    f, t = law.force(i, j, pos, state, sys_law)
                    nh = h_ab

                f_map[(a, b)] = jnp.asarray(f, dtype=float)
                t_map[(a, b)] = jnp.asarray(t, dtype=float)

                mask = ((si == a) * (sj == b)) | ((si == b) * (sj == a))
                nh_flat, tree_def = jax.tree.flatten(nh)
                h_ab_flat, _ = jax.tree.flatten(h_ab)

                new_nh_flat: list[Any] = []
                for nh_arr, h_ab_arr in zip(nh_flat, h_ab_flat):
                    if nh_arr is None:
                        new_nh_flat.append(None)
                        continue
                    m = mask
                    while m.ndim < nh_arr.ndim:
                        m = m[..., None]
                    new_nh_flat.append(jnp.where(m, nh_arr, h_ab_arr))

                h_map[(a, b)] = jax.tree.unflatten(tree_def, new_nh_flat)

                if a != b:
                    f_map[(b, a)] = f_map[(a, b)]
                    t_map[(b, a)] = t_map[(a, b)]
                    h_map[(b, a)] = h_map[(a, b)]

        f_results = [f_map[(a, b)] for a in range(S) for b in range(S)]
        t_results = [t_map[(a, b)] for a in range(S) for b in range(S)]

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

        new_history = []
        for a in range(S):
            row = []
            for b in range(S):
                row.append(h_map[(a, b)])
            new_history.append(tuple(row))

        return f_final, t_final, tuple(new_history)

    @property
    def required_material_properties(self) -> tuple[str, ...]:
        """A static tuple of strings specifying the material properties required by this force model.

        The sorted union of the material properties required by all laws in the
        table. These properties must be present in the :attr:`System.mat_table`
        for the model to function correctly. This is used for validation.
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

        The mapping is symmetrized: entry ``(a, b)`` also populates ``(b, a)``.
        Pairs not present in the mapping default to an empty
        :class:`LawCombiner` (zero force, torque, and energy).

        Parameters
        ----------
        S : int
            Number of species. The resulting table has shape ``S x S``.
        mapping : dict[tuple[int, int], ForceModel]
            Mapping from species-index pairs to the force law governing
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
    @partial(jax.named_call, name="ForceRouter.force")
    def force(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute the force and torque acting on particle :math:`i` due to particle :math:`j` using the law selected by their species.

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
        Tuple[jax.Array, jax.Array]
            A tuple ``(force, torque)`` computed by the law at
            ``table[species_id[i]][species_id[j]]``.

        """
        router = cast(ForceRouter, system.force_model)
        S = len(router.table)

        si = state.species_id[i]
        sj = state.species_id[j]
        idx = si * S + sj

        f_map = {}
        t_map = {}

        # Evaluate upper triangle to save computation
        for a in range(S):
            for b in range(a, S):
                law = router.table[a][b]
                sys_law = dataclasses.replace(system, force_model=law)

                f, t = law.force(i, j, pos, state, sys_law)

                f_map[(a, b)] = jnp.asarray(f, dtype=float)
                t_map[(a, b)] = jnp.asarray(t, dtype=float)
                if a != b:
                    f_map[(b, a)] = f_map[(a, b)]
                    t_map[(b, a)] = t_map[(a, b)]

        f_results = [f_map[(a, b)] for a in range(S) for b in range(S)]
        t_results = [t_map[(a, b)] for a in range(S) for b in range(S)]

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

        return f_final, t_final

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
        """Compute the potential energy of the interaction between particle :math:`i` and particle :math:`j` using the law selected by their species.

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
            Scalar JAX array representing the potential energy computed by the
            law at ``table[species_id[i]][species_id[j]]``.

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
        e_final = jax.lax.select_n(idx, *e_results)
        return e_final


__all__ = ["ForceRouter"]
