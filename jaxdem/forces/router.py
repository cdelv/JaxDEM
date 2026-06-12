# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Force model router selecting laws based on species pairs."""

from __future__ import annotations

import dataclasses
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

    table: tuple[tuple[ForceModel, ...], ...] = field(default=(()))
    """A symmetric :math:`S \\times S` table where entry ``table[a][b]`` is the :class:`ForceModel` governing interactions between species ``a`` and ``b``."""

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

        if idx.ndim == 0:
            # Per-pair scalar call: dispatch with lax.switch so only the
            # selected law is evaluated at runtime (the symmetric duplicates in
            # the S x S table cost nothing).
            branches = []
            for a in range(S):
                for b in range(S):
                    law = router.table[a][b]
                    sys_law = dataclasses.replace(system, force_model=law)

                    def _branch(
                        law: ForceModel = law, sys_law: System = sys_law
                    ) -> tuple[jax.Array, jax.Array]:
                        f, t = law.force(i, j, pos, state, sys_law)
                        return jnp.asarray(f, dtype=float), jnp.asarray(t, dtype=float)

                    branches.append(_branch)
            return jax.lax.switch(idx, branches)

        all_f = []
        all_t = []
        for a in range(S):
            for b in range(S):
                law = router.table[a][b]
                sys_law = dataclasses.replace(system, force_model=law)
                f, t = law.force(i, j, pos, state, sys_law)
                all_f.append(f)
                all_t.append(t)

        f_shape = jnp.broadcast_shapes(*(f.shape for f in all_f))
        t_trail = jnp.broadcast_shapes(*(t.shape[-1:] for t in all_t))
        t_shape = f_shape[:-1] + t_trail
        stacked_f = jnp.stack([jnp.broadcast_to(f, f_shape) for f in all_f])
        stacked_t = jnp.stack([jnp.broadcast_to(t, t_shape) for t in all_t])

        n_idx = jnp.arange(idx.shape[0])
        return stacked_f[idx, n_idx], stacked_t[idx, n_idx]

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

        if idx.ndim == 0:
            # Per-pair scalar call: dispatch with lax.switch so only the
            # selected law is evaluated at runtime.
            branches = []
            for a in range(S):
                for b in range(S):
                    law = router.table[a][b]
                    sys_law = dataclasses.replace(system, force_model=law)
                    branches.append(
                        lambda law=law, sys_law=sys_law: jnp.asarray(
                            law.energy(i, j, pos, state, sys_law), dtype=float
                        )
                    )
            return jax.lax.switch(idx, branches)

        all_e = []
        for a in range(S):
            for b in range(S):
                law = router.table[a][b]
                sys_law = dataclasses.replace(system, force_model=law)
                e = law.energy(i, j, pos, state, sys_law)
                all_e.append(e)

        e_shape = jnp.broadcast_shapes(*(e.shape for e in all_e))
        stacked_e = jnp.stack([jnp.broadcast_to(e, e_shape) for e in all_e])

        n_idx = jnp.arange(idx.shape[0])
        return stacked_e[idx, n_idx]


__all__ = ["ForceRouter"]
