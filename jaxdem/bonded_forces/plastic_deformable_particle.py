# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Implementation of the plastic deformable particle model."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast

from . import BondedForceModel
from .deformable_particle import (
    DeformableParticleModel,
    _build_idx_map,
    _merge_metric_field,
)
from ..utils.linalg import norm

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@BondedForceModel.register("PlasticDeformableParticleModel")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PlasticDeformableParticleModel(DeformableParticleModel):
    r"""Deformable particle model with per-edge plasticity.

    Elastic forces, topology, reference configuration, and coefficient
    broadcasting are identical to :class:`DeformableParticleModel` (see its
    docstring for the full energy definition and shape conventions). The only
    difference is the plastic update rule.

    **Plasticity:**

    The plasticity comes from integrating a spring-dashpot equation for the initial length
    of each edge, evaluated at every force calculation step:

    .. math::
        L_{e,0}(t+dt) = L_{e,0}(t) + \frac{1}{\tau_{s,e}} (L_e(t) - L_{e,0}(t)) dt

    where :math:`L_{e,0}` is the initial (reference) edge length (``initial_edge_lengths``),
    :math:`L_e` is the current edge length, :math:`\tau_{s,e}` is the relaxation time (``tau_s``),
    and :math:`dt` is the simulation time step.

    Shapes (see :class:`DeformableParticleModel` for definitions of K, M, E, A):

    - ``tau_s``: ``(E,)`` — one relaxation time per edge.
    """

    tau_s: jax.Array | None = None
    """
    Plastic relaxation time for each edge. Shape: (E,).
    """

    @classmethod
    def Create(
        cls,
        *,
        vertices: ArrayLike | None = None,
        # topology
        elements: ArrayLike | None = None,
        edges: ArrayLike | None = None,
        element_adjacency: ArrayLike | None = None,
        element_adjacency_edges: ArrayLike | None = None,
        # ID mappings
        elements_id: ArrayLike | None = None,
        # reference configuration
        initial_body_contents: ArrayLike | None = None,
        initial_element_measures: ArrayLike | None = None,
        initial_edge_lengths: ArrayLike | None = None,
        initial_bendings: ArrayLike | None = None,
        # coefficients
        em: ArrayLike | None = None,
        ec: ArrayLike | None = None,
        eb: ArrayLike | None = None,
        el: ArrayLike | None = None,
        gamma: ArrayLike | None = None,
        tau_s: ArrayLike | None = None,
        w_b: ArrayLike | None = None,  # compatibility with checkpointer.
    ) -> Self:
        # Explicit two-arg super: dataclass(slots=True) recreates the class, so
        # zero-arg super()'s __class__ cell points at the discarded original
        # and raises TypeError on Python < 3.14.
        base = super(PlasticDeformableParticleModel, cls).Create(
            vertices=vertices,
            elements=elements,
            edges=edges,
            element_adjacency=element_adjacency,
            element_adjacency_edges=element_adjacency_edges,
            elements_id=elements_id,
            initial_body_contents=initial_body_contents,
            initial_element_measures=initial_element_measures,
            initial_edge_lengths=initial_edge_lengths,
            initial_bendings=initial_bendings,
            em=em,
            ec=ec,
            eb=eb,
            el=el,
            gamma=gamma,
            w_b=w_b,
        )

        tau_s_arr = None
        if base.el is not None and tau_s is not None:
            tau_s_arr = jnp.asarray(tau_s, dtype=float)
            if tau_s_arr.ndim == 0 or tau_s_arr.shape == (1,):
                assert base.edges is not None
                tau_s_arr = jnp.full(base.edges.shape[:-1], tau_s_arr, dtype=float)
            assert base.edges is not None
            assert (
                tau_s_arr.shape == base.edges.shape[:-1]
            ), f"tau_s.shape={tau_s_arr.shape} does not match expected edges.shape[:-1]={base.edges.shape[:-1]}."

        return dataclasses.replace(base, tau_s=tau_s_arr)

    @classmethod
    def _merge_extra_fields(
        cls,
        current: DeformableParticleModel,
        nxt: DeformableParticleModel,
    ) -> dict[str, Any]:
        current = cast(PlasticDeformableParticleModel, current)
        nxt = cast(PlasticDeformableParticleModel, nxt)
        n_edge_cur = 0 if current.edges is None else int(current.edges.shape[0])
        n_edge_new = 0 if nxt.edges is None else int(nxt.edges.shape[0])
        # Use inf as fill so that 1/tau_s = 0 (no plasticity) for padded edges.
        return {
            "tau_s": _merge_metric_field(
                current.tau_s,
                nxt.tau_s,
                n_edge_cur,
                n_edge_new,
                float("inf"),
            )
        }

    def update_reference_state(
        self,
        pos: jax.Array,
        state: State,
        system: System,
    ) -> DeformableParticleModel:
        """Relax each reference edge length toward the current edge length."""
        if (
            self.edges is None
            or self.initial_edge_lengths is None
            or self.tau_s is None
        ):
            return self

        idx_map = _build_idx_map(state)
        current_edge_indices = idx_map[self.edges]
        v1 = pos[current_edge_indices[:, 0]]
        v2 = pos[current_edge_indices[:, 1]]
        edge_length = norm(v2 - v1)

        L_e_0 = self.initial_edge_lengths
        new_initial_edge_lengths = (
            L_e_0 + (1.0 / self.tau_s) * (edge_length - L_e_0) * system.dt
        )
        return dataclasses.replace(self, initial_edge_lengths=new_initial_edge_lengths)
