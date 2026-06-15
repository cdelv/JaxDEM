# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Implementation of the plastic perimeter deformable particle model."""

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
    _cat_optional,
    _merge_metric_field,
)
from ..utils.linalg import norm

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@BondedForceModel.register("PlasticPerimeterDeformableParticleModel")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PlasticPerimeterDeformableParticleModel(DeformableParticleModel):
    r"""Deformable particle model with perimeter-level plasticity.

    Elastic forces are identical to :class:`DeformableParticleModel` (individual
    edge springs, measure, content, bending, and surface-tension terms are all
    preserved; see its docstring for the full energy definition and shape
    conventions). The only difference is in the plastic update rule.

    Instead of relaxing each edge rest length independently (as in
    :class:`PlasticDeformableParticleModel`), the **total reference perimeter**
    of each body relaxes toward the current total perimeter, and the change is
    distributed back to individual edges by uniform rescaling:

    .. math::

        P_{K,0}^{\text{new}} &= P_{K,0} + \frac{1}{\tau_{s,K}} (P_K - P_{K,0})\,dt \\
        L_{e,0}^{\text{new}} &= L_{e,0}\;\frac{P_{K,0}^{\text{new}}}{P_{K,0}}
        \qquad \forall\, e \in K

    where :math:`P_K = \sum_{e \in K} L_e` is the current perimeter,
    :math:`P_{K,0} = \sum_{e \in K} L_{e,0}` the reference perimeter,
    :math:`\tau_{s,K}` the per-body relaxation time, and :math:`dt` the time
    step. Uniform rescaling preserves relative edge proportions within each body.

    This model requires an ``edges_id`` mapping (shape ``(E,)``) that assigns
    each edge to its owning body, analogous to ``elements_id`` for elements.
    When only a single body is present and ``edges_id`` is omitted, all edges
    are assumed to belong to body 0.

    Shapes (see :class:`DeformableParticleModel` for definitions of K, M, E, A):

    - ``edges_id``: ``(E,)``
    - ``tau_s``: ``(K,)``
    """

    edges_id: jax.Array | None = None
    """
    Array of body IDs for each edge. Shape: (E,).
    ``edges_id[e] == k`` means edge ``e`` belongs to body ``k``.
    Required for perimeter-level plasticity.
    """

    tau_s: jax.Array | None = None
    """
    Plastic relaxation time for each body. Shape: (K,).
    Controls how fast the total reference perimeter of each body relaxes toward
    the current perimeter.
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
        edges_id: ArrayLike | None = None,
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
        base = super(PlasticPerimeterDeformableParticleModel, cls).Create(
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

        edges_id_arr = None
        tau_s_arr = None
        if base.el is not None and tau_s is not None:
            assert base.edges is not None

            tau_s_arr = jnp.asarray(tau_s, dtype=float)
            tau_s_arr = jnp.atleast_1d(tau_s_arr)

            if edges_id is not None:
                edges_id_arr = jnp.asarray(edges_id, dtype=int)
                _, edges_id_arr = jnp.unique(edges_id_arr, return_inverse=True)
            else:
                edges_id_arr = jnp.zeros(base.edges.shape[0], dtype=int)

            assert (
                edges_id_arr.shape == base.edges.shape[:-1]
            ), f"edges_id.shape={edges_id_arr.shape} does not match edges.shape[:-1]={base.edges.shape[:-1]}."

            num_bodies = int(jnp.max(edges_id_arr)) + 1
            if tau_s_arr.ndim == 0 or tau_s_arr.shape == (1,):
                tau_s_arr = jnp.full((num_bodies,), tau_s_arr.squeeze(), dtype=float)
            assert tau_s_arr.shape == (
                num_bodies,
            ), f"tau_s.shape={tau_s_arr.shape} does not match number of edge-bodies ({num_bodies},)."

        return dataclasses.replace(base, edges_id=edges_id_arr, tau_s=tau_s_arr)

    @classmethod
    def _merge_extra_fields(
        cls,
        current: DeformableParticleModel,
        nxt: DeformableParticleModel,
    ) -> dict[str, Any]:
        current = cast(PlasticPerimeterDeformableParticleModel, current)
        nxt = cast(PlasticPerimeterDeformableParticleModel, nxt)

        n_edge_cur = 0 if current.edges is None else int(current.edges.shape[0])
        n_edge_new = 0 if nxt.edges is None else int(nxt.edges.shape[0])
        n_bodies_cur = _num_edge_bodies(current)
        n_bodies_new = _num_edge_bodies(nxt)

        cur_eid = current.edges_id
        new_eid = nxt.edges_id

        # If any side uses perimeter plasticity, we need edge-body IDs for all edges.
        need_edge_ids = current.tau_s is not None or nxt.tau_s is not None
        if need_edge_ids:
            if cur_eid is None and n_edge_cur > 0:
                cur_eid = jnp.zeros((n_edge_cur,), dtype=int)
                n_bodies_cur = max(n_bodies_cur, 1)
            if new_eid is None and n_edge_new > 0:
                new_eid = jnp.zeros((n_edge_new,), dtype=int)
                n_bodies_new = max(n_bodies_new, 1)

        if new_eid is not None and n_bodies_cur > 0:
            new_eid = new_eid + n_bodies_cur

        # Use inf as fill so that 1/tau_s = 0 (no plasticity) for padded bodies.
        merged_tau_s = _merge_metric_field(
            current.tau_s,
            nxt.tau_s,
            n_bodies_cur,
            n_bodies_new,
            float("inf"),
        )

        return {
            "edges_id": _cat_optional(cur_eid, new_eid),
            "tau_s": merged_tau_s,
        }

    def update_reference_state(
        self,
        pos: jax.Array,
        state: State,
        system: System,
    ) -> DeformableParticleModel:
        """Relax each body's reference perimeter toward its current perimeter."""
        if (
            self.edges is None
            or self.initial_edge_lengths is None
            or self.tau_s is None
            or self.edges_id is None
        ):
            return self

        idx_map = _build_idx_map(state)
        current_edge_indices = idx_map[self.edges]
        v1 = pos[current_edge_indices[:, 0]]
        v2 = pos[current_edge_indices[:, 1]]
        L_e = norm(v2 - v1)

        L_e_0 = self.initial_edge_lengths
        dt = system.dt
        num_bodies = self.tau_s.shape[0]

        P_K = jax.ops.segment_sum(L_e, self.edges_id, num_segments=num_bodies)
        P_K_0 = jax.ops.segment_sum(L_e_0, self.edges_id, num_segments=num_bodies)

        P_K_0_new = P_K_0 + (1.0 / self.tau_s) * (P_K - P_K_0) * dt

        scale = P_K_0_new / jnp.where(P_K_0 == 0.0, 1.0, P_K_0)
        new_initial_edge_lengths = L_e_0 * scale[self.edges_id]
        return dataclasses.replace(self, initial_edge_lengths=new_initial_edge_lengths)


@jax.jit(inline=True)
def _num_edge_bodies(model: PlasticPerimeterDeformableParticleModel) -> int:
    if model.tau_s is not None:
        return int(model.tau_s.shape[0])
    if model.edges_id is not None and model.edges_id.size > 0:
        return int(jnp.max(model.edges_id)) + 1
    return 0
