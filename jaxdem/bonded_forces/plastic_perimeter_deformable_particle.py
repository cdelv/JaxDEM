# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Implementation of the plastic perimeter deformable particle model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast
from collections.abc import Sequence
from functools import partial

from . import BondedForceModel
from . import DeformableParticleModel
from ..utils.linalg import norm

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System
    from .force_manager import ForceFunction
    from .force_manager import EnergyFunction


@BondedForceModel.register("PlasticPerimeterDeformableParticleModel")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PlasticPerimeterDeformableParticleModel(BondedForceModel):
    r"""Deformable particle model with perimeter-level plasticity.

    Elastic forces are identical to :class:`DeformableParticleModel` (individual
    edge springs, measure, content, bending, and surface-tension terms are all
    preserved). The only difference is in the plastic update rule.

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

    # --- Topology ---
    elements: jax.Array | None
    """
    Array of vertex indices forming the boundary elements.
    Shape: (M, 3) for 3D (Triangles) or (M, 2) for 2D (Segments).
    Indices refer to the particle unique_id. Vertices correspond to ``State.pos``.
    """

    edges: jax.Array | None
    """
    Array of vertex indices forming the edges. Shape: (E, 2).
    Each row contains the indices of the two vertices forming the edge.
    Note: In 2D, the set of edges often overlaps with the set of elements (segments).
    """

    element_adjacency: jax.Array | None
    """
    Array of element adjacency pairs (for bending/dihedral angles). Shape: (A, 2).
    Each row contains the indices of the two elements sharing a connection.
    """

    element_adjacency_edges: jax.Array | None
    """
    Array of vertex IDs forming the shared edge for each adjacency. Shape: (A, 2).
    This can be independent from ``edges`` because we could have extra edge springs
    that do not correspond to the mesh connectivity.
    """

    # --- ID Mappings ---
    elements_id: jax.Array | None
    """
    Array of body IDs for each boundary element. Shape: (M,).
    ``elements_id[i] == k`` means element ``i`` belongs to body ``k``.
    """

    edges_id: jax.Array | None
    """
    Array of body IDs for each edge. Shape: (E,).
    ``edges_id[e] == k`` means edge ``e`` belongs to body ``k``.
    Required for perimeter-level plasticity.
    """

    # --- Reference Configuration ---
    initial_body_contents: jax.Array | None
    """
    Array of reference (stress-free) bulk content for each body. Shape: (K,).
    Represents Volume in 3D or Area in 2D.
    """

    initial_element_measures: jax.Array | None
    """
    Array of reference (stress-free) measures for each element. Shape: (M,).
    Represents Area in 3D or Length in 2D.
    """

    initial_edge_lengths: jax.Array | None
    """
    Array of reference (stress-free) lengths for each unique edge. Shape: (E,).
    """

    initial_bendings: jax.Array | None
    """
    Array of reference (stress-free) bending angles for each adjacency. Shape: (A,).
    Represents Dihedral Angle in 3D or Vertex Angle in 2D.
    """

    w_b: jax.Array | None
    """
    Precomputed bending normalization coefficient per adjacency. Shape: (A,).
    3D: hinge_length0 / dual_length0 with dual_length0 between face centers.
    2D: 1 / dual_length0 with dual_length0 = 0.5 * (L_left0 + L_right0).
    """

    # --- Coefficients ---
    em: jax.Array | None
    """
    Measure elasticity coefficient for each element. Shape: (M,).
    (Controls Area stiffness in 3D; Length stiffness in 2D).
    """

    ec: jax.Array | None
    """
    Content elasticity coefficient for each body. Shape: (K,).
    (Controls Volume stiffness in 3D; Area stiffness in 2D).
    """

    eb: jax.Array | None
    """
    Bending elasticity coefficient for each hinge. Shape: (A,).
    """

    el: jax.Array | None
    """
    Edge length elasticity coefficient for each edge. Shape: (E,).
    """

    gamma: jax.Array | None
    """
    Surface/Line tension coefficient for each element. Shape: (M,).
    """

    tau_s: jax.Array | None
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
        base = DeformableParticleModel.Create(
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

        return cls(
            elements=base.elements,
            edges=base.edges,
            element_adjacency=base.element_adjacency,
            element_adjacency_edges=base.element_adjacency_edges,
            elements_id=base.elements_id,
            edges_id=edges_id_arr,
            initial_body_contents=base.initial_body_contents,
            initial_element_measures=base.initial_element_measures,
            initial_edge_lengths=base.initial_edge_lengths,
            initial_bendings=base.initial_bendings,
            w_b=base.w_b,
            em=base.em,
            ec=base.ec,
            eb=base.eb,
            el=base.el,
            gamma=base.gamma,
            tau_s=tau_s_arr,
        )

    @staticmethod
    @partial(jax.named_call, name="PlasticPerimeterDeformableParticleModel.merge")
    def merge(
        model1: BondedForceModel,
        model2: BondedForceModel | Sequence[BondedForceModel],
    ) -> BondedForceModel:
        from .deformable_particle import _merge_metric_field, _cat_optional

        def to_base(m: BondedForceModel) -> DeformableParticleModel:
            m = cast(PlasticPerimeterDeformableParticleModel, m)
            return DeformableParticleModel(
                elements=m.elements,
                edges=m.edges,
                element_adjacency=m.element_adjacency,
                element_adjacency_edges=m.element_adjacency_edges,
                elements_id=m.elements_id,
                initial_body_contents=m.initial_body_contents,
                initial_element_measures=m.initial_element_measures,
                initial_edge_lengths=m.initial_edge_lengths,
                initial_bendings=m.initial_bendings,
                w_b=m.w_b,
                em=m.em,
                ec=m.ec,
                eb=m.eb,
                el=m.el,
                gamma=m.gamma,
            )

        base_model1 = to_base(model1)
        base_model2: DeformableParticleModel | Sequence[DeformableParticleModel]
        if isinstance(model2, BondedForceModel):
            base_model2 = to_base(model2)
            models_to_merge_plastic = [
                cast(PlasticPerimeterDeformableParticleModel, model2)
            ]
            models_to_merge_base: Sequence[DeformableParticleModel] = [base_model2]
        else:
            base_model2 = [to_base(m) for m in model2]
            models_to_merge_plastic = [
                cast(PlasticPerimeterDeformableParticleModel, m) for m in model2
            ]
            models_to_merge_base = base_model2

        base_merged = DeformableParticleModel.merge(base_model1, base_model2)
        base_merged = cast(DeformableParticleModel, base_merged)

        current_plastic = cast(PlasticPerimeterDeformableParticleModel, model1)
        current_tau_s = current_plastic.tau_s
        current_edges_id = current_plastic.edges_id
        n_bodies_cur = _num_edge_bodies(current_plastic)
        n_edge_cur = (
            0 if base_model1.edges is None else int(base_model1.edges.shape[0])
        )

        for nxt_plastic, nxt_base in zip(
            models_to_merge_plastic, models_to_merge_base, strict=False
        ):
            n_bodies_new = _num_edge_bodies(nxt_plastic)
            n_edge_new = 0 if nxt_base.edges is None else int(nxt_base.edges.shape[0])

            need_edge_ids = (
                current_tau_s is not None or nxt_plastic.tau_s is not None
            )

            cur_eid = current_edges_id
            new_eid = nxt_plastic.edges_id

            if need_edge_ids:
                if cur_eid is None and n_edge_cur > 0:
                    cur_eid = jnp.zeros((n_edge_cur,), dtype=int)
                    n_bodies_cur = max(n_bodies_cur, 1)
                if new_eid is None and n_edge_new > 0:
                    new_eid = jnp.zeros((n_edge_new,), dtype=int)
                    n_bodies_new = max(n_bodies_new, 1)

            if new_eid is not None and n_bodies_cur > 0:
                new_eid = new_eid + n_bodies_cur

            current_edges_id = _cat_optional(cur_eid, new_eid)

            # Use inf as fill so that 1/tau_s = 0 (no plasticity) for padded bodies.
            current_tau_s = _merge_metric_field(
                current_tau_s, nxt_plastic.tau_s, n_bodies_cur, n_bodies_new, float("inf")
            )

            n_bodies_cur += n_bodies_new
            n_edge_cur += n_edge_new

        return PlasticPerimeterDeformableParticleModel(
            elements=base_merged.elements,
            edges=base_merged.edges,
            element_adjacency=base_merged.element_adjacency,
            element_adjacency_edges=base_merged.element_adjacency_edges,
            elements_id=base_merged.elements_id,
            edges_id=current_edges_id,
            initial_body_contents=base_merged.initial_body_contents,
            initial_element_measures=base_merged.initial_element_measures,
            initial_edge_lengths=base_merged.initial_edge_lengths,
            initial_bendings=base_merged.initial_bendings,
            w_b=base_merged.w_b,
            em=base_merged.em,
            ec=base_merged.ec,
            eb=base_merged.eb,
            el=base_merged.el,
            gamma=base_merged.gamma,
            tau_s=current_tau_s,
        )

    @staticmethod
    @partial(jax.named_call, name="PlasticPerimeterDeformableParticleModel.add")
    def add(
        model: BondedForceModel,
        **kwargs: Any,
    ) -> BondedForceModel:
        new_model = PlasticPerimeterDeformableParticleModel.Create(**kwargs)
        return PlasticPerimeterDeformableParticleModel.merge(model, new_model)

    @staticmethod
    @partial(
        jax.named_call,
        name="PlasticPerimeterDeformableParticleModel.compute_potential_energy_w_aux",
    )
    def compute_potential_energy_w_aux(
        pos: jax.Array,
        state: State,
        system: System,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        return DeformableParticleModel.compute_potential_energy_w_aux(
            pos, state, system
        )

    @staticmethod
    @partial(
        jax.named_call,
        name="PlasticPerimeterDeformableParticleModel.compute_potential_energy",
    )
    def compute_potential_energy(
        pos: jax.Array,
        state: State,
        system: System,
    ) -> jax.Array:
        return DeformableParticleModel.compute_potential_energy(pos, state, system)

    @staticmethod
    @partial(
        jax.named_call,
        name="PlasticPerimeterDeformableParticleModel.compute_forces",
    )
    def compute_forces(
        pos: jax.Array,
        state: State,
        system: System,
    ) -> tuple[jax.Array, jax.Array]:
        dp_model = cast(
            PlasticPerimeterDeformableParticleModel, system.bonded_force_model
        )

        if (
            dp_model.edges is not None
            and dp_model.initial_edge_lengths is not None
            and dp_model.tau_s is not None
            and dp_model.edges_id is not None
        ):
            idx_map = (
                jnp.zeros((state.N,), dtype=int)
                .at[state.unique_id]
                .set(jnp.arange(state.N))
            )
            current_edge_indices = idx_map[dp_model.edges]
            v1 = pos[current_edge_indices[:, 0]]
            v2 = pos[current_edge_indices[:, 1]]
            L_e = norm(v2 - v1)

            L_e_0 = dp_model.initial_edge_lengths
            dt = system.dt
            num_bodies = dp_model.tau_s.shape[0]

            P_K = jax.ops.segment_sum(L_e, dp_model.edges_id, num_segments=num_bodies)
            P_K_0 = jax.ops.segment_sum(
                L_e_0, dp_model.edges_id, num_segments=num_bodies
            )

            P_K_0_new = P_K_0 + (1.0 / dp_model.tau_s) * (P_K - P_K_0) * dt

            scale = P_K_0_new / jnp.where(P_K_0 == 0.0, 1.0, P_K_0)
            dp_model.initial_edge_lengths = L_e_0 * scale[dp_model.edges_id]

        return DeformableParticleModel.compute_forces(pos, state, system)

    @property
    @partial(
        jax.named_call,
        name="PlasticPerimeterDeformableParticleModel.force_and_energy_fns",
    )
    def force_and_energy_fns(self) -> tuple[ForceFunction, EnergyFunction, bool]:
        return (
            PlasticPerimeterDeformableParticleModel.compute_forces,
            PlasticPerimeterDeformableParticleModel.compute_potential_energy,
            False,
        )


def _num_edge_bodies(model: PlasticPerimeterDeformableParticleModel) -> int:
    if model.tau_s is not None:
        return int(model.tau_s.shape[0])
    if model.edges_id is not None and model.edges_id.size > 0:
        return int(jnp.max(model.edges_id)) + 1
    return 0
