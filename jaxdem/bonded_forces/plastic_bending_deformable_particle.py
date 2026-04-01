# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Implementation of the deformable particle model with plastic bending angles."""

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
from .deformable_particle import compute_element_properties_2D, compute_element_properties_3D
from ..utils.linalg import cross, dot, norm, unit

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System
    from ..forces.force_manager import ForceFunction
    from ..forces.force_manager import EnergyFunction


@BondedForceModel.register("PlasticBendingDeformableParticleModel")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PlasticBendingDeformableParticleModel(BondedForceModel):
    r"""Deformable particle model with plastic bending angles.

    Elastic forces are identical to :class:`DeformableParticleModel` (individual
    edge springs, measure, content, bending, and surface-tension terms are all
    preserved). The only difference is in the plastic update rule for the
    reference bending angles.

    Instead of relaxing each edge rest length (as in
    :class:`PlasticDeformableParticleModel`), the **reference bending angle**
    of each adjacency relaxes toward the current bending angle:

    .. math::

        \theta_{a,0}(t+dt) = \theta_{a,0}(t)
            + \frac{1}{\tau_{s,a}} (\theta_a(t) - \theta_{a,0}(t))\,dt

    where :math:`\theta_{a,0}` is the reference (rest) bending angle
    (``initial_bendings``), :math:`\theta_a` is the current bending angle,
    :math:`\tau_{s,a}` is the relaxation time (``tau_s``), and :math:`dt` is the
    simulation time step.

    Shapes (see :class:`DeformableParticleModel` for definitions of K, M, E, A):

    - ``tau_s``: ``(A,)`` — one relaxation time per adjacency (bending hinge).
    """

    # --- Topology ---
    elements: jax.Array | None
    """
    Array of vertex indices forming the boundary elements.
    Shape: (M, 3) for 3D (Triangles) or (M, 2) for 2D (Segments).
    Indices refer to the particle unique_id. Vertices correspond to `State.pos`.
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
    This can be independent from `edges` because we could have extra edge springs
    that do not correspond to the mesh connectivity.
    """

    # --- ID Mappings ---
    elements_id: jax.Array | None
    """
    Array of body IDs for each boundary element. Shape: (M,).
    `elements_id[i] == k` means element `i` belongs to body `k`.
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
    Plastic relaxation time for each adjacency (bending hinge). Shape: (A,).
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

        tau_s_arr = None
        if base.eb is not None and tau_s is not None:
            tau_s_arr = jnp.asarray(tau_s, dtype=float)
            if tau_s_arr.ndim == 0 or tau_s_arr.shape == (1,):
                assert base.element_adjacency is not None
                tau_s_arr = jnp.full(base.element_adjacency.shape[:-1], tau_s_arr, dtype=float)
            assert base.element_adjacency is not None
            assert (
                tau_s_arr.shape == base.element_adjacency.shape[:-1]
            ), f"tau_s.shape={tau_s_arr.shape} does not match expected element_adjacency.shape[:-1]={base.element_adjacency.shape[:-1]}."

        return cls(
            elements=base.elements,
            edges=base.edges,
            element_adjacency=base.element_adjacency,
            element_adjacency_edges=base.element_adjacency_edges,
            elements_id=base.elements_id,
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
    @partial(jax.named_call, name="PlasticBendingDeformableParticleModel.merge")
    def merge(
        model1: BondedForceModel,
        model2: BondedForceModel | Sequence[BondedForceModel],
    ) -> BondedForceModel:
        from .deformable_particle import _merge_metric_field

        def to_base(m: BondedForceModel) -> DeformableParticleModel:
            m = cast(PlasticBendingDeformableParticleModel, m)
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
            models_to_merge_plastic = [cast(PlasticBendingDeformableParticleModel, model2)]
            models_to_merge_base: Sequence[DeformableParticleModel] = [base_model2]
        else:
            base_model2 = [to_base(m) for m in model2]
            models_to_merge_plastic = [
                cast(PlasticBendingDeformableParticleModel, m) for m in model2
            ]
            models_to_merge_base = base_model2

        base_merged = DeformableParticleModel.merge(base_model1, base_model2)
        base_merged = cast(DeformableParticleModel, base_merged)

        # Merge tau_s manually (per-adjacency)
        current_tau_s = cast(PlasticBendingDeformableParticleModel, model1).tau_s
        n_adj_cur = (
            0 if base_model1.element_adjacency is None
            else int(base_model1.element_adjacency.shape[0])
        )

        for nxt_plastic, nxt_base in zip(
            models_to_merge_plastic, models_to_merge_base, strict=False
        ):
            n_adj_new = (
                0 if nxt_base.element_adjacency is None
                else int(nxt_base.element_adjacency.shape[0])
            )
            current_tau_s = _merge_metric_field(
                current_tau_s, nxt_plastic.tau_s, n_adj_cur, n_adj_new, 0.0
            )
            n_adj_cur += n_adj_new

        return PlasticBendingDeformableParticleModel(
            elements=base_merged.elements,
            edges=base_merged.edges,
            element_adjacency=base_merged.element_adjacency,
            element_adjacency_edges=base_merged.element_adjacency_edges,
            elements_id=base_merged.elements_id,
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
    @partial(jax.named_call, name="PlasticBendingDeformableParticleModel.add")
    def add(
        model: BondedForceModel,
        **kwargs: Any,
    ) -> BondedForceModel:
        new_model = PlasticBendingDeformableParticleModel.Create(**kwargs)
        return PlasticBendingDeformableParticleModel.merge(model, new_model)

    @staticmethod
    @partial(
        jax.named_call,
        name="PlasticBendingDeformableParticleModel.compute_potential_energy_w_aux",
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
        jax.named_call, name="PlasticBendingDeformableParticleModel.compute_potential_energy"
    )
    def compute_potential_energy(
        pos: jax.Array,
        state: State,
        system: System,
    ) -> jax.Array:
        return DeformableParticleModel.compute_potential_energy(pos, state, system)

    @staticmethod
    @partial(jax.named_call, name="PlasticBendingDeformableParticleModel.compute_forces")
    def compute_forces(
        pos: jax.Array,
        state: State,
        system: System,
    ) -> tuple[jax.Array, jax.Array]:
        dp_model = cast(PlasticBendingDeformableParticleModel, system.bonded_force_model)
        dim = state.dim

        has_bending_reqs = (
            dp_model.elements is not None
            and dp_model.element_adjacency is not None
            and dp_model.eb is not None
            and dp_model.initial_bendings is not None
            and dp_model.w_b is not None
            and dp_model.tau_s is not None
        )
        if dim == 3 and has_bending_reqs:
            has_bending_reqs = dp_model.element_adjacency_edges is not None

        if has_bending_reqs:
            vertices = pos
            if dim == 3:
                compute_element_properties = compute_element_properties_3D
            else:
                compute_element_properties = compute_element_properties_2D

            idx_map = (
                jnp.zeros((state.N,), dtype=int)
                .at[state.unique_id]
                .set(jnp.arange(state.N))
            )
            current_element_indices = idx_map[cast(jax.Array, dp_model.elements)]
            element_normal, _, _ = jax.vmap(compute_element_properties)(
                vertices[current_element_indices]
            )

            element_adjacency = cast(jax.Array, dp_model.element_adjacency)
            n1 = element_normal[element_adjacency[:, 0]]
            n2 = element_normal[element_adjacency[:, 1]]
            cos = dot(n1, n2)

            if dim == 3:
                adjacency_edges = cast(jax.Array, dp_model.element_adjacency_edges)
                hinge_idx = idx_map[adjacency_edges]  # (A, 2)
                h_verts = vertices[hinge_idx]  # (A, 2, 3)
                tangent_vec = h_verts[:, 1, :] - h_verts[:, 0, :]
                tangent = unit(tangent_vec)
                cross_prod = cross(n1, n2)
                sin = dot(cross_prod, tangent)
            else:
                sin = cross(n1, n2)
                sin = jnp.squeeze(sin)

            theta = jnp.atan2(sin, cos)
            theta_0 = dp_model.initial_bendings
            dt = system.dt

            dp_model.initial_bendings = (
                theta_0 + (1.0 / dp_model.tau_s) * (theta - theta_0) * dt
            )

        return DeformableParticleModel.compute_forces(pos, state, system)

    @property
    @partial(jax.named_call, name="PlasticBendingDeformableParticleModel.force_and_energy_fns")
    def force_and_energy_fns(self) -> tuple[ForceFunction, EnergyFunction, bool]:
        return (
            PlasticBendingDeformableParticleModel.compute_forces,
            PlasticBendingDeformableParticleModel.compute_potential_energy,
            False,
        )
