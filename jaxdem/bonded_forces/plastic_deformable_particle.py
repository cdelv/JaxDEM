# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Implementation of the plastic deformable particle model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple, Optional, Dict, Self, Sequence, cast
from functools import partial

from . import BondedForceModel
from . import DeformableParticleModel
from ..utils.linalg import norm

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System
    from .force_manager import ForceFunction
    from .force_manager import EnergyFunction


@BondedForceModel.register("PlasticDeformableParticleModel")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PlasticDeformableParticleModel(BondedForceModel):
    r"""
    This model assumes triangular meshes for 3D bodies and linear segment
    meshes for 2D bodies.
    Meshes do not need to be closed, but content-based forces will fail if the
    mesh is not closed.
    Vertices should be ordered consistently (e.g., CCW) to define positive
    normals and bending angles.

    The container manages the mesh connectivity (`elements`, `edges`, etc.) and
    reference properties (initial measures, contents, lengths, angles) required
    to compute forces.
    It supports both 3D (volumetric bodies bounded by triangles) and 2D
    (planar bodies bounded by line segments).

    The general form of the deformable particle potential energy per particle is:

    .. math::
        &E_K = E_{K,measure} + E_{K,content} + E_{K,bending} + E_{K,edge} + E_{K,surface}

        &E_{K,measure} = \sum_{m} \frac{em_m}{2{\mathcal{M}_{m,0}}} \left(\mathcal{M}_m - \mathcal{M}_{m,0}\right)^2

        &E_{K,surface} = -\sum_{m} \gamma_m \mathcal{M}_m

        &E_{K,content} = \frac{e_c}{2{\mathcal{C}_{K,0}}} \left(\mathcal{C}_K - \mathcal{C}_{K,0}\right)^2

        &E_{K,bending} = \frac{1}{2} \sum_{a} eb_a wb_{a} \left(\theta_a -\theta_{a,0}\right)^2

        &E_{K,edge} = \frac{1}{2} \sum_{e} el_e \left(L_e - L_{e,0}\right)^2

    **Plasticity:**

    The plasticity comes from integrating a spring-dashpot equation for the initial length
    of each edge, evaluated at every force calculation step:

    .. math::
        L_{e,0}(t+dt) = L_{e,0}(t) + \frac{1}{\tau_{s,e}} (L_e(t) - L_{e,0}(t)) dt

    where :math:`L_{e,0}` is the initial (reference) edge length (``initial_edge_lengths``),
    :math:`L_e` is the current edge length, :math:`\tau_{s,e}` is the relaxation time (``tau_s``),
    and :math:`dt` is the simulation time step.

    **Definitions per Dimension:**

    * **3D:** Measure ($\mathcal{M}$) is Face Area; Content ($\mathcal{C}$) is
      Volume; Elements are Triangles.
    * **2D:** Measure ($\mathcal{M}$) is Segment Length; Content ($\mathcal{C}$)
      is Enclosed Area; Elements are Segments.

    Bending normalization is precomputed from the reference configuration:
    * **3D:** :math:`w_b = l_0 / h_0`, where :math:`l_0` is the initial shared-edge
      (hinge) length and :math:`h_0` is the initial distance between adjacent face
      centroids.
    * **2D:** :math:`w_b = 1/h_0`, with
      :math:`h_0 = 0.5 (L_{\text{left},0}+L_{\text{right},0})`, where
      :math:`L_{\text{left},0}` and :math:`L_{\text{right},0}` are initial lengths
      of adjacent segments.

    Shapes:
        - K: Number of deformable bodies, which can be defined by their vertices and connectivity (elements, edges, etc.)
        - M: Number of boundary elements (:math:`K \sum_K m_K`)
        - E: Number of unique edges (:math:`K \sum_K e_K`)
        - A: Number of element adjacencies (:math:`K \sum_K a_K`)

    All mesh properties of each body are concatenated along axis=0. We do not
    need to distinguish between bodies to compute forces, except for the
    content-related term, which requires the body ID for each element to sum up
    the content contributions per body.
    The body ID for each element is stored in `elements_id`, which maps each
    element to its corresponding body.

    Coefficient broadcasting:
    - Scalar coefficients are broadcast to the corresponding geometric entities.
      `em` and `gamma` are broadcast to shape `(M,)`, `eb` to `(A,)`, and `el`
      to `(E,)`.
    - `ec` is special: its shape is `(K,)`, one value per body, where `K` is the
      number of unique IDs in `elements_id`. `elements_id[m]` maps each element
      `m` to the body index used to read `ec`.
    """

    # --- Topology ---
    elements: Optional[jax.Array]
    """
    Array of vertex indices forming the boundary elements.
    Shape: (M, 3) for 3D (Triangles) or (M, 2) for 2D (Segments).
    Indices refer to the particle unique_id. Vertices correspond to `State.pos`.
    """

    edges: Optional[jax.Array]
    """
    Array of vertex indices forming the edges. Shape: (E, 2).
    Each row contains the indices of the two vertices forming the edge.
    Note: In 2D, the set of edges often overlaps with the set of elements (segments).
    """

    element_adjacency: Optional[jax.Array]
    """
    Array of element adjacency pairs (for bending/dihedral angles). Shape: (A, 2).
    Each row contains the indices of the two elements sharing a connection.
    """

    element_adjacency_edges: Optional[jax.Array]
    """
    Array of vertex IDs forming the shared edge for each adjacency. Shape: (A, 2).
    This can be independent from `edges` because we could have extra edge springs
    that do not correspond to the mesh connectivity.
    """

    # --- ID Mappings ---
    elements_id: Optional[jax.Array]
    """
    Array of body IDs for each boundary element. Shape: (M,).
    `elements_id[i] == k` means element `i` belongs to body `k`.
    """

    # --- Reference Configuration ---
    initial_body_contents: Optional[jax.Array]
    """
    Array of reference (stress-free) bulk content for each body. Shape: (K,).
    Represents Volume in 3D or Area in 2D.
    """

    initial_element_measures: Optional[jax.Array]
    """
    Array of reference (stress-free) measures for each element. Shape: (M,).
    Represents Area in 3D or Length in 2D.
    """

    initial_edge_lengths: Optional[jax.Array]
    """
    Array of reference (stress-free) lengths for each unique edge. Shape: (E,).
    """

    initial_bendings: Optional[jax.Array]
    """
    Array of reference (stress-free) bending angles for each adjacency. Shape: (A,).
    Represents Dihedral Angle in 3D or Vertex Angle in 2D.
    """

    w_b: Optional[jax.Array]
    """
    Precomputed bending normalization coefficient per adjacency. Shape: (A,).
    3D: hinge_length0 / dual_length0 with dual_length0 between face centers.
    2D: 1 / dual_length0 with dual_length0 = 0.5 * (L_left0 + L_right0).
    """

    # --- Coefficients ---
    em: Optional[jax.Array]
    """
    Measure elasticity coefficient for each element. Shape: (M,).
    (Controls Area stiffness in 3D; Length stiffness in 2D).
    """

    ec: Optional[jax.Array]
    """
    Content elasticity coefficient for each body. Shape: (K,).
    (Controls Volume stiffness in 3D; Area stiffness in 2D).
    """

    eb: Optional[jax.Array]
    """
    Bending elasticity coefficient for each hinge. Shape: (A,).
    """

    el: Optional[jax.Array]
    """
    Edge length elasticity coefficient for each edge. Shape: (E,).
    """

    gamma: Optional[jax.Array]
    """
    Surface/Line tension coefficient for each element. Shape: (M,).
    """

    tau_s: Optional[jax.Array]
    """
    Plastic relaxation time for each edge. Shape: (E,).
    """

    @classmethod
    def Create(
        cls,
        *,
        vertices: Optional[ArrayLike] = None,
        # topology
        elements: Optional[ArrayLike] = None,
        edges: Optional[ArrayLike] = None,
        element_adjacency: Optional[ArrayLike] = None,
        element_adjacency_edges: Optional[ArrayLike] = None,
        # ID mappings
        elements_id: Optional[ArrayLike] = None,
        # reference configuration
        initial_body_contents: Optional[ArrayLike] = None,
        initial_element_measures: Optional[ArrayLike] = None,
        initial_edge_lengths: Optional[ArrayLike] = None,
        initial_bendings: Optional[ArrayLike] = None,
        # coefficients
        em: Optional[ArrayLike] = None,
        ec: Optional[ArrayLike] = None,
        eb: Optional[ArrayLike] = None,
        el: Optional[ArrayLike] = None,
        gamma: Optional[ArrayLike] = None,
        tau_s: Optional[ArrayLike] = None,
        w_b: Optional[ArrayLike] = None,  # compatibility with checkpointer.
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
        if base.el is not None and tau_s is not None:
            tau_s_arr = jnp.asarray(tau_s, dtype=float)
            if tau_s_arr.ndim == 0 or tau_s_arr.shape == (1,):
                assert base.edges is not None
                tau_s_arr = jnp.full(base.edges.shape[:-1], tau_s_arr, dtype=float)
            assert base.edges is not None
            assert (
                tau_s_arr.shape == base.edges.shape[:-1]
            ), f"tau_s.shape={tau_s_arr.shape} does not match expected edges.shape[:-1]={base.edges.shape[:-1]}."

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
    @partial(jax.named_call, name="PlasticDeformableParticleModel.merge")
    def merge(
        model1: BondedForceModel,
        model2: BondedForceModel | Sequence[BondedForceModel],
    ) -> BondedForceModel:
        from .deformable_particle import _merge_metric_field

        def to_base(m: BondedForceModel) -> DeformableParticleModel:
            m = cast(PlasticDeformableParticleModel, m)
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
            models_to_merge_plastic = [cast(PlasticDeformableParticleModel, model2)]
            models_to_merge_base: Sequence[DeformableParticleModel] = [base_model2]
        else:
            base_model2 = [to_base(m) for m in model2]
            models_to_merge_plastic = [
                cast(PlasticDeformableParticleModel, m) for m in model2
            ]
            models_to_merge_base = base_model2

        base_merged = DeformableParticleModel.merge(base_model1, base_model2)
        base_merged = cast(DeformableParticleModel, base_merged)

        # Merge tau_s manually
        current_tau_s = cast(PlasticDeformableParticleModel, model1).tau_s
        n_edge_cur = 0 if base_model1.edges is None else int(base_model1.edges.shape[0])

        for nxt_plastic, nxt_base in zip(models_to_merge_plastic, models_to_merge_base):
            n_edge_new = 0 if nxt_base.edges is None else int(nxt_base.edges.shape[0])
            current_tau_s = _merge_metric_field(
                current_tau_s, nxt_plastic.tau_s, n_edge_cur, n_edge_new, 0.0
            )
            n_edge_cur += n_edge_new

        return PlasticDeformableParticleModel(
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
    @partial(jax.named_call, name="PlasticDeformableParticleModel.add")
    def add(
        model: BondedForceModel,
        **kwargs: Any,
    ) -> BondedForceModel:
        new_model = PlasticDeformableParticleModel.Create(**kwargs)
        return PlasticDeformableParticleModel.merge(model, new_model)

    @staticmethod
    @partial(
        jax.named_call,
        name="PlasticDeformableParticleModel.compute_potential_energy_w_aux",
    )
    def compute_potential_energy_w_aux(
        pos: jax.Array,
        state: State,
        system: System,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        return DeformableParticleModel.compute_potential_energy_w_aux(
            pos, state, system
        )

    @staticmethod
    @partial(
        jax.named_call, name="PlasticDeformableParticleModel.compute_potential_energy"
    )
    def compute_potential_energy(
        pos: jax.Array,
        state: State,
        system: System,
    ) -> jax.Array:
        return DeformableParticleModel.compute_potential_energy(pos, state, system)

    @staticmethod
    @partial(jax.named_call, name="PlasticDeformableParticleModel.compute_forces")
    def compute_forces(
        pos: jax.Array,
        state: State,
        system: System,
    ) -> Tuple[jax.Array, jax.Array]:
        dp_model = cast(PlasticDeformableParticleModel, system.bonded_force_model)

        if (
            dp_model.edges is not None
            and dp_model.initial_edge_lengths is not None
            and dp_model.tau_s is not None
        ):
            idx_map = (
                jnp.zeros((state.N,), dtype=int)
                .at[state.unique_id]
                .set(jnp.arange(state.N))
            )
            current_edge_indices = idx_map[dp_model.edges]
            v1 = pos[current_edge_indices[:, 0]]
            v2 = pos[current_edge_indices[:, 1]]
            edge_vector = v2 - v1
            edge_length = norm(edge_vector)

            L_e_0 = dp_model.initial_edge_lengths
            L_e = edge_length
            dt = system.dt

            dp_model.initial_edge_lengths = (
                L_e_0 + (1.0 / dp_model.tau_s) * (L_e - L_e_0) * dt
            )

        return DeformableParticleModel.compute_forces(pos, state, system)

    @property
    @partial(jax.named_call, name="PlasticDeformableParticleModel.force_and_energy_fns")
    def force_and_energy_fns(self) -> Tuple[ForceFunction, EnergyFunction, bool]:
        return (
            PlasticDeformableParticleModel.compute_forces,
            PlasticDeformableParticleModel.compute_potential_energy,
            False,
        )
