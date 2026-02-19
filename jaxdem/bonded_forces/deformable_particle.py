# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Implementation of the deformable particle model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Optional, Dict, Self, Sequence, cast
from functools import partial

from . import BonndedForceModel
from ..utils.linalg import cross, unit

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System
    from .force_manager import ForceFunction
    from .force_manager import EnergyFunction


@BonndedForceModel.register("DeformableParticleModel")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DeformableParticleModel(BonndedForceModel):
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

        &E_{K,measure} = \frac{1}{2} \sum_{m} em_m \mathcal{M}_{m,0} \left(\frac{\mathcal{M}_m}{\mathcal{M}_{m,0}} - 1\right)^2

        &E_{K,surface} = -\sum_{m} \gamma_m \mathcal{M}_m

        &E_{K,content} = \frac{e_c}{2} \mathcal{C}_{K,0} \left(\frac{\mathcal{C}_K}{\mathcal{C}_{K,0}} - 1\right)^2

        &E_{K,bending} = \frac{1}{2} \sum_{a} eb_a \frac{l_a}{h_a} \left(\theta_a/\theta_{a,0} - 1\right)^2

        &E_{K,edge} = \frac{1}{2} \sum_{e} el_e L_{e,0}\left(\frac{L_e}{L_{e,0}} - 1\right)^2

    **Definitions per Dimension:**

    * **3D:** Measure ($\mathcal{M}$) is Face Area; Content ($\mathcal{C}$) is
      Volume; Elements are Triangles.
    * **2D:** Measure ($\mathcal{M}$) is Segment Length; Content ($\mathcal{C}$)
      is Enclosed Area; Elements are Segments.

    The factor :math:`\frac{l_a}{h_a}` is the ratio between the hinge length and the dual length.
    The dual length is the distance between the two adjacent element centroids,
    and the hinge length is the length of the shared edge between the two adjacent elements.
    This factor is important to ensure that the bending energy scales correctly with mesh resolution.

    Shapes:
        - K: Number of deformable bodies, which can be defined by their vertices and connectivity (elements, edges, etc.)
        - M: Number of boundary elements (:math:`K \sum_K m_K`)
        - E: Number of unique edges (:math:`K \sum_K e_K`)
        - A: Number of element adjacencies (:math:`K \sum_K a_K`)

    All mesh properties of each body are concatenated along axis=0. We do not
    need to distinguish between bodies to compute forces, except for the
    content-related term, which requires the body ID for each element to sum up
    the content contributions per body.
    The body ID for each element is stored in `elements_ID`, which maps each
    element to its corresponding body.

    Coefficient broadcasting:
    - Scalar coefficients are broadcast to the corresponding geometric entities.
      `em` and `gamma` are broadcast to shape `(M,)`, `eb` to `(A,)`, and `el`
      to `(E,)`.
    - `ec` is special: its shape is `(K,)`, one value per body, where `K` is the
      number of unique IDs in `elements_ID`. `elements_ID[m]` maps each element
      `m` to the body index used to read `ec`.
    """

    # --- Topology ---
    elements: Optional[jax.Array]
    """
    Array of vertex indices forming the boundary elements.
    Shape: (M, 3) for 3D (Triangles) or (M, 2) for 2D (Segments).
    Indices refer to the particle unique_ID. Vertices correspond to `State.pos`.
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
    elements_ID: Optional[jax.Array]
    """
    Array of body IDs for each boundary element. Shape: (M,).
    `elements_ID[i] == k` means element `i` belongs to body `k`.
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
        elements_ID: Optional[ArrayLike] = None,
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
    ) -> Self:
        vertices = jnp.asarray(vertices, dtype=float) if vertices is not None else None
        elements = jnp.asarray(elements, dtype=int) if elements is not None else None
        edges = jnp.asarray(edges, dtype=int) if edges is not None else None
        element_adjacency = (
            jnp.asarray(element_adjacency, dtype=int)
            if element_adjacency is not None
            else None
        )
        element_adjacency_edges = (
            jnp.asarray(element_adjacency_edges, dtype=int)
            if element_adjacency_edges is not None
            else None
        )
        if elements_ID is not None:
            elements_ID = jnp.asarray(elements_ID, dtype=int)
            _, elements_ID = jnp.unique(elements_ID, return_inverse=True)
        initial_body_contents = (
            jnp.asarray(initial_body_contents, dtype=float)
            if initial_body_contents is not None
            else None
        )
        initial_element_measures = (
            jnp.asarray(initial_element_measures, dtype=float)
            if initial_element_measures is not None
            else None
        )
        initial_edge_lengths = (
            jnp.asarray(initial_edge_lengths, dtype=float)
            if initial_edge_lengths is not None
            else None
        )
        initial_bendings = (
            jnp.asarray(initial_bendings, dtype=float)
            if initial_bendings is not None
            else None
        )
        em = jnp.asarray(em, dtype=float) if em is not None else None
        ec = jnp.asarray(ec, dtype=float) if ec is not None else None
        eb = jnp.asarray(eb, dtype=float) if eb is not None else None
        el = jnp.asarray(el, dtype=float) if el is not None else None
        gamma = jnp.asarray(gamma, dtype=float) if gamma is not None else None
        em = jnp.atleast_1d(em) if em is not None else None
        ec = jnp.atleast_1d(ec) if ec is not None else None
        eb = jnp.atleast_1d(eb) if eb is not None else None
        el = jnp.atleast_1d(el) if el is not None else None
        gamma = jnp.atleast_1d(gamma) if gamma is not None else None

        # Check if we need to compute mesh properties
        if (
            (em is not None and initial_element_measures is None)
            or (ec is not None and initial_body_contents is None)
            or (eb is not None and initial_bendings is None)
        ):
            assert (
                vertices is not None and elements is not None
            ), "Vertices and elements must be provided to compute initial measures, contents, or bending."
            assert (
                vertices.shape[-1] == 2 or vertices.shape[-1] == 3
            ), f"vertices.shape[-1]={vertices.shape[-1]}, but should be 2 or 3."
            assert (
                elements.shape[-1] == 2 or elements.shape[-1] == 3
            ), f"elements.shape[-1]={elements.shape[-1]}, but should be 2 or 3."
            assert (
                vertices.shape[-1] == elements.shape[-1]
            ), f"vertices.shape[-1]={vertices.shape[-1]} does not match elements.shape[-1]={elements.shape[-1]}."
            compute_fn = (
                compute_element_properties_3D
                if vertices.shape[-1] == 3
                else compute_element_properties_2D
            )
            (
                initial_element_normals,
                computed_initial_element_measures,
                partial_body_contents,
            ) = jax.vmap(compute_fn)(vertices[elements])

            if em is not None and initial_element_measures is None:
                initial_element_measures = computed_initial_element_measures

            if ec is not None and initial_body_contents is None:
                elements_arr = elements
                if elements_ID is None:
                    assert ec.shape == (
                        1,
                    ), f"Content elasticity coefficient (ec) has shape {ec.shape}, but should have shape (..., 1) if elements_ID is not provided."
                    elements_ID = jnp.zeros(elements_arr.shape[0], dtype=int)
                elements_id_arr = cast(jax.Array, elements_ID)
                assert (
                    elements_id_arr.shape == elements_arr.shape[:-1]
                ), f"elements_ID.shape={elements_id_arr.shape} does not match elements.shape[:-1]={elements_arr.shape[:-1]}. elements.shape={elements_arr.shape}."
                assert jnp.max(elements_id_arr) + 1 == ec.shape[-1], (
                    "Number of unique body IDs in elements_ID does not match "
                    "number of content elasticity coefficients (ec). "
                    f"ec.shape={ec.shape}."
                )
                initial_body_contents = jax.ops.segment_sum(
                    partial_body_contents, elements_id_arr, num_segments=ec.shape[-1]
                )

        # Contents
        if ec is not None and initial_body_contents is not None:
            assert (
                elements is not None
            ), "Content elasticity coefficient (ec) provided but elements is None."
            assert (
                elements_ID is not None
            ), "Content elasticity coefficient (ec) provided but elements_ID is None."
            assert (
                elements.shape[-1] == 2 or elements.shape[-1] == 3
            ), f"elements.shape={elements.shape}, but should have shape=(..., M, 2 or 3)."
            assert (
                jnp.max(elements_ID) + 1 == ec.shape[-1]
            ), f"Number of unique body IDs in elements_ID does not match number of content elasticity coefficients (ec). ec.shape={ec.shape}."
            assert (
                initial_body_contents.shape[-1] == ec.shape[-1]
            ), f"initial_body_contents.shape[-1]={initial_body_contents.shape[-1]} does not match ec.shape[-1]={ec.shape[-1]}."

        # Measure
        if em is not None:
            assert (
                elements is not None
            ), "Measure elasticity coefficient (em) provided but elements is None."
            assert (
                elements.shape[-1] == 2 or elements.shape[-1] == 3
            ), f"elements.shape={elements.shape}, but should have shape=(..., M, 2 or 3)."
            if em.ndim == 0 or em.shape == (1,):
                em = jnp.full(elements.shape[:-1], em, dtype=float)
            assert (
                em.shape == elements.shape[:-1]
            ), f"em.shape={em.shape} does not match expected element.shape[:-1]={elements.shape[:-1]}. elements.shape={elements.shape}."

        # Bending
        if eb is not None:
            assert (
                element_adjacency is not None
            ), "Bending elasticity coefficient (eb) provided but element_adjacency is None."
            assert (
                elements is not None
            ), "Bending elasticity coefficient (eb) provided but elements is None."
            assert (
                element_adjacency.shape[-1] == 2
            ), f"element_adjacency.shape={element_adjacency.shape}, but should have shape=(..., A, 2)."
            if eb.ndim == 0 or eb.shape == (1,):
                eb = jnp.full(element_adjacency.shape[:-1], eb, dtype=float)
            if elements.shape[-1] == 3 and element_adjacency_edges is None:
                f1 = elements[element_adjacency[:, 0]]
                matches = (
                    f1[:, :, None] == elements[element_adjacency[:, 1]][:, None, :]
                )
                missing_idx = jnp.argmin(jnp.any(matches, axis=2).astype(int), axis=1)
                v_start = jnp.take_along_axis(
                    f1, ((missing_idx + 1) % 3)[:, None], axis=1
                )
                v_end = jnp.take_along_axis(
                    f1, ((missing_idx + 2) % 3)[:, None], axis=1
                )
                element_adjacency_edges = jnp.concatenate([v_start, v_end], axis=1)
                assert (
                    element_adjacency_edges.shape == element_adjacency.shape
                ), f"element_adjacency_edges.shape={element_adjacency_edges.shape} does not match expected element_adjacency.shape={element_adjacency.shape}."
            if initial_bendings is None:
                n1 = initial_element_normals[element_adjacency[:, 0]]
                n2 = initial_element_normals[element_adjacency[:, 1]]
                initial_bendings = angle_between_normals(n1, n2)
            assert (
                initial_bendings.shape == element_adjacency.shape[:-1]
            ), f"initial_bendings.shape={initial_bendings.shape}, but should have element_adjacency.shape[:-1]={element_adjacency.shape[:-1]}. element_adjacency.shape={element_adjacency.shape}."
            assert (
                eb.shape == element_adjacency.shape[:-1]
            ), f"eb.shape={eb.shape} does not match expected element_adjacency.shape[:-1]={element_adjacency.shape[:-1]}. element_adjacency.shape={element_adjacency.shape}."

        # Surface tension
        if gamma is not None:
            assert (
                elements is not None
            ), "Surface tension coefficient (gamma) provided but elements is None."
            assert (
                elements.shape[-1] == 2 or elements.shape[-1] == 3
            ), f"elements.shape={elements.shape}, but should have shape=(..., M, 2 or 3)."
            if gamma.ndim == 0 or gamma.shape == (1,):
                gamma = jnp.full(elements.shape[:-1], gamma, dtype=float)
            assert (
                gamma.shape == elements.shape[:-1]
            ), f"gamma.shape={gamma.shape} does not match expected element.shape[:-1]={elements.shape[:-1]}. elements.shape={elements.shape}."

        # Edges
        if el is not None:
            assert (
                edges is not None
            ), "Edge elasticity coefficient (el) provided but edges is None."
            assert (
                edges.shape[-1] == 2
            ), f"edges.shape={edges.shape}, but should have shape=(..., E, 2)."
            if el.ndim == 0 or el.shape == (1,):
                el = jnp.full(edges.shape[:-1], el, dtype=float)
            if initial_edge_lengths is None:
                assert (
                    vertices is not None
                ), "Vertices must be provided to compute initial edge lengths."
                v1 = vertices[edges[:, 0]]
                v2 = vertices[edges[:, 1]]
                v12 = v2 - v1
                initial_edge_lengths = jnp.sum(v12 * v12, axis=-1)
                initial_edge_lengths = jnp.sqrt(initial_edge_lengths)
            assert (
                initial_edge_lengths.shape == edges.shape[:-1]
            ), f"initial_edge_lengths.shape={initial_edge_lengths.shape}, but should have edges.shape[:-1]={edges.shape[:-1]}. edges.shape={edges.shape}."
            assert (
                el.shape == edges.shape[:-1]
            ), f"el.shape={el.shape} does not match expected edges.shape[:-1]={edges.shape[:-1]}. edges.shape={edges.shape}."

        # Keep only data required by active terms, even if the user provided extra fields.
        need_elements = any(x is not None for x in (em, ec, eb, gamma))
        need_edges = el is not None
        need_adjacency = eb is not None
        need_elements_ID = ec is not None
        need_initial_body_contents = ec is not None
        need_initial_element_measures = em is not None

        if not need_elements:
            elements = None
        if not need_edges:
            edges = None
            initial_edge_lengths = None
        if not need_adjacency:
            element_adjacency = None
            element_adjacency_edges = None
            initial_bendings = None
        if not need_elements_ID:
            elements_ID = None
        if not need_initial_body_contents:
            initial_body_contents = None
        if not need_initial_element_measures:
            initial_element_measures = None

        return cls(
            elements=elements,
            edges=edges,
            element_adjacency=element_adjacency,
            element_adjacency_edges=element_adjacency_edges,
            elements_ID=(
                jnp.asarray(elements_ID, dtype=int) if elements_ID is not None else None
            ),
            initial_body_contents=initial_body_contents,
            initial_element_measures=initial_element_measures,
            initial_edge_lengths=initial_edge_lengths,
            initial_bendings=initial_bendings,
            em=em,
            ec=ec,
            eb=eb,
            el=el,
            gamma=gamma,
        )

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleModel.merge")
    def merge(
        model1: DeformableParticleModel,
        model2: DeformableParticleModel | Sequence[DeformableParticleModel],
    ) -> DeformableParticleModel:
        models_to_merge = (
            [model2] if isinstance(model2, DeformableParticleModel) else list(model2)
        )
        current = model1

        for nxt in models_to_merge:
            vertex_offset = _max_vertex_id(current) + 1
            element_offset = (
                0 if current.elements is None else int(current.elements.shape[0])
            )

            n_current = _num_bodies(current)
            n_new = _num_bodies(nxt)

            # If any side uses content terms, we need body IDs for all elements.
            need_body_ids = (
                current.ec is not None
                or nxt.ec is not None
                or current.initial_body_contents is not None
                or nxt.initial_body_contents is not None
            )
            cur_elements_ID = current.elements_ID
            new_elements_ID = nxt.elements_ID
            if need_body_ids:
                if cur_elements_ID is None and current.elements is not None:
                    cur_elements_ID = jnp.zeros((current.elements.shape[0],), dtype=int)
                    n_current = max(n_current, 1)
                if new_elements_ID is None and nxt.elements is not None:
                    new_elements_ID = jnp.zeros((nxt.elements.shape[0],), dtype=int)
                    n_new = max(n_new, 1)

            body_offset = n_current
            if new_elements_ID is not None and body_offset > 0:
                new_elements_ID = new_elements_ID + body_offset

            next_elements = (
                nxt.elements + vertex_offset
                if nxt.elements is not None and vertex_offset > 0
                else nxt.elements
            )
            next_edges = (
                nxt.edges + vertex_offset
                if nxt.edges is not None and vertex_offset > 0
                else nxt.edges
            )
            next_adjacency = (
                nxt.element_adjacency + element_offset
                if nxt.element_adjacency is not None and element_offset > 0
                else nxt.element_adjacency
            )
            next_adj_edges = (
                nxt.element_adjacency_edges + vertex_offset
                if nxt.element_adjacency_edges is not None and vertex_offset > 0
                else nxt.element_adjacency_edges
            )

            # Per-element/edge/adjacency fields: pad missing terms.
            n_elem_cur = (
                0 if current.elements is None else int(current.elements.shape[0])
            )
            n_elem_new = 0 if nxt.elements is None else int(nxt.elements.shape[0])
            n_edge_cur = 0 if current.edges is None else int(current.edges.shape[0])
            n_edge_new = 0 if nxt.edges is None else int(nxt.edges.shape[0])
            n_adj_cur = (
                0
                if current.element_adjacency is None
                else int(current.element_adjacency.shape[0])
            )
            n_adj_new = (
                0
                if nxt.element_adjacency is None
                else int(nxt.element_adjacency.shape[0])
            )

            merged_em = _merge_metric_field(
                current.em, nxt.em, n_elem_cur, n_elem_new, 0.0
            )
            merged_gamma = _merge_metric_field(
                current.gamma, nxt.gamma, n_elem_cur, n_elem_new, 0.0
            )
            merged_initial_element_measures = _merge_metric_field(
                current.initial_element_measures,
                nxt.initial_element_measures,
                n_elem_cur,
                n_elem_new,
                1.0,
            )
            merged_el = _merge_metric_field(
                current.el, nxt.el, n_edge_cur, n_edge_new, 0.0
            )
            merged_initial_edge_lengths = _merge_metric_field(
                current.initial_edge_lengths,
                nxt.initial_edge_lengths,
                n_edge_cur,
                n_edge_new,
                1.0,
            )
            merged_eb = _merge_metric_field(
                current.eb, nxt.eb, n_adj_cur, n_adj_new, 0.0
            )
            merged_initial_bendings = _merge_metric_field(
                current.initial_bendings,
                nxt.initial_bendings,
                n_adj_cur,
                n_adj_new,
                1.0,
            )

            # Per-body fields (ec/content): pad missing with zero/one.
            merged_ec = _merge_metric_field(current.ec, nxt.ec, n_current, n_new, 0.0)
            merged_initial_body_contents = _merge_metric_field(
                current.initial_body_contents,
                nxt.initial_body_contents,
                n_current,
                n_new,
                1.0,
            )

            current = DeformableParticleModel(
                elements=_cat_optional(current.elements, next_elements),
                edges=_cat_optional(current.edges, next_edges),
                element_adjacency=_cat_optional(
                    current.element_adjacency, next_adjacency
                ),
                element_adjacency_edges=_cat_optional(
                    current.element_adjacency_edges, next_adj_edges
                ),
                elements_ID=_cat_optional(cur_elements_ID, new_elements_ID),
                initial_body_contents=merged_initial_body_contents,
                initial_element_measures=merged_initial_element_measures,
                initial_edge_lengths=merged_initial_edge_lengths,
                initial_bendings=merged_initial_bendings,
                em=merged_em,
                ec=merged_ec,
                eb=merged_eb,
                el=merged_el,
                gamma=merged_gamma,
            )

        return current

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleModel.add")
    def add(
        model: DeformableParticleModel,
        *,
        vertices: Optional[ArrayLike] = None,
        elements: Optional[ArrayLike] = None,
        edges: Optional[ArrayLike] = None,
        element_adjacency: Optional[ArrayLike] = None,
        element_adjacency_edges: Optional[ArrayLike] = None,
        elements_ID: Optional[ArrayLike] = None,
        initial_body_contents: Optional[ArrayLike] = None,
        initial_element_measures: Optional[ArrayLike] = None,
        initial_edge_lengths: Optional[ArrayLike] = None,
        initial_bendings: Optional[ArrayLike] = None,
        em: Optional[ArrayLike] = None,
        ec: Optional[ArrayLike] = None,
        eb: Optional[ArrayLike] = None,
        el: Optional[ArrayLike] = None,
        gamma: Optional[ArrayLike] = None,
    ) -> DeformableParticleModel:
        new_model = DeformableParticleModel.Create(
            vertices=vertices,
            elements=elements,
            edges=edges,
            element_adjacency=element_adjacency,
            element_adjacency_edges=element_adjacency_edges,
            elements_ID=elements_ID,
            initial_body_contents=initial_body_contents,
            initial_element_measures=initial_element_measures,
            initial_edge_lengths=initial_edge_lengths,
            initial_bendings=initial_bendings,
            em=em,
            ec=ec,
            eb=eb,
            el=el,
            gamma=gamma,
        )
        return DeformableParticleModel.merge(model, new_model)

    @staticmethod
    @partial(
        jax.named_call, name="DeformableParticleModel.compute_potential_energy_w_aux"
    )
    def compute_potential_energy_w_aux(
        pos: jax.Array,
        state: State,
        system: System,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        dp_model = cast(DeformableParticleModel, system.bonded_force_model)
        vertices = pos
        dim = state.dim
        if dim == 3:
            compute_element_properties = compute_element_properties_3D
        elif dim == 2:
            compute_element_properties = compute_element_properties_2D
        else:
            raise ValueError(
                f"DeformableParticleModel only supports 2D or 3D, got dim={dim}."
            )
        idx_map = (
            jnp.zeros((state.N,), dtype=int)
            .at[state.unique_ID]
            .set(jnp.arange(state.N))
        )
        E_element = jnp.array(0.0, dtype=float)
        E_content = jnp.array(0.0, dtype=float)
        E_gamma = jnp.array(0.0, dtype=float)
        E_bending = jnp.array(0.0, dtype=float)
        E_edge = jnp.array(0.0, dtype=float)

        if dp_model.elements is not None:
            current_element_indices = idx_map[dp_model.elements]
            element_normal, element_measure, partial_content = jax.vmap(
                compute_element_properties
            )(vertices[current_element_indices])

        # Content
        if (
            dp_model.elements is not None
            and dp_model.elements_ID is not None
            and dp_model.ec is not None
            and dp_model.initial_body_contents is not None
        ):
            # 1/2 * sum_K ec_K * C_{K,0} * (C_K/C_{K,0} - 1)^2
            body_content = jax.ops.segment_sum(
                partial_content,
                dp_model.elements_ID,
                num_segments=dp_model.ec.shape[-1],
            )
            norm_content_energy = (
                dp_model.ec
                * jnp.square(body_content - dp_model.initial_body_contents)
                / dp_model.initial_body_contents
            )
            E_content = jnp.sum(norm_content_energy) / 2

        # Measure
        if (
            dp_model.elements is not None
            and dp_model.em is not None
            and dp_model.initial_element_measures is not None
        ):
            # 1/2 * sum_m em_m * (M_m - M_{m,0})^2 / M_{m,0}
            norm_measure_energy = (
                dp_model.em
                * jnp.square(element_measure - dp_model.initial_element_measures)
                / dp_model.initial_element_measures
            )
            E_element = jnp.sum(norm_measure_energy) / 2

        # Surface tension
        if dp_model.elements is not None and dp_model.gamma is not None:
            # - sum_m gamma_m * M_m
            E_gamma = -jnp.sum(dp_model.gamma * element_measure)

        # Bending
        if (
            dp_model.elements is not None
            and dp_model.element_adjacency is not None
            and dp_model.element_adjacency_edges is not None
            and dp_model.eb is not None
            and dp_model.initial_bendings is not None
        ):
            # 1/2 * sum_a eb_a * l_a/h_a * (theta_a - theta_{a,0})^2
            n1 = element_normal[dp_model.element_adjacency[:, 0]]
            n2 = element_normal[dp_model.element_adjacency[:, 1]]
            cos = jnp.sum(n1 * n2, axis=-1)

            if dim == 3:
                hinge_idx = idx_map[dp_model.element_adjacency_edges]  # (A, 2)
                h_verts = vertices[hinge_idx]  # (A, 2, 3)
                tangent_vec = h_verts[:, 1, :] - h_verts[:, 0, :]
                tangent = unit(tangent_vec)
                cross_prod = cross(n1, n2)
                sin = jnp.sum(cross_prod * tangent, axis=-1)
            else:
                sin = cross(n1, n2)
                sin = jnp.squeeze(sin)

            bending = jnp.atan2(sin, cos)

            # compute scaling factor l_a/h_a
            adj_elem_1 = current_element_indices[dp_model.element_adjacency[:, 0]]
            adj_elem_2 = current_element_indices[dp_model.element_adjacency[:, 1]]
            C1 = jnp.mean(vertices[adj_elem_1], axis=-2)
            C2 = jnp.mean(vertices[adj_elem_2], axis=-2)
            dual_length = jnp.sum((C2 - C1) ** 2, axis=-1)
            dual_length = jnp.sqrt(dual_length)
            adj_edge_idx = idx_map[dp_model.element_adjacency_edges]
            hinge_length = vertices[adj_edge_idx[:, 1]] - vertices[adj_edge_idx[:, 0]]
            hinge_length = jnp.sum(hinge_length * hinge_length, axis=-1)
            hinge_length = jnp.sqrt(hinge_length)
            bending_scaling = hinge_length / dual_length
            norm_bending_energy = (
                dp_model.eb
                * bending_scaling
                * jnp.square(bending - dp_model.initial_bendings)
            )
            E_bending = jnp.sum(norm_bending_energy) / 2

        # Edges
        if (
            dp_model.edges is not None
            and dp_model.el is not None
            and dp_model.initial_edge_lengths is not None
        ):
            # 1/2 * sum_e el_e * (L_e - L_{e,0})^2 / L_{e,0}
            current_edge_indices = idx_map[dp_model.edges]
            v1 = vertices[current_edge_indices[:, 0]]
            v2 = vertices[current_edge_indices[:, 1]]
            edge_vector = v2 - v1
            edge_length = jnp.sum(edge_vector * edge_vector, axis=-1)
            edge_length = jnp.sqrt(edge_length)
            norm_edge_strain_energy = (
                dp_model.el
                * jnp.square(edge_length - dp_model.initial_edge_lengths)
                / dp_model.initial_edge_lengths
            )
            E_edge = jnp.sum(norm_edge_strain_energy) / 2

        aux = dict(
            E_content=E_content,
            E_element=E_element,
            E_gamma=E_gamma,
            E_bending=E_bending,
            E_edge=E_edge,
        )

        return E_content + E_element + E_gamma + E_bending + E_edge, aux

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleModel.compute_potential_energy")
    def compute_potential_energy(
        pos: jax.Array,
        state: State,
        system: System,
    ) -> jax.Array:
        pe_energy, _ = DeformableParticleModel.compute_potential_energy_w_aux(
            pos, state, system
        )
        return pe_energy

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleModel.compute_forces")
    def compute_forces(
        pos: jax.Array,
        state: State,
        system: System,
    ) -> Tuple[jax.Array, jax.Array]:
        force = -jax.grad(DeformableParticleModel.compute_potential_energy)(
            pos, state, system
        )
        return force, jnp.zeros_like(state.torque)

    @property
    @partial(jax.named_call, name="DeformableParticleModel.force_and_energy_fns")
    def force_and_energy_fns(self) -> Tuple[ForceFunction, EnergyFunction, bool]:
        return (
            DeformableParticleModel.compute_forces,
            DeformableParticleModel.compute_potential_energy,
            False,
        )


@partial(jax.named_call, name="DeformableParticleModel.angle_between_normals")
def angle_between_normals(n1: jax.Array, n2: jax.Array) -> jax.Array:
    cos = jnp.sum(n1 * n2, axis=-1)
    sin = cross(n1, n2)
    if sin.ndim > cos.ndim:
        sin = jnp.sum(sin * sin, axis=-1)
        sin = jnp.sqrt(sin)
    return jnp.atan2(sin, cos)


@partial(jax.named_call, name="DeformableParticleModel.compute_element_properties_3D")
def compute_element_properties_3D(
    simplex: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r1 = simplex[0]
    r2 = simplex[1] - simplex[0]
    r3 = simplex[2] - simplex[0]
    face_normal = cross(r2, r3) / 2
    partial_vol = jnp.sum(face_normal * r1, axis=-1) / 3
    area_face2 = jnp.sum(face_normal * face_normal, axis=-1)
    area_face = jnp.sqrt(area_face2)
    return (
        face_normal / jnp.where(area_face == 0, 1, area_face),
        area_face,
        partial_vol,
    )


@partial(jax.named_call, name="DeformableParticleModel.compute_element_properties_2D")
def compute_element_properties_2D(
    simplex: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r1 = simplex[0]
    r2 = simplex[1]
    edge = r2 - r1
    length2 = jnp.sum(edge * edge, axis=-1)
    length = jnp.sqrt(length2)
    normal = jnp.array([edge[1], -edge[0]])
    normal /= jnp.where(length == 0, 1.0, length)
    partial_area = 0.5 * (r1[0] * r2[1] - r1[1] * r2[0])
    return normal, length, partial_area


@partial(jax.named_call, name="DeformableParticleModel._cat_optional")
def _cat_optional(
    a: Optional[jax.Array], b: Optional[jax.Array]
) -> Optional[jax.Array]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return jnp.concatenate((a, b), axis=0)


@partial(jax.named_call, name="DeformableParticleModel._merge_metric_field")
def _merge_metric_field(
    a: Optional[jax.Array],
    b: Optional[jax.Array],
    n_a: int,
    n_b: int,
    fill: float,
) -> Optional[jax.Array]:
    if a is None and b is None:
        return None
    left = a if a is not None else jnp.full((n_a,), fill, dtype=float)
    right = b if b is not None else jnp.full((n_b,), fill, dtype=float)
    return jnp.concatenate((left, right), axis=0)


@partial(jax.named_call, name="DeformableParticleModel._num_bodies")
def _num_bodies(model: DeformableParticleModel) -> int:
    candidates: list[int] = []
    if model.ec is not None:
        candidates.append(int(model.ec.shape[0]))
    if model.initial_body_contents is not None:
        candidates.append(int(model.initial_body_contents.shape[0]))
    if model.elements_ID is not None and model.elements_ID.size > 0:
        candidates.append(int(jnp.max(model.elements_ID)) + 1)
    return max(candidates) if candidates else 0


@partial(jax.named_call, name="DeformableParticleModel._max_vertex_id")
def _max_vertex_id(model: DeformableParticleModel) -> int:
    candidates: list[int] = []
    for arr in (model.elements, model.edges, model.element_adjacency_edges):
        if arr is not None and arr.size > 0:
            candidates.append(int(jnp.max(arr)))
    return max(candidates) if candidates else -1
