# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Implementation of the deformable particle model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast
from collections.abc import Sequence
from functools import partial

from . import BondedForceModel
from ..utils.linalg import cross, dot, norm, norm2, unit

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System
    from ..forces.force_manager import ForceFunction
    from ..forces.force_manager import EnergyFunction


@BondedForceModel.register("DeformableParticleModel")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DeformableParticleModel(BondedForceModel):
    r"""This model assumes triangular meshes for 3D bodies and linear segment
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

        E_K &= E_{K,measure} + E_{K,content} + E_{K,bending} + E_{K,edge} + E_{K,surface} \\
        E_{K,measure} &= \sum_{m} \frac{em_m}{2{\mathcal{M}_{m,0}}} \left(\mathcal{M}_m - \mathcal{M}_{m,0}\right)^2 \\
        E_{K,surface} &= \sum_{m} \gamma_m \mathcal{M}_m \\
        E_{K,content} &= \frac{e_c}{2{\mathcal{C}_{K,0}}} \left(\mathcal{C}_K - \mathcal{C}_{K,0}\right)^2 \\
        E_{K,bending} &= \frac{1}{2} \sum_{a} eb_a wb_{a} \left(\theta_a -\theta_{a,0}\right)^2 \\
        E_{K,edge} &= \frac{1}{2} \sum_{e} el_e \left(L_e - L_{e,0}\right)^2

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
        w_b: ArrayLike | None = None,  # compatibility with checkpointer.
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
        if elements_id is not None:
            elements_id = jnp.asarray(elements_id, dtype=int)
            _, elements_id = jnp.unique(elements_id, return_inverse=True)
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
        w_b = jnp.asarray(w_b, dtype=float) if w_b is not None else None
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

        # Check if we need to compute mesh properties.
        # Note: the bending term needs element properties both when the
        # reference bendings are missing and when the bending normalization
        # w_b is missing (the 2D w_b path uses the computed element measures).
        if (
            (em is not None and initial_element_measures is None)
            or (ec is not None and initial_body_contents is None)
            or (eb is not None and (initial_bendings is None or w_b is None))
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
                if elements_id is None:
                    assert ec.shape == (
                        1,
                    ), f"Content elasticity coefficient (ec) has shape {ec.shape}, but should have shape (..., 1) if elements_id is not provided."
                    elements_id = jnp.zeros(elements_arr.shape[0], dtype=int)
                elements_id_arr = cast(jax.Array, elements_id)
                assert (
                    elements_id_arr.shape == elements_arr.shape[:-1]
                ), f"elements_id.shape={elements_id_arr.shape} does not match elements.shape[:-1]={elements_arr.shape[:-1]}. elements.shape={elements_arr.shape}."
                assert jnp.max(elements_id_arr) + 1 == ec.shape[-1], (
                    "Number of unique body IDs in elements_id does not match "
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
                elements_id is not None
            ), "Content elasticity coefficient (ec) provided but elements_id is None."
            assert (
                elements.shape[-1] == 2 or elements.shape[-1] == 3
            ), f"elements.shape={elements.shape}, but should have shape=(..., M, 2 or 3)."
            assert (
                jnp.max(elements_id) + 1 == ec.shape[-1]
            ), f"Number of unique body IDs in elements_id does not match number of content elasticity coefficients (ec). ec.shape={ec.shape}."
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
            if eb.ndim == 0 or jnp.asarray(eb).shape == (1,):
                eb = jnp.full(element_adjacency.shape[:-1], eb, dtype=float)
            if elements.shape[-1] == 3 and element_adjacency_edges is None:
                assert (
                    vertices is not None
                ), "Bending elasticity coefficient (eb) provided but vertices is None."
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
                # Compute SIGNED reference bending angles with the exact same
                # sign convention used at runtime in
                # compute_potential_energy_w_aux, so that concave (valley)
                # hinges are stress-free in the reference configuration.
                assert (
                    vertices is not None
                ), "Bending elasticity coefficient (eb) provided but vertices is None."
                if elements.shape[-1] == 3:
                    assert (
                        element_adjacency_edges is not None
                    ), "3D bending requires element_adjacency_edges."
                    hinge_vertices = vertices[element_adjacency_edges]
                else:
                    hinge_vertices = None
                initial_bendings = signed_bending_angles(
                    initial_element_normals,
                    element_adjacency,
                    hinge_vertices,
                    int(vertices.shape[-1]),
                )

            # Precompute reference bending normalization w_b if not provided
            if w_b is None:
                assert (
                    vertices is not None
                ), "Bending elasticity coefficient (eb) provided but vertices is None."
                if elements.shape[-1] == 3:
                    assert (
                        element_adjacency_edges is not None
                    ), "3D bending requires element_adjacency_edges."
                    adj_elem_1 = elements[element_adjacency[:, 0]]
                    adj_elem_2 = elements[element_adjacency[:, 1]]
                    c1_0 = jnp.mean(vertices[adj_elem_1], axis=-2)
                    c2_0 = jnp.mean(vertices[adj_elem_2], axis=-2)
                    dual_length0 = norm(c2_0 - c1_0)

                    h_verts0 = vertices[element_adjacency_edges]
                    hinge_vec0 = h_verts0[:, 1, :] - h_verts0[:, 0, :]
                    hinge_length0 = norm(hinge_vec0)
                    w_b = hinge_length0 / jnp.where(
                        dual_length0 == 0, 1.0, dual_length0
                    )
                else:
                    left_len0 = computed_initial_element_measures[
                        element_adjacency[:, 0]
                    ]
                    right_len0 = computed_initial_element_measures[
                        element_adjacency[:, 1]
                    ]
                    dual_length0 = 0.5 * (left_len0 + right_len0)
                    w_b = 1.0 / jnp.where(dual_length0 == 0, 1.0, dual_length0)

            assert (
                initial_bendings.shape == element_adjacency.shape[:-1]
            ), f"initial_bendings.shape={initial_bendings.shape}, but should have element_adjacency.shape[:-1]={element_adjacency.shape[:-1]}. element_adjacency.shape={element_adjacency.shape}."
            assert (
                jnp.asarray(eb).shape == element_adjacency.shape[:-1]
            ), f"eb.shape={jnp.shape(eb)} does not match expected element_adjacency.shape[:-1]={element_adjacency.shape[:-1]}. element_adjacency.shape={element_adjacency.shape}."
            assert (
                jnp.asarray(w_b).shape == element_adjacency.shape[:-1]
            ), f"w_b.shape={jnp.shape(w_b)} does not match expected element_adjacency.shape[:-1]={element_adjacency.shape[:-1]}."

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
            if el.ndim == 0 or jnp.asarray(el).shape == (1,):
                el = jnp.full(edges.shape[:-1], el, dtype=float)
            if initial_edge_lengths is None:
                assert (
                    vertices is not None
                ), "Vertices must be provided to compute initial edge lengths."
                v1 = vertices[edges[:, 0]]
                v2 = vertices[edges[:, 1]]
                v12 = v2 - v1
                initial_edge_lengths = norm(v12)
            assert (
                jnp.asarray(initial_edge_lengths).shape == edges.shape[:-1]
            ), f"initial_edge_lengths.shape={jnp.shape(initial_edge_lengths)}, but should have edges.shape[:-1]={edges.shape[:-1]}. edges.shape={edges.shape}."
            assert (
                jnp.asarray(el).shape == edges.shape[:-1]
            ), f"el.shape={jnp.shape(el)} does not match expected edges.shape[:-1]={edges.shape[:-1]}. edges.shape={edges.shape}."

        # Validate positivity of reference measures used as energy normalizations.
        # Negative reference contents typically indicate clockwise (CW) vertex
        # winding; the energy divides by these quantities, so a negative value
        # silently produces a negative-stiffness potential.
        _require_positive_reference(
            "initial_element_measures",
            initial_element_measures if em is not None else None,
        )
        _require_positive_reference(
            "initial_body_contents", initial_body_contents if ec is not None else None
        )
        _require_positive_reference(
            "initial_edge_lengths", initial_edge_lengths if el is not None else None
        )

        # Keep only data required by active terms, even if the user provided extra fields.
        need_elements = any(x is not None for x in (em, ec, eb, gamma))
        need_edges = el is not None
        need_adjacency = eb is not None
        need_elements_id = ec is not None
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
            w_b = None
        if not need_elements_id:
            elements_id = None
        if not need_initial_body_contents:
            initial_body_contents = None
        if not need_initial_element_measures:
            initial_element_measures = None

        return cls(
            elements=elements,
            edges=edges,
            element_adjacency=element_adjacency,
            element_adjacency_edges=element_adjacency_edges,
            elements_id=(
                jnp.asarray(elements_id, dtype=int) if elements_id is not None else None
            ),
            initial_body_contents=initial_body_contents,
            initial_element_measures=initial_element_measures,
            initial_edge_lengths=(
                jnp.asarray(initial_edge_lengths)
                if initial_edge_lengths is not None
                else None
            ),
            initial_bendings=initial_bendings,
            w_b=jnp.asarray(w_b) if w_b is not None else None,
            em=em,
            ec=ec,
            eb=eb,
            el=el,
            gamma=gamma,
        )

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleModel.merge")
    def merge(
        model1: BondedForceModel,
        model2: BondedForceModel | Sequence[BondedForceModel],
    ) -> BondedForceModel:
        if not isinstance(model1, DeformableParticleModel):
            raise TypeError(
                f"DeformableParticleModel.merge expects model1 to be DeformableParticleModel, got {type(model1).__name__}."
            )

        model_cls = type(model1)

        if isinstance(model2, BondedForceModel):
            models_to_merge: list[DeformableParticleModel] = [
                cast(DeformableParticleModel, model2)
            ]
        else:
            models_to_merge = [cast(DeformableParticleModel, m) for m in model2]

        if any(type(m) is not model_cls for m in models_to_merge):
            raise TypeError(
                f"{model_cls.__name__}.merge only supports merging {model_cls.__name__} instances."
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
            cur_elements_id = current.elements_id
            new_elements_id = nxt.elements_id
            if need_body_ids:
                if cur_elements_id is None and current.elements is not None:
                    cur_elements_id = jnp.zeros((current.elements.shape[0],), dtype=int)
                    n_current = max(n_current, 1)
                if new_elements_id is None and nxt.elements is not None:
                    new_elements_id = jnp.zeros((nxt.elements.shape[0],), dtype=int)
                    n_new = max(n_new, 1)

            body_offset = n_current
            if new_elements_id is not None and body_offset > 0:
                new_elements_id = new_elements_id + body_offset

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
            merged_w_b = _merge_metric_field(
                current.w_b,
                nxt.w_b,
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

            merged_kwargs: dict[str, Any] = dict(
                elements=_cat_optional(current.elements, next_elements),
                edges=_cat_optional(current.edges, next_edges),
                element_adjacency=_cat_optional(
                    current.element_adjacency, next_adjacency
                ),
                element_adjacency_edges=_cat_optional(
                    current.element_adjacency_edges, next_adj_edges
                ),
                elements_id=_cat_optional(cur_elements_id, new_elements_id),
                initial_body_contents=merged_initial_body_contents,
                initial_element_measures=merged_initial_element_measures,
                initial_edge_lengths=merged_initial_edge_lengths,
                initial_bendings=merged_initial_bendings,
                w_b=merged_w_b,
                em=merged_em,
                ec=merged_ec,
                eb=merged_eb,
                el=merged_el,
                gamma=merged_gamma,
            )
            # Subclasses (plastic variants) merge their extra fields through
            # this single hook so the merge logic cannot diverge between them.
            merged_kwargs.update(model_cls._merge_extra_fields(current, nxt))
            current = model_cls(**merged_kwargs)

        return current

    @classmethod
    def _merge_extra_fields(
        cls,
        current: DeformableParticleModel,
        nxt: DeformableParticleModel,
    ) -> dict[str, Any]:
        """Merge subclass-specific fields for one merge step.

        Subclasses override this to merge their extra dataclass fields (e.g.
        plastic relaxation times). The base model has no extra fields.
        """
        return {}

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleModel.add")
    def add(
        model: BondedForceModel,
        **kwargs: Any,
    ) -> BondedForceModel:
        if not isinstance(model, DeformableParticleModel):
            raise TypeError(
                f"DeformableParticleModel.add expects model to be DeformableParticleModel, got {type(model).__name__}."
            )

        model_cls = type(model)
        new_model = model_cls.Create(**kwargs)
        return model_cls.merge(model, new_model)

    @staticmethod
    @partial(
        jax.named_call, name="DeformableParticleModel.compute_potential_energy_w_aux"
    )
    def compute_potential_energy_w_aux(
        pos: jax.Array,
        state: State,
        system: System,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
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
        idx_map = _build_idx_map(state)
        e_element = jnp.array(0.0, dtype=float)
        e_content = jnp.array(0.0, dtype=float)
        e_gamma = jnp.array(0.0, dtype=float)
        e_bending = jnp.array(0.0, dtype=float)
        e_edge = jnp.array(0.0, dtype=float)

        if dp_model.elements is not None:
            current_element_indices = idx_map[dp_model.elements]
            element_normal, element_measure, partial_content = jax.vmap(
                compute_element_properties
            )(vertices[current_element_indices])

        # Content
        if (
            dp_model.elements is not None
            and dp_model.elements_id is not None
            and dp_model.ec is not None
            and dp_model.initial_body_contents is not None
        ):
            # 1/2 * sum_K ec_K * C_{K,0} * (C_K/C_{K,0} - 1)^2
            body_content = jax.ops.segment_sum(
                partial_content,
                dp_model.elements_id,
                num_segments=dp_model.ec.shape[-1],
            )
            norm_content_energy = (
                dp_model.ec
                * jnp.square(body_content - dp_model.initial_body_contents)
                / dp_model.initial_body_contents
            )
            e_content = jnp.sum(norm_content_energy) / 2

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
            e_element = jnp.sum(norm_measure_energy) / 2

        # Surface tension
        if dp_model.elements is not None and dp_model.gamma is not None:
            # sum_m gamma_m * M_m
            e_gamma = jnp.sum(dp_model.gamma * element_measure)

        # Bending
        has_bending_reqs = (
            dp_model.elements is not None
            and dp_model.element_adjacency is not None
            and dp_model.eb is not None
            and dp_model.initial_bendings is not None
            and dp_model.w_b is not None
        )
        if dim == 3 and has_bending_reqs:
            has_bending_reqs = dp_model.element_adjacency_edges is not None

        if has_bending_reqs:
            # Narrow types for mypy
            element_adjacency = cast(jax.Array, dp_model.element_adjacency)
            eb = cast(jax.Array, dp_model.eb)
            initial_bendings = cast(jax.Array, dp_model.initial_bendings)
            w_b = cast(jax.Array, dp_model.w_b)

            # 1/2 * sum_a eb_a * w_b,a * (theta_a - theta_{a,0})^2
            if dim == 3:
                # We checked this is not None in has_bending_reqs
                adjacency_edges = cast(jax.Array, dp_model.element_adjacency_edges)
                hinge_vertices = vertices[idx_map[adjacency_edges]]  # (A, 2, 3)
            else:
                hinge_vertices = None

            bending = signed_bending_angles(
                element_normal, element_adjacency, hinge_vertices, dim
            )

            norm_bending_energy = eb * w_b * jnp.square(bending - initial_bendings)
            e_bending = jnp.sum(norm_bending_energy) / 2

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
            edge_length = norm(edge_vector)
            norm_edge_strain_energy = dp_model.el * jnp.square(
                edge_length - dp_model.initial_edge_lengths
            )
            e_edge = jnp.sum(norm_edge_strain_energy) / 2

        aux = {
            "E_content": e_content,
            "E_element": e_element,
            "E_gamma": e_gamma,
            "E_bending": e_bending,
            "E_edge": e_edge,
        }

        return e_content + e_element + e_gamma + e_bending + e_edge, aux

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

    def update_reference_state(
        self,
        pos: jax.Array,
        state: State,
        system: System,
    ) -> DeformableParticleModel:
        """Return a model with the plastically updated reference configuration.

        This is a *pure* functional update: the model itself is never mutated.
        The base (elastic) model has no plastic flow and returns ``self``.
        Plastic subclasses override this hook.
        """
        return self

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleModel.compute_forces")
    def compute_forces(
        pos: jax.Array,
        state: State,
        system: System,
    ) -> tuple[jax.Array, jax.Array]:
        dp_model = cast(DeformableParticleModel, system.bonded_force_model)

        # Plastic flow: compute the updated model functionally, then thread it
        # through the system explicitly. ``ForceManager.apply`` returns this
        # ``system``, which is how the update persists across steps; the model
        # instance itself is never mutated in place.
        new_model = dp_model.update_reference_state(pos, state, system)
        if new_model is not dp_model:
            system.bonded_force_model = new_model

        force = -jax.grad(DeformableParticleModel.compute_potential_energy)(
            pos, state, system
        )
        return force, jnp.zeros_like(state.torque)

    @property
    @partial(jax.named_call, name="DeformableParticleModel.force_and_energy_fns")
    def force_and_energy_fns(self) -> tuple[ForceFunction, EnergyFunction, bool]:
        return (
            DeformableParticleModel.compute_forces,
            DeformableParticleModel.compute_potential_energy,
            False,
        )


@partial(jax.named_call, name="DeformableParticleModel.angle_between_normals")
def angle_between_normals(n1: jax.Array, n2: jax.Array) -> jax.Array:
    """Unsigned angle between normals.

    .. warning::
        This loses the sign of the bending angle in 2D and 3D. For reference
        bendings consistent with the runtime energy, use
        :func:`signed_bending_angles` instead.
    """
    cos = dot(n1, n2)
    sin = cross(n1, n2)
    if sin.ndim > cos.ndim:
        sin = norm(sin)
    return jnp.atan2(sin, cos)


@partial(jax.named_call, name="DeformableParticleModel.signed_bending_angles")
def signed_bending_angles(
    element_normals: jax.Array,
    element_adjacency: jax.Array,
    hinge_vertices: jax.Array | None,
    dim: int,
) -> jax.Array:
    """Signed bending angle per element adjacency.

    This is the single source of truth for the bending-angle sign convention,
    shared by the energy, the reference-bending computation at ``Create`` time,
    and the plastic bending update.

    Parameters
    ----------
    element_normals : jax.Array
        Unit normals per element. Shape ``(M, dim)``.
    element_adjacency : jax.Array
        Adjacent element pairs. Shape ``(A, 2)``.
    hinge_vertices : jax.Array or None
        Positions of the two shared-edge (hinge) vertices per adjacency,
        ordered as in ``element_adjacency_edges``. Shape ``(A, 2, 3)``.
        Required in 3D; ignored in 2D.
    dim : int
        Spatial dimension (2 or 3).
    """
    n1 = element_normals[element_adjacency[:, 0]]
    n2 = element_normals[element_adjacency[:, 1]]
    cos = dot(n1, n2)

    if dim == 3:
        assert hinge_vertices is not None, "3D bending requires hinge vertices."
        tangent_vec = hinge_vertices[:, 1, :] - hinge_vertices[:, 0, :]
        tangent = unit(tangent_vec)
        cross_prod = cross(n1, n2)
        sin = dot(cross_prod, tangent)
    else:
        # 2D cross returns shape (A, 1); drop the trailing axis only so that
        # single-adjacency meshes keep shape (A,).
        sin = jnp.squeeze(cross(n1, n2), axis=-1)

    return jnp.atan2(sin, cos)


@partial(jax.named_call, name="DeformableParticleModel.current_bending_angles")
def current_bending_angles(
    model: DeformableParticleModel,
    pos: jax.Array,
    state: State,
) -> jax.Array | None:
    """Signed bending angles of the current configuration.

    Runs the same element-normal/bending-angle pipeline used by the energy
    (single source of truth: :func:`signed_bending_angles`). Returns ``None``
    when the model lacks the required bending topology.
    """
    dim = state.dim
    has_reqs = model.elements is not None and model.element_adjacency is not None
    if dim == 3:
        has_reqs = has_reqs and model.element_adjacency_edges is not None
    if not has_reqs:
        return None

    compute_fn = (
        compute_element_properties_3D if dim == 3 else compute_element_properties_2D
    )
    idx_map = _build_idx_map(state)
    elements = cast(jax.Array, model.elements)
    element_normal, _, _ = jax.vmap(compute_fn)(pos[idx_map[elements]])

    if dim == 3:
        adjacency_edges = cast(jax.Array, model.element_adjacency_edges)
        hinge_vertices = pos[idx_map[adjacency_edges]]
    else:
        hinge_vertices = None

    return signed_bending_angles(
        element_normal,
        cast(jax.Array, model.element_adjacency),
        hinge_vertices,
        dim,
    )


@partial(jax.named_call, name="DeformableParticleModel._build_idx_map")
def _build_idx_map(state: State) -> jax.Array:
    """Map ``unique_id`` -> current row index in ``state.pos``."""
    return jnp.zeros((state.N,), dtype=int).at[state.unique_id].set(jnp.arange(state.N))


def _require_positive_reference(name: str, arr: ArrayLike | None) -> None:
    """Raise a clear error when a reference measure is not strictly positive."""
    if arr is None:
        return
    arr = jnp.asarray(arr)
    if not bool(jnp.all(arr > 0)):
        raise ValueError(
            f"{name} must be strictly positive, got min={float(jnp.min(arr))}. "
            "Negative reference measures/contents usually indicate clockwise "
            "(CW) vertex winding; reorder element vertices counter-clockwise "
            "(CCW) so normals point outward."
        )


@partial(jax.named_call, name="DeformableParticleModel.compute_element_properties_3D")
def compute_element_properties_3D(
    simplex: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r1 = simplex[0]
    r2 = simplex[1] - simplex[0]
    r3 = simplex[2] - simplex[0]
    face_normal = cross(r2, r3) / 2
    partial_vol = dot(face_normal, r1) / 3
    area_face2 = norm2(face_normal)
    # Double-where: sqrt/rsqrt must never see 0, even in the untaken branch,
    # otherwise reverse-mode gradients of degenerate elements are NaN.
    safe_area2 = jnp.where(area_face2 == 0.0, 1.0, area_face2)
    area_face = jnp.where(area_face2 == 0.0, 0.0, jnp.sqrt(safe_area2))
    inv_area = jnp.where(area_face2 == 0.0, 0.0, jax.lax.rsqrt(safe_area2))
    return (
        face_normal * inv_area,
        area_face,
        partial_vol,
    )


@partial(jax.named_call, name="DeformableParticleModel.compute_element_properties_2D")
def compute_element_properties_2D(
    simplex: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r1 = simplex[0]
    r2 = simplex[1]
    edge = r2 - r1
    length2 = norm2(edge)
    # Double-where: sqrt/rsqrt must never see 0, even in the untaken branch,
    # otherwise reverse-mode gradients of degenerate elements are NaN.
    safe_length2 = jnp.where(length2 == 0.0, 1.0, length2)
    length = jnp.where(length2 == 0.0, 0.0, jnp.sqrt(safe_length2))
    inv_length = jnp.where(length2 == 0.0, 0.0, jax.lax.rsqrt(safe_length2))
    normal = jnp.array([edge[1], -edge[0]]) * inv_length
    partial_area = 0.5 * (r1[0] * r2[1] - r1[1] * r2[0])
    return normal, length, partial_area


@partial(jax.named_call, name="DeformableParticleModel._cat_optional")
def _cat_optional(a: jax.Array | None, b: jax.Array | None) -> jax.Array | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return jnp.concatenate((a, b), axis=0)


@partial(jax.named_call, name="DeformableParticleModel._merge_metric_field")
def _merge_metric_field(
    a: jax.Array | None,
    b: jax.Array | None,
    n_a: int,
    n_b: int,
    fill: float,
) -> jax.Array | None:
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
    if model.elements_id is not None and model.elements_id.size > 0:
        candidates.append(int(jnp.max(model.elements_id)) + 1)
    return max(candidates) if candidates else 0


@partial(jax.named_call, name="DeformableParticleModel._max_vertex_id")
def _max_vertex_id(model: DeformableParticleModel) -> int:
    candidates: list[int] = []
    for arr in (model.elements, model.edges, model.element_adjacency_edges):
        if arr is not None and arr.size > 0:
            candidates.append(int(jnp.max(arr)))
    return max(candidates) if candidates else -1
