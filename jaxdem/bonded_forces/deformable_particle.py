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
from ..utils.linalg import cross

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

        &E_{K,measure} = \frac{1}{2} \sum_{m} em_m \mathcal{M}_{m,0}} \left(\frac{\mathcal{M}_m}{\mathcal{M}_{m,0}} - 1\right)^2

        &E_{K,surface} = \sum_{m} \gamma_m \mathcal{M}_m

        &E_{K,content} = \frac{e_c}{2} \mathcal{C}_{K,0} \left(\frac{\mathcal{C}_K}{\mathcal{C}_{K,0}} - 1\right)^2

        &E_{K,bending} = \frac{1}{2} \sum_{a} eb_a \frac{l_a}{h_a} \left(\theta_a/\theta_{a,0} - 1\right)^2

        &E_{K,edge} = \frac{1}{2} \sum_{e} el_e L_{e,0}\left(\frac{L_e}{L_{e,0}} - 1\right)^2

    **Definitions per Dimension:**

    * **3D:** Measure ($\mathcal{M}$) is Face Area; Content ($\mathcal{C}$) is
      Volume; Elements are Triangles.
    * **2D:** Measure ($\mathcal{M}$) is Segment Length; Content ($\mathcal{C}$)
      is Enclosed Area; Elements are Segments.

    The factor :math:`\frac{l_a}{h_a}` is the quotient between the hinge lenght and the dual lenght.
    The dual lenght is the distance between the two adjacent element centroids,
    and the hinge length is the length of the shared edge between the two adjacent elements.
    This factor is important to ensure that the bending energy scales correctly with mesh resolution.

    Shapes:
        - K: Number of deformable bodies, which can be defined by its vertices and connectivity (elements, edges, etc.)
        - M: Number of boundary elements (:math:`K \sum_K m_K`)
        - E: Number of unique edges (:math:`K \sum_K e_K`)
        - A: Number of element adjacencies (:math:`K \sum_K a_K`)

    All mesh properties of each body are concatenated along axis=0. We do not
    need to distinguish between bodies to compute forces, except for the
    content-related term, which requires the body ID for each element to sum up
    the content contributions per body.
    The body ID for each element is stored in `elements_ID`, which maps each
    element to its corresponding body.
    """

    # --- Topology ---
    elements: Optional[jax.Array]
    """
    Array of vertex indices forming the boundary elements.
    Shape: (M, 3) for 3D (Triangles) or (M, 2) for 2D (Segments).
    Indices refer to the particle unique_ID. Verices correspond to `State.pos`.
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
    that dont correspond to the mesh connectivity.
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
        elements_ID = (
            jnp.asarray(elements_ID, dtype=int) if elements_ID is not None else None
        )
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
                vertices.shape[-1] == vertices.shape[-1]
            ), f"vertices.shape[-1]={vertices.shape[-1]}, should have the same dimension as elements.shape[-1]={elements.shape[-1]}."
            compute_fn = (
                compute_element_properties_3D
                if vertices.shape[-1] == 3
                else compute_element_properties_2D
            )
            initial_element_normals, initial_element_measures, partial_body_contents = (
                jax.vmap(compute_fn)(vertices[elements])
            )

            # This needs to be corrected
            if ec is not None and initial_body_contents is None:
                initial_body_contents = jax.ops.segment_sum(
                    partial_body_contents, elements_ID, num_segments=num_bodies
                )

        # Contents

        # Meassure
        if em is not None:
            assert (
                elements is not None
            ), "Measure elasticity coefficient (em) provided but elements is None."
            assert (
                elements.shape[-1] == 2 or elements.shape[-1] == 3
            ), f"elements.shape={elements.shape}, but should should have shape=(..., M, 2 or 3)."
            if em.ndim == 0 or em.shape == (1,):
                em = jnp.full(elements.shape[:-1], em, dtype=float)
            assert (
                em.shape == elements.shape[:-1]
            ), f"em.shape={em.shape} does not match expected element.shape[:-1]={elements.shape[:-1]}. elements.shape={elements.shape}."
        else:
            initial_element_measures = None

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
            ), f"element_adjacency.shape={element_adjacency.shape}, but should should have shape=(..., A, 2)."
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
            ), f"elements.shape={elements.shape}, but should should have shape=(..., M, 2 or 3)."
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
            ), f"edges.shape={edges.shape}, but should should have shape=(..., E, 2)."
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

        return cls(
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
            n1 = _num_bodies(current)
            n2 = _num_bodies(nxt)
            body_offset = n1
            vertex_offset = _max_vertex_id(current) + 1
            element_offset = (
                0 if current.element is None else int(current.element.shape[0])
            )

            # Shift IDs in second model to preserve merge semantics consistent with State.merge.
            next_element = (
                nxt.element + vertex_offset
                if nxt.element is not None and vertex_offset > 0
                else nxt.element
            )
            next_edges = (
                nxt.edges + vertex_offset
                if nxt.edges is not None and vertex_offset > 0
                else nxt.edges
            )
            next_element_adjacency = (
                nxt.element_adjacency + element_offset
                if nxt.element_adjacency is not None and element_offset > 0
                else nxt.element_adjacency
            )
            next_element_adjacency_edges = (
                nxt.element_adjacency_edges + vertex_offset
                if nxt.element_adjacency_edges is not None and vertex_offset > 0
                else nxt.element_adjacency_edges
            )
            next_elements_ID = (
                nxt.elements_ID + body_offset
                if nxt.elements_ID is not None and body_offset > 0
                else nxt.elements_ID
            )

            merged_initial_body_contents = _merge_body_field(
                current.initial_body_contents,
                nxt.initial_body_contents,
                n1,
                n2,
                fill=1.0,
            )
            merged_em = _merge_body_field(current.em, nxt.em, n1, n2, fill=0.0)
            merged_ec = _merge_body_field(current.ec, nxt.ec, n1, n2, fill=0.0)
            merged_eb = _merge_body_field(current.eb, nxt.eb, n1, n2, fill=0.0)
            merged_el = _merge_body_field(current.el, nxt.el, n1, n2, fill=0.0)
            merged_gamma = _merge_body_field(current.gamma, nxt.gamma, n1, n2, fill=0.0)

            current = DeformableParticleModel(
                element=_cat_optional(current.element, next_element),
                edges=_cat_optional(current.edges, next_edges),
                element_adjacency=_cat_optional(
                    current.element_adjacency, next_element_adjacency
                ),
                element_adjacency_edges=_cat_optional(
                    current.element_adjacency_edges, next_element_adjacency_edges
                ),
                elements_ID=_cat_optional(current.elements_ID, next_elements_ID),
                initial_body_contents=merged_initial_body_contents,
                initial_element_measures=_cat_optional(
                    current.initial_element_measures, nxt.initial_element_measures
                ),
                initial_edge_lengths=_cat_optional(
                    current.initial_edge_lengths, nxt.initial_edge_lengths
                ),
                initial_bending=_cat_optional(
                    current.initial_bending, nxt.initial_bending
                ),
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
        initial_bending: Optional[ArrayLike] = None,
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
            initial_bending=initial_bending,
            em=em,
            ec=ec,
            eb=eb,
            el=el,
            gamma=gamma,
        )
        return DeformableParticleModel.merge(model, new_model)

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleModel.compute_potential_energy")
    @staticmethod
    def compute_potential_energy(
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
                f"DeformableParticleContainer only supports 2D or 3D, got dim={dim}."
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

        current_element_indices = idx_map[dp_model.elements]
        element_normal, element_measure, partial_content = jax.vmap(
            compute_element_properties
        )(vertices[current_element_indices])

        return jnp.zeros_like(state.mass)

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleModel.compute_potential_energy")
    @staticmethod
    def compute_forces(
        pos: jax.Array,
        state: State,
        system: System,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        return jnp.zeros_like(state.force), jnp.zeros_like(state.torque)

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleModel.create_force_and_energy_fns")
    def create_force_and_energy_fns(
        container: DeformableParticleModel,
    ) -> Tuple[ForceFunction, EnergyFunction, bool]:
        return (
            DeformableParticleModel.compute_forces,
            DeformableParticleModel.compute_potential_energy,
            False,
        )


def angle_between_normals(n1: jax.Array, n2: jax.Array) -> jax.Array:
    cos = jnp.sum(n1 * n2, axis=-1)
    sin = cross(n1, n2)
    if sin.ndim > cos.ndim:
        sin = jnp.sum(sin * sin, axis=-1)
        sin = jnp.sqrt(sin)
    return jnp.atan2(sin, cos)


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
