# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Implementation of the deformable particle container."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Optional
from functools import partial

from ..utils.linalg import cross

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System
    from .force_manager import ForceFunction


def compute_face_properties(
    triangle: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r"""
    Computes normal, area, and signed partial volume for a single triangle.

    Parameters
    ----------
    triangle : jax.Array
        Shape (3, 3) representing the coordinates of the 3 vertices.

    Returns
    -------
    Tuple[jax.Array, jax.Array, jax.Array]
        (unit_normal, area, partial_volume)
    """
    r1 = triangle[0]
    r2 = triangle[1] - triangle[0]
    r3 = triangle[2] - triangle[0]
    face_normal = cross(r2, r3) / 2
    partial_vol = jnp.sum(face_normal * r1, axis=-1) / 3
    area_face2 = jnp.sum(face_normal * face_normal, axis=-1, keepdims=True)
    area_face = jnp.where(area_face2 == 0, 1.0, jnp.sqrt(area_face2))
    return face_normal / area_face, area_face, partial_vol


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DeformableParticleContainer:  # type: ignore[misc]
    r"""
    Registry holding topology and reference configuration for deformable particles.

    This container manages the mesh connectivity (`elements`, `edges`, etc.) and reference
    properties (initial measures, contents, lengths, angles) required to compute forces.
    It supports both 3D (volumetric bodies bounded by triangles) and 2D (planar bodies
    bounded by line segments).

    Indices in elements, edges, etc. correspond to the ID of particles in :class:`~jaxdem.state.State`.

    The general form of the deformable particle potential energy per particle is:

    .. math::
        &E_K = E_{K,measure} + E_{K,content} + E_{K,bending} + E_{K,edge}

        &E_{K,measure} = \frac{e_m}{2} \sum_{m} \left(\frac{\mathcal{M}_m}{\mathcal{M}_{m,0}} - 1 \right)^2 - \gamma \sum_{m} \mathcal{M}_m

        &E_{K,content} = \frac{e_c}{2} \left(\frac{\mathcal{C}_K}{\mathcal{C}_{K,0}} - 1 \right)^2

        &E_{K,bending} = \frac{e_b}{2} \sum_{a} \left( \theta_a/\theta_{a,0} - 1\right)^2

        &E_{K,edge} = \frac{e_l}{ 2}\sum_{e} \left( \frac{L_e}{L_{e,0}} - 1 \right)^2

    **Definitions per Dimension:**

    * **3D:** Measure ($\mathcal{M}$) is Face Area; Content ($\mathcal{C}$) is Volume; Elements are Triangles.
    * **2D:** Measure ($\mathcal{M}$) is Segment Length; Content ($\mathcal{C}$) is Enclosed Area; Elements are Segments.

    Shapes:
        - K: Number of deformable bodies
        - M: Number of boundary elements (:math:`K \sum_K m_K`)
        - E: Number of unique edges (:math:`K \sum_K e_K`)
        - A: Number of element adjacencies (:math:`K \sum_K a_K`)

    All mesh properties are concatenated along axis=0.
    """

    elements_ID: Optional[jax.Array]
    """
    Array of body IDs for each boundary element. Shape: (M,).
    `elements_ID[i] == k` means element `i` belongs to body `k`.
    """

    elements: Optional[jax.Array]
    """
    Array of vertex indices forming the boundary elements.
    Shape: (M, 3) for 3D (Triangles) or (M, 2) for 2D (Segments).
    Indices refer to the particle unique ID corresponding to the `State.pos` array.
    """

    initial_element_measures: Optional[jax.Array]
    """
    Array of reference (stress-free) measures for each element. Shape: (M,).
    Represents Area in 3D or Length in 2D.
    """

    element_adjacency_ID: Optional[jax.Array]
    """
    Array of body IDs for each adjacency (bending hinge). Shape: (A,).
    `element_adjacency_ID[a] == k` means adjacency `a` belongs to body `k`.
    """

    element_adjacency: Optional[jax.Array]
    """
    Array of element adjacency pairs (for bending/dihedral angles). Shape: (A, 2).
    Each row contains the indices of the two elements sharing a connection.
    """

    initial_bending: Optional[jax.Array]
    """
    Array of reference (stress-free) bending angles for each adjacency. Shape: (A,).
    Represents Dihedral Angle in 3D or Vertex Angle in 2D.
    """

    edges_ID: Optional[jax.Array]
    """
    Array of body IDs for each unique edge. Shape: (E,).
    `edges_ID[e] == k` means edge `e` belongs to body `k`.
    """

    edges: Optional[jax.Array]
    """
    Array of vertex indices forming the unique edges (wireframe). Shape: (E, 2).
    Each row contains the indices of the two vertices forming the edge.
    Note: In 2D, the set of edges often overlaps with the set of elements (segments).
    """

    initial_edge_lengths: Optional[jax.Array]
    """
    Array of reference (stress-free) lengths for each unique edge. Shape: (E,).
    """

    initial_body_contents: Optional[jax.Array]
    """
    Array of reference (stress-free) bulk content for each body. Shape: (K,).
    Represents Volume in 3D or Area in 2D.
    """

    em: Optional[jax.Array]
    """
    Measure elasticity coefficient for each body. Shape: (K,).
    (Controls Area stiffness in 3D; Length stiffness in 2D).
    """

    ec: Optional[jax.Array]
    """
    Content elasticity coefficient for each body. Shape: (K,).
    (Controls Volume stiffness in 3D; Area stiffness in 2D).
    """

    eb: Optional[jax.Array]
    """
    Bending elasticity coefficient for each body. Shape: (K,).
    """

    el: Optional[jax.Array]
    """
    Edge length elasticity coefficient for each body. Shape: (K,).
    """

    gamma: Optional[jax.Array]
    """
    Surface/Line tension coefficient for each body. Shape: (K,).
    """

    num_bodies: int
    """
    Total number of distinct deformable bodies in the container. Shape: (K,).
    """

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleContainer.create")
    def create(
        vertices: ArrayLike,
        *,
        elements_ID: Optional[ArrayLike] = None,
        elements: Optional[ArrayLike] = None,
        initial_element_measures: Optional[ArrayLike] = None,
        element_adjacency_ID: Optional[ArrayLike] = None,
        element_adjacency: Optional[ArrayLike] = None,
        initial_bending: Optional[ArrayLike] = None,
        edges_ID: Optional[ArrayLike] = None,
        edges: Optional[ArrayLike] = None,
        initial_edge_lengths: Optional[ArrayLike] = None,
        initial_body_contents: Optional[ArrayLike] = None,
        em: Optional[ArrayLike] = None,
        ec: Optional[ArrayLike] = None,
        eb: Optional[ArrayLike] = None,
        el: Optional[ArrayLike] = None,
        gamma: Optional[ArrayLike] = None,
    ) -> "DeformableParticleContainer":
        r"""
        Factory method to create a new :class:`DeformableParticleContainer`.

        Calculates initial geometric properties (areas, volumes, bending angles, and edge lenghts) from the provided
        `vertices` if they are not explicitly provided.

        Parameters
        -----------
        vertices : jax.ArrayLike
            The particle positions corresponding to `State.pos`. Shape: (V, dim).
        elements_ID: jax.ArrayLike, optional
            Array of body IDs for each boundary element. Shape: (M,).
            If None, defaults to all zeros (single body).
        elements: jax.ArrayLike, optional
            Array of vertex indices forming the boundary elements.
            Shape: (M, 3) for 3D (Triangles) or (M, 2) for 2D (Segments).
        initial_element_measures: jax.ArrayLike, optional
            Reference (stress-free) measures for each element (Area in 3D, Length in 2D).
            If None, calculated from `vertices` and `elements`.
        element_adjacency_ID: jax.ArrayLike, optional
            Array of body IDs for each adjacency (bending hinge). Shape: (A,).
            If None, defaults to all zeros.
        element_adjacency: jax.ArrayLike, optional
            Array of element adjacency pairs for bending. Shape: (A, 2).
        initial_bending: jax.ArrayLike, optional
            Reference (stress-free) bending angles. Shape: (A,).
            **Defaults to 0.0 (flat) if not provided.**
        edges_ID: jax.ArrayLike, optional
            Array of body IDs for each unique edge. Shape: (E,).
            If None, defaults to all zeros.
        edges: jax.ArrayLike, optional
            Array of vertex indices forming the unique edges (wireframe). Shape: (E, 2).
        initial_edge_lengths: jax.ArrayLike, optional
            Reference (stress-free) lengths for each unique edge. Shape: (E,).
            If None, calculated as the Euclidean distance between vertices.
        initial_body_contents: jax.ArrayLike, optional
            Reference (stress-free) bulk content (Volume in 3D, Area in 2D) for each body.
            If None, calculated by summing element contributions.
        em: jax.ArrayLike, optional
            Elasticity coefficient for element measures (Area/Length stiffness). Shape: (K,).
        ec: jax.ArrayLike, optional
            Elasticity coefficient for body contents (Volume/Area stiffness). Shape: (K,).
        eb: jax.ArrayLike, optional
            Bending elasticity coefficient. Shape: (K,).
        el: jax.ArrayLike, optional
            Edge length elasticity coefficient. Shape: (K,).
        gamma: jax.ArrayLike, optional
            Surface/Line tension coefficient. Shape: (K,).

        Returns
        --------
        DeformableParticleContainer
            A new container instance with densified IDs and computed properties.
        """
        vertices = jnp.asarray(vertices, dtype=float)
        dim = vertices.shape[-1]

        # --- Densify IDs and Compute num_bodies ---
        valid_ids = [
            jnp.array(x)
            for x in [elements_ID, element_adjacency_ID, edges_ID]
            if x is not None
        ]

        if valid_ids:
            # Concatenate all IDs to map them to a single dense range (0..N-1)
            unique_vals, dense_all = jnp.unique(
                jnp.concatenate(valid_ids), return_inverse=True
            )
            num_bodies = unique_vals.size

            # Split dense indices back to original variables
            split_indices = np.cumsum([x.size for x in valid_ids])[:-1]
            split_arrays = iter(jnp.split(dense_all, split_indices))

            # Reassign only if the original variable was provided
            elements_ID = next(split_arrays) if elements_ID is not None else None
            element_adjacency_ID = (
                next(split_arrays) if element_adjacency_ID is not None else None
            )
            edges_ID = next(split_arrays) if edges_ID is not None else None
        else:
            num_bodies = 1

        # --- Compute Geometric Properties related to elements and contents ---
        if em is not None or ec is not None or gamma is not None:
            assert (
                elements is not None
            ), "Elements must be provided if em, ec, or gamma are specified."
            elements = jnp.asarray(elements, dtype=int)
            M = elements.shape[0]
            elements_ID = (
                jnp.asarray(elements_ID, dtype=int)
                if elements_ID is not None
                else jnp.zeros(M, dtype=int)
            )

            # TO DO: Support 2D elements (segments)
            element_normal, element_measure, partial_content = jax.vmap(
                compute_face_properties
            )(vertices[elements])

            if em is not None:
                em = jnp.asarray(em, dtype=float)
                initial_element_measures = (
                    jnp.asarray(initial_element_measures, dtype=float)
                    if initial_element_measures is not None
                    else element_measure
                )
                assert initial_element_measures.shape[0] == M
                assert em.shape[0] == num_bodies

            if ec is not None:
                ec = jnp.asarray(ec, dtype=float)
                initial_body_contents = (
                    jnp.asarray(initial_body_contents, dtype=float)
                    if initial_body_contents is not None
                    else jax.ops.segment_sum(
                        partial_content, elements_ID, num_segments=num_bodies
                    )
                )
                assert initial_body_contents.shape[0] == num_bodies
                assert ec.shape[0] == num_bodies

            if gamma is not None:
                gamma = jnp.asarray(gamma, dtype=float)
                assert gamma.shape[0] == num_bodies

        if el is not None:
            el = jnp.asarray(el, dtype=float)
            assert edges is not None, "Edges must be provided if el is specified."
            edges = jnp.asarray(edges, dtype=int)
            E = edges.shape[0]
            edges_ID = (
                jnp.asarray(edges_ID, dtype=int)
                if edges_ID is not None
                else jnp.zeros(E, dtype=int)
            )
            edge_lengths = jnp.linalg.norm(
                vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=-1
            )
            initial_edge_lengths = (
                jnp.asarray(initial_edge_lengths, dtype=float)
                if initial_edge_lengths is not None
                else edge_lengths
            )
            assert initial_edge_lengths.shape[0] == E
            assert edges_ID.shape[0] == E
            assert el.shape[0] == num_bodies

        if eb is not None:
            eb = jnp.asarray(eb, dtype=float)
            assert (
                element_adjacency is not None
            ), "Element adjacency must be provided if eb is specified."
            element_adjacency = jnp.asarray(element_adjacency, dtype=int)
            A = element_adjacency.shape[0]
            element_adjacency_ID = (
                jnp.asarray(element_adjacency_ID, dtype=int)
                if element_adjacency_ID is not None
                else jnp.zeros(A, dtype=int)
            )
            initial_bending = (
                jnp.asarray(initial_bending, dtype=float)
                if initial_bending is not None
                else jnp.zeros(A, dtype=float)
            )
            assert initial_bending.shape[0] == A
            assert element_adjacency_ID.shape[0] == A
            assert eb.shape[0] == num_bodies

        return DeformableParticleContainer(
            elements_ID=(
                jnp.asarray(elements_ID, dtype=int) if elements_ID is not None else None
            ),
            elements=(
                jnp.asarray(elements, dtype=int) if elements is not None else None
            ),
            initial_element_measures=(
                jnp.asarray(initial_element_measures, dtype=int)
                if initial_element_measures is not None
                else None
            ),
            element_adjacency_ID=(
                jnp.asarray(element_adjacency_ID, dtype=int)
                if element_adjacency_ID is not None
                else None
            ),
            element_adjacency=(
                jnp.asarray(element_adjacency, dtype=int)
                if element_adjacency is not None
                else None
            ),
            initial_bending=(
                jnp.asarray(initial_bending, dtype=int)
                if initial_bending is not None
                else None
            ),
            edges_ID=(
                jnp.asarray(edges_ID, dtype=int) if edges_ID is not None else None
            ),
            edges=(jnp.asarray(edges, dtype=int) if edges is not None else None),
            initial_edge_lengths=(
                jnp.asarray(initial_edge_lengths, dtype=int)
                if initial_edge_lengths is not None
                else None
            ),
            initial_body_contents=(
                jnp.asarray(initial_body_contents, dtype=int)
                if initial_body_contents is not None
                else None
            ),
            em=(jnp.asarray(em, dtype=int) if em is not None else None),
            ec=(jnp.asarray(ec, dtype=int) if ec is not None else None),
            eb=(jnp.asarray(eb, dtype=int) if eb is not None else None),
            el=(jnp.asarray(el, dtype=int) if el is not None else None),
            gamma=(jnp.asarray(gamma, dtype=int) if gamma is not None else None),
            num_bodies=num_bodies,
        )

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleContainer.merge")
    def merge(
        c1: "DeformableParticleContainer",
        c2: "DeformableParticleContainer",
    ) -> "DeformableParticleContainer":
        r"""
        Merges two :class:`DeformableParticleContainer` instances. This combines the topologies of two systems.

        Parameters
        -----------
        c1 : DeformableParticleContainer
            The first container. Its body IDs are preserved.
        c2 : DeformableParticleContainer
            The second container. Its body IDs are shifted by `c1.num_bodies`.
            Its face indices are shifted by `particle_offset`.

        Returns
        --------
        DeformableParticleContainer
            A merged container containing all bodies and faces.
        """
        # c2.elements_ID += c1.num_bodies

        def cat(a: jax.Array, b: jax.Array) -> jax.Array:
            if isinstance(a, jax.Array):
                return jnp.concatenate((a, b), axis=0)
            else:
                return a + b

        return jax.tree_util.tree_map(cat, c1, c2)

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleContainer.add")
    def add(
        container: "DeformableParticleContainer",
        vertices: ArrayLike,
        *,
        elements_ID: Optional[ArrayLike] = None,
        elements: Optional[ArrayLike] = None,
        initial_element_measures: Optional[ArrayLike] = None,
        element_adjacency_ID: Optional[ArrayLike] = None,
        element_adjacency: Optional[ArrayLike] = None,
        initial_bending: Optional[ArrayLike] = None,
        edges_ID: Optional[ArrayLike] = None,
        edges: Optional[ArrayLike] = None,
        initial_edge_lengths: Optional[ArrayLike] = None,
        initial_body_contents: Optional[ArrayLike] = None,
        em: Optional[ArrayLike] = None,
        ec: Optional[ArrayLike] = None,
        eb: Optional[ArrayLike] = None,
        el: Optional[ArrayLike] = None,
        gamma: Optional[ArrayLike] = None,
    ) -> "DeformableParticleContainer":
        r"""
        Factory method to create a new :class:`DeformableParticleContainer`.

        Calculates initial geometric properties (areas, volumes, bending angles, and edge lenghts) from the provided
        `state` if they are not explicitly provided.

        Parameters
        -----------
        vertices : jax.ArrayLike
            The particle positions corresponding `State.pos` that act as the vertices. Shape: (V, dim).
        elements_ID: jax.Array
            Array of body IDs for each boundary element. Shape: (M,)
            `elements_ID[i] == k` means element `i` belongs to body `k`.
        elements: jax.Array
            Array of vertex indices forming the boundary elements.
            Shape: (M, 3) for 3D (Triangles) or (M, 2) for 2D (Segments).
            Indices refer to the particle unique ID corresponding to the `State.pos` array.
        initial_element_measures: jax.Array
            Array of reference (stress-free) measures for each element. Shape: (M,)
            Represents Area in 3D or Length in 2D.
        element_adjacency_ID: jax.Array
            Array of body IDs for each adjacency (bending hinge). Shape: (A,)
            `element_adjacency_ID[a] == k` means adjacency `a` belongs to body `k`.
        element_adjacency: jax.Array
            Array of element adjacency pairs (for bending/dihedral angles). Shape: (A, 2)
            Each row contains the indices of the two elements sharing a connection.
        initial_bending: jax.Array
            Array of reference (stress-free) bending angles for each adjacency. Shape: (A,)
            Represents Dihedral Angle in 3D or Vertex Angle in 2D.
        edges_ID: jax.Array
            Array of body IDs for each unique edge. Shape: (E,)
            `edges_ID[e] == k` means edge `e` belongs to body `k`.
        edges: jax.Array
            Array of vertex indices forming the unique edges (wireframe). Shape: (E, 2)
            Each row contains the indices of the two vertices forming the edge.
            Note: In 2D, the set of edges often overlaps with the set of elements (segments).
        initial_edge_lengths: jax.Array
            Array of reference (stress-free) lengths for each unique edge. Shape: (E,)
        initial_body_contents: jax.Array
            Array of reference (stress-free) bulk content for each body. Shape: (K,)
            Represents Volume in 3D or Area in 2D.
        em: jax.Array
            Measure elasticity coefficient for each body. Shape: (K,)
            (Controls Area stiffness in 3D; Length stiffness in 2D).
        ec: jax.Array
            Content elasticity coefficient for each body. Shape: (K,)
            (Controls Volume stiffness in 3D; Area stiffness in 2D).
        eb: jax.Array
            Bending elasticity coefficient for each body. Shape: (K,)
        el: jax.Array
            Edge length elasticity coefficient for each body. Shape: (K,)
        gamma: jax.Array
            Surface/Line tension coefficient for each body. Shape: (K,)

        Returns
        --------
        DeformableParticleContainer
            A new container with the added bodies.
        """
        new_part = DeformableParticleContainer.create(
            vertices=vertices,
            elements=elements,
            edges=edges,
            elements_ID=elements_ID,
            initial_element_measures=initial_element_measures,
            element_adjacency_ID=element_adjacency_ID,
            element_adjacency=element_adjacency,
            initial_bending=initial_bending,
            edges_ID=edges_ID,
            initial_edge_lengths=initial_edge_lengths,
            initial_body_contents=initial_body_contents,
            em=em,
            ec=ec,
            eb=eb,
            el=el,
            gamma=gamma,
        )
        return DeformableParticleContainer.merge(container, new_part)

    @staticmethod
    def create_force_function(
        container: "DeformableParticleContainer",
    ) -> ForceFunction:
        r"""
        Creates a force function for use in simulations based on the deformable particle container.

        Parameters
        -----------
        container : DeformableParticleContainer
            The deformable particle container defining the topology and reference configuration.

        Returns
        --------
        Callable[[State, System], Tuple[jax.Array, jax.Array]]
            A force function that computes forces and torques based on the deformable particle model.
        """
        from ..forces.deformeble_particle_force import deformable_particle_force

        return partial(deformable_particle_force, DP_container=container)
