# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Implementation of the deformable particle container."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Optional, Dict
from functools import partial

from ..utils.linalg import cross

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System
    from .force_manager import ForceFunction
    from .force_manager import EnergyFunction


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
        if dim == 3:
            compute_element_properties = compute_element_properties_3D
        elif dim == 2:
            compute_element_properties = compute_element_properties_2D
        else:
            raise ValueError(
                f"DeformableParticleContainer only supports 2D or 3D, got dim={dim}."
            )

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
        if em is not None or ec is not None or gamma is not None or eb is not None:
            assert (
                elements is not None
            ), "Elements must be provided if em, ec, or gamma are specified."
            elements = jnp.asarray(elements, dtype=int)
            M = elements.shape[0]
            if elements.shape[-1] != dim:
                raise ValueError(
                    f"Invalid elements shape for {dim}D simulation. "
                    f"Expected shape (M, {dim}) [M, vertices_per_simplex], "
                    f"got {elements.shape}."
                )

            elements_ID = (
                jnp.asarray(elements_ID, dtype=int)
                if elements_ID is not None
                else jnp.zeros(M, dtype=int)
            )

            element_normal, element_measure, partial_content = jax.vmap(
                compute_element_properties
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
                element_adjacency is not None and elements is not None
            ), "Element adjacency and elements must be provided if eb is specified."
            element_adjacency = jnp.asarray(element_adjacency, dtype=int)
            A = element_adjacency.shape[0]
            element_adjacency_ID = (
                jnp.asarray(element_adjacency_ID, dtype=int)
                if element_adjacency_ID is not None
                else jnp.zeros(A, dtype=int)
            )
            element_normal, element_measure, partial_content = jax.vmap(
                compute_element_properties
            )(vertices[elements])
            angles = jax.vmap(angle_between_normals)(
                element_normal[element_adjacency[:, 0]],
                element_normal[element_adjacency[:, 1]],
            )
            initial_bending = (
                jnp.asarray(initial_bending, dtype=float)
                if initial_bending is not None
                else angles
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
                jnp.asarray(initial_element_measures, dtype=float)
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
                jnp.asarray(initial_bending, dtype=float)
                if initial_bending is not None
                else None
            ),
            edges_ID=(
                jnp.asarray(edges_ID, dtype=int) if edges_ID is not None else None
            ),
            edges=(jnp.asarray(edges, dtype=int) if edges is not None else None),
            initial_edge_lengths=(
                jnp.asarray(initial_edge_lengths, dtype=float)
                if initial_edge_lengths is not None
                else None
            ),
            initial_body_contents=(
                jnp.asarray(initial_body_contents, dtype=float)
                if initial_body_contents is not None
                else None
            ),
            em=(jnp.asarray(em, dtype=float) if em is not None else None),
            ec=(jnp.asarray(ec, dtype=float) if ec is not None else None),
            eb=(jnp.asarray(eb, dtype=float) if eb is not None else None),
            el=(jnp.asarray(el, dtype=float) if el is not None else None),
            gamma=(jnp.asarray(gamma, dtype=float) if gamma is not None else None),
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
        if c2.elements_ID is not None:
            c2.elements_ID += c1.num_bodies

        if c2.edges_ID is not None:
            c2.edges_ID += c1.num_bodies

        if c2.element_adjacency_ID is not None:
            c2.element_adjacency_ID += c1.num_bodies

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
    def compute_potential_energy(
        pos: jax.Array,
        state: State,
        _system: System,
        container: DeformableParticleContainer,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
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

        current_element_indices = idx_map[container.elements]
        element_normal, element_measure, partial_content = jax.vmap(
            compute_element_properties
        )(vertices[current_element_indices])

        # Element elastic energy
        if (
            container.em is not None
            and container.initial_element_measures is not None
            and container.elements_ID is not None
        ):
            temp_elements = jax.ops.segment_sum(
                jnp.square(element_measure - container.initial_element_measures),
                container.elements_ID,
                num_segments=container.num_bodies,
            )
            E_element = 0.5 * jnp.sum(container.em * temp_elements)

        # Content elastic energy
        if (
            container.ec is not None
            and container.initial_body_contents is not None
            and container.elements_ID is not None
        ):
            content = jax.ops.segment_sum(
                partial_content,
                container.elements_ID,
                num_segments=container.num_bodies,
            )
            E_content = 0.5 * jnp.sum(
                container.ec * jnp.square(content - container.initial_body_contents)
            )

        # Surface tension
        if container.gamma is not None and container.elements_ID is not None:
            element = jax.ops.segment_sum(
                element_measure,
                container.elements_ID,
                num_segments=container.num_bodies,
            )
            E_gamma = -jnp.sum(container.gamma * element)

        # Bending energy
        if (
            container.eb is not None
            and container.element_adjacency is not None
            and container.initial_bending is not None
            and container.element_adjacency_ID is not None
        ):
            # angles = jax.vmap(angle_between_normals)(
            #     element_normal[container.element_adjacency[:, 0]],
            #     element_normal[container.element_adjacency[:, 1]],
            # )
            # temp_angles = jax.ops.segment_sum(
            #     jnp.square(angles - container.initial_bending),
            #     container.element_adjacency_ID,
            #     num_segments=container.num_bodies,
            # )
            temp_angles = jax.ops.segment_sum(
                (
                    1.0  # jnp.cos(container.initial_bending)
                    - jnp.sum(
                        element_normal[container.element_adjacency[:, 0]]
                        * element_normal[container.element_adjacency[:, 1]],
                        axis=-1,
                    )
                ),
                container.element_adjacency_ID,
                num_segments=container.num_bodies,
            )
            E_bending = 0.5 * jnp.sum(container.eb * temp_angles)

        # Edge length energy
        if (
            container.el is not None
            and container.edges is not None
            and container.initial_edge_lengths is not None
            and container.edges_ID is not None
        ):
            current_edge_indices = idx_map[container.edges]
            edge_vecs = (
                vertices[current_edge_indices[:, 0]]
                - vertices[current_edge_indices[:, 1]]
            )
            edge_lengths2 = jnp.sum(edge_vecs * edge_vecs, axis=-1)
            edge_lengths = jnp.sqrt(edge_lengths2)
            temp_edges = jax.ops.segment_sum(
                jnp.square(edge_lengths - container.initial_edge_lengths),
                container.edges_ID,
                num_segments=container.num_bodies,
            )
            E_edge = 0.5 * jnp.sum(container.el * temp_edges)

        aux = dict(
            E_element=E_element,
            E_content=E_content,
            E_gamma=E_gamma,
            E_bending=E_bending,
            E_edge=E_edge,
        )
        return E_element + E_content + E_gamma + E_bending + E_edge, aux

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
        ForceFunction
            A force function that computes forces and torques based on the deformable particle model.
        """

        force_fn, _ = DeformableParticleContainer.create_force_energy_functions(
            container
        )
        return force_fn

    @staticmethod
    def create_force_energy_functions(
        container: "DeformableParticleContainer",
    ) -> Tuple["ForceFunction", "EnergyFunction"]:
        """
        Create a force function and a matching energy function.

        Critical guarantee
        ------------------
        Both functions use :meth:`compute_potential_energy` as the single source of truth.
        The force is computed as ``-grad(total_energy)`` using the exact same energy
        expression as the returned energy function.
        """

        def force_function(
            pos: jax.Array, state: "State", system: "System"
        ) -> Tuple[jax.Array, jax.Array]:
            energy_grad, _ = jax.grad(
                DeformableParticleContainer.compute_potential_energy, has_aux=True
            )(pos, state, system, container)
            return -energy_grad, jnp.zeros_like(state.torque)

        def energy_function(
            pos: jax.Array, state: "State", system: "System"
        ) -> jax.Array:
            # ForceManager expects per-particle energy. The DP energy is naturally a scalar
            # for the whole container; we distribute it uniformly over particles referenced
            # by the DP topology (and return 0 for unrelated particles, e.g. extra spheres).
            total, _ = DeformableParticleContainer.compute_potential_energy(
                pos, state, system, container
            )

            idx_map = (
                jnp.zeros((state.N,), dtype=int)
                .at[state.unique_ID]
                .set(jnp.arange(state.N))
            )

            mask = jnp.zeros((state.N,), dtype=bool)
            if container.elements is not None:
                mask = mask.at[idx_map[container.elements].reshape(-1)].set(True)
            if container.edges is not None:
                mask = mask.at[idx_map[container.edges].reshape(-1)].set(True)

            count = jnp.sum(mask.astype(float))
            count = jnp.where(count == 0.0, 1.0, count)

            return (total / count) * mask.astype(float)

        return force_function, energy_function


def angle_between_normals(n1: jax.Array, n2: jax.Array) -> jax.Array:
    r"""
    Computes the angle between two normals.

    Parameters
    ----------
    n1 : jax.Array
        First normal vector.
    n2 : jax.Array
        Second normal vector.

    Returns
    -------
    jax.Array
        Angle between the two normals in radians.
    """
    y = n1 - n2
    x = n1 + n2
    y_norm2 = jnp.sum(y * y, axis=-1)
    x_norm2 = jnp.sum(x * x, axis=-1)
    y_norm = jnp.sqrt(y_norm2)
    x_norm = jnp.sqrt(x_norm2)
    a = 2.0 * jnp.atan2(y_norm, x_norm)
    a = jnp.where((a >= 0.0) * (a <= jnp.pi), a, jnp.where(a < 0.0, 0.0, jnp.pi))
    return a

    # dot = jnp.sum(n1 * n2, axis=-1)
    # dot = jnp.clip(dot, -1.0, 1.0)
    # return jnp.arccos(dot)

    # dot = jnp.sum(n1 * n2, axis=-1)
    # cross_norm = jnp.linalg.norm(cross(n1, n2), axis=-1)
    # return jnp.atan2(cross_norm, dot)

    # diff_norm = jnp.sqrt(jnp.sum(jnp.square(n1 - n2), axis=-1) + 1e-18)
    # sum_norm = jnp.sqrt(jnp.sum(jnp.square(n1 + n2), axis=-1) + 1e-18)
    # return 2.0 * jnp.atan2(diff_norm, sum_norm)


def compute_element_properties_3D(
    simplex: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r"""
    Computes normal, area, and signed partial volume for a single simplex.

    Parameters
    ----------
    simplex : jax.Array
        Shape (3, 3) representing the coordinates of the 3 vertices.

    Returns
    -------
    Tuple[jax.Array, jax.Array, jax.Array]
        (unit_normal, area, partial_volume)
    """
    r1 = simplex[0]
    r2 = simplex[1] - simplex[0]
    r3 = simplex[2] - simplex[0]
    face_normal = cross(r2, r3) / 2
    partial_vol = jnp.sum(face_normal * r1, axis=-1) / 3
    area_face2 = jnp.sum(face_normal * face_normal, axis=-1)
    area_face = jnp.sqrt(area_face2)
    return face_normal / jnp.where(area_face == 0, 1, area_face), area_face, partial_vol


def compute_element_properties_2D(
    simplex: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r"""
    Computes normal, length, and signed partial area for a single simplex.

    Parameters
    ----------
    simplex : jax.Array
        Shape (2, 2) representing the coordinates of the 2 vertices.

    Returns
    -------
    Tuple[jax.Array, jax.Array, jax.Array]
        (unit_normal, length, partial_area)
    """
    r1 = simplex[0]
    r2 = simplex[1]
    edge = r2 - r1
    length = jnp.linalg.norm(edge)
    normal = jnp.array([edge[1], -edge[0]])
    unit_normal = normal / jnp.where(length == 0, 1.0, length)
    partial_area = 0.5 * (r1[0] * r2[1] - r1[1] * r2[0])
    return unit_normal, length, partial_area
