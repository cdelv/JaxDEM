# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Implementation of the deformable particle container."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Optional, Dict
from functools import partial

from ..utils.linalg import cross, unit

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
        &E_K = E_{K,measure} + E_{K,content} + E_{K,bending} + E_{K,edge} + E_{K,strain}

    **Definitions per Dimension:**

    * **3D:** Measure ($\mathcal{M}$) is Face Area; Content ($\mathcal{C}$) is Volume; Elements are Triangles.
    * **2D:** Measure ($\mathcal{M}$) is Segment Length; Content ($\mathcal{C}$) is Enclosed Area; Elements are Segments.

    Strain Energy (StVK) on Elements (Triangles):
    .. math::
        W = A_0 \cdot \left( \mu \mathrm{tr}(E^2) + \frac{\lambda}{2} (\mathrm{tr} E)^2 \right)

    All mesh properties are concatenated along axis=0.
    """

    # --- Topology ---
    elements: Optional[jax.Array]
    """
    Array of vertex indices forming the boundary elements.
    Shape: (M, 3) for 3D (Triangles) or (M, 2) for 2D (Segments).
    Indices refer to the particle unique ID corresponding to the `State.pos` array.
    """

    edges: Optional[jax.Array]
    """
    Array of vertex indices forming the unique edges (wireframe). Shape: (E, 2).
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
    Array of vertex IDs forming the shared edge for each adjacency.
    Shape: (A, 2).
    """

    # --- ID Mappings ---
    elements_ID: Optional[jax.Array]
    """
    Array of body IDs for each boundary element. Shape: (M,).
    `elements_ID[i] == k` means element `i` belongs to body `k`.
    """

    edges_ID: Optional[jax.Array]
    """
    Array of body IDs for each unique edge. Shape: (E,).
    `edges_ID[e] == k` means edge `e` belongs to body `k`.
    """

    element_adjacency_ID: Optional[jax.Array]
    """
    Array of body IDs for each adjacency (bending hinge). Shape: (A,).
    `element_adjacency_ID[a] == k` means adjacency `a` belongs to body `k`.
    """

    num_bodies: int
    """
    Total number of distinct deformable bodies in the container. Shape: (K,).
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

    initial_bending: Optional[jax.Array]
    """
    Array of reference (stress-free) bending angles for each adjacency. Shape: (A,).
    Represents Dihedral Angle in 3D or Vertex Angle in 2D.
    """

    # --- Strain Energy Data (SVK) ---
    inv_ref_shape: Optional[jax.Array]
    """
    Inverse of the reference shape matrix for each element.
    Shape: (M, 2, 2) for triangles, or (M, 1, 1) for segments.
    Used to compute the deformation gradient F or Green strain E.
    """
    inv_ref_tet_shape: Optional[jax.Array]
    """
    Inverse of the reference shape matrix for tetrahedra formed by each
    boundary triangle and the corresponding body center. Shape: (M, 3, 3).
    """
    initial_tet_volumes: Optional[jax.Array]
    """
    Reference volumes for tetrahedra formed by each boundary triangle and
    the corresponding body center. Shape: (M,).
    """

    # --- Coefficients ---
    em: Optional[jax.Array]
    """
    Measure elasticity coefficient (Modulus) for each body. Shape: (K,).
    (Controls Area stiffness in 3D; Length stiffness in 2D).
    """

    ec: Optional[jax.Array]
    """
    Content elasticity coefficient (Modulus) for each body. Shape: (K,).
    (Controls Volume stiffness in 3D; Area stiffness in 2D).
    """

    eb: Optional[jax.Array]
    """
    Bending elasticity coefficient (Rigidity) for each body. Shape: (K,).
    """

    el: Optional[jax.Array]
    """
    Edge length elasticity coefficient (Modulus) for each body. Shape: (K,).
    """

    gamma: Optional[jax.Array]
    """
    Surface/Line tension coefficient for each body. Shape: (K,).
    """
    lame_lambda: Optional[jax.Array]
    """
    First Lamé parameter for StVK model. Shape: (K,).
    """
    lame_mu: Optional[jax.Array]
    """
    Second Lamé parameter (Shear Modulus) for StVK model. Shape: (K,).
    """
    use_tetrahedral_svk: bool
    """
    If True, compute StVK strain energy on tetrahedra formed by each boundary
    triangle and the mesh center of its body (3D only). If False, use the
    existing shell-like element StVK model.
    """

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleContainer.create")
    def create(
        vertices: ArrayLike,
        *,
        elements: Optional[ArrayLike] = None,
        edges: Optional[ArrayLike] = None,
        element_adjacency: Optional[ArrayLike] = None,
        # ID mappings
        elements_ID: Optional[ArrayLike] = None,
        edges_ID: Optional[ArrayLike] = None,
        element_adjacency_ID: Optional[ArrayLike] = None,
        # Reference states (computed if None)
        initial_element_measures: Optional[ArrayLike] = None,
        initial_body_contents: Optional[ArrayLike] = None,
        initial_bending: Optional[ArrayLike] = None,
        initial_edge_lengths: Optional[ArrayLike] = None,
        # Coefficients
        em: Optional[ArrayLike] = None,
        ec: Optional[ArrayLike] = None,
        eb: Optional[ArrayLike] = None,
        el: Optional[ArrayLike] = None,
        gamma: Optional[ArrayLike] = None,
        lame_lambda: Optional[ArrayLike] = None,
        lame_mu: Optional[ArrayLike] = None,
        use_tetrahedral_svk: bool = False,
    ) -> DeformableParticleContainer:
        r"""
        Factory method to create a new :class:`DeformableParticleContainer`.

        Calculates initial geometric properties (areas, volumes, bending angles, and edge lengths) from the provided
        `vertices` if they are not explicitly provided.
        """
        v_ref = jnp.asarray(vertices, dtype=float)
        dim = v_ref.shape[-1]

        # 1. Standardize Topologies
        elements = jnp.asarray(elements, dtype=int) if elements is not None else None
        edges = jnp.asarray(edges, dtype=int) if edges is not None else None
        element_adjacency = (
            jnp.asarray(element_adjacency, dtype=int)
            if element_adjacency is not None
            else None
        )

        # 2. Densify Body IDs
        ids_to_check = [
            x for x in [elements_ID, edges_ID, element_adjacency_ID] if x is not None
        ]
        if ids_to_check:
            unique_ids, dense_map = jnp.unique(
                jnp.concatenate([jnp.atleast_1d(x) for x in ids_to_check]),
                return_inverse=True,
            )
            num_bodies = unique_ids.size

            # Helper to extract mapped slices
            cursor = 0

            def get_dense(orig: Optional[ArrayLike]) -> Optional[jax.Array]:
                nonlocal cursor
                if orig is None:
                    return None
                size = jnp.atleast_1d(orig).size
                res = dense_map[cursor : cursor + size]
                cursor += size
                return res

            elements_ID = get_dense(elements_ID)
            edges_ID = get_dense(edges_ID)
            element_adjacency_ID = get_dense(element_adjacency_ID)
        else:
            num_bodies = 1
            elements_ID = (
                jnp.zeros(elements.shape[0], dtype=int)
                if elements is not None
                else None
            )
            edges_ID = (
                jnp.zeros(edges.shape[0], dtype=int) if edges is not None else None
            )
            element_adjacency_ID = (
                jnp.zeros(element_adjacency.shape[0], dtype=int)
                if element_adjacency is not None
                else None
            )

        # 3. Geometric computations
        # Trigger calculation if any relevant coeff is present or if purely creating topology
        calc_geom = True
        adj_edges = None

        if calc_geom and elements is not None:
            compute_fn = (
                compute_element_properties_3D
                if dim == 3
                else compute_element_properties_2D
            )
            norms, measures, partial_contents = jax.vmap(compute_fn)(v_ref[elements])

            initial_element_measures = (
                jnp.asarray(initial_element_measures)
                if initial_element_measures is not None
                else measures
            )

            if elements_ID is not None:
                if initial_body_contents is None:
                    initial_body_contents = jax.ops.segment_sum(
                        partial_contents, elements_ID, num_segments=num_bodies
                    )
                else:
                    initial_body_contents = jnp.asarray(initial_body_contents)

            if element_adjacency is not None:
                # Bending angles
                n1, n2 = norms[element_adjacency[:, 0]], norms[element_adjacency[:, 1]]
                initial_bending = (
                    jnp.asarray(initial_bending)
                    if initial_bending is not None
                    else jax.vmap(angle_between_normals)(n1, n2)
                )

                # 3D Winding preservation for bending
                if dim == 3:
                    f1 = elements[element_adjacency[:, 0]]
                    matches = (
                        f1[:, :, None] == elements[element_adjacency[:, 1]][:, None, :]
                    )
                    missing_idx = jnp.argmin(
                        jnp.any(matches, axis=2).astype(int), axis=1
                    )
                    v_start = jnp.take_along_axis(
                        f1, ((missing_idx + 1) % 3)[:, None], axis=1
                    )
                    v_end = jnp.take_along_axis(
                        f1, ((missing_idx + 2) % 3)[:, None], axis=1
                    )
                    adj_edges = jnp.concatenate([v_start, v_end], axis=1)

        if edges is not None:
            lengths = jnp.linalg.norm(v_ref[edges[:, 1]] - v_ref[edges[:, 0]], axis=-1)
            initial_edge_lengths = (
                jnp.asarray(initial_edge_lengths)
                if initial_edge_lengths is not None
                else lengths
            )

        # 4. Precompute SVK Reference Shape Inverses
        inv_ref_shape = None
        inv_ref_tet_shape = None
        initial_tet_volumes = None
        if (lame_lambda is not None or lame_mu is not None) and elements is not None:
            inv_ref_shape = jax.vmap(compute_inverse_reference_shape)(v_ref[elements])
            if use_tetrahedral_svk and dim == 3 and elements.shape[1] == 3:
                ref_centers = compute_body_centers_from_elements(
                    v_ref, elements, elements_ID, num_bodies
                )
                ref_tets = jnp.concatenate(
                    [ref_centers[elements_ID][:, None, :], v_ref[elements]], axis=1
                )
                inv_ref_tet_shape = jax.vmap(compute_inverse_reference_shape_tet)(
                    ref_tets
                )
                initial_tet_volumes = jax.vmap(compute_tetra_volume)(ref_tets)

        return DeformableParticleContainer(
            elements=elements,
            edges=edges,
            element_adjacency=element_adjacency,
            element_adjacency_edges=adj_edges,
            elements_ID=jnp.asarray(elements_ID) if elements_ID is not None else None,
            edges_ID=jnp.asarray(edges_ID) if edges_ID is not None else None,
            element_adjacency_ID=(
                jnp.asarray(element_adjacency_ID)
                if element_adjacency_ID is not None
                else None
            ),
            num_bodies=num_bodies,
            initial_element_measures=(
                jnp.asarray(initial_element_measures)
                if initial_element_measures is not None
                else None
            ),
            initial_body_contents=(
                jnp.asarray(initial_body_contents)
                if initial_body_contents is not None
                else None
            ),
            initial_bending=(
                jnp.asarray(initial_bending) if initial_bending is not None else None
            ),
            initial_edge_lengths=(
                jnp.asarray(initial_edge_lengths)
                if initial_edge_lengths is not None
                else None
            ),
            inv_ref_shape=inv_ref_shape,
            inv_ref_tet_shape=inv_ref_tet_shape,
            initial_tet_volumes=initial_tet_volumes,
            em=jnp.asarray(em) if em is not None else None,
            ec=jnp.asarray(ec) if ec is not None else None,
            eb=jnp.asarray(eb) if eb is not None else None,
            el=jnp.asarray(el) if el is not None else None,
            gamma=jnp.asarray(gamma) if gamma is not None else None,
            lame_lambda=jnp.asarray(lame_lambda) if lame_lambda is not None else None,
            lame_mu=jnp.asarray(lame_mu) if lame_mu is not None else None,
            use_tetrahedral_svk=bool(use_tetrahedral_svk),
        )

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleContainer.merge")
    def merge(
        c1: DeformableParticleContainer,
        c2: DeformableParticleContainer,
    ) -> DeformableParticleContainer:
        r"""
        Merges two :class:`DeformableParticleContainer` instances.
        """
        if c2.elements_ID is not None:
            c2.elements_ID += c1.num_bodies

        if c2.edges_ID is not None:
            c2.edges_ID += c1.num_bodies

        if c2.element_adjacency_ID is not None:
            c2.element_adjacency_ID += c1.num_bodies

        def cat(a: jax.Array, b: jax.Array) -> jax.Array:
            if isinstance(a, jax.Array) and isinstance(b, jax.Array):
                return jnp.concatenate((a, b), axis=0)
            elif a is None and b is None:
                return None
            else:
                return a if a is not None else b

        merged = jax.tree_util.tree_map(cat, c1, c2)
        merged.num_bodies = c1.num_bodies + c2.num_bodies

        return merged

    @staticmethod
    @partial(jax.named_call, name="DeformableParticleContainer.add")
    def add(
        container: DeformableParticleContainer,
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
        lame_lambda: Optional[ArrayLike] = None,
        lame_mu: Optional[ArrayLike] = None,
        use_tetrahedral_svk: bool = False,
    ) -> DeformableParticleContainer:
        r"""
        Factory method to add bodies to a container.
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
            lame_lambda=lame_lambda,
            lame_mu=lame_mu,
            use_tetrahedral_svk=use_tetrahedral_svk,
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
        E_strain = jnp.array(0.0, dtype=float)

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
            # (M - M0)^2 / M0
            diff = element_measure - container.initial_element_measures
            norm_strain_energy = jnp.square(diff) / container.initial_element_measures

            temp_elements = jax.ops.segment_sum(
                norm_strain_energy,
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
            # (V - V0)^2 / V0
            diff = content - container.initial_body_contents
            norm_vol_energy = jnp.square(diff) / container.initial_body_contents
            E_content = 0.5 * jnp.sum(container.ec * norm_vol_energy)

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
            n1 = element_normal[container.element_adjacency[:, 0]]
            n2 = element_normal[container.element_adjacency[:, 1]]
            cos = jnp.sum(n1 * n2, axis=-1)

            if dim == 3 and container.element_adjacency_edges is not None:
                hinge_idx = idx_map[container.element_adjacency_edges]  # (A, 2)
                h_verts = vertices[hinge_idx]  # (A, 2, 3)
                tangent_vec = h_verts[:, 1, :] - h_verts[:, 0, :]
                tangent = unit(tangent_vec)
                cross_prod = cross(n1, n2)
                sin = jnp.sum(cross_prod * tangent, axis=-1)
            else:
                sin = cross(n1, n2)
                sin = jnp.squeeze(sin)

            bending = jnp.atan2(sin, cos)
            diff = bending - container.initial_bending
            temp_angles = jax.ops.segment_sum(
                jnp.square(diff),
                container.element_adjacency_ID,
                num_segments=container.num_bodies,
            )
            E_bending = 0.5 * jnp.sum(container.eb * temp_angles) / 2

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

            diff = edge_lengths - container.initial_edge_lengths
            norm_edge_energy = jnp.square(diff) / container.initial_edge_lengths

            temp_edges = jax.ops.segment_sum(
                norm_edge_energy,
                container.edges_ID,
                num_segments=container.num_bodies,
            )
            E_edge = 0.5 * jnp.sum(container.el * temp_edges)

        # StVK Strain Energy (Corrected per-element)
        if (
            container.lame_lambda is not None
            and container.lame_mu is not None
            and container.elements is not None
            and container.elements_ID is not None
        ):
            if (
                container.use_tetrahedral_svk
                and dim == 3
                and container.inv_ref_tet_shape is not None
                and container.initial_tet_volumes is not None
            ):
                curr_verts = vertices[current_element_indices]
                curr_centers = compute_body_centers_from_elements(
                    vertices,
                    current_element_indices,
                    container.elements_ID,
                    container.num_bodies,
                )
                curr_d_vecs = jnp.swapaxes(
                    curr_verts - curr_centers[container.elements_ID][:, None, :], -1, -2
                )
                F = curr_d_vecs @ container.inv_ref_tet_shape
                C = jnp.swapaxes(F, -1, -2) @ F
                E = 0.5 * (C - jnp.eye(3))
                tr_E = jnp.trace(E, axis1=-2, axis2=-1)
                tr_E2 = jnp.sum(E * E, axis=(-1, -2))
                mu = container.lame_mu[container.elements_ID]
                lam = container.lame_lambda[container.elements_ID]
                W = mu * tr_E2 + 0.5 * lam * (tr_E**2)
                E_strain = jnp.sum(W * container.initial_tet_volumes)
            elif container.inv_ref_shape is not None:
                # Compute deformation using vectorized batch operations (M, dim, rank)
                # 1. Gather current vertices: (M, rank+1, dim)
                curr_verts = vertices[current_element_indices]

                # 2. Compute current edge vectors d: (M, dim, rank)
                # d_j = x_{j+1} - x_0
                d_vecs = jnp.swapaxes(curr_verts[:, 1:] - curr_verts[:, 0:1], -1, -2)

                # 3. Compute Deformation Gradient F = d @ D_inv
                # d_vecs: (M, dim, rank), inv_ref_shape: (M, rank, rank) -> F: (M, dim, rank)
                F = d_vecs @ container.inv_ref_shape

                # 4. Compute Green-Lagrange Strain E = 0.5 * (F.T @ F - I)
                # C = F.T @ F (Right Cauchy-Green, pulled back to local 2D/1D ref manifold)
                C = jnp.swapaxes(F, -1, -2) @ F

                rank = container.inv_ref_shape.shape[-1]
                I = jnp.eye(rank)
                E = 0.5 * (C - I)

                # 5. Compute Invariants
                # tr(E): Trace of (M, rank, rank) -> (M,)
                tr_E = jnp.trace(E, axis1=-2, axis2=-1)

                # tr(E^2) = sum(E_ij * E_ji) -> sum(E_ij^2) for symmetric E
                tr_E2 = jnp.sum(E * E, axis=(-1, -2))

                # 6. Map coefficients and compute Energy Density W
                mu = container.lame_mu[container.elements_ID]
                lam = container.lame_lambda[container.elements_ID]

                # Energy Density W (Energy per unit measure)
                # W = mu * tr(E^2) + 0.5 * lambda * tr(E)^2
                W = mu * tr_E2 + 0.5 * lam * (tr_E**2)

                # 7. Total Strain Energy
                # E_total = sum(W_i * A0_i)
                # Note: Thickness is excluded as requested.
                E_strain = jnp.sum(W * container.initial_element_measures)

        aux = dict(
            E_element=E_element,
            E_content=E_content,
            E_gamma=E_gamma,
            E_bending=E_bending,
            E_edge=E_edge,
            E_strain=E_strain,
        )
        return E_element + E_content + E_gamma + E_bending + E_edge + E_strain, aux

    @staticmethod
    def create_force_function(
        container: DeformableParticleContainer,
    ) -> ForceFunction:
        force_fn, _ = DeformableParticleContainer.create_force_energy_functions(
            container
        )
        return force_fn

    @staticmethod
    def create_force_energy_functions(
        container: DeformableParticleContainer,
    ) -> Tuple[ForceFunction, EnergyFunction]:
        def force_function(
            pos: jax.Array, state: State, system: System
        ) -> Tuple[jax.Array, jax.Array]:
            energy_grad, _ = jax.grad(
                DeformableParticleContainer.compute_potential_energy, has_aux=True
            )(pos, state, system, container)
            return -energy_grad, jnp.zeros_like(state.torque)

        def energy_function(
            pos: jax.Array, state: State, system: System
        ) -> jax.Array:
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
    cos = jnp.sum(n1 * n2, axis=-1)
    sin = cross(n1, n2)
    sin = jnp.sum(sin * sin, axis=-1)
    sin = jnp.sqrt(sin)
    return jnp.atan2(sin, cos)


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
    return (
        face_normal / jnp.where(area_face == 0, 1, area_face),
        area_face,
        partial_vol,
    )


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
    length2 = jnp.sum(edge * edge, axis=-1)
    length = jnp.sqrt(length2)
    normal = jnp.array([edge[1], -edge[0]])
    normal /= jnp.where(length == 0, 1.0, length)
    partial_area = 0.5 * (r1[0] * r2[1] - r1[1] * r2[0])
    return normal, length, partial_area


def compute_inverse_reference_shape(simplex: jax.Array) -> jax.Array:
    """
    Computes the inverse of the reference shape matrix (mapping local edge basis to reference coordinates).

    For a triangle (3 verts), constructs a local 2D basis and inverts the mapping [X2-X1, X3-X1].
    For a segment (2 verts), inverts the length.

    Returns:
        (2, 2) matrix for triangles, or (1, 1) for segments.
    """
    n_verts = simplex.shape[0]

    if n_verts == 3:
        # Triangle (Rank 2)
        d1 = simplex[1] - simplex[0]
        d2 = simplex[2] - simplex[0]

        # Construct local orthonormal basis (u, v) aligned with d1
        u = unit(d1)
        # Project d2 onto plane (assuming 3D embedding)
        # For a triangle, they define a plane.
        # Gram-Schmidt for v
        v_temp = d2 - jnp.dot(d2, u) * u
        v = unit(v_temp)

        # Local coordinates of the edge vectors:
        # edge1_loc = ( |d1|, 0 )
        # edge2_loc = ( d2.u, d2.v )

        # Shape Matrix B = [edge1_loc, edge2_loc] (Columns are edge vectors)
        # B = [[|d1|, d2.u],
        #      [0,    d2.v]]

        b11 = jnp.linalg.norm(d1)
        b12 = jnp.dot(d2, u)
        b22 = jnp.dot(d2, v)

        # Analytic Inverse of Upper Triangular 2x2
        # invB = 1/(b11*b22) * [[b22, -b12], [0, b11]]
        det = b11 * b22
        # Avoid NaN in degenerate case
        safe_det = jnp.where(jnp.abs(det) < 1e-12, 1.0, det)

        inv_shape = (1.0 / safe_det) * jnp.array([[b22, -b12], [0.0, b11]])
        return inv_shape

    elif n_verts == 2:
        # Segment (Rank 1)
        d1 = simplex[1] - simplex[0]
        l0 = jnp.linalg.norm(d1)
        safe_l0 = jnp.where(l0 < 1e-12, 1.0, l0)
        return jnp.array([[1.0 / safe_l0]])

    else:
        # Fallback (Should not happen given fixed element shapes)
        return jnp.eye(n_verts - 1)


def compute_body_centers_from_elements(
    vertices: jax.Array,
    elements: jax.Array,
    elements_ID: jax.Array,
    num_bodies: int,
) -> jax.Array:
    node_mask = jnp.zeros((num_bodies, vertices.shape[0]), dtype=bool)
    body_rows = jnp.broadcast_to(elements_ID[:, None], elements.shape)
    node_mask = node_mask.at[body_rows, elements].set(True)
    counts = jnp.sum(node_mask.astype(vertices.dtype), axis=1)
    counts = jnp.where(counts == 0.0, 1.0, counts)
    centers = node_mask.astype(vertices.dtype) @ vertices
    return centers / counts[:, None]


def compute_inverse_reference_shape_tet(tet: jax.Array) -> jax.Array:
    d1 = tet[1] - tet[0]
    d2 = tet[2] - tet[0]
    d3 = tet[3] - tet[0]
    D = jnp.stack([d1, d2, d3], axis=-1)
    det = jnp.linalg.det(D)
    safe_D = jnp.where(jnp.abs(det) < 1e-12, jnp.eye(3), D)
    return jnp.linalg.inv(safe_D)


def compute_tetra_volume(tet: jax.Array) -> jax.Array:
    d1 = tet[1] - tet[0]
    d2 = tet[2] - tet[0]
    d3 = tet[3] - tet[0]
    return jnp.abs(jnp.dot(d1, cross(d2, d3))) / 6.0
