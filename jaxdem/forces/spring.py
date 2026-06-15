# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Linear spring force model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ..utils.linalg import cross, norm, unit, unit_and_norm
from . import ForceModel
from .facet_contact import (
    get_facet_indices,
    point_segment_distance,
    point_triangle_distance,
    segment_segment_distance,
    triangle_triangle_distance,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@ForceModel.register("spring")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SpringForce(ForceModel):
    r"""A `ForceModel` implementation for a linear spring-like interaction between particles.

    Notes
    -----
    - The 'effective Young's modulus' (:math:`k_{eff,\; ij}`) is retrieved from the
      :attr:`jaxdem.System.mat_table` based on the material IDs of the interacting particles.
    - The force is zero if :math:`i == j`.
    - Distances and normals are computed with the zero-safe double-``where``
      helpers in :mod:`jaxdem.utils.linalg`, so the force and its gradients
      remain finite when particles are perfectly co-located.

    The penetration :math:`\delta` (overlap) between two particles :math:`i` and :math:`j` is:

    .. math::
        \delta = \max\left(0, (R_i + R_j) - r\right)

    where :math:`R_i` and :math:`R_j` are the radii of particles :math:`i` and :math:`j` respectively,
    and :math:`r = ||r_{ij}||` is the distance between their centers.

    The force :math:`F_{ij}` acting on particle :math:`i` due to particle :math:`j` is:

    .. math::
        F_{ij} = k_{eff,\; ij}\, \delta\, \hat{n}_{ij}

    where :math:`\hat{n}_{ij} = \vec{r}_{ij} / r` is the unit vector from particle
    :math:`j` to particle :math:`i`.

    The potential energy :math:`E_{ij}` of the interaction is:

    .. math::
        E_{ij} = \frac{1}{2} k_{eff,\; ij} \delta^2

    where :math:`k_{eff,\; ij}` is the effective Young's modulus for the particle pair.

    """

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SpringForce.force")
    def force(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> tuple[jax.Array, jax.Array]:
        """Compute linear spring-like interaction force acting on particle :math:`i` due to particle :math:`j`.

        Returns zero when :math:`i = j`.

        Parameters
        ----------
        i : int
            Index of the first particle.
        j : int
            Index of the second particle.
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            ``(force, torque)`` with shapes ``(dim,)`` and ``(ang_dim,)``.
            The torque is always zero for this model.

        """
        mi, mj = state.mat_id[i], state.mat_id[j]
        k = system.mat_table.young_eff[mi, mj]
        R = state.rad[i] + state.rad[j]

        rij = system.domain.displacement(pos[i], pos[j], system)
        n, r = unit_and_norm(rij)
        r = r[..., 0]
        delta = jnp.maximum(0.0, R - r) * (i != j)
        return (k * delta)[..., None] * n, jnp.zeros_like(state.torque[i])

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SpringForce.energy")
    def energy(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> jax.Array:
        """Compute linear spring-like interaction potential energy between particle :math:`i` and particle :math:`j`.

        Returns zero when :math:`i = j`.

        Parameters
        ----------
        i : int
            Index of the first particle.
        j : int
            Index of the second particle.
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        jax.Array
            Scalar JAX array representing the potential energy of the interaction
            between particles :math:`i` and :math:`j`.

        """
        mi, mj = state.mat_id[i], state.mat_id[j]
        k = system.mat_table.young_eff[mi, mj]
        R = state.rad[i] + state.rad[j]

        rij = system.domain.displacement(pos[i], pos[j], system)
        r = norm(rij)
        s = R - r
        s *= (s > 0) * (i != j)
        return 0.5 * k * s**2

    @property
    def required_material_properties(self) -> tuple[str, ...]:
        """A static tuple of strings specifying the material properties required by this force model.

        These properties (e.g., 'young_eff', 'restitution', ...) must be present in the
        :attr:`System.mat_table` for the model to function correctly. This is used
        for validation.
        """
        return ("young_eff",)


@jax.jit(inline=True)
def _sphere_facet_pair(
    i: jax.Array, j: jax.Array, pos: jax.Array, state: "State", system: "System"
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Shared sphere-facet pair kernel used by both ``force`` and ``energy``.

    Returns ``(k, delta, is_contact, w, n, c_1, thick_i, is_rigid)`` where
    ``delta`` is the (masked) overlap, ``w`` the partition-of-unity weight,
    ``n`` the contact normal and ``c_1`` the contact point on body ``i``.
    Quantities unused by a caller are removed by XLA dead-code elimination.
    """
    dim = pos.shape[-1]

    is_facet_i, idxs_i, is_primary_i = get_facet_indices(i, state)
    is_facet_j, idxs_j, is_primary_j = get_facet_indices(j, state)

    idx0_i = jnp.where(is_facet_i, idxs_i[..., 0], i)
    idx0_j = jnp.where(is_facet_j, idxs_j[..., 0], j)

    is_rigid_i = jnp.where(
        is_facet_i,
        jnp.all(state.clump_id[idxs_i] == state.clump_id[idxs_i[..., 0:1]], axis=-1),
        True,
    )
    is_rigid_j = jnp.where(
        is_facet_j,
        jnp.all(state.clump_id[idxs_j] == state.clump_id[idxs_j[..., 0:1]], axis=-1),
        True,
    )
    is_rigid = is_rigid_i * is_rigid_j

    compute_interaction = jnp.where(
        is_rigid,
        jnp.where(is_facet_i, is_primary_i, True)
        * jnp.where(is_facet_j, is_primary_j, True),
        True,
    ) * (is_facet_i ^ is_facet_j)

    mi, mj = state.mat_id[idx0_i], state.mat_id[idx0_j]
    k = system.mat_table.young_eff[mi, mj]

    v_a = jnp.where(is_facet_i[..., None], pos[idxs_i[..., 0]], pos[idxs_j[..., 0]])
    v_b = jnp.where(is_facet_i[..., None], pos[idxs_i[..., 1]], pos[idxs_j[..., 1]])
    if dim == 3:
        v_c = jnp.where(is_facet_i[..., None], pos[idxs_i[..., 2]], pos[idxs_j[..., 2]])
    else:
        v_c = None

    p = jnp.where(is_facet_i[..., None], pos[idx0_j], pos[idx0_i])

    if dim == 3:
        d_sf, c_f, coords_f = point_triangle_distance(p, v_a, v_b, v_c, system)
    else:
        d_sf, c_f, coords_f = point_segment_distance(p, v_a, v_b, system)

    c_1 = jnp.where(is_facet_i[..., None], c_f, p)
    c_2 = jnp.where(is_facet_j[..., None], c_f, p)

    thick_i = state.rad[idx0_i]
    thick_j = state.rad[idx0_j]

    rij = system.domain.displacement(c_1, c_2, system)
    n, r = unit_and_norm(rij)

    fallback_rij = system.domain.displacement(
        state.pos_c[idx0_i], state.pos_c[idx0_j], system
    )
    n = jnp.where(r[..., 0:1] > 1e-7, n, unit(fallback_rij))

    delta = thick_i + thick_j - d_sf
    is_contact = (delta > 0) * compute_interaction
    delta *= is_contact

    idxs_facet = jnp.where(is_facet_i[..., None], idxs_i, idxs_j)
    particle_facet = jnp.where(is_facet_i, i, j)
    v_idx = jnp.argmax(idxs_facet == particle_facet[..., None], axis=-1)
    w_vertex = jnp.sum(
        coords_f * jax.nn.one_hot(v_idx, dim, dtype=coords_f.dtype), axis=-1
    )

    w = jnp.where(is_rigid, 1.0, w_vertex)

    return k, delta, is_contact, w, n, c_1, thick_i, is_rigid


@ForceModel.register("sphere_facet_spring")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SphereFacetSpringForce(ForceModel):
    r"""Linear spring contact between spheres and facets.

    .. warning::
        Facet contacts are detected through the facet's *vertex spheres*
        (in particular the facet's primary vertex). The collider's neighbor
        cutoff must therefore be at least the facet circumradius (the largest
        vertex-to-contact-point distance) plus the contact thickness
        (``state.rad``). If the primary vertex lies outside the cutoff while
        the contact point is in range, the contact is silently missed or
        applied asymmetrically. Cell-list based colliders must be configured
        with ``cutoff >= max facet circumradius + thickness``.

    The contact thickness is taken from the facet vertices' ``state.rad``
    (set via ``State.add_facet(thickness=...)``); the force model itself
    has no thickness parameter.
    """

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SphereFacetSpringForce.force")
    def force(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> tuple[jax.Array, jax.Array]:
        k, delta, is_contact, w, n, c_1, thick_i, is_rigid = _sphere_facet_pair(
            i, j, pos, state, system
        )

        fn_mag = jnp.maximum(0.0, k * delta)
        f_total = (fn_mag * is_contact * w)[..., None] * n

        r_ci = (
            system.domain.displacement(c_1, state.pos[i], system)
            - thick_i[..., None] * n
        )

        t_total = cross(r_ci, f_total)

        is_rigid_mask = is_rigid
        while is_rigid_mask.ndim < t_total.ndim:
            is_rigid_mask = is_rigid_mask[..., None]
        t_total = t_total * is_rigid_mask

        return f_total, t_total

    @staticmethod
    @jax.jit(inline=True)
    def energy(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> jax.Array:
        k, delta, is_contact, w, n, c_1, thick_i, is_rigid = _sphere_facet_pair(
            i, j, pos, state, system
        )
        return 0.5 * k * delta * delta * w


@jax.jit(inline=True)
def _facet_facet_pair(
    i: jax.Array, j: jax.Array, pos: jax.Array, state: "State", system: "System"
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Shared facet-facet pair kernel used by both ``force`` and ``energy``.

    Returns ``(k, delta, is_contact, w, n, c_ff_1, thick_i, is_rigid)`` where
    ``delta`` is the (masked) overlap, ``w`` the partition-of-unity weight,
    ``n`` the contact normal and ``c_ff_1`` the contact point on facet ``i``.
    Quantities unused by a caller are removed by XLA dead-code elimination.
    """
    dim = pos.shape[-1]
    i_arr = jnp.asarray(i)
    j_arr = jnp.asarray(j)

    is_facet_i, idxs_i, is_primary_i = get_facet_indices(i, state)
    is_facet_j, idxs_j, is_primary_j = get_facet_indices(j, state)

    idx0_i = jnp.where(is_facet_i, idxs_i[..., 0], i)
    idx0_j = jnp.where(is_facet_j, idxs_j[..., 0], j)

    is_rigid_i = jnp.where(
        is_facet_i,
        jnp.all(state.clump_id[idxs_i] == state.clump_id[idxs_i[..., 0:1]], axis=-1),
        True,
    )
    is_rigid_j = jnp.where(
        is_facet_j,
        jnp.all(state.clump_id[idxs_j] == state.clump_id[idxs_j[..., 0:1]], axis=-1),
        True,
    )
    is_rigid = is_rigid_i & is_rigid_j

    compute_interaction = (
        jnp.where(
            is_rigid,
            is_primary_i & is_primary_j,
            True,
        )
        & is_facet_i
        & is_facet_j
        & (idx0_i != idx0_j)
    )

    mi, mj = state.mat_id[idx0_i], state.mat_id[idx0_j]
    k = system.mat_table.young_eff[mi, mj]

    t1_a = pos[idxs_i[..., 0]]
    t1_b = pos[idxs_i[..., 1]]
    t2_a = pos[idxs_j[..., 0]]
    t2_b = pos[idxs_j[..., 1]]

    swap = idx0_i > idx0_j

    tA_a = jnp.where(swap[..., None], t2_a, t1_a)
    tA_b = jnp.where(swap[..., None], t2_b, t1_b)
    tB_a = jnp.where(swap[..., None], t1_a, t2_a)
    tB_b = jnp.where(swap[..., None], t1_b, t2_b)

    if dim == 3:
        t1_c = pos[idxs_i[..., 2]]
        t2_c = pos[idxs_j[..., 2]]
        tA_c = jnp.where(swap[..., None], t2_c, t1_c)
        tB_c = jnp.where(swap[..., None], t1_c, t2_c)
        tA_a, tA_b, tA_c, tB_a, tB_b, tB_c = jnp.broadcast_arrays(
            tA_a, tA_b, tA_c, tB_a, tB_b, tB_c
        )
        d_ff, c_A, c_B, coords_A, coords_B = triangle_triangle_distance(
            tA_a, tA_b, tA_c, tB_a, tB_b, tB_c, system
        )
    else:
        tA_a, tA_b, tB_a, tB_b = jnp.broadcast_arrays(tA_a, tA_b, tB_a, tB_b)
        d_ff, c_A, c_B, coords_A, coords_B = segment_segment_distance(
            tA_a, tA_b, tB_a, tB_b, system
        )

    c_ff_1 = jnp.where(swap[..., None], c_B, c_A)
    c_ff_2 = jnp.where(swap[..., None], c_A, c_B)
    coords_1 = jnp.where(swap[..., None], coords_B, coords_A)
    coords_2 = jnp.where(swap[..., None], coords_A, coords_B)

    thick_i = state.rad[idx0_i]
    thick_j = state.rad[idx0_j]

    delta = thick_i + thick_j - d_ff
    is_contact = (delta > 0) & compute_interaction
    delta *= is_contact

    rij = system.domain.displacement(c_ff_1, c_ff_2, system)
    n, r = unit_and_norm(rij)

    fallback_rij = system.domain.displacement(
        state.pos_c[idx0_i], state.pos_c[idx0_j], system
    )
    n = jnp.where(r[..., 0:1] > 1e-7, n, unit(fallback_rij))

    v_idx_i = jnp.argmax(idxs_i == i_arr[..., None], axis=-1)
    w_vertex_i = jnp.sum(
        coords_1 * jax.nn.one_hot(v_idx_i, dim, dtype=coords_1.dtype), axis=-1
    )

    v_idx_j = jnp.argmax(idxs_j == j_arr[..., None], axis=-1)
    w_vertex_j = jnp.sum(
        coords_2 * jax.nn.one_hot(v_idx_j, dim, dtype=coords_2.dtype), axis=-1
    )

    w = jnp.where(is_rigid, 1.0, w_vertex_i * w_vertex_j)

    return k, delta, is_contact, w, n, c_ff_1, thick_i, is_rigid


@ForceModel.register("facet_facet_spring")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class FacetFacetSpringForce(ForceModel):
    r"""Linear spring contact between facets.

    .. warning::
        Facet contacts are detected through the facets' *vertex spheres*
        (in particular each facet's primary vertex). The collider's neighbor
        cutoff must therefore be at least the facet circumradius (the largest
        vertex-to-contact-point distance) plus the contact thickness
        (``state.rad``). If a primary vertex lies outside the cutoff while
        the contact point is in range, the contact is silently missed or
        applied asymmetrically. Cell-list based colliders must be configured
        with ``cutoff >= max facet circumradius + thickness``.

    The contact thickness is taken from the facet vertices' ``state.rad``
    (set via ``State.add_facet(thickness=...)``); the force model itself
    has no thickness parameter.
    """

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="FacetFacetSpringForce.force")
    def force(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> tuple[jax.Array, jax.Array]:
        k, delta, is_contact, w, n, c_ff_1, thick_i, is_rigid = _facet_facet_pair(
            i, j, pos, state, system
        )

        fn_mag = jnp.maximum(0.0, k * delta)
        f_total = (fn_mag * is_contact * w)[..., None] * n

        r_ci = (
            system.domain.displacement(c_ff_1, state.pos[i], system)
            - thick_i[..., None] * n
        )

        t_total = cross(r_ci, f_total)

        is_rigid_mask = is_rigid
        while is_rigid_mask.ndim < t_total.ndim:
            is_rigid_mask = is_rigid_mask[..., None]
        t_total = t_total * is_rigid_mask

        return f_total, t_total

    @staticmethod
    @jax.jit(inline=True)
    def energy(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> jax.Array:
        k, delta, is_contact, w, n, c_ff_1, thick_i, is_rigid = _facet_facet_pair(
            i, j, pos, state, system
        )
        return 0.5 * k * delta * delta * w


__all__ = ["SpringForce", "SphereFacetSpringForce", "FacetFacetSpringForce"]
