# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Linear spring force model."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

from ..utils.linalg import cross, dot, norm, norm2, unit, unit_and_norm
from . import ForceModel
from ._facet_incidence import get_facet_incidence
from .facet_contact import (
    facet_normal,
    facet_search_radii,
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
    r"""Linear spring-like interaction between particles.

    Notes
    -----
    - The model reads the 'effective Young's modulus' (:math:`k_{eff,\; ij}`)
      from the :attr:`jaxdem.System.mat_table <jaxdem.system.System.mat_table>` using the material IDs of the
      interacting particles.
    - The force is zero if :math:`i == j`.
    - The model computes distances and normals with the zero-safe
      double-``where`` helpers in :mod:`jaxdem.utils.linalg`. The force and
      its gradients stay finite when particles are perfectly co-located.

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

    def search_radii(self, state: State, system: System) -> jax.Array:
        """Conservative search extent for this law's finite interaction range."""
        return jnp.maximum(state._rad, (1.0) * state.rad)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SpringForce.force")
    def force(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
        *,
        advance_history: bool = True,
        quasistatic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute the linear spring force on particle :math:`i` from particle :math:`j`.

        Returns zero when :math:`i = j`.

        Parameters
        ----------
        i : int
            Index of the first particle.
        j : int
            Index of the second particle.
        pos : jax.Array
            Particle positions.
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.
        history : jax.Array
            Pair history, returned unchanged by this stateless law.
        advance_history, quasistatic : bool, optional
            Unused by this stateless law.

        Returns
        -------
        tuple[jax.Array, jax.Array, jax.Array]
            ``(force, torque, history)``. Force and torque have shapes
            ``(dim,)`` and ``(ang_dim,)`` for a single pair. Torque is zero.

        """
        R = state.rad[i] + state.rad[j]
        rij = system.domain._displacement(pos[i], pos[j], system)
        n2 = norm2(rij)
        zero = n2 == 0.0
        safe_n2 = jnp.where(zero, 1.0, n2)
        inv = jnp.where(zero, 0.0, jax.lax.rsqrt(safe_n2))
        r = jnp.where(zero, 0.0, jnp.sqrt(safe_n2))

        mi, mj = state.mat_id[i], state.mat_id[j]
        k = system.mat_table.young_eff[mi, mj]
        delta = jnp.maximum(0.0, R - r) * (i != j)
        coefficient = (k * delta) * inv
        t_shape = jnp.shape(j) + jnp.shape(state.torque[i])
        return (
            coefficient[..., None] * rij,
            jnp.zeros(t_shape, dtype=state.torque.dtype),
            history,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SpringForce.energy")
    def energy(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
    ) -> jax.Array:
        """Compute the linear spring potential energy between particle :math:`i` and particle :math:`j`.

        Returns zero when :math:`i = j`.

        Parameters
        ----------
        i : int
            Index of the first particle.
        j : int
            Index of the second particle.
        pos : jax.Array
            Particle positions.
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        jax.Array
            Scalar potential energy of the interaction between particles
            :math:`i` and :math:`j`.

        """
        R = state.rad[i] + state.rad[j]
        rij = system.domain._displacement(pos[i], pos[j], system)
        r = norm(rij)

        mi, mj = state.mat_id[i], state.mat_id[j]
        k = system.mat_table.young_eff[mi, mj]
        s = jnp.maximum(0.0, R - r) * (i != j)
        return 0.5 * k * s * s

    @property
    def required_material_properties(self) -> tuple[str, ...]:
        """Names of the material properties this force model needs.

        Each name (for example 'young_eff' or 'restitution') must be present
        in :attr:`System.mat_table <jaxdem.system.System.mat_table>`. Used for validation.
        """
        return ("young_eff",)


def _inactive_facet_pair(i: jax.Array, j: jax.Array, pos: jax.Array) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Return finite, broadcast-compatible data when all queries are disabled."""
    shape = jnp.broadcast_shapes(jnp.shape(i), jnp.shape(j))
    zero = jnp.zeros(shape, dtype=pos.dtype)
    vector = jnp.zeros((*shape, pos.shape[-1]), dtype=pos.dtype)
    return zero, zero, zero > 0, zero, vector, vector, zero, zero == 0


_FacetPairData = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
_FacetPairKernel = Callable[
    [
        int | jax.Array,
        int | jax.Array,
        jax.Array,
        "State",
        "System",
        int | jax.Array,
        int | jax.Array,
    ],
    _FacetPairData,
]


def _sum_facet_forces(
    pair_kernel: _FacetPairKernel,
    i: int | jax.Array,
    j: int | jax.Array,
    pos: jax.Array,
    state: State,
    system: System,
) -> tuple[jax.Array, jax.Array]:
    """Sum incident face contacts onto the existing physical vertex keys."""
    slots = 1 + state.facet_ids_extra.shape[-1]

    def evaluate(
        slot_i: int | jax.Array, slot_j: int | jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        k, delta, contact, w, n, point, thickness, rigid = pair_kernel(
            i, j, pos, state, system, slot_i, slot_j
        )
        force = (jnp.maximum(0.0, k * delta) * contact * w)[..., None] * n
        arm = (
            system.domain._displacement(point, state.pos[i], system)
            - thickness[..., None] * n
        )
        torque = cross(arm, force)
        rigid = jnp.broadcast_to(rigid, force.shape[:-1])
        return force, torque * rigid[..., None]

    # Keep the original compiled path for states without shared vertices.
    if slots == 1:
        return evaluate(0, 0)
    if pair_kernel is _sphere_facet_pair:
        # One member must be a sphere, whose only active incidence is slot zero.
        slot_i = jnp.concatenate((jnp.arange(slots), jnp.zeros(slots - 1, dtype=int)))
        slot_j = jnp.concatenate((jnp.zeros(slots, dtype=int), jnp.arange(1, slots)))
    else:
        slot = jnp.arange(slots * slots)
        slot_i, slot_j = slot // slots, slot % slots
    forces, torques = jax.vmap(evaluate)(slot_i, slot_j)
    return jnp.sum(forces, axis=0), jnp.sum(torques, axis=0)


def _sum_facet_energies(
    pair_kernel: _FacetPairKernel,
    i: int | jax.Array,
    j: int | jax.Array,
    pos: jax.Array,
    state: State,
    system: System,
) -> jax.Array:
    slots = 1 + state.facet_ids_extra.shape[-1]

    def evaluate(slot_i: int | jax.Array, slot_j: int | jax.Array) -> jax.Array:
        k, delta, _, w, *_ = pair_kernel(i, j, pos, state, system, slot_i, slot_j)
        return 0.5 * k * delta * delta * w

    if slots == 1:
        return evaluate(0, 0)
    if pair_kernel is _sphere_facet_pair:
        slot_i = jnp.concatenate((jnp.arange(slots), jnp.zeros(slots - 1, dtype=int)))
        slot_j = jnp.concatenate((jnp.zeros(slots, dtype=int), jnp.arange(1, slots)))
    else:
        slot = jnp.arange(slots * slots)
        slot_i, slot_j = slot // slots, slot % slots
    return jnp.sum(jax.vmap(evaluate)(slot_i, slot_j), axis=0)


@jax.jit(inline=True)
def _sphere_facet_pair(
    i: jax.Array,
    j: jax.Array,
    pos: jax.Array,
    state: "State",
    system: "System",
    incidence_i: int | jax.Array = 0,
    incidence_j: int | jax.Array = 0,
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
    At zero separation, a deterministic facet normal supplies a reciprocal
    repulsive force. It is a contact convention, not the derivative of the
    unsigned-distance energy at an intersection. Positive-distance normals
    follow the closest-point displacement. XLA removes unused quantities.
    """
    dim = pos.shape[-1]
    model = cast(SphereFacetSpringForce, system.force_model)
    if not (model.edge or model.vertex or (dim == 3 and model.face)):
        return _inactive_facet_pair(i, j, pos)

    is_facet_i, idxs_i, is_primary_i, face_i, active_i = get_facet_incidence(
        i, state, incidence_i
    )
    is_facet_j, idxs_j, is_primary_j, face_j, active_j = get_facet_incidence(
        j, state, incidence_j
    )

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

    compute_interaction = (
        jnp.where(
            is_rigid,
            jnp.where(is_facet_i, is_primary_i, True)
            * jnp.where(is_facet_j, is_primary_j, True),
            True,
        )
        * (is_facet_i ^ is_facet_j)
        * active_i
        * active_j
    )

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
        d_sf, c_f, coords_f = point_triangle_distance(
            p,
            v_a,
            v_b,
            v_c,
            system,
            face=model.face,
            edge=model.edge,
            vertex=model.vertex,
        )
    else:
        d_sf, c_f, coords_f = point_segment_distance(
            p, v_a, v_b, system, edge=model.edge, vertex=model.vertex
        )

    c_1 = jnp.where(is_facet_i[..., None], c_f, p)
    c_2 = jnp.where(is_facet_j[..., None], c_f, p)

    thick_i = state.rad[idx0_i]
    thick_j = state.rad[idx0_j]

    rij = system.domain._displacement(c_1, c_2, system)
    n, r = unit_and_norm(rij)

    fallback_rij = system.domain._displacement(
        state.pos_c[idx0_i], state.pos_c[idx0_j], system
    )
    normal = facet_normal(v_a, v_b, v_c, system)
    normal = jnp.where(is_facet_i[..., None], normal, -normal)
    normal = jnp.where(norm2(normal)[..., None] > 0.0, normal, unit(fallback_rij))
    axis = jnp.zeros_like(normal).at[..., 0].set(1.0)
    axis = jnp.where(is_facet_i[..., None], axis, -axis)
    normal = jnp.where(norm2(normal)[..., None] > 0.0, normal, axis)
    normal = jnp.where(dot(normal, fallback_rij)[..., None] < 0.0, -normal, normal)
    n = jnp.where(r[..., None] > 0.0, n, normal)

    delta = thick_i + thick_j - d_sf
    is_contact = (delta > 0) * compute_interaction
    delta = jnp.where(is_contact, delta, 0.0)

    idxs_facet = jnp.where(is_facet_i[..., None], idxs_i, idxs_j)
    particle_facet = jnp.where(is_facet_i, i, j)
    v_idx = jnp.argmax(idxs_facet == particle_facet[..., None], axis=-1)
    w_vertex = jnp.take_along_axis(
        coords_f,
        jnp.broadcast_to(v_idx[..., None], coords_f.shape[:-1] + (1,)),
        axis=-1,
    )[..., 0]

    w = jnp.where(is_rigid, 1.0, w_vertex)

    return k, delta, is_contact, w, n, c_1, thick_i, is_rigid


@ForceModel.register("sphere_facet_spring")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SphereFacetSpringForce(ForceModel):
    r"""Linear spring contact between spheres and facets.

    Contacts are evaluated through vertex keys, using the primary key for
    rigid facets. Search bounds automatically include the current facet extent
    and contact thickness. ``state._rad`` supplies an optional conservative
    lower bound on that search radius.

    Shared faces accumulate loads on the same physical vertex keys. Flexible
    nodal loads use closest-point barycentric weights. Penalties add per face,
    including multiple faces contacted near a shared edge or vertex.

    Contact thickness is ``state.rad`` at each face's first vertex
    (set through ``State.add_facet(thickness=...)`` or retained when attaching
    faces to existing nodes). The force model itself
    has no thickness parameter. Zero-distance normals are force conventions,
    so physical relaxation requires a gradient-only optimizer. Value-based
    optimization requires a custom energy target differentiated through AD.

    ``face``, ``edge``, and ``vertex`` enable contacts whose closest point is
    on that feature. Disabling a feature suppresses its contacts without
    substituting a farther feature. In 2D, the open segment is an ``edge``
    and its endpoints are ``vertex`` contacts; ``face`` has no effect.
    All switches default to true and are static JAX configuration, so changing
    them recompiles the kernel and omits disabled feature calculations.
    """

    face: bool = jax.tree.static(default=True)
    edge: bool = jax.tree.static(default=True)
    vertex: bool = jax.tree.static(default=True)

    def __post_init__(self) -> None:
        for name in ("face", "edge", "vertex"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{type(self).__name__}.{name} must be a Python bool.")

    @property
    def supports_analytical_energy_gradient(self) -> bool:
        """Zero-distance contact normals are force conventions, not energy gradients."""
        return False

    @property
    def required_material_properties(self) -> tuple[str, ...]:
        """Require the effective Young modulus for the contact spring."""
        return ("young_eff",)

    def search_radii(self, state: State, system: System) -> jax.Array:
        """Include current facet extent and contact thickness in search bounds."""
        return facet_search_radii(state, system)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SphereFacetSpringForce.force")
    def force(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
        *,
        advance_history: bool = True,
        quasistatic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        force, torque = _sum_facet_forces(_sphere_facet_pair, i, j, pos, state, system)
        return force, torque, history

    @staticmethod
    @jax.jit(inline=True)
    def energy(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
    ) -> jax.Array:
        return _sum_facet_energies(_sphere_facet_pair, i, j, pos, state, system)


@jax.jit(inline=True)
def _facet_facet_pair(
    i: jax.Array,
    j: jax.Array,
    pos: jax.Array,
    state: "State",
    system: "System",
    incidence_i: int | jax.Array = 0,
    incidence_j: int | jax.Array = 0,
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
    At zero separation, a deterministic facet normal supplies a reciprocal
    repulsive force. It is a contact convention, not the derivative of the
    unsigned-distance energy at an intersection. Positive-distance normals
    follow the closest-point displacement. XLA removes unused quantities.
    """
    dim = pos.shape[-1]
    model = cast(FacetFacetSpringForce, system.force_model)
    if not (
        model.edge_edge or (dim == 3 and (model.vertex_triangle or model.edge_triangle))
    ):
        return _inactive_facet_pair(i, j, pos)
    i_arr = jnp.asarray(i)
    j_arr = jnp.asarray(j)

    is_facet_i, idxs_i, is_primary_i, face_i, active_i = get_facet_incidence(
        i, state, incidence_i
    )
    is_facet_j, idxs_j, is_primary_j, face_j, active_j = get_facet_incidence(
        j, state, incidence_j
    )

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
        & (face_i != face_j)
        & ~jnp.any(idxs_i[..., :, None] == idxs_j[..., None, :], axis=(-2, -1))
        & active_i
        & active_j
    )

    mi, mj = state.mat_id[idx0_i], state.mat_id[idx0_j]
    k = system.mat_table.young_eff[mi, mj]

    t1_a = pos[idxs_i[..., 0]]
    t1_b = pos[idxs_i[..., 1]]
    t2_a = pos[idxs_j[..., 0]]
    t2_b = pos[idxs_j[..., 1]]

    swap = face_i > face_j

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
            tA_a,
            tA_b,
            tA_c,
            tB_a,
            tB_b,
            tB_c,
            system,
            vertex_triangle=model.vertex_triangle,
            edge_edge=model.edge_edge,
            edge_triangle=model.edge_triangle,
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
    delta = jnp.where(is_contact, delta, 0.0)

    rij = system.domain._displacement(c_ff_1, c_ff_2, system)
    n, r = unit_and_norm(rij)

    center_a = jnp.where(swap[..., None], state.pos_c[idx0_j], state.pos_c[idx0_i])
    center_b = jnp.where(swap[..., None], state.pos_c[idx0_i], state.pos_c[idx0_j])
    centers = system.domain._displacement(center_a, center_b, system)
    normal = facet_normal(tA_a, tA_b, tA_c if dim == 3 else None, system)
    other = facet_normal(tB_a, tB_b, tB_c if dim == 3 else None, system)
    normal = jnp.where(norm2(normal)[..., None] > 0.0, normal, other)
    normal = jnp.where(norm2(normal)[..., None] > 0.0, normal, unit(centers))
    normal = jnp.where(
        norm2(normal)[..., None] > 0.0,
        normal,
        jnp.zeros_like(normal).at[..., 0].set(1.0),
    )
    normal = jnp.where(dot(normal, centers)[..., None] < 0.0, -normal, normal)
    normal = jnp.where(swap[..., None], -normal, normal)
    n = jnp.where(r[..., None] > 0.0, n, normal)

    v_idx_i = jnp.argmax(idxs_i == i_arr[..., None], axis=-1)
    w_vertex_i = jnp.take_along_axis(
        coords_1,
        jnp.broadcast_to(v_idx_i[..., None], coords_1.shape[:-1] + (1,)),
        axis=-1,
    )[..., 0]

    v_idx_j = jnp.argmax(idxs_j == j_arr[..., None], axis=-1)
    w_vertex_j = jnp.take_along_axis(
        coords_2,
        jnp.broadcast_to(v_idx_j[..., None], coords_2.shape[:-1] + (1,)),
        axis=-1,
    )[..., 0]

    w = jnp.where(is_rigid, 1.0, w_vertex_i * w_vertex_j)

    return k, delta, is_contact, w, n, c_ff_1, thick_i, is_rigid


@ForceModel.register("facet_facet_spring")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class FacetFacetSpringForce(ForceModel):
    r"""Linear spring contact between facets.

    Contacts are evaluated through vertex keys, using the primary key for
    rigid facets. Search bounds automatically include the current facet extent
    and contact thickness. ``state._rad`` supplies an optional conservative
    lower bound on that search radius.

    Shared faces accumulate loads on the same physical vertex keys. Flexible
    nodal loads use closest-point barycentric weights. Penalties add per face,
    including multiple faces contacted near a shared edge or vertex.

    Contact thickness is ``state.rad`` at each face's first vertex
    (set through ``State.add_facet(thickness=...)`` or retained when attaching
    faces to existing nodes). The force model itself
    has no thickness parameter. Zero-distance normals are force conventions,
    so physical relaxation requires a gradient-only optimizer. Value-based
    optimization requires a custom energy target differentiated through AD.

    Faces sharing any physical node are excluded from facet-facet contact.

    Static switches select geometry query groups: ``vertex_triangle`` measures
    all vertices against the opposite triangle, ``edge_edge`` measures all
    edge pairs, and ``edge_triangle`` detects edge piercings of the opposite
    face. The minimum remaining candidate defines the contact. These groups
    overlap physical feature types: vertex-triangle queries include boundary
    contacts, and parallel face-face separation can be found by edge queries.
    In 2D, only ``edge_edge`` is used. All switches default to true; disabled
    groups are omitted when JAX compiles the kernel.
    """

    vertex_triangle: bool = jax.tree.static(default=True)
    edge_edge: bool = jax.tree.static(default=True)
    edge_triangle: bool = jax.tree.static(default=True)

    def __post_init__(self) -> None:
        for name in ("vertex_triangle", "edge_edge", "edge_triangle"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{type(self).__name__}.{name} must be a Python bool.")

    @property
    def supports_analytical_energy_gradient(self) -> bool:
        """Zero-distance contact normals are force conventions, not energy gradients."""
        return False

    @property
    def required_material_properties(self) -> tuple[str, ...]:
        """Require the effective Young modulus for the contact spring."""
        return ("young_eff",)

    def search_radii(self, state: State, system: System) -> jax.Array:
        """Include current facet extent and contact thickness in search bounds."""
        return facet_search_radii(state, system)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="FacetFacetSpringForce.force")
    def force(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
        *,
        advance_history: bool = True,
        quasistatic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        force, torque = _sum_facet_forces(_facet_facet_pair, i, j, pos, state, system)
        return force, torque, history

    @staticmethod
    @jax.jit(inline=True)
    def energy(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
    ) -> jax.Array:
        return _sum_facet_energies(_facet_facet_pair, i, j, pos, state, system)


__all__ = ["SpringForce", "SphereFacetSpringForce", "FacetFacetSpringForce"]
