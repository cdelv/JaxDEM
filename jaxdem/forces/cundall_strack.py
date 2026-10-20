# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Cundall-Strack linear spring-dashpot contact force model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp

from ..utils.linalg import cross, dot, norm, norm2, unit, unit_and_norm
from ..utils.linalg import harmonic_mean as _harmonic_mean
from . import ForceModel

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System
    from ..utils.contacts import ContactData


FrictionMixingRule = Callable[[jax.Array, jax.Array], jax.Array]
CundallStrackParameterization = Literal["elastic", "coefficients"]


class CundallStrackContactData(NamedTuple):
    """Directed active contacts resolved into Cundall-Strack components.

    Attributes
    ----------
    pair_ids : jax.Array
        Sphere indices ``(i, j)``, shape ``(M, 2)``. Each row describes the
        contact force and torque on ``i`` from ``j``. Reciprocal contacts are
        represented by separate rows.
    normal_forces, tangential_forces : jax.Array
        Normal and Coulomb-limited tangential force vectors, shape ``(M, dim)``.
    torques : jax.Array
        Force-law torques about the center of sphere ``i``, shape
        ``(M, 1 | 3)``. These include tangential contact torque and rolling
        resistance, but not a clump-member offset moment.
    friction_coefficients : jax.Array
        Effective sliding-friction coefficient ``mu_ij`` selected by the
        configured mixing rule, shape ``(M,)``.
    mobilized : jax.Array
        Whether the tangential trial force is on or beyond the Coulomb surface,
        shape ``(M,)``.
    """

    pair_ids: jax.Array
    normal_forces: jax.Array
    tangential_forces: jax.Array
    torques: jax.Array
    friction_coefficients: jax.Array
    mobilized: jax.Array

    def as_contact_data(self, state: State, system: System) -> ContactData:
        """Return the matching generic contact snapshot without reevaluation.

        The resolved normal and tangential components are combined into total
        pair forces. Torques are shifted from each source sphere's center to
        its clump center of mass, matching :class:`~jaxdem.utils.ContactData`.
        Pair displacements use the configured domain's image convention.

        ``state`` and ``system`` must describe the unchanged configuration
        from which this snapshot was collected. Contact history is not read or
        advanced.
        """
        from ..utils.contacts import ContactData

        source, destination = self.pair_ids.T
        position = state.pos
        forces = self.normal_forces + self.tangential_forces
        member_offset = position[source] - state.pos_c[source]
        torques = self.torques + cross(member_offset, forces)
        displacements = system.domain.displacement(
            position[source], position[destination], system
        )
        return ContactData(self.pair_ids, forces, torques, displacements)


def minimum_friction(mu_i: jax.Array, mu_j: jax.Array) -> jax.Array:
    """Mix sliding-friction coefficients with their minimum."""
    return jnp.minimum(mu_i, mu_j)


def maximum_friction(mu_i: jax.Array, mu_j: jax.Array) -> jax.Array:
    """Mix sliding-friction coefficients with their maximum."""
    return jnp.maximum(mu_i, mu_j)


def arithmetic_mean_friction(mu_i: jax.Array, mu_j: jax.Array) -> jax.Array:
    """Mix sliding-friction coefficients with their arithmetic mean."""
    return 0.5 * (mu_i + mu_j)


def geometric_mean_friction(mu_i: jax.Array, mu_j: jax.Array) -> jax.Array:
    """Mix sliding-friction coefficients with their geometric mean."""
    return jnp.sqrt(mu_i) * jnp.sqrt(mu_j)


def _rotate_about_axis(v: jax.Array, axis: jax.Array, angle: jax.Array) -> jax.Array:
    """Rotate a 3-vector about a unit axis with Rodrigues' formula."""
    c = jnp.cos(angle)[..., None]
    s = jnp.sin(angle)[..., None]
    return v * c + jnp.cross(axis, v) * s + axis * dot(axis, v)[..., None] * (1 - c)


def _transport_tangent(
    xi: jax.Array, previous_normal: jax.Array, normal: jax.Array
) -> jax.Array:
    """Shortest-rotation transport between contact planes in 2D or 3D."""
    old, new, value = previous_normal, normal, xi
    if xi.shape[-1] == 2:
        # A planar rotation needs only its sine and cosine. Avoid padding every
        # contact to 3D and evaluating three vector cross products.
        c = jnp.clip(dot(old, new), -1.0, 1.0)
        s = cross(old, new)
        perpendicular = jnp.stack((-value[..., 1], value[..., 0]), axis=-1)
        rotated = c[..., None] * value + s * perpendicular
    else:
        # Reflect the old tangent across the normal bisector. In the tangent
        # plane this gives the shortest rotation. The squared bisector length
        # avoids cancellation in 1 + dot(old, new) near opposite normals.
        tangent = value - dot(value, old)[..., None] * old
        bisector = old + new
        bisector_norm2 = norm2(bisector)
        regular = (
            tangent
            - (2.0 * dot(tangent, bisector) / jnp.maximum(bisector_norm2, 1e-12))[
                ..., None
            ]
            * bisector
        )
        # Antiparallel normals have no unique shortest rotation in 3D. Choose
        # a deterministic axis using the least-aligned Cartesian basis vector.
        basis = jax.nn.one_hot(jnp.argmin(jnp.abs(old), axis=-1), 3, dtype=old.dtype)
        axis = unit(cross(old, basis))
        antiparallel = 2.0 * dot(axis, value)[..., None] * axis - value
        rotated = jnp.where((bisector_norm2 < 2e-6)[..., None], antiparallel, regular)
    rotated = jnp.where((norm2(old) > 0)[..., None], rotated, value)
    return rotated - dot(rotated, normal)[..., None] * normal


@jax.jit(inline=True, static_argnames=("advance_history",))
@partial(jax.named_call, name="cundall_strack_contact_response")
def _contact_response_with_coefficients(
    i: int,
    j: int,
    pos: jax.Array,
    state: State,
    system: System,
    history: jax.Array,
    kn: jax.Array,
    kt: jax.Array,
    gamma_n: jax.Array,
    gamma_t: jax.Array,
    mu_ij: jax.Array,
    mu_r_ij: jax.Array,
    *,
    advance_history: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate the shared linear spring-dashpot Cundall--Strack law."""
    R_i, R_j = state.rad[i], state.rad[j]

    # Geometry & overlap
    rij = system.domain._displacement(pos[i], pos[j], system)
    n, r = unit_and_norm(rij)
    delta = R_i + R_j - r
    is_contact = (delta > 0) * (i != j)
    delta *= is_contact

    # Contact-point arms
    r_ci = -R_i[..., None] * n
    r_cj = R_j[..., None] * n

    # COM-to-contact arm = COM-to-member-center offset + sphere surface arm.
    v_ci = state.velocity_at(i, r_ci)
    v_cj = state.velocity_at(j, r_cj)

    v_rel = system.domain.relative_velocity(pos[i], pos[j], v_ci, v_cj, system)
    vn = dot(v_rel, n)
    vt_vec = v_rel - vn[..., None] * n
    # Normal force (strictly repulsive, zero when not in contact)
    Fn = jnp.maximum(0.0, kn * delta - gamma_n * vn) * is_contact

    dim = pos.shape[-1]
    # Layout: [tangential spring displacement (dim), previous normal (dim)].
    xi_old = history[..., :dim]
    previous_normal = history[..., dim:]
    xi = _transport_tangent(xi_old, previous_normal, n)
    if dim == 3:
        spin = 0.5 * dot(state.ang_vel[i] + state.ang_vel[j], n)
        xi = _rotate_about_axis(
            xi, n, spin * system.dt if advance_history else jnp.zeros_like(spin)
        )
    xi_trial = xi + vt_vec * system.dt
    xi_eval = xi_trial if advance_history else xi

    Ft_trial = -kt[..., None] * xi_eval - gamma_t[..., None] * vt_vec
    Ft_norm = norm(Ft_trial)
    Ft_max = mu_ij * Fn
    safe_norm = jnp.where(Ft_norm > 0.0, Ft_norm, 1.0)
    Ft = Ft_trial * jnp.minimum(1.0, Ft_max / safe_norm)[..., None]
    Ft *= is_contact[..., None]

    # Return-map the spring at sliding contacts so stored displacement
    # cannot wind up beyond the Coulomb surface.
    safe_kt = jnp.where(kt > 0.0, kt, 1.0)[..., None]
    xi_returned = jnp.where(
        kt[..., None] > 0.0, -(Ft + gamma_t[..., None] * vt_vec) / safe_kt, 0.0
    )
    sliding = Ft_norm > Ft_max
    mobilized = is_contact & (Ft_norm > 0.0) & (Ft_norm >= Ft_max)
    xi_next = jnp.where(sliding[..., None], xi_returned, xi_trial)
    next_history = jnp.concatenate([xi_next, n], axis=-1)
    next_history = jnp.where(is_contact[..., None], next_history, 0.0)
    new_history = next_history if advance_history else history

    normal_force = Fn[..., None] * n
    F = normal_force + Ft
    torque = cross(r_ci, F)

    # Rolling friction: resistive torque opposing relative angular velocity
    R_eff = 0.5 * _harmonic_mean(R_i, R_j)
    omega_rel = state.ang_vel[i] - state.ang_vel[j]
    omega_hat = unit(omega_rel)
    torque = torque - (mu_r_ij * R_eff * Fn)[..., None] * omega_hat

    return normal_force, Ft, torque, new_history, mobilized


@jax.jit(inline=True, static_argnames=("advance_history",))
@partial(jax.named_call, name="cundall_strack_force")
def _force_with_coefficients(
    i: int,
    j: int,
    pos: jax.Array,
    state: State,
    system: System,
    history: jax.Array,
    kn: jax.Array,
    kt: jax.Array,
    gamma_n: jax.Array,
    gamma_t: jax.Array,
    mu_ij: jax.Array,
    mu_r_ij: jax.Array,
    *,
    advance_history: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the combined force, torque, and updated pair history."""
    normal, tangential, torque, new_history, _ = _contact_response_with_coefficients(
        i,
        j,
        pos,
        state,
        system,
        history,
        kn,
        kt,
        gamma_n,
        gamma_t,
        mu_ij,
        mu_r_ij,
        advance_history=advance_history,
    )
    return normal + tangential, torque, new_history


@jax.jit(inline=True)
@partial(jax.named_call, name="cundall_strack_stiffnesses")
def _pair_stiffnesses(
    i: int,
    j: int,
    state: State,
    system: System,
) -> tuple[jax.Array, jax.Array]:
    """Return normal and tangential pair stiffnesses."""
    law = cast(CundallStrackForce, system.force_model)
    mi, mj = state.mat_id[i], state.mat_id[j]
    table = system.mat_table

    if law.parameterization == "elastic":
        E_i, E_j = table.young[mi], table.young[mj]
        nu_i, nu_j = table.poisson[mi], table.poisson[mj]
        R_i, R_j = state.rad[i], state.rad[j]

        ER_i = E_i * R_i
        ER_j = E_j * R_j
        kn = _harmonic_mean(ER_i, ER_j)
        kt = _harmonic_mean(ER_i / (2.0 * (1.0 + nu_i)), ER_j / (2.0 * (1.0 + nu_j)))
    else:
        kn = _harmonic_mean(table.k_n[mi], table.k_n[mj])
        kt = _harmonic_mean(table.k_t[mi], table.k_t[mj])
    return kn, kt


@jax.jit(inline=True)
@partial(jax.named_call, name="cundall_strack_coefficients")
def _pair_coefficients(
    i: int,
    j: int,
    state: State,
    system: System,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return pair stiffness, damping, and friction coefficients."""
    law = cast(CundallStrackForce, system.force_model)
    mi, mj = state.mat_id[i], state.mat_id[j]
    table = system.mat_table
    kn, kt = _pair_stiffnesses(i, j, state, system)

    if law.parameterization == "elastic":
        e_i, e_j = table.e[mi], table.e[mj]
        m_i, m_j = state.mass[i], state.mass[j]
        m_eff = 0.5 * _harmonic_mean(m_i, m_j)
        e_eff = jnp.minimum(e_i, e_j)
        e_safe = jnp.where(e_eff > 0.0, e_eff, 1.0)
        ln_e = jnp.log(e_safe)
        beta = jnp.where(
            e_eff > 0.0,
            -ln_e / jnp.sqrt(jnp.pi * jnp.pi + ln_e * ln_e),
            1.0,
        )
        gamma_n = 2.0 * beta * jnp.sqrt(kn) * jnp.sqrt(m_eff)
        gamma_t = 2.0 * beta * jnp.sqrt(kt) * jnp.sqrt(m_eff)
    else:
        gamma_n = _harmonic_mean(table.b_n[mi], table.b_n[mj])
        gamma_t = _harmonic_mean(table.b_t[mi], table.b_t[mj])

    mu_ij = jnp.asarray(
        law.friction_mixing(table.mu[mi], table.mu[mj]), dtype=state.pos.dtype
    )
    mu_r_ij = jnp.asarray(
        law.rolling_friction_mixing(table.mu_r[mi], table.mu_r[mj]),
        dtype=state.pos.dtype,
    )
    return kn, kt, gamma_n, gamma_t, mu_ij, mu_r_ij


@jax.jit
def _evaluate_contact_diagnostics(
    state: State,
    system: System,
    pairs: jax.Array,
    history: jax.Array,
    valid: jax.Array,
) -> CundallStrackContactData:
    """Evaluate a packed block without advancing its contact history."""
    if not pairs.shape[0]:
        return _empty_contact_diagnostics(state)

    def evaluate(
        args: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        pair, pair_history, pair_valid = args
        i = jnp.clip(pair[0], 0, state.N - 1)
        j = jnp.clip(pair[1], 0, state.N - 1)
        kn, kt, gamma_n, gamma_t, mu_ij, mu_r_ij = _pair_coefficients(
            i, j, state, system
        )
        normal, tangential, torque, _, mobilized = _contact_response_with_coefficients(
            i,
            j,
            state.pos,
            state,
            system,
            pair_history,
            kn,
            kt,
            gamma_n,
            gamma_t,
            mu_ij,
            mu_r_ij,
            advance_history=False,
        )
        return (
            jnp.where(pair_valid, normal, 0.0),
            jnp.where(pair_valid, tangential, 0.0),
            jnp.where(pair_valid, torque, 0.0),
            jnp.where(pair_valid, mu_ij, 0.0),
            pair_valid & mobilized,
        )

    normal, tangential, torque, friction, mobilized = jax.lax.map(
        evaluate,
        (pairs, history, valid),
        batch_size=min(pairs.shape[0], 4096),
    )
    return CundallStrackContactData(
        pairs, normal, tangential, torque, friction, mobilized
    )


def _empty_contact_diagnostics(state: State) -> CundallStrackContactData:
    """Return a correctly shaped empty contact snapshot."""
    dtype = state.pos_c.dtype
    return CundallStrackContactData(
        jnp.empty((0, 2), dtype=int),
        jnp.empty((0, state.dim), dtype=dtype),
        jnp.empty((0, state.dim), dtype=dtype),
        jnp.empty((0, state.ang_vel.shape[-1]), dtype=dtype),
        jnp.empty((0,), dtype=dtype),
        jnp.empty((0,), dtype=bool),
    )


@ForceModel.register("cundallstrack")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class CundallStrackForce(ForceModel):
    r"""History-dependent linear spring/dashpot friction with rolling resistance.

    The tangential spring displacement is accumulated per contact, transported
    with the changing contact frame, and return-mapped at the Coulomb limit.
    ``parameterization="elastic"`` derives contact coefficients from ``young``,
    ``poisson``, and ``e``. ``parameterization="coefficients"`` reads ``k_n``,
    ``k_t``, ``b_n``, and ``b_t`` directly from the material table and combines
    particle values with a zero-safe harmonic mean. ``friction_mixing`` and
    ``rolling_friction_mixing`` select the pair sliding- and rolling-friction
    coefficients.

    Computes the interaction between two spheres with a linear elastic
    assumption, viscous damping, and Coulomb friction.

    **Elastic Parameterization**
    The elastic parameterization computes the effective mass
    :math:`m_{eff}` and restitution coefficient :math:`e_{eff}` as:

    .. math::
        m_{eff} = \left( \frac{1}{m_i} + \frac{1}{m_j} \right)^{-1}, \quad
        e_{eff} = \min(e_i, e_j)

    The model computes the shear modulus :math:`G` per particle from Young's
    modulus :math:`E` and Poisson's ratio :math:`\nu`:

    .. math::
        G = \frac{E}{2(1 + \nu)}

    The model treats the effective stiffnesses for the normal (:math:`k_n`) and
    tangential (:math:`k_t`) directions as springs in series:

    .. math::
        k_n = \frac{2 E_i R_i E_j R_j}{E_i R_i + E_j R_j}, \quad
        k_t = \frac{2 G_i R_i G_j R_j}{G_i R_i + G_j R_j}

    The viscous damping coefficient :math:`\beta` and the directional damping
    coefficients are:

    .. math::
        \beta = \frac{-\ln(e_{eff})}{\sqrt{\pi^2 + \ln^2(e_{eff})}}
    .. math::
        \gamma_n = 2 \beta \sqrt{k_n m_{eff}}, \quad
        \gamma_t = 2 \beta \sqrt{k_t m_{eff}}

    **Forces**
    The normal force :math:`F_n` includes spring repulsion and viscous damping,
    limited to repulsive values:

    .. math::
        F_n = \max(0, k_n \delta_n - \gamma_n v_n)

    The tangential force combines the history spring and viscous dashpot and is
    capped by the Coulomb sliding friction limit:

    .. math::
        \mathbf{F}_{t, trial} = -k_t\boldsymbol{\xi}_t-\gamma_t \mathbf{v}_t
    .. math::
        \mathbf{F}_t = \min\!\left(1,\frac{\mu F_n}{\|\mathbf{F}_{t,trial}\|}\right)
        \mathbf{F}_{t,trial}

    **Rolling Friction**
    The rolling friction torque resists the relative angular velocity at the
    contact:

    .. math::
        \boldsymbol{\tau}_{\text{roll}} =
            -\mu_r \, R_{\text{eff}} \, F_n \, \hat{\omega}_{\text{rel}}

    The configured ``rolling_friction_mixing`` callable determines
    :math:`\mu_r`; its default is :math:`\min(\mu_{r,i}, \mu_{r,j})`.
    The effective radius is :math:`R_{\text{eff}} = R_i R_j / (R_i + R_j)`,
    and :math:`\hat{\omega}_{\text{rel}}` is the unit relative angular velocity.

    References
    ----------
    .. .. [1] Cundall, P. A., & Strack, O. D. (1979). A discrete numerical model
           for granular assemblies. Geotechnique, 29(1), 47-65.

    """

    parameterization: CundallStrackParameterization = jax.tree.static(default="elastic")
    friction_mixing: FrictionMixingRule = jax.tree.static(default=minimum_friction)
    rolling_friction_mixing: FrictionMixingRule = jax.tree.static(
        default=minimum_friction
    )

    def __post_init__(self) -> None:
        if self.parameterization not in ("elastic", "coefficients"):
            raise ValueError(
                "parameterization must be 'elastic' or 'coefficients', "
                f"got {self.parameterization!r}."
            )

    @property
    def supports_analytical_energy_gradient(self) -> bool:
        return False

    def history_shape(self, dim: int) -> tuple[int, ...]:
        """Store tangential displacement and the previous contact normal."""
        return (2 * dim,)

    def search_radii(self, state: State, system: System) -> jax.Array:
        """Conservative search extent for this law's finite interaction range."""
        return jnp.maximum(state._rad, (1.0) * state.rad)

    @staticmethod
    @jax.jit(inline=True, static_argnames=("advance_history", "quasistatic"))
    @partial(jax.named_call, name="CundallStrackForce.force")
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
        r"""Compute Cundall-Strack normal and tangential forces and torque.

        Parameters
        ----------
        i, j : int
            Particle indices.
        pos : jax.Array
            Particle positions.
        state : State
            Current simulation state.
        system : System
            System configuration.
        history : jax.Array
            Stored tangential spring and previous contact normal.
        advance_history : bool, optional
            Advance the stored spring with this step's relative motion.
        quasistatic : bool, optional
            Evaluate with no time passing: the normal and tangential dashpots
            and rolling friction are switched off and the Coulomb cap uses
            the elastic normal force only.

        Returns
        -------
        tuple[jax.Array, jax.Array, jax.Array]
            ``(force, torque, history)`` with dimension-agnostic shapes. When
            ``advance_history`` is false the returned history is exactly the
            input, while force uses its contact-frame-transported spring.

        """
        kn, kt, gamma_n, gamma_t, mu_ij, mu_r_ij = _pair_coefficients(
            i, j, state, system
        )
        if quasistatic:
            gamma_n = jnp.zeros_like(gamma_n)
            gamma_t = jnp.zeros_like(gamma_t)
            mu_r_ij = jnp.zeros_like(mu_r_ij)
        return _force_with_coefficients(
            i,
            j,
            pos,
            state,
            system,
            history,
            kn,
            kt,
            gamma_n,
            gamma_t,
            mu_ij,
            mu_r_ij,
            advance_history=advance_history,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="CundallStrackForce.energy")
    def energy(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
    ) -> jax.Array:
        """Return normal plus stored tangential spring energy.

        The stored spring is transported to the current contact normal but not
        advanced or return-mapped. Dissipated work is not part of the energy.
        """
        kn, kt = _pair_stiffnesses(i, j, state, system)
        R_i, R_j = state.rad[i], state.rad[j]
        rij = system.domain._displacement(pos[i], pos[j], system)
        normal, distance = unit_and_norm(rij)
        contact = (distance < R_i + R_j) & (i != j)
        overlap = jnp.where(contact, R_i + R_j - distance, 0.0)

        dim = pos.shape[-1]
        xi = _transport_tangent(history[..., :dim], history[..., dim:], normal)
        tangential = jnp.where(contact, 0.5 * kt * dot(xi, xi), 0.0)
        return 0.5 * kn * overlap * overlap + tangential

    @property
    def required_material_properties(self) -> tuple[str, ...]:
        if self.parameterization == "elastic":
            return ("young", "poisson", "e", "mu", "mu_r")
        return ("k_n", "k_t", "b_n", "b_t", "mu", "mu_r")


def get_cundall_strack_contacts(
    state: State, system: System
) -> tuple[State, System, CundallStrackContactData]:
    """Collect resolved Cundall-Strack forces for all active directed contacts.

    The configured collider supplies pair candidates and stored tangential
    history. Neighbor-list caches are rebuilt when necessary, but contact
    history is never advanced. The returned ``system`` retains any refreshed
    cache. Results are sorted by source and destination sphere index.

    This is a host-side operation because it returns variable-length arrays.

    Raises
    ------
    TypeError
        The system does not use :class:`CundallStrackForce` directly.
    ValueError
        Pair search results are incomplete or a reported value is nonfinite.
    """
    from ..utils._pair_candidates import _collect_pair_candidates

    if not isinstance(system.force_model, CundallStrackForce):
        raise TypeError(
            "get_cundall_strack_contacts requires a system whose force model "
            "is CundallStrackForce."
        )

    system, pairs, valid, history = _collect_pair_candidates(state, system)
    data = _evaluate_contact_diagnostics(state, system, pairs, history, valid)
    active = valid & (
        jnp.any(data.normal_forces != 0, axis=1)
        | jnp.any(data.tangential_forces != 0, axis=1)
        | jnp.any(data.torques != 0, axis=1)
    )
    data = jax.tree.map(lambda value: value[active], data)
    finite = (
        jnp.all(jnp.isfinite(data.normal_forces))
        & jnp.all(jnp.isfinite(data.tangential_forces))
        & jnp.all(jnp.isfinite(data.torques))
        & jnp.all(jnp.isfinite(data.friction_coefficients))
    )
    if not bool(finite):
        raise ValueError("Cundall-Strack contact diagnostics must be finite.")
    order = jnp.lexsort((data.pair_ids[:, 1], data.pair_ids[:, 0]))
    return state, system, jax.tree.map(lambda value: value[order], data)


__all__ = [
    "CundallStrackContactData",
    "CundallStrackForce",
    "CundallStrackParameterization",
    "FrictionMixingRule",
    "arithmetic_mean_friction",
    "geometric_mean_friction",
    "get_cundall_strack_contacts",
    "maximum_friction",
    "minimum_friction",
]
