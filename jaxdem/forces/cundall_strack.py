# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Cundall-Strack linear spring-dashpot contact force model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Literal, cast

import jax
import jax.numpy as jnp

from ..utils.linalg import cross, dot, norm, unit, unit_and_norm
from . import ForceModel

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


FrictionMixingRule = Callable[[jax.Array, jax.Array], jax.Array]
CundallStrackParameterization = Literal["elastic", "coefficients"]


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
    return jnp.sqrt(mu_i * mu_j)


def _harmonic_mean(value_i: jax.Array, value_j: jax.Array) -> jax.Array:
    """Return a zero-safe harmonic mean that preserves equal inputs."""
    is_zero = (value_i == 0.0) | (value_j == 0.0)
    safe_i = jnp.where(is_zero, 1.0, value_i)
    safe_j = jnp.where(is_zero, 1.0, value_j)
    return jnp.where(is_zero, 0.0, 2.0 * safe_i * safe_j / (safe_i + safe_j))


def _rotate_about_axis(v: jax.Array, axis: jax.Array, angle: jax.Array) -> jax.Array:
    """Rotate a 3-vector about a unit axis with Rodrigues' formula."""
    c = jnp.cos(angle)[..., None]
    s = jnp.sin(angle)[..., None]
    return v * c + jnp.cross(axis, v) * s + axis * dot(axis, v)[..., None] * (1 - c)


def _transport_tangent(
    xi: jax.Array, previous_normal: jax.Array, normal: jax.Array
) -> jax.Array:
    """Shortest-rotation transport between contact planes in 2D or 3D."""
    dim = xi.shape[-1]
    pad = 3 - dim
    padding = [(0, 0)] * (xi.ndim - 1) + [(0, pad)]
    old = jnp.pad(previous_normal, padding)
    new = jnp.pad(normal, padding)
    value = jnp.pad(xi, padding)
    old_norm = norm(old)
    c = jnp.clip(dot(old, new), -1.0, 1.0)
    k = jnp.cross(old, new)
    regular = (
        value
        + jnp.cross(k, value)
        + jnp.cross(k, jnp.cross(k, value)) / jnp.maximum(1.0 + c, 1e-12)[..., None]
    )

    # Antiparallel normals have no unique shortest rotation. In 2D the plane
    # fixes the convention; in 3D choose a deterministic axis perpendicular
    # to the old normal using its least-aligned Cartesian basis vector.
    if dim == 2:
        antiparallel = -value
    else:
        basis = jax.nn.one_hot(jnp.argmin(jnp.abs(old), axis=-1), 3, dtype=old.dtype)
        axis = unit(jnp.cross(old, basis))
        antiparallel = 2.0 * dot(axis, value)[..., None] * axis - value
    rotated = jnp.where((c < -1.0 + 1e-6)[..., None], antiparallel, regular)
    rotated = jnp.where((old_norm > 0)[..., None], rotated, value)
    transported = rotated[..., :dim]
    return transported - dot(transported, normal)[..., None] * normal


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
    Ft_norm = jnp.linalg.norm(Ft_trial, axis=-1)
    Ft_max = mu_ij * Fn
    Ft = Ft_trial * jnp.minimum(
        1.0, Ft_max / jnp.maximum(Ft_norm, 1e-30)
    )[..., None]
    Ft *= is_contact[..., None]

    # Return-map the spring at sliding contacts so stored displacement
    # cannot wind up beyond the Coulomb surface.
    xi_returned = -(Ft + gamma_t[..., None] * vt_vec) / jnp.maximum(
        kt[..., None], 1e-30
    )
    sliding = Ft_norm > Ft_max
    xi_next = jnp.where(sliding[..., None], xi_returned, xi_trial)
    next_history = jnp.concatenate([xi_next, n], axis=-1)
    next_history = jnp.where(is_contact[..., None], next_history, 0.0)
    new_history = next_history if advance_history else history

    F = Fn[..., None] * n + Ft
    torque = cross(r_ci, F)

    # Rolling friction: resistive torque opposing relative angular velocity
    R_eff = (R_i * R_j) / (R_i + R_j)
    omega_rel = state.ang_vel[i] - state.ang_vel[j]
    omega_hat = unit(omega_rel)
    torque = torque - (mu_r_ij * R_eff * Fn)[..., None] * omega_hat

    return F, torque, new_history


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
        ER_product = ER_i * ER_j
        kn = (2.0 * ER_product) / (ER_i + ER_j)
        kt = ER_product / (ER_i * (1.0 + nu_j) + ER_j * (1.0 + nu_i))
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
        m_eff = (m_i * m_j) / (m_i + m_j)
        e_eff = jnp.minimum(e_i, e_j)
        e_safe = jnp.where(e_eff > 0.0, e_eff, 1.0)
        ln_e = jnp.log(e_safe)
        beta = jnp.where(
            e_eff > 0.0,
            -ln_e / jnp.sqrt(jnp.pi * jnp.pi + ln_e * ln_e),
            1.0,
        )
        gamma_n = 2.0 * beta * jnp.sqrt(kn * m_eff)
        gamma_t = 2.0 * beta * jnp.sqrt(kt * m_eff)
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

    where :math:`\mu_r = \min(\mu_{r,i}, \mu_{r,j})` is the effective
    rolling friction coefficient, :math:`R_{\text{eff}} = R_i R_j / (R_i + R_j)`,
    and :math:`\hat{\omega}_{\text{rel}}` is the unit relative angular velocity.

    References
    ----------
    .. .. [1] Cundall, P. A., & Strack, O. D. (1979). A discrete numerical model
           for granular assemblies. Geotechnique, 29(1), 47-65.

    """

    parameterization: CundallStrackParameterization = jax.tree.static(
        default="elastic"
    )
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

    @property
    def has_history_dependent_energy(self) -> bool:
        """Include the stored tangential spring in pair energy."""
        return True

    def history_shape(self, dim: int) -> tuple[int, ...]:
        """Store tangential displacement and the previous contact normal."""
        return (2 * dim,)

    def search_radii(self, state: State, system: System) -> jax.Array:
        """Conservative search extent for this law's finite interaction range."""
        return jnp.maximum(state._rad, (1.0) * state.rad)

    @staticmethod
    @jax.jit(inline=True, static_argnames=("advance_history",))
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
    @partial(jax.named_call, name="CundallStrackForce.energy_with_history")
    def energy_with_history(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
    ) -> jax.Array:
        """Return normal plus stored tangential spring energy."""
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


__all__ = [
    "CundallStrackForce",
    "CundallStrackParameterization",
    "FrictionMixingRule",
    "arithmetic_mean_friction",
    "geometric_mean_friction",
    "maximum_friction",
    "minimum_friction",
]
