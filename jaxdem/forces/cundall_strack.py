# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Cundall-Strack linear spring-dashpot contact force model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ..utils.linalg import cross, dot, norm, unit, unit_and_norm
from . import ForceModel

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


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


@ForceModel.register("cundallstrack")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class CundallStrackForce(ForceModel):
    r"""History-dependent linear spring/dashpot friction with rolling resistance.

    The tangential spring displacement is accumulated per contact, transported
    with the changing contact frame, and return-mapped at the Coulomb limit.

    Computes the interaction between two spheres with a linear elastic
    assumption, viscous damping, and Coulomb friction.

    **Effective Properties**
    The model computes the effective mass :math:`m_{eff}`, restitution
    coefficient :math:`e_{eff}`, and friction coefficient :math:`\mu` as:

    .. math::
        m_{eff} = \left( \frac{1}{m_i} + \frac{1}{m_j} \right)^{-1}, \quad
        e_{eff} = \min(e_i, e_j), \quad
        \mu = \min(\mu_i, \mu_j)

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
    Setting :math:`\mu_r = 0` (the default) disables rolling friction.

    References
    ----------
    .. .. [1] Cundall, P. A., & Strack, O. D. (1979). A discrete numerical model
           for granular assemblies. Geotechnique, 29(1), 47-65.

    """

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
        mi, mj = state.mat_id[i], state.mat_id[j]
        E_i, E_j = system.mat_table.young[mi], system.mat_table.young[mj]
        nu_i, nu_j = system.mat_table.poisson[mi], system.mat_table.poisson[mj]
        e_i, e_j = system.mat_table.e[mi], system.mat_table.e[mj]
        mu_i, mu_j = system.mat_table.mu[mi], system.mat_table.mu[mj]
        mu_r_i, mu_r_j = system.mat_table.mu_r[mi], system.mat_table.mu_r[mj]

        m_i, m_j = state.mass[i], state.mass[j]
        R_i, R_j = state.rad[i], state.rad[j]

        # Effective material properties
        ER_i = E_i * R_i
        ER_j = E_j * R_j
        ER_product = ER_i * ER_j
        kn = (2.0 * ER_product) / (ER_i + ER_j)
        kt = ER_product / (ER_i * (1.0 + nu_j) + ER_j * (1.0 + nu_i))
        m_eff = (m_i * m_j) / (m_i + m_j)
        e_eff = jnp.minimum(e_i, e_j)
        mu_eff = jnp.minimum(mu_i, mu_j)

        # Damping coefficients
        # Guard e = 0: log(0) = -inf would give beta = inf/inf = NaN.
        # The analytic limit of beta as e -> 0+ is 1.
        e_safe = jnp.where(e_eff > 0.0, e_eff, 1.0)
        ln_e = jnp.log(e_safe)
        beta = jnp.where(
            e_eff > 0.0,
            -ln_e / jnp.sqrt(jnp.pi * jnp.pi + ln_e * ln_e),
            1.0,
        )
        gamma_n = 2.0 * beta * jnp.sqrt(kn * m_eff)
        gamma_t = 2.0 * beta * jnp.sqrt(kt * m_eff)

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

        v_rel = v_ci - v_cj
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
        Ft_max = mu_eff * Fn
        Ft = (
            Ft_trial * jnp.minimum(1.0, Ft_max / jnp.maximum(Ft_norm, 1e-30))[..., None]
        )
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
        mu_r_eff = jnp.minimum(mu_r_i, mu_r_j)
        R_eff = (R_i * R_j) / (R_i + R_j)
        omega_rel = state.ang_vel[i] - state.ang_vel[j]
        omega_hat = unit(omega_rel)
        torque = torque - (mu_r_eff * R_eff * Fn)[..., None] * omega_hat

        return F, torque, new_history

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="CundallStrackForce.energy")
    def energy(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> jax.Array:
        r"""Compute only the normal elastic energy of the interaction.

        Tangential spring energy is path-dependent and the current energy API
        does not receive pair history. This diagnostic therefore excludes it,
        as well as viscous and rolling dissipation; this model explicitly opts
        out of analytical energy-gradient minimization.

        .. math::
            U_{ij} = \frac{1}{2} k_n \delta_n^2

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
        jax.Array
            Scalar potential energy.

        """
        mi, mj = state.mat_id[i], state.mat_id[j]
        E_i, E_j = system.mat_table.young[mi], system.mat_table.young[mj]
        R_i, R_j = state.rad[i], state.rad[j]

        kn = (2.0 * E_i * R_i * E_j * R_j) / (E_i * R_i + E_j * R_j)

        rij = system.domain._displacement(pos[i], pos[j], system)
        r = norm(rij)

        delta = R_i + R_j - r
        delta *= (delta > 0) * (i != j)
        return 0.5 * kn * delta * delta

    @property
    def required_material_properties(self) -> tuple[str, ...]:
        return ("young", "poisson", "e", "mu", "mu_r")


__all__ = ["CundallStrackForce"]
