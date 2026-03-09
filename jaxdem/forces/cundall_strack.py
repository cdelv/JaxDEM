# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Cundall-Strack linear spring-dashpot contact force model."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple
from functools import partial

from . import ForceModel
from ..utils.linalg import cross, cross_3X3D_1X2D

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@ForceModel.register("cundallstrack")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class CundallStrackForce(ForceModel):
    r"""
    Cundall-Strack linear spring-dashpot normal and tangential force model.

    Computes the interaction between two spheres using a linear elastic
    assumption combined with viscous damping and Coulomb friction.

    **Effective Properties**
    The effective mass :math:`m_{eff}`, restitution coefficient :math:`e_{eff}`,
    and friction coefficient :math:`\mu` are taken as:

    .. math::
        m_{eff} = \left( \frac{1}{m_i} + \frac{1}{m_j} \right)^{-1}, \quad
        e_{eff} = \min(e_i, e_j), \quad
        \mu = \min(\mu_i, \mu_j)

    The Shear Modulus :math:`G` is computed per particle from Young's modulus
    :math:`E` and Poisson's ratio :math:`\nu`:

    .. math::
        G = \frac{E}{2(1 + \nu)}

    The effective stiffnesses for the normal (:math:`k_n`) and tangential
    (:math:`k_t`) directions are modelled as springs in series:

    .. math::
        k_n = \frac{2 E_i R_i E_j R_j}{E_i R_i + E_j R_j}, \quad
        k_t = \frac{2 G_i R_i G_j R_j}{G_i R_i + G_j R_j}

    The viscous damping coefficient :math:`\beta` and resulting directional
    damping coefficients are:

    .. math::
        \beta = \frac{-\ln(e_{eff})}{\sqrt{\pi^2 + \ln^2(e_{eff})}}
    .. math::
        \gamma_n = 2 \beta \sqrt{k_n m_{eff}}, \quad
        \gamma_t = 2 \beta \sqrt{k_t m_{eff}}

    **Forces**
    The normal force :math:`F_n` includes spring repulsion and viscous damping,
    constrained to be strictly repulsive:

    .. math::
        F_n = \max(0, k_n \delta_n - \gamma_n v_n)

    *Note on Tangential Force*: True Cundall-Strack uses an integrated shear
    displacement history. Because this stateless implementation does not track
    tangential history per pair, the tangential force is approximated using the
    dynamic viscous dashpot strictly capped by the Coulomb sliding friction limit:

    .. math::
        \mathbf{F}_{t, trial} = -\gamma_t \mathbf{v}_t
    .. math::
        \mathbf{F}_t = \min(\|\mathbf{F}_{t, trial}\|, \mu F_n) \frac{\mathbf{v}_t}{\|\mathbf{v}_t\|}

    References
    ----------
    .. [1] Cundall, P. A., & Strack, O. D. (1979). A discrete numerical model
           for granular assemblies. Geotechnique, 29(1), 47-65.
    """

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="CundallStrackForce.force")
    def force(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> Tuple[jax.Array, jax.Array]:
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
        tuple[jax.Array, jax.Array]
            ``(force, torque)`` with dimension-agnostic shapes.
        """
        mi, mj = state.mat_id[i], state.mat_id[j]
        E_i, E_j = system.mat_table.young[mi], system.mat_table.young[mj]
        nu_i, nu_j = system.mat_table.poisson[mi], system.mat_table.poisson[mj]
        e_i, e_j = system.mat_table.e[mi], system.mat_table.e[mj]
        mu_i, mu_j = system.mat_table.mu[mi], system.mat_table.mu[mj]

        m_i, m_j = state.mass[i], state.mass[j]
        R_i, R_j = state.rad[i], state.rad[j]

        # Effective material properties
        G_i = E_i / (2.0 * (1.0 + nu_i))
        G_j = E_j / (2.0 * (1.0 + nu_j))

        kn = (2.0 * E_i * R_i * E_j * R_j) / (E_i * R_i + E_j * R_j)
        kt = (2.0 * G_i * R_i * G_j * R_j) / (G_i * R_i + G_j * R_j)

        m_eff = (m_i * m_j) / (m_i + m_j)
        e_eff = jnp.minimum(e_i, e_j)
        mu = jnp.minimum(mu_i, mu_j)

        # Damping coefficients
        ln_e = jnp.log(e_eff)
        beta = -ln_e / jnp.sqrt(jnp.pi**2 + ln_e**2)
        gamma_n = 2.0 * beta * jnp.sqrt(kn * m_eff)
        gamma_t = 2.0 * beta * jnp.sqrt(kt * m_eff)

        # Geometry & overlap
        rij = system.domain.displacement(pos[i], pos[j], system)
        r_sq = jnp.sum(rij * rij, axis=-1)
        r = jnp.where(r_sq == 0, 1.0, jnp.sqrt(r_sq))
        n = rij / r[..., None]

        delta = R_i + R_j - r
        is_contact = delta > 0
        delta = delta * is_contact

        # Contact-point arms
        r_ci = -R_i[..., None] * n
        r_cj = R_j[..., None] * n

        # Contact-point velocities (dimension-agnostic via cross utilities)
        vi, vj = state.vel[i], state.vel[j]
        v_ci = vi + cross_3X3D_1X2D(state.ang_vel[i], r_ci)
        v_cj = vj + cross_3X3D_1X2D(state.ang_vel[j], r_cj)

        v_rel = v_ci - v_cj
        vn = jnp.sum(v_rel * n, axis=-1)
        vt_vec = v_rel - vn[..., None] * n

        # Normal force (strictly repulsive)
        Fn = jnp.maximum(0.0, kn * delta - gamma_n * vn)
        Fn = jnp.where(is_contact, Fn, 0.0)

        # Tangential force (viscous dashpot capped by Coulomb friction)
        Ft_trial = -gamma_t[..., None] * vt_vec
        Ft_mag = jnp.sqrt(jnp.sum(Ft_trial**2, axis=-1) + 1e-12)
        F_max = mu * Fn
        scale = jnp.where(Ft_mag > F_max, F_max / Ft_mag, 1.0)
        Ft_vec = Ft_trial * scale[..., None] * is_contact[..., None]

        F = Fn[..., None] * n + Ft_vec
        torque = cross(r_ci, Ft_vec)

        return F, torque

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="CundallStrackForce.energy")
    def energy(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> jax.Array:
        r"""Compute conservative potential energy for the interaction.

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

        rij = system.domain.displacement(pos[i], pos[j], system)
        r = jnp.sum(rij * rij, axis=-1)
        r = jnp.sqrt(r)

        delta = R_i + R_j - r
        delta *= delta > 0
        return 0.5 * kn * jnp.pow(delta, 2.0)

    @property
    def required_material_properties(self) -> Tuple[str, ...]:
        return ("young", "poisson", "e", "mu")


__all__ = ["CundallStrackForce"]
