# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Hertzian (nonlinear) normal contact force model."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple
from functools import partial

from . import ForceModel

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@ForceModel.register("hertz")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class HertzianForce(ForceModel):
    r"""
    Hertzian nonlinear normal contact force between elastic spheres.

    The effective Young's modulus :math:`E^*` is computed directly from
    the per-particle Young's modulus :math:`E` and Poisson's ratio
    :math:`\nu`:

    .. math::
        \frac{1}{E^*} = \frac{1 - \nu_i^2}{E_i} + \frac{1 - \nu_j^2}{E_j}

    The effective radius is:

    .. math::
        \frac{1}{R^*} = \frac{1}{R_i} + \frac{1}{R_j}

    The Hertzian stiffness combines both:

    .. math::
        k = \tfrac{4}{3}\, E^* \sqrt{R^*}

    The penetration depth :math:`\delta` between particles :math:`i`
    and :math:`j` is:

    .. math::
        \delta = \max(0,\; R_i + R_j - r)

    where :math:`r = \|r_{ij}\|`.

    The Hertzian normal force and contact energy are:

    .. math::
        \mathbf{F}_{ij} = k \; \delta^{3/2} \; \hat{n}_{ij}, \qquad
        U_{ij} = \tfrac{2}{5}\, k \; \delta^{5/2}

    where :math:`\hat{n}_{ij} = \mathbf{r}_{ij} / r`.

    Notes
    -----
    The material ``young`` and ``poisson`` properties are read directly
    from :attr:`System.mat_table` per particle; no matchmaker effective
    value is used.
    """

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="HertzianForce.force")
    def force(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> Tuple[jax.Array, jax.Array]:
        r"""Compute Hertzian normal contact force on particle *i* from *j*.

        .. math::
            \mathbf{F}_{ij} = \tfrac{4}{3}\, E^*\, \sqrt{R^*}\;
            \delta^{3/2}\; \hat{n}_{ij}

        Parameters
        ----------
        i, j : int
            Particle indices.
        pos : jax.Array
            Particle positions (rotated to lab frame).
        state : State
            Current simulation state.
        system : System
            System configuration.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            ``(force, torque)`` with shapes ``(dim,)`` and ``(ang_dim,)``.
        """
        mi, mj = state.mat_id[i], state.mat_id[j]
        E_i = system.mat_table.young[mi]
        E_j = system.mat_table.young[mj]
        nu_i = system.mat_table.poisson[mi]
        nu_j = system.mat_table.poisson[mj]
        R_i, R_j = state.rad[i], state.rad[j]

        R_star = (R_i * R_j) / (R_i + R_j)
        E_star = 1.0 / ((1.0 - nu_i**2) / E_i + (1.0 - nu_j**2) / E_j)
        k = (4.0 / 3.0) * E_star * jnp.sqrt(R_star)

        rij = system.domain.displacement(pos[i], pos[j], system)
        r = jnp.sum(rij * rij, axis=-1)
        r = jnp.where(r == 0, 1.0, jnp.sqrt(r))
        delta = R_i + R_j - r
        delta *= delta > 0

        mag = k * jnp.pow(delta, 1.5)
        F = (mag / r)[..., None] * rij

        return F, jnp.zeros_like(state.torque[i])

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="HertzianForce.energy")
    def energy(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> jax.Array:
        r"""Hertzian contact energy.

        .. math::
            U_{ij} = \tfrac{2}{5} \cdot \tfrac{4}{3}\, E^*\, \sqrt{R^*}\, \delta^{5/2}

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
        E_i = system.mat_table.young[mi]
        E_j = system.mat_table.young[mj]
        nu_i = system.mat_table.poisson[mi]
        nu_j = system.mat_table.poisson[mj]
        R_i, R_j = state.rad[i], state.rad[j]

        R_star = (R_i * R_j) / (R_i + R_j)
        E_star = 1.0 / ((1.0 - nu_i**2) / E_i + (1.0 - nu_j**2) / E_j)
        k = (4.0 / 3.0) * E_star * jnp.sqrt(R_star)

        rij = system.domain.displacement(pos[i], pos[j], system)
        r = jnp.sum(rij * rij, axis=-1)
        r = jnp.sqrt(r)
        delta = R_i + R_j - r
        delta *= delta > 0
        return 0.4 * k * jnp.pow(delta, 2.5)

    @property
    def required_material_properties(self) -> Tuple[str, ...]:
        return ("young", "poisson")


__all__ = ["HertzianForce"]
