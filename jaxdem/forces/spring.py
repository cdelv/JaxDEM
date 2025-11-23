# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Linear spring force model."""

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


@ForceModel.register("spring")
@partial(jax.tree_util.register_dataclass, drop_fields=["required_material_properties"])
@dataclass(slots=True)
class SpringForce(ForceModel):
    r"""
    A `ForceModel` implementation for a linear spring-like interaction between particles.

    Notes
    -----
    - The 'effective Young's modulus' (:math:`k_{eff,\; ij}`) is retrieved from the
      :attr:`jaxdem.System.mat_table` based on the material IDs of the interacting particles.
    - The force is zero if :math:`i == j`.
    - A small epsilon is added to the squared distance (:math:`r^2`) before taking the square root
      to prevent division by zero or NaN issues when particles are perfectly co-located.

    The penetration :math:`\delta` (overlap) between two particles :math:`i` and :math:`j` is:

    .. math::
        \delta = (R_i + R_j) - r

    where :math:`R_i` and :math:`R_j` are the radii of particles :math:`i` and :math:`j` respectively,
    and :math:`r = ||r_{ij}||` is the distance between their centers.

    The scalar overlap :math:`s` is defined as:

    .. math::
        s = \max \left(0, \frac{R_i + R_j}{r} - 1 \right)

    The force :math:`F_{ij}` acting on particle :math:`i` due to particle :math:`j` is:

    .. math::
        F_{ij} = k_{eff,\; ij} s r_{ij}

    The potential energy :math:`E_{ij}` of the interaction is:

    .. math::
        E_{ij} = \frac{1}{2} k_{eff,\; ij} s^2

    where :math:`k_{eff,\; ij}` is the effective Young's modulus for the particle pair.
    """

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="SpringForce.force")
    def force(
        i: int, j: int, state: "State", system: "System"
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Compute linear spring-like interaction force acting on particle :math:`i` due to particle :math:`j`.

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
            Force vector acting on particle :math:`i` due to particle :math:`j`.
        """
        mi = jnp.take(state.mat_id, i, axis=-1)
        mj = jnp.take(state.mat_id, j, axis=-1)
        k = jnp.take_along_axis(
            jnp.take_along_axis(
                system.mat_table.young_eff, mi[..., None, None], axis=-2
            ),
            mj[..., None, None],
            axis=-1,
        )[..., 0, 0]
        R = jnp.take(state.rad, i, axis=-1) + jnp.take(state.rad, j, axis=-1)

        rij = system.domain.displacement(
            jnp.take(state.pos, i, axis=-2),
            jnp.take(state.pos, j, axis=-2),
            system,
        )
        r = jnp.sum(rij * rij, axis=-1)
        r = jnp.sqrt(r + jnp.finfo(state.pos.dtype).eps)
        # s = jnp.maximum(0.0, R / r - 1.0)
        s = R / r - 1.0
        s *= s > 0
        scalar = (k * s)[..., None]
        return scalar * rij, jnp.zeros_like(jnp.take(state.angVel, i, axis=-2))

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="SpringForce.energy")
    def energy(i: int, j: int, state: "State", system: "System") -> jax.Array:
        """
        Compute linear spring-like interaction potential energy between particle :math:`i` and particle :math:`j`.

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
        mi = jnp.take(state.mat_id, i, axis=-1)
        mj = jnp.take(state.mat_id, j, axis=-1)
        k = jnp.take_along_axis(
            jnp.take_along_axis(
                system.mat_table.young_eff, mi[..., None, None], axis=-2
            ),
            mj[..., None, None],
            axis=-1,
        )[..., 0, 0]
        R = jnp.take(state.rad, i, axis=-1) + jnp.take(state.rad, j, axis=-1)

        rij = system.domain.displacement(
            jnp.take(state.pos, i, axis=-2),
            jnp.take(state.pos, j, axis=-2),
            system,
        )
        r = jnp.sum(rij * rij, axis=-1)
        r = jnp.sqrt(r)
        # s = jnp.maximum(0.0, R - r)
        s = R - r
        s *= s > 0
        return 0.5 * k * s**2

    @property
    def required_material_properties(self) -> Tuple[str, ...]:
        """
        A static tuple of strings specifying the material properties required by this force model.

        These properties (e.g., 'young_eff', 'restitution', ...) must be present in the
        :attr:`System.mat_table` for the model to function correctly. This is used
        for validation.
        """
        return ("young_eff",)


__all__ = ["SpringForce"]
