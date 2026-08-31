# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Reflective boundary-condition domain."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from . import Domain
from ._toc import verlet_collision_fraction

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("reflectsphere")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ReflectSphereDomain(Domain):
    """A `Domain` implementation that enforces reflective boundary conditions only for spheres.
    This dedicated version exists for performance.

    When a particle moves beyond the defined `box_size`, the domain reflects
    its position back into the box. It also reverses the velocity component
    normal to the boundary, scaled by `restitution_coefficient`.

    Notes
    -----
    - The reflection occurs at the boundaries defined by `anchor` and `anchor + box_size`.

    """

    restitution_coefficient: jax.Array

    @classmethod
    def Create(
        cls,
        dim: int,
        box_size: jax.Array | None = None,
        anchor: jax.Array | None = None,
        restitution_coefficient: float = 1.0,
        **kw: Any,
    ) -> Self:
        """Default factory method for the ReflectSphereDomain class.

        Parameters
        ----------
        dim : int
            The dimensionality of the domain (e.g., 2, 3).
        box_size : jax.Array, optional
            The size of the domain along each dimension. If not provided,
            defaults to an array of ones with shape `(dim,)`.
        anchor : jax.Array, optional
            The anchor (origin) of the domain. If not provided,
            defaults to an array of zeros with shape `(dim,)`.
        restitution_coefficient : float
            Restitution coefficient between 0 and 1 to modulate energy conservation with wall.

        Returns
        -------
        ReflectSphereDomain
            A new instance with the specified or default configuration.

        Raises
        ------
        ValueError
            If `box_size` or `anchor` have the wrong shape, or if
            `restitution_coefficient` is outside `(0, 1]`.

        """
        if not (0.0 < restitution_coefficient <= 1.0):
            raise ValueError(
                "restitution_coefficient must be in (0, 1], got "
                f"{restitution_coefficient}."
            )
        # Explicit two-arg super: dataclass(slots=True) recreates the class, so
        # zero-arg super()'s __class__ cell points at the discarded original
        # and raises TypeError on Python < 3.14.
        return super(ReflectSphereDomain, cls).Create(
            dim,
            box_size=box_size,
            anchor=anchor,
            restitution_coefficient=jnp.asarray(restitution_coefficient, dtype=float),
            **kw,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="ReflectSphereDomain.apply")
    def apply(state: State, system: System) -> tuple[State, System]:
        r"""Apply reflective boundary conditions to particles.

        The method checks particles against the domain boundaries.
        When a particle moves beyond a boundary, the method reflects its
        position back into the box. It also reverses the velocity component
        normal to that boundary, scaled by the restitution coefficient :math:`e`.

        .. math::
            l &= a + R \\
            u &= a + B - R \\
            v' &= \begin{cases} -e\,v & \text{if } r < l \text{ or } r > u \\ v & \text{otherwise} \end{cases} \\
            r' &= \begin{cases} 2l - r & \text{if } r < l \\ r & \text{otherwise} \end{cases} \\
            r'' &= \begin{cases} 2u - r' & \text{if } r' > u \\ r' & \text{otherwise} \end{cases} \\
            r &= r''

        where:
            - :math:`r` is the current particle position (:attr:`jaxdem.State.pos`)
            - :math:`v` is the current particle velocity (:attr:`jaxdem.State.vel`)
            - :math:`a` is the domain anchor (:attr:`Domain.anchor`)
            - :math:`B` is the domain box size (:attr:`Domain.box_size`)
            - :math:`R` is the particle radius (:attr:`jaxdem.State.rad`)
            - :math:`l` is the lower boundary for the particle center
            - :math:`u` is the upper boundary for the particle center
            - :math:`e` is the restitution coefficient.

        **Verlet Time-of-Collision Correction**

        The shared Verlet-consistent solver
        :func:`jaxdem.domains._toc.verlet_collision_fraction` (also used by
        :class:`ReflectDomain`) computes the collision time fraction
        :math:`\alpha \in [0, 1]` and the velocity at the moment of collision.
        The method reconstructs the pre-collision velocity as
        :math:`v_{col} = v + (\alpha - 1) \Delta t\, a`.

        TO DO: Check correctness when adding different shape types and angular velocity

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            The updated `State` object with reflected positions and velocities,
            and the `System` object.

        Note
        -----
        - Only works for states with *ONLY* spheres.

        """
        domain = cast(ReflectSphereDomain, system.domain)
        e = domain.restitution_coefficient
        pos = state.pos_c

        rad = state.rad[:, None]
        lo = system.domain.anchor + rad
        hi = system.domain.anchor + system.domain.box_size - rad

        over_lo = jnp.maximum(0.0, lo - pos)
        over_hi = jnp.maximum(0.0, pos - hi)

        inv_mass = 1.0 / state.mass
        acc = state.force * inv_mass[:, None]

        wall_sign = (over_lo > 0).astype(float) - (over_hi > 0).astype(float)
        delta = over_lo + over_hi

        alpha = verlet_collision_fraction(state.vel, acc, delta, wall_sign, system.dt)
        alpha = jnp.where(wall_sign != 0.0, alpha, 1.0)

        dt_remaining = (1.0 - alpha) * system.dt
        v_col = state.vel - dt_remaining * acc

        closing_mask = (v_col * wall_sign) < 0.0

        dv = -(1.0 + e) * v_col * closing_mask
        dv_flat = jnp.where(state.fixed[:, None], 0.0, dv)

        state.vel += dv_flat

        state.pos_c += dv_flat * dt_remaining

        return state, system


__all__ = ["ReflectSphereDomain"]
