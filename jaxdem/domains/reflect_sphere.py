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
    We have this dedicated version for performance reasons.

    Particles that attempt to move beyond the defined `box_size` will have their
    positions reflected back into the box and their velocities reversed in the
    direction normal to the boundary, modulated by `restitution_coefficient`.

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
        r"""Applies reflective boundary conditions to particles.

        Particles are checked against the domain boundaries.
        If a particle attempts to move beyond a boundary, its position is reflected
        back into the box, and its velocity component normal to that boundary is reversed
        (scaled by the restitution coefficient :math:`e`).

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

        The collision time fraction :math:`\alpha \in [0, 1]` and the velocity at the
        moment of collision are obtained from the shared Verlet-consistent solver
        :func:`jaxdem.domains._toc.verlet_collision_fraction` (also used by
        :class:`ReflectDomain`), and the pre-collision velocity is reconstructed as
        :math:`v_{col} = v + (\alpha - 1) \Delta t\, a`.

        TO DO: Ensure correctness when adding different types of shapes and angular vel

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

        v_col = state.vel + (alpha - 1.0) * system.dt * acc

        closing_mask = (v_col * wall_sign) < 0.0

        dv = -(1.0 + e) * v_col * closing_mask
        dv_flat = jnp.where(state.fixed[:, None], 0.0, dv)
        
        state.vel += dv_flat

        dt_remaining = (1.0 - alpha) * system.dt
        state.pos_c += dv_flat * dt_remaining

        return state, system


__all__ = ["ReflectSphereDomain"]
