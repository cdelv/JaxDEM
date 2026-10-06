# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Periodic boundary-condition domain."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

import jax
import jax.numpy as jnp

from . import Domain, SearchGeometry

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("periodic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PeriodicDomain(Domain):
    """A `Domain` implementation that enforces periodic boundary conditions.

    Particles that move out of one side of the simulation box re-enter from the
    opposite side. The domain computes the displacement vector between
    particles with the minimum image convention.

    ``box_rate`` is the time derivative of ``box_size``. Periodic images of a
    particle move at ``image * box_rate``, which :meth:`relative_velocity`
    subtracts. Time stepping does not advance ``box_size``; set ``box_rate``
    when a protocol changes the box continuously, and leave it zero otherwise.
    Discrete rescaling goes through :meth:`Domain.resize`.
    """

    search_geometry = SearchGeometry.ORTHOGONAL

    box_rate: jax.Array = field(default_factory=lambda: jnp.zeros((0,), dtype=float))
    """Time derivative of ``box_size`` for moving periodic images."""

    @classmethod
    def Create(
        cls,
        dim: int,
        box_size: jax.Array | None = None,
        anchor: jax.Array | None = None,
        box_rate: jax.Array | None = None,
        **kw: Any,
    ) -> Self:
        """Construct a periodic box; ``box_rate`` defaults to zero."""
        rate = (
            jnp.zeros(dim, dtype=float)
            if box_rate is None
            else jnp.asarray(box_rate, dtype=float)
        )
        if rate.shape != (dim,):
            raise ValueError(
                f"box_rate must have shape ({dim},), got shape {rate.shape}."
            )
        # Two-arg super: dataclass(slots=True) recreates the class.
        return super(PeriodicDomain, cls).Create(
            dim, box_size=box_size, anchor=anchor, box_rate=rate, **kw
        )

    def rates_between(self, old: Domain, dt: jax.Array) -> Domain:
        """Set ``box_rate`` from the box change since ``old``."""
        return replace(self, box_rate=(self.box_size - old.box_size) / dt)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="PeriodicDomain.relative_velocity")
    def relative_velocity(
        ri: jax.Array,
        rj: jax.Array,
        vi: jax.Array,
        vj: jax.Array,
        system: System,
    ) -> jax.Array:
        """Subtract the velocity of the image selected by ``displacement``."""
        domain = system.domain
        image = jnp.round((ri - rj) / domain.box_size)
        return vi - vj - image * domain.box_rate

    def search_geometry_snapshot(self) -> jax.Array:
        """Return periodic box geometry used by search caches."""
        return jnp.concatenate(
            (self.box_size, jnp.zeros((3,), dtype=self.box_size.dtype))
        )

    @staticmethod
    @jax.jit(inline=True)
    def _shift(pos: jax.Array, system: System) -> jax.Array:
        domain = system.domain
        return domain.anchor + jnp.mod(pos - domain.anchor, domain.box_size)

    @property
    def periodic(self) -> bool:
        """Whether the domain enforces periodic boundary conditions."""
        return True

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="PeriodicDomain.displacement")
    def displacement(ri: jax.Array, rj: jax.Array, system: System) -> jax.Array:
        r"""Compute the minimum image displacement vector between two particles :math:`r_i` and :math:`r_j`.

        For periodic boundary conditions, the displacement is the
        shortest vector that connects :math:`r_j` to :math:`r_i`, possibly by
        crossing periodic boundaries.

        Parameters
        ----------
        ri : jax.Array
            Position vector of the first particle :math:`r_i`.
        rj : jax.Array
            Position vector of the second particle :math:`r_j`.
        system : System
            The configuration of the simulation, containing the `domain` instance
            with `anchor` and `box_size` for periodicity.

        Returns
        -------
        jax.Array
            The minimum image displacement vector:

            .. math::
                & r_{ij} = (r_i - a) - (r_j - a) \\
                & r_{ij} = r_{ij} - B \cdot \text{round}(r_{ij}/B)

            where:
                - :math:`a` is the domain anchor (:attr:`Domain.anchor`)
                - :math:`B` is the domain box size (:attr:`Domain.box_size`)

        """
        rij = ri - rj
        return rij - system.domain.box_size * jnp.round(rij / system.domain.box_size)

    @staticmethod
    @jax.jit(inline=True)
    def _displacement(ri: jax.Array, rj: jax.Array, system: System) -> jax.Array:
        rij = ri - rj
        return rij - system.domain.box_size * jnp.round(rij / system.domain.box_size)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="PeriodicDomain.shift")
    def shift(state: State, system: System) -> tuple[State, System]:
        r"""Wrap particles back into the primary simulation box.

        .. math::
            r = r - B \cdot \text{floor}((r - a)/B)

        where:
            - :math:`a` is the domain anchor (:attr:`Domain.anchor`)
            - :math:`B` is the domain box size (:attr:`Domain.box_size`)

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            The updated `State` object with wrapped particle positions, and the
            `System` object.

        """
        # Wrap by the clump representative (the body center `pos_c`, shared by all
        # members of a clump) rather than per-sphere positions: per-sphere wrapping
        # would give clump members straddling a boundary different shifts and tear
        # the clump apart.
        state.pos_c = system.domain._shift(state.pos_c, system)
        return state, system


__all__ = ["PeriodicDomain"]
