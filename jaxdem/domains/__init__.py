# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Simulation domains and boundary-condition implementations."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Tuple

from ..factory import Factory

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover - fallback for older Python
    from typing_extensions import Self  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class Domain(Factory, ABC):
    """
    The base interface for defining the simulation domain and the effect of its boundary conditions.

    The `Domain` class defines how:
        - Relative displacement vectors between particles are calculated.
        - Particles' positions are "shifted" or constrained to remain within the
          defined simulation boundaries based on the boundary condition type.

    Example
    -------
    To define a custom domain, inherit from `Domain` and implement its abstract methods:

    >>> @Domain.register("my_custom_domain")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True)
    >>> class MyCustomDomain(Domain):
            ...
    """

    box_size: jax.Array
    """Length of the simulation domain along each dimension."""

    anchor: jax.Array
    """Anchor position (minimum coordinate) of the simulation domain."""

    periodic: ClassVar[bool] = False
    """Whether the domain enforces periodic boundary conditions."""

    @classmethod
    def Create(
        cls,
        dim: int,
        box_size: Optional[jax.Array] = None,
        anchor: Optional[jax.Array] = None,
    ) -> Self:
        """
        Default factory method for the Domain class.

        This method constructs a new Domain instance with a box-shaped domain
        of the given dimensionality. If `box_size` or `anchor` are not provided,
        they are initialized to default values.

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

        Returns
        -------
        Domain
            A new instance of the Domain subclass with the specified
            or default configuration.

        Raises
        ------
        AssertionError
            If `box_size` and `anchor` do not have the same shape.
        """
        if box_size is None:
            box_size = jnp.ones(dim, dtype=float)
        box_size = jnp.asarray(box_size, dtype=float)

        if anchor is None:
            anchor = jnp.zeros_like(box_size, dtype=float)
        anchor = jnp.asarray(anchor, dtype=float)

        assert box_size.shape == anchor.shape
        return cls(box_size=box_size, anchor=anchor)

    @staticmethod
    @abstractmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, system: "System") -> jax.Array:
        r"""
        Computes the displacement vector between two particles :math:`r_i` and :math:`r_j`,
        considering the domain's boundary conditions.

        Parameters
        ----------
        ri : jax.Array
            Position vector of the first particle :math:`r_i`. Shape `(dim,)`.
        rj : jax.Array
            Position vector of the second particle :math:`r_j`. Shape `(dim,)`.
        system : System
            The configuration of the simulation, containing the `domain` instance.

        Returns
        -------
        jax.Array
            The displacement vector :math:`r_{ij} = r_i - r_j`,
            adjusted for boundary conditions. Shape `(dim,)`.

        Example
        -------
        >>> rij = system.domain.displacement(ri, rj, system)
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def shift(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Applies boundary conditions to particles state.

        This method updates the `state` based on the domain's rules, ensuring
        particles remain within the simulation box or handle interactions
        at boundaries appropriately (e.g., reflection, wrapping).

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated `State` object adjusted by the boundary conditions and the `System` object.

        Example
        -------
        >>> state, system = system.domain.shift(state, system)
        """
        raise NotImplementedError


from .free import FreeDomain
from .periodic import PeriodicDomain
from .reflect import ReflectDomain

__all__ = ["Domain", "FreeDomain", "PeriodicDomain", "ReflectDomain"]
