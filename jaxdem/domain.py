# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining domains. The domain performs boundary conditions coordinate transformation and computes the displacement vector according to the boundary conditions.
"""

import jax
import jax.numpy as jnp

from typing import ClassVar, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from .factory import Factory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import State
    from .system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class Domain(Factory["Domain"], ABC):
    """
    The base interface for defining the simulation domain and the effect of its boundary conditions.

    The `Domain` class defines how:
        - Relative displacement vectors between particles are calculated.
        - Particles' positions are "shifted" or constrained to remain within the
          defined simulation boundaries based on the boundary condition type.

    Notes
    -----
    All concrete `Domain` implementations must support both 2D and 3D simulations.
    All methods must be JIT-compatible.

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
    """Length of the simulation domain along each dimension. Defines the size of the simulation box."""

    anchor: jax.Array
    """Anchor position of the simulation domain. This represents the minimum coordinate
    (e.g., the "left-down corner") of the domain in each dimension."""

    periodic: ClassVar[bool] = False
    """
    Whether or not the domain is periodic.

    This is a class-level attribute that should be set to `True` for periodic
    boundary condition implementations.
    """


@classmethod
def _create(
    cls,
    dim: int,
    box_size: Optional[jax.Array] = None,
    anchor: Optional[jax.Array] = None,
):
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

    assert (
        box_size.shape == anchor.shape
    ), f"box_size.shape={box_size.shape} does not match anchor.shape={anchor.shape}"

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

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by subclasses.

        Example
        -------
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

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by subclasses.

        Example
        -------
        """
        raise NotImplementedError


@Domain.register("free")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class FreeDomain(Domain):
    """
    A `Domain` implementation representing an unbounded, "free" space.

    In a `FreeDomain`, there are no explicit boundary conditions applied to
    particles. Particles can move indefinitely in any direction, and the
    concept of a "simulation box" is only used to define the bounding box of the system.

    Notes
    -----
    - The `box_size` and `anchor` attributes are dynamically updated in
      the `shift` method to encompass all particles. Some hashing tools require the domain size.

    Example
    -------
    >>> system = jaxdem.System.create(dim=2, domain_type="free", domain_kw=dict(box_size=jnp.array([10., 10.]), anchor=jnp.array([0., 0.])))
    >>>
    >>> # After a step, particle moves, and the domain's effective box_size and anchor update
    >>> state, system = sim_system.domain.shift(state, system)
    >>> print("Updated Domain Anchor:", system.domain.anchor)
    >>> print("Updated Domain Box Size:", system.domain.box_size)
    """

    @staticmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, _: "System") -> jax.Array:
        r"""
        Computes the displacement vector between two particles.

        In a free domain, the displacement is simply the direct vector difference
        between the particle positions.

        Parameters
        ----------
        ri : jax.Array
            Position vector of the first particle :math:`r_i`.
        rj : jax.Array
            Position vector of the second particle :math:`r_j`.
        _ : System
            The system object.

        Returns
        -------
        jax.Array
            The direct displacement vector :math:`r_i - r_j`.
        """
        return ri - rj

    @staticmethod
    @jax.jit
    def shift(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Updates the `System`'s domain `anchor` and `box_size` to encompass all particles. Does not apply any transformations to the state.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The current system configuration.

        Returns
        -------
        Tuple[State, System]
            The original `State` object (unchanged) and the `System` object
            with updated `domain.anchor` and `domain.box_size`.
        """
        p_min = jnp.min(state.pos - state.rad[..., None], axis=-2)
        p_max = jnp.max(state.pos + state.rad[..., None], axis=-2)
        domain = replace(system.domain, box_size=p_max - p_min, anchor=p_min)
        system = replace(system, domain=domain)
        return state, system


@Domain.register("reflect")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class ReflectDomain(Domain):
    """
    A `Domain` implementation that enforces reflective boundary conditions.

    Particles that attempt to move beyond the defined `box_size` will have their
    positions reflected back into the box and their velocities reversed in the
    direction normal to the boundary.

    Notes
    -----
    - The reflection occurs at the boundaries defined by `anchor` and `anchor + box_size`.

    Example
    -------

    >>> system = jaxdem.System.create(dim=2, domain_type="reflect", domain_kw=dict(box_size=jnp.array([10., 7.]), anchor=jnp.array([1., 0.])))
    """

    @staticmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, _: "System") -> jax.Array:
        r"""
        Computes the displacement vector between two particles.

        In a reflective domain, the displacement is simply the direct vector difference.

        Parameters
        ----------
        ri : jax.Array
            Position vector of the first particle :math:`r_i`.
        rj : jax.Array
            Position vector of the second particle :math:`r_j`.
        _ : System
            The system object.

        Returns
        -------
        jax.Array
            The direct displacement vector :math:`r_i - r_j`.
        """
        return ri - rj

    @staticmethod
    @jax.jit
    def shift(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Applies reflective boundary conditions to particles.

        Particles are checked against the domain boundaries.
        If a particle attempts to move beyond a boundary, its position is reflected
        back into the box, and its velocity component normal to that boundary is reversed.

        .. math::
            l &= a + R \\\\
            u &= a + B - R \\\\
            v' &= \\begin{cases} -v & \\text{if } r < l \\text{ or } r > u \\\\ v & \\text{otherwise} \\end{cases} \\\\
            r' &= \\begin{cases} 2l - r & \\text{if } r < l \\\\ r & \\text{otherwise} \\end{cases} \\\\
            r'' &= \\begin{cases} 2u - r' & \\text{if } r' > u \\\\ r' & \\text{otherwise} \\end{cases}
            r = r''

        where:
            - :math:`r` is the current particle position (:attr:`state.pos`)
            - :math:`v` is the current particle velocity (:attr:`state.vel`)
            - :math:`a` is the domain anchor (:attr:`system.domain.anchor`)
            - :math:`B` is the domain box size (:attr:`system.domain.box_size`)
            - :math:`R` is the particle radius (:attr:`state.rad`)
            - :math:`l` is the lower boundary for the particle center
            - :math:`u` is the upper boundary for the particle center

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
        """
        lower_bound = system.domain.anchor + state.rad[:, None]
        upper_bound = system.domain.anchor + system.domain.box_size - state.rad[:, None]
        outside_lower = state.pos < lower_bound
        outside_upper = state.pos > upper_bound
        hit = jnp.logical_or(outside_lower, outside_upper)
        state = replace(state, vel=jnp.where(hit, -state.vel, state.vel))
        reflected_pos = jnp.where(
            outside_lower, 2.0 * lower_bound - state.pos, state.pos
        )
        reflected_pos = jnp.where(
            outside_upper, 2.0 * upper_bound - reflected_pos, reflected_pos
        )
        state = replace(state, pos=reflected_pos)
        return state, system


@Domain.register("periodic")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class PeriodicDomain(Domain):
    """
    A `Domain` implementation that enforces periodic boundary conditions.

    Particles that move out of one side of the simulation box re-enter from the
    opposite side. The displacement vector between particles is computed using the minimum image convention.

    Notes
    -----
    - This domain type is periodic (`periodic = True`).
    - The `shift` method wraps particles back into the primary simulation box.

    Example
    -------

    >>> system = jaxdem.System.create(dim=2, domain_type="periodic", domain_kw=dict(box_size=jnp.array([10., 7.]), anchor=jnp.array([1., 0.])))
    """

    periodic: ClassVar[bool] = True

    @staticmethod
    @jax.jit
    def displacement(ri: jax.Array, rj: jax.Array, system: "System") -> jax.Array:
        """
        Computes the minimum image displacement vector between two particles :math:`r_i` and :math:`r_j`.

        For periodic boundary conditions, the displacement is calculated as the
        shortest vector that connects :math:`r_j` to :math:`r_i`, potentially by crossing
        periodic boundaries.

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
                & r_{ij} = (r_i - a) - (r_j - a) \\\\
                & r_{ij} = r_{ij} - B \\cdot \\text{round}(r_{ij}/B)

            where:
                - :math:`a` is the domain anchor (:attr:`system.domain.anchor`)
                - :math:`B` is the domain box size (:attr:`system.domain.box_size`)
        """
        rij = ri - rj
        return rij - system.domain.box_size * jnp.floor(
            rij / system.domain.box_size + 0.5
        )

    @staticmethod
    @jax.jit
    def shift(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Wraps particles back into the primary simulation box.

        .. math::
            r = r - B \\cdot \\text{floor}((r - a)/B) \\\\

        where:
            - :math:`a` is the domain anchor (:attr:`system.domain.anchor`)
            - :math:`B` is the domain box size (:attr:`system.domain.box_size`)

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
        state = replace(
            state,
            pos=state.pos
            - system.domain.box_size
            * jnp.floor((state.pos - system.domain.anchor) / system.domain.box_size),
        )
        return state, system
