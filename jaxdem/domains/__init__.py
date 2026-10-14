# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Simulation domains and boundary-condition implementations."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp

from ..factory import Factory

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


class SearchGeometry(Enum):
    """Spatial-search geometry implemented by a domain."""

    ORTHOGONAL = "orthogonal"
    SHEAR_PERIODIC = "shear_periodic"


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Domain(Factory, ABC):
    """The base interface for the simulation domain and its boundary conditions.

    The `Domain` class defines:
        - How to compute relative displacement vectors between particles.
        - How to "shift" or constrain particle positions so they stay within
          the simulation boundaries.

    Example:
    --------
    To define a custom domain, inherit from `Domain` and implement its abstract methods:

    >>> @Domain.register("my_custom_domain")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True)
    >>> class MyCustomDomain(Domain):
            ...

    """

    box_size: jax.Array
    """Length of the simulation domain along each dimension."""

    inv_box_size: jax.Array
    """Inverse length of the simulation domain along each dimension."""

    anchor: jax.Array
    """Anchor position (minimum coordinate) of the simulation domain."""

    search_geometry: ClassVar[SearchGeometry | None] = None

    def search_geometry_snapshot(self) -> jax.Array:
        """Return fixed-shape geometry data used to validate search caches."""
        return jnp.zeros((self.box_size.shape[0] + 3,), dtype=self.box_size.dtype)

    def search_geometry_drift(
        self, previous: jax.Array, old_pos: jax.Array
    ) -> jax.Array:
        """Bound image motion since the geometry snapshot of the last cache build."""
        return jnp.asarray(0.0, dtype=self.box_size.dtype)

    def neighbor_cells(
        self,
        coords: jax.Array,
        image: jax.Array,
        grid_dims: jax.Array,
        cell_size: jax.Array,
    ) -> jax.Array:
        """Adjust wrapped stencil cells for the domain's periodic image rule.

        ``coords`` and signed ``image`` counts have shape ``(N, stencil, dim)``.
        The base rule keeps the orthogonal stencil; shear-periodic domains may
        expand it to cover fractional image offsets.
        """
        return coords

    def rates_between(self, old: Domain, dt: jax.Array) -> Domain:
        """Return this domain with image rates for the change from ``old`` over ``dt``.

        Domains with moving periodic images (shear strain, box length) set
        the rate fields that :meth:`relative_velocity` reads, so a finite
        geometry change can be expressed as velocities. The base domain has
        no moving images and returns itself.
        """
        return self

    @property
    def periodic(self) -> bool:
        """Whether the domain enforces periodic boundary conditions."""
        return False

    @staticmethod
    @jax.jit(inline=True)
    def _shift(pos: jax.Array, system: System) -> jax.Array:
        """Map query points according to this domain's image rule.

        ``system`` is part of the uniform domain operation signature; the base
        no-op has no geometry to read from it.
        """
        return pos

    @staticmethod
    @jax.jit(inline=True)
    def update_bounds(
        pos: jax.Array, system: System, padding: float | jax.Array = 0.0
    ) -> System:
        """Update query bounds when the domain derives them from particle positions."""
        return system

    @classmethod
    def Create(
        cls,
        dim: int,
        box_size: jax.Array | None = None,
        anchor: jax.Array | None = None,
        **kw: Any,
    ) -> Self:
        """Default factory method for the Domain class.

        This method constructs a new Domain instance with a box-shaped domain
        of the given dimensionality. If you do not provide `box_size` or
        `anchor`, they default to the values below.

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
        **kw : Any
            Extra keyword arguments passed to the subclass constructor
            (e.g. ``restitution_coefficient`` for reflective domains).

        Returns
        -------
        Domain
            A new instance of the Domain subclass with the specified
            or default configuration.

        Raises
        ------
        ValueError
            If `box_size` or `anchor` do not have shape `(dim,)`.

        """
        if box_size is None:
            box_size = jnp.ones(dim, dtype=float)
        box_size = jnp.asarray(box_size, dtype=float)

        if box_size.shape != (dim,):
            raise ValueError(
                f"box_size must have shape ({dim},), got shape {box_size.shape}."
            )

        if anchor is None:
            anchor = jnp.zeros_like(box_size, dtype=float)
        anchor = jnp.asarray(anchor, dtype=float)

        if anchor.shape != (dim,):
            raise ValueError(
                f"anchor must have shape ({dim},), got shape {anchor.shape}."
            )

        if not isinstance(box_size, jax.core.Tracer):
            if not bool(jnp.all(jnp.isfinite(box_size))) or not bool(
                jnp.all(box_size > 0)
            ):
                raise ValueError("box_size must contain only finite positive values.")
        if not isinstance(anchor, jax.core.Tracer) and not bool(
            jnp.all(jnp.isfinite(anchor))
        ):
            raise ValueError("anchor must contain only finite values.")

        return cls(box_size=box_size, inv_box_size=1.0 / box_size, anchor=anchor, **kw)

    @staticmethod
    @jax.jit(inline=True)
    def displacement(ri: jax.Array, rj: jax.Array, system: System) -> jax.Array:
        r"""Compute the displacement vector between two particles :math:`r_i` and :math:`r_j`,
        respecting the domain's boundary conditions.

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
        return ri - rj

    @staticmethod
    @jax.jit(inline=True)
    def _displacement(ri: jax.Array, rj: jax.Array, system: System) -> jax.Array:
        return system.domain.displacement(ri, rj, system)

    @staticmethod
    @jax.jit(inline=True)
    def relative_velocity(
        ri: jax.Array,
        rj: jax.Array,
        vi: jax.Array,
        vj: jax.Array,
        system: System,
    ) -> jax.Array:
        """Return relative velocity for the image selected by ``displacement``.

        ``ri`` and ``rj`` are the particle centers used to select that image.
        ``vi`` and ``vj`` may include rotational contact-point velocities.
        Domains with moving periodic images override this operation to add
        the image's translational velocity offset.
        """
        return vi - vj

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="Domain.resize")
    def resize(
        state: State,
        system: System,
        box_size: jax.Array,
        group_id: jax.Array | None = None,
    ) -> tuple[State, System]:
        """Rescale the box to ``box_size`` and move bodies affinely about the anchor.

        Each body (a clump, or the group given by ``group_id``) is translated
        so that its center of mass follows the affine map; body size and
        orientation are unchanged. Pair history is advanced once for the
        resulting relative motion without time passing, so velocities,
        ``dt``, and rate-dependent forces play no part. Caches rebuild when
        the geometry change exhausts their skin.

        The history update is not a no-op even though the map is affine.
        Bodies move rigidly, so two member spheres of different bodies are
        displaced by the difference of their body offsets, which is not
        parallel to their contact normal; the tangential spring of such a
        contact changes. For single spheres under isotropic scaling the
        relative displacement is purely normal and the stored spring is
        unchanged. Periodic images move with the box, so the domain's
        ``box_rate`` supplies their velocity and a pair interacting across a
        boundary sees only its true relative motion.

        Parameters
        ----------
        state, system : State, System
            Configuration to rescale.
        box_size : jax.Array
            New box lengths, shape ``(dim,)``.
        group_id : jax.Array, optional
            Per-particle body labels, shape ``(N,)``. Defaults to
            ``state.clump_id``. Deformable particles pass their bond-graph
            component labels so each body moves as a unit.
        """
        from ..colliders import advance_history

        domain = system.domain
        new_box = jnp.asarray(box_size, dtype=domain.box_size.dtype)
        scale = new_box / domain.box_size
        group = state.clump_id if group_id is None else group_id
        total = jax.ops.segment_sum(state.pos_c, group, num_segments=state.N)
        counts = jax.ops.segment_sum(
            jnp.ones((state.N,), dtype=state.pos_c.dtype), group, num_segments=state.N
        )
        com = total / jnp.maximum(counts[:, None], 1.0)
        offset = (com - domain.anchor) * (scale - 1.0)
        new_state = replace(state, pos_c=state.pos_c + offset[group])
        new_system = replace(
            system, domain=replace(domain, box_size=new_box, inv_box_size=1.0 / new_box)
        )
        _, new_system = advance_history(state, system, new_state, new_system)
        return new_state, new_system

    @staticmethod
    @jax.jit(inline=True)
    def apply(state: State, system: System) -> tuple[State, System]:
        """Apply boundary conditions during the simulation step.

        This method updates the `state` with the domain's rules so particles
        handle boundary interactions (e.g., reflection).

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

        Note
        ----
        - Periodic domains do not need to wrap coordinates during time
          stepping, so their ``apply`` is a no-op. :meth:`shift` wraps the
          coordinates instead (e.g. when saving, so positions are displayed
          inside the box). Reflective domains, in contrast, must update
          positions and velocities here.

        Example
        -------
        >>> state, system = system.domain.apply(state, system)

        """
        return state, system

    @staticmethod
    @jax.jit(inline=True)
    def shift(state: State, system: System) -> tuple[State, System]:
        """Shift particles according to the domain's boundary-condition rules.

        This method updates the `state` with the domain's rules so particles
        stay within the simulation box or handle boundary interactions
        (e.g., reflection, wrapping).

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
        return state, system


from .free import FreeDomain
from .lees_edwards import LeesEdwardsDomain
from .periodic import PeriodicDomain
from .reflect import ReflectDomain
from .reflect_sphere import ReflectSphereDomain

__all__ = [
    "Domain",
    "SearchGeometry",
    "FreeDomain",
    "LeesEdwardsDomain",
    "PeriodicDomain",
    "ReflectDomain",
    "ReflectSphereDomain",
]
