# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Lees-Edwards shear-periodic boundary-condition domain."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from functools import partial
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

from . import Domain

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("leesedwards")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LeesEdwardsDomain(Domain):
    """A `Domain` implementation that enforces Lees-Edwards boundary conditions.

    The domain is periodic in all directions. Across the shear-gradient axis
    ``beta``, periodic images are offset along the shear-flow axis ``alpha`` by
    the current shear strain ``gamma``.

    ``gamma`` is a plain state field that both :meth:`displacement` and
    :meth:`shift` read directly. The domain does not advance it; the shear
    protocol is imposed externally by updating ``gamma`` between steps (e.g. in
    ``user_post_step_actions``). For example, constant-rate shear is::

        from dataclasses import replace

        def shear(state, system):
            gamma = system.domain.gamma + gamma_dot * system.dt
            return state, replace(system, domain=replace(system.domain, gamma=gamma))

    while oscillatory shear sets ``gamma = gamma_amp * jnp.sin(omega * system.time)``.
    """

    gamma: jax.Array  # float
    """Current shear strain; the Lees-Edwards image offset along ``alpha`` is
    ``gamma * L_beta``. Updated externally to impose the desired shear protocol."""

    alpha_axis: jax.Array
    """One-hot vector for the shear-flow coordinate."""

    beta_axis: jax.Array
    """One-hot vector for the shear-gradient coordinate."""

    alpha: int = jax.tree.static(default=0)  # type: ignore[attr-defined]
    """Index of the shear-flow coordinate."""

    beta: int = jax.tree.static(default=1)  # type: ignore[attr-defined]
    """Index of the shear-gradient coordinate."""

    @classmethod
    def Create(
        cls,
        dim: int,
        box_size: jax.Array | None = None,
        anchor: jax.Array | None = None,
        gamma: float | jax.Array = 0.0,
        alpha: int = 0,
        beta: int = 1,
        **kwargs: Any,
    ) -> "LeesEdwardsDomain":
        """Construct a Lees-Edwards domain with validated shear axes."""
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

        alpha = int(alpha)
        beta = int(beta)
        if not 0 <= alpha < dim:
            raise ValueError(f"alpha must be in [0, {dim}), got {alpha}.")
        if not 0 <= beta < dim:
            raise ValueError(f"beta must be in [0, {dim}), got {beta}.")
        if alpha == beta:
            raise ValueError("alpha and beta must be distinct axes.")

        dtype = box_size.dtype
        return cls(
            box_size=box_size,
            inv_box_size=1.0 / box_size,
            anchor=anchor,
            gamma=jnp.asarray(gamma, dtype=float),
            alpha_axis=jax.nn.one_hot(alpha, dim, dtype=dtype),
            beta_axis=jax.nn.one_hot(beta, dim, dtype=dtype),
            alpha=alpha,
            beta=beta,
        )

    @property
    def periodic(self) -> bool:
        """Whether the domain enforces periodic boundary conditions."""
        return True

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="LeesEdwardsDomain.displacement")
    def displacement(ri: jax.Array, rj: jax.Array, system: System) -> jax.Array:
        r"""Computes the shear-periodic minimum image displacement vector.

        When the minimum image crosses the shear-gradient axis ``beta``, the
        displacement is shifted along the shear-flow axis ``alpha`` by
        :math:`\gamma L_\beta` per crossed image.

        Parameters
        ----------
        ri : jax.Array
            Position vector of the first particle :math:`r_i`.
        rj : jax.Array
            Position vector of the second particle :math:`r_j`.
        system : System
            The configuration of the simulation, containing the `domain` instance
            with `box_size` and Lees-Edwards shear parameters.

        Returns
        -------
        jax.Array
            The shear-periodic minimum image displacement vector:

            .. math::
                & r_{ij} = r_i - r_j \\\\
                & r_{ij,\alpha} = r_{ij,\alpha}
                    - \operatorname{round}(r_{ij,\beta}/L_\beta)\gamma L_\beta \\\\
                & r_{ij} = r_{ij} - L \left\lfloor 0.5 + r_{ij}/L \right\rfloor

            where:
                - :math:`L` is the domain box size (:attr:`Domain.box_size`)

        """
        le_domain = cast("LeesEdwardsDomain", system.domain)
        rij = ri - rj
        beta_rij = jnp.sum(rij * le_domain.beta_axis, axis=-1, keepdims=True)
        beta_length = jnp.sum(le_domain.box_size * le_domain.beta_axis)
        gamma = le_domain.gamma
        rij = (
            rij
            - jnp.round(beta_rij / beta_length)
            * beta_length
            * gamma
            * le_domain.alpha_axis
        )
        return rij - le_domain.box_size * jnp.round(rij * le_domain.inv_box_size)

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="LeesEdwardsDomain.shift")
    def shift(state: State, system: System) -> tuple[State, System]:
        r"""Wraps particles back into the primary shear-periodic simulation box.

        .. math::
            & n_\beta = \left\lfloor (r_\beta - a_\beta)/L_\beta \right\rfloor \\\\
            & r_\alpha = r_\alpha - n_\beta \gamma L_\beta \\\\
            & r = r - L \left\lfloor (r-a)/L \right\rfloor

        where:
            - :math:`a` is the domain anchor (:attr:`Domain.anchor`)
            - :math:`L` is the domain box size (:attr:`Domain.box_size`)

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
        pos = state.pos
        le_domain = cast("LeesEdwardsDomain", system.domain)
        image = jnp.floor((pos - le_domain.anchor) * le_domain.inv_box_size)
        beta_image = jnp.sum(image * le_domain.beta_axis, axis=-1, keepdims=True)
        beta_length = jnp.sum(le_domain.box_size * le_domain.beta_axis)
        gamma = le_domain.gamma
        shear_offset = beta_image * gamma * beta_length * le_domain.alpha_axis

        shifted_pos = pos - shear_offset
        image = jnp.floor((shifted_pos - le_domain.anchor) * le_domain.inv_box_size)
        pos_c = state.pos_c - shear_offset - le_domain.box_size * image
        return replace(state, pos_c=pos_c), system


__all__ = ["LeesEdwardsDomain"]
