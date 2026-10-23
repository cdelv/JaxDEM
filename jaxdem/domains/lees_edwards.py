# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Lees-Edwards shear-periodic boundary-condition domain."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp

from . import Domain, SearchGeometry
from .periodic import PeriodicDomain, _image_span

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("leesedwards")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LeesEdwardsDomain(PeriodicDomain):
    r"""A `Domain` implementation that enforces Lees-Edwards boundary conditions.

    The domain is periodic in all directions. Across the shear-gradient axis
    ``beta``, the current shear strain ``gamma`` offsets the periodic images
    along the shear-flow axis ``alpha``.

    ``gamma`` is the current shear strain and ``gamma_dot`` its time derivative.
    Time stepping does not advance these fields automatically. Set the initial
    values before initializing forces and update the protocol in
    ``user_pre_step_actions``, before each force evaluation. For constant-rate
    shear, set ``gamma_dot`` when creating the domain and use::

        from dataclasses import replace

        def shear(state, system):
            gamma = system.domain.gamma + system.domain.gamma_dot * system.dt
            return state, replace(system, domain=replace(system.domain, gamma=gamma))

    For oscillatory shear, set ``gamma = gamma_amp * jnp.sin(omega * system.time)``
    and ``gamma_dot = gamma_amp * omega * jnp.cos(omega * system.time)`` in the
    same callback. Contact laws use :meth:`relative_velocity <jaxdem.domains.lees_edwards.LeesEdwardsDomain.relative_velocity>` to account for
    the moving image even when particle coordinates remain unwrapped.

    For discrete affine shear, call :meth:`shear <jaxdem.domains.lees_edwards.LeesEdwardsDomain.shear>` to apply one strain increment
    to particle centers, periodic images, and any pair history. Quasistatic
    protocols can then minimize the returned state at fixed strain. Box
    rescaling goes through :meth:`Domain.resize <jaxdem.domains.Domain.resize>`; ``box_rate`` is the box
    length rate for continuous protocols and is zero otherwise.

    ``box_rate`` is inherited from :class:`PeriodicDomain <jaxdem.domains.periodic.PeriodicDomain>`. Periodic images
    move at ``image * box_rate`` when the box changes, and the shear offset
    ``gamma * L_beta`` of an image across the gradient boundary moves with
    ``L_beta`` as well. :meth:`relative_velocity <jaxdem.domains.lees_edwards.LeesEdwardsDomain.relative_velocity>` subtracts both so a pair
    interacting across a boundary sees only its true relative motion.
    See :attr:`PeriodicDomain.box_rate <jaxdem.domains.periodic.PeriodicDomain.box_rate>`.

    For fixed integer image counts :math:`\mathbf n`, the image position and
    velocity are

    .. math::

        \mathbf r_{j,\mathrm{image}}
        &= \mathbf r_j + \mathbf n \odot \mathbf L
           + n_\beta \gamma L_\beta \mathbf e_\alpha, \\
        \mathbf v_{\mathrm{image}}
        &= \mathbf n \odot \dot{\mathbf L}
           + n_\beta (\dot\gamma L_\beta + \gamma\dot L_\beta)
             \mathbf e_\alpha.

    Here :math:`\mathbf v_{\mathrm{image}}` is the velocity offset relative to
    particle ``j``, so the contact relative velocity is
    :math:`\mathbf v_i - \mathbf v_j - \mathbf v_{\mathrm{image}}`.
    """

    search_geometry = SearchGeometry.SHEAR_PERIODIC

    @staticmethod
    @jax.jit(inline=True)
    def _shift(pos: jax.Array, system: System) -> jax.Array:
        r"""Wrap coordinates after removing their shear-image offset.

        The inherited :meth:`PeriodicDomain.shift <jaxdem.domains.periodic.PeriodicDomain.shift>` calls this shear-specific
        rule on body centers, preserving each clump's internal geometry.

        .. math::

            n_\beta &= \left\lfloor(r_\beta-a_\beta)/L_\beta\right\rfloor, \\
            \widetilde{\mathbf r}
            &= \mathbf r - n_\beta\gamma L_\beta\mathbf e_\alpha, \\
            \mathbf r' &= \mathbf a +
                \operatorname{mod}(\widetilde{\mathbf r}-\mathbf a,\mathbf L).
        """
        domain = cast("LeesEdwardsDomain", system.domain)
        beta_image = jnp.floor(
            (pos[..., domain.beta] - domain.anchor[domain.beta])
            / domain.box_size[domain.beta]
        )
        shifted = pos.at[..., domain.alpha].add(
            -beta_image * domain.gamma * domain.box_size[domain.beta]
        )
        return domain.anchor + jnp.mod(shifted - domain.anchor, domain.box_size)

    gamma: jax.Array  # float
    """Current shear strain. The Lees-Edwards image offset along ``alpha`` is
    ``gamma * L_beta``. Update it externally to impose the shear protocol."""

    alpha_axis: jax.Array
    """Legacy serialized one-hot cache; runtime geometry uses canonical ``alpha``."""

    beta_axis: jax.Array
    """Legacy serialized one-hot cache; runtime geometry uses canonical ``beta``."""

    alpha: int = jax.tree.static(default=0)
    """Index of the shear-flow coordinate."""

    beta: int = jax.tree.static(default=1)
    """Index of the shear-gradient coordinate."""

    gamma_dot: jax.Array = field(default_factory=lambda: jnp.asarray(0.0, dtype=float))
    """Current shear rate, used for image velocities. Keep it consistent with
    the externally imposed derivative of ``gamma``; it does not advance strain."""

    def rates_between(self, old: Domain, dt: jax.Array) -> Domain:
        """Set ``gamma_dot`` and ``box_rate`` from the geometry change since ``old``."""
        le_old = cast("LeesEdwardsDomain", old)
        return replace(
            self,
            gamma_dot=(self.gamma - le_old.gamma) / dt,
            box_rate=(self.box_size - old.box_size) / dt,
        )

    def search_geometry_snapshot(self) -> jax.Array:
        """Return box and canonical shear geometry used by search caches."""
        return jnp.concatenate(
            (
                self.box_size,
                jnp.reshape(self.gamma, (1,)),
                jnp.asarray((self.alpha, self.beta), dtype=self.box_size.dtype),
            )
        )

    def search_geometry_drift(
        self, previous: jax.Array, old_pos: jax.Array
    ) -> jax.Array:
        """Bound box and shear image motion, including unwrapped coordinates."""
        old_box = previous[: self.box_size.size]
        images, lengths = _image_span(old_box, self.box_size, old_pos)
        old_offset = previous[self.box_size.size] * old_box[self.beta]
        offset = self.gamma * self.box_size[self.beta]
        beta_images = images[self.beta]
        flow_images = jnp.ceil(
            images[self.alpha]
            + beta_images
            * jnp.maximum(jnp.abs(old_offset), jnp.abs(offset))
            / lengths[self.alpha]
        )
        images = images.at[self.alpha].set(flow_images)
        box_drift = jnp.sum(images * jnp.abs(self.box_size - old_box))
        return box_drift + beta_images * jnp.abs(offset - old_offset)

    def neighbor_cells(
        self,
        coords: jax.Array,
        image: jax.Array,
        grid_dims: jax.Array,
        cell_size: jax.Array,
    ) -> jax.Array:
        """Cover both flow-axis cells touched by each fractional shear image."""
        alpha_shift = (
            -image[..., self.beta].astype(cell_size.dtype)
            * self.gamma
            * self.box_size[self.beta]
            / cell_size[self.alpha]
        )
        lo = jnp.floor(alpha_shift).astype(image.dtype)
        hi = jnp.ceil(alpha_shift).astype(image.dtype)
        offsets = jnp.stack((lo, hi), axis=-1).reshape(lo.shape[0], 2 * lo.shape[1])
        expanded = jnp.repeat(coords, 2, axis=1)
        base = expanded[..., self.alpha]
        positive = offsets >= 0
        magnitude = jnp.where(positive, offsets, -offsets).astype(coords.dtype)
        dim = grid_dims[self.alpha]
        remainder = magnitude % dim
        forward = jnp.where(
            (remainder == 0) | (base < dim - remainder),
            base + remainder,
            base - (dim - remainder),
        )
        backward = jnp.where(
            remainder > base,
            dim - (remainder - base),
            base - remainder,
        )
        return expanded.at[..., self.alpha].set(jnp.where(positive, forward, backward))

    @classmethod
    def Create(
        cls,
        dim: int,
        box_size: jax.Array | None = None,
        anchor: jax.Array | None = None,
        gamma: float | jax.Array = 0.0,
        alpha: int = 0,
        beta: int = 1,
        gamma_dot: float | jax.Array = 0.0,
        box_rate: jax.Array | None = None,
        **kwargs: Any,
    ) -> "LeesEdwardsDomain":
        """Construct a Lees-Edwards domain with validated shear axes."""
        alpha = int(alpha)
        beta = int(beta)
        if not 0 <= alpha < dim:
            raise ValueError(f"alpha must be in [0, {dim}), got {alpha}.")
        if not 0 <= beta < dim:
            raise ValueError(f"beta must be in [0, {dim}), got {beta}.")
        if alpha == beta:
            raise ValueError("alpha and beta must be distinct axes.")

        gamma = jnp.asarray(gamma, dtype=float)
        if gamma.ndim != 0:
            raise ValueError(f"gamma must be a scalar, got shape {gamma.shape}.")
        if not isinstance(gamma, jax.core.Tracer) and not bool(jnp.isfinite(gamma)):
            raise ValueError("gamma must be finite.")

        gamma_dot = jnp.asarray(gamma_dot, dtype=float)
        if gamma_dot.ndim != 0:
            raise ValueError(
                f"gamma_dot must be a scalar, got shape {gamma_dot.shape}."
            )
        if not isinstance(gamma_dot, jax.core.Tracer) and not bool(
            jnp.isfinite(gamma_dot)
        ):
            raise ValueError("gamma_dot must be finite.")

        # Explicit super is required for slotted dataclasses, as in ReflectDomain.
        return super(LeesEdwardsDomain, cls).Create(
            dim=dim,
            box_size=box_size,
            anchor=anchor,
            gamma=gamma,
            gamma_dot=gamma_dot,
            box_rate=box_rate,
            alpha_axis=jax.nn.one_hot(alpha, dim, dtype=float),
            beta_axis=jax.nn.one_hot(beta, dim, dtype=float),
            alpha=alpha,
            beta=beta,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="LeesEdwardsDomain.shear")
    def shear(
        state: State, system: System, dgamma: float | jax.Array
    ) -> tuple[State, System]:
        """Apply an affine strain increment about the domain anchor.

        Move body centers along ``alpha`` by ``dgamma * (r_beta - anchor_beta)``
        and increment ``gamma`` by ``dgamma``. Rigid-body offsets and
        orientations are preserved. Pair history is advanced once for the
        resulting relative motion, including that of periodic shear images,
        without time passing: velocities, ``dt``, ``gamma_dot``, and
        rate-dependent forces play no part. Caches rebuild when the strain
        change exhausts their skin. For quasistatic shear, minimize the
        returned state at its new strain before applying the next increment.
        """
        from ..colliders import advance_history

        domain = cast("LeesEdwardsDomain", system.domain)
        affine_shift = dgamma * (
            state.pos_c[..., domain.beta] - domain.anchor[domain.beta]
        )
        sheared_state = replace(
            state, pos_c=state.pos_c.at[..., domain.alpha].add(affine_shift)
        )
        sheared_system = replace(
            system, domain=replace(domain, gamma=domain.gamma + dgamma)
        )
        _, sheared_system = advance_history(
            state, system, sheared_state, sheared_system
        )
        return sheared_state, sheared_system

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="LeesEdwardsDomain.displacement")
    def displacement(ri: jax.Array, rj: jax.Array, system: System) -> jax.Array:
        r"""Compute the shear-periodic minimum image displacement vector.

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
                & r_{ij} = r_{ij} - L\operatorname{round}(r_{ij}/L)

            where:
                - :math:`L` is the domain box size (:attr:`Domain.box_size <jaxdem.domains.Domain.box_size>`)
                - Rounding uses ties to even at half-box displacements.

        """
        le_domain = cast("LeesEdwardsDomain", system.domain)
        rij = ri - rj
        gamma = le_domain.gamma
        beta_length = le_domain.box_size[le_domain.beta]
        shear_image = jnp.round(rij[..., le_domain.beta] / beta_length)
        rij = rij.at[..., le_domain.alpha].add(-shear_image * beta_length * gamma)
        return rij - le_domain.box_size * jnp.round(rij / le_domain.box_size)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="LeesEdwardsDomain.relative_velocity")
    def relative_velocity(
        ri: jax.Array,
        rj: jax.Array,
        vi: jax.Array,
        vj: jax.Array,
        system: System,
    ) -> jax.Array:
        """Subtract the velocity of the same image used by ``displacement``.

        The image of ``j`` sits at ``rj + n * L + n_beta * gamma * L_beta e_alpha``
        with integer image counts ``n``. Its velocity is ``n * box_rate`` plus
        ``n_beta * (gamma_dot * L_beta + gamma * box_rate_beta)`` along
        ``alpha``. Image counts use the supplied unwrapped particle centers,
        including when they span multiple boxes.
        """
        domain = cast("LeesEdwardsDomain", system.domain)
        rij = ri - rj
        beta_length = domain.box_size[domain.beta]
        shear_image = jnp.round(rij[..., domain.beta] / beta_length)
        rij = rij.at[..., domain.alpha].add(-shear_image * beta_length * domain.gamma)
        image = jnp.round(rij / domain.box_size)
        image_velocity = image * domain.box_rate
        image_velocity = image_velocity.at[..., domain.alpha].add(
            shear_image
            * (
                domain.gamma_dot * beta_length
                + domain.gamma * domain.box_rate[domain.beta]
            )
        )
        return vi - vj - image_velocity


__all__ = ["LeesEdwardsDomain"]
