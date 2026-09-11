# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Unbounded (free) simulation domain."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from . import Domain

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("free")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class FreeDomain(Domain):
    """A `Domain` implementation for an unbounded, "free" space.

    A `FreeDomain` applies no explicit boundary conditions to particles.
    Particles can move indefinitely in any direction. The "simulation box"
    only defines the bounding box of the system.

    Notes
    -----
    - The `apply` method updates the `box_size` and `anchor` attributes to
      encompass all particles. Some hashing tools require the domain size.

    """

    @staticmethod
    @jax.jit(inline=True)
    def update_bounds(
        pos: jax.Array, system: System, padding: float | jax.Array = 0.0
    ) -> System:
        """Resize the free-space bounds around arbitrary query points."""
        if pos.ndim != 2:
            raise ValueError(
                f"resize expects one snapshot with shape (N, dim); got {pos.shape}. "
                "Use jax.vmap for batched snapshots."
            )
        pad = jnp.asarray(padding, dtype=pos.dtype)
        if pad.ndim == pos.ndim - 1:
            pad = pad[..., None]
        p_min = jnp.min(pos - pad, axis=-2)
        p_max = jnp.max(pos + pad, axis=-2)
        scale = jnp.maximum(jnp.max(jnp.abs(pos)), jnp.asarray(1.0, pos.dtype))
        eps = jnp.finfo(pos.dtype).eps * scale * 16  # type: ignore[no-untyped-call]
        box_size = jnp.maximum(p_max - p_min, eps)
        domain = replace(
            system.domain,
            box_size=box_size,
            inv_box_size=1.0 / box_size,
            anchor=p_min,
        )
        return replace(system, domain=domain)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="FreeDomain.apply")
    def apply(state: State, system: System) -> tuple[State, System]:
        """Update the domain `anchor` and `box_size` of the `System` to encompass all particles.

        This method does not transform the state.

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
        return state, FreeDomain.update_bounds(state.pos, system, state.rad)


__all__ = ["FreeDomain"]
