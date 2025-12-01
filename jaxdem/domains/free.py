# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Unbounded (free) simulation domain."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple
from functools import partial

from . import Domain

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Domain.register("free")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
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
    """

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"), inline=True)
    @partial(jax.named_call, name="FreeDomain.apply")
    def apply(state: "State", system: "System") -> Tuple["State", "System"]:
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

        Note
        -----
        - This method donates state and system
        """
        pos = state.pos
        p_min = jnp.min(pos - state.rad[..., None], axis=-2)
        p_max = jnp.max(pos + state.rad[..., None], axis=-2)
        system.domain.box_size = p_max - p_min
        system.domain.anchor = p_min
        return state, system


__all__ = ["FreeDomain"]
