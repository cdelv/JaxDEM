# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Utilities for managing per-particle force contributions."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence, Tuple, Optional
import operator
from functools import partial

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


ForceFunction = Callable[["State", "System", int], jax.Array]


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ForceManager:
    """
    Manage per-particle force contributions prior to pairwise interactions.
    """

    gravity: jax.Array
    """
    Constant acceleration applied to all particles. Shape ``(dim,)``.
    """

    external_accel: jax.Array
    """
    Accumulated external acceleration applied uniformly to all particles. This
    buffer is cleared when :meth:`apply` is invoked.
    """

    iota: jax.Array
    """
    iota(N, int) where N is the number of particles.
    """

    force_functions: Tuple[ForceFunction, ...] = field(
        default=(), metadata={"static": True}
    )
    """
    Optional tuple of callables with signature ``(state, system, i)`` returning
    per-particle acceleration contributions for particle ``i``.
    """

    @staticmethod
    def create(
        dim: int,
        shape: Tuple,
        *,
        gravity: jax.Array | None = None,
        force_functions: Sequence[ForceFunction] = (),
    ) -> "ForceManager":
        """Create a :class:`ForceManager` for a system with ``dim`` spatial dimensions.

        Parameters
        ----------
        dim:
            Spatial dimension of the managed system.
        gravity:
            Optional initial gravitational acceleration. Defaults to zeros.
        force_functions:
            Sequence of callables with signature (state, system, index) returning per-particle acceleration contributions.
        """
        gravity = (
            jnp.zeros(dim, dtype=float)
            if gravity is None
            else jnp.asarray(gravity, dtype=float)
        )
        external_accel = jnp.zeros(shape, dtype=float)
        return ForceManager(
            gravity=gravity,
            external_accel=external_accel,
            iota=jax.lax.iota(dtype=int, size=shape[-2]),
            force_functions=tuple(force_functions),
        )

    @staticmethod
    def add_force(
        state: "State",
        system: "System",
        force: jax.Array,
        idx: Optional[jax.Array] = None,
    ) -> "System":
        """
        Accumulate an external acceleration to be applied on the next ``apply`` call.

        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.
        force:
            Acceleration contribution to accumulate. Can be a single ``(dim,)`` vector
            applied uniformly or an array broadcastable to ``external_force[idx]``.
        idx:
            Optional integer indices of the particles receiving the contribution. When
            omitted the force is applied to all particles.

        Returns
        -------
        System
            A new :class:`jaxdem.System` instance with the updated accumulated
            accelerations.
        """
        force = jnp.asarray(force, dtype=float)
        if idx is None:
            idx = system.force_manager.iota
        idx = jnp.asarray(idx, dtype=int)
        system.force_manager.external_force.at[idx].set(
            system.force_manager.external_force[idx] + force / state.mass[idx]
        )
        return system

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    def apply(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Overwrite ``state.accel`` with managed per-particle contributions.

        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        Tuple[State, System]
            The updated state and system after one time step.
        """
        accel = system.force_manager.external_accel
        accel += system.force_manager.gravity
        if system.force_manager.force_functions:
            accel += (
                jax.tree.reduce(
                    operator.add,
                    jax.tree.map(
                        lambda func: jax.vmap(lambda i: func(state, system, i))(
                            system.force_manager.iota
                        ),
                        system.force_manager.force_functions,
                    ),
                )
                / state.mass
            )
        system.force_manager.external_accel *= 0.0
        return state, system


__all__ = ["ForceManager"]
