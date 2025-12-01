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


ForceFunction = Callable[["State", "System", int], Tuple[jax.Array, jax.Array]]


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

    external_force: jax.Array
    """
    Accumulated external force applied to all particles. This
    buffer is cleared when :meth:`apply` is invoked.
    """

    external_torque: jax.Array
    """
    Accumulated external torque applied to all particles. This
    buffer is cleared when :meth:`apply` is invoked.
    """

    force_functions: Tuple[ForceFunction, ...] = field(
        default=(), metadata={"static": True}
    )
    """
    Optional tuple of callables with signature ``(state, system, i)`` returning
    per-particle force and torque contributions for particle ``i``.
    """

    @staticmethod
    @partial(jax.named_call, name="ForceManager.create")
    def create(
        dim: int,
        shape: Tuple,
        *,
        gravity: jax.Array | None = None,
        force_functions: Sequence[ForceFunction] = (),
    ) -> "ForceManager":
        """
        Create a :class:`ForceManager` for a system with ``dim`` spatial dimensions.

        Parameters
        ----------
        dim:
            Spatial dimension of the managed system.
        gravity:
            Optional initial gravitational acceleration. Defaults to zeros.
        force_functions:
            Sequence of callables with signature (state, system, index) returning per-particle force/torque contributions.
        """
        gravity = (
            jnp.zeros(dim, dtype=float)
            if gravity is None
            else jnp.asarray(gravity, dtype=float)
        )
        external_force = jnp.zeros(shape, dtype=float)

        ang_dim = 1 if dim == 2 else 3
        external_torque = jnp.zeros(shape[:-1] + (ang_dim,), dtype=float)

        return ForceManager(
            gravity=gravity,
            external_force=external_force,
            external_torque=external_torque,
            force_functions=tuple(force_functions),
        )

    @staticmethod
    @partial(jax.named_call, name="ForceManager.add_force")
    def add_force(
        state: "State",
        system: "System",
        force: jax.Array,
        *,
        torque: Optional[jax.Array] = None,
    ) -> "System":
        """
        Accumulate an external force (and optionally torque) to be applied on the next ``apply`` call for all particles.

        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.
        force:
            Force contribution to accumulate. Must be a single ``(dim,)`` vector applied uniformly.
        torque:
            Optional torque contribution to accumulate. Must be broadcast-compatible with ``external_torque``.

        Returns
        -------
        System
            A new :class:`jaxdem.System` instance with the updated accumulated
            forces and torques.
        """
        force = jnp.asarray(force, dtype=float)
        system.force_manager.external_force += force
        if torque is not None:
            torque = jnp.asarray(torque, dtype=float)
            system.force_manager.external_torque += torque
        return system

    @staticmethod
    @partial(jax.named_call, name="ForceManager.add_force")
    def add_force_at(
        state: "State",
        system: "System",
        force: jax.Array,
        idx: jax.Array,
        *,
        torque: Optional[jax.Array] = None,
    ) -> "System":
        """
        Accumulate an external force (and optionally torque) to be applied on the next ``apply`` call over ``idx`` particles.

        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.
        force:
            Force contribution to accumulate. Must be an array broadcastable to ``external_force[idx]``.
        idx:
            integer indices of the particles receiving the contribution.
        torque:
            Optional torque contribution to accumulate. Must be an array
            broadcastable to ``external_torque[idx]``.

        Returns
        -------
        System
            A new :class:`jaxdem.System` instance with the updated accumulated
            forces and torques.
        """
        force = jnp.asarray(force, dtype=float)
        idx = jnp.asarray(idx, dtype=int)

        full_force = jnp.zeros_like(system.force_manager.external_force)
        full_force = full_force.at[idx].add(force)

        full_torque = None
        if torque is not None:
            torque = jnp.asarray(torque, dtype=float)
            full_torque = jnp.zeros_like(system.force_manager.external_torque)
            full_torque = full_torque.at[idx].add(torque)

        return ForceManager.add_force(state, system, full_force, torque=full_torque)

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="ForceManager.apply")
    def apply(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Overwrite ``state.force`` with managed per-particle contributions.

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

        Note
        -----
        - This method donates state and system
        """
        state.force = (
            system.force_manager.external_force
            + system.force_manager.gravity * state.mass[..., None]
        )
        state.torque = system.force_manager.external_torque

        if system.force_manager.force_functions:
            # per-function contributions: tuple of dicts [{"F": (...,N,dim), "T": (...,N,ang_dim)}, ...]
            contribs = jax.tree_util.tree_map(
                lambda func: (
                    # vmap over particles; each func returns (force, torque)
                    lambda F_T: {"F": F_T[0], "T": F_T[1]}
                )(
                    jax.vmap(lambda i: func(state, system, i))(
                        jax.lax.iota(dtype=int, size=state.N)
                    )
                ),
                tuple(system.force_manager.force_functions),
            )

            # sum across functions
            summed = jax.tree_util.tree_reduce(
                lambda a, b: jax.tree_util.tree_map(operator.add, a, b),
                contribs,
            )
            F_tot, T_tot = summed["F"], summed["T"]

            # forces
            state.force += F_tot

            # torques: inertia shape (..., N, 1) in 2D or (..., N, 3) in 3D
            state.torque += T_tot

        system.force_manager.external_force *= 0.0
        system.force_manager.external_torque *= 0.0
        return state, system


__all__ = ["ForceManager"]
