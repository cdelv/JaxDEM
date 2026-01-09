# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Utilities for managing external and custom force contributions that dont deppend on the collider."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence, Tuple, Optional, Union
from functools import partial

from ..utils.linalg import cross

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


# Updated Signature: (state, system) -> (Force, Torque)
# Returns arrays of shape (N, dim) and (N, ang_dim)
ForceFunction = Callable[["State", "System"], Tuple[jax.Array, jax.Array]]


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ForceManager:  # type: ignore[misc]
    """
    Manage per-particle force contributions prior to pairwise interactions.
    """

    gravity: jax.Array
    """
    Constant acceleration applied to all particles. Shape ``(dim,)``.
    """

    external_force: jax.Array
    """
    Accumulated external force applied to all particles (at particle position).
    This buffer is cleared when :meth:`apply` is invoked.
    """

    external_force_com: jax.Array
    """
    Accumulated external force applied to Center of Mass (does not induce torque).
    This buffer is cleared when :meth:`apply` is invoked.
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
    Tuple of callables with signature ``(state, system)`` returning
    per-particle force and torque arrays.
    """

    is_com_force: Tuple[bool, ...] = field(default=(), metadata={"static": True})
    """
    Tuple of booleans corresponding to ``force_functions``.
    If True, the force is applied to the Center of Mass (no induced torque).
    If False, the force is applied to the constituent particle (induces torque via lever arm).
    """

    @staticmethod
    @partial(jax.named_call, name="ForceManager.create")
    def create(
        state_shape: Tuple[int, ...],
        *,
        gravity: jax.Array | None = None,
        # Allow passing (Func, bool) or just Func
        force_functions: Sequence[
            Union[ForceFunction, Tuple[ForceFunction, bool]]
        ] = (),
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
            Sequence of callables or tuples (callable, bool).

            - If a callable is provided, it is assumed to be applied at the **Particle Position** (is_com=False).
            - If a tuple (callable, bool) is provided, the boolean determines if it is a COM force.

            Signature of callable: ``(state, system) -> (Force, Torque)`` for every particle in state.
        """
        dim = state_shape[-1]
        gravity = (
            jnp.zeros(dim, dtype=float)
            if gravity is None
            else jnp.asarray(gravity, dtype=float)
        )
        external_force = jnp.zeros(state_shape, dtype=float)
        external_force_com = jnp.zeros(state_shape, dtype=float)

        ang_dim = 1 if dim == 2 else 3
        external_torque = jnp.zeros(state_shape[:-1] + (ang_dim,), dtype=float)

        # Parse force functions
        funcs = []
        is_com = []
        for item in force_functions:
            if isinstance(item, tuple):
                funcs.append(item[0])
                is_com.append(item[1])
            else:
                funcs.append(item)
                is_com.append(False)  # Default: Apply at particle

        return ForceManager(
            gravity=gravity,
            external_force=external_force,
            external_force_com=external_force_com,
            external_torque=external_torque,
            force_functions=tuple(funcs),
            is_com_force=tuple(is_com),
        )

    @staticmethod
    @partial(jax.named_call, name="ForceManager.add_force")
    def add_force(
        state: "State",
        system: "System",
        force: jax.Array,
        *,
        torque: Optional[jax.Array] = None,
        is_com: bool = False,
    ) -> "System":
        """
        Accumulate an external force (and optionally torque) to be applied on the next ``apply`` call for all particles.

        Parameters
        ----------
        is_com : bool, optional
            If True, force is applied to Center of Mass (no induced torque).
            If False (default), force is applied to Particle Position (induces torque).
        """
        force = jnp.asarray(force, dtype=float)
        system.force_manager.external_force_com += force * is_com
        system.force_manager.external_force += force * (1 - is_com)
        if torque is not None:
            torque = jnp.asarray(torque, dtype=float)
            system.force_manager.external_torque += torque
        return system

    @staticmethod
    @partial(jax.named_call, name="ForceManager.add_force_at")
    def add_force_at(
        state: "State",
        system: "System",
        force: jax.Array,
        idx: jax.Array,
        *,
        torque: Optional[jax.Array] = None,
        is_com: bool = False,
    ) -> "System":
        """
        Accumulate an external force at specific indices.

        Parameters
        ----------
        is_com : bool, optional
            If True, force is applied to Center of Mass (no induced torque).
            If False (default), force is applied to Particle Position (induces torque).
        """
        force = jnp.asarray(force, dtype=float)
        idx = jnp.asarray(idx, dtype=int)

        # Create the full force contribution array
        delta_force = (
            jnp.zeros_like(system.force_manager.external_force).at[idx].add(force)
        )
        system.force_manager.external_force_com += delta_force * is_com
        system.force_manager.external_force += delta_force * (1.0 - is_com)

        if torque is not None:
            torque = jnp.asarray(torque, dtype=float)
            full_torque = jnp.zeros_like(system.force_manager.external_torque)
            full_torque = full_torque.at[idx].add(torque)
            system.force_manager.external_torque += full_torque

        return system

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="ForceManager.apply")
    def apply(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Overwrite ``state.force`` with managed per-particle contributions,
        correctly aggregating for rigid bodies (clumps).

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
        # 1. Initialize Accumulators
        # Particle Frame Accumulators (will induce torque via lever arm)
        F_part = system.force_manager.external_force
        T_total = system.force_manager.external_torque

        # COM Frame Accumulators (direct addition to Clump COM)
        # Gravity is naturally a COM force
        F_com = system.force_manager.external_force_com + (
            system.force_manager.gravity * state.mass[..., None]
        )

        # 2. Apply Force Functions using tree_map
        if system.force_manager.force_functions:

            def eval_force(
                func: ForceFunction, is_com: jax.Array
            ) -> Tuple[jax.Array, jax.Array, jax.Array]:
                f, t = func(state, system)
                zeros_f = jnp.zeros_like(f)

                if is_com:
                    return zeros_f, f, t
                else:
                    return f, zeros_f, t

            # Map over the tuple of functions to get a tuple of structures
            results = jax.tree_util.tree_map(
                eval_force,
                system.force_manager.force_functions,
                system.force_manager.is_com_force,
            )

            # Reduce (sum) across the tuple of results
            # We map 'sum' over the leaves of the unpacked results tuple
            fp, fc, tp = jax.tree_util.tree_map(lambda *args: sum(args), *results)

            F_part += fp
            F_com += fc
            T_total += tp

        # 3. Rigid Body Aggregation
        # A. Handle Particle Forces (Induce Torque)
        # Lever arm: Vector from Clump COM to Particle in World Frame
        # pos_p is in Principal Frame, so we rotate it to World Frame
        r_i = state.q.rotate(state.q, state.pos_p)

        # B. Combine with COM forces
        T_total += cross(r_i, F_part)

        # C. Segment Sum (Aggregate to Clump ID)
        # This sums contributions from all particles belonging to the same ID
        F_com = jax.ops.segment_sum(F_com + F_part, state.ID, num_segments=state.N)
        T_total = jax.ops.segment_sum(T_total, state.ID, num_segments=state.N)

        # 4. Update State
        # Broadcast aggregated clump forces back to all constituents
        state.force += F_com[state.ID]
        state.torque += T_total[state.ID]

        # 5. Clear External Buffers
        system.force_manager.external_force *= 0.0
        system.force_manager.external_force_com *= 0.0
        system.force_manager.external_torque *= 0.0

        return state, system


__all__ = ["ForceManager"]
