# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Utilities for managing external and custom force contributions that dont deppend on the collider."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence, Tuple, Optional, Union, Any
from functools import partial

from ..utils.linalg import cross

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


# Updated Signature: (state, system) -> (Force, Torque)
# Returns arrays of shape (N, dim) and (N, ang_dim)
ForceFunction = Callable[[jax.Array, "State", "System"], Tuple[jax.Array, jax.Array]]
EnergyFunction = Callable[[jax.Array, "State", "System"], jax.Array]


@jax.jit
def default_energy_func(pos: jax.Array, state: State, system: System) -> jax.Array:
    return jnp.zeros_like(state.mass)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ForceManager:  # type: ignore[misc]
    """
    Manage per-particle force contributions prior to pairwise interactions.
    It also resets the accumulated forces in the state after application.
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

    is_com_force: Tuple[bool, ...] = field(default=(), metadata={"static": True})
    """
    Tuple of booleans corresponding to ``force_functions``.
    If True, the force is applied to the Center of Mass (no induced torque).
    If False, the force is applied to the constituent particle (induces torque via lever arm).
    """

    force_functions: Tuple[ForceFunction, ...] = field(
        default=(), metadata={"static": True}
    )
    """
    Tuple of callables with signature ``(pos, state, system)`` returning
    per-particle force and torque arrays.
    """

    energy_functions: Tuple[Optional[EnergyFunction], ...] = field(
        default=(), metadata={"static": True}
    )
    """
    Tuple of callables (or None) with signature ``(pos, state, system)`` returning
    per-particle potential energy arrays. Corresponds to ``force_functions``.
    """

    @staticmethod
    @partial(jax.named_call, name="ForceManager.create")
    def create(
        state_shape: Tuple[int, ...],
        *,
        gravity: jax.Array | None = None,
        # Allow passing ForceFunc, (ForceFunc, bool), (ForceFunc, EnergyFunc), or (ForceFunc, EnergyFunc, bool)
        force_functions: Sequence[
            Union[
                ForceFunction,
                Tuple[ForceFunction, bool],
                Tuple[ForceFunction, EnergyFunction],
                Tuple[ForceFunction, EnergyFunction, bool],
            ]
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
            Sequence of callables or tuples. Supported formats:

            - ``ForceFunc``: Applied at particle, no potential energy.
            - ``(ForceFunc, bool)``: Boolean specifies if it is a COM force.
            - ``(ForceFunc, EnergyFunc)``: Includes potential energy function.
            - ``(ForceFunc, EnergyFunc, bool)``: Includes energy and COM specifier.

            Signature of ForceFunc: ``(pos, state, system) -> (Force, Torque)``
            Signature of EnergyFunc: ``(pos, state, system) -> Energy``

            Supported formats for force_functions items:
            - func                  -> (func, None, False)
            - (func,)               -> (func, None, False)
            - (func, bool)          -> (func, None, bool)
            - (func, energy)        -> (func, energy, False)
            - (func, energy, bool)  -> (func, energy, bool)
            - (func, None, bool)    -> (func, None, bool)
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
        energies = []
        is_com = []

        # RENAME loop variable to 'entry' or 'raw_item' to avoid conflict
        for entry in force_functions:
            f_func = None
            e_func: Union[EnergyFunction, Any] = default_energy_func
            com_flag = False  # Default

            # 1. Normalize input to a unified sequence type
            # We use 'args' to hold the tuple version.
            # We hint as Sequence[Any] to allow flexible indexing logic below without casting.
            args: Sequence[Any]
            if not isinstance(entry, (tuple, list)):
                args = (entry,)
            else:
                args = entry

            # 2. Extract Force Function (Always 1st arg)
            f_func = args[0]

            # 3. Handle Variadic Arguments based on Length
            if len(args) == 1:
                # Format: (Force,)
                pass
            elif len(args) == 2:
                # Format: (Force, Bool) OR (Force, Energy)
                second_arg = args[1]
                if isinstance(second_arg, bool):
                    com_flag = second_arg
                else:
                    if second_arg is not None:
                        e_func = second_arg

            elif len(args) == 3:
                # Format: (Force, Energy, Bool)
                if args[1] is not None:
                    e_func = args[1]
                com_flag = args[2]
            else:
                raise ValueError(
                    f"Force function entry has invalid length: {len(args)}"
                )

            funcs.append(f_func)
            energies.append(e_func)
            is_com.append(com_flag)

        return ForceManager(
            gravity=gravity,
            external_force=external_force,
            external_force_com=external_force_com,
            external_torque=external_torque,
            force_functions=tuple(funcs),
            energy_functions=tuple(energies),
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
        idx = jnp.asarray(idx, dtype=int)

        force = jnp.asarray(force, dtype=float)
        force = jnp.zeros_like(system.force_manager.external_force).at[idx].add(force)
        system.force_manager.external_force_com += force * is_com
        system.force_manager.external_force += force * (1.0 - is_com)

        if torque is not None:
            torque = jnp.asarray(torque, dtype=float)
            torque = (
                jnp.zeros_like(system.force_manager.external_torque).at[idx].add(torque)
            )
            system.force_manager.external_torque += torque

        return system

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="ForceManager.apply")
    def apply(state: "State", system: "System") -> Tuple["State", "System"]:
        """
        Accumulate managed per-particle contributions on top of collider/contact forces,
        then perform final clump aggregation + broadcast.

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
        # 0. Start from collider/contact contributions (computed earlier in the step)
        F_contact = state.force
        T_contact = state.torque

        # Per-sphere lever arm in world frame
        r_i = state.q.rotate(state.q, state.pos_p)

        # 1. Initialize accumulators for managed contributions
        # Particle-frame forces (applied at particle location; induce torque via lever arm)
        F_part = system.force_manager.external_force
        T_part = system.force_manager.external_torque

        # COM-frame forces (applied at clump COM; should not induce torque via lever arm)
        F_com = system.force_manager.external_force_com

        # 2. Apply Force Functions using tree_map (Vicsek can be appended last by convention)
        if system.force_manager.force_functions:
            pos = state.pos_c + r_i

            def eval_force(
                func: ForceFunction, is_com: jax.Array
            ) -> Tuple[jax.Array, jax.Array, jax.Array]:
                f, t = func(pos, state, system)
                return (1 - is_com) * f, is_com * f, t

            # Map over the tuple of functions to get a tuple of structures
            results = jax.tree_util.tree_map(
                eval_force,
                system.force_manager.force_functions,
                system.force_manager.is_com_force,
            )

            # Reduce (sum) across the tuple of results
            # We map 'sum' over the leaves of the unpacked results tuple
            fp, fc, tp = jax.tree_util.tree_map(lambda *args: sum(args), *results)

            F_part = F_part + fp
            F_com = F_com + fc
            T_part = T_part + tp

        # 3. Gravity (COM force)
        F_com = F_com + system.force_manager.gravity * state.mass[..., None]

        # 4. Compose per-sphere totals in a way that matches the previous pipeline:
        # - COM forces are divided by count so that a later segment_sum yields the correct clump force.
        # - Managed torques are also divided by count so that a later segment_sum yields the correct clump torque.
        count = jnp.bincount(state.clump_ID, length=state.N)[state.clump_ID]
        count_f = count.astype(F_contact.dtype)[..., None]

        # Particle forces induce torque via lever arm (but collider/contact torques already include their own lever arms)
        T_part = T_part + cross(r_i, F_part)

        F_total = F_contact + F_part + (F_com / count_f)
        T_total = T_contact + (T_part / count_f)

        # 5. Final rigid-body aggregation and broadcast
        F_clump = jax.ops.segment_sum(F_total, state.clump_ID, num_segments=state.N)
        T_clump = jax.ops.segment_sum(T_total, state.clump_ID, num_segments=state.N)
        state.force = F_clump[state.clump_ID]
        state.torque = T_clump[state.clump_ID]

        # 6. Clear external buffers
        system.force_manager.external_force *= 0.0
        system.force_manager.external_force_com *= 0.0
        system.force_manager.external_torque *= 0.0

        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="ForceManager.compute_potential_energy")
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        """
        Compute the total potential energy of the system.

        Notes
        ------
        - The energy of clump members is divided by the number of spheres in the clump.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        jax.Array
            A scalar JAX array representing the total potential energy of each particle.
        """
        # 1. Gravitational Potential Energy
        # U = -m (g . r)
        r_i = state.q.rotate(state.q, state.pos_p)
        pos = state.pos_c + r_i
        pe_grav = -jnp.sum(system.force_manager.gravity * pos, axis=-1) * state.mass

        # 2. Custom Energy Functions
        if not system.force_manager.energy_functions:
            return pe_grav

        count = jnp.bincount(state.clump_ID, length=state.N)[state.clump_ID]

        def eval_energy(func: Optional[EnergyFunction], is_com: bool) -> jax.Array:
            if func is None:
                return jnp.zeros_like(state.mass)

            e = func(pos, state, system)

            # If force was applied to COM, distribute energy across constituents
            return jnp.where(is_com, e / count, e)

        # Evaluate all energy functions
        custom_energies = jax.tree_util.tree_map(
            eval_energy,
            system.force_manager.energy_functions,
            system.force_manager.is_com_force,
        )

        # Sum contributions
        if isinstance(custom_energies, (list, tuple)):
            pe_custom = sum(custom_energies)
        else:
            pe_custom = custom_energies

        return pe_grav + pe_custom


__all__ = [
    "ForceManager",
]
