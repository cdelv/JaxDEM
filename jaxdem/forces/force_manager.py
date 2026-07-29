# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""External and custom force contributions that do not depend on the collider."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from ..utils.linalg import cross, dot

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


ForceFunction = Callable[[jax.Array, "State", "System"], tuple[jax.Array, jax.Array]]
EnergyFunction = Callable[[jax.Array, "State", "System"], jax.Array]


@jax.jit
def default_energy_func(pos: jax.Array, state: State, system: System) -> jax.Array:
    return jnp.array(0.0, dtype=float)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ForceManager:
    """Manage custom force contributions outside the collider.

    After the collider runs, :meth:`apply` adds these contributions to the
    state forces and aggregates them over rigid bodies.
    """

    gravity: jax.Array
    """
    Constant acceleration applied to all particles. Shape ``(dim,)``.
    """

    external_force: jax.Array
    """
    Accumulated external force applied to all particles (at particle position).
    :meth:`apply` clears this buffer.
    """

    external_force_com: jax.Array
    """
    Accumulated external force applied to the center of mass (induces no torque).
    :meth:`apply` clears this buffer.
    """

    external_torque: jax.Array
    """
    Accumulated external torque applied to all particles.
    :meth:`apply` clears this buffer.
    """

    is_com_force: tuple[bool, ...] = jax.tree.static(default=())
    """
    Boolean array corresponding to ``force_functions`` with shape ``(n_forces,)``.
    If True, apply the force to the center of mass (no induced torque).
    If False, apply the force at the particle position (induces torque through the lever arm).
    """

    force_functions: tuple[ForceFunction, ...] = jax.tree.static(default=())
    """
    Tuple of callables with signature ``(pos, state, system)`` returning
    per-particle force and torque arrays.
    """

    energy_functions: tuple[EnergyFunction | None, ...] = jax.tree.static(default=())
    """
    Tuple of callables (or None) with signature ``(pos, state, system)`` returning
    per-particle potential energy arrays. Corresponds to ``force_functions``.
    """

    @staticmethod
    @partial(jax.named_call, name="ForceManager.create")
    def create(
        state_shape: tuple[int, ...],
        *,
        gravity: jax.Array | None = None,
        # Allow passing ForceFunc, (ForceFunc, bool), (ForceFunc, EnergyFunc), or (ForceFunc, EnergyFunc, bool)
        force_functions: Sequence[
            ForceFunction
            | tuple[ForceFunction, bool]
            | tuple[ForceFunction, EnergyFunction | None]
            | tuple[ForceFunction, EnergyFunction | None, bool]
        ] = (),
    ) -> ForceManager:
        """Create a :class:`ForceManager` for a state with the given shape.

        Parameters
        ----------
        state_shape:
            Shape of the state position array, typically ``(..., dim)``.
        gravity:
            Optional initial gravitational acceleration. Defaults to zeros of shape ``(dim,)``.
        force_functions:
            Sequence of callables or tuples. Signature of ForceFunc:
            ``(pos, state, system) -> (Force, Torque)``. Signature of
            EnergyFunc: ``(pos, state, system) -> Energy``. Supported formats:

            - ``func``                  -> ``(func, None, False)``
            - ``(func,)``               -> ``(func, None, False)``
            - ``(func, bool)``          -> ``(func, None, bool)``
            - ``(func, energy)``        -> ``(func, energy, False)``
            - ``(func, energy, bool)``  -> ``(func, energy, bool)``
            - ``(func, None, bool)``    -> ``(func, None, bool)``

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
        external_torque = jnp.zeros((*state_shape[:-1], ang_dim), dtype=float)

        # Parse force functions
        funcs = []
        energies = []
        is_com = []

        for entry in force_functions:
            f_func: ForceFunction | None = None
            e_func: EnergyFunction | Any = default_energy_func
            com_flag: bool = False

            # Normalize to tuple if not already
            args: Sequence[Any] = (
                entry if isinstance(entry, (tuple, list)) else (entry,)
            )

            # Parse based on structure
            # Case 1: (func,) or func
            if len(args) == 1:
                f_func = args[0]

            # Case 2: (func, bool) or (func, energy)
            elif len(args) == 2:
                f_func = args[0]
                if isinstance(args[1], bool):
                    com_flag = args[1]
                else:
                    e_func = args[1] if args[1] is not None else default_energy_func

            # Case 3: (func, energy, bool)
            elif len(args) == 3:
                f_func, e_func_arg, com_flag = args
                e_func = e_func_arg if e_func_arg is not None else default_energy_func

            else:
                raise ValueError(
                    f"Force function entry has invalid length: {len(args)}. "
                    "Expected (Force,), (Force, Bool), (Force, Energy), or (Force, Energy, Bool)."
                )

            if not callable(f_func):
                raise TypeError(
                    f"First element must be a callable force function, got {type(f_func)}"
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
        state: State,
        system: System,
        force: jax.Array,
        *,
        is_com: bool = False,
    ) -> System:
        """Buffer an external force on all particles for the next ``apply`` call.

        The method returns only ``system``. The state does not change because
        the force waits in the :class:`ForceManager` buffer until :meth:`apply`
        runs.

        Parameters
        ----------
        state : State
            Current state of the simulation. Used to normalize COM forces by
            the clump member count.
        system : System
            Simulation system configuration.
        force : jax.Array
            External force to add to every particle (in the state's
            current particle order).
        is_com : bool, optional
            If True, apply the force to the center of mass (no induced
            torque). The force goes to every clump member, so each clump
            receives ``force`` in total, not ``force`` per member.
            If False (default), apply the force at the particle position
            (induces torque).

        """
        force = jnp.asarray(force, dtype=float)
        # COM writes are replicated on every clump member: store the
        # per-member share so ``apply`` can segment-sum without dividing.
        count = jnp.bincount(state.clump_id, length=state.N)[state.clump_id]
        system.force_manager.external_force_com += force * is_com / count[..., None]
        system.force_manager.external_force += force * (1 - is_com)
        return system

    @staticmethod
    @partial(jax.named_call, name="ForceManager.add_force_at")
    def add_force_at(
        state: State,
        system: System,
        force: jax.Array,
        idx: jax.Array,
        *,
        is_com: bool = False,
    ) -> System:
        """Buffer an external force on the particles with array index ``idx``
        for the next ``apply`` call.

        The method returns only ``system``. The state does not change because
        the force waits in the :class:`ForceManager` buffer until :meth:`apply`
        runs.

        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.
        force : jax.Array
            External force to add to the particles with array index ``idx``.
        idx : jax.Array
            Array indices of the particles the external force acts on.
        is_com : bool, optional
            If True, apply the force to the center of mass (no induced torque).
            If False (default), apply the force at the particle position
            (induces torque).

        """
        idx = jnp.asarray(idx, dtype=int)
        force = jnp.asarray(force, dtype=float)
        force = jnp.zeros_like(system.force_manager.external_force).at[idx].add(force)
        system.force_manager.external_force_com += force * is_com
        system.force_manager.external_force += force * (1.0 - is_com)
        return system

    @staticmethod
    @partial(jax.named_call, name="ForceManager.add_torque")
    def add_torque(
        state: State,
        system: System,
        torque: jax.Array,
    ) -> System:
        """Buffer an external torque on all particles for the next ``apply`` call.

        The method returns only ``system``. The state does not change because
        the torque waits in the :class:`ForceManager` buffer until :meth:`apply`
        runs.

        Parameters
        ----------
        state : State
            Current state of the simulation. Used to normalize the torque by
            the clump member count.
        system : System
            Simulation system configuration.
        torque : jax.Array
            External torque to add to every particle (in the state's
            current particle order). The torque goes to every clump member,
            so each clump receives ``torque`` in total, not ``torque`` per
            member.

        """
        torque = jnp.asarray(torque, dtype=float)
        # Replicated on every clump member: store the per-member share so
        # ``apply`` can segment-sum without dividing.
        count = jnp.bincount(state.clump_id, length=state.N)[state.clump_id]
        system.force_manager.external_torque += torque / count[..., None]
        return system

    @staticmethod
    @partial(jax.named_call, name="ForceManager.add_torque_at")
    def add_torque_at(
        state: State,
        system: System,
        torque: jax.Array,
        idx: jax.Array,
    ) -> System:
        """Buffer an external torque on the particles with array index ``idx``
        for the next ``apply`` call.

        The method returns only ``system``. The state does not change because
        the torque waits in the :class:`ForceManager` buffer until :meth:`apply`
        runs.

        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.
        torque : jax.Array
            External torque to add to the particles with array index ``idx``.
        idx : jax.Array
            Array indices of the particles the external torque acts on.

        """
        idx = jnp.asarray(idx, dtype=int)
        torque = jnp.asarray(torque, dtype=float)
        torque = (
            jnp.zeros_like(system.force_manager.external_torque).at[idx].add(torque)
        )
        system.force_manager.external_torque += torque
        return system

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="ForceManager.apply")
    def apply(state: State, system: System) -> tuple[State, System]:
        """Add the managed per-particle contributions to the collider forces,
        then aggregate over clumps and broadcast back to the members.

        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        Tuple[State, System]
            The updated state and system.

        """
        # 0. Start from collider/contact contributions (computed earlier in the step)
        F_contact = state.force
        T_contact = state.torque
        r_i = state._pos_p_rot

        # 1. Initialize accumulators for managed contributions
        # Particle-frame forces (applied at particle location; induce torque via lever arm)
        F_part = system.force_manager.external_force
        T_part = system.force_manager.external_torque

        # COM-frame forces (applied at clump COM; should not induce torque via lever arm)
        F_com = system.force_manager.external_force_com

        # 2. Apply Force Functions using tree_map
        if system.force_manager.force_functions:
            pos = state.pos_c + r_i

            def eval_force(
                func: ForceFunction, is_com: bool
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                f, t = func(pos, state, system)
                return (1.0 - is_com) * f, is_com * f, t

            results = jax.tree.map(
                eval_force,
                system.force_manager.force_functions,
                system.force_manager.is_com_force,
            )

            # Reduce (sum) across the tuple of results
            fp, fc, tp = jax.tree.map(lambda *args: sum(args), *results)
            F_part += fp
            F_com += fc
            T_part += tp

        # 3. Gravity (COM force). Every clump member stores the *total* clump
        # mass, so divide by the member count; the segment_sum below then
        # yields M_total * g per clump.
        count = jnp.bincount(state.clump_id, length=state.N)[state.clump_id]
        F_com += system.force_manager.gravity * (state.mass / count)[..., None]

        # 4. All accumulators now hold genuinely per-particle contributions
        # (clump-replicated writes — add_force(is_com=True), add_torque, and
        # gravity — were already normalized by the clump member count), so the
        # segment_sum below needs no further division.
        # Particle forces induce torque via lever arm (but collider/contact torques already include their own lever arms)
        T_part += cross(r_i, F_part)
        F_total = F_contact + F_part + F_com
        T_total = T_contact + T_part

        # 5. Final rigid-body aggregation and broadcast
        F_clump = jax.ops.segment_sum(F_total, state.clump_id, num_segments=state.N)
        T_clump = jax.ops.segment_sum(T_total, state.clump_id, num_segments=state.N)
        state.force = F_clump[state.clump_id]
        state.torque = T_clump[state.clump_id]

        # 6. Clear external buffers (zeros_like, not *= 0, so NaN/Inf do not persist)
        system.force_manager.external_force = jnp.zeros_like(
            system.force_manager.external_force
        )
        system.force_manager.external_force_com = jnp.zeros_like(
            system.force_manager.external_force_com
        )
        system.force_manager.external_torque = jnp.zeros_like(
            system.force_manager.external_torque
        )

        return state, system

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="ForceManager.compute_potential_energy")
    def compute_potential_energy(state: State, system: System) -> jax.Array:
        """Compute the total potential energy of the system.

        Notes
        -----
        - The energy of clump members is divided by the number of spheres in the clump.

        Parameters
        ----------
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        jax.Array
            Scalar total potential energy.

        """
        # 1. Gravitational Potential Energy
        # U = -M (g . r_com) per clump. Gravity is applied as a pure COM
        # force, so use the clump COM ``pos_c`` (identical on all members).
        # Every clump member stores the *total* clump mass, so divide by the
        # member count to avoid overcounting clump contributions.
        pos = state.pos
        count = jnp.bincount(state.clump_id, length=state.N)[state.clump_id]
        pe = -jnp.sum(
            dot(system.force_manager.gravity, state.pos_c) * state.mass / count
        )

        # 2. Custom Energy Functions
        if system.force_manager.energy_functions:

            def eval_energy(func: EnergyFunction | None, is_com: bool) -> jax.Array:
                if func is None:
                    return jnp.array(0.0, dtype=float)
                e = func(pos, state, system)
                return jnp.sum(e)

            # Evaluate all energy functions
            custom_energies = jax.tree.map(
                eval_energy,
                system.force_manager.energy_functions,
                system.force_manager.is_com_force,
            )
            pe_custom = jax.tree.map(lambda *args: sum(args), *custom_energies)
            pe += pe_custom

        return pe


__all__ = [
    "ForceManager",
]
