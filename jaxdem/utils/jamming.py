# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Jamming routines.
https://doi.org/10.1103/PhysRevE.68.011306.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial

from typing import TYPE_CHECKING, Any, NamedTuple

from .packing_utils import (
    _host_body_grouping,
    _scale_to_packing_fraction_grouped,
    compute_packing_fraction,
)

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


class JamResult(NamedTuple):
    """Result of :func:`bisection_jam`.

    Behaves like the historical 6-tuple (same field order), but the named
    fields make the intent explicit at the call site, e.g.
    ``result.jammed_state`` instead of ``result[2]``.
    """

    unjammed_state: "State"
    """Last *unjammed* state visited by the bisection."""
    unjammed_system: "System"
    """System matching :attr:`unjammed_state`."""
    jammed_state: "State"
    """The jammed state (usually what you want)."""
    jammed_system: "System"
    """System matching :attr:`jammed_state`."""
    packing_fraction: jax.Array
    """Packing fraction of the jammed state."""
    potential_energy: jax.Array
    """Per-particle potential energy of the jammed state."""


@partial(
    jax.jit, static_argnames=["n_minimization_steps", "n_jamming_steps", "verbose"]
)
def bisection_jam(
    state: State,
    system: System,
    n_minimization_steps: int = 1000000,
    pe_tol: float = 1e-16,
    pe_diff_tol: float = 1e-16,
    n_jamming_steps: int = 10000,
    packing_fraction_tolerance: float = 1e-10,
    packing_fraction_increment: float = 1e-3,
    verbose: bool = True,
) -> JamResult:
    """Find the nearest jammed state for a given state and system.
    Uses bisection search with state reversion.

    Parameters
    ----------
    state : State
        The state to jam.
    system : System
        The system to jam.
    n_minimization_steps : int, optional
        The number of steps to take in the minimization.  Should be large.  Typically 1e6.
    pe_tol : float, optional
        The tolerance for the potential energy.  Should be very small.  Typically 1e-16.
    pe_diff_tol : float, optional
        The tolerance for the difference in potential energy across subsequent steps.  Should be very small.  Typically 1e-16.
    n_jamming_steps : int, optional
        The number of steps in the jamming loop.  Typically 1e4.
    packing_fraction_tolerance : float, optional
        The tolerance for the packing fraction to determine convergence.  Typically 1e-10
    packing_fraction_increment : float, optional
        The initial increment for the packing fraction.  Typically 1e-3.  Larger increments make it faster in the unjammed region, but makes minimization of the earliest detected jammed states take much longer.
    verbose : bool, optional
        If ``True`` (default), print per-iteration progress via
        ``jax.debug.print``. Set to ``False`` to silence the prints and avoid
        the per-iteration host callbacks they incur.

    Returns
    -------
    JamResult
        A named tuple ``(unjammed_state, unjammed_system, jammed_state,
        jammed_system, packing_fraction, potential_energy)``; unpacking it
        like the historical 6-tuple keeps working.

    """
    # cannot proceed if the initial state is jammed
    state, system, n_steps, final_pe = system.minimize(
        state,
        system,
        max_steps=n_minimization_steps,
        pe_tol=pe_tol,
        pe_diff_tol=pe_diff_tol,
    )
    is_initially_jammed = final_pe > pe_tol

    def print_warning() -> None:
        jax.debug.print(
            "Warning: Initial state is already jammed (PE={pe} > tol={tol}). Skipping.",
            pe=final_pe,
            tol=pe_tol,
        )
        return

    if verbose:
        jax.lax.cond(is_initially_jammed, print_warning, lambda: None)
        jax.debug.print("Initial minimization took {n_steps} steps.", n_steps=n_steps)
    initial_packing_fraction = compute_packing_fraction(state, system)

    # Body grouping depends only on the (static) bond/clump topology: pay the
    # host callback once here instead of once per bisection iteration (which
    # would force a host round-trip per loop step and break async dispatch).
    group_id = jax.pure_callback(
        _host_body_grouping,
        jax.ShapeDtypeStruct((state.N,), jnp.int32),  # type: ignore[no-untyped-call]
        state.clump_id,
        state.bond_id,
        vmap_method="sequential",
    )

    init_carry = (
        0,  # iteration
        is_initially_jammed,  # is_jammed
        state,
        system,  # current state/system
        state,
        system,  # last unjammed state/system
        initial_packing_fraction,  # current packing fraction
        initial_packing_fraction,  # low packing fraction
        -1.0,  # high packing fraction (initially set to -1.0)
        final_pe,  # final potential energy
    )

    def cond_fun(carry: tuple[Any, ...]) -> jax.Array:
        i, is_jammed, _, _, _, _, _, _, _, _ = carry
        return (i < n_jamming_steps) & (~is_jammed)

    def body_fun(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        i, _, state, system, last_state, last_system, pf, pf_low, pf_high, _ = carry

        # minimize the state
        state, system, n_steps, final_pe = system.minimize(
            state,
            system,
            max_steps=n_minimization_steps,
            pe_tol=pe_tol,
            pe_diff_tol=pe_diff_tol,
        )

        is_jammed = final_pe > pe_tol

        def jammed_branch(_: None) -> tuple[Any, ...]:
            new_pf_high = pf
            new_pf = (new_pf_high + pf_low) / 2.0
            return (
                new_pf,
                pf_low,
                new_pf_high,
                last_state,
                last_system,
                last_state,
                last_system,
            )

        def unjammed_branch(
            _: None,
        ) -> tuple[
            Any, ...
        ]:  # if unjammed, save current as last unjammed, increment or bisect
            new_last_state = state
            new_last_system = system
            new_pf_low = pf

            def bisect() -> (
                jax.Array
            ):  # if a jammed state is known, perform a bisection search
                return (pf_high + new_pf_low) / 2.0

            def increment() -> (
                jax.Array
            ):  # if no jammed state is known, increment the packing fraction
                return new_pf_low + packing_fraction_increment

            new_pf = jax.lax.cond(pf_high > 0, bisect, increment)
            return (
                new_pf,
                new_pf_low,
                pf_high,
                state,
                system,
                new_last_state,
                new_last_system,
            )

        (
            new_pf,
            new_pf_low,
            new_pf_high,
            new_state,
            new_system,
            new_last_state,
            new_last_system,
        ) = jax.lax.cond(is_jammed, jammed_branch, unjammed_branch, operand=None)

        # check if the packing fraction is converged and print
        ratio = new_pf_high / new_pf_low
        is_jammed = (jnp.abs(ratio - 1.0) < packing_fraction_tolerance) & (
            new_pf_high > 0
        )
        if verbose:
            jax.debug.print(
                "Step: {i} -  phi={pf}, PE={pe} after {n_steps} steps",
                i=i + 1,
                pf=pf,
                pe=final_pe,
                n_steps=n_steps,
            )

        next_state, next_system = _scale_to_packing_fraction_grouped(
            new_state, new_system, new_pf, group_id
        )

        return (
            i + 1,
            is_jammed,
            next_state,
            next_system,
            new_last_state,
            new_last_system,
            new_pf,
            new_pf_low,
            new_pf_high,
            final_pe,
        )

    final_carry = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    _, _, _, _, last_state, last_system, final_pf, _, pf_high, final_pe = final_carry
    last_jammed_pf = jnp.where(pf_high > 0, pf_high, final_pf)
    last_jammed_state, last_jammed_system = _scale_to_packing_fraction_grouped(
        last_state, last_system, last_jammed_pf, group_id
    )
    last_jammed_state, last_jammed_system, _, final_pe = last_jammed_system.minimize(
        last_jammed_state,
        last_jammed_system,
        max_steps=n_minimization_steps,
        pe_tol=pe_tol,
        pe_diff_tol=pe_diff_tol,
    )
    return JamResult(
        unjammed_state=last_state,
        unjammed_system=last_system,
        jammed_state=last_jammed_state,
        jammed_system=last_jammed_system,
        packing_fraction=final_pf,
        potential_energy=final_pe,
    )
