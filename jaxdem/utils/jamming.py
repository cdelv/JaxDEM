# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Jamming routines.
https://doi.org/10.1103/PhysRevE.68.011306
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import replace
from functools import partial

from typing import TYPE_CHECKING, Tuple

from ..minimizers import minimize

if TYPE_CHECKING:
    from ..state import State
    from ..system import System

@partial(jax.jit, static_argnames=["n_minimization_steps", "n_jamming_steps"])
def bisection_jam(state: State, system: System, n_minimization_steps: int = 1000000, pe_tol: float = 1e-16, pe_diff_tol: float = 1e-16, n_jamming_steps: int = 10000, packing_fraction_tolerance: float = 1e-10, packing_fraction_increment: float = 1e-3) -> Tuple[State, System]:
    """
    Find the nearest jammed state for a given state and system.
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
        The number of steps in the jamming loop.  Typically 1e3.
    packing_fraction_tolerance : float, optional
        The tolerance for the packing fraction to determine convergence.  Typically 1e-10
    packing_fraction_increment : float, optional
        The initial increment for the packing fraction.  Typically 1e-3.  Larger increments make it faster in the unjammed region, but makes minimization of the earliest detected jammed states take much longer.
    Returns
    -------
    Tuple[State, System]
        The jammed state and system.
    """

    # cannot proceed if the initial state is jammed
    state, system, n_steps, final_pe = minimize(state, system, max_steps=n_minimization_steps, pe_tol=pe_tol, pe_diff_tol=pe_diff_tol, initialize=True)
    is_initially_jammed = final_pe > pe_tol
    def print_warning():
        jax.debug.print("Warning: Initial state is already jammed (PE={pe} > tol={tol}). Skipping.", pe=final_pe, tol=pe_tol)
        return None
    jax.lax.cond(is_initially_jammed, print_warning, lambda: None)
    jax.debug.print("Initial minimization took {n_steps} steps.", n_steps=n_steps)
    dim = state.dim
    initial_packing_fraction = jnp.sum(state.volume) / jnp.prod(system.domain.box_size)
    init_carry = (
        0,                                   # iteration
        is_initially_jammed,                 # is_jammed
        state, system,                       # current state/system
        state, system,                       # last unjammed state/system
        initial_packing_fraction,            # current packing fraction
        initial_packing_fraction,            # low packing fraction
        -1.0,                                # high packing fraction (initially set to -1.0)
        final_pe,                            # final potential energy
    )

    def cond_fun(carry):
        i, is_jammed, _, _, _, _, _, _, _, _ = carry
        return (i < n_jamming_steps) & (~is_jammed)

    def body_fun(carry):
        (i, _, state, system, last_state, last_system, pf, pf_low, pf_high, _) = carry

        # minimize the state
        state, system, n_steps, final_pe = minimize(
            state, system, max_steps=n_minimization_steps, pe_tol=pe_tol, pe_diff_tol=pe_diff_tol, initialize=True
        )

        is_jammed = final_pe > pe_tol

        def jammed_branch(_):  # if jammed, revert to last unjammed state and bisect
            new_pf_high = pf
            new_pf = (new_pf_high + pf_low) / 2.0
            return new_pf, pf_low, new_pf_high, last_state, last_system, last_state, last_system

        def unjammed_branch(_):  # if unjammed, save current as last unjammed, increment or bisect
            new_last_state = state
            new_last_system = system
            new_pf_low = pf
            
            def bisect():  # if a jammed state is known, perform a bisection search
                return (pf_high + new_pf_low) / 2.0
            def increment():  # if no jammed state is known, increment the packing fraction
                return new_pf_low + packing_fraction_increment
            
            new_pf = jax.lax.cond(pf_high > 0, bisect, increment)
            return new_pf, new_pf_low, pf_high, state, system, new_last_state, new_last_system

        new_pf, new_pf_low, new_pf_high, new_state, new_system, new_last_state, new_last_system = jax.lax.cond(
            is_jammed, jammed_branch, unjammed_branch, operand=None
        )

        # check if the packing fraction is converged and print
        ratio = new_pf_high / new_pf_low
        is_jammed = (jnp.abs(ratio - 1.0) < packing_fraction_tolerance) & (new_pf_high > 0)
        # jax.debug.print("Step: {i} -  phi={pf}, PE={pe}", i=i+1, pf=pf, pe=final_pe)
        jax.debug.print("Step: {i} -  phi={pf}, PE={pe} after {n_steps} steps", i=i+1, pf=pf, pe=final_pe, n_steps=n_steps)
        
        # scale the box and positions
        new_box_size_scalar = (jnp.sum(new_state.volume) / new_pf) ** (1 / dim)
        current_box_L = new_system.domain.box_size[0]
        scale_factor = new_box_size_scalar / current_box_L
        
        new_box_size = jnp.ones_like(new_system.domain.box_size) * new_box_size_scalar
        new_domain = replace(new_system.domain, box_size=new_box_size)
        
        next_system = replace(new_system, domain=new_domain)
        next_state = replace(new_state, pos_c=new_state.pos_c * scale_factor)

        return (i + 1, is_jammed, next_state, next_system, new_last_state, new_last_system, new_pf, new_pf_low, new_pf_high, final_pe)

    final_carry = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    (_, _, _, _, last_state, last_system, final_pf, _, _, final_pe) = final_carry
    return last_state, last_system, final_pf, final_pe