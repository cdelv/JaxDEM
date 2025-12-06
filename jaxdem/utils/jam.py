# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Jamming routines.
https://doi.org/10.1103/PhysRevE.68.011306
"""

from __future__ import annotations

import jax.numpy as jnp
from dataclasses import replace

from typing import TYPE_CHECKING, Tuple

from ..minimizers import minimize

if TYPE_CHECKING:
    from ..state import State
    from ..system import System

def jam(state: State, system: System, n_minimization_steps: int = 1000000, pe_tol: float = 1e-16, pe_diff_tol: float = 1e-16, n_jamming_steps: int = 1000, packing_fraction_tolerance: float = 1e-10, packing_fraction_increment: float = 1e-3) -> Tuple[State, System]:
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

    # initial minimization
    state, system, _, final_pe = minimize(state, system, max_steps=n_minimization_steps, pe_tol=pe_tol, pe_diff_tol=pe_diff_tol, initialize=True)
    if final_pe > pe_tol:
        raise ValueError("Initial state is jammed")
    
    packing_fraction = jnp.sum(state.volume) / jnp.prod(system.domain.box_size)

    # jamming loop
    dim = state.dim
    packing_fraction_low = packing_fraction
    packing_fraction_high = -1.0
    last_unjammed_state = state
    last_unjammed_system = system
    jamming_iteration = 0
    while jamming_iteration < n_jamming_steps:
        jamming_iteration += 1
        state, system, _, final_pe = minimize(state, system, max_steps=n_minimization_steps, pe_tol=pe_tol, pe_diff_tol=pe_diff_tol, initialize=False)
        if final_pe > pe_tol:  # jammed
            packing_fraction_high = packing_fraction
            packing_fraction = (packing_fraction_high + packing_fraction_low) / 2.0
            # revert to last unjammed state
            state = last_unjammed_state
            system = last_unjammed_system
            print(f"Jammed on iteration {jamming_iteration} with packing fraction {packing_fraction} and PE {final_pe}")
        else:  # unjammed
            # save the last unjammed state and system
            last_unjammed_state = state
            last_unjammed_system = system
            packing_fraction_low = packing_fraction
            if packing_fraction_high > 0:  # if we have found a jammed state, bisection search
                packing_fraction = (packing_fraction_high + packing_fraction_low) / 2.0
            else:  # increment packing fraction
                packing_fraction += packing_fraction_increment
            print(f"Unjammed on iteration {jamming_iteration} with packing fraction {packing_fraction} and PE {final_pe}")

        if (abs(packing_fraction_high / packing_fraction_low - 1) < packing_fraction_tolerance and packing_fraction_high > 0):
            break

        new_box_size = (jnp.sum(state.volume) / packing_fraction) ** (1 / dim)
        scale_factor = new_box_size / system.domain.box_size
        new_domain = replace(system.domain, box_size=jnp.ones(dim) * new_box_size)
        system = replace(system, domain=new_domain)
        state = replace(state, pos_c=state.pos_c * scale_factor)
    
    return state, system