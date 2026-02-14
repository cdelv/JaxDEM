# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Minimization routines and drivers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import TYPE_CHECKING, Tuple, cast

from ..utils import thermal

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@partial(jax.jit, static_argnames=["max_steps", "initialize"])
def minimize(
    state: State,
    system: System,
    max_steps: int = 10000,
    pe_tol: float = 1e-16,
    pe_diff_tol: float = 1e-16,
    initialize: bool = True,
) -> Tuple[State, System, int, float]:
    """
    Minimize the energy of the system until either of the following conditions are met:
    1. step_count >= max_steps
    2. PE <= PE_tol (Energy is low enough) and |PE / prev_PE - 1| < pe_diff_tol (Energy stopped changing)
    Parameters
    ----------
    state : State
        The state of the system.
    system : System
        The system to minimize.
    max_steps : int, optional
        The maximum number of steps to take.
    pe_tol : float, optional
        The tolerance for the potential energy.
    pe_diff_tol : float, optional
        The tolerance for the difference in potential energy.
    initialize : bool, optional
        Whether to initialize the integrator before minimizing.
    Returns
    -------
    Tuple[State, System, int, float]
        The final state, system, number of steps, and potential energy.

    Notes
    -----
    - The potential energy is computed using the `compute_potential_energy` method of the `collider` object.
    - The `step` method of the `system` object is used to take a single step in the minimization.
    - The `jax.lax.while_loop` function is used to take steps until the conditions are met.
    - The `jax.jit` function is used to compile the minimization routine.
    """

    if initialize:  # not sure i like this
        state, system = system.linear_integrator.initialize(state, system)
        state, system = system.rotation_integrator.initialize(state, system)

    N = state.N  # TODO: change this to the number of clumps
    initial_pe = 1e9
    init_carry = (state, system, 0, initial_pe, jnp.inf)

    def cond_fun(
        carry: Tuple[State, System, int, float, float],
    ) -> jax.Array:  # TODO: change this to a custom condition class
        state, system, step_count, pe, prev_pe = carry
        is_running = step_count < max_steps
        not_minimized = pe > pe_tol
        not_stable = jnp.abs(pe / prev_pe - 1.0) >= pe_diff_tol
        return is_running * not_minimized * not_stable

    def body_fun(
        carry: Tuple[State, System, int, float, float],
    ) -> Tuple[State, System, int, float, float]:
        state, system, step_count, pe, _ = carry
        prev_pe = pe
        state, system = system.step(state, system, n=1)
        pe_force_manager = system.force_manager.compute_potential_energy(state, system)
        pe_collider = system.collider.compute_potential_energy(state, system)
        new_pe = cast(float, jnp.sum(pe_force_manager + pe_collider) / N)
        return state, system, step_count + 1, new_pe, prev_pe

    final_state, final_system, steps, final_pe, _ = jax.lax.while_loop(
        cond_fun, body_fun, init_carry
    )
    return final_state, final_system, steps, final_pe
