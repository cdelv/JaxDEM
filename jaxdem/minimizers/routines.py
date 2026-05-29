# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Minimization routines and drivers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import TYPE_CHECKING, Any

from ..utils.thermal import compute_potential_energy
from .optimizers import _state_to_params, _params_to_state

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@partial(jax.custom_vjp)
def _objective_energy(
    trial_params: jax.Array,
    state: State,
    system: System,
) -> jax.Array:
    """Evaluate potential energy of the trial parameters."""
    trial_state = _params_to_state(state, trial_params)
    return compute_potential_energy(trial_state, system)


def _objective_energy_fwd(
    trial_params: jax.Array,
    state: State,
    system: System,
) -> tuple[jax.Array, tuple[jax.Array, State, System]]:
    value = _objective_energy(trial_params, state, system)
    return value, (trial_params, state, system)


def _objective_energy_bwd(
    res: tuple[jax.Array, State, System],
    g: jax.Array,
) -> tuple[jax.Array, None, None]:
    """Return analytical forces/torques as the gradient."""
    trial_params, state, system = res
    trial_state = _params_to_state(state, trial_params)

    trial_state, eval_system = system.collider.compute_force(trial_state, system)
    trial_state, eval_system = eval_system.force_manager.apply(trial_state, eval_system)

    grads = jnp.concatenate([-trial_state.force, -trial_state.torque], axis=-1)

    return (grads * g, None, None)


_objective_energy.defvjp(_objective_energy_fwd, _objective_energy_bwd)


@partial(jax.jit, static_argnames=["max_steps", "initialize"])
def minimize(
    state: State,
    system: System,
    max_steps: int = 10000,
    pe_tol: float = 1e-16,
    pe_diff_tol: float = 1e-16,
    initialize: bool = True,
) -> tuple[State, System, int, float]:
    r"""Minimize the energy of the system using the configured optax optimizer.

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
        Unused now (maintained for API backward compatibility).

    Returns
    -------
    Tuple[State, System, int, float]
        The final state, system, number of steps, and potential energy.

    """
    import optax  # type: ignore[import-untyped]

    if system.minimizer is None:
        raise ValueError(
            "No minimizer configured in System. Please configure `minimizer` in System.create."
        )

    params = _state_to_params(state)
    opt_state = system.minimizer.init(params)

    N = state.N
    if system.target_fn is None:
        initial_pe = compute_potential_energy(state, system) / N
    else:
        initial_pe = system.target_fn(state, system)

    init_carry = (state, system, 0, initial_pe, jnp.inf, params, opt_state)

    def cond_fun(
        carry: tuple[State, System, int, float, float, jax.Array, Any],
    ) -> jax.Array:
        _, _, step_count, pe, prev_pe, _, _ = carry
        is_running = step_count < max_steps
        not_minimized = pe > pe_tol
        not_stable = jnp.abs(pe / prev_pe - 1.0) >= pe_diff_tol
        return is_running & not_minimized & not_stable

    def body_fun(
        carry: tuple[State, System, int, float, float, jax.Array, Any],
    ) -> tuple[State, System, int, float, float, jax.Array, Any]:
        state, system, step_count, pe, _, params, opt_state = carry
        prev_pe = pe

        mask = ~state.fixed[..., None]

        def value_fn(optim_params: jax.Array) -> jax.Array:
            if system.target_fn is None:
                return _objective_energy(optim_params, state, system)
            else:
                trial_state = _params_to_state(state, optim_params)
                return system.target_fn(trial_state, system)

        pe_current, grads = jax.value_and_grad(value_fn)(params)
        grads *= mask

        updates, new_opt_state = system.minimizer.update(
            grads,
            opt_state,
            params,
            value=pe_current,
            grad=grads,
            value_fn=value_fn,
        )
        updates *= mask

        new_params = optax.apply_updates(params, updates)
        new_params = jnp.where(mask, new_params, params)

        new_state = _params_to_state(state, new_params)

        if system.target_fn is None:
            new_pe = compute_potential_energy(new_state, system) / N
        else:
            new_pe = system.target_fn(new_state, system)

        return (
            new_state,
            system,
            step_count + 1,
            new_pe,
            prev_pe,
            new_params,
            new_opt_state,
        )

    final_state, final_system, steps, final_pe, _, _, _ = jax.lax.while_loop(
        cond_fun, body_fun, init_carry
    )
    return final_state, final_system, steps, final_pe
