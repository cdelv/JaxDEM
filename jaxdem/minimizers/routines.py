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
) -> tuple[jax.Array, tuple[State, System]]:
    """Evaluate potential energy of the trial parameters.

    Parameters
    ----------
    trial_params : jax.Array
        The packed trial parameters of shape `(N, 3)` or `(N, 6)`.
    state : State
        The simulation state.
    system : System
        The system configuration.

    Returns
    -------
    Tuple[jax.Array, Tuple[State, System]]
        A tuple containing the potential energy and a tuple of the evaluated State and System.
    """
    trial_state = _params_to_state(state, trial_params)
    trial_state, eval_system = system.collider.compute_force(trial_state, system)
    trial_state, eval_system = eval_system.force_manager.apply(trial_state, eval_system)
    pe = compute_potential_energy(trial_state, eval_system)
    return pe, (trial_state, eval_system)


def _objective_energy_fwd(
    trial_params: jax.Array,
    state: State,
    system: System,
) -> tuple[tuple[jax.Array, tuple[State, System]], tuple[jax.Array, State, System]]:
    pe, (trial_state, eval_system) = _objective_energy(trial_params, state, system)
    return (pe, (trial_state, eval_system)), (trial_params, state, eval_system)


def _objective_energy_bwd(
    res: tuple[jax.Array, State, System],
    g: tuple[jax.Array, Any],
) -> tuple[jax.Array, None, None]:
    """Backward pass returning analytical forces/torques as the gradient.

    Parameters
    ----------
    res : Tuple[jax.Array, State, System]
        The residuals from the forward pass.
    g : Tuple[jax.Array, Any]
        The incoming gradient from the VJP.

    Returns
    -------
    Tuple[jax.Array, None, None]
        The gradient with respect to the parameters, and None for the state and system.
    """
    trial_params, state, eval_system = res
    trial_state = _params_to_state(state, trial_params)

    trial_state, eval_system = eval_system.collider.compute_force(
        trial_state, eval_system
    )
    trial_state, eval_system = eval_system.force_manager.apply(trial_state, eval_system)

    # Map forces and torques back to the original order of the input state particles
    sorted_indices = jnp.argsort(trial_state.unique_id)
    pos_in_sorted = jnp.searchsorted(
        trial_state.unique_id, state.unique_id, sorter=sorted_indices
    )
    original_to_sorted = sorted_indices[pos_in_sorted]

    unsorted_force = trial_state.force[original_to_sorted]
    unsorted_torque = trial_state.torque[original_to_sorted]

    grads = jnp.concatenate([-unsorted_force, -unsorted_torque], axis=-1)

    g_val, _ = g
    return (grads * g_val, None, None)


_objective_energy.defvjp(_objective_energy_fwd, _objective_energy_bwd)


@partial(jax.jit, static_argnames=["max_steps", "initialize"])
def minimize(
    state: State,
    system: System,
    max_steps: int = 10000,
    pe_tol: float = 1e-16,
    pe_diff_tol: float = 1e-16,
    initialize: bool = True,
) -> tuple[State, System, int, float | jax.Array]:
    r"""Minimize the energy of the system using the configured optax optimizer.

    This function runs a JAX-compatible optimization loop using the minimizer specified
    in `system.minimizer`. The positions and orientations are packed into a flat parameter array,
    optimized, and then unpacked back to the returned `State`.

    The optimization loop terminates when any of the following conditions are met:

    1. The number of steps reaches `max_steps`.
    2. The potential energy per particle drops below `pe_tol` (or overall energy if `system.target_fn` is defined).
    3. The relative change in potential energy between successive steps drops below `pe_diff_tol`:

       .. math::
           \left|\frac{E_k}{E_{k-1}} - 1\right| < \text{pe\_diff\_tol}

    Parameters
    ----------
    state : State
        The state of the system.
    system : System
        The system to minimize.
    max_steps : int, default 10000
        The maximum number of optimization steps to take.
    pe_tol : float, default 1e-16
        The absolute potential energy tolerance.
    pe_diff_tol : float, default 1e-16
        The relative potential energy difference tolerance for convergence.
    initialize : bool, default True
        Unused now (maintained for API backward compatibility).

    Returns
    -------
    Tuple[State, System, int, float | jax.Array]
        A tuple containing:
        - The energy-minimized `State`.
        - The updated `System`.
        - The number of steps actually taken.
        - The final potential energy.
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

    init_carry: tuple[
        State, System, int, float | jax.Array, float | jax.Array, jax.Array, Any
    ] = (state, system, 0, initial_pe, jnp.inf, params, opt_state)

    def cond_fun(
        carry: tuple[
            State, System, int, float | jax.Array, float | jax.Array, jax.Array, Any
        ],
    ) -> jax.Array:
        _, _, step_count, pe, prev_pe, _, _ = carry
        is_running = step_count < max_steps
        not_minimized = pe > pe_tol
        not_stable = jnp.abs(pe / prev_pe - 1.0) >= pe_diff_tol
        return is_running & not_minimized & not_stable

    def body_fun(
        carry: tuple[
            State, System, int, float | jax.Array, float | jax.Array, jax.Array, Any
        ],
    ) -> tuple[
        State, System, int, float | jax.Array, float | jax.Array, jax.Array, Any
    ]:
        state, system, step_count, pe, _, params, opt_state = carry
        prev_pe = pe

        mask = ~state.fixed[..., None]

        def value_fn(optim_params: jax.Array) -> tuple[jax.Array, tuple[State, System]]:
            if system.target_fn is None:
                return _objective_energy(optim_params, state, system)
            else:
                trial_state = _params_to_state(state, optim_params)
                pe = system.target_fn(trial_state, system)
                trial_state, eval_system = system.collider.compute_force(
                    trial_state, system
                )
                trial_state, eval_system = eval_system.force_manager.apply(
                    trial_state, eval_system
                )
                return pe, (trial_state, eval_system)

        (pe_current, (current_state, new_system)), grads = jax.value_and_grad(
            value_fn, has_aux=True
        )(params)
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

        new_pe, (new_state, new_system) = value_fn(new_params)
        if system.target_fn is None:
            new_pe = new_pe / N

        return (
            new_state,
            new_system,
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
