# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Minimization routines and drivers."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ..utils.quaternion import Quaternion
from ..utils.thermal import compute_potential_energy

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.jit
def _state_to_delta_params(state: State) -> dict[str, jax.Array]:
    """Extract anchored rotation delta and positions as a PyTree dictionary.

    Returns
    -------
    dict
        A dictionary with keys 'pos_c' and 'rotvec' containing arrays.
    """
    rot_dim = 1 if state.dim == 2 else 3
    zeros = jnp.zeros(state.pos_c.shape[:-1] + (rot_dim,), dtype=state.pos_c.dtype)
    return {"pos_c": state.pos_c, "rotvec": zeros}


@jax.jit
def _delta_params_to_state(state: State, params: dict[str, jax.Array]) -> State:
    """Unpack anchored parameter dictionary back to a state.

    The rotation block of ``params['rotvec']`` is interpreted as a delta rotation vector
    applied (left-multiplied) to the reference state's current orientation.

    Parameters
    ----------
    state : State
        The reference (anchor) state from which to copy fields.
    params : dict
        A dictionary with keys 'pos_c' and 'rotvec'.

    Returns
    -------
    State
        The updated simulation state.
    """
    pos_c = params["pos_c"]
    rotvec = params["rotvec"]
    if state.dim == 2:
        rotvec = jnp.concatenate(
            [jnp.zeros_like(pos_c), rotvec],
            axis=-1,
        )
    q = Quaternion.from_rotvec(rotvec) @ state.q
    return replace(state, pos_c=pos_c, q=q.unit(q))


@partial(jax.custom_vjp)
def _objective_energy(
    trial_params: dict[str, jax.Array],
    state: State,
    system: System,
) -> tuple[jax.Array, tuple[State, System]]:
    """Evaluate potential energy of the trial parameters.

    The rotation block of ``trial_params`` is interpreted as a delta rotation
    vector anchored at ``state.q`` (see ``_delta_params_to_state``).

    Parameters
    ----------
    trial_params : dict
        The dict of trial parameters containing 'pos_c' and 'rotvec'.
    state : State
        The simulation state (anchor for the rotation parameters).
    system : System
        The system configuration.

    Returns
    -------
    Tuple[jax.Array, Tuple[State, System]]
        A tuple containing the potential energy and a tuple of the evaluated State and System.
    """
    trial_state = _delta_params_to_state(state, trial_params)
    trial_state, eval_system = system.collider.compute_force(trial_state, system)
    trial_state, eval_system = eval_system.force_manager.apply(trial_state, eval_system)
    pe = compute_potential_energy(trial_state, eval_system)
    return pe, (trial_state, eval_system)


def _objective_energy_fwd(
    trial_params: dict[str, jax.Array],
    state: State,
    system: System,
) -> tuple[tuple[jax.Array, tuple[State, System]], tuple[State, State]]:
    pe, (trial_state, eval_system) = _objective_energy(trial_params, state, system)
    # The forward pass already evaluated forces/torques; carry the evaluated
    # trial state so the backward pass does not need a second force evaluation.
    return (pe, (trial_state, eval_system)), (trial_state, state)


def _objective_energy_bwd(
    res: tuple[State, State],
    g: tuple[jax.Array, Any],
) -> tuple[dict[str, jax.Array], None, None]:
    """Backward pass returning analytical forces/torques as the gradient.

    Reuses the forces/torques stored on the trial state evaluated in the
    forward pass (no force recomputation).

    Parameters
    ----------
    res : Tuple[State, State]
        The residuals from the forward pass: the evaluated trial state and the
        original (anchor) state.
    g : Tuple[jax.Array, Any]
        The incoming gradient from the VJP.

    Returns
    -------
    Tuple[dict, None, None]
        The gradient with respect to the parameters, and None for the state and system.
    """
    trial_state, state = res
    force = trial_state.force
    torque = trial_state.torque

    if state.dim == 2:
        torque = torque

    grads = {"pos_c": -force, "rotvec": -torque}

    g_val, _ = g
    grads = jax.tree.map(lambda x: x * g_val, grads)

    return (grads, None, None)


_objective_energy.defvjp(_objective_energy_fwd, _objective_energy_bwd)


@jax.jit(static_argnames=["max_steps"])
def minimize(
    state: State,
    system: System,
    max_steps: int = 10000,
    pe_tol: float = 1e-16,
    pe_diff_tol: float = 1e-16,
    force_tol: float = 0.0,
) -> tuple[State, System, int, float | jax.Array]:
    r"""Minimize the energy of the system using the configured optax optimizer.

    This function runs a JAX-compatible optimization loop using the minimizer specified
    in `system.minimizer`. The positions and orientations are packed into a flat parameter array,
    optimized, and then unpacked back to the returned `State`. The rotation parameters are
    re-anchored at the current orientation each iteration (delta rotation vectors), so the
    torque-as-gradient identity stays exact regardless of the accumulated rotation.

    Exactly **one** force/energy evaluation is performed per iteration (plus one initial
    evaluation); the value and gradient are carried through the loop state.

    The optimization loop terminates when any of the following conditions are met:

    1. The number of steps reaches `max_steps`.
    2. The magnitude of the potential energy per particle drops below `pe_tol`
       (or of the overall objective if `system.target_fn` is defined):
       :math:`|E_k| \le \text{pe\_tol}`.
    3. The relative change in potential energy between successive steps drops below
       `pe_diff_tol` (with a safe denominator, so a zero-energy state does not produce NaN):

       .. math::
           \frac{|E_k - E_{k-1}|}{\max(|E_k|, |E_{k-1}|, \epsilon)} < \text{pe\_diff\_tol}

    4. The maximum absolute gradient component (force/torque) drops to `force_tol` or
       below: :math:`\max_i |g_i| \le \text{force\_tol}`.

    Parameters
    ----------
    state : State
        The state of the system.
    system : System
        The system to minimize.
    max_steps : int, default 10000
        The maximum number of optimization steps to take.
    pe_tol : float, default 1e-16
        The absolute potential energy tolerance (applied to the magnitude, so
        negative-energy objectives such as Lennard-Jones do not exit prematurely).
    pe_diff_tol : float, default 1e-16
        The relative potential energy difference tolerance for convergence.
    force_tol : float, default 0.0
        Force-norm (max absolute gradient component) tolerance. The default of 0.0
        only triggers for an exactly force-free configuration.

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

    N = state.N

    def make_value_fn(anchor_state: State, anchor_system: System) -> Any:
        def value_fn(
            optim_params: dict[str, jax.Array],
        ) -> tuple[jax.Array, tuple[State, System]]:
            if anchor_system.target_fn is None:
                return _objective_energy(optim_params, anchor_state, anchor_system)
            else:
                trial_state = _delta_params_to_state(anchor_state, optim_params)
                pe = anchor_system.target_fn(trial_state, anchor_system)
                trial_state, eval_system = anchor_system.collider.compute_force(
                    trial_state, anchor_system
                )
                trial_state, eval_system = eval_system.force_manager.apply(
                    trial_state, eval_system
                )
                return pe, (trial_state, eval_system)

        return value_fn

    def eval_step(
        anchor_state: State, anchor_system: System, params: dict[str, jax.Array]
    ) -> tuple[jax.Array, dict[str, jax.Array], State, System]:
        """Single force/energy evaluation returning value, gradient and the evaluated state."""
        value_fn = make_value_fn(anchor_state, anchor_system)
        if anchor_system.target_fn is None:
            # The analytical gradient of the energy w.r.t. the (re-anchored)
            # parameters is just -[force, -torque] of the evaluated state, so no
            # second (autodiff) force evaluation is needed.
            pe, (trial_state, eval_system) = value_fn(params)
            torque = trial_state.torque
            grads = {"pos_c": -trial_state.force, "rotvec": -torque}
        else:
            (pe, (trial_state, eval_system)), grads = jax.value_and_grad(
                value_fn, has_aux=True
            )(params)
        return pe, grads, trial_state, eval_system

    params = _state_to_delta_params(state)
    opt_state = system.minimizer.init(params)

    # Initial (and only per-iteration) force/energy evaluation.
    pe0, grads0, state0, system0 = eval_step(state, system, params)
    params0 = _state_to_delta_params(state0)

    init_carry: tuple[
        State,
        System,
        int,
        jax.Array,
        float | jax.Array,
        dict[str, jax.Array],
        Any,
        dict[str, jax.Array],
    ] = (
        state0,
        system0,
        0,
        pe0,
        jnp.asarray(jnp.inf, dtype=jnp.asarray(pe0).dtype),
        params0,
        opt_state,
        grads0,
    )

    def cond_fun(
        carry: tuple[
            State,
            System,
            int,
            jax.Array,
            float | jax.Array,
            dict[str, jax.Array],
            Any,
            dict[str, jax.Array],
        ],
    ) -> jax.Array:
        _, _, step_count, pe, prev_pe, _, _, grads = carry
        pe_n = pe / N if system.target_fn is None else pe

        is_running = step_count < max_steps
        converged_pe = jnp.abs(pe_n) <= pe_tol
        # Relative energy change with a safe denominator (no NaN at pe == 0).
        denom = jnp.maximum(
            jnp.maximum(jnp.abs(pe), jnp.abs(prev_pe)),
            np.finfo(jnp.asarray(pe).dtype).tiny,
        )
        converged_rel = jnp.abs(pe - prev_pe) / denom < pe_diff_tol
        max_grad = jnp.max(
            jnp.array(
                [jnp.max(jnp.abs(x), initial=0.0) for x in jax.tree.leaves(grads)]
            ),
            initial=0.0,
        )
        converged_force = max_grad <= force_tol
        return is_running & ~(converged_pe | converged_rel | converged_force)

    def body_fun(
        carry: tuple[
            State,
            System,
            int,
            jax.Array,
            float | jax.Array,
            dict[str, jax.Array],
            Any,
            dict[str, jax.Array],
        ],
    ) -> tuple[
        State,
        System,
        int,
        jax.Array,
        float | jax.Array,
        dict[str, jax.Array],
        Any,
        dict[str, jax.Array],
    ]:
        state, system, step_count, pe, _, params, opt_state, grads = carry

        mask = ~state.fixed[..., None]
        grads = jax.tree.map(lambda x: x * mask, grads)

        # Line-search minimizers (e.g. conjugate gradient) call ``value_fn`` and
        # need a scalar objective; ``make_value_fn`` returns ``(value, aux)``, so
        # expose the value alone here. First-order minimizers (FIRE, damped
        # Newtonian) ignore ``value_fn`` entirely, so this is a no-op for them.
        vfn = make_value_fn(state, system)
        updates, new_opt_state = system.minimizer.update(
            grads,
            opt_state,
            params,
            value=pe,
            grad=grads,
            value_fn=lambda p, *args, **kw: vfn(p)[0],
        )
        updates = jax.tree.map(lambda x: x * mask, updates)

        new_params = optax.apply_updates(params, updates)
        new_params = jax.tree.map(
            lambda n, p: jnp.where(mask, n, p), new_params, params
        )

        new_pe, new_grads, new_state, new_system = eval_step(state, system, new_params)
        # Re-anchor: rotation parameters become a zero delta about the new
        # orientation; the gradient (-force/-torque) is exact at this anchor.
        next_params = _state_to_delta_params(new_state)

        return (
            new_state,
            new_system,
            step_count + 1,
            new_pe,
            pe,
            next_params,
            new_opt_state,
            new_grads,
        )

    final_state, final_system, steps, final_pe, _, _, _, _ = jax.lax.while_loop(
        cond_fun, body_fun, init_carry
    )
    if system.target_fn is None:
        final_pe = final_pe / N
    return final_state, final_system, steps, final_pe
