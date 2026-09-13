# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Minimization routines and drivers."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..utils.quaternion import Quaternion
from ..utils.thermal import compute_potential_energy

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


CONVERGED = 0
MAX_STEPS = 2
NONFINITE = 3
COLLIDER_OVERFLOW = 4


class MinimizeInfo(NamedTuple):
    """Mechanical diagnostics; status 0=converged, 2=step limit,
    3=nonfinite, 4=collider overflow.
    """

    converged: jax.Array
    finite: jax.Array
    force_max: jax.Array
    torque_max: jax.Array
    status: jax.Array


@jax.jit
def convergence_info(force, torque, fixed, force_tol, torque_tol) -> MinimizeInfo:
    """Check free-body force and torque norms using already evaluated forces."""
    f = jnp.linalg.norm(jnp.where(fixed[..., None], 0.0, force), axis=-1)
    t = jnp.linalg.norm(jnp.where(fixed[..., None], 0.0, torque), axis=-1)
    fmax = jnp.max(f, initial=0.0)
    tmax = jnp.max(t, initial=0.0)
    finite = jnp.isfinite(fmax) & jnp.isfinite(tmax)
    converged = finite & jnp.all(f <= force_tol) & jnp.all(t <= torque_tol)
    return MinimizeInfo(
        converged,
        finite,
        fmax,
        tmax,
        jnp.where(~finite, NONFINITE, jnp.where(converged, CONVERGED, MAX_STEPS)),
    )


@jax.jit
def _state_to_delta_params(state: State) -> dict[str, jax.Array]:
    """Pack positions and a zero rotation delta into a parameter dictionary.

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
    """Unpack an anchored parameter dictionary back into a state.

    The rotation block of ``params['rotvec']`` is a delta rotation vector that
    the function applies (left-multiplies) to the current orientation of the
    reference state.

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
    """Evaluate the potential energy of the trial parameters.

    The rotation block of ``trial_params`` is a delta rotation vector anchored
    at ``state.q`` (see ``_delta_params_to_state``).

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
    """Backward pass that returns the analytical forces and torques as the gradient.

    Reuses the forces and torques stored on the trial state from the forward
    pass (no force recomputation).

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


@jax.jit(static_argnames=["max_steps", "return_info"])
def minimize(
    state: State,
    system: System,
    max_steps: int = 10000,
    force_tol: float = 1e-12,
    torque_tol: float | None = None,
    *,
    return_info: bool = False,
) -> (
    tuple[State, System, int, float | jax.Array]
    | tuple[State, System, int, float | jax.Array, MinimizeInfo]
):
    r"""Minimize the energy of the system using the configured optax optimizer.

    This function runs a JAX-compatible optimization loop using the minimizer in
    `system.minimizer`. The function packs the positions and orientations into a
    parameter dictionary, optimizes them, and unpacks them into the returned
    `State`. The function re-anchors the rotation parameters at the current
    orientation each iteration (delta rotation vectors), so the
    torque-as-gradient identity stays exact regardless of the accumulated rotation.

    FIRE and damped Newtonian relaxation evaluate forces/torques once per step,
    plus one initial evaluation. Their physical potential energy is evaluated
    once after relaxation, for the returned energy. Line-search and other
    optimizers retain objective evaluations required by their update interface;
    custom targets retain automatic differentiation of the objective.

    Mechanical convergence requires BOTH maximum free-body force and torque
    norms to pass their respective absolute tolerances. ``torque_tol=None``
    uses the numerical value of ``force_tol``; specify it explicitly when the
    force and torque units/scales differ. These reductions reuse the evaluated
    gradients: no additional force calculation or host synchronization occurs.
    Fixed bodies do not contribute to the convergence norms.

    ``return_info=True`` appends a :class:`MinimizeInfo` to the historical
    four-tuple ``(state, system, steps, energy)``. Check ``info.converged`` before
    accepting a mechanically equilibrated state: hitting ``max_steps`` or
    encountering a nonfinite value or collider overflow is not convergence.
    Collider overflow stops relaxation immediately with status 4; retry from
    a valid state with sufficient collider capacity. The reported energy
    remains per constituent sphere (or the unnormalized custom objective).
    Tolerances are in the user's units; for a relative residual target, choose
    them from the relevant contact-force and particle-length scales.
    """
    import optax  # type: ignore[import-untyped]

    if system.minimizer is None:
        raise ValueError(
            "No minimizer configured in System. Please configure `minimizer` in System.create."
        )

    from .optimizers import CustomGradientTransformation, damped_newtonian, fire

    torque_tol = force_tol if torque_tol is None else torque_tol
    force_only = (
        system.target_fn is None
        and isinstance(system.minimizer, CustomGradientTransformation)
        and system.minimizer._constructor in (fire, damped_newtonian)
    )

    def make_value_fn(anchor_state: State, anchor_system: System) -> Any:
        def value_fn(params: dict[str, jax.Array]) -> Any:
            if anchor_system.target_fn is None:
                return _objective_energy(params, anchor_state, anchor_system)
            trial_state = _delta_params_to_state(anchor_state, params)
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
    ) -> tuple[Any, dict[str, jax.Array], State, System]:
        if anchor_system.target_fn is None:
            trial_state = _delta_params_to_state(anchor_state, params)
            trial_state, eval_system = anchor_system.collider.compute_force(
                trial_state, anchor_system
            )
            trial_state, eval_system = eval_system.force_manager.apply(
                trial_state, eval_system
            )
            grads = {"pos_c": -trial_state.force, "rotvec": -trial_state.torque}
            pe = (
                None
                if force_only
                else compute_potential_energy(trial_state, eval_system)
            )
        else:
            (pe, (trial_state, eval_system)), grads = jax.value_and_grad(
                make_value_fn(anchor_state, anchor_system), has_aux=True
            )(params)
        return pe, grads, trial_state, eval_system

    params = _state_to_delta_params(state)
    opt_state = system.minimizer.init(params)
    pe0, grads0, state0, system0 = eval_step(state, system, params)
    init_carry = (
        state0,
        system0,
        0,
        pe0,
        _state_to_delta_params(state0),
        opt_state,
        grads0,
    )

    def cond_fun(carry: tuple[Any, ...]) -> jax.Array:
        cur_state, cur_system, step_count, pe, _, _, grads = carry
        info = convergence_info(
            grads["pos_c"], grads["rotvec"], cur_state.fixed, force_tol, torque_tol
        )
        finite = info.finite if force_only else info.finite & jnp.isfinite(pe)
        return (
            (step_count < max_steps) & finite & ~info.converged
            & ~cur_system.collider.overflow
        )

    def body_fun(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        state, system, step_count, pe, params, opt_state, grads = carry
        mask = ~state.fixed[..., None]
        grads = jax.tree.map(lambda x: jnp.where(mask, x, 0.0), grads)
        if force_only:
            updates, new_opt_state = system.minimizer.update(grads, opt_state, params)
        else:
            vfn = make_value_fn(state, system)
            updates, new_opt_state = system.minimizer.update(
                grads,
                opt_state,
                params,
                value=pe,
                grad=grads,
                value_fn=lambda p, *args, **kw: vfn(p)[0],
            )
        updates = jax.tree.map(lambda x: jnp.where(mask, x, 0.0), updates)
        new_params = optax.apply_updates(params, updates)
        new_params = jax.tree.map(
            lambda n, p: jnp.where(mask, n, p), new_params, params
        )
        new_pe, new_grads, new_state, new_system = eval_step(state, system, new_params)
        # Re-anchor the rotation coordinates so force/torque remain exact gradients.
        return (
            new_state,
            new_system,
            step_count + 1,
            new_pe,
            _state_to_delta_params(new_state),
            new_opt_state,
            new_grads,
        )

    final_state, final_system, steps, final_pe, _, _, grads = jax.lax.while_loop(
        cond_fun, body_fun, init_carry
    )
    if force_only:
        final_pe = compute_potential_energy(final_state, final_system)
    if system.target_fn is None:
        final_pe = final_pe / final_state.N
    info = convergence_info(
        grads["pos_c"], grads["rotvec"], final_state.fixed, force_tol, torque_tol
    )
    # Validate the reported objective once, including on the force-only path.
    finite = info.finite & jnp.isfinite(final_pe)
    overflow = final_system.collider.overflow
    converged = info.converged & finite & ~overflow
    info = info._replace(
        finite=finite,
        converged=converged,
        status=jnp.where(
            overflow, COLLIDER_OVERFLOW,
            jnp.where(~finite, NONFINITE, jnp.where(converged, CONVERGED, MAX_STEPS)),
        ),
    )
    result = (final_state, final_system, steps, final_pe)
    return (*result, info) if return_info else result
