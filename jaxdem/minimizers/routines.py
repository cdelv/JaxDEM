# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Minimization routines and drivers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from functools import partial
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from ..forces.force_manager import default_energy_func
from ..utils.quaternion import Quaternion
from ..topology import BodyTopology, body_topology

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


class TerminationReason(IntEnum):
    """Reason an energy minimization stopped."""

    MAX_STEPS = 0
    ENERGY_TOLERANCE = 1
    ENERGY_CHANGE_TOLERANCE = 2
    FORCE_TOLERANCE = 3
    NONFINITE = 4
    SEARCH_OVERFLOW = 5


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class MinimizationResult:
    """Minimization output with a JIT-safe termination reason code.

    Iteration yields the historical four result values for compatibility.
    Inspect ``reason`` with :class:`TerminationReason` to distinguish success
    criteria from exhaustion of ``max_steps``.
    """

    state: State
    system: System
    steps: int | jax.Array
    energy: jax.Array
    reason: jax.Array

    def __iter__(self) -> Iterator[Any]:
        yield self.state
        yield self.system
        yield self.steps
        yield self.energy

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int | slice) -> Any:
        values = (self.state, self.system, self.steps, self.energy)
        return values[index]


def _evaluate_readonly_forces(state: State, system: System) -> tuple[State, System]:
    """Evaluate conservative forces without initialization or one-shot loads."""
    conservative_state = replace(
        state,
        vel=jnp.zeros_like(state.vel),
        ang_vel=jnp.zeros_like(state.ang_vel),
    )
    conservative_state, eval_system = system.collider.evaluate_force(
        conservative_state, system
    )

    force_manager = eval_system.force_manager
    empty_manager = replace(
        force_manager,
        external_force=jnp.zeros_like(force_manager.external_force),
        external_force_com=jnp.zeros_like(force_manager.external_force_com),
        external_torque=jnp.zeros_like(force_manager.external_torque),
    )
    eval_system = replace(eval_system, force_manager=empty_manager)
    conservative_state, eval_system = eval_system.force_manager.apply(
        conservative_state, eval_system
    )
    evaluated_state = replace(
        state,
        force=conservative_state.force,
        torque=conservative_state.torque,
    )
    return evaluated_state, replace(
        eval_system,
        force_manager=force_manager,
        search_overflow=eval_system.search_overflow | eval_system.collider.overflow,
    )


@jax.jit
def _state_to_delta_params(
    state: State, topology: BodyTopology | None = None
) -> dict[str, jax.Array]:
    """Pack positions and a zero rotation delta into a parameter dictionary.

    Returns
    -------
    dict
        A dictionary with keys 'pos_c' and 'rotvec' containing arrays.
    """
    topology = (
        body_topology(state.clump_id, state.fixed) if topology is None else topology
    )
    body_pos = topology.gather_representatives(state.pos_c)
    rot_dim = 1 if state.dim == 2 else 3
    zeros = jnp.zeros(body_pos.shape[:-1] + (rot_dim,), dtype=body_pos.dtype)
    return {"pos_c": body_pos, "rotvec": zeros}


@jax.jit
def _delta_params_to_state(
    state: State,
    params: dict[str, jax.Array],
    topology: BodyTopology | None = None,
) -> State:
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
    topology = (
        body_topology(state.clump_id, state.fixed) if topology is None else topology
    )
    pos_c = params["pos_c"]
    rotvec = params["rotvec"]
    if state.dim == 2:
        rotvec = jnp.concatenate(
            [jnp.zeros_like(pos_c), rotvec],
            axis=-1,
        )
    body_q = Quaternion.create(
        topology.gather_representatives(state.q.w),
        topology.gather_representatives(state.q.xyz),
    )
    q = Quaternion.from_rotvec(rotvec) @ body_q
    q = q.unit(q)
    member_q = Quaternion.create(
        topology.gather_members(q.w), topology.gather_members(q.xyz)
    )
    return replace(state, pos_c=topology.gather_members(pos_c), q=member_q)


@partial(jax.custom_vjp)
def _objective_energy(
    trial_params: dict[str, jax.Array],
    state: State,
    system: System,
    topology: BodyTopology | None = None,
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
    topology = (
        body_topology(state.clump_id, state.fixed) if topology is None else topology
    )
    trial_state = _delta_params_to_state(state, trial_params, topology)
    trial_state, eval_system, pe_collider = system.collider.compute_potential_energy(
        trial_state, system
    )
    pe = pe_collider + eval_system.force_manager.compute_potential_energy(
        trial_state, eval_system
    )
    return pe, (trial_state, eval_system)


def _objective_energy_fwd(
    trial_params: dict[str, jax.Array],
    state: State,
    system: System,
    topology: BodyTopology | None = None,
) -> tuple[
    tuple[jax.Array, tuple[State, System]],
    tuple[jax.Array, jax.Array],
]:
    topology = (
        body_topology(state.clump_id, state.fixed) if topology is None else topology
    )
    pe, (trial_state, eval_system) = _objective_energy(
        trial_params, state, system, topology
    )
    conservative_state, eval_system = _evaluate_readonly_forces(
        trial_state, eval_system
    )
    return (pe, (conservative_state, eval_system)), (
        -topology.gather_representatives(conservative_state.force),
        -topology.gather_representatives(conservative_state.torque),
    )


def _objective_energy_bwd(
    residual: tuple[jax.Array, jax.Array], g: tuple[jax.Array, Any]
) -> tuple[dict[str, jax.Array], None, None, None]:
    grad_pos, grad_rot = residual
    scale, _ = g
    return (
        {"pos_c": grad_pos * scale, "rotvec": grad_rot * scale},
        None,
        None,
        None,
    )


_objective_energy.defvjp(_objective_energy_fwd, _objective_energy_bwd)


@jax.jit(static_argnames=["max_steps"])
def minimize(
    state: State,
    system: System,
    max_steps: int = 10000,
    pe_tol: float = 1e-16,
    pe_diff_tol: float = 1e-16,
    force_tol: float = 0.0,
) -> MinimizationResult:
    r"""Minimize the energy of the system using the configured optax optimizer.

    This function runs a JAX-compatible optimization loop using the minimizer in
    `system.minimizer`. The function packs the positions and orientations into a
    parameter dictionary, optimizes them, and unpacks them into the returned
    `State`. The function re-anchors the rotation parameters at the current
    orientation each iteration (delta rotation vectors), so the
    torque-as-gradient identity stays exact regardless of the accumulated rotation.

    The loop performs exactly **one** force and energy evaluation per iteration,
    plus one initial evaluation. It carries the value and the gradient through
    the loop state.

    The default objective uses the zero-velocity analytical derivative of the
    built-in pair-law potential, which excludes constitutive damping and keeps
    dynamic collider line searches performant. Custom pair laws must provide a
    ``target_fn`` so JAX can differentiate their scalar objective. Likewise,
    every custom force-manager force used by the default objective must have a
    matching energy function.

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
    MinimizationResult
        An iterable result containing the historical four values:
        - The energy-minimized `State`.
        - The updated `System`.
        - The number of steps actually taken.
        - The final potential energy.
        Its ``reason`` field is a scalar code from :class:`TerminationReason`.
    """
    import optax  # type: ignore[import-untyped]

    if system.minimizer is None:
        raise ValueError(
            "No minimizer configured in System. Please configure `minimizer` in System.create."
        )
    if system.target_fn is None:
        if not system.force_model.supports_analytical_energy_gradient:
            raise ValueError(
                "Default minimization requires a force model with an analytical "
                "energy gradient; provide target_fn or explicitly opt in a "
                "conservative custom force model."
            )
        if any(
            energy_fn is default_energy_func
            for energy_fn in system.force_manager.energy_functions
        ):
            raise ValueError(
                "Every custom force used by default minimization must provide a "
                "matching energy function, or the system must provide target_fn."
            )

    N = state.N
    topology = body_topology(state.clump_id, state.fixed)
    body_free_mask = (topology.valid & ~topology.fixed)[..., None]

    def make_value_fn(anchor_state: State, anchor_system: System) -> Any:
        def value_fn(
            optim_params: dict[str, jax.Array],
        ) -> tuple[jax.Array, tuple[State, System]]:
            if anchor_system.target_fn is None:
                return _objective_energy(
                    optim_params, anchor_state, anchor_system, topology
                )
            else:
                trial_state = _delta_params_to_state(
                    anchor_state, optim_params, topology
                )
                pe = anchor_system.target_fn(trial_state, anchor_system)
                return pe, (trial_state, anchor_system)

        return value_fn

    def eval_step(
        anchor_state: State, anchor_system: System, params: dict[str, jax.Array]
    ) -> tuple[jax.Array, dict[str, jax.Array], State, System]:
        """Single force and energy evaluation that returns the value, the gradient, and the evaluated state."""
        value_fn = make_value_fn(anchor_state, anchor_system)
        (pe, (trial_state, eval_system)), grads = jax.value_and_grad(
            value_fn, has_aux=True
        )(params)
        return pe, grads, trial_state, eval_system

    params = _state_to_delta_params(state, topology)
    opt_state = system.minimizer.init(params)

    # Initial (and only per-iteration) force/energy evaluation.
    pe0, grads0, state0, system0 = eval_step(state, system, params)
    params0 = _state_to_delta_params(state0, topology)

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
        state, eval_system, step_count, pe, prev_pe, _, _, grads = carry
        pe_n = pe / N if system.target_fn is None else pe

        is_running = step_count < max_steps
        converged_pe = jnp.abs(pe_n) <= pe_tol
        # Relative energy change with a safe denominator (no NaN at pe == 0).
        denom = jnp.maximum(jnp.abs(pe), jnp.abs(prev_pe))
        denom = jnp.where(denom > 0, denom, jnp.ones_like(denom))
        converged_rel = jnp.abs(pe - prev_pe) / denom < pe_diff_tol
        free_grads = jax.tree.map(lambda x: x * body_free_mask, grads)
        max_grad = jnp.max(
            jnp.array(
                [jnp.max(jnp.abs(x), initial=0.0) for x in jax.tree.leaves(free_grads)]
            ),
            initial=0.0,
        )
        converged_force = max_grad <= force_tol
        finite = jnp.isfinite(pe) & jnp.all(
            jnp.array(
                [jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads)],
                dtype=bool,
            )
        )
        overflow = eval_system.search_overflow | eval_system.collider.overflow
        return (
            is_running
            & finite
            & ~overflow
            & ~(converged_pe | converged_rel | converged_force)
        )

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

        mask = body_free_mask
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
        next_params = _state_to_delta_params(new_state, topology)

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

    final_state, final_system, steps, final_pe, prev_pe, _, _, final_grads = (
        jax.lax.while_loop(cond_fun, body_fun, init_carry)
    )
    pe_n = final_pe / N if system.target_fn is None else final_pe
    denom = jnp.maximum(jnp.abs(final_pe), jnp.abs(prev_pe))
    denom = jnp.where(denom > 0, denom, jnp.ones_like(denom))
    free_grads = jax.tree.map(lambda x: x * body_free_mask, final_grads)
    max_grad = jnp.max(
        jnp.array(
            [jnp.max(jnp.abs(x), initial=0.0) for x in jax.tree.leaves(free_grads)]
        ),
        initial=0.0,
    )
    reason = jnp.select(
        [
            final_system.search_overflow | final_system.collider.overflow,
            ~jnp.isfinite(final_pe)
            | ~jnp.all(
                jnp.array(
                    [jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(final_grads)],
                    dtype=bool,
                )
            ),
            jnp.abs(pe_n) <= pe_tol,
            jnp.abs(final_pe - prev_pe) / denom < pe_diff_tol,
            max_grad <= force_tol,
        ],
        [
            int(TerminationReason.SEARCH_OVERFLOW),
            int(TerminationReason.NONFINITE),
            int(TerminationReason.ENERGY_TOLERANCE),
            int(TerminationReason.ENERGY_CHANGE_TOLERANCE),
            int(TerminationReason.FORCE_TOLERANCE),
        ],
        default=int(TerminationReason.MAX_STEPS),
    )
    # Report forces for the accepted configuration without evolving contact
    # history or consuming one-shot loads.
    final_state, final_system = _evaluate_readonly_forces(final_state, final_system)
    if system.target_fn is None:
        final_pe = final_pe / N
    return MinimizationResult(final_state, final_system, steps, final_pe, reason)
