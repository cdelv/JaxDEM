# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Minimization routines and drivers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import IntEnum
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp

from ..forces.force_manager import default_energy_func
from ..topology import BodyTopology, body_topology
from ..utils.quaternion import Quaternion

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


class TerminationReason(IntEnum):
    """Reason an energy minimization stopped."""

    MAX_STEPS = 0
    # Legacy energy-based termination codes.
    ENERGY_TOLERANCE = 1
    ENERGY_CHANGE_TOLERANCE = 2
    FORCE_TOLERANCE = 3
    FORCE_TORQUE_TOLERANCE = 3
    NONFINITE = 4
    SEARCH_OVERFLOW = 5


CONVERGED = 0
MAX_STEPS = 2
NONFINITE = 3
COLLIDER_OVERFLOW = 4


class MinimizeInfo(NamedTuple):
    """Residual norms and convergence status for a mechanical relaxation.

    Attributes
    ----------
    converged : jax.Array
        Boolean scalar indicating that both residual norms meet their
        tolerances and the evaluation is valid.
    finite : jax.Array
        Boolean scalar indicating finite residual norms. Results from
        :func:`minimize` also require finite objective gradients and energy.
    force_max : jax.Array
        Maximum Euclidean force norm over free bodies. For a custom target,
        this is the maximum norm of its gradient with respect to body position.
    torque_max : jax.Array
        Maximum Euclidean torque norm over free bodies. For a custom target,
        this is the maximum norm of its gradient with respect to body rotation.
    status : jax.Array
        Integer scalar describing convergence:

        * 0 (``CONVERGED``): both tolerances are satisfied.
        * 2 (``MAX_STEPS``): residuals exceed tolerance; a completed
          minimization exhausted its step budget.
        * 3 (``NONFINITE``): an evaluated quantity is nonfinite.
        * 4 (``COLLIDER_OVERFLOW``): spatial search results are incomplete.
    reason : jax.Array
        Property mapping ``status`` to a :class:`TerminationReason` code.
        The two sets of integer codes differ.

    Notes
    -----
    Fields are scalar arrays for one system. Applying ``jax.vmap`` to the
    evaluation adds a batch axis to each field.
    """

    converged: jax.Array
    finite: jax.Array
    force_max: jax.Array
    torque_max: jax.Array
    status: jax.Array

    @property
    def reason(self) -> jax.Array:
        """Translate the diagnostic status to the public termination enum."""
        return jnp.select(
            [
                self.status == COLLIDER_OVERFLOW,
                self.status == NONFINITE,
                self.converged,
            ],
            [
                int(TerminationReason.SEARCH_OVERFLOW),
                int(TerminationReason.NONFINITE),
                int(TerminationReason.FORCE_TORQUE_TOLERANCE),
            ],
            default=int(TerminationReason.MAX_STEPS),
        )


@jax.jit
def convergence_info(
    force: jax.Array,
    torque: jax.Array,
    fixed: jax.Array,
    force_tol: float | jax.Array,
    torque_tol: float | jax.Array,
) -> MinimizeInfo:
    """Evaluate force and torque convergence for one set of bodies.

    Parameters
    ----------
    force : jax.Array
        Total force on each body, with shape ``(n_bodies, dim)``.
    torque : jax.Array
        Total torque on each body, with shape ``(n_bodies, 1)`` in two
        dimensions or ``(n_bodies, 3)`` in three dimensions.
    fixed : jax.Array
        Boolean mask with shape ``(n_bodies,)``. Entries marked ``True``
        are excluded from both residual norms, including nonfinite entries.
    force_tol : float or jax.Array
        Scalar absolute tolerance on the maximum body force norm.
    torque_tol : float or jax.Array
        Scalar absolute tolerance on the maximum body torque norm.

    Returns
    -------
    MinimizeInfo
        Maximum Euclidean force and torque norms, their finiteness, and
        convergence status. Convergence requires both norms to be finite,
        ``force_max <= force_tol``, and ``torque_max <= torque_tol``.

    Notes
    -----
    Empty or fully fixed body sets have zero residual norms. Finite residuals
    exceeding either tolerance produce ``MAX_STEPS``; nonfinite residuals
    produce ``NONFINITE``. This function evaluates residuals only; optimizer
    iteration counts and spatial search overflow are handled by :func:`minimize`.
    Use ``jax.vmap`` to evaluate independent systems.
    """
    force_max = jnp.max(
        jnp.linalg.norm(jnp.where(fixed[..., None], 0.0, force), axis=-1), initial=0.0
    )
    torque_max = jnp.max(
        jnp.linalg.norm(jnp.where(fixed[..., None], 0.0, torque), axis=-1), initial=0.0
    )
    finite = jnp.isfinite(force_max) & jnp.isfinite(torque_max)
    converged = finite & (force_max <= force_tol) & (torque_max <= torque_tol)
    return MinimizeInfo(
        converged,
        finite,
        force_max,
        torque_max,
        jnp.where(~finite, NONFINITE, jnp.where(converged, CONVERGED, MAX_STEPS)),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class MinimizationResult:
    """Final configuration, objective value, and minimization diagnostics.

    Attributes
    ----------
    state : State
        Final configuration with evaluated conservative forces and torques.
    system : System
        System associated with ``state``, including updated spatial search
        caches and overflow status.
    steps : int or jax.Array
        Number of optimizer updates performed. The initial objective or force
        evaluation is excluded from this count.
    energy : jax.Array
        Final potential energy per constituent sphere for the physical
        objective, or the unnormalized value of a custom ``target_fn``.
    reason : jax.Array
        Integer :class:`TerminationReason` code identifying convergence,
        step exhaustion, a nonfinite evaluation, or spatial search overflow.
    info : MinimizeInfo
        Force and torque residuals, finiteness, and convergence status.
    converged : jax.Array
        Boolean property equal to ``info.converged``.

    Notes
    -----
    Supports ``jax.jit`` and ``jax.vmap``. Iteration and indexing expose
    ``(state, system, steps, energy)``.
    """

    state: State
    system: System
    steps: int | jax.Array
    energy: jax.Array
    reason: jax.Array
    info: MinimizeInfo

    @property
    def converged(self) -> jax.Array:
        return self.info.converged

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


def _evaluate_energy(state: State, system: System) -> tuple[State, System, jax.Array]:
    """Return physical energy, updated search caches, and overflow status."""
    state, system, pair_energy = system.collider.compute_potential_energy(state, system)
    energy = pair_energy + system.force_manager.compute_potential_energy(state, system)
    system = replace(
        system, search_overflow=system.search_overflow | system.collider.overflow
    )
    return state, system, energy


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
    trial_state, eval_system, pe = _evaluate_energy(trial_state, system)
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
    force_tol: float = 1e-12,
    torque_tol: float | None = None,
) -> MinimizationResult:
    """Minimize the configured objective over body positions and orientations.

    Parameters
    ----------
    state : State
        Initial configuration for a single system. Forces and torques are
        evaluated at the initial configuration before any optimizer update.
    system : System
        System containing the optimizer and force models. The objective is
        physical potential energy unless ``system.target_fn`` is set.
    max_steps : int, optional
        Maximum number of optimizer updates. Defaults to 10000.
    force_tol : float, optional
        Absolute tolerance on the maximum Euclidean force norm over free
        bodies. Defaults to ``1e-12``.
    torque_tol : float or None, optional
        Absolute tolerance on the maximum Euclidean torque norm over free
        bodies. ``None`` uses the numerical value of ``force_tol``.

    Returns
    -------
    MinimizationResult
        Final state, system, update count, energy, and convergence diagnostics.
        Iteration yields ``(state, system, steps, energy)``. Physical energy
        is reported per constituent sphere; custom target values are
        unnormalized.

    Raises
    ------
    ValueError
        If no optimizer is configured, or if the physical objective uses a
        force model without an analytical energy gradient or a managed force
        without a matching energy function.

    Notes
    -----
    Convergence requires both ``force_max <= force_tol`` and
    ``torque_max <= torque_tol`` with finite evaluations and complete spatial
    searches. Fixed bodies are excluded from the residual norms. Custom
    targets use their translational and rotational objective gradients as
    residuals. Energy magnitude and energy changes do not determine
    convergence. Step exhaustion, nonfinite evaluations, and spatial search
    overflow terminate the minimization unsuccessfully.

    For the physical objective, FIRE and damped Newtonian evaluate conservative
    forces initially and after each update, and energy once at exit.
    Line-search optimizers and custom targets evaluate the objective during
    optimization.

    Each rigid body has one set of optimization coordinates. Rotation
    increments are anchored to the current orientation after each update.
    Force evaluation holds contact history fixed and excludes damping and
    queued loads. Velocities, queued loads, time, and integrator state are
    unchanged by relaxation.
    """
    import optax  # type: ignore[import-untyped]

    from .optimizers import CustomGradientTransformation, damped_newtonian, fire

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

    torque_tol = force_tol if torque_tol is None else torque_tol
    topology = body_topology(state.clump_id, state.fixed)
    fixed = ~topology.valid | topology.fixed
    mask = ~fixed[..., None]
    force_only = (
        system.target_fn is None
        and isinstance(system.minimizer, CustomGradientTransformation)
        and system.minimizer._constructor in (fire, damped_newtonian)
    )

    def make_value_fn(anchor_state: State, anchor_system: System) -> Any:
        def value_fn(params: dict[str, jax.Array]) -> Any:
            if anchor_system.target_fn is None:
                return _objective_energy(params, anchor_state, anchor_system, topology)
            trial = _delta_params_to_state(anchor_state, params, topology)
            return anchor_system.target_fn(trial, anchor_system), (trial, anchor_system)

        return value_fn

    def eval_step(
        anchor_state: State, anchor_system: System, params: dict[str, jax.Array]
    ) -> tuple[Any, dict[str, jax.Array], State, System]:
        if force_only:
            trial = _delta_params_to_state(anchor_state, params, topology)
            trial, evaluated = _evaluate_readonly_forces(trial, anchor_system)
            grads = {
                "pos_c": -topology.gather_representatives(trial.force),
                "rotvec": -topology.gather_representatives(trial.torque),
            }
            return None, grads, trial, evaluated
        (pe, (trial, evaluated)), grads = jax.value_and_grad(
            make_value_fn(anchor_state, anchor_system), has_aux=True
        )(params)
        return pe, grads, trial, evaluated

    def diagnostics(
        grads: dict[str, jax.Array], pe: Any, evaluated: System
    ) -> MinimizeInfo:
        info = convergence_info(
            grads["pos_c"], grads["rotvec"], fixed, force_tol, torque_tol
        )
        finite = info.finite & jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads)])
        )
        if pe is not None:
            finite = finite & jnp.isfinite(pe)
        overflow = evaluated.search_overflow | evaluated.collider.overflow
        converged = info.converged & finite & ~overflow
        return info._replace(
            finite=finite,
            converged=converged,
            status=jnp.where(
                overflow,
                COLLIDER_OVERFLOW,
                jnp.where(
                    ~finite, NONFINITE, jnp.where(converged, CONVERGED, MAX_STEPS)
                ),
            ),
        )

    params = _state_to_delta_params(state, topology)
    opt_state = system.minimizer.init(params)
    pe, grads, state, system = eval_step(state, system, params)
    carry = (
        state,
        system,
        0,
        pe,
        _state_to_delta_params(state, topology),
        opt_state,
        grads,
    )

    def condition(c: tuple[Any, ...]) -> jax.Array:
        _, evaluated, steps, pe, _, _, grads = c
        info = diagnostics(grads, pe, evaluated)
        return (steps < max_steps) & (info.status == MAX_STEPS)

    def step(c: tuple[Any, ...]) -> tuple[Any, ...]:
        current, evaluated, steps, pe, params, opt_state, grads = c
        free_grads = jax.tree.map(lambda x: jnp.where(mask, x, 0.0), grads)
        if force_only:
            updates, opt_state = evaluated.minimizer.update(
                free_grads, opt_state, params
            )
        else:
            value_fn = make_value_fn(current, evaluated)
            updates, opt_state = evaluated.minimizer.update(
                free_grads,
                opt_state,
                params,
                value=pe,
                grad=free_grads,
                value_fn=lambda p, *args, **kw: value_fn(p)[0],
            )
        updates = jax.tree.map(lambda x: jnp.where(mask, x, 0.0), updates)
        next_params = optax.apply_updates(params, updates)
        next_params = jax.tree.map(
            lambda n, p: jnp.where(mask, n, p), next_params, params
        )
        pe, grads, current, evaluated = eval_step(current, evaluated, next_params)
        return (
            current,
            evaluated,
            steps + 1,
            pe,
            _state_to_delta_params(current, topology),
            opt_state,
            grads,
        )

    state, system, steps, pe, _, _, grads = jax.lax.while_loop(condition, step, carry)
    if force_only:
        state, system, pe = _evaluate_energy(state, system)
    elif system.target_fn is not None:
        # Custom objective gradients need not have evaluated physical forces.
        state, system = _evaluate_readonly_forces(state, system)
    if system.target_fn is None:
        pe = pe / max(state.N, 1)
    info = diagnostics(grads, pe, system)
    return MinimizationResult(state, system, steps, pe, info.reason, info)
