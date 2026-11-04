# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Minimization routines and drivers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp

from ..forces.force_manager import default_energy_func
from ..topology import BodyTopology, body_topology
from ..utils.quaternion import Quaternion
from ..utils.linalg import cross, dot, norm, norm2

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
        Maximum Euclidean force norm over free bodies for mechanical
        relaxation. Objectives differentiated by AD use the maximum norm of
        their gradient with respect to body position.
    torque_max : jax.Array
        Maximum Euclidean torque norm over free bodies for mechanical
        relaxation. Objectives differentiated by AD use the maximum norm of
        their world angular gradient.
    status : jax.Array
        Integer scalar describing convergence:

        * 0 (``CONVERGED``): both tolerances are satisfied.
        * 2 (``MAX_STEPS``): residuals exceed tolerance; a completed
          minimization exhausted its step budget.
        * 3 (``NONFINITE``): an evaluated quantity is nonfinite.
        * 4 (``COLLIDER_OVERFLOW``): spatial search results are incomplete.
    pressure : jax.Array
        Contact pressure of the evaluated configuration when
        :func:`minimize` controls pressure, otherwise NaN. Convergence then
        also requires ``abs(pressure - target) <= pressure_tol``.
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
    pressure: jax.Array | float = float("nan")

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

    def maximum_norm(value: jax.Array) -> jax.Array:
        free = jnp.where(fixed[..., None], 0.0, value)
        return jnp.max(norm(free), initial=0.0)

    force_max = maximum_norm(force)
    torque_max = maximum_norm(torque)
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
        Final configuration with evaluated quasistatic forces and torques.
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


def _conservative_manager_forces(state: State, system: System) -> tuple[State, System]:
    """Add rate-independent managed forces without consuming buffered loads."""
    moving_state = state
    state = replace(
        state, vel=jnp.zeros_like(state.vel), ang_vel=jnp.zeros_like(state.ang_vel)
    )
    force_manager = system.force_manager
    domain = system.domain
    empty_manager = replace(
        force_manager,
        external_force=jnp.zeros_like(force_manager.external_force),
        external_force_com=jnp.zeros_like(force_manager.external_force_com),
        external_torque=jnp.zeros_like(force_manager.external_torque),
    )
    system = replace(
        system,
        force_manager=empty_manager,
        domain=domain.rates_between(domain, jnp.ones_like(system.dt)),
    )
    state, system = system.force_manager.apply(state, system)
    state = replace(state, vel=moving_state.vel, ang_vel=moving_state.ang_vel)
    return state, replace(system, force_manager=force_manager, domain=domain)


def _evaluate_readonly_forces(
    state: State, system: System, *, virial: bool = False
) -> tuple[Any, ...]:
    """Evaluate rate-independent forces without touching history or loads.

    With ``virial`` the collider's pair virial is returned as a third value.
    """
    # Only name the flag when set, so colliders without it keep working.
    conservative_state, eval_system, *extra = system.collider.compute_force(
        state,
        system,
        advance_history=False,
        quasistatic=True,
        **({"virial": True} if virial else {}),
    )
    conservative_state, eval_system = _conservative_manager_forces(
        conservative_state, eval_system
    )
    evaluated_state = replace(
        state,
        force=conservative_state.force,
        torque=conservative_state.torque,
    )
    eval_system = replace(
        eval_system,
        search_overflow=eval_system.search_overflow | eval_system.collider.overflow,
    )
    return (evaluated_state, eval_system, *extra)


def _advance_history_forces(
    anchor_state: State,
    anchor_system: System,
    trial_state: State,
    trial_system: System,
    *,
    virial: bool = False,
) -> tuple[Any, ...]:
    """Advance pair history from anchor to trial and keep that traversal's forces."""
    from ..colliders import advance_history

    evaluated_state, evaluated, *extra = advance_history(
        anchor_state, anchor_system, trial_state, trial_system, virial=virial
    )
    evaluated_state, evaluated = _conservative_manager_forces(
        evaluated_state, evaluated
    )
    return (
        replace(
            trial_state, force=evaluated_state.force, torque=evaluated_state.torque
        ),
        evaluated,
        *extra,
    )


@jax.jit
def _state_to_delta_params(
    state: State, topology: BodyTopology | None = None
) -> dict[str, jax.Array]:
    """Pack body positions and identity rotations about a fixed reference.

    Two-dimensional rotations are unwrapped angles. Three-dimensional
    rotations are raw quaternions, normalized only when constructing a State.
    Optimizer coordinates remain continuous throughout a minimization.
    """
    topology = (
        body_topology(state.clump_id, state.fixed) if topology is None else topology
    )
    body_pos = topology.gather_representatives(state.pos_c)
    width = 1 if state.dim == 2 else 4
    rotation = jnp.zeros(body_pos.shape[:-1] + (width,), dtype=body_pos.dtype)
    if state.dim == 3:
        # Radius two gives the initial xyz coordinates unit angular scale.
        rotation = rotation.at[..., 0].set(2.0)
    return {"pos_c": body_pos, "rotation": rotation}


@jax.jit
def _delta_params_to_state(
    state: State,
    params: dict[str, jax.Array],
    topology: BodyTopology | None = None,
) -> State:
    """Map continuous optimizer coordinates to normalized body orientations.

    Rotations left-multiply the fixed reference ``state.q``. Keep this
    reference unchanged throughout minimization; normalizing a raw quaternion
    here does not modify the optimizer's parameters or accumulated state.
    A zero raw quaternion is invalid and produces a nonfinite evaluation.
    """
    topology = (
        body_topology(state.clump_id, state.fixed) if topology is None else topology
    )
    rotation = params["rotation"]
    if state.dim == 2:
        rotvec = jnp.concatenate([jnp.zeros_like(params["pos_c"]), rotation], axis=-1)
        delta_q = Quaternion.from_rotvec(rotvec)
    else:
        unit = rotation * jax.lax.rsqrt(norm2(rotation)[..., None])
        delta_q = Quaternion.create(unit[..., :1], unit[..., 1:])
    body_q = Quaternion.create(
        topology.gather_representatives(state.q.w),
        topology.gather_representatives(state.q.xyz),
    )
    q = Quaternion.unit(delta_q @ body_q)
    member_q = Quaternion.create(
        topology.gather_members(q.w), topology.gather_members(q.xyz)
    )
    return replace(state, pos_c=topology.gather_members(params["pos_c"]), q=member_q)


class _BoxReference(NamedTuple):
    """Reference geometry for the box coordinate of pressure-controlled relaxation.

    The optimizer coordinate ``params["box"]`` is ``scale * log(L / L0)``,
    with ``L0`` the box lengths at the start of the minimization. ``scale``
    is the square root of ``sum(|R_i - anchor|^2)`` over bodies, so under
    the unit-mass dynamics of :func:`~jaxdem.minimizers.fire` the box
    coordinate carries the kinetic energy of the affine motion it induces.
    """

    box_size: jax.Array
    scale: jax.Array
    pressure: jax.Array


def _box_reference(
    state: State, system: System, topology: BodyTopology, pressure: Any
) -> _BoxReference:
    body_pos = topology.gather_representatives(state.pos_c)
    offset = jnp.where(topology.valid[..., None], body_pos - system.domain.anchor, 0.0)
    mass = jnp.sum(offset**2)
    return _BoxReference(
        system.domain.box_size,
        jnp.sqrt(jnp.where(mass > 0.0, mass, 1.0)),
        jnp.asarray(pressure, dtype=body_pos.dtype),
    )


def _box_factor(params: dict[str, jax.Array], box: _BoxReference) -> jax.Array:
    """Uniform box scale ``L / L0`` encoded by the box coordinate."""
    return jnp.exp(params["box"][0] / box.scale)


def _trial_configuration(
    state: State,
    system: System,
    params: dict[str, jax.Array],
    topology: BodyTopology,
    box: _BoxReference | None,
) -> tuple[State, System]:
    """Map optimizer coordinates to a trial state and its system.

    Without a box coordinate the system is returned unchanged. With one, the
    box lengths are ``L0 * factor`` and every body centre moves affinely
    about the domain anchor, as in :meth:`Domain.resize`.
    """
    trial = _delta_params_to_state(state, params, topology)
    if box is None:
        return trial, system
    factor = _box_factor(params, box)
    anchor = system.domain.anchor
    box_size = box.box_size * factor
    domain = replace(system.domain, box_size=box_size, inv_box_size=1.0 / box_size)
    return (
        replace(trial, pos_c=anchor + (trial.pos_c - anchor) * factor),
        replace(system, domain=domain),
    )


def _mechanical_gradient(
    params: dict[str, jax.Array],
    trial: State,
    evaluated: System,
    topology: BodyTopology,
    box: _BoxReference | None,
    virial: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """Gradient of the (enthalpy) objective from evaluated forces and virial.

    With a box coordinate the objective is ``U + P V``. Positions pick up
    the box factor from the affine map, and the box coordinate's gradient is
    ``(dim * P * V - W) / scale`` with ``W`` the pair virial, which vanishes
    when the contact pressure ``W / (dim * V)`` equals the target.
    """
    factor = 1.0 if box is None else _box_factor(params, box)
    grads = {
        "pos_c": -topology.gather_representatives(trial.force) * factor,
        "rotation": _rotation_pullback(
            params["rotation"], -topology.gather_representatives(trial.torque)
        ),
    }
    if box is not None:
        volume = jnp.prod(evaluated.domain.box_size)
        dim = box.box_size.shape[-1]
        grads["box"] = ((dim * box.pressure * volume - virial) / box.scale)[None]
    return grads


def _rotation_pullback(rotation: jax.Array, gradient: jax.Array) -> jax.Array:
    """Pull a world angular gradient back to fixed optimizer coordinates."""
    if rotation.shape[-1] == 1:
        return gradient
    w, xyz = rotation[..., :1], rotation[..., 1:]
    scalar = -dot(xyz, gradient)[..., None]
    vector = w * gradient + cross(gradient, xyz)
    return 2.0 * jnp.concatenate([scalar, vector], axis=-1) / norm2(rotation)[..., None]


def _rotation_residual(rotation: jax.Array, gradient: jax.Array) -> jax.Array:
    """Recover the world angular residual independently of quaternion scale."""
    if rotation.shape[-1] == 1:
        return gradient
    w, xyz = rotation[..., :1], rotation[..., 1:]
    gw, gxyz = gradient[..., :1], gradient[..., 1:]
    return 0.5 * (w * gxyz - xyz * gw + cross(xyz, gxyz))


@jax.custom_vjp
def _objective_energy(
    trial_params: dict[str, jax.Array],
    state: State,
    system: System,
    topology: BodyTopology | None = None,
    box: _BoxReference | None = None,
) -> tuple[jax.Array, tuple[State, System]]:
    """Evaluate the potential energy (or enthalpy) of the trial parameters.

    Rotation coordinates are anchored at ``state.q``: an unwrapped angle in
    2D or a raw normalized quaternion in 3D (see ``_delta_params_to_state``).
    The analytical VJP includes the normalization pullback at every trial.

    Parameters
    ----------
    trial_params : dict
        The dict of trial parameters containing 'pos_c' and 'rotation', and
        'box' when ``box`` is given.
    state : State
        The simulation state (anchor for the rotation parameters).
    system : System
        The system configuration.
    box : _BoxReference, optional
        Pressure-control reference; the objective is then ``U + P V``.

    Returns
    -------
    Tuple[jax.Array, Tuple[State, System]]
        A tuple containing the objective and a tuple of the evaluated State and System.
    """
    topology = (
        body_topology(state.clump_id, state.fixed) if topology is None else topology
    )
    trial_state, trial_system = _trial_configuration(
        state, system, trial_params, topology, box
    )
    trial_state, eval_system, pe = _evaluate_energy(trial_state, trial_system)
    if box is not None:
        pe = pe + box.pressure * jnp.prod(eval_system.domain.box_size)
    return pe, (trial_state, eval_system)


def _objective_energy_fwd(
    trial_params: dict[str, jax.Array],
    state: State,
    system: System,
    topology: BodyTopology | None = None,
    box: _BoxReference | None = None,
) -> tuple[tuple[jax.Array, tuple[State, System]], dict[str, jax.Array]]:
    topology = (
        body_topology(state.clump_id, state.fixed) if topology is None else topology
    )
    pe, (trial_state, eval_system) = _objective_energy(
        trial_params, state, system, topology, box
    )
    conservative_state, eval_system, *extra = _evaluate_readonly_forces(
        trial_state, eval_system, **({"virial": True} if box is not None else {})
    )
    grads = _mechanical_gradient(
        trial_params, conservative_state, eval_system, topology, box, *extra
    )
    return (pe, (conservative_state, eval_system)), grads


def _objective_energy_bwd(
    residual: dict[str, jax.Array], g: tuple[jax.Array, Any]
) -> tuple[dict[str, jax.Array], None, None, None, None]:
    scale, _ = g
    return (
        jax.tree.map(lambda grad: grad * scale, residual),
        None,
        None,
        None,
        None,
    )


_objective_energy.defvjp(_objective_energy_fwd, _objective_energy_bwd)


@jax.jit(static_argnames=["max_steps", "advance_history"])
def minimize(
    state: State,
    system: System,
    max_steps: int = 10000,
    force_tol: float = 1e-12,
    torque_tol: float | None = None,
    *,
    advance_history: bool = False,
    pressure: float | jax.Array | None = None,
    pressure_tol: float | jax.Array | None = None,
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
        bodies. AD objectives use their body-position gradient instead.
        Defaults to ``1e-12``.
    torque_tol : float or None, optional
        Absolute tolerance on the maximum Euclidean torque norm over free
        bodies. AD objectives use their world angular gradient instead.
        ``None`` uses the numerical value of ``force_tol``.
    advance_history : bool, optional
        Whether accepted optimizer updates advance pair history. Defaults to
        ``False``, preserving stored history without accumulating slip from
        optimizer motion. Updates occur once after each accepted step; trial
        objective evaluations hold history fixed. Stateless force models
        ignore this option.
    pressure : float or jax.Array, optional
        Target contact pressure. When set, the box size becomes an optimizer
        coordinate and the objective is the enthalpy ``U + pressure * V``.
        Defaults to ``None``, a fixed box.
    pressure_tol : float or jax.Array, optional
        Absolute tolerance on ``abs(contact pressure - pressure)``. ``None``
        uses the numerical value of ``force_tol``.

    Returns
    -------
    MinimizationResult
        Final state, system, update count, energy, and convergence diagnostics.
        Iteration yields ``(state, system, steps, energy)``. Physical energy
        is reported per constituent sphere; custom target values are
        unnormalized. With pressure control the returned system carries the
        final box and ``info.pressure`` the final contact pressure.

    Raises
    ------
    ValueError
        If no optimizer is configured, or if the physical objective uses a
        managed force without a matching energy function, or if pressure
        control is requested with a custom target, managed forces, or bonded
        forces.

    Notes
    -----
    Convergence requires both ``force_max <= force_tol`` and
    ``torque_max <= torque_tol`` with finite evaluations and complete spatial
    searches. Fixed bodies are excluded from the residual norms. Custom
    targets and energies without analytical physical gradients use their
    translational and world angular AD gradients as residuals. These can
    differ from mechanical forces for history-dependent laws. Energy
    magnitude and energy changes do not determine convergence. Step exhaustion,
    nonfinite evaluations, and spatial search overflow terminate the
    minimization unsuccessfully.

    For the physical objective, gradient-only optimizers
    (``requires_value=False``, including FIRE and damped Newtonian) evaluate
    residual forces initially and after each update, and energy once at exit.
    Gradient-only optimizers also support force conventions that are not
    analytical energy gradients. Value-based optimizers use the analytical
    physical gradient when available and otherwise differentiate the energy
    directly with stored history fixed. Custom targets use ordinary AD
    through the State mapping. These optimizers evaluate the objective during
    optimization. AD through built-in collider energy requires Naive or
    NeighborList; direct CellList and MultiCellList traversals do not support
    reverse-mode differentiation.

    Each rigid body has one set of continuous optimization coordinates.
    Two-dimensional rotations use an unwrapped angle about the initial
    orientation; three-dimensional rotations use a raw quaternion normalized
    only when constructing the trial state. The initial orientation reference
    and optimizer coordinates are retained, so transformations that remember
    parameter differences (including L-BFGS) receive consistent coordinates.
    A zero raw quaternion is invalid and terminates with a nonfinite result.
    Quaternion sign and radial redundancy do not guarantee convergence for
    every optimizer; convergence uses world angular residuals independently
    of the raw quaternion scale.
    Forces are always evaluated rate-independently: dashpots and other
    velocity terms are off, so relaxation does not depend on ``dt``. Pair
    history is frozen by default; existing spring memory still contributes to
    force evaluation. Search rebuilds remap surviving pairs and initialize new
    pairs. With ``advance_history=True``, history advances once per accepted
    coordinate update, using configuration changes from the anchor to the
    accepted state. Energy and gradients are then evaluated with that updated
    history fixed. Line-search trials never commit history, and the initial
    evaluation always holds history fixed.

    With ``pressure`` set, one extra coordinate scales every box length by
    the same factor and moves all body centres affinely about the domain
    anchor, as :meth:`Domain.resize` does; orientations and member offsets
    are unchanged. Its gradient is ``dim * V * (pressure - P)`` with ``P``
    the contact pressure from the pair virial, which the colliders
    accumulate in the same pass as the forces. The coordinate is scaled by
    ``sqrt(sum |R_i - anchor|^2)`` so it carries the kinetic energy of the
    affine motion it causes under the unit-mass optimizer dynamics; wrap
    positions into the box first for the best conditioning. Convergence
    additionally requires ``abs(P - pressure) <= pressure_tol``. Fixed bodies
    move with the box. Only collider pair forces contribute to the virial,
    so managed and bonded forces are rejected; gravity is not part of it.
    """
    import optax  # type: ignore[import-untyped]

    if system.minimizer is None:
        raise ValueError(
            "No minimizer configured in System. Please configure `minimizer` in System.create."
        )
    force_only = system.target_fn is None and not getattr(
        system.minimizer, "requires_value", True
    )
    analytical_gradient = (
        system.target_fn is None
        and system.force_model.supports_analytical_energy_gradient
    )
    mechanical_residuals = force_only or analytical_gradient
    history_relaxation = advance_history and system.force_model.history_shape(
        state.dim
    ) != (0,)
    if system.target_fn is None and any(
        energy_fn is default_energy_func
        for energy_fn in system.force_manager.energy_functions
    ):
        raise ValueError(
            "Every custom force used by default minimization must provide a "
            "matching energy function, or the system must provide target_fn."
        )
    if pressure is not None:
        if system.target_fn is not None:
            raise ValueError("Pressure control requires the physical energy objective.")
        if system.force_manager.force_functions or system.bonded_force_model:
            raise ValueError(
                "Pressure control supports collider pair forces only; managed "
                "and bonded forces do not report a virial."
            )

    torque_tol = force_tol if torque_tol is None else torque_tol
    pressure_tol = force_tol if pressure_tol is None else pressure_tol
    topology = body_topology(state.clump_id, state.fixed)
    fixed = ~topology.valid | topology.fixed
    mask = ~fixed[..., None]
    box = (
        None if pressure is None else _box_reference(state, system, topology, pressure)
    )
    free = {"pos_c": mask, "rotation": mask}
    if box is not None:
        free["box"] = jnp.ones((1,), dtype=bool)

    def restrict(tree: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return jax.tree.map(lambda keep, x: jnp.where(keep, x, 0.0), free, tree)

    reference_q = state.q

    def make_value_fn(anchor_state: State, anchor_system: System) -> Any:
        anchor_state = replace(anchor_state, q=reference_q)

        def value_fn(params: dict[str, jax.Array]) -> Any:
            if analytical_gradient:
                return _objective_energy(
                    params, anchor_state, anchor_system, topology, box
                )
            trial, trial_system = _trial_configuration(
                anchor_state, anchor_system, params, topology, box
            )
            if anchor_system.target_fn is None:
                trial, evaluated, energy = _evaluate_energy(trial, trial_system)
                if box is not None:
                    energy = energy + box.pressure * jnp.prod(evaluated.domain.box_size)
                return energy, (trial, evaluated)
            return anchor_system.target_fn(trial, anchor_system), (trial, anchor_system)

        return value_fn

    def eval_step(
        anchor_state: State,
        anchor_system: System,
        params: dict[str, jax.Array],
        *,
        advance: bool,
    ) -> tuple[Any, dict[str, jax.Array], State, System]:
        if force_only:
            trial, trial_system = _trial_configuration(
                replace(anchor_state, q=reference_q),
                anchor_system,
                params,
                topology,
                box,
            )
            virial_kw = {"virial": True} if box is not None else {}
            if history_relaxation and advance:
                trial, evaluated, *extra = _advance_history_forces(
                    anchor_state, anchor_system, trial, trial_system, **virial_kw
                )
            else:
                trial, evaluated, *extra = _evaluate_readonly_forces(
                    trial, trial_system, **virial_kw
                )
            grads = _mechanical_gradient(
                params, trial, evaluated, topology, box, *extra
            )
            return None, grads, trial, evaluated
        if history_relaxation and advance:
            trial, trial_system = _trial_configuration(
                replace(anchor_state, q=reference_q),
                anchor_system,
                params,
                topology,
                box,
            )
            _, anchor_system = _advance_history_forces(
                anchor_state, anchor_system, trial, trial_system
            )
        (pe, (trial, evaluated)), grads = jax.value_and_grad(
            make_value_fn(anchor_state, anchor_system), has_aux=True
        )(params)
        return pe, grads, trial, evaluated

    def diagnostics(
        current: State,
        params: dict[str, jax.Array],
        grads: dict[str, jax.Array],
        pe: Any,
        evaluated: System,
    ) -> MinimizeInfo:
        if mechanical_residuals:
            force = topology.gather_representatives(current.force)
            torque = topology.gather_representatives(current.torque)
        else:
            force = grads["pos_c"]
            torque = _rotation_residual(params["rotation"], grads["rotation"])
        info = convergence_info(
            force,
            torque,
            fixed,
            force_tol,
            torque_tol,
        )
        finite = info.finite & jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads)])
        )
        if pe is not None:
            finite = finite & jnp.isfinite(pe)
        overflow = evaluated.search_overflow | evaluated.collider.overflow
        converged = info.converged & finite & ~overflow
        pressure_value = jnp.asarray(jnp.nan, dtype=force.dtype)
        if box is not None:
            # Recover P from the box gradient so every objective path agrees.
            volume = jnp.prod(evaluated.domain.box_size)
            dim = box.box_size.shape[-1]
            pressure_value = box.pressure - grads["box"][0] * box.scale / (dim * volume)
            converged = converged & (
                jnp.abs(pressure_value - box.pressure) <= pressure_tol
            )
        return info._replace(
            finite=finite,
            converged=converged,
            pressure=pressure_value,
            status=jnp.where(
                overflow,
                COLLIDER_OVERFLOW,
                jnp.where(
                    ~finite, NONFINITE, jnp.where(converged, CONVERGED, MAX_STEPS)
                ),
            ),
        )

    params = _state_to_delta_params(state, topology)
    if box is not None:
        params["box"] = jnp.zeros((1,), dtype=params["pos_c"].dtype)
    opt_state = system.minimizer.init(params)
    pe, grads, state, system = eval_step(state, system, params, advance=False)
    carry = (
        state,
        system,
        0,
        pe,
        params,
        opt_state,
        grads,
    )

    def condition(c: tuple[Any, ...]) -> jax.Array:
        current, evaluated, steps, pe, params, _, grads = c
        info = diagnostics(current, params, grads, pe, evaluated)
        return (steps < max_steps) & (info.status == MAX_STEPS)

    def step(c: tuple[Any, ...]) -> tuple[Any, ...]:
        current, evaluated, steps, pe, params, opt_state, grads = c
        free_grads = restrict(grads)
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
        updates = restrict(updates)
        next_params = optax.apply_updates(params, updates)
        pe, grads, current, evaluated = eval_step(
            current, evaluated, next_params, advance=True
        )
        return (
            current,
            evaluated,
            steps + 1,
            pe,
            next_params,
            opt_state,
            grads,
        )

    state, system, steps, pe, params, _, grads = jax.lax.while_loop(
        condition, step, carry
    )
    if force_only or box is not None:
        # Report potential energy, not the enthalpy objective.
        state, system, pe = _evaluate_energy(state, system)
    if not force_only and not mechanical_residuals:
        # AD objectives need not have evaluated physical forces.
        state, system = _evaluate_readonly_forces(state, system)
    if system.target_fn is None:
        pe = pe / max(state.N, 1)
    info = diagnostics(state, params, grads, pe, system)
    return MinimizationResult(state, system, steps, pe, info.reason, info)
