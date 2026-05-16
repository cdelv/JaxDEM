# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Optax-based minimizer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, cast

import jax
import jax.numpy as jnp

from ..integrators import LinearIntegrator, RotationIntegrator
from ..utils.linalg import unit_and_norm
from ..utils.quaternion import Quaternion
from ..utils.thermal import compute_potential_energy
from . import LinearMinimizer, RotationMinimizer

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.jit
def _quaternion_to_rotvec(q: Quaternion) -> jax.Array:
    """
    Map a unit quaternion to its axis-angle rotation vector.

    Parameters
    ----------
    q : Quaternion
        The input unit quaternion.

    Returns
    -------
    jax.Array
        The 3D axis-angle rotation vector.
    """
    q_u = q.unit(q)

    # q and -q represent the same rotation; keep the shortest representation.
    sign = jnp.where(q_u.w < 0.0, -1.0, 1.0)
    w = q_u.w * sign
    xyz = q_u.xyz * sign

    axis, sin_half = unit_and_norm(xyz)
    angle = 2.0 * jnp.arctan2(sin_half, w)
    return axis * angle


@jax.jit
def _rotvec_to_quaternion(rotvec: jax.Array) -> Quaternion:
    """
    Map an axis-angle rotation vector to a unit quaternion via exponential map.

    Parameters
    ----------
    rotvec : jax.Array
        The 3D axis-angle rotation vector.

    Returns
    -------
    Quaternion
        The corresponding unit quaternion.
    """
    axis, angle = unit_and_norm(rotvec)
    half_angle = 0.5 * angle
    return Quaternion(jnp.cos(half_angle), axis * jnp.sin(half_angle))


@jax.jit
def _state_to_params(state: State) -> jax.Array:
    """
    Particle pos and orientation to optimization vector.

    Parameters
    ----------
    state : State
        The simulation state containing particle positions and orientations.

    Returns
    -------
    jax.Array
        A packed array of optimization parameters. For 2D, it is a 3D vector
        (x, y, theta). For 3D, it is a 6D vector (x, y, z, rx, ry, rz).
    """
    rotvec = _quaternion_to_rotvec(state.q)
    if state.dim == 2:
        return jnp.concatenate([state.pos_c, rotvec[..., 2:3]], axis=-1)
    else:
        return jnp.concatenate([state.pos_c, rotvec], axis=-1)


@jax.jit
def _params_to_state(state: State, params: jax.Array) -> State:
    """
    Unpack an optimization vector to state pos and orientation.

    Parameters
    ----------
    state : State
        The baseline simulation state.
    params : jax.Array
        The packed optimization parameters.

    Returns
    -------
    State
        A new state with updated center of mass positions and quaternions.
    """
    if state.dim == 2:
        pos_c = params[..., 0:2]
        rotvec = jnp.concatenate(
            [jnp.zeros_like(pos_c), params[..., 2:3]],
            axis=-1,
        )
    else:
        pos_c = params[..., 0:3]
        rotvec = params[..., 3:6]

    q = _rotvec_to_quaternion(rotvec)
    return replace(state, pos_c=pos_c, q=q.unit(q))


@partial(jax.custom_vjp)
def _objective_energy(
    trial_params: jax.Array,
    state: State,
    system: System,
) -> jax.Array:
    """
    Evaluate potential energy with a custom backward pass returning analytical forces.

    Notes
    -----
    The backward pass directly returns the analytical negative force and torque
    as the gradient of the parameters, bypassing automatic differentiation of
    the energy functions.

    Parameters
    ----------
    trial_params : jax.Array
        The trial optimization parameters.
    state : State
        The baseline simulation state.
    system : System
        The system configuration.

    Returns
    -------
    jax.Array
        The evaluated total potential energy of the system.
    """
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


@LinearMinimizer.register("optax")
@LinearIntegrator.register("optax")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class OptaxOptimizer(LinearMinimizer):
    """
    Optax-based minimizer for coupled position and orientation updates.

    Notes
    -----
    This minimizer leverages the Optax library to perform gradient-based energy
    minimization. By default, it maps analytical forces and torques directly to
    gradients of the optimization parameters to bypass autodiff.
    """

    opt_state: Any
    """State of the underlying optax optimizer."""

    params: jax.Array
    """The optimization parameters representing particle poses (3D or 6D)."""

    optimizer: Any = jax.tree.static()  # type: ignore[attr-defined]
    """The optax optimizer instance (GradientTransformation)."""

    target_fn: Callable[[State, System], jax.Array] | None = jax.tree.static(
        default=None
    )  # type: ignore[attr-defined]
    """Optional custom target evaluation function."""

    @classmethod
    def Create(
        cls,
        optimizer: Any,
        state: State,
        target_fn: Callable[[State, System], jax.Array] | None = None,
    ) -> OptaxOptimizer:
        """
        Create a new OptaxOptimizer.

        Parameters
        ----------
        optimizer : Any
            An instantiated optax GradientTransformation (e.g., ``optax.fire(...)``).
        state : State
            The initial state of the system, used to initialize the optimizer.
        target_fn : Callable[[State, System], jax.Array] or None, optional
            A custom function to compute the target value (e.g., loss, structural error)
            from the state and system.
            The signature must be `(state: State, system: System) -> jax.Array`.
            If None, defaults to using the internal custom VJP which calculates
            the system's potential energy alongside the analytical forces as gradients.

        Returns
        -------
        OptaxOptimizer
            An initialized OptaxOptimizer instance.
        """
        params = _state_to_params(state)
        opt_state = optimizer.init(params)
        return cls(
            opt_state=opt_state,
            params=params,
            optimizer=optimizer,
            target_fn=target_fn,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="OptaxOptimizer.step_after_force")
    def step_after_force(state: State, system: System) -> tuple[State, System]:
        """
        Perform a single minimization step.

        Parameters
        ----------
        state : State
            The current simulation state.
        system : System
            The simulation system configuration.

        Returns
        -------
        tuple[State, System]
            The updated state and system after applying the optimizer step.
        """
        import optax

        opt = cast(OptaxOptimizer, system.linear_integrator)
        params = _state_to_params(state)
        mask = ~state.fixed[..., None]

        def value_fn(optim_params: jax.Array) -> jax.Array:
            if opt.target_fn is None:
                return _objective_energy(optim_params, state, system)
            else:
                trial_state = _params_to_state(state, optim_params)
                return opt.target_fn(trial_state, system)

        pe_current, grads = jax.value_and_grad(value_fn)(params)
        grads *= mask

        # Let Optax do its magic
        updates, new_opt_state = opt.optimizer.update(
            grads,
            opt.opt_state,
            params,
            value=pe_current,
            grad=grads,
            value_fn=value_fn,
        )

        updates *= mask

        new_params = optax.apply_updates(params, updates)
        new_params = jnp.where(mask, new_params, params)

        state = _params_to_state(state, new_params)

        new_opt = replace(opt, opt_state=new_opt_state, params=new_params)
        system = replace(system, linear_integrator=new_opt)

        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="OptaxOptimizer.initialize")
    def initialize(state: State, system: System) -> tuple[State, System]:
        """
        Initialize or sync the optimizer state with the simulation state.

        Parameters
        ----------
        state : State
            The current simulation state.
        system : System
            The simulation system configuration.

        Returns
        -------
        tuple[State, System]
            The updated state and system with a synchronized optimizer.
        """
        opt = cast(OptaxOptimizer, system.linear_integrator)
        params = _state_to_params(state)
        synced_opt = replace(opt, params=params, opt_state=opt.optimizer.init(params))
        return state, replace(system, linear_integrator=synced_opt)


@RotationMinimizer.register("optax")
@RotationIntegrator.register("optax")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class OptaxRotationNoOp(RotationMinimizer):
    """
    Dummy rotation minimizer.

    Notes
    -----
    The OptaxOptimizer couples both linear and rotational updates into a single
    step via 6D optimization parameters. This no-op class is registered to prevent
    the simulation loop from performing duplicate rotational updates.
    """

    @classmethod
    def Create(cls) -> OptaxRotationNoOp:
        """
        Create a new OptaxRotationNoOp.

        Returns
        -------
        OptaxRotationNoOp
            An initialized dummy rotation minimizer.
        """
        return cls()

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="OptaxRotationNoOp.step_after_force")
    def step_after_force(state: State, system: System) -> tuple[State, System]:
        """
        Perform a no-op step after forces are calculated.

        Parameters
        ----------
        state : State
            The current simulation state.
        system : System
            The simulation system configuration.

        Returns
        -------
        tuple[State, System]
            The unchanged state and system.
        """
        return state, system
