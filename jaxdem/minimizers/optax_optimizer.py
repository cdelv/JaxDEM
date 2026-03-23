# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Optax-based energy minimizer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

from . import LinearMinimizer, RotationMinimizer
from ..integrators import LinearIntegrator, RotationIntegrator

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@LinearMinimizer.register("optax")
@LinearIntegrator.register("optax")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class OptaxOptimizer(LinearMinimizer):
    """Optax-based energy minimizer for coupled position and orientation updates.

    This minimizer leverages optax to step both positions and quaternions
    simultaneously based on energy gradients.
    """

    opt_state: Any
    optimizer: Any = jax.tree.static()  # type: ignore[attr-defined]

    @classmethod
    def Create(cls, optimizer: Any, state: State) -> OptaxOptimizer:
        """Create an OptaxOptimizer with a given optax optimizer.

        Parameters
        ----------
        optimizer : optax.GradientTransformation
            The optax optimizer to use (e.g. optax.adam(1e-3)).
        state : State
            The initial state, used to initialize the optimizer state.

        Returns
        -------
        OptaxOptimizer
            A new minimizer instance.
        """

        params = {"pos_c": state.pos_c, "q": state.q}
        opt_state = optimizer.init(params)
        return cls(opt_state=opt_state, optimizer=optimizer)

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="OptaxOptimizer.step_after_force")
    def step_after_force(state: State, system: System) -> tuple[State, System]:
        import optax
        from ..utils.quaternion import Quaternion

        opt = cast(OptaxOptimizer, system.linear_integrator)

        # We can extract exact energy gradients directly from JaxDEM calculated forces!
        # This completely avoids double-evaluating the energy functions and neighbors.
        mask = (1 - state.fixed)[..., None]
        
        # Position gradient is simply the negative force
        grad_pos = -state.force * mask

        # Orientation gradient (dE/dq) is derived from torque (tau = -dE/dtheta).
        # We can map lab-frame torque back to the 4D quaternion Euclidean gradient
        if state.dim == 2:
            torque_3d = jnp.pad(state.torque, ((0, 0), (2, 0)), constant_values=0.0)
        else:
            torque_3d = state.torque
            
        tau_q = Quaternion(jnp.zeros_like(state.q.w), torque_3d)
        grad_q_obj = tau_q @ state.q
        
        grad_q = Quaternion(
            w=-0.5 * grad_q_obj.w * mask,
            xyz=-0.5 * grad_q_obj.xyz * mask
        )

        grads = {"pos_c": grad_pos, "q": grad_q}
        params = {"pos_c": state.pos_c, "q": state.q}

        updates, new_opt_state = opt.optimizer.update(grads, opt.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        new_q = new_params["q"].unit(new_params["q"])
        state = replace(state, pos_c=new_params["pos_c"], q=new_q)

        new_opt = replace(opt, opt_state=new_opt_state)
        system = replace(system, linear_integrator=new_opt)

        return state, system


@RotationMinimizer.register("optax")
@RotationIntegrator.register("optax")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class OptaxRotationNoOp(RotationMinimizer):
    """Dummy rotation minimizer.

    Since OptaxOptimizer handles both positions and orientations,
    this rotation minimizer acts as a no-op to prevent double updates.
    """

    @classmethod
    def Create(cls) -> OptaxRotationNoOp:
        return cls()

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="OptaxRotationNoOp.step_after_force")
    def step_after_force(state: State, system: System) -> tuple[State, System]:
        return state, system
