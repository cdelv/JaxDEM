# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Basic gradient-descent energy minimizer."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Tuple, cast

from . import LinearMinimizer, RotationMinimizer
from ..integrators import LinearIntegrator, RotationIntegrator
from ..integrators.velocity_verlet_spiral import omega_dot
from ..utils.quaternion import Quaternion

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@LinearMinimizer.register("lineargradientdescent")
@LinearIntegrator.register("lineargradientdescent")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LinearGradientDescent(LinearMinimizer):

    learning_rate: jax.Array

    @classmethod
    def Create(cls, learning_rate: float = 1e-3) -> "LinearGradientDescent":
        """Create a LinearGradientDescent minimizer with JAX array parameters.

        Parameters
        ----------
        learning_rate : float, optional
            Learning rate for gradient descent updates. Default is 1e-3.

        Returns
        -------
        LinearGradientDescent
            A new minimizer instance with JAX array parameters.
        """
        return cls(learning_rate=jnp.array(learning_rate))

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="LinearGradientDescent.step_after_force")
    def step_after_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """Gradient-descent update using the integrator's learning rate.

        The learning rate is stored on the LinearGradientDescent dataclass
        attached to ``system.linear_integrator``, so no mutable state is kept
        outside the System PyTree.

        The update equation is simply:

        .. math::
            r_{t+1} = r_{t} + \\gamma F_{t}
        """
        gd = cast(LinearGradientDescent, system.linear_integrator)
        lr = gd.learning_rate
        mask = (1 - state.fixed)[..., None]
        state.pos_c += lr * state.force * mask
        return state, system


@RotationMinimizer.register("rotationgradientdescent")
@RotationIntegrator.register("rotationgradientdescent")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class RotationGradientDescent(RotationMinimizer):

    learning_rate: jax.Array

    @classmethod
    def Create(cls, learning_rate: float = 1e-3) -> "RotationGradientDescent":
        """Create a RotationGradientDescent minimizer with JAX array parameters.

        Parameters
        ----------
        learning_rate : float, optional
            Learning rate for gradient descent updates. Default is 1e-3.

        Returns
        -------
        RotationGradientDescent
            A new minimizer instance with JAX array parameters.
        """
        return cls(learning_rate=jnp.array(learning_rate))

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="RotationGradientDescent.step_after_force")
    def step_after_force(state: "State", system: "System") -> Tuple["State", "System"]:
        r"""Gradient-descent update using the integrator's learning rate.

        The learning rate is stored on the RotationGradientDescent dataclass
        attached to ``system.rotation_integrator``, so no mutable state is kept
        outside the System PyTree.

        The update equation is:

        .. math::
            q_{t+1} = q_{t} \cdot e^\left(\\gamma \\tau_t I^{-1})

        Where the torque term is a purely imaginary quaternion (scalar part is zero and the vector part is equal to the vector). The exponential map of a purely imaginary quaternion is

        .. math::
            e^u = \cos(|u|) + \frac{\vec{u}}{|u|}\sin(|u|)
        """
        gd = cast(RotationGradientDescent, system.rotation_integrator)
        lr = gd.learning_rate

        # pad torques to 3d if needed
        if state.dim == 2:
            torque_lab_3d = jnp.pad(state.torque, ((0, 0), (2, 0)), constant_values=0.0)
        else:  # state.dim == 3
            torque_lab_3d = state.torque

        torque = state.q.rotate_back(
            state.q, torque_lab_3d
        )  # rotate torques to body frame

        # calculate angular acceleration due to torques
        # no angular velocity dependence
        k = (
            0.5
            * lr
            * omega_dot(torque * 0.0, torque, state.inertia, 1 / state.inertia)
            * (1 - state.fixed)[..., None]
        )

        k_norm2 = jnp.sum(k * k, axis=-1, keepdims=True)
        k_norm = jnp.sqrt(k_norm2)
        k_norm = jnp.where(k_norm == 0, 1.0, k_norm)

        # calculate orientation update
        state.q @= Quaternion(
            jnp.cos(k_norm),
            jnp.sin(k_norm) * k / k_norm,
        )

        # normalize the quarternion
        state.q = state.q.unit(state.q)

        return state, system
