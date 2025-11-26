# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Basic gradient-descent energy minimizer."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Tuple

from . import LinearMinimizer
from ..integrators import LinearIntegrator

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@LinearMinimizer.register("lineargradientdescent")
@LinearIntegrator.register("lineargradientdescent")  # also register as linear integrator
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LinearGradientDescent(LinearMinimizer):

    learning_rate: float = 1e-3

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="LinearGradientDescent.step_after_force")
    def step_after_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """Gradient-descent update using the integrator's learning rate.

        The learning rate is stored on the LinearGradientDescent dataclass
        attached to ``system.linear_integrator``, so no mutable state is kept
        outside the System PyTree.
        """
        gd = system.linear_integrator
        lr = gd.learning_rate
        mask = (1 - state.fixed)[..., None]
        state.pos = state.pos + lr * state.force * mask
        return state, system