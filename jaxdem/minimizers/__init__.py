"""Energy-minimizer interfaces and implementations."""

from __future__ import annotations

import jax

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

from ..factory import Factory
from ..integrators import Integrator

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


# ADD AN IS_CONVERGED METHOD TO THE MINIMIZER BASE CLASS OR MAYBE A BASE CLASS FOR CONVERGENCE CHECKS (i.e. PE OR PRESSURE)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Minimizer(Integrator, ABC):
    """
    Abstract base class for energy minimizers.

    Notes
    -----
    - `Minimizer` subclasses the generic `Integrator` interface, so it can be
      plugged in anywhere an `Integrator` is expected (e.g., as `System.linear_integrator`).
    - The default implementations of `step_before_force`, `step_after_force`,
      and `initialize` are inherited from `Integrator` and act as no-ops.
    - Concrete minimizers should typically override `step_after_force` to
      update the state based on the current forces in an energy-decreasing way.
    """


class LinearMinimizer(Minimizer):
    """
    Namespace for translation/linear-state minimizers.

    Concrete minimizers (e.g., GradientDescent) should subclass this to
    signal that they operate on linear kinematics.
    """


class RotationMinimizer(Minimizer):
    """
    Namespace for rotational-state minimizers.

    Concrete minimizers that relax orientations / angular DOFs should
    subclass this.
    """


from .gradient_descent import LinearGradientDescent, RotationGradientDescent
from .fire import LinearFIRE
from .routines import minimize

__all__ = ["Minimizer", "LinearMinimizer", "RotationMinimizer", "LinearGradientDescent", "RotationGradientDescent", "LinearFIRE", "minimize"]