# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional, Dict
from functools import partial

from .Domain import Domain
from .Simulator import Simulator
from .Integrator import Integrator
from .Forces import ForceModel
from .State import State
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .ContactDetection import Grid

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class System:
    """
    Class that holds all the configuration information for the simulation.
    """
    k: float = 500.0   # TO DO: move to Material class
    dt: float = 0.01

    simulator: Optional['Simulator'] = field(
        default = Simulator.create('naive'),
        metadata = {'static': True}
    )
    integrator: Optional['Integrator'] = field(
        default = Integrator.create('euler'),
        metadata = {'static': True}
    )
    force_model: Optional['ForceModel'] = field(
        default = ForceModel.create('spring'),
        metadata = {'static': True}
    )
    domain: Optional['Domain'] = field(
        default = Domain.create('free', dim = 3),
        metadata = {'static': False}
    )
    grid: Optional['Grid'] = field(
        default = None,
        metadata = {'static': False}
    )

    @staticmethod
    @partial(jax.jit, static_argnames=('steps'))
    def step(state: 'State', system: 'System', steps: int = 1) -> Tuple['State', 'System']:
        """
        Advance the simulation for the specified number of steps.
        """
        def body_fun(i, carry):
            state, system = carry
            state, system = system.simulator.step(state, system)
            return (state, system)

        return jax.lax.fori_loop(0, steps, body_fun, (state, system))
