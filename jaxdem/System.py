# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional
from functools import partial

from .Space import Domain
from .Simulate import Simulator
from .Integrator import Integrator
from .Forces import ForceModel
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .State import State

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class System:
    """
    Class that holds all the configuration information for the simulation.
    """
    k: float = 500.0
    dt:float = 0.01
    domain: Optional['Domain'] = field(
        default = Domain.create('free'),
        metadata = {'static': True}
    )

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

    @staticmethod
    @partial(jax.jit, static_argnames=('steps'))
    def step(s: 'State', sys: 'System', steps: int = 1) -> Tuple['State', 'System']:
        """
        Advance the simulation for the specified number of steps.
        """
        def body_fun(i, carry):
            s, sys = carry
            s, sys = sys.simulator.step(s, sys)
            return (s, sys)
        
        # Run 'steps' iterations of the body function
        s, sys = jax.lax.fori_loop(0, steps, body_fun, (s, sys))
        
        return s, sys
