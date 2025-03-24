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

from jaxdem.State import state

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class system:
    """
    Class that holds all the configuration information for the simulation.
    ...
    """
    k: float = 500.0
    dt:float = 0.01
    step_fn: Optional[Callable[['state', 'system'], Tuple['state', 'system']]] = field(
        default = None,
        metadata = {'static': True}
    )

    @staticmethod
    @partial(jax.jit, static_argnames=('steps'))
    def step(s: 'state', sys: 'system', steps: int = 1) -> Tuple['state', 'system']:
        """
        Advance the simulation for the specified number of steps.
        """
        def body_fun(i, carry):
            s, sys = carry
            s, sys = sys.step_fn(s, sys)
            return (s, sys)
        
        # Run 'steps' iterations of the body function
        s, sys = jax.lax.fori_loop(0, steps, body_fun, (s, sys))
        
        return s, sys
