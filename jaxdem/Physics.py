# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from functools import partial
from typing import Tuple

from jaxdem.State import state
from jaxdem.System import system
from jaxdem.Forces import spring_force

@partial(jax.jit, inline=True)
def compute_force_naive(s: 'state', sys: 'system') -> Tuple['state', 'system']:
    s.accel = jax.vmap(
        lambda i: jax.vmap(
            lambda j: 
                spring_force(i, j, s, sys)
        )(jax.lax.iota(dtype=int, size=s.N)).sum(axis=0)
    )(jax.lax.iota(dtype=int, size=s.N))
    return s, sys

@partial(jax.jit, inline=True)
def step_naive(s: 'state', sys: 'system') -> Tuple['state', 'system']:
    s, sys = compute_force_naive(s, sys)

    s.vel += sys.dt*s.accel
    s.pos += sys.dt*s.vel
    
    return s, sys