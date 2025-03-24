# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from functools import partial

from jaxdem.State import state
from jaxdem.System import system

@partial(jax.jit, inline=True)
def spring_force(i, j, s: 'state', sys: 'system') -> jnp.ndarray:
    r_ij = s.pos[i] - s.pos[j]
    r = jnp.linalg.norm(r_ij)
    s = jnp.maximum(0.0, (s.rad[i] + s.rad[j])/(r + jnp.finfo(s.pos.dtype).eps) - 1.0)
    return sys.k * s * r_ij