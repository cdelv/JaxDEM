# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions to help with linear algebra.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def unit(v: jax.Array) -> jax.Array:
    """
    Normalize vectors along the last axis.
    v: (..., D)
    returns: (..., D), unit vectors; zeros map to zeros.
    """
    norm2 = jnp.vecdot(v, v)
    return v * jnp.where(norm2 == 0, 1.0, jax.lax.rsqrt(norm2))
