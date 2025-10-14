# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions to help with linear algebra.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from functools import partial


@jax.jit
@partial(jax.named_call, name="utils.unit")
def unit(v: jax.Array) -> jax.Array:
    """
    Normalize vectors along the last axis.
    v: (..., D)
    returns: (..., D), unit vectors; zeros map to zeros.
    """
    # v: (..., D) -> (..., D)
    norm2 = jnp.sum(v * v, axis=-1, keepdims=True)  # (..., 1)
    scale = jnp.where(norm2 == 0, 1.0, jax.lax.rsqrt(norm2))  # (..., 1)
    return v * scale  # broadcast-safe
