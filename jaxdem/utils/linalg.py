# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions to help with linear algebra.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from functools import partial


@partial(jax.jit, inline=True)
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


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.cross_3X3D_1X2D")
def cross_3X3D_1X2D(w, r):
    if r.shape[-1] == 3:
        return jnp.linalg.cross(w, r)
    elif r.shape[-1] == 2:
        # (0, 0, c) x (a, b, 0) = (-bc, ac, 0)
        return w * jnp.stack([-r[..., 1], r[..., 0]], axis=-1)
