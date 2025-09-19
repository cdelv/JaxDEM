# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions to compute angles between vectors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .linalg import unit


@jax.jit
def signed_angle(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    r"""
    Directional angle from v1 -> v2 around normal :math:`\hat{z}` (right-hand rule), in :math:`[-\pi, \pi)`.
    """
    v1 = unit(v1)
    v2 = unit(v2)
    dot = jnp.vecdot(v1, v2)
    sin = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]  # ẑ·(a×b)
    return jnp.arctan2(sin, dot)  # (-π, π]


@jax.jit
def signed_angle_x(v1: jnp.ndarray) -> jnp.ndarray:
    r"""Directional angle from v1 -> :math:`\hat{x}` around normal :math:`\hat{z}`, in :math:`(-\pi, \pi]`."""
    return jnp.arctan2(-v1[..., 1], v1[..., 0])


@jax.jit
def angle(v1: jax.Array, v2: jax.Array) -> jax.Array:
    r"""
    angle from v1 -> v2 in :math:`[0, \pi]`
    """
    v1 = unit(v1)
    v2 = unit(v2)
    y = jnp.linalg.norm(v1 - v2, axis=-1)
    x = jnp.linalg.norm(v1 + v2, axis=-1)
    return 2.0 * jnp.atan2(y, x)


@jax.jit
def angle_x(v1: jax.Array) -> jax.Array:
    r"""
    angle from v1 -> :math:`\hat{x}` in :math:`[0, \pi]`
    """
    v1 = unit(v1)
    v2 = jnp.zeros(v1.shape[-1], dtype=v1.dtype).at[0].set(1.0)
    y = jnp.linalg.norm(v1 - v2, axis=-1)
    x = jnp.linalg.norm(v1 + v2, axis=-1)
    return 2.0 * jnp.atan2(y, x)


__all__ = ["signed_angle", "signed_angle_x", "angle", "angle_x"]
