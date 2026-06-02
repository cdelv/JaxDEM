# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Utility functions to compute angles between vectors."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from functools import partial

from .linalg import dot, norm, unit


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.signed_angle")
def signed_angle(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    r"""Directional angle from v1 -> v2 around normal :math:`\hat{z}` (right-hand rule), in :math:`[-\pi, \pi)`.

    Calculates the signed angle between two 2D vectors :math:`\vec{v}_1$ and :math:`\vec{v}_2$ using the dot product and the 2D cross product:

    .. math::
        \hat{v}_1 &= \text{unit}(\vec{v}_1) \\
        \hat{v}_2 &= \text{unit}(\vec{v}_2) \\
        d &= \hat{v}_1 \cdot \hat{v}_2 \\
        s &= \hat{v}_{1,x} \hat{v}_{2,y} - \hat{v}_{1,y} \hat{v}_{2,x} \\
        \theta &= \text{atan2}(s, d)

    Parameters
    ----------
    v1 : jnp.ndarray
        First vector. Shape `(..., 2)`.
    v2 : jnp.ndarray
        Second vector. Shape `(..., 2)`.

    Returns
    -------
    jnp.ndarray
        Signed angle in radians.
    """
    v1 = unit(v1)
    v2 = unit(v2)
    d = dot(v1, v2)
    sin = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]  # ẑ·(a×b)
    return jnp.arctan2(sin, d)  # (-π, π]


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.signed_angle_x")
def signed_angle_x(v1: jnp.ndarray) -> jnp.ndarray:
    r"""Directional angle from v1 -> :math:`\hat{x}` around normal :math:`\hat{z}`, in :math:`(-\pi, \pi]`.

    Calculates the signed angle of a 2D vector :math:`\vec{v}_1$ relative to the positive x-axis :math:`(1, 0)$:

    .. math::
        \theta = \text{atan2}(-v_{1,y}, v_{1,x})

    Parameters
    ----------
    v1 : jnp.ndarray
        The input vector. Shape `(..., 2)`.

    Returns
    -------
    jnp.ndarray
        Signed angle in radians.
    """
    return jnp.arctan2(-v1[..., 1], v1[..., 0])


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.angle")
def angle(v1: jax.Array, v2: jax.Array) -> jax.Array:
    r"""Angle from v1 -> v2 in :math:`[0, \pi]`.

    Calculates the unsigned angle between two vectors :math:`\vec{v}_1$ and :math:`\vec{v}_2$ using a numerically stable half-angle formula:

    .. math::
        \hat{v}_1 &= \text{unit}(\vec{v}_1) \\
        \hat{v}_2 &= \text{unit}(\vec{v}_2) \\
        y &= \|\hat{v}_1 - \hat{v}_2\| \\
        x &= \|\hat{v}_1 + \hat{v}_2\| \\
        \theta &= 2 \cdot \text{atan2}(y, x)

    Parameters
    ----------
    v1 : jax.Array
        First vector. Shape `(..., dim)`.
    v2 : jax.Array
        Second vector. Shape `(..., dim)`.

    Returns
    -------
    jax.Array
        Unsigned angle in radians.
    """
    v1 = unit(v1)
    v2 = unit(v2)
    y = norm(v1 - v2)
    x = norm(v1 + v2)
    return 2.0 * jnp.atan2(y, x)


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.angle_x")
def angle_x(v1: jax.Array) -> jax.Array:
    r"""Angle from v1 -> :math:`\hat{x}` in :math:`[0, \pi]`.

    Calculates the unsigned angle of a vector :math:`\vec{v}_1$ relative to the positive x-axis :math:`(1, 0, \dots)$:

    .. math::
        \hat{v}_1 &= \text{unit}(\vec{v}_1) \\
        y &= \sqrt{2(1 - \hat{v}_{1,x})} \\
        x &= \sqrt{2(1 + \hat{v}_{1,x})} \\
        \theta &= 2 \cdot \text{atan2}(y, x)

    Parameters
    ----------
    v1 : jax.Array
        The input vector. Shape `(..., dim)`.

    Returns
    -------
    jax.Array
        Unsigned angle in radians.
    """
    v1 = unit(v1)
    return 2.0 * jnp.atan2(
        jnp.sqrt(2.0 - 2.0 * v1[..., 0]), jnp.sqrt(2.0 + 2.0 * v1[..., 0])
    )


__all__ = ["angle", "angle_x", "signed_angle", "signed_angle_x"]
