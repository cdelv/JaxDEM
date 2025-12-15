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
@partial(jax.named_call, name="utils.cross")
def cross(a: jax.Array, b: jax.Array) -> jax.Array:
    """
    Computes the cross product of two vectors, 'a' and 'b', along their last axis.

    For 3D vectors (D=3), the result is a vector orthogonal to both 'a' and 'b'.
    For 2D vectors (D=2), the result is the scalar magnitude of the 3D cross
    product when a third zero component is assumed, often interpreted as the
    signed area of the parallelogram spanned by the vectors.

    Parameters
    -----------
        a: JAX Array with shape (..., D), where D is the dimension (2 or 3).
        b: JAX Array with shape (..., D), where D must match a's dimension.

    Returns
    --------
        A JAX Array representing the cross product.
        - If D=3: shape is (..., 3).
        - If D=2: shape is (..., 1) (a scalar wrapped in an array).

    Raises
    -------
        ValueError: If the last dimension (D) is not 2 or 3, or if the last dimensions of 'a' and 'b' do not match.
    """
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            f"The last dimension of 'a' ({a.shape[-1]}) must match the last dimension of 'b' ({b.shape[-1]})."
        )

    if a.shape[-1] not in (2, 3):
        raise ValueError(
            f"Cross product is only defined for 2D or 3D vectors. The last dimension of the input array is {a.shape[-1]}."
        )

    if a.shape[-1] == 2:
        result = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        return jnp.expand_dims(result, axis=-1)
    else:
        a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
        b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
        c1 = a1 * b2 - a2 * b1
        c2 = a2 * b0 - a0 * b2
        c3 = a0 * b1 - a1 * b0
        return jnp.stack([c1, c2, c3], axis=-1)


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
    return v * scale


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.cross_3X3D_1X2D")
def cross_3X3D_1X2D(w: jax.Array, r: jax.Array) -> jax.Array:
    """
    Computes the cross product of angular velocity vector (w) and a position
    vector (r), often used to find tangential velocity: v = w x r.

    This function handles two scenarios based on the dimension of 'r':

    1.  **3D Case (r.shape[-1] == 3):**
        - w must be a 3D vector (w.shape[-1] == 3).
        - Computes the standard 3D cross product: w x r.

    2.  **2D Case (r.shape[-1] == 2):**
        - w is treated as a *scalar* (the z-component of angular velocity, w_z).
        - The computation is equivalent to: (0, 0, w_z) x (r_x, r_y, 0).
        - The result is the 2D tangential velocity vector (v_x, v_y) in the xy-plane.

    Parameters
    -----------
        w: JAX Array. In the 3D case, shape is (..., 3). In the 2D case, shape is (..., 1) or (...).
        r: JAX Array. Shape is (..., 3) or (..., 2).

    Returns
    --------
        A JAX Array representing the tangential velocity (w x r).
        - If r is 3D, the output shape is (..., 3).
        - If r is 2D, the output shape is (..., 2).

    Raises:
        ValueError: If r is not 2D or 3D, or if dimensions are incompatible.
    """
    if r.shape[-1] == 2:
        dim_w = w.shape[-1] if w.ndim > 0 else 0
        if dim_w not in (0, 1):
            raise ValueError(
                f"For a 2D position vector (r.shape[-1]=2), angular velocity 'w' must be a scalar or have a last dimension of 1 (representing w_z). Got last dimension: {dim_w}."
            )

        # (0, 0, w_z) x (r_x, r_y, 0) = (-w_z * r_y, w_z * r_x, 0)
        r_perp = jnp.stack([-r[..., 1], r[..., 0]], axis=-1)
        return w * r_perp

    else:
        return cross(r, w)
