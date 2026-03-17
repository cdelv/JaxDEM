# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Utility functions to help with linear algebra."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from functools import partial


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.cross")
def cross(a: jax.Array, b: jax.Array) -> jax.Array:
    """Computes the cross product of two vectors, 'a' and 'b', along their last axis.

    For 3D vectors (D=3), the result is a vector orthogonal to both 'a' and 'b'.
    For 2D vectors (D=2), the result is the scalar magnitude of the 3D cross
    product when a third zero component is assumed, often interpreted as the
    signed area of the parallelogram spanned by the vectors.

    Parameters
    ----------
        a: JAX Array with shape (..., D), where D is the dimension (2 or 3).
        b: JAX Array with shape (..., D), where D must match a's dimension.

    Returns
    -------
        A JAX Array representing the cross product.
        - If D=3: shape is (..., 3).
        - If D=2: shape is (..., 1) (a scalar wrapped in an array).

    Raises
    ------
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
        return a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1]

    a_ext = jnp.concatenate([a, a], axis=-1)
    b_ext = jnp.concatenate([b, b], axis=-1)
    a_left = a_ext[..., 1:4]  # [y, z, x]
    b_left = b_ext[..., 1:4]
    a_right = a_ext[..., 2:5]  # [z, x, y]
    b_right = b_ext[..., 2:5]
    return a_left * b_right - a_right * b_left


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.dot")
def dot(a: jax.Array, b: jax.Array) -> jax.Array:
    """Dot product of vectors along the last axis.

    a, b: (..., D)
    returns: (...), the dot product.
    """
    return jnp.vecdot(a, b)


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.norm2")
def norm2(v: jax.Array) -> jax.Array:
    """Squared norm of vectors along the last axis.

    v: (..., D)
    returns: (...), the squared norm.
    """
    return dot(v, v)


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.norm")
def norm(v: jax.Array) -> jax.Array:
    """Norm of vectors along the last axis.

    v: (..., D)
    returns: (...), the norm.
    """
    return jnp.sqrt(norm2(v))


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.unit")
def unit(v: jax.Array) -> jax.Array:
    """Normalize vectors along the last axis.

    v: (..., D)
    returns: (..., D), unit vectors; zeros map to zeros.
    """
    n2 = norm2(v)[..., None]
    safe_n2 = jnp.where(n2 == 0.0, 1.0, jax.lax.rsqrt(n2))
    return v * safe_n2


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.unit_and_norm")
def unit_and_norm(v: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Normalize vectors along the last axis and return the norm.

    v: (..., D)
    returns: ((..., D), (..., 1)), unit vectors and their norms; zeros map to zeros.
    """
    n2 = norm2(v)[..., None]
    norm_v = jnp.sqrt(n2)
    safe_inv_scale = jnp.where(n2 == 0.0, 1.0, jax.lax.rsqrt(n2))
    return v * safe_inv_scale, norm_v


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.cross_3X3D_1X2D")
def cross_3X3D_1X2D(w: jax.Array, r: jax.Array) -> jax.Array:
    """Computes the cross product of angular velocity vector (w) and a position
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
    ----------
        w: JAX Array. In the 3D case, shape is (..., 3). In the 2D case, shape is (..., 1) or (...).
        r: JAX Array. Shape is (..., 3) or (..., 2).

    Returns
    -------
        A JAX Array representing the tangential velocity (w x r).
        - If r is 3D, the output shape is (..., 3).
        - If r is 2D, the output shape is (..., 2).

    Raises
    ------
        ValueError: If r is not 2D or 3D, or if dimensions are incompatible.

    """
    if r.shape[-1] == 2:
        dim_w = w.shape[-1] if w.ndim > 0 else 0
        if dim_w not in (0, 1):
            raise ValueError(
                f"For a 2D position vector (r.shape[-1]=2), angular velocity 'w' must be a scalar or have a last dimension of 1 (representing w_z). Got last dimension: {dim_w}."
            )

        # (0, 0, w_z) x (r_x, r_y, 0) = (-w_z * r_y, w_z * r_x, 0)
        v_x = -w * r[..., 1:2]
        v_y = w * r[..., 0:1]
        return jnp.concatenate([v_x, v_y], axis=-1)

    return cross(w, r)
