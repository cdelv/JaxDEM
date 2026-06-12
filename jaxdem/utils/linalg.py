# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Utility functions to help with linear algebra."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.cross")
def cross(a: jax.Array, b: jax.Array) -> jax.Array:
    r"""Computes the cross product of two vectors, 'a' and 'b', along their last axis.

    For 3D vectors (:math:`D=3`), the result is a vector orthogonal to both 'a' and 'b':

    .. math::
        \vec{c} = \vec{a} \times \vec{b} = (a_y b_z - a_z b_y) \mathbf{i} + (a_z b_x - a_x b_z) \mathbf{j} + (a_x b_y - a_y b_x) \mathbf{k}

    For 2D vectors (:math:`D=2`), the result is the scalar magnitude of the 3D cross product:

    .. math::
        c = a_x b_y - a_y b_x

    Parameters
    ----------
    a : jax.Array
        First vector. Shape `(..., D)`, where `D` is the dimension (2 or 3).
    b : jax.Array
        Second vector. Shape `(..., D)`.

    Returns
    -------
    jax.Array
        The cross product.
        - If D=3: shape is `(..., 3)`.
        - If D=2: shape is `(..., 1)`.

    Raises
    ------
    ValueError
        If the last dimension is not 2 or 3, or if the dimensions of 'a' and 'b' do not match.
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
    r"""Dot product of vectors along the last axis.

    .. math::
        c = \vec{a} \cdot \vec{b} = \sum_{i} a_i b_i

    Parameters
    ----------
    a : jax.Array
        First vector. Shape `(..., D)`.
    b : jax.Array
        Second vector. Shape `(..., D)`.

    Returns
    -------
    jax.Array
        The dot product. Shape `(...)`.
    """
    return jnp.vecdot(a, b)


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.norm2")
def norm2(v: jax.Array) -> jax.Array:
    r"""Squared norm of vectors along the last axis.

    .. math::
        \|v\|^2 = \vec{v} \cdot \vec{v} = \sum_{i} v_i^2

    Parameters
    ----------
    v : jax.Array
        Input vector. Shape `(..., D)`.

    Returns
    -------
    jax.Array
        The squared norm. Shape `(...)`.
    """
    return dot(v, v)


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.norm")
def norm(v: jax.Array) -> jax.Array:
    r"""Norm (magnitude) of vectors along the last axis.

    .. math::
        \|v\| = \sqrt{\sum_{i} v_i^2}

    Parameters
    ----------
    v : jax.Array
        Input vector. Shape `(..., D)`.

    Returns
    -------
    jax.Array
        The norm. Shape `(...)`.
    """
    n2 = norm2(v)
    # Double-where: sqrt must never see 0, even in the untaken branch,
    # otherwise reverse-mode gradients at v=0 are NaN (d√x/dx -> inf).
    safe_n2 = jnp.where(n2 == 0.0, 1.0, n2)
    return jnp.where(n2 == 0.0, 0.0, jnp.sqrt(safe_n2))


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.unit")
def unit(v: jax.Array) -> jax.Array:
    r"""Normalize vectors to unit vectors along the last axis.

    If the vector is zero, the result is zero.

    .. math::
        \hat{v} = \begin{cases} \frac{\vec{v}}{\|v\|} & \text{if } \|v\| > 0 \\ \vec{0} & \text{otherwise} \end{cases}

    Parameters
    ----------
    v : jax.Array
        Input vector. Shape `(..., D)`.

    Returns
    -------
    jax.Array
        Unit vector. Shape `(..., D)`.
    """
    n2 = norm2(v)[..., None]
    # Double-where: rsqrt must never see 0, even in the untaken branch,
    # otherwise reverse-mode gradients at v=0 are NaN.
    safe_n2 = jnp.where(n2 == 0.0, 1.0, n2)
    inv_norm = jnp.where(n2 == 0.0, 0.0, jax.lax.rsqrt(safe_n2))
    return v * inv_norm


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.unit_and_norm")
def unit_and_norm(v: jax.Array) -> tuple[jax.Array, jax.Array]:
    r"""Normalize vectors along the last axis and return both unit vectors and norms.

    Parameters
    ----------
    v : jax.Array
        Input vector. Shape `(..., D)`.

    Returns
    -------
    Tuple[jax.Array, jax.Array]
        A tuple of (unit vectors, norms).
    """
    n2 = norm2(v)[..., None]
    # Double-where: sqrt/rsqrt must never see 0, even in the untaken branch,
    # otherwise reverse-mode gradients at v=0 are NaN.
    safe_n2 = jnp.where(n2 == 0.0, 1.0, n2)
    norm_v = jnp.where(n2 == 0.0, 0.0, jnp.sqrt(safe_n2))
    inv_norm = jnp.where(n2 == 0.0, 0.0, jax.lax.rsqrt(safe_n2))
    return v * inv_norm, norm_v


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="utils.cross_3X3D_1X2D")
def cross_3X3D_1X2D(w: jax.Array, r: jax.Array) -> jax.Array:
    r"""Computes the cross product of angular velocity vector (w) and a position
    vector (r), often used to find tangential velocity: v = w x r.

    For 3D vectors, standard 3D cross product is used:

    .. math::
        \vec{v} = \vec{w} \times \vec{r}

    For 2D vectors, angular velocity :math:`w` is a scalar (z-component) and position :math:`\vec{r}` is 2D:

    .. math::
        \vec{v} = (-w \cdot r_y, \, w \cdot r_x)

    Parameters
    ----------
    w : jax.Array
        Angular velocity. Shape `(..., 3)` in 3D; in 2D either a true scalar
        (shape `()`) or with a trailing singleton axis, shape `(..., 1)`.
        A bare `(N,)` array is rejected.
    r : jax.Array
        Position vector. Shape `(..., 3)` or `(..., 2)`.

    Returns
    -------
    jax.Array
        Tangential velocity. Shape matches `r`.

    Raises
    ------
    ValueError
        If dimensions are incompatible.
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
