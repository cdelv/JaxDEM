# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Kernel typing + small helpers for JAX analysis.

In the JAX engine, kernels are *pure* functions that operate on arrays directly:

    kernel(arrays, t0, t1, **kwargs) -> pytree of arrays
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from typing import Any, Mapping, Protocol

from .bessel import j0 as j0_bessel


class KernelFn(Protocol):
    def __call__(
        self, arrays: Mapping[str, jax.Array], t0: Any, t1: Any, **kwargs: Any
    ) -> Any: ...


# ---- Example kernels (Option A layout: (T,N,...) or (T,S,N,...) ) ----


def msd_kernel(arrays: Mapping[str, jax.Array], t0: Any, t1: Any) -> jax.Array:
    """Mean-squared displacement.

    Works for both:
    - pos[t]: (N,d) -> returns () scalar
    - pos[t]: (S,N,d) -> returns (S,) vector
    """

    pos0 = arrays["pos"][t0]
    pos1 = arrays["pos"][t1]
    dr = pos1 - pos0
    dr -= jnp.mean(dr, axis=-2, keepdims=True)  # subtract drift
    dr2 = jnp.sum(dr * dr, axis=-1)  # (N,) or (S,N)
    return jnp.mean(dr2, axis=-1)  # () or (S,)


def _spherical_j0(x: jax.Array) -> jax.Array:
    """Spherical Bessel j0(x) = sin(x)/x with a safe x=0 value."""

    return jnp.where(x == 0, 1.0, jnp.sin(x) / x)


def isf_self_isotropic_kernel(
    arrays: Mapping[str, jax.Array], t0: Any, t1: Any, *, k: Any
) -> jax.Array:
    """Self intermediate scattering function (isotropic average).

    For isotropic averaging:
    - 2D: Fs(k, t) = < J0(k * |dr|) >
    - 3D: Fs(k, t) = < sin(k|dr|)/(k|dr|) >

    Supports:
    - pos[t]: (N,d)  -> returns () if k scalar else (K,)
    - pos[t]: (S,N,d)-> returns (S,) if k scalar else (S,K)
    """

    pos = arrays["pos"]
    dr = pos[t1] - pos[t0]  # (N,d) or (S,N,d)
    d = int(dr.shape[-1])

    r = jnp.linalg.norm(dr, axis=-1)  # (N,) or (S,N)
    k_arr = jnp.asarray(k)

    if k_arr.ndim == 0:
        x = r * k_arr  # (N,) or (S,N)
        if d == 2:
            phi = j0_bessel(x)
        elif d == 3:
            phi = _spherical_j0(x)
        else:
            raise ValueError(
                f"isf_self_isotropic_kernel only supports d=2 or d=3, got d={d}"
            )
        return jnp.mean(phi, axis=-1)  # () or (S,)

    x = r[..., None] * k_arr  # (N,K) or (S,N,K)
    if d == 2:
        phi = j0_bessel(x)
    elif d == 3:
        phi = _spherical_j0(x)
    else:
        raise ValueError(
            f"isf_self_isotropic_kernel only supports d=2 or d=3, got d={d}"
        )
    return jnp.mean(phi, axis=-2)  # (K,) or (S,K)


def isf_self_kvecs_kernel(
    arrays: Mapping[str, jax.Array], t0: Any, t1: Any, *, kvecs: jax.Array
) -> jax.Array:
    """Self ISF for explicit k-vectors: Fs({k}, t) = <cos(k·dr)>."""

    pos = arrays["pos"]
    dr = pos[t1] - pos[t0]  # (N,d) or (S,N,d)
    phase = jnp.einsum("...nd,kd->...nk", dr, kvecs)  # codespell:ignore nd
    return jnp.mean(jnp.cos(phase), axis=-2)  # (K,) or (S,K)


def unwrap_angles_2d(q_w: jax.Array, q_xyz: jax.Array) -> jax.Array:
    """Convert (T, N, 1) and (T, N, 3) quaternion trajectory to unwrapped cumulative angle (T, N)."""
    theta_wrapped = 2.0 * jnp.arctan2(q_xyz[..., 2], q_w[..., 0])
    dtheta = jnp.diff(theta_wrapped, axis=0)
    dtheta = (dtheta + jnp.pi) % (2 * jnp.pi) - jnp.pi
    cumulative = jnp.concatenate(
        [theta_wrapped[0:1], theta_wrapped[0:1] + jnp.cumsum(dtheta, axis=0)], axis=0
    )
    return cumulative


def msad_kernel_2d(arrays: Mapping[str, jax.Array], t0: Any, t1: Any) -> jax.Array:
    """Mean-squared angular displacement on unwrapped cumulative angle."""
    theta0 = arrays["theta"][t0]
    theta1 = arrays["theta"][t1]
    dtheta = theta1 - theta0
    return jnp.mean(dtheta * dtheta, axis=-1)


def isf_angular_kernel_2d(
    arrays: Mapping[str, jax.Array], t0: Any, t1: Any, *, theta_0: Any
) -> jax.Array:
    """Angular ISF: <cos(θ₀ · Δθ)>"""
    dtheta = arrays["theta"][t1] - arrays["theta"][t0]
    theta_0_arr = jnp.asarray(theta_0)
    if theta_0_arr.ndim == 0:
        return jnp.mean(jnp.cos(theta_0_arr * dtheta), axis=-1)
    return jnp.mean(jnp.cos(dtheta[..., None] * theta_0_arr), axis=-2)
