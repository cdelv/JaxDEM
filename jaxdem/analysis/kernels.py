# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Kernel typing + small helpers for JAX analysis.

In the JAX engine, kernels are *pure* functions that operate on arrays directly:

    kernel(arrays, t0, t1, **kwargs) -> pytree of arrays
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import jax.numpy as jnp

from .bessel import j0 as j0_bessel


class KernelFn(Protocol):
    def __call__(self, arrays: Mapping[str, jnp.ndarray], t0: Any, t1: Any, **kwargs: Any) -> Any:
        ...


# ---- Example kernels (Option A layout: (T,N,...) or (T,S,N,...) ) ----


def msd_kernel(arrays: Mapping[str, jnp.ndarray], t0: Any, t1: Any) -> jnp.ndarray:
    """Mean-squared displacement.

    Works for both:
    - pos[t]: (N,d) -> returns () scalar
    - pos[t]: (S,N,d) -> returns (S,) vector
    """

    pos0 = arrays["pos"][t0]
    pos1 = arrays["pos"][t1]
    dr = pos1 - pos0
    dr2 = jnp.sum(dr * dr, axis=-1)  # (N,) or (S,N)
    return jnp.mean(dr2, axis=-1)    # () or (S,)

def _spherical_j0(x: jnp.ndarray) -> jnp.ndarray:
    """Spherical Bessel j0(x) = sin(x)/x with a safe x=0 value."""

    return jnp.where(x == 0, 1.0, jnp.sin(x) / x)


def isf_self_isotropic_kernel(
    arrays: Mapping[str, jnp.ndarray], t0: Any, t1: Any, *, k: Any
) -> jnp.ndarray:
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
            raise ValueError(f"isf_self_isotropic_kernel only supports d=2 or d=3, got d={d}")
        return jnp.mean(phi, axis=-1)  # () or (S,)

    x = r[..., None] * k_arr  # (N,K) or (S,N,K)
    if d == 2:
        phi = j0_bessel(x)
    elif d == 3:
        phi = _spherical_j0(x)
    else:
        raise ValueError(f"isf_self_isotropic_kernel only supports d=2 or d=3, got d={d}")
    return jnp.mean(phi, axis=-2)  # (K,) or (S,K)


def isf_self_kvecs_kernel(
    arrays: Mapping[str, jnp.ndarray], t0: Any, t1: Any, *, kvecs: jnp.ndarray
) -> jnp.ndarray:
    """Self ISF for explicit k-vectors: Fs({k}, t) = <cos(k·dr)>."""

    pos = arrays["pos"]
    dr = pos[t1] - pos[t0]  # (N,d) or (S,N,d)
    phase = jnp.einsum("...nd,kd->...nk", dr, kvecs)  # (N,K) or (S,N,K)
    return jnp.mean(jnp.cos(phase), axis=-2)  # (K,) or (S,K)
