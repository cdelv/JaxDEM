# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Quaternion math utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass
from functools import partial

from .linalg import cross, dot


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Quaternion:
    """Quaternion representing particle orientation (body frame to lab frame)."""

    w: jax.Array  # (..., N, 1)
    xyz: jax.Array  # (..., N, 3)

    @staticmethod
    @partial(jax.named_call, name="Quaternion.create")
    def create(w: ArrayLike | None = None, xyz: ArrayLike | None = None) -> Quaternion:
        if w is None:
            w = jnp.ones((1, 1), dtype=float)
        w = jnp.asarray(w, dtype=float)

        if w.ndim == 0:
            w = w[None]

        if xyz is None:
            xyz = jnp.zeros((1, 3), dtype=float)
        xyz = jnp.asarray(xyz, dtype=float)

        return Quaternion(w, xyz)

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.unit")
    def unit(q: Quaternion) -> Quaternion:
        n2 = q.w * q.w + dot(q.xyz, q.xyz)[..., None]
        safe_inv_scale = jnp.where(n2 == 0.0, 1.0, jax.lax.rsqrt(n2))
        return Quaternion(q.w * safe_inv_scale, q.xyz * safe_inv_scale)

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.conj")
    def conj(q: Quaternion) -> Quaternion:
        return Quaternion(q.w, -q.xyz)

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.inv")
    def inv(q: Quaternion) -> Quaternion:
        q = Quaternion.conj(q)
        return Quaternion.unit(q)

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.rotate")
    def rotate(q: Quaternion, v: jax.Array) -> jax.Array:
        """Rotates a vector v from the body reference frame to the lab reference frame."""
        dim = v.shape[-1]
        if dim == 2:
            qw = q.w
            qz = q.xyz[..., 2:3]
            c = qw * qw - qz * qz
            s = 2.0 * qw * qz
            vx = v[..., 0:1]
            vy = v[..., 1:2]
            rx = c * vx - s * vy
            ry = s * vx + c * vy
            return jnp.concatenate([rx, ry], axis=-1)
        if dim == 3:
            # Slices are "views" (no-cost), Concatenates are "copies" (high-cost)
            qx, qy, qz = q.xyz[..., 0:1], q.xyz[..., 1:2], q.xyz[..., 2:3]
            vx, vy, vz = v[..., 0:1], v[..., 1:2], v[..., 2:3]

            # T = q.xyz x v (6 mul, 3 sub)
            tx = qy * vz - qz * vy
            ty = qz * vx - qx * vz
            tz = qx * vy - qy * vx

            # B = q.xyz x T (6 mul, 3 sub)
            bx = qy * tz - qz * ty
            by = qz * tx - qx * tz
            bz = qx * ty - qy * tx

            # Quaternion rotation formula: v + 2wT + 2B
            rx = vx + 2.0 * (q.w * tx + bx)
            ry = vy + 2.0 * (q.w * ty + by)
            rz = vz + 2.0 * (q.w * tz + bz)

            return jnp.concatenate([rx, ry, rz], axis=-1)

        return v

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.rotate_back")
    def rotate_back(q: Quaternion, v: jax.Array) -> jax.Array:
        """Rotates a vector v from the lab reference frame to the body reference frame."""
        q = Quaternion.conj(q)
        return Quaternion.rotate(q, v)

    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.__matmul__")
    def __matmul__(self, other: Quaternion) -> Quaternion:  # q @ r
        w = self.w * other.w - dot(self.xyz, other.xyz)[..., None]
        xyz = self.w * other.xyz + other.w * self.xyz + cross(self.xyz, other.xyz)
        return Quaternion(w, xyz)
