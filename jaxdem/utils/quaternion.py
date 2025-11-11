# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions to handle environments.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .linalg import unit


@jax.tree_util.register_dataclass
@dataclass
class Quaternion:
    w: jax.Array  # (..., N, 1)
    xyz: jax.Array  # (..., N, 3)

    @staticmethod
    def create(w=None, xyz=None) -> Quaternion:
        if w is None:
            w = jnp.ones((1, 1), dtype=float)
        w = jnp.asarray(w, dtype=float)

        if xyz is None:
            xyz = jnp.zeros((1, 3), dtype=float)
        xyz = jnp.asarray(xyz, dtype=float)
        return Quaternion(w, xyz)

    @staticmethod
    @jax.jit
    def unit(q: Quaternion) -> Quaternion:
        qvec = unit(jnp.concatenate([q.w, q.xyz], axis=-1))
        w, xyz = jnp.split(qvec, [1], axis=-1)
        return Quaternion(w, xyz)

    @staticmethod
    @jax.jit
    def conj(q: Quaternion) -> Quaternion:
        return Quaternion(q.w, -q.xyz)

    @staticmethod
    @jax.jit
    def inv(q: Quaternion) -> Quaternion:
        q = Quaternion.conj(q)
        return Quaternion.unit(q)

    @staticmethod
    @jax.jit
    def rotate(q: Quaternion, v: jax.Array) -> jax.Array:
        dim = v.shape[-1]
        if dim == 2:
            angle = 2.0 * jnp.arctan2(q.xyz[..., 2], q.w[..., 0])
            c, s = jnp.cos(angle), jnp.sin(angle)
            x, y = v[..., 0], v[..., 1]
            return jnp.stack([c * x - s * y, s * x + c * y], axis=-1)

        if dim == 3:
            return (
                v
                + 2 * q.w * jnp.linalg.cross(q.xyz, v)
                + jnp.linalg.cross(2 * q.xyz, jnp.linalg.cross(q.xyz, v))
            )

        return v

    @staticmethod
    @jax.jit
    def rotate_back(q: Quaternion, v: jax.Array) -> jax.Array:
        q = Quaternion.conj(q)
        return Quaternion.rotate(q, v)

    @staticmethod
    @jax.jit
    def multiply(a: Quaternion, b: Quaternion):
        w = a.w * b.w - jnp.sum(a.xyz * b.xyz, axis=-1, keepdims=True)
        xyz = a.w * b.xyz + b.w * a.xyz + jnp.linalg.cross(a.xyz, b.xyz, axis=-1)
        return Quaternion(w, xyz)

    def __matmul__(self, other):  # q @ r
        return Quaternion.multiply(self, other)
