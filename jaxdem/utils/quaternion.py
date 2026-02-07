# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions to handle environments.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass
from functools import partial

from .linalg import unit, cross


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Quaternion:
    """
    Quaternion representing the orientation of a particle. Stores the rotation body to lab.
    """

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
        qvec = unit(jnp.concatenate([q.w, q.xyz], axis=-1))
        w, xyz = jnp.split(qvec, [1], axis=-1)
        return Quaternion(w, xyz)

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
        """
        Rotates a vector v from the body reference frame to the lab reference frame.
        """
        dim = v.shape[-1]
        if dim == 2:
            qw = q.w[..., 0]
            qz = q.xyz[..., -1]
            c = qw * qw - qz * qz
            s = 2.0 * qw * qz
            vx = v[..., 0]
            vy = v[..., 1]
            return jnp.stack([c * vx - s * vy, s * vx + c * vy], axis=-1)
        if dim == 3:
            xyz = q.xyz
            T = (
                xyz[..., [1, 2, 0]] * v[..., [2, 0, 1]]
                - xyz[..., [2, 0, 1]] * v[..., [1, 2, 0]]
            )
            B = (
                xyz[..., [1, 2, 0]] * T[..., [2, 0, 1]]
                - xyz[..., [2, 0, 1]] * T[..., [1, 2, 0]]
            )
            return v + 2.0 * (q.w * T + B)

        return v

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.rotate_back")
    def rotate_back(q: Quaternion, v: jax.Array) -> jax.Array:
        """
        Rotates a vector v from the lab reference frame to the body reference frame.
        """
        q = Quaternion.conj(q)
        return Quaternion.rotate(q, v)

    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.__matmul__")
    def __matmul__(self, other: Quaternion) -> Quaternion:  # q @ r
        w1, w2 = self.w, other.w
        xyz1, xyz2 = self.xyz, other.xyz
        w = w1 * w2 - jnp.sum(xyz1 * xyz2, axis=-1)[..., None]
        xyz = w1 * xyz2 + w2 * xyz1 + cross(xyz1, xyz2)
        return Quaternion(w, xyz)
