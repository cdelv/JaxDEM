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
    # shape (4,), order (w, x, y, z)
    q: jax.Array = field(
        default_factory=lambda: jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    )

    @staticmethod
    @jax.jit
    def unit(q: Quaternion) -> Quaternion:
        q.q = unit(q.q)
        return q

    @staticmethod
    @jax.jit
    def inv(q: Quaternion) -> Quaternion:
        with jax.ensure_compile_time_eval():
            mask = jnp.asarray([1.0, -1.0, -1.0, -1.0], dtype=float)
        q.q = q.q * mask
        return q

    @staticmethod
    @jax.jit
    def rotate(q: Quaternion, v: jax.Array) -> jax.Array:
        w, x, y, z = jnp.split(q.q, 4, -1)
        t = 2.0 * jnp.cross(jnp.concatenate([x, y, z], -1), v)
        return v + w[..., 0:1] * t + jnp.cross(jnp.concatenate([x, y, z], -1), t)

    @staticmethod
    @jax.jit
    def rotate_back(q: Quaternion, v: jax.Array) -> jax.Array:
        q = q.inv(q)
        return q.rotate(q, v)

    @staticmethod
    @jax.jit
    def multiply(a, b):
        aw, ax, ay, az = jnp.split(a, 4, axis=-1)
        bw, bx, by, bz = jnp.split(b, 4, axis=-1)
        w = aw * bw - ax * bx - ay * by - az * bz
        x = aw * bx + ax * bw + ay * bz - az * by
        y = aw * by - ax * bz + ay * bw + az * bx
        z = aw * bz + ax * by - ay * bx + az * bw
        return Quaternion(jnp.concatenate([w, x, y, z], -1))

    def __matmul__(self, other):  # q @ r
        return Quaternion.multiply(self, other)
