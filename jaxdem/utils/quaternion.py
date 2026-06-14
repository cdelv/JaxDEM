# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Quaternion math utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .linalg import cross, dot


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Quaternion:
    r"""Quaternion representation for 2D and 3D particle orientations.

    A quaternion :math:`q` is represented as a scalar part :math:`w` and a vector part
    :math:`\vec{v}_{xyz} = (x, y, z)`:

    .. math::
        q = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}

    In 2D, the rotation axis is restricted to the z-axis: :math:`\vec{v}_{xyz} = (0, 0, z)`.

    Attributes
    ----------
    w : jax.Array
        The scalar component of the quaternion. Shape is `(..., N, 1)`.
    xyz : jax.Array
        The vector components of the quaternion. Shape is `(..., N, 3)`.
    """

    w: jax.Array  # (..., N, 1)
    xyz: jax.Array  # (..., N, 3)

    @staticmethod
    @partial(jax.named_call, name="Quaternion.create")
    def create(w: ArrayLike | None = None, xyz: ArrayLike | None = None) -> Quaternion:
        """Create a Quaternion instance.

        Parameters
        ----------
        w : ArrayLike, optional
            The scalar component. If None, defaults to ones.
        xyz : ArrayLike, optional
            The vector component. If None, defaults to zeros.

        Returns
        -------
        Quaternion
            The created quaternion.
        """
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
        r"""Normalize a quaternion to have unit norm.

        .. math::
            q_{unit} = \frac{q}{\|q\|} = \frac{q}{\sqrt{w^2 + x^2 + y^2 + z^2}}

        Parameters
        ----------
        q : Quaternion
            The quaternion to normalize.

        Returns
        -------
        Quaternion
            The normalized unit quaternion.
        """
        n2 = q.w[..., 0] * q.w[..., 0] + dot(q.xyz, q.xyz)
        # Double-where: rsqrt must never see 0, even in the untaken branch,
        # otherwise reverse-mode gradients at q=0 are NaN.
        safe_n2 = jnp.where(n2 == 0.0, 1.0, n2)
        inv_norm = jnp.where(n2 == 0.0, 0.0, jax.lax.rsqrt(safe_n2))
        return Quaternion(q.w * inv_norm[..., None], q.xyz * inv_norm[..., None])

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.from_rotvec")
    def from_rotvec(rotvec: jax.Array) -> Quaternion:
        r"""Build a unit quaternion from an axis-angle rotation vector.

        For a rotation vector :math:`\vec{\theta} = \theta \hat{u}` (angle
        :math:`\theta = \|\vec{\theta}\|` about the unit axis :math:`\hat{u}`):

        .. math::
            q = \left(\cos\frac{\theta}{2},\; \hat{u} \sin\frac{\theta}{2}\right)

        Parameters
        ----------
        rotvec : jax.Array
            Rotation vector(s) of shape `(..., 3)`.

        Returns
        -------
        Quaternion
            The corresponding unit quaternion with `w` of shape `(..., 1)`
            and `xyz` of shape `(..., 3)`.
        """
        # 1. Compute squared norm safely
        n2 = dot(rotvec, rotvec)
        safe_n2 = jnp.where(n2 == 0.0, 1.0, n2)
        theta = jnp.sqrt(safe_n2)

        half = 0.5 * theta

        # 2. Evaluate sin(theta/2) / theta safely.
        # At theta=0, this explicitly routes to 0.5, ensuring the correct
        # forward value and correct reverse-mode gradient.
        sinc_factor = jnp.where(n2 == 0.0, 0.5, jnp.sin(half) / theta)

        return Quaternion(jnp.cos(half)[..., None], rotvec * sinc_factor[..., None])

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.conj")
    def conj(q: Quaternion) -> Quaternion:
        r"""Compute the conjugate of a quaternion.

        .. math::
            q^* = w - x\mathbf{i} - y\mathbf{j} - z\mathbf{k}

        Parameters
        ----------
        q : Quaternion
            The quaternion.

        Returns
        -------
        Quaternion
            The conjugate quaternion.
        """
        return Quaternion(q.w, -q.xyz)

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.inv")
    def inv(q: Quaternion) -> Quaternion:
        r"""Compute the inverse of a quaternion.

        For a unit quaternion, the inverse is equal to its conjugate:

        .. math::
            q^{-1} = \frac{q^*}{\|q\|^2}

        Parameters
        ----------
        q : Quaternion
            The quaternion.

        Returns
        -------
        Quaternion
            The inverse quaternion.
        """
        n2 = q.w[..., 0] * q.w[..., 0] + dot(q.xyz, q.xyz)
        safe_n2 = jnp.where(n2 == 0.0, 1.0, n2)
        inv_n2 = jnp.where(n2 == 0.0, 0.0, 1.0 / safe_n2)
        q = Quaternion.conj(q)
        return Quaternion(q.w * inv_n2[..., None], q.xyz * inv_n2[..., None])

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.rotate")
    def rotate(q: Quaternion, v: jax.Array) -> jax.Array:
        r"""Rotates a vector :math:`\vec{v}` from the body reference frame to the lab reference frame.

        In 3D, the rotation of a vector :math:`\vec{v}` by a unit quaternion :math:`q = (w, \vec{q}_{xyz})` is:

        .. math::
            \vec{v}' = \vec{v} + 2 w (\vec{q}_{xyz} \times \vec{v}) + 2 (\vec{q}_{xyz} \times (\vec{q}_{xyz} \times \vec{v}))

        In 2D, where rotation is restricted to the z-axis, the rotation by angle :math:`\theta`
        (corresponding to quaternion components :math:`w = \cos(\theta/2)` and :math:`q_z = \sin(\theta/2)`) is:

        .. math::
            x' &= x \cos(\theta) - y \sin(\theta) \\
            y' &= x \sin(\theta) + y \cos(\theta)

        Parameters
        ----------
        q : Quaternion
            The rotation quaternion.
        v : jax.Array
            The vector to rotate. Shape is `(..., dim)`.

        Returns
        -------
        jax.Array
            The rotated vector in the lab frame. Shape is `(..., dim)`.
        """
        dim = v.shape[-1]
        if dim == 0:
            # Wildcard empty state sentinel (State.create() with no particles
            # and no dim): shape (..., 0) carries no vectors to rotate.
            return v
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

        raise ValueError(
            "Quaternion rotation is only defined for 2D or 3D vectors. "
            f"The last dimension of the input array is {dim}."
        )

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.rotate_back")
    def rotate_back(q: Quaternion, v: jax.Array) -> jax.Array:
        r"""Rotates a vector :math:`\vec{v}` from the lab reference frame to the body reference frame.

        This performs the inverse rotation using the quaternion conjugate :math:`q^* = (w, -\vec{q}_{xyz})`:

        .. math::
            \vec{v}' = \vec{v} - 2 w (\vec{q}_{xyz} \times \vec{v}) + 2 (\vec{q}_{xyz} \times (\vec{q}_{xyz} \times \vec{v}))

        Parameters
        ----------
        q : Quaternion
            The rotation quaternion.
        v : jax.Array
            The vector to rotate back. Shape is `(..., dim)`.

        Returns
        -------
        jax.Array
            The rotated vector in the body frame. Shape is `(..., dim)`.
        """
        q = Quaternion.conj(q)
        return Quaternion.rotate(q, v)


    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.from_rotation_matrix")
    def from_rotation_matrix(v_rot: jax.Array) -> "Quaternion":
        def safe_sqrt(x: jax.Array) -> jax.Array:
            return jnp.sqrt(jnp.maximum(x, 1e-8))

        trace = v_rot[..., 0, 0] + v_rot[..., 1, 1] + v_rot[..., 2, 2]

        val_trace = 1.0 + trace
        S_trace = safe_sqrt(val_trace) * 2.0
        q_trace = jnp.stack(
            [
                0.25 * S_trace,
                (v_rot[..., 2, 1] - v_rot[..., 1, 2]) / S_trace,
                (v_rot[..., 0, 2] - v_rot[..., 2, 0]) / S_trace,
                (v_rot[..., 1, 0] - v_rot[..., 0, 1]) / S_trace,
            ],
            axis=-1,
        )

        val_col0 = 1.0 + v_rot[..., 0, 0] - v_rot[..., 1, 1] - v_rot[..., 2, 2]
        S_col0 = safe_sqrt(val_col0) * 2.0
        q_col0 = jnp.stack(
            [
                (v_rot[..., 2, 1] - v_rot[..., 1, 2]) / S_col0,
                0.25 * S_col0,
                (v_rot[..., 0, 1] + v_rot[..., 1, 0]) / S_col0,
                (v_rot[..., 0, 2] + v_rot[..., 2, 0]) / S_col0,
            ],
            axis=-1,
        )

        val_col1 = 1.0 - v_rot[..., 0, 0] + v_rot[..., 1, 1] - v_rot[..., 2, 2]
        S_col1 = safe_sqrt(val_col1) * 2.0
        q_col1 = jnp.stack(
            [
                (v_rot[..., 0, 2] - v_rot[..., 2, 0]) / S_col1,
                (v_rot[..., 0, 1] + v_rot[..., 1, 0]) / S_col1,
                0.25 * S_col1,
                (v_rot[..., 1, 2] + v_rot[..., 2, 1]) / S_col1,
            ],
            axis=-1,
        )

        val_col2 = 1.0 - v_rot[..., 0, 0] - v_rot[..., 1, 1] + v_rot[..., 2, 2]
        S_col2 = safe_sqrt(val_col2) * 2.0
        q_col2 = jnp.stack(
            [
                (v_rot[..., 1, 0] - v_rot[..., 0, 1]) / S_col2,
                (v_rot[..., 0, 2] + v_rot[..., 2, 0]) / S_col2,
                (v_rot[..., 1, 2] + v_rot[..., 2, 1]) / S_col2,
                0.25 * S_col2,
            ],
            axis=-1,
        )

        q_arr = jnp.where(
            trace[..., None] > 0,
            q_trace,
            jnp.where(
                (v_rot[..., 0, 0] > v_rot[..., 1, 1])[..., None]
                & (v_rot[..., 0, 0] > v_rot[..., 2, 2])[..., None],
                q_col0,
                jnp.where(
                    (v_rot[..., 1, 1] > v_rot[..., 2, 2])[..., None],
                    q_col1,
                    q_col2,
                ),
            ),
        )
        return Quaternion(w=q_arr[..., 0:1], xyz=q_arr[..., 1:])

    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="Quaternion.__matmul__")
    def __matmul__(self, other: Quaternion) -> Quaternion:  # q @ r
        r"""Multiplies two quaternions :math:`q_1 = (w_1, \vec{v}_1)` and :math:`q_2 = (w_2, \vec{v}_2)`.

        The quaternion multiplication (Hamilton product) is defined as:

        .. math::
            q_{new} = (w_1 w_2 - \vec{v}_1 \cdot \vec{v}_2, \, w_1 \vec{v}_2 + w_2 \vec{v}_1 + \vec{v}_1 \times \vec{v}_2)

        Parameters
        ----------
        other : Quaternion
            The other quaternion to multiply by.

        Returns
        -------
        Quaternion
            The product quaternion.
        """
        w = self.w * other.w - dot(self.xyz, other.xyz)[..., None]
        xyz = self.w * other.xyz + other.w * self.xyz + cross(self.xyz, other.xyz)
        return Quaternion(w, xyz)
