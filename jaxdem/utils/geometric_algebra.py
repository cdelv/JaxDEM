import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from functools import partial
from typing import Union


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Bivector:
    data: jax.Array

    @classmethod
    def create(cls, data: jax.Array, dim: int):
        expected_size = (dim * (dim - 1)) // 2
        if data.shape[-1] != expected_size:
            raise ValueError(f"Shape mismatch: dim={dim} requires size {expected_size}")
        return cls(data)

    @property
    def dim(self: "Bivector"):
        from math import sqrt

        size = self.data.shape[-1]
        dim: int = int((1 + sqrt(8 * size + 1)) / 2)
        return dim

    @partial(jax.jit, inline=True)
    def __neg__(self: "Bivector") -> "Bivector":
        """-B"""
        return Bivector(-self.data)

    @partial(jax.jit, inline=True)
    def __add__(self: "Bivector", other: "Bivector") -> "Bivector":
        """B1 + B2"""
        return Bivector(self.data + other.data)

    @partial(jax.jit, inline=True)
    def __sub__(self: "Bivector", other: "Bivector") -> "Bivector":
        """B1 - B2"""
        return Bivector(self.data - other.data)

    @partial(jax.jit, inline=True)
    def __mul__(
        self: "Bivector", other: Union[jax.Array, "Rotor"]
    ) -> Union["Bivector", "Rotor"]:
        """
        Handles two cases:
        1. Scalar product: B * Scalar -> Bivector
        2. Geometric Product: B1 B2 -> Rotor = B1.B2 + B1xB2
        """
        if isinstance(other, Bivector):
            # Geometric Product
            MatA = Bivector.to_matrix(self)
            MatB = Bivector.to_matrix(other)
            P = MatA @ MatB

            # 1. Scalar Part (Dot): Tr(P) / 2
            # Note: This returns the GA scalar product (negative definite)
            s = jnp.trace(P, axis1=-2, axis2=-1) / 2.0

            # 2. Bivector Part (Commutator): (P - P.T) / 2
            P_skew = P - jnp.swapaxes(P, -1, -2)
            b = Bivector.from_matrix(P_skew)

            return Rotor(s, b)
        else:
            # Scalar product
            return Bivector(self.data * other)

    @partial(jax.jit, inline=True)
    def __rmul__(self, other: jax.Array) -> "Bivector":
        """Scalar product: a * B"""
        return Bivector(self.data * other)

    @staticmethod
    @partial(jax.jit, inline=True)
    def wedge(u: jax.Array, v: jax.Array) -> "Bivector":
        """
        Vector ^ Vector -> Bivector
        u, v are JAX Arrays of shape (..., dim)
        """
        # Infer dimension from the vector input
        dim = u.shape[-1]

        with jax.ensure_compile_time_eval():
            r_idx, c_idx = jnp.triu_indices(dim, k=1)

        data = u[..., r_idx] * v[..., c_idx] - u[..., c_idx] * v[..., r_idx]
        return Bivector(data)

    @staticmethod
    @partial(jax.jit, inline=True)
    def dot_vector(b: "Bivector", v: jax.Array) -> jax.Array:
        B_mat = Bivector.to_matrix(b)
        return jnp.einsum("...ij,...j->...i", B_mat, v)

    @staticmethod
    @partial(jax.jit, inline=True)
    def commutator(A: "Bivector", B: "Bivector") -> "Bivector":
        """
        Bivector x Bivector -> Bivector
        Computes the Lie Bracket [A, B] = AB - BA.
        """
        MatA = Bivector.to_matrix(A)
        MatB = Bivector.to_matrix(B)

        # [A, B] = A @ B - B @ A
        # The result of two skew-symmetric matrices commuted is also skew-symmetric.
        MatC = MatA @ MatB - MatB @ MatA
        return Bivector.from_matrix(MatC)

    @staticmethod
    @partial(jax.jit, inline=True)
    def to_matrix(b: "Bivector") -> jax.Array:
        with jax.ensure_compile_time_eval():
            dim = b.dim
            r_idx, c_idx = jnp.triu_indices(dim, k=1)

        batch_shape = b.data.shape[:-1]
        mat = jnp.zeros(batch_shape + (dim, dim), dtype=b.data.dtype)
        mat = mat.at[..., r_idx, c_idx].set(b.data)
        mat = mat.at[..., c_idx, r_idx].set(-b.data)
        return mat

    @staticmethod
    @partial(jax.jit, inline=True)
    def from_matrix(mat: jax.Array) -> "Bivector":
        dim = mat.shape[-1]
        with jax.ensure_compile_time_eval():
            r_idx, c_idx = jnp.triu_indices(dim, k=1)

        data = mat[..., r_idx, c_idx]
        return Bivector(data)

    @staticmethod
    @partial(jax.jit, inline=True)
    def to_bivector(v: jax.Array) -> "Bivector":
        dim = v.shape[-1]

        if dim == 3:
            # 3D: [wx, wy, wz] -> [-wz (01), wy (02), -wx (12)]
            # Indices for 3D triu: (0,1)->XY, (0,2)->XZ, (1,2)->YZ
            # w_z corresponds to XY plane (Index 0) with sign -1
            # w_y corresponds to XZ plane (Index 1) with sign +1
            # w_x corresponds to YZ plane (Index 2) with sign -1
            data = jnp.stack([-v[..., 2], v[..., 1], -v[..., 0]], axis=-1)
            return Bivector(data)
        elif dim == 1:  # 2D scalar "vector"
            return Bivector(-v)
        else:
            raise NotImplementedError

    @staticmethod
    @partial(jax.jit, inline=True)
    def to_vector(b: "Bivector") -> jax.Array:
        dim = b.dim
        data = b.data
        if dim == 3:
            # Inverse of above:
            # v[0] (wx) = -data[2]
            # v[1] (wy) = +data[1]
            # v[2] (wz) = -data[0]
            return jnp.stack([-data[..., 2], data[..., 1], -data[..., 0]], axis=-1)
        elif dim == 2:
            return -data
        else:
            raise NotImplementedError

    @staticmethod
    @partial(jax.jit, inline=True)
    def exp(b: "Bivector") -> "Rotor":
        """
        Exponential Map: Lie Algebra (Bivector) -> Lie Group (Rotor).
        R = exp(B) = cos(|B|) + B/|B| * sin(|B|)
        """
        # Calculate magnitude (theta)
        theta2 = jnp.sum(b.data * b.data, axis=-1)
        theta = jnp.sqrt(theta2)

        # Scalar part (cos(|B|))
        w = jnp.cos(theta)

        # Bivector part (B/|B| * sin(|B|)
        theta_safe = jnp.where(theta == 0, 1.0, theta)
        bivec = b.data * jnp.sin(theta) / theta_safe

        return Rotor(w, Bivector(bivec))


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Rotor:
    w: jax.Array
    bivec: "Bivector"

    @classmethod
    def create(cls, w: jax.Array, bivec: "Bivector"):
        if w.ndim == 0 and bivec.data.ndim > 1:
            w = jnp.broadcast_to(w, bivec.data.shape[:-1])
        return cls(w, bivec)

    @staticmethod
    @partial(jax.jit, inline=True, static_argnames=("dim",))
    def identity(dim: int, batch_shape=()):
        size = (dim * (dim - 1)) // 2
        b_data = jnp.zeros(batch_shape + (size,))
        return Rotor(jnp.ones(batch_shape), Bivector(b_data))

    @staticmethod
    @partial(jax.jit, inline=True)
    def normalize(rot: "Rotor") -> "Rotor":
        """
        Re-normalizes the Rotor so R ~R = 1.
        Essential for preventing numerical drift in physics simulations.
        """
        # |R|^2 = a^2 + |B|^2
        norm2 = rot.w**2 + jnp.sum(rot.bivec.data**2, axis=-1)
        scale = jnp.where(norm2 == 0, 1.0, jax.lax.rsqrt(norm2))
        return Rotor(rot.w * scale, rot.bivec * scale[..., None])

    @staticmethod
    @partial(jax.jit, inline=True)
    def conj(rot: "Rotor") -> "Rotor":
        return Rotor(rot.w, -rot.bivec)

    @staticmethod
    @partial(jax.jit, inline=True)
    def inverse(rot: "Rotor") -> "Rotor":
        rot = rot.normalize(rot)
        return rot.conj(rot)

    @staticmethod
    @partial(jax.jit, inline=True)
    def rotate(rot: "Rotor", vec: jax.Array) -> jax.Array:
        """
        Apply rotation to vector v.
        Formula: v' = v + 2 * ( B.(B.v) + a(B.v) )
        """
        T = Bivector.dot_vector(rot.bivec, vec)  # B . v
        BT = Bivector.dot_vector(rot.bivec, T)  # B.(B.v)
        aT = rot.w[..., None] * T  # a(B.v)
        return vec + 2.0 * (BT + aT)

    @partial(jax.jit, inline=True)
    def __mul__(self: "Rotor", other: "Rotor") -> "Rotor":
        """
        Group Composition: R_new = self * other
        (a1 + B1)(a2 + B2)
        """
        w1, bivec1 = self.w, self.bivec
        w2, bivec2 = other.w, other.bivec

        # 1. Scalar Part: a1a2 + B1.B2
        dot_b = jnp.sum(bivec1.data * bivec2.data, axis=-1)
        new_a = w1 * w2 - dot_b

        # 2. Bivector Part: a1B2 + a2B1 + [B1, B2]
        cross_term = Bivector.commutator(bivec1, bivec2)
        new_b = (w1 * bivec2) + (w2 * bivec1) + cross_term

        return Rotor(new_a, new_b)

    @partial(jax.jit, inline=True)
    def log(self: "Rotor") -> "Bivector":
        """
        Logarithmic Map: Lie Group (Rotor) -> Lie Algebra (Bivector).
        Input: Normalized Rotor R = exp(B).
        Output: Bivector B (representing angle * axis).
        """
        # |B_im| (Magnitude of the Bivector part)
        norm2 = jnp.sum(self.bivec.data * self.bivec.data, axis=-1)
        b_norm = jnp.sqrt(norm2)

        # Angle theta = arctan(|B_im|, a)
        theta = jnp.arctan2(b_norm, self.w)
        safe_norm = jnp.where(b_norm == 0.0, 1.0, b_norm)
        scale = theta / safe_norm

        return Bivector(self.bivec.data * scale[..., None])

    @partial(jax.jit, inline=True)
    def to_matrix(self) -> jax.Array:
        """
        Converts Rotor to Rotation Matrix (NxN).
        Formula: R = I + 2*B_mat @ (B_mat + a*I)
        """
        dim = self.bivec.dim  # Corrected attribute access
        B_mat = Bivector.to_matrix(self.bivec)
        I = jnp.eye(dim)

        # a * I
        aI = self.w[..., None, None] * I

        return I + 2.0 * (B_mat @ (B_mat + aI))

    @staticmethod
    @partial(jax.jit, inline=True)
    def from_matrix(matrix: jax.Array) -> "Rotor":
        """
        Robustly converts Rotation Matrix to Rotor.
        Handles 2D (2x2) by padding to 3D (3x3), solving, and slicing back.
        """
        orig_dim = matrix.shape[-1]

        # 1. Pad 2D matrices to 3D
        if orig_dim == 2:
            shape_diff = 3 - orig_dim
            matrix = jnp.pad(
                matrix,
                ((0, 0),) * (matrix.ndim - 2) + ((0, shape_diff), (0, shape_diff)),
                mode="constant",
            )
            matrix = matrix.at[..., 2, 2].set(1.0)

        # 2. Run Robust Shepperd's Algorithm (3D)
        m00, m11, m22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]
        trace = m00 + m11 + m22

        def case_tr():
            s = jnp.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (matrix[..., 2, 1] - matrix[..., 1, 2]) / s
            y = (matrix[..., 0, 2] - matrix[..., 2, 0]) / s
            z = (matrix[..., 1, 0] - matrix[..., 0, 1]) / s
            return jnp.stack([w, x, y, z], axis=-1)

        def case_diag(i):
            j = (i + 1) % 3
            k = (j + 1) % 3
            s = (
                jnp.sqrt(
                    1.0 + matrix[..., i, i] - matrix[..., j, j] - matrix[..., k, k]
                )
                * 2.0
            )

            # Use matrix shape to determine batch size
            batch_shape = matrix.shape[:-2]
            q = jnp.zeros(batch_shape + (4,), dtype=matrix.dtype)

            q = q.at[..., 0].set((matrix[..., k, j] - matrix[..., j, k]) / s)
            q = q.at[..., 1 + i].set(0.25 * s)
            q = q.at[..., 1 + j].set((matrix[..., j, i] + matrix[..., i, j]) / s)
            q = q.at[..., 1 + k].set((matrix[..., k, i] + matrix[..., i, k]) / s)
            return q

        decision = jnp.array([m00, m11, m22, trace], dtype=matrix.dtype)
        choice = jnp.argmax(decision, axis=0)  # Shape: (Batch...)

        # FIX: Expand dims of choice to (Batch..., 1) for broadcasting against (Batch..., 4)
        cond_3 = (choice == 3)[..., None]
        cond_0 = (choice == 0)[..., None]
        cond_1 = (choice == 1)[..., None]

        q_raw = jnp.where(
            cond_3,
            case_tr(),
            jnp.where(
                cond_0, case_diag(0), jnp.where(cond_1, case_diag(1), case_diag(2))
            ),
        )

        q_norm = q_raw / jnp.linalg.norm(q_raw, axis=-1, keepdims=True)
        res_3d = Rotor.from_quat(q_norm)

        if orig_dim == 2:
            return Rotor(res_3d.w, Bivector(res_3d.bivec.data[..., 0:1]))
        return res_3d

    @staticmethod
    @partial(jax.jit, inline=True)
    def from_axis_angle(axis: jax.Array, angle: jax.Array) -> "Rotor":
        """
        Creates Rotor from a vector axis and an angle.
        R = cos(a/2) - sin(a/2) * (AxisDual)
        """
        # Normalize axis
        norm2 = jnp.sum(axis * axis, axis=-1, keepdims=True)  # (..., 1)
        scale = jnp.where(norm2 == 0, 1.0, jax.lax.rsqrt(norm2))  # (..., 1)
        axis = axis * scale
        B_plane = Bivector.to_bivector(axis)

        half_angle = angle / 2
        w = jnp.cos(half_angle)
        s = jnp.sin(half_angle)

        # Note: GA convention R = c - sB
        return Rotor(w, Bivector(s[..., None] * B_plane.data))

    @staticmethod
    @partial(jax.jit, inline=True)
    def from_quat(q: jax.Array) -> "Rotor":
        """
        Import from Quaternion [w, x, y, z].
        Assumes 3D. Maps (x, y, z) -> Bivectors (-YZ, XZ, -XY).
        """
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        # 3D Bivector Layout: [XY, XZ, YZ]
        # XY (idx 0) <- -z
        # XZ (idx 1) <- y
        # YZ (idx 2) <- -x
        b_data = jnp.stack([-z, y, -x], axis=-1)
        return Rotor(w, Bivector(b_data))

    @partial(jax.jit, inline=True)
    def to_quat(self) -> jax.Array:
        """Export to Quaternion [w, x, y, z] (3D only)"""
        b = self.bivec.data
        # w=w, x=-YZ, y=XZ, z=-XY
        return jnp.stack([self.w, -b[..., 2], b[..., 1], -b[..., 0]], axis=-1)
