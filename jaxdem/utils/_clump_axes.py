# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Stable principal frames for clump inertia tensors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

import numpy as np


def _canonical_columns(vectors: jax.Array) -> jax.Array:
    """Fix eigenvector signs, using the last axis to enforce handedness."""
    pivot_index = jnp.argmax(jnp.abs(vectors), axis=-2)
    pivots = jnp.take_along_axis(vectors, pivot_index[..., None, :], axis=-2)[..., 0, :]
    vectors = vectors * jnp.where(pivots < 0.0, -1.0, 1.0)[..., None, :]
    handedness = jnp.sum(
        jnp.cross(vectors[..., :, 0], vectors[..., :, 1]) * vectors[..., :, 2],
        axis=-1,
    )
    return vectors.at[..., :, 2].multiply(
        jnp.where(handedness < 0.0, -1.0, 1.0)[..., None]
    )


def _plane_frame(axis: jax.Array, *, unique_last: bool) -> jax.Array:
    """Complete a unique eigenaxis with a deterministic orthonormal plane."""
    reference_index = jnp.argmin(jnp.abs(axis), axis=-1)
    reference = jax.nn.one_hot(reference_index, 3, dtype=axis.dtype)
    first = reference - jnp.sum(reference * axis, axis=-1, keepdims=True) * axis
    first = first / jnp.sqrt(jnp.sum(first * first, axis=-1, keepdims=True))
    second = jnp.cross(axis, first)
    columns = (first, second, axis) if unique_last else (axis, first, second)
    return jnp.stack(columns, axis=-1)


@jax.jit
def _principal_frame(
    tensor: jax.Array,
    isotropic_moment: jax.Array,
    positive_semidefinite: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return sorted principal moments and a deterministic body-to-world frame.

    ``tensor`` has shape ``(..., 3, 3)`` and excludes the isotropic intrinsic
    moment, which is added after diagonalization. This separation preserves
    tiny positive intrinsic moments about nearly singular point-mass axes.
    For tensors known to be positive semidefinite, negative eigenvalues from
    roundoff are clamped before that addition. Sampled correction tensors
    may have physical negative eigenvalues, so callers can disable clamping.

    Eigenvector signs and handedness are canonicalized. At numerical
    degeneracy (64 machine epsilons relative to the tensor spectrum), a
    repeated eigenspace uses a deterministic projected world axis; a fully
    isotropic tensor uses the identity. The returned quaternion has order
    ``[w, x, y, z]`` and a positive first nonzero component.
    """
    tensor = 0.5 * (tensor + jnp.swapaxes(tensor, -1, -2))
    values, vectors = jnp.linalg.eigh(tensor)
    values = jnp.where(
        jnp.asarray(positive_semidefinite)[..., None],
        jnp.maximum(values, 0.0),
        values,
    )
    vectors = _canonical_columns(vectors)

    scale = jnp.max(jnp.abs(values), axis=-1)
    tolerance = 64.0 * np.finfo(tensor.dtype).eps * scale
    equal_low = values[..., 1] - values[..., 0] <= tolerance
    equal_high = values[..., 2] - values[..., 1] <= tolerance
    low_plane = _plane_frame(vectors[..., :, 2], unique_last=True)
    high_plane = _plane_frame(vectors[..., :, 0], unique_last=False)
    vectors = jnp.where(equal_low[..., None, None], low_plane, vectors)
    vectors = jnp.where(equal_high[..., None, None], high_plane, vectors)
    vectors = jnp.where(
        (equal_low & equal_high)[..., None, None],
        jnp.eye(3, dtype=tensor.dtype),
        vectors,
    )

    xyzw = Rotation.from_matrix(vectors).as_quat()
    quaternion = jnp.concatenate((xyzw[..., 3:], xyzw[..., :3]), axis=-1)
    quaternion = quaternion / jnp.sqrt(
        jnp.sum(quaternion * quaternion, axis=-1, keepdims=True)
    )
    first_index = jnp.argmax(quaternion != 0.0, axis=-1)
    first = jnp.take_along_axis(quaternion, first_index[..., None], axis=-1)
    quaternion = quaternion * jnp.where(first < 0.0, -1.0, 1.0)
    moments = values + jnp.asarray(isotropic_moment)[..., None]
    return moments, quaternion
