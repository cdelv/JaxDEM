# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""
Utility functions to randomize particle orientations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from typing import TYPE_CHECKING
from functools import partial

from .quaternion import Quaternion

if TYPE_CHECKING:
    from ..state import State


@jax.jit
@partial(jax.named_call, name="utils.randomize_orientations")
def randomize_orientations(state: State, key: jax.Array) -> State:
    """
    Randomize orientations for clumps (particles with repeated ``state.clump_id``),
    leaving spheres unchanged.
    """
    N = state.N
    dim = state.dim  # static at trace time (derived from shapes)

    def _one(
        k: jax.Array, id_i: jax.Array, w_i: jax.Array, xyz_i: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        counts = jnp.bincount(id_i, length=N)
        is_clump_member = counts[id_i] > 1  # (N,)

        if dim == 2:
            theta_by_id = jax.random.uniform(k, (N,), minval=0.0, maxval=2.0 * jnp.pi)
            theta = theta_by_id[id_i]
            w_s = jnp.cos(0.5 * theta)[:, None]
            xyz_s = jnp.stack(
                [jnp.zeros_like(theta), jnp.zeros_like(theta), jnp.sin(0.5 * theta)],
                axis=-1,
            )
        else:  # dim == 3
            q4_by_id = jax.random.normal(k, (N, 4))
            q4_by_id = q4_by_id / jnp.linalg.norm(
                q4_by_id, axis=-1, keepdims=True
            )  # uniform rotation
            q4 = q4_by_id[id_i]  # same orientation for same clump ID
            w_s = q4[:, 0:1]
            xyz_s = q4[:, 1:4]

        w_new = jnp.where(is_clump_member[:, None], w_s, w_i)  # spheres: unchanged
        xyz_new = jnp.where(
            is_clump_member[:, None], xyz_s, xyz_i
        )  # spheres: unchanged
        q_new = Quaternion.unit(Quaternion(w_new, xyz_new))
        return q_new.w, q_new.xyz

    # Match common batching conventions: vmap over axis 0 if present.
    lead_ndim = state.clump_id.ndim - 1  # leading axes before particle axis N

    if lead_ndim == 0:
        w_new, xyz_new = _one(key, state.clump_id, state.q.w, state.q.xyz)
        state.q = Quaternion(w_new, xyz_new)

    elif lead_ndim == 1:
        keys = jax.random.split(key, state.clump_id.shape[0])
        w_new, xyz_new = jax.vmap(_one)(keys, state.clump_id, state.q.w, state.q.xyz)
        state.q = Quaternion(w_new, xyz_new)

    else:
        # For stacked trajectories (or other multi-leading-dim states), flatten
        # and then reshape back to preserve the original layout.
        lead_shape = state.clump_id.shape[:-1]
        ids = state.clump_id.reshape((-1, N))
        w0 = state.q.w.reshape((-1, N, 1))
        xyz0 = state.q.xyz.reshape((-1, N, 3))
        keys = jax.random.split(key, ids.shape[0])

        w_flat, xyz_flat = jax.vmap(_one)(keys, ids, w0, xyz0)
        w_new = w_flat.reshape(lead_shape + (N, 1))
        xyz_new = xyz_flat.reshape(lead_shape + (N, 3))
        state.q = Quaternion(w_new, xyz_new)

    return state
